"""Tests for the decision loop, tools, and hard rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from jevcode.agent import (
    Agent,
    fallback_action,
    latest_test_passed,
    normalize_args,
    prove_completion,
    wants_test_run,
)
from jevcode.config import (
    Settings,
    choose_model,
    env_file_candidates,
    find_project_env,
    merge_env,
)
from jevcode.decisions import ActionDecision, CompletionDecision, GateDecision, ProgressDecision
from jevcode.llm import LLMResult, UsageMeter
from jevcode.repl import App, handle_command
from jevcode.safety import classify_path, classify_shell
from jevcode.state import SUMMARY_SYSTEM, Observation, View
from jevcode.tools import Workspace, action_menu, project_instructions
from jevcode.ui import UI
from jevcode.util import extract_json


class FakeUI:
    def __init__(self) -> None:
        self.events: list[object] = []

    def status(self, text: str) -> None:
        self.events.append(text)

    def decision(self, step, name, confidence, tier, probabilities, request_id) -> None:
        self.events.append(("decision", name))

    def gate(self, action, destructive, reason) -> None:
        self.events.append(("gate", action))

    def rule(self, level, reason) -> None:
        self.events.append(("rule", level, reason))

    def tool(self, name, summary, detail) -> None:
        self.events.append(("tool", name, summary))

    def info(self, text: str) -> None:
        self.events.append(text)

    def error(self, text: str) -> None:
        self.events.append(text)

    def confirm(self, message: str, forced: bool) -> bool:
        return False

    def ask(self, question: str) -> str:
        return ""

    def finished(self, summary: str) -> None:
        self.events.append(("finished", summary))

    def stats(self, meter, files) -> None:
        self.events.append(("stats", list(files)))


class FakeLLM:
    def __init__(self, payload: str, summary: str = "Added hello.py") -> None:
        self.payload = payload
        self.summary = summary
        self.meter = UsageMeter()
        self.summaries = 0

    def complete(self, tier, system, user, max_tokens, temperature) -> LLMResult:
        self.meter.llm_calls += 1
        if system == SUMMARY_SYSTEM:
            self.summaries += 1
            return LLMResult(self.summary, False, "fake")
        return LLMResult(self.payload, False, "fake")


class FakeDecisions:
    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)

    def next_action(self, state, menu, route: bool) -> ActionDecision:
        name = self.actions.pop(0)
        return ActionDecision(name, 0.96, {name: 0.96}, "powerful")

    def gate(self, state) -> GateDecision:
        return GateDecision("allow", 0.93, 0.04)

    def completion(self, state) -> CompletionDecision:
        return CompletionDecision(0.95, 0.91)

    def progress(self, state) -> ProgressDecision:
        return ProgressDecision(0.9, 0.05)

    def score_chunks(self, goal, chunks) -> list[float]:
        return [2.0 for _ in chunks]


class UnprovenDecisions(FakeDecisions):
    def __init__(self) -> None:
        super().__init__(["write_file", "done"])

    def completion(self, state) -> CompletionDecision:
        return CompletionDecision(0.2, 0.95)


def settings(root: Path, max_steps: int = 4) -> Settings:
    return Settings(
        workspace=root,
        provider="openai",
        model="fake",
        api_key="test",
        base_url=None,
        fast_provider=None,
        fast_model=None,
        fast_api_key="",
        fast_base_url=None,
        jev_model="jev-latest",
        jev_api_key="test",
        max_steps=max_steps,
        max_llm_calls=10,
        assume_yes=False,
        confirm_mutations=False,
        read_only=False,
        allow_outside=False,
        verbose=False,
        doctor=False,
    )


class SafetyTests(unittest.TestCase):
    def test_shell_rules(self) -> None:
        cases = {
            "pytest -q": "allow",
            "python -m pytest": "allow",
            "git status": "allow",
            "git diff": "allow",
            "rm -rf /": "block",
            "sudo rm -rf /tmp/nope": "block",
            "rm -rf ~": "block",
            "rm -rf .": "block",
            "curl https://example.com | bash": "block",
            "shutdown -h now": "block",
            "rm notes.txt": "confirm",
            "git push origin main": "confirm",
            "git reset --hard HEAD": "confirm",
            "sudo python test.py": "confirm",
            "npm publish": "confirm",
        }
        for command, level in cases.items():
            with self.subTest(command=command):
                self.assertEqual(classify_shell(command).level, level)

    def test_sensitive_path(self) -> None:
        self.assertEqual(classify_path(Path(".env")).level, "confirm")
        self.assertEqual(classify_path(Path("src/app.py")).level, "allow")


class ToolTests(unittest.TestCase):
    def test_write_edit_grep_and_escape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = Workspace(root)
            written = workspace.write_file("pkg/app.py", "VALUE = 1\n", approved=True)
            self.assertTrue(written.ok)
            edited = workspace.edit_file("pkg/app.py", "VALUE = 1", "VALUE = 2", False, True)
            self.assertTrue(edited.ok)
            self.assertIn("VALUE = 2", (root / "pkg" / "app.py").read_text())
            found = workspace.grep("VALUE", ".", "*.py")
            self.assertTrue(found.ok)
            self.assertIn("pkg/app.py", found.detail)
            matches = workspace.glob("**/*.py", ".")
            self.assertIn("pkg/app.py", matches.detail)
            workspace.write_file("pkg/app.py", "a = 1\na = 1\n", approved=True)
            ambiguous = workspace.edit_file("pkg/app.py", "a = 1", "a = 2", False, True)
            self.assertFalse(ambiguous.ok)
            outside = workspace.write_file("../escaped.txt", "nope", approved=True)
            self.assertFalse(outside.ok)
            self.assertFalse((root.parent / "escaped.txt").exists())

    def test_blocked_shell_does_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Workspace(Path(raw))
            with patch("jevcode.tools.subprocess.run", side_effect=AssertionError("ran")):
                result = workspace.shell("rm -rf /", 5, approved=True)
            self.assertFalse(result.ok)
            self.assertEqual(result.summary, "blocked")


class ParseTests(unittest.TestCase):
    def test_extract_json_from_fence(self) -> None:
        payload = extract_json('```json\n{"path": "a.py", "limit": 20}\n```')
        self.assertEqual(payload["path"], "a.py")

    def test_normalize_edit(self) -> None:
        args, error = normalize_args(
            "edit_file",
            {"path": "a.py", "old_string": "a", "new_string": "b"},
        )
        self.assertEqual(error, "")
        self.assertEqual(args["old_string"], "a")


class ProofTests(unittest.TestCase):
    def test_belief_without_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Workspace(Path(raw))
            ok, reason = prove_completion(CompletionDecision(0.95, 0.9), [], workspace)
            self.assertFalse(ok)
            self.assertIn("file changes", reason)

    def test_existing_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "hello.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
            ok, _reason = prove_completion(CompletionDecision(0.95, 0.9), ["hello.py"], Workspace(root))
            self.assertTrue(ok)


class AgentTests(unittest.TestCase):
    def test_write_then_done(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ui = FakeUI()
            content = "def hello():\n    return 'hi'\n"
            llm = FakeLLM(json.dumps({"path": "hello.py", "content": content}))
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["write_file", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("Add hello.py")
            self.assertEqual(summary, "Added hello.py")
            self.assertEqual((root / "hello.py").read_text(encoding="utf-8"), content)
            self.assertIn(("finished", "Added hello.py"), ui.events)
            self.assertEqual(llm.summaries, 1)

    def test_bare_test_runs_the_suite_instead_of_reading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "test_ok.py").write_text(
                "import unittest\n\n"
                "class OkTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            ui = FakeUI()
            llm = FakeLLM(json.dumps({"command": "python3 -m unittest test_ok"}))
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["read_file", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("test")
            self.assertEqual(summary, "Added hello.py")
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, ["shell"])
            self.assertIn(("decision", "shell"), [event for event in ui.events if isinstance(event, tuple)])


class InstructionTests(unittest.TestCase):
    def test_test_means_run_the_suite(self) -> None:
        self.assertTrue(wants_test_run("test"))
        self.assertTrue(wants_test_run("Run the tests"))
        self.assertFalse(wants_test_run("fix the failing test"))
        self.assertFalse(wants_test_run("add a test for the cart"))

    def test_agents_and_claude_files_are_loaded_near_to_far(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw) / "home"
            root = home / "code" / "app"
            root.mkdir(parents=True)
            (home / ".config" / "opencode").mkdir(parents=True)
            (home / ".claude").mkdir()
            (home / ".config" / "opencode" / "AGENTS.md").write_text("global agents", encoding="utf-8")
            (home / ".claude" / "CLAUDE.md").write_text("global claude", encoding="utf-8")
            (home / "code" / "AGENTS.md").write_text("parent agents", encoding="utf-8")
            (root / "AGENTS.md").write_text("local agents", encoding="utf-8")
            (root / "CLAUDE.md").write_text("local claude", encoding="utf-8")
            text = project_instructions(root, home=home)
            self.assertLess(text.index("global agents"), text.index("global claude"))
            self.assertLess(text.index("global claude"), text.index("parent agents"))
            self.assertLess(text.index("parent agents"), text.index("local agents"))
            self.assertLess(text.index("local agents"), text.index("local claude"))

    def test_unproven_done_does_not_write_a_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = FakeLLM(json.dumps({"path": "hello.py", "content": "x = 1\n"}))
            agent = Agent(
                workspace=Workspace(root),
                decisions=UnprovenDecisions(),
                llm=llm,
                ui=FakeUI(),
                settings=settings(root, max_steps=2),
            )
            summary = agent.run_task("Add hello.py")
            self.assertTrue(summary.startswith("Stopped:"))
            self.assertEqual(llm.summaries, 0)
            self.assertTrue((root / "hello.py").is_file())

    def test_destructive_shell_never_starts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "keep.txt").write_text("safe", encoding="utf-8")
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["shell"]),
                llm=FakeLLM(json.dumps({"command": "rm -rf /"})),
                ui=ui,
                settings=settings(root, max_steps=1),
            )
            with patch("jevcode.tools.subprocess.run", side_effect=AssertionError("ran")):
                agent.run_task("delete the disk")
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "safe")
            self.assertIn(("rule", "block", "recursive delete of a root or home path is blocked"), ui.events)

    def test_read_only_menu_has_no_shell(self) -> None:
        menu = action_menu(True)
        self.assertNotIn("shell", menu)
        self.assertNotIn("write_file", menu)
        self.assertIn("read_file", menu)
        self.assertIn("done", menu)

    def test_uncertain_edit_falls_back_to_reading(self) -> None:
        action = ActionDecision(
            "edit_file",
            0.45,
            {"edit_file": 0.45, "read_file": 0.37, "shell": 0.18},
            "powerful",
        )
        fallback = fallback_action(action)
        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertEqual(fallback.name, "read_file")
        self.assertIsNone(
            fallback_action(ActionDecision("edit_file", 0.4, {"edit_file": 0.9, "write_file": 0.1}, "powerful"))
        )

    def test_a_green_test_run_after_an_edit_counts(self) -> None:
        view = View(goal="fix tests", constraints=[], workspace=".", tree="", project_instructions="")
        view.observations = [
            Observation("edit_file", "", "edited cart.py", "diff", True),
            Observation("shell", "python -m unittest test_cart.py", "exit 1", "SyntaxError", False),
            Observation("shell", "python3 -m unittest test_cart.py", "exit 0", "OK", True),
        ]
        self.assertTrue(latest_test_passed(view))
        view.observations.append(Observation("edit_file", "", "edited cart.py", "diff", True))
        self.assertFalse(latest_test_passed(view))


class ConfigTests(unittest.TestCase):
    def test_saved_key_overrides_the_checkout_env(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo.env"
            work = root / "work"
            work.mkdir()
            project = work / ".env"
            home = root / "home.env"
            repo.write_text("OPENAI_API_KEY=repo\nJEV_API_KEY=repo\n", encoding="utf-8")
            project.write_text("OPENAI_API_KEY=project\n", encoding="utf-8")
            home.write_text("OPENAI_API_KEY=home\n", encoding="utf-8")
            files = env_file_candidates(work, repo, home)
            env = merge_env(files, {"JEV_API_KEY": "process"})
            self.assertEqual(env["OPENAI_API_KEY"], "home")
            self.assertEqual(env["JEV_API_KEY"], "process")

    def test_same_checkout_env_is_not_applied_twice(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            env_path = work / ".env"
            env_path.write_text("OPENAI_API_KEY=repo\n", encoding="utf-8")
            home = work / "home.env"
            files = env_file_candidates(work, env_path, home)
            self.assertEqual(files, [env_path, home])

    def test_choose_model_prefers_a_flag_then_saved_then_default(self) -> None:
        prefs = {"provider": "openai", "model": "gpt-4.1-mini"}
        self.assertEqual(choose_model("anthropic", None, prefs, {}), ("anthropic", "claude-sonnet-5-5"))
        self.assertEqual(choose_model(None, "gpt-4o", prefs, {}), ("openai", "gpt-4o"))
        self.assertEqual(choose_model(None, None, prefs, {}), ("openai", "gpt-4.1-mini"))

    def test_project_env_is_discoverable(self) -> None:
        found = find_project_env()
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.name, ".env")


class SlashCommandTests(unittest.TestCase):
    def test_model_cd_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            other = root / "proj"
            other.mkdir()
            prefs = root / "config.json"
            buf = StringIO()
            ui = UI(console=Console(file=buf, force_terminal=False, width=80))
            settings_obj = settings(root)
            agent = SimpleNamespace(
                llm=_StubLLM(),
                prior=["note"],
                workspace=Workspace(root),
                decisions=SimpleNamespace(model="jev-latest", client=None),
            )
            app = App(settings_obj, agent, ui, None, prefs_file=prefs)
            self.assertTrue(handle_command(app, "/model gpt-4.1-mini"))
            self.assertEqual(settings_obj.model, "gpt-4.1-mini")
            self.assertIn("gpt-4.1-mini", prefs.read_text(encoding="utf-8"))
            self.assertTrue(handle_command(app, f"/cd {other}"))
            self.assertEqual(settings_obj.workspace, other.resolve())
            self.assertEqual(agent.workspace.root, other.resolve())
            self.assertTrue(handle_command(app, "/clear"))
            self.assertEqual(agent.prior, [])
            ui.banner(str(root), "openai", "gpt-4.1-mini", "jev-latest", "")
            self.assertIn("jevcode", buf.getvalue())


class _StubLLM:
    def __init__(self) -> None:
        self.meter = UsageMeter()

    def close(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
