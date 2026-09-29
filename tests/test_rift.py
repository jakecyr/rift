"""Tests for the decision loop, tools, and hard rules."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from rift.agent import (
    Agent,
    latest_test_passed,
    normalize_args,
    prove_completion,
)
from rift.config import (
    Settings,
    choose_model,
    env_file_candidates,
    find_project_env,
    merge_env,
)
from rift.decisions import ActionDecision, CompletionDecision, GateDecision, ProgressDecision
from rift.gitstatus import parse_porcelain, secret_staging
from rift.llm import GenerationError, LLM, LLMResult, Profile, UsageMeter, _openai_text, writer_rates
from rift.prompt import argument_system
from rift.repl import App, _toolbar, handle_command, run_bang
from rift.safety import classify_path, classify_shell
from rift.state import (
    ARG_SYSTEM,
    PLAN_SYSTEM,
    SUMMARY_SYSTEM,
    Observation,
    View,
    jev_state,
    llm_user_message,
)
from rift.tools import Workspace, action_menu, project_instructions
from rift.ui import UI
from rift.util import extract_json
from rift.web import html_to_text, parse_search_results, reject_url


class FakeUI:
    def __init__(self) -> None:
        self.events: list[object] = []

    def status(self, text: str) -> None:
        self.events.append(text)

    def plan(self, task, steps, done_when) -> None:
        self.events.append(("plan", task, list(steps)))

    def decision(self, step, name, confidence, tier, probabilities, request_id) -> None:
        self.events.append(("decision", name))

    def gate(self, action, destructive, reason) -> None:
        self.events.append(("gate", action))

    def rule(self, level, reason) -> None:
        self.events.append(("rule", level, reason))

    def tool(self, name, summary, detail, paths=()) -> None:
        self.events.append(("tool", name, summary))

    def trace(self, text: str) -> None:
        self.events.append(("trace", text))

    def info(self, text: str) -> None:
        self.events.append(text)

    def error(self, text: str) -> None:
        self.events.append(text)

    def confirm(self, message: str, forced: bool) -> bool:
        self.events.append(("confirm", message))
        return False

    def ask(self, question: str) -> str:
        return ""

    def finished(self, summary: str) -> None:
        self.events.append(("finished", summary))

    def stats(self, meter, files) -> None:
        self.events.append(("stats", list(files)))


class FakeLLM:
    def __init__(self, payload: str, summary: str = "Added hello.py", plan: str = "{}") -> None:
        self.payload = payload
        self.summary = summary
        self.plan = plan
        self.plan_users: list[str] = []
        self.meter = UsageMeter()
        self.summaries = 0

    def complete(self, tier, system, user, max_tokens, temperature, effort=None) -> LLMResult:
        self.meter.llm_calls += 1
        if system == PLAN_SYSTEM:
            self.plan_users.append(user)
            return LLMResult(self.plan, False, "fake")
        if system == SUMMARY_SYSTEM:
            self.summaries += 1
            return LLMResult(self.summary, False, "fake")
        return LLMResult(self.payload, False, "fake")


class ToolReplyLLM(FakeLLM):
    def __init__(self, replies: dict[str, list[str]], summary: str = "Added the files", plan: str = "{}") -> None:
        super().__init__("{}", summary, plan)
        self.replies = {name: list(payloads) for name, payloads in replies.items()}
        self.users: list[str] = []

    def complete(self, tier, system, user, max_tokens, temperature, effort=None) -> LLMResult:
        self.meter.llm_calls += 1
        if system == PLAN_SYSTEM:
            self.plan_users.append(user)
            return LLMResult(self.plan, False, "fake")
        self.users.append(user)
        if system == SUMMARY_SYSTEM:
            self.summaries += 1
            return LLMResult(self.summary, False, "fake")
        tool = "write_file"
        prefix = "Selected tool:"
        for line in str(user).splitlines():
            if line.startswith(prefix):
                tool = line[len(prefix) :].strip()
                break
        return LLMResult(self.replies[tool].pop(0), False, "fake")


class FakeDecisions:
    def __init__(self, actions: list[str], confidence: float = 0.96) -> None:
        self.actions = list(actions)
        self.confidence = confidence
        self.states: list[dict] = []

    def next_action(self, state, menu, route: bool) -> ActionDecision:
        self.states.append(state)
        name = self.actions.pop(0)
        return ActionDecision(name, self.confidence, {name: self.confidence}, "powerful")

    def gate(self, state) -> GateDecision:
        return GateDecision("allow", 0.93, 0.04)

    def completion(self, state) -> CompletionDecision:
        return CompletionDecision(0.95, 0.91)

    def progress(self, state) -> ProgressDecision:
        return ProgressDecision(0.9, 0.05)

    def score_chunks(self, goal, chunks) -> list[float]:
        return [2.0 for _ in chunks]


class BlockingDecisions(FakeDecisions):
    def gate(self, state) -> GateDecision:
        return GateDecision("block", 0.9, 0.8)


class AnsweredDecisions(FakeDecisions):
    """A question, or a goal the workspace already satisfies, needs no new file changes."""

    def completion(self, state) -> CompletionDecision:
        return CompletionDecision(0.95, 0.1)


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
            with patch("rift.tools.subprocess.run", side_effect=AssertionError("ran")):
                result = workspace.shell("rm -rf /", 5, approved=True)
            self.assertFalse(result.ok)
            self.assertEqual(result.summary, "blocked")

    def test_delete_file_removes_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = Workspace(root)
            (root / "gone.py").write_text("x = 1\n", encoding="utf-8")
            (root / "keep").mkdir()
            removed = workspace.delete_file("gone.py", approved=True)
            self.assertTrue(removed.ok)
            self.assertFalse((root / "gone.py").exists())
            directory = workspace.delete_file("keep", approved=True)
            self.assertFalse(directory.ok)
            outside = workspace.delete_file("../nope.py", approved=True)
            self.assertFalse(outside.ok)

    def test_replace_text_updates_contents_and_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pkg = root / "alpha"
            pkg.mkdir()
            (pkg / "app.py").write_text("import alpha\n", encoding="utf-8")
            (root / "note.md").write_text("run alpha\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=alpha\n", encoding="utf-8")
            result = Workspace(root).replace_text("alpha", "beta", "", False)
            self.assertTrue(result.ok)
            self.assertEqual((root / "beta" / "app.py").read_text(encoding="utf-8"), "import beta\n")
            self.assertEqual((root / "note.md").read_text(encoding="utf-8"), "run beta\n")
            self.assertEqual((root / ".env").read_text(encoding="utf-8"), "TOKEN=alpha\n")

    def test_edit_batch_applies_each_block(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = Workspace(root)
            (root / "a.py").write_text("one\n", encoding="utf-8")
            (root / "b.py").write_text("two\n", encoding="utf-8")
            result = workspace.edit_batch(
                [
                    {"path": "a.py", "old_string": "one", "new_string": "1", "replace_all": False},
                    {"path": "b.py", "old_string": "two", "new_string": "2", "replace_all": False},
                ],
                True,
            )
            self.assertTrue(result.ok)
            self.assertEqual((root / "a.py").read_text(encoding="utf-8"), "1\n")
            self.assertEqual((root / "b.py").read_text(encoding="utf-8"), "2\n")
            self.assertEqual(result.paths, ("a.py", "b.py"))


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
            self.assertIn("nothing was written or run", reason)

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

    def test_a_test_request_runs_the_suite_through_shell(self) -> None:
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
                decisions=FakeDecisions(["shell", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("test")
            self.assertEqual(summary, "Added hello.py")
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, ["shell"])
            self.assertIn(("decision", "shell"), [event for event in ui.events if isinstance(event, tuple)])

    def test_repo_replace_is_one_step(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "app.py").write_text("print('alpha')\n", encoding="utf-8")
            (root / "note.md").write_text("see alpha\n", encoding="utf-8")
            ui = FakeUI()
            llm = FakeLLM(json.dumps({"old": "alpha", "new": "beta"}))
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["replace_text", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("change all instances of alpha to beta")
            self.assertEqual(summary, "Added hello.py")
            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print('beta')\n")
            self.assertEqual((root / "note.md").read_text(encoding="utf-8"), "see beta\n")
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, ["replace_text"])

class InstructionTests(unittest.TestCase):
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
            with patch("rift.tools.subprocess.run", side_effect=AssertionError("ran")):
                agent.run_task("delete the disk")
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "safe")
            self.assertIn(("rule", "block", "recursive delete of a root or home path is blocked"), ui.events)

    def test_read_only_menu_has_no_shell(self) -> None:
        menu = action_menu(True)
        self.assertNotIn("shell", menu)
        self.assertNotIn("write_file", menu)
        self.assertIn("read_file", menu)
        self.assertIn("done", menu)

    def test_a_green_test_run_counts_for_every_writing_tool(self) -> None:
        for tool in ("edit_file", "write_file", "edit_batch", "replace_text"):
            with self.subTest(tool=tool):
                view = View(goal="rename", constraints=[], workspace=".", tree="", project_instructions="")
                view.observations = [
                    Observation(tool, "", f"{tool} changed 2 files", "diff", True),
                    Observation("shell", "python3 -m unittest", "exit 0", "OK", True),
                ]
                self.assertTrue(latest_test_passed(view))

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
            self.assertIn("rift", buf.getvalue())
            self.assertTrue(handle_command(app, "/effort high"))
            self.assertEqual(settings_obj.effort, "high")
            self.assertEqual(agent.llm.effort, "high")
            self.assertIn("high", prefs.read_text(encoding="utf-8"))

    def test_bang_runs_a_command_and_keeps_the_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "note.txt").write_text("hello\n", encoding="utf-8")
            buf = StringIO()
            ui = UI(console=Console(file=buf, force_terminal=False, width=80))
            agent = SimpleNamespace(prior=[], workspace=Workspace(root))
            app = App(settings(root), agent, ui, None)
            run_bang(app, "!ls")
            self.assertIn("note.txt", buf.getvalue())
            self.assertEqual(len(agent.prior), 1)
            self.assertIn("note.txt", agent.prior[0])
            self.assertIn("User ran `ls`", agent.prior[0])
            run_bang(app, "!")
            self.assertEqual(len(agent.prior), 1)
            run_bang(app, "!rm -rf /")
            self.assertEqual(len(agent.prior), 1)
            self.assertIn("blocked", buf.getvalue().lower())


class WebAndMenuTests(unittest.TestCase):
    def test_menu_includes_the_agent_actions(self) -> None:
        menu = action_menu(False)
        for name in ("read_file", "grep", "write_file", "web_search", "web_fetch", "think", "todo", "delete_file"):
            self.assertIn(name, menu)

    def test_fetch_rejects_non_public_urls(self) -> None:
        self.assertTrue(reject_url("file:///etc/passwd"))
        self.assertTrue(reject_url("http://169.254.169.254/latest"))
        self.assertFalse(reject_url("https://example.com/docs"))

    def test_search_results_and_page_text(self) -> None:
        html = (
            '<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">Docs</a>'
            '<div class="result__snippet">The <b>docs</b> page.</div>'
        )
        hits = parse_search_results(html)
        self.assertEqual(hits[0][1], "https://example.com/docs")
        self.assertIn("docs", hits[0][2])
        page = html_to_text("<style>x</style><p>Hello <b>there</b></p>")
        self.assertIn("Hello", page)
        self.assertNotIn("<b>", page)

    def test_a_second_thought_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = FakeLLM(json.dumps({"thought": "Rename the package, then run the tests."}))
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["think", "think"]),
                llm=llm,
                ui=ui,
                settings=settings(root, max_steps=2),
            )
            agent.run_task("rename the package")
            self.assertEqual(llm.meter.llm_calls, 2)
            self.assertEqual(len(llm.plan_users), 1)
            self.assertTrue(any(event == ("trace", "skipped a second thought") for event in ui.events))


class MultiFileWriteTests(unittest.TestCase):
    def test_writer_prompt_fills_one_file_when_several_remain(self) -> None:
        self.assertIn("fill the single next file", ARG_SYSTEM)
        self.assertIn('Do not return {"need":"ask_user"} because other files remain', ARG_SYSTEM)
        self.assertIn("one file per call", argument_system(""))

    def test_several_new_files_are_written_one_per_turn(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = ToolReplyLLM(
                {
                    "write_file": [
                        json.dumps({"path": "left.txt", "content": "left\n"}),
                        json.dumps({"path": "right.txt", "content": "right\n"}),
                    ]
                }
            )
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["write_file", "write_file", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("add left.txt and right.txt")
            self.assertEqual(summary, "Added the files")
            self.assertEqual((root / "left.txt").read_text(encoding="utf-8"), "left\n")
            self.assertEqual((root / "right.txt").read_text(encoding="utf-8"), "right\n")
            writes = [user for user in llm.users if "Selected tool: write_file" in user]
            self.assertEqual(len(writes), 2)
            self.assertIn("only one path", writes[0])
            self.assertNotIn("could not fill", " ".join(str(event) for event in ui.events))

    def test_claude_and_agents_files_are_created_without_asking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "README.md").write_text("# demo\n\nTests: python3 -m unittest\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {
                    "read_file": [json.dumps({"path": "README.md"})],
                    "write_file": [
                        json.dumps({"path": "CLAUDE.md", "content": "# Claude\n"}),
                        json.dumps({"path": "AGENTS.md", "content": "# Agents\n"}),
                    ],
                }
            )
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["read_file", "write_file", "write_file", "done"]),
                llm=llm,
                ui=ui,
                settings=settings(root),
            )
            summary = agent.run_task("add claude.md and agents.md files")
            self.assertEqual(summary, "Added the files")
            self.assertEqual((root / "CLAUDE.md").read_text(encoding="utf-8"), "# Claude\n")
            self.assertEqual((root / "AGENTS.md").read_text(encoding="utf-8"), "# Agents\n")
            self.assertNotIn("ask_user", [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "decision"])

    def test_identical_write_refusal_stops(self) -> None:
        reason = (
            "The selected write_file tool accepts only one path, "
            "but the request requires creating both alpha.txt and beta.txt."
        )
        refusal = json.dumps({"need": "ask_user", "path": "", "reason": reason})
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=FakeDecisions(["write_file"] * 8),
                llm=FakeLLM(refusal),
                ui=ui,
                settings=settings(root, max_steps=8),
            )
            summary = agent.run_task("create alpha.txt and beta.txt")
            self.assertIn("repeated the same action", summary)
            decisions = [event for event in ui.events if isinstance(event, tuple) and event[0] == "decision"]
            self.assertEqual(len(decisions), 4)
            self.assertFalse((root / "alpha.txt").exists())
            self.assertFalse((root / "beta.txt").exists())


class CostTests(unittest.TestCase):
    def test_writer_rates_use_the_model_then_the_provider(self) -> None:
        self.assertEqual(writer_rates("openai", "gpt-6-astra"), (10.0, 50.0))
        self.assertEqual(writer_rates("openai", "gpt-6-luna"), (0.10, 0.50))
        self.assertEqual(writer_rates("anthropic", "claude-sonnet-5-5"), (2.0, 10.0))
        self.assertEqual(writer_rates("openai", "gpt-4.1-mini"), (10.0, 50.0))
        self.assertEqual(writer_rates("grok", "grok-custom"), (2.0, 6.0))
        self.assertEqual(writer_rates("ollama", "qwen2.5-coder"), (0.0, 0.0))

    def test_footer_shows_session_jev_and_writer_cost(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings_obj = settings(root)
            settings_obj.provider = "openai"
            settings_obj.model = "gpt-6-luna"
            settings_obj.effort = "high"
            agent = SimpleNamespace(llm=_StubLLM(), prior=[], workspace=Workspace(root))
            app = App(settings_obj, agent, FakeUI(), None)
            meter = agent.llm.meter
            meter.jev_input_tokens = 1_000_000
            meter.llm_input_tokens = 2_000_000
            meter.llm_output_tokens = 1_000_000
            text = _toolbar(app).value
            self.assertIn("gpt-6-luna (high)", text)
            self.assertIn("jev $0.0420", text)
            self.assertIn("writer $0.7000", text)
            meter.jev_input_tokens += 1_000_000
            later = _toolbar(app).value
            self.assertIn("jev $0.0840", later)
            self.assertIn("writer $0.7000", later)
            settings_obj.provider = "anthropic"
            settings_obj.model = "claude-sonnet-5-5"
            priced = _toolbar(app).value
            self.assertIn("claude-sonnet-5-5 (high)", priced)
            self.assertIn("writer $14.0000", priced)


class GitTaskTests(unittest.TestCase):
    def test_status_paths_and_secret_staging(self) -> None:
        self.assertEqual(
            parse_porcelain(' M README.md\n?? "my file.txt"\nR  old.py -> new.py\n'),
            ["README.md", "my file.txt", "old.py", "new.py"],
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git_init(root)
            (root / "notes.txt").write_text("notes\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            self.assertEqual(secret_staging("git add -A", root), [".env"])
            self.assertEqual(secret_staging("git add .", root), [".env"])
            self.assertEqual(secret_staging("git add .env", root), [".env"])
            self.assertEqual(secret_staging("git add notes.txt && git commit -m x", root), [])
            self.assertEqual(secret_staging("git status", root), [])
            self.assertEqual(secret_staging("ls", root), [])

    def test_hooks_cannot_be_skipped_silently(self) -> None:
        self.assertEqual(classify_shell("git commit --no-verify -m x").level, "confirm")
        self.assertEqual(classify_shell("git commit -n -m x").level, "confirm")
        self.assertEqual(classify_shell("git push").level, "confirm")
        self.assertEqual(classify_shell("git commit -m 'ship it'").level, "allow")

    def test_commit_runs_through_the_plan_and_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git_init(root)
            (root / "README.md").write_text("hello\nshipped\n", encoding="utf-8")
            (root / "notes.txt").write_text("ship the notes\n", encoding="utf-8")
            (root / "my file.txt").write_text("spaced\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            plan = json.dumps(
                {
                    "task": "Stage the changed files except .env and commit them",
                    "steps": ["git add the changed files except .env", "git commit with a subject"],
                    "done_when": "git log shows the new commit",
                }
            )
            llm = ToolReplyLLM(
                {
                    "shell": [
                        json.dumps({"command": "git add -- README.md notes.txt 'my file.txt'"}),
                        json.dumps({"command": "git commit -m 'Ship the notes'"}),
                    ]
                },
                summary="Committed the notes.",
                plan=plan,
            )
            decisions = FakeDecisions(["shell", "shell", "done"])
            ui = FakeUI()
            agent = Agent(Workspace(root), decisions, llm, ui, settings(root))
            with patch.dict(os.environ, _GIT_ENV):
                summary = agent.run_task("commit changes")
            self.assertEqual(summary, "Committed the notes.")
            self.assertEqual(_git(root, ["log", "-1", "--format=%s"]), "Ship the notes")
            names = _git(root, ["show", "--name-only", "--pretty=format:", "HEAD"])
            for name in ("README.md", "notes.txt", "my file.txt"):
                self.assertIn(name, names)
            self.assertNotIn(".env", names)
            self.assertIn(("plan", "Stage the changed files except .env and commit them", json.loads(plan)["steps"]), ui.events)
            self.assertEqual(decisions.states[0]["plan"], json.loads(plan)["steps"])
            self.assertEqual(decisions.states[0]["done_when"], "git log shows the new commit")
            self.assertIn("M README.md", decisions.states[0]["workspace_snapshot"])
            self.assertNotIn("TOKEN=secret", agent.last_view.snapshot)
            self.assertIn("Plan:\n1. git add", llm.users[0])

    def test_staging_a_secret_needs_a_person(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git_init(root)
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            ui = FakeUI()
            agent = Agent(
                Workspace(root),
                FakeDecisions(["shell"]),
                FakeLLM(json.dumps({"command": "git add -A"})),
                ui,
                settings(root, max_steps=1),
            )
            agent.run_task("stage everything")
            confirms = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "confirm"]
            self.assertEqual(len(confirms), 1)
            self.assertIn("git add -A", confirms[0])
            self.assertEqual(_git(root, ["diff", "--cached", "--name-only"]), "")

    def test_push_is_not_held_when_jev_is_unsure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git_init(root)
            ui = FakeUI()
            llm = FakeLLM(json.dumps({"command": "git push"}))
            agent = Agent(
                Workspace(root),
                FakeDecisions(["shell"], confidence=0.3),
                llm,
                ui,
                settings(root, max_steps=1),
            )
            agent.run_task("push changes")
            confirms = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "confirm"]
            self.assertEqual(confirms, ["        run shell?\n        git push"])
            self.assertNotIn("skipped a low-confidence change", " ".join(str(event) for event in ui.events))
            self.assertEqual(agent.last_view.observations[-1].summary, "denied")

    def test_a_failing_hook_reaches_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git_init(root)
            hook = root / ".git" / "hooks" / "pre-commit"
            hook.write_text("#!/bin/sh\necho hook refused\nexit 1\n", encoding="utf-8")
            hook.chmod(0o755)
            (root / "notes.txt").write_text("blocked by the hook\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {
                    "shell": [
                        json.dumps({"command": "git add notes.txt"}),
                        json.dumps({"command": "git commit -m 'Should not land'"}),
                    ]
                }
            )
            agent = Agent(Workspace(root), FakeDecisions(["shell", "shell"]), llm, FakeUI(), settings(root, max_steps=2))
            with patch.dict(os.environ, _GIT_ENV):
                agent.run_task("commit changes")
            self.assertEqual(_git(root, ["log", "-1", "--format=%s"]), "init")
            last = agent.last_view.observations[-1]
            self.assertFalse(last.ok)
            self.assertIn("hook refused", last.detail)

    def test_policy_blocks_stop_instead_of_spinning(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ui = FakeUI()
            agent = Agent(
                workspace=Workspace(root),
                decisions=BlockingDecisions(["shell", "shell", "shell", "shell"]),
                llm=FakeLLM(json.dumps({"command": "ls"})),
                ui=ui,
                settings=settings(root, max_steps=6),
            )
            with patch("rift.tools.subprocess.run", side_effect=AssertionError("ran")):
                summary = agent.run_task("list the workspace")
            self.assertIn("repeated the same action", summary)
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, [])
            gates = [event for event in ui.events if isinstance(event, tuple) and event[0] == "gate"]
            self.assertEqual(len(gates), 1)

    def test_snapshot_is_visible_before_any_tool(self) -> None:
        view = View(
            goal="review the changes",
            constraints=[],
            workspace=".",
            tree="",
            project_instructions="",
            snapshot="branch: main\nstatus:\n M README.md",
        )
        self.assertIn("M README.md", llm_user_message(view, "done", "summary"))
        self.assertIn("M README.md", jev_state(view)["workspace_snapshot"])

    def test_a_successful_command_counts_as_work(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            verdict = CompletionDecision(0.95, 0.9)
            ok, _reason = prove_completion(verdict, [], Workspace(Path(raw)))
            self.assertFalse(ok)
            ok, _reason = prove_completion(verdict, [], Workspace(Path(raw)), ran_command=True)
            self.assertTrue(ok)


class PlanTests(unittest.TestCase):
    def test_follow_ups_are_resolved_from_earlier_requests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = FakeLLM("{}", plan=json.dumps({"task": "stage and commit", "steps": [], "done_when": ""}))
            agent = Agent(Workspace(root), FakeDecisions(["ask_user", "ask_user"]), llm, FakeUI(), settings(root))
            agent.run_task("stage and commit")
            agent.run_task("try agaon")
            self.assertEqual(agent.last_view.request, "try agaon")
            self.assertEqual(agent.last_view.goal, "stage and commit")
            self.assertIn("Request: stage and commit", llm.plan_users[1])

    def test_an_unreadable_plan_keeps_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = FakeLLM("{}", plan="I would read the file first.")
            agent = Agent(Workspace(root), FakeDecisions(["ask_user"]), llm, FakeUI(), settings(root))
            agent.run_task("add a retry helper")
            self.assertEqual(agent.last_view.goal, "add a retry helper")
            self.assertEqual(agent.last_view.plan, [])

    def test_an_empty_reasoning_reply_is_retried(self) -> None:
        reasoning_only = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": [{"type": "reasoning", "text": "thinking"}]},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 200},
        }

        def reply(text: str) -> dict:
            return {
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = LLM(Profile("openai", "gpt-6-luna", "test", "http://example.invalid/v1"), None, UsageMeter())
            calls: list[dict] = []

            def fake_post(url, headers, payload):
                calls.append(dict(payload))
                system = payload["messages"][0]["content"]
                if system == PLAN_SYSTEM:
                    return reply(json.dumps({"task": "list files", "steps": ["run ls"], "done_when": "ls exits 0"}))
                if system == SUMMARY_SYSTEM:
                    return reply("Listed the files.")
                arg_calls = [call for call in calls if call["messages"][0]["content"] not in {PLAN_SYSTEM, SUMMARY_SYSTEM}]
                if len(arg_calls) == 1:
                    return reasoning_only
                return reply(json.dumps({"command": "ls"}))

            llm._post = fake_post
            ui = FakeUI()
            agent = Agent(Workspace(root), FakeDecisions(["shell", "done"]), llm, ui, settings(root))
            try:
                summary = agent.run_task("list files")
            finally:
                llm.close()
            self.assertEqual(summary, "Listed the files.")
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, ["shell"])
            arg_calls = [call for call in calls if call["messages"][0]["content"] not in {PLAN_SYSTEM, SUMMARY_SYSTEM}]
            self.assertEqual(arg_calls[1].get("reasoning_effort"), "low")
            self.assertEqual(llm.effort, "medium")


class OpenAITextTests(unittest.TestCase):
    def test_visible_text_is_recovered_when_content_is_reasoning(self) -> None:
        text, truncated = _openai_text(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"type": "reasoning", "text": "hidden"},
                                {"type": "output_text", "text": "Ship the notes"},
                            ]
                        },
                    }
                ]
            }
        )
        self.assertEqual(text, "Ship the notes")
        self.assertFalse(truncated)
        text, _truncated = _openai_text(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None, "refusal": None},
                    }
                ],
                "output_text": "From output_text",
            }
        )
        self.assertEqual(text, "From output_text")
        text, _truncated = _openai_text(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "refusal": "Cannot write that subject"},
                    }
                ]
            }
        )
        self.assertEqual(text, "Cannot write that subject")
        with self.assertRaises(GenerationError) as raised:
            _openai_text(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": [{"type": "reasoning", "text": "only thinking"}]},
                        }
                    ]
                }
            )
        self.assertEqual(str(raised.exception), "model returned no text")


class ActionCoverageTests(unittest.TestCase):
    """One test per kind of request a user types, run end to end against a test project."""

    def test_commit_and_push(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, remote = _project(Path(raw), with_remote=True)
            (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {
                    "shell": [
                        json.dumps({"command": "git add calc.py"}),
                        json.dumps({"command": "git commit -m 'Add calc'"}),
                        json.dumps({"command": "git push origin HEAD"}),
                    ]
                },
                summary="Committed and pushed calc.py.",
                plan=_plan("Commit calc.py and push it", ["git add calc.py", "git commit", "git push"], "origin has the commit"),
            )
            ui = ApprovingUI()
            agent = Agent(Workspace(root), FakeDecisions(["shell", "shell", "shell", "done"]), llm, ui, settings(root, max_steps=5))
            with patch.dict(os.environ, _GIT_ENV):
                summary = agent.run_task("commit and push")
            self.assertEqual(summary, "Committed and pushed calc.py.")
            self.assertEqual(_git(root, ["log", "-1", "--format=%s"]), "Add calc")
            self.assertEqual(_git(remote, ["log", "-1", "--format=%s"]), "Add calc")
            confirms = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "confirm"]
            self.assertEqual(len(confirms), 1)
            self.assertIn("git push origin HEAD", confirms[0])

    def test_stage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "notes.txt").write_text("notes\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {"shell": [json.dumps({"command": "git add notes.txt"})]},
                summary="Staged notes.txt.",
                plan=_plan("Stage notes.txt", ["git add notes.txt"], "git diff --cached lists notes.txt"),
            )
            agent = Agent(Workspace(root), FakeDecisions(["shell", "done"]), llm, FakeUI(), settings(root))
            summary = agent.run_task("stage")
            self.assertEqual(summary, "Staged notes.txt.")
            self.assertEqual(_git(root, ["diff", "--cached", "--name-only"]), "notes.txt")
            self.assertEqual(_git(root, ["log", "-1", "--format=%s"]), "init")

    def test_update_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {
                    "read_file": [json.dumps({"path": "calc.py"})],
                    "edit_file": [
                        json.dumps(
                            {
                                "path": "calc.py",
                                "old_string": "    return a + b",
                                "new_string": "    return int(a) + int(b)",
                            }
                        )
                    ],
                },
                summary="Coerced the arguments in calc.py.",
                plan=_plan("Coerce the arguments in add", ["read calc.py", "edit calc.py"], "calc.py calls int()"),
            )
            ui = FakeUI()
            agent = Agent(Workspace(root), FakeDecisions(["read_file", "edit_file", "done"]), llm, ui, settings(root))
            summary = agent.run_task("update calc.py so add coerces its arguments to int")
            self.assertEqual(summary, "Coerced the arguments in calc.py.")
            self.assertIn("int(a) + int(b)", (root / "calc.py").read_text(encoding="utf-8"))
            self.assertEqual(("stats", ["calc.py"]), next(e for e in ui.events if isinstance(e, tuple) and e[0] == "stats"))

    def test_replace_all_instances(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "app.py").write_text("print('alpha')\n", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "alpha.md").write_text("see alpha\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {"replace_text": [json.dumps({"old": "alpha", "new": "beta"})]},
                summary="Renamed alpha to beta.",
                plan=_plan("Replace alpha with beta everywhere", ["replace_text alpha -> beta"], "no file mentions alpha"),
            )
            agent = Agent(Workspace(root), FakeDecisions(["replace_text", "done"]), llm, FakeUI(), settings(root))
            summary = agent.run_task("replace all instances of alpha with beta")
            self.assertEqual(summary, "Renamed alpha to beta.")
            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print('beta')\n")
            self.assertTrue((root / "docs" / "beta.md").is_file())
            self.assertFalse((root / "docs" / "alpha.md").exists())

    def test_bang_git_diff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "README.md").write_text("hello\nshipped\n", encoding="utf-8")
            buf = StringIO()
            ui = UI(console=Console(file=buf, force_terminal=False, width=80))
            agent = SimpleNamespace(prior=[], workspace=Workspace(root))
            app = App(settings(root), agent, ui, None)
            run_bang(app, "! git diff")
            self.assertIn("+shipped", buf.getvalue())
            self.assertEqual(len(agent.prior), 1)
            self.assertIn("User ran `git diff`", agent.prior[0])
            self.assertIn("+shipped", agent.prior[0])
            self.assertEqual(_git(root, ["status", "--porcelain"]), "M README.md")

    def test_add_tests_for_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            test_body = (
                "import unittest\n\nfrom calc import add\n\n\n"
                "class AddTest(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
            )
            llm = ToolReplyLLM(
                {
                    "read_file": [json.dumps({"path": "calc.py"})],
                    "write_file": [json.dumps({"path": "test_calc.py", "content": test_body})],
                    "shell": [json.dumps({"command": "python3 -m unittest test_calc -v"})],
                },
                summary="Added test_calc.py and the suite passes.",
                plan=_plan(
                    "Add a unittest for calc.py",
                    ["read calc.py", "write test_calc.py", "run python3 -m unittest test_calc"],
                    "the test command exits 0",
                ),
            )
            agent = Agent(
                Workspace(root),
                FakeDecisions(["read_file", "write_file", "shell", "done"]),
                llm,
                FakeUI(),
                settings(root, max_steps=5),
            )
            summary = agent.run_task("add tests for calc.py")
            self.assertEqual(summary, "Added test_calc.py and the suite passes.")
            self.assertIn("assertEqual(add(1, 2), 3)", (root / "test_calc.py").read_text(encoding="utf-8"))
            ran = [item for item in agent.last_view.observations if item.tool == "shell"]
            self.assertTrue(ran[-1].ok)
            self.assertIn("OK", ran[-1].detail)

    def test_run_all_tests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "test_ok.py").write_text(
                "import unittest\n\n\nclass OkTest(unittest.TestCase):\n"
                "    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            llm = ToolReplyLLM(
                {"shell": [json.dumps({"command": "python3 -m unittest discover -p 'test_*.py'"})]},
                summary="The suite passes.",
                plan=_plan("Run the test suite", ["python3 -m unittest discover"], "the command exits 0"),
            )
            ui = FakeUI()
            agent = Agent(Workspace(root), FakeDecisions(["shell", "done"]), llm, ui, settings(root))
            summary = agent.run_task("run all tests")
            self.assertEqual(summary, "The suite passes.")
            tools = [event[1] for event in ui.events if isinstance(event, tuple) and event[0] == "tool"]
            self.assertEqual(tools, ["shell"])
            self.assertTrue(agent.last_view.observations[-1].ok)
            self.assertEqual(agent.last_view.files_changed, [])

    def test_create_a_new_node_project(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            package = json.dumps(
                {"name": "demo", "version": "1.0.0", "main": "index.js", "scripts": {"start": "node index.js"}},
                indent=2,
            )
            llm = ToolReplyLLM(
                {
                    "write_file": [
                        json.dumps({"path": "package.json", "content": package + "\n"}),
                        json.dumps({"path": "index.js", "content": "console.log('demo');\n"}),
                    ],
                    "shell": [json.dumps({"command": "node index.js"})],
                },
                summary="Created package.json and index.js.",
                plan=_plan(
                    "Create a minimal node project",
                    ["write package.json", "write index.js", "run node index.js"],
                    "node index.js prints demo",
                ),
            )
            agent = Agent(
                Workspace(root),
                FakeDecisions(["write_file", "write_file", "shell", "done"]),
                llm,
                FakeUI(),
                settings(root, max_steps=5),
            )
            summary = agent.run_task("create a new node project")
            self.assertEqual(summary, "Created package.json and index.js.")
            self.assertEqual(json.loads((root / "package.json").read_text(encoding="utf-8"))["main"], "index.js")
            self.assertIn("console.log", (root / "index.js").read_text(encoding="utf-8"))
            self.assertEqual(agent.last_view.files_changed, ["package.json", "index.js"])
            ran = agent.last_view.observations[-1]
            self.assertEqual(ran.tool, "shell")
            if shutil.which("node"):
                self.assertTrue(ran.ok)
                self.assertIn("demo", ran.detail)

    def test_a_refusal_points_the_loop_at_done(self) -> None:
        """The work already landed, so the writer has no edit to make. That is not a dead end."""
        refusal = json.dumps(
            {"need": "ask_user", "path": "", "reason": "the rename already happened; there is no edit to make"}
        )
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            (root / "calc.py").write_text("def multiply(a, b):\n    return a * b\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {"edit_batch": [refusal]},
                summary="The rename was already complete.",
                plan=_plan("Rename mul to multiply", ["replace mul with multiply"], "no file mentions mul"),
            )
            decisions = AnsweredDecisions(["edit_batch", "done"])
            ui = FakeUI()
            agent = Agent(Workspace(root), decisions, llm, ui, settings(root))
            summary = agent.run_task("replace all instances of mul with multiply")
            self.assertEqual(summary, "The rename was already complete.")
            guidance = decisions.states[-1]["guidance"]
            self.assertIn("could not fill edit_batch", guidance)
            self.assertIn("Do not pick edit_batch again", guidance)
            self.assertIn("pick done", guidance)
            self.assertNotIn("repeated the same action", summary)

    def test_a_repeat_is_told_to_check_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            llm = FakeLLM(
                json.dumps({"path": "README.md"}),
                summary="Read the readme.",
                plan=_plan("Read README.md", ["read README.md"], "the file is loaded"),
            )
            decisions = AnsweredDecisions(["read_file", "read_file", "done"])
            agent = Agent(Workspace(root), decisions, llm, FakeUI(), settings(root))
            summary = agent.run_task("show me the readme")
            self.assertEqual(summary, "Read the readme.")
            guidance = decisions.states[-1]["guidance"]
            self.assertIn("already ran", guidance)
            self.assertIn("pick done", guidance)

    def test_every_request_is_planned_before_a_tool_runs(self) -> None:
        requests = ["commit and push", "stage", "replace all instances of x with y", "run all tests"]
        with tempfile.TemporaryDirectory() as raw:
            root, _remote = _project(Path(raw))
            for request in requests:
                with self.subTest(request=request):
                    llm = FakeLLM(
                        json.dumps({"command": "git status --porcelain"}),
                        summary="Reported the status.",
                        plan=_plan(request, ["git status"], "the command exits 0"),
                    )
                    decisions = FakeDecisions(["shell", "done"])
                    agent = Agent(Workspace(root), decisions, llm, FakeUI(), settings(root))
                    agent.run_task(request)
                    self.assertEqual(len(llm.plan_users), 1)
                    self.assertIn(f"User request:\n{request}", llm.plan_users[0])
                    self.assertEqual(decisions.states[0]["plan"], ["git status"])
                    self.assertEqual(decisions.states[0]["goal"], request)


class RobustnessTests(unittest.TestCase):
    def test_status_spins_only_on_a_terminal(self) -> None:
        buf = StringIO()
        ui = UI(verbose=True, console=Console(file=buf, force_terminal=False, width=80))
        with ui.status("planning with fake"):
            pass
        self.assertIn("planning with fake", buf.getvalue())

    def test_planning_reports_status_before_the_first_tool(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ui = FakeUI()
            agent = Agent(
                Workspace(root),
                AnsweredDecisions(["done"]),
                FakeLLM("{}", summary="Nothing to do.", plan=_plan("Look around", [], "no files change")),
                ui,
                settings(root, max_steps=2),
            )
            agent.run_task("look around")
            self.assertTrue(any(isinstance(event, str) and event.startswith("planning with ") for event in ui.events))

    def test_a_failed_command_waits_for_a_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fail = "python3 -c 'raise SystemExit(1)'"
            llm = ToolReplyLLM(
                {
                    "shell": [json.dumps({"command": fail}), json.dumps({"command": fail}), json.dumps({"command": fail})],
                    "write_file": [json.dumps({"path": "notes.txt", "content": "fixed\n"})],
                },
                summary="Updated notes and reran the check.",
                plan=_plan("Fix the check", ["run the check", "edit notes", "run the check"], "the check exits 0"),
            )
            agent = Agent(
                Workspace(root),
                FakeDecisions(["shell", "shell", "write_file", "shell", "done"]),
                llm,
                FakeUI(),
                settings(root, max_steps=6),
            )
            summary = agent.run_task("fix the failing check")
            self.assertEqual(summary, "Updated notes and reran the check.")
            shells = [item for item in agent.last_view.observations if item.tool == "shell"]
            ran = [item for item in shells if item.summary != "duplicate"]
            skipped = [item for item in shells if item.summary == "duplicate"]
            self.assertEqual(len(ran), 2)
            self.assertEqual(len(skipped), 1)
            self.assertFalse(ran[0].ok)
            self.assertEqual((root / "notes.txt").read_text(encoding="utf-8"), "fixed\n")

    def test_a_third_failing_test_command_stops(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = ToolReplyLLM(
                {
                    "shell": [
                        json.dumps({"command": "python3 -c 'raise SystemExit(\"pytest failed\")'"}),
                        json.dumps({"command": "python3 -c 'raise SystemExit(\"pytest -x failed\")'"}),
                        json.dumps({"command": "python3 -c 'raise SystemExit(\"pytest -k failed\")'"}),
                    ]
                }
            )
            agent = Agent(
                Workspace(root),
                FakeDecisions(["shell", "shell", "shell", "shell"]),
                llm,
                FakeUI(),
                settings(root, max_steps=5),
            )
            summary = agent.run_task("run the tests")
            self.assertIn("repeated the same action", summary)
            ran = [
                item
                for item in agent.last_view.observations
                if item.tool == "shell" and item.summary != "duplicate"
            ]
            self.assertEqual(len(ran), 2)
            self.assertTrue(all(not item.ok for item in ran))

    def test_distinct_blocked_commands_still_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = ToolReplyLLM(
                {"shell": [json.dumps({"command": "ls"}), json.dumps({"command": "pwd"})]}
            )
            agent = Agent(
                Workspace(root),
                BlockingDecisions(["shell", "shell", "shell"]),
                llm,
                FakeUI(),
                settings(root, max_steps=4),
            )
            with patch("rift.tools.subprocess.run", side_effect=AssertionError("ran")):
                summary = agent.run_task("list the workspace")
            self.assertIn("policy blocked an action twice", summary)

    def test_a_missing_read_points_back_at_the_file(self) -> None:
        refusal = json.dumps(
            {"need": "read_file", "path": "src/rift/ui.py", "reason": "truncated before the status method"}
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "src" / "rift").mkdir(parents=True)
            (root / "src" / "rift" / "ui.py").write_text("class UI:\n    pass\n", encoding="utf-8")
            llm = ToolReplyLLM(
                {"edit_file": [refusal]},
                summary="Nothing to change.",
                plan=_plan("Edit ui.py", ["edit ui.py"], "status exists"),
            )
            decisions = AnsweredDecisions(["edit_file", "done"])
            agent = Agent(Workspace(root), decisions, llm, FakeUI(), settings(root))
            summary = agent.run_task("add a planning loader")
            self.assertEqual(summary, "Nothing to change.")
            guidance = decisions.states[-1]["guidance"]
            self.assertIn("Pick read_file", guidance)
            self.assertIn("src/rift/ui.py", guidance)
            self.assertNotIn("Do not pick edit_file again", guidance)

    def test_loaded_text_keeps_every_region(self) -> None:
        view = View(goal="edit", constraints=[], workspace=".", tree="", project_instructions="")
        view.remember_file("ui.py", "class UI:\n    def status(self):\n        pass\n", note="lines 1-3 of 3")
        view.remember_file("ui.py", "def status(self):", note="lines 2-2 of 3")
        self.assertIn("class UI:", view.loaded["ui.py"])
        self.assertEqual(view.loaded_notes["ui.py"], "lines 1-3 of 3")
        view.remember_file("ui.py", "tail of the file")
        self.assertIn("class UI:", view.loaded["ui.py"])
        self.assertIn("tail of the file", view.loaded["ui.py"])
        view.remember_file("ui.py", "just the new file\n", replace=True)
        self.assertEqual(view.loaded["ui.py"], "just the new file\n")
        self.assertNotIn("ui.py", view.loaded_notes)

    def test_a_small_file_survives_beside_a_large_one(self) -> None:
        view = View(goal="edit ui", constraints=[], workspace=".", tree="", project_instructions="")
        view.remember_file("big.py", "B" * 60_000, note="lines 1-1 of 1")
        view.remember_file("ui.py", "def status(self, text: str):\n    return spinner\n", note="lines 1-2 of 2")
        text = llm_user_message(view, "edit_file", "path, old_string, new_string")
        self.assertIn("def status(self, text: str):", text)
        self.assertIn("lines 1-2 of 2", text)


class ApprovingUI(FakeUI):
    def confirm(self, message: str, forced: bool) -> bool:
        self.events.append(("confirm", message))
        return True


def _plan(task: str, steps: list[str], done_when: str) -> str:
    return json.dumps({"task": task, "steps": steps, "done_when": done_when})


def _project(tmp: Path, with_remote: bool = False) -> tuple[Path, Path]:
    """A git repo with one commit, and optionally a bare remote it can push to."""
    root = tmp / "work"
    root.mkdir()
    _git_init(root)
    remote = tmp / "remote.git"
    if with_remote:
        env = os.environ.copy()
        env.update(_GIT_ENV)
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, env=env)
        _git(root, ["remote", "add", "origin", str(remote)])
        _git(root, ["push", "-u", "origin", "HEAD"])
    return root, remote


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git_init(root: Path) -> None:
    env = os.environ.copy()
    env.update(_GIT_ENV)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, env=env)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True, capture_output=True, env=env)
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, env=env)


def _git(root: Path, args: list[str]) -> str:
    env = os.environ.copy()
    env.update(_GIT_ENV)
    completed = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env)
    return completed.stdout.strip()


class _StubLLM:
    def __init__(self) -> None:
        self.meter = UsageMeter()
        self.effort = "medium"

    def close(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
