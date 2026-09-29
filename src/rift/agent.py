"""The agent loop.

Jev chooses and gates. The language model only writes arguments and the final note.
Code owns limits, path sandboxing, and the proof that a finished task really landed on disk.
"""

from __future__ import annotations

import json

from rift.decisions import (
    ActionDecision,
    CompletionDecision,
    DecisionError,
    GateDecision,
)
from rift.llm import GenerationError
from rift.prompt import argument_system
from rift.safety import Verdict, allow, block, classify_path, classify_shell, confirm
from rift.state import (
    ASK_SYSTEM,
    BASE_CONSTRAINTS,
    SUMMARY_SYSTEM,
    Observation,
    View,
    jev_state,
    llm_user_message,
    state_is_large,
)
from rift.tools import SPECS, ToolError, ToolResult, Workspace, action_menu, project_instructions, unified_diff
from rift.util import clip, extract_json
from rift.web import web_fetch, web_search

MUTATING = {"edit_file", "edit_batch", "replace_text", "write_file", "delete_file", "shell"}


class Agent:
    def __init__(self, workspace: Workspace, decisions, llm, ui, settings) -> None:
        self.workspace = workspace
        self.decisions = decisions
        self.llm = llm
        self.ui = ui
        self.settings = settings
        self.prior: list[str] = []
        self._held_edits = 0

    def run_task(self, goal: str) -> str:
        started = self.llm.meter.snapshot()
        view = View(
            goal=goal.strip(),
            constraints=[*BASE_CONSTRAINTS, *self.settings.constraints],
            workspace=str(self.workspace.root),
            tree=self.workspace.tree(),
            project_instructions=project_instructions(self.workspace.root),
            prior_tasks=list(self.prior),
        )
        files_changed: list[str] = []
        seen: dict[str, int] = {}
        done_attempts = 0
        stuck_strikes = 0
        executed = 0
        summary = ""
        try:
            for step in range(1, self.settings.max_steps + 1):
                if self._llm_calls_since(started) >= self.settings.max_llm_calls:
                    summary = self._stop("generation call limit reached", view)
                    break
                view.tree = self.workspace.tree()
                self._compact(view)
                state = jev_state(view)
                if stuck_strikes >= 2:
                    action = ActionDecision("ask_user", 1.0, {}, "powerful")
                    stuck_strikes = 0
                else:
                    try:
                        action = self.decisions.next_action(
                            state,
                            action_menu(self.settings.read_only),
                            route=self._route_enabled(),
                        )
                    except DecisionError as error:
                        self.ui.error(str(error))
                        summary = self._stop(str(error), view)
                        break
                if wants_test_run(view.goal) and not _ran_shell(view):
                    if self.settings.read_only:
                        summary = self._stop("running tests needs shell, and this session is read-only", view)
                        break
                    if action.name != "shell":
                        action = ActionDecision("shell", 0.95, {"shell": 0.95}, action.tier, action.request_id)
                if wants_repo_replace(view.goal) and not _ran_replace(view):
                    if self.settings.read_only:
                        summary = self._stop("a repo-wide replace needs write access", view)
                        break
                    if action.name != "replace_text":
                        action = ActionDecision(
                            "replace_text", 0.95, {"replace_text": 0.95}, action.tier, action.request_id
                        )
                self.ui.decision(
                    step,
                    action.name,
                    action.confidence,
                    action.tier,
                    action.probabilities,
                    action.request_id,
                )
                if action.name == "done":
                    outcome = self._try_finish(view, state, files_changed)
                    if outcome is not None:
                        summary = outcome
                        break
                    done_attempts += 1
                    if done_attempts >= 3:
                        summary = self._stop("completion was rejected three times", view)
                        break
                    continue
                if action.name == "ask_user":
                    if not self._ask(view, action.tier):
                        summary = self._stop(view.observations[-1].detail or "stopped to ask the user", view)
                        break
                    continue
                if action.name == "think" and view.observations and view.observations[-1].tool == "think":
                    view.guidance = "A thought is already recorded. Choose a tool."
                    view.observations.append(Observation("think", "", "skipped", view.guidance, False))
                    self.ui.trace("skipped a second thought")
                    continue
                uncertain_edit = action.name in MUTATING and action.confidence < 0.6
                mechanical = action.name in {"edit_batch", "replace_text"} or wants_repo_replace(view.goal)
                if (
                    not mechanical
                    and uncertain_edit
                    and (action.confidence < 0.45 or not _spec_is_loaded(view))
                ):
                    fallback = None if _spec_is_loaded(view) else fallback_action(action)
                    if fallback is not None:
                        self.ui.trace(
                            f"held {action.name} at {action.confidence:.2f}; using {fallback.name}"
                        )
                        action = fallback
                    else:
                        ranked = _ranked(action.probabilities)
                        view.guidance = (
                            "Do not edit yet. Read the test file or run the tests first. "
                            f"The last uncertain choice was {action.name} at {action.confidence:.2f} ({ranked})."
                        )
                        view.observations.append(
                            Observation(action.name, "", "skipped", view.guidance, False)
                        )
                        self.ui.trace("skipped a low-confidence change")
                        self._held_edits += 1
                        if self._held_edits >= 4:
                            summary = self._stop("held an uncertain edit four times", view)
                            break
                        continue
                args, refusal = self._arguments(view, action)
                if refusal is not None:
                    view.observations.append(refusal)
                    self.ui.info(f"        {refusal.summary}: {clip(refusal.detail, 200)}")
                    continue
                assert args is not None
                signature = action.name + ":" + json.dumps(args, sort_keys=True, default=str)[:800]
                if action.name != "shell":
                    seen[signature] = seen.get(signature, 0) + 1
                    if seen[signature] > 1:
                        view.observations.append(
                            Observation(
                                action.name,
                                clip(signature, 200),
                                "duplicate",
                                "This exact action already ran. Choose a different one.",
                                False,
                            )
                        )
                        self.ui.trace("skipped a repeated action")
                        if seen[signature] >= 4:
                            summary = self._stop("repeated the same action", view)
                            break
                        continue
                if not self._permit(view, state, action.name, args):
                    continue
                result = self._execute(action.name, args)
                preview = json.dumps(_preview_args(args), default=str)
                view.observations.append(
                    Observation(action.name, clip(preview, 500), result.summary, result.detail, result.ok)
                )
                self.ui.tool(action.name, result.summary, result.detail, result.paths)
                if result.ok and action.name == "think":
                    view.guidance = result.detail
                if result.ok and action.name == "todo":
                    view.todos = list(args["todos"])
                if result.ok and action.name == "delete_file":
                    rel = self._display(str(args.get("path", "")))
                    if rel:
                        marker = f"deleted {rel}"
                        if marker not in files_changed:
                            files_changed.append(marker)
                    view.files_changed = list(files_changed)
                if result.ok and action.name in {"edit_file", "write_file", "edit_batch", "replace_text"}:
                    paths = list(result.paths)
                    if not paths and args.get("path"):
                        paths = [self._display(str(args.get("path", "")))]
                    for rel in paths:
                        if rel and rel not in files_changed:
                            files_changed.append(rel)
                        if rel:
                            self._refresh_loaded(view, rel)
                    view.files_changed = list(files_changed)
                if result.ok and action.name == "read_file":
                    rel = self._display(str(args.get("path", "")))
                    if rel:
                        view.remember_file(rel, result.detail)
                if result.ok:
                    self._held_edits = 0
                executed += 1
                if result.ok and executed % 4 == 0:
                    stuck_strikes = self._check_progress(view, stuck_strikes)
            else:
                summary = self._stop("step limit reached", view)
        except GenerationError as error:
            self.ui.error(str(error))
            summary = self._stop(str(error), view)
        except KeyboardInterrupt:
            summary = self._stop("interrupted", view)
        self.ui.stats(self.llm.meter.delta(started), files_changed)
        if summary:
            note = clip(summary, 500)
            if note not in self.prior:
                self.prior.append(note)
            self.prior = self.prior[-6:]
        return summary

    def _try_finish(self, view: View, state: dict, files_changed: list[str]) -> str | None:
        check = dict(state)
        check["files_changed"] = files_changed
        try:
            verdict = self.decisions.completion(check)
        except DecisionError as error:
            view.observations.append(Observation("done", "", "completion check failed", str(error), False))
            self.ui.error(str(error))
            return None
        passed, reason = prove_completion(verdict, files_changed, self.workspace)
        if not passed and files_changed and latest_test_passed(view):
            passed = True
            reason = "the test command exited 0 after the last edit"
        if wants_test_run(view.goal):
            if _test_command_passed(view):
                passed = True
                reason = "the test command exited 0"
            else:
                passed = False
                reason = "the test command has not exited 0"
        self.ui.trace(
            f"complete {verdict.complete:.2f}  needs_file_changes {verdict.needs_file_changes:.2f}"
        )
        if verdict.complete <= 0.8 and passed:
            self.ui.trace(reason)
        if not passed:
            view.observations.append(Observation("done", "", "completion rejected", reason, False))
            self.ui.info(f"        rejected: {reason}")
            return None
        result = self.llm.complete(
            "powerful",
            SUMMARY_SYSTEM,
            llm_user_message(view, "done", "Plain text summary of the work."),
            800,
            0.2,
        )
        self.ui.finished(result.text)
        return result.text.strip()

    def _arguments(self, view: View, action: ActionDecision) -> tuple[dict | None, Observation | None]:
        spec = SPECS.get(action.name)
        if spec is None:
            return None, Observation(action.name, "", "unknown tool", action.name, False)
        user = llm_user_message(view, action.name, spec.arg_help)
        if action.name == "shell" and wants_test_run(view.goal):
            user += (
                "\n\nThe user asked to run the test suite. "
                "Set command to the test command in the project instructions when one is given. "
                "Otherwise use the runner this repo already uses. Use python3, not python."
            )
        if action.name == "replace_text":
            user += (
                "\n\nSet old and new from the user message. "
                "One call replaces the text in every file and in file names. Leave glob empty."
            )
        max_tokens = 16384 if action.name in {"edit_file", "write_file", "edit_batch"} else 2048
        self.ui.status(f"        writing arguments with {self._model_name(action.tier)}")
        result = self.llm.complete(action.tier, argument_system(view.project_instructions), user, max_tokens, 0.0)
        try:
            args = extract_json(result.text)
        except ValueError:
            result = self.llm.complete(
                action.tier,
                argument_system(view.project_instructions),
                user + "\n\nYour previous reply was not one JSON object. Reply with only JSON.",
                max_tokens,
                0.0,
            )
            try:
                args = extract_json(result.text)
            except ValueError as error:
                return None, Observation(action.name, "", "invalid arguments", str(error), False)
        if isinstance(args.get("need"), str) and not _has_payload(action.name, args):
            detail = f"need {args.get('need', '')} {args.get('path', '')}: {args.get('reason', '')}".strip()
            return None, Observation(action.name, "", f"could not fill {action.name}", detail, False)
        normalized, error = normalize_args(action.name, args)
        if error:
            return None, Observation(action.name, "", "invalid arguments", error, False)
        return normalized, None

    def _permit(self, view: View, state: dict, tool: str, args: dict) -> bool:
        code = self._code_verdict(tool, args)
        if code.level == "block":
            self.ui.rule("block", code.reason)
            view.observations.append(Observation(tool, _args_text(args), "blocked", code.reason, False))
            return False
        gate: GateDecision | None = None
        if tool in MUTATING:
            proposal = dict(state)
            proposal["proposed_action"] = {"tool": tool, "arguments": _preview_args(args)}
            if tool == "shell":
                excerpt = self.workspace.file_excerpt(str(args.get("command", "")))
                if excerpt:
                    proposal["proposed_action"]["file_excerpt"] = excerpt
            try:
                gate = self.decisions.gate(proposal)
            except DecisionError as error:
                self.ui.error(str(error))
                view.observations.append(Observation(tool, "", "gate failed", str(error), False))
                return False
            self.ui.gate(gate.action, gate.destructive, "")
            if gate.action == "block":
                view.observations.append(
                    Observation(tool, _args_text(args), "blocked by policy", "Jev blocked this action", False)
                )
                return False
        need_human = code.level == "confirm"
        if self.settings.confirm_mutations and tool in MUTATING:
            need_human = True
        if gate is not None and (gate.action == "confirm" or gate.destructive > 0.4):
            need_human = True
        if not need_human:
            return True
        forced = code.level == "confirm" or (self.settings.confirm_mutations and tool in MUTATING)
        if not self.ui.confirm(self._confirm_message(tool, args), forced):
            view.observations.append(Observation(tool, _args_text(args), "denied", "a person denied this action", False))
            return False
        return True

    def _execute(self, tool: str, args: dict) -> ToolResult:
        try:
            if tool == "read_file":
                return self.workspace.read_file(args["path"], args["offset"], args["limit"])
            if tool == "grep":
                return self.workspace.grep(args["pattern"], args["path"], args["glob"])
            if tool == "glob":
                return self.workspace.glob(args["pattern"], args["path"])
            if tool == "list_dir":
                return self.workspace.list_dir(args["path"])
            if tool == "edit_file":
                return self.workspace.edit_file(
                    args["path"], args["old_string"], args["new_string"], args["replace_all"], True
                )
            if tool == "edit_batch":
                return self.workspace.edit_batch(args["edits"], True)
            if tool == "replace_text":
                return self.workspace.replace_text(args["old"], args["new"], args["glob"], True)
            if tool == "write_file":
                return self.workspace.write_file(args["path"], args["content"], True)
            if tool == "delete_file":
                return self.workspace.delete_file(args["path"], True)
            if tool == "web_search":
                return web_search(args["query"])
            if tool == "web_fetch":
                return web_fetch(args["url"])
            if tool == "think":
                thought = args["thought"]
                return ToolResult(True, clip(thought, 200), thought)
            if tool == "todo":
                lines = [f"{item['status']}: {item['content']}" for item in args["todos"]]
                return ToolResult(True, f"{len(lines)} tasks", "\n".join(lines))
            if tool == "shell":
                return self.workspace.shell(args["command"], args["timeout_sec"], True)
        except ToolError as error:
            return ToolResult(False, "tool error", str(error))
        return ToolResult(False, "unknown tool", tool)

    def _code_verdict(self, tool: str, args: dict) -> Verdict:
        if tool == "shell":
            return classify_shell(str(args.get("command", "")))
        raw_path = str(args.get("path", "."))
        try:
            path = self.workspace.resolve(raw_path)
        except ToolError as error:
            return block(str(error))
        if self.workspace.allow_outside:
            try:
                path.relative_to(self.workspace.root)
            except ValueError:
                if tool in {"edit_file", "write_file", "delete_file", "shell"}:
                    return confirm("path is outside the workspace")
        if tool in {"edit_file", "write_file", "delete_file"}:
            return classify_path(path)
        if tool == "edit_batch":
            worst = allow()
            for edit in args.get("edits") or []:
                try:
                    verdict = classify_path(self.workspace.resolve(str(edit.get("path", "."))))
                except ToolError as error:
                    return block(str(error))
                if verdict.level == "block":
                    return verdict
                if verdict.level == "confirm":
                    worst = verdict
            return worst
        return allow()

    def _ask(self, view: View, tier: str) -> bool:
        try:
            result = self.llm.complete(
                tier,
                ASK_SYSTEM,
                llm_user_message(view, "ask_user", "One question."),
                400,
                0.2,
            )
            question = result.text.strip()
        except GenerationError as error:
            self.ui.error(str(error))
            question = f"What should I do next to finish this: {view.goal}"
        answer = self.ui.ask(question)
        if not answer:
            view.observations.append(Observation("ask_user", "", "no answer", question, False))
            return False
        view.observations.append(Observation("ask_user", "", "user answered", f"Q: {question}\nA: {answer}", True))
        return True

    def _check_progress(self, view: View, strikes: int) -> int:
        try:
            verdict = self.decisions.progress(jev_state(view))
        except DecisionError as error:
            self.ui.error(str(error))
            return strikes
        self.ui.trace(f"progress {verdict.progressing:.2f}  repeating {verdict.repeating:.2f}")
        if verdict.repeating > 0.7 or verdict.progressing < 0.3:
            view.guidance = (
                "The last actions did not move the goal forward. "
                "Do not repeat them. Read something new, make a narrower edit, or ask_user."
            )
            return strikes + 1
        view.guidance = ""
        return 0

    def _compact(self, view: View) -> None:
        if len(view.observations) <= 10 and not state_is_large(jev_state(view)):
            return
        if len(view.observations) <= 4:
            return
        tail = view.observations[-4:]
        head = view.observations[:-4][-12:]
        try:
            scores = self.decisions.score_chunks(view.goal, [item.summary for item in head])
        except DecisionError as error:
            self.ui.error(f"compaction skipped: {error}")
            view.observations = view.observations[-6:]
            return
        kept: list[Observation] = []
        for item, score in zip(head, scores):
            if score >= 1.5:
                kept.append(item)
            elif score >= 0.75:
                kept.append(Observation(item.tool, item.args_preview, item.summary, item.summary, item.ok))
        if not kept:
            kept = head[-2:]
        view.observations = kept + tail

    def _confirm_message(self, tool: str, args: dict) -> str:
        if tool == "shell":
            return "        run shell?\n        " + str(args.get("command", ""))
        if tool == "edit_file":
            diff = unified_diff(args["old_string"], args["new_string"], str(args["path"]), max_lines=40)
            return f"        edit {args['path']}?\n{diff}"
        if tool == "write_file":
            return f"        write {args['path']} ({len(args['content'])} chars)?"
        if tool == "replace_text":
            return f"        replace {args.get('old', '')!r} with {args.get('new', '')!r} across the workspace?"
        if tool == "edit_batch":
            return f"        apply {len(args.get('edits') or [])} edits?"
        if tool == "delete_file":
            return f"        delete {args.get('path', '')}?"
        return f"        run {tool}?"

    def _refresh_loaded(self, view: View, raw: str) -> None:
        rel = self._display(raw)
        if not rel:
            return
        try:
            text = self.workspace.read_text_if_small(raw)
        except ToolError:
            return
        if text:
            view.remember_file(rel, text)

    def _display(self, raw: str) -> str:
        try:
            return self.workspace.display(self.workspace.resolve(raw))
        except ToolError:
            return ""

    def _route_enabled(self) -> bool:
        return bool(self.settings.fast_model and self.settings.fast_provider)

    def _model_name(self, tier: str) -> str:
        if tier == "fast" and self.settings.fast_model:
            return self.settings.fast_model
        return self.settings.model

    def _llm_calls_since(self, earlier) -> int:
        return self.llm.meter.llm_calls - earlier.llm_calls

    def _stop(self, reason: str, view: View) -> str:
        last = view.observations[-1].summary if view.observations else "no actions yet"
        text = f"Stopped: {reason}. Last action: {last}."
        self.ui.error(text)
        return text


def fallback_action(action: ActionDecision) -> ActionDecision | None:
    """When an edit is too uncertain, take the best read instead of spinning."""
    best_name = ""
    best_prob = 0.0
    for name, prob in action.probabilities.items():
        if name not in {"read_file", "grep", "glob", "list_dir"} or prob < 0.15:
            continue
        if prob > best_prob:
            best_name = name
            best_prob = float(prob)
    if not best_name:
        return None
    return ActionDecision(best_name, best_prob, action.probabilities, action.tier, action.request_id)


def _spec_is_loaded(view: View) -> bool:
    for path in view.loaded:
        name = path.rsplit("/", 1)[-1]
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
    return False


def _ranked(probabilities: dict[str, float]) -> str:
    ordered = sorted(probabilities.items(), key=lambda item: -item[1])[:4]
    return ", ".join(f"{name}={prob:.2f}" for name, prob in ordered)


def wants_repo_replace(goal: str) -> bool:
    """The user asked to change one string everywhere, not to edit files one by one."""
    text = " ".join(goal.strip().lower().split())
    return "all instances" in text or "every instance" in text or "everywhere" in text


def _ran_replace(view: View) -> bool:
    return any(item.tool == "replace_text" for item in view.observations)


def wants_test_run(goal: str) -> bool:
    """A bare request to run the suite, not a request to change tests."""
    text = " ".join(goal.strip().lower().strip(".!?").split())
    return text in {
        "test",
        "tests",
        "run test",
        "run tests",
        "run the test",
        "run the tests",
        "run the test suite",
        "test suite",
        "npm test",
        "pytest",
        "run pytest",
        "run unittest",
    }


def _ran_shell(view: View) -> bool:
    return any(item.tool == "shell" for item in view.observations)


def _test_command_passed(view: View) -> bool:
    passed = False
    for item in view.observations:
        if item.tool == "shell" and _is_test_command(f"{item.args_preview} {item.summary}"):
            passed = item.ok
    return passed


def latest_test_passed(view: View) -> bool:
    """A green test run after the last edit is proof the model does not have to supply."""
    saw_edit = False
    passed = False
    for item in view.observations:
        if item.tool in {"edit_file", "write_file"} and item.ok:
            saw_edit = True
            passed = False
        if item.tool == "shell" and _is_test_command(f"{item.args_preview} {item.summary}"):
            passed = item.ok
    return saw_edit and passed


def _is_test_command(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("unittest", "pytest", "npm test", "go test", "cargo test"))


def prove_completion(verdict: CompletionDecision, files_changed: list[str], workspace: Workspace) -> tuple[bool, str]:
    """Jev's belief is not proof. The files have to exist."""
    if verdict.complete <= 0.8:
        return False, f"completion belief is {verdict.complete:.2f}; need above 0.80"
    if verdict.needs_file_changes >= 0.7 and not files_changed:
        return False, "the goal still needs file changes and none were written"
    missing: list[str] = []
    empty: list[str] = []
    for rel in files_changed:
        if rel.startswith("deleted "):
            target = rel.removeprefix("deleted ")
            if (workspace.root / target).exists():
                missing.append(target + " still exists")
            continue
        path = workspace.root / rel
        if not path.exists():
            missing.append(rel)
            continue
        if path.is_file() and path.stat().st_size == 0:
            empty.append(rel)
    if missing:
        return False, "missing files: " + ", ".join(missing)
    if empty:
        return False, "empty files: " + ", ".join(empty)
    return True, "artifacts exist"


def normalize_args(tool: str, args: dict) -> tuple[dict | None, str]:
    if tool == "read_file":
        path = _string(args.get("path"))
        if not path:
            return None, "path is required"
        return {
            "path": path,
            "offset": _int(args.get("offset"), 1, 1, 1_000_000),
            "limit": _int(args.get("limit"), 400, 1, 800),
        }, ""
    if tool == "grep":
        pattern = _string(args.get("pattern"))
        if not pattern:
            return None, "pattern is required"
        return {"pattern": pattern, "path": _string(args.get("path")) or ".", "glob": _string(args.get("glob"))}, ""
    if tool == "glob":
        pattern = _string(args.get("pattern"))
        if not pattern:
            return None, "pattern is required"
        return {"pattern": pattern, "path": _string(args.get("path")) or "."}, ""
    if tool == "list_dir":
        return {"path": _string(args.get("path")) or "."}, ""
    if tool == "edit_file":
        path = _string(args.get("path"))
        old = args.get("old_string")
        new = args.get("new_string")
        if not path:
            return None, "path is required"
        if not isinstance(old, str) or old == "":
            return None, "old_string must be a non-empty string"
        if not isinstance(new, str):
            return None, "new_string must be a string"
        if old == new:
            return None, "old_string and new_string are identical"
        return {
            "path": path,
            "old_string": old,
            "new_string": new,
            "replace_all": _bool(args.get("replace_all")),
        }, ""
    if tool == "edit_batch":
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return None, "edits must be a non-empty list"
        if len(edits) > 40:
            return None, "too many edits"
        cleaned: list[dict] = []
        for item in edits:
            if not isinstance(item, dict):
                return None, "each edit must be an object"
            normalized, error = normalize_args("edit_file", item)
            if error or normalized is None:
                return None, error or "invalid edit"
            cleaned.append(normalized)
        return {"edits": cleaned}, ""
    if tool == "replace_text":
        old = args.get("old")
        new = args.get("new")
        if not isinstance(old, str) or old == "":
            return None, "old must be a non-empty string"
        if not isinstance(new, str):
            return None, "new must be a string"
        if old == new:
            return None, "old and new are identical"
        if len(old) > 10_000 or len(new) > 10_000:
            return None, "replacement is too long"
        return {"old": old, "new": new, "glob": _string(args.get("glob"))}, ""
    if tool == "write_file":
        path = _string(args.get("path"))
        content = args.get("content")
        if not path:
            return None, "path is required"
        if not isinstance(content, str):
            return None, "content must be a string"
        if len(content) > 1_000_000:
            return None, "content is too large"
        return {"path": path, "content": content}, ""
    if tool == "shell":
        command = _string(args.get("command"))
        if not command:
            return None, "command is required"
        if len(command) > 4000:
            return None, "command is too long"
        return {"command": command, "timeout_sec": _int(args.get("timeout_sec"), 120, 1, 300)}, ""
    if tool == "delete_file":
        path = _string(args.get("path"))
        if not path:
            return None, "path is required"
        return {"path": path}, ""
    if tool == "web_search":
        query = _string(args.get("query"))
        if not query:
            return None, "query is required"
        if len(query) > 400:
            return None, "query is too long"
        return {"query": query}, ""
    if tool == "web_fetch":
        url = _string(args.get("url"))
        if not url:
            return None, "url is required"
        return {"url": url}, ""
    if tool == "think":
        thought = _string(args.get("thought"))
        if not thought:
            return None, "thought is required"
        if len(thought) > 4000:
            return None, "thought is too long"
        return {"thought": thought}, ""
    if tool == "todo":
        items = args.get("todos")
        if not isinstance(items, list) or not items:
            return None, "todos must be a non-empty list"
        if len(items) > 20:
            return None, "too many todos"
        cleaned: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                return None, "each todo must be an object"
            content = _string(item.get("content"))
            status = _string(item.get("status")) or "pending"
            if not content:
                return None, "todo content is required"
            if status not in {"pending", "in_progress", "completed"}:
                return None, "status must be pending, in_progress, or completed"
            cleaned.append({"content": content, "status": status})
        return {"todos": cleaned}, ""
    return None, f"unknown tool {tool}"


def _has_payload(tool: str, args: dict) -> bool:
    if tool == "edit_file":
        return isinstance(args.get("old_string"), str)
    if tool == "edit_batch":
        return isinstance(args.get("edits"), list)
    if tool == "replace_text":
        return isinstance(args.get("old"), str)
    if tool == "write_file":
        return isinstance(args.get("content"), str)
    if tool == "shell":
        return isinstance(args.get("command"), str)
    if tool == "delete_file":
        return isinstance(args.get("path"), str)
    if tool == "web_search":
        return isinstance(args.get("query"), str)
    if tool == "web_fetch":
        return isinstance(args.get("url"), str)
    if tool == "think":
        return isinstance(args.get("thought"), str)
    if tool == "todo":
        return isinstance(args.get("todos"), list)
    if tool in {"grep", "glob"}:
        return isinstance(args.get("pattern"), str)
    if tool in {"read_file", "list_dir"}:
        return isinstance(args.get("path"), str) and "need" not in args
    return False


def _preview_args(args: dict) -> dict:
    shown: dict = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 1500:
            shown[key] = value[:1500] + f"...[{len(value)} chars]"
        else:
            shown[key] = value
    return shown


def _args_text(args: dict) -> str:
    return clip(json.dumps(_preview_args(args), default=str), 500)


def _string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _int(value: object, default: int, low: int, high: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False
