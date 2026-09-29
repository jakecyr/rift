"""The agent loop.

The writer plans the task before any tool runs. Jev chooses each action against
that plan and gates edits and shell calls. The writer fills arguments and the
final note. Code owns limits, path sandboxing, hard safety rules, and the proof
that a finished task really landed.
"""

from __future__ import annotations

import json

from rift.decisions import (
    ActionDecision,
    CompletionDecision,
    DecisionError,
    GateDecision,
)
from rift.gitstatus import capture_git, secret_staging
from rift.llm import GenerationError, LLMResult
from rift.prompt import argument_system
from rift.safety import Verdict, allow, block, classify_path, classify_shell, confirm
from rift.state import (
    ASK_SYSTEM,
    BASE_CONSTRAINTS,
    PLAN_SYSTEM,
    SUMMARY_SYSTEM,
    Observation,
    View,
    jev_state,
    llm_user_message,
    plan_user_message,
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
        self._blocks = 0
        self._failures = 0

    def run_task(self, goal: str) -> str:
        request = goal.strip()
        started = self.llm.meter.snapshot()
        self._blocks = 0
        self._failures = 0
        view = View(
            goal=request,
            request=request,
            constraints=[*BASE_CONSTRAINTS, *self.settings.constraints],
            workspace=str(self.workspace.root),
            tree=self.workspace.tree(),
            project_instructions=project_instructions(self.workspace.root),
            snapshot=capture_git(self.workspace.root).text,
            prior_tasks=list(self.prior),
        )
        self.last_view = view
        try:
            self._plan(view)
        except KeyboardInterrupt:
            return self._finish_run(started, view, self._stop("interrupted", view))
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
                args, refusal = self._arguments(view, action)
                if refusal is not None:
                    signature = action.name + ":refusal:" + clip(refusal.detail, 800)
                else:
                    assert args is not None
                    signature = action.name + ":" + json.dumps(args, sort_keys=True, default=str)[:800]
                if refusal is not None or action.name != "shell":
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
                        view.guidance = (
                            f"{action.name} with those arguments already ran. "
                            "Check recent_actions for its result. "
                            "If done_when already holds, pick done. Otherwise pick a different action."
                        )
                        self.ui.trace("skipped a repeated action")
                        if seen[signature] >= 4:
                            summary = self._stop("repeated the same action", view)
                            break
                        continue
                if refusal is not None:
                    view.observations.append(refusal)
                    view.guidance = (
                        f"The writer could not fill {action.name}: {clip(refusal.detail, 400)} "
                        f"Do not pick {action.name} again for this step. "
                        "If done_when already holds, pick done. Otherwise pick a different tool."
                    )
                    self.ui.info(f"        {refusal.summary}: {clip(refusal.detail, 200)}")
                    continue
                assert args is not None
                if not self._permit(view, state, action.name, args):
                    if self._blocks >= 2:
                        summary = self._stop("policy blocked an action twice", view)
                        break
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
                    self._failures = 0
                else:
                    self._failures += 1
                    if self._failures >= 2:
                        view.guidance = (
                            "Those actions failed. Do not repeat them. "
                            "Use the error output for one narrower step, or ask_user."
                        )
                executed += 1
                if executed % 4 == 0:
                    stuck_strikes = self._check_progress(view, stuck_strikes)
            else:
                summary = self._stop("step limit reached", view)
        except GenerationError as error:
            self.ui.error(str(error))
            summary = self._stop(str(error), view)
        except KeyboardInterrupt:
            summary = self._stop("interrupted", view)
        return self._finish_run(started, view, summary, files_changed)

    def _plan(self, view: View) -> None:
        """Think before acting: restate the request, list the steps, and say what proves it done.

        A reply that cannot be parsed leaves the request as the goal and the loop runs unplanned.
        """
        tools = list(action_menu(self.settings.read_only))
        self.ui.status(f"        planning with {self._model_name('powerful')}")
        result = self._generate("powerful", PLAN_SYSTEM, plan_user_message(view, tools), 4096, 0.2)
        if result is None:
            self.ui.trace("no plan; running the request as written")
            return
        try:
            data = extract_json(result.text)
        except ValueError:
            self.ui.trace("plan was not JSON; running the request as written")
            return
        task = _string(data.get("task"))
        if task:
            view.goal = clip(task, 2000)
        steps = data.get("steps")
        if isinstance(steps, list):
            view.plan = [clip(_string(step), 300) for step in steps if _string(step)][:8]
        view.done_when = clip(_string(data.get("done_when")), 400)
        restated = view.goal if view.goal != view.request else ""
        self.ui.plan(restated, view.plan, view.done_when)

    def _generate(self, tier: str, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult | None:
        """One generation with one retry. Reasoning models can spend a budget and return no text."""
        try:
            return self.llm.complete(tier, system, user, max_tokens, temperature)
        except GenerationError as error:
            self.ui.trace(f"retrying after: {error}")
        try:
            return self.llm.complete(tier, system, user, max_tokens, temperature, effort="low")
        except GenerationError as error:
            self.ui.error(str(error))
            return None

    def _finish_run(self, started, view: View, summary: str, files_changed: list[str] | None = None) -> str:
        files = list(view.files_changed if files_changed is None else files_changed)
        self.ui.stats(self.llm.meter.delta(started), files)
        if summary:
            note = f"Request: {clip(view.goal, 400)}\nResult: {clip(summary, 500)}"
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
        passed, reason = prove_completion(verdict, files_changed, self.workspace, _ran_a_command(view))
        if not passed and files_changed and latest_test_passed(view):
            passed = True
            reason = "the test command exited 0 after the last edit"
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
        user += (
            "\n\nFill this tool for the first plan step that the recent actions have not done. "
            "If a recent result shows the plan is wrong, fill it for the step that the evidence calls for."
        )
        if action.name in {"write_file", "edit_file"}:
            user += (
                "\n\nThis tool accepts only one path. "
                "Fill the single next file that is not already written. "
                "Do not return need ask_user because other files remain."
            )
        max_tokens = 16384 if action.name in {"edit_file", "write_file", "edit_batch"} else 4096
        system = argument_system(view.project_instructions)
        self.ui.status(f"        writing arguments with {self._model_name(action.tier)}")
        result = self._generate(action.tier, system, user, max_tokens, 0.0)
        if result is None:
            return None, Observation(action.name, "", f"could not fill {action.name}", "the model returned no text", False)
        try:
            args = extract_json(result.text)
        except ValueError:
            retry = self._generate(
                action.tier,
                system,
                user + "\n\nYour previous reply was not one JSON object. Reply with only JSON.",
                max_tokens,
                0.0,
            )
            if retry is None:
                return None, Observation(action.name, "", "invalid arguments", "the model returned no text", False)
            try:
                args = extract_json(retry.text)
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
            self._blocks += 1
            self.ui.rule("block", code.reason)
            view.observations.append(
                Observation(
                    tool,
                    _args_text(args),
                    "blocked",
                    code.reason + " Do not retry this command.",
                    False,
                )
            )
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
                self._blocks += 1
                view.observations.append(
                    Observation(
                        tool,
                        _args_text(args),
                        "blocked by policy",
                        (
                            f"Jev blocked this {tool} (destructive {gate.destructive:.2f}). "
                            "Do not retry it. Choose a different action."
                        ),
                        False,
                    )
                )
                view.guidance = "The last action was blocked. Do not retry it."
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
            command = str(args.get("command", ""))
            verdict = classify_shell(command)
            if verdict.level != "allow":
                return verdict
            secrets = secret_staging(command, self.workspace.root)
            if secrets:
                return confirm("this would stage files that can hold secrets: " + ", ".join(secrets))
            return verdict
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


def _ran_a_command(view: View) -> bool:
    """A command that exited 0 is work on the machine, even when no file in the tree changed."""
    return any(item.tool == "shell" and item.ok for item in view.observations)


def latest_test_passed(view: View) -> bool:
    """A green test run after the last change is proof the model does not have to supply.

    Every tool that writes counts, not only the single-file ones.
    """
    saw_edit = False
    passed = False
    for item in view.observations:
        if item.tool in MUTATING - {"shell"} and item.ok:
            saw_edit = True
            passed = False
        if item.tool == "shell" and _is_test_command(f"{item.args_preview} {item.summary}"):
            passed = item.ok
    return saw_edit and passed


def _is_test_command(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("unittest", "pytest", "npm test", "go test", "cargo test"))


def prove_completion(
    verdict: CompletionDecision,
    files_changed: list[str],
    workspace: Workspace,
    ran_command: bool = False,
) -> tuple[bool, str]:
    """Jev's belief is not proof. The files have to exist, or a command has to have run."""
    if verdict.complete <= 0.8:
        return False, f"completion belief is {verdict.complete:.2f}; need above 0.80"
    if verdict.needs_file_changes >= 0.7 and not files_changed and not ran_command:
        return False, "the goal still needs changes and nothing was written or run"
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
