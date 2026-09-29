"""State sent to Jev, and the longer prompt sent to the writer model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rift.util import clip

BASE_CONSTRAINTS = [
    "Stay inside the workspace.",
    "Do not treat the task as finished until the files or the answer exist.",
    "Prefer edit_file over write_file when the file already exists.",
    "Use shell to run tests and builds, not to read files.",
    "Run Python with python3. The command python may be an older interpreter.",
    "Copy old_string from loaded file text. Do not invent surrounding code.",
]

ARG_SYSTEM = """You fill JSON arguments for one tool that was already chosen.
Return one JSON object and no other text.
Do not choose a different tool.
old_string must be copied exactly from the loaded file text, including whitespace.
If the loaded text is missing or truncated so you cannot copy an exact snippet, return {"need":"read_file","path":"the/file","reason":"why"}.
write_file and edit_file take one path. When the goal needs several files, fill the single next file that is not already in files changed. Do not return {"need":"ask_user"} because other files remain. The loop will call you again for the rest.
If the tool cannot be filled for any other reason, return {"need":"ask_user","path":"","reason":"why"}."""

SUMMARY_SYSTEM = """Write a short completion note for the user.
Plain text, no JSON. Say what changed, which files, and anything left unfinished.
Stay under 150 words."""

PLAN_SYSTEM = """You plan one coding-agent task before any tool runs. Think it through, then return one JSON object and no other text:
{"task": "...", "steps": ["...", "..."], "done_when": "..."}

task: the user's request as a standalone instruction. Resolve words such as "again", "it", or "that" from the earlier requests. Keep the user's scope. Do not add work they did not ask for.
steps: 0 to 8 short steps in order. Each step is one tool action with a concrete target: read a named file, grep for a symbol, run a named command, edit a named file. Use the workspace snapshot and the project instructions for real paths, branches, and commands. Leave out a step when its result is already in the snapshot. When a step writes or runs something, say what it is.
done_when: one observable condition that proves the task is finished, for example a command exiting 0, a commit existing, or a file containing the change.

A question the snapshot already answers needs no steps.
Do not plan destructive, irreversible, or history-rewriting commands the user did not ask for. Do not plan to skip hooks or checks."""

ASK_SYSTEM = """Write one specific question for the user.
Plain text, no preamble. Ask only for a fact you need in order to continue."""


@dataclass
class Observation:
    tool: str
    args_preview: str
    summary: str
    detail: str
    ok: bool


@dataclass
class View:
    goal: str
    constraints: list[str]
    workspace: str
    tree: str
    project_instructions: str
    snapshot: str = ""
    request: str = ""
    plan: list[str] = field(default_factory=list)
    done_when: str = ""
    files_changed: list[str] = field(default_factory=list)
    loaded: dict[str, str] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    guidance: str = ""
    prior_tasks: list[str] = field(default_factory=list)
    todos: list[dict[str, str]] = field(default_factory=list)

    def remember_file(self, path: str, content: str) -> None:
        self.loaded.pop(path, None)
        self.loaded[path] = content[:120_000]
        while len(self.loaded) > 6:
            oldest = next(iter(self.loaded))
            del self.loaded[oldest]


def jev_state(view: View) -> dict:
    recent = []
    for item in view.observations[-8:]:
        entry = {
            "tool": item.tool,
            "ok": item.ok,
            "args": clip(item.args_preview, 240),
            "summary": clip(item.summary, 400),
        }
        if item.detail and item.tool in {
            "read_file",
            "grep",
            "edit_file",
            "write_file",
            "shell",
            "web_search",
            "web_fetch",
            "think",
            "todo",
        }:
            entry["evidence"] = clip(item.detail, 700)
        recent.append(entry)
    state = {
        "goal": view.goal,
        "plan": view.plan,
        "done_when": view.done_when,
        "constraints": view.constraints,
        "workspace": view.workspace,
        "tree": clip(view.tree, 8000),
        "files_changed": view.files_changed[-20:],
        "files_loaded": list(view.loaded.keys()),
        "recent_actions": recent,
    }
    if view.snapshot:
        state["workspace_snapshot"] = clip(view.snapshot, 4000)
    if view.project_instructions:
        state["project_instructions"] = clip(view.project_instructions, 4000)
    if view.guidance:
        state["guidance"] = view.guidance
    if view.prior_tasks:
        state["prior_tasks"] = view.prior_tasks[-4:]
    if view.todos:
        state["todos"] = view.todos
    return state


def llm_user_message(view: View, tool_name: str, arg_help: str) -> str:
    sections = [
        f"Selected tool: {tool_name}",
        f"Argument shape: {arg_help}",
        f"Goal:\n{view.goal}",
    ]
    if view.plan:
        sections.append("Plan:\n" + "\n".join(f"{index}. {step}" for index, step in enumerate(view.plan, 1)))
    if view.done_when:
        sections.append(f"Done when:\n{view.done_when}")
    sections += [
        "Constraints:\n" + "\n".join(f"- {item}" for item in view.constraints),
        f"Workspace: {view.workspace}",
        f"Tree:\n{clip(view.tree, 8000)}",
    ]
    if view.snapshot:
        sections.append("Workspace snapshot:\n" + clip(view.snapshot, 16_000))
    if view.project_instructions:
        sections.append("Project instructions:\n" + clip(view.project_instructions, 4000))
    if view.files_changed:
        sections.append("Files changed this task:\n" + "\n".join(view.files_changed))
    if view.guidance:
        sections.append("Guidance:\n" + view.guidance)
    if view.prior_tasks:
        sections.append("Earlier tasks:\n" + "\n".join(view.prior_tasks[-4:]))
    if view.todos:
        lines = [f"- {item['status']}: {item['content']}" for item in view.todos]
        sections.append("Task list:\n" + "\n".join(lines))
    if view.loaded:
        sections.append("Loaded files (exact text you may copy):\n" + _loaded_block(view))
    if view.observations:
        sections.append("Recent actions:\n" + _observation_block(view))
    return clip("\n\n".join(sections), 80_000)


def plan_user_message(view: View, tools: list[str]) -> str:
    sections = [
        f"User request:\n{view.request or view.goal}",
        "Tools the loop can run:\n" + ", ".join(tools),
        f"Workspace: {view.workspace}",
        f"Tree:\n{clip(view.tree, 8000)}",
    ]
    if view.prior_tasks:
        sections.append("Earlier requests and results, oldest first:\n" + "\n\n".join(view.prior_tasks[-4:]))
    if view.snapshot:
        sections.append("Workspace snapshot:\n" + clip(view.snapshot, 16_000))
    if view.project_instructions:
        sections.append("Project instructions:\n" + clip(view.project_instructions, 4000))
    return clip("\n\n".join(sections), 60_000)


def state_is_large(state: dict) -> bool:
    return len(json.dumps(state, default=str)) > 40_000


def _loaded_block(view: View) -> str:
    blocks: list[str] = []
    budget = 48_000
    for path, content in reversed(list(view.loaded.items())):
        header = f"--- {path} ({len(content)} chars) ---"
        room = budget - len(header) - 2
        if room < 500:
            break
        body = content if len(content) <= room else content[:room] + "\n...[file truncated, read a narrower offset]"
        blocks.append(f"{header}\n{body}")
        budget -= len(blocks[-1])
    return "\n\n".join(reversed(blocks))


def _observation_block(view: View) -> str:
    lines: list[str] = []
    for item in view.observations[-6:]:
        lines.append(f"- {item.tool} ok={item.ok} {clip(item.summary, 300)}")
        if item.detail and item.tool in {
            "shell",
            "grep",
            "edit_file",
            "write_file",
            "ask_user",
            "web_search",
            "web_fetch",
            "think",
            "todo",
        }:
            lines.append(clip(item.detail, 2500))
    return "\n".join(lines)
