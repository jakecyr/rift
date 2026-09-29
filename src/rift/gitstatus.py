"""Git context and the stage/commit sequence.

Claude Code, Codex, and OpenCode put git status in the prompt before the
model chooses a tool. Reviewing a diff is not a reason to read source files,
and staging a commit is one add plus one commit, not a search of the repo.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as _spawn

from rift.safety import classify_path
from rift.util import clip


@dataclass(frozen=True)
class GitSnapshot:
    text: str
    is_repo: bool
    porcelain: str


@dataclass(frozen=True)
class GitPlan:
    review: bool
    commit: bool
    is_repo: bool
    paths: tuple[str, ...]
    held: tuple[str, ...]


def capture_git(root: Path) -> GitSnapshot:
    """Status, diff, and recent subjects. Empty when this folder is not a repo."""
    try:
        probe = _git(root, ["rev-parse", "--is-inside-work-tree"])
    except (OSError, subprocess.TimeoutExpired):
        return GitSnapshot("git is not available", False, "")
    if probe.code != 0 or probe.out.strip() != "true":
        return GitSnapshot("not a git repository", False, "")
    branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).out.strip() or "(no branch)"
    porcelain = _git(root, ["status", "--porcelain"]).out
    log = _git(root, ["log", "-8", "--oneline"]).out.strip()
    diff = _git(root, ["diff", "HEAD", "--"]).out if _git(root, ["rev-parse", "--verify", "HEAD"]).code == 0 else ""
    if not diff.strip():
        diff = _git(root, ["diff", "--"]).out
    untracked = _untracked_preview(root, porcelain)
    parts = [
        f"branch: {branch}",
        "status:",
        porcelain.strip() or "(clean)",
        "recent commits:",
        log or "(no commits yet)",
        "diff:",
        diff.strip() or "(no tracked diff)",
    ]
    if untracked:
        parts.extend(["untracked:", untracked])
    return GitSnapshot(clip("\n".join(parts), 16_000), True, porcelain)


def plan_git(goal: str, root: Path, snapshot: GitSnapshot) -> GitPlan | None:
    """A review or a commit. Ordinary edits that mention those words stay on the tool loop."""
    review = wants_git_review(goal)
    commit = wants_git_commit(goal)
    if not review and not commit:
        return None
    paths: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    if snapshot.is_repo:
        paths, held = split_status(root, snapshot.porcelain)
    return GitPlan(review, commit, snapshot.is_repo, paths, held)


def wants_git_review(goal: str) -> bool:
    text = _norm(goal)
    if any(phrase in text for phrase in ("git diff", "what changed", "what did i change", "show the diff")):
        return True
    if text in {"git status", "status"}:
        return True
    mentions_work = "change" in text or "diff" in text
    asks_to_look = "revie" in text or text.startswith("what ") or "show " in text or "look " in text
    if not (mentions_work and asks_to_look):
        return False
    if re.search(r"\b(function|class|bug|rename|implement|refactor|helper)\b", text):
        return False
    return True


def wants_git_commit(goal: str) -> bool:
    text = _norm(goal)
    if re.search(r"\b(do not|don't|dont|never)\b.{0,20}\bcommit\b", text):
        return False
    if any(phrase in text for phrase in ("commit hook", "pre-commit", "commit message", "commit-msg")):
        return False
    if not re.search(r"\b(stage|commit|check-in|check in)\b", text):
        return False
    code_task = re.search(
        r"\b(function|class|bug|rename|implement|refactor|helper|variable|hook|method)\b",
        text,
    )
    git_work = re.search(r"\b(stage|diff|changes|uncommitted|git)\b", text)
    if code_task and not git_work:
        return False
    return True


def split_status(root: Path, porcelain: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Paths to stage, and paths left alone because they look like secrets."""
    stage: list[str] = []
    held: list[str] = []
    for rel in parse_porcelain(porcelain):
        if _sensitive(root, rel):
            held.append(rel)
        else:
            stage.append(rel)
    return tuple(stage), tuple(held)


def parse_porcelain(text: str) -> list[str]:
    paths: list[str] = []
    for raw in text.splitlines():
        if len(raw) < 4:
            continue
        body = raw[3:]
        pieces = body.split(" -> ") if " -> " in body else [body]
        for piece in pieces:
            path = _unquote(piece)
            if path and path not in paths:
                paths.append(path)
    return paths


def git_add_command(paths: tuple[str, ...] | list[str]) -> str:
    quoted = " ".join(shlex.quote(path) for path in paths)
    return f"git add -- {quoted}"


def git_commit_command(message: str, *, identity: bool = False) -> str:
    prefix = "git "
    if identity:
        prefix += "-c user.name=rift -c user.email=rift@localhost "
    return prefix + "commit -m " + shlex.quote(message)


def mixed_shell_problem(command: str) -> str:
    """A review must not become `git diff && pytest`. Those are different jobs."""
    lowered = command.lower()
    has_git = bool(re.search(r"(^|[;&|]\s*)git\b", lowered))
    has_test = any(token in lowered for token in ("pytest", "unittest", "npm test", "go test", "cargo test"))
    if has_git and has_test:
        return "Run git and the tests as separate commands. Do not chain them."
    return ""


def _sensitive(root: Path, rel: str) -> bool:
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return True
    try:
        resolved = (root / candidate).resolve()
        resolved.relative_to(root.resolve())
    except ValueError:
        return True
    return classify_path(resolved).level != "allow"


def _line_for(porcelain: str, rel: str) -> str:
    quoted = f'"{rel}"'
    for item in porcelain.splitlines():
        body = item[3:] if len(item) > 3 else ""
        if body in {rel, quoted}:
            return item
        if " -> " in body and rel in {_unquote(part) for part in body.split(" -> ")}:
            return item
    return ""


def _unquote(path: str) -> str:
    text = path.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return text


def _untracked_preview(root: Path, porcelain: str, budget: int = 4000) -> str:
    blocks: list[str] = []
    for rel in parse_porcelain(porcelain):
        line = _line_for(porcelain, rel)
        if not line.startswith("??"):
            continue
        path = root / rel
        if not path.is_file() or _sensitive(root, rel):
            blocks.append(rel)
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            blocks.append(rel)
            continue
        piece = f"--- {rel} ---\n{clip(body, 1500)}"
        if sum(len(item) for item in blocks) + len(piece) > budget:
            blocks.append(f"{rel} ({path.stat().st_size} bytes)")
            break
        blocks.append(piece)
    return "\n".join(blocks)


def _norm(goal: str) -> str:
    return " ".join(goal.lower().replace(",", " ").split())


@dataclass(frozen=True)
class _GitOutput:
    code: int
    out: str


def _git(root: Path, args: list[str]) -> _GitOutput:
    completed = _spawn(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Keep stdout only. A leading space is a porcelain column, and stderr hints are not paths.
    return _GitOutput(completed.returncode, (completed.stdout or "").rstrip("\n"))
