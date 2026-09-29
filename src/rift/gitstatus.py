"""Git context for the planner and the loop, plus one hard rule about secrets.

Claude Code, Codex, and OpenCode put git status in the prompt before the model
chooses a tool. Any git work, from a review to a push, then goes through the
ordinary loop and the ordinary gates.
"""

from __future__ import annotations

import fnmatch
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


def capture_git(root: Path) -> GitSnapshot:
    """Branch, status, recent subjects, and the diff. Plain text when this is not a repo."""
    porcelain = changed_porcelain(root)
    if porcelain is None:
        return GitSnapshot("not a git repository", False, "")
    branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).out.strip() or "(no branch)"
    upstream = _git(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    ahead = ""
    if upstream.code == 0 and upstream.out.strip():
        counts = _git(root, ["rev-list", "--left-right", "--count", "@{u}...HEAD"]).out.split()
        if len(counts) == 2:
            ahead = f" (upstream {upstream.out.strip()}, behind {counts[0]}, ahead {counts[1]})"
    else:
        ahead = " (no upstream)"
    log = _git(root, ["log", "-8", "--oneline"]).out.strip()
    has_head = _git(root, ["rev-parse", "--verify", "HEAD"]).code == 0
    diff = _git(root, ["diff", "HEAD", "--"]).out if has_head else ""
    if not diff.strip():
        diff = _git(root, ["diff", "--"]).out
    untracked = _untracked_preview(root, porcelain)
    parts = [
        f"branch: {branch}{ahead}",
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


def changed_porcelain(root: Path) -> str | None:
    """`git status --porcelain` with every untracked file listed. None outside a repo."""
    try:
        probe = _git(root, ["rev-parse", "--is-inside-work-tree"])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.code != 0 or probe.out.strip() != "true":
        return None
    return _git(root, ["status", "--porcelain", "--untracked-files=all"]).out


def secret_staging(command: str, root: Path) -> list[str]:
    """Changed files that look like secrets and that this command would stage or commit."""
    if "git" not in command:
        return []
    porcelain = changed_porcelain(root)
    if not porcelain:
        return []
    sensitive = [(line, rel) for line, rel in _status_entries(porcelain) if _sensitive(root, rel)]
    if not sensitive:
        return []
    hits: list[str] = []
    for segment in re.split(r"[;&|\n]", command):
        tokens = _tokens(segment)
        if not tokens or tokens[0] != "git":
            continue
        sub, rest = _subcommand(tokens)
        if sub == "add":
            broad = any(token in {"-A", "--all", "-u", "--update"} for token in rest)
            specs = [token for token in rest if not token.startswith("-")]
            for line, rel in sensitive:
                if broad or any(_spec_covers(spec, rel) for spec in specs):
                    hits.append(rel)
        if sub == "commit" and any(_short_flag(token, "a") or token == "--all" for token in rest):
            hits.extend(rel for line, rel in sensitive if not line.startswith("??"))
    return sorted(set(hits))


def parse_porcelain(text: str) -> list[str]:
    paths: list[str] = []
    for _line, path in _status_entries(text):
        if path not in paths:
            paths.append(path)
    return paths


def _status_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw in text.splitlines():
        if len(raw) < 4:
            continue
        body = raw[3:]
        pieces = body.split(" -> ") if " -> " in body else [body]
        for piece in pieces:
            path = _unquote(piece)
            if path:
                entries.append((raw, path))
    return entries


def _spec_covers(spec: str, rel: str) -> bool:
    cleaned = spec.strip("\"'")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned in {".", "", "*", ":/", ":"}:
        return True
    if rel == cleaned or rel.startswith(cleaned.rstrip("/") + "/"):
        return True
    return fnmatch.fnmatch(rel, cleaned) or fnmatch.fnmatch(Path(rel).name, cleaned)


def _short_flag(token: str, letter: str) -> bool:
    return token.startswith("-") and not token.startswith("--") and letter in token[1:]


def _subcommand(tokens: list[str]) -> tuple[str, list[str]]:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-C", "-c", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token, tokens[index + 1 :]
    return "", []


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


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


def _unquote(path: str) -> str:
    text = path.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return text


def _untracked_preview(root: Path, porcelain: str, budget: int = 4000) -> str:
    blocks: list[str] = []
    for line, rel in _status_entries(porcelain):
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
