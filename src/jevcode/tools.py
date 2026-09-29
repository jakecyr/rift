"""Workspace tools. The model picks one; this module does the work."""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jevcode.safety import classify_path, classify_shell
from jevcode.util import clip

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".jevcode",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    criteria: str
    arg_help: str
    readonly: bool


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "read_file",
        "Read a specific file whose contents are not already loaded and are needed next.",
        'Keys: "path" (required), "offset" (1-based line, default 1), "limit" (lines, default 400, max 800).',
        True,
    ),
    ToolSpec(
        "grep",
        "Find a symbol, string, or error inside file contents.",
        'Keys: "pattern" (required, Python regex), "path" (default "."), "glob" (optional, for example *.py).',
        True,
    ),
    ToolSpec(
        "glob",
        "Find files by name when the path is not already known.",
        'Keys: "pattern" (required, for example **/*.py or *.md), "path" (default ".").',
        True,
    ),
    ToolSpec(
        "list_dir",
        "List one directory when the workspace tree does not show what you need.",
        'Keys: "path" (directory, default ".").',
        True,
    ),
    ToolSpec(
        "edit_file",
        "Change an existing file with one exact replacement. Only after that file and the test or spec for the change are both loaded.",
        'Keys: "path" (required), "old_string" (exact text, required), "new_string" (required), "replace_all" (default false).',
        False,
    ),
    ToolSpec(
        "write_file",
        "Create a new file, or replace a whole file when a small edit cannot express the change.",
        'Keys: "path" (required), "content" (full new text, required).',
        False,
    ),
    ToolSpec(
        "shell",
        "Run a command. A request to run tests, including the word test alone, means run the project's test command. Not for reading files.",
        'Keys: "command" (required), "timeout_sec" (default 120, max 300).',
        False,
    ),
)

SPECS = {spec.name: spec for spec in TOOL_SPECS}

ASK_CRITERIA = (
    "The goal is ambiguous or missing a fact only the user knows, and guessing would write the wrong code."
)
DONE_CRITERIA = (
    "The requested change is already saved, or the question is fully answered, and no further tool is needed."
)


def action_menu(read_only: bool) -> dict[str, str]:
    menu = {
        spec.name: spec.criteria
        for spec in TOOL_SPECS
        if not read_only or spec.readonly
    }
    menu["ask_user"] = ASK_CRITERIA
    menu["done"] = DONE_CRITERIA
    return menu


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    summary: str
    detail: str


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: Path, allow_outside: bool = False) -> None:
        self.root = root.resolve()
        self.allow_outside = allow_outside

    def tree(self, limit: int = 200) -> str:
        lines: list[str] = []
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(name for name in dirnames if not _skip_dir(name))
            rel_dir = Path(dirpath).relative_to(self.root)
            depth = 0 if rel_dir == Path(".") else len(rel_dir.parts)
            if depth >= 5:
                dirnames.clear()
                continue
            indent = "  " * depth
            for name in sorted(filenames):
                if name == ".DS_Store":
                    continue
                if count >= limit:
                    lines.append("...")
                    return "\n".join(lines) or "(empty)"
                lines.append(f"{indent}{name}")
                count += 1
            for name in dirnames:
                if count >= limit:
                    lines.append("...")
                    return "\n".join(lines) or "(empty)"
                lines.append(f"{indent}{name}/")
                count += 1
        return "\n".join(lines) or "(empty)"

    def list_dir(self, raw: str) -> ToolResult:
        try:
            path = self.resolve(raw or ".")
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        if not path.is_dir():
            return ToolResult(False, "not a directory", str(raw))
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.name == ".DS_Store":
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
            if len(entries) >= 200:
                entries.append("...")
                break
        rel = self.display(path)
        body = "\n".join(entries) or "(empty)"
        return ToolResult(True, f"listed {rel} ({len(entries)} entries)", body)

    def glob(self, pattern: str, raw: str) -> ToolResult:
        try:
            base = self.resolve(raw or ".")
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        if not base.is_dir():
            return ToolResult(False, "not a directory", str(raw))
        matches: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if not _skip_dir(name)]
            for name in filenames:
                full = Path(dirpath) / name
                rel = full.relative_to(base).as_posix()
                if _glob_match(rel, name, pattern):
                    matches.append(self.display(full))
                    if len(matches) >= 100:
                        matches.append("...")
                        return ToolResult(True, f"{len(matches) - 1}+ matches", "\n".join(matches))
        body = "\n".join(matches) or "(no matches)"
        return ToolResult(True, f"{len(matches)} matches for {pattern}", body)

    def grep(self, pattern: str, raw: str, file_glob: str) -> ToolResult:
        try:
            base = self.resolve(raw or ".")
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        try:
            regex = re.compile(pattern)
        except re.error as error:
            return ToolResult(False, "bad regex", str(error))
        hits: list[str] = []
        files: list[Path]
        if base.is_file():
            files = [base]
        else:
            files = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [name for name in dirnames if not _skip_dir(name)]
                for name in filenames:
                    rel = (Path(dirpath) / name).relative_to(base).as_posix()
                    if file_glob and not _glob_match(rel, name, file_glob):
                        continue
                    files.append(Path(dirpath) / name)
        for path in files:
            if len(hits) >= 40:
                break
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{self.display(path)}:{number}: {clip(line.strip(), 200)}")
                    if len(hits) >= 40:
                        break
        body = "\n".join(hits) or "(no matches)"
        suffix = "" if len(hits) < 40 else " (capped)"
        return ToolResult(True, f"{len(hits)} hits{suffix}", body)

    def read_file(self, raw: str, offset: int, limit: int) -> ToolResult:
        try:
            path = self.resolve(raw)
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        if not path.is_file():
            return ToolResult(False, "file not found", self.display(path) if path.exists() else raw)
        if path.stat().st_size > 2_000_000:
            return ToolResult(False, "file too large to read", "use grep or a smaller offset")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(False, "binary file", raw)
        lines = text.splitlines()
        start = max(0, offset - 1)
        window = lines[start : start + limit]
        body = "\n".join(window)
        end = start + len(window)
        summary = f"{self.display(path)} lines {start + 1}-{end} of {len(lines)}"
        return ToolResult(True, summary, body)

    def edit_file(self, raw: str, old: str, new: str, replace_all: bool, approved: bool) -> ToolResult:
        try:
            path = self.resolve(raw)
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        sensitive = classify_path(path)
        if sensitive.level == "confirm" and not approved:
            return ToolResult(False, "not approved", sensitive.reason)
        if not path.is_file():
            return ToolResult(False, "file not found", raw)
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(False, "binary file", raw)
        newline = "\r\n" if "\r\n" in original else "\n"
        current = original.replace("\r\n", "\n")
        old_norm = old.replace("\r\n", "\n")
        new_norm = new.replace("\r\n", "\n")
        count = current.count(old_norm)
        if count == 0:
            return ToolResult(False, "old_string not found", f"no exact match in {self.display(path)}")
        if count > 1 and not replace_all:
            return ToolResult(
                False,
                "old_string matched more than once",
                f"{count} matches in {self.display(path)}; add context or set replace_all",
            )
        updated = current.replace(old_norm, new_norm) if replace_all else current.replace(old_norm, new_norm, 1)
        if newline == "\r\n":
            updated = updated.replace("\n", "\r\n")
        path.write_text(updated, encoding="utf-8")
        diff = unified_diff(original, updated, self.display(path))
        return ToolResult(True, f"edited {self.display(path)}", diff)

    def write_file(self, raw: str, content: str, approved: bool) -> ToolResult:
        try:
            path = self.resolve(raw)
        except ToolError as error:
            return ToolResult(False, "path rejected", str(error))
        sensitive = classify_path(path)
        if sensitive.level == "confirm" and not approved:
            return ToolResult(False, "not approved", sensitive.reason)
        if path.exists() and path.is_dir():
            return ToolResult(False, "path is a directory", raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        original = ""
        if path.is_file():
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult(False, "refusing to overwrite a binary file", raw)
        path.write_text(content, encoding="utf-8")
        diff = unified_diff(original, content, self.display(path))
        return ToolResult(True, f"wrote {self.display(path)} ({len(content)} chars)", diff)

    def shell(self, command: str, timeout: int, approved: bool) -> ToolResult:
        verdict = classify_shell(command)
        if verdict.level == "block":
            return ToolResult(False, "blocked", verdict.reason)
        if verdict.level == "confirm" and not approved:
            return ToolResult(False, "not approved", verdict.reason)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"timed out after {timeout}s", command)
        chunks = [completed.stdout or "", completed.stderr or ""]
        output = "\n".join(chunk for chunk in chunks if chunk).strip() or "(no output)"
        summary = f"exit {completed.returncode}: {clip(command, 120)}"
        return ToolResult(completed.returncode == 0, summary, clip(output, 16000))

    def file_excerpt(self, command: str) -> str:
        excerpts: list[str] = []
        for token in command.replace("\\", " ").split():
            if token.startswith("-") or len(token) > 180:
                continue
            try:
                path = self.resolve(token.strip("\"'"))
            except ToolError:
                continue
            if not path.is_file() or path.stat().st_size > 100_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            excerpts.append(f"--- {self.display(path)} ---\n{clip(text, 2000)}")
            if len(excerpts) >= 2:
                break
        return "\n".join(excerpts)

    def read_text_if_small(self, raw: str, limit: int = 120_000) -> str:
        path = self.resolve(raw)
        if not path.is_file() or path.stat().st_size > limit:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def resolve(self, raw: str) -> Path:
        text = (raw or ".").strip()
        if "\x00" in text:
            raise ToolError("path contains a null byte")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if not self.allow_outside:
            try:
                resolved.relative_to(self.root)
            except ValueError as error:
                raise ToolError(f"path is outside the workspace: {raw}") from error
        return resolved

    def display(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


def unified_diff(old: str, new: str, rel: str, max_lines: int = 80) -> str:
    if old == new:
        return f"{rel} unchanged"
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    extra = ""
    if len(lines) > max_lines:
        extra = f"\n...[{len(lines) - max_lines} more diff lines]"
        lines = lines[:max_lines]
    return "\n".join(lines) + extra


def project_instructions(root: Path, home: Path | None = None) -> str:
    """Load AGENTS.md and CLAUDE.md the way OpenCode and Claude Code do.

    Global files come first, then each directory from the home directory down
    to the workspace. Both names are included when a directory has both.
    """
    home_path = (home or Path.home()).expanduser().resolve()
    root_path = root.expanduser().resolve()
    files: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(resolved)

    add(home_path / ".config" / "opencode" / "AGENTS.md")
    add(home_path / ".claude" / "CLAUDE.md")
    for directory in _instruction_directories(root_path, home_path):
        for name in ("AGENTS.md", "agents.md", "CLAUDE.md", "claude.md"):
            add(directory / name)
    for name in ("jevcode.md", ".cursorrules"):
        add(root_path / name)

    parts: list[str] = []
    budget = 32_000
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        piece = f"# {path}\n{text}"
        if len(piece) > budget:
            piece = piece[:budget]
        parts.append(piece)
        budget -= len(piece)
        if budget <= 0:
            break
    return "\n\n".join(parts)


def _instruction_directories(root: Path, home: Path) -> list[Path]:
    try:
        root.relative_to(home)
    except ValueError:
        return [root]
    chain = [root]
    current = root
    while current != home and current.parent != current:
        current = current.parent
        chain.append(current)
    chain.reverse()
    return chain


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def _glob_match(rel: str, name: str, pattern: str) -> bool:
    cleaned = pattern.replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    candidates = [cleaned]
    if cleaned.startswith("**/"):
        candidates.append(cleaned[3:])
    return any(fnmatch.fnmatch(rel, item) or fnmatch.fnmatch(name, item) for item in candidates)
