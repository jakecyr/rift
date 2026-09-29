"""Hard rules that never go through a model.

Jev can still ask for a confirmation on top of these. It cannot overrule a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SENSITIVE_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

_SHELL_WORDS = {"bash", "sh", "zsh", "dash"}
_SKIP_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class Verdict:
    level: str  # allow, confirm, block
    reason: str


def allow(reason: str = "no hard rule matched") -> Verdict:
    return Verdict("allow", reason)


def confirm(reason: str) -> Verdict:
    return Verdict("confirm", reason)


def block(reason: str) -> Verdict:
    return Verdict("block", reason)


def classify_shell(command: str) -> Verdict:
    text = command.strip()
    if not text:
        return block("empty command")

    lowered = text.lower()
    squashed = re.sub(r"\s+", "", lowered)
    if "rm-rf/" in squashed or "rm-fr/" in squashed or "rm-rf~" in squashed or "rm-fr~" in squashed:
        return block("recursive delete of a root or home path is blocked")
    if ":(){" in squashed:
        return block("fork bomb is blocked")
    if re.search(r">\s*/dev/(?:sd|nvme|disk|hd)", lowered):
        return block("writing to a disk device is blocked")
    if re.search(r"\b(?:mkfs|fdisk)\b", lowered):
        return block("disk formatting is blocked")
    if re.search(r"\bdd\b[^\n]*\bof=/dev/", lowered):
        return block("writing to a disk device is blocked")
    if re.search(r"\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b", lowered):
        return block("piping into a shell is blocked")
    if re.search(r"\b(?:shutdown|reboot|halt|poweroff)\b", lowered):
        return block("power commands are blocked")
    if re.search(r"\bchmod\b[^\n]*\s+777\s+/(?:\s|$)", lowered):
        return block("chmod 777 / is blocked")
    if _rm_targets_are_dangerous(text):
        return block("recursive delete of a root or home path is blocked")

    for segment in _segments(text):
        tokens = _command_tokens(segment)
        if not tokens:
            continue
        head = tokens[0]
        if head == "sudo" or "sudo" in tokens[:2]:
            return confirm("sudo needs a person")
        if head in {"rm", "unlink", "shred"}:
            return confirm("deleting files needs a person")
        if head in {"chmod", "chown", "chgrp"}:
            return confirm("permission changes need a person")
        if head in {"kill", "pkill", "killall"}:
            return confirm("kill needs a person")
        if head == "git":
            sub = _git_subcommand(tokens)
            if sub in {"push", "clean", "rebase"}:
                return confirm(f"git {sub} needs a person")
            if sub == "reset" and "--hard" in tokens:
                return confirm("git reset --hard needs a person")
        if head in {"npm", "pnpm", "yarn"} and "publish" in tokens:
            return confirm("publishing needs a person")
        if head == "docker" and any(token in tokens for token in ("system", "rm", "rmi", "prune")):
            return confirm("docker cleanup needs a person")
        if head == "kubectl" and any(token in tokens for token in ("delete", "apply")):
            return confirm("kubectl changes need a person")
    return allow("shell command has no hard-rule block")


def classify_path(path: Path) -> Verdict:
    name = path.name
    parts = set(path.parts)
    if name in SENSITIVE_NAMES or name.startswith(".env") or ".ssh" in parts:
        return confirm(f"{name} can hold secrets and needs a person")
    lowered = name.lower()
    if any(token in lowered for token in ("secret", "credential", "password")):
        return confirm(f"{name} looks sensitive and needs a person")
    return allow()


def _rm_targets_are_dangerous(command: str) -> bool:
    dangerous = {"/", "/*", "~", "~/", "$home", "${home}", ".", "..", "*", "./", "./*"}
    for segment in _segments(command):
        tokens = _command_tokens(segment)
        if tokens and tokens[0] == "sudo":
            tokens = tokens[1:]
        if not tokens or tokens[0] != "rm":
            continue
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            target = token.lower()
            if target in dangerous or target.startswith("/") or target.startswith("~"):
                return True
    return False


def _segments(command: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;&|\n]", command) if part.strip()]


def _command_tokens(segment: str) -> list[str]:
    tokens = segment.split()
    while tokens and _SKIP_ENV.match(tokens[0]):
        tokens.pop(0)
    if tokens and tokens[0] in {"command", "time", "nohup"}:
        tokens = tokens[1:]
    if len(tokens) >= 2 and tokens[0] == "sudo":
        return tokens
    if tokens and tokens[0] in _SHELL_WORDS and len(tokens) >= 3 and tokens[1] == "-c":
        return _command_tokens(tokens[2].strip("'\""))
    return tokens


def _git_subcommand(tokens: list[str]) -> str:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-C", "-c", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return ""
