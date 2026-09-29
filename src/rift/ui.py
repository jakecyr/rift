"""Terminal output."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from rift.llm import JEV_USD_PER_MILLION_INPUT, UsageMeter


class UI:
    def __init__(
        self,
        verbose: bool = False,
        assume_yes: bool = False,
        console: Console | None = None,
    ) -> None:
        self.verbose: bool = verbose
        self.assume_yes: bool = assume_yes
        self.console: Console = console or Console()
        self.session = None
        self._in_tty = sys.stdin.isatty()

    def banner(self, workspace: str, provider: str, model: str, jev_model: str, fast: str) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right")
        grid.add_column()
        grid.add_row("workspace", escape(workspace))
        grid.add_row("writer", f"[bold]{escape(provider)}[/]  {escape(model)}")
        if fast:
            grid.add_row("fast", escape(fast))
        grid.add_row("jev", escape(jev_model))
        self.console.print(
            Panel(
                grid,
                title="[bold]rift[/]",
                subtitle="[dim]/help[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def help(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        rows = [
            ("/provider", "show or set openai, anthropic, grok, ollama"),
            ("/model", "show or set the writer model"),
            ("/effort", "reasoning level: off, low, medium, high, xhigh, max"),
            ("/fast", "cheaper model for Jev to route to, or /fast off"),
            ("/key", "set openai, anthropic, grok, or jev key"),
            ("/jev", "Jev model id"),
            ("/cd", "change the workspace"),
            ("/pwd", "show the workspace"),
            ("/clear", "forget this session and clear the screen"),
            ("/yes", "toggle auto-approve for ordinary confirms"),
            ("/confirm", "toggle ask-before-every-edit"),
            ("/read-only", "toggle edits and shell"),
            ("/verbose", "toggle Jev probabilities"),
            ("/doctor", "which keys are set"),
            ("/status", "show the current session"),
            ("!", "run a shell command and keep the output for the next task"),
            ("/quit", "leave"),
        ]
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(Panel(table, title="commands", border_style="cyan", padding=(1, 1)))
        self.console.print(
            "[dim]Anything else is a task. !command runs in this workspace. test runs the project's tests. "
            "Up-arrow recalls history.[/]"
        )

    def doctor(self, settings) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right")
        grid.add_column()
        grid.add_row("workspace", escape(str(settings.workspace)))
        grid.add_row("writer", f"{escape(settings.provider)}  {escape(settings.model)}")
        grid.add_row("writer key", "set" if settings.api_key or settings.provider == "ollama" else "[red]missing[/]")
        grid.add_row("jev", escape(settings.jev_model))
        grid.add_row("jev key", "set" if settings.jev_api_key else "[red]missing[/]")
        if settings.fast_model:
            grid.add_row("fast", f"{escape(settings.fast_provider or '')}  {escape(settings.fast_model)}")
        else:
            grid.add_row("fast", "off")
        for name, present in settings.key_status.items():
            grid.add_row(name, "set" if present else "missing")
        if settings.env_files:
            grid.add_row("env files", "\n".join(escape(path) for path in settings.env_files))
        self.console.print(Panel(grid, title="doctor", border_style="cyan", padding=(1, 2)))

    def status(self, text: str) -> None:
        self.trace(text)

    def plan(self, task: str, steps: list[str], done_when: str) -> None:
        if not task and not steps:
            return
        self.console.print("\n[bold]plan[/]", highlight=False)
        if task:
            self.console.print(f"  [dim]{escape(task)}[/]", highlight=False)
        for index, step in enumerate(steps, 1):
            self.console.print(f"  {index}. {escape(step)}", highlight=False)
        if done_when and self.verbose:
            self.console.print(f"  [dim]done when {escape(done_when)}[/]", highlight=False)

    def trace(self, text: str) -> None:
        if self.verbose and text.strip():
            self.console.print(f"  [dim]{escape(text.strip())}[/]")

    def decision(
        self,
        step: int,
        name: str,
        confidence: float,
        tier: str,
        probabilities: dict,
        request_id: str,
    ) -> None:
        if self.verbose:
            self.console.print(
                f"\n[bold cyan]step {step}[/]  [bold]{escape(name)}[/]  "
                f"[dim]{confidence:.2f}  {escape(tier)}[/]"
            )
            if probabilities:
                ranked = "  ".join(
                    f"{escape(key)} [dim]{value:.2f}[/]"
                    for key, value in sorted(probabilities.items(), key=lambda item: -item[1])
                )
                extra = f"  [dim]{escape(request_id)}[/]" if request_id else ""
                self.console.print(f"  {ranked}{extra}")
            return
        if name == "done":
            return
        self.console.print(f"\n[bold]{escape(_label(name))}[/]", highlight=False)

    def gate(self, action: str, destructive: float, reason: str) -> None:
        if action == "allow" and not self.verbose:
            return
        color = {"allow": "green", "confirm": "yellow", "block": "red"}.get(action, "white")
        tail = f"  {escape(reason)}" if reason else ""
        score = f"  [dim]destructive {destructive:.2f}[/]" if self.verbose else ""
        self.console.print(f"  [{color}]{escape(action)}[/]{score}{tail}")

    def rule(self, level: str, reason: str) -> None:
        if level == "allow" and not self.verbose:
            return
        color = {"allow": "green", "confirm": "yellow", "block": "red"}.get(level, "white")
        self.console.print(f"  [{color}]{escape(level)}[/]  {escape(reason)}")

    def tool(self, name: str, summary: str, detail: str, paths: tuple[str, ...] = ()) -> None:
        if name in {"replace_text", "edit_batch"} and paths:
            self._file_list(paths)
            return
        if name in {"edit_file", "write_file"}:
            self.console.print(f"  {escape(_short_summary(summary))}", highlight=False)
            self._diff(detail, 18 if self.verbose else 10)
            return
        if name == "grep":
            self.console.print(f"  {escape(summary)}", highlight=False)
            if self.verbose:
                self._body(detail, 24)
            return
        if name == "shell":
            self.console.print(f"  {escape(summary)}", highlight=False)
            failed = not summary.startswith("exit 0")
            if self.verbose or failed:
                self._body(detail, 24 if self.verbose else 12)
            return
        if name in {"web_search", "web_fetch", "think", "todo", "delete_file"}:
            self.console.print(f"  {escape(summary)}", highlight=False)
            if detail.strip() and detail.strip() != summary.strip():
                self._body(detail, 16 if self.verbose else 6)
            return
        if name == "read_file":
            self.console.print(f"  [dim]{escape(summary)}[/]", highlight=False)
            return
        if summary:
            self.console.print(f"  {escape(summary)}", highlight=False)

    def _file_list(self, paths: tuple[str, ...]) -> None:
        shown = list(paths)[:8]
        for path in shown:
            self.console.print(f"  [dim]{escape(path)}[/]", highlight=False)
        extra = len(paths) - len(shown)
        if extra:
            self.console.print(f"  [dim]+{extra} more[/]")

    def _diff(self, detail: str, limit: int) -> None:
        if not detail or not _looks_like_diff(detail):
            return
        lines = detail.splitlines()
        preview = "\n".join(lines[:limit])
        self.console.print(Syntax(preview, "diff", theme="monokai", word_wrap=True, padding=(0, 1)))
        if len(lines) > limit:
            self.console.print(f"  [dim]+{len(lines) - limit} lines[/]")

    def _body(self, detail: str, limit: int) -> None:
        if not detail:
            return
        lines = detail.splitlines()
        preview = "\n".join(lines[:limit])
        self.console.print(Text(preview, style="dim"))
        if len(lines) > limit:
            self.console.print(f"  [dim]+{len(lines) - limit} lines[/]")

    def command_result(self, command: str, summary: str, detail: str) -> None:
        self.console.print(f"\n[bold]![/] {escape(command)}", highlight=False)
        self.console.print(f"  {escape(summary)}", highlight=False)
        self._body(detail, 40)

    def info(self, text: str) -> None:
        self.console.print(f"  {escape(text)}")

    def error(self, text: str) -> None:
        self.console.print(f"[red]{escape(text)}[/]")

    def confirm(self, message: str, forced: bool) -> bool:
        self.console.print(Text(message))
        if not forced and self.assume_yes:
            self.console.print("  [dim]auto-approved[/]")
            return True
        if not self._in_tty:
            self.console.print("  [red]denied[/] [dim](no terminal to approve it)[/]")
            return False
        try:
            if self.session is not None:
                answer = self.session.prompt("  approve? [y/N] ").strip().lower()
            else:
                answer = input("  approve? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes"}

    def ask(self, question: str) -> str:
        self.console.print(Panel(escape(question), title="question", border_style="yellow"))
        if not self._in_tty:
            return ""
        try:
            if self.session is not None:
                return self.session.prompt("  you › ").strip()
            return input("you> ").strip()
        except EOFError:
            return ""

    def finished(self, summary: str) -> None:
        self.console.print()
        self.console.print(Panel(escape(summary.strip()), title="done", border_style="green", padding=(1, 2)))

    def stats(self, meter: UsageMeter, files: list[str]) -> None:
        cost = meter.jev_input_tokens / 1_000_000 * JEV_USD_PER_MILLION_INPUT
        parts = [f"jev {meter.jev_calls}", f"writer {meter.llm_calls}"]
        if files:
            parts.append(f"{len(files)} files")
        if meter.jev_calls:
            parts.append(f"~${cost:.4f}")
        self.console.print("\n[dim]" + "  ·  ".join(parts) + "[/]", highlight=False)

    def clear(self) -> None:
        self.console.clear()

    def say(self, text: str) -> None:
        self.console.print(text, highlight=False)


def _label(name: str) -> str:
    return {
        "read_file": "read",
        "edit_file": "edit",
        "edit_batch": "edit",
        "write_file": "write",
        "delete_file": "delete",
        "replace_text": "replace",
        "list_dir": "list",
        "web_search": "web search",
        "web_fetch": "fetch",
        "think": "think",
        "todo": "todo",
    }.get(name, name)


def _short_summary(summary: str) -> str:
    for prefix in ("edited ", "wrote "):
        if summary.startswith(prefix):
            return summary[len(prefix) :]
    return summary


def short_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return "~/" + candidate.resolve().relative_to(Path.home()).as_posix()
    except ValueError:
        return str(candidate)


def _looks_like_diff(text: str) -> bool:
    return text.startswith("---") or text.startswith("+++") or "\n@@" in text
