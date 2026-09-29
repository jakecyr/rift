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

from jevcode.llm import UsageMeter


class UI:
    def __init__(
        self,
        verbose: bool = False,
        assume_yes: bool = False,
        console: Console | None = None,
    ) -> None:
        self.verbose = verbose
        self.assume_yes = assume_yes
        self.console = console or Console()
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
                title="[bold]jevcode[/]",
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
            ("/quit", "leave"),
        ]
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(Panel(table, title="commands", border_style="cyan", padding=(1, 1)))
        self.console.print("[dim]Anything else is a task. test runs the project's tests. Up-arrow recalls history.[/]")

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
        self.console.print(
            f"\n[bold cyan]step {step}[/]  [bold]{escape(name)}[/]  "
            f"[dim]{confidence:.2f}  {escape(tier)}[/]"
        )
        if self.verbose and probabilities:
            ranked = "  ".join(
                f"{escape(key)} [dim]{value:.2f}[/]"
                for key, value in sorted(probabilities.items(), key=lambda item: -item[1])
            )
            extra = f"  [dim]{escape(request_id)}[/]" if request_id else ""
            self.console.print(f"  {ranked}{extra}")

    def gate(self, action: str, destructive: float, reason: str) -> None:
        color = {"allow": "green", "confirm": "yellow", "block": "red"}.get(action, "white")
        tail = f"  {escape(reason)}" if reason else ""
        self.console.print(f"  [{color}]gate {escape(action)}[/]  [dim]destructive {destructive:.2f}[/]{tail}")

    def rule(self, level: str, reason: str) -> None:
        color = {"allow": "green", "confirm": "yellow", "block": "red"}.get(level, "white")
        self.console.print(f"  [{color}]rule {escape(level)}[/]  {escape(reason)}")

    def tool(self, name: str, summary: str, detail: str) -> None:
        self.console.print(f"  [dim]{escape(name)}[/]  {escape(summary)}")
        if not detail or name not in {"shell", "grep", "edit_file", "write_file", "glob"}:
            return
        preview = "\n".join(detail.splitlines()[:40])
        if _looks_like_diff(preview):
            self.console.print(Syntax(preview, "diff", theme="monokai", word_wrap=True, padding=1))
            return
        self.console.print(Text(preview, style="dim"))

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
        cost = meter.jev_input_tokens / 1_000_000 * 0.042
        self.console.print(
            "  [dim]"
            f"jev {meter.jev_calls} calls · {meter.jev_input_tokens} tokens · ~${cost:.4f}"
            f"    writer {meter.llm_calls} calls · {meter.llm_input_tokens} in / {meter.llm_output_tokens} out"
            "[/]"
        )
        if files:
            self.console.print("  [dim]files[/]  " + escape(", ".join(files)))

    def clear(self) -> None:
        self.console.clear()

    def say(self, text: str) -> None:
        self.console.print(text, highlight=False)


def short_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return "~/" + candidate.resolve().relative_to(Path.home()).as_posix()
    except ValueError:
        return str(candidate)


def _looks_like_diff(text: str) -> bool:
    return text.startswith("---") or text.startswith("+++") or "\n@@" in text
