"""Interactive session: slash commands change model, keys, and workspace."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as prompt_secret
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from typesafe_sdk import TypeSafeClient

from jevcode.config import (
    DEFAULT_MODELS,
    PROVIDERS,
    refresh_settings,
    save_prefs,
    save_user_key,
    user_config_dir,
)
from jevcode.llm import LLM, Profile
from jevcode.tools import Workspace
from jevcode.ui import short_path


class App:
    def __init__(self, settings, agent, ui, jev_client, prefs_file: Path | None = None) -> None:
        self.settings = settings
        self.agent = agent
        self.ui = ui
        self.jev_client = jev_client
        self.prefs_file = prefs_file

    def rebuild_writer(self) -> None:
        meter = self.agent.llm.meter
        new = llm_from_settings(self.settings, meter)
        old = self.agent.llm
        self.agent.llm = new
        old.close()

    def rebind_jev(self) -> None:
        new = TypeSafeClient(
            api_key=self.settings.jev_api_key,
            model=self.settings.jev_model,
            timeout=60.0,
        )
        new.__enter__()
        old = self.jev_client
        self.jev_client = new
        self.agent.decisions.client = new
        self.agent.decisions.model = self.settings.jev_model
        if old is not None:
            old.__exit__(None, None, None)

    def close(self) -> None:
        self.agent.llm.close()
        if self.jev_client is not None:
            self.jev_client.__exit__(None, None, None)

    def remember(self) -> None:
        save_prefs(
            {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "fast_model": self.settings.fast_model or "",
                "fast_provider": self.settings.fast_provider or "",
                "jev_model": self.settings.jev_model,
            },
            path=self.prefs_file,
        )


def llm_from_settings(settings, meter) -> LLM:
    powerful = Profile(settings.provider, settings.model, settings.api_key, settings.base_url)
    fast = None
    if settings.fast_model and settings.fast_provider:
        fast = Profile(
            settings.fast_provider,
            settings.fast_model,
            settings.fast_api_key,
            settings.fast_base_url,
        )
    return LLM(powerful, fast, meter)


def handle_command(app: App, line: str) -> bool:
    """Run one slash command. Return False when the session should end."""
    try:
        parts = shlex.split(line)
    except ValueError:
        app.ui.error("Could not parse that command.")
        return True
    if not parts:
        return True
    command, *args = parts
    command = command.lower()
    if command in {"/quit", "/exit"}:
        return False
    if command == "/help":
        app.ui.help()
        return True
    if command == "/status":
        _show_status(app)
        return True
    if command == "/doctor":
        refresh_settings(app.settings)
        app.ui.doctor(app.settings)
        return True
    if command == "/pwd":
        app.ui.info(str(app.settings.workspace))
        return True
    if command == "/clear":
        app.agent.prior.clear()
        app.ui.clear()
        _show_status(app)
        return True
    if command == "/yes":
        app.settings.assume_yes = not app.settings.assume_yes
        app.ui.assume_yes = app.settings.assume_yes
        app.ui.info("auto-approve on" if app.settings.assume_yes else "auto-approve off")
        return True
    if command == "/verbose":
        app.settings.verbose = not app.settings.verbose
        app.ui.verbose = app.settings.verbose
        app.ui.info("verbose on" if app.settings.verbose else "verbose off")
        return True
    if command == "/read-only":
        app.settings.read_only = not app.settings.read_only
        app.ui.info("read-only on" if app.settings.read_only else "read-only off")
        return True
    if command == "/confirm":
        app.settings.confirm_mutations = not app.settings.confirm_mutations
        app.ui.info("confirm every edit on" if app.settings.confirm_mutations else "confirm every edit off")
        return True
    if command in {"/provider", "/backend"}:
        _set_provider(app, args)
        return True
    if command == "/model":
        _set_model(app, args)
        return True
    if command == "/fast":
        _set_fast(app, args)
        return True
    if command == "/jev":
        _set_jev(app, args)
        return True
    if command == "/key":
        _set_key(app, args)
        return True
    if command == "/cd":
        _set_workspace(app, args)
        return True
    app.ui.error(f"Unknown command {command}. Try /help.")
    return True


def run_repl(app: App) -> int:
    _show_status(app)
    history_path = user_config_dir() / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=_completer(),
        complete_while_typing=True,
        style=_PROMPT_STYLE,
    )
    app.ui.session = session
    while True:
        try:
            if sys.stdin.isatty():
                line = session.prompt(
                    HTML("<arrow>› </arrow>"),
                    bottom_toolbar=lambda: _toolbar(app),
                ).strip()
            else:
                line = input("jevcode> ").strip()
        except (EOFError, KeyboardInterrupt):
            app.ui.console.print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            try:
                if not handle_command(app, line):
                    return 0
            except KeyboardInterrupt:
                app.ui.error("interrupted")
            continue
        try:
            app.agent.run_task(line)
        except KeyboardInterrupt:
            app.ui.error("interrupted")


def _show_status(app: App) -> None:
    fast = ""
    if app.settings.fast_model and app.settings.fast_provider:
        fast = f"{app.settings.fast_provider} {app.settings.fast_model}"
    app.ui.banner(
        short_path(app.settings.workspace),
        app.settings.provider,
        app.settings.model,
        app.settings.jev_model,
        fast,
    )


def _set_provider(app: App, args: list[str]) -> None:
    if not args:
        app.ui.info(f"{app.settings.provider}  {app.settings.model}")
        return
    provider = args[0].lower()
    if provider not in PROVIDERS:
        app.ui.error(f"Provider must be one of: {', '.join(PROVIDERS)}")
        return
    previous = app.settings.provider
    app.settings.provider = provider
    if len(args) > 1:
        app.settings.model = " ".join(args[1:])
    elif previous != provider:
        app.settings.model = DEFAULT_MODELS[provider]
        app.settings.fast_model = None
        app.settings.fast_provider = None
    refresh_settings(app.settings)
    app.rebuild_writer()
    app.remember()
    if provider != "ollama" and not app.settings.api_key:
        app.ui.error(f"No {provider} key yet. Set one with /key {provider}")
        return
    app.ui.info(f"writer {provider}  {app.settings.model}")


def _set_model(app: App, args: list[str]) -> None:
    if not args:
        app.ui.info(f"{app.settings.provider}  {app.settings.model}")
        return
    app.settings.model = " ".join(args)
    app.rebuild_writer()
    app.remember()
    app.ui.info(f"model {app.settings.model}")


def _set_fast(app: App, args: list[str]) -> None:
    if not args:
        if app.settings.fast_model:
            app.ui.info(f"{app.settings.fast_provider}  {app.settings.fast_model}")
        else:
            app.ui.info("fast routing is off")
        return
    if args[0].lower() == "off":
        app.settings.fast_model = None
        app.settings.fast_provider = None
        app.rebuild_writer()
        app.remember()
        app.ui.info("fast routing off")
        return
    model = " ".join(args)
    if model == app.settings.model and (app.settings.fast_provider in {None, app.settings.provider}):
        app.ui.error("The fast model is the same as the writer. Pick a smaller one, or /fast off.")
        return
    app.settings.fast_model = model
    app.settings.fast_provider = app.settings.provider
    refresh_settings(app.settings)
    app.rebuild_writer()
    app.remember()
    app.ui.info(f"fast {app.settings.provider}  {model}")


def _set_jev(app: App, args: list[str]) -> None:
    if not args:
        app.ui.info(app.settings.jev_model)
        return
    app.settings.jev_model = args[0]
    app.agent.decisions.model = app.settings.jev_model
    app.remember()
    app.ui.info(f"jev {app.settings.jev_model}")


def _set_key(app: App, args: list[str]) -> None:
    if not args:
        app.ui.doctor(app.settings)
        return
    kind = args[0].lower()
    if kind not in {"openai", "anthropic", "grok", "jev"}:
        app.ui.error("Use /key openai, anthropic, grok, or jev.")
        return
    if not sys.stdin.isatty():
        app.ui.error("Setting a key needs a terminal so the secret is not saved in history.")
        return
    try:
        value = prompt_secret(f"{kind} api key: ", is_password=True).strip()
    except (EOFError, KeyboardInterrupt):
        app.ui.console.print()
        return
    if not value:
        app.ui.error("No key entered.")
        return
    save_user_key(kind, value)
    refresh_settings(app.settings)
    if kind == "jev":
        app.rebind_jev()
    else:
        app.rebuild_writer()
    app.ui.info(f"{kind} key saved in ~/.jevcode/.env")


def _set_workspace(app: App, args: list[str]) -> None:
    if not args:
        app.ui.info(str(app.settings.workspace))
        return
    raw = Path(args[0]).expanduser()
    target = raw if raw.is_absolute() else app.settings.workspace / raw
    try:
        target = target.resolve()
    except OSError as error:
        app.ui.error(str(error))
        return
    if not target.is_dir():
        app.ui.error(f"Not a directory: {target}")
        return
    app.settings.workspace = target
    app.agent.workspace = Workspace(target, allow_outside=app.settings.allow_outside)
    app.ui.info(f"workspace {short_path(target)}")


def _toolbar(app: App) -> HTML:
    workspace = short_path(app.settings.workspace)
    return HTML(
        f" <b>{app.settings.provider}</b> {app.settings.model}"
        f"   <b>cwd</b> {workspace}"
        "   <b>/help</b> "
    )


def _completer() -> NestedCompleter:
    providers = {name: None for name in PROVIDERS}
    return NestedCompleter.from_nested_dict(
        {
            "/help": None,
            "/status": None,
            "/provider": providers,
            "/backend": providers,
            "/model": None,
            "/fast": {"off": None},
            "/key": {"openai": None, "anthropic": None, "grok": None, "jev": None},
            "/jev": None,
            "/cd": None,
            "/pwd": None,
            "/clear": None,
            "/yes": None,
            "/confirm": None,
            "/read-only": None,
            "/verbose": None,
            "/doctor": None,
            "/quit": None,
            "/exit": None,
        }
    )


_PROMPT_STYLE = Style.from_dict(
    {
        "arrow": "bold ansicyan",
        "bottom-toolbar": "noreverse",
        "bottom-toolbar.text": "#9aa4b2",
    }
)
