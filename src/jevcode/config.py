"""Runtime configuration loaded from flags, saved preferences, and env files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jevcode import __version__

PROVIDERS = ("openai", "anthropic", "grok", "ollama")
DEFAULT_MODELS = {
    "openai": "gpt-6-astra",
    "anthropic": "claude-sonnet-5-5",
    "grok": "grok-4",
    "ollama": "qwen2.5-coder",
}


@dataclass
class Settings:
    workspace: Path
    provider: str
    model: str
    api_key: str
    base_url: str | None
    fast_provider: str | None
    fast_model: str | None
    fast_api_key: str
    fast_base_url: str | None
    jev_model: str
    jev_api_key: str
    max_steps: int
    max_llm_calls: int
    assume_yes: bool
    confirm_mutations: bool
    read_only: bool
    allow_outside: bool
    verbose: bool
    doctor: bool
    constraints: list[str] = field(default_factory=list)
    task: str = ""
    key_status: dict[str, bool] = field(default_factory=dict)
    env_files: list[str] = field(default_factory=list)


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevcode",
        description=(
            "Agentic coding CLI. A frontier model writes code. "
            "Jev chooses the next action, gates risky calls, and checks completion. "
            "Hard rules stay in code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  jevcode
  jevcode --provider anthropic "Add retry logic to src/http.py"
  jevcode --model gpt-4.1-mini "Fix the failing test"
  jevcode --provider ollama --model qwen2.5-coder "Explain this repository"
  jevcode --fast-model gpt-4.1-mini "Rename the helper and update callers"
  jevcode --doctor
"""
    )
    parser.add_argument("task", nargs="*", help="Task to run. Omit it to open the REPL.")
    parser.add_argument("--provider", choices=PROVIDERS, help="Model that writes code and tool arguments.")
    parser.add_argument("--model", help="Generation model id. Defaults depend on the provider.")
    parser.add_argument("--fast-provider", choices=PROVIDERS, help="Provider for the cheaper generation model.")
    parser.add_argument("--fast-model", help="Cheaper model. Jev routes between this and --model.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for the main model.")
    parser.add_argument("--jev-model", default="jev-latest", help="Jev model id (default: jev-latest).")
    parser.add_argument("--workspace", default=".", help="Directory the agent may edit (default: .).")
    parser.add_argument("--max-steps", type=int, default=25, help="Action limit for one task (default: 25).")
    parser.add_argument("--max-llm-calls", type=int, default=40, help="Generation-call cap for one task.")
    parser.add_argument("--constraint", action="append", default=[], help="Extra rule, repeatable.")
    parser.add_argument("--yes", action="store_true", help="Skip Jev confirm prompts. Hard rules still stop.")
    parser.add_argument("--confirm", action="store_true", help="Ask before every edit, write, and shell command.")
    parser.add_argument("--read-only", action="store_true", help="No edits, writes, or shell.")
    parser.add_argument("--allow-outside", action="store_true", help="Permit paths outside the workspace.")
    parser.add_argument("--verbose", action="store_true", help="Print Jev probabilities and request ids.")
    parser.add_argument("--doctor", action="store_true", help="Show which keys and models are configured.")
    parser.add_argument("--version", action="version", version=f"jevcode {__version__}")
    return parser


def load_settings(argv: list[str] | None = None) -> Settings:
    args = main_parser().parse_args(argv)
    cwd = Path.cwd()
    env_files = env_file_candidates(cwd, find_project_env(), user_env_path())
    env = merge_env(env_files, os.environ)
    prefs = load_prefs()
    provider, model = choose_model(
        provider_flag=args.provider,
        model_flag=args.model,
        prefs=prefs,
        env=env,
    )
    if args.fast_model:
        fast_model = None if args.fast_model == "off" else args.fast_model
        fast_provider = None if fast_model is None else (args.fast_provider or provider)
    elif args.fast_provider:
        fast_provider = args.fast_provider
        fast_model = prefs.get("fast_model") or None
    else:
        fast_model = prefs.get("fast_model") or None
        fast_provider = (prefs.get("fast_provider") or provider) if fast_model else None
    if fast_model == "":
        fast_model = None
        fast_provider = None
    workspace = Path(args.workspace).expanduser().resolve()
    if args.jev_model != "jev-latest":
        jev_model = args.jev_model
    elif prefs.get("jev_model"):
        jev_model = str(prefs["jev_model"])
    else:
        jev_model = "jev-latest"
    return _settings_from_env(
        env,
        env_files=env_files,
        provider=provider,
        model=model,
        workspace=workspace,
        fast_provider=fast_provider,
        fast_model=fast_model,
        jev_model=jev_model,
        base_url_override=args.base_url,
        max_steps=max(1, args.max_steps),
        max_llm_calls=max(1, args.max_llm_calls),
        assume_yes=args.yes,
        confirm_mutations=args.confirm,
        read_only=args.read_only,
        allow_outside=args.allow_outside,
        verbose=args.verbose,
        doctor=args.doctor,
        constraints=list(args.constraint),
        task=" ".join(args.task).strip(),
    )


def refresh_settings(settings: Settings) -> None:
    """Reload keys after /key or a provider change. The workspace stays put."""
    env_files = env_file_candidates(Path.cwd(), find_project_env(), user_env_path())
    env = merge_env(env_files, os.environ)
    updated = _settings_from_env(
        env,
        env_files=env_files,
        provider=settings.provider,
        model=settings.model,
        workspace=settings.workspace,
        fast_provider=settings.fast_provider,
        fast_model=settings.fast_model,
        jev_model=settings.jev_model,
        base_url_override=settings.base_url,
        max_steps=settings.max_steps,
        max_llm_calls=settings.max_llm_calls,
        assume_yes=settings.assume_yes,
        confirm_mutations=settings.confirm_mutations,
        read_only=settings.read_only,
        allow_outside=settings.allow_outside,
        verbose=settings.verbose,
        doctor=settings.doctor,
        constraints=list(settings.constraints),
        task=settings.task,
    )
    for name in (
        "api_key",
        "base_url",
        "fast_api_key",
        "fast_base_url",
        "jev_api_key",
        "key_status",
        "env_files",
    ):
        setattr(settings, name, getattr(updated, name))


def _settings_from_env(env: dict[str, str], env_files: list[Path], **kwargs: object) -> Settings:
    provider = str(kwargs["provider"])
    fast_provider = kwargs["fast_provider"]
    fast_provider_name = str(fast_provider) if fast_provider else None
    base_override = kwargs["base_url_override"]
    base_override = str(base_override) if base_override else None
    return Settings(
        workspace=kwargs["workspace"],  # type: ignore[arg-type]
        provider=provider,
        model=str(kwargs["model"]),
        api_key=_provider_key(provider, env),
        base_url=_provider_base_url(provider, env, base_override),
        fast_provider=fast_provider_name,
        fast_model=str(kwargs["fast_model"]) if kwargs["fast_model"] else None,
        fast_api_key=_provider_key(fast_provider_name, env) if fast_provider_name else "",
        fast_base_url=_provider_base_url(fast_provider_name, env, None) if fast_provider_name else None,
        jev_model=str(kwargs["jev_model"]),
        jev_api_key=env.get("TYPESAFE_API_KEY") or env.get("JEV_API_KEY") or "",
        max_steps=int(kwargs["max_steps"]),  # type: ignore[arg-type]
        max_llm_calls=int(kwargs["max_llm_calls"]),  # type: ignore[arg-type]
        assume_yes=bool(kwargs["assume_yes"]),
        confirm_mutations=bool(kwargs["confirm_mutations"]),
        read_only=bool(kwargs["read_only"]),
        allow_outside=bool(kwargs["allow_outside"]),
        verbose=bool(kwargs["verbose"]),
        doctor=bool(kwargs["doctor"]),
        constraints=list(kwargs["constraints"]),  # type: ignore[arg-type]
        task=str(kwargs["task"]),
        key_status=_key_status(env),
        env_files=[str(path) for path in env_files if path.is_file()],
    )


def validate(settings: Settings) -> list[str]:
    errors: list[str] = []
    if not settings.workspace.is_dir():
        errors.append(f"workspace does not exist: {settings.workspace}")
    if not settings.jev_api_key:
        errors.append("Set TYPESAFE_API_KEY or JEV_API_KEY. Get one at console.typesafe.ai/settings/keys.")
    if settings.provider != "ollama" and not settings.api_key:
        errors.append(
            f"No API key for {settings.provider} ({_key_name(settings.provider)}). "
            "Pass --provider openai|anthropic|grok|ollama, or set the matching key. "
            "Ollama does not need a key."
        )
    if settings.fast_provider and settings.fast_provider != "ollama" and not settings.fast_api_key:
        errors.append(
            f"Set the API key for --fast-provider {settings.fast_provider} ({_key_name(settings.fast_provider)})."
        )
    if settings.fast_model and settings.fast_provider == settings.provider and settings.fast_model == settings.model:
        errors.append("--fast-model is the same as --model, so there is nothing to route between.")
    return errors


def print_doctor(settings: Settings) -> None:
    print(f"workspace: {settings.workspace}")
    print(f"jev:       {settings.jev_model}  key={'set' if settings.jev_api_key else 'missing'}")
    print(
        f"writer:    {settings.provider} {settings.model}  "
        f"key={'set' if settings.api_key or settings.provider == 'ollama' else 'missing'}"
    )
    if settings.fast_model and settings.fast_provider:
        fast_key = "set" if settings.fast_api_key or settings.fast_provider == "ollama" else "missing"
        print(f"fast:      {settings.fast_provider} {settings.fast_model}  key={fast_key}")
    else:
        print("fast:      off (pass --fast-model to let Jev route cheap vs strong)")
    for name, present in settings.key_status.items():
        print(f"env {name}: {'set' if present else 'missing'}")
    if settings.env_files:
        print("env files:")
        for path in settings.env_files:
            print(f"  {path}")
    if settings.provider == "ollama":
        print(f"ollama:    {settings.base_url}")


def user_config_dir() -> Path:
    return Path.home() / ".jevcode"


def user_env_path() -> Path:
    return user_config_dir() / ".env"


def prefs_path() -> Path:
    return user_config_dir() / "config.json"


def load_prefs(path: Path | None = None) -> dict:
    file = path or prefs_path()
    if not file.is_file():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(updates: dict, path: Path | None = None) -> None:
    file = path or prefs_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    data = load_prefs(file)
    data.update(updates)
    file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    file.chmod(0o600)


def save_user_key(kind: str, value: str, path: Path | None = None) -> None:
    names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY",
        "jev": "JEV_API_KEY",
    }
    if kind not in names:
        raise ValueError(kind)
    file = path or user_env_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    existing = _parse_env_file(file)
    existing[names[kind]] = value
    if kind == "jev":
        existing["TYPESAFE_API_KEY"] = value
    lines = [f"{key}={existing[key]}" for key in existing]
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    file.chmod(0o600)


def choose_model(
    provider_flag: str | None,
    model_flag: str | None,
    prefs: dict,
    env: dict[str, str],
) -> tuple[str, str]:
    saved_provider = prefs.get("provider") if isinstance(prefs.get("provider"), str) else None
    saved_model = prefs.get("model") if isinstance(prefs.get("model"), str) else None
    provider = provider_flag or saved_provider or _infer_provider(env)
    if provider not in PROVIDERS:
        provider = "openai"
    if model_flag:
        return provider, model_flag
    if saved_model and provider == (provider_flag or saved_provider or provider):
        if provider_flag and saved_provider and provider_flag != saved_provider:
            return provider, DEFAULT_MODELS[provider]
        return provider, saved_model
    return provider, DEFAULT_MODELS[provider]


def env_file_candidates(cwd: Path, project_env: Path | None, home_env: Path) -> list[Path]:
    """Later files override earlier ones. Keys saved in ~/.jevcode win over the repo file."""
    ordered: list[Path] = []
    if project_env is not None:
        ordered.append(project_env)
    cwd_env = cwd / ".env"
    if project_env is None or not _same_file(cwd_env, project_env):
        ordered.append(cwd_env)
    ordered.append(home_env)
    return ordered


def merge_env(files: list[Path], process: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in files:
        merged.update(_parse_env_file(path))
    for key, value in process.items():
        if value:
            merged[key] = value
    return merged


def find_project_env() -> Path | None:
    """The .env next to this checkout, so keys work when jevcode is started elsewhere."""
    start = Path(__file__).resolve()
    for parent in start.parents:
        manifest = parent / "pyproject.toml"
        if not manifest.is_file() or not _is_jevcode_project(manifest):
            continue
        env_path = parent / ".env"
        return env_path if env_path.is_file() else None
    return None


def _is_jevcode_project(manifest: Path) -> bool:
    try:
        text = manifest.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return 'name = "jevcode"' in text or "name = 'jevcode'" in text


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _key_status(env: dict[str, str]) -> dict[str, bool]:
    return {
        "TYPESAFE_API_KEY": bool(env.get("TYPESAFE_API_KEY")),
        "JEV_API_KEY": bool(env.get("JEV_API_KEY")),
        "OPENAI_API_KEY": bool(env.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(env.get("ANTHROPIC_API_KEY")),
        "XAI_API_KEY": bool(env.get("XAI_API_KEY") or env.get("GROK_API_KEY")),
        "OLLAMA_HOST": bool(env.get("OLLAMA_HOST")),
    }


def _infer_provider(env: dict[str, str]) -> str:
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("XAI_API_KEY") or env.get("GROK_API_KEY"):
        return "grok"
    if env.get("OLLAMA_HOST"):
        return "ollama"
    return "openai"


def _provider_key(provider: str, env: dict[str, str]) -> str:
    if provider == "openai":
        return env.get("OPENAI_API_KEY", "")
    if provider == "anthropic":
        return env.get("ANTHROPIC_API_KEY", "")
    if provider == "grok":
        return env.get("XAI_API_KEY") or env.get("GROK_API_KEY") or ""
    if provider == "ollama":
        return env.get("OLLAMA_API_KEY", "")
    return ""


def _key_name(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY or GROK_API_KEY",
        "ollama": "none",
    }[provider]


def _provider_base_url(provider: str, env: dict[str, str], override: str | None) -> str | None:
    if provider == "anthropic":
        return None
    if override and provider == "openai":
        return override.rstrip("/")
    if provider == "openai":
        return (env.get("OPENAI_BASE_URL") or "").rstrip("/") or None
    if provider == "grok":
        return "https://api.x.ai/v1"
    if provider == "ollama":
        return (env.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def fail(messages: list[str]) -> None:
    for message in messages:
        print(message, file=sys.stderr)
    raise SystemExit(2)
