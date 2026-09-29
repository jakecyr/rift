"""Command line entrypoint."""

from __future__ import annotations

from typesafe_sdk import TypeSafeClient

from rift.agent import Agent
from rift.config import fail, load_settings, print_doctor, validate
from rift.decisions import DecisionLayer
from rift.llm import UsageMeter
from rift.repl import App, llm_from_settings, run_repl
from rift.tools import Workspace
from rift.ui import UI


def main(argv: list[str] | None = None) -> int:
    settings = load_settings(argv)
    if settings.doctor:
        print_doctor(settings)
        return 0
    errors = validate(settings)
    if errors:
        fail(errors)
    meter = UsageMeter()
    llm = llm_from_settings(settings, meter)
    ui = UI(verbose=settings.verbose, assume_yes=settings.assume_yes)
    workspace = Workspace(settings.workspace, allow_outside=settings.allow_outside)
    client = TypeSafeClient(api_key=settings.jev_api_key, model=settings.jev_model, timeout=60.0)
    client.__enter__()
    app = App(
        settings,
        Agent(
            workspace=workspace,
            decisions=DecisionLayer(client=client, meter=meter, model=settings.jev_model),
            llm=llm,
            ui=ui,
            settings=settings,
        ),
        ui,
        client,
    )
    try:
        if settings.task:
            app.agent.run_task(settings.task)
            return 0
        return run_repl(app)
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
