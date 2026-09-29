# rift

A coding agent for the repo in front of you. A frontier model writes. [Jev](https://typesafe.ai) decides. Hard rules stay in Python.

Jev picks the next action from the live tool menu, gates anything that can change the machine, notices when the loop is stuck, and says whether the task looks done. The writer model (OpenAI, Anthropic, Grok, or Ollama) only fills tool arguments and the final note. A separate check then proves the files actually exist.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Python 3.10 or newer. The decision client is the official `typesafe-sdk` package.

Put the virtualenv on your `PATH` once, then start it from any project:

```bash
export PATH="$HOME/code/my-assistant/.venv/bin:$PATH"
cd ~/some/other/project
rift
```

## Keys

Files are read first, then the process environment wins. A command the agent runs does not inherit those files.

From lowest priority to highest:

1. The `.env` next to this rift checkout. An editable install still finds it when you start `rift` from another folder.
2. `.env` in the folder you launched from, when that file is a different path.
3. `~/.rift/.env`, which is where `/key` saves a key.
4. The process environment.

```bash
TYPESAFE_API_KEY=...     # or JEV_API_KEY
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
OLLAMA_HOST=http://127.0.0.1:11434
```

```bash
rift --doctor
```

## Use it

`/model` and `--model` pick the writer. `/provider` (or `/backend`) switches OpenAI, Anthropic, Grok, or Ollama. Those choices are saved in `~/.rift/config.json`.

```bash
rift
rift --model gpt-4.1-mini "Fix the failing test"
rift --provider anthropic "Add retry logic to src/http.py"
rift --provider ollama --model qwen2.5-coder "Explain this repository"
rift --fast-model gpt-4.1-mini "Rename the helper and update callers"
```

Inside the prompt: `/model`, `/provider`, `/fast`, `/key`, `/cd`, `/clear`, `/doctor`, `/help`, `/quit`. `!ls` runs that command in the workspace and keeps the output for the next task. Up-arrow recalls history. Tab completes commands. `test` runs the project's test command instead of opening test files.

`AGENTS.md` and `CLAUDE.md` are loaded automatically: `~/.config/opencode/AGENTS.md`, `~/.claude/CLAUDE.md`, then each directory from your home folder down to the workspace. When a directory has both, both are included and the nearer file comes last.

`--yes` skips Jev's confirm prompts. It does not skip hard rules: `rm`, `git push`, `sudo`, secret files, and anything irreversible still ask. `--confirm` asks before every edit and shell command. `--read-only` removes write and shell from the menu.

## What one turn does

1. Jev chooses the next tool, and the model tier if you set `--fast-model`.
2. The writer fills arguments for that one tool.
3. Code blocks catastrophic commands and paths outside the workspace.
4. Jev gates edits and shell calls (`allow`, `confirm`, `block`).
5. The tool runs.
6. Every few steps Jev scores progress. When it picks done, code checks that the files are really there.

A repo-wide rename is one step. Several different edits go out together in one step.

Defaults are `gpt-6-astra`, `claude-sonnet-5-5`, `grok-4`, and `qwen2.5-coder`. Override them with `--model` or `/model`.
