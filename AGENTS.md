# Project guide

## Layout

- `src/rift/` contains the application package. The `rift` command calls `rift.cli:main`.
- `tests/` contains the test suite, including `tests/test_rift.py` and `tests/rift/`.
- `pyproject.toml` defines the package and its runtime dependencies. Python 3.10 or newer is required.

## Tests

Run the test suite with:

```bash
python3 -m pytest
```

## Project behavior

- The writer model fills arguments and the final note; Jev selects actions and gates edits and shell calls. Keep hard safety rules in Python.
- `AGENTS.md` and `CLAUDE.md` instructions are loaded automatically, with nearer workspace instructions taking precedence over home-directory instructions.
