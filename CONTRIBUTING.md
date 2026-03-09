# Contributing to Nexus AI

Hey, thanks for checking this out! This is mostly my solo project but contributions are welcome.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Python 3.10+, type hints where it makes sense
- `black` for formatting, `ruff` for linting
- Keep docstrings short - one-liners are fine for simple functions

```bash
black src/ tests/
ruff check src/ tests/
```

## Making Changes

1. Fork + clone
2. Create a branch (`git checkout -b fix/your-fix`)
3. Make changes, add tests if needed
4. Make sure `pytest tests/ -v` passes
5. Open a PR

## What I Look For in PRs

- Does it work? Tests pass?
- Is the code readable?
- No unnecessary complexity

If you have questions, just open an issue. Thanks!
