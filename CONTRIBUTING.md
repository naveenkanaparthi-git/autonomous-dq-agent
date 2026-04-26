# Contributing to autonomous-dq-agent

## Development Setup

```bash
git clone https://github.com/naveenkanaparthi-git/autonomous-dq-agent.git
cd autonomous-dq-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Standards

- **Formatting**: `black src/ tests/ --line-length 88`
- **Linting**: `ruff check src/ tests/ --fix`
- **Type checking**: `mypy src/ --ignore-missing-imports --no-strict-optional`
- **Tests must pass**: `pytest tests/ -v`

## Adding an Expectation Type

1. Add the type to `ExpectationType` enum in `models/validation.py`
2. Implement a handler `_expect_<name>` in `DataValidator` in `core/validator.py`
3. Register the handler in `DataValidator._evaluate`'s `handler_map`
4. Add `SuiteBuilder` method in `services/suite_builder.py`
5. Write unit tests in `tests/unit/test_core.py`

## Pull Request Checklist

- [ ] `make all` passes (format + lint + typecheck + tests)
- [ ] New expectation types have unit tests for pass AND fail cases
- [ ] New features are documented in README.md
- [ ] Docstrings added for all new public functions/classes
