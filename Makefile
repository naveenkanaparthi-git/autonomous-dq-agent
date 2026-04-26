.PHONY: install format lint typecheck test all clean

install:
	pip install -e ".[dev]"

format:
	black src/ tests/ --line-length 88

lint:
	ruff check src/ tests/ --fix

typecheck:
	mypy src/ --ignore-missing-imports --no-strict-optional

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

all: format lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml
