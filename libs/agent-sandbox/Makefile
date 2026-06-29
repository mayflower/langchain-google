.PHONY: all check_imports coverage format format_diff integration_test integration_tests lint lint_diff lint_package lint_tests lint_examples test tests help

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE = tests/integration_tests/

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest $(EXTRA_PYTEST_ARGS) $(TEST_FILE)

check_imports: $(shell find langchain_google_agent_sandbox -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/agent-sandbox --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_google_agent_sandbox
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test
lint_examples: PYTHON_FILES=examples
lint_examples: MYPY_CACHE=.mypy_cache_examples

lint lint_diff lint_package lint_tests lint_examples:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

# Run unit tests and generate a coverage report.
coverage:
	uv run --with pytest-cov pytest --cov \
		--cov-report xml \
		--cov-report html \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)
	open htmlcov/index.html

help:
	@echo '----'
	@echo 'check_imports                - check imports'
	@echo 'coverage                     - run unit tests and generate coverage reports'
	@echo 'format                       - run code formatters'
	@echo 'format_diff                  - run code formatters on changed files'
	@echo 'lint                         - run linters'
	@echo 'lint_diff                    - run linters on changed files'
	@echo 'lint_package                 - lint package source'
	@echo 'lint_tests                   - lint tests'
	@echo 'lint_examples                - lint examples'
	@echo 'test                         - run unit tests with sockets disabled'
	@echo 'integration_test             - run environment-gated integration tests'
