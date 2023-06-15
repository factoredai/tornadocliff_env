# A Self-Documenting Makefile: http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
SHELL = /bin/bash
OS = $(shell uname | tr A-Z a-z)

.PHONY: test
test: ## Run tests
	poetry run pytest --cov=patterns tests/

.PHONY: clean
clean: ## Cleans project folder mainly cache
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf cache-directory
	rm -rf cache
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -f coverage.xml
	find . -name ".DS_Store" -delete

# .PHONY: new-pattern
# new-pattern: ## Creates folder to implement new pattern
# 	poetry run python scripts/gen_new_pattern.py

.PHONY: lint
lint: ## Checks code linting
	poetry run black --check .
	poetry run isort --check-only .
	make lint-types

.PHONY: format
format: ## Formats code
	poetry run black .
	poetry run isort .

.PHONY: test-env
test-env: ## Test env
	python -m src.tornado_cliff.tornadocliffenv

.PHONY: help
.DEFAULT_GOAL := help
help:
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'
