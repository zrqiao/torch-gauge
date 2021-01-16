.DEFAULT_GOAL := all
src = torch_gauge
isort = isort $(src)
black = black $(src)
autoflake = autoflake -ir --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables $(src)
mypy = mypy torch_gauge

.PHONY: install
install:
	pip install -e .

.PHONY: format
format:
	$(autoflake)
	$(isort)
	$(black)
#	$(mypy)

.PHONY: lint
lint:
	$(isort) --check-only
	$(black) --check

.PHONY: check-dist
check-dist:
	python setup.py check -ms
	python setup.py sdist
	twine check dist/*

.PHONY: mypy
mypy:
	$(mypy)

.PHONY: test
test:
	pytest -v torch_gauge

.PHONY: docs
docs:
	black -l 80 docs/examples
	python docs/build/evaluate_examples.py
	mkdocs build
