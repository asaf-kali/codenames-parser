PYTHON_TEST_COMMAND=pytest
ifeq ($(OS),Windows_NT)
	OPEN_FILE_COMMAND=start
	DEL_COMMAND=del
else
	OPEN_FILE_COMMAND=xdg-open
	DEL_COMMAND=gio trash
endif
SYNC=--sync
.PHONY: build

# gif arguments
CASE=cut_rotated
FPS=1.5

# Install

upgrade-pip:
	pip install --upgrade pip

install-ci: upgrade-pip
	pip install poetry==1.8.3
	poetry config virtualenvs.create false

install-run:
	poetry install --only main

install-test:
	poetry install --only main --only test

install-lint:
	poetry install --only lint

install-dev: upgrade-pip
	poetry install --all-extras	$(SYNC)
	pre-commit install

install: lock-check install-dev lint cover-fast

# Poetry

lock:
	poetry lock --no-update

lock-check:
	poetry check --lock

# Test

test:
	python -m $(PYTHON_TEST_COMMAND) $(PYTEST_EXTRA)

test-fast:
	@make test PYTEST_EXTRA="-m 'not slow'"

cover-base:
	coverage run -m $(PYTHON_TEST_COMMAND) $(PYTEST_EXTRA)
	coverage report

cover-xml: cover-base
	coverage xml

cover-html: cover-base
	coverage html

cover: cover-html
	$(OPEN_FILE_COMMAND) htmlcov/index.html &
	$(DEL_COMMAND) .coverage*

cover-fast:
	@make cover PYTEST_EXTRA="-m 'not slow'"

# Packaging

build:
	$(DEL_COMMAND) -f dist/*
	poetry build

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

build-and-upload: build upload

# Video

gif:
	bash ./scripts/create_gif.sh $(CASE) $(FPS)
	$(OPEN_FILE_COMMAND) exports/$(CASE).gif

# Semantic release

semrel:
	@echo "Releasing version..."
	semantic-release version

semrel-dev:
	@echo "Releasing dev version..."
	semantic-release version --no-commit --no-push
	# Replace "-dev.1" with epoch time in version
	sed -i 's/-dev.1/.dev.$(shell date +%s)/g' pyproject.toml
	make build

# Lint

format:
	black .
	isort .
	ruff . --fix

check-ruff:
	ruff .

check-black:
	black --check .

check-isort:
	isort --check .

check-mypy:
	mypy .

check-pylint:
	pylint codenames_parser/ --fail-under=10

lint: format
	pre-commit run --all-files
	@make check-pylint --no-print-directory

# Quick and dirty

wip: format
	git add .
	git commit -m "Auto commit." --no-verify

amend: format
	git add .
	git commit --amend --no-edit --no-verify
