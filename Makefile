# ==============================================================================
# TorchSig Models Project Makefile
# ==============================================================================

# --- Variables ---
# Use virtual environment Python if available, otherwise fall back to python3
PYTHON ?= python3
ifneq ($(wildcard .venv/bin/python),)
    PYTHON := .venv/bin/python
endif
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest

# Directories
SRC_DIR = torchsig_models
TEST_DIR = tests

# --- Phony Targets ---
# .PHONY tells Make that these are "commands" and not "files" on disk.
# This prevents conflicts if you have a folder named 'tests' or 'clean'.
.PHONY: install test test-cov test-notebooks test-notebooks-clean clean-notebooks \
		lint format fix clean help check-lfs

# Default target: show help
help:
	@echo "TorchSig Project Orchestration"
	@echo "-----------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Install dependencies and the package in editable mode"
	@echo "  test       Run all tests in the $(TEST_DIR) directory"
	@echo "  test-cov   Run tests and generate a coverage report"
	@echo "  test-notebooks"
	@echo "             Executes all Jupyter notebooks to verify they run without errors."
	@echo "  test-notesbooks-clean"
	@echo "             Removes stamp files created by notebook execution"
	@echo "  clean-notebooks"
	@echo "             Removes all output from executed notebooks"
	@echo "  lint       Run static analysis (ruff check)"
	@echo "  format     Auto-format code (ruff format)"
	@echo "  fix        Apply automatic fixes and formatting (ruff check --fix)"
	@echo "  check-lfs  Checks for any remaining Git LFS references in the repository"
	@echo "  clean      Remove caches, test artifacts, model files, and build outputs"
	@echo "  help       Show this help message"

# --- Targets ---

# Setup
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# Testing
TEST_MODE ?= fast

test:
	@echo "Running tests ($(TEST_MODE) mode)..."
	$(PYTEST) $(TEST_DIR) --test-mode=$(TEST_MODE)

test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

# test-notebooks target
NOTEBOOKS        := $(wildcard examples/*.ipynb)
EXECUTED_STAMPS  := $(patsubst %.ipynb,%.executed,$(NOTEBOOKS))

test-notebooks: .cleaned $(EXECUTED_STAMPS)

# Changed this line: remove "examples/" from the dependency
%.executed: %.ipynb .cleaned
	jupyter trust $< || { echo "❌ $< is not a valid notebook"; exit 1; }
	jupyter nbconvert --inplace --execute $< || { rm -f $@; exit 1; }
	touch $@

test-notebooks-clean:
	rm -f $(EXECUTED_STAMPS) .cleaned

clean-notebooks:
	@if [ -n "$(NOTEBOOKS)" ]; then \
		echo "🧹 Running nbclean on all notebooks…"; \
		nb-clean clean $(NOTEBOOKS) || { echo "❌ nb-clean failed – see the messages above"; exit 1; }; \
	else \
		echo "No notebooks to clean."; \
	fi

# The *file* that tells make the cleaning has finished.
# It is touched only after the clean step succeeded.
.cleaned: clean-notebooks
	touch $@


# Quality Assurance
lint:
	@echo "Running Ruff check..."
	$(PYTHON) -m ruff check $(SRC_DIR)

format:
	@echo "Formatting with Ruff..."
	$(PYTHON) -m ruff format $(SRC_DIR)

fix:
	@echo "Automatically fixing lint errors..."
	$(PYTHON) -m ruff check --fix $(SRC_DIR)
	$(PYTHON) -m ruff format $(SRC_DIR)

# Maintenance
clean:
	@echo "Cleaning up..."
	# Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	# Remove pytest and coverage artifacts
	rm -rf .pytest_cache .coverage htmlcov
	# Remove temporary directories created by file-handler tests
	# (Adjust the pattern if your tests use a different tmp naming convention)
	rm -fr "/tmp/pytest-of-$(USER)"
	# Remove model files that shouldn't be in release
	find . -type f \( -name "*.pt" -o -name "*.pth" \) ! -path "./.venv/*" -delete
	# Remove other large/misc files not for release
	find . -type f \( -name "*.h5" -o -name "*.npz" -o -name "*.zarr" \) ! -path "./.pytest_cache/*" ! -path "./.venv/*" -delete
	# Remove coverage and test reports
	rm -f coverage.xml report.xml report.json
	# Remove egg-info
	rm -rf *.egg-info torchsig_models.egg-info
	@echo "Cleanup complete."

# Git LFS
.PHONY: check-lfs

check-lfs:
	@echo "🔍 Checking branch: $(shell git rev-parse --abbrev-ref HEAD) (post filter-repo)"
	@LFS_FOUND=0; \
	# 1. Check current .gitattributes \
	printf "\n1. Checking current .gitattributes...\n"; \
	if [ -f .gitattributes ]; then \
		if grep -q "filter=lfs" .gitattributes; then \
			printf "\033[0;31m   ❌ Current .gitattributes contains LFS rules\033[0m\n"; \
			LFS_FOUND=1; \
		else \
			printf "\033[0;32m   ✅ Current .gitattributes has no LFS rules\033[0m\n"; \
		fi; \
	else \
		printf "\033[0;32m   ✅ No .gitattributes file exists\033[0m\n"; \
	fi; \
	# 2. Check .gitattributes history \
	printf "\n2. Checking .gitattributes history...\n"; \
	if git log --all --oneline -- .gitattributes >/dev/null 2>&1; then \
		printf "   Found .gitattributes in history...\n"; \
		git log --all --pretty=format:'%H' -- .gitattributes | while read -r commit; do \
			if git show "$$commit:.gitattributes" 2>/dev/null | grep -q "filter=lfs"; then \
				SHORT_COMMIT=$${commit:0:7}; \
				printf "\033[0;31m   ❌ LFS rule found in .gitattributes (commit: $$SHORT_COMMIT)\033[0m\n"; \
				LFS_FOUND=1; \
			fi; \
		done; \
	else \
		printf "\033[0;32m   ✅ No .gitattributes in history\033[0m\n"; \
	fi; \
	# 3. Check for LFS pointers in current files \
	printf "\n3. Checking current files for LFS pointers...\n"; \
	git ls-tree -r HEAD --name-only 2>/dev/null | while read -r file; do \
		if [ "$$file" = ".gitattributes" ]; then continue; fi; \
		blob=$(git rev-parse "HEAD:$$file" 2>/dev/null); \
		if [ -n "$$blob" ]; then \
			content=$(git cat-file -p "$$blob" 2>/dev/null | head -1); \
			if [[ "$$content" == _* ]]; then \
				printf "\033[0;31m   ❌ LFS pointer found in current file: $$file\033[0m\n"; \
				LFS_FOUND=1; \
			fi; \
		fi; \
	done; \
	if [ "$$LFS_FOUND" -eq 0 ]; then \
		printf "\033[0;32m   ✅ No LFS pointers in current files\033[0m\n"; \
	fi; \
	# 4. Check Git LFS cache \
	printf "\n4. Checking Git LFS cache...\n"; \
	if [ -d .git/lfs/objects ]; then \
		LFS_OBJECTS=$$(find .git/lfs/objects -type f 2>/dev/null | wc -l); \
		if [ "$$LFS_OBJECTS" -gt 0 ]; then \
			printf "\033[0;31m   ❌ Found $$LFS_OBJECTS LFS objects in .git/lfs/objects\033[0m\n"; \
			LFS_FOUND=1; \
		else \
			printf "\033[0;32m   ✅ No LFS objects in cache\033[0m\n"; \
		fi; \
	else \
		printf "\033[0;32m   ✅ No LFS cache directory\033[0m\n"; \
	fi; \
	# 5. Check Git config \
	printf "\n5. Checking Git config...\n"; \
	if git config --get-regexp 'filter.lfs' >/dev/null 2>&1; then \
		printf "\033[0;31m   ❌ LFS filter found in Git config\033[0m\n"; \
		LFS_FOUND=1; \
		git config --get-regexp 'filter.lfs' | while read -r key value; do \
			printf "\033[0;31m      $$key=$$value\033[0m\n"; \
		done; \
	else \
		printf "\033[0;32m   ✅ No LFS in Git config\033[0m\n"; \
	fi; \
	# Final result \
	printf "\n=== Summary for $(shell git rev-parse --abbrev-ref HEAD) (post filter-repo) ===\n"; \
	if [ "$$LFS_FOUND" -eq 0 ]; then \
		printf "\033[0;32m✅ No active Git LFS objects or references found\033[0m\n"; \
		printf "\033[0;32m   (History may have had LFS, but it's been cleaned)\033[0m\n"; \
		exit 0; \
	else \
		printf "\033[0;31m❌ Active Git LFS references still exist\033[0m\n"; \
		printf "\033[1;33m   To fix:\033[0m\n"; \
		printf "\033[1;33m   1. Remove LFS config: git config --remove-section filter.lfs\033[0m\n"; \
		printf "\033[1;33m   2. Prune LFS cache: git lfs prune\033[0m\n"; \
		printf "\033[1;33m   3. If needed, run: git filter-repo --invert-paths --path .gitattributes\033[0m\n"; \
		exit 1; \
	fi

# ==============================================================================
# Notes:
# - Use 'make install' when first setting up the repo.
# - Use 'make test' before every commit.
# - Use 'make clean' if you encounter weird filesystem issues.
# ==============================================================================

