# loom-code — release helper.
#
# Daily commits / pushes are plain git. This Makefile only adds a
# `release` target — the explicit "publish to PyPI" flag. Mirrors the
# loomflow repo's release flow.
#
# Usage:
#   make release BUMP=patch    # X.Y.Z → X.Y.(Z+1)
#   make release BUMP=minor    # X.Y.Z → X.(Y+1).0
#   make release BUMP=major    # X.Y.Z → (X+1).0.0
#
# What it does:
#   1. Refuses to run with a dirty working tree.
#   2. Runs the gates (ruff + pytest).
#   3. bump-my-version edits pyproject.toml + loom_code/__init__.py
#      atomically, commits, and tags v<new_version>.
#   4. Pushes the commit and the tag to origin.
#   5. The push of the v* tag fires .github/workflows/release.yml,
#      which builds and uploads to PyPI via trusted publishing.
#
# Prerequisites: bump-my-version + ruff (in the [dev] extras —
# `pip install -e '.[dev]'`).

.PHONY: release check show-version release-help

release:
	@if [ -z "$(BUMP)" ]; then \
	  echo ""; \
	  echo "  ✗ missing BUMP="; \
	  echo "  usage: make release BUMP=patch|minor|major"; \
	  echo ""; \
	  exit 1; \
	fi
	@if ! command -v bump-my-version >/dev/null 2>&1; then \
	  echo ""; \
	  echo "  ✗ bump-my-version not installed."; \
	  echo "    install with: pip install -e '.[dev]'"; \
	  echo ""; \
	  exit 1; \
	fi
	@echo "→ Pre-flight: working tree clean?"
	@git diff-index --quiet HEAD -- || (echo "  ✗ uncommitted changes; commit or stash first" && exit 1)
	@echo "→ Pre-flight: gates green?"
	@ruff check loom_code tests
	@pytest -q
	@echo "→ Bumping version ($(BUMP)) + committing + tagging..."
	bump-my-version bump $(BUMP) --verbose
	@echo "→ Pushing commit + tag to origin..."
	git push origin main
	git push origin --tags
	@echo ""
	@echo "  ✓ Released. PyPI workflow firing on the new tag."
	@echo "    Check: https://github.com/Anurich/loomflow-cli/actions"
	@echo ""

# Show what `make release BUMP=...` would do without changing anything.
check:
	@if [ -z "$(BUMP)" ]; then \
	  echo "usage: make check BUMP=patch|minor|major"; \
	  exit 1; \
	fi
	bump-my-version show-bump

show-version:
	@bump-my-version show current_version 2>/dev/null || \
	  grep '^version = ' pyproject.toml

release-help:
	@echo ""
	@echo "  make release BUMP=patch     # X.Y.Z → X.Y.(Z+1)"
	@echo "  make release BUMP=minor     # X.Y.Z → X.(Y+1).0"
	@echo "  make release BUMP=major     # X.Y.Z → (X+1).0.0"
	@echo ""
	@echo "  make check BUMP=...         # dry-run; show new version"
	@echo "  make show-version           # print current version"
	@echo ""
