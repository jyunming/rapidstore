#!/usr/bin/env bash
# Install TurboQuantDB git hooks by pointing git at scripts/hooks/.
#
# Usage: bash scripts/install_hooks.sh
# Run from the repo root.

set -euo pipefail

HOOKS_DIR="scripts/hooks"

if [ ! -f "$HOOKS_DIR/pre-commit" ]; then
    echo "ERROR: $HOOKS_DIR/pre-commit not found. Run from the repo root." >&2
    exit 1
fi

chmod +x "$HOOKS_DIR/pre-commit"
git config core.hooksPath "$HOOKS_DIR"
echo "✓ Git hooks installed (core.hooksPath = $HOOKS_DIR)"
echo "  pre-commit hook will run on every commit."
