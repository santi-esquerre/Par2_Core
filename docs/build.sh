#!/usr/bin/env bash
# docs/build.sh — Build Par2_Core documentation (Doxygen + Sphinx).
#
# Usage:
#   cd docs && bash build.sh          # full build
#   cd docs && bash build.sh clean    # remove build artifacts
#
# Prerequisites (pip install -r docs/requirements.txt):
#   sphinx, breathe, sphinx-rtd-theme, myst-parser, doxygen

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning docs build artifacts..."
    rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/_doxygen"
    echo "Done."
    exit 0
fi

echo "=== Step 1/2: Doxygen (XML for Breathe) ==="
cd "$REPO_ROOT"
doxygen docs/Doxyfile

echo ""
echo "=== Step 2/2: Sphinx (HTML) ==="
cd "$SCRIPT_DIR"
sphinx-build -b html source build/html

echo ""
echo "Documentation built successfully."
echo "Open: file://$SCRIPT_DIR/build/html/index.html"
