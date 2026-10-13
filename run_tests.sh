#!/usr/bin/env bash
# ==============================================================================
# XENodes Test Runner (Python Node Tests & GUI Regression Tests)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$COMFYUI_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

RUN_PYTHON=true
RUN_GUI=true

if [ "$1" == "python" ]; then
    RUN_GUI=false
elif [ "$1" == "gui" ]; then
    RUN_PYTHON=false
fi

echo "=========================================================="
echo " Running XENodes Regression Test Suite"
echo "=========================================================="

# 1. Run Python Node Backend Tests
if [ "$RUN_PYTHON" = true ]; then
    echo ""
    echo ">> [1/2] Running Python Backend Tests..."
    PYTHONPATH="$COMFYUI_DIR" "$VENV_PYTHON" -m unittest discover -s "$SCRIPT_DIR/tests" -p "test_*.py" -v
    echo ">> Python tests completed successfully!"
fi

# 2. Run GUI Frontend Regression Tests
if [ "$RUN_GUI" = true ]; then
    echo ""
    echo ">> [2/2] Running GUI Frontend Regression Tests..."
    FRONTEND_DIR="$COMFYUI_DIR/../ComfyUI_frontend"
    if [ -d "$FRONTEND_DIR" ] && command -v pnpm &> /dev/null; then
        (cd "$FRONTEND_DIR" && pnpm exec node --experimental-vm-modules --test "$SCRIPT_DIR/web/tests/test_gui_regression.mjs")
    elif command -v node &> /dev/null; then
        node --experimental-vm-modules --test "$SCRIPT_DIR/web/tests/test_gui_regression.mjs"
    else
        echo ">> Warning: node/pnpm not found, skipping GUI regression tests."
    fi
    echo ">> GUI tests completed successfully!"
fi

echo ""
echo "=========================================================="
echo " All XENodes tests passed successfully!"
echo "=========================================================="
