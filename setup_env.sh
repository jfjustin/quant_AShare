#!/usr/bin/env bash
# One-shot environment setup for ashare_quant.
# Usage:  bash setup_env.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SDK="$(cd "$HERE/.." && pwd)"          # EMQuantAPI_Python root
PY="$SDK/python3"

echo "==> 1/4  Registering EmQuantAPI on the Python path (.pth)"
python3 "$PY/installEmQuantAPI.py"

echo "==> 2/4  Clearing macOS quarantine on native libraries (fixes dlopen policy block)"
xattr -dr com.apple.quarantine "$SDK" 2>/dev/null || true

echo "==> 3/4  Installing Python dependencies"
python3 -m pip install -q -r "$HERE/requirements.txt"

echo "==> 4/4  Creating config.yaml from template (edit it with your Choice login)"
[ -f "$HERE/config.yaml" ] || cp "$HERE/config.example.yaml" "$HERE/config.yaml"

echo
echo "Done. Next:"
echo "  - Put your East Money Choice username/password in config.yaml (or export EM_USERNAME / EM_PASSWORD)"
echo "  - Or activate once with the GUI tool:  $PY/libs/mac/loginactivator_mac"
echo
echo "Smoke tests (no Choice login needed):"
echo "  python3 tests/test_core.py"
echo "  python3 scripts/run_sentiment.py --seed"
