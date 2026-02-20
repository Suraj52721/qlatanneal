#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "==> qanneal setup (CPU)"

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake not found. Install it first (macOS: brew install cmake)."
  exit 1
fi

python -V
python -m pip install -U pip setuptools wheel
python -m pip install -U scikit-build-core pybind11 numpy

# Build + install the Python package (includes native extension).
python -m pip install . --no-build-isolation

python - <<'PY'
import qanneal
import qanneal._qanneal
print("qanneal version:", qanneal.__version__)
PY

echo "==> Setup complete"
