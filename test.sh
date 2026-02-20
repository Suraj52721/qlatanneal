#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

PYBIND11_DIR="$(python -m pybind11 --cmakedir)"
cmake -S . -B build -DQANNEAL_BUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build
ctest --test-dir build

python -c "import qanneal; import qanneal._qanneal; print(qanneal.__version__)"
python examples/python/sa_multi.py
python examples/python/sqa_basic.py
python examples/python/metrics_plot.py
python examples/python/parallel_tempering.py
