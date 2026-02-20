Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "==> qanneal setup (CPU)"

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
  Write-Host "CMake not found. Install with: winget install Kitware.CMake"
  exit 1
}

python -V
python -m pip install -U pip setuptools wheel
python -m pip install -U scikit-build-core pybind11 numpy

# Build + install the Python package (includes native extension).
python -m pip install . --no-build-isolation

python -c "import qanneal, qanneal._qanneal; print('qanneal version:', qanneal.__version__)"

Write-Host "==> Setup complete"
