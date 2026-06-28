@echo off
setlocal

echo ==> qanneal setup (CPU) - Windows

where cmake >nul 2>nul
if errorlevel 1 (
  echo CMake not found. Install with: winget install Kitware.CMake
  exit /b 1
)

python -V
python -m pip install -U pip setuptools wheel
python -m pip install -U scikit-build-core pybind11 numpy

python -m pip install . --no-build-isolation

python -c "import qanneal, qanneal._qanneal; print('qanneal version:', qanneal.__version__)"

echo ==> Setup complete
endlocal
