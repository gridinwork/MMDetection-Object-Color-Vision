@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
echo ============================================
echo MMDetection Object ^& Color Vision Studio
echo ============================================

where py >nul 2>&1
if errorlevel 1 (
  echo Python launcher py.exe was not found.
  pause
  exit /b 1
)

py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 (
  echo Python 3.11 is required.
  echo MMCV 2.1 CUDA wheels used by MMDetection 3.3 are not published for Python 3.12.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)

set "PY=%~dp0.venv\Scripts\python.exe"

echo Updating pip...
"%PY%" -m pip install -U pip setuptools wheel
if errorlevel 1 goto fail

set "HAS_NVIDIA=0"
if exist "%SystemRoot%\System32\nvidia-smi.exe" set "HAS_NVIDIA=1"
if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "HAS_NVIDIA=1"

if "%HAS_NVIDIA%"=="1" (
  echo NVIDIA GPU found. Installing PyTorch 2.1.2 with CUDA 12.1.
  "%PY%" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
) else (
  echo No NVIDIA driver found. Installing the CPU build of PyTorch.
  "%PY%" -m pip install torch==2.1.2 torchvision==0.16.2
)
if errorlevel 1 goto fail

echo Pinning NumPy 1.x. PyTorch 2.1 does not run on NumPy 2.
"%PY%" -m pip install "numpy>=1.23.5,<2"
if errorlevel 1 goto fail

echo Installing MMEngine...
"%PY%" -m pip install "mmengine==0.10.7"
if errorlevel 1 goto fail

set "MMCV_URL=https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp311-cp311-win_amd64.whl"
"%PY%" -c "import torch,sys; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if not errorlevel 1 set "MMCV_URL=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp311-cp311-win_amd64.whl"
echo MMCV package: %MMCV_URL%

echo Installing MMCV 2.1.0...
"%PY%" -m pip install --force-reinstall --no-deps --no-cache-dir "%MMCV_URL%"
if errorlevel 1 goto fail

echo Installing MMDetection 3.3.0...
"%PY%" -m pip install "mmdet==3.3.0"
if errorlevel 1 goto fail

echo Restoring the matching MMCV build...
"%PY%" -m pip install --force-reinstall --no-deps --no-cache-dir "%MMCV_URL%"
if errorlevel 1 goto fail

echo Installing OpenCV, NumPy and PySide6...
"%PY%" -m pip install "numpy>=1.23.5,<2" "opencv-python==4.10.0.84" "PySide6>=6.6,<6.10"
if errorlevel 1 goto fail

echo Creating folders...
"%PY%" -c "from utils.paths import ensure_dirs; ensure_dirs()"
if errorlevel 1 goto fail

echo Running import test, DetInferencer, CUDA inference and a sample image...
"%PY%" -m utils.smoke_test
if errorlevel 1 goto fail

echo.
echo Installation finished. Start the program with start.bat
pause
exit /b 0

:fail
echo.
echo Installation failed. Read the messages above. A copy of runtime errors also goes to logs\app.log after the app starts.
pause
exit /b 1
