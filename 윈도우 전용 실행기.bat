@echo off
chcp 65001 >nul
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"

set "FILE=stockTest.py"

where python >nul 2>&1 || (
  echo ❌ Python not found.
  pause
  exit /b 1
)

python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip

if exist requirements.txt (
  python -m pip install -r requirements.txt
)

if not exist "%FILE%" (
  echo ❌ No file found: %FILE%
  pause
  exit /b 1
)

echo.
echo ✅ Starting Streamlit app...
python -m streamlit run "%FILE%"

echo.
echo Server finished. Press any key to close.
pause
