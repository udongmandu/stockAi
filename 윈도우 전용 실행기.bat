@echo on
setlocal ENABLEDELAYEDEXPANSION

REM ────────────────────────────────
REM 작업 폴더를 이 배치 파일 위치로 고정
cd /d "%~dp0"

REM 실행할 Streamlit 파일명
set "FILE=stockTest.py"

REM 로그 파일
set "LOG=run_log.txt"
echo. > "%LOG%"

echo [STEP] Python check  ^>^> %LOG%
where python >> "%LOG%" 2>&1 || (
  echo ❌ check python installed path . >> "%LOG%"
  echo ❌ check python installed path .
  pause
  exit /b 1
)

echo [STEP] pip 버전 확인  ^>^> %LOG%
python -m ensurepip --default-pip >> "%LOG%" 2>&1
python -m pip --version >> "%LOG%" 2>&1 || (
  echo ❌ pip cant use >> "%LOG%"
  echo ❌ pip cant use
  pause
  exit /b 1
)

echo [STEP] pip 업그레이드  ^>^> %LOG%
python -m pip install --upgrade pip >> "%LOG%" 2>&1 || (
  echo ❌ pip cant upgrade >> "%LOG%"
  echo ❌ pip cant upgrade
  type "%LOG%"
  pause
  exit /b 1
)

if exist requirements.txt (
  echo [STEP] requirements install  ^>^> %LOG%
  python -m pip install -r requirements.txt >> "%LOG%" 2>&1 || (
    echo ❌ requirements 설치 실패 >> "%LOG%"
    echo ❌ requirements 설치 실패
    type "%LOG%"
    pause
    exit /b 1
  )
) else (
  echo [INFO] no requirements.txt -> skipped >> "%LOG%"
)

echo [STEP] Streamlit 확인  ^>^> %LOG%
python -c "import streamlit,sys;print(streamlit.__version__)" >> "%LOG%" 2>&1 || (
  echo ❌ streamlit cant be imported >> "%LOG%"
  echo ❌ check the streamlit is installed
  type "%LOG%"
  pause
  exit /b 1
)

if not exist "%FILE%" (
  echo ❌ there is no file: %FILE% >> "%LOG%"
  echo ❌ there is no file for start: %FILE%
  type "%LOG%"
  pause
  exit /b 1
)

echo [STEP] Streamlit app start  ^>^> %LOG%
python -m streamlit run "%FILE%"

echo [INFO] Log (recent):
type "%LOG%"

echo.
echo server terminated
pause
endlocal
