@echo off

REM ────────────────────────────────
REM 파이썬 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    pause
    exit /b 1
)

REM ────────────────────────────────
REM pip 설치 및 최신화
echo 🔄 pip 설치 및 최신화 중...
python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip

REM ────────────────────────────────
REM requirements.txt로 패키지 설치
echo 📦 requirements.txt 기반 패키지 설치 중...
python -m pip install -r requirements.txt

REM ────────────────────────────────
REM 실행할 Streamlit 파일명
set FILE=stockTest.py

REM ────────────────────────────────
REM Streamlit 앱 실행
echo 🚀 Streamlit 앱을 실행합니다. 종료하려면 Ctrl+C 후 Y 입력
python -m streamlit run %FILE%

echo ✅ 서버 종료됨. 아무 키나 누르세요.
pause >nul
