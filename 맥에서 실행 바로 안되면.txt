@echo off
REM 파이썬 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.8 이상 설치 후 PATH에 등록해 주세요.
    pause
    exit /b 1
)

REM pip 최신화 및 필요한 패키지 설치
python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip
python -m pip install --quiet --disable-pip-version-check streamlit pandas requests beautifulsoup4 openai xlrd

REM streamlit 실행 (python 모듈 방식)
echo Streamlit 서버를 시작합니다. 종료하려면 Ctrl+C 누른 후 Y 입력
python -m streamlit run stockTest.py

echo 서버 종료됨. 아무 키 누르면 닫힙니다.
pause >nul
