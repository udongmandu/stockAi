#!/bin/bash

# ────────────────────────────────
# Python 3 설치 확인
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3이 설치되어 있지 않습니다. 설치 후 다시 시도하세요."
    exit 1
fi

# ────────────────────────────────
# pip 최신화
echo "🔄 pip 설치 및 업그레이드..."
python3 -m ensurepip --default-pip
python3 -m pip install --upgrade pip

# ────────────────────────────────
# requirements.txt로 패키지 설치
echo "📦 requirements.txt 기반 패키지 설치..."
python3 -m pip install -r requirements.txt

# ────────────────────────────────
# 실행할 Streamlit 파일명 설정
FILE="stockTest.py"

# ────────────────────────────────
# Streamlit 앱 실행
echo "🚀 Streamlit 서버를 시작합니다. 종료하려면 Ctrl+C 누르세요."
python3 -m streamlit run "$FILE"
