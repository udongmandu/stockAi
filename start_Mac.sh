#!/bin/bash

# Python 3 설치 확인
if ! command -v python3 &> /dev/null
then
    echo "Python 3이 설치되어 있지 않습니다. 설치 후 다시 시도하세요."
    exit 1
fi

# pip 최신화 및 필요한 패키지 설치
python3 -m ensurepip --default-pip
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet streamlit pandas requests beautifulsoup4 openai xlrd

# streamlit 실행
echo "Streamlit 서버를 시작합니다. 종료하려면 Ctrl+C 누르세요."
python3 -m streamlit run stockTest.py
