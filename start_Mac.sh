#!/bin/bash

# Python 3 설치 확인
if ! command -v python3 &> /dev/null
then
    echo "Python 3이 설치되어 있지 않습니다. 설치 후 다시 시도하세요."
    exit 1
fi

# 패키지 설치 (필요시)
pip3 install --upgrade pip
pip3 install streamlit pandas requests beautifulsoup4 openai xlrd

# 앱 실행
python3 -m streamlit run stockTest.py

