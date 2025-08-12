#!/bin/bash

# ────────────────────────────────
# 현재 스크립트 위치로 이동
cd "$(dirname "$0")" || exit 1

# 실행할 Streamlit 파일명
FILE="stockTest.py"

# 로그 파일
LOG="run_log.txt"
> "$LOG"

echo "[STEP] Python check" | tee -a "$LOG"
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python3 not found. Please install and ensure it's in your PATH." | tee -a "$LOG"
    exit 1
fi

echo "[STEP] pip check" | tee -a "$LOG"
python3 -m ensurepip --default-pip >> "$LOG" 2>&1
if ! python3 -m pip --version >> "$LOG" 2>&1; then
    echo "❌ pip is not available." | tee -a "$LOG"
    exit 1
fi

echo "[STEP] pip upgrade" | tee -a "$LOG"
if ! python3 -m pip install --upgrade pip >> "$LOG" 2>&1; then
    echo "❌ Failed to upgrade pip." | tee -a "$LOG"
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "[STEP] Installing requirements.txt" | tee -a "$LOG"
    if ! python3 -m pip install -r requirements.txt >> "$LOG" 2>&1; then
        echo "❌ Failed to install requirements." | tee -a "$LOG"
        exit 1
    fi
else
    echo "[INFO] requirements.txt not found -> skipped" | tee -a "$LOG"
fi

echo "[STEP] Streamlit check" | tee -a "$LOG"
if ! python3 -c "import streamlit,sys;print(streamlit.__version__)" >> "$LOG" 2>&1; then
    echo "❌ Streamlit not installed or cannot be imported." | tee -a "$LOG"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "❌ File not found: $FILE" | tee -a "$LOG"
    exit 1
fi

echo "[STEP] Starting Streamlit app" | tee -a "$LOG"
python3 -m streamlit run "$FILE"

echo "[INFO] Log (recent):"
tail -n 20 "$LOG"

echo "✅ Server terminated"
