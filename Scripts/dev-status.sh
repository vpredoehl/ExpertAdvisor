#!/usr/bin/env bash
set -euo pipefail

echo "== Git =="
git status --short
git branch --show-current
git log -1 --oneline

echo
echo "== Changed files =="
git diff --stat
git diff --cached --stat

echo
echo "== LSTM processes =="
pgrep -alf LSTM_Release || true

echo
echo "== Scheduler status =="
BIN="./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release"

if [[ -x "$BIN" ]]; then
    "$BIN" --scheduler-status || true
else
    echo "Release executable not found: $BIN"
fi

