#!/bin/bash

ROOT="/Volumes/Developer SSD/ExpertAdvisor"

BIN="$(
    find \
        "$ROOT/DerivedData/Production" \
        "$ROOT/DerivedData/Release" \
        -type f \
        -path '*/Build/Products/Release/LSTM_Release' \
        -perm -111 \
        -print0 2>/dev/null \
    | xargs -0 stat -f '%m %N' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

if [ -z "$BIN" ]; then
    echo "ERROR: No Release LSTM_Release binary found." >&2
    return 1 2>/dev/null || exit 1
fi

export BIN

echo "BIN=$BIN"

