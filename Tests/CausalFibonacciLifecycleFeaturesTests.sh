#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/CausalFibonacciLifecycleFeaturesTests"
CXX="${CXX:-clang++}"

"$CXX" \
  -std=c++20 \
  -Wall -Wextra -Werror -pedantic \
  -I"$ROOT/Headers" \
  "$ROOT/Tests/CausalFibonacciLifecycleFeaturesTests.cpp" \
  -o "$OUT"

"$OUT"
