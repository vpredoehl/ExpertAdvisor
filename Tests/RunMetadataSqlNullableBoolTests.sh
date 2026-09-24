#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
clang++ -std=c++20 -I"$root/Headers" -I"$root/Sources" \
  "$root/Tests/RunMetadataSqlNullableBoolTests.cpp" -o "$scratch/run-metadata-test"
"$scratch/run-metadata-test"
