#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_lstm_model_input_compat.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/LSTMModelInputCompatibilityTests.cpp" \
    -o "${test_dir}/LSTMModelInputCompatibilityTests"
"${test_dir}/LSTMModelInputCompatibilityTests"
