#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_lstm_input_width_expansion.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/LSTMInputWidthExpansionTests.cpp" \
    -o "${test_dir}/LSTMInputWidthExpansionTests"
"${test_dir}/LSTMInputWidthExpansionTests"

# CLI/workflow wiring checks protect the opt-in boundary: ordinary resume uses
# the unchanged three-argument loadAll call behavior, while only the explicit
# mode selects parameter expansion and the scheduler persists/forwards it.
grep -q 'arg == "--resume-expand-input-width"' "${repo_root}/LSTM/main.cpp"
grep -q 'resumeConfig->parameterExpansionRequired' "${repo_root}/LSTM/main.cpp"
grep -q 'AddCliFlag(argv, "--resume-expand-input-width")' \
    "${repo_root}/Sources/ExperimentScheduler.cpp"
grep -q 'resume_expand_input_width boolean NOT NULL' \
    "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"
