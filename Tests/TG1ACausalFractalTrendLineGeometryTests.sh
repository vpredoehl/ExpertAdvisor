#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg1a_geometry.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TG1ACausalFractalTrendLineGeometryTests.cpp" \
    -o "${test_dir}/TG1ACausalFractalTrendLineGeometryTests"

"${test_dir}/TG1ACausalFractalTrendLineGeometryTests"

fractal_source="${repo_root}/Database/indicators/fractal.plpgsql"
candlestick_source="${repo_root}/Database/forex/candlestick.plpgsql"
grep -Eiq 'max\(high\).*< high as fractal_high' "${fractal_source}"
grep -Eiq 'min\(low\).* > low as fractal_low' "${fractal_source}"
grep -Eiq 'order by dt rows between 2 preceding and 2 following exclude current row' "${fractal_source}"
grep -Eiq 'create or replace function candlestick\(' "${candlestick_source}"
printf '%s\n' "TG1A database fractal semantic audit passed"
