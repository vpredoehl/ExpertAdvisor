#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Bypass Apple /usr/bin tool-discovery shims: changing TMPDIR must not
# launch xcodebuild metadata discovery. No Xcode/xcrun invocation is needed.
research_git="${EA_RESEARCH_GIT:-/Applications/Xcode.app/Contents/Developer/usr/bin/git}"
research_cxx="${CXX:-/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++}"
research_sdk="${EA_RESEARCH_SDKROOT:-/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk}"
test -x "${research_git}" && test -x "${research_cxx}" && test -d "${research_sdk}"
export EA_RESEARCH_GIT="${research_git}"
guard() {
python3 - "${repo_root}" <<'PY'
import hashlib, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
def git(*args): return subprocess.check_output([os.environ['EA_RESEARCH_GIT'], '-C', str(root), *args], text=True).strip()
assert git('rev-parse', 'HEAD') == '291fbd286ed19041ba8f275cc59760ebdc4025ae', 'STOP: HEAD changed'
assert git('branch', '--show-current') == 'lstm-feature-development', 'STOP: branch changed'
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only'), 'STOP: tracked changes'
for name in ['Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp', 'Headers/CausalFibonacciTrendlineHistoricalArtifact.hpp',
             'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp', 'Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp',
             'Scripts/run_causal_fibonacci_trendline_study.sh', 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
             'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']:
    assert (root / name).is_file() and not (root / name).is_symlink(), 'STOP: missing research file'
    assert not git('ls-files', '--', name), 'STOP: research file tracked'
sql = root / 'layout13_replication.sql'
assert sql.is_file() and not sql.is_symlink() and hashlib.sha256(sql.read_bytes()).hexdigest() == 'afcebbcb591149c9b54873ee19d707bea58d2ab33676ea1598de696b28843590', 'STOP: SQL changed/missing'
PY
}
guard
test_dir="$(mktemp -d /tmp/ea_fib_v2_tests.XXXXXX)"
cleanup() {
    if [[ "${test_dir}" == /tmp/ea_fib_v2_tests.* && -d "${test_dir}" && ! -L "${test_dir}" ]]; then rm -rf -- "${test_dir}"; fi
}
trap cleanup EXIT
"${research_cxx}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -isysroot "${research_sdk}" \
    -I"${repo_root}/Headers" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp" \
    "${repo_root}/Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp" -o "${test_dir}/tests"
guard
"${test_dir}/tests" "${repo_root}/Scripts/tg4_analysis_config.frozen_v1.conf" "${test_dir}"
guard
python3 - "${test_dir}" <<'PY'
import csv, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / 'first'
p = json.loads((root / 'protocol.json').read_text())
m = json.loads((root / 'provenance_manifest.json').read_text())
assert p['schema'] == 'fib-trendline-decision-study-v2'
assert p['warmup_completed_candles'] == 2048
assert p['calibration_embargo_completed_candles'] == 4
assert p['final_2025_disabled'] and p['statistical_estimation'] == 'NOT_EXECUTED'
assert p['support_model']['status'] == 'NOT_EXECUTED'
assert p['adequacy']['minimum_B'] == p['adequacy']['minimum_C'] == 500
assert p['calendar_status'] == 'UNVERIFIED'
assert m['canonical_candle_content_sha256'] and m['coverage'] == 'UNVERIFIED'
for name in ('decisions', 'outcomes', 'geometry_measurements', 'interaction_episodes', 'coverage', 'exclusions', 'gaps'):
    rows = list(csv.reader((root / (name + '.csv')).open()))
    assert rows and all(len(row) == len(rows[0]) for row in rows), name
for line in (root / 'checksums.sha256').read_text().splitlines():
    expected, file = line.split('  ', 1)
    assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected, file
decisions = list(csv.DictReader((root / 'decisions.csv').open()))
outcomes = list(csv.DictReader((root / 'outcomes.csv').open()))
assert len({r['decision_id'] for r in decisions}) == len(decisions)
assert len(outcomes) == sum(r['status'] == 'classified' for r in decisions)
assert all(int(r['bar_start_utc_epoch']) % 3600 == 2700 for r in decisions)
assert 'NOT_EXECUTED' in (root / 'COMPLETE').read_text()
print('Deterministic v2 artifact JSON/CSV/checksum/reconciliation validation passed')
PY
guard
