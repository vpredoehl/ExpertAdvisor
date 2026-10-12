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
export EA_RESEARCH_GUARD_HEAD="$("${research_git}" -C "${repo_root}" rev-parse HEAD)"
guard() {
python3 - "${repo_root}" <<'PY'
import os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
def git(*args): return subprocess.check_output([os.environ['EA_RESEARCH_GIT'], '-C', str(root), *args], text=True).strip()
assert git('rev-parse', 'HEAD') == os.environ['EA_RESEARCH_GUARD_HEAD'], 'STOP: HEAD changed during invocation'
assert git('branch', '--show-current') == 'lstm-feature-development', 'STOP: branch changed'
files = ['Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp', 'Headers/CausalFibonacciTrendlineHistoricalArtifact.hpp',
             'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp', 'Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp',
             'Scripts/run_causal_fibonacci_trendline_study.sh', 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
             'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']
assert not git('diff', '--cached', '--name-only'), 'STOP: staged changes'
assert not git('ls-files', '--others', '--exclude-standard'), 'STOP: unexpected untracked files'
assert set(git('diff', '--name-only').splitlines()) <= set(files), 'STOP: non-research tracked changes'
gitdir = pathlib.Path(git('rev-parse', '--absolute-git-dir'))
for name in ('MERGE_HEAD', 'REBASE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
             'BISECT_LOG', 'index.lock', 'rebase-merge', 'rebase-apply', 'sequencer'):
    assert not (gitdir / name).exists(), 'STOP: active Git operation: ' + name
for name in files:
    assert (root / name).is_file() and not (root / name).is_symlink(), 'STOP: missing research file'
    assert git('ls-files', '--', name) == name, 'STOP: research file is not tracked: ' + name
PY
}
guard
test_dir="$(mktemp -d /tmp/ea_fib_v2_tests.XXXXXX)"
cleanup() {
    if [[ "${test_dir}" == /tmp/ea_fib_v2_tests.* && -d "${test_dir}" && ! -L "${test_dir}" ]]; then rm -rf -- "${test_dir}"; fi
}
trap cleanup EXIT
python3 - "${repo_root}" <<'PYGUARD'
import os, pathlib, sys
from unittest.mock import patch
root = pathlib.Path(sys.argv[1])
head = os.environ['EA_RESEARCH_GUARD_HEAD']
files = ['Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp', 'Headers/CausalFibonacciTrendlineHistoricalArtifact.hpp',
         'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp', 'Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp',
         'Scripts/run_causal_fibonacci_trendline_study.sh', 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
         'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']
checks = 0
for script in ['Scripts/run_causal_fibonacci_trendline_study.sh', 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']:
    code = (root / script).read_text().split('python3 - "${repo_root}" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    assert 'layout13_replication.sql' not in code
    runner = script.startswith('Scripts/')
    def verify(expected, validation=True, changed='', staged='', untracked='', actual_head=head,
               expected_head=head, branch='lstm-feature-development', missing_tracking=False):
        global checks
        def fake_git(command, **kwargs):
            args = tuple(command[3:])
            replies = {('rev-parse', 'HEAD'): actual_head, ('branch', '--show-current'): branch,
                       ('rev-parse', '--absolute-git-dir'): str(root / '.git'),
                       ('diff', '--name-only'): changed, ('diff', '--cached', '--name-only'): staged,
                       ('ls-files', '--others', '--exclude-standard'): untracked}
            if args[:2] == ('ls-files', '--'):
                return '' if missing_tracking else args[2] + '\n'
            return replies[args] + '\n'
        passed = True
        with patch('subprocess.check_output', fake_git), patch.dict(os.environ,
             {'EA_RESEARCH_GUARD_HEAD': expected_head, 'EA_RESEARCH_VALIDATION_ONLY': '1' if validation else '0'}), \
             patch.object(sys, 'argv', ['guard', str(root)]):
            try: exec(compile(code, script, 'exec'), {})
            except AssertionError: passed = False
        assert passed == expected, (script, validation, changed, staged, untracked, actual_head, passed)
        checks += 1
    verify(True)
    verify(True, changed=files[3])
    verify(False, changed='Sources/TrainingPipeline.cpp')
    verify(False, staged=files[3])
    verify(False, untracked='unexpected.cpp')
    verify(False, actual_head='f' * 40)
    verify(True, actual_head='f' * 40, expected_head='f' * 40) # Future committed HEAD is accepted.
    verify(False, branch='other')
    verify(False, missing_tracking=True)
    if runner:
        verify(True, validation=False)
        verify(False, validation=False, changed=files[3])
        verify(False, validation=False, changed='Sources/TrainingPipeline.cpp')
print(f'RESEARCH_GUARD_VALIDATION checks={checks} strict_execution=true protected_sql_access=false')
PYGUARD
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
