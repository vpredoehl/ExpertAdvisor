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
new_files = ['Headers/CausalFibonacciResearchCandleSnapshot.hpp', 'Headers/CausalFractalHistoryShadowAudit.hpp', 'Sources/CausalFractalHistoryShadowAudit.cpp', 'Tests/CausalFractalHistoryShadowAuditTests.cpp', 'Headers/CausalABCapacityCounterfactual.hpp', 'Sources/CausalABCapacityCounterfactual.cpp', 'Tests/CausalABCapacityCounterfactualTests.cpp', 'Sources/CausalABCapacityInterventionCLI.cpp', 'Tests/CausalABCapacityInterventionTests.cpp', 'Scripts/run_causal_ab_capacity_interventions.py']
files += new_files + ['docs/phases/target-generation/PhaseTG3/AB-Capacity-Counterfactual.md']
assert not git('diff', '--cached', '--name-only'), 'STOP: staged changes'
untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
validation = True
assert untracked <= set(new_files + ['docs/phases/target-generation/PhaseTG3/AB-Capacity-Counterfactual.md']) if validation else not untracked, 'STOP: unexpected untracked files'
assert set(git('diff', '--name-only').splitlines()) <= set(files), 'STOP: non-research tracked changes'
gitdir = pathlib.Path(git('rev-parse', '--absolute-git-dir'))
for name in ('MERGE_HEAD', 'REBASE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
             'BISECT_LOG', 'index.lock', 'rebase-merge', 'rebase-apply', 'sequencer'):
    assert not (gitdir / name).exists(), 'STOP: active Git operation: ' + name
for name in files:
    assert (root / name).is_file() and not (root / name).is_symlink(), 'STOP: missing research file'
    assert git('ls-files', '--', name) == name or (validation and name in new_files + ['docs/phases/target-generation/PhaseTG3/AB-Capacity-Counterfactual.md'] and name in untracked), 'STOP: research file is not tracked: ' + name
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
new_files = ['Headers/CausalFibonacciResearchCandleSnapshot.hpp', 'Headers/CausalFractalHistoryShadowAudit.hpp', 'Sources/CausalFractalHistoryShadowAudit.cpp', 'Tests/CausalFractalHistoryShadowAuditTests.cpp', 'Headers/CausalABCapacityCounterfactual.hpp', 'Sources/CausalABCapacityCounterfactual.cpp', 'Tests/CausalABCapacityCounterfactualTests.cpp', 'Sources/CausalABCapacityInterventionCLI.cpp', 'Tests/CausalABCapacityInterventionTests.cpp', 'Scripts/run_causal_ab_capacity_interventions.py']
files += new_files + ['docs/phases/target-generation/PhaseTG3/AB-Capacity-Counterfactual.md']
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
    verify(True, untracked=new_files[0])
    if runner: verify(False, validation=False, untracked=new_files[0])
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
    "${repo_root}/Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp" -lz -o "${test_dir}/tests"
guard
"${test_dir}/tests" "${repo_root}/Scripts/tg4_analysis_config.frozen_v1.conf" "${test_dir}"
guard
"${research_cxx}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -isysroot "${research_sdk}" \
    -I"${repo_root}/Headers" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp" \
    "${repo_root}/Sources/CausalFractalHistoryShadowAudit.cpp" \
    "${repo_root}/Tests/CausalFractalHistoryShadowAuditTests.cpp" -lz -o "${test_dir}/shadow-tests"
"${test_dir}/shadow-tests" "${repo_root}/Scripts/tg4_analysis_config.frozen_v1.conf" "${test_dir}"
guard
python3 - "${test_dir}" <<'PY'
import csv, gzip, hashlib, json, pathlib, sys
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
for name in ('decisions', 'outcomes', 'episode_summary', 'coverage', 'exclusions', 'gaps', 'compression_manifest'):
    rows = list(csv.reader((root / (name + '.csv')).open()))
    assert rows and all(len(row) == len(rows[0]) for row in rows), name
assert p['artifact_format'] == 'fib-trendline-artifacts-v3'
assert p['episode_publication'] == 'summary' and p['episode_detail_omitted']
assert not (root / 'interaction_episodes.csv').exists() and not (root / 'interaction_episodes.csv.gz').exists()
for content in csv.DictReader((root / 'compression_manifest.csv').open()):
    raw = gzip.decompress((root / content['filename']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == content['decoded_sha256']
    assert len(raw) == int(content['decoded_bytes'])
    assert raw.count(b'\n') - 1 == int(content['decoded_rows'])
    direct = m['compressed_artifacts'][content['filename']]
    assert direct['compressed_sha256'] == content['compressed_sha256']
    assert direct['decoded_sha256'] == content['decoded_sha256']
    assert direct['decoded_rows'] == int(content['decoded_rows'])
    rows = list(csv.reader(raw.decode().splitlines()))
    assert all(len(row) == len(rows[0]) for row in rows)
assert m['compression']['mtime'] == 0 and m['compression']['level'] == 6
assert m['diagnostic_tracking_complete'] and m['episode_detail_omitted']
assert m['capacity_high_watermark_values']['retained_ab_structures']['full_replay'] == 1
assert m['capacity_high_watermark_values']['eligible_geometry_pairs']['scoring'] == 1
assert m['capacity_high_watermark_values']['episode_ledger']['full_replay'] == 3
for name in ('saturated-summary', 'saturated-full'):
    sat = json.loads((root.parent / name / 'provenance_manifest.json').read_text())
    assert not sat['diagnostic_tracking_complete']
    assert sat['episode_detail_omitted'] == (name == 'saturated-summary')
    assert sat['capacity_high_watermark_values']['retained_ab_structures'] == dict(full_replay=3, prefix=3, scoring=2)
    assert sat['capacity_high_watermark_values']['eligible_geometry_pairs'] == dict(full_replay=9, prefix=9, scoring=4)
    summary = {r['metric']: r['value'] for r in csv.DictReader((root.parent / name / 'episode_summary.csv').open()) if r['phase'] == 'full_replay'}
    assert summary['episode_saturated'] == '1' and summary['skipped_observation_opportunities'] == '38'
    assert summary['tracking_complete'] == '0'
assert p['protocol_fingerprint'] == json.loads((root.parent / 'full' / 'protocol.json').read_text())['protocol_fingerprint']
for line in (root / 'checksums.sha256').read_text().splitlines():
    expected, file = line.split('  ', 1)
    assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected, file
decisions = list(csv.DictReader((root / 'decisions.csv').open()))
outcomes = list(csv.DictReader((root / 'outcomes.csv').open()))
assert len({r['decision_id'] for r in decisions}) == len(decisions)
assert len(outcomes) == sum(r['status'] == 'classified' for r in decisions)
assert all(int(r['bar_start_utc_epoch']) % 3600 == 2700 for r in decisions)
assert 'NOT_EXECUTED' in (root / 'COMPLETE').read_text()
audit = root.parent / 'shadow-pressure'
protocol = json.loads((audit / 'audit_protocol.json').read_text())
provenance = json.loads((audit / 'audit_provenance.json').read_text())
assert protocol['schema'] == 'causal-fractal-history-shadow-audit-v1'
assert protocol['semantic_protocol'] == 'fib-trendline-decision-study-v2'
assert protocol['primary_artifact_format'] == 'fib-trendline-artifacts-v3'
assert protocol['TG3_history_per_kind'] == 128 and protocol['TG1A_history_per_kind'] == 64
assert protocol['TG3_candidate_capacity'] == 512 and protocol['TG1A_candidate_capacity'] == 4096
assert protocol['compression']['mtime'] == 0 and protocol['compression']['zlib_version']
assert provenance['mirror_parity'] and provenance['diagnostic_complete'] and not provenance['capacity_exhausted']
assert provenance['creation_losses_TG1A'] == 19474
assert provenance['lost_candidate_totals'] == 'EXACT_WITHIN_OBSERVED_REPLAY'
parent = root.parent / 'shadow-pressure-parent'
assert provenance['parent_primary_checksums_sha256'] == hashlib.sha256((parent / 'checksums.sha256').read_bytes()).hexdigest()
assert provenance['canonical_candle_sha256'] == provenance['audited_prefix_candle_sha256']
for name in ('history_summary.csv',):
    with (audit / name).open() as f:
        rows = list(csv.reader(f))
    assert all(len(row) == len(rows[0]) for row in rows)
for name, claim in provenance['compressed_contents'].items():
    with gzip.open(audit / name, 'rt') as f:
        reader = csv.reader(f); header = next(reader); count = 0
        for row in reader:
            assert len(row) == len(header), name
            count += 1
    assert count == claim['decoded_rows']
for line in (audit / 'audit_checksums.sha256').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((audit / name).read_bytes()).hexdigest() == expected
assert not (root.parent / 'isolation-failed-audit' / 'AUDIT_COMPLETE').exists()
assert not (root.parent / 'isolation-allocation-audit' / 'AUDIT_COMPLETE').exists()
print('Deterministic artifact-v3/semantic-v2 JSON/CSV/gzip/checksum/reconciliation validation passed')
print('Shadow-audit-v1 JSON/CSV/gzip/parent-linkage/completeness validation passed')
for folder in ('isolation-decision-audit', 'decision-pipeline'):
    audit = root.parent / folder
    meta = json.loads((audit / 'audit_provenance.json').read_text())
    protocol = json.loads((audit / 'decision_counterfactual_protocol.json').read_text())
    assert meta['decision_geometry_complete'] and protocol['schema'] == 'causal-decision-geometry-counterfactual-v1'
    assert meta['decision_geometry_extension'] == protocol['schema']
    assert protocol['ratios'] == [0.382, 0.500, 0.618] and protocol['threshold_atr'] == 0.25
    summary = {r['metric']: r for r in csv.DictReader((audit / 'decision_counterfactual_summary.csv').open())}
    assert summary['diagnostic_complete']['value'] == '1'
    for filename, claim in meta['decision_compressed_contents'].items():
        digest = hashlib.sha256(); rows = 0
        with gzip.open(audit / filename, 'rb') as f:
            for line in f: digest.update(line); rows += 1
        assert digest.hexdigest() == claim['decoded_sha256'] and rows - 1 == claim['decoded_rows']
        with gzip.open(audit / filename, 'rt') as f:
            reader = csv.reader(f); header = next(reader)
            assert all(len(row) == len(header) for row in reader), filename
    for line in (audit / 'audit_checksums.sha256').read_text().splitlines():
        expected, filename = line.split('  ', 1)
        assert hashlib.sha256((audit / filename).read_bytes()).hexdigest() == expected
    if folder == 'decision-pipeline':
        assert int(summary['decisions_with_restored_candidate_presence']['value']) == 7
        assert int(summary['decisions_with_restored_candidate_proximity']['value']) == 7
        assert int(summary['added_pair_evidence_rows']['value']) > 0
        assert all(summary[k]['value'] == '0' for k in ('decisions_changed_Lline','decisions_changed_I','decisions_changed_cohort'))
        with gzip.open(audit / 'decision_counterfactual_candidates.csv.gz', 'rt') as f:
            candidate_rows = list(csv.DictReader(f))
        assert any(r['actual_present'] == '0' and r['direct_history_suppression'] == '1' for r in candidate_rows)
        assert any(r['actual_present'] == '1' and r['counterfactual_present'] == '0' for r in candidate_rows)
    else:
        assert summary['decisions_with_restored_candidate_presence']['value'] == '0'
for failure in range(3):
    failed = root.parent / f'decision-failure-audit-{failure}'
    assert not (failed / 'AUDIT_COMPLETE').exists()
    summary = {r['metric']: r['value'] for r in csv.DictReader((failed / 'decision_counterfactual_summary.csv').open())}
    assert summary['decisions_changed_cohort'] == 'UNKNOWN'
assert not (root.parent / 'isolation-decision-failed-audit' / 'AUDIT_COMPLETE').exists()
print('Counterfactual decision-extension JSON/CSV/gzip/positive-pair-evidence/UNKNOWN/checksum validation passed')

PY
guard

"${research_cxx}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -isysroot "${research_sdk}" \
    -I"${repo_root}/Headers" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp" \
    "${repo_root}/Sources/CausalFractalHistoryShadowAudit.cpp" \
    "${repo_root}/Sources/CausalABCapacityCounterfactual.cpp" \
    "${repo_root}/Tests/CausalABCapacityCounterfactualTests.cpp" -lz -o "${test_dir}/ab-capacity-tests"
"${test_dir}/ab-capacity-tests" "${repo_root}/Scripts/tg4_analysis_config.frozen_v1.conf" "${test_dir}"
python3 - "${test_dir}/AB-capacity" <<'PYAB'
import csv, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for d in sorted(root.glob('*-audit')):
    for name in ('decisions.csv', 'restored_candidates.csv', 'source_evictions.csv'):
        rows = list(csv.reader((d / name).open()))
        assert all(len(r) == len(rows[0]) for r in rows), (d.name, name)
    if not (d / 'AUDIT_COMPLETE').exists():
        continue
    if d.name == 'enabled-audit': # Intentionally corrupted by the C++ integrity test.
        continue
    p = json.loads((d / 'audit_protocol.json').read_text())
    provenance = json.loads((d / 'audit_provenance.json').read_text())
    summary = {r['metric']: r['value'] for r in csv.DictReader((d / 'summary.csv').open())}
    assert p['schema'] == 'causal-ab-capacity-counterfactual-v1'
    assert p['source_AB_capacity'] == 512 and p['TG3_history_per_kind'] == 128
    assert p['line_capacity'] == 4096 and p['TG1A_history_per_kind'] == 64
    assert not p['restore_fractal_history'] and p['ratios'] == [.382, .5, .618]
    assert int(summary['peak_accounted_memory_bytes']) <= p['memory_byte_budget']
    assert int(summary['peak_live_AB_records']) <= p['live_AB_budget']
    assert int(summary['total_pairs']) <= p['pair_work_total']
    assert int(summary['peak_decision_pairs']) <= p['pair_work_per_decision']
    assert int(summary['decoded_output_bytes']) == sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    assert int(summary['decoded_output_bytes']) <= p['decoded_output_byte_budget']
    for line in (d / 'audit_checksums.sha256').read_text().splitlines():
        h, name = line.split('  ', 1)
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == h
    rows = list(csv.DictReader((d / 'decisions.csv').open()))
    assert len(rows) == int(summary['scheduled']) == int(summary['decision_rows'])
    assert all(r['status'] in ('RECONSTRUCTED', 'UNOBSERVABLE_MISSING_SLOT') for r in rows)
    if d.name == 'observer-boundary-audit':
        assert rows[0]['actual_cohort'] == 'B' and rows[0]['counterfactual_cohort'] == 'C'
        assert rows[1]['actual_cohort'] == rows[1]['counterfactual_cohort'] == 'B'
        victims = list(csv.DictReader((d / 'restored_candidates.csv').open()))
        assert any(r['age'] == '2048' and r['conjunction'] == '1' for r in victims)
        assert not any(int(r['age']) > 2048 for r in victims)
    if d.name == 'missing-slot-audit':
        assert sum(r['status'] == 'UNOBSERVABLE_MISSING_SLOT' for r in rows) == 1
        assert all(r['actual_Ffib'] == r['counterfactual_Ffib'] == 'UNKNOWN' for r in rows if r['status'] == 'UNOBSERVABLE_MISSING_SLOT')
    if d.name == 'positive-audit':
        assert summary['changed_Ffib'] == summary['changed_I'] == summary['changed_cohort'] == '5'
        assert summary['changed_Lline'] == '0'
        assert all(r['actual_cohort'] == 'B' and r['counterfactual_cohort'] == 'C' for r in rows)
    if d.name == 'repeat-audit':
        assert provenance['parent_status'] == 'INVALID'
        assert all(r['primary_status'] == 'capacity_invalid' for r in rows)
        evictions = list(csv.DictReader((d / 'source_evictions.csv').open()))
        assert {r['direction'] for r in evictions} == {'0', '1'}
        assert len(evictions) == int(summary['source_evictions_full'])
        assert int(evictions[0]['bar']) == int(summary['first_eviction_bar'])
        assert all(r['reason'] == 'SOURCE_AB_CAPACITY' for r in evictions)
        assert len({r['AB_identity'] for r in evictions}) == len(evictions)
        victims = list(csv.DictReader((d / 'restored_candidates.csv').open()))
        assert any(r['upward'] == '0' for r in victims) and any(r['upward'] == '1' for r in victims)
        assert any(r['conjunction'] == '1' for r in victims)
        assert summary['changed_Ffib'] == summary['changed_I'] == summary['changed_cohort'] == '0'
        assert all(r['eligible_pairs'] == '0' and r['fib_near'] == 'INELIGIBLE' and r['conjunction'] == 'INELIGIBLE' for r in victims if r['upward'] == '0')
        assert int(summary['source_evictions_full']) == int(summary['source_evictions_prefix']) + int(summary['source_evictions_scoring'])
cf_rows = list(csv.DictReader((root / 'CF-failure-audit/decisions.csv').open()))
assert cf_rows[0]['actual_status'] == 'RECONSTRUCTED' and cf_rows[0]['counterfactual_status'] == 'UNKNOWN'
assert all(r['changed_cohort'] == 'UNKNOWN' for r in cf_rows)
print('AB-capacity-v1 CSV/JSON/checksum/budget/INVALID-linkage/UNKNOWN validation passed')
PYAB

guard
