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
# Freeze this invocation's HEAD, rather than pinning all future research commits.
export EA_RESEARCH_GUARD_HEAD="$("${research_git}" -C "${repo_root}" rev-parse HEAD)"
export EA_RESEARCH_VALIDATION_ONLY=0
if [[ "${1:-}" == "--build-only" ]]; then
    [[ "$#" == 1 ]] || { echo 'STOP: --build-only takes no study arguments' >&2; exit 1; }
    export EA_RESEARCH_VALIDATION_ONLY=1
fi
# Study execution requires a clean worktree. Build-only permits unstaged edits
# confined to the explicit research source/test scope, without executing a database study.
guard() {
python3 - "${repo_root}" <<'PY'
import os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
files = ['Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp',
         'Headers/CausalFibonacciTrendlineHistoricalArtifact.hpp',
         'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp',
         'Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp',
         'Scripts/run_causal_fibonacci_trendline_study.sh',
         'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
         'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']
new_files = ['Headers/CausalFractalHistoryShadowAudit.hpp', 'Sources/CausalFractalHistoryShadowAudit.cpp', 'Tests/CausalFractalHistoryShadowAuditTests.cpp']
files += new_files
def git(*args):
    return subprocess.check_output([os.environ['EA_RESEARCH_GIT'], '-C', str(root), *args], text=True).strip()
assert git('rev-parse', 'HEAD') == os.environ['EA_RESEARCH_GUARD_HEAD'], 'STOP: HEAD changed during invocation'
assert git('branch', '--show-current') == 'lstm-feature-development', 'STOP: branch changed'
assert not git('diff', '--cached', '--name-only'), 'STOP: staged changes'
untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
validation = os.environ.get('EA_RESEARCH_VALIDATION_ONLY', '1') == '1'
assert untracked <= set(new_files) if validation else not untracked, 'STOP: unexpected untracked files'
changed = set(git('diff', '--name-only').splitlines())
if os.environ['EA_RESEARCH_VALIDATION_ONLY'] == '1':
    assert changed <= set(files), 'STOP: non-research tracked changes'
else:
    assert not changed, 'STOP: research execution requires a clean tracked working tree'
gitdir = pathlib.Path(git('rev-parse', '--absolute-git-dir'))
for name in ('MERGE_HEAD', 'REBASE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
             'BISECT_LOG', 'index.lock', 'rebase-merge', 'rebase-apply', 'sequencer'):
    assert not (gitdir / name).exists(), 'STOP: active Git operation: ' + name
for name in files:
    assert (root / name).is_file() and not (root / name).is_symlink(), 'STOP: research file missing or symlink: ' + name
    assert git('ls-files', '--', name) == name or (validation and name in new_files and name in untracked), 'STOP: research file is not tracked: ' + name
PY
}
guard
build_dir="$(mktemp -d /tmp/ea_fib_v2_build.XXXXXX)"
cleanup() {
    # This process owns this exact mktemp directory; never remove another path.
    if [[ "${build_dir}" == /tmp/ea_fib_v2_build.* && -d "${build_dir}" && ! -L "${build_dir}" ]]; then
        chmod -R u+w "${build_dir}"
        rm -rf -- "${build_dir}"
    fi
}
trap cleanup EXIT
export GIT_COMMIT="$("${research_git}" -C "${repo_root}" rev-parse HEAD)"
export EA_RESEARCH_SOURCE_MANIFEST="${build_dir}/source_manifest.sha256"
export EA_RESEARCH_SOURCE_ARCHIVE="${build_dir}/source_snapshot.tar.gz"
python3 - "${repo_root}" "${build_dir}" <<'PY'
import gzip, hashlib, pathlib, shutil, sys, tarfile
root, task = map(pathlib.Path, sys.argv[1:])
paths = sorted(p for directory in ('Headers', 'Sources', 'Common')
               for p in (root / directory).rglob('*')
               if p.is_file() and p.suffix in ('.hpp', '.h', '.cpp', '.mm'))
paths += [root / 'Scripts/tg4_analysis_config.frozen_v1.conf', root / 'Scripts/run_causal_fibonacci_trendline_study.sh',
          root / 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
          root / 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh',
          root / 'Tests/CausalFractalHistoryShadowAuditTests.cpp']
paths = sorted(set(paths))
snapshot = task / 'snapshot'
rows = []
for p in paths:
    if p.is_symlink() and not p.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('external source symlinks are not accepted')
    # Materialize in-repository aliases as read-only bytes; no mutable links survive in the snapshot.
    data = p.read_bytes()
    destination = snapshot / p.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    rows.append(hashlib.sha256(data).hexdigest() + '  ' + str(p.relative_to(root)) + '\n')
(task / 'source_manifest.sha256').write_text(''.join(rows))
# Deterministic archive: sorted names, zero timestamps/IDs, fixed permissions.
with (task / 'source_snapshot.tar.gz').open('wb') as raw:
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as archive:
            for p in sorted(snapshot.rglob('*')):
                if p.is_file():
                    info = archive.gettarinfo(str(p), arcname=str(p.relative_to(snapshot)))
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mode = 0o444
                    with p.open('rb') as source:
                        archive.addfile(info, source)
for p in sorted(snapshot.rglob('*'), reverse=True):
    p.chmod(0o555 if p.is_dir() else 0o444)
snapshot.chmod(0o555)
PY
export EA_RESEARCH_SOURCE_SHA256="$(shasum -a 256 "${EA_RESEARCH_SOURCE_MANIFEST}" | awk '{print $1}')"
export EA_RESEARCH_CONFIG_SHA256="$(shasum -a 256 "${build_dir}/snapshot/Scripts/tg4_analysis_config.frozen_v1.conf" | awk '{print $1}')"
export EA_RESEARCH_COMPILER="$("${research_cxx}" --version)
flags=-std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic -Wno-c++23-attribute-extensions -lz
live_root=${repo_root}
snapshot_root=${build_dir}/snapshot
sysroot=${research_sdk}"
read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
export EA_RESEARCH_EXECUTABLE="${build_dir}/fib-trendline-study"
"${research_cxx}" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic -isysroot "${research_sdk}" \
    -Wno-c++23-attribute-extensions \
    "-DEA_RESEARCH_LIVE_ROOT=\"${repo_root}\"" \
    "-DEA_RESEARCH_SNAPSHOT_ROOT=\"${build_dir}/snapshot\"" \
    -I"${build_dir}/snapshot/Headers" -I"${build_dir}/snapshot/Sources" "${pqxx_cflags[@]}" \
    "${build_dir}/snapshot/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${build_dir}/snapshot/Sources/TG4HistoricalMarketDataRepository.cpp" \
    "${build_dir}/snapshot/Sources/RuntimeDatabaseConnection.cpp" \
    "${build_dir}/snapshot/Common/HistoricalFxTimestamp.cpp" \
    "${build_dir}/snapshot/Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp" \
    "${build_dir}/snapshot/Sources/CausalFractalHistoryShadowAudit.cpp" \
    "${build_dir}/snapshot/Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp" \
    "${pqxx_libs[@]}" -lz -o "${EA_RESEARCH_EXECUTABLE}"
# Verify snapshot bytes after compilation, before any possible database access.
(cd "${build_dir}/snapshot" && shasum -a 256 -c "${EA_RESEARCH_SOURCE_MANIFEST}" > "${build_dir}/snapshot-verification.log")
research_libraries="$(python3 - "${EA_RESEARCH_EXECUTABLE}" <<'PYLIB'
import hashlib, os, pathlib, subprocess, sys
binary = pathlib.Path(sys.argv[1])
def dependencies(path):
    return subprocess.check_output(['otool', '-L', str(path)], text=True).splitlines()[1:]
pqxx = next(pathlib.Path(line.strip().split(' (', 1)[0]) for line in dependencies(binary) if 'libpqxx' in line)
pq = next(pathlib.Path(line.strip().split(' (', 1)[0]) for line in dependencies(pqxx) if '/libpq.' in line)
config = pq.resolve().parent.parent / 'bin' / 'pg_config'
if not config.is_file():
    raise RuntimeError('cannot identify linked libpq version; provenance publication prohibited')
version = subprocess.check_output([str(config), '--version'], text=True).strip()
print('libpqxx=' + subprocess.check_output(['pkg-config', '--modversion', 'libpqxx'], text=True).strip())
print('linked_libpq=' + str(pq) + '; ' + version)
print('linked_libpq_sha256=' + hashlib.sha256(pq.read_bytes()).hexdigest())
print('linked_libpqxx_sha256=' + hashlib.sha256(pqxx.read_bytes()).hexdigest())
PYLIB
)"
export EA_RESEARCH_LIBRARIES="${research_libraries}"
export EA_RESEARCH_EXECUTABLE_SHA256="$(shasum -a 256 "${EA_RESEARCH_EXECUTABLE}" | awk '{print $1}')"
guard
if [[ "${1:-}" == "--build-only" ]]; then
    "${EA_RESEARCH_EXECUTABLE}" --help
    python3 - "${EA_RESEARCH_EXECUTABLE}" <<'PYCLI'
import datetime, subprocess, sys
binary = sys.argv[1]
def invoke(args):
    # --help guarantees no database execution even if a rejection regresses.
    return subprocess.run([binary, *args, '--help'], text=True, capture_output=True, check=False)
def bounded(start, end):
    return ['--bounded-development', '--start-utc', start, '--end-utc', end]
start, end = '2018-01-01T00:00:00Z', '2018-02-01T00:00:00Z'
for tier in ('summary', 'full'):
    result = invoke(['--episode-publication', tier])
    assert result.returncode == 0 and f'episode_publication={tier}' in result.stdout
assert 'episode_publication=summary' in invoke([]).stdout
import tempfile, pathlib
with tempfile.TemporaryDirectory(prefix='ea_shadow_cli_help_') as task:
    plan = pathlib.Path(task) / 'sidecar'
    result = invoke(['--history-shadow-audit-dir', str(plan)])
    assert result.returncode == 0 and not plan.exists(), 'help must neither create artifacts nor connect to PostgreSQL'
invalid = [
    (['--history-shadow-audit-dir'], 'duplicate or incomplete'),
    (['--history-shadow-audit-dir', '/tmp/a', '--history-shadow-audit-dir', '/tmp/b'], 'duplicate or incomplete'),
    (['--episode-publication', 'sample'], 'requires summary|full'),
    (['--episode-publication'], 'requires summary|full'),
    (['--episode-publication', 'summary', '--episode-publication', 'full'], 'duplicate or incomplete'),
    (['--partition', 'final2025'], '2025 disabled before detector/database access'),
    (['--bounded-development'], 'requires both'),
    (['--bounded-development', '--start-utc', start], 'requires both'),
    (['--bounded-development', '--end-utc', end], 'requires both'),
    (bounded(end, start), 'start < end'),
    (bounded(start, start), 'start < end'),
    (bounded('2009-12-31T23:59:59Z', end), 'within development'),
    (bounded(start, '2020-01-01T00:00:00Z'), 'within development'),
    (bounded('2020-01-01T00:00:00Z', '2020-02-01T00:00:00Z'), 'within development'),
    (bounded('2025-01-01T00:00:00Z', '2025-02-01T00:00:00Z'), 'within development'),
    (bounded(start, end) + ['--partition', 'calibration'], 'development only'),
    (bounded(start, end) + ['--partition', 'final2025'], '2025 disabled'),
    (['--start-utc', start, '--end-utc', end], 'require --bounded-development'),
    (bounded(start, end) + ['--start-utc', start], 'duplicate or incomplete'),
    (bounded(start, end) + ['--bounded-development'], 'duplicate or incomplete'),
    (bounded(start, end) + ['--replay-origin', start], 'unknown'),
    (bounded(start, end) + ['--warmup-bars', '0'], 'unknown'),
]
for bad in ['2018-01-01 00:00:00', '2018-01-01T00:00:00+00:00',
            '2018-01-01T00:00:00-05:00', '2018-11-04T01:30:00',
            '2018-01-01T00:00:00.0Z', '2018-01-01T00:00:00Z ', '',
            '2018-01-01T0x:00:00Z']:
    invalid.append((bounded(bad, end), 'YYYY-MM-DDTHH:MM:SSZ'))
for bad in ['2018-02-30T00:00:00Z', '2018-01-01T24:00:00Z',
            '2018-01-01T00:60:00Z', '2018-01-01T00:00:60Z', '2019-02-29T00:00:00Z']:
    invalid.append((bounded(start, bad), 'invalid UTC calendar boundary'))
for args, message in invalid:
    result = invoke(args)
    assert result.returncode == 1 and message in result.stderr, (args, result.stdout, result.stderr)
valid = [('2010-01-01T00:00:00Z', '2010-01-02T00:00:00Z'), (start, end),
         ('2012-02-29T00:00:00Z', '2012-03-01T00:00:00Z'),
         ('2019-12-31T22:00:00Z', '2019-12-31T23:59:59Z')]
epoch = lambda value: int(datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp())
for lo, hi in valid:
    result = invoke(bounded(lo, hi))
    expected = f'bounded_development_plan replay_origin_utc_epoch=1262304000 score_start_utc_epoch={epoch(lo)} score_end_exclusive_utc_epoch={epoch(hi)} input_end_exclusive_utc_epoch={epoch(hi)}'
    assert result.returncode == 0 and expected in result.stdout, (lo, hi, result.stdout, result.stderr)
print(f'BOUNDED_CLI_VALIDATION accepted_plans={len(valid)} rejected_requests={len(invalid)} database_access=false replay_origin=2010')
PYCLI
    echo 'Immutable standalone CLI build and pre-database 2025 rejection passed'
else
    "${EA_RESEARCH_EXECUTABLE}" "$@"
fi
guard
