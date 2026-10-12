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
# Only seven untracked research paths are permitted to differ. No clean/reset,
# production build, Xcode invocation, process control, or worktree cleanup.
guard() {
python3 - "${repo_root}" <<'PY'
import hashlib, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
files = ['Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp',
         'Headers/CausalFibonacciTrendlineHistoricalArtifact.hpp',
         'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp',
         'Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp',
         'Scripts/run_causal_fibonacci_trendline_study.sh',
         'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.cpp',
         'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']
def git(*args):
    return subprocess.check_output([os.environ['EA_RESEARCH_GIT'], '-C', str(root), *args], text=True).strip()
assert git('rev-parse', 'HEAD') == '291fbd286ed19041ba8f275cc59760ebdc4025ae', 'STOP: HEAD changed'
assert git('branch', '--show-current') == 'lstm-feature-development', 'STOP: branch changed'
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only'), 'STOP: tracked files changed'
for name in files:
    assert (root / name).is_file() and not (root / name).is_symlink(), 'STOP: research file missing or symlink: ' + name
    assert not git('ls-files', '--', name), 'STOP: research file became tracked: ' + name
sql = root / 'layout13_replication.sql'
assert sql.is_file() and not sql.is_symlink(), 'STOP: protected SQL missing'
assert hashlib.sha256(sql.read_bytes()).hexdigest() == 'afcebbcb591149c9b54873ee19d707bea58d2ab33676ea1598de696b28843590', 'STOP: protected SQL changed'
assert not git('ls-files', '--', 'layout13_replication.sql'), 'STOP: SQL became tracked'
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
          root / 'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh']
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
flags=-std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic -Wno-c++23-attribute-extensions
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
    "${build_dir}/snapshot/Sources/CausalFibonacciTrendlineHistoricalEvaluationCLI.cpp" \
    "${pqxx_libs[@]}" -o "${EA_RESEARCH_EXECUTABLE}"
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
    if "${EA_RESEARCH_EXECUTABLE}" --partition final2025 --output-dir "${build_dir}/forbidden" > "${build_dir}/heldout-rejection.log" 2>&1; then
        echo "STOP: held-out option was accepted" >&2
        exit 1
    fi
    rg -q '2025 disabled before detector/database access' "${build_dir}/heldout-rejection.log"
    echo 'Immutable standalone CLI build and pre-database 2025 rejection passed'
else
    "${EA_RESEARCH_EXECUTABLE}" "$@"
fi
guard
