#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build-lstm-release.sh [--help] [--dry-run]

Build the development LSTM Release scheme using incremental Release caching.
  --help     Show this help without building.
  --dry-run  Print resolved paths and command without building or creating logs.

Build output is retained under DerivedData/ExpertAdvisor/BuildLogs/.
Existing Release provenance checks apply. No publication or executable launch.
EOF
}

dry_run=false
for argument in "$@"; do
  case "$argument" in
    --help) usage; exit 0 ;;
    --dry-run) dry_run=true ;;
    *) printf 'Unknown option: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
project="$repo_root/ExpertAdvisor.xcodeproj"
derived_data="$repo_root/DerivedData/ExpertAdvisor"
executable="$derived_data/Build/Products/Release/LSTM_Release"
log_directory="$derived_data/BuildLogs"
commit="$(/usr/bin/git -C "$repo_root" rev-parse --verify HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
build_command=(
  xcodebuild
  -project "$project"
  -scheme "LSTM Release"
  -configuration Release
  -derivedDataPath "$derived_data"
  CODE_SIGNING_ALLOWED=NO
  PUBLISH_CANONICAL_LSTM_RELEASE=NO
  PUBLISH_CANONICAL_BINARY=NO
  build
)

summary() {
  printf '\nGit commit: %s\nBuild result: %s\nExecutable path: %s\nSHA-256: %s\nLog path: %s\n' \
    "$commit" "$1" "$executable" "$2" "$3"
}

if "$dry_run"; then
  printf 'Repository: %s\nProject: %s\nScheme: LSTM Release\nConfiguration: Release\nDerivedData: %s\n' \
    "$repo_root" "$project" "$derived_data"
  printf 'Command:'
  printf ' %q' "${build_command[@]}"
  printf '\n'
  summary 'DRY RUN (not executed)' 'not computed' "$log_directory/lstm-release-$timestamp.XXXXXX (not created)"
  exit 0
fi

cd "$repo_root"
mkdir -p "$log_directory"
log_path="$(mktemp "$log_directory/lstm-release-$timestamp.XXXXXX")"

# Capture both pipeline statuses before any other command changes PIPESTATUS.
set +e
"${build_command[@]}" 2>&1 | tee "$log_path"
pipeline_status=("${PIPESTATUS[@]}")
set -e
build_status="${pipeline_status[0]}"
if (( build_status != 0 )); then
  summary "FAILED (xcodebuild exit $build_status)" 'not computed' "$log_path"
  exit "$build_status"
fi
if (( pipeline_status[1] != 0 )); then
  summary 'FAILED (build output logging failed; xcodebuild exit 0)' 'not computed' "$log_path"
  exit 1
fi
if [[ ! -f "$executable" || ! -x "$executable" ]]; then
  summary 'FAILED (expected executable missing or not executable; xcodebuild exit 0)' 'not computed' "$log_path"
  exit 1
fi
if ! hash_output="$(/usr/bin/shasum -a 256 "$executable")"; then
  summary 'FAILED (SHA-256 verification failed; xcodebuild exit 0)' 'not computed' "$log_path"
  exit 1
fi
summary 'SUCCESS (xcodebuild exit 0; executable verified)' "${hash_output%% *}" "$log_path"
