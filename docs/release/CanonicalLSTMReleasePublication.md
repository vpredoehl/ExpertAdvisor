# Ordinary canonical `LSTM_Release` publication

## Development Release build

Run `Scripts/build-lstm-release.sh` from any working directory to build
`ExpertAdvisor.xcodeproj`, scheme `LSTM Release`, configuration `Release`, with
`CODE_SIGNING_ALLOWED=NO`. It resolves the repository from its script location
and reuses `DerivedData/ExpertAdvisor` without cleaning or overriding
`CONFIGURATION_BUILD_DIR`, preserving incremental caches. The legacy
`Scripts/build-release.sh` delegates to this command.

Use `--help` for usage or `--dry-run` to inspect the resolved paths and exact
command without building or creating logs. Real builds retain full stdout and
stderr in a unique UTC-timestamped log under
`DerivedData/ExpertAdvisor/BuildLogs/`. The command reports Git HEAD, result,
executable path, SHA-256 and log path. A successful build must produce an
executable at `DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`.
Xcode failures retain their original exit code; logging or output-verification
failures after Xcode succeeds return 1.

The existing Xcode provenance phase still requires a clean source tree for
Release. Publication settings are explicitly disabled; the command does not
run the executable, deploy artifacts, push changes or control schedulers.

## Publication

`Scripts/PublishCanonicalLSTMRelease.py` publishes only the ordinary
compatibility/operator/training `LSTM_Release`. It is intentionally distinct
from `Scripts/PublishSemanticWorker.py`; it neither reads nor changes
`Builds/SemanticWorkers/registry.json` or any semantic artifact.

The publisher requires a clean Git checkout and a Release executable whose
embedded source commit exactly matches clean `HEAD`. It runs the executable's
harmless `--help` command, hashes it, and retains it at:

```
Builds/Canonical/LSTM_Release/<source-commit>/<sha256>/LSTM_Release
```

The adjacent immutable manifest records the source commit, SHA-256, role, and
schema. A pre-existing content-addressed directory is verified, never replaced;
a byte or manifest conflict fails closed. The publisher stages and fsyncs the
artifact before its atomic directory rename. It then atomically replaces only
the convenience symlink `DerivedData/Canonical/LSTM_Release`, and verifies its
resolved artifact, hash, embedded commit, and harmless invocation afterward.

To invoke this from Xcode, build the `Publish LSTM Canonical` aggregate target
in Release with `PUBLISH_CANONICAL_LSTM_RELEASE=YES`. The target's `LSTM Release`
dependency supplies the candidate product. Routine Release builds do not
publish artifacts. The canonical symlink changes path resolution only; it never
overwrites a retained artifact or a running executable image.
