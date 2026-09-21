# Semantic-worker registry and executable identities

The scheduler reads `Builds/SemanticWorkers/registry.json` once, validates all
entries before acquiring scheduler authority, and keeps the resulting typed
`SemanticWorkerRegistry` in memory. The registry maps exact semantic-layout and
executable-role pairs; it
it never derives compatibility from layout ordering, experiment IDs, directory
contents, timestamps, or `layout <= current` rules.

The three executable identities are deliberately separate:

- `schedulerExecutablePath` is the executable that owns the scheduler lease
  and fencing identity.
- `currentWorkerExecutablePath` remains the existing immutable training/reference
  `LSTM_Release` binding. New train and final-analysis attempts use it.
  Checkpoint analysis remains in-process scheduler work and therefore uses
  scheduler identity.
- `selectedWorkerExecutablePath` is chosen per dispatch. Final and checkpoint
  inference both resolve the model's exact semantic layout through the same
  registry and persist that canonical immutable path before launch.

Every registered executable is stored at:

```text
Builds/SemanticWorkers/layout<N>/<role>/<git-commit>/<sha256>/<executable>
Builds/SemanticWorkers/layout<N>/<role>/<git-commit>/<sha256>/manifest.json
```

The executable directory also contains deterministic `default.metallib` and
`MetaNN.metallib` symbolic links. They resolve to a shared, content-addressed
runtime package:

```text
Builds/SemanticWorkers/runtime/<runtime-manifest-sha256>/default.metallib
Builds/SemanticWorkers/runtime/<runtime-manifest-sha256>/MetaNN.metallib
Builds/SemanticWorkers/runtime/<runtime-manifest-sha256>/manifest.json
```

`MetaNN_metal.metallib` is the build-product identity and is published under
the runtime name `MetaNN.metallib` required by MetaNN. Each registry worker
entry binds an exact runtime identity. Runtime packages may therefore be
shared while different executable generations can retain different runtime
identities if the Metal libraries change.

The operational registry and binaries are ignored by Git. The checked-in v4
schema is `docs/semantic-workers/semantic-worker-registry.schema.json`. Registry paths are
artifact-root-relative and must exactly match the content-addressed structure.
Startup canonicalizes them, rejects escapes and missing/non-executable files,
compares each manifest, and hashes each executable and runtime resource.
Admission revalidates the selected worker's two runtime links and resource
hashes before capacity reservation or process spawn. An incomplete runtime is
reported as `SCHEDULER_SEMANTIC_WORKER_RUNTIME_UNAVAILABLE`; the experiment is
left pending and no worker attempt is created.

The publisher is the sole registry writer; schedulers are read-only consumers
of the startup snapshot. A startup validation failure occurs before authority
acquisition or dispatch. Recovery means restoring the accepted immutable
artifact/manifest or republishing from a successful clean Release build, never
editing a registered artifact in place. The `current` symlink is only an
operator convenience; `registry.json` remains authoritative.

Current layout 7 has two explicit bindings: `train` points at the preserved
immutable `LSTM_Release` training/reference artifact and `infer` points at the
immutable `lstm-infer-worker`. Historical v2/v3 layout-only entries remain
readable: their established capability binding is interpreted as the legacy
`LSTM_Release` artifact and is never inferred from its filename. Layout 6 is the
accepted inference-only artifact from commit
`7645265bca0c2529523e1d2cdb37e7d023dfd559`, SHA-256
`945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7`.
Normal routing does not depend on the separate layout-6 worktree.

The optional scheduler-only setting is:

```bash
--semantic-worker-registry=/absolute/path/to/registry.json
```

Without it, the scheduler uses
`$PWD/Builds/SemanticWorkers/registry.json`. This option is never placed in a
child command. `--legacy-layout6-infer-worker` remains temporarily accepted as
an identity assertion: the supplied canonical path must equal the registry's
layout-6 path. It cannot override the registry and is never inherited by a
child. Remove the flag after production launch configuration no longer passes
the old external path and a registry-backed scheduler restart has been
operationally validated.

## Publishing and rollover

`Publish LSTM Canonical` is now reserved for ordinary `LSTM_Release`
publication.  With `PUBLISH_CANONICAL_LSTM_RELEASE=YES`, it runs only after its
normal Release dependency succeeds and calls
`Scripts/PublishCanonicalLSTMRelease.py`; it never calls the semantic-worker
publisher or changes the semantic-worker registry. Semantic workers remain
published explicitly with `Scripts/PublishSemanticWorker.py`. Release
provenance still requires a clean checkout. The semantic publisher determines
the current semantic contract from the source headers, obtains clean `HEAD`,
verifies the commit is embedded in the built executable, and computes SHA-256.
It then:

1. publishes and verifies the content-addressed shared runtime package;
2. copies the executable into a same-filesystem staging directory;
3. verifies the staged hash and fsyncs executable, manifest, and directory;
4. atomically renames the completed directory into its immutable final path,
   refusing to overwrite or repair a conflicting existing path;
5. attaches deterministic resource links to the new worker and, during legacy
   registry migration, to existing worker directories without changing their
   executable or manifest identity;
6. validates the prior registry, retains all prior artifacts, changes only the
   old current binding for the same role to historical on a role/layout rollover,
   and atomically replaces `registry.json`;
7. only after the registry replacement, atomically updates the `current`
   convenience symlink.

Any build, provenance, copy, hash, manifest, or registry failure leaves the
previous registry authoritative. An artifact staged successfully before a
registry failure is merely unreachable and safe. Already-running attempts keep
their persisted immutable executable path across current-worker publication.
Retention is indefinite/manual and reachability-based; the publisher performs
no deletion.
