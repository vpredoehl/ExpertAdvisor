# Semantic-worker registry and executable identities

The scheduler reads `Builds/SemanticWorkers/registry.json` once, validates all
entries before acquiring scheduler authority, and keeps the resulting typed
`SemanticWorkerRegistry` in memory. The registry maps exact semantic layouts;
it never derives compatibility from layout ordering, experiment IDs, directory
contents, timestamps, or `layout <= current` rules.

The three executable identities are deliberately separate:

- `schedulerExecutablePath` is the executable that owns the scheduler lease
  and fencing identity.
- `currentWorkerExecutablePath` is the registry's current published worker.
  New train and final-analysis attempts use it. Checkpoint analysis remains
  in-process scheduler work and therefore uses scheduler identity.
- `selectedWorkerExecutablePath` is chosen per dispatch. Final and checkpoint
  inference both resolve the model's exact semantic layout through the same
  registry and persist that canonical immutable path before launch.

Every registered executable is stored at:

```text
Builds/SemanticWorkers/layout<N>/<git-commit>/<sha256>/LSTM_Release
Builds/SemanticWorkers/layout<N>/<git-commit>/<sha256>/manifest.json
```

The operational registry and binaries are ignored by Git. The checked-in v1
schema is `docs/semantic-worker-registry.schema.json`. Registry paths are
artifact-root-relative and must exactly match the content-addressed structure.
Startup canonicalizes them, rejects escapes and missing/non-executable files,
compares each manifest, and hashes each executable once. Polling and dispatch
reuse the validated in-memory identity, so immutable artifacts are not hashed
on every scheduler poll.

The publisher is the sole registry writer; schedulers are read-only consumers
of the startup snapshot. A startup validation failure occurs before authority
acquisition or dispatch. Recovery means restoring the accepted immutable
artifact/manifest or republishing from a successful clean Release build, never
editing a registered artifact in place. The `current` symlink is only an
operator convenience; `registry.json` remains authoritative.

Current layout 7 and historical layout 6 are explicit entries. Layout 6 is the
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

`Publish LSTM Canonical` runs only after its normal Release dependency succeeds
and calls `Scripts/PublishSemanticWorker.py`. Release provenance still requires
a clean checkout. The publisher determines the current semantic contract from
the source headers, obtains clean `HEAD`, verifies the commit is embedded in the
built executable, and computes SHA-256. It then:

1. copies into a same-filesystem staging directory;
2. verifies the staged hash and fsyncs executable, manifest, and directory;
3. atomically renames the completed directory into its immutable final path,
   refusing to overwrite or repair a conflicting existing path;
4. validates the prior registry, retains all prior artifacts, changes the old
   current rule to historical on a layout rollover, and atomically replaces
   `registry.json`;
5. only after the registry replacement, atomically updates the `current`
   convenience symlink.

Any build, provenance, copy, hash, manifest, or registry failure leaves the
previous registry authoritative. An artifact staged successfully before a
registry failure is merely unreachable and safe. Already-running attempts keep
their persisted immutable executable path across current-worker publication.
Retention is indefinite/manual and reachability-based; the publisher performs
no deletion.
