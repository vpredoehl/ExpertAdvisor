# Fibonacci incremental-information V2 — 2025 confirmation runner

This operational note describes the dedicated execution boundary for the frozen
protocol `causal-fibonacci-layout9-incremental-information-v2`. It does not
change the protocol, its interpretation contract, or the status of its
pre-2025 evidence.

The command is intentionally separate from the sealed pre-2025 runner:

```text
LSTM_Release --fibonacci-incremental-information-confirm-2025 \
  --artifact-dir /absolute/path/to/causal-fibonacci-layout9-incremental-information-v1 \
  --frozen-pre2025-result-dir /absolute/path/to/causal-fibonacci-layout9-incremental-information-v1-pre2025-results-v2 \
  --output-dir /absolute/path/to/new-confirmation-output \
  --code-commit COMMITTED_SOURCE_SHA \
  --authorize-fibonacci-confirmation-2025 \
  --enforce-no-refit
```

This command is for a later, separately authorized, one-time read-only
execution. It must not be run as part of build or qualification work.

Before any row is read, the runner requires both explicit flags and verifies
the frozen protocol document, immutable extraction manifest/row hashes, and
the complete V2 pre-2025 result identity. The latter binds runner identity,
protocol ID/SHA, source manifest/row hashes, code lineage, solver settings,
checksum-file identity, and `sealed_and_discarded_before_parsing` state.

The data boundary is fixed by construction:

```text
development rows -> development-only transforms and fixed diagnostic fits
confirmation_2025 rows -> evaluation-only frozen diagnostics
```

Validation and pre-2025 lock-test rows are not admitted to either operation.
The confirmation reader and development reader each inspect the partition
first and reject all other partitions before parsing row values. No 2025 value
can influence scaling, bin edges, reconstruction fits, multinomial fits,
feature selection, or solver/model specification.

The output path must not exist. Results are written to a same-filesystem
incomplete staging directory, checksummed and verified there, marked complete,
then atomically renamed to the requested output path. Output contains only the
frozen structural, Fibonacci-ledger, reconstruction, conditional, monthly,
association, coverage, and equal-symbol diagnostics; no profitability artifact
is produced. The manifest binds input and pre-2025 provenance, execution
commit, authorization/no-refit state, frozen solver settings, and confirmation
partition. A completed directory can be independently verified from its
manifest, completion marker, and checksums.

The runner reports frozen metrics only. It does not assign a final protocol
classification or adapt any rule using confirmation results.
