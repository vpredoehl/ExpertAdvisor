# Semantic-layout inference worker routing

The canonical scheduler remains the current `LSTM_Release` process and owns
the generation-52 scheduler lease. Phase 20N adds one optional inference-only
route:

- semantic layout 7, width 77: the canonical scheduler executable;
- semantic layout 6, width 77: the explicitly configured legacy executable;
- every other layout, a missing/incomplete identity, or a width mismatch:
  fail closed without reserving capacity or launching a child.

Configure the legacy route only on the canonical scheduler invocation:

```bash
--legacy-layout6-infer-worker=/absolute/canonical/path/to/LSTM_Release
```

The configured path is resolved with `realpath` and must identify an existing,
executable regular file. No PATH search, directory scan, timestamp inference,
or automatic binary discovery occurs. Train and analyze always use the
canonical scheduler executable. Final and checkpoint inference use the same
semantic selector.

The selected executable is persisted on the reservation and exact worker
attempt, copied to the experiment or checkpoint worker fields, and used by
spawn persistence and process observation. Launch requires exact equality
between the reserved path and `argv[0]`, re-canonicalizes the path, and checks
that it remains executable. Scheduler invocation and lease identity continue
to use the canonical scheduler executable and are not derived from the worker
selection.

## Historical layout-6 worker compatibility

The approved historical source is commit
`7645265bca0c2529523e1d2cdb37e7d023dfd559`. Its worker contract was audited
against current HEAD:

- it parses and requires `--scheduler-worker-attempt-id` for scheduler-managed
  work;
- it resolves its own canonical executable and performs exact active-attempt
  registration against worker ID, experiment/checkpoint identity, worker kind,
  phase, PID, process group, process-start identity, and persisted executable;
- it persists final and checkpoint inference results through the same
  attempt-aware inference workflow used by the current scheduler;
- the inference argv emitted by the scheduler uses only options already
  accepted by that commit.

The verified binary at the time of the audit was:

```text
/Volumes/Developer SSD/ExpertAdvisor-layout6/DerivedData/Layout6/Build/Products/Release/LSTM_Release
SHA-256 945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7
```

Its source checkout is clean at the approved commit and declares semantic
layout version 6. The binary contains the exact worker-attempt registration
CLI and diagnostics. Layout 6 is not merely relabeled layout 7: the registry
records layouts 6 and 7 as incompatible same-width siblings descending from
layout 5. Layout 6 retains the historical exact-cutoff first-release surprise
contract, while layout 7 applies the corrected half-open completed-bar cutoff.

Do not use the layout-6 executable as a scheduler. Do not replace it in place
while an attempt is active. A production change should verify the checkout,
binary hash, executable permissions, and canonical path before starting the
current scheduler with the option above.
