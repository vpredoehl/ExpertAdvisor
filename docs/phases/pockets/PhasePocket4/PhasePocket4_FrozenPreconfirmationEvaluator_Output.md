---
title: "Phase Pocket 4 Frozen Preconfirmation Evaluator"
document_type: "implementation report"
status: "implemented"
---

# Phase Pocket 4 — frozen Phase Pocket 3 read-only preconfirmation evaluator

## Scope and authority

This implementation uses the frozen Phase Pocket 3 protocol at
`docs/phases/pockets/PhasePocket3/PocketProspectiveEmpiricalEvaluationProtocolFreeze.md`.
That artifact remains the scientific authority.  Phase Pocket 4 adds no rule,
outcome, selection criterion, detector predicate, or model feature.

`CausalPocketDetector` remains the Phase Pocket 2 **project operational
definition**, not a mathematical transcription of the MTI manual.  Its pure
implementation now accepts the protocol's frozen 10/15/20 lookback values;
its default remains the Phase Pocket 2 source-default 15 bars.

## Architecture and firewall

```text
canonical immutable config -> read-only source preflight -> causal Pocket replay
    -> bounded retrospective outcome labels -> deterministic aggregates -> atomic artifact
```

The command opens PostgreSQL in `REPEATABLE READ, READ ONLY` mode and uses only
the canonical 15-minute completed-candlestick source and six frozen tables.
For preconfirmation, the query's exclusive upper bound is always
`2025-01-01T00:00:00Z`.  The validation command follows that same boundary and
does not instantiate the outcome labeler.  The labeler is a separate API and
reads no more than its explicit 4, 16, or 64 future-bar horizon.

The fixed config is [pocket_prospective_preconfirmation_v1.conf](../../../../Scripts/pocket_prospective_preconfirmation_v1.conf).
It has byte-stable canonical serialization, a SHA-256 identity, exact source
mapping, partitions, horizons, warmup, censoring, ordering, and bootstrap
contract.  Parsing rejects any altered, missing, unknown, malformed, or
noncanonical field and validates the frozen protocol-document hash.

## CLI

The evaluator is isolated in the `PocketResearch Release` Xcode target.  It
links only its entry point, the shared Pocket evaluator headers, timestamp
parsing, and libpq/libpqxx; it does not link the LSTM runtime, scheduler,
training, inference, Tensor, or experiment components. `LSTM_Release` no
longer exposes this Pocket command.

Preconfirmation review (read-only and outcome-blind):

```text
DerivedData/PocketResearch/Build/Products/Release/PocketResearch_Release --pocket-prospective-evaluator --validate-only --config Scripts/pocket_prospective_preconfirmation_v1.conf --output-dir /absolute/new/output-dir --git-commit <40-hex-HEAD> --executable-identity <sha256-of-PocketResearch_Release>
```

Real preconfirmation evaluation (implemented, but not run by Phase Pocket 4):

```text
DerivedData/PocketResearch/Build/Products/Release/PocketResearch_Release --pocket-prospective-evaluator --execute --config Scripts/pocket_prospective_preconfirmation_v1.conf --output-dir /absolute/new/output-dir --git-commit <40-hex-HEAD> --executable-identity <sha256-of-PocketResearch_Release>
```

Verification is read-only and does not rerun an evaluation:

```text
DerivedData/PocketResearch/Build/Products/Release/PocketResearch_Release --verify-pocket-prospective-artifact /absolute/output-dir
```

The output target must be absent.  Publication writes a sibling temporary
directory, hashes and verifies its complete content, then atomically renames
it. Existing targets, incomplete directories, and digest mismatch fail closed.
The manifest captures config/protocol/detector identities, source audit data,
database identity, full Git commit, executable SHA-256, row/timestamp bounds,
and read-only snapshot status.

## Frozen outcomes and reporting

The separate bounded evaluator implements inclusive directional touch and
close contact, 4/16/64 future-bar windows, right-censoring at the first gap,
invalid bar, source boundary, or tail, nonnegative directional MFE/MAE,
signed directional close return, and the frozen 64-bar race.  Future labels
cannot affect the causal detector.  Metrics use complete-window denominators;
censors remain explicit.  The deterministic 2,000-replicate UTC-week block
bootstrap is seeded from the immutable configuration SHA-256.  Greedy thinning
retains the nonoverlap equality boundary (`next >= last + 64`).

## Tests and non-goals

Synthetic tests cover detector 15-parity and 10/20 behavior, confirmation
availability, malformed/duplicate data rejection, firewall range rejection,
touch/fill equality for both directions, bounded horizon reads, gap/tail
censoring, deterministic metrics/bootstrap/thinning, config/hash validation,
atomic publication, existing-target rejection, and tamper detection. Existing
Causal Pocket detector and market-structure registry regression tests also run.

No empirical Pocket evaluation was run.  No Pocket outcome/result artifact was
read.  No database write, migration, Forex-data edit, Tensor/layout change,
training/inference/scheduler/experiment action, TG4 comparator, or profitability
ranking action is part of this phase.
