# Phase 20 current-state audit (pre-implementation)

Audit date: 2026-09-01

## Repository and operational state

- Branch: `lstm-feature-development`.
- Starting worktree: clean.
- Starting commit: `c0e9cd9c771360b61b1b21f1dc2c9f734ed1b35a`.
- Production schema ledger ends at migration 088.
- Production corpus contains 2,601 USD economic events, 1,521 selected
  consensus rows, and 1,230 provenance-certified initial actual rows.
- The active scheduler uses the audited `DerivedData/ExpertAdvisor` Release
  binary. Two production training workers are active. Experiment 610 is the
  first width-75 continuation and is currently expanding model 1702 from width
  71 to width 75. No Phase 20 validation may contend with those workers.

## Existing Phase 19 contract

Phase 19 already implemented the production feature values and most LSTM
plumbing. The current layout is:

- 49 pre-economic Tensor columns.
- 10 event occurrence/recency columns (Tensor 49-58).
- 8 consensus-era columns (Tensor 59-66), of which the four historical
  provider-surprise positions 63-66 are permanently closed zeros.
- 4 authoritative initial-release surprise columns (Tensor 67-70).
- 4 model-only causal return columns (model inputs 71-74).
- 71 Tensor columns and 75 total model inputs.

`Tensor::Add` constructs all 22 economic columns in one append-only order.
Training, classification inference, regression inference, final inference, and
checkpoint inference project Tensor rows through the same
`ModelInputContract` and append the same four causal return columns. Widths 53,
63, 71, and 75 remain registered structural prefixes.

The event engine is causal at a strict completed-bar cutoff. Future events are
not consumed. Consensus is exposed at an exact release boundary or from the
most recent released event. A certified actual requires `available_at` to be
strictly earlier than the per-bar cutoff. Revision rows are excluded by the
schema-088 feature view. Provider actuals are ignored. Malformed values and
provenance fail with an exception; missing consensus, missing actual, and
incompatible semantics have deterministic availability-indicator/zero
semantics.

Model persistence already stores structural width in `model_meta` and semantic
layout V5 in `model_input_semantics_meta`. Explicit append-only expansion
validates the source layout, moves the four return rows, zero-initializes only
the appended Tensor rows, persists ancestry/provenance, and never mutates the
source model. Scheduler train, final-infer, and checkpoint-infer commands use
the same executable and database contract; analysis consumes persisted
inference rather than rebuilding features.

## Confirmed integration gaps

1. A newly queued experiment does not persist expected model input width or
   semantic layout. Queue duplicate identity therefore cannot distinguish an
   old width-71 fresh experiment from an otherwise identical width-75 fresh
   experiment, and a pending fresh experiment is not bound against a later
   scheduler binary's feature layout.
2. `model_input_semantics_meta` is enforced for explicit width expansion and
   downstream paired-evaluation reads, but ordinary `PgModelIO::loadAll` does
   not reject an incompatible marker-bearing model. Structural width still
   rejects shape mismatches, but the persisted semantic identity is not
   uniformly enforced at normal inference/resume load.
3. Database/query failures and malformed rows fail explicitly, but a completely
   empty USD `economic_event` corpus returns an empty vector and is
   indistinguishable from a legitimate event-free interval. The normal runtime
   would consequently construct valid-looking all-zero event features.
4. Existing tests prove width-75 Tensor construction, training/inference row
   parity, width expansion, model persistence, and the real production
   forward/BPTT paths separately. They do not combine nonzero authoritative
   event columns with a fresh width-75 LSTM training/inference/persistence
   smoke in one focused test.

## No gaps found

- No scheduler-only economic-event option exists or is needed: currency,
  family mapping, normalization constants, recency constant, ordering, and
  layout are compile-time feature semantics, while event values come from the
  same read-only LSTM database for direct and scheduler workers.
- Resume already uses the established append-only expansion mechanism; a
  second mechanism is neither needed nor permitted.
- Final and checkpoint inference already resolve persisted width and use the
  same Tensor construction and model-input materialization code.
- Campaign/profitability paths consume persisted model/inference evidence and
  already carry model input width where population homogeneity requires it.
- No migration is needed for event values, actual revisions, consensus, or
  model matrices. Persistent experiment feature identity does require a new,
  forward-only migration.
