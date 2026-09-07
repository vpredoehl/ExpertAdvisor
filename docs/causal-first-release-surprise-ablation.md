# Causal first-release surprise contract

The layout-v7 pair is:

```text
causal_first_release_surprise_available
causal_first_release_surprise
```

For a 15-minute input bar beginning at `barStart`, the completed information
set is the half-open interval ending at `cutoff = barStart + 15 minutes`.
An economic event and its first-release actual are usable only when their
respective timestamps are strictly earlier than `cutoff`. Evidence published
exactly at the cutoff first becomes usable on the next bar. Snapshot creation
and ingestion timestamps never determine feature availability.

The actual must come from `economic_event_first_release_actual_at(...)` (or the
equivalent immutable snapshot materialization) with state
`proven_first_release`. Selection admits only authoritative observations
explicitly classified as initial/revision-sequence zero with exact source
publication evidence. The earliest source publication wins; conflicting
semantic values at the earliest instant and unresolved possible-first values
are `ambiguous` and fail closed. Canonical/latest revised actuals and secondary
provider actuals are not feature inputs.

The forecast is the provider-neutral `economic_event_selected_consensus` row
associated with that event. It is exposed no earlier than the authoritative
release boundary, preventing a final historical pre-release forecast from
appearing on arbitrary earlier bars. The selected row is immutable and unique
per event; provider identity and retained artifact provenance are diagnostic,
not numeric inputs. Only scalar forecast/actual pairs with identical unit,
source scale, and qualifier are compatible.

For compatible evidence:

```text
raw_difference = first_release_actual_canonical - consensus_canonical
surprise = clamp(raw_difference / fixed_family_scale, -10, +10)
available = 1
```

The sign is positive when the actual exceeds consensus. Fixed scales are 10
percentage points for percent families, 1,000,000 counts for employment and
Weekly Claims, and 10,000,000 counts for JOLTS. This is fixed-scale feature
normalization, not division by consensus, so there is no zero-denominator
case. Missing consensus, unavailable or ambiguous first release, semantic
incompatibility, and non-finite values produce `(available, surprise) = (0,0)`.
A genuine zero surprise produces `(1,0)`.

## Controlled ablation and identity

The treatment mask is exactly:

```text
causal_first_release_surprise_available,causal_first_release_surprise
```

The formal ablation hook copies the persisted-width Tensor prefix and then
zeros those two columns. It does not remove columns or change model width; all
other Tensor-derived inputs remain byte-identical. The canonical mask is part
of experiment identity and model lineage. Resume, checkpoint retry, and
inference reload the mask from that lineage and reject mismatches. Width 77 and
semantic layout 7 independently identify the corrected half-open channel
semantics. Layout 6 used the same width and columns but exposed a first-release
actual at an exact completed-bar cutoff; it is an incompatible pre-fix sibling
of layout 7, not a resume-compatible predecessor.

Economic-calendar snapshot ID/hash is a separate corpus identity. Corrected
layout-7 evidence requires both arms to carry the same non-null persisted
snapshot ID and tagged hash. A missing, partial, or mismatched snapshot
identity fails closed. Snapshot identity must never stand in for the feature
treatment.

## Replication interpretation

Use the modern read-only comparison form:

```text
--compare-feature-ablation-pair=CONTROL_ID:ABLATION_ID \
--expected-ablation-mask=causal_first_release_surprise_available,causal_first_release_surprise
```

For multiple pairs, use the same expected mask with
`--compare-feature-ablation-replications`. The output preserves each pair's
snapshot and model-input identities and reports corrected-valid and historical
pre-fix counts separately.

Layout 6 evidence is classified as `pre_fix_causal_surprise_evidence`; it is
historical evidence and contributes zero to the corrected replication count.
Experiments 622/623 are the protected production layout-6 pair. Layout 7 is
classified as `corrected_causal_surprise_pair_evidence`. Experiments 624/625
are the first protected production corrected pair while their persisted
identities remain width 77, layout 7, and snapshot 1 with the expected hash.
They must complete before a performance conclusion is drawn.

The corrected replication minimum remains three complete, valid layout-7
pairs. Reaching it produces
`corrected_causal_surprise_replication_evidence`; a single corrected pair is
not replicated evidence. Additional independent corrected replications are a
separate scientific step and are not launched by the evaluation commands.
