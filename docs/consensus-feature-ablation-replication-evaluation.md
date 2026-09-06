# Consensus feature-ablation replication evaluation

Phase 7 adds this read-only command:

```text
--compare-feature-ablation-replications=601:602,603:604,605:606,607:608
```

Current generic feature replications provide the exact treatment explicitly
and use conventional control/ablation order:

```text
--compare-feature-ablation-replications=619:620,622:623 \
--expected-ablation-mask=causal_first_release_surprise_available,causal_first_release_surprise
```

Within each generic pair, the control mask must be empty and the ablation mask
must exactly match the request. Economic-calendar snapshot ID/hash must match
within a pair, but may differ across replication members. Output retains those
corpus contracts and explicitly classifies snapshot identity as reproducibility
provenance, not treatment. Omitting the expected mask retains the historical
consensus-specific argument convention below.

The declared order is authoritative and identity-significant. Every experiment
ID must be positive and globally unique across the set; duplicate pairs or an
experiment reused in another pair are rejected. Each member is evaluated with
the Phase 6 comparator: the control has exactly the four active consensus
channels ablated and the treatment has an empty ablation mask. A cross-member
ablation-identity mismatch fails closed.

The service loads all members in one repeatable-read `pqxx::read_transaction`.
It exposes no persistence or activation operation. Incomplete, missing,
invalid, and profitability-unavailable members are emitted individually and
are never silently discarded or counted as negative profitability evidence.

## Version 1 profitability policy

The canonical policy requires at least three complete comparable pairs with
both profitability deltas. `promising` requires a strict positive pair
majority on aggregate terminal-horizon return and average return per
actionable prediction, positive signed sums on both measures, and no invalid
or missing-evidence member. `not_promising` symmetrically requires strict
negative majorities and negative signed sums on both measures. With sufficient
evidence, disagreement or an integrity failure is `mixed`; fewer than three
valid pairs is `insufficient_evidence`.

Inference accuracy, leader score, neutral proportion, and actionable count are
corroborating summaries only. They cannot override profitability. Exact zero
is neutral; incomplete and invalid members do not enter any signed summary.
The command emits the complete canonical policy and its tagged FNV-1a-64 hash.

The membership identity includes every pair in declared order. The evaluation
identity includes that membership hash, every pair-evaluation identity/state,
the common ablation identity and canonical treatment mask, policy identity,
pair/replication semantic versions, per-pair validated input contracts, the
snapshot-provenance role, and the software-readiness identity.

## Readiness gate

The advisory gate deliberately separates software readiness from evidence:

- incomplete evidence with ready software: `await_replication`;
- promising evidence with ready software: `eligible_for_activation_review`;
- mixed/not-promising evidence: `do_not_enable`;
- failed software audit: `blocked_software_readiness`.

`eligible_for_activation_review` is not activation. Output always states
`activation_performed=false`; the command cannot change scheduler defaults,
experiment generation, Campaign Manager scoring, model configuration, or the
database.

Exit 0 means every declared member is complete and valid. Exit 4 means the
evaluation executed successfully but one or more members are incomplete or
lack profitability. Exit 3 means invalid or missing evidence. Argument errors
remain exit 1 and database/tool failures remain exit 2. Every scientific exit
still emits `software_success=true`.

## EconomicEventFeatures readiness audit

The audited feature contract uses the authoritative `economic_event` release
instant. Events become causal only under strict `event_time < completed-bar
cutoff`; an event exactly on a boundary first affects the next bar. A final
historical forecast is visible only at the exact release cutoff and afterward,
never on arbitrarily early bars. Later provider actuals/revisions cannot enter
the four reserved surprise channels, which remain zero until first-release
actual provenance exists.

`economic_event_selected_consensus` is the provider-neutral selected layer.
The underlying immutable row retains provider, observation identity, artifact,
semantic contract, and provider JSON; the feature loader preserves provider
identity for diagnostics. One populated provider per event prevents silent
overwrite. A later archival import may enrich a previously missing historical
forecast, but the imported value is the source-backed pre-release forecast,
not a revised actual; its legal model visibility remains anchored to the
authoritative release instant.

Missing consensus uses presence=0 with numeric channels zero, so it is distinct
from a genuine scalar zero (presence=1). Scalars duplicate low/high; ranges
require ordered finite endpoints. Source scale must be finite and positive,
the family/unit scale is fixed and dataset-independent, and normalized values
must fit finite `float` channels. Malformed shape, unit, scale, missing provider,
or float overflow fails closed.

The append-only ordering remains deterministic, width 71/layout v4 metadata is
persisted and validated, ablation zeros the four positions without changing
width, and resume/input expansion remains fail-closed. Row diagnostics count
completed rows with relevant events, selected/scalar/range/missing consensus,
and selected-provider distribution. Identical evidence produces identical
features and diagnostics.

When 603:608 finish, rerun the same ordered command under production read-only
protection (for example `PGOPTIONS='-c default_transaction_read_only=on'`) and
review the emitted scientific decision. Any activation remains a separate,
explicit future workflow.
