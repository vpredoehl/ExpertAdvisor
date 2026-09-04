# Economic-event first-release actual provenance

Migration 090 adds a provider-neutral, append-only actual-observation ledger
and a fail-closed first-release selection contract. It does not change the
existing width-75 Tensor layout, semantic layout version 5, migration-088
feature view, scheduler, or any experiment row.

## Terminology

- **Event release time** is `economic_event.event_timestamp_utc`, the
  authoritative release instant or conservative causal boundary normalized to
  UTC from the source-local date, time, and IANA time zone.
- **Canonical actual** is the latest proven authoritative value exposed by the
  read-only `economic_event_canonical_actual` audit view. It may be a revision
  or correction. No existing runtime consumer is redirected to this view.
- **First-release actual** is the earliest directly evidenced authoritative
  initial value selected by `economic_event_first_release_actual`. It is not
  synonymous with the current canonical value.
- **Revision** is an explicitly source-supported later value with a positive
  source revision sequence. Its `source_publication_at` is the revision's
  publication/revision time. A revision remains a separate immutable row.
- **Correction** is a source-supported correction whose source does not supply
  the same ordered revision contract. Corrections are retained but never
  retroactively replace first-release history.
- **Observation time** (`observed_at`) is the first proved time this system
  retrieved or observed the exact evidence. Filesystem mtime is never evidence.
- **Source publication time** (`source_publication_at`) is the instant the
  source directly proves for publication of that exact value. It is NULL when
  absent; neither event time nor import time is substituted.
- **Ingestion time** (`ingested_at`) is when the observation entered this
  database. It is distinct from publication and observation.
- **Proven availability time** (`proven_available_at`) is source publication
  time when that direct proof exists, otherwise the system observation time.
  The fallback proves only late availability; it does not prove first release.

## Durable observation identity

Every observation retains the logical `economic_event_id`, source and source
role, source-native event ID, source observation ID, an observation-level
`evidence_key`, artifact path and optional SHA-256, semantic contract, raw and
canonical values, timestamps, and structured source provenance.

The unique key is `(economic_event_id, source_name, evidence_key)`. Reimporting
the same artifact observation uses the same key and is an exact duplicate.
A changed artifact, revision, correction, or alternate-source observation must
have a different evidence key and is retained separately. Reusing an evidence
key for different payload is an immutable identity conflict, not an update.
Rows cannot be updated or deleted.

The existing authoritative importer writes migration-088 compatibility data
and the migration-090 observation in one transaction. Its preflight reports an
identical retry as `duplicate_identical`, emits no inserts for that retry, and
reports a changed payload under an existing identity as `conflict`.

## Conservative backfill

Migration 090 copies migration-088 rows because those rows already require a
direct authoritative initial/revision classification, exact causal
`available_at`, and immutable source evidence. It does not modify them.

Parsed actual snapshots in `economic_event_consensus` are retained as
`secondary` / `alternate_observation`. Their source publication timestamp and
initial/revision status were not proved, so they receive
`source_publication_time_status='unavailable'`. Their historical `imported_at`
is both the conservative observation and availability boundary. They can
never qualify as first-release actuals. Myfxbook currently contributes
forecast observations only, so migration 090 manufactures no Myfxbook actual.

## Deterministic first-release selection

For each exact `economic_event_id`, selection performs these steps:

1. Consider only observations whose source role is `authoritative` and whose
   source exactly matches the event's authoritative agency.
2. Require direct `observation_kind='initial'`, `revision_sequence=0`, exact
   source publication time, and `availability_proof='source_publication'`.
3. Select the minimum source publication time, never minimum database ID,
   ingestion time, filesystem mtime, or numeric value.
4. If one semantic value exists at that earliest instant, select the
   deterministic source-observation/evidence-key order. Identical corroborating
   observations may coexist and produce the same selected value.
5. If different semantic values claim the same earliest instant, return
   `ambiguous` and no first-release value.
6. If no qualifying observation exists, return `provenance_unavailable` and no
   first-release value, even when secondary, late-backfilled, revised, or
   unclassified values exist.

A later revision or correction changes the audit-only canonical value but does
not alter the selected first release. An earlier directly proved initial may
supersede a later initial because it supplies stronger ordering evidence. A
same-time conflicting claim makes the result ambiguous instead of rewriting
history. There is no inference from “first row imported” or “oldest value.”

## Point-in-time contract

`economic_event_first_release_actual_at(T)` is the only migration-090 API that
future causal feature code may use for actuals. It returns a value only when:

- provenance state is `proven_first_release`; and
- `proven_available_at <= T`.

The boundary is inclusive: immediately before availability the row is absent;
at availability and afterward it is visible. A 2015 event first observed in
2026 with no direct publication timestamp can be audited as an observation no
earlier than 2026, but cannot be selected as first release at any cutoff.

Callers must supply the historical information cutoff, not wall-clock query
time. They must not query `economic_event_canonical_actual`, select the latest
observation, use the provider `actual_*` fields in the consensus table, or
substitute event/retrieval/ingestion time for missing source publication proof.

## Source precedence

Existing policy gives the event's authoritative government agency precedence
for release actuals. Secondary OANDA/Myfxbook observations remain distinct
evidence but are never promoted by numeric agreement or earlier ingestion.
The existing selected-consensus provider policy is unchanged.

## Read-only audit

Aggregate coverage and one-event inspection are JSON and read-only:

```sh
python3 EconomicCalendar/audit_economic_event_actual_provenance.py --db LSTM

python3 EconomicCalendar/audit_economic_event_actual_provenance.py \
  --db LSTM --event-id 123 --cutoff 2015-01-01T13:30:00Z
```

The event report includes release identity, canonical and first-release values,
provenance state/reason, all observations and classifications, every timestamp,
artifact/source identifiers, and PIT visibility at the optional cutoff.

## Phase-2 retrieval contract

Phase 2 must obtain the pre-release consensus from the existing provider-neutral
selected view and the actual only from the cutoff function:

```sql
SELECT
    e.economic_event_id,
    e.event_timestamp_utc AS event_release_at,
    c.consensus_value_kind,
    c.consensus_value_low,
    c.consensus_value_high,
    c.consensus_unit,
    c.consensus_scale,
    c.consensus_qualifier,
    c.consensus_source,
    a.first_release_actual_value_kind,
    a.first_release_actual_value_low,
    a.first_release_actual_value_high,
    a.first_release_actual_unit,
    a.first_release_actual_scale,
    a.first_release_actual_qualifier,
    a.proven_available_at,
    CASE
        WHEN a.economic_event_id IS NOT NULL THEN a.provenance_state
        WHEN f.provenance_state = 'proven_first_release'
            THEN 'not_yet_available'
        ELSE f.provenance_state
    END AS pit_provenance_state,
    f.selection_reason,
    a.source_name,
    a.source_native_event_id,
    a.source_observation_id,
    a.evidence_key
FROM economic_event e
JOIN economic_event_selected_consensus c USING (economic_event_id)
JOIN economic_event_first_release_actual f USING (economic_event_id)
LEFT JOIN economic_event_first_release_actual_at($1::timestamptz) a
    USING (economic_event_id)
WHERE e.event_timestamp_utc <= $1::timestamptz;
```

`economic_event_selected_consensus` remains the audited pre-release-consensus
contract established by migrations 081/082; its provider `actual_*` fields are
not part of the selected view. Phase 2 must additionally enforce compatible
value kind, unit, scale, and qualifier before computing surprise. This phase
does not compute surprise or add any feature channel.

The assessment view is joined only for state and reason. Phase 2 must never
read its first-release value columns directly because that view is intentionally
cutoff-independent for audit. The actual value and its proven availability time
must come exclusively from `economic_event_first_release_actual_at($1)`.
