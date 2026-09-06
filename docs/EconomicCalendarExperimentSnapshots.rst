Immutable Economic-Calendar Experiment Snapshots
================================================

Purpose
-------

Migration 092 adds a durable identity for the economic-calendar corpus used to
construct model features.  The identity exists because the live calendar is an
append-only evidence system: a later canonical event, historical consensus
observation, or authoritative first-release backfill can legitimately improve
the live corpus while making an older experiment impossible to reproduce.

Once an experiment is bound to a finalized economic-calendar snapshot, later
economic-calendar production imports or provenance remediation cannot alter the
economic-calendar feature corpus observed by that experiment.

Existing pre-092 experiments remain unbound and retain legacy live-corpus
behavior; Phase 9 does not retroactively freeze them.  A NULL snapshot id and
NULL snapshot hash are the explicit ``legacy_live_corpus`` identity.  Phase 9
does not bind, rewrite, or reinterpret historical experiment or model rows.

Materialized corpus
-------------------

``economic_calendar_snapshot`` is the immutable header.  Its four child tables
materialize the model-facing state selected in the same repeatable-read
transaction:

* ``economic_calendar_snapshot_event`` copies canonical event content and a
  deterministic ``canonical_event_order``;
* ``economic_calendar_snapshot_consensus`` copies the selected forecast,
  provider identity, stable provenance, and historical availability proof;
* ``economic_calendar_snapshot_release_actual`` copies the migration-088
  feature projection, including its strict known-at instant; and
* ``economic_calendar_snapshot_first_release_actual`` copies the complete
  migration-090 assessment, including unavailable/ambiguous states and the
  deterministically selected proven first release.

The bound read path uses only these child tables.  It does not join back to live
calendar views.  Snapshot inspection recomputes the hash and counts from the
materialization and fails if they differ from the finalized header.

Hash contract version 1
-----------------------

The header hash is ``fnv1a64:`` followed by 16 lower-case hexadecimal digits.
FNV-1a-64 is applied to a length-delimited serialization beginning with
``economic-calendar-snapshot-v1;``.  Rows are tagged as ``event``,
``consensus``, ``release_actual``, or ``first_release`` and ordered under the
PostgreSQL ``C`` collation by tag and canonical JSONB-array text.

The canonical representation uses:

* UTC epoch microseconds for timestamps;
* explicit ``YYYY-MM-DD`` dates and ``HH24:MI:SS.US`` local release times;
* JSONB canonical text for numeric, Boolean, null, and provenance values;
* every event field projected by ``EconomicEventRepository``;
* every selected consensus value field plus stable source observation,
  artifact, semantic, provider, observed-at, forecast-available-at, and
  availability-proof fields;
* every feature release-actual field, including stable source provenance; and
* every first-release assessment/value/availability/provenance field used to
  reconstruct point-in-time state.

The canonical event order is derived from canonical event content rather than
surrogate database ids.  Child rows identify their event by that order.  Local
surrogate ids (event, consensus, report, release-actual, and observation row
ids), physical row order, ``imported_at``/``ingested_at``, and
``source_retrieved_at`` are excluded.  Those values are either insertion-order
artifacts or acquisition-time metadata and cannot alter model features.
Consequently the hash is independent of insertion order, table layout,
machine, session timezone display, and locale collation.  A change to any
frozen model-relevant value, causal availability boundary, or stable
provenance field changes the hash.

Binding lifecycle
-----------------

Fresh queue workflows acquire the snapshot-creation table lock before their
first repeatable-read data query.  They compute or reuse one finalized current
snapshot and insert the experiment's id/hash in the same transaction.  The
snapshot and experiment therefore commit together; a failure or rollback
leaves neither a partial finalized snapshot nor an experiment binding.
Duplicate identity includes both snapshot id and hash, so experiments observing
different calendar corpora are not scientifically identical.

A resume resolves the source model identity and copies it exactly.  It never
selects the latest live snapshot.  A continuation child uses the selected
source model in the same way, so the child inherits the parent/source snapshot
rather than creating a current snapshot.  A newly persisted model inherits the
snapshot from its experiment in the database trigger.  Source model and source
experiment disagreement is rejected.

Runtime train, final inference, checkpoint inference, and resume paths resolve
experiment/model/checkpoint lineage before bulk feature loading.  A complete
finalized identity is required whenever a binding exists.  Missing snapshots,
unsupported hash versions, incomplete id/hash pairs, hash mismatches, corrupt
materializations, and lineage conflicts fail closed.  NULL-bound historical
lineage continues through the legacy live loader.

Point-in-time and range behavior
--------------------------------

Snapshotting changes the evidence source, not causality.  First-release actuals
remain unavailable before ``proven_available_at`` and become visible at the
existing inclusive boundary.  Migration-088 release actuals retain their
existing strict upper-bound behavior.  Historical consensus remains associated
with its pre-release ``forecast_available_at`` proof and is not exposed before
the canonical event becomes the relevant event.

The bulk range query remains one SQL query.  It returns the latest prior row for
each authoritative ``(source_agency,event_family)`` stream plus all events in
inclusive ``[start,end]`` order.  It introduces no per-bar SQL and preserves
feature warmup and causal ordering semantics.

Explicit creation and observability
-----------------------------------

Use ``--create-economic-calendar-snapshot --dry-run`` to compute and report the
current hash and counts without writing.  Omit ``--dry-run`` only in an approved
write environment to create or reuse the finalized snapshot.  Output includes
the id (or ``DRY_RUN``), hash, canonical-event, consensus, release-actual,
proven-first-release, unavailable-provenance and ambiguous counts,
source/family counts, reuse state, and dry-run state.

``--experiment-metadata`` reports the snapshot id/hash and either
``immutable_snapshot`` or ``legacy_live_corpus``.  Read-only causal-surprise
observability and meta-analysis inspection resolve an existing experiment
binding only.  They never create a snapshot merely because an experiment is
legacy/unbound.  Snapshot creation is limited to explicit creation and queue
workflows.

Production cutover
------------------

Migration 092 must be deployed before a Phase-9-aware scheduler binary is
enabled.  Existing workers and pre-092 experiments remain compatible through
NULL-bound live behavior.  Deployment does not itself create a production
snapshot; the first approved fresh queue or explicit creation workflow does.
Phase-7 authoritative Weekly Claims actual imports and Phase-8 historical
Weekly Claims consensus imports are separate operational changes and are not
performed by Phase 9.

Phase 9 does not change model width 77, semantic layout version 6, Tensor's 73
physical features, causal-surprise columns 71/72, normalization, the
``[-10,+10]`` clamp, PIT boundaries, warmup behavior, ablation masks,
first-release selection, selected-consensus semantics, or per-bar query count.
