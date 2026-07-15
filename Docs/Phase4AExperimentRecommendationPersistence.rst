Phase 4A Step 3: advisory recommendation persistence
=====================================================

Phase 4A remains recommendation-only.  Generation is an explicit command: it
does not create experiments, enter the scheduler loop, consume worker slots,
change an experiment, evaluate a continuation, or approve or queue anything.
Scoring and recommendation-to-experiment conversion are deliberately deferred.

Storage and provenance
----------------------

Migration 026 adds ``experiment_recommendation_scan`` and upgrades or creates
``experiment_recommendation``.  Additive migration 027 strengthens positive
scan-filter, nonempty failure-error, and symmetric recommendation-status
metadata constraints without rewriting legacy prototype rows.  Every explicit
generation invocation first
creates a scan row, even when no source is eligible or no proposal is stored.
The scan records the exact Step 1 policy canonical text and hash, filters,
counters, completion state, and any fatal error.  A recommendation records the
scan, final source experiment/model/analysis, evidence metrics, Step 2 mutation
and distance metadata, both canonical configuration identities and hashes, and
the exact policy identity.

Only ``proposed`` rows are created.  ``approved`` is reserved for a later
explicit workflow.  Existing ``rejected`` and ``expired`` rows are historical
dispositions: generation reports ``historical_recommendation`` and never
reactivates or changes them.  Existing ``proposed`` or ``approved`` rows report
``active_recommendation`` and block another active proposal.

Canonical text is authoritative equality.  FNV-1a hashes are lookup
accelerators, not uniqueness proof.  The active partial unique index contains
the full semantic and policy canonical texts.  Persistence also rechecks those
texts inside a short transaction protected by an advisory lock.  Equal hashes
with unequal canonical texts are reported and counted as collisions; they are
never merged.  Collision records identify semantic, invocation, or policy
identity explicitly.  Legacy rows lacking Step 3 canonical fields remain untouched and
are excluded from the active canonical index rather than assigned guessed
identities.

The live migration history may contain historical migration 025,
``025_experiment_recommendation.sql``.  Its authoritative source is not present
in this repository and must not be reconstructed or rewritten.  Migration 026
therefore supports both fresh schema creation and upgrade from that legacy
prototype shape.  Migration 027 adds its hardening checks as ``NOT VALID``:
PostgreSQL does not scan historical rows when adding them, but it does enforce
the checks on every new insert or update.  They remain unvalidated until a
separate, deliberate ``VALIDATE CONSTRAINT`` operation proves all historical
rows conform.

Unmapped legacy prototype rows are likewise omitted from Step 3 list/detail
mapping, whose typed contract requires a scan and both authoritative canonical
identities.

Source evidence and dates
-------------------------

The repository reads only experiments in ``completed``/``done`` and joins the
unique ``final`` analysis whose model ID equals ``experiment.last_model_id``.
Missing final models, missing or incomplete final analysis, invalid metrics,
and unsupported date mappings produce structured source skips.  Checkpoint
analysis and analysis belonging to an older final model cannot supply source
evidence.  Neutral proportion and evidence count are derived from the three
persisted final prediction counts.

The current experiment/model rows do not provide an authoritative, complete
round trip for all historical architecture, feature, label, class-weight,
head-bias, optimizer, and normalization settings identified by the Step 1
audit.  Step 3 does not invent a universal configuration hash or infer those
values from build metadata.  Recommendation mutation remains restricted to the
Step 1 allowlist and preserves source/model provenance; enforcing additional
implementation-contract variants is deferred until authoritative persisted
metadata exists.

The four experiment range columns are physically ``timestamptz``.  Existing
workers consume Chicago-local ``YYYY-MM-DD`` boundaries.  Step 3 therefore
extracts each boundary explicitly with ``AT TIME ZONE 'America/Chicago'`` and
``to_char(..., 'YYYY-MM-DD')``.  It also verifies that the stored value is local
midnight.  A non-midnight historical value is skipped as
``ambiguous_experiment_date_mapping`` instead of being silently truncated or
made dependent on the PostgreSQL session timezone.

Selection and deterministic order
---------------------------------

Eligible sources are grouped by the policy scope: ``symbol_horizon``,
``symbol``, or ``global``.  Within each group the top-source order is:

1. leader score descending;
2. inference accuracy descending;
3. evidence count descending;
4. source experiment ID ascending.

The service retains ``top_sources_per_scope`` and reports overflow with
``outside_top_sources_per_scope``.  It invokes the unchanged pure Step 2
generator for each retained source.  All resulting candidates are then ordered
by group key, source rank, Step 2 parameter order, proposed numeric value,
semantic canonical text, invocation canonical text, and source experiment ID.
The per-scan maximum is applied only after this deterministic order.  A
``structural_rank`` is an ordering position, not a recommendation score.

Duplicate semantics
-------------------

Experiment duplicate detection reconstructs complete Step 1 identities and
compares canonical text; it never compares only the mutated column.  Semantic
configuration identity answers whether the same research configuration already
exists and therefore blocks a Phase 4A proposal even if resume-model or
checkpoint invocation identity differs.  Exact invocation equality is retained
as diagnostics for a future policy extension.

Pending, paused, running, and completed semantic matches are
``existing_experiment``.  Failed and cancelled matches are
``excluded_terminal_experiment`` and block only when
``terminal_experiments_are_duplicates`` is enabled.  A nonterminal/completed
match takes precedence over a terminal match; ties use the lowest experiment
ID.  Recommendation duplicate identity is semantic canonical configuration
plus canonical policy.  A materially different policy may preserve a separate
proposal and provenance.

Because active uniqueness is semantic configuration plus policy, multiple
sources proposing the same candidate under the same policy share one active
row and its primary source provenance.  Multi-source attribution is deferred;
it is not inferred or appended by Step 3.

Transactions and concurrency
----------------------------

Source loading uses a read-only transaction.  Grouping, eligibility,
generation, and sorting occur in memory.  Each candidate uses a short explicit
read-write transaction that rechecks experiments and recommendations before an
idempotent insert.  Concurrent identical generation commands serialize only on
the candidate-and-policy advisory key and return the already existing row.
Scan completion or failure is persisted separately.  The authoritative checks
do not lock or update source experiments.

The transaction-lock key is derived by PostgreSQL from the full semantic and
policy canonical texts.  A lock-key collision can only serialize unrelated
transactions: full canonical values are rechecked, and the partial unique
index stores those full values.  Hash equality never becomes semantic
equality.  If an existing experiment cannot be mapped to the Step 1
date/configuration contract, duplicate lookup fails closed for that candidate.

Candidate-level persistence errors are counted without rolling back unrelated
recommendations.  A handled scan-level orchestration failure marks the scan
``failed`` with a nonempty error.  A completed scan may have a nonzero
``persistence_errors`` counter, making partial failure observable.  Policy
parse and validation errors occur before database persistence and do not
create a scan; every validated invocation that reaches persistence does.

Commands
--------

Generate recommendations::

  LSTM_Release --generate-experiment-recommendations \
    --recommendation-policy=minimum_leader_score=0.2,minimum_inference_accuracy=0.4 \
    --recommendation-symbol=eurusd --recommendation-horizon=12 \
    --recommendation-max=10

Optional generation filters are ``--recommendation-source-experiment=ID``,
``--recommendation-symbol=SYMBOL``, and ``--recommendation-horizon=N``.

Inspect stored advisory data::

  LSTM_Release --list-experiment-recommendations \
    --recommendation-status-filter=proposed --recommendation-limit=100
  LSTM_Release --recommendation-status=ID
  LSTM_Release --list-experiment-recommendation-scans --recommendation-limit=100
  LSTM_Release --recommendation-scan-status=ID

Generation, list, and detail commands emit stable
``EXPERIMENT_RECOMMENDATION_*`` records.  No scoring field is synthesized and
no approval, queueing, or scheduler-managed scan is provided in Step 3.
Text values percent-escape delimiters, percent signs, non-ASCII bytes, and
control bytes.  ``NULL`` is reserved for an absent optional value; concrete
text with that exact spelling is ``%4E%55%4C%4C``, while an empty string remains
explicitly empty.  A scan-finalization failure is emitted separately from the
original scan failure so the initiating error is not hidden.
