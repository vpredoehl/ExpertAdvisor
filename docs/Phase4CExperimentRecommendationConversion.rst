Phase 4C Step 1 Manual Recommendation Conversion Contract
=========================================================

Purpose and boundary
--------------------

Phase 4C Step 1 defines a pure deterministic contract that can turn supplied,
already-reviewed recommendation evidence into an in-memory proposed experiment
specification. It does not persist a conversion, assign an experiment ID,
create or queue an experiment, expose a CLI command, connect to PostgreSQL, or
interact with the scheduler.

The explicit call with complete authorization provenance is the manual act.
An effective latest ``approve`` review and ``approved`` recommendation status
are mandatory prerequisites, but approval alone does not invoke this contract
and never authorizes execution. Ranking membership, bucket, and position are
optional advisory provenance and cannot substitute for manual authorization.

Supported mutations
-------------------

The contract supports only the four families already owned by the Phase 4A
candidate generator:

* ``core_lr_mult``;
* ``head_lr_mult``;
* ``label_threshold``;
* ``prediction_horizon``.

Exactly one mutation is required. Learning-rate multipliers and label threshold
must be finite, positive, strict canonical doubles. Prediction horizon must be
a positive canonical integer. Every non-targeted semantic field and both
invocation fields (checkpoint interval and resume-model provenance) are copied
unchanged from the supplied source invocation.

Source consistency and eligibility
----------------------------------

Supplied evidence is untrusted. The converter rejects a request unless:

* the recommendation exists, has positive consistent source IDs, and is
  ``approved``;
* the latest effective review action is an approval for that recommendation;
* evaluation and score evidence are completed, valid, canonically identified,
  and belong to the recommendation/source;
* evaluation is ``eligible`` and ``advisory_ready`` rather than blocked, stale,
  incomplete, unsupported, or invalid;
* the source invocation has a valid existing semantic and invocation identity;
* the recorded source value exactly equals the canonical value in that source
  configuration;
* the proposed value is canonical, valid, and different;
* applying it changes exactly one semantic field;
* the resulting semantic and invocation identities exactly match the supplied
  persisted recommendation identity; and
* the exact conversion canonical text is not in the supplied existing identity
  set.

Stable rejection codes distinguish missing recommendation or authorization,
non-convertible lifecycle, superseded authorization, missing/incomplete or
invalid evaluation/score, blocked evidence, missing/multiple/unsupported
mutation, malformed/out-of-range/equal values, incomplete source configuration,
source-value mismatch, duplicate identity, inconsistent provenance, and
inconsistent source-experiment identity.

Identity
--------

``experiment_recommendation_conversion_identity_v1`` is the authoritative,
length-framed canonical representation. It contains:

* contract version, recommendation ID, and source experiment ID;
* recommendation semantic canonical identity;
* evaluation identity and evaluation-policy canonical text;
* scoring-policy canonical text;
* typed latest-review authorization and its canonical event provenance;
* changed parameter and canonical source/proposed values; and
* complete canonical source and proposed invocation configurations.

The tagged FNV-1a hash uses the existing recommendation helper and is only a
compact accelerator. Exact canonical text decides equality and duplicate input.
Ranking provenance is deliberately absent from conversion identity: changing a
display bucket or rank cannot change authorization or the proposed experiment.
No timestamp, process ID, locale, path, unordered-container order, or generated
conversion/experiment row ID participates.

Pure-domain contract and deferred work
--------------------------------------

The implementation consists only of value types and pure functions under
``EA::ExperimentRecommendation``. Its unit test links no PostgreSQL library and
requires no environment, file, network, executable, database, or scheduler.

Phase 4C Step 2 durably persists completed proposals as immutable audit records;
see ``Phase4CExperimentRecommendationConversionPersistence.rst``. Step 3 adds
an append-only manual review history and inspection CLI; see
``Phase4CExperimentRecommendationConversionProposalReview.rst``. Proposal
preparation through CLI, experiment creation, budgets, queueing, scheduler
capacity, and any autonomous workflow remain deferred. A later operational
conversion capability still requires the architecture decision and service
boundary required by Volume VIII §11.3. Profitability and rank never imply
authorization.

References
----------

* ``docs/architecture/Volume_I_Foundation.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
* ``docs/Phase4AExperimentRecommendationFoundation.rst``
* ``docs/Phase4AExperimentRecommendationReview.rst``
* ``docs/Phase4BExperimentRecommendationEvaluation.rst``
* ``docs/Phase4BExperimentRecommendationRanking.rst``
