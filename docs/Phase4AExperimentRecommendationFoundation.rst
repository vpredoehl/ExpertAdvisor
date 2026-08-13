Phase 4A Steps 1-2: recommendation foundation
===============================================

Phase 4A remains advisory only. The foundation defines database-independent
recommendation policy and identity types plus pure, deterministic candidate
generation. It does not score or persist candidates, access PostgreSQL, add
schema, expose commands, interact with the scheduler, approve recommendations,
or queue experiments.

Identity model
--------------

The audit found that one identity cannot accurately represent both a research
question and an execution:

* ``EffectiveExperimentConfiguration`` is the **semantic configuration**. It
  contains the experiment-row inputs that Phase 4A can reproduce: canonical
  symbol, prediction horizon, label threshold, nullable core and head learning
  rate multipliers, target epochs, and Gregorian train/inference date
  boundaries.
* ``ExperimentInvocationConfiguration`` contains that semantic configuration
  plus checkpoint interval and nullable ``resume_model_id``. Resuming restores
  learned weights and the SGD update counter. It does not restore an RNG state.
  Consequently, distinct resume models are distinct invocations even when they
  ask the same semantic hyperparameter/data question. Checkpoint interval
  distinguishes operational executions without changing ordinary training
  mathematics through the same target epoch.

Future duplicate detection must name which equivalence it needs. Semantic
deduplication compares semantic canonical text. Exact execution deduplication
compares invocation canonical text. A database uniqueness constraint that
happens to include ``resume_model_id`` does not define semantic equivalence.

The word "effective" is deliberately limited to settings that can currently
be round-tripped through the scheduler experiment row. Completed models contain
additional compatibility metadata, but the experiment schema and scheduler
cannot recreate all of it. Phase 4A must reject rather than infer such settings
if a source differs from supported runtime invariants. A future configuration
contract/hash in the experiment schema is required before identity can safely
span arbitrary binaries or feature/label versions.

Schema-to-identity classification
---------------------------------

The classifications below are exclusive:

``identity_included``
  Participates in semantic configuration identity.
``runtime_state_excluded``
  Result or mutable execution state.
``provenance_excluded``
  Describes where/when a run came from, not the proposed semantic question.
``lineage_excluded``
  Parent/continuation ancestry; may participate in invocation provenance.
``operational_control_excluded``
  Scheduler/monitoring control that does not define the semantic candidate.
``invariant_not_persisted``
  Current binary constant; not reconstructible from an experiment row.
``behavior_affecting_but_not_reconstructible``
  Persisted only on a completed model or otherwise unavailable to a new
  experiment-row launch.
``intentionally_unsupported``
  Phase 4A cannot propose mutation of the setting.
``requires_future_schema_support``
  Identity cannot be made authoritative until additional version/snapshot data
  is persisted.

Experiment table and migrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InsertExperimentRecord`` writes configuration/control columns. Queue and
resume helpers read them, ``LoadPendingExperiments`` maps launch inputs, and
``BuildTrainCommand``/``BuildInferCommand`` construct worker invocations.

.. list-table:: Experiment-column audit
   :header-rows: 1
   :widths: 24 13 22 18 23

   * - Column(s)
     - Type
     - Written / read
     - Effect
     - Classification and justification
   * - ``symbol``
     - text
     - queue insert / train and infer launch
     - input table and instrument
     - ``identity_included``; canonical symbol selects the data source.
   * - ``prediction_horizon``
     - integer
     - queue insert / train launch
     - label lookahead and inference metadata
     - ``identity_included``.
   * - ``c_next_threshold``
     - double precision
     - queue insert / ``--threshold``
     - first-hit label boundary
     - ``identity_included`` as ``label_threshold``.
   * - ``core_lr_mult``, ``head_lr_mult``
     - double precision nullable
     - queue insert / train launch
     - optimizer learning rates
     - ``identity_included``; NULL remains distinct from a concrete value.
   * - ``target_epochs``
     - integer
     - queue/continuation insert / ``--epochs`` or ``--target-epochs``
     - training extent
     - ``identity_included``.
   * - ``checkpoint_interval``
     - integer
     - queue insert / ``--checkpoint-every``
     - checkpoint persistence cadence
     - ``operational_control_excluded`` from semantic identity and included in invocation identity. It controls checkpoint evidence, inference opportunities, and checkpoint-stop timing, not ordinary training math.
   * - ``train_start``, ``train_end``
     - timestamptz
     - queue insert / reconstructed as text, truncated to ten characters, and passed as training positional dates
     - Gregorian training date boundaries
     - ``identity_included`` as ``train_start_date``/``train_end_date``. The physical timestamp instant is not semantic.
   * - ``infer_start``, ``infer_end``
     - timestamptz nullable
     - queue insert / reconstructed as text, truncated to ten characters, and passed as inference positional dates
     - Gregorian evaluation date boundaries
     - ``identity_included`` as nullable ``infer_start_date``/``infer_end_date``. The physical timestamp instant is not semantic.
   * - ``experiment_id``
     - bigserial
     - database / all scheduler lookups
     - row identity only
     - ``provenance_excluded``.
   * - ``resume_model_id``
     - bigint nullable
     - queue/continuation insert / resume launch
     - initial weights and optimizer progress
     - ``lineage_excluded`` from semantic identity; included in the separate invocation identity.
   * - ``last_model_id``
     - bigint nullable
     - training completion / infer and analysis launch
     - output/result linkage
     - ``runtime_state_excluded``.
   * - ``status``, ``phase``
     - text
     - scheduler transitions / queue selection
     - lifecycle only
     - ``runtime_state_excluded``.
   * - ``current_epoch``, ``worker_pid``, ``worker_started_at``, ``current_operation``
     - integer/timestamptz/text
     - workers and scheduler / lifecycle recovery
     - attempt state
     - ``runtime_state_excluded``.
   * - ``train_log_path``, ``infer_log_path``, ``analysis_log_path``, ``exit_code``, ``error_message``
     - text/integer
     - scheduler workers / diagnostics
     - observability only
     - ``runtime_state_excluded``.
   * - ``created_at``, ``started_at``, ``completed_at``, ``updated_at``
     - timestamptz
     - database/scheduler / ordering and display
     - lifecycle time
     - ``runtime_state_excluded``.
   * - ``duplicate_nonce``
     - bigint
     - explicit duplicate queue path / unique index only
     - uniqueness escape hatch
     - ``operational_control_excluded``.
   * - ``git_commit``, ``git_branch``, ``git_dirty``
     - text/boolean
     - run-metadata capture / status and reports
     - source provenance
     - ``provenance_excluded``; commit may explain behavior but is not a future proposed configuration.
   * - ``build_config``, ``compiler_version``, ``schema_version``, ``scheduler_version``, ``binary_name``
     - text
     - run-metadata capture / status and reports
     - build/runtime provenance
     - ``provenance_excluded``.
   * - ``invocation_mode``, ``run_metadata_captured_at``
     - text/timestamptz
     - queue metadata / status
     - invocation provenance
     - ``provenance_excluded``.
   * - ``stop_after_checkpoint_epoch``, ``stopped_at_checkpoint_epoch``, ``stopped_at_checkpoint_model_id``
     - integer/bigint
     - checkpoint policy/operator / training stop and recovery
     - stop request/result
     - ``operational_control_excluded``; actual stopping is reflected by results, not the proposed semantic target.
   * - ``opportunistic_checkpoint_infer``, ``checkpoint_infer_enabled``, ``checkpoint_infer_min_epoch``, ``checkpoint_infer_interval``
     - boolean/integer
     - queue policy / checkpoint scheduler
     - monitoring/evidence cadence
     - ``operational_control_excluded``.
   * - ``checkpoint_policy_enabled``, ``checkpoint_policy_min_leader_score``, ``checkpoint_policy_min_infer_accuracy``, ``checkpoint_policy_top_n``, ``checkpoint_policy_scope``, ``checkpoint_policy_stop_mode``, ``checkpoint_policy_grace_evals``
     - boolean/double/integer/text
     - policy configuration / checkpoint evaluator
     - orchestration and early-stop policy
     - ``operational_control_excluded``.
   * - ``checkpoint_policy_last_decision``, ``checkpoint_policy_last_decision_at``, ``checkpoint_policy_last_checkpoint_eval_id``, ``checkpoint_policy_last_reason``
     - text/timestamptz/bigint
     - checkpoint evaluator / status
     - policy result state
     - ``runtime_state_excluded``.
   * - ``continuation_policy_enabled``, ``continuation_policy_target_epochs``, ``continuation_policy_min_evals``, ``continuation_policy_patience``, ``continuation_policy_min_leader_score``, ``continuation_policy_min_infer_accuracy``, ``continuation_policy_min_improvement``, ``continuation_policy_max_degradation``, ``continuation_policy_top_n``, ``continuation_policy_scope``, ``continuation_policy_trend_mode``, ``continuation_policy_source_mode``, ``continuation_policy_include_excluded``, ``continuation_candidate_excluded``
     - boolean/integer/double/text
     - continuation CLI/inheritance / Phase 3A/3B
     - continuation orchestration
     - ``operational_control_excluded``; Phase 3 identity remains separate.
   * - ``continuation_policy_inherit_to_child``, ``continuation_policy_target_increment``, ``continuation_policy_max_target_epochs``, ``continuation_policy_progression_mode``, ``continuation_policy_target_sequence``
     - boolean/integer/integer[]/text
     - policy/inheritance persistence / Phase 3 planner
     - continuation target planning
     - ``operational_control_excluded``.
   * - ``continuation_policy_revision``, ``continuation_policy_last_decision``, ``continuation_policy_last_decision_at``, ``continuation_policy_last_reason``, ``continuation_policy_selected_model_id``, ``continuation_policy_queued_experiment_id``
     - bigint/text/timestamptz
     - Phase 3 persistence / status and preflight
     - policy runtime state
     - ``runtime_state_excluded``.
   * - ``parent_experiment_id``, ``continuation_source_experiment_id``, ``continuation_source_model_id``, ``continuation_source_epoch``, ``continuation_generation``, ``continuation_decision_id``
     - bigint/integer
     - continuation child insert / lineage checks
     - ancestry
     - ``lineage_excluded``.
   * - ``continuation_policy_inherited``, ``continuation_policy_inherited_from_experiment_id``, ``continuation_policy_inherited_from_revision``, ``continuation_policy_inherited_from_hash``, ``continuation_policy_inheritance_status``
     - boolean/bigint/text
     - child policy initialization / Phase 3 status
     - immutable policy provenance/terminal state
     - ``provenance_excluded``.

The checkpoint-evaluation, analysis-result, inference-result, continuation-
decision, and checkpoint-decision tables contain evidence/results rather than
proposed experiment configuration. Their IDs, metrics, watermarks, decisions,
timestamps, ranks, and worker state are therefore ``runtime_state_excluded``;
their parent/source IDs are ``lineage_excluded``. They are recommendation
source evidence, never candidate identity.

Model metadata and runtime configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Behavior/configuration audit beyond the experiment row
   :header-rows: 1
   :widths: 25 16 22 16 25

   * - Field/source
     - Persistence
     - Written / read
     - Effect
     - Classification and justification
   * - model parameter matrices and head weights/biases
     - ``matrix`` rows
     - model save / resume and inference load
     - learned initial state
     - ``behavior_affecting_but_not_reconstructible``; represented indirectly by ``resume_model_id`` only in invocation identity.
   * - model ``parent_model_id``, ``experiment_id``
     - ``model`` columns
     - model save / ownership and lineage checks
     - ancestry/ownership
     - ``lineage_excluded``.
   * - model name, comment, created time
     - ``model`` columns
     - model save / display
     - provenance
     - ``provenance_excluded``.
   * - target type / label mode
     - ``target_meta`` matrix
     - model save / resume and inference compatibility
     - loss/head/label behavior
     - ``behavior_affecting_but_not_reconstructible`` from an experiment row.
   * - label threshold and prediction horizon
     - ``train_config_meta`` and experiment row
     - model save/resume / queue and launch
     - label behavior
     - ``identity_included`` through the authoritative experiment fields.
   * - first-hit/lookahead label rule ID
     - ``train_config_meta``
     - model save / resume compatibility
     - label assignment
     - ``behavior_affecting_but_not_reconstructible``; current value is fixed at rule 1.
   * - class weights including neutral class weight
     - ``train_config_meta``
     - model save / resume compatibility
     - loss weighting
     - ``intentionally_unsupported``; no experiment-row/queue fields exist.
   * - window size, hidden size, number of layers
     - ``train_config_meta``/``model_meta``
     - model save / resume compatibility
     - data windows and architecture
     - ``behavior_affecting_but_not_reconstructible`` from an experiment row.
   * - core/head-weight LR multipliers
     - experiment row and ``train_config_meta``
     - queue/model save / launch/resume
     - optimizer rates
     - ``identity_included`` through experiment fields.
   * - head-bias LR multiplier
     - ``train_config_meta`` only
     - model save / resume compatibility
     - optimizer rate
     - ``intentionally_unsupported``; no experiment-row/queue persistence.
   * - normalization version
     - ``train_config_meta``
     - model save / resume/inference compatibility
     - feature normalization
     - ``behavior_affecting_but_not_reconstructible`` from an experiment row.
   * - optimizer type and update count
     - ``optimizer_meta``
     - model save / resume load
     - SGD compatibility/progress
     - ``behavior_affecting_but_not_reconstructible``; update count is carried by resume invocation, not semantic configuration.
   * - completed epoch
     - ``train_config_meta``
     - model save / resume start epoch
     - training progress
     - ``runtime_state_excluded``.
   * - training symbol/range metadata
     - ``train_symbol_meta``/``train_range_meta``
     - model save / resume validation
     - input selection
     - ``identity_included`` through canonical experiment symbol/range.
   * - feature count/model input width
     - ``model_meta`` shape
     - model save / resume/inference compatibility
     - architecture compatibility only
     - ``behavior_affecting_but_not_reconstructible``; count does not identify feature meanings.
   * - target scale/bias/z-score statistics
     - ``target_meta``
     - model save / resume/inference load
     - target transformation
     - ``behavior_affecting_but_not_reconstructible`` from an experiment row.
   * - base learning rate, batch size, mini-batch cap, gradient/clipping constants
     - compiled constants
     - LSTM implementation only
     - optimizer math
     - ``invariant_not_persisted``.
   * - deterministic initializer seed (42)
     - compiled constant/thread-local RNG
     - LSTM construction only
     - initial weights
     - ``invariant_not_persisted``; no RNG state is serialized/restored.
   * - feature set, return-feature horizons, lookbacks/scales, higher-timeframe behavior
     - compiled code/macros
     - tensor/LSTM feature construction
     - feature semantics
     - ``requires_future_schema_support`` via a durable feature/configuration version or hash.
   * - gate/state execution mode and reset-state mode
     - build constants/macros
     - LSTM execution
     - execution implementation
     - ``invariant_not_persisted``; git/build provenance is insufficient as semantic configuration.
   * - model/serialization schema versions
     - model metadata plus code
     - save/load compatibility
     - representation compatibility
     - ``provenance_excluded`` from the research question; loaders must still validate compatibility.
   * - underlying input-row snapshot/revision
     - not persisted
     - external price tables / tensor construction
     - exact data contents
     - ``requires_future_schema_support``; symbol and time range alone do not freeze mutable source data.
   * - git commit/build/compiler/invocation mode
     - experiment run metadata
     - queue capture / reporting
     - implementation provenance
     - ``provenance_excluded``; future cross-build equivalence needs a separate implementation contract, not source labels in candidate identity.

Supported recommendation mutations remain ``core_lr_mult``, ``head_lr_mult``,
and ``label_threshold``. ``prediction_horizon`` is supported only with explicit
enablement and a nonempty permitted-horizon list. Head-bias LR, class weights,
architecture, label mode/rule, normalization, feature definitions, optimizer,
and other compiled settings are not legal mutation names.

Canonicalization and hashes
---------------------------

Policy canonical text begins with ``experiment_recommendation_policy_v2``.
Semantic configuration begins with
``experiment_recommendation_semantic_configuration_v3``. Version 2 introduced
shortest-round-trip floating formatting and separated resume lineage; version 3
models ranges as consumed Gregorian dates and removes checkpoint cadence from
semantic identity. Invocation identity uses
``experiment_recommendation_invocation_v2`` because it now includes checkpoint
interval. No Phase 4 identity was persisted under the provisional versions.

Floating values use the locale-independent C++ ``to_chars`` general overload,
which emits the shortest representation that round-trips to the same finite
``double``. Thus common values remain ``0.8``, ``0.1``, and ``0.0001`` while
adjacent representable values remain distinct. Negative zero becomes ``0``;
NaN and infinities are rejected. Semantic collections are sorted and
deduplicated, NULL differs from zero, and symbols use project canonicalization.

Hashes are tagged ``fnv1a64:<16 lowercase hexadecimal digits>`` and are FNV-1a
64-bit over the exact canonical bytes. FNV is retained because it is stable,
small, dependency-free, and adequate as a lookup accelerator. It is **not** an
authoritative equality proof. Future persistence must:

* store both canonical text and tagged hash;
* use the hash only to find possible matches;
* compare canonical text after every hash match;
* avoid a hash-only uniqueness constraint, or explicitly include canonical text
  in collision handling;
* log/surface a collision when equal hashes have different canonical text; and
* never merge, reject, approve, or queue one configuration as another solely
  because their 64-bit hashes match.

Policy parsing whitespace
-------------------------

The parser trims ASCII space, tab, CR, LF, form-feed, and vertical-tab around
assignments, keys, scalar values, and colon-delimited list items. Duplicate
keys are detected after trimming. It rejects empty keys/required values, empty
assignments, repeated commas, empty list items, malformed internal tokens,
unknown keys, and unsupported parameters. It deliberately does not recognize
Unicode whitespace.

Date-boundary grammar
---------------------

``train_start``, ``train_end``, ``infer_start``, and ``infer_end`` are all
physically PostgreSQL ``timestamptz`` columns. The scheduler reconstructs them
as text and passes only the first ten characters to workers. Training and
inference therefore consume Gregorian date boundaries, and semantic identity
accepts exactly:

``YYYY-MM-DD``

The complete Gregorian date is validated, including leap-year rules, and is
emitted unchanged in canonical form. Timestamps, timezone offsets, surrounding
whitespace, trailing characters, invalid months/days, and invalid leap days are
rejected. The pure component never truncates timestamps.

Future database mapping must explicitly extract the same calendar-date
representation consumed by scheduler workers. It must not depend on a
PostgreSQL session timezone followed by ``timestamptz::text`` and substring
truncation, because the resulting date can otherwise change with session
timezone. Resolving that mapping belongs to the future persistence layer, not
this pure component.

Checkpoint interval semantics
-----------------------------

Checkpoint interval belongs to invocation identity. It controls checkpoint
persistence cadence, available checkpoint evidence, checkpoint inference
opportunities, and the effective timing of an independently requested
checkpoint stop. Saving a checkpoint reads a ``const`` LSTM and does not alter
labels, features, loss calculations, optimizer updates, the model trajectory
through the same epochs, or the final model reached normally at the same target
epoch. Changing checkpoint interval therefore leaves semantic identity stable
while changing invocation identity.

Pure candidate generation
-------------------------

``EvaluateRecommendationSource`` returns a structured eligibility result. A
source is eligible only when the policy is enabled and valid, its experiment
ID and semantic/invocation configurations are valid, its leader score and
inference accuracy are present and finite, its evidence count and both metrics
meet policy minima, and any required predicted-neutral proportion is present,
finite, within ``[0,1]``, and at or below the configured maximum. Rejections
use stable machine-readable reasons and do not throw away the underlying policy
or identity validation detail.

``GenerateRecommendationCandidates`` supports only these mutations:

* additive offsets to ``core_lr_mult``;
* additive offsets to ``head_lr_mult``;
* additive offsets to ``label_threshold``; and
* explicit permitted ``prediction_horizon`` values, only when horizon changes
  are enabled and the parameter is allowlisted.

Learning-rate multipliers and thresholds follow the queue/worker launch
contract: results must be finite, positive, and different from the source.
Horizons must be positive and different from the source. No campaign-specific
upper or lower caps are invented. A nullable source multiplier is never
guessed; each requested mutation is rejected as ``missing_source_value``.
Symbol, target epochs, all train/inference dates, checkpoint interval, and
resume model are copied unchanged.

Every emitted candidate differs from the normalized source semantic
configuration in exactly one field. Both semantic and invocation identities
are built through the Step 1 canonical identity APIs. Structural distance
contains absolute delta, relative delta when the source is nonzero, and signed
horizon delta for horizon changes; it is metadata only and is not a score.

Offsets and horizons are sorted and deduplicated before construction. Valid
candidates are ordered by parameter (core LR, head LR, threshold, horizon),
numeric proposed value, semantic canonical text, then invocation canonical
text. All valid candidates are constructed and sorted before the per-source
limit retains the first N; overflow candidates are returned as
``per_source_limit`` rejections. There is intentionally no cross-source limit
in this pure step.

In-memory duplicate handling follows the Step 1 collision contract. Semantic
canonical text is authoritative. Hash buckets accelerate lookup, but every
hash match is checked by canonical text. Equal canonical text is rejected as a
duplicate; equal hashes with different canonical text retain both candidates
and produce an explicit collision record.

Deferred work
-------------

Scoring, PostgreSQL evidence loading, database duplicate lookup,
collision-aware persistence, migrations, CLI inspection, scheduler scans,
approval, and experiment conversion remain deferred. Before approval/queueing,
future work must persist or validate a complete implementation/feature contract
for behavior-affecting settings that the experiment row cannot currently
reconstruct.
