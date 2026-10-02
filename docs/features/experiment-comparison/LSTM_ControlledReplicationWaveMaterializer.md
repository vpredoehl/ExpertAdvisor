# Controlled Replication-Wave Materializer

## Command

```text
LSTM_Debug --materialize-experiment-replications=SOURCE_A_ID:SOURCE_B_ID \
  --replication-seeds=SEED[,SEED...] [--allow-existing-equivalent]
```

Attached and separated option/value forms are accepted. Requested seed order is
preserved. The command is mutually exclusive with every other scheduler CLI
command through the existing single-command dispatcher.

## Layout-11 confluence replication

The frozen two-channel Layout-11 study has a separate constrained constructor:

```text
LSTM_Release --plan-layout11-confluence-replication=TEMPLATE_ID \
  --replication-seeds=SEED[,SEED...]

LSTM_Release --materialize-layout11-confluence-replication=TEMPLATE_ID \
  --replication-seeds=SEED[,SEED...]
```

The plan is repeatable-read and reports zero mutations. The materialization
command is separately authorized operational work; it creates only
`paused/train` rows and never queues or starts a worker.

`TEMPLATE_ID` must be a completed fresh, empty-mask Layout-10/114 control that
exactly matches the frozen H4/20-epoch baseline: threshold `0.0008`, LR
multipliers `120`/`25`, checkpoint interval `20`, full-history warmup,
calendar snapshot `1` / `fnv1a64:67610f94f5c8e7cc`, 2010--2025 training and
2025--2026 inference windows, disabled continuation, and the persisted
objective contract. The template is locked, reloaded, and never modified.

For every requested fresh seed the materializer constructs a matched Layout-11
width-116 pair. The control has an empty mask; treatment is exactly:

```text
confluence_tg4_structural_fibonacci_retracement_support_available,confluence_tg4_structural_fibonacci_retracement_contradiction_available
```

No raw TG4, Fibonacci, Pocket, economic, or other feature is accepted. Pair
identity must match exactly outside that mask. The Layout-11 TRAIN selector
must select one shared `train_feature_ablation_v1` capable worker for both
arms. Any incompatible baseline, layout/width, worker, identity, or existing
equivalent fails closed before an insert; the write path retains the existing
single serializable transaction and experiment-table lock.

By default, any exact configured/scientific equivalent or ambiguous equivalent
aborts the complete wave. `--allow-existing-equivalent` is valid only with
this materialization command and permits an exact equivalent resolved to one
experiment ID to be materialized again as a fresh execution replication.
Ambiguous equivalence remains fail-closed. The option does not alter
scientific identity, equivalence lookup, or execution provenance.

## Prospective TRAIN-worker routing evidence

Planning and materialization also load the configured semantic-worker registry
and invoke the same `selectTrainingReferenceWorker()` path used by scheduler
admission. Output records the proposed arm and seed, layout, width, effective
TRAIN capabilities, selected role/rule/priority, source commit, executable
SHA-256, runtime identity, canonical executable/manifest paths, and registry
schema/path. An empty-mask control therefore shows the same
`train_feature_ablation_v1` capability-qualified routing as its TG4-ablation
partner when the registry requires that domain.

Wave evidence uses `train_worker_routing_state`,
`pair_train_execution_identity_homogeneous`, and
`wave_train_execution_identity_homogeneous`, including distinct selected TRAIN
execution identities. A heterogeneous wave is visible but is not silently
treated as one producer. These fields are deterministic routing evidence only:
they are not proof of statistical independence, efficacy, or completed
execution provenance.

Routing is necessarily a statement about the registry loaded at planning or
materialization time. The registry schema version and canonical path are
reported. The completed experiment's actual worker-attempt provenance remains
authoritative and may differ if the registry or runtime changes before
execution.

## Contract

`ProposedExperimentSpecification` is the sole proposed-experiment object. It
extends the planner's exact configured scientific `ArmResultSet` with the
authoritative source experiment ID and typed fresh-initialization seed. Planner
preflight, planner rendering, equivalence lookup, materializer preflight, and
persistence all consume this same object. Persistence never parses rendered
planner output.

For each source arm, the persistence adapter clones the authoritative persisted
scientific/configuration columns and substitutes only
`fresh_initialization_seed`. New rows use fresh administrative and operational
state: a fresh administrative `duplicate_nonce`, `status=paused`, `phase=train`,
normal scheduler priority, no resume model, and no copied model, result, progress, worker,
timestamp, pause/resume, preemption, or continuation-execution lineage state.
The command creates records only; it does not queue or start them.

## Transaction and concurrency boundary

The complete wave uses one PostgreSQL transaction. Before reloading either
source, it obtains:

```sql
LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE;
```

The transaction then reloads authoritative evidence, rebuilds and revalidates
the plan, performs the same TRAIN-worker routing preflight, rechecks every
proposed arm for equivalence, inserts every row, and retrieves every new ID
before committing. Any non-valid preflight,
unavailable/ambiguous/incompatible TRAIN routing, unauthorized or ambiguous
equivalent, missing evidence, or insertion error rolls back the whole wave.

The legacy production unique index does not contain
`fresh_initialization_seed`. While the table lock is held, each inserted row is
therefore assigned the next administrative `duplicate_nonce` so distinct seed
replications do not collide with their source or one another. The nonce is not
part of scientific equivalence. With the explicit override, an exact
equivalent seed is still not reused or modified; it is only an audit reference
for the newly inserted paused/train execution replication.

This PostgreSQL table lock conflicts with the `ROW EXCLUSIVE` lock acquired by
every `INSERT`, `UPDATE`, and `DELETE` on `experiment`. It therefore serializes
the materializer against the existing scheduler, recommendation conversion,
continuation, direct SQL, and future PostgreSQL experiment writers without
requiring those paths to adopt a new advisory-lock convention. PostgreSQL table
locks are mandatory; an ordinary writer cannot bypass them. The boundary does
not serialize transactions that mutate only model/result/evidence tables, but
those transactions cannot create an equivalent experiment row. Concurrent
missing or changing evidence remains fail-closed through the authoritative
loader and ambiguous-equivalence result.

The lock prevents any experiment-row writer from interleaving with the
materializer's reload/check/insert boundary. It does not impose the
materializer's scientific-equivalence policy on an unrelated writer after the
lock is released; those paths retain their own duplicate rules.

The isolation/lock ordering was exercised against a disposable PostgreSQL 17
cluster: a materializer-shaped transaction waiting behind a generic writer saw
the writer's committed experiment after acquiring the lock. Two concurrent
invocations of the actual materialization command produced one two-row wave;
the waiter reloaded the committed rows and aborted with an equivalence conflict.

## Equivalence and retry behavior

Equivalence is rechecked only after serialization protection is held and before
the first insert. Every arm reports exactly one of:

- `no_equivalent_experiment_found`
- `equivalent_experiment_found=<experiment_id>`
- `equivalent_experiment_ambiguous=<ids/reason>`

Only the first state is insertable by default. With
`--allow-existing-equivalent`, `equivalent_experiment_found` is also
insertable, while ambiguity always aborts the complete wave. Without the
override, a second identical command and a partially pre-existing wave create
zero rows. A newly created paused experiment does not yet have materialized
model evidence, so the current authoritative equivalence loader reports it as
ambiguous/missing candidate identity evidence rather than pretending model
identity is proven; this is a deterministic fail-closed conflict.

## Exit and output semantics

Success returns 0. Input, source-evidence, scientific-preflight, and equivalence
conflicts return 3 with zero committed inserts. Database or insertion failures
return 2 after rollback. An indeterminate PostgreSQL commit reports
`materialization_outcome_unknown` and never emits the staged success mapping.
Output is deterministic and explicitly reports
`statistical_independence=not_inferred`, transaction/atomicity state,
preflight, per-arm equivalence, ordered seed-to-ID mapping, and
`queued=false,started=false`. It makes no winner, ranking, recommendation,
expected-result, significance, or independence claim.
