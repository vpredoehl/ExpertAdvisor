# Controlled Replication-Wave Materializer

## Command

```text
LSTM_Debug --materialize-experiment-replications=SOURCE_A_ID:SOURCE_B_ID \
  --replication-seeds=SEED[,SEED...]
```

Attached and separated option/value forms are accepted. Requested seed order is
preserved. The command is mutually exclusive with every other scheduler CLI
command through the existing single-command dispatcher.

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
the plan, rechecks every proposed arm for equivalence, inserts every row, and
retrieves every new ID before committing. Any non-valid preflight, found or
ambiguous equivalent, missing evidence, or insertion error rolls back the whole
wave.

The legacy production unique index does not contain
`fresh_initialization_seed`. While the table lock is held, each inserted row is
therefore assigned the next administrative `duplicate_nonce` so distinct seed
replications do not collide with their source or one another. The nonce is not
part of scientific equivalence; an existing equivalent seed still aborts the
whole wave before insertion.

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

Only the first state is insertable. Any other state aborts the complete wave.
A second identical command and a partially pre-existing wave therefore create
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
