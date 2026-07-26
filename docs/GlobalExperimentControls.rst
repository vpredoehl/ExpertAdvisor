Global experiment execution controls
====================================

The global pause, resume, and cancellation commands are one-shot
administrative operations.  They connect directly to PostgreSQL and therefore
do not require a scheduler process to be running.

Commands
--------

::

  LSTM_Release --pause-all-experiments --yes
  LSTM_Release --resume-all-experiments --yes
  LSTM_Release --cancel-all-experiments --immediate --yes
  LSTM_Release --cancel-all-experiments --after-next-checkpoint --yes

``--infer-before-cancel`` is valid with either cancellation mode.
``--dry-run`` replaces ``--yes`` and performs no database mutation, signaling,
or worker launch.  A dry run reports each authoritative experiment/worker PID,
process group, identity-validation result, intended signal, checkpoint target,
and inference action.

Database authority and locking
------------------------------

Migration ``046_global_experiment_control.sql`` creates the singleton
``experiment_global_control`` row, immutable request identity and mutable
completion accounting in ``experiment_admin_request``, and per-worker outcomes
in ``experiment_admin_worker_outcome``.  The global desired state is
``running`` or ``paused``.  An active request is a separate gate, so lifecycle
status is never used to represent Unix suspension.

The applying invocation owns a short renewable database lease.  A concurrent
same-shaped invocation is rejected while that lease is live; after an
interrupted invocation's lease expires, the same command can claim and resume
only its still-planned outcomes.

The scheduler and every administrative command take the same transaction-level
PostgreSQL advisory lock.  Worker claims, launches, PID/process-group identity
persistence, control-state transitions, cancellation target assignment, and
continuation queueing are serialized by that lock.  Signals and bounded waits
occur after the request transaction commits; a second locked transaction
records outcomes and finalizes the request.  Thus the database blocks new work
without keeping a transaction open during potentially blocking OS operations.

A scheduler started while desired state is ``paused`` remains alive, performs
observation/reconciliation, and launches no train, final-infer, analyze,
checkpoint-infer, or continuation work.  During cancel-with-inference, only
checkpoint inference rows linked to the active cancellation request are
eligible to launch.

Process identity and signal safety
----------------------------------

The worker PID persisted in ``experiment`` or ``experiment_checkpoint_eval`` is
the primary lookup.  The launcher also persists the process group, executable,
command line, and a macOS kernel process-start identity derived from
``proc_pidinfo(PROC_PIDTBSDINFO)``.  The database row supplies the worker phase,
experiment identifier, and checkpoint-evaluation identifier where applicable.
Scheduler launch and conservative process adoption both persist the start
identity before making the worker administratively manageable.  Before every
``SIGSTOP``, ``SIGCONT``, ``SIGTERM``, or ``SIGKILL``, the command:

* confirms that the database row is still active;
* checks PID existence without signaling it;
* reads the current PID, process group, executable, command, and process-start
  identity;
* matches the experiment/phase or checkpoint-evaluation identity;
* requires exact agreement with the persisted process group, executable, and
  process-start identity;
* rejects scheduler commands, incomplete persisted identity, PID reuse, a
  changed process group, process group 0/1, and the caller's PID or process
  group.

Workers use ``setsid()``, so a validated worker group can be signaled without
including the scheduler.  Pause sends ``SIGSTOP`` and resume sends ``SIGCONT``.
Immediate cancellation sends ``SIGCONT`` first for a database-recorded stopped
worker, revalidates all identity components, then sends ``SIGTERM``.  After the
bounded grace period it revalidates all components again and waits for the
complete process group—not only its leader—to exit before ``SIGKILL``
escalation.  If the leader is gone or any identity field differs, escalation is
rejected because the group can no longer be proven to be the persisted worker.
Missing processes, stale PIDs,
identity failures, permissions, signaling failures, and escalation are distinct
audit outcomes.  A PID is cleared only after conclusive absence or cancellation
reconciliation.  Existing active rows without a process-start identity are not
signaled; a scheduler must first conservatively rediscover and adopt the exact
worker.

Pause and resume semantics
--------------------------

Pause persists ``desired_state=paused`` before signaling.  Zero-worker pause
succeeds, and a repeated pause is an idempotent, auditable success.  A stopped
worker remains lifecycle ``running`` in its existing phase while its separate
``worker_control_state`` is ``paused``.  The scheduler does not reclaim it or
classify it as failed.

Resume persists ``desired_state=running`` while the active request continues to
block scheduler launches.  It revalidates every stopped worker and sends
``SIGCONT`` only to validated groups.  The active gate is removed after outcome
accounting, after which a running scheduler can claim new work.  With no
scheduler, existing workers continue but queued work waits for a future
scheduler.

Cancellation semantics
----------------------

Immediate cancellation atomically gates scheduling, marks queued/lifecycle
paused experiments cancelled, and records every active worker before signaling.
Durable checkpoints and completed inference rows are not removed.  One worker
failure is recorded as a partial result and does not prevent other targets from
being cancelled.

With ``--infer-before-cancel``, inference uses the latest unambiguous durable
database checkpoint; it never uses unsaved in-memory weights.  An already
completed result is reused.  A pending/running checkpoint evaluation is linked
to the request rather than duplicated.  No checkpoint produces
``no_checkpoint`` and cancellation still proceeds.  Inference failure is a
partial administrative result and never prevents cancellation of the remaining
targets.

After-next-checkpoint cancellation reuses ``stop_after_checkpoint_epoch``.
Queued experiments are cancelled without launch.  Running training selects the
current epoch only when that epoch is already a durable periodic checkpoint;
otherwise it selects the first periodic checkpoint strictly after current
progress.  If no future checkpoint exists before ``target_epochs``, the worker
is cancelled immediately, using its latest durable checkpoint for optional
inference.  Non-training phases use immediate phase-safe termination rather than
inventing checkpoints.  A database-recorded stopped training worker is resumed
so it can reach its target.

The worker records each durable checkpoint stop-decision epoch under the shared
coordination lock.  This distinguishes a checkpoint the worker is about to
evaluate from one whose stop decision has already passed, avoiding an
unnecessary extra interval.  If a checkpoint-bound worker disappears, the
scheduler may relaunch only that cancellation-authorized training row from its
latest durable checkpoint.  With no restart checkpoint, cancellation completes
partially rather than retraining from unsaved state.

At the target, the training worker saves through the normal checkpoint path,
records the exact model, queues request-authorized inference when requested,
and transitions to lifecycle ``cancelled`` without final inference/analysis or
continuation.  The request, target, and outcome survive scheduler or worker
restart.  Scheduler reconciliation completes or partially completes the audit
only from durable checkpoint-inference evidence.

Status and recovery
-------------------

``--scheduler-status`` shows desired global state, active request ID, stopped
managed-worker count, active cancellation mode, infer-before-cancel state,
pending checkpoint-cancellation count, and the latest administrative request.
Machine output exposes the same fields.

Crashes leave the active request and per-worker plans visible.  Scheduler
restart reconstructs its launch gate and cancellation-inference authority from
the database.  Repeating the same command shape resumes any still-planned
signals from that persisted request; already-accounted outcomes are not
reapplied.  Duplicate completed pause/resume commands remain idempotent and
auditable.  Before rejecting a conflicting new administrative command,
reconciliation atomically retires an active cancellation whose worker outcomes
are already terminal; genuinely active requests are still rejected.  Partial
failures remain queryable for diagnosis.

Upgrade reconciliation also repairs pre-correction ``awaiting_inference`` rows
whose cancellation checkpoint epoch, model ID, or both were not persisted.
Missing identity is filled only when one cancellation-owned evaluation proves
the request, experiment, checkpoint epoch, checkpoint model, model ownership,
and persisted model-epoch metadata, or when the experiment's durable
after-next-checkpoint stop evidence proves those same relationships.  A
non-null identity value is never replaced.  Partial persisted identity must
agree with the recovered half; multiple candidates, conflicting values,
foreign ownership, cross-experiment linkage, or malformed model metadata
produce a terminal diagnostic ``partial`` outcome instead of a heuristic
latest-model choice.

The same classifier repairs an after-next-checkpoint outcome left
``pending_checkpoint`` after its experiment becomes terminal.  A successful
repair requires an exact request link, a valid terminal
``completed``/``done`` or ``cancelled``/``train`` lifecycle, agreement among
the requested, cancel, stop, current, and stopped checkpoint epochs, and a
stopped checkpoint model owned by the experiment with exactly matching epoch
metadata.  Missing, conflicting, or foreign stopped-checkpoint evidence is
classified terminally; it is not filtered out to await a worker that can no
longer run.

Cancellation inference attaches an unowned exact evaluation to the current
request and treats the same owner as an idempotent replay.  A non-null owner
belonging to another request is immutable: the owner is preserved and the
current worker outcome becomes terminal ``partial``.  Duplicate exact rows are
ambiguous and are never selected arbitrarily.  Completed reconciliation
requires ``status='completed'``, ``phase='done'``, a completion timestamp, no
worker PID, exact request/experiment/epoch/model linkage, and valid model
ownership and epoch metadata.  Failed reconciliation requires an allowed
terminal inference failure phase, a completion timestamp, no worker PID, and
the same exact linkage.  Pending and running evaluations remain nonterminal
only when their phase, worker, and timestamp lifecycle is internally valid.
Malformed terminal or queued materializations become deterministic diagnostic
``partial`` outcomes.

Every reconciliation pass rebuilds request counts from worker outcomes.
``pending_count`` includes only ``planned``, ``pending_checkpoint``, and
``awaiting_inference`` work; both ``partial`` and ``failed`` contribute to
``failed_count``.  Mixed terminal results produce request ``partial`` and fully
successful results produce ``completed``.  ``completed_at`` is set only after
``pending_count`` reaches zero and is preserved on replay.  Clearing
``active_request_id`` occurs in that same transaction, once, after terminal
accounting.  Scheduler restarts and repeated command/reconciliation replay
therefore preserve evaluation counts, ownership, terminal timestamps, and
diagnostics while allowing the next administrative request only after the
previous request has converged.
