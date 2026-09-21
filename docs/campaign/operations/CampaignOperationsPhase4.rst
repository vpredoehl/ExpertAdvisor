Campaign Operations Phase 4
===========================

Campaign Operations Phase 4 is architectural Phase F: Controls,
Cancellation, and Reconciliation. It adds durable control over future
Campaign Operations actions, cancellation coordination for one exact durable
request, and bounded reconciliation of interrupted Phase E dispatch. It does
not enable production dispatch, signal a scheduler or worker, stop a running
experiment, complete a campaign, or implement archival, reporting, analytics,
budgeting changes, forecasting, optimization, or automation.

The integrated migration is ``053``. Migrations ``049`` through ``052`` retain
their global-control, canonical-operation, scheduler-ownership, and
generation-52 exact-attempt meanings. That scheduler-hardened baseline is a
prerequisite, not Campaign Operations authority. Architectural Phase G is
implemented separately by Campaign Operations Phase 5 migration ``054`` and
does not change this Phase F contract.

Pause and resume
----------------

Pause and resume append one immutable, versioned control event. The first
control event must be ``pause``; subsequent events alternate. The command
uses an expected control version, actor, reason, and ``--yes``. An exact
retry returns the existing event. A stale version, invalid transition, or
changed replay conflicts.

The current pause gate is checked by new request acceptance and every Phase E
dispatch selection, acquisition, and handoff transaction. A pause therefore
blocks only future Campaign Operations work. It neither revokes an existing
lease nor signals, pauses, resumes, or stops scheduler and worker processes.

Cancellation
------------

Cancellation records intent separately from settlement. The command requires
one campaign, exact request ID and expected request version, stable operation
key, actor, reason, and ``--yes``. The operation key is the durable
idempotency key: an exact replay returns the original request and settlement;
a changed target or payload conflicts.
One target can have only one durable cancellation owner. A distinct operation
key targeting a request or campaign that already has cancellation ownership
is rejected under the same cancellation lock domain before another intent can
commit.

For an unbound ``ready`` request, or a ``dispatching`` request whose lease has
expired with no binding, downstream execution, or current-attempt outcome,
one transaction locks budget, campaign, reservation, and request in the
accepted order. It releases the complete held reservation, cancels the
request, and appends transition, settlement, and audit evidence. An active
dispatch lease is never stolen; cancellation intent remains durable and
reports ``waiting_for_lease_expiry`` only while PostgreSQL proves that lease
remains active. Once the lease has expired, downstream or ambiguous evidence
that prevents safe settlement reports ``reconciliation_required``.

For a ``bound`` request, committed units are never released. Campaign
Operations first commits cancellation intent and releases its locks. It then
invokes the narrow Experiment Lifecycle cancellation capability once per
permanent control owner in separate transactions, and finally settles from
the complete immutable lifecycle evidence. ``pending`` and ``paused``
experiments become ``cancelled``; ``completed``, ``failed``, and already
``cancelled`` experiments are recorded as already terminal. Running
experiment cancellation is explicitly unsupported and is recorded as
``running_cancellation_not_supported``; no process signal is sent.

Reconciliation and restart
--------------------------

Reconciliation scans by ascending request ID with an explicit cursor and a
batch limit from 1 through 1000. ``--campaign-operations-reconcile-observe``
only appends deterministic observations and cursor evidence. One transaction
creates the durable cursor identity and every observation linked to that
identity. For a multi-request batch it locks every campaign in ascending ID
order before locking every request in ascending ID order. The exact membership
and cursor commit atomically before any requested recovery. A deferred
PostgreSQL constraint verifies the observation count, run key, request bounds,
and last-target value at commit, so a pre-commit crash or malformed direct
batch leaves neither and replay never reconstructs membership from a run key,
request-ID range, or current request state.
``--campaign-operations-reconcile-recover`` observes first and invokes only
the named safe lease-recovery transition when PostgreSQL time proves the
current lease expired and exact evidence proves there is no binding,
downstream execution, current-attempt outcome, or expired reservation.

Safe recovery clears the lease, returns the request to ``ready``, appends an
immutable recovery outcome, resolution, and audit, and leaves the held
reservation unchanged. Cancellation settlement is repaired only by replaying
the cancellation coordinator. A successful cancellation replay atomically
resolves every still-unresolved observation whose exact request, cancellation,
and point-in-time evidence corresponds to that settlement. An expired held
reservation with a ready,
unbound request is observed and delegated to the reservation service; Phase F
reconciliation does not expire or release it. Partial, progressed,
bound-cardinality, or causally ambiguous evidence is likewise
observation-only and remains fail-closed for its owning service or an
operator. Reusing a run key and prior cursor is idempotent, including empty
batches, and restart loads observations only through the cursor's durable
identity.
Overlapping cursors may hold separate observations of the same request
version and evidence. After one observation performs recovery, every
equivalent observation resolves to that same immutable recovery outcome
without repeating the request transition.

Cancellation retries are whole-transaction retries only for the accepted SQL
states. Intent IDs, replay disposition, candidate evidence, and settlement are
attempt-local and become visible to later coordination only after commit.
Concurrent replay of pending unbound intent rechecks settlement after taking
the cancellation-domain locks, so identical replays converge on one request
and one settlement.
Reconciliation cursor persistence and each safe recovery also retry the whole
transaction at most three times for serialization conflicts and deadlocks.
Every attempt reloads authoritative state on a fresh connection. If commit
acknowledgement is lost, the service looks up the exact durable cursor key or
observation resolution before deciding whether a retry is safe. Timestamp
components in canonical evidence use fixed UTC microsecond text and therefore
do not depend on the session time zone or process locale.

CLI
---

::

   LSTM_Release --campaign-operations-pause CAMPAIGN_ID \
     --campaign-operations-expected-control-version N \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON --yes

   LSTM_Release --campaign-operations-resume CAMPAIGN_ID \
     --campaign-operations-expected-control-version N \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON --yes

   LSTM_Release --campaign-operations-cancel CAMPAIGN_ID \
     --campaign-operations-request-id REQUEST_ID \
     --campaign-operations-expected-request-version N \
     --campaign-operations-operation-key KEY \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON --yes

   LSTM_Release --campaign-operations-control-status CAMPAIGN_ID

   LSTM_Release --campaign-operations-reconcile-observe RUN_KEY \
     [--campaign-operations-reconcile-after-request-id ID] \
     [--campaign-operations-reconcile-limit N] --yes

   LSTM_Release --campaign-operations-reconcile-recover RUN_KEY \
     [--campaign-operations-reconcile-after-request-id ID] \
     [--campaign-operations-reconcile-limit N] --yes

The status command reports the current pause/version head, cancellation
intent and settlement, and unresolved reconciliation-observation count.
It reads through ``campaign_operations_reader`` in one read-only,
repeatable-read snapshot.
Control status accepts neither ``--dry-run`` nor ``--yes``. Mutations require
``--yes`` and do not support dry-run.

Persistence and privilege boundary
----------------------------------

Migration 053 adds append-only control, cancellation, lifecycle-cancellation,
settlement, reconciliation observation/resolution/cursor, and audit evidence.
Deferred constraints require every externally observable mutation and its
causal audit to commit together. Guard functions enforce request,
reservation, and recovery transitions at the database boundary. Resolution
rows additionally carry a typed foreign key to either their exact cancellation
settlement or dispatch-attempt outcome. Capability-specific security-definer
functions validate that upstream transition and derive the persisted owning
service and capability; the workflow roles have no direct resolution-table
insert or sequence privilege.

Controller, cancellation coordinator, reconciler, recovery, lifecycle
cancellation executor, and lifecycle cancellation owner are separate NOLOGIN
capabilities. They receive only the reads, column-scoped writes, sequence use,
and guarded functions required by their owning workflows. None is granted to
``pqxx`` or a login principal by the migration. The existing read-model role
receives only the Phase 4 ``SELECT`` privileges required by control status;
the cancellation coordinator receives only its guarded resolution function and
the causal-audit append capability needed to close exact cancellation
observations.
