Campaign Operations Phase H3
============================

H3 implements the bounded Campaign Manager run-once command. It is a single
invocation with one optimistic, read-only ``REPEATABLE READ`` candidate
snapshot, followed by sequential processing of at most the supplied bound.
Candidates use the existing Phase E eligibility predicates and are ordered by
``operational_request_id``. The snapshot commits before the first request is
processed; no batch row, claim, lease, advisory lock, ``FOR UPDATE``,
``SKIP LOCKED``, polling loop, or sleep is used.

Command boundary
----------------

::

   LSTM_Release --campaign-operations-manager-run-once LIMIT --yes

``LIMIT`` is required, positive, and bounded by the reviewed maximum of 100.
Zero, negative, malformed, overflowed, and larger values are rejected. The
command has no default, daemon form, continuous mode, autostart, supervision,
worker control, scheduler mutation, or automatic request acceptance/completion.
The literal ``--yes`` acknowledgement follows the existing production
mutation convention; no test hook is available through the CLI or Manager
configuration.

Identity and common engine
--------------------------

Each candidate derives:

::

   campaign_operations_manager_request_operation_v1
   ;request_identity_canonical=<framed complete request canonical>
   ;expected_request_version=<N>

The operation key is ``mgr-v1:<fnv1a64:16-lowercase-hex>:<N>`` using the
project FNV-1a-64 representation. Complete source canonical evidence is
stored in the H3 Attempt V2 evidence relation and compared byte-for-byte on
replay; the hash is lookup material only. The ``mgr-v1:`` namespace is
reserved for new H3 Manager operations after migration 058. Valid pre-058 H2
caller-keyed operations which happened to use that prefix are recorded once at
the migration boundary and retain only their exact H2 replay/recovery authority.
They remain H2 operations: they are never adopted into H3, never receive
Manager source evidence, and cannot be used to create a new H2 acquisition.
New caller-keyed acquisition in the namespace remains prohibited. The H3
Manager source-evidence requirements and H4 exclusion are unchanged.
The namespace remains reserved to this contract: a Manager-shaped Attempt V2 and its one exact
source-evidence row are transactionally inseparable at COMMIT. A missing,
duplicate, malformed, or mismatched row fails closed; an existing attempt is
never adopted or repaired by backfilling evidence. Recovery and complete
binding replay re-read and full-compare that evidence before handoff.
Request/version changes therefore produce a new key, while reuse of an old
key replays the old operation.

H3 calls the same internal Phase E engine used by isolated tests and H2 exact
production dispatch. It does not duplicate acquisition, handoff, retry,
replay, or uncertain-commit recovery logic. Request-local semantic failures
are reported and processing continues. Disablement, ineffective scheduler
protocol evidence, privilege failure, and database-wide failure stop the
batch; committed earlier requests remain authoritative.

The result reports the bound, ordered candidate IDs, processed count, each
request's operation key/outcome/replay classification, and any stable global
stop reason. Multiple Managers may select the same optimistic snapshot rows;
deterministic operation identity, existing request/version CAS, immutable
evidence, and exact replay provide correctness without a Manager claim table.

Disposable runtime verification
--------------------------------

``Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.sh`` builds a fresh
H1/H2 disposable PostgreSQL cluster, installs and checks migration 058, clones
isolated databases per scenario, and runs the actual H3 Manager run-once loop
through a test-only approved-build fixture seam. The harness uses explicit
condition-variable snapshot barriers, independent sessions, controlled
transaction hooks, backend PIDs, and ``pg_blocking_pids()``; it uses no timing
sleep to infer ordering or concurrency.

The executable proof covers request-local continuation, global disable,
scheduler-protocol invalidation, privilege failure, database-wide failure,
overlapping Managers, unrelated-request progress, migration 058 source
canonical bidirectional COMMIT completeness/tamper rejection/atomic rollback,
caller-key namespace rejection, acquisition-boundary recovery, and an
adversarial legacy missing-source Manager recovery that fails before binding
or backfill, bounded sequential order, and H4 negative structural assertions.
The frozen classification map is
request-local semantic failure -> continue; global disable -> stop;
scheduler-protocol/effective-enable mismatch -> stop; privilege failure ->
stop; database-wide or commit-unknown failure -> stop.

Run the disposable proof with:

::

   Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.sh

The required runtime marker is ``H3_RUNTIME_HARNESS_OK A=PASS B=PASS C=PASS
D=PASS E=PASS F=PASS G=PASS H=PASS I=PASS J=PASS``. The harness preserves
historical H1/H2 Attempt V1 canonical bytes and never writes production
experiment rows, scheduler state, workers, or production roles.
