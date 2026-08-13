Campaign Operations Phase H Step 1
==================================

Phase H Step 1 (H1) implements the authority and persistence foundation from
ADR-0019 as narrowly amended by ADR-0019A and ADR-0019B. Migration ``055`` is additive to
migration ``054`` and remains
disabled after installation: it records no enable event, admits no request,
creates no Attempt V2 row, grants no capability to a LOGIN role, and does not
change scheduler, lifecycle, experiment, or worker data.

H1 does not implement production dispatch, exact canary dispatch, enable,
disable, the Campaign Manager, bounded run-once, continuous mode, scheduler
behavior, or lifecycle behavior.

Canonical evidence
------------------

The C++ and PostgreSQL implementations reconstruct the exact ADR-0019
canonicals for scheduler protocol evidence, Manager build identity, production
enable/disable events, first request admission, and Attempt V2. Canonical text
is authoritative; tagged FNV-1a hashes are indexes and integrity checks. Full
canonical comparison is required on hydration and hash equality never makes
different canonical text equivalent.

The production enablement chain, its one-to-one audit, and first request
admission are immutable. Attempt V2 extends the existing attempt table with an
exclusive nullable V2 shape while preserving every Attempt V1 row and
canonical. Deferred constraints enforce the Boolean/admission equivalence and
complete admission/attempt/audit evidence at transaction commit. Owner-level
update, delete, and truncate operations cannot rewrite production evidence.

The existing Attempt V1 path remains test-only. PostgreSQL additionally
requires the literal UTF-8 database-name prefix
``expertadvisor_campaign_operations_phase3_test_``, explicit direct or
inherited membership in the isolated dispatcher capability, a false production
witness, no enablement event, and no explicit membership in any production
role. The isolated C++ adapter continues to require its exact acknowledgement.

Privileges
----------

Migration ``055`` creates a missing exact sealed infrastructure owner and seven
exact NOLOGIN capabilities. It never normalizes a pre-existing incompatible
role. The sealed ``campaign_operations_h1_boundary_authority`` is exactly
``NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
NOBYPASSRLS CONNECTION LIMIT -1`` with no password verifier, validity deadline,
role/database setting, credential, membership in either direction, member, or
``ADMIN OPTION`` edge. PostgreSQL superuser/cluster-administrator
power is explicitly outside the supported threat model; no ordinary or
reachable role owns a protected H1 object.

* ``campaign_operations_production_enabler``
* ``campaign_operations_production_disabler``
* ``campaign_operations_production_dispatcher``
* ``campaign_operations_production_phase5_transactional``
* ``campaign_operations_production_reader``
* ``campaign_operations_scheduler_protocol_evidence_owner``
* ``campaign_operations_scheduler_protocol_evidence_reader``

H1 grants the production mutation roles no production DML or transition
function. The reader receives only the H1 evidence and readiness/status views.
The sealed owner owns the narrow, pinned scheduler snapshot/lock functions and
can perform ``SELECT FOR SHARE`` without granting scheduler DML to an ordinary
role. The legacy-named scheduler evidence owner owns no H1 object; the callable
evidence-reader role has no scheduler DML.
``pqxx`` receives no new membership. LOGIN creation and role membership remain reviewed
deployment operations outside migration ``055``.

Deployment audit and restore
----------------------------

``Scripts/CampaignOperationsH1DeploymentAudit.sh`` is the versioned,
read-only deployment gate. It exits nonzero with one of the stable
``H1A001``--``H1A011`` diagnostics when the role tuple, graph, all-schema
ownership/entry-point allowlist, explicit/default ACLs, fixed transitions,
migration ledger/checksum, or requested historical-byte proof is not exact.
The supported stage commands are:

::

   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-upgrade --host HOST --port PORT --user ADMIN --database DATABASE
   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-upgrade --host HOST --port PORT --user ADMIN --database DATABASE
   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-restore --host HOST --port PORT --user ADMIN --database DATABASE
   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-role-recreation --host HOST --port PORT --user ADMIN --database DATABASE
   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-database-restore --host HOST --port PORT --user ADMIN --database DATABASE
   Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-enablement --host HOST --port PORT --user ADMIN --database DATABASE

The command first validates the independent versioned manifest set and its
embedded migration digest. The frozen object inventory, explicit ACL tuples,
default ACL states, and column ACL declarations have separate exact keys,
counts, reference checks, and a combined SHA-256. Actual catalog discovery is
not driven by expectation rows, so deleting an expectation cannot hide an
object. The command computes the local migration SHA-256 and, for database-bearing
post stages, calls
``campaign_operations_h1_deployment_audit_v1(checksum, true, false)`` inside a
``READ ONLY`` transaction, then runs
``Database/manifests/055_campaign_operations_h1_acl_manifest.sql`` against the
same target. The manifest expands NULL ACLs with ``acldefault()``, uses
``aclexplode()``, preserves ``catalog_acl IS NULL`` for every scanned ACL
column, and reports every row from both expected-minus-actual and
actual-minus-expected differences as stable ``H1A006`` or ``H1A007`` output.
For ``pre-enablement`` after H2 is installed, the historical 055 function is
not reinterpreted as if the H2 readiness adapter had existed at H1. The H1
manifest remains frozen and is validated unchanged; the audit then invokes the
versioned H2 deployment audit (which binds migration 056/057 ledger identity)
and permits exactly
``public.campaign_operations_production_readiness_snapshot_v1()`` in the
sealed-owner function set. Its owner, SECURITY DEFINER/STABLE properties,
search path, and ACL remain H2-exact. Any other sealed-owner function still
fails closed as ``H1A004``.
Historical restore verification calls the same
surface with the final argument ``true`` after restoring the captured pre-055
fixture.

Logical PostgreSQL dumps can normalize an explicit owner-only ACL to NULL.
After a database restore, ``Scripts/CampaignOperationsH1RestoreAclOrigin.sh``
materializes the frozen explicit origin without broadening privileges, before
the read-only post-database audit.

Supported restore workflow A creates an empty target database, recreates the
exact roles separately, runs ``pre-restore`` and ``post-role-recreation``,
restores a database-only dump, then runs ``post-database-restore``. Workflow B
restores the roles-only dump before the database dump and uses the same gates.
A safe exact pre-existing role set is exercised independently as workflow C;
workflow D proves an incompatible pre-existing boundary role rejects before
database creation or restore. Incompatible attributes,
membership or ``ADMIN OPTION``, unexpected ownership/default ACLs, PUBLIC
execution, and alternate-schema wrappers block before restore where observable
and always block the post-restore deployment gate.

Readiness and status
--------------------

::

   LSTM_Release --campaign-operations-production-readiness
   LSTM_Release --campaign-operations-production-status

Both commands use a repeatable-read, read-only transaction. They acquire no
tuple or advisory lock, advance no sequence, and perform no repair or mutation.
Readiness validates migration filename/checksum, all H1 contract versions,
generation-52 scheduler evidence, the current enablement head and independent
verification reference, the approved versus actual Manager build contract,
the actual session principal's explicit required/prohibited role graph,
Completion V1 nested-V2 proof, and unresolved reconciliation evidence. A
missing actual Manager build contract fails closed in H1 because no Manager is
implemented or launched by this increment.

Its single machine-readable readiness row reports the evidence it evaluates:
``scheduler_evidence_canonical`` is the scheduler canonical identity;
``manager_service_contract`` is the persisted approved Manager service
contract; and ``production_attempt_contract_version`` is the Attempt V2
contract version.  It also reports the migration version/checksum, scheduler,
enablement, Manager-build, admission, and completion-proof contract versions,
approved and actual build canonical/hash identities, role/deployment state,
and the deterministic semicolon-delimited blocker set.  Values are rendered
from the loaded readiness snapshot and hydrated enablement/build evidence; a
missing value is explicitly rendered as ``missing``, ``none``, or
``unavailable`` and continues to block readiness where required.

Each ``*_contract_version`` value without an ``expected_`` prefix is the
observed authoritative value. Scheduler version is the typed marker on the
persisted ``experiment_scheduler_protocol`` singleton; Manager-build and
enablement versions are read from persisted enablement evidence; admission and
Attempt V2 versions are aggregates across their relevant persisted evidence;
and the completion nested-V2 proof version is the aggregate of persisted
Completion V1 rows, falling back only when there are no completed campaigns to
the deployed Completion V1 column-contract catalog evidence. An entirely
uninstantiated Admission or production Attempt V2 family is valid at genesis:
its explicit evidence count is zero and its displayed version is
``genesis-empty``, not a version mismatch. Once a family has persisted
evidence, the canonical version must be exact and readable; wrong, mixed,
stale, malformed, or structurally unreadable evidence remains fail-closed.
Genesis-empty changes only this evidence-family interpretation and waives no
other readiness requirement. The corresponding
``expected_*_contract_version`` fields are normative constants used only for
comparison. Canonical/hash/audit corruption remains an integrity error rather
than a fabricated snapshot.

The readiness command exits ``0`` only for complete readiness, ``2`` for a
valid blocked snapshot, and ``1`` for an execution or integrity failure. Status
returns one read-only row per operational request plus a final count and reports
admission, latest V2 attempt, enable event, lease expiry, Phase F recovery
eligibility, and reconciliation state.

This is an intentional split interface: readiness is the aggregate deployment
row, while status owns per-request old-event lease expiry, Phase F recovery
eligibility, and reconciliation detail. Together they satisfy the Phase H
readiness/status reporting contract without changing the readiness row shape.

Migration verification
----------------------

``Tests/CampaignOperationsPhaseH1MigrationTests.sh`` creates and removes an
isolated temporary PostgreSQL cluster and disposable database. It applies
``055`` transactionally twice before evidence
to prove supported replay, then checks upgrade/no-backfill behavior, PostgreSQL
canonical reconstruction and golden vectors, malformed evidence rejection,
exact replay for all three fixed transitions, exact 61-byte acquisition catalog
identity, deferred equations, rollback stability, sealed-owner/context/default-
ACL guards for all three owners and all supported future object classes,
ACL/search-path/role catalogs, direct/two-level/three-level/ADMIN membership
rejection, all-schema wrapper/overload rejection, genuine custom-format dump
and restore scenarios A--J, exact lock boundaries 0a--5 using independent
connections and catalog blocker evidence, literal-prefix V1 isolation,
genuine pre-055 Attempt V1 and Completion V1 byte preservation through
migration and restore, and repository/service reconstruction.
Its base schema comes from the checked repository backup plus migrations
``050``--``054`` inside the new cluster; it does not read or change the live
database, production rows, production roles, or runtime scheduler evidence.

Runtime assurance records
-------------------------

The migration harness emits ``h1-runtime-result-v2`` records with 24 fields and reconciles
every fixture against ``Tests/fixtures/CampaignOperationsH1Traceability.tsv``.
It rejects missing, duplicate, stale, malformed, unexpected, or artifact-digest
mismatched records and generates ``docs/CampaignOperationsH1Traceability.md``
from the reconciled data. Restore A--J use ``h1-restore-runtime-v2``. Lock rows
use ``h1-lock-runtime-v3`` records with 43 fields; the strict validator
deliberately rejects any row classified as a non-final seam.

``Tests/CampaignOperationsPhaseH1Tests.cpp`` provides the independent C++
canonical/golden-vector and malformed-input suite.

Final architecture consistency correction
-----------------------------------------

One immutable first admission now survives committed Phase F recovery and
later acquisitions. The advanced request version produces a new Attempt V2
and acquisition audit; exact replay hydrates the stored service principal and
approved build rather than comparing them with the recovering Manager.

Repository reads require the complete enablement, admission, Attempt V2,
audit, canonical-evidence, authority, and replay graph. Deployment preflight
and audit use the same exact 48-function catalog contract, readiness reports
approved and actual build evidence, and recursive authority diagnostics retain
every shortest-path edge and its ADMIN OPTION value.
