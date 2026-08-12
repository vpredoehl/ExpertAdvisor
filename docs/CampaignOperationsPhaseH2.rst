Campaign Operations Phase H2
============================

The exact privilege and deployment reachability contract is frozen by
``docs/architecture/adr/ADR-0019C-h2-privilege-deployment-contract.md``.
Migration 056 is its additive catalog implementation. This document is the
implementation-facing H2 summary; the ADR is authoritative for roles, ACLs,
LOGIN deployment, recovery, audit, rollback, and prohibited combinations.

Status
------

This document describes the repository's H2 exact production-dispatch
implementation. H2 is a caller-driven, single-request increment. It does not
claim that production rollout, pre-enablement audit, role assignment, an
enable event, or a canary has occurred in a production database.

Commands
--------

The H2 CLI exposes exactly these production mutations:

``--campaign-operations-production-enable``
  Requires ``--campaign-operations-operation-key``,
  ``--campaign-operations-expected-production-version``,
  ``--campaign-operations-independent-verification-reference``, actor,
  reason, and the literal ``--yes`` acknowledgement.

``--campaign-operations-production-disable``
  Requires ``--campaign-operations-operation-key``,
  ``--campaign-operations-expected-production-version``, actor, reason, and
  the literal ``--yes`` acknowledgement. Disable evidence is recorded before
  any later operational rollback decision; this command does not stop
  workers, schedulers, or processes.

``--campaign-operations-dispatch-request``
  Dispatches one explicitly named request. It requires request ID, expected
  request version, ``--campaign-operations-operation-key``, actor, and the
  literal ``--yes`` acknowledgement.

Production mutations reject ``--dry-run``. Operation keys are never generated
by the executable. They must match the frozen ASCII form
``^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`` and are compared as exact immutable
identity inputs. Reusing a key with any changed request, actor, reason,
version, build, or evidence field is a conflicting replay.

Execution and recovery
----------------------

The isolated/test adapter and the production adapter enter one common Phase E
dispatch engine. Production policy, readiness, role, acknowledgement, and
operation-key admission remain outside that engine.

Production acquisition uses the H1 authorized transition and Attempt V2
evidence. Handoff reacquires the Phase H scheduler and production-gate
boundaries before the established Phase E lock order. A completed immutable
binding is returned as exact replay without a second experiment, binding,
handoff, admission, or contradictory audit row.

Direct SQL authority correction
-------------------------------

Migration 059 preserves the atomic H1 production V2 transition but removes
``EXECUTE`` from ``campaign_operations_production_dispatcher``. The deployed
Manager/dispatcher login cannot invoke the mutation-capable wrapper. A distinct
deployment LOGIN is granted only the NOLOGIN
``campaign_operations_production_dispatch_service`` capability, and the
reviewed C++ process uses that service connection only after its application
build preflight succeeds.
That sealed SECURITY DEFINER service boundary validates the database-side
readiness contract, canonical evidence, current enablement, completion proof,
reconciliation state, role graph, and supplied approved build identity before
entering the raw transition. The raw transition is owned by, and executable
only through, the sealed H1 boundary authority; PUBLIC, ``pqxx``, and the
dispatcher role have no direct path to it. The C++ readiness evaluator remains
the application-only actual-running-build preflight, while migration 059 makes
the database-observable gate unavoidable for both direct dispatch and H3
Manager run-once. The ordinary Manager login retains read/candidate-selection
authority but no wrapper or raw-transition authority.

Migration 059 must be applied and recorded in ``schema_migrations`` after 058.
Deployment audits verify the corrected function ACL and deny raw transition
execution, including through inherited dispatcher membership.

State-contract seam
-------------------

The common handoff engine has explicit mode-specific predecessors. The
isolated fixture transitions ``ready/v3/false`` to ``dispatching/v4/false``;
the existing Phase E bound transition requires that exact state/version,
matching unexpired Attempt V1 lease, and produces ``bound/v5/false``. H1
production acquisition starts from the same ready row but atomically produces
``dispatching/v4/true`` with the unexpired lease, one request admission, the
current enablement event, and Attempt V2 ``3/4`` evidence. H2 production
binding requires those exact identities and operation key, then produces
``bound/v5/true`` while clearing the lease. The Boolean is the immutable
one-way production admission/Attempt V2 witness; it is not production
authorization by itself. Migration 055 and the isolated V1 transition remain
unchanged, and exact completed-binding replay does not re-run mutation.

Every meaningful uncertain-commit path abandons the original connection and
uses a fresh independent connection for authoritative readback. Recovery
returns exact replay only after complete evidence hydration and exact caller
identity comparison. Partial, corrupt, stale, cross-principal, or conflicting
evidence fails closed; after the bounded recovery policy, unresolved state is
reported as indeterminate rather than guessed as rollback.

Deployment safety and boundaries
--------------------------------

H1 migration 055 remains authoritative for schema, protected transition
functions, readiness, role graph, ACL catalog, and default-off behavior. H2
does not create a LOGIN, grant role membership, bypass protected functions, or
change live scheduler/worker state. The dedicated production roles remain an
operational deployment prerequisite, and the normal repository state remains
production-inert until an authorized deployment grants the exact H2 reachability
surface and readiness succeeds. Migration 057 adds only the versioned
production predecessor transition; it does not alter migration 055.

H2 has no Manager run-once command, candidate-selection loop, deterministic
Manager-generated operation key, polling, daemon, cadence/backoff loop,
supervision, autostart, or continuous command. Those behaviors remain H3/H4
boundaries.

The production CLI never reuses the generic ``pqxx`` connection identity.
Readiness, status, and Manager candidate selection use
``CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER``. Direct dispatch and H3
Manager run-once perform the application preflight through that Manager LOGIN,
then route the mutation transaction through the distinct
``CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER``. Enable requires
``CAMPAIGN_OPERATIONS_PRODUCTION_ENABLER_DB_USER``; disable requires
``CAMPAIGN_OPERATIONS_PRODUCTION_DISABLER_DB_USER``. Each is required and has
no fallback. The service LOGIN must not equal the Manager LOGIN and must not
reach the dispatcher role. The accepted direct membership tuples remain
ADR-0019C authority, with the additional service tuple documented below.

::

   GRANT campaign_operations_production_dispatch_service,
         campaign_operations_production_phase5_transactional
     TO <DISPATCH_SERVICE_LOGIN>;
