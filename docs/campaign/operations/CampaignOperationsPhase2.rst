Campaign Operations Phase 2
===========================

Scope
-----

Phase 2 is the first Campaign Operations operational authority. It can
administer a campaign's materialized-member dispatch-unit budget and accept
one durable request for the campaign's exact immutable Phase 4D
materialization.

Acceptance commits one ``held`` reservation, one ``ready`` request, one
``acquired`` reservation event, and one audit reference in the same
transaction. The persisted request has
``production_dispatch_enabled=false``. This phase does not dispatch work,
invoke a Phase 5 workflow, create or activate experiments, poll or signal the
scheduler, launch workers, or implement campaign control or later lifecycle
transitions.

Capabilities
------------

The migration creates two hardened NOLOGIN roles:

* ``campaign_operations_budget_administrator``
* ``campaign_operations_request_acceptor``

Neither role is granted to the runtime login ``pqxx``. A database administrator
must separately review and grant the appropriate role to the principal used
for a command. Budget administration and request acceptance remain separate
capabilities. Both are enforced even for an idempotent replay.

The ordinary ``pqxx`` connection is not a Phase 2 principal. A reviewed
pre-Phase-H LOGIN must be selected with
``CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER``; that setting is used only by the
Campaign Operations pre-Phase-H command family, with no ``pqxx`` fallback.
``LSTM_DB_USER`` does not select this principal.
The command preflight rejects superusers and any principal that inherits a
Phase-H production capability.
The deployment LOGIN may receive the two Phase 2 NOLOGIN capabilities when
both budget administration and request acceptance are intentionally operated
by that reviewed service. It must not be a Phase-H production LOGIN.

The same reviewed ``campaign_operations_pre_phase_h_login`` is also the
deployment principal for explicit campaign admission. Grant it exactly the
``campaign_operations_campaign_creator`` capability in addition to the two
Phase 2 capabilities; do not grant any Phase-H production capability. The
runtime requires that principal through
``CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`` and enforces the creator boundary
on both first creation and exact replay.

The deployment grant is:

.. code-block:: sql

   GRANT campaign_operations_campaign_creator
   TO campaign_operations_pre_phase_h_login;

Campaign admission
------------------

Admission is the explicit handoff from Recommendation Governance's immutable
Phase 4D materialization to Campaign Operations' immutable operational
campaign. The input is always the recommendation campaign materialization ID,
not an operational campaign ID:

.. code-block:: console

   LSTM_Release \
     --campaign-operations-admit MATERIALIZATION_ID \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON \
     --yes

The command reloads and validates the complete durable materialization, builds
the canonical V1 operational campaign with
``phase4d_materialization_v1``, ``dispatch_full_materialization``, and
``complete_materialization``, and persists it through the Campaign Operations
repository. Exact replay reports ``existing_identical``; a changed binding is
rejected. Admission creates no budget, reservation, request, dispatch,
experiment, activation, or scheduler activity.

The intended progression is: materialization -> explicit operational campaign
admission -> budget grant -> request acceptance -> production admission and
dispatch. These IDs remain distinct: the materialization ID is the upstream
Recommendation Governance identity, while the operational campaign ID is the
Campaign Operations identity created by admission.

Budget operations
-----------------

Budget is an append-only ledger in exact materialization member-dispatch units.
Every mutation supplies the currently expected ledger version:

.. code-block:: console

   LSTM_Release \
     --campaign-operations-budget-grant CAMPAIGN_ID \
     --campaign-operations-expected-budget-version 0 \
     --campaign-operations-budget-value UNITS \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON \
     --yes

Use ``--campaign-operations-budget-amend`` with a signed non-zero delta,
``--campaign-operations-budget-revoke`` with no budget value, or
``--campaign-operations-budget-supersede`` with a new non-negative total.
Revocation preserves committed and held obligations. An amendment or
supersession cannot reduce the ledger below those obligations.

Request acceptance
------------------

.. code-block:: console

   LSTM_Release \
     --campaign-operations-accept-request CAMPAIGN_ID \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON \
     [--campaign-operations-reservation-expires-at \
       YYYY-MM-DDTHH:MM:SS.ffffffZ] \
     --yes

Acceptance requires the current effective operational grant and an active
budget with enough reservable units for every materialization member. The
global lock order is authorization, budget, campaign, reservation, request.
PostgreSQL also takes the authorization-domain lock for every authorization
event insert, so a direct column-scoped authorizer-capability write cannot race
request acceptance around the repository lock.
PostgreSQL transaction time is authoritative for grant and optional expiry
validation; a non-null reservation expiry must be later than
``transaction_timestamp()``.

The request's serialized authorization, action, scope, prerequisite-policy,
and optional governance-provenance evidence must exactly match the accepting
authorization event. PostgreSQL also binds each budget audit row to its ledger
entry and each request-acceptance audit row to its request, reservation,
acquisition event, authorization, and budget evidence. These audit rows remain
append-only causal indexes rather than state authorities.

The logical operation is one campaign, action kind, and action-contract
version. An identical retry returns the same reservation and request without
consuming budget again. A changed actor, reason, grant, expiry, or payload for
that logical operation conflicts, including changed prerequisite or
provenance evidence.

Status and reconciliation
-------------------------

.. code-block:: console

   LSTM_Release --campaign-operations-budget-status CAMPAIGN_ID
   LSTM_Release --campaign-operations-request-status REQUEST_ID

Budget accounting reports granted, ever reserved, committed, released or
expired, held, and currently reservable units. Repository reads verify that
each request matches its accepting authorization and each reservation has its
request and acquisition evidence; inconsistent evidence fails closed as
reconciliation-required corruption. The approved request-status view compares
the exact request, reservation, authorization, prerequisite/provenance, budget,
acquisition-event, and audit evidence before reporting consistency. Request
status includes its associated reservation and budget evidence. Phase 2 does
not perform a reservation transition or automatic repair.

Operational constraints
-----------------------

Mutation commands require ``--yes`` and reject ``--dry-run``. Status commands
are read-only and accept neither flag. Duplicate Phase 2 command, actor,
reason, expected-version, budget-value, expiry, and ID options are rejected.
Successful machine output percent-encodes unsafe string bytes and reports the
durable campaign, authorization, budget, reservation, acquisition-event,
request, version, replay, and disabled-production-dispatch evidence needed for
operator traceability. The scheduler continues to ignore Campaign Operations
requests. Do not infer dispatch or experiment state from a ``ready`` request.
