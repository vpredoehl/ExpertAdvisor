# ADR-0017: Campaign Operations privilege, audit, and read-model boundaries

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume I §§6–10; Volume X §§3–10; Volume XII §§3–10
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Campaign Operations spans upstream governance evidence, its own append-only
history, guarded reservations/requests, the accepted Phase 5 transaction
primitive, lifecycle observation, and audit. A broad runtime login or
security-definer business workflow could collapse service boundaries and
permit direct experiment or scheduler mutation.

The architecture also needs an explicit institutional capability-assignment
rule and a single narrow mechanism for Phase 5 invocation.

## 2. Decision

Campaign Operations uses positive-allowlist, NOLOGIN capability roles owned by
`campaign_operations_owner`. Separate capabilities exist for campaign
creation, authorization, budget administration, reservation/request
acceptance, dispatch/adoption, campaign control, cancellation, reconciliation,
completion/audit, read-only status, and optional projection writing.

### 2.1 Assignment and separation

- The database/security administrator is the institutional authority that
  grants a NOLOGIN capability role to an authenticated application service
  principal according to reviewed deployment configuration.
- Application actor identity and reason remain mandatory audit payload; a
  database role does not replace actor authentication.
- Capability separation is mandatory. No natural person or runtime principal
  receives a capability by implication.
- V1 requires no additional human-person inequality beyond the accepted Phase
  6C reviewer/Phase 6D ratifier rule. Additional institutional separation must
  be accepted explicitly rather than inferred.
- Migration creation of a role does not grant it to `pqxx` or enable a
  workflow. Membership/enabling is a separate reviewed deployment action.

### 2.2 Database privileges

- `PUBLIC` and unrelated roles are revoked before positive grants.
- Append-only history permits payload-column `INSERT`, required `SELECT`, and
  sequence `USAGE` only; runtime `UPDATE`, `DELETE`, `TRUNCATE`, generated-ID
  insertion, and sequence `UPDATE` are denied.
- Guarded mutable reservation/request/projection columns are writable only
  through exact expected-state/version transitions paired with immutable
  events.
- Functions use a pinned safe `search_path`; trigger/function execution is
  denied to `PUBLIC` and unrelated roles.
- Reconciliation may read accepted evidence, execute only the campaign and
  request lock helpers required by global ordering, and atomically append an
  exact cursor batch plus observations. It has no reservation, budget, or
  resolution lock/mutation capability.
  Scheduler/workers have no Campaign Operations privileges; Campaign
  Operations has no scheduler-claim, capacity, process, or worker privileges.
- Status readers use approved views/read models and read-only repeatable
  snapshots. Projections expose source high-water marks, are rebuildable, and
  never authorize writes.

### 2.3 Phase 5 invocation

V1 uses an invoker-rights application adapter. While holding the caller-owned
libpqxx transaction and ADR-0013 locks, the dedicated dispatcher service calls
the existing `LaunchRecommendationCampaignInTransaction` workflow through a
dedicated NOLOGIN transactional capability role.

That role receives only the exact table/column/sequence/function privileges
required by the accepted Phase 5 repositories plus Campaign Operations
binding/settlement writes. It does not inherit the existing broad `pqxx`
experiment privileges and cannot issue arbitrary experiment, lifecycle, or
scheduler SQL. A security-definer business procedure that reimplements Phase 5
is not the V1 mechanism.

### 2.4 Audit

Every authoritative mutation records, where applicable, the actor, fixed
capability, reason, contract version, expected/prior/resulting identity and
version, campaign, grant, budget, reservation, request, attempt, cancellation,
observation, exact downstream IDs, outcome, stable diagnostic, database time,
and replay disposition.

Domain events remain authority. The consolidated audit reference stream is an
append-only causal index and never a competing current-state table.

## 3. Rationale and decision drivers

- Preserve service/repository ownership under least privilege.
- Prevent broad database credentials from bypassing authoritative workflows.
- Make deployment enablement separate from schema presence.
- Provide complete causal audit without duplicating state authority.

## 4. Consequences

### 4.1 Positive consequences

- Compromise of one capability does not grant all campaign or scheduler power.
- Phase 5 is reused transactionally without direct Campaign Operations SQL.
- Audit can reconstruct who acted under which exact authority and evidence.
- Read models remain safe to rebuild.

### 4.2 Negative consequences and trade-offs

- Migrations and deployment require detailed ACL and membership management.
- The Phase 5 adapter needs exact column-level privilege analysis.
- More roles increase operational configuration complexity.

### 4.3 Risks and mitigations

- Accidental enablement: NOLOGIN roles are not granted to runtime by migration.
- Privilege drift: catalog-level exact ACL/NULL-ACL tests.
- Security-definer escalation: invoker-rights V1 adapter and pinned function
  paths where narrow transition functions are unavoidable.
- Audit as authority: no mutation path consults audit references to decide
  current state.

## 5. Compatibility and migration

Migration 045 already establishes the owner and initial NOLOGIN roles without
granting them to `pqxx`. Later migrations may add the remaining bounded roles,
views, and grants additively.

Existing Phase 4C/5 runtime behavior and privileges remain unchanged until a
separately reviewed dispatcher deployment enables the narrow adapter. No
production principal is altered merely by accepting this ADR.

## 6. Implementation implications

- Each service has an explicit allowlist of tables, columns, sequences, and
  functions derived from its transaction in the accepted specification.
- Migration tests inspect owners, memberships, grants, default/NULL ACLs,
  search paths, index keys, and absence of broad mutation privileges.
- Operator-facing authentication maps to application actor identity before a
  service call; it does not accept caller-supplied database role text.
- Audit writes commit in the same transaction as their authoritative cause.

## 7. Verification and operational evidence

- Catalog-level owner, role-hardening, membership, ACL, default ACL, sequence,
  function, trigger, and search-path tests.
- Positive tests for every allowed service action and negative tests for every
  prohibited cross-capability action.
- Tests proving schema installation alone leaves workflows disabled.
- Invoker-rights Phase 5 adapter tests proving accepted handoff succeeds while
  arbitrary experiment and scheduler SQL fails.
- Audit causal-completeness, exact replay/conflict, and projection rebuild
  tests.
- Scheduler/worker and unrelated runtime-role isolation tests.

## 8. Alternatives considered

### 8.1 Reuse the broad `pqxx` role

Rejected because it would give Campaign Operations unrelated experiment and
lifecycle mutation power.

### 8.2 Implement Phase 5 as a security-definer SQL procedure

Rejected because it would duplicate the accepted C++ workflow and bypass the
service/repository layering.

### 8.3 One Campaign Operations super-role

Rejected because authorization, budget, dispatch, cancellation,
reconciliation, and completion require distinct powers.

## 9. Relationships to other ADRs

- ADR-0001 and Volume I define PostgreSQL and least-privilege authority.
- ADR-0010 through ADR-0015 define the service-specific powers.
- ADR-0016 defines the scheduler privileges that remain disjoint.

## 10. References

- [Accepted Campaign Operations specification §§24–25, 27–29](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Volume XII](../Volume_XII_Database.md)
- [Migration 045](../../../Database/migrations/045_campaign_operations_foundation.sql)
- [Phase 5 launch](../../Phase5ExperimentRecommendationCampaignLaunch.rst)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted explicit capability assignment, invoker-rights Phase 5 invocation, least privilege, audit, and read-model boundaries. |
