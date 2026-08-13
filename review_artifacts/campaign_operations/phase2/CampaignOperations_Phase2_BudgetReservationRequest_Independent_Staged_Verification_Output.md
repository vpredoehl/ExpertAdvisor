---
title: "Campaign Operations Phase 2 Budget Reservation Request Independent Staged Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase2_BudgetReservationRequest_Independent_Staged_Verification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 2 Budget Reservation Request Independent Staged Verification

## Executive summary

The Phase 2 boundary is respected, and the transaction/locking design is fundamentally sound. The implementation does not dispatch, acquire leases, invoke Phase 5, mutate experiments, schedule workers, or implement later reservation lifecycle transitions.

However, two database-integrity/security defects allow the request-acceptor capability to persist internally inconsistent immutable evidence or incorrectly attributed audit evidence. These violate the accepted authority and audit contracts and should block commit.

No Critical findings. Two High, two Medium, and two Low findings.

## Findings by severity

### Critical

None.

### High

1. Request prerequisite evidence is not bound to the accepting authorization.

The request table stores `prerequisite_policy` and optional provenance, but the request trigger only validates the reservation relationship and scope digest. It never verifies those request fields against the referenced authorization event: [migration 047:619](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:619>).

The request-acceptor role can insert all affected columns directly: [migration 047:1251](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:1251>). It could therefore reference a valid exact-Phase-6D grant while persisting a request claiming `phase4d_materialization_only_v1`, or substitute different provenance.

The repository also accepts this inconsistency when reloading and replaying:

- Request mapping reconstructs the supplied prerequisite fields without comparing them to the authorization: [CampaignOperationsRepository.cpp:1177](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:1177>).
- Replay compares the current authorization identity but not the request’s prerequisite policy/provenance against that authorization: [CampaignOperationsRepository.cpp:1753](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:1753>).

This violates immutable request provenance and could later provide misleading handoff authority.

2. Audit evidence is not fully bound to its authoritative mutation.

The deferred acceptance check validates reference IDs but not that audit capability, actor, reason, and version agree with the accepted request: [migration 047:719](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:719>).

The budget audit check requires the budget capability and version, but still does not compare audit actor/reason to the ledger entry: [migration 047:788](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:788>).

Because both operational roles receive direct column-level audit insertion privileges, a buggy or compromised capability principal can commit authoritative evidence with falsely attributed audit evidence. This violates ADR-0017’s mandatory causal audit contract.

### Medium

1. Reservation expiry is only validated in C++, not by the database authority.

The service rejects `expires_at <= transaction_timestamp()`: [CampaignOperationsRepository.cpp:1779](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:1779>). The reservation trigger performs authorization, budget, and campaign checks but never enforces future semantic expiry: [migration 047:484](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:484>).

A request-acceptor principal using its granted insert columns can therefore create an already-expired held reservation and ready request, consuming budget outside the accepted workflow contract.

2. Required concurrency, ACL, and CLI behaviors are insufficiently tested.

The staged tests cover exact duplicate acceptance, rollback by explicit transaction abort, basic authorization/budget denial, safe migration rerun, and one cross-role denial. They do not cover:

- authorization revoke versus acceptance;
- budget amendment/revocation versus acceptance;
- competing changed-payload acceptance;
- deferred commit failure after each partial evidence combination;
- direct role attempts to forge prerequisite, expiry, or audit evidence;
- request-acceptor attempts to mutate budget;
- sequence `UPDATE`, table `TRUNCATE`, and broader exact ACL assertions;
- CLI mutual exclusion, confirmation, dry-run, duplicate options, and exit codes.

The current “rollback” test aborts a complete successful acceptance rather than proving each deferred completeness constraint independently: [CampaignOperationsRepositoryTests.cpp:1778](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1778>).

### Low

1. New CLI options silently permit repeated values.

The new parser does not track duplicate command or metadata options, unlike several neighboring command families. Repeating actor, reason, expected version, or budget value silently uses the last value. For confirmed budget mutations, duplicate options should be rejected deterministically.

2. Machine-readable output is incomplete and diagnostics are not safely framed.

Acceptance output omits the exact authorization event, budget entry, acquisition event, and reservation identity evidence: [CampaignOperationsService.cpp:284](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsService.cpp:284>). Request-status output also omits the reservation and budget details already available in its projection.

Additionally, raw `error.what()` is embedded in comma-delimited output without escaping, so SQL diagnostics containing commas or newlines can break machine parsing: [CampaignOperationsService.cpp:57](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsService.cpp:57>).

## Positive observations

- The Phase 2 boundary is clean. No staged Campaign Operations code invokes Phase 5, creates experiments, dispatches, leases requests, or touches scheduler/worker authority.
- Layering remains CLI → service → repository → PostgreSQL.
- Acceptance uses one caller-owned libpqxx transaction; exceptions and deferred-constraint failures roll back the entire operation.
- Deferred constraints require reservation, request, acquisition event, and audit references to coexist at commit.
- Locking follows authorization → budget → campaign using the same semantic authorization key as Phase 1. Budget administration consistently uses budget → campaign.
- Advisory locks serialize duplicate acceptance, budget mutation, and authorization changes without an observed lock inversion.
- Budget accounting correctly derives `A`, `C`, `L`, `H`, `M`, and `V` from the append-only head and reservations. No competing balance snapshot was introduced.
- Ledger version and predecessor uniqueness prevent forks. Revocation preserves obligations, and supersession is allowed only after a revoked head.
- Idempotency uses typed natural keys and complete canonical comparisons rather than trusting hashes.
- Runtime capability roles are hardened `NOLOGIN` roles, are not granted to `pqxx`, and receive no update/delete/dispatch/experiment privileges in the staged migration.
- Status reads use repeatable-read, read-only transactions.
- Documentation consistently states that accepted requests remain undispatched.

## Verification performed

Passed:

```text
git diff --cached --check
```

```text
clang++ -std=c++20 -Wall -Wextra -Werror \
  -I Sources -I Headers \
  Tests/CampaignOperationsTests.cpp \
  Sources/CampaignOperations.cpp \
  Sources/ExperimentRecommendation.cpp \
  -o /tmp/campaign_operations_phase2_review_domain_test
```

The resulting pure test executable passed with exit code 0.

The repository/integration test compiled cleanly with `-Wall -Wextra -Werror` and libpqxx.

## Missing verification

- Repository/migration integration tests were not run because no `LSTM_TEST_DB_NAME` disposable database was configured.
- The Release build was not rerun independently. A production scheduler and six active training workers are using the configured Release product path; rebuilding there could replace the executable used for subsequent worker launches.
- The existing Release binary is newer than the staged source files, but that prior build was not treated as independent verification.
- No staged CLI test suite exercises the new parser paths.

## Recommended fixes

1. Bind request prerequisite policy and provenance exactly to the referenced accepting authorization in both the database trigger and repository reload/replay validation.
2. Strengthen deferred audit constraints so cause, capability, actor, reason, prior/resulting version, and all causal IDs exactly match the authoritative ledger or acceptance evidence.
3. Reject non-null reservation expiry at or before PostgreSQL transaction time in the reservation trigger.
4. Add negative direct-capability tests for all three defects.
5. Add independent-connection races for authorization revocation and budget amend/revoke versus acceptance.
6. Add deferred atomicity, exact ACL/sequence/TRUNCATE, and CLI parser/output tests.
7. Reject duplicate Campaign Operations CLI options and expose the exact grant, budget, reservation, acquisition, and request evidence in framed machine output.

## Overall readiness assessment

The architecture, atomicity model, budget arithmetic, lock order, and phase boundary are strong. The immutable-evidence and audit-integrity gaps are nevertheless commit-blocking because the database grants the affected capability principals enough direct insert authority to exercise them.

Review file changes: none.

`git status --short`:

```text
M  Database/README.md
A  Database/migrations/047_campaign_operations_budget_request_acceptance.sql
M  ExpertAdvisor.xcodeproj/project.pbxproj
M  Sources/CampaignOperations.cpp
M  Sources/CampaignOperations.hpp
M  Sources/CampaignOperationsRepository.cpp
M  Sources/CampaignOperationsRepository.hpp
A  Sources/CampaignOperationsService.cpp
A  Sources/CampaignOperationsService.hpp
M  Sources/ExperimentScheduler.cpp
A  Tests/CampaignOperationsPhase2MigrationTests.sql
M  Tests/CampaignOperationsRepositoryTests.cpp
M  Tests/CampaignOperationsTests.cpp
A  docs/CampaignOperationsPhase2.rst
M  docs/architecture/Volume_XII_Database.md
M  docs/architecture/Volume_X_Research_Automation.md
?? CampaignOperations_Phase2_Budget_Reservation_Request_Implementation_Output.md
?? review_artifacts/campaign_operations/authority_closure/
```

`git diff --cached --stat`:

```text
16 files changed, 5147 insertions(+), 16 deletions(-)
```

READY AFTER IDENTIFIED ISSUES ARE RESOLVED