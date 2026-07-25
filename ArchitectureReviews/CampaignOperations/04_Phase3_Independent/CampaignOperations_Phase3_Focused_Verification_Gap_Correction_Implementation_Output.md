---
title: "Campaign Operations Phase 3 Focused Verification-Gap Correction Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase3_Focused_Verification_Gap_Correction_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 3 Focused Verification-Gap Correction Implementation

## Executive summary

The staged Campaign Operations Phase 3 / architectural Phase E implementation has been corrected and fully restaged. Focused verification now passes for retry, rollback, authorization/budget concurrency, evidence classification, adoption/control ownership, unknown commits, restart behavior, corruption rejection, duplicate acquisition, ACLs, migrations, and relevant regressions.

No commit was made. HEAD remains `3a761d145ccfa3bc05cf6f37cd5d8106055eaa48`.

## Defects corrected

- Fixed exhausted `40001`/`40P01` retries leaking SQL exceptions instead of returning stable `transient_database_retry_exhausted`.
- Changed retries to use fresh connections and authoritative reloads, allowing whole-operation replay after proven-absent unknown commits.
- Strengthened downstream evidence classification to validate complete Phase 4D/5 causality and canonical provenance.
- Added stable control-owner collision detection before binding insertion.
- Strengthened binding hydration against malformed attempts, outcomes, audits, owners, projections, canonical bytes, and version mismatches.
- Corrected reservation commitment ordering and the deferred completeness trigger’s expected bound state.
- Prevented the transactional Phase 5 role from committing arbitrary experiment/execution/activation mutations without atomic Campaign Operations binding.
- Replaced table-level INSERT grants with column-level grants that deny generated IDs and timestamps.
- Removed unnecessary sequence `SELECT`; retained only required `USAGE`.
- Added hardened default ACLs and expanded catalog verification.

## Verification-only seams

Deterministic hooks cover four acquisition and fifteen handoff positions, SQLSTATE injection, Phase 5 mutation stages, lost responses, and simulated broken connections.

They are production-inert because the only adapter:

- accepts one explicit request;
- requires database prefix `expertadvisor_campaign_operations_phase3_test_`;
- requires acknowledgement `I_UNDERSTAND_PHASE3_TEST_ONLY`;
- exposes no production CLI, database, scheduler, or configuration control surface.

## Verification results

- Retry: deterministic `40001` and `40P01` passed with three whole-operation attempts, three authoritative reloads, deterministic identities, and one final durable result. Exhaustion and arbitrary-SQLSTATE behavior passed.
- Authorization races: handoff-first, revoke-first, successor-first, and PostgreSQL-time expiry cases passed.
- Budget races: amend, revoke, supersede, and handoff-first cases passed without reversed lock order, deadlock, partial mutation, or retroactive uncommit.
- Rollback: all 4 acquisition and 15 handoff points passed all-or-none assertions.
- Evidence/adoption: create, exact adoption, missing/revoked/expired/non-head authority, partial, paused, mixed, progressed, mismatched, replay, and ownership-collision cases passed fail-closed assertions.
- Unknown commit/restart: lost response after commit, proven-absent retry, partial binding corruption, downstream-without-binding, durable lease restart, and progressed immutable-binding replay passed.
- Corruption: malformed canonical/version/digest/outcome/audit/binding/owner/reservation/request fixtures were rejected without repair or Phase 5 reinvocation.
- Duplicate selection/acquisition: stable ordering, advisory duplicate observation, two-connection contention, exactly one lease winner, and exactly one attempt ordinal passed.
- ACLs: ownership, NOLOGIN, membership denial, PUBLIC/default ACLs, column grants, generated-column denial, sequence privileges, function security, role separation, and arbitrary experiment mutation denial passed.
- Migration: clean harness installation, migration 047 upgrade path, migration 048 rerun, preservation assertions, and absence of backfill/production enablement passed.

## Commands and results

- Release build:

  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`

  Result: `BUILD SUCCEEDED`. Only warning was an environment-level missing `Info.plist` for the unrelated LLVM22 toolchain registration.

- Campaign Operations pure tests: passed.
- Phase 1–3 repository/migration/service tests: passed.
- Phase 2 CLI parser tests: passed.
- Phase 4C conversion workflow pure/repository tests: passed.
- Phase 5 launch pure/repository/service integration: passed.
- Global experiment-control pure regression: passed.
- Project validation: `plutil -lint` passed.
- `git diff --cached --check`: passed.

The disposable database `expertadvisor_campaign_operations_phase3_test_correction` used PID-scoped schemas with deterministic `DROP SCHEMA ... CASCADE` cleanup. It was dropped after testing.

## Production safety

The pre-existing scheduler PID `3428` and training-worker PIDs `4570, 4575, 4581, 4587, 4593, 4599, 4605` retained their original parents and start times. No scheduler or worker was started, signalled, paused, resumed, or disturbed.

Production dispatch remains disabled, `pqxx` has no Phase 3 capabilities, and ADR-0016 remains unsatisfied.

## Files staged

The staged diff contains 20 Phase 3-related files:

- Migration 048 and Xcode project wiring.
- Dispatch, binding, repository, and service sources.
- Transaction-bound Phase 5 test hook changes.
- Campaign Operations and Phase 5 integration tests.
- Phase 3 migration/ACL tests.
- Phase 3 and architecture documentation.

Staged diff: `7,372 insertions, 11 deletions`.

## Final worktree state

- Staged: all 20 intended Phase 3 implementation, correction, test, project, and documentation files.
- Unstaged tracked changes: none.
- Untracked and intentionally untouched:
  - `CampaignOperations_Phase3_ArchitecturalPhaseE_Implementation_Output.md`
  - `CampaignOperations_Phase3_Architecture_Readiness_CEE_Review_Output.md`
- No commit made; HEAD unchanged.
- Remaining limitation: physical network severance was not performed; unknown-connection behavior was exercised deterministically through the verification seam and fresh-connection recovery path.

READY FOR INDEPENDENT PHASE 3 VERIFICATION