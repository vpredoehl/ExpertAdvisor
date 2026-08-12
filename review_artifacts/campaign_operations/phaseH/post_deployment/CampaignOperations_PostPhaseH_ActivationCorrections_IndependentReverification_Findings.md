# Campaign Operations — Post-Phase-H Activation Corrections Independent Reverification

## Executive disposition

**POST_PHASE_H_ACTIVATION_CORRECTIONS_REVERIFIED**

The packaged correction set is acceptable for the next deployment/archive step. The previously identified blockers have been addressed without evidence of a new architectural blocker in the uploaded package.

## 1. Scope reviewed

This reverification covers the packaged current worktree at:

- source baseline: `fa2bf5ee8a3267a618d3980c36cc48f31ca6a254`
- migration 059 direct-SQL readiness boundary
- genesis-empty readiness correction
- distinct production dispatch-service principal
- C++ Manager/service routing
- H1/H2 deployment audits and ACL manifest
- H2 explicit dispatch/replay/concurrency behavior
- H3 Manager compatibility/runtime integration
- H4 deployment configuration/supervisor integration
- documentation consistency
- D2 concurrency harness correction

The review is based on the uploaded packaged source/tests/artifacts and their recorded isolated test evidence. No live production database mutation was performed by this reverification.

## 2. Genesis-readiness correction

The original genesis defect was valid: zero Admission V1 and zero production Attempt V2 rows produced missing aggregate versions, which were interpreted as `canonical_contract_versions` and made the sanctioned first dispatch unreachable.

The current correction distinguishes an actually empty production-evidence family from an existing malformed/wrong/mixed family. This preserves the intended rollout sequence:

1. valid enablement;
2. zero production history;
3. readiness may become true;
4. first production dispatch creates Admission V1 / Attempt V2.

Existing evidence remains fail-closed.

**Disposition: PASS.**

## 3. Direct-SQL boundary

The prior independent reverification correctly rejected the first migration-059 design because the ordinary Manager/dispatcher authority could directly execute the mutation-capable v3 wrapper and supply the approved build canonical as ordinary SQL input.

The corrected authority model removes that defect:

- raw V2 transition remains unavailable to Manager/dispatcher/PUBLIC/pqxx;
- the final v3 mutation wrapper is granted only to the sealed H1 owner and the distinct `campaign_operations_production_dispatch_service` capability;
- ordinary Manager/dispatcher authority cannot execute the wrapper;
- ordinary Manager cannot `SET ROLE` into the dispatch-service capability;
- C++ performs actual-running-build readiness on the Manager connection, then uses a separate dispatch-service connection for the mutation path;
- deployment configuration requires a distinct dispatch-service login;
- H4 configuration validates that the Manager login and dispatch-service login are distinct.

This satisfies the threat model that originally exposed the direct-SQL bypass: possession of the ordinary Manager login no longer gives direct SQL reachability to the final production mutation boundary.

**Disposition: PASS.**

## 4. SQL readiness gate

Migration 059 retains the database-observable readiness gate and does not replace it with a caller-controlled GUC, token, application name, PID, hostname, or build-path string.

The gate preserves checks for:

- migration identity;
- scheduler evidence;
- effective enablement;
- approved build canonical;
- permitted production role graph;
- completion nested-V2 proof;
- reconciliation blockers;
- enablement-chain validity;
- already-materialized Admission V1 integrity;
- already-materialized production Attempt V2 integrity.

Genesis-empty Admission/Attempt families are allowed because the existing-evidence scans are vacuously clean when no relevant rows exist.

**Disposition: PASS.**

## 5. C++ routing and service separation

`ExperimentScheduler.cpp` now supplies two distinct connection strings for production dispatch/Manager operation:

- `CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER`
- `CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER`

The normal production path still captures and validates the actual Release build before production readiness succeeds. Acquisition switches to the production dispatch-service capability; normal hydration/handoff continues through the established production Phase-5 transactional role.

The service separation is therefore operational rather than documentation-only.

**Disposition: PASS.**

## 6. ACL / deployment model

Migration 059 creates a NOLOGIN dispatch-service capability and grants the authorized v3 wrapper to that capability rather than to the normal dispatcher role. The explicit ACL manifest and H1/H2 deployment audits were updated consistently.

The packaged H4 connection environment example includes the distinct dispatch-service login, and the supervisor configuration rejects a dispatch-service login equal to the Manager login.

No evidence in the package shows a Manager-to-service membership or ADMIN OPTION path that would restore the bypass.

**Disposition: PASS.**

## 7. D2 concurrency correction

The D2 failure was correctly classified as a test-fixture/version assumption defect rather than a production authorization/concurrency defect.

The corrected isolated suite reaches the actual D2 race proof:

- durable acquisition is established;
- independent handoff and disable actors are identified;
- lock/blocker evidence establishes disable-first ordering;
- stale handoff is rejected with `campaign_operations_production_dispatch_authority_invalid`;
- final request authority remains consistent;
- no partial downstream state is created.

Recorded full-suite result:

`H2_FOUR_RACE_SUITE_OK C1=PASS C2=PASS D1=PASS D2=PASS E_SAME_KEY=PASS F_DIFFERENT_KEY=PASS`

and:

`H2_CONCURRENCY_HARNESS_OK ... four_races=PASS`

**Disposition: PASS.**

## 8. Focused security regression

The recorded isolated direct-SQL boundary regression exits 0 with:

`H8_DIRECT_SQL_BOUNDARY_REGRESSION_OK raw_denied=PASS inherited_denied=PASS manager_wrapper_denied=PASS dispatcher_wrapper_denied=PASS service_path=PASS public_denied=PASS readiness_gate=PASS replay=PASS isolated_clones=PASS`

This directly addresses the earlier independent-reverification blocker and verifies that the legitimate service path still functions.

**Disposition: PASS.**

## 9. Remaining risk / deployment condition

This reverification does not constitute proof that migration 059 is already correctly installed in the live LSTM catalog; the package intentionally avoids live production mutation. The deployment step must therefore create/provision the distinct dispatch-service LOGIN and apply migration 059 using the reviewed deployment/audit procedure before production Manager dispatch is resumed.

That is a deployment action, not a source-level correction blocker.

No remaining code/test/architecture blocker is identified from the uploaded package.

## 10. Recommendation

The correction set is ready to be archived/staged/committed as the post-Phase-H activation correction baseline, followed by database backup if desired for the schema-changing migration, then controlled migration-059 deployment and production-principal provisioning/audit.

**READY_FOR_POST_PHASE_H_ACTIVATION_CORRECTIONS_ARCHIVE_AND_DEPLOYMENT**
