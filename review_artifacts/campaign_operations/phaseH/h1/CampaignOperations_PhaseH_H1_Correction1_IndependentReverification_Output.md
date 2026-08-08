---
title: "Campaign Operations Phase H H1 Correction 1 Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_Correction1_IndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Correction 1 Independent Reverification

## 1. Executive summary

Correction #1 is not complete.

The catalog preflight is correctly positioned before the first migration mutation and nominally classifies all requested function attributes. The dedicated hostile suite passes. However, an independently reproduced ACL defect allows expected grants to disappear whenever the named grantee role is absent:

- One replay silently recreated a missing reader role and restored its function grant.
- Another replay succeeded while `campaign_operations_completion_writer` and its required function grants remained absent.
- The post-DDL audit accepted that incomplete final ACL state.

The anti-drift test also omits several duplicated contract families, and hostile coverage does not exercise many required incompatibilities.

## 2. Findings

### Critical

None.

### High — Missing grantee roles bypass exact ACL validation

Both ACL comparisons construct expected grants by inner-joining the literal grantee name to `pg_roles`:

- Preflight: [migration 055:895](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:895>)
- Post-DDL audit: [migration 055:4871](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:4871>)

If the expected role is absent, the corresponding expected ACL tuple vanishes from the comparison.

Independent disposable-cluster results:

```text
before_role_drop=t
hostile_precondition_role_absent_and_grant_absent=t
replay_exit=0
after_role_recreated_and_grant_restored=t
```

Thus replay mutated an incompatible protected-function ACL state instead of rejecting it during preflight.

A stronger case used `campaign_operations_completion_writer`, which migration 055 does not recreate:

```text
hostile_precondition_role_absent=t
replay_exit=0
final_role_still_absent=t
final_expected_grant_absent=t
```

Migration replay—including its post-DDL audit—succeeded with manifest-required explicit function grants missing.

This fails required verification items 1, 2, 3, and 7.

### Medium — Duplicate contract drift protection is incomplete

Current literal values match mechanically, but the checked-in test only compares:

- Return contracts
- Identity-argument contracts
- Final extra ACL contracts

See [ProtectedFunctionPreflightTests.sh:27](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:27>).

It does not compare these duplicated families:

- `protected_functions` versus `allowed_functions`
- `protected_function_signatures` versus either `allowed_function_signatures` copy
- `protected_function_contracts` versus either `allowed_function_contracts` copy

The duplicated declarations occur at [migration 055:48](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:48>), [migration 055:97](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:97>), [migration 055:195](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:195>), and [migration 055:4172](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:4172>).

They match now, but they can drift without the focused test failing. Required verification item 4 is therefore not satisfied.

### Medium — Hostile tests are not comprehensive

The dedicated suite covers eight cases:

- Return type
- Procedure/object kind
- Leakproof
- Unexpected explicit grantee
- PUBLIC execution
- Grant option
- OUT metadata
- Owner

See [ProtectedFunctionPreflightTests.sh:203](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:203>).

It does not independently exercise:

- Aggregate or window-function kind
- Alternate schema
- Overload or input-signature mismatch
- Input argument-name/identity mismatch
- Volatility
- Parallel safety
- SECURITY DEFINER/INVOKER
- Missing required explicit grant
- Missing expected grantee role
- NULL-versus-explicit ACL origin
- INOUT and TABLE metadata specifically
- Drift mutations for the omitted literal families

The existing snapshot comparison is strong for covered fixtures: it captures complete catalog and definition state, compares before/after bytes, and checks that the first H1 relation was not created. But eight fixtures do not comprehensively prove the full protected tuple.

### Low

None.

## 3. Evidence by requirement

| Requirement | Result | Evidence |
|---|---|---|
| Preflight before mutation | Partially verified | The prerequisite block at lines 7–30 is read-only; protected-function preflight ends at line 946; first mutation is `CREATE ROLE` at [line 1031](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1031>). Missing-role ACL states nevertheless bypass rejection. |
| Function vs procedure | Implemented | Exact `prokind = 'f'` enforced at [line 761](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:761>); procedure tested, aggregate not tested. |
| Ownership | Implemented and tested | Owner phase checks at [line 694](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:694>). |
| Schema/name/signature | Implemented, incompletely tested | Alternate-schema, uniqueness, overload, defaults and variadic checks at [line 647](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:647>). |
| Identity arguments | Implemented, incompletely tested | `pg_get_function_identity_arguments` comparison at [line 821](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:821>). |
| Return and OUT/TABLE metadata | Implemented | Return type, set-returning state, all argument types, modes and output names at [line 799](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:799>). |
| Volatility, parallel, leakproof, security mode | Implemented | Catalog tuple comparison at [line 735](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:735>). Only leakproof is hostile-tested. |
| ACL/PUBLIC/explicit grants | Defective | Bidirectional set comparison exists at [line 867](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:867>), but absent grantee roles make required tuples disappear. |
| Rejection preserves object | Verified for covered rejection paths | Before/after catalog snapshots and no-new-relation assertion at [test:142](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:142>). The missing-role cases are not rejected. |
| Literal drift | Not fully verified | Only three of at least six duplicated families are compared. |
| Checksums/manifests | Consistent now | Migration SHA and embedded SHA both equal `e696c851…18fdbb`; manifest digest and migration marker both equal `ec6e34b1…4fb9f`. Manifest validator passed. |

## 4. Exact files requiring changes

- [Database/migrations/055_campaign_operations_production_admission_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql>)
  - Fail closed when any role named by an expected protected-function ACL tuple is absent.
  - Ensure preflight and post-DDL audit cannot omit expected grants through role joins.

- [Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh>)
  - Add missing-role and missing-required-grant hostile fixtures.
  - Compare every duplicated preflight/audit contract family.
  - Add hostile cases for currently uncovered tuple components.

- [Sources/CampaignOperationsProductionAdmission.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.hpp:21>)
  - Regenerate the embedded migration SHA-256 after changing migration 055.

No manifest-content change is presently indicated; its current digest is consistent.

## 5. Verification performed

Passed:

- `bash -n Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Scripts/CampaignOperationsH1ManifestValidator.sh`
- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
  - Supported schema-054 upgrade and replay
  - All eight checked-in hostile fixtures
- Mechanical comparison of all current duplicate arrays: all currently match
- Correction-scoped `git diff --cached --check`
- Two independent missing-grantee disposable-cluster probes

No repository files were changed by this review. No Xcode build or `LSTM_Release` command was needed.

Repository state remains 114 staged files plus 31 untracked entries. Staged diff:

```text
114 files changed, 30122 insertions(+), 20 deletions(-)
```

## Final disposition

CORRECTION_REQUIRED