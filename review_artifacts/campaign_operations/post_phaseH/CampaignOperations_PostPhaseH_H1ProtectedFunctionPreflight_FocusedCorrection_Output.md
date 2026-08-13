---
title: "Campaign Operations Post-Phase-H H1 Protected Function Preflight Focused Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PostPhaseH_H1ProtectedFunctionPreflight_FocusedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Post-Phase-H H1 Protected Function Preflight Focused Correction

## 1. First failing command/assertion

`psql ... h1_preflight_legacy_* -f Database/migrations/055_campaign_operations_production_admission_foundation.sql`

It failed with `H1A006 protected function preflight ACL mismatch: public.lock_campaign_operations_authorization_head(bigint,text)` before the new hardened-lock fixture ran.

## 2. Root cause

Defect in migration 055 predecessor-ACL acceptance logic. It conflated the ordinary explicit-PUBLIC predecessor ACL with the hardened no-PUBLIC form, causing the valid schema-054 baseline from `LSTM_latest.dump` to fail.

## 3. Correction made

Updated migration 055 to distinguish exact predecessor forms:

- Ordinary lock ACLs with PUBLIC retain the normal expected ACL.
- Hardened lock ACLs without PUBLIC require the exact predecessor-only grants.
- The dispatcher transition still requires its predecessor dispatcher grant whenever explicit.
- Historical owner-only NULL/default-ACL hardening remains accepted.

The full set comparison remains intact; arbitrary grants, PUBLIC expansion, grant options, missing required grants, and explicit-default ACL origin remain rejected.

## 4. Files changed

- [055_campaign_operations_production_admission_foundation.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql)
- [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh) — pre-existing fixture/role changes preserved; no additional final test-script edit was needed.

## 5. Verification results

- `bash -n Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh` — passed
- `git diff --check` — passed
- `bash Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh` — passed, exit `0`

The focused test confirmed hardened locks are accepted, predecessor-only grants are absent after sealing, and all ACL negative fixtures pass.

## 6. Final H1 ACL contract

Unchanged. `protected_function_final_extra_acl_contracts` was not modified; predecessor-only lock grants are accepted only pre-sealing and explicitly revoked during sealing.

## 7. Remaining blocker

None. No production database was accessed or modified.

`git status --short` shows the two modified task files plus the pre-existing `.pre_lock_predecessor_fix` untracked copies. `git diff --stat`: 263 insertions, 9 deletions across the two modified files.