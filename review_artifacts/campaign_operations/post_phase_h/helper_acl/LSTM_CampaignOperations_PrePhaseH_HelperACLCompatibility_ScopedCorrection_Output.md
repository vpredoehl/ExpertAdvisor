---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Compatibility Scoped Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_HelperACLCompatibility_ScopedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Compatibility Scoped Correction

Implemented the scoped correction.

- Root cause: both Phase A–G budget and request repositories directly call `lock_campaign_operations_campaign(bigint)` after H1 revoked capability-role execution.
- Minimal ACL closure: grant `EXECUTE` only on that helper to:
  - `campaign_operations_budget_administrator`
  - `campaign_operations_request_acceptor`
- No grants were added for the other four lock helpers; their paths use advisory locks or separate Phase-H capabilities.

Migration added: `064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql`.

Updated H1 audit integration, ACL overlay manifest, README, and focused privilege/replay/CLI tests. The correction preserves H1 ownership, `SECURITY DEFINER`, `search_path`, PUBLIC/`pqxx` denial, dispatcher/Phase-5 isolation, and pre-Phase-H LOGIN separation.

Verification passed:

- Migration replay and ACL regression.
- H1 manifest validator.
- Production connection-routing test.
- Phase 2 CLI parser test.
- Release build:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
- `git diff --check`.

Environment-backed budget/request mutations were not run because no disposable populated fixture was available.

`git status --short` still includes pre-existing backup/review artifacts. Tracked diff: 6 files, 195 insertions, 32 deletions; new migration, manifest, and regression scripts are untracked additions.