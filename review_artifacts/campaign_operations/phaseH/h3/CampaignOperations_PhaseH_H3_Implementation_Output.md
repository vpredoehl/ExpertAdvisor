---
title: "Campaign Operations Phase H H3 Bounded Campaign Manager Run-Once Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H3_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations Phase H H3 Bounded Campaign Manager Run-Once Implementation

Implemented H3 bounded Manager run-once.

Summary:

- Added `--campaign-operations-manager-run-once LIMIT --yes`, bounded to 100.
- Added one optimistic, read-only `REPEATABLE READ` candidate snapshot.
- Added deterministic UTF-8/FNV-1a operation identity and full canonical replay validation.
- Reused the common Phase E engine; no test hooks exposed.
- Added sequential processing, local-failure continuation, global-stop reporting, and replay/new-operation reporting.
- Added immutable Attempt V2-linked source evidence via migration 058.
- Added H3 documentation and structural/identity/CLI tests.
- Updated standalone H1/H2 test link sources for the shared Manager identity unit.

Schema change was strictly necessary because H1/H2 Attempt V2 has no Manager source-canonical field. Migration 058 adds an immutable one-to-one evidence relation without changing historical V1/V2 bytes, granting no LOGIN or scheduler privileges.

Verification passed:

```text
CampaignOperationsPhaseH3Tests.cpp
CampaignOperationsPhaseH3CliTests.sh
CampaignOperationsPhaseH3ContractTests.sh
CampaignOperationsPhaseH1CliTests.sh
CampaignOperationsPhaseH2WorkflowTests.sh
CampaignOperationsPhaseH2ConcurrencyTests.sh
Disposable migration 058 verification
```

Build passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor build
```

Active scheduler/training workers were inspected and left undisturbed. No production data or roles were modified.

Changed H3 files include:

- `Sources/CampaignOperationsManager*`
- `Sources/CampaignOperationsManagerService*`
- `Sources/CampaignOperationsDispatchRepository.*`
- `Sources/CampaignOperationsDispatchService.*`
- `Sources/ExperimentScheduler.cpp`
- `Database/migrations/058_campaign_operations_h3_manager_run_once.sql`
- `docs/CampaignOperationsPhaseH3.rst`
- H1/H2 regression test harness updates and H3 tests

Pre-existing change preserved:

```text
Database/backups/LSTM_latest.dump.json
```

`git diff --check`: passed.

`git status --short` and `git diff --stat`: reviewed; no unrelated files staged or discarded.

Remaining MEDIUM verification gap: full injected H3 runtime scenarios for local/global continuation and concurrent Manager run-once require a dedicated disposable H3 workflow harness; existing H2 independent-connection concurrency coverage passes.

READY_FOR_H3_INDEPENDENT_REVERIFICATION