---
title: "Campaign Operations Admission Top-Level Dispatch Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_AdmissionTopLevelDispatch_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations Admission Top-Level Dispatch Correction

PASS

1. Root cause confirmed: the top-level dispatcher omitted `campaignOperationsAdmitMaterializationId`, causing valid admission commands to return 1 without output.

2. Files changed:

- [Sources/ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:23935)
- [Tests/CampaignOperationsPhase2CliTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhase2CliTests.sh:54)

3. Correction: admission now routes through `RunCampaignOperationsCommand()`, preserving the existing pre-Phase-H connection/principal validation and `RunOperationalCampaignAdmissionCommand()` path.

4. Regression coverage: a valid admission invocation with the pre-Phase-H principal unset must emit the existing missing-principal diagnostic. It would fail with silent exit 1 if dispatch were removed.

5. Dispatch audit: all 21 neighboring Campaign Operations command selectors are represented; no additional analogous omissions found.

6. Verification:

- Release build: succeeded.
- Phase 2 CLI tests: passed.
- Phase 4 CLI tests: passed.
- Phase 5 CLI tests: passed.
- Phase H connection-routing tests: passed.
- `git diff --check`: passed.

No schema, ACL, role, grant, migration, privilege, or production-authority changes were made by this task. No live database admission was executed.

The working tree contains pre-existing unrelated changes and untracked database artifacts; they were preserved.