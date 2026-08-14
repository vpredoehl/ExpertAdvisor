---
title: "LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture Link Closure Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_H1_HistoricalFixture_LinkClosure_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture Link Closure Correction

CORRECTED. The H1 link commands now use the real production materialization repository and its complete dependency closure; focused adapters are disabled only in that production-repository mode.

Report: [H1 link-closure correction](</Volumes/Developer SSD/ExpertAdvisor/LSTM_CampaignOperations_PrePhaseH_H1_HistoricalFixture_LinkClosure_Correction_Output.md>)

Changed:

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- `Tests/CampaignOperationsRepositoryTests.cpp`
- Added the requested report.

Verified:

- Focused workflow and broad links with `-Wall -Wextra -Werror`
- `bash -n Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- `git diff --check`
- Full H1 suite passed, including broad Phase 1–5 regression and final reference-graph checks.

The final preserved H1 cluster is `/tmp/ea-h1-pg.xqoNJK`. No unrelated clusters were touched.

`git status --short` and `git diff --stat` were captured in the report; unrelated existing worktree changes remain unmodified.