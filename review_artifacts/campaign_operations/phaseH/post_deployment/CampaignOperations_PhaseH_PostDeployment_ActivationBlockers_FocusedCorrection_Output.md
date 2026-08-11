---
title: "Campaign Operations Phase H Post-Deployment Activation Blockers Focused Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_PostDeployment_ActivationBlockers_FocusedCorrection_Codex_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H Post-Deployment Activation Blockers Focused Correction

## Result

READY_FOR_INDEPENDENT_REVERIFICATION

Both activation blockers are corrected without changing migrations 055–058 or production data.

- H1 pre-enablement now treats the H2 readiness wrapper as a ledger-bound post-H1 evolution, while the frozen 055 manifest remains unchanged. Unknown sealed-owner functions still fail closed.
- Production commands now use explicit deployment LOGIN environment variables; generic `pqxx` routing remains unchanged.

## Production LOGIN mapping

- Manager/readiness/status/dispatch/run-once: `CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER`
  - dispatcher, phase5 transactional, reader, scheduler-evidence reader
- Enable: `CAMPAIGN_OPERATIONS_PRODUCTION_ENABLER_DB_USER`
  - enabler, reader, scheduler-evidence reader
- Disable: `CAMPAIGN_OPERATIONS_PRODUCTION_DISABLER_DB_USER`
  - disabler, reader

No fallback to `pqxx`; missing configuration fails closed. H4 requires and verifies the Manager login value.

## Migration safety

No new migration was required. The correction is audit/deployment routing only. Verified unchanged SHA-256 values for migrations 055–058 match the supplied values.

## Files changed

- `Sources/ExperimentScheduler.cpp`
- `Scripts/CampaignOperationsH1DeploymentAudit.sh`
- `Scripts/CampaignOperationsH4Supervisor.py`
- `Deployment/CampaignOperationsH4/campaign-operations-h4.connection.env.example`
- `Deployment/CampaignOperationsH4/campaign-operations-h4.example.json`
- `Tests/CampaignOperationsPhaseH2WorkflowTests.sh`
- `Tests/CampaignOperationsPhaseH3CliTests.sh`
- `Tests/CampaignOperationsPhaseH4SupervisorTests.py`
- `Tests/CampaignOperationsPhaseHProductionConnectionTests.sh`
- `docs/CampaignOperationsPhaseH1.rst`
- `docs/CampaignOperationsPhaseH2.rst`
- `docs/CampaignOperationsPhaseH4.rst`

## Validation

Passed:

- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- Disposable H1/H2 pre-enablement audit: accepted wrapper and Manager tuple passed.
- Disposable negative cases: arbitrary sealed-owner function rejected with `H1A004`; altered wrapper rejected with `H2A005`.
- `Tests/CampaignOperationsPhaseH2PrivilegeDeploymentTests.sh`
- `bash Tests/CampaignOperationsPhaseH3CliTests.sh …/LSTM_Release`
- `bash Tests/CampaignOperationsPhaseH3ContractTests.sh`
- `bash Tests/CampaignOperationsPhaseHProductionConnectionTests.sh`
- `python3 Tests/CampaignOperationsPhaseH4SupervisorTests.py` — 31 tests passed.
- Release build succeeded:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
- `git diff --check` passed.

The aggregate `CampaignOperationsPhaseH2WorkflowTests.sh` could not complete in this session because its preserved-H1 temporary-cluster setup exited before producing harness output; the focused H2 privilege/audit and new workflow cases were validated separately.

No production enablement, dispatch, Manager run-once, scheduler action, or `pqxx` capability grant was performed.

`git status --short` shows the intended modified files above plus the new production-connection test. `git diff --stat` reports 11 tracked files changed, 293 insertions, and 32 deletions; the new untracked test is not included in that stat.