---
title: "Campaign Operations Post-Phase-H H2 Production Enable Predecessor Mismatch Diagnostic"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PostPhaseH_H2_ProductionEnablePredecessorMismatch_Diagnostic_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Post-Phase-H H2 Production Enable Predecessor Mismatch Diagnostic

## 1. Verdict

**Conclusive:** this is a stale H2 fixture-state assumption. H2 now correctly reuses the canonical completed H1 database, which already contains a valid active H1 enablement event. The H2 C++ test still assumes an empty enablement history and requests initial version `0`.

## 2. Diagnostic origin

Call path:

1. [H2 workflow shell test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.sh:338) runs the compiled H2 C++ test.
2. [H2 C++ test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.cpp:99) creates `h2-enable-001` with `expectedPriorVersion = 0`.
3. `EnableProduction` → `EnableProductionOnce` in [ProductionAdmissionService](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:265).
4. [FindCurrentProductionEnablementHead](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:712) loads the greatest `resulting_version`.
5. [Service check](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:279) throws `production_enable_predecessor_mismatch`.
6. [H2 test catch](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.cpp:271) prints `H2_WORKFLOW_CPP_FAIL diagnostic=...`.

Migration 055’s SQL transition independently enforces the same rule at [lines 3501–3510](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3501), but C++ rejects first, so the SQL function is not reached.

## 3. Expected predecessor contract

For the first H2 enable request:

- Operation key: `h2-enable-001`
- Requested prior version: `0`
- Required head: **no predecessor event at all**
- Therefore candidate predecessor ID/canonical would be absent/empty.
- If prior version were positive, the head must instead be:
  - present,
  - `event_kind = disable`,
  - `resulting_version = expectedPriorVersion`.
- The candidate’s predecessor canonical is derived from the loaded head only after this check.

Prior to the predecessor check, the service also requires:

- migration `055`, exact filename, checksum `dd01812…c3ff0`;
- complete scheduler evidence with the same canonical identity loaded twice;
- valid H2 request build contract:
  - canonical hash `fnv1a64:1d532a5c1b9dc908`;
  - source `777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d`;
  - compiler `clang++-h2-fixture`.

These prerequisites all passed.

## 4. Observed H1→H2 fixture state

Immediately before the failing executable call, the canonical numeric H1 database contained exactly one enablement event:

```text
id=1
operation_key=enable-001
event_kind=enable
predecessor_event_id=NULL
expected_prior_version=0
resulting_version=1
enablement_contract_version=1
scheduler_hash=fnv1a64:af614995e691e378
build_hash=fnv1a64:9afc5f65cbaaece3
enablement_hash=fnv1a64:05e8f676db9ba009
history_valid=true
```

Its stored canonical matches `campaign_operations_production_enablement_canonical_v1`, and it has its required immutable audit row.

Scheduler evidence was complete:

```text
required_generation=52
cutover_state=complete
evidence_hash=fnv1a64:af614995e691e378
```

Relevant migration ledger entries were exact:

```text
055  dd01812b04f0f48ab8caac40a5280ff5c6831ed2fc4f53ceb9774e0dc92c3ff0
056  e382ea14cfe80bf9d4ef01a01861679cc23be19ff190beb8c21df73d15ad4310
057  ab58c6e7433bd81925497de9b4eb825fa0c8b908d202da61a261cfd4b18368fb
058  3cd147d7b4bf118242965eeeb49b6028b4547f03d6e1c672dc8560a611337014
```

The workflow’s pre-populated 058 ledger row is not consulted by this production-enable predecessor check and is not causal.

The event is intentionally created by the H1 positive regression fixture in [CampaignOperationsPhaseH1MigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:877), which H1 runs against its primary database at [lines 1426–1428](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:1426).

## 5. Exact mismatch

| Contract field | Expected | Observed |
|---|---:|---:|
| Current predecessor/head | absent | event `1` exists |
| Head kind | n/a | `enable` |
| Head resulting version | n/a | `1` |
| H2 requested prior version | `0` | `0` |

The first failing expression is `request.expectedPriorVersion == 0 && predecessor`, which is true.

Changing only the H2 expected version from `0` to `1` would still fail: a positive-version enable requires a **disable** head, but the observed head is an enable.

## 6. Root cause

The H2 workflow change now requires H1 success and selects the canonical numeric-suffix H1 database rather than an auxiliary `_lock` database. The completed canonical H1 fixture legitimately retains its `enable-001` positive replay/canonicalization event.

The H2 C++ fixture still encodes a genesis enable (`0 → 1`). That assumption was hidden by the old H1REG027-failure workflow/selection behavior. This is not caused by the migration-055 checksum propagation, schema-migrations validation, scheduler evidence, build contract, replay canonicalization, or production code.

## 7. Minimal correction recommendation

A later correction should change only the H2 fixture boundary:

- [Tests/CampaignOperationsPhaseH2WorkflowTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.sh): after creating/granting `h2_disabler_login`, append a legitimate disposable **disable** event through `record_campaign_operations_production_disable_v1`, using event `1` and its canonical identity. Do not delete, bypass, or rewrite H1 evidence.
- [Tests/CampaignOperationsPhaseH2WorkflowTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.cpp): update the H2 enable request to expect version `2`, and its following disable request to expect version `3`.

This creates the valid sequence:

```text
H1 enable v1 → H2 fixture disable v2 → H2 enable v3 → H2 disable v4
```

Production service/repository code should remain unchanged.

## 8. Scope integrity

- H1REG027 remains resolved; it was not re-diagnosed or changed.
- H1A006 remains intact.
- No production ownership or ACL change is indicated.
- No migration/checksum correction is indicated.
- No C++ production-enable logic change is indicated.

## 9. Validation / reproduction evidence

Ran after confirming no active `LSTM_Release` worker:

```bash
bash Tests/CampaignOperationsPhaseH2WorkflowTests.sh
```

Result: exit `1`, after migration runner, H2 manifest/audit, privilege matrix, hostile-role checks, and Phase-E setup passed:

```text
H2_WORKFLOW_CPP_FAIL diagnostic=production_enable_predecessor_mismatch
```

Used the workflow’s `H2_PRESERVE_BASELINE_ROOT` / `H2_PRESERVE_CLUSTER_ROOT` hooks to inspect the disposable database before the executable call. The preserved temporary cluster was stopped afterward.

## 10. Working-tree status

`git diff --check`: passed with no output.

`git status --short` and `git diff --stat` are unchanged from the initial state: 13 pre-existing modified files and 5 pre-existing untracked files. No repository file was modified by this diagnostic.

## 11. Next step

The repository is ready for a focused H2 fixture/state correction prompt targeting:

- `Tests/CampaignOperationsPhaseH2WorkflowTests.sh`
- `Tests/CampaignOperationsPhaseH2WorkflowTests.cpp`

After that correction, rerun in this order:

1. H2 workflow;
2. H3 compatibility;
3. H3 migration-058 execution;
4. independent reverification.