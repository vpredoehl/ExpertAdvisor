---
title: "LSTM Campaign Operations Post-064 Residual ACL State Remediation and H1 Post-Upgrade Closure"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_Post064_ResidualACL_StateRemediation_and_H1_PostUpgrade_Closure_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Campaign Operations Post-064 Residual ACL State Remediation and H1 Post-Upgrade Closure

## Verdict: PASS

Migration 065 durably corrected localhost production ACL state. The unchanged post-064 authority composition now passes the H1/H2 post-upgrade audit, while injected unauthorized ACLs remain fail-closed.

### Exact remediation

| Tuple | Prior class | Checked-in authority | Action | Result |
|---|---|---|---|---|
| `schema public → pqxx USAGE` (grantor `pg_database_owner`) | C, actual-minus | None; no 045–064 authority | `REVOKE USAGE ON SCHEMA public FROM pqxx` | Absent |
| `dispatch_audit_reference_even_seq → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `dispatch_audit_reference_even_seq → pqxx USAGE` (H1 authority) | C | None | Revoke | Absent |
| `dispatch_attempt_id_seq → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `dispatch_attempt_id_seq → pqxx USAGE` (H1 authority) | C | None | Revoke | Absent |
| `operational_request_id_seq → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `operational_request_id_seq → pqxx USAGE` (H1 authority) | C | None | Revoke | Absent |
| `dispatch_attempt → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `dispatch_audit_reference_event → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `schema_migrations → pqxx SELECT` (grantor `vjp`) | C | None | Revoke | Absent |
| `production_readiness_v1 → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `production_status_v1 → pqxx SELECT` (H1 authority) | C | None | Revoke | Absent |
| `operational_request.dispatcher_identity → owner UPDATE` | C, expected-minus | 055:4269–4273; frozen 055 column manifest | Grant direct non-grantable UPDATE | Present |
| `operational_request.lease_expires_at → owner UPDATE` | C | Same | Grant | Present |
| `operational_request.lease_token_hash → owner UPDATE` | C | Same | Grant | Present |
| `operational_request.request_state → owner UPDATE` | C | Same | Grant | Present |
| `operational_request.state_version → owner UPDATE` | C | Same | Grant | Present |
| `operational_request.updated_at → owner UPDATE` | C | Same | Grant | Present |

The six missing grants were absent from `pg_attribute.attacl`; 055 explicitly grants them and no checked-in 056–064 migration revokes them. The available evidence therefore identifies post-055 production ACL drift; a more specific historic external operation is not recoverable from the catalog.

### Changed files

- [065_campaign_operations_post064_residual_acl_state_remediation.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/065_campaign_operations_post064_residual_acl_state_remediation.sql) — guarded, transactional remediation with direct-catalog postconditions.
- [CampaignOperationsPhaseH2WorkflowTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.sh) — reproduces all 18 discrepancies, validates migration/replay/ledger, and exercises fail-closed regressions.

Migration 065 requires exact ledger identities/checksums for 055, 056, 059, and 064; changes no ownership, membership, default ACL, PUBLIC grant, or grant option.

055 and frozen manifests are unchanged. H2 audit/ledger validation was not modified and passed. The 064 overlay remains validated through the existing audit path. The composition authority remains exactly 118 tuples (`A=83`, `B1=35`), with no `pqxx` tuple added.

### Verification

Passed:

- `python3 Tests/CampaignOperationsH1Post064AclProvenanceTests.py`
- `bash Tests/CampaignOperationsPhaseH2WorkflowTests.sh`
  - migration 065 applies and replays safely
  - 12 `pqxx` ACLs revoked; six owner updates restored
  - injected unauthorized `pqxx` schema ACL fails closed
  - removed frozen-055 owner update fails closed
  - H2 validation and 064 overlay remain active
- `bash -n Scripts/CampaignOperationsH1DeploymentAudit.sh`
- `git diff --check`

Localhost preflight: `::1|5432|f|off`.

Local migration result:

```text
MIGRATION_APPLY,version=065,...
MIGRATION_DONE,applied=1,skipped=60,database=LSTM,host=localhost
```

Ledger/checksum:

```text
065|065_campaign_operations_post064_residual_acl_state_remediation.sql|c1565e4a...|matches=true
```

Direct catalog verification:

```text
unauthorized_pqxx_remaining=0
required_owner_update_missing=0
owner_update_grant_option_present=0
```

Final audit command exited `0`:

```bash
bash Scripts/CampaignOperationsH1DeploymentAudit.sh \
  --stage post-upgrade --host localhost --port 5432 --user vjp --database LSTM
```

Salient output:

```text
H2_DEPLOYMENT_AUDIT_OK stage=post-upgrade ...
H1_DEPLOYMENT_AUDIT_V1_OK stage=post-upgrade
```

`git diff --check`: passed.

`git status --short` retains pre-existing review artifacts and modifications, plus the new untracked migration. `git diff --stat` reports the tracked pre-existing-plus-current test/audit work; migration 065 is untracked and therefore omitted from that stat. No commit was created.