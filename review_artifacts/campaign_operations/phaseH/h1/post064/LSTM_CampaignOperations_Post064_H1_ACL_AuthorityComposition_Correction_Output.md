---
title: "LSTM Campaign Operations Post-064 H1 ACL Authority-Composition Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_Post064_H1_ACL_AuthorityComposition_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Campaign Operations Post-064 H1 ACL Authority-Composition Correction

## Root cause

The localhost catalog contains ACL drift beyond the authority checked in for migrations 056–064. A permissive H1 compatibility filter would violate the requested fail-closed contract.

The captured audit is [campaign_operations_h1_post064_pre_correction_audit.log](/Volumes/Developer%20SSD/ExpertAdvisor/Artifacts/campaign_operations_h1_post064_pre_correction_audit.log).

## Authority map

| Authority | Observed tuples | Result |
|---|---:|---|
| 056 H2 manifest | 11 | Exactly backed: fixed-transition EXECUTE, scheduler lock, H2 evidence reads, readiness snapshot |
| 056 explicit Phase-E grants | 21 | Exactly backed: production Phase-5 table reads, dispatch-audit INSERT columns, five lock helpers, sequence usage |
| 059 direct-SQL boundary | 3 | Exactly backed: authorized-dispatch wrapper owner/service EXECUTE and readiness-gate owner EXECUTE |
| 064 overlay | 2 | Already separately validated and removed only after exact overlay validation |
| No 056–064 authority | 95 | Blocker |
| Frozen H1 tuples now absent | 6 | Blocker; no 056–064 supersession/revocation exists |

The 95 unbacked additions include:

- 64 legacy/pre-Phase-H column INSERT tuples for `campaign_operations_dispatcher`, `campaign_operations_phase5_transactional`, and `campaign_operations_request_acceptor`.
- 11 `pqxx` table/view/sequence privileges.
- 8 `public` schema USAGE grants.
- Legacy dispatcher/Phase-5 table and sequence privileges.
- Budget/request capability reads and an extra request-acceptor sequence privilege.

The six missing frozen tuples are `campaign_operations_owner` UPDATE on the specified `campaign_operations_operational_request` columns. No later migration revokes or supersedes them.

All local 056–064 migration SHA-256 values match their localhost ledger rows, so this is not a checksum ambiguity.

## Files changed

- No source files changed.
- Created the requested runtime evidence artifact noted above.

## Why H1 remains frozen

No migration-055 file or frozen H1 manifest was modified.

## Why H2 is not weakened

H2 ran successfully before H1 failed. No H2 script, checksum binding, role-graph validation, or audit was changed.

## Why 064 remains exact

No change was made to its existing exact overlay or its direct, non-grantable campaign-lock EXECUTE checks.

## Unauthorized-extra-ACL negative regression

Not added: implementing composition or a regression while 95 live tuples lack checked-in authority would require accepting unproven ACLs, contrary to the task’s explicit stop condition.

## Commands run and results

- `bash -n Scripts/CampaignOperationsH1DeploymentAudit.sh` — passed.
- `git diff --check` — passed.
- Required localhost post-upgrade audit — H2 passed; H1 failed `H1A006` on residual unbacked ACLs.
- Verified all migration 056–064 checksums against localhost ledger — all matched.

`git status --short`:

```text
?? Artifacts/
```

`git diff --stat`: no tracked changes.

## Final verdict

The post-064 H1 post-upgrade audit is **not yet safe to use as a composed authority check**. The blocking 95 additions and six removed H1 tuples need an exact, checked-in, ledger-bound authority or must be remediated as drift before any composition change can be safely implemented.