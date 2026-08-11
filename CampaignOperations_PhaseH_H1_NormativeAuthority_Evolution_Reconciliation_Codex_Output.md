---
title: "Campaign Operations Phase H H1 Normative Authority Evolution Reconciliation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_NormativeAuthority_Evolution_Reconciliation_Codex_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 Normative Authority Evolution Reconciliation

## Decision

Implemented a frozen, fail-closed source-binding registry:

- Ordinary rows remain strict live-exact source/digest checks.
- `PHASEH-INCREMENTS` is `historical_superseded`, retaining frozen H1 text/digest and requiring the exact accepted ADR-0020 external-supervision semantics.
- `VOLUMEXII-MIGRATION` is `semantic_equivalent_revision`, with an exact reviewed current excerpt/digest that proves the same migration-055 invariant.

## Row disposition

| Clause | Binding | Anchor |
|---|---|---|
| `PHASEH-INCREMENTS` | live-exact → historical-superseded | unchanged |
| `VOLUMEXII-MIGRATION` | live-exact → semantic-equivalent-revision | 97–107 → 97–105 |
| `PHASEH-LOCK-ORDER` | live-exact | 508–537 → 511–540 |
| `PHASEH-PRIVILEGES` | live-exact | 586–607 → 589–610 |
| `PHASEH-MIGRATION` | live-exact | 726–750 → 729–753 |
| `PHASEH-VERIFICATION` | live-exact | 800–822 → 808–830 |
| `VOLUMEXII-SEALED` | live-exact | 128–145 → 144–161 |
| `VOLUMEXII-AUDIT` | live-exact | 147–156 → 163–172 |
| `VOLUMEXII-TESTING` | live-exact | 355–366 → 371–382 |
| `VOLUMEXII-PERMISSIONS` | live-exact | 388–397 → 404–413 |

Canonical excerpts and canonical clause digests were unchanged for every row.

## `PHASEH-INCREMENTS`

Its frozen H1 assertion remains intact. The new binding validates ADR-0020’s exact path, `Accepted` status, external deployment-owned repeated H3 invocation, preservation of H1–H3 boundaries, and prohibition of an in-process daemon/CLI, scheduler polling, or DB singleton/heartbeat/lease/leader-election authority.

## `VOLUMEXII-MIGRATION`

Accepted as the sole semantic-equivalent revision. The reviewed current text is pinned by exact excerpt and raw digest, preserving the additive migration-055/no-login-membership/no-backfill/no-V1-reinterpretation/no-production-state-mutation proposition. Mutation of that invariant fails.

## Files changed

- `Scripts/CampaignOperationsH1EvidenceAuthority.py`
- `Tests/CampaignOperationsPhaseH1EvidenceAuthorityTests.py`
- `Tests/fixtures/CampaignOperationsH1NormativeClauses.tsv`
- `Tests/fixtures/CampaignOperationsH1SourceBindings.tsv` (new)

## Digest changes

- `AUTHORITY_DIGEST`: `34be85602ebf1c492530ebf17955aaaa2e407c1ee8d7cbc569627f6d8934df4f` → `d071538dfedcd2125bf87f476cf3750ff585d2c06acb4d690925916308ab14ef`
- New source-binding policy digest: `5702ebdb1137205199a6e0c532bc890b90c2489c5b0421d640abf704fa832c3e`

I preserved an unrelated pre-existing `EvidenceObligations` policy-digest change in the dirty workspace.

## Validation

All exit 0:

- `git diff --check`
- `bash -n Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- `python3 -m py_compile Scripts/CampaignOperationsH1EvidenceAuthority.py Tests/CampaignOperationsPhaseH1EvidenceAuthorityTests.py`
- `python3 Tests/CampaignOperationsPhaseH1EvidenceAuthorityTests.py` — 51 tests passed
- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh` — passed

## Scope confirmation

No production behavior or migration was changed for this reconciliation. No broad H3/H4 rewrite was performed, no commit was made, and the pre-existing dirty-worktree changes remain preserved.

`git status --short` still includes unrelated existing modifications plus the four authority-reconciliation files above; full `git diff --stat` reports 16 tracked files changed / 409 insertions / 56 deletions, including those unrelated pre-existing edits.