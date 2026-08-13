# LSTM Donchian-20 Ablation Provenance — Final Independent Reverification Findings

## Disposition

**PASS WITH ONE NON-BLOCKING EXTERNAL TEST-HARNESS FINDING**

The packaged Donchian-20 ablation/provenance implementation is internally coherent and the targeted correction closes the previously identified campaign-originating gap. The evidence supports proceeding to archive/migration/commit preparation, while preserving the unrelated Campaign Operations launch-fixture failure as an explicit open test-infrastructure item.

This reverification is based on the supplied source/diff package and the execution evidence embedded in its prior correction artifacts. It does not independently execute the user's local repository from this environment.

## Findings

### Recommendation semantic identity v4 — CLOSED

`donchian20_mode` is part of semantic canonical identity. Tests now prove that otherwise identical `enabled` and `zero_ablation` configurations have different canonical texts and hashes.

### Post-060 fixtures — CLOSED

Affected conversion/campaign fixtures now include `donchian20_mode`, its allowed-value check, and the mode in experiment uniqueness identity.

### Campaign deliberate arm origin — CLOSED

Campaign planning supports `enabled`, `zero_ablation`, or `enabled:zero_ablation`. Policy canonicalization advances to v2 and includes the arm specification. Unspecified arms preserve prior behavior and do not expand a campaign.

### One ranking member → two materialized arms — CLOSED

Migration 061 removes the old uniqueness constraint that prohibited multiple materialization members from sharing one ranking member. Proposal identity and materialization ordinal uniqueness remain authoritative. Downstream validators were updated so a valid paired arm is not rejected merely because both members share upstream ranking/recommendation provenance.

### Conversion provenance — CLOSED

The normal recommendation mutation is validated against the recommendation's canonical identity before the campaign arm is applied. The requested Donchian arm then becomes part of the proposed invocation and conversion identity.

### Experiment/model provenance — CLOSED

Experiment persistence, scheduler command construction, resume, checkpoint inference, final inference, duplicate identity, and model metadata all carry or validate the Donchian mode. Missing legacy model mode metadata resolves to `enabled`, preserving the established compatibility rule.

## End-to-end paired-arm evidence

The materialization integration fixture demonstrates that one ranked candidate expands to two members, produces `enabled` and `zero_ablation` proposals with distinct conversion identities, replays idempotently, reviews and executes both proposals, and persists exactly two paused training experiments with the two corresponding modes. No scheduler or worker is required for this proof.

## Scientific-control behavior

`zero_ablation` preserves feature width and Donchian column positions while writing exact zero to columns 32 and 33. `enabled` retains the causal prior-window Donchian calculation. The causality regression remains present. This makes the A/B comparison an information ablation without a geometry change.

## Migration review

Migration 060 establishes experiment-level Donchian mode and incorporates it into experiment identity. Migration 061 narrowly removes the materialization/ranking-member uniqueness rule that prevented deliberate paired arms. Remaining proposal/member identity protections continue to distinguish the pair.

## Validation evidence

The supplied correction evidence reports passes for Donchian feature and Tensor tests, recommendation identity tests, campaign planning tests, paired campaign materialization integration on isolated PostgreSQL, campaign status/handoff/outcome/follow-up tests, conversion execution/activation tests, campaign activation/execution tests, Release build, and `git diff --check`.

## Non-blocking external finding

`ExperimentRecommendationCampaignLaunchRepositoryTests` still fails in a Campaign Operations Phase 3 atomic-handoff fixture with PostgreSQL `permission denied for table experiment`. The evidence attributes this to the fixture granting the needed SELECT privilege to `campaign_operations_phase5_transactional` but not `campaign_operations_dispatcher`. The failure occurs before a Donchian-specific assertion.

I classify this as **non-blocking for the Donchian correction**, but it should remain a separately tracked Campaign Operations fixture issue.

## Final verdict

**PASS WITH ONE NON-BLOCKING EXTERNAL TEST-HARNESS FINDING.**

The Donchian-20 ablation/provenance correction is suitable to proceed to archival and production migration preparation. Before launching controlled experiments, migrations 060 and 061 should be applied in order to the production LSTM database, followed by the project's schema-change backup procedure and canonical Release rebuild/verification.
