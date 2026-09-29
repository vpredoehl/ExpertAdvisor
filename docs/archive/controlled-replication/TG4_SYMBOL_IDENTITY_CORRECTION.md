# TG4 symbol-identity correction

This note records a prospective-specification clerical correction. It does not
alter the original frozen study or any experiment record.

## Original freeze

- Study identifier: `tg4_cross_context_controlled_replication`
- Semantic identity: `fnv1a64:ae4943096a276492`
- Freeze timestamp: `2026-09-29T01:37:31Z`
- Artifact SHA-256: `b3ddca8de75385f6bf35932cc719f98eb4d25f3bb3e912c97c11015b1c69ee35`
- Artifact: `studies/study_ae4943096a276492.txt`

The original artifact encoded display symbols (`CADCHF` and `AUDCAD`) for its
`symbol` context dimensions.

## Diagnosis chronology

After the prospective freeze, one comparison invocation failed closed before
producing a comparison:

`study_specification_context_identity_mismatch`

No outcome or performance result was produced. The diagnosis used only the
configuration fields `experiment_id`, `symbol`, `prediction_horizon`,
`fresh_initialization_seed`, and `feature_ablation_mask`. Those persisted
configured symbols are the literal values `cadchfrmp` for experiments 676–681
and `audcadrmp` for experiments 670–675.

## Corrected freeze

The corrected prospective study is:

- Study identifier: `tg4_cross_context_controlled_replication_corrected_symbol_identity`
- Freeze timestamp: `2026-09-29T01:43:04Z`
- Semantic identity: `fnv1a64:6fd11fe711483cb3`
- Artifact SHA-256: `ec1019f8fdd1089416ffcfd11177cd284ccfb40dc161e7506c46712d59af3479`
- Artifact: `studies/study_6fd11fe711483cb3.txt`

Only the literal `symbol` values changed (`CADCHF` to `cadchfrmp` and
`AUDCAD` to `audcadrmp`), together with the explicitly derived study
identifier, freeze timestamp, semantic identity, archive filename, and byte
SHA-256. Membership, pair ordering, context ordering, seeds, intervention and
treatment mask, family boundaries, and analysis/aggregation policy are
unchanged.

The original artifact and registry entry remain immutable and independently
verify with their original identity and SHA-256. This correction is additive;
it does not rewrite or conceal the failed first comparison attempt. No outcome
evidence was inspected and no comparison was run as part of the correction.
