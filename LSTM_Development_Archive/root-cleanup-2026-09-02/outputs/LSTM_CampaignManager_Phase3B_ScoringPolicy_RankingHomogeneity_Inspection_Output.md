---
title: "LSTM Campaign Manager Phase 3B Scoring Policy Ranking Homogeneity Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3B Scoring Policy Ranking Homogeneity Inspection

# Campaign Manager Phase 3B inspection report

## 1. Executive conclusion

The current system does **not** guarantee scoring-policy ranking homogeneity.

Normal score runs and normal evaluation runs are internally single-policy. The defect appears when ranking loads evaluation results:

- `evaluation_run` scope is homogeneous under the normal CLI/service path.
- `recommendation_scan`, `symbol`, `horizon`, `family`, `symbol_horizon`, and `global` scopes can combine evaluation results from multiple evaluation runs with different scoring-policy definitions.
- Ranking compares their numeric `final_score` values without validating policy equality.
- The ranking database trigger validates scope, copied values, membership, buckets, and tie-breaks, but does not validate scoring-policy homogeneity.
- A completed heterogeneous snapshot is accepted by downstream campaign planning as authoritative ordering.

The correct Phase 3B behavior is fail-closed: reject the entire requested ranking operation before creating a snapshot if its full pre-limit population is heterogeneous, missing semantic identity, or internally inconsistent. Silent partitioning would change the meaning of the requested scope and is not supported by the current single-scope snapshot architecture.

Profitability remains entirely observational, weight-zero, and non-ranking-bearing.

Final verdict: `READY_FOR_PHASE_3B_IMPLEMENTATION`.

---

## 2. Current recommendation source/provenance architecture

Recommendation generation creates an `experiment_recommendation_scan` containing:

- recommendation policy canonical text;
- policy hash;
- policy version;
- source filters and run counters.

Each `experiment_recommendation` freezes:

- scan ID;
- source experiment/model/analysis IDs;
- symbol and horizon;
- leader score, inference accuracy, neutral proportion, and evidence count;
- changed parameter and source/proposed values;
- semantic configuration canonical/hash;
- invocation configuration canonical/hash;
- recommendation policy canonical/hash;
- source rank, generation ordinal, and structural rank.

See [026_experiment_recommendation_persistence.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/026_experiment_recommendation_persistence.sql:1>) and the authoritative policy canonicalization in [ExperimentRecommendation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp:798>).

Source discovery binds completed experiments to their final model and final analysis, then loads exact FINAL profitability evidence separately. [ExperimentRecommendationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:514>)

Phase 3A provenance is persisted as either:

- legacy: every Phase 3A field is `NULL`;
- explicitly unavailable: provenance version 1, FINAL scope, reason, no observation values;
- available: exact FINAL inference result plus exact profitability observation and all frozen metric/source hashes.

Migration 076 enforces shape, exact FINAL inference binding, exact observation binding, and recommendation-side immutability. [076_campaign_manager_final_profitability_provenance.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/076_campaign_manager_final_profitability_provenance.sql:86>)

---

## 3. Current recommendation scoring architecture

`--score-experiment-recommendations`:

1. Parses a supplied policy or constructs `RecommendationScoringPolicy{}` defaults.
2. Creates a new serial score run with canonical policy, hash, and version.
3. Loads proposed recommendations by optional symbol, horizon, scan, or recommendation ID.
4. Scores every candidate with the one request policy.
5. Ranks the run internally.
6. Persists score and component rows.

The default policy contains seven positive weights, two penalty weights, evidence/neutral/mutation normalization parameters, missing-neutral behavior, bounds, and family preferences. [ExperimentRecommendationScoring.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.hpp:13>)

Canonicalization expands every current configurable value, so omitted CLI options do not leave implicit values in persisted policy text. The FNV-1a hash is documented as an accelerator; canonical text is authoritative. [ExperimentRecommendationScoring.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:281>)

The nine current components are:

- `leader_quality`: leader score clamped to `[0,1]`;
- `inference_accuracy`: persisted accuracy;
- `evidence_strength`: linear saturation between minimum and saturation counts;
- `neutral_balance`: preferred band, linear decline, or `0.5` for allowed missing data;
- `structural_proximity`: relative-delta or absolute-delta proximity;
- `parameter_preference`: configured family preference;
- `source_rank`: reciprocal rank;
- `horizon_change_penalty`: normalized absolute horizon delta;
- `relative_mutation_penalty`: relative or absolute mutation normalization.

Positive contributions are divided by total positive weight. Penalties use the same denominator. The result is clamped by `score_floor` and `score_ceiling`. [ExperimentRecommendationScoring.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:370>)

Score-run ordering uses:

1. final score descending;
2. raw positive score descending;
3. raw penalty score ascending;
4. source leader score;
5. inference accuracy;
6. evidence count;
7. structural distance;
8. recommendation semantic canonical;
9. recommendation policy canonical;
10. recommendation ID.

[ExperimentRecommendationScoring.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:498>)

### Important separation

Persisted score rows are **not consumed by evaluation**. Evaluation recomputes the score from recommendation evidence using its embedded scoring policy. There is no `recommendation_score_id` relationship between evaluation and `experiment_recommendation_score`.

Consequences:

- Old and new score-table rows do not directly enter evaluation rankings.
- Equivalent old and new **evaluation** rows do enter broad rankings.
- Score listings without `score_run_id` can display heterogeneous score runs together.
- `--recommendation-score-min` applies a numeric threshold across such heterogeneous rows, even though list ordering is by run ID and run-local ordinal rather than score.

### Score persistence weaknesses

`experiment_recommendation_score` duplicates the run policy fields but has no database constraint or trigger requiring them to equal its parent run. The repository also does not load and compare the run policy before insertion. [ExperimentRecommendationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:1212>)

Migration 028 grants the application role update/delete rights on score runs, scores, and components despite describing the history as immutable. [028_experiment_recommendation_scoring.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/028_experiment_recommendation_scoring.sql:135>)

Thus score identity is immutable by normal command convention, not by the current database contract.

---

## 4. Current scoring-policy identity model

Current scoring identity consists of:

- `scoring_version`;
- complete scoring-policy canonical text;
- FNV-1a hash of that canonical text.

The canonical text commits to current configurable weights, thresholds, normalization constants, missing-neutral handling, bounds, and family preferences.

It does not independently spell out:

- component algorithm definitions;
- aggregation formula;
- component names/order;
- source metric semantic definitions;
- missing-value behavior outside the configurable neutral flag;
- a source-metric definition version/hash.

Those behaviors are only implicitly represented by `scoring_version=1` and the `...scoring_policy_v1` canonical prefix. Therefore any formula or source-metric semantic change must bump the algorithm/version contract. The code and database do not enforce that discipline.

Hash integrity is also incomplete:

- Score creation detects same-hash/different-canonical history, but only logs it and continues. [ExperimentRecommendationScoringService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoringService.cpp:137>)
- Evaluation and ranking do not perform equivalent collision checks.
- Database rows do not prove `hash == FNV(canonical)`.
- Existing comparison deliberately treats equal canonical text with different hashes as comparable.

Canonical equality is correctly treated as stronger than hash equality, but inconsistent canonical/hash pairs should still be classified as invalid provenance in Phase 3B.

---

## 5. Current evaluation provenance model

Both:

- `--evaluate-experiment-recommendations`
- `--evaluate-experiment-recommendation`

use `RecommendationEvaluationPolicy`, containing:

- evaluation version;
- evaluator version;
- embedded scoring policy.

The evaluation-policy canonical text length-prefixes the complete scoring-policy canonical text. [ExperimentRecommendationEvaluation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:170>)

Evaluation then:

1. validates recommendation evidence;
2. classifies invalid, unsupported, missing, stale, and duplicate cases;
3. recomputes the score with the embedded scoring policy;
4. persists scores/components only for `advisory_ready`.

[ExperimentRecommendationEvaluation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:335>)

An evaluation run freezes:

- run identity canonical/hash;
- evaluation policy canonical/hash;
- evaluation and evaluator versions;
- scoring policy canonical/hash/version;
- evidence snapshot canonical/hash.

An evaluation result freezes:

- evaluation identity canonical/hash;
- recommendation semantic and recommendation-policy provenance;
- source IDs and evidence;
- disposition and eligibility;
- recomputed score and components;
- ranking ordinal;
- Phase 3A profitability snapshot.

The evaluation identity includes the evaluation-policy canonical text, so it indirectly includes the scoring-policy canonical text. [ExperimentRecommendationEvaluation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:99>)

### Evaluation provenance conclusion

Under the normal service path, every result in one evaluation run uses the run’s single embedded scoring policy.

However:

- The result has no direct scoring-policy fields; they are reconstructed through its mandatory run FK.
- `PersistRecommendationEvaluation` checks recommendation semantic provenance but does not verify the result’s scoring-policy identity against the parent run. [ExperimentRecommendationEvaluationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:462>)
- The schema has no trigger enforcing evaluation-result/run scoring-policy coherence.
- CLI status/explain output shows scoring hash/version but not the authoritative scoring canonical text.
- Evaluation run status data structures do not expose the canonical policy.

So score-policy identity is preserved in the normal workflow but is operationally obscured and not fully defended at repository/database boundaries.

---

## 6. Current ranking architecture and exact populations

Ranking loads persisted evaluation results and their parent evaluation-run policy fields. [ExperimentRecommendationRankingRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingRepository.cpp:44>)

The selection query applies only scope predicates; it has no scoring-policy predicate or grouping. [ExperimentRecommendationRankingRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingRepository.cpp:353>)

| Scope | Exact population | Current homogeneity |
|---|---|---|
| `evaluation_run` | `er.recommendation_evaluation_run_id = requested ID` | Normally homogeneous because one run owns one policy; not DB-enforced |
| `recommendation_scan` | All evaluation results whose copied scan ID matches | Can span many evaluation runs/policies |
| `symbol` | All results whose recommendation source symbol matches | Can span scans, runs, and policies |
| `horizon` | All results whose source horizon matches | Can span scans, runs, and policies |
| `family` | All results whose changed parameter matches | Can span scans, runs, and policies |
| `symbol_horizon` | Intersection of symbol and horizon | Can span scans, runs, and policies |
| `global` | All persisted evaluation results | Unrestricted heterogeneity |

There is no “latest evaluation per recommendation” selection. The same recommendation may appear multiple times through different evaluation runs.

The repository first selects up to 1,001 evaluation identities in deterministic identity order. The ranking layer rejects over 1,000 inputs. Output limit is applied only after complete ranking.

### Buckets and ordering

The ranking policy defines:

1. `advisory_ready`;
2. `blocked`;
3. `non_actionable`.

Within `advisory_ready`:

1. final score descending;
2. recommendation semantic hash;
3. evaluation identity hash;
4. evaluation result ID.

Within `blocked`:

1. pending duplicate;
2. active duplicate;
3. completed duplicate;
4. semantic/evaluation/result identity tie-breaks.

Within `non_actionable`:

1. insufficient evidence;
2. stale evidence;
3. unsupported family;
4. invalid persisted evidence;
5. identity tie-breaks.

[ExperimentRecommendationRanking.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:243>) and [ExperimentRecommendationRanking.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:406>)

Scoring-policy homogeneity matters for all three buckets, not just scored rows: scoring minimum-evidence and missing-value rules can change eligibility, disposition, and bucket membership.

---

## 7. Snapshot and member identity

A ranking snapshot persists:

- ranking policy canonical/hash/version;
- scope canonical/hash and filter fields;
- requested limit;
- source membership canonical/hash;
- counts and lifecycle.

The membership canonical is an ordered set of:

- evaluation identity canonical;
- evaluation result ID.

The snapshot identity combines ranking policy, scope, limit, and membership. [ExperimentRecommendationRanking.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:473>)

Because evaluation identity embeds evaluation policy, and evaluation policy embeds scoring policy, a normal snapshot identity **indirectly records** each member’s scoring policy. But it does not:

- declare one common scoring identity;
- validate that one exists;
- prevent heterogeneous identities;
- make homogeneity visible;
- protect against an evaluation identity inconsistent with its parent run fields.

A ranking member stores the evaluation-result FK, copied score/disposition/source fields, bucket ranks, and tie-break values. Its scoring policy is reconstructable only by joining through evaluation result to evaluation run.

The ranking-member trigger verifies scope, membership, copied data, tie-breaks, and component summaries, but never compares the evaluation run’s scoring policy with a snapshot-level policy. [035_experiment_recommendation_ranking.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/035_experiment_recommendation_ranking.sql:347>)

Downstream campaign planning requires only a completed ranking snapshot and consumes `global_ordinal` directly. It does not validate score-policy homogeneity. [ExperimentRecommendationCampaignPlanningRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:149>)

---

## 8. Core homogeneity answers

| Question | Finding |
|---|---|
| A. Different scoring policies in one snapshot? | **Yes** for every scope except a normally produced single evaluation run. Even evaluation-run scope is not defended against repository/direct-SQL inconsistency. |
| B. Equal numeric scores with different semantics? | **Yes.** Broad scopes compare equal or unequal values without policy validation. |
| C. Old and new score rows mixed? | Persisted score-table rows are not ranking inputs, but score listings and minimum-score filtering can mix them. Old/new evaluation results are mixed by broad ranking scopes. |
| D. Defaults change undetected? | Numeric default changes alter canonical text, but ranking merely records and mixes the changed identities. Formula/source-metric changes without a version bump are undetected. |
| E. Evaluation loses score-policy identity? | Not in the normal FK graph, but it is only run-level/indirect and canonical text is poorly exposed. Persistence does not verify result/run coherence. |
| F. Ranking validates homogeneity? | **No.** |
| G. Snapshot commits to scoring identity? | Indirectly per evaluation identity, but not as a single asserted population identity. This records heterogeneity rather than preventing it. |
| H. Comparison prevents heterogeneous comparison? | Mostly yes: it rejects differing scoring/evaluation canonical text or versions before calculating deltas. It does not reject equal canonical text with inconsistent hashes. |
| I. Can NULL provenance enter? | Current evaluation-run scoring fields are `NOT NULL`. A NULL would fail mapping rather than receive an explicit compatibility classification. Snapshot rows have no scoring identity field at all. |
| J. Legacy handling safe and explicit? | Recommendations distinguish legacy Phase 3A state, but scoring/ranking semantic legacy handling is not explicit. Existing snapshots cannot be declared homogeneous from snapshot columns alone. |

Comparison code correctly uses canonical text rather than trusting hashes. [ExperimentRecommendationRanking.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:564>) However, a heterogeneous snapshot can exist even though a subsequent pairwise member comparison reports `incomparable_policy_version`.

---

## 9. Concrete heterogeneous-ranking failure modes

1. Re-evaluate one scan using policy A.
2. Re-evaluate the same scan using policy B.
3. The evaluation policy changes the evaluation run identity, so both runs persist.
4. Rank by `recommendation_scan`.
5. Both sets are loaded and sorted together.

The same occurs for symbol, horizon, family, symbol+horizon, and global scopes.

A single recommendation can therefore occupy multiple positions in one broad snapshot, potentially with different scores and policy semantics.

A scoring-default change produces the same failure: old evaluation results remain, new results use a different expanded policy canonical, and the broad ranking still accepts both.

A normalization or weight change can reorder members directly. Even when final scores happen to be equal, the deterministic tie-break selects an order without first establishing that equality is meaningful.

Because campaign planning trusts the resulting ordinal, heterogeneity is not merely a display defect; it can affect subsequent recommendation selection.

---

## 10. Adversarial cases

1. **Same numeric score, different policy hash:** currently rankable together. Phase 3B must reject if the canonical policies differ or if either hash is inconsistent.

2. **Same hash, different canonical policy text:** currently rankable; pairwise comparison rejects because canonical text differs. Phase 3B must reject the population. Hash collision must never establish compatibility.

3. **Same weights, different scoring algorithm/version:** currently rankable across evaluation runs. Phase 3B must reject.

4. **Different normalization semantics:** configurable threshold differences change canonical text but are still mixed. Formula changes without a version bump are currently invisible.

5. **Legacy/NULL scoring identity plus current identity:** current schema normally prevents it, but Phase 3B must explicitly classify NULL/invalid provenance as non-comparable and reject rather than crash or infer defaults.

6. **Evaluations created at different times with different scoring policies:** currently mixed by every broad scope.

7. **Global ranking across differently scored scans:** currently allowed without restriction.

8. **Scan ranking after later rescoring/re-evaluation:** persisted score rows themselves are irrelevant, but a later evaluation under the new policy creates a second evaluation run; scan ranking mixes both.

9. **Ranking rerun after defaults change:** the snapshot identity changes indirectly because evaluation membership changes, but the ranking operation still compares old and new semantics.

10. **Comparing heterogeneous ranking members:** pairwise comparison returns incomparable, but only after a snapshot has already ranked them. The command reports the incomparable state rather than calculating score/component deltas.

11. **Profitability available versus unavailable:** scoring input contains no profitability field; evaluation evidence and decision identity deliberately exclude it; ranking does not load it. Score, eligibility, disposition, bucket, order, rank, and snapshot identity remain unchanged.

12. **Materially different profitability values:** same conclusion. Profitability is neither a component nor a tie-break. Phase 3A tests already verify that extreme opposite profitability values cannot reverse evaluation ordering. [ExperimentRecommendationPhase3AProfitabilityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3AProfitabilityTests.cpp:188>)

---

## 11. Legacy compatibility findings

There are three separate legacy concerns:

- Pre-Step-3 recommendations may have nullable recommendation identities. Scoring explicitly skips missing source rank; singular evaluation can encounter less explicit mapping failures.
- Evaluation policy columns are non-null from table creation, but consistency between canonical/hash/version fields is not DB-enforced.
- Existing ranking snapshots have no population scoring identity column.

Existing snapshots must not be assumed homogeneous merely because their persisted top members appear homogeneous. `source_membership_canonical` describes the entire pre-limit population, whereas `experiment_recommendation_ranking_member` contains only the post-limit members.

A future migration should inspect the complete membership population:

- exactly one valid scoring/evaluation semantic identity: classify as verified;
- zero-member snapshot: classify explicitly as empty;
- multiple identities: classify as legacy heterogeneous;
- malformed, incomplete, or unresolvable membership: classify as legacy unverified.

It must not rewrite historical order or silently assign a current default policy. New campaign planning or comparison use should reject legacy heterogeneous/unverified snapshots. Already materialized historical workflows should remain untouched.

---

## 12. Phase 3A preservation findings

Phase 3A is currently correctly separated from scoring and ranking:

- compile-time constants keep profitability weight and contribution at exactly zero. [ExperimentRecommendation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:15>)
- scoring input has no profitability field;
- profitability is excluded from evaluation decision evidence and evaluation identity;
- ranking does not load profitability;
- tie-breaks do not reference profitability;
- evaluation persistence copies the observational snapshot separately;
- available, unavailable, zero-actionable, and legacy states remain distinguishable;
- migration 076 validates exact FINAL inference and observation binding;
- recommendation profitability snapshots are immutable.

The Phase 3B design must leave migration 076, profitability canonicalization, evaluation disposition logic, scoring components, and ranking tie-breaks unchanged.

---

## 13. Required Phase 3B invariant

The preferred invariant should be strengthened to cover unscored members:

> Every evaluation admitted to one ranking population—including advisory-ready, blocked, and non-actionable evaluations—MUST carry the same valid, immutable evaluation semantic identity and scoring semantic identity. Canonical identity equality is authoritative; every accompanying hash and version MUST be internally consistent. The identity MUST describe every behavior capable of changing score, eligibility, disposition, bucket, or numeric comparability.

For the current implementation, scoring semantic identity must commit to:

- scoring algorithm name and version;
- complete scoring-policy canonical text;
- component contract and component names;
- all weights;
- each normalization formula and its parameters;
- positive/penalty aggregation and clamping;
- missing/unavailable handling;
- source leader-score, inference-accuracy, neutral proportion, and evidence-count semantic contract versions/hashes;
- structural-distance derivation;
- parameter-family preference interpretation.

Evaluation semantic identity must additionally commit to:

- evaluation algorithm/version;
- evaluator version;
- embedded scoring semantic identity;
- eligibility/disposition precedence and classification semantics.

Profitability identity must remain excluded because its weight and decision contribution are zero.

---

## 14. Recommended persistence/schema changes

A subsequent migration, likely 077, should narrowly add:

1. A complete scoring-semantic identity canonical/hash/version to score runs and evaluation runs, or formally elevate the current scoring-policy canonical to that role after adding the missing algorithm/source-metric contract.

2. Copied scoring/evaluation semantic identity fields on ranking snapshots:

   - semantic state: `identified`, `empty`, `legacy_heterogeneous`, or `legacy_unverified`;
   - scoring semantic canonical/hash/version;
   - evaluation policy canonical/hash/version;
   - evaluator version.

3. A new ranking snapshot identity version that explicitly includes the common scoring/evaluation semantic identity, rather than depending only on individual membership identities.

4. Optionally, the semantic hash on ranking members for direct observability; canonical authority can remain on the snapshot plus evaluation run.

5. Shape constraints requiring new Phase 3B snapshots to be either:

   - empty with explicit empty state; or
   - non-empty, verified homogeneous, and fully identified.

6. A deterministic audit/classification of historical snapshots using the entire source membership, not only persisted top members.

Do not rewrite historical ranks or alter existing recommendations/evaluations.

---

## 15. Recommended C++ enforcement

### Score creation

- Validate every score result has the same exact scoring semantic canonical/hash/version as its score run.
- Validate policy hash coherence rather than merely logging collisions.
- Make `RankRecommendationScores` reject heterogeneous inputs even if current service construction normally prevents them.
- Harden score history permissions to match its claimed immutability.
- Do not allow cross-policy `minimumScore` filtering without an explicit semantic identity or score-run filter.

### Evaluation creation

- Carry scoring-policy canonical text in `RecommendationEvaluationResult`, not only hash/version.
- Before result insertion, load the parent evaluation run and compare full evaluation/scoring identities.
- Validate canonical/hash coherence on begin, retry, and persistence.
- Keep recomputation behavior unchanged; do not introduce a score-row FK.

### Ranking population

Add a single validation step after loading and before ranking, membership construction, or snapshot creation:

- validate each identity;
- compute distinct exact scoring/evaluation semantic tuples over the full population;
- require exactly one for a non-empty population;
- reject missing, malformed, or inconsistent identities;
- report the offending run/result IDs and identity hashes.

Apply this to dry runs as well.

Do not silently partition. Separate snapshots should only be supported later through an explicit policy-selector or explicit multi-snapshot command.

### Snapshot/member handling

- Build snapshot identity v2 with the common semantic identity.
- Persist that identity on the snapshot.
- Validate every member’s parent evaluation run against the snapshot identity.
- Make campaign planning require `verified_homogeneous` for newly consumed snapshots.

### Comparison

- Validate hash/canonical/version coherence on both sides.
- Require exact scoring semantic canonical equality before score, rank, or component deltas.
- Retain family compatibility checks.
- Return a dedicated `incomparable_scoring_semantics` or `invalid_scoring_provenance` state rather than overloading policy-version mismatch.

---

## 16. Recommended database enforcement

Database enforcement should be defense in depth:

- A score-row insert trigger should require exact equality with its score run’s canonical/hash/version.
- An evaluation-result insert trigger should require identity consistency with its evaluation run.
- A ranking-snapshot insert trigger should verify that the entire declared membership resolves to exactly one scoring/evaluation semantic identity.
- The ranking-member trigger should join through the evaluation run and require exact equality with the snapshot semantic identity.
- New snapshot/member identity fields should be application-role immutable.
- Revoke application-role update/delete rights on score results/components and restrict score-run updates to lifecycle counters/status.
- Add hash-format checks and, if a stable database FNV helper is accepted as authoritative, recompute hashes at insertion. Exact canonical equality remains primary.

The repository should still validate first so failures are clear before persistence.

---

## 17. Observability changes

Status/list/explain output should expose:

- scoring semantic canonical/hash/version;
- evaluation semantic canonical/hash/version;
- population semantic state;
- distinct semantic identity count;
- source evaluation-run IDs;
- homogeneity validation result;
- explicit rejection reason and conflicting hashes/canonicals.

Ranking snapshot status currently shows ranking policy but no common scoring policy. [ExperimentRecommendationRankingService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingService.cpp:55>)

Score and evaluation list output should warn when results span multiple semantic identities. Any score-based filter or ordering across such groups should fail closed.

Profitability output should continue showing:

`profitability_weight=0,profitability_score_contribution=0`

with availability and provenance separately observable.

---

## 18. Required deterministic and adversarial tests

A subsequent implementation should add:

1. Homogeneous members rank successfully in every supported scope.

2. Two evaluation runs with different scoring canonical texts cannot share scan, symbol, horizon, family, symbol+horizon, or global ranking populations.

3. A legacy/NULL/malformed semantic identity fails closed with a precise error.

4. Snapshot identity changes deterministically when scoring semantics change, even if membership IDs and numeric scores are otherwise held constant.

5. Comparison rejects:

   - different canonical/same hash;
   - same canonical/different or invalid hash;
   - different scoring versions;
   - different normalization contracts;
   - different source-metric contracts.

6. Repeated ranking with identical population and semantic identity produces identical membership, order, ranks, and snapshot identity.

7. A scan evaluated again under a later policy is rejected as heterogeneous rather than combined.

8. Ranking after changed compiled defaults rejects old/new population mixing.

9. Direct repository insertion cannot attach a score/evaluation to a parent run with another policy.

10. Database triggers reject a ranking member whose evaluation-run semantic identity differs from its snapshot.

11. Existing snapshot migration classification examines full source membership, including members omitted by output limit.

12. Empty populations receive an explicit deterministic empty semantic state.

13. Available versus unavailable profitability produces identical score, eligibility, disposition, bucket, order, rank, and snapshot identity when pre-Phase-3C inputs are identical.

14. Changing profitability values alone cannot change score, components, ordering, tie-breaks, bucket, rank, or ranking identity.

15. Profitability never appears in scoring semantic identity, ranking policy identity, or tie-break canonical text.

Existing tests already confirm canonical-policy comparison behavior and Phase 3A profitability non-bearing behavior, but there is no ranking homogeneity test. [ExperimentRecommendationRankingTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationRankingTests.cpp:424>)

---

## 19. Narrow implementation plan

1. Define one complete versioned scoring-semantic canonical contract; preserve scoring formula and all weights unchanged.
2. Add pure canonical/hash/coherence and population-homogeneity validators.
3. Strengthen score/evaluation persistence checks against parent runs.
4. Add a migration for snapshot semantic identity, historical classification, triggers, constraints, and score-history privilege hardening.
5. Introduce ranking policy/snapshot identity v2 with explicit common semantics.
6. Reject heterogeneous populations before ranking or snapshot creation.
7. Enforce the same identity in member persistence and campaign planning intake.
8. Extend status/list/explain output.
9. Add targeted pure C++, repository integration, and migration tests.
10. Run the smallest new tests, then existing scoring/evaluation/ranking/Phase 3A suites, then the prescribed Release build.

No scoring formula, recommendation eligibility rule, profitability rule, or ranking order within a homogeneous population needs to change.

## Profitability boundary

Profitability remains exactly weight-zero and non-ranking-bearing in Phase 3B.

It must not affect:

- `recommendation_score`;
- score components;
- eligibility or evaluation disposition;
- ranking bucket;
- ordering or rank;
- tie-breaking;
- ranking population identity;
- campaign selection.

Profitability activation remains exclusively Phase 3C work.

## Final verdict

`READY_FOR_PHASE_3B_IMPLEMENTATION`

Reasons:

- The heterogeneity defect is reproducible directly from current selection and ranking code.
- Existing evaluation-run provenance supplies enough information to enforce an initial fail-closed invariant.
- The required migration can classify legacy snapshots without altering historical decisions.
- No profitability distribution or Phase 3C decision is required.

## Inspection execution record

Files changed: none.

Commands run: read-only `rg`, `nl`, `sed`, `ls`, and Git inspection commands. No executable, scheduler command, database command, test, build, or migration was run.

Actual branch: `lstm-feature-development` at `edbda9c`; this differs from the `phase6` branch noted in `AGENTS.md`.

`git status --short`:

```text
(no output; worktree clean)
```

`git diff --stat`:

```text
(no output)
```