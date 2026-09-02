# Independent Verification — Campaign Manager Phase 3B Scoring-Policy Ranking Homogeneity

## Verdict

**CHANGES REQUIRED BEFORE COMMIT**

The Phase 3B implementation is directionally strong and satisfies the main application-level invariant, but independent verification found one material fail-closed gap in migration 077:

> The SQL-side scoring-policy provenance validator checks textual shape, version, and FNV-1a hash, but it does **not** enforce the same semantic validity and exact canonical reconstruction rules as the C++ validator.

Because migration 077 uses that SQL validator to establish evaluation-run validity, historical ranking classification, new snapshot verification, and member/snapshot coherence, a direct SQL caller can potentially create a syntactically shaped and correctly hashed scoring policy that the C++ application would reject, yet the database can treat it as valid Phase 3B semantic provenance.

This violates the requested Phase 3B boundary that malformed or internally inconsistent semantic provenance must fail closed.

## Package integrity

Archive reviewed:

`LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Verification.tar.gz`

Observed archive SHA-256:

`6a84f7e874060a588a220962e65265093f5cf835c8130a8f2aedfde1d5650812`

All packaged file hashes in `SHA256SUMS.txt` verified successfully.

The package contains the intended production sources, migration 077, migration test, modified ranking/evaluation/campaign-planning tests, and Phase 3B documentation.

`GIT_DIFF_CHECK.txt` is empty, consistent with `git diff --check` passing.

Base commit recorded by the package:

`edbda9cba2906b11c4535f22c58528f7728f19b0`

Branch:

`lstm-feature-development`

## What verified successfully

### 1. Full pre-limit population validation — PASS

`RunRankExperimentRecommendationEvaluationsCommand()` loads the ranking population and calls `ValidateRecommendationRankingPopulationSemantics(evaluations)` **before**:

- `RankRecommendationEvaluationEvidence(...)`
- ranking limit application
- membership canonical construction
- ranking snapshot creation/persistence.

The repository loader selects up to `kMaximumRecommendationRankingInputs + 1`; `MapEvaluations()` throws once the population exceeds the supported maximum. Thus a population larger than the supported ranking bound fails rather than being silently truncated into a supposedly complete ranking population.

### 2. All ranking scopes pass through the same validation — PASS

The common ranking command/service path protects:

- evaluation_run
- recommendation_scan
- symbol
- horizon
- family
- symbol_horizon
- global

No scope-specific bypass was found in the packaged implementation.

### 3. Heterogeneous numeric scores fail closed — PASS

Pure tests explicitly construct equal numeric `finalScore` values under different scoring semantics and expect:

`heterogeneous_scoring_semantics`

The ranking operation rejects before ranking.

### 4. Canonical/hash mismatch detection in C++ — PASS

The C++ implementation correctly treats canonical text as authoritative.

`ValidateRecommendationScoringPolicyProvenance()`:

1. parses the persisted canonical policy;
2. validates the policy through the existing scoring-policy parser;
3. reconstructs canonical text;
4. requires exact canonical equality;
5. requires persisted scoring version to match;
6. requires the stored hash to equal the hash of the authoritative canonical text.

The C++ ranking validator also rejects inconsistent scoring and evaluation semantic hashes/canonicals.

### 5. Snapshot identity v2 — PASS

New snapshot identity commits to:

- ranking policy
- ranking scope
- requested limit
- population semantic state
- common scoring semantic canonical
- common evaluation semantic canonical
- complete pre-limit membership canonical.

Changing semantic meaning while keeping member IDs/numeric scores fixed therefore changes or invalidates the snapshot identity.

### 6. New snapshot DB enforcement — PASS, subject to Finding 1

The migration trigger requires new snapshots to be identity version 2 and only admits:

- `verified_homogeneous`
- `empty`

For non-empty verified populations it resolves the complete membership against persisted evaluation results/runs and verifies one common scoring/evaluation semantic canonical before accepting the snapshot.

A heterogeneous historical membership cannot be relabeled as a new verified homogeneous snapshot under the tested path.

### 7. Ranking member ↔ snapshot binding — PASS, subject to Finding 1

The member trigger resolves each inserted ranking member through its evaluation result/run and requires its derived scoring/evaluation semantics to equal the snapshot semantics.

A mismatched member raises SQLSTATE `23514`.

### 8. Historical classification uses full membership — PASS

Migration 077 classifies old snapshots from `source_membership_canonical`, not from persisted top-N ranking members.

It distinguishes:

- `verified_homogeneous`
- `empty`
- `legacy_heterogeneous`
- `legacy_unverified`

Historical rank/order/member rows are not rewritten.

### 9. Campaign-planning trust boundary — PASS

Campaign planning rejects ranking snapshots whose population semantic state is neither:

- `verified_homogeneous`
- `empty`

For verified homogeneous snapshots it checks canonical/hash identity shape and requires distinct scoring/evaluation semantic counts to be exactly one.

### 10. Profitability isolation / Phase 3C boundary — PASS

No profitability field was found in the new scoring/evaluation semantic identity contracts.

Evaluation observability still reports:

- `profitability_weight=0`
- `profitability_score_contribution=0`

No Phase 3C gate, threshold, normalization, ranking component, tie-break, or campaign-selection behavior was introduced in the packaged Phase 3B changes.

---

# Material Finding 1 — SQL scoring-policy provenance validation is weaker than C++

**Severity: High for the Phase 3B fail-closed trust boundary**

## C++ behavior

The C++ function `ParseCanonicalScoringPolicy()` does the correct thing:

- requires the `experiment_recommendation_scoring_policy_v1;` prefix;
- parses all assignments using `ParseRecommendationScoringPolicy(...)`;
- therefore executes the full `ValidateRecommendationScoringPolicy(...)` rules;
- reconstructs canonical text using `RecommendationScoringPolicyCanonicalText(policy)`;
- rejects unless the reconstructed canonical text exactly equals the persisted canonical.

Examples of policy states the C++ path rejects include:

- negative scoring weights;
- zero/nonpositive `minimum_evidence_count`;
- evidence saturation below the minimum;
- preferred/maximum neutral proportions outside their legal relationship/ranges;
- nonpositive mutation/structural maxima;
- invalid score bounds;
- parameter preferences outside `[0,1]`;
- noncanonical alternate numeric spellings that do not reconstruct exactly.

## SQL behavior

Migration 077 defines:

`recommendation_scoring_policy_provenance_valid_v1(...)`

Its current validation is essentially:

- `policy_version = 1`;
- one large regular expression matching the expected field order/text shape;
- `policy_hash = recommendation_semantic_tagged_fnv1a64(policy_canonical)`.

The regular expression **does not enforce the C++ semantic constraints**.

It even intentionally permits forms that can be semantically invalid, for example:

- negative numeric fields due to `-?`;
- `minimum_evidence_count=0`;
- arbitrary score floor/ceiling relationships;
- preferences outside `[0,1]`;
- structurally valid but noncanonical numeric spellings such as forms the C++ canonicalizer would normalize differently.

Therefore a caller able to insert directly into evaluation-run persistence can construct:

1. a policy string satisfying the SQL regex;
2. a matching FNV-1a hash;
3. version 1;
4. an evaluation policy canonical embedding that policy;
5. matching hashes.

The SQL trigger can accept this as valid evaluation-run semantic provenance even though the C++ application would reject the same scoring policy.

## Why this matters beyond the evaluation-run row

This validator feeds:

- `recommendation_evaluation_run_semantics_valid_v1(...)`
- `recommendation_evaluation_result_semantics_valid_v1(...)`
- migration-time historical snapshot classification
- new Phase 3B ranking snapshot validation
- ranking member semantic enforcement.

Thus the database can potentially derive a scoring/evaluation semantic identity from policy provenance that is not actually a valid canonical scoring policy according to the authoritative application contract.

That weakens the requested invariant:

> heterogeneous, missing, malformed, or internally inconsistent scoring/evaluation semantics must fail closed.

## Required correction

Migration 077 needs a SQL-side policy validator that enforces the same accepted canonical domain as C++, or an equally strong database representation that makes invalid policy text impossible to treat as verified.

At minimum, the correction should prove all of the following:

1. policy canonical is not merely regex-shaped;
2. every numeric/domain constraint from `ValidateRecommendationScoringPolicy()` is enforced;
3. field ordering and exact canonical number spelling are enforced;
4. the persisted version equals the canonical embedded version;
5. the hash equals the authoritative canonical;
6. the evaluation-policy canonical exactly embeds that already-validated scoring canonical.

A preferable implementation is a narrowly defined SQL parser/validator for v1 canonical policy text rather than trying to grow the already-large regex further without tests.

## Required adversarial migration tests

Add migration tests that construct a correctly hashed but C++-invalid policy canonical and prove direct SQL insertion fails.

At minimum test:

- `leader_score_weight=-0.25`;
- `minimum_evidence_count=0`;
- `evidence_saturation_count < minimum_evidence_count`;
- `score_floor >= score_ceiling`;
- a parameter preference greater than 1;
- a noncanonical-but-regex-shaped numeric representation if the canonical C++ formatter would normalize it differently.

Also prove such a row cannot:

- be used as an evaluation run;
- produce an evaluation result accepted by the semantic trigger;
- cause a historical snapshot to be classified `verified_homogeneous`;
- produce a new `verified_homogeneous` snapshot.

---

# Verification Gap 2 — evaluation-result subordinate hash coherence should be adversarially tested

**Severity: Medium verification gap; not confirmed as a production defect from this package alone**

The new `recommendation_evaluation_result_semantics_valid_v1(...)` function validates:

- parent evaluation-run semantic validity;
- result/run ID relationship;
- exact evaluation identity canonical reconstruction;
- evaluation identity hash.

However, in migration 077 itself it does not visibly verify every subordinate persisted hash, such as:

- `evidence_hash` against `evidence_canonical`;
- recommendation semantic hash against recommendation semantic canonical;
- recommendation policy hash against recommendation policy canonical.

The C++ repository path does validate at least the evaluation/evidence identity before persistence, and pre-existing migration 034 may already defend some of these fields. The verification package did not include migration 034 itself, so this review cannot conclusively call this a production defect.

The Phase 3B migration test currently proves rejection of a deliberately wrong evaluation identity, but it does not independently prove rejection of a correctly reconstructed evaluation identity accompanied by an incorrect subordinate evidence hash.

Before commit, add or confirm an adversarial database test for these subordinate canonical/hash pairs. If migration 034 already enforces them, the new test should document that dependency. If it does not, migration 077 should close the gap.

---

# Other observations

## Application-level implementation quality

The C++ implementation is substantially stronger than the SQL gap above. The application path:

- reconstructs scoring semantics from validated canonical policy provenance;
- reconstructs evaluation semantics;
- validates every ranking input;
- computes distinct semantic identities over the full population;
- rejects malformed or heterogeneous populations before sorting;
- includes semantic state in snapshot identity;
- uses dedicated comparison incompatibility states.

No application-level homogeneity bypass was found in the packaged files.

## Empty population handling

The implementation preserves an explicit `empty` semantic state with zero distinct identities and no fabricated scoring/evaluation semantic canonical. Campaign planning has a corresponding shape check.

This is consistent and fail-closed.

## Score-history hardening

General score-run/result/component immutability was intentionally deferred. Nothing in this package showed ranking directly trusting historical score-table rows for its semantic identity, so the deferral is acceptable for the narrow Phase 3B invariant.

## Release state

The package records Debug success and Release pending because the worktree is intentionally uncommitted. Independent verification does not recommend committing yet due to Finding 1.

---

# Final disposition

| Area | Result |
|---|---|
| Package integrity | PASS |
| Full pre-limit validation | PASS |
| All ranking scopes | PASS |
| Heterogeneous score semantics rejection | PASS |
| C++ canonical/hash/version validation | PASS |
| Snapshot identity v2 | PASS |
| Historical full-membership classification | PASS subject to SQL validator correction |
| Member ↔ snapshot DB enforcement | PASS subject to SQL validator correction |
| Campaign-planning fail-closed boundary | PASS |
| Profitability remains weight-zero | PASS |
| Phase 3C excluded | PASS |
| SQL scoring-policy semantic validity parity | **FAIL — correction required** |
| Subordinate result hash DB tests | VERIFY/HARDEN |
| Ready to commit | **NO** |

## Recommended next action

Do **one narrow correction prompt**, not a redesign.

The correction should:

1. make migration 077's SQL scoring-policy provenance validation equivalent in accepted domain to the C++ canonical policy validator;
2. add adversarial migration tests for correctly hashed but semantically invalid/noncanonical scoring policies;
3. explicitly verify subordinate evaluation-result canonical/hash coherence at the DB boundary;
4. change no ranking order, score formula, profitability behavior, Phase 3A behavior, or Phase 3C behavior.

After that correction, repackage migration 077, the migration test, and any production/test files changed by the correction for independent reverification.
