---
title: "LSTM Profitability Phase 2A Final Inference Provenance Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2A_FinalInferenceProvenance_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2A Final Inference Provenance Targeted Correction

Implemented the targeted Phase 2A provenance correction. Profitability remains diagnostic-only.

### Findings and correction

1. **Defect confirmed.** FINAL continuation evidence selected the newest completed final inference row by model ID.

2. **Root cause.** `experiment_analysis_result` does not retain the exact `inference_eval_result.id`; the loader used recency instead of reconstructing scheduler persistence identity.

3. **Authoritative FINAL identity now requires:**

   - source experiment ID;
   - `experiment.last_model_id = final model`;
   - model ownership by that experiment;
   - source symbol and inference date range;
   - persisted model horizon, threshold, window size, label rule, target type, and completed epoch;
   - `status='completed'`;
   - `inference_scope='final'`;
   - null `checkpoint_eval_id`;
   - null `parent_experiment_id`.

   The inference row must match exactly and uniquely. No recency fallback remains.

4. **Unavailable behavior:**

   - zero matches: `no_exact_final_inference_result`;
   - multiple matches: `ambiguous_final_inference_result`;
   - inconsistent source/model configuration: `final_inference_context_mismatch`;
   - observation-level missing/mismatch reasons remain unchanged.

   These affect only profitability availability. Missing evidence is not zero and remains non-blocking.

5. **Checkpoint behavior unchanged.** It already binds exact checkpoint evaluation ID, checkpoint model, checkpoint scope, and completed result.

6. **Migration:** None required.

### Files touched by this correction

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [InferenceProfitabilityRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp>)
- [InferenceProfitabilityRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.hpp>)
- [InferenceProfitabilityRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/InferenceProfitabilityRepositoryTests.cpp>)
- [InferenceProfitabilityRepositoryTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/InferenceProfitabilityRepositoryTests.sh>)
- [ContinuationPolicyInheritanceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ContinuationPolicyInheritanceTests.cpp>)
- [ContinuationProfitabilityPolicyIsolationTests.py](</Volumes/Developer SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyIsolationTests.py>)
- [InferenceProfitabilityPersistence.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/InferenceProfitabilityPersistence.rst>)

### Behavior-neutral proof

Tests confirm that changing or removing the corrected FINAL diagnostic inference identity does not change:

- source selection or ranking;
- evidence count or watermark;
- deduplication or trend results;
- continuation policy evaluation or queue behavior;
- recommendation scoring;
- checkpoint policy behavior.

No Campaign Manager, recommendation, checkpoint-policy, or training implementation was changed.

### Verification

Passed:

- `./Tests/InferenceProfitabilityTests.sh`
- `./Tests/InferenceProfitabilityRepositoryTests.sh`
- `ContinuationPolicyInheritanceTests`
- `ExperimentRecommendationScoringTests`
- `python3 Tests/ContinuationProfitabilityPolicyIsolationTests.py`
- `git diff --check`
- Full Debug build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor \
  ENABLE_USER_SCRIPT_SANDBOXING=NO build

** BUILD SUCCEEDED **
```

The unmodified Debug command initially encountered the existing Xcode user-script sandbox/temp-file issue before compilation.

Release was not run: Release provenance requires a clean committed tree, while this worktree intentionally contains uncommitted Phase 2A changes. Production scheduler and training workers were also active, so scheduler integration executables were not launched.

### Before Phase 2B

- Apply migration 073 operationally; the inspected database reports it absent.
- Obtain independent verification of this corrected worktree.
- Run the canonical Release build from a clean committed tree.
- Exercise read-only continuation diagnostics when production workers permit safe integration testing.

Current branch remains `lstm-feature-development`, not the `phase6` branch named in `AGENTS.md`.

`git status --short`:

```text
 M Sources/ContinuationPolicy.cpp
 M Sources/ContinuationPolicy.hpp
 M Sources/ExperimentScheduler.cpp
 M Sources/InferenceProfitabilityRepository.cpp
 M Sources/InferenceProfitabilityRepository.hpp
 M Tests/ContinuationPolicyInheritanceTests.cpp
 M Tests/InferenceProfitabilityRepositoryTests.cpp
 M Tests/InferenceProfitabilityRepositoryTests.sh
 M docs/InferenceProfitabilityPersistence.rst
?? LSTM_Profitability_Phase2A_ContinuationEvidence_Observability_Implementation_Output.md
?? LSTM_Profitability_Phase2A_Independent_Verification_Findings.md
?? LSTM_Profitability_Phase2A_Verification_Package.tar.gz
?? Tests/ContinuationProfitabilityPolicyIsolationTests.py
```

`git diff --stat`:

```text
9 files changed, 1357 insertions(+), 234 deletions(-)
```

The stat includes pre-existing Phase 2A worktree changes and excludes untracked files.