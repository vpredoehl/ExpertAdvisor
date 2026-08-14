---
title: "Return Feature Correction Production Runtime Deployment Scope Review"
document_type: "architecture review"
status: "final"
generated_from: "ReturnFeature_Correction_ProductionRuntime_DeploymentScope_Review_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Return Feature Correction Production Runtime Deployment Scope Review

# Return Feature Correction — Production Runtime Deployment Scope Review

## Executive Decision

Primary recommendation:

    KEEP_CURRENT_H4_RUNTIME

Confidence: HIGH

The correction is not in H4’s runtime execution closure. It changes LSTM training/inference feature construction and a standalone baseline diagnostic, not Campaign Operations readiness, manager, dispatch, handoff, scheduler-control, or startup dispatch behavior.

## 1. Commit 87c8b06 Change Classification

| Changed file | Behavior changed | Runtime category | H4 reachable | Evidence |
|---|---|---|---|---|
| `Headers/LSTM.hpp` | Private helper signatures changed from window-local to Tensor-global positions. | Shared model/feature infrastructure | NO | Declarations only; no H4 call path constructs or invokes these helpers. |
| `Headers/ReturnFeatureHistory.hpp` | New inline causal helper computes return-1/4/8/16 from `currentGlobalPosition`. | Shared model/feature infrastructure | NO | No dynamic initialization; instantiated only from LSTM paths and baseline diagnostic. |
| `LSTM/LSTM.cpp` | `CalculateBatch`, `PredictNextDirectionProbs`, `PredictNextReturn`, and `PredictNextRelativeMove` now use Tensor-global history coordinates. | Training and inference | NO | H4’s Campaign Operations commands do not instantiate `EA::LSTM` or call these methods. |
| `LSTM/main.cpp` | `--baseline-3class` diagnostic now uses Tensor-global return history. | Training/diagnostic only | NO | H4 commands are recognized by `IsExperimentSchedulerCommand` before ordinary LSTM launch parsing; no baseline path runs. |
| `Tests/LSTMFeatureVectorParityTests.cpp/.sh` | Adds parity/no-look-ahead regression coverage. | Test-only | NO | Not linked into production executable behavior. |
| Three review artifacts | Evidence only. | Non-executable | NO | No runtime effect. |

The archived independent review confirms the correction preserves width, feature order, scaling, clamp behavior, schema, Campaign Operations behavior, and H4 state. See [independent reverification findings](/Volumes/Developer%20SSD/ExpertAdvisor/review_artifacts/lstm_feature_development/return_feature_equivalence/LSTM_ReturnFeature_TrainInferenceEquivalence_Correction_IndependentReverification_Findings.md:1).

## 2. H4 Runtime Reachability Analysis

Actual H4 execution closure:

```text
CampaignOperationsH4Supervisor.py
  -> LSTM_Release --campaign-operations-production-readiness
     -> main() -> IsExperimentSchedulerCommand()
     -> RunExperimentSchedulerCli()
     -> RunCampaignOperationsCommand()
     -> CaptureActualManagerBuildContract + RunProductionReadinessCommand
     -> Campaign Operations admission/repository read path

  -> LSTM_Release --campaign-operations-manager-run-once LIMIT --yes
     -> same Scheduler CLI route
     -> RunCampaignOperationsManagerOnceCommand
     -> candidate selection -> production dispatch/readiness validation
     -> Campaign Operations dispatch/handoff services
```

Neither branch calls `CalculateBatch`, `PredictNextDirectionProbs`, `PredictNextReturn`, `PredictNextRelativeMove`, or the baseline helpers. `LSTM/main.cpp` checks scheduler-command ownership before it parses normal train/infer arguments or opens the normal LSTM workflow.

The newly included header has only an `inline constexpr` lookback array and inline templates; it introduces no changed global/static runtime initialization. `LSTM/LSTM.cpp` adds no changed startup initialization.

Manager dispatch can advance Campaign Operations workflows, but its contract does not start a scheduler or model worker. Any later training/inference worker is a separate research-runtime deployment path, not code executed by H4 itself.

## 3. Current Production Runtime Safety

The version-11 H4 runtime can remain deployed unchanged.

- Its authorized identity remains source commit `97aaf723aad11f9b04379bda7eb5a29181f5339e` and SHA `sha256:91c8…45db`.
- The deployed executable continues to run the same Campaign Operations code and preserves the authorization match required by readiness.
- Leaving it installed leaves H4 unaffected by the Return Feature correction.
- Repository HEAD at `87c8b06` is expected to differ from the installed production executable under this model.

The archived H4 steady-state evidence records `ready=true`, `build_comparison=match`, version 11, and unchanged authorization. See [H4 steady-state findings](/Volumes/Developer%20SSD/ExpertAdvisor/review_artifacts/campaign_operations/phaseH/post_deployment/CampaignOperations_PostPhaseH_H4_SteadyState_Observation_Findings.md:37).

## 4. New Release Promotion Consequences

If promoted, the new executable would require the established controlled deployment boundary:

- fresh clean-source Release build with embedded source provenance;
- regeneration and verification of the self-contained runtime stage, including Mach-O closure;
- final staged executable SHA verification;
- controlled replacement of `/opt/expertadvisor/LSTM_Release`;
- H4 JSON `executable_sha256` update;
- durable H4 stop/start handling;
- immutable production disable then enable authorization refresh, with a new approved source commit, SHA, operation keys, and independent-verification reference;
- launchd reactivation and post-start/autonomous-cycle verification.

Installing the new binary without this authorization refresh would cause the build-contract comparison to fail readiness. The H4 configuration also independently checks the configured executable SHA.

That operational churn is not justified now: it provides no H4 runtime benefit while it introduces avoidable authorization, service-continuity, and deployment risk.

## 5. Training / Inference Deployment Scope

The correction should be used independently for research/model runtime work:

- **Future training:** recommend the corrected build; it restores train/inference equivalence.
- **Future standalone and checkpoint inference:** use deliberately and record the feature-pipeline boundary.
- **Resumed experiments:** control separately. A resume under the corrected executable changes the feature values used for continued optimization even though model width/order are unchanged; treat it as a changed scientific lineage or retrain.
- **Already-running experiments:** unaffected until they start a new executable; do not assume replacement changes an in-memory worker.
- **Pre-correction models:** structurally compatible, but corrected inference may produce different inputs/predictions for windows with prior Tensor history. Historical comparisons need explicit old-versus-corrected pipeline labeling.

There is no persisted complete feature-layout/version hash beyond structural compatibility, so scientific reproducibility—not H4 deployment—argues for controlled adoption and clear cohort separation.

## 6. Build Provenance / Authorization Interpretation

Repository HEAD advancement alone does not require H4 reauthorization.

The implemented build contract is formed from the executable’s build-time embedded source commit and the SHA of the explicitly supplied executable. Readiness compares that actual executable identity with the immutable approved-build identity. It does not inspect the current development repository HEAD at H4 runtime.

Therefore, authorization follows the exact installed executable, not development-tree advancement.

## 7. Risks / Caveats

- The current production identity/readiness conclusion relies on the supplied deployment facts and archived same-day evidence; this review intentionally performed no live H4 probe.
- Corrected feature semantics require research-version discipline for resumes and comparisons of pre-correction models.
- A future H4 replacement must be a fresh, clean, self-contained Release artifact; an older staged binary must not be reused.

No files were changed.

Tests/build commands run: none; this was a source-and-artifact reachability review. The provided archive records independent correction verification and successful Release-build closure.

`git status --short`: clean

`git diff --stat`: empty

RETURN_FEATURE_H4_DEPLOYMENT_SCOPE=KEEP_CURRENT_H4_RUNTIME

H4_PRODUCTION_REPLACEMENT_REQUIRED=NO

RETURN_FEATURE_DEVELOPMENT_BUILD_USE=RECOMMEND_NEW_TRAINING_AND_CONTROLLED_VERSIONED_INFERENCE_ONLY

REVIEW_DISPOSITION=PASS