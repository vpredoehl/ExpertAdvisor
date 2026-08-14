---
title: "LSTM Return Feature Research Runtime Boundary and Experiment Inventory Review"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ReturnFeature_ResearchRuntimeBoundary_ExperimentInventory_Review_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Return Feature Research Runtime Boundary and Experiment Inventory Review

# 1. Executive Decision

Commit `87c8b06` establishes a real scientific feature-semantics boundary: return-1/4/8/16 now use Tensor-global causal history in both training and inference.

Treat all existing experiments and models as pre-boundary. Do not mix them with post-boundary training, resumes, or inference on a single leaderboard.

No repository or database state was changed.

# 2. Semantic Boundary Definition

Canonical label: `RETURN_HISTORY_TENSOR_GLOBAL_V1`.

Unchanged:

- Model widths: legacy 36 and current 38.
- Feature order: return-1, return-4, return-8, return-16.
- Return scale (`kFeatScale`), invalid/startup zero behavior, and final `[-10, 10]` clamp.
- Model/checkpoint structural compatibility.

Changed:

| Path | Pre-87c8b06 | Post-87c8b06 |
|---|---|---|
| Training | Return history indexed relative to enclosing batch | Indexed by Tensor-global observation position |
| Standalone inference | Indexed relative to local inference window | Indexed by Tensor-global observation position |
| Checkpoint inference | Same local-window inference behavior | Same global behavior |
| Resume | Could continue legacy checkpoint with changed code undetected | Would change feature values mid-lineage if resuming a legacy checkpoint |
| Historical comparison | Training/inference mismatch possible | New results are semantically distinct from legacy results |

The shared helper is causal: it uses only `currentGlobalPosition` and `currentGlobalPosition - lookback`.

# 3. Current Runtime / Scheduler Identity

Snapshot: 2026-08-14 14:47 CDT.

| Process | PID | Started | Mapped image conclusion |
|---|---:|---|---|
| Scheduler | 42864 | Aug 14 07:29 | Pre-boundary image |
| Training experiment 549 | 68338 | Aug 13 01:06 | Pre-boundary image |
| Training experiment 554 | 32973 | Aug 14 00:46 | Pre-boundary image |

`lsof` shows all three processes retain separately mapped temporary build images, not the current DerivedData product. Their mapped image sizes/inodes differ from the current artifact, so rebuilding/replacing the on-disk executable did not change these running processes.

Current launch target recorded by the scheduler:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
```

Current on-disk target:

```text
SHA-256: 243679a1eb8e2a862011577bf54961f2756d343c9a05ce203d03ae72b7961ade
mtime:   2026-08-14 14:03:24 -0500
embedded source commit: 87c8b06090bd037c484721e82794908ba10f6d91
```

Therefore, at the observation point, a newly dispatched worker would execute post-correction semantics. The still-running scheduler is pre-correction, but its `execv` child target is the current post-correction DerivedData artifact.

# 4. Experiment Inventory

Database totals:

| Category | Count |
|---|---:|
| Experiments | 331 |
| Completed with final model | 306 |
| Cancelled | 14 |
| Running training | 2 |
| Running inference | 0 |
| Paused training | 7 |
| Failed training | 1 |
| Failed inference | 1 |
| Pending experiments | 0 |
| Completed checkpoint evaluations | 448 |
| Failed checkpoint evaluations | 37 |
| Pending/running checkpoint evaluations | 0 |

Live/nonterminal inventory:

| ID | Symbol | Horizon | State | Epoch / target | Resume model | Worker | Provenance |
|---:|---|---:|---|---|---:|---:|---|
| 544 | gbpchfrmp | 8 | paused/train | 7 / 80 | — | — | `fec53d…` pre |
| 545 | gbpjpyrmp | 8 | paused/train | 4 / 80 | — | — | `fec53d…` pre |
| 546 | gbpcadrmp | 12 | failed/train | — / 80 | 1562 | — | `1cca34…` pre |
| 547 | eurjpyrmp | 12 | paused/train | — / 80 | 1560 | — | `1cca34…` pre |
| 548 | chfjpyrmp | 12 | paused/train | — / 80 | 1563 | — | `1cca34…` pre |
| 549 | eurchfrmp | 12 | running/train | 61 / 80 | — | 68338 | `1cca34…` pre |
| 550 | nzdcadrmp | 12 | paused/train | — / 80 | 1571 | — | `1cca34…` pre |
| 551 | cadchfrmp | 4 | failed/infer | 57 / 80 | — | — | `126265…` pre |
| 552 | cadchfrmp | 4 | paused/train | — / 80 | — | — | no captured provenance; created pre-boundary |
| 553 | cadchfrmp | 4 | paused/train | — / 80 | — | — | `2817bc…` pre |
| 554 | cadchfrmp | 4 | running/train | 79 / 80 | 1601 | 32973 | `2817bc…` pre |

Additional legacy lineage indicators:

- 98 terminal rows retain a `resume_model_id`.
- 21 rows retain a stopped checkpoint model.
- 82 rows are continuation descendants.
- All 306 completed final models are pre-boundary candidates for future inference.
- No experiment row records post-boundary commit `87c8b06`.

Checkpoint policy exposure:

- 549 has completed checkpoint inference at epochs 20/40/60 for models 1602/1603/1604.
- 554 has completed checkpoint inference at epoch 60 for model 1605.
- Both active training rows have checkpoint inference enabled and opportunistic checkpoint inference enabled.
- At epoch 79, experiment 554 is immediately exposed to a post-boundary checkpoint-inference dispatch when it reaches epoch 80.

# 5. Cohort Classification

| Cohort | Current membership |
|---|---|
| `PRE_BOUNDARY_COMPLETED` | 306 completed final-model experiments |
| `PRE_BOUNDARY_ACTIVE` | 549, 554 |
| `PRE_BOUNDARY_PAUSED_OR_FAILED` | 544–548, 550–553 |
| `PRE_BOUNDARY_RESUMABLE_HISTORY` | 98 resume-pointer rows; 21 stopped-checkpoint rows; 82 continuation descendants |
| `PRE_BOUNDARY_PENDING` | None at snapshot |
| `PRE_BOUNDARY_CHECKPOINT_INFERENCE` | 448 completed, 37 failed, none pending/running |
| `POST_BOUNDARY_NEW` | None |
| `CROSS_BOUNDARY_RESUME` | None confirmed; would result from resuming legacy checkpoints under the current artifact |
| `CROSS_BOUNDARY_INFERENCE` | None confirmed; would result from dispatching legacy-model inference/checkpoint inference now |

# 6. Per-Cohort Disposition

| Cohort | Disposition |
|---|---|
| `PRE_BOUNDARY_ACTIVE` | `ALLOW_TO_FINISH_PRE_BOUNDARY`, but hold follow-on checkpoint inference |
| `PRE_BOUNDARY_PENDING` | `HOLD_PENDING_FOR_REQUEUE`; none currently exist |
| `PRE_BOUNDARY_PAUSED_OR_FAILED` | `RETRAIN_FROM_EPOCH_0_POST_BOUNDARY` for canonical research |
| Legacy resume exception | `ALLOW_RESUME_ONLY_AS_NEW_SCIENTIFIC_LINEAGE` |
| `PRE_BOUNDARY_COMPLETED` | `NO_ACTION_REQUIRED`; retain immutable historical cohort |
| Pre-boundary model inference | `INFER_PRE_MODEL_WITH_PRE_RUNTIME_ONLY` for lineage-preserving results |
| Corrected inference of pre-boundary model | `ALLOW_CORRECTED_INFERENCE_WITH_CROSS_BOUNDARY_LABEL` only |
| Pending legacy checkpoint inference | `HOLD_PENDING_FOR_REVIEW`; do not silently run under corrected runtime |
| `POST_BOUNDARY_NEW` | Allow only after an explicit research-runtime boundary/freeze is recorded |

# 7. Resume / Checkpoint Policy

Do not continue a pre-boundary checkpoint as though it were the same experiment under `RETURN_HISTORY_TENSOR_GLOBAL_V1`.

A resume restores weights, optimizer state, architectural width, target settings, range, and Donchian mode, but does not restore or validate return-feature semantic identity. Continuing a pre-boundary checkpoint under the corrected executable would change the input vectors used for all subsequent optimization.

Primary recommendation: retrain from epoch 0 for canonical post-boundary results.

If continuation is operationally necessary, preserve the parent checkpoint but create a separately labeled cross-boundary lineage. It must not replace the legacy result or enter the same leaderboard.

# 8. Inference Policy for Pre-Correction Models

Post-correction inference of a pre-correction model is structurally possible: model width/order and checkpoint matrices still load. It is not scientifically equivalent, because the four input values can differ whenever prior Tensor history exists.

Policy:

- Preserve existing legacy inference as historical evidence.
- Do not replace legacy inference results with corrected-runtime results.
- Prefer pre-runtime inference for legacy model lineage, if that exact runtime can be reproduced.
- If corrected inference is intentionally performed, store it as `CROSS_BOUNDARY_INFERENCE` with both model lineage and runtime semantics explicitly identified.
- Do not dispatch a pending legacy checkpoint-inference action through the current executable without this designation.

# 9. Scientific Comparability / Leaderboard Policy

The clean comparison boundary is feature semantics, not tensor width, model ID, branch, or scheduler phase.

Only compare/rank together:

```text
RETURN_HISTORY_TENSOR_GLOBAL_V1 training
+ RETURN_HISTORY_TENSOR_GLOBAL_V1 inference
```

Legacy training/inference and cross-boundary inference/resume must be separate cohorts. A structurally compatible 38-wide model is not semantically comparable merely because its parameter shapes load.

# 10. Persistence / Provenance Gap

**Gap: YES.**

Persisted experiment metadata includes `git_commit`, branch, dirty state, build configuration, worker executable path, PID/PGID, command line, and process-start identity. Checkpoints persist structural input width and selected train/runtime configuration.

It does not persist:

- a feature semantics/pipeline version;
- a feature-vector contract hash;
- executable SHA-256 for a research worker;
- an immutable, embedded source-commit identity for the actual research executable.

Additionally, `RunMetadata::Capture` obtains `git_commit` from `git rev-parse HEAD` at runtime. That records checkout state, not necessarily the binary’s build-time source identity.

Minimal future improvement ranking:

1. `feature_semantics_version` — recommended; explicitly identifies this scientific boundary.
2. `research_runtime_executable_sha256` — strongest binary reproducibility companion.
3. `research_runtime_source_commit` — useful only if embedded/build-time captured.
4. `feature_vector_contract_hash` — most comprehensive but requires a broader contract design.

Recommendation: add immutable `feature_semantics_version`, captured for experiment creation, model save, resume, inference result, and checkpoint inference. Pair it later with executable SHA-256.

# 11. Recommended Staged Adoption Plan

1. Preserve this inventory and label all existing rows/models as legacy by observation-time cohort.
2. Establish a research dispatch freeze point before another child process launches.
3. Let 549 and 554 finish only as pre-boundary work; prevent their next checkpoint-inference action from silently crossing.
4. Hold any newly appearing pending work until it is recreated/labeled for post-boundary semantics.
5. Do not resume legacy checkpoints for canonical results; retrain from epoch 0.
6. Promote the corrected artifact only for research workers, independently of H4.
7. Begin a new post-boundary experiment cohort using `RETURN_HISTORY_TENSOR_GLOBAL_V1`.
8. Preserve legacy inference; allow corrected legacy-model inference only as an explicitly cross-boundary comparison cohort.
9. Segregate leaderboard/report outputs by semantic cohort.
10. Later implement the minimal persistence improvement through the established schema/workflow path.

# 12. Exact Next Operational Boundary

Before experiment 554 reaches its epoch-80 checkpoint and the scheduler can dispatch its next inference worker:

```text
FREEZE_RESEARCH_DISPATCH_AND_CHECKPOINT_INFERENCE;
RECORD_RETURN_HISTORY_TENSOR_GLOBAL_V1_AS_THE_NEW_COHORT_BOUNDARY;
ALLOW_ONLY_ALREADY_MAPPED_PRE_BOUNDARY_TRAINING_TO_FINISH.
```

This is a recommendation only; no scheduler, queue, worker, or database state was altered.

# 13. Risks / Caveats

- The active scheduler itself is pre-boundary, while its child launch path is now post-boundary. This is the immediate cross-boundary risk.
- Mapped temporary build images are no longer available at their original staging paths, so they cannot now be SHA-256 hashed; process start times, mapped-image identity, and row metadata establish their pre-boundary status.
- Existing `git_commit` provenance is informative but not authoritative executable provenance.
- Tensor-global return history fixes the reviewed return coordinate defect only. The separate query-start prehistory limitation for EMA/ATR/rolling/Donchian remains unchanged.
- H4 Campaign Operations is outside this research-runtime adoption decision.

# 14. Final Decision Markers

```text
RETURN_FEATURE_RESEARCH_RUNTIME_BOUNDARY=RETURN_HISTORY_TENSOR_GLOBAL_V1_AT_87c8b06
ACTIVE_PRE_BOUNDARY_WORKERS=ALLOW_TO_FINISH
PENDING_PRE_BOUNDARY_EXPERIMENTS=NONE_CURRENTLY_HOLD_AND_RECREATE_IF_PRESENT
PRE_BOUNDARY_RESUME_POLICY=RETRAIN_FROM_EPOCH_0_POST_BOUNDARY
PRE_BOUNDARY_MODEL_INFERENCE_POLICY=PRE_RUNTIME_ONLY_OR_EXPLICIT_CROSS_BOUNDARY_LABEL
CHECKPOINT_INFERENCE_CROSS_BOUNDARY_POLICY=HOLD_PENDING_PRE_BOUNDARY_CHECKPOINT_INFERENCE
LEADERBOARD_COMPARABILITY_POLICY=SEGREGATE_BY_FEATURE_SEMANTICS_VERSION
FEATURE_PIPELINE_PROVENANCE_PERSISTENCE_GAP=YES
H4_PRODUCTION_RUNTIME_ACTION=KEEP_CURRENT
NEXT_OPERATIONAL_BOUNDARY=FREEZE_RESEARCH_DISPATCH_BEFORE_NEXT_POST_BOUNDARY_CHILD_LAUNCH
REVIEW_DISPOSITION=PASS_WITH_ACTIONS
```

Files changed: none.

Tests/build commands run: none; this was read-only.

`git status --short`:

```text
?? ReturnFeature_Correction_ProductionRuntime_DeploymentScope_Review_Output.md
```

`git diff --stat`: empty.