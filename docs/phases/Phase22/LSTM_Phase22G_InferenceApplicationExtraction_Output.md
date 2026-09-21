---
title: "LSTM Phase 22G — InferenceApplication Extraction"
document_type: "implementation / validation report"
status: "no-go"
---

# LSTM Phase 22G — InferenceApplication Extraction

## Baseline and disposition

- Branch: `lstm-feature-development`
- Baseline commit: `84f8c0fdbc659a019d844ee200fbafc8360ac4cb` (`Document Phase 22F inference composition audit`)
- Pre-flight `git status --short`: empty.
- Baseline history recorded: `84f8c0f Document Phase 22F inference composition audit`; `97770aa Add role-aware semantic worker artifacts`; `2760c82 Route scheduler analysis to standalone worker`; `3b883e3 Add standalone analyze worker`; `5352225 Document Phase 22A worker executable audit`.

**Phase 22G InferenceApplication extraction: NO-GO**

No production source, Xcode target, registry, publisher, artifact, scheduler, or database change was made. This report is the only worktree change.

## Direct-source finding

The Phase 22F audit was read first and then verified against the current `LSTM/main.cpp`. Its proposed seam cannot be extracted in this increment without violating a stated stop condition.

The executable's live inference branch is not self-contained at the listed evaluation and persistence functions. In particular, `main.cpp:8691-9197` is a shared, monolithic setup path that:

- resolves persisted inference configuration and applies it to the process globals;
- resolves scheduler final/checkpoint persistence context;
- chooses symbol, Donchian mode/lookback, warmup, and economic snapshot;
- builds the market/event tensor and logical output indexes; and
- constructs and loads the LSTM.

That block is entered before the `gRuntimeInferenceMode` branch at `9198` and its same initialized model/tensor state continues directly into the training path at `9438` and after. It therefore is neither inference-only composition nor a lower-level reusable component with an existing interface.

Moving the later inference-only functions (`RunInferenceEvaluation`, `RunInferAllForSymbol`, and scheduler/frozen-outcome persistence) alone would leave the required composition and DB/transaction setup in `main.cpp`; moving the shared setup as-is would either duplicate it between the new application and training or extract substantial training composition into a new shared runtime/application layer. Both outcomes are explicitly outside Phase 22G's “move existing inference composition; do not redesign it” constraint.

The process-state prerequisite compounds this: `gRuntimeInferenceMode`, active evaluation-label state, BuildConfig globals, diagnostics, and profiler setup are consumed on both sides of the shared setup. A standalone `RunInference(const Options&)` that owns its connections/transactions would need a new explicit runtime/configuration abstraction spanning that shared path. That is broader than a tightly scoped setter/restorer and would change the existing training ownership boundary.

## Requested boundary and compatibility implications

No `Sources/InferenceApplication/InferenceApplication.hpp/.cpp` was created because a public API there would have been a façade over still-main-owned composition, rather than an independently linkable application. No static target was added. `LSTM/main.cpp` remains unchanged; consequently there is no duplicate implementation and no misleading compatibility adapter.

Existing compatibility routes remain untouched: direct `--infer --model`, final scheduler inference, checkpoint scheduler inference, `--infer-all`, force/start-after controls, strategy/Phase 19 modes, and frozen-model-outcome inference. Final/checkpoint persistence remains exactly where it was, as does worker registration/lifecycle at `main.cpp:8457-8511` and SchedulerCore. No scheduler routing, semantic registry/publisher, migration/schema, or historical layout-6 artifact policy changed.

## Validation

- `git diff --check`: deferred until after this report was written; run below as the final repository validation.
- No executable, scheduler command, DB test, or process-interacting test was launched. The requested production-safety process inspection is unnecessary because no runtime validation is safe or meaningful for a non-implementation.
- No build was run. A build would validate only the unchanged production source and cannot establish the requested independently-linkable extraction. The canonical full Release build additionally requires a clean source tree and this uncommitted report intentionally makes it dirty.

## Prerequisite and recommended next phase

A separately reviewed prerequisite phase must first establish a small shared model-input/runtime composition below both training and inference, with an explicit ownership contract for persisted configuration, feature/tensor construction, and process runtime state. It must prove that training continues to own its checkpoint/save loop and that the shared component does not take scheduler lifecycle authority. Only after that can Phase 22G extract a genuine independently-linkable `InferenceApplication`, followed by Phase 22H's thin worker.

Recommended next phase: **Phase 22G prerequisite — shared inference/training runtime composition boundary audit and extraction**, not Phase 22H.

## Final repository state

- `git diff --check`: return code 0 (there are no tracked diffs).
- `git diff --no-index --check /dev/null docs/Phase22/LSTM_Phase22G_InferenceApplicationExtraction_Output.md`: content check passed; its normal no-index difference status was normalized for the command.
- `git status --short`: `?? docs/Phase22/LSTM_Phase22G_InferenceApplicationExtraction_Output.md`.
- `git diff --stat`: empty, because the sole report is intentionally untracked and unstaged.
