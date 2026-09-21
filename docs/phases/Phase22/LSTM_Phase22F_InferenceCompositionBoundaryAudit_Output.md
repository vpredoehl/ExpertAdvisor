---
title: "LSTM Phase 22F Inference Composition Boundary Audit"
document_type: "architecture audit / design only"
status: "final"
---

# LSTM Phase 22F — Inference Composition Boundary Audit

## Baseline and disposition

- Branch: `lstm-feature-development`
- Commit: `554b0710292078248bff682dba9ee55fa37f6af6` (`Add role-aware semantic worker artifacts`)
- Pre-flight: `git status --short` was empty. No source, Xcode project, test,
  database, scheduler, process, artifact, registry, or production state was
  changed by this audit.
- Historical compatibility binding: layout 6 continues to mean the immutable
  `LSTM_Release` from `7645265bca0c2529523e1d2cdb37e7d023dfd559`, SHA-256
  `945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7`.

The narrow feasible seam is an **inference-only application composition**, not
a wrapper around `LSTM_Release` and not a general Phase-21-style decomposition.
It must own the existing infer branch currently interleaved in `LSTM/main.cpp`;
a thin compatibility adapter and a thin new worker adapter can then invoke it.
The inference kernels, feature construction, persistence, and validation remain
the exact existing implementations.

**Phase 22F inference composition boundary audit: GO WITH PREREQUISITES**

Prerequisite means the extraction itself must first make the infer branch
independently linkable and prove `LSTM_Release --infer...` parity before adding
or routing a worker. No registry/database prerequisite was found.

## Current architecture and entry paths

```text
operator `LSTM_Release --infer...` ──────────────┐
scheduler final INFER ── BuildInferCommand ──────┤
scheduler checkpoint INFER ─ BuildCheckpoint... ─┤
                                                 v
LSTM/main.cpp: ParseLaunchArgs / register attempt / global runtime setup
                                                 v
model + experiment admission / persisted configuration resolution
                                                 v
MarketDataCore -> Tensor + economic-calendar snapshot/events -> EA::LSTM
                                                 v
RunInferenceEvaluation -> ProcessBatchPredict -> model inference
                                                 v
final/checkpoint/infer-all persistence + profitability persistence
                                                 v
return status / profiler finalizer
```

The direct compatibility routes are `--infer --model`, `--infer --infer-all`
(with `--force-infer` / `--infer-start-after-model-id`), direct strategy and
Phase-19 evaluation modes, and `--run-frozen-model-outcome-inference`. They all
parse in `LSTM/main.cpp:4039-4691` and converge on the runtime infer branch.
`--infer --model` resolves model metadata at `8704-8749`, builds the tensor at
`8850-9060`, loads the model at `9112-9196`, and evaluates at `9198-9435`.
`--infer-all` instead calls `RunInferAllForSymbol` at `9060`; candidates call
the same `RunInferenceEvaluation` at `8052-8115`.

Final scheduler work is built in
`Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:3434-3460` as:

```text
<selected executable> --infer --model=<last_model_id>
  --scheduler-experiment-id=<experiment_id>
  --donchian20-mode=<persisted> --feature-warmup-scope=<persisted>
  --donchian-lookback=<persisted> --log-level=summary [profile options]
  <infer-start> <infer-end>
```

Checkpoint work is built at `3489-3515` and differs only in model identity and
`--scheduler-checkpoint-eval-id=<checkpoint_eval_id>`. The launcher appends
`--scheduler-worker-attempt-id=<id>` only after canonical-executable validation
(`3073-3115`). Final work reserves via `ReserveExperimentWorkerAttempt`
(`2835-2923`); checkpoint via `ReserveCheckpointWorkerAttempt` (`2925-2992`).
Both select the inference role before reservation.

The child currently self-registers in `main.cpp:8454-8499` using the existing
`EA::SchedulerCore::RegisterSchedulerWorker`, whose exact PID/process-group/
start-identity/canonical-executable matching is in
`Sources/SchedulerCore/SchedulerWorkerRegistration.cpp:38-143`. Scheduler
reconciliation and orphan/missing-process handling consume that same persisted
attempt identity in `GlobalExperimentControl.cpp` and
`ProductionSchedulerDaemon.cpp`; they do not need an inference-specific new
lifecycle protocol.

Continuation-related inference is scheduler final work after a resumed model;
the inference command still uses `experiment.last_model_id`. Training's
checkpoint save/queue path remains in `main.cpp:3248-3583` and
`5439-5570`; it creates checkpoint-eval work, but it does not execute inference.
Forced/retry final inference requeues the existing `infer` phase and therefore
re-enters `BuildInferCommand`. Recovery reuses the reserved command/executable,
not a separate inference implementation.

## What is embedded in `main.cpp`

`main.cpp` is 9,682 lines. The independently needed inference composition is
not one function: the live infer application path is roughly 2,100 lines of
inference-only or inference-dominant orchestration plus its nearby private
types/helpers, interleaved with training in the main routine. The important
inventory follows.

| Main.cpp area | Classification | Extraction treatment |
|---|---|---|
| `3766-4691` `LaunchArgs`, parsing, validation | CLI/operator-only, shared with training | Keep compatibility parser in `main.cpp`; introduce an explicit public inference-options value and adapt to it. Worker gets a narrow separate parser. Do not copy the broad parser. |
| `4694-4718`, `6388-6418` global runtime configuration | reusable composition, currently global | Move infer application use behind a scoped/explicit inference runtime configuration; its state is the main small prerequisite refactor. |
| `6210-6704` model/config/scheduler admission helpers | reusable inference composition | Move with the application: persisted model configuration, model symbol resolution, scheduler persistence context, semantic/model-width/feature-ablation checks. |
| `6718-7035`, `7190-7705` | reusable inference composition | Move with it: identity, final/checkpoint result persistence, profitability observation, `RunInferenceEvaluation`, feature/Tensor decisions. |
| `7712-8347` | reusable compatibility inference composition | Move with it so `--infer-all` retains exact implementation; the future worker must reject it. |
| `8433-9435` infer portions of `main` | reusable inference composition plus lifecycle adapter | Split into compatibility dispatch, worker registration adapter, and `RunInferenceApplication`. |
| `3248-3758`, `4798-5570`, `9438-9682` | training/checkpoint production | Remain in `main.cpp`; only existing persisted scheduler data is read by inference. |
| baseline/label-grid/trainability diagnostics and meta/import/scheduler dispatch | unrelated | Leave untouched. |
| `2783-3112` `ProcessBatchPredict` and evaluation diagnostics | shared low-level inference implementation | Move with inference composition, not reimplement. It calls `EA::LSTM`, `Tensor`, label, profitability and strategy adapter code. |

Existing reusable components already outside `main.cpp` include `EA::LSTM`
(`LSTM/LSTM.cpp`), `Tensor` (`LSTM/Tensor.cpp`), `DBIO::PgModelIO`, the static
`MarketDataCore` target, economic-calendar repository/features, profitability
repository/core, StrategyEvaluationCore and adapters, runtime logging,
SchedulerWorkerRegistration, and SchedulerCore lifecycle/reconciliation. They
are lower-level dependencies, not an inference application boundary.

## Minimum extraction seam

Create `Sources/InferenceApplication/InferenceApplication.hpp` and
`Sources/InferenceApplication/InferenceApplication.cpp`, and one static
`InferenceApplication` target. These locations are proposed because the code is
application composition rather than a new model/repository layer. The exact
names are intentionally based on the existing `LaunchArgs` and
`RunInferenceEvaluation`, rather than inventing a generic WorkerApplication.

Pseudocode for the public seam:

```cpp
namespace EA::InferenceApplication {
struct Options { // public, value-only infer projection of existing LaunchArgs
  std::string fromDate, toDate;
  std::optional<long long> modelId, schedulerExperimentId,
      schedulerCheckpointEvalId;
  // existing inference controls: model/config redundancies, infer-all,
  // force/start-after, eval/Phase-19/frozen outcome, logging/profiling.
};

int RunInference(const Options& options); // owns connections and transactions
}
```

`Options` should retain current field semantics; it is not permission to rename
or normalize behavior. `main.cpp` may retain `LaunchArgs` while it is shared by
training, then perform a lossless conversion only when `--infer` is selected.
The worker parser directly creates only the scheduler-safe subset of `Options`.
`RunInference` contains the existing setup/evaluation/persistence infer paths,
including infer-all and direct compatibility modes. It opens the current Forex
and LSTM connections and retains their transaction boundaries: read-only
metadata/market/economic transactions, one read-write evaluation transaction
unless a controlled evaluation explicitly makes it read-only, then existing
final/checkpoint/profitability commits.

Ownership after extraction:

- CLI parsing: `main.cpp` keeps the broad compatibility parser; new
  `LSTM/InferWorkerMain.cpp` owns only the narrow worker parser (both split and
  equals forms). Neither parses by invoking the other executable.
- Worker-attempt registration: adapters call the existing
  `RegisterSchedulerWorker` before `RunInference`; the shared application does
  not reserve, launch, or reconcile attempts. This preserves lifecycle logic
  without coupling direct inference to scheduler state.
- Logging/profiling: `RunInference` configures the existing runtime log level
  and `EA::LSTM` profiler/finalizer exactly once. Callers supply options.
- Semantic admission: Scheduler owns registry selection/admission before
  reservation; the application owns existing persisted model/experiment,
  model-input-width/layout, ablation, Donchian/warmup, symbol/range and
  checkpoint/final identity validation.
- DB connections: composition owns them; no caller-supplied `pqxx` transaction
  crosses the public seam.

The thinnest resulting `main.cpp` adapter is: preserve non-inference command
dispatch and training; parse compatibility args; prohibit unmanaged scheduler
work; self-register if applicable; map an inference `LaunchArgs` to `Options`;
call `RunInference`. The immutable old layout-6 binary naturally cannot receive
this adapter; that is not a compatibility violation because it remains selected
and executes its already-shipped layout-6 inference implementation unchanged.

## Global/static and implicit state

No broad dependency-injection framework is needed, but these implicit inputs
must be made composition-local or explicit:

- `gRuntimeInferenceMode` (`122`) drives range labels, mode branching and
  configuration. Replace inference use with an application-local constant or
  scoped setting established by `RunInference`; training retains its current
  state. Do not leave infer behavior dependent on whichever executable called
  it.
- `prediction_horizon`, `c_next_threshold`, `window_size`, `hidden_size`,
  learning-rate and related BuildConfig globals are overwritten by
  `ApplyPersistedInferenceRuntimeConfig`. This established behavior must move
  together with config loading; a small explicit runtime-config object (or
  narrow scoped setter/restorer) is required so the application can link alone.
- `ActiveEvalLabelConfig()` static (`265-279`), `DiagnosticOut()` static null
  stream (`165-193`), per-process diagnostic counters in
  `ProcessBatchPredict` and LSTM binding, and profiler state are safe to move
  with the application/runtime logging. They are process-local, not scheduler
  identity state.
- `ForexDbConnectionString`, `LstmDbConnectionString`, `CurrentUtcDate`, and
  economic snapshot resolution are private helpers required by inference; move
  or give the application private equivalents, preserving environment behavior.
- Training-only checkpoint queues, resume objective state, model-save state,
  and checkpoint-stop test hook stay in `main.cpp`. They are not worker-boundary
  blockers.

## Final versus checkpoint convergence

Both paths use the same persisted model configuration, market data, economic
snapshot, Tensor, `EA::LSTM` load, `RunInferenceEvaluation`, prediction math,
acceptance computation and profitability accumulator. They diverge only at
`ResolveSchedulerInferencePersistenceContext` (`6542-6704`) and persistence:
final calls `PersistCompletedInferenceResult` then profitability scope `final`;
checkpoint calls `PersistCompletedCheckpointInferenceResult` then scope
`checkpoint`, with checkpoint-eval/parent-experiment identity. Thus the one
application seam supports both; the scheduler cutover must change both
`BuildInferCommand` and `BuildCheckpointEvalInferCommand` callers, never only
the former.

## Future `lstm-infer-worker` contract

The worker accepts exactly:

```text
--infer
--model MODEL_ID                         (split or --model=MODEL_ID)
exactly one of:
  --scheduler-experiment-id EXPERIMENT_ID
  --scheduler-checkpoint-eval-id CHECKPOINT_EVAL_ID
--scheduler-worker-attempt-id ATTEMPT_ID
--donchian20-mode VALUE
--feature-warmup-scope VALUE
--donchian-lookback VALUE
--log-level VALUE
--lstm-profile-hotspots
--lstm-profile-output PATH
FROM_DATE TO_DATE
```

All value options must accept the split and equals forms because scheduler uses
equals (`AddCliOption`) while compatibility CLI supports both. Require explicit
`--infer`, one explicit model, two dates, attempt id, and exactly one scheduler
identity; invoke `RegisterSchedulerWorker` with `checkpoint_infer/infer` or
`experiment/infer` respectively, then call `RunInference`.

Reject training/resume/model-creation flags; `--infer-all`, `--force-infer`,
`--infer-start-after-model-id`; raw `--symbol`, width/horizon/threshold/hidden
or learning-rate overrides; direct strategy/Phase-19/frozen-outcome modes; all
scheduler admin/daemon/analyze commands; duplicate options; absent attempt id;
and both/neither final/checkpoint identity. The scheduler currently propagates
only the listed persisted mode/warmup/lookback/log/profile fields; no other
legitimate scheduler inference option was found.

## Target, link, and runtime composition

The new executable should comprise thin `LSTM/InferWorkerMain.cpp` plus
`InferenceApplication` and the existing implementation dependencies required by
the moved code: `LSTM.cpp`, `Tensor.cpp`, `PricePoint.cpp`, `PgModelIO` and
model/input helpers, economic event repository/features, runtime logging,
inference profitability repository, profitability verification pieces used by
compatibility modes, StrategyEvaluationCore/adapters, `MarketDataCore`,
`libMetaNN.a`, `libMetalBuffer.a`, libpq/libpqxx/libomp, and macOS Foundation,
Metal and MetalPerformanceShaders. Exact source membership should be proven via
the new target link map; do not copy the whole LSTM Release source phase.

Compared with `LSTM Release`, omit training-loop/checkpoint-save and unrelated
admin/meta/import sources. Compared with `lstm-analyze-worker`, inference needs
MetaNN/Metal, Tensor/LSTM, MarketDataCore, profitability and strategy runtime;
the analyze worker needs SchedulerCore analysis composition instead.

Metal is a real worker requirement: the Release target currently links
Metal.framework, MetalPerformanceShaders.framework, Foundation.framework and
`libMetalBuffer.a`, and depends on MetaNN/libMetaNN. `MetaNN/metal/metal_add.mm`
loads `MetaNN.metallib` from `NSBundle.mainBundle`; the current tool targets use
a CopyFiles phase and MetaNN metallib dependency. The new worker must receive an
equivalent copy/resource installation and be tested from its published runtime
directory, not merely from DerivedData. Its runtime package must contain the
resource(s) whose hashes are recorded by the Phase-22E runtime manifest.

## Semantic artifact readiness and routing changes

Phase 22E is sufficient once a real executable exists. Registry v3 represents
role explicitly; `SemanticWorkerRegistry::selectInferenceWorker` (`853-885`)
selects by persisted layout/input width and infer capability, not filename or
layout-number heuristic. Role-aware manifest v2 requires infer executable
identity `lstm-infer-worker`; legacy manifest v1 remains supported. Publisher
already supports only infer-role artifacts at
`layout<layout>/infer/<commit>/<sha256>/lstm-infer-worker`.

For layout 6 the registry retains an immutable historical `LSTM_Release`
binding and returns it as `immutable_historical_semantic_worker`; it must not be
rebuilt, renamed, wrapped or republished. For new layouts, role-aware selection
will return the published infer worker. Reservation persists this selected
canonical path; launch validates it; self-registration compares its own real
path to it; reconciliation/recovery uses the persisted attempt path. No registry
or publisher change is indicated.

After extraction/worker addition, the only scheduler routing work is to give
the daemon a role-specific infer-worker default/override analogous to analyze,
then pass the selected semantic executable to both existing builders. The
selection/reservation code at `2835-2992` and exact identity flow at
`3073-3335` should remain authoritative. Tests must verify command observation
and recovery still recognize `--infer` plus the appropriate scheduler identity.

## Training coupling

Inference materially shares `EA::LSTM`, Tensor/model feature semantics,
`PgModelIO`, BuildConfig runtime values, MarketDataCore and economic-calendar
features with training. It also reads training-persisted configuration and
model/checkpoint rows. That requires the small runtime-config seam described
above, not extraction of training, checkpoint creation, resume orchestration or
model saving. The train branch remains in `main.cpp`.

## Test plan and implementation slices

Extend rather than duplicate `Tests/SchedulerSemanticAdmissionIntegrationTests.sh`,
`Tests/SemanticWorkerRegistryTests.cpp`, `Tests/SemanticWorkerPublisherTests.py`,
the existing scheduler ownership/recovery integration tests, and the
`StandaloneAnalyzeWorkerCliTests.sh` pattern. Add focused tests for:

- compatibility `LSTM_Release --infer --model` parity and direct inference
  persistence/profitability;
- worker split/equals parser forms, rejection matrix, registration success and
  registration mismatch exit `125`;
- exact final and checkpoint argv contracts, including profiler fields;
- final and checkpoint result/profitability scopes and semantic admission;
- exact canonical executable persisted at reservation/spawn/self-registration;
- layout-6 `LSTM_Release` selection and new role-aware infer-worker selection;
- executable mismatch rejection, daemon restart/reconciliation, orphan and
  missing-process recovery for both infer scopes;
- Metal/metallib resource presence and launched-published-artifact lookup;
- ANALYZE and TRAIN routing non-regression; and no publication during extraction
  and target/CLI implementation.

Recommended reversible slices:

1. Extract `InferenceApplication` plus compatibility adapter only; build and
   run parity/focused inference tests. Stop if output, persistence or profiler
   behavior differs.
2. Add `lstm-infer-worker` target, narrow CLI, self-registration, target link
   map and packaged-runtime test; no scheduler routing or publication.
3. Route **both** final and checkpoint builders through role-aware selection;
   run lifecycle/recovery/legacy-layout tests, still without publication.
4. After clean-tree review and full provenance-valid Release build, obtain
   separate authorization for immutable artifact publication and production
   cutover.

Expected implementation files: new `Sources/InferenceApplication/InferenceApplication.{hpp,cpp}`;
new `LSTM/InferWorkerMain.cpp`; `LSTM/main.cpp`; `ExpertAdvisor.xcodeproj/project.pbxproj`;
only necessary scheduler daemon configuration/routing files; and the focused
tests above. No database migration, registry/publisher change, or training
worker file is expected.

Rollback points are after each slice: retain the original compatibility adapter;
do not select/publish the new target until slice 2 is green; and retain the
registry's immutable layout-6 selection throughout. Primary risks are hidden
global BuildConfig coupling, an omitted compatibility inference mode, and
metallib resource lookup outside the build directory; the parity, narrow-CLI
and packaged-artifact tests directly address them.

## Final checks

Final `git diff --check`: passed.

Final `git status --short`:

```text
?? docs/Phase22/LSTM_Phase22F_InferenceCompositionBoundaryAudit_Output.md
```

Final `git diff --stat`: empty because this required report is intentionally
untracked. This audit intentionally ran no build or executable: project
instructions prohibit launching work without checking active scheduler/training
state, and the requested scope is source inspection/design only.
