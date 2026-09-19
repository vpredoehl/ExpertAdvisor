---
title: "LSTM Phase 22H — Shared Inference/Training Composition Boundary Audit"
document_type: "architecture audit / design only"
status: "final"
---

# LSTM Phase 22H — Shared Inference/Training Composition Boundary Audit

## 1. Baseline, scope, and disposition

- Repository: `/Volumes/Developer SSD/ExpertAdvisor`
- Branch: `lstm-feature-development`
- Baseline `HEAD`: `88d22508a08c6477b9c7a6fe9210645efd8d90c8`
  (`Document Phase 22G inference extraction blocker`).
- Preflight `git status --short`: empty.
- Latest relevant history: `88d2250 Document Phase 22G inference extraction
  blocker`; `84f8c0f Document Phase 22F inference composition audit`;
  `97770aa Add role-aware semantic worker artifacts`; `2760c82 Route scheduler
  analysis to standalone worker`; `3b883e3 Add standalone analyze worker`.
- The Phase 22F and 22G reports were read before this audit. All conclusions
  below were then checked against current source; the reports are not used as
  substitutes for that inspection.

**Phase 22H shared inference/training composition boundary audit: GO WITH
PREREQUISITES.**

The verified prerequisite is deliberately smaller than a shared runtime or
application: extract only a **value-driven model-input preparation component**
which builds a `Tensor` from an already-resolved plan. It may own the short
read-only market/economic reads that it initiates and the resulting `Tensor`,
logical-output start index, and calendar snapshot. It must not resolve
persisted model configuration, mutate process configuration, construct/load an
LSTM, own an evaluation transaction, or know scheduler identity.

This is the one common operation with identical present ownership after mode
configuration has been resolved. It removes the large shared data/Tensor block
from the future inference extraction path without copying it into an inference
application or carrying training lifecycle into one. Configuration resolution,
mode state, and model loading are not cohesive shared components: source shows
material inference/training differences there, so they remain mode-owned.

No production source, project/target, scheduler, registry, publisher, schema,
database, executable, or runtime artifact was changed. This report is the
sole worktree change.

## 2. Verified current control flow and ownership

```text
compatibility CLI / scheduler-owned command
  -> main.cpp:8453 ParseLaunchArgs; 8457-8511 worker admission/registration/log
  -> 8512-8535 common validation, launch globals, profiler configuration
  -> 8537-8689 two DB connections, optional frozen outcome, training-only
     objective and resume configuration
  -> 8691-8816 shared setup with mode-specific configuration branches
  -> 8826-9196 symbol selection, market/economic reads, Tensor, runtime DB
     transaction, model construction and loading
  -> 9198-9435 inference evaluation and inference-specific persistence, OR
  -> 9438-9677 training diagnostics, batch loop, checkpoint queue/save/final save.
```

The Phase 22G line-range finding remains materially correct, with current
source ranges as follows:

| Current source range | Verified role | Ownership conclusion |
|---|---|---|
| `LSTM/main.cpp:8433-8515` | top-level command dispatch, broad CLI parsing, scheduler managed-work guard, worker self-registration, worker-start logs, CLI globals | compatibility CLI/adapter plus scheduler lifecycle concern; keep outside all proposed components |
| `8530-8535` | LSTM hotspot profiler configuration and RAII finalizer | process diagnostic state; inference application may later own its own setup/finalizer, but not a shared input component |
| `8537-8689` | Forex/LSTM connections; frozen-outcome job; training objective; resume config and global mutation | mixed setup, substantially training-only from `8587-8689`; not a cohesive shared seam |
| `8691-8816` | available symbols, inference persisted configuration/application, scheduler final/checkpoint validation, Donchian/warmup/input identity | shared entry location but mode-specific configuration and scheduler validation; retain mode-owned |
| `8826-8879` | select resolved, resume, CLI, or all symbols | common selection algorithm but its input policy is mode-owned; include only the selected symbol in the proposed explicit input plan |
| `8880-9002` | load candlesticks; resolve/load economic corpus; create/populate Tensor; carry logical output index | genuinely shared model-input preparation; selected prerequisite |
| `9003-9046` | derive/validate model-input width against Tensor | common validation primitive, but expected width and scheduler identity are mode-owned; caller retains validation or passes expected width as a value |
| `9048-9196` | runtime transaction; infer-all / Phase-19 exits; feature mask; LSTM construction/load; resume optimizer state | mixed; construction factory is already a small helper, while loading behavior differs by mode; do not extract as one component |
| `9198-9435` | `RunInferenceEvaluation`, controlled/frozen/scheduler persistence | inference-only; later `InferenceApplication`, not the prerequisite |
| `9438-9677` | label reset, diagnostics, train batches, progress, checkpoints/queueing, final save | training-only; permanently outside an inference component |

`gRuntimeInferenceMode` is set by `ApplyLaunchRuntimeConfig` at
`4694-4726`, switched to false by resume configuration at `5128-5157`, and
checked through the shared path at `8589`, `8705`, `8739`, `8765`, `8777`,
`9087`, `9120`, `9130`, and `9198`. It is therefore a mode switch controlling
both sides, rather than state that a proposed common input component may own.

## 3. Shared-region classification inventory

| Block/helper/state | Classification | Evidence and disposition |
|---|---|---|
| `LaunchArgs` / `ParseLaunchArgs` (`3766-4691`) | compatibility CLI/adapter | One broad type holds inference, training/resume, scheduler, and diagnostic flags. Retain in `main.cpp`; a later inference adapter maps a lossless inference projection. |
| scheduler registration and started logs (`8457-8511`) | scheduler lifecycle/worker-registration | Calls `RegisterSchedulerWorker` before work. It remains an executable adapter responsibility. |
| `LoadPersistedInferenceConfig` / `ApplyPersistedInferenceRuntimeConfig` (`6294-6403`) | inference-only composition | Reads model metadata and validates redundant CLI fields, then mutates BuildConfig globals. Training instead uses launch/default globals or `ApplyResumeRuntimeConfig`. Move only later with inference composition after its dependency cleanup. |
| `ResolveSchedulerInferencePersistenceContext` (`6452-6686`) | inference-only persistence-context resolution | Final/checkpoint identity is checked here and later directs persistence. It is not scheduler daemon authority, but it is inference-only and does not belong in a generic shared setup. |
| resume config and training objective (`8587-8689`, `5128-5157`) | training-only | Includes target-epoch, optimizer restoration and input-width expansion semantics. Keep in `main.cpp`. |
| available symbol discovery and selected-symbol policy (`8693-8700`, `8826-8879`) | genuinely shared prerequisite, split at plan boundary | Discovery is common, but selection precedence is mode-specific. The component must accept a selected symbol, not an entire `LaunchArgs`. |
| scheduler Donchian/warmup/lookback/feature mask/input identity checks (`8753-8815`) | mixed, mode-owned validation | Inference config and resume config use different sources; scheduler checks are conditional. Keep callers responsible, pass resolved values. |
| `MarketDataCore::LoadCandlesticks` (`8946-8951`) | lower-level reusable component already present | `MarketDataCore` is already reusable. The missing seam is composition of its request with economic events and `Tensor`, not re-extraction of MarketDataCore. |
| economic snapshot/event loading (`8952-8985`) | genuinely shared prerequisite | Same data needed by the common Tensor construction; component can own this short LSTM read transaction. |
| `Tensor` construction/population (`8986-9002`) | genuinely shared prerequisite | Identical Tensor constructor and row feed are consumed by inference and training. Return an owned Tensor rather than a context object. |
| logical output start index (`8950`, `9001`) | genuinely shared prerequisite output | Originates in MarketDataCore and is used by `RunInferenceEvaluation` (`7236`) and training batches (`9468`). Return by value alongside Tensor. |
| runtime input-width validation (`9003-9046`) | common primitive, caller-owned policy | Tensor width calculation is reusable; `resumeConfig`, persisted config, and scheduler identity determine the expected width. Keep expectation/identity checks outside. |
| `CreateLstmForRuntimeLogLevel` (`6706-6738`) | lower-level reusable helper candidate only | It already has a narrow Tensor/value API. It may move with the eventual inference composition or a small model factory, but does not justify a new shared composition object. |
| `DBIO::PgModelIO::loadAll` plus optimizer/meta loading (`9112-9196`) | mixed | Base model load is common, but training adds objective compatibility, optimizer state, completed epoch and expansion. Do not extract a shared loader that disguises these ownership differences. |
| `RunInferenceEvaluation` (`7190-7705`) and `ProcessBatchPredict` (`2783-3078`) | inference-only / lower-level inference implementation | Evaluation orchestrates inference diagnostics and strategy paths; batch prediction is a low-level kernel called by it. Both move later with InferenceApplication, not now. |
| `RunInferAllForSymbol` (`8137-8349`) | compatibility inference-only | It uses the same Tensor but is a compatibility-only multi-model/persistence route. Preserve it in later inference composition; future narrow worker rejects it. |
| final/checkpoint persistence (`9351-9432`) | inference-only persistence | Existing idempotent persistence remains with later inference app, not shared input preparation. |
| training/checkpoint/save lifecycle (`9438-9677`; helpers `3248-3583`, `5439-5570`) | training-only | Training loop, progress, checkpoint creation/queueing, stop hooks, and model save remain in `main.cpp`. |

## 4. Process-global/static-state inventory

| State | Current use | Required treatment |
|---|---|---|
| `gRuntimeInferenceMode` (`122`) | range label (`196-199`), mode branches, launch/resume configuration | retain process-global behavior for the prerequisite. A later inference application must establish inference state internally or replace its inference consumers; do not pass it into input preparation. |
| BuildConfig globals: `prediction_horizon`, `c_next_threshold`, `window_size`, `hidden_size`, `n_out`, `num_layers`, `normalization_version`, LR multipliers, epoch count | mutated by CLI (`4694-4726`), resume (`5128-5157`), and persisted inference config (`6388-6403`); consumed by validation, labels, evaluation, construction and training | no broad value-context conversion in the prerequisite. Later extraction needs a narrowly scoped, reviewed configuration solution; the input component receives only resolved `Donchian20Mode`, lookback, dates and warmup policy. |
| active eval label static (`265-279`) | changed by `RunInferenceEvaluation` (`7203-7216`), reset on training entry (`9438`) | remain with evaluation/training. Not input preparation state. |
| `DiagnosticOut` static null stream (`165-193`), diagnostic counters (`2818-2819`, `9090-9103`) | process-local output/rate limits | retain; no public component state. Later move only the counters/helpers that move with inference. |
| LSTM hotspot profiler (`8530-8535`) and LSTM diagnostic bucket/distribution resets | lifetime spans full process/main invocation | retain at adapter/main until the later inference extraction can preserve exactly-once finalization. |

The `prediction_horizon`/threshold/window globals make a single
`ModelRuntimeComposition` misleading: their consumers span diagnostics,
evaluation, persistence identity, phase-19 tooling, and the training loop.
Replacing all of them with broad dependency injection is not a safe
prerequisite. The selected input component intentionally avoids them.

## 5. DB, transaction, and object-lifetime analysis

| Resource | Current lifetime / owner | Boundary requirement |
|---|---|---|
| `pqxx::connection c_forex`, `c_LSTM` | constructed in `main.cpp:8537-8538`; live through the outer work loop and destroyed on main return | selected component should receive connection strings (or a tiny private connector factory), create its own short read-only connections/transactions, and return materialized values. It must not expose a connection publicly. |
| Forex metadata transaction | `8695-8699`, read-only, committed before configuration resolution | candidate component need not own it: caller resolves/selects a symbol. This keeps selection policy mode-owned. |
| LSTM configuration transaction | `8702-8816`, read-only, contains inference config, scheduler validation, metadata/masks | must remain caller-owned; its contents are mode-specific and no transaction crosses a component seam. |
| Forex data transaction | one per selected symbol, `8882-9000`, read-only; contains coverage check and `LoadCandlesticks`; committed after Tensor rows are added | selected component owns this transaction internally and returns a fully materialized Tensor / logical start index. No `pqxx::work` leaks. |
| Economic event transaction | `8956-8984`, read-only; snapshot and events materialized before Tensor construction | selected component owns this transaction internally. Snapshot identity is returned by value for infer-all compatibility checks. |
| runtime LSTM work | `9048` starts read-only only for controlled strategy evaluation, otherwise read-write; held through model load/evaluation and inference persistence; training commits it at `9455` before long training | remains inference/training caller-owned. The public input seam must not accept it. |
| save/checkpoint transactions | training helper work and final `wSave` at `9580-9664` | training-only, outside every proposed component. |
| market rows / economic events | `marketData` materializes locally; event vector moves into `Tensor` (`8946-9000`) | component owns transient rows/events and returns only Tensor; no borrowed data remains. |
| `Tensor t` | created per symbol at `8986`, stack lifetime through either inference branch or training loop | return by value/move in `PreparedModelInput`; resulting owner is the immediate inference/training caller. |
| `EA::LSTM l` | constructed at `9084`, binds `t`; valid until evaluation or training finishes | do not put in selected component. Its construction/load has different inference, resume, and training-objective behavior. |
| profiler/diagnostic finalizer | `8530-8535`, enclosing `main` scope | do not transfer into selected component. |

The safe public seam therefore takes and returns values only. It requires no
`pqxx::transaction_base`, `pqxx::work`, connection, reference-to-row, or
LSTM reference across its interface.

## 6. Candidate boundary designs

| Candidate | Responsibility, inputs, outputs, callers | Source moved / remains | Behavioral and dependency assessment | Verdict |
|---|---|---|---|---|
| A. One `ModelRuntimeComposition` | Resolve config, scheduler context, input data/Tensor, LSTM build/load; input broad `LaunchArgs`/connections; output runtime context | Would move most of `8691-9196`; training loop remains | Requires a monolithic object holding globals, transactions, Tensor/LSTM and scheduler-conditioned state; model load differs for resume/objective/optimizer. Violates ownership and creates forbidden runtime/application abstraction. | Reject |
| B. Shared persisted runtime/model config | Load/apply persisted inference config as a shared service | Moves inference helpers `6210-6403`; training resume stays | Not genuinely shared: training uses launch/default config or resume config and has target-epoch/objective/optimizer rules. Global BuildConfig mutation would remain cross-cutting. | Reject |
| C. Model-input preparation (selected) | Given already-resolved symbol, range, warmup, Donchian mode/lookback and optional calendar selection, fetch market/events and build Tensor | Moves the common mechanical core of `8880-9002`; leaves selection/config/width policy/model load and both mode branches in main | No training/inference algorithm change. Explicit small values, owned materialized output, short internal RO transactions. Links existing MarketDataCore, EconomicCalendar, Tensor and pqxx only. Fully reversible. | Select |
| D. Shared model build/load composition | Build LSTM and load persisted model | Would move `9084-9196` | Inference and normal load share `loadAll`, but training adds objective validation; resume adds source/id choice, parameter expansion, optimizer metadata and completed epoch. A single loader would be conditional training ownership. | Reject; retain existing narrow factory/helper and repository calls |
| E. Move inference-only tail first | Move `RunInferenceEvaluation`, infer-all, persistence (`7190-8349`, `9198-9435`) to InferenceApplication | Leaves prior required setup in main | Recreates 22G's no-go: application is not independently linkable without duplicating setup or exposing a broad main-owned runtime. | Reject |

### Selected interface shape (proposal only)

The later prerequisite should use names comparable to the following, not an
application/runtime context:

```cpp
namespace EA::ModelInputPreparation {
struct Request {
  std::string symbol;
  std::string outputStart;
  std::string outputEnd;
  EA::FeatureWarmupScope warmupScope;
  Donchian20Mode donchian20Mode;
  std::size_t donchianLookback;
  std::optional<EconomicCalendarSnapshotIdentity> calendarSnapshot;
};
struct Result {
  Tensor tensor;
  std::size_t logicalOutputStartIndex;
  std::optional<EconomicCalendarSnapshotIdentity> calendarSnapshot;
};
Result Prepare(const Request&, const DatabaseConnectionSettings&);
}
```

The precise connection-settings type should be a small private/common value
already derivable from `ForexDbConnectionString()` and `LstmDbConnectionString()`;
it is not a service locator. If direct `pqxx::connection` construction proves
necessary for current connection-string behavior, keep that construction
private to the component. `Request` must not accept `LaunchArgs`, a scheduler
context, a transaction, or an LSTM. A caller retains `selectedSymbols` policy,
the read-only config transaction, model input width decision, and the
read-write evaluation/train transaction.

## 7. Explicit non-goals and retained ownership

The selected prerequisite must not absorb scheduler daemon authority,
reservation/launch/reconciliation, worker registration, semantic-role
selection, registry/publisher behavior, final/checkpoint inference result or
profitability persistence, training loop/checkpoint/save, continuation policy,
priority/preemption, analyze-worker behavior, publication/cutover, or database
schema work. It also must not move `RunInferenceEvaluation` or
`ProcessBatchPredict` yet.

Training behavior must remain unchanged: its caller prepares a plan from
resume/default configuration, receives a Tensor, sets its own training
objective, performs resume model/optimizer loading as today, commits the
startup work before training, and owns checkpoint/final-save lifecycles.
Inference behavior must remain unchanged: its caller resolves persisted
configuration/context, prepares the same plan, retains its evaluation work and
then executes unchanged evaluation/persistence routes.

## 8. Proposed later implementation scope, links, tests, and stop conditions

Proposed later files/targets only (not created here):

- `Sources/ModelInputPreparation/ModelInputPreparation.hpp/.cpp`, in a small
  static target linked by the existing LSTM executable target.
- Move only the composition around current `8880-9002`, with a small unit test
  target/source following existing C++ test patterns. Do not change the Xcode
  project until that implementation phase is separately authorized.
- Existing dependencies remain `MarketDataCore`, `Tensor`, EconomicCalendar
  repository/features, libpqxx, logging and BuildConfig declarations already
  used by the executable. No SchedulerCore dependency is justified for this
  component. It must not link an inference application target yet.

Focused proof required in that future implementation:

1. deterministic request-to-Tensor parity test covering full-history and
   range warmup, logical output index, Donchian mode/lookback and economic
   snapshot identity;
2. unit/fixture test that errors do not leak an open transaction and that the
   returned Tensor has no borrowed market/event data;
3. existing compatible direct inference and training/resume regression tests,
   demonstrating unchanged model-input width validation and model binding;
4. an infer-all compatibility test preserving snapshot mismatch behavior; and
5. a link/build test proving the new target depends only on its declared
   lower-level libraries, not SchedulerCore or a `main.cpp` symbol.

Stop immediately with NO-GO in that implementation if preserving the exact
connection/environment behavior requires passing `pqxx::work` through a public
seam; if model construction/loading or scheduler state must be added to make
the component useful; if training and inference need divergent Tensor
algorithms; if the project requires scheduler/registry/schema changes; if
BuildConfig global removal expands beyond a narrowly reviewed later inference
step; or if compatibility tests expose altered logical-output, calendar,
feature, or persistence behavior.

## 9. Recommended Phase 22 sequence

1. Authorize and implement only selected Candidate C; prove direct
   training/inference input parity and preserve process-global behavior.
2. Re-audit the now-reduced `main.cpp` boundary. Move inference-only persisted
   configuration resolution, evaluation, model construction/loading and
   persistence into `InferenceApplication`, using the shared input component;
   do not move training ownership.
3. Add the narrow `lstm-infer-worker` adapter and its strict scheduler-only
   parser/registration adapter.
4. Route both final and checkpoint inference through role-aware semantic
   selection.
5. Seek separate authorization for publication/cutover.

This differs from treating a shared configuration/runtime object as step 1:
direct source shows configuration and model loading are not single common
ownership domains. The selected Tensor-input boundary is the smallest
reversible prerequisite that materially reduces the verified 22G blocker.

## 10. Audit validation and final state

- Read the requested 22F/22G reports and inspected current source directly,
  including `main.cpp` control flow, configuration helpers, inference helpers,
  MarketDataCore/EconomicCalendar call sites, and current target file layout.
- No build, executable, scheduler/worker action, artifact publication, or
  database interaction was performed, as required for this audit-only phase.
- `git diff --check`: passed after report creation.
- New-report content check: `git diff --no-index --check /dev/null
  docs/Phase22/LSTM_Phase22H_SharedInferenceTrainingCompositionBoundaryAudit_Output.md`
  passed (the expected no-index difference status was normalized).
- Files changed: `docs/Phase22/LSTM_Phase22H_SharedInferenceTrainingCompositionBoundaryAudit_Output.md` only.
- Final `git status --short`:
  `?? docs/Phase22/LSTM_Phase22H_SharedInferenceTrainingCompositionBoundaryAudit_Output.md`
- Final `git diff --stat`: empty, because the sole report is untracked and
  unstaged.
