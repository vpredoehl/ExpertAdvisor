---
title: "LSTM Phase 22I — ModelInputPreparation Extraction"
document_type: "implementation / validation report"
status: "final"
---

# LSTM Phase 22I — ModelInputPreparation Extraction

## Disposition

**Phase 22I ModelInputPreparation extraction: GO WITH PREREQUISITES**

The selected shared market/economic-input-to-Tensor composition is now an
independently linkable static-library boundary used by the existing shared
per-symbol path before it branches into inference or training. Runtime parity
against a database-backed fixture remains a prerequisite because this checkout
has no isolated database fixture/test target, and project executables were not
launched against potentially production-connected settings.

## Baseline and preflight

- Branch: `lstm-feature-development`
- Commit: `e08b990dc1b1f009bc1403a10f1bab3059825a13`
  (`Document Phase 22H shared input composition audit`)
- Initial `git status --short`: empty.
- The requested Phase 22F, 22G, and 22H reports, `AGENTS.md`, current source,
  and Xcode project were read before editing.

## Before and after

Before, `LSTM/main.cpp` constructed a Forex read transaction, loaded the
candlestick slice, resolved and loaded the economic corpus, constructed and
populated `Tensor`, and carried its logical output start index directly into
both the inference and training continuations.

After, `main.cpp` still selects the symbol, validates/resolves the calendar
lineage, owns model-input-width policy, and performs the frozen-outcome coverage
check. It passes only resolved values to:

```text
main inference/training shared path
  -> EA::ModelInputPreparation::Prepare
  -> private Forex RO transaction -> MarketDataCore
  -> private LSTM RO transaction -> economic events
  -> owned Tensor + logical output index + snapshot identity
  -> unchanged inference evaluation/persistence OR training/load/checkpoint/save
```

Both inference and training consume the exact same `preparedInput.tensor` and
`preparedInput.logicalOutputStartIndex` in the existing shared loop; no
second Tensor-building implementation remains.

## Public API

`Sources/ModelInputPreparation/ModelInputPreparation.hpp` defines:

```cpp
namespace EA::ModelInputPreparation {
struct DatabaseConnectionSettings {
  std::string forexConnectionString;
  std::string lstmConnectionString;
};
struct Request {
  std::string symbol, outputStart, outputEnd;
  EA::FeatureWarmupScope warmupScope;
  Donchian20Mode donchian20Mode;
  std::size_t donchianLookback;
  std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity>
      calendarSnapshot;
};
struct Result {
  Tensor tensor;
  std::size_t logicalOutputStartIndex;
  std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity>
      calendarSnapshot;
};
Result Prepare(const Request&, const DatabaseConnectionSettings&);
}
```

The API contains no `LaunchArgs`, scheduler state, `pqxx::work`, connection,
`EA::LSTM`, optimizer/checkpoint state, persistence state, or registry state.

## Changed files and target

- `Sources/ModelInputPreparation/ModelInputPreparation.hpp/.cpp`: new
  value-driven preparation component.
- `LSTM/main.cpp`: replaced the old shared market/event/Tensor materialization
  region with snapshot resolution plus `Prepare`; width validation, LSTM
  construction/loading, infer-all, inference evaluation/persistence, and all
  training/checkpoint/save code remain there.
- `ExpertAdvisor.xcodeproj/project.pbxproj`: new static
  `ModelInputPreparation` target, linked by Debug and Release LSTM targets.
  Its sole explicit target dependency is `MarketDataCore`; it has no
  SchedulerCore or `main.cpp` dependency.

No scheduler, registry, publisher, schema/migration, or production-state file
was changed.

## DB and lifetime ownership

`Prepare` creates and commits its own short read-only Forex transaction and
its own short read-only LSTM economic-event transaction using only copied
connection strings. Calendar snapshot lineage selection remains in the caller's
existing LSTM read transaction and is supplied as a copied identity. Market
rows and event vectors are local, event data is moved into `Tensor`, and the
returned `Tensor` is moved by value. No transaction, connection, row, or event
reference crosses the public seam.

The frozen-outcome coverage check remains caller-owned and is committed before
preparation. No global/static behavior was redesigned: runtime logging remains
the existing `EA::RuntimeDiagnosticLoggingEnabled()` policy; BuildConfig,
`gRuntimeInferenceMode`, labels, diagnostics, and profiler lifetime are
unchanged.

## Behavior retained

- Full-history/range query selection remains controlled by the same warmup
  enum and `kTensorFeatureHistoryQueryStart`.
- Donchian mode/lookback, event feature loading, feature ordering, Tensor row
  feed order, and logical output start validation are the original calls and
  values.
- Immutable calendar identity is propagated back unchanged, preserving the
  existing infer-all snapshot mismatch comparison.
- Model input-width validation remains at its former caller-owned position.
- Inference evaluation/final/checkpoint persistence and training model,
  optimizer, loop, checkpoint, and save ownership are unchanged.

## Validation

Commands and return codes:

1. `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj` — `0`.
2. `xcodebuild -project ExpertAdvisor.xcodeproj -scheme ModelInputPreparation -configuration Release -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" build` — `0`.
   The target graph contains only `ModelInputPreparation -> MarketDataCore`.
3. `xcrun clang++ -std=c++20 -fsyntax-only -IHeaders -ISources
   -ISources/ModelInputPreparation -ISources/MarketDataCore -IMetaNN
   -IMetaNN/MetaNN -IMetaNN/MetaNN/data/facilities -IMetaNN/MetaNN/data
   -IMetaNN/MetaNN/operation -IMetaNN/MetaNN/operation/tensor
   -I/opt/homebrew/include -I/opt/homebrew/opt/libpqxx@7.10.1/include
   -I/opt/homebrew/opt/libpq/include -I/opt/homebrew/opt/libomp/include
   LSTM/main.cpp` — `0`; existing libpqxx deprecation warnings were emitted.
4. `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug"
   -configuration Debug -derivedDataPath "$PWD/DerivedData/ExpertAdvisor"
   build` — `0`. The consumer executable compiled and linked
   `-lModelInputPreparation`; existing libpqxx deprecation warnings were
   emitted, but no new warning/error was introduced.
5. `git diff --check` — `0` before report creation; rerun after report below.

`nm -u libModelInputPreparation.a` shows only lower-level unresolved
MarketData/EconomicCalendar/Tensor symbols; it shows no SchedulerCore or
`main.cpp` symbol. The public header has no `pqxx::work` declaration.

No executable or test was launched: no isolated deterministic database fixture
or existing focused training/resume/direct-inference/infer-all test target was
found, and invoking the production-configured executable would violate the
production-safety constraint. Consequently, runtime parity, lifetime/error,
training/resume, direct-inference, and infer-all mismatch tests are explicitly
deferred—not claimed.

The provenance-valid canonical Release command was also deferred because its
provenance guard requires a clean source tree and this implementation is
intentionally uncommitted:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release"
  -configuration Release -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" build
```

## Next prerequisite and recommendation

Before the next inference-boundary phase, add an isolated, disposable
market/economic fixture that can establish Tensor parity for full-history and
range warmup, output index, Donchian values, snapshot identity, and failure
cleanup without touching scheduler or production data. Then re-run direct
inference/training-resume and infer-all mismatch compatibility tests against
that fixture. The recommended next Phase 22 increment remains the re-audit and
narrow InferenceApplication extraction; do not create a worker or route the
scheduler in this phase.

## Final worktree state

All changes are unstaged and uncommitted.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M LSTM/main.cpp
?? Sources/ModelInputPreparation/
?? docs/Phase22/LSTM_Phase22I_ModelInputPreparationExtraction_Output.md
```

`git diff --stat` (tracked files) reports:

```text
ExpertAdvisor.xcodeproj/project.pbxproj | 118 insertions
LSTM/main.cpp                           | 66 lines changed
```

The untracked component and this report are intentionally absent from Git's
default unstaged diff statistic; both were reviewed directly. `git diff
--check` passed for tracked changes, and `git diff --no-index --check` passed
for this untracked report (its normal difference exit status is `1`).
