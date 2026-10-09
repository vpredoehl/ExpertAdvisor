# Phase 25B-3: production-representative Metal training qualification

**Status: BLOCKED on 2026-10-09. Production readiness is not established.**

Initial preconditions passed. A production scheduler subsequently started during the first paired trial's canonical-input preparation. The benchmark guard terminated only its own fixture and stopped the trial runner. Zero paired trials completed. Numerical comparisons, checkpoint restoration and sustained-memory measurements were not started. No Phase 25B-4 work was started.

## Baseline and safety

| Item | Verified evidence |
|---|---|
| Development directory | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` |
| Branch | `dedicated-train-layout-rollover-squashed-v1` |
| HEAD / Phase 25B-2 | `4493601b30fd70dd44d3a7567d6d53f8510e9baf` |
| Phase 25B-1 ancestor | `091234b54fad122b377aec0044b7ab9dc19e6cb7` |
| Initial development tree | Clean; `git status --short` produced no output |
| Production / shared MetaNN source | Git status clean at initial and post-abort inspections; this work modified neither |
| Installed toolchain | Xcode 27.0 (27A266a), Apple clang 21.0.0, macOS 27.0.1 (26A434), ARM64 |
| Memory | 51,539,607,552 bytes / 48 GiB; initial `memory_pressure -Q` reported 89% system-wide memory free; zero swap |
| Initial workers | No LSTM worker or scheduler process present; Ollama serve present with zero loaded models / no runner |
| Initial scheduler service | `com.vjp.lstm.scheduler`: active count 0, no PID, last exit 78 / EX_CONFIG, job state spawn failed |
| Initial persisted scheduler state | Global desired state running; no running experiments. An owner invocation without an end timestamp and a lease marked active remained, but its expiry was 2026-10-09 10:21:30 America/Chicago and its PID was absent. These records were left untouched. |
| Production PostgreSQL | Explicit `default_transaction_read_only=on`; `BEGIN READ ONLY` / `SET TRANSACTION READ ONLY`; both read-only settings verified on |

The installed Xcode version differs from the general AGENTS.md description (26.5), and matches the toolchain recorded for Phase 25B-2. No toolchain, production scheduler configuration or shared dependency source was changed.

### Exact execution blocker

The safety census captured production scheduler **PID 14494**, launched around **20:50:16 UTC / 15:50:16 America/Chicago**, with:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
--schedule-experiments --max-train-procs=0 --max-infer-procs=0
--max-analyze-procs=18 --scheduler-poll-seconds=30
```

Its SCREEN/login parent processes were also visible. The qualification policy defers on a production scheduler; the guard does not assume that a zero TRAIN cap makes concurrent scheduler work acceptable. Detection occurred before the first trial completed preparation: its state and input evidence files remained empty. The failed attempt contributes no timing sample. This work sent no signal to a production process and did not start, stop or reconfigure a production scheduler. The final process census found no LSTM executable processes; qualification was still not resumed after the captured conflict.

The exact rejection and command lines remain in `DerivedData/ExpertAdvisor/phase25b3-benchmark.log`.

## Authoritative configuration and historical provenance

Read-only inspection selected completed **experiment 746**, final model **2090**, with the following persisted contract:

| Configuration | Value |
|---|---|
| Symbol / horizon | `cadchfrmp` / 4 |
| Semantic layout / model input width | 13 / 171 |
| Hidden width / sequence length / layers | 64 / 64 / 1 |
| Production training range | `2010-01-01` through `2025-01-01`, using the established civil/inclusive candlestick reader |
| Production completed / target epochs | 20 / 20 |
| Production optimizer update count | 28,900 |
| Fresh initialization seed | 1002 |
| Warmup | `full_history_warmup`; query start `-infinity` |
| Donchian | Enabled, lookback 20 |
| Objective | `legacy_first_hit_weighted_ce_v1`; hash `fnv1a64:65818f2e1fa1a324` |
| Target | UpNeutralDownReturn; existing strict high/low first-hit rule, upward tie rule; threshold float `0.0007999999797903001` |
| Class weights / normalization version | Down/neutral/up = 1/1/1; normalization version 1 |
| SGD base learning rate | `0.0003333333588670939`, confirmed in captured fresh-model state |
| Core / directional head weight / head bias multipliers | 120 / 25 / 2.5 |
| Gradient clipping | Existing componentwise clipping at 10 after normalization |
| Target normalization | Scale 1, bias 0, z-score off, mean 0, std 1 |
| Economic calendar | Immutable finalized snapshot 1, hash `fnv1a64:67610f94f5c8e7cc`, 2,601 canonical events |
| Original experiment source commit | `0eb13a7ddc44a6b47e60152a0ad1d761644c2372` |

The experiment's concrete Fibonacci-lifecycle feature-ablation mask is preserved through `ReadPersistedModelMaterialization`; its exact 38-feature expression is retained in `Phase25B3/Checkpoint/baseline-mask.txt`. It is not replaced by an empty mask or a wildcard. Feature definitions, ordering, warmup, normalization, target generation and appended return-feature construction use existing implementations.

The pilot called the existing `ModelInputPreparation::Prepare` service with the persisted range, Donchian configuration, full-history warmup and snapshot identity. It loaded **369,905 actual historical 15-minute rows**, with logical output start index **0**. Captured source `PriceTP` bounds are **2010-01-03T22:00:00Z** through **2024-12-31T21:45:00Z**. These timestamps preserve the existing historical timestamp conversion.

The full canonical physical feature matrices plus raw OHLC and target timestamps were serialized outside training timers. Their SHA-256 is:

```text
244b20028c2985d1edf7192918712a97afe348b1ec4ca7d3da69763d69e9e932
```

No synthetic input tensors were used. The prepared inputs are a current read-only extraction; the original experiment does not provide an immutable market-data hash that proves the tick/materialized-view corpus has remained unchanged since its original execution. The calendar snapshot is immutable and its identity was verified.

## Prepared benchmark methodology

The standalone fixture links unchanged `LSTM.cpp`, `Tensor.cpp`, canonical input preparation, market/economic readers and `MetalForwardAffine.mm`. Once preparation finishes, its PostgreSQL connections are closed and `CalculateBatch` performs database-free training. No production training worker is launched.

Both affine paths use the same timing executable, Release libraries/kernels, seed, objective, ablation mask, parameters, chronological input order and optimizer configuration. Each path runs in a fresh process because affine selection is cached. Runtime diagnostic logging and numerical observers are absent from the timing build; existing remaining diagnostic computation is retained.

The configured primary run has five independent pairs. Odd pairs use MetaNN then combined; even pairs reverse that order. Each process takes eight warmup and 120 measured updates from the first 128 chronological outer batches of the full prepared tensor. Each outer batch has 256 rows and 189 candidate windows at T=64 / horizon=4; the existing internal window chunks are 128 and 61. There is one SGD update per successful outer `CalculateBatch` call. This planned prefix qualification does not rerun the production experiment's complete 20 epochs.

`steady_clock` surrounds each complete `CalculateBatch` call. Summed durations report active training wall time, excluding preparation and between-update evidence writing; process elapsed time and preparation time are recorded separately. Internal chunk timings are not separately instrumented. CPU usage is process user+system time divided by timed call duration, with 100% representing one CPU core. Each update captures resident memory and Metal `currentAllocatedSize`; process peak RSS uses macOS `getrusage` bytes.

An exclusive lock shared with Phase 25B-2 prevents overlapping fixture runners. Safety checks run before/after each process and every five seconds during execution, inspecting production/fixture processes, the launch service, Ollama and memory pressure. VM/swap snapshots and system-wide AGX utilization counters are retained. Desktop activity and the counters' sampling interval remain limitations; the counters cannot attribute utilization to a particular process.

## Measurements actually completed

| Run | Path | Warmup / measured updates | Result |
|---|---|---|---|
| Untallied cold pilot | MetaNN | 0 / 1 | Passed finite-value, accepted-window and optimizer-counter assertions |
| Pair 1, first path | MetaNN | Planned 8 / 120 | Aborted in canonical preparation on production scheduler detection; no update sample |
| Pair 1, second path | Combined | Planned 8 / 120 | Not started |
| Pairs 2–5 | Alternating | Planned 8 / 120 per path | Not started |

Pilot safety timestamps span **20:37:55–20:40:03 UTC**:

| Pilot measurement | Value |
|---|---|
| Canonical preparation | 123.962 seconds |
| Process elapsed time | 127.946524 seconds |
| Cold complete outer-batch update | 992.070042 ms |
| Cold optimizer updates/second | 1.007993 |
| CPU utilization during the timed call | 54.4408% of one core |
| Accepted windows / loss / update count | 189 / `1.09900808334351` / 1 |
| Peak resident memory | 4,469,915,648 bytes |
| Resident memory before training | 4,429,709,312 bytes |
| Metal allocations before training | 3,342,991,360 bytes |
| Metal allocations after update | 3,343,777,792 bytes |
| Metal allocations after model release, while the Tensor remained alive | 3,343,532,032 bytes |
| Memory pressure / swap | Normal level 1 throughout retained snapshots; zero swap usage |
| Sampled system-wide device utilization | 0–10% across preparation, the single cold update and completion; not a training-only estimate |

This cold pilot is not a throughput qualification or a paired result. No speedup, confidence interval, paired win count or production throughput conclusion is calculated from it.

## Numerical, checkpoint and memory qualification

The pilot captured all six parameter families, public recurrent state, loss, learning rate, target mapping and counters. Every serialized matrix value was checked finite. Initial core matrix SHA-256 is `fca6e5acdad9af435ac890304a650f1d27e80975c6896add3668c6fa09ec1ada`; full initial state SHA-256 is `d46bee1b3955df1644ec145322c34c06a8ea55e99a4970c74cd4ab946bf2cc10`. Full initial-plus-one-update state evidence SHA-256 is `2b8051de2af461532f9e5b46ef161ed659382d10665a83a3f21371e42560a12b`.

| Required comparison | Status |
|---|---|
| Initial parameters, losses, final parameters, learning rates and update counters across affine paths | Blocked; no combined run |
| Forward recurrent caches and pre-/post-clipping gradients | Observer build prepared; no observer run executed |
| Exact difference locations, magnitudes and reproducibility | Not measured; absence of a comparison is not zero numerical difference |
| A: original checkpoint into original | Not executed |
| B: combined checkpoint into combined | Not executed |
| C: original checkpoint into combined | Not executed |
| D: combined checkpoint into original | Not executed |
| Uninterrupted/restored trajectory and optimizer/input-contract compatibility | Not established |
| Sustained resident memory, swap growth or unbounded retention | Not measured |
| Temporary allocation churn and within-call Metal peaks | Not instrumented by boundary snapshots |

The bulk of Metal allocations existed before the first training update, with the full historical Tensor retained. The small post-release increase is compatible with caching, but one update cannot establish its cause or stability. No allocation growth is classified as a memory leak, and no leak-free conclusion is claimed.

### Isolated checkpoint preparation and cleanup

Checkpoint preparation reused the established Phase 24Q isolation/routing approach, with fresh private SCRAM credentials, a separate TCP port **55483**, and data directory `DerivedData/ExpertAdvisor/Phase25B3/Checkpoint/pgdata`. Its server identifier was **7694771646283355739**, postmaster **13930**. The port, directory, user and server identity were verified before fixture administration. No production credentials or workflow rows were copied.

The isolated instance received exact historical candlesticks through binary COPY, the existing candlestick functions/materialized-view read path, and economic reference tables including immutable snapshot 1. The supported isolated protocol-cutover and queue workflows succeeded. One benchmark-owned experiment was queued for a bounded real-data checkpoint range, `2010-01-01`–`2010-04-01`, two epochs, seed 1002, the exact mask, objective and calendar identity. This setup is not a completed checkpoint test, nor the original experiment's full-duration configuration. Native selector propagation and restoration remain unverified.

Before removal, read-only checks showed **zero models, zero worker attempts, zero active worker bindings and vacant private scheduler authority**. The exact postmaster PID/directory and server identifier were rechecked, only that postmaster was stopped, and its absence was verified. Private databases, socket directory, routing fixture and temporary credential files were then removed. Sanitized reference data, setup source and logs remain as qualification evidence. No production process was signaled.

## Comparison and recommendation

[Phase 25B-2](LSTMTrainingThroughputBenchmark.rst) reported +5.726% complete-training throughput with 10/10 paired wins and bitwise equivalence on its warmed synthetic, 128-window, horizon-2 fixture. This attempt uses real CADCHFRMP data, horizon 4, its persisted ablation mask and 189-window outer batches. Its only completed update is cold. Comparing the cold pilot latency with Phase 25B-2's warmed means would be invalid.

**Generalization remains unresolved.** The prior findings are neither confirmed nor contradicted by this attempt. Keep the current runtime default unchanged. Production readiness requires the five clean paired trials, full numerical comparisons, all four isolated checkpoint restoration cases and sustained-memory measurements. Resume qualification only in an uncontended scheduler/GPU window; this report does not authorize changing production workload state.

## Commands and build results

Executed build and fixture commands:

```bash
bash Scripts/build-lstm-release.sh
# Script's normal build, with stable DerivedData and publication disabled:
xcodebuild -project "$PWD/ExpertAdvisor.xcodeproj" -scheme "LSTM Release" \
  -configuration Release -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" \
  CODE_SIGNING_ALLOWED=NO PUBLISH_CANONICAL_LSTM_RELEASE=NO \
  PUBLISH_CANONICAL_BINARY=NO build

xcodebuild -project "$PWD/ExpertAdvisor.xcodeproj" -scheme "LSTM Train Worker" \
  -configuration Release -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" \
  CODE_SIGNING_ALLOWED=NO PUBLISH_CANONICAL_LSTM_RELEASE=NO \
  PUBLISH_CANONICAL_BINARY=NO PUBLISH_SEMANTIC_WORKER=NO build

bash Tests/LSTMProductionTrainingQualification.sh
PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMProductionTrainingQualification.py --pilot
PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMProductionTrainingQualification.py
git diff --check
bash -n Tests/LSTMProductionTrainingQualification.sh
# Python runner syntax was also checked with ast.parse, without executing it.
```

Both ordinary Xcode builds succeeded. Release provenance enforcement was retained. The TRAIN target was built from the unchanged clean baseline by temporarily preserving the three new untracked fixture files under ignored DerivedData and restoring them afterward. No Clean, output-directory override, provenance bypass or publication occurred. The development TRAIN binary contains `EA_LSTM_FORWARD_AFFINE` / combined selection; it was not run for training.

The standalone timing and observer builds succeeded with `-O3 -DNDEBUG`, C++20, ARC, warnings-as-errors and the same narrow legacy/MetaNN warning exclusions as Phase 25B-2; their final build log contains no diagnostics. An initial incorrect calendar identity member reference was corrected before the successful builds/pilot. The Xcode builds retain existing warning defects: malformed LLVM23 toolchain discovery, libpqxx deprecations/unused diagnostics and the libomp deployment mismatch (2 warning occurrences in the incremental CLI build; 660 in the TRAIN build). This is not a warning-clean native build, and no adjacent warning refactor was performed in this qualification-only increment. Static shell/Python syntax checks passed; they did not launch qualification workloads.

Isolated setup commands used `Phase25B3/Checkpoint/setup.py`, `import-history.py`, `native-fixture.py` and `isolated.py cli complete-cutover --complete-scheduler-protocol-cutover --yes`, followed by the supported `--queue-experiment` workflow. Sanitized command arrays and results are retained in `Checkpoint/commands.jsonl`. Cleanup ownership assertions and `pg_ctl` results are in `Checkpoint/cleanup.json` and associated logs. No `--evidence`, `--sustained`, native checkpoint training or broader GPU regression command was run after the conflict.

## Artifacts, changes and Git review

Retained local evidence is under `DerivedData/ExpertAdvisor/Phase25B3`: `qualification-status.json`, `provenance.json`, `pilot-initialization.json`, pilot input/state/log/safety files, timing/observer binaries, executed source snapshots and sanitized checkpoint setup/cleanup evidence. Build/run logs are `phase25b3-release-build.log`, `phase25b3-train-worker-build.log`, `phase25b3-fixture-build.log`, `phase25b3-pilot.log` and `phase25b3-benchmark.log` under stable DerivedData.

New files:

- `Tests/LSTMProductionTrainingQualification.mm`: canonical read-only preparation and unchanged database-free training, timing/state/resource evidence and optional existing numerical observers.
- `Tests/LSTMProductionTrainingQualification.sh`: development-only standalone builds using stable Release products.
- `Tests/LSTMProductionTrainingQualification.py`: sequential alternating-pair runner with resource monitoring and workload-conflict termination of only its own fixture.
- `docs/architecture/LSTMProductionTrainingQualification.md`: this blocked qualification report.

No existing tracked file was modified. Training mathematics, semantic definitions, checkpoint formats, default affine selection, production data and worker registries were not changed. Nothing was committed, pushed, merged or deployed.

Final `git status --short`:

```text
?? Tests/LSTMProductionTrainingQualification.mm
?? Tests/LSTMProductionTrainingQualification.py
?? Tests/LSTMProductionTrainingQualification.sh
?? docs/architecture/LSTMProductionTrainingQualification.md
```

`git diff --stat` produces no output because all additions remain untracked. `git diff --check` passes. Production and shared MetaNN `git status --short` produce no output.
