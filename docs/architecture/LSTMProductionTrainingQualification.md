# Phase 25B-3: production-representative Metal training qualification

**Status: INCONCLUSIVE for Phase 25B-3T on 2026-10-09; Phase 25B-3R remains BLOCKED. Production readiness is not established.**

## Phase 25B-3T Metal command-buffer timing investigation

**Final status: INCONCLUSIVE.** Direct timestamps rule out the
ExpertAdvisor-owned combined affine command buffer as the dominant source of
the slow Combined process. The slow process instead spent materially more time
in the surrounding forward and backward recurrent stages. MetaNN command
buffers remain intentionally uninstrumented because they are in shared MetaNN;
therefore the exact CPU-versus-GPU cause of the shared runtime regime is not
fully observable.

### Timestamp API and validation

The installed macOS 27 SDK (`MacOSX27.0.sdk`) exposes all four relevant
`MTLCommandBuffer` properties: `GPUStartTime`, `GPUEndTime`,
`kernelStartTime`, and `kernelEndTime`. An isolated Metal probe, run with GPU
access, returned nonzero ordered values for an empty command buffer:

| Field | Probe result |
|---|---:|
| GPU start → end | 208 ns |
| Kernel start → end | 14.875 µs |
| Timestamp validity | all nonzero; both intervals ordered |

The probe established API availability and plausible values. CPU timestamps use
`std::chrono::steady_clock`; Metal GPU timestamps use Metal's GPU clock. The
diagnostic never subtracts those clocks or labels CPU wait as GPU execution.

### Instrumentation and controlled run

The isolated CADCHFRMP fixture used the Phase 25B-3S configuration and ran
three fresh alternating pairs with 8 warmup and 64 measured updates. The
opt-in instrumentation is disabled unless
`EA_LSTM_COMMAND_BUFFER_TIMING` is set. It adds no command buffers and no
additional waits. For the combined path it records CPU command-buffer creation,
encoding, commit, and `waitUntilCompleted` intervals, plus GPU and kernel
timestamp durations. MetaNN command-buffer internals are reported as
unobservable; its existing wall-clock hotspot scopes remain available.

| Pair | Order | MetaNN ms/update | Combined ms/update | Combined change |
|---:|---|---:|---:|---:|
| 1 | MetaNN → combined | 760.973 | 729.551 | +4.126% |
| 2 | Combined → MetaNN | 759.978 | 710.768 | +6.475% |
| 3 | MetaNN → combined | 775.064 | 986.502 | −27.280% |

Pair 3 reproduced a slow Combined regime after the first 16-update block:
block means were 746.7, 1094.7, 1139.9, and 964.8 ms/update. All six
processes produced the same input and final-state hashes; Metal allocation and
swap remained stable.

### CPU/GPU timing breakdown

The following values are per combined affine command buffer, averaged across
18,432 buffers (72 updates including warmup). CPU and GPU durations are shown
separately and are not subtracted across clock domains.

| Combined process | CPU create | CPU encode | CPU commit | CPU wait | GPU execution | GPU kernel |
|---|---:|---:|---:|---:|---:|---:|
| Pair 1 fast | 0.397 µs | 17.182 µs | 1.925 µs | 159.137 µs | 31.086 µs | 22.062 µs |
| Pair 2 fast | 0.391 µs | 17.069 µs | 1.891 µs | 157.801 µs | 30.014 µs | 22.081 µs |
| Pair 3 slow | 0.388 µs | 17.089 µs | 2.011 µs | 155.817 µs | 31.889 µs | 18.653 µs |

All 18,432 pair-3 command buffers had valid ordered GPU timestamps. CPU
creation, encoding, commit, and wait timings did not increase in the slow
process. GPU execution increased only about 6% while kernel duration decreased
about 15%; neither explains the roughly 35% total-update slowdown.

### Stage comparison and conclusion

Existing hotspot totals for pair 3 versus the two fast Combined processes were:

| Stage total per process | Fast Combined mean | Slow Combined pair 3 | Change |
|---|---:|---:|---:|
| Forward step | 21,808 ms | 28,700 ms | +31.6% |
| Affine scope | 3,319 ms | 3,274 ms | −1.4% |
| Backward step | 25,145 ms | 32,766 ms | +30.3% |
| Backward GEMMs | 8,733 ms | 10,229 ms | +17.1% |
| Gradient clipping | 2.254 ms | 2.183 ms | −3.2% |
| Optimizer | 0.486 ms | 0.429 ms | −11.7% |

**Measured:** the combined affine command buffer is not the dominant source of
the slow Combined regime; CPU submission/encoding/wait timings are stable,
GPU timestamps are valid, and the slow process spends extra wall time in
forward and backward scopes. **Inferred:** the additional latency is in other
recurrent command buffers or host/runtime scheduling around them. **Unknown:**
whether that latency is GPU execution, CPU scheduling, queue scheduling, or a
Metal synchronization effect, because MetaNN and the remaining recurrent
operations were not given command-buffer timestamp probes. No thermal,
Spotlight, or GPU scheduling attribution is made.

Numerical equivalence remains bitwise exact, with matching diagnostic input and
final-state hashes; checkpoint restoration remains passed. The next action is a
focused probe of the recurrent gate-state and backward command buffers using
ExpertAdvisor-owned timing hooks where possible, or an explicit Metal capture
if shared MetaNN visibility is required. Keep MetaNN as the production default
and do not begin Phase 25B-4.

## Phase 25B-3S Metal performance diagnostics

**Final status: INCONCLUSIVE.** The diagnostic run did not reproduce the
combined slow regime within three short pairs, so it does not isolate a
combined-specific bottleneck. It did reproduce a slow regime in the MetaNN
path under the same workload, which is evidence that the regime is shared by
the training process and Metal runtime rather than caused by the combined
affine operation alone.

### Evidence analysis before new runs

The 504-update traces contain abrupt transitions. In sustained pair 3,
combined moved from roughly 715–729 ms/update to roughly 1.0–1.15 s/update
around update 256 and later returned to the fast range. In sustained pair 4,
combined briefly reached about 990 ms/update around update 72 before returning
to about 720–750 ms/update. Pair 2 MetaNN also transitioned from about
1.4 s/update to about 0.76–0.78 s/update after update 320. These are step-like
changes, not monotonic warmup or gradual memory growth. Metal allocation was
constant at 3,343,761,408 bytes in the completed processes.

Both implementations perform synchronous Metal work. The combined affine
path encodes MPS GEMM and row-bias in one command buffer, commits it, and calls
`waitUntilCompleted`. The MetaNN path performs its GEMM and bias stages through
separate command buffers, each followed by `waitUntilCompleted`. The existing
timing records did not contain command-buffer or GPU timestamps, so they could
not previously separate GPU execution from host submission and waiting.

### Diagnostic method

The isolated read-only fixture used experiment 746’s CADCHFRMP data, horizon 4,
layout 13, width 171, sequence length 64, seed 1002, and the same canonical
input hash. Three fresh-process pairs ran 8 warmup and 64 measured updates in
alternating order. An opt-in profiler enabled the existing `HotspotScope`
timers; it added CPU elapsed-time counters only and no extra Metal waits. The
default fixture and production runtime remain unchanged. Profiles include
`forward_step_batch`, `matmul_bias_gate_preactivation`, `backward_step_batch`,
`backward_gemms`, `gradient_clipping`, and `optimizer_update`. The profiler
reports wall time, not GPU hardware timestamps.

| Pair | Order | MetaNN ms/update | Combined ms/update | Combined change |
|---:|---|---:|---:|---:|
| 1 | MetaNN → combined | 771.334 | 729.576 | +5.407% |
| 2 | Combined → MetaNN | 1061.574 | 744.568 | +29.863% |
| 3 | MetaNN → combined | 767.204 | 698.039 | +9.016% |

All six diagnostic processes used identical input and final-state hashes.
Combined won all three short pairs. The slow regime appeared in pair 2
MetaNN, with 16-update block means of 1183.1, 1190.7, 1091.3, and 781.3 ms;
combined remained between 713 and 863 ms in that pair. No process showed swap
growth or Metal allocation growth, and the safety snapshots remained clear of
production workers and competing training processes.

### Stage timing

Hotspot totals below are per diagnostic process across 64 updates. “Forward
rest” is `forward_step_batch` minus the affine scope. The totals overlap by
design because nested scopes are reported separately; they are used to locate
changes, not to sum to the fixture wall time.

| Regime | Total update mean | Forward step | Affine scope | Forward rest | Backward step | Backward GEMMs | Clip | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaNN fast (pairs 1,3) | 769.269 ms | 24,495.7 ms | 5,576.0 ms | 18,919.7 ms | 25,786.3 ms | 8,933.3 ms | 2.236 ms | 0.485 ms |
| MetaNN slow (pair 2) | 1061.574 ms | 33,254.1 ms | 5,257.3 ms | 27,996.8 ms | 35,217.7 ms | 10,791.2 ms | 2.133 ms | 0.423 ms |
| Combined (pairs 1–3) | 724.061 ms | 21,889.4 ms | 3,276.6 ms | 18,612.8 ms | 25,365.5 ms | 8,849.3 ms | 2.196 ms | 0.476 ms |

The slow MetaNN process spent about 8.8 seconds more in forward-step scopes
and 9.4 seconds more in backward-step scopes than fast MetaNN processes. Its
affine scope was slightly faster, not slower. Clipping and optimizer scopes
were negligible and unchanged. This excludes the affine call, clipping, and
optimizer as the cause of the observed short-run slow regime. It supports a
shared runtime-scheduling or host/GPU execution-state hypothesis affecting
multiple recurrent stages, but the data do not distinguish CPU scheduling,
Metal queue scheduling, GPU occupancy, or thermal behavior. No thermal or GPU
hardware conclusion is inferred from wall time alone.

No GPU start/end timestamps were exposed by the existing fixture profiler, so
CPU submission/wait time and GPU execution time remain combined in each
hotspot. Separating those components requires a follow-up Metal command-buffer
timestamp probe that covers both paths without modifying shared MetaNN.

### Correctness and next action

Each diagnostic pair produced matching input and final-state SHA-256 values;
the prior observer qualification remains bitwise exact with zero differences,
and checkpoint restoration remains passed. The diagnostic run made no changes
to training mathematics, tensor lifetimes, checkpoint behavior, production
repositories, or the default runtime.

The recommended next action is a narrowly scoped command-buffer timestamp
study, first for one fast and one slow process when the slow regime is
actually present. Keep the production default on MetaNN and do not begin
Phase 25B-4. The current evidence is **INCONCLUSIVE**, not a basis for an
optimization or a production runtime change.

## Phase 25B-3R sustained-performance requalification

The six-pair sustained runner was allowed to continue without restarting or
changing any trial. Five pairs completed 8 warmup plus 504 measured updates;
pair 6 reached canonical dataset preparation for its combined process and then
stopped before its first measured update. Its partial input and state files,
log, and all five completed pair records are preserved under
`DerivedData/ExpertAdvisor/Phase25B3/SustainedR`. Because the sixth pair has no
timing sample, this is incomplete evidence and the final status is **BLOCKED**.

The controlled preflight found no LSTM worker, scheduler worker, or Ollama
model; production PostgreSQL remained read-only. CoreSpotlight was resident but
near zero CPU, aggregate host samples were predominantly idle, memory pressure
was level 1, and swap remained 12.12 MB without growth. No competing training or
GPU workload was observed. This makes substantial background interference an
unlikely explanation for the completed measurements, while not proving that
driver, thermal, or Metal runtime state was identical between processes.

### Completed sustained pairs

Training time excludes canonical preparation. The order alternated as
MetaNN → combined, then combined → MetaNN. “Combined change” is positive when
combined is faster.

| Pair | Order | MetaNN ms/update | Combined ms/update | Combined change | MetaNN early → late | Combined early → late |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MetaNN → combined | 801.545 | 1363.075 | −70.056% | 781.624 → 821.467 | 1371.010 → 1355.140 |
| 2 | Combined → MetaNN | 1207.819 | 1396.388 | −15.612% | 1406.116 → 1009.523 | 1395.387 → 1397.390 |
| 3 | MetaNN → combined | 751.839 | 828.689 | −10.222% | 756.361 → 747.316 | 718.717 → 938.662 |
| 4 | Combined → MetaNN | 786.179 | 733.598 | +6.688% | 782.697 → 789.660 | 724.034 → 743.163 |
| 5 | MetaNN → combined | 787.728 | 765.675 | +2.799% | 796.270 → 779.186 | 785.661 → 745.689 |
| 6 | Combined → MetaNN | — | — | — | not completed | not completed |

Across the five completed pairs, MetaNN averaged **867.022 ms/update**
(1.1534 updates/s) and combined averaged **1017.485 ms/update** (0.9828
updates/s). Combined won 2/5 pairs. The mean of per-trial p90/p95/p99
latencies was 935.0/950.1/995.3 ms for MetaNN and 1119.7/1132.7/1170.8 ms
for combined. Early versus late means were 904.6 → 829.4 ms for MetaNN and
999.0 → 1036.0 ms for combined. These aggregate values are descriptive only:
the trial means have standard deviations of 191.4 ms and 332.7 ms, and the
first two pairs were in a materially slower state than pairs 3–5 for both
implementations.

The order-balanced data therefore do not reproduce a stable combined-only
regression. They also do not establish a sustained combined advantage. Pair 1
and pair 2 show combined at 1.36–1.40 s/update; pair 3–5 show both paths near
0.73–0.83 s/update, with MetaNN itself reaching 1.21 s/update in pair 2.
The pattern is consistent with process-order and Metal runtime state (queue,
command-buffer scheduling, driver/thermal state) dominating this workload.
The timing fixture measures the complete update; source inspection localizes
the implementation difference to the forward affine call, but this run does
not separate GEMM execution from synchronization or command-buffer wait time.
No optimization or diagnostic change was made, and no shared MetaNN source was
modified.

Every completed pair used the same input SHA-256
`244b20028c2985d1edf7192918712a97afe348b1ec4ca7d3da69763d69e9e932` and final
state SHA-256
`ac8f8ab2bb5194dd18bca2c0424978a9ba7d8efa7d2678417a174acd7e0c5d67`.
The existing Phase 25B-3 observer result remains bitwise exact:
`numerical-equivalence.json` reports `bitwise_equal: true`, zero differences,
and maximum absolute difference zero. The prior four checkpoint restoration
cases also remain byte-for-byte equivalent. Metal allocation stayed at about
3.344 GB in every completed process; measured RSS declined from about 4.436 GB
to about 3.94 GB and showed no unbounded growth. These results do not indicate
memory pressure or a leak.

### Recommendation for Phase 25B-4

Do not start Phase 25B-4 from this result and do not change the production
default. A future qualification should first complete a fresh six-pair run in
a controlled window, then add disabled-by-default lightweight stage timing to
separate forward affine work from command-buffer synchronization. The next
study should retain alternating order and record thermal/driver state where
available. The present evidence supports neither PASS nor FAIL for sustained
performance, so the required status is **BLOCKED**.

The qualification was completed after the first guarded attempt was stopped when a production scheduler appeared during preparation. The resumed run completed five alternating real-data pairs, numerical observation, all four checkpoint restoration cases, and a 504-update sustained-memory run for each affine path. No production worker, scheduler, database row, checkpoint, registry, or shared MetaNN source was modified.

## Baseline and safety

| Item | Verified value |
|---|---|
| Development branch / Phase 25B-3S start | `dedicated-train-layout-rollover-squashed-v1` / `f39cda1f00c804596c1d23a85590b910dfca668d` |
| Phase 25B-3T starting commit | `2f71b3519690491dd22879e2d4a9d756b710085e` |
| Phase 25B-1 / 25B-2 | `091234b54fad122b377aec0044b7ab9dc19e6cb7` / `4493601b30fd70dd44d3a7567d6d53f8510e9baf` ancestors |
| Production / shared MetaNN | `/Volumes/Developer SSD/ExpertAdvisor` and its `MetaNN/MetaNN`; clean before and after |
| Production database | Explicit `BEGIN READ ONLY`, `default_transaction_read_only=on`; no writes |
| Initial competing workloads | No LSTM worker or scheduler; Ollama had no loaded model/runner |
| Hardware state | 48 GiB RAM; pressure level 1; host swap later showed 12.12 MB used, with no benchmark-process growth |
| Build | Stable `DerivedData/ExpertAdvisor`; Release provenance retained; no Clean, publish, deploy, or output-directory override |

The earlier production scheduler event is retained as a historical safety stop in `phase25b3-benchmark.log`. The guard signaled only its own fixture. The resumed qualification ran only while production scheduler and worker censuses were clear. The temporary checkpoint database used port `55483`, a separate data directory and generated credentials; it was stopped and removed after a read-only ownership check. Its scheduler had zero active worker attempts at shutdown.

## Production configuration and data provenance

The authoritative source was completed production experiment **746**, final model **2090**, read through PostgreSQL without mutation:

- Symbol `CADCHFRMP`, horizon 4, semantic layout 13, model input width 171.
- Hidden width 64, one layer, sequence length 64; training range 2010-01-01 through 2025-01-01.
- Full-history warmup, Donchian enabled with lookback 20, objective `legacy_first_hit_weighted_ce_v1` (`fnv1a64:65818f2e1fa1a324`).
- Seed 1002; SGD; base learning rate `0.0003333333588670939`; multipliers 120 / 25 / 2.5; clipping 10.
- Immutable calendar snapshot 1, hash `fnv1a64:67610f94f5c8e7cc`, containing 2,601 events.
- The persisted 38-feature ablation mask is retained in `DerivedData/ExpertAdvisor/Phase25B3/Checkpoint/baseline-mask.txt`.

Canonical preparation used the existing input-preparation service and real historical data: **369,905** 15-minute rows, from `2010-01-03T22:00:00Z` through `2024-12-31T21:45:00Z`. The serialized physical input tensor and raw target inputs have SHA-256 `244b20028c2985d1edf7192918712a97afe348b1ec4ca7d3da69763d69e9e932`. No synthetic tensor was presented as production data. Preparation was outside training timers.

## Benchmark methodology

`EA_LSTM_FORWARD_AFFINE=metann` and `combined` used the same Release fixture, real input tensor, seed, parameters, objective, chronological minibatch order, optimizer, hardware and process isolation. Odd pairs ran MetaNN first; even pairs ran combined first. Each process used 8 warmup and 120 measured `CalculateBatch` updates, each processing 189 windows. Training wall time excludes canonical preparation. CPU time, RSS, Metal `currentAllocatedSize`, process peak RSS, pressure, swap and safety censuses were recorded. No GPU workload overlapped the pairs.

## Paired timing results

| Pair | Order | MetaNN ms | Combined ms | Combined improvement |
|---:|---|---:|---:|---:|
| 1 | MetaNN → combined | 770.604 | 708.060 | 8.833% |
| 2 | Combined → MetaNN | 1,275.039 | 706.806 | 80.394% |
| 3 | MetaNN → combined | 1,270.933 | 1,268.898 | 0.160% |
| 4 | Combined → MetaNN | 1,297.168 | 1,267.069 | 2.376% |
| 5 | MetaNN → combined | 876.922 | 754.187 | 16.274% |

Aggregate means were MetaNN **1,098.133 ms/update** and combined **941.004 ms/update**, corresponding to **0.9106** versus **1.0627 updates/s**, **+16.698%** combined throughput, and five combined wins. Trial-to-trial standard deviations were 253.466 ms and 299.101 ms; CPU utilization averaged 45.67% and 51.52%. Input hashes and loss/state hashes matched in every pair.

The result is a real-data prefix qualification, not a claim that every production run improves by 16.7%. The sustained run below demonstrates material order/workload sensitivity.

## Numerical equivalence

The observer run compared forward recurrent caches, gradients before clipping, gradients after clipping, losses, recurrent states, parameters, learning rate and counters. Both paths produced 16,384 forward-cache records, 128 pre-clip records and 128 post-clip records, totaling 1,172,104,192 observed float elements and 4,691,543,936 bytes. Both raw observer streams had SHA-256 `6d1000634064f641c805c0a29be9c4d5b27cd0f567534ffb8f48837afad6797e`.

`numerical-equivalence.json` reports `bitwise_equal: true`, `observer_on_off_equal: true`, zero numerical differences and maximum absolute difference zero. No gradient value exceeded the clipping threshold in this prefix, so clipping behavior was equivalently exercised but not stress-tested with clipped elements.

## Checkpoint restoration

An isolated PostgreSQL instance copied the historical candlestick and economic reference data, used the exact feature mask and calendar identity, and ran the supported scheduler/checkpoint workflow. Fresh source experiments were 13 (MetaNN, epoch-1 model 30, final 32) and 14 (combined, epoch-1 model 33, final 35). Each case had its own restored experiment and persisted final model:

| Case | Restore | Private experiment | Path after restore | Result |
|---|---|---:|---|---|
| A | Original checkpoint → original | 15 | MetaNN | 24 resumed updates bitwise equal |
| B | Combined checkpoint → combined | 16 | Combined | 24 resumed updates bitwise equal |
| C | Original checkpoint → combined | 17 | Combined | 24 resumed updates bitwise equal |
| D | Combined checkpoint → original | 18 | MetaNN | 24 resumed updates bitwise equal |

The fixture verified parameter matrices, recurrent state, optimizer state, learning rate, counters, objective, target normalization, input width, hidden width, semantic layout and calendar identity before and after restoration. Uninterrupted and restored trajectories matched byte-for-byte. Checkpoint formats were not changed. The private database, routing fixture and credentials were removed after verifying its port, data directory and system identifier.

## Sustained memory and longer-run behavior

Each path ran 8 warmup plus **504 measured updates** (512 total) over the same canonical data. MetaNN mean measured latency was **751.670 ms/update** (1.3304 updates/s; 56.98% CPU); combined was **988.555 ms/update** (1.0116 updates/s; 48.27% CPU). Thus the longer run favored MetaNN by approximately 31.5% on this execution, despite the five-pair prefix favoring combined. This is the main performance-generalization limitation.

Metal allocation boundaries were stable: about **3,343,761,408 bytes** during measured training for both paths, returning to about **3,343,515,648 bytes** after model release. RSS did not grow without bound: MetaNN measured RSS ranged from approximately 3.961 to 4.434 GB and combined from 3.958 to 4.434 GB, with both ending below their first measured sample. The observed retained allocation is consistent with the canonical tensor and allocator pooling; no leak was observed. Host pressure stayed at level 1 and the small 12.12 MB host swap observation did not increase during the runs. Boundary sampling cannot exclude transient within-kernel peaks.

## Comparison, recommendation and limitations

Phase 25B-2 measured **+5.73%** complete-training throughput with 10/10 wins and bitwise equivalence on its warmed synthetic fixture. Phase 25B-3 extends the evidence to real CADCHFRMP data, horizon 4, the persisted ablation mask, independent checkpoint persistence/restoration, and 504-update memory runs. It confirms mathematical equivalence, checkpoint compatibility and memory stability. It does **not** establish a universal throughput advantage: the five-pair prefix favored combined, while the longer sustained execution favored MetaNN, with substantial trial variance.

Recommendation: keep the current default runtime selection unchanged. The combined implementation is qualified for further controlled use and has no numerical, checkpoint or memory blocker, but production-wide performance readiness should remain conditional until an order-balanced sustained study explains the reversal and the existing Release warning debt is reviewed. No production deployment or scheduler change is authorized by this report.

## Commands and files changed

Successful checks included the stable Release and TRAIN builds, standalone fixture builds, the original five-pair timing, `--evidence`, `--sustained`, isolated checkpoint suite, the Phase 25B-3R sustained-pairs runner, the Phase 25B-3S three-pair diagnostic runner, the Phase 25B-3T timestamp probe and three timestamp pairs, `git diff --check`, shell syntax and Python AST checks. The first guarded attempt was stopped before producing a timing sample; it is not included in the results above. The Phase 25B-3R runner was not restarted after pair 6 stopped before measurement.

Changed files:

- `Tests/LSTMProductionTrainingQualification.mm`
- `Headers/MetalForwardAffine.hpp`
- `LSTM/MetalForwardAffine.mm`
- `Tests/LSTMProductionTrainingQualification.sh`
- `Tests/LSTMProductionCheckpointQualification.mm`
- `Tests/LSTMProductionTrainingEvidence.py`
- `Tests/LSTMProductionTrainingQualification.py`
- `docs/architecture/LSTMProductionTrainingQualification.md`

No production or shared MetaNN file changed. No checkpoint format, training mathematics, semantic layout, default runtime selection, production database, worker registry, scheduler or canonical binary changed. The command-buffer timing is opt-in and disabled by default. The Phase 25B-3T instrumentation and report changes are committed locally; nothing was pushed, merged or deployed.

Generated evidence remains under `DerivedData/ExpertAdvisor/Phase25B3`.
