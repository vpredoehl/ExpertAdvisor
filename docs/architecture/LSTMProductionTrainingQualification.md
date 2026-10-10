# Phase 25B-3: production-representative Metal training qualification

**Status: BLOCKED for Phase 25B-3R on 2026-10-09. Production readiness is not established.**

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
| Development branch / HEAD | `dedicated-train-layout-rollover-squashed-v1` / `c5dbe1773b24b11aad710b44791d5fab53b83977` |
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

Successful checks included the stable Release and TRAIN builds, standalone fixture builds, the original five-pair timing, `--evidence`, `--sustained`, isolated checkpoint suite, the Phase 25B-3R sustained-pairs runner, `git diff --check`, shell syntax and Python AST checks. The first guarded attempt was stopped before producing a timing sample; it is not included in the results above. The Phase 25B-3R runner was not restarted after pair 6 stopped before measurement.

Changed files:

- `Tests/LSTMProductionTrainingQualification.mm`
- `Tests/LSTMProductionTrainingQualification.sh`
- `Tests/LSTMProductionCheckpointQualification.mm`
- `Tests/LSTMProductionTrainingEvidence.py`
- `Tests/LSTMProductionTrainingQualification.py`
- `docs/architecture/LSTMProductionTrainingQualification.md`

No production or shared MetaNN file changed. No checkpoint format, training mathematics, semantic layout, default runtime selection, production database, worker registry, scheduler or canonical binary changed. The validated Phase 25B-3R documentation and runner change are committed locally; nothing was pushed, merged or deployed.

Generated evidence remains under `DerivedData/ExpertAdvisor/Phase25B3`.
