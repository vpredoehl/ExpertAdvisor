Phase 25B-2: complete LSTM training throughput benchmark
========================================================

Completed on 2026-10-09. Combined provided a statistically consistent improvement in this database-free complete-training fixture: 1.057256x throughput (+5.726%), with 5.416% less time per update. It was faster in all ten independent paired trials. This qualifies the tested fixture and hardware; it does not establish production campaign throughput.

Environment and safety
----------------------

* Development branch: dedicated-train-layout-rollover-squashed-v1.
* Baseline: 091234b54fad122b377aec0044b7ab9dc19e6cb7. Initial development tree was clean.
* Hardware: Mac Studio, Apple M5 Max, 18 CPU cores, 40 GPU cores, 48 GiB unified memory (51,539,607,552 bytes).
* Actual installed tools: Xcode 27.0 (27A266a), Apple clang 21.0.0; macOS 27.0.1 (26A434). The supplied project instructions mentioned Xcode 26.5.
* Shared MetaNN source: /Volumes/Developer SSD/ExpertAdvisor/MetaNN/MetaNN, followed through the existing development symlink. No dependency source was changed.
* No LSTM training, inference, analysis or scheduler process was present. launchctl showed active count 0, no scheduler PID, last exit 78 (EX_CONFIG), and job state spawn failed. Its launch configuration was not changed.
* Ollama serve remained running with zero loaded models; no runner was present. It was never terminated.
* Memory pressure level remained normal (1) in all retained safety snapshots. Initial/final memory_pressure reported 89%/90% system-wide memory free, with zero swapins/swapouts.
* The runner holds an exclusive lock, launches only one GPU fixture at a time, and checks processes, scheduler, Ollama and memory pressure before/after each fixture and every five seconds during longer runs. If a conflict appears, it defers and terminates only its own fixture.
* No LSTM_Release executable was launched. No production database, experiment, checkpoint, scheduler state, production binary or source was modified. Production and shared MetaNN Git status remained clean.

Fixture and measurement boundary
--------------------------------

* Actual unchanged EA::LSTM::CalculateBatch implementation, linked with Tensor.cpp, EconomicEventFeatures.cpp, PricePoint.cpp and MetalForwardAffine.mm. This includes feature-row/window assembly, forward propagation, output head/loss, full BPTT, normalized componentwise clipping and SGD parameter updates.
* Input width 171; hidden width 64; sequence length 64; 128 overlapping windows in one minibatch; recurrent affine dimensions 128 x 235 times 235 x 256. No width or feature layout changes.
* Tensor symbol usdcadrmp; 520 synthetic bars. For row r, close = 1.25f + 0.003f*sin(float(r)*0.7f); open = close-0.0001f; high/low = close +/- 0.0003f; tick volume = 100 + r%7; timestamp = 900*r seconds since epoch. Tensor::Add constructs the existing feature layout without database access.
* Each update uses the final 193 bars (zero-based rows 327 through 519), yielding exactly 128 valid windows with prediction horizon 2. CalculateBatch epoch argument is fixed at 2. Every update asserts 128 accepted windows, zero skipped windows, a finite loss and an incremented optimizerUpdateCount.
* Initialization seed 42, initial_long_term=1, initial_short_term=0; existing initializer and forget-gate bias offset 1.5. Only inactive output-head families are explicitly zeroed, following the existing numerical fixture pattern.
* Primary target: UpNeutralDownReturn; legacy_first_hit_weighted_ce_v1 objective, class weights 1/1/1, targetScale=1, targetBias=0, z-score off, no ablation.
* Actual optimizer is the authoritative SGD implementation. Base learning rate = 0.000333333358867094; core multiplier 120; direction-head weight/bias multipliers 25/2.5. Existing normalization and gradient scales remain unchanged. Componentwise clipping threshold is 10 after normalization; mixed-precision core accumulation remains double. No optimizer setting was overridden.
* Both paths use the same timing binary and hardware, selected with EA_LSTM_FORWARD_AFFINE=metann or combined in fresh processes. Selection is cached within a process, so the runner never switches paths in an existing process.
* LstmRuntimeDiagnosticLoggingEnabled returns false. cout is silenced with the same stream-buffer behavior as the application. Existing unconditionally computed legacy diagnostic statistics remain part of CalculateBatch; their mathematics or gating was not changed. Batch profiling is off. No numerical observer calls are compiled into training in the timing build.
* Each process performs eight warmup updates, then sixteen measured updates. Ten independent processes per path: 80 warmup and 160 measured updates per path. Models continue from warmup rather than resetting; both paths follow identical parameter trajectories.
* Odd-numbered pairs execute metann then combined; even pairs execute combined then metann. Trials ran sequentially approximately 17:54:30--17:56:26 UTC. No trial or update was discarded.
* steady_clock surrounds only CalculateBatch. Its existing synchronous completion behavior is retained. No new Metal synchronization was added. Timings include host work and the complete training update, and exclude model construction, evidence serialization, resource queries, process launch and cleanup.
* Full state/loss evidence is serialized after each update, outside its timer. It can influence caches and wall time between updates; the same work occurs for both paths. The separate observer build is used only for equivalence checks and contributes no throughput samples.

Per-trial measurements
----------------------

Latency is the arithmetic mean of sixteen complete updates in each process. RSS is lifetime peak resident memory from macOS getrusage, in bytes.

.. csv-table:: Ten alternating-order pairs
   :header: "Pair", "Order", "MetaNN ms", "Combined ms", "MetaNN updates/s", "Combined updates/s", "Speedup", "MetaNN peak RSS bytes", "Combined peak RSS bytes"

   1, "metann then combined", 230.052039, 222.930602, 4.346843, 4.485701, 1.031945, 66011136, 65077248
   2, "combined then metann", 236.109419, 220.639883, 4.235324, 4.532272, 1.070112, 65224704, 65273856
   3, "metann then combined", 234.176718, 223.158336, 4.270279, 4.481123, 1.049375, 65093632, 65372160
   4, "combined then metann", 234.840982, 221.586281, 4.258201, 4.512915, 1.059817, 65388544, 65208320
   5, "metann then combined", 233.770263, 222.380130, 4.277704, 4.496805, 1.051219, 64962560, 65257472
   6, "combined then metann", 235.747419, 225.490526, 4.241828, 4.434776, 1.045487, 65028096, 65093632
   7, "metann then combined", 239.377003, 224.087365, 4.177511, 4.462545, 1.068231, 65208320, 65126400
   8, "combined then metann", 235.995287, 222.928867, 4.237373, 4.485736, 1.058613, 65175552, 65323008
   9, "metann then combined", 240.496221, 221.683987, 4.158069, 4.510926, 1.084861, 65421312, 64995328
   10, "combined then metann", 233.994099, 222.161974, 4.273612, 4.501220, 1.053259, 65388544, 65110016

Aggregates and consistency
--------------------------

.. csv-table:: Summary
   :header: "Statistic", "MetaNN", "Combined"

   "Mean update latency (ms)", 235.455945, 222.704795
   "Median of 160 measured updates (ms)", 234.131771, 222.640833
   "Median of ten trial means (ms)", 235.294201, 222.654499
   "Sample SD of ten trial means (ms)", 2.939714, 1.371768
   "CV of ten trial means (%)", 1.248520, 0.615958
   "Sample SD of 160 measured updates (ms)", 9.974338, 5.167203
   "Updates/s from mean latency", 4.247079, 4.490249
   "Minimum trial mean (ms)", 230.052039, 220.639883
   "Maximum trial mean (ms)", 240.496221, 225.490526

Mean-latency speedup: 1.057256x; trial-median speedup: 1.056768x. Mean paired saving: 12.751150 ms/update. The 95% paired t interval is [10.432666, 15.069633] ms/update (ten pairs, df=9).

A two-sided sign-flip calculation over all 1024 paired assignments gives p=0.001953125. The paired percentile-bootstrap 95% interval for the ratio of means is [1.048815, 1.065991]x (20,000 resamples, seed 42). These statistics use independent process trials as the units; the 160 within-process updates are not treated as independent experiments. Sign-flip symmetry and approximate independence are assumptions, since order was alternated rather than randomized. Every paired difference is positive, and both intervals exclude no improvement.

Memory and Metal allocation observations
----------------------------------------

* MetaNN process peak RSS range: 61.953125--62.953125 MiB.
* Combined process peak RSS range: 61.984375--62.343750 MiB.
* Every measured trial in both paths reported Metal currentAllocatedSize = 1,359,872 bytes after warmup and after the final update, maximum observed at update boundaries = 1,376,256 bytes (including warmup), and after model release = 1,114,112 bytes.
* Metal values are device allocation snapshots in each fixture process, not allocation counts, per-operation allocation attribution or a peak measured inside CalculateBatch. Transient cache/buffer allocations are not captured by these boundary snapshots. RSS includes initialization, training, evidence writing and process lifetime; it is not exclusively GPU memory. No meaningful memory reduction is established.

Numerical equivalence
---------------------

* All twenty timed processes had exactly equal initial state bytes. Initial-state SHA-256: 331475f77025f408c2ad50cd975839c7fa50912c70f4f5307a1c64998c8ef246. Core initial parameter matrix SHA-256: 7ce733ccbf6c31fc9a586bbb8bd26aa1ac71bf364cab2c8b8d8281acc68f61ea.
* All twenty complete serialized trajectories matched exactly, including all warmup/measured losses, six parameter families, public recurrent state fields, learning rate, target scale, update counts and completedEpochs. State SHA-256: ae4992eac10a07154834b02c9927571cb101fb7d69b0469a21672f37d4510ed5. Parameters changed from their initial values; every process completed exactly 24 updates.
* Separate observer-on runs repeated the identical 24-update primary trajectory and matched every timed trajectory. Their complete raw tensor files also matched bitwise: 1536 forward-cache records per path (64 steps x 24 updates), 24 preclip and 24 postclip records with all six gradient families. Each forward record includes input, previous hidden/cell state, all four gates, new cell/hidden state and forget logits. Maximum absolute/relative differences are zero for finite compared values.
* Public prevHiddenState/prevCellState are reset by CalculateBatch; their equality alone is not the recurrent-state test. Actual nontrivial batched recurrent caches were compared through every step of the observer runs.
* The primary trajectory had no components above the clipping threshold. Every captured postclip gradient was independently checked against clamp(preclip, -10, 10), not only compared across paths.
* Three-update, 128-window validation also passed for auxiliary classification, log-return regression and percent-return regression. Both paths matched bitwise, and observer-on state matched observer-off state for each target.
* Supplementary H=64/T=64 regression runs used three windows (68 bars) and targetScale=100000 to exercise real clipping. Both log and percent cases clipped 408 components over three updates, with identical pre/postclip tensors and state trajectories. This fixture-only target mapping matches the existing numerical test pattern; it is excluded from throughput measurements.
* All matrix/loss evidence checked for finite values. Raw tensor trajectory SHA-256: d585b77ad62d362170f150a298b241adcecd71a7b98f77dcae438872fdc932b0. Full raw evidence was checked by SHA-256 and direct cmp comparisons; no tolerance was needed. Database checkpoint serialization/resumption was not exercised.

Builds, checks and retained evidence
------------------------------------

The clean baseline normal Release build succeeded before adding benchmark files. It used the existing DerivedData/ExpertAdvisor and normal Build/Products/Release paths, with unchanged provenance enforcement and no CONFIGURATION_BUILD_DIR override. All canonical publication was disabled. The standalone fixtures use existing development libraries and metallibs, C++20, -O3, -DNDEBUG, Objective-C ARC and deployment target 26.2. Their builds use warnings-as-errors and the same narrow legacy/MetaNN warning exclusions as the existing numerical fixture; they emitted no warnings.

Executed commands::

    bash Scripts/build-lstm-release.sh
    # Exact xcodebuild command used by that script:
    xcodebuild -project "$PWD/ExpertAdvisor.xcodeproj" -scheme "LSTM Release" -configuration Release -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" CODE_SIGNING_ALLOWED=NO PUBLISH_CANONICAL_LSTM_RELEASE=NO PUBLISH_CANONICAL_BINARY=NO build
    bash Tests/LSTMTrainingThroughputBenchmark.sh
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMTrainingThroughputBenchmark.py --pilot
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMTrainingThroughputBenchmark.py --validate
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMTrainingThroughputBenchmark.py
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMTrainingThroughputBenchmark.py --validate-trajectory
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/LSTMTrainingDiagnosticOverheadTests.py
    PYTHONDONTWRITEBYTECODE=1 python3 Tests/TrainingDistributionLoggingTests.py
    bash -n Tests/LSTMTrainingThroughputBenchmark.sh
    git diff --check

All final fixture builds, numerical checks, ten measured pairs, full-trajectory checks and CPU regressions passed. An initial validation assumption that the 128-window regression case must clip failed; inspection confirmed normalization kept its gradients below 10. The explicit supplementary three-window cases resolved that coverage gap; the measured configuration was not changed.

The unchanged baseline Xcode build retained existing unused diagnostic-local/function warnings, deprecated -Ofast/infinity warnings in dependency targets, a libomp deployment-version warning (26.2 target versus a 27.0 library on the installed 27.0.1 system), and an existing malformed LLVM23 toolchain registration. These originate from unchanged baseline configuration/source and were not redesigned during a benchmark-only increment. The database-free fixtures do not claim byte-identical code generation to every Xcode dependency target.

Retained local artifacts are under DerivedData/ExpertAdvisor/Phase25B2: trials.json, summary.json, initialization.json, equivalence.json, trajectory-equivalence.json, provenance.json, per-process logs/safety snapshots, complete state and raw tensor files, and the timing/evidence binaries. Main build logs: DerivedData/ExpertAdvisor/phase25b2-release-build.log and phase25b2-fixture-build.log. Release executable SHA-256: a5ce98d51797f3cdfd3eb42aaa32436d9219a47b298db8841d7b8dd427d4ad86. Fixture/source/library/kernel hashes are in provenance.json.

Limitations and measurement bias
--------------------------------

* One hardware/software configuration, one synthetic primary target/data batch, B=128 and a short warmed training trajectory. Different batch sizes, heads, data distributions and sustained training may yield different relative improvements.
* Database loading, checkpoint I/O, service/workflow orchestration, scheduler concurrency and epoch-level data traversal are excluded. The result measures complete CalculateBatch updates, not full experiment elapsed time.
* Eight warmup updates initialize normal kernels/caches, but are not a thermal-equilibrium guarantee. All fresh processes start identically and order is counterbalanced; residual temperature, allocator, filesystem and OS scheduling effects remain possible.
* Desktop/WindowServer, Terminal, mactop and other normal system activity remained present, including the 5K display. GPU utilization, clock rate, power and temperature were not instrumented or held constant.
* Parent safety polling and between-update evidence serialization can affect CPU scheduling/cache state. They are symmetric; numerical observers were entirely excluded from measured training.
* Runtime diagnostic logging is disabled, but unchanged legacy statistics that lack a runtime guard still execute. The benchmark reflects that existing disabled-logging behavior; it does not measure a hypothetical removal of all diagnostic computation.
* Metal boundary snapshots cannot establish transient allocation peaks or allocation-rate differences. Timing differences are attributed to path selection under these controls, not a separate profiling decomposition of every operation.
* The statistical intervals describe this ten-pair sample under their stated assumptions; they are not a guarantee of production throughput on other machines or workloads.

Files added and operational change
----------------------------------

* Tests/LSTMTrainingThroughputBenchmark.mm: deterministic database-free complete-training fixture with timing and optional existing tensor observers.
* Tests/LSTMTrainingThroughputBenchmark.sh: isolated builds using normal development Release products.
* Tests/LSTMTrainingThroughputBenchmark.py: guarded sequential trial runner, equivalence checks and paired analysis.
* docs/architecture/LSTMTrainingThroughputBenchmark.rst: methodology, measurements and limitations.

No existing tracked file was modified. No training mathematics, synchronization, allocation strategy, model layout, checkpoint format, production state or default path was changed. Changes remain uncommitted, unstaged and unpublished. No additional optimization was started.

Final Git review
----------------

git status --short::

    ?? Tests/LSTMTrainingThroughputBenchmark.mm
    ?? Tests/LSTMTrainingThroughputBenchmark.py
    ?? Tests/LSTMTrainingThroughputBenchmark.sh
    ?? docs/architecture/LSTMTrainingThroughputBenchmark.rst

git diff --stat produced no output because all four additions remain untracked. The untracked additions contain 567 lines: fixture 155, runner 223, build launcher 35, report 154. git diff --check passed; production/shared-dependency git status --short produced no output.
