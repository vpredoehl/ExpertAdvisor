# Phase 25B-3: production-representative Metal training qualification

**Status: BLOCKED for Phase 25B-3U new measurements on 2026-10-09; historic root-cause analysis and Phase 25B-3T remain INCONCLUSIVE. Phase 25B-3R remains BLOCKED. Production readiness is not established.**

## Phase 25B-3U recurrent synchronization analysis

**Final status: BLOCKED for new controlled performance measurements (2026-10-09).
The historic root-cause assessment remains INCONCLUSIVE.** A sustained competing
CPU workload prevented a valid new qualification window. Existing measurements
localize the additional recurrent wall time, but do not establish its dominant
underlying cause. No new qualification benchmark was launched.

### A. Objective and preflight

Investigate the approximately 14.6-second forward/backward increase in the slow
Phase T Combined process without changing mathematics, numerical precision,
production defaults, checkpoint semantics, input layout, or shared MetaNN.
Development started clean at `5dd91afe9fff87ea2e31580251e6af6506a7c820` on
`dedicated-train-layout-rollover-squashed-v1` in the requested Rollover worktree.
Production started clean at `b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`;
shared MetaNN started clean at `a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`.
`MetaNN/MetaNN` is a symlink to production's separate MetaNN repository and was
inspected read-only.

Sandbox process/swap inspection failed; normal-access escalation was used,
not an assumption of idleness. The saved launchctl status had no scheduler PID,
active count zero, and `spawn failed` / last exit 78. No LSTM worker, qualification
training process, or Ollama runner was present; Ollama's residency API returned
no models. CoreSpotlight was at 0% CPU. Memory pressure was level 1; swap remained
12.12 MB and VM swap counters remained 256 in / 776 out. No competing compute
GPU process was identified, although device utilization was 11% with desktop
activity, so zero GPU activity is not claimed.

**MEASURED:** the initial census recorded `fileproviderd` at 102.4% CPU. A bounded
five-second CPU sample confirmed 103.3%, its Provider process 11.6%, and
WindowServer 37.8%. This is substantial competing work under the requested
preflight rule. The new alternating qualification pairs were stopped before
launch. These current observations are a safety blocker, **not evidence that
file-provider activity caused either historic slow trial**. No external process
was stopped or reconfigured. Correctness tests do not supply causal latency
measurements.

### B. Complete existing evidence and population correction

All twelve complete S/T hotspot reports, logs, 72-entry per-update JSON traces,
safety snapshots, serialized inputs and state trajectories were inspected.
`DiagnosticU/analyze_existing.py` independently parses every UPDATE log record,
checks it against JSON, hashes the actual files, and literally compares all
12 input/state streams. Its complete results are in `existing-analysis.json`
and `existing-analysis.txt`; no previous evidence was overwritten.

Fast Combined is the mean of T pairs 1/2; slow Combined is T pair 3.
Fast MetaNN is the mean of S pairs 1/3; slow MetaNN is S pair 2.
S Combined never exhibits the corresponding slow regime, and T MetaNN remains
near the fast regime. Measured 16-update blocks confirm T slow Combined's
746.7 / 1094.7 / 1139.9 / 964.8 ms and S slow MetaNN's
1183.1 / 1190.7 / 1091.3 / 781.3 ms transitions.

**MEASURED:** hotspot totals include **8 warmup + 64 measured updates**, not just
64. Counts are 18,432 forward calls and 9,216 backward calls. The fixture's
published means describe only 64 measured updates. Hotspots have no per-update
or warmup-separated stage records; a measured-only stage decomposition cannot
be recovered. Multiplying stage totals by 64/72 would assume a stationary
regime contradicted by the traces. Earlier S wording that described its hotspot
totals as “across 64 updates” should be read with this correction.

### C. Actual recurrent execution-path inventory

`CalculateBatch` processes 189 accepted windows as minibatches of 128 and 61,
each of sequence length 64, input width 171, hidden width 64. It therefore runs
128 training forward and 128 backward timesteps per update. In this exact
legacy classification fixture with epoch index 0, the existing epoch checkpoint
hidden-geometry diagnostic replays both minibatches after the update, adding
**128 forward calls per update**. Suppressed logging does not remove that replay.
The 256-forward / 128-backward count is measured, not an assumed single pass.
The existing replay is left intact; this investigation does not redesign it.

| Region / operation | Calls per fixture update | Execution and synchronization | Existing visibility |
|---|---:|---|---|
| Forward concatenate `[x,h]` | 256 | CPU row memcpy through shared `[MTLBuffer contents]`; small C++ vectors; row weight view shares ownership | `concat_cols` also aggregates backward concatenation |
| Forward affine `(B,235) × (235,256)` plus bias | 256 | Combined: one buffer with MPS GEMM then row-bias encoder, commit and one blocking wait. MetaNN: GEMM and bias each have their own buffer/commit/wait | affine wall scope; Combined CPU/GPU timestamps only |
| Forward i/f/o sigmoid, g tanh, cell and hidden state | 256 | one `GateStateFused` compute buffer, commit and blocking wait; persistent training scratch buffers | `gate_state_fused` inclusive wall time |
| Forget-logit slice plus recurrent cache | 256 | one CPU SliceCols allocation/copy and nine `DeepCopyMatrix` allocations/copies; concrete Matrix Evaluate drains EvalPlan, normally already empty; no copy command buffer | hidden inside forward before U; now `forward_cache_materialization` |
| Replay scratch setup | 128 | a fresh scratch object at each replay timestep; seven Metal-backed scratch matrices allocated before affine | still forward residual, outside new cache scope |
| Backward gate expressions and cell carry | 128 | shared expression DAG: tanh, products, scalar subtraction, addition; CPU loops and temporary Metal-backed matrices, then EvalPlan bookkeeping; no NSMetalAdd dispatch | hidden inside backward before U; now `backward_gate_expressions` |
| Backward parameter / bias / hidden GEMMs | 128 sets | three CPU transposes/materializations and three synchronous MPS MatMul buffers, each committed and waited separately | `backward_gemms` includes transpose/allocation, CPU encode/wait, GPU execution |
| Backward gate packing, `[x,h]`, ones column | 128 | new matrices, CPU memcpy/fill; no GPU copy buffers | packing and concat scopes exclude some allocation |
| Gate gradient accumulation | 256; plus 2 merges | CPU loops split dW/db into four accumulators, then merge into update gradients with existing casts | aggregate `gate_accumulator_split_merge` mixes backward and outer merges |
| Clip and SGD | 1 clip; 2 optimizer scopes | CPU in-place float loops; no optimizer GPU buffers | existing clip/optimizer scopes |
| Host norm / finite / matrix diagnostics | variable | `WaitForAll` submits an **empty** Metal buffer and blocks; FroNormEvalHost does it twice; some helpers copy to host vectors/CPU tensors | inclusive `host_diagnostics_data_copy`, with mixed parentage |

**Source-derived counts (INFERRED):** regular recurrent work requires at least
896 buffer commits/waits per Combined update: 256 affine + 256 gate-state +
384 backward GEMMs. MetaNN requires at least 1,152: 512 affine + 256 gate-state
+ 384 backward GEMMs. Over 72 updates these are 64,512 and 82,944 respectively.
These are lower bounds, excluding heads and diagnostic empty buffers. Each
forward/backward operation returns after its existing wait, so this path does
execute many small synchronous GPU operations. The CPU gate expressions and
cache copies themselves do **not** add command buffers.

Each forward cache allocates ten matrices: 2,560 per fixture update, 184,320 per
72-update process. The recurrent backward DAG has 23 unique materialized
expression outputs (16 products, five scalar subtractions, one tanh, one add),
assuming the shared EvalBuffer deduplication observed in source; approximately
2,944 temporary matrices per update. EvalPlan uses sets/maps, heap-owned nodes,
and shared evaluation handles. Metal ContinuousMemory constructs an Impl,
obtains the device, and allocates shared MTLBuffer storage; its pool only covers
buffers up to 4,096 bytes. Even the smaller minibatch's B×H float buffers are
15,616 bytes, so these allocations are outside that pool. Cache copies,
expression computation, resource allocation/release and host bookkeeping are
plausible hidden costs, **not measured causes of the historic variation**.

Relevant read-only interfaces are `operation/math/{multiply,add,substract,tanh}.h`,
`operation/tensor/{dot,permute}.h`, `evaluate/{eval_plan,eval_buffer,eval_handle}.h`,
`data/facilities/continuous_memory_metal.mm`, and `metal/metal_matmul.mm`.
The recurrent expression evaluators operate on CPU raw pointers despite their
Metal device tag. `metal_add.mm` has synchronous unary/binary kernels, but those
kernels are **not invoked by these gate expressions**. No generic Metal-tag
assumption was used to label them GPU work. Production build OpenMP settings
can affect CPU execution; the historical qualification script uses its existing
O3 flags. Neither path is changed here.

### D–F. Ranked hotspots and fast/slow decomposition

Times below are seconds. Counts cover the complete 72-update process. The ranked
Combined table uses **17.117265 s full-update increase** as its percentage
denominator. Residuals are exclusive only of the explicitly named child scopes;
the operations inside each residual remain inclusive and unresolved.

| Rank / operation | Calls | Fast total | Slow total | Absolute increase | % full-update increase | Instrumentation visibility |
|---|---:|---:|---:|---:|---:|---|
| 1 Forward minus affine and fused gate-state | 18,432 parents | 15.939245 | 22.877000 | +6.937755 | 40.53% | reliable parent-minus-two-children residual; cache/scratch/CPU hidden |
| 2 Backward minus GEMMs and gate packing | 9,216 parents | 16.351277 | 22.478719 | +6.127442 | 35.80% | reliable residual; CPU gate DAG/allocation/bookkeeping hidden |
| 3 Outside forward/backward/clip/optimizer/window build | 72 updates | 4.741283 | 7.092209 | +2.350926 | 13.73% | outer residual; heads, replay setup, diagnostics, normalization and other bookkeeping |
| 4 Backward GEMM scope | 9,216 sets | 8.733035 | 10.228600 | +1.495565 | 8.74% | inclusive CPU transpose/allocation plus three GPU submissions/waits |
| 5 Window batch build | 144 | 0.581529 | 0.835796 | +0.254267 | 1.49% | CPU construction inclusive; disjoint from recurrent calls |
| Forward affine | 18,432 | 3.319240 | 3.274200 | −0.045040 | −0.26% | nested forward; already removed from rank 1 |
| Fused gate/state | 18,432 | 2.549515 | 2.548700 | −0.000815 | −0.005% | nested forward; already removed from rank 1 |
| Gate packing | 9,216 | 0.060889 | 0.058181 | −0.002707 | −0.016% | nested backward; already removed from rank 2 |
| Clip / optimizer | 72 / 144 | 0.002254 / 0.000486 | 0.002183 / 0.000429 | −0.000071 / −0.000057 | <0.001% each | disjoint CPU scopes |

Additional aggregate leaf checks, **not additive to that partition**:
host diagnostic copies/norms, 20,126 calls, 0.725503 → 0.725745 s (+0.000243 s,
0.0014%); accumulator split/merge, 18,576 calls, 0.155569 → 0.148886 s
(−0.006683 s); concat, 27,648 calls, 0.104968 → 0.099474 s (−0.005493 s);
appended return features, 18,432 calls, 0.000454 → 0.000458 s (+0.000003 s).
These aggregate leaves have mixed parents and cannot be subtracted from a
specific parent's total reliably. Existing profiler `percent` columns divide
by the sum of overlapping scopes; they are **not percentages of update time**.

| Quantity / scope | Combined fast | Combined slow | Difference | MetaNN fast | MetaNN slow | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Measured mean (ms/update, 64 only) | 720.159 | 986.502 | +266.343 | 769.269 | 1061.574 | +292.305 |
| Total measured updates (64) | 46.090183 | 63.136113 | +17.045930 | 49.233235 | 67.940745 | +18.707510 |
| Total update time (all 72) | 52.278752 | 69.396017 | +17.117265 | 55.712254 | 77.313604 | +21.601350 |
| Forward, inclusive | 21.808000 | 28.699900 | +6.891900 | 24.495700 | 33.254100 | +8.758400 |
| Backward, inclusive | 25.145200 | 32.765500 | +7.620300 | 25.786250 | 35.217700 | +9.431450 |
| Affine child | 3.319240 | 3.274200 | −0.045040 | 5.576040 | 5.257310 | −0.318730 |
| Backward GEMM child | 8.733035 | 10.228600 | +1.495565 | 8.933275 | 10.791200 | +1.857925 |
| Net measured matrix-scope contribution | — | — | +1.450525 | — | — | +1.539195 |
| Fused forward elementwise/state child | 2.549515 | 2.548700 | −0.000815 | 2.554145 | 2.456000 | −0.098145 |
| Clip / optimizer | 0.002254 / 0.000486 | 0.002183 / 0.000429 | −0.000128 combined | 0.002236 / 0.000485 | 0.002133 / 0.000423 | −0.000165 combined |
| Forward residual (minus affine/state) | 15.939245 | 22.877000 | +6.937755 | 16.365515 | 25.540790 | +9.175275 |
| Backward residual (minus GEMMs/packing) | 16.351277 | 22.478719 | +6.127442 | 16.791176 | 24.367960 | +7.576784 |
| Outside forward/backward/clip/optimizer | 5.322812 | 7.928005 | +2.605193 | 5.427584 | 8.839249 | +3.411665 |

**MEASURED:** the recurrent difference is **14.512200 s** for Combined, not an
independent 14.6-second measured-only total. Forward contributes 47.49% and
backward 52.51% of it. Backward GEMMs increase **17.13%**, while backward stage
wall time increases **30.31%**. Their +1.495565 s explains only **19.63% of the
backward increase**, or 10.31% of the recurrent increase. They do not grow
proportionally or dominate the extra backward latency. For MetaNN, GEMMs grow
20.80% versus backward's 36.58%, explaining 19.70% of its backward increase;
its recurrent increase is 18.189850 s.

The forward/backward residual increases together are **13.065197 s**, or
**90.03% of Combined's recurrent difference** (76.33% of its full-update
difference). MetaNN shows the same pattern: 16.752059 s, 92.10% of its recurrent
difference. These residuals include some already measured mixed-parent CPU
leaves, so they are not a claim of entirely uninstrumented execution. Their
stable/smaller aggregate leaf timings cannot explain the extra seconds.
The CPU-only backward elementwise contribution cannot be separated historically;
its pure arithmetic time is **UNKNOWN**. Existing instrumentation therefore
does **not fully explain the recurrent difference at operation level**.

### G. CPU/GPU synchronization findings

The preserved T Combined timestamps, across 18,432 buffers in each process:

| Duration | Fast mean total | Slow total | Difference |
|---|---:|---:|---:|
| CPU create | 0.007262 | 0.007153 | −0.000110 |
| CPU encode | 0.315652 | 0.314984 | −0.000668 |
| CPU commit | 0.035172 | 0.037059 | +0.001886 |
| CPU blocking wait | 2.920904 | 2.872028 | −0.048876 |
| GPU execution | 0.563097 | 0.587787 | +0.024690 |
| GPU kernel interval | 0.406820 | 0.343811 | −0.063009 |

**MEASURED:** the Combined affine wait is smaller in the slow trial. Its tiny
GPU-duration increase cannot explain the recurrent difference. These durations
are children of the affine scope, **not additional decomposition rows**. GPU
kernel and execution intervals can overlap; they are not added together or
subtracted from CPU clocks. All 18,432 slow-trial GPU timestamps were valid.

**UNKNOWN:** CPU wait versus GPU execution for backward GEMMs and GateStateFused;
shared MetaNN exposes no command-buffer handles through these interfaces. The
GEMM wall scope includes CPU transpose/temporary allocation as well as GPU work,
so its increase does not prove slower GEMM kernels or waits. No ExpertAdvisor-only
hook can read those buffers' timestamps without intercepting/replacing shared
internals, which is outside this task. The forward residual predominantly
executes host allocation/copies/bookkeeping around already synchronous calls;
wall time in those regions does not identify which underlying operation blocks.

Whole-process CPU time increases only 3.395045 s for Combined's 17.117265 s wall
increase, and 1.758386 s for MetaNN's 21.601350 s increase. Aggregate process CPU
time is not a per-scope synchronization measurement and cannot be subtracted
from wall time to infer exact waits. No thermal, CoreSpotlight, CPU scheduling,
macOS GPU scheduling, bandwidth, or driver attribution is established.

### H. Instrumentation decision and changes

A. Existing scopes locate the dominant **regions**, not their individual costs.
B. Extra time is spread across both recurrent stages, not one measured operation.
C. Only Combined affine has separated CPU wait / GPU duration evidence.
D. ExpertAdvisor can time cache materialization and backward expression evaluation
at their existing call boundaries. E. Those CPU wall scopes add no command
buffers, barriers, waits, copies, or altered operation ordering.

Only two probes were added, using the existing opt-in profiler:
`forward_cache_materialization` covers the original forget-logit slice and nine
cache copies; `backward_gate_expressions` covers unchanged expression creation,
registration, temporary allocation, and the existing EvalPlan evaluation.
Both remain inclusive; neither is GPU timing or a pure allocation timer.
When the profiler is disabled, their optional scopes are empty and no new clock
reads/counters are executed. Phase T command-buffer instrumentation, its
`EA_LSTM_COMMAND_BUFFER_TIMING` switch, and the default MetaNN affine selection
are unchanged. Ordinary training still enables profiling only through its
existing explicit launch option.

Two new counters would run 256 and 128 times per update in this fixture, adding
384 timer/counter operations and two map keys. Performance perturbation is
**UNKNOWN**, because the competing workload prohibited controlled on/off timing.
The new scopes are exercised by correctness-only profiles; their latencies are
excluded from causal conclusions. No optimization follows from those samples.
No canonical-data trial, sustained trial, or new production database read/write
was launched in U. A bounded, unexecuted three-pair runner is retained as a
DiagnosticU investigation artifact, not a validated new benchmark result.

### I. Numerical equivalence and regression validation

**PASS:** actual S/T input bytes and loss/state trajectory bytes match across
all twelve processes. Input SHA-256 is
`244b20028c2985d1edf7192918712a97afe348b1ec4ca7d3da69763d69e9e932`;
72-update state SHA-256 is
`ef94c3a8361b6bf8a89d354a5be7324abf8465591a8f403c580951e2eb9d3317`.
This state stream includes initial state, every loss/counter, and matrices after
every update, not just a rounded final loss. Prior full observer and private
PostgreSQL checkpoint restoration evidence is preserved; the database suite
was not rerun.

**PASS:** the existing database-free numerical suite compares starting-source
and current-source results, MetaNN/Combined paths, logging on/off, legacy and
auxiliary classification, log/percent regression with real clipping, and matrix
restore continuation. Production dimensions H=64/T=64 also pass. Added checks
verify profiler defaults off, both new counters are emitted when enabled, and
profiler-on/off state and full forward-cache/preclip/postclip tensor streams
match literally for both paths. No mathematics, synchronization calls, tensor
ownership, layout, or checkpoint code was edited.

Validation commands (all retained evidence under DiagnosticU):

- `PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase25B3/DiagnosticU/analyze_existing.py`
- `bash DerivedData/ExpertAdvisor/Phase25B3/DiagnosticU/build-fixture.sh` — existing qualification O3/Wall/Wextra/Werror build, successful with no emitted warnings.
- `LSTM_TEST_EVIDENCE_DIR="$PWD/DerivedData/ExpertAdvisor/Phase25B3/DiagnosticU/numerical" bash Tests/LSTMTrainingDiagnosticNumericalTests.sh DerivedData/ExpertAdvisor/Phase25B3/DiagnosticU/baseline-LSTM.cpp` — all cases PASS, exit 0.
- `bash -n Tests/LSTMTrainingDiagnosticNumericalTests.sh`, Python AST checks, and `git diff --check`.
- `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build` — initial sandbox attempt denied build metadata access; normal-access attempt reached the intentional clean-source Release provenance guard. The clean committed Release build passed (exit 0, `release-build-final.log`); the final documentation commit is also rebuilt, with evidence retained in `release-build-completion.log`.

The existing Release warning debt (`-Ofast` deprecation, infinity under fast-math,
unused diagnostic locals/functions, empty metal_copy archive object, and stale
external LLVM toolchain discovery, and libomp built for macOS 27 linked into the macOS-26.2 target) is outside the two probes. Changing those
numerical flags or shared code would violate this increment's constraints.
The dedicated O3 fixture compiles under the existing warning-as-error policy;
no new probe warning is accepted. The preserved evidence-directory option in
the numerical runner lets this task keep all diagnostics under DerivedData.

### J–L. Assessment, unknowns and next action

**MEASURED:** affine and fused forward gate/state wall time remain stable or
smaller; clipping, SGD, packing, accumulation and measured diagnostic-copy
leaves cannot explain the added seconds. Backward GEMM wall time accounts for
only about one fifth of the backward increase. Approximately 90% of the extra
recurrent time lies in the forward cache/scratch/host region and backward
expression/allocation/host region. The current competing workload blocks new
controlled timing; it is not an attribution for the historic traces.

**INFERRED:** repeated shared-buffer allocation/copy/materialization and the CPU
expression DAG are higher-value investigation targets than more affine or fused
activation GPU probes. Source inspection disproves a blanket interpretation of
all recurrent Metal-tagged elementwise work as synchronous GPU kernels. The
fixture's repeated hidden-geometry replay substantially increases forward
allocation/copy counts. Stable retained Metal allocation does not establish
stable allocation latency or eliminate transient allocation/release overhead.

**UNKNOWN:** which of resource allocation/release, CPU arithmetic/copy loops,
EvalPlan registration/dispatch, and host blocking accounts for each increase;
CPU/GPU split inside backward GEMMs; per-update stage location of the regime
transition; new probe overhead; and causality of the slow regime. No direct
measurement identifies the dominant underlying cause, so
ROOT_CAUSE_IDENTIFIED is not warranted.

Recommended next action: in a verified idle window, use the same isolated
CADCHFRMP/horizon-4/layout-13/width-171/sequence-64/seed-1002/calendar-1 fixture
with these two scopes and the preserved T timing switch, at most three
alternating pairs of 8 warmup + 64 measured updates. First establish negligible
probe perturbation; then compare cache/expression/residual growth. If the slow
regime does not appear, stop and retain T/S as primary evidence. Split allocation
from CPU arithmetic only if those results justify another ExpertAdvisor-owned
probe; obtaining shared-buffer GPU/wait timestamps would require separate
approved visibility work. Keep the production default MetaNN. Do not start
Phase 25B-4, deploy, push, merge, change scheduler configuration, or modify
production records or shared MetaNN.

Files changed in U: `LSTM/LSTM.cpp`,
`Tests/LSTMTrainingDiagnosticNumericalTests.cpp`,
`Tests/LSTMTrainingDiagnosticNumericalTests.sh`, and this report. All new evidence
is in `DerivedData/ExpertAdvisor/Phase25B3/DiagnosticU/`. Production and shared
MetaNN Git identity/status are checked again at completion.

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
