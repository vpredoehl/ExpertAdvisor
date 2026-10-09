Phase 25B-1: combined forward GEMM and bias
========================================

The recurrent batched LSTM forward affine operation can opt into an
ExpertAdvisor-owned Objective-C++ adapter. The original MetaNN implementation
remains the default. Output heads, single-row inference, gate/state evaluation,
backward propagation, pooling, diagnostics and checkpoint formats retain their
existing production behavior.

Runtime selection
-----------------

``EA_LSTM_FORWARD_AFFINE=metann`` selects the original implementation. An unset
variable also selects MetaNN. ``EA_LSTM_FORWARD_AFFINE=combined`` selects the
adapter. Other values raise an error before this affine operation submits work.
Selection is cached on first use and cannot be changed within that process.
The existing diagnostic logging mode emits one
``DIAG_LSTM_FORWARD_AFFINE,path=metann|combined`` line. This setting is not
persisted in models or experiment rows. There is no automatic retry on a GPU
error: the runtime switch is an explicit fallback selection.

Execution and ownership
-----------------------

The adapter constructs the same contiguous float32 MPS matrices, including
element offsets and rowBytes of ``k*sizeof(float)`` for A and ``n*sizeof(float)``
for B/C. GEMM retains transposeLeft=NO, transposeRight=NO, alpha=1 and beta=0.
The existing compiled ``add_row_bias_f32`` function is loaded from
``default.metallib``; no kernel is copied or recompiled by the adapter.

One command buffer encodes MPS GEMM followed by a separate serial bias compute
encoder, then commits once and waits once. Buffers must be shared, tracked,
on the adapter's device, and within checked byte ranges; the output cannot
overlap inputs. Shared C++ memory owners and Metal resources remain alive until
completion. No matrix host pointer is accessed in the adapter. Status/error
inspection occurs after the blocking wait; failure raises a runtime error.

The original affine path uses two submissions and waits. The optimized affine
path uses one of each. With gate/state evaluation unchanged, the recurrent
forward compute path becomes two rather than three submissions/waits per step,
or 128 rather than 192 for 64 steps. Diagnostic barriers remain additional.

Validation performed on 2026-10-09
----------------------------------

The worktree began at ``8a6fe73f0ced875dab2e7cc4a839e9335229ae97``. Process and
read-only scheduler inspection found no active train/infer/analyze workers,
no unresolved launches and no scheduler process. Publication was disabled for
every Xcode invocation. No production process, experiment state, database data,
production binary or shared MetaNN source was changed.

Matrix validation passed bitwise equality for 45 combinations: production
``k=235,n=256`` with m=1/17/128, irregular dimensions, offsets of 0/4/12 float
elements, and scales 1/1e-12/1e12. Each case reused storage for eight operations
with changing inputs. Sixteen sequential producer/consumer operations passed.
Guard regions remained intact. Both runtime selections passed. Zero-work calls
submitted nothing; invalid ranges, output aliasing and oversized dimensions
failed before submission. All instrumented nonempty combined calls recorded
one submission, one wait and successful completion. Real GPU failure was not
deliberately induced; the post-wait status/error branch was inspected.

The extended Phase 25A fixtures passed for legacy classification, auxiliary
classification, log-return regression and percent-return regression, with
diagnostics on/off and both affine paths. The small fixture executes 66 updates
plus restored-matrix continuation. An additional H=64/T=64 fixture executes
three updates plus continuation. Losses, parameter trajectories, learning rates,
update counts, cached timestep states/gates/forget logits and all six pre/postclip
gradient matrices matched bitwise. Clipping diagnostics matched, and regression
fixtures verified actual clipping. Current original-path state and diagnostic
output also matched the Phase 25A source baseline. Fixture-only observers are
compiled under ``LSTM_NUMERICAL_TEST_OBSERVERS`` and are absent from normal builds.
Database checkpoint serialization was not exercised.

Warmed isolated performance
--------------------------

Hardware: Mac Studio / Apple M5 Max / 48 GB, macOS 27.0.1 (26A434).
Compiler: Apple clang 21.0.0. Baseline ``metal_matmul.mm`` and adapter were
compiled together with C++20, -O3, -DNDEBUG, -fobjc-arc and deployment target
26.2. Both used identical matrices, existing compiled bias kernel and allocator.
Each shape used 32 warmup operations per path, then 12 alternating-order trials
of 100 operations per path. Latencies include synchronous return and host
encoding; they are not GPU-only timings. All benchmark outputs matched bitwise.

==================  ==================  ==================  ===============
m / k / n           MetaNN median us    Combined median us  Median speedup
==================  ==================  ==================  ===============
1 / 235 / 256       248.089             130.266             1.904x
17 / 235 / 256      260.111             130.141             1.999x
128 / 235 / 256     258.727             129.865             1.992x
==================  ==================  ==================  ===============

Trial standard deviations (MetaNN/combined, us): 5.637/0.497, 1.410/0.175 and
1.385/0.394 respectively. Trial ranges were 238.715--260.228 / 129.213--130.950,
256.670--260.418 / 129.839--130.333, and 255.875--260.730 / 129.221--130.417 us.
For each shape's 1200 measured operations per path, source-derived counts are
2400 submissions/waits for MetaNN versus 1200 for combined. Combined per-call
counts were independently checked by the instrumented correctness tests.

Device currentAllocatedSize before/after the warmed trials was 720896/720896,
786432/786432 and 1146880/1130496 bytes respectively. These are whole-device
snapshots for this shared test process, not attribution to each path or a leak
test. Process peak RSS reached 17858560 bytes. No allocation optimization or
full-training throughput benchmark was performed; these speedups apply only to
the isolated affine operation.

Commands and build limitations
------------------------------

Executed successfully::

    bash Tests/MetalForwardAffineTests.sh --benchmark
    bash Tests/LSTMTrainingDiagnosticNumericalTests.sh /tmp/ea_phase25b1_baseline_LSTM.cpp
    python3 Tests/LSTMTrainingDiagnosticOverheadTests.py
    python3 Tests/TrainingDistributionLoggingTests.py
    LSTM_TEST_PRODUCTS_DIR="$PWD/DerivedData/ExpertAdvisor/Build/Products/Release" bash Tests/FreshModelInitializationTests.sh
    xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Train Worker" -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor -destination 'platform=macOS,arch=arm64' PUBLISH_CANONICAL_LSTM_RELEASE=NO PUBLISH_CANONICAL_BINARY=NO build
    plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj
    git diff --check

The baseline source was extracted read-only with
``git show 8a6fe73f:LSTM/LSTM.cpp``. Numerical and matrix fixture builds use
warnings-as-errors with the existing narrow MetaNN/legacy warning exclusions;
the adapter emitted no warnings. Existing standalone LSTM fixture launchers
were updated to link the adapter and follow the shared-dependency include link.
The persistence fixture was not run because it creates a database; its launcher
was updated for the new link dependency only.

Broader ``LSTM Release`` / Debug and ``LSTM Debug`` / Debug builds failed on
unrelated missing comparison/archive symbols in the analysis-worker dependency
and missing ModelRuntimeValidation symbols in the legacy Debug target,
respectively. The latter source is absent from that target in the baseline
project as well. The training worker Debug build succeeded, retaining existing
unused diagnostic-local warnings. Xcode also reported an existing malformed
LLVM23 toolchain registration. Those unrelated issues were not redesigned in
this increment. Initial sandboxed attempts could not access Metal/Xcode services;
authorized retries provided the executed validation above.

The unchanged Release provenance generator was executed and rejected the dirty
tree with ``Release provenance requires a clean source tree``. No guard was
bypassed and no normal Release executable was published. Development Xcode
products stayed under ``DerivedData/ExpertAdvisor`` with no
``CONFIGURATION_BUILD_DIR`` override. Shared MetaNN was clean after validation;
the inspected GEMM helper and kernel source hashes remained unchanged.

Qualification is limited to the tested hardware/software and isolated fixtures.
The default remains MetaNN. All changes are left uncommitted; Phase 25B-2 was
not started.
