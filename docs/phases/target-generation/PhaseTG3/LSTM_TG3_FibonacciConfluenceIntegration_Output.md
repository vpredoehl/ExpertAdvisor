# LSTM TG3 causal Fibonacci confluence integration

## Disposition

**IMPLEMENTATION COMPLETE — PENDING CLEAN RELEASE VALIDATION**

Implementation and source-level validation are complete. Final closure still
requires review/commit followed by the prescribed normal Release build from an
unchanged clean source tree during an operationally safe window. Provenance was
not weakened or bypassed.

Baseline branch and commit: `lstm-feature-development` at
`bb9805274c96d72347ccf4da4308827ea72f0c1b` (TG2), above TG1B `ed7d2e9`
and TG1A `7058413`.

## Files changed

- `Headers/CausalFibonacciConfluenceIntegration.hpp`
- `Tests/TG3FibonacciConfluenceIntegrationTests.cpp`
- `Tests/TG3FibonacciConfluenceIntegrationTests.sh`
- `docs/phases/target-generation/PhaseTG3/TG3-Causal-Fibonacci-Confluence-Integration.md`
- `docs/phases/target-generation/PhaseTG3/LSTM_TG3_FibonacciConfluenceIntegration_Output.md`

TG1A, TG1B, and TG2 headers were not modified.

## Audit findings and implemented semantics

The source/manual audit found only the supplied sentence “See if Outer UTL is
within the up AB Fibonacci.” No available manual, diagram, repository module,
database function, or prior phase defines A/B selection, a ratio set,
retracement versus extension, “within,” observation time, or DTL symmetry.

TG3 therefore implements the source direction as the default
`SourceUTLUpABOnly` study and labels the remaining choices as configurable
implementation conventions. A separately enabled symmetric DTL/down-AB path
is a diagnostic hypothesis only.

The neutral A/B convention pairs each newly confirmed B fractal with the most
recent earlier confirmed opposite-kind TG1A fractal producing the directional
price move. UpAB is confirmed low A to later confirmed high B; DownAB is
confirmed high A to later confirmed low B. The structure becomes available at
B confirmation and its identity is never rewritten.

Ratios are mandatory caller configuration, finite, deduplicated, sorted, and
limited to retracements in `[0,1]`. There is no default ratio set and no
extension implementation. For range `R = abs(B-A)`:

```text
UpAB:   level(r) = B - rR
DownAB: level(r) = B + rR
```

TG3 observes confluence on the completed TG2 Inner-break bar. It copies TG2's
already selected paired Outer and its break-bar projection, selects only an AB
available by that bar, and freezes all confluence geometry before later
behavior is known.

The neutral “within” policy is inclusive absolute price tolerance around each
exact configured level: `abs(Outer - level) <= tolerance`. It records the
level-specific zone, distance, match, minimum raw distance, and ATR-normalized
minimum distance when observation-time ATR is causally available. Tolerance is
never calibrated from later outcome.

No pair, no eligible AB, and unsupported direction are distinct structural
ineligibility reasons. They are not counted as non-confluence or TG2 failure.

## Empirical conditioning

TG3 synchronizes but never rewrites TG2's Outer-target and
retest-then-Outer outcomes. Confluence and no-confluence groups separately
report successes, failures, censored, pending, and resolved counts. Structural
ineligibility remains separately visible. The only rate is:

```text
successes / (successes + failures)
```

and is absent for a zero denominator. No measured market sample or outcome
improvement claim is part of this implementation increment.

## Causality, bounds, and performance

Focused tests cover B-confirmation availability, UpAB and DownAB timing, exact
level equations, ratio validation/deduplication/ordering, stable identities,
multiple-AB selection, unchanged TG2 pairing, exact and adjacent tolerance
boundaries, structural ineligibility, outcome grouping, raw rates, immutable
confluence under later pivots/outcomes, historical/streaming prefix parity,
duplicate timestamps, and deterministic capacity censoring.

State bounds cover retained fractals, AB age/count, active observations, and
retained observations. Capacity loss of pending empirical state is explicit
censoring. The representative 50,000-bar test completed with bounded state;
the final timing will be recorded in the validation section.

## Validation record

Completed so far:

- `Tests/TG3FibonacciConfluenceIntegrationTests.sh` — PASS with C++20,
  `-Wall -Wextra -Werror -pedantic`; final 50,000-bar run completed in 381 ms with
  8 AB structures, 8 retained observations, and 49,993 cumulative Inner-break
  observations.
- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh` — PASS; 50,000-bar bound
  test and PostgreSQL fractal semantic audit passed.
- `Tests/TG1BTrendLineAngleClassificationTests.sh` — PASS.
- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh` — PASS; 50,000-bar bound
  test passed.
- `bash Tests/LSTMInputWidthExpansionTests.sh` — PASS. Direct invocation first
  reported the pre-existing non-executable file mode; explicit Bash execution
  passed.
- `Tests/LSTMModelInputCompatibilityTests.sh` — PASS; compile-time assertions
  retain Tensor `feature_size == 73` and current model input width 77.
- `bash Tests/FeatureAblationPairEvaluationTests.sh` — PASS. Direct invocation
  first reported the pre-existing non-executable file mode.
- `bash Tests/FeatureAblationReplicationEvaluationTests.sh` — PASS. Direct
  invocation first reported the pre-existing non-executable file mode.
- `Tests/SemanticWorkerRegistryTests.sh` — PASS.
- `Tests/SchedulerStatusProcessRecognitionTests.sh` — PASS.
- TG3 AddressSanitizer + UndefinedBehaviorSanitizer — PASS under the same
  strict warnings; final instrumented 50,000-bar run completed in 1,117 ms.
  `ASAN_OPTIONS=detect_leaks=0` was used because Apple ASan does not support
  leak detection; address and undefined-behavior checks remained enabled.
- Clang static analyzer — PASS; Apple Clang 21 returned an empty diagnostics
  array.
- `git diff --check`, explicit untracked-file whitespace checks, shell syntax,
  and final strict focused compilation — PASS.
- `clang-format --dry-run --Werror` — unavailable because `clang-format` is
  not installed; the source has no lines over 100 columns and strict compiler
  warnings/static analysis provide the available checks.
- Read-only `LSTM_Release --scheduler-status` — PASS; active scheduler PID
  61531, validated canonical scheduler identity, two managed training workers,
  no unmanaged workers, no identity mismatches, and no expected-missing
  workers. The command reported high system memory usage but no TG3-related
  scheduler fault.

## Semantic and non-production evidence

The new implementation is an isolated header used only by its standalone
focused test. No include or call site was added to Tensor, model input
preparation, feature layouts, feature ablation, scheduler, worker, registry,
database, or publication code. Semantic layout remains 7 and intended model
input width is unchanged by construction; regression evidence will be listed
after completion.

`kModelInputSemanticLayoutVersion` remains 7, Tensor `feature_size` remains
73, and `kCurrentModelInputWidth` remains 77. Repository search finds no TG3
include or call site in any production source, project target, database,
scheduler, worker, registry, or publication path.

Read-only process inspection and scheduler status found the production
scheduler and two managed training workers active. No process, experiment,
database row, stash, or production artifact was altered. A dirty-tree Release
build was not run.

## Release and repository state

Release build status: not run. The worktree is intentionally dirty with the
five new TG3 files, and `DerivedData/ExpertAdvisor/Build/Products/Release` is
also serving the active scheduler/status products. Running the build now would
violate the clean-source provenance gate and could replace active products.
After review/commit, and when operationally safe, run exactly:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Then verify a clean worktree and unchanged intended HEAD.

Final repository status is five untracked TG3 files and no tracked-file
modification. Consequently literal `git diff --stat` is empty until the files
are staged or committed; the untracked footprint is five files and will be
reported with exact line counts in the handoff. Intentional stashes remain
untouched.

## Remaining risks and follow-on research

The source leaves A/B selection, ratios, “within,” and DTL symmetry undefined.
TG3 makes these explicit configuration/policy choices, but they remain
measurement conventions requiring domain review. No real symbol/timeframe/date
population was authorized or measured, so there is no evidence here that
confluence improves TG2 outcomes. A later research run may compare the frozen
groups by stated population, period, numerator, denominator, censoring, and
limitations. Any proposal to make a validated measurement a model feature or
trading rule must be a separate phase with semantic-layout/input-compatibility
review.
