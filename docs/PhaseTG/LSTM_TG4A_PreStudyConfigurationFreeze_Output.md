# LSTM TG4A pre-study configuration freeze output

## Verdict

**GO / CLOSED.** All three methodology inputs are prospectively frozen,
reproducible, tested, committed, and Release provenance-valid. No real TG4
historical study has been run and no TG4 outcome-bearing observation has been
inspected.

## Baseline and safety

- Branch: `lstm-feature-development`.
- Requested TG4 implementation commit:
  `8cc823239ed34b068009c92c50a185216db01158`.
- Starting HEAD: `f68c9d8c547c58e79b10fae8c3eb7f57104546cd`.
- Intervening commits: `3959a73` (live inference child regression) and
  `f68c9d8` (authoritative live inference child completion).
- The starting worktree was clean. Existing stashes were inventoried and not
  changed.
- Read-only scheduler status found PID 68183 healthy, with no active managed
  or unmanaged train, inference, or analysis worker. No scheduler, worker,
  experiment row, or published binary was changed or restarted.

## Frozen decisions

| Input | Frozen value | Provenance |
|---|---|---|
| TG1B reference-bar scale | `14` completed bars | Experimental; tied prospectively to the existing 14-bar creation-time Wilder ATR horizon. |
| TG3 ratio set | `0.6180339887498949` | Experimental one-element set; nearest binary64 decimal for `(sqrt(5)-1)/2`. |
| TG3 tolerance | `1` canonical FX pip | Experimental; materialized as 0.0001 for non-JPY canonical pairs and 0.01 for USDJPY. |

The source defines the angle bands and UTL/up-AB direction, but does not define
these three numeric inputs. The implementation defines ATR normalization,
causal AB selection, exact retracement equations, observation timing, and the
inclusive absolute-distance comparison. The complete rationale and quote-unit
audit are in `TG4A-Pre-Study-Methodology-Freeze.md`.

## Implementation

TG4 now accepts a canonical-pip tolerance count instead of one global raw
price tolerance. It derives the absolute value per canonical symbol before
constructing TG3, so TG3's causal and inclusive absolute-price semantics are
unchanged. Metadata serializes the convention, pip count, all six effective
values, stable study/configuration identities, and a deterministic
configuration fingerprint.

The tracked freeze is
`Scripts/tg4_analysis_config.frozen_v1.conf`. The example remains deliberately
non-runnable until its experimental placeholders are replaced.

The named first-study range `tg4-preconfirmation-2010-2025-v1` is exactly:

```text
warmup  [2010-01-01, 2025-01-01)
score   [2010-01-01, 2025-01-01)
outcome [2010-01-01, 2025-01-01)
```

Therefore no 2025 confirmation bar can be loaded or used even to resolve a
pre-boundary outcome. Pending finite windows are censored at end of input.

## Validation record

Pre-commit validation:

- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh` — PASS, including the
  read-only PostgreSQL fractal semantic audit.
- `Tests/TG1BTrendLineAngleClassificationTests.sh` — PASS.
- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh` — PASS.
- `Tests/TG3FibonacciConfluenceIntegrationTests.sh` — PASS.
- `Tests/TG4HistoricalEmpiricalEvaluationTests.sh` — PASS with strict C++20
  warnings; the tracked frozen configuration loaded and produced fingerprint
  `fnv1a64:bf809ce38a4a444a`.
- `Tests/TG4HistoricalEmpiricalEvaluationBoundaryTests.sh` — PASS.
- `Scripts/run_tg4_historical_empirical_evaluation.sh --help` — PASS; the
  standalone strict-warning CLI build completed and only printed usage.
- `Tests/LSTMInputWidthExpansionTests.sh` — PASS.
- `Tests/LSTMModelInputCompatibilityTests.sh` — PASS.
- TG4 AddressSanitizer and UndefinedBehaviorSanitizer run — PASS with
  `ASAN_OPTIONS=detect_leaks=0`; address and UB checks remained enabled.
- Clang static analyzer over the changed TG4 core and tests — PASS with no
  diagnostics.
- Changed shell script syntax and `git diff --check` — PASS.

The frozen file SHA-256 is
`78187108b908b4298eb9ae2f1e9fa1de63eadb1aceb29d09ab5261c42a7a9444`.
- Normal provenance build:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release"
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build` —
  PASS (`** BUILD SUCCEEDED **`) from a clean committed tree; post-build tree
  remained clean and the intended HEAD was unchanged.

TG4A did not run the real study as validation and did not publish or restart
the scheduler or any worker.

## First outcome-bearing study command — not executed

```bash
Scripts/run_tg4_historical_empirical_evaluation.sh \
  --config Scripts/tg4_analysis_config.frozen_v1.conf \
  --output-dir /absolute/new/path/tg4-preconfirmation-2010-2025-v1 \
  --all-canonical-symbols \
  --study tg4-preconfirmation-2010-2025-v1
```

## Remaining limitations

- The three inputs are prospective experimental conventions, not source-
  authored methodology and not evidence of optimality.
- No TG4 outcome has yet tested their empirical usefulness.
- The one-ratio set deliberately excludes other ratios; changing it requires a
  new frozen configuration and study identity.
- The USDJPY source has a previously documented long historical gap whose
  cause remains unclassified; TG4 will report but not fill it.
