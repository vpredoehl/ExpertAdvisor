---
title: "LSTM TG1B Trend-Line Angle Calibration and Classification"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_TG1B_TrendLineAngleCalibrationClassification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM TG1B Trend-Line Angle Calibration and Classification

IMPLEMENTATION COMPLETE — PENDING CLEAN RELEASE VALIDATION

## Implementation

TG1B is implemented as a separate composition layer over unchanged TG1A geometry:

- [CausalFractalTrendLineAngleClassification.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/CausalFractalTrendLineAngleClassification.hpp:24)
- [TG1BTrendLineAngleClassificationTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TG1BTrendLineAngleClassificationTests.cpp:104)
- [TG1BTrendLineAngleClassificationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TG1BTrendLineAngleClassificationTests.sh)
- [TG1B methodology documentation](/Volumes/Developer%20SSD/ExpertAdvisor/docs/TG1B-Trend-Line-Angle-Calibration-and-Classification.md:1)
- TG1A documentation now references the separate TG1B layer.
- Corrected one stale scheduler source path in `LSTMInputWidthExpansionTests.sh`; no production behavior changed.

The TG1A audit found no implementation defect, so its geometry code and raw field meanings remain unchanged.

## Calibration and classification

The creation-time convention is:

```text
n = rawSlopePerBar / creationAtr
angleMagnitude = atan(abs(n) * referenceBarScale) * 180 / pi
```

`referenceBarScale` is mandatory, finite, positive, configurable, and emitted diagnostically. It means completed bars per normalized horizontal unit; the vertical unit is one creation-time ATR. No chart pixels, zoom, quote magnitude, or display state are involved.

Direction remains independently UTL/DTL. Signed normalized slope is preserved, while angle uses magnitude.

Exact inclusive bands:

- `LongTerm`: 12°–20°
- `Outer`: 25°–40°
- `Inner`: 45°–85°
- Everything below 12°, in `(20,25)` or `(40,45)`, above 85°, or unavailable/invalid is `Unclassified`.

Classification freezes when anchor 2 confirms. ATR includes completed bars through that confirmation bar, never future bars. Later TG1A observation-time ATR changes do not rewrite TG1B classification.

Creation events support deterministic distribution collection; the live classified view mirrors TG1A expiry and eviction.

Representative output:

```text
symbol=EURUSD,timeframe=1h,direction=UTL,anchor1_bar=2,anchor2_bar=7,raw_slope_per_bar=0.4,creation_atr=10.6793679825,creation_atr_normalized_slope=0.0374554000438,calibration_reference_bar_scale=20,calibrated_angle_magnitude_degrees=36.8371746562,classification=Outer,classification_timing=second_anchor_confirmation
```

Scale 20 is a test/example calibration, not an empirical recommendation.

## Validation

Passed:

- `bash Tests/TG1BTrendLineAngleClassificationTests.sh`
- `bash Tests/TG1ACausalFractalTrendLineGeometryTests.sh`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/FeatureAblationPairEvaluationTests.sh`
- `bash Tests/SemanticWorkerRegistryTests.sh`
- TG1B AddressSanitizer and UndefinedBehaviorSanitizer run
- Clang static analysis with strict warnings
- `git diff --check`, including untracked TG1B files

Tests cover all requested boundaries using adjacent representable floating-point values, UTL/DTL symmetry, missing/invalid normalization, calibration effects, monotonicity, confirmation timing, frozen future behavior, causally reproduced Wilder ATR, lifecycle retention, and historical/streaming parity at every prefix.

Semantic layout remains 7, model input width remains 77, feature-ablation semantics are unchanged, and protected production/registry files have no diff.

## Production and Release status

The production scheduler remains running at PID 61531. Layout-7 training workers for experiments 650 and 651 remain running at PIDs 95926 and 95985. Nothing was stopped, restarted, paused, published, or replaced; the registry was not modified.

The normal Release build was intentionally not run because the implementation is uncommitted and therefore cannot pass the required clean-source provenance gate. After committing, the remaining prerequisite is:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build
```

## Repository state

Branch/head: `lstm-feature-development` at `be70daf`. Existing pre-TG1A stashes remain untouched.

`git status --short`:

```text
 M Tests/LSTMInputWidthExpansionTests.sh
 M docs/TG1A-Causal-Fractal-Trend-Line-Geometry.md
?? Headers/CausalFractalTrendLineAngleClassification.hpp
?? Tests/TG1BTrendLineAngleClassificationTests.cpp
?? Tests/TG1BTrendLineAngleClassificationTests.sh
?? docs/TG1B-Trend-Line-Angle-Calibration-and-Classification.md
```

`git diff --stat`—which excludes the four untracked new files:

```text
 Tests/LSTMInputWidthExpansionTests.sh           | 2 +-
 docs/TG1A-Causal-Fractal-Trend-Line-Geometry.md | 4 +++-
 2 files changed, 4 insertions(+), 2 deletions(-)
```

The four new files contain 932 lines. The only remaining acceptance prerequisite is the committed, clean-source Release build.