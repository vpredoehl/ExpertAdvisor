---
title: "LSTM TG4A Pre-Study Configuration Freeze"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_TG4A_PreStudyConfigurationFreeze_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM TG4A Pre-Study Configuration Freeze

## Verdict: GO / CLOSED

TG4A is prospectively frozen, tested, committed, and Release provenance-valid. No real TG4 study was run, and no TG4 outcome-bearing data—including 2025 confirmation outcomes—was inspected.

Commit: `35aada569905c1ae522500cb11f39a0a5379d296`
Configuration fingerprint: `fnv1a64:bf809ce38a4a444a`

### Frozen decisions

- `referenceBarScale=14`
  - Experimental.
  - Tied prospectively to the existing 14-bar Wilder ATR horizon.
  - Represents projected 14-bar movement measured in creation-time ATRs, independent of pixels, zoom, aspect ratio, and quote magnitude.
  - The example/test value 20 was not adopted.

- `tg3_retracement_ratios=0.6180339887498949`
  - Experimental one-element set.
  - Nearest binary64 serialization of `(sqrt(5)-1)/2`.
  - No generic 0.382, 0.5, 0.786, or extension levels were imported.

- `tg3_price_tolerance_convention=canonical_fx_pips`
  - Experimental tolerance: `1` pip, inclusive.
  - Effective tolerance is `0.0001` for AUDCAD, AUDUSD, EURUSD, GBPUSD, and USDCAD; `0.01` for USDJPY.
  - This replaces the economically inconsistent global raw-price tolerance while preserving TG3’s causal `abs(P-L) <= T` semantics.

The source defines the angle bands and UTL/up-AB direction. ATR normalization, causal AB selection, level equations, and inclusive comparison are implementation-defined. The three frozen numeric choices are explicitly experimental.

### Principal artifacts

- [Frozen configuration](</Volumes/Developer SSD/ExpertAdvisor/Scripts/tg4_analysis_config.frozen_v1.conf>)
- [Methodology freeze](</Volumes/Developer SSD/ExpertAdvisor/docs/phases/target-generation/PhaseTG4/TG4A-Pre-Study-Methodology-Freeze.md>)
- [TG4A validation output](</Volumes/Developer SSD/ExpertAdvisor/docs/phases/target-generation/PhaseTG4/LSTM_TG4A_PreStudyConfigurationFreeze_Output.md>)

The frozen file SHA-256 is `78187108b908b4298eb9ae2f1e9fa1de63eadb1aceb29d09ab5261c42a7a9444`.

### Validation

All passed:

- TG1A, TG1B, TG2, TG3, TG4, and TG4 boundary tests
- Frozen-config loading, symbol-aware tolerance, fingerprint, and pre-2025 boundary tests
- Model input-width and compatibility regressions
- Standalone TG4 CLI strict-warning compilation
- ASan and UBSan
- Clang static analyzer
- Shell syntax checks
- `git diff --check` and `git diff --cached --check`
- Required `LSTM Release` build: `** BUILD SUCCEEDED **`

The scheduler remained running at PID 68183 with zero active or unmanaged workers. Nothing was published or restarted.

### Files changed

Thirteen TG4A-scoped files were committed: TG4 header/core/CLI, frozen and example configs, three TG4 test files, and five TG4/TG4A documentation files.

Commit stat:

```text
13 files changed, 601 insertions(+), 41 deletions(-)
```

Final repository state:

```text
git status --short
# clean; no output

git diff --stat
# empty; no output
```

### First outcome-bearing command — not executed

The proposed output directory is currently absent.

```bash
Scripts/run_tg4_historical_empirical_evaluation.sh \
  --config "/Volumes/Developer SSD/ExpertAdvisor/Scripts/tg4_analysis_config.frozen_v1.conf" \
  --output-dir "/Volumes/Developer SSD/ExpertAdvisor/Artifacts/tg4-preconfirmation-2010-2025-v1" \
  --all-canonical-symbols \
  --study tg4-preconfirmation-2010-2025-v1
```

This named study sets both scoring and outcome-end boundaries exclusively to `2025-01-01`; no 2025 bar can be loaded even to resolve a pre-boundary outcome.

Remaining limitations are explicit: all three inputs remain experimental, the ratio set intentionally covers only one level, and the previously documented USDJPY historical gap remains unclassified.