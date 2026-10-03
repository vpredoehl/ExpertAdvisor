---
title: "Price-Level Structure Phase 5A — Characterization Harness"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_PriceLevelStructure_Phase5A_CharacterizationHarness_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Price-Level Structure Phase 5A — Characterization Harness

Implemented and staged a read-only Phase 5A characterization harness. Frozen v1, Layout11, Tensor/LSTM, scheduler, DB schema, and experiments 702/703 are unchanged.

Authoritative data path: reused the canonical 15-minute completed-candlestick contract in `CanonicalMarketDataRange.hpp`, following the existing streaming pattern in `TG4HistoricalMarketDataRepository.cpp`. `MarketDataCore` materializes this same path; the harness streams it to avoid whole-history buffering.

Changed files:

- [PriceLevelCharacterization.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/PriceLevelCharacterization.hpp>) — one-pass strict-pivot and causal preceding-range scale research helpers.
- [PriceLevelCharacterizationCLI.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PriceLevelCharacterizationCLI.cpp>) — standalone read-only CLI; CSV/manifest output and optional v1 candidate replay metrics.
- [build_price_level_characterization.sh](</Volumes/Developer SSD/ExpertAdvisor/Scripts/build_price_level_characterization.sh>) — standalone build outside product targets.
- [PriceLevelCharacterizationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/PriceLevelCharacterizationTests.cpp>) and runner — v1 pivot parity, causality, chronology, determinism.
- [Phase5A_Characterization_Harness.md](</Volumes/Developer SSD/ExpertAdvisor/docs/phases/PriceLevel/Phase5A_Characterization_Harness.md>) — usage/contract documentation.

Usage:

```bash
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5a_characterization

/tmp/price_level_phase5a_characterization \
  --start 2023-01-01 --end 2026-01-01 \
  --output-dir /tmp/price_level_phase5a_2023_2025
```

Outputs: `pivot_radius.csv`, `scale.csv`, `detector_metrics.csv` (populated only with explicit `--v1-candidate`), and `manifest.txt`. Scale probes are preceding completed `high-low` ranges; pivot-time and confirmation-time semantics are separately reported. No adaptive detector was introduced.

Initial read-only run, all 28 symbols, 2023–2025:

- 1,998,797 bars; 18.13 seconds wall-clock.
- Aggregate strict pivots:

| Radius | Total pivots | Per 100 bars |
|---:|---:|---:|
| 1 | 872,487 | 43.65 |
| 2 | 516,388 | 25.83 |
| 3 | 368,061 | 18.41 |
| 4 | 284,820 | 14.25 |
| 6 | 198,085 | 9.91 |
| 8 | 152,694 | 7.64 |

The 2025 per-symbol median 15-minute range spans 0.00033 (EURGBP) to 0.13 (GBPJPY), about 394×. This further rules out a single raw-price width; it does not select an adaptive lookback or multiplier. No detector candidate was run, because no defensible production-compatible configuration is authorized.

Validation run:

```bash
Tests/PriceLevelCharacterizationTests.sh
Tests/CausalPriceLevelEngineTests.sh
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5a_characterization
```

All passed. Two independent canonical one-symbol replays produced byte-identical CSVs. No final product binaries or `xcodebuild` targets were built.

Git evidence:

- Starting branch/HEAD: `lstm-feature-development` / `2d6ddaef113d51a52dd97fa03ace7bdd3af039af`
- Starting worktree: clean.
- Final `git status --short`: six staged additions listed above.
- Staged diff: 1,043 insertions across 6 files; unstaged `git diff --stat` is empty.
- Proposed commit: `Add Price-Level Phase 5A characterization harness`

Remaining decision: select/configure a candidate detector study only after explicitly choosing research parameters; this implementation intentionally makes no production configuration recommendation.