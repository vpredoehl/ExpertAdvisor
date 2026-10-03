# Price-Level Phase 5A characterization harness

`Scripts/build_price_level_characterization.sh` builds the standalone,
read-only Phase 5A executable. It uses the canonical absolute half-open
15-minute candlestick query and processes only completed OHLC bars. It never
selects `target`, writes the database, or loads Tensor/LSTM code.

Build outside the repository products:

```bash
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5a_characterization
```

Run an explicit interval; `--end` is exclusive. The default symbol list is
the 28 production FX pairs and the default pivot-radius study is
`1,2,3,4,6,8`.

```bash
/tmp/price_level_phase5a_characterization \
  --start 2023-01-01 --end 2026-01-01 \
  --output-dir /tmp/price_level_phase5a_2023_2025
```

Optional controls are `--symbols`, `--pivot-radii`, and
`--scale-lookbacks`; comma-separated values are required. The default scale
lookbacks (`32,96,384`) are research probes, not approved configuration
values.

`--v1-candidate radius,width,max_active,max_age,max_evidence` enables a
separate replay of the unchanged `causal-price-level/v1` engine and writes
structural interaction metrics. It is deliberately opt-in: the option does
not approve or persist the supplied values. `completedBarDuration` is fixed at
900 seconds for this harness because the source is the 15-minute production
candlestick layer.

Successful runs atomically publish the requested directory after writing:

- `pivot_radius.csv` — symbol and aggregate strict-pivot counts and rates.
- `scale.csv` — yearly raw-range and preceding-range scale distributions,
  including pivot-time and confirmation-time samples.
- `detector_metrics.csv` — populated only when a v1 candidate was requested;
  it records interactions, active-level distribution, ended lifetimes,
  pivot observations per ended level, and retained-evidence saturation.
- `manifest.txt` — source contract, causal-scale boundary, date range, bar
  count, elapsed runtime, and the explicitly requested candidate identity.

The adaptive-width investigation is scale-only. Pivot-time scale means the
rolling statistic before the originating pivot bar; confirmation-time scale
means the statistic before the later confirmation bar. In both cases the
statistic contains only preceding completed `high-low` ranges. No adaptive
width detector is implemented here, so v1 zones and identities remain frozen.
