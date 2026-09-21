# TG1A causal fractal trend-line geometry

TG1A is a reusable diagnostic geometry layer in
`Headers/CausalFractalTrendLineGeometry.hpp`. It is deliberately not connected
to `Tensor`, `FeatureLayout`, feature ablation, model input widths, semantic
layout version 7, or the semantic-worker registry. Adding TG1A therefore does
not change an existing model's inputs or the meaning of a published worker.

## Fractal and causal contract

The implementation uses the project's strict five-candle definition. A center
high must be greater than all four neighbors and a center low must be less than
all four neighbors. Equality prevents detection. The engine consumes completed
bars in increasing timestamp order and examines the center of the most recent
five-bar window only when the fifth bar closes.

Consequently, a center at bar `i` retains `i` as its geometric anchor but is
published as a `ConfirmedFractal` only at bar `i + 2`. Anchor and confirmation
bar/timestamp fields are separate. Historical ingestion stable-sorts rows by
timestamp and then calls the identical streaming update path; duplicate
timestamps are rejected.

The database audit found the same semantics in
`Database/indicators/fractal.plpgsql`: strict `max(high) < high` and
`min(low) > low` comparisons over `ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
EXCLUDE CURRENT ROW`. Its source row is named `dt` by the `cst` return type in
`Database/forex/candlestick.plpgsql`, so the actual window says `ORDER BY dt`;
the positional output field in `fractal_type` is named `ts`. The window is
therefore chronologically ordered even though the two composite types use
different timestamp field names.

The legacy `current_high` and `current_low` helpers filter the output of an
otherwise unbounded `fractal(...)` call. That is safe for retrospective queries
but is not a point-in-time availability contract: a caller can ask for a time
whose latest returned center was evaluated using later rows. TG1A does not use
those helpers and enforces confirmation in its consumer-side streaming state.
The fixed-year `fractal_mv` helper also remains legacy date-range code. Neither
legacy area was rewritten because it is outside TG1A and does not block the new
causal path.

## Candidate construction and validity

When a low fractal confirms, it is paired with every retained earlier low whose
price is strictly lower. Each valid pair creates a UTL. A high fractal is paired
symmetrically with every retained earlier high whose price is strictly higher
to create a DTL. A third touch is not required.

For anchors `(i1, p1)` and `(i2, p2)`, the canonical horizontal coordinate is
the sequential completed-bar index:

```text
slope = (p2 - p1) / (i2 - i1)
projected(j) = p1 + slope * (j - i1)
```

Intervening bars satisfy:

```text
UTL: low[j] + interveningPriceTolerance >= projected(j)
DTL: high[j] - interveningPriceTolerance <= projected(j)
```

Both `interveningPriceTolerance` and `touchPriceTolerance` are explicit,
nonnegative absolute price-unit configuration values; both default to zero.
TG1A does not infer tick size and does not introduce ATR-dependent validity.
Callers must select values appropriate for the instrument's price precision.

## Touches and diagnostics

`candleTouchCount` counts non-anchor candle extremes within
`touchPriceTolerance` of the line, including interactions already observable
when anchor 2 confirms and later completed bars. `fractalTouchCount` counts
confirmed same-kind fractals within that tolerance, including the two anchors.
The two measurements are independent. A line continues to be measured after a
later cross; interpreting a cross as a break or a trade signal is outside TG1A.

The current raw distance uses a directionally consistent sign:

```text
UTL: current low - projected line
DTL: projected line - current high
```

Positive values are on the supporting/resisting side, and negative values are
beyond the line. A completed-bar ATR diagnostic uses `atrPeriod` (default 14)
and the same Wilder recursive update convention as the existing `Tensor` ATR
(`alpha = 1 / period`, seeded by the first true range). Raw slope and distance
are divided by a positive ATR for optional diagnostics. ATR never changes
fractal detection, candidate validity, or touch status. No slope-to-angle
conversion or angle class exists.

`FormatDiagnostic` emits symbol, timeframe, both anchors and their separate
confirmations, creation time, separation, raw and ATR-normalized slope, age,
the two touch counts, current projection, and raw/ATR-normalized distance.

## Deterministic bounds and complexity

The defaults are:

- anchor lookback: 512 bars;
- retained confirmed fractals: 64 per kind;
- candidate lifetime: 512 bars after creation;
- total live candidates: 4096.

All candidates produced by retained anchors coexist. The engine discards
anchors outside the configured bar horizon and, if necessary, the oldest
confirmed fractals first. Candidates expire only after their configured
creation-age limit. If the explicit candidate cap is exceeded, eviction is
deterministic by `(creation bar, direction, anchor1 bar, anchor2 bar)`, oldest
first. Public candidate ordering is deterministic by
`(direction, anchor1 bar, anchor2 bar, creation bar)`.

With `F` retained fractals per kind and `C` live candidates, a normal bar update
is `O(C)`, and a confirming bar adds `O(F)` pairing work plus bounded
intervening-bar validation. Storage is bounded by the configured bar, fractal,
and candidate horizons. It cannot grow quadratically over the complete input
history.

## Scope boundary

TG1A contains no Long-Term/Outer/Inner classification, chart angles,
Fibonacci logic, probabilities, break/retest strategy, labels, entries, exits,
stops, targets, or sizing. TG1B provides a separate diagnostic calibration and
classification layer without changing this TG1A contract; the remaining
strategy semantics remain later-phase concerns.
