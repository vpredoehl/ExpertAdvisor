# TG4A pre-study methodology freeze

## Decision and anti-leakage record

TG4A freezes the first outcome-bearing TG4 study prospectively. No real TG4
historical study was run, and no TG4 exploratory, calibration, validation, or
confirmation observation, cohort, rate, target, retest, confluence, return, or
profitability outcome was inspected or used. In particular, no 2025 TG4
confirmation outcome was accessed.

The frozen configuration identity is
`tg4a-first-study-preconfirmation-frozen-v1` in
`Scripts/tg4_analysis_config.frozen_v1.conf`. The first study identity is
`tg4-preconfirmation-2010-2025-v1`; both scored input and outcome resolution
end exclusively at `2025-01-01T00:00:00Z`.

The audited TG4 implementation commit is
`8cc823239ed34b068009c92c50a185216db01158`. The working baseline also contains
two documented later scheduler/inference commits, `3959a73` and `f68c9d8`,
which do not alter TG4 methodology. Existing stashes were inventoried and left
untouched.

## Evidence classes

- Source-defined: the inclusive LongTerm 12-20 degree, Outer 25-40 degree,
  and Inner 45-85 degree bands; and the supplied direction statement, “See if
  Outer UTL is within the up AB Fibonacci.”
- Implementation-defined: creation-time Wilder ATR normalization; the TG1B
  arctangent formula; causal five-bar fractals; the most-recent-prior-opposite-
  confirmed-fractal AB policy; retracement equations; break-bar observation;
  TG2 Outer pairing; and inclusive absolute-price comparison.
- Experimental prospective conventions: `referenceBarScale=14`, the single
  ratio `0.6180339887498949`, and a tolerance of one canonical FX pip. These
  values are not attributed to the supplied trading source.

## Decision 1: TG1B reference-bar scale

Frozen value: `14` completed 15-minute bars.

The implementation first forms slope in ATRs per completed bar:

```text
normalizedSlope = rawSlopePerBar / creationAtr
angle = atan(abs(normalizedSlope) * referenceBarScale) * 180 / pi
```

The frozen horizontal normalized unit is the same 14-bar horizon already used
by the frozen Wilder ATR period. Consequently the angle compares the projected
14-bar price change with one creation-time ATR. It is independent of pixels,
zoom, aspect ratio, price level, and quote magnitude. This is an experimental
coordinate convention anchored to an existing implementation horizon, not an
empirically selected classifier scale. The test/example value 20 was not used
as evidence and is not the frozen value.

With scale 14, the fixed bands mean the following absolute projected movement
over 14 bars, measured in creation-time ATRs:

| Class | Degrees | ATR movement over 14 bars | Absolute ATR slope per bar |
|---|---:|---:|---:|
| LongTerm | 12-20 | 0.2125566-0.3639702 | 0.0151826-0.0259979 |
| Outer | 25-40 | 0.4663077-0.8390996 | 0.0333077-0.0599357 |
| Inner | 45-85 | 1.0000000-11.4300523 | 0.0714286-0.8164323 |

The gaps between bands remain unclassified exactly as TG1B defines them.

## Decision 2: TG3 Fibonacci ratio set

Frozen serialization: `0.6180339887498949`.

The source supplies no numeric ratio. TG4A therefore uses the smallest
possible experimental set: one retracement ratio. It is the canonical
mathematical limiting ratio of consecutive Fibonacci numbers,
`(sqrt(5)-1)/2`, serialized as the nearest binary64 round-trip decimal. No
additional folklore levels (including 0.382, 0.5, 0.786, extensions, or a
multi-level ladder) are imported. TG3's implementation-defined equations
remain unchanged: for an up AB range `R`, the level is `B - ratio * R`.

This choice is a deliberately narrow experimental operationalization of the
source's otherwise unspecified word “Fibonacci,” not a claim that the source
named 61.8 percent and not a value selected from historical success.

## Decision 3: TG3 price tolerance

Frozen convention: `canonical_fx_pips`, value `1` pip, inclusive.

The repository has five non-JPY canonical pairs and one JPY pair. All six raw
tables store bid/ask as `numeric(10,6)`, but a read-only unit/precision audit
found symbol-dependent quote grids. January 2024 source quotes contain
0.00001 increments for the five non-JPY pairs and 0.001 increments for
USDJPY; January 2010 source quotes use 0.0001 and 0.01 respectively. No TG4
event or outcome was produced by this audit.

A single raw absolute value would therefore have different pip meaning across
the universe. TG4 now converts the frozen pip count deterministically:

| Symbols | Canonical pip size | Effective inclusive absolute tolerance |
|---|---:|---:|
| audcadrmp, audusdrmp, eurusdrmp, gbpusdrmp, usdcadrmp | 0.0001 | 0.0001 |
| usdjpyrmp | 0.01 | 0.01 |

One pip is an experimental proximity convention. It is prospective, uses a
common market unit, and has consistent pip meaning across symbols. TG3 still
performs its existing causal check `abs(Outer projection - level) <= effective
absolute tolerance`; no TG3 pairing, AB, timing, or outcome semantics changed.

The effective per-symbol values, pip convention, pip count, ratio set,
reference scale, and all other configuration values are serialized in
`metadata.json` and covered by `configuration_fingerprint`. The fingerprint
is FNV-1a-64 and is an identity checksum, not a security primitive.

## Temporal lock and first-study plan

The half-open partitions remain:

- exploratory `[2010-01-01, 2020-01-01)`;
- calibration `[2020-01-01, 2023-01-01)`;
- locked-rule validation `[2023-01-01, 2025-01-01)`;
- untouched confirmation `[2025-01-01, 2026-01-01)`.

The preconfirmation named study loads and scores only timestamps before
`2025-01-01`, and its outcome end is the same exclusive boundary. A finite TG2
window that cannot resolve before that boundary is censored at end of input;
no 2025 bar can resolve it. This is stricter than the older full named study,
whose outcome-only tail can extend beyond a score boundary.

After review, the operator may run this exact command. TG4A did not run it:

```bash
Scripts/run_tg4_historical_empirical_evaluation.sh \
  --config Scripts/tg4_analysis_config.frozen_v1.conf \
  --output-dir /absolute/new/path/tg4-preconfirmation-2010-2025-v1 \
  --all-canonical-symbols \
  --study tg4-preconfirmation-2010-2025-v1
```

The output directory must be new. Database defaults may be replaced by an
explicit read-only `--connection` string without changing the study identity.

## Remaining limitations

- The three frozen inputs remain experimental because the supplied source is
  silent; prospective freezing makes them reproducible, not source-authored.
- The 14-bar scale and one-pip zone are measurement conventions, not claims of
  optimality.
- A single ratio intentionally tests only the canonical Fibonacci limit
  retracement and cannot characterize other commonly discussed levels.
- No TG4 outcome evidence exists yet. Any later change to these values creates
  a new configuration identity and requires a separately declared study;
  validation or confirmation results must never rewrite this freeze.
