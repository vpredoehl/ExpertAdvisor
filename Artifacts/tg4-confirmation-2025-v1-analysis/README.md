# TG4A 2025 Confirmation Robustness Analysis

## Status

CLOSED — 2025 confirmation analysis frozen.

No further parameter selection, alternative temporal blocking, or
optimization is to be performed against this confirmation period.

## Source Study

Source artifact:

    Artifacts/tg4-confirmation-2025-v1

Primary input:

    Artifacts/tg4-confirmation-2025-v1/observations.csv

Observation SHA-256:

    487323b573a9b5df61fd60e82729827f4cfdebb222f7ef22937e993af10a263d

Source study configuration:

    tg4a-first-study-preconfirmation-frozen-v1

Baseline commit:

    8cc823239ed34b068009c92c50a185216db01158

Score period:

    2025-01-01T00:00:00Z through 2026-01-01T00:00:00Z exclusive

The source study was read-only and performed no parameter optimization.

## Frozen TG4A Semantics

TG4A confluence used the predefined inverse golden ratio:

    0.6180339887498949

with canonical FX tolerance of 1 pip.

TG4A was directionally defined only for UTL observations under:

    source_utl_up_ab_only

DTL observations were structurally ineligible for TG4A confluence.

No TG4A parameters were selected using the 2025 confirmation outcome.

## Primary Result

The original event-weighted comparison produced:

    confluent outer-target rate      0.8816739
    non-confluent outer-target rate  0.7268192
    absolute difference             +0.1548547

Because TG4 observations can share break events, trend structures,
A/B structures, and market episodes, additional post-confirmation
dependence diagnostics were performed without changing the frozen
TG4A specification.

## Dependence Diagnostic

Primary comparable population:

    UTL paired resolved observations  41,058
    confluent                           2,772
    non-confluent                      38,286

Structural multiplicity:

    unique break clusters              20,175
    unique outer candidates             4,124
    unique inner candidates            14,108
    unique A/B structures               9,303

Break clusters had zero mixed confluence classifications and zero
mixed outcomes.

## Unique-Break Robustness

After reducing the primary comparison to one observation per
(symbol, break_bar):

    confluent breaks                    1,456
    success rate                       0.867445

    non-confluent breaks               18,719
    success rate                       0.720712

    absolute difference               +0.146733
                                      +14.67 percentage points

The difference remained positive for every canonical symbol:

    AUDCAD  +12.28 pp
    AUDUSD  +15.69 pp
    EURUSD  +17.65 pp
    GBPUSD  +12.40 pp
    USDCAD  +15.42 pp
    USDJPY  +12.43 pp

## Within-Outer Robustness

Outer-candidate structure:

    outer candidates                    4,124
    confluent-only                        107
    non-confluent-only                  3,174
    mixed-confluence                      843
    mixed-outcome                       1,069

Within the 843 outer candidates containing both confluence states:

    pooled break difference            +15.24 pp
    equal-outer mean difference         +3.06 pp

Among non-tied outer candidates:

    informative outer candidates          356
    positive                               244
    negative                               112
    positive fraction                 0.685393
    exact two-sided sign-test p       2.16017241085e-12

The sign test is a directional robustness diagnostic under its unit
assumptions; it does not establish complete independence between
outer structures.

## Symbol-Week Temporal Robustness

A final temporal robustness test was predeclared using symbol x ISO
calendar week as the blocking unit. Unique break events were used,
and only blocks containing both confluence states participated.

Results:

    matched symbol-weeks                  286
    informative symbol-weeks              275
    positive                               213
    negative                                62
    ties                                    11
    positive fraction                 0.774545

    equal-symbol-week mean difference   +11.98 pp
    median difference                   +15.29 pp

    matched-week pooled difference      +14.60 pp

    exact two-sided sign-test p
        1.47168923742e-20

No alternative temporal block size was selected after inspecting
this result.

## Frozen Empirical Conclusion

Under the pre-confirmation frozen TG4A specification, UTL break
events with predefined 0.618 Fibonacci confluence were associated
with materially higher subsequent outer-target hit rates than
comparable non-confluent UTL break events.

The association persisted after:

- deduplication to unique break events;
- separate examination of all six canonical FX symbols;
- comparisons within shared outer structures; and
- symbol-calendar-week temporal blocking.

TG4A is therefore retained as a robust empirical feature candidate
under the frozen specification.

This result does NOT establish:

- causality;
- trading profitability;
- independence of every observation or analysis unit;
- validity for DTL observations; or
- validity of alternative Fibonacci ratios, tolerances, horizons,
  angle bands, or other parameterizations.

The 2025 confirmation period must not be reused to tune TG4A.

## Preserved Analysis

Analysis program:

    tg4_break_weighted.py

Program SHA-256:

    283f79d3dc6c7523500e9bfcc13b942ace69dfa450cfd44787ab5eb52b787bbe

Captured output:

    tg4_break_weighted_output.txt

Output SHA-256:

    0d1dfbcd2085ef11c3d0c84fd66a346f8725437c8310f07a77ff3a7e3c3a64e9

See SHA256SUMS for preserved provenance.

