# AUDCAD 2023 Local/Dukascopy Data Integrity Certification

Saved: 2026-09-16  
Project: LSTM Trader / ExpertAdvisor

## Frozen conclusions

The 2023 investigation established a defensible repair and activity architecture for AUDCAD.

**Price side:** ASK is the production side. Keep local ASK OHLC when the local candle exists; use Dukascopy ASK OHLC only when the local candle is absent; otherwise mark the observation missing. Do not interpolate, forward-fill, synthesize OHLC, or substitute correlated pairs.

**Clock mapping (empirically certified for 2023):**
- Through the March 24 trading period: local = Dukascopy UTC − 5 hours.
- From the March 26 trading period through October 27: local = Dukascopy UTC − 4 hours.
- From the October 29 trading period: local = Dukascopy UTC − 5 hours.

This mapping was determined empirically against price data and should not be replaced by an assumed timezone rule without re-certification.

## Tick-volume result

The existing production candle definition is `SUM(raw vol)`. The user confirmed that `vol` represents the number of ticks in the unit of time.

On 7,650 strict-overlap 15-minute candles:
- 7,645 matched Dukascopy tick count exactly.
- Exact-match rate: **99.9346%**.
- Correlation, local `SUM(vol)` vs Dukascopy tick count: **0.999941**.
- Correlation, local `SUM(vol)` vs Dukascopy native ASK volume: **0.925357**.
- Causal 96-bar z-score correlation, local tick volume vs Dukascopy tick count: **0.999973**.
- Causal 96-bar z-score correlation, local tick volume vs Dukascopy ASK volume: **0.962359**.

Therefore the homogeneous reconstructed activity feature should use **Dukascopy tick count**. Dukascopy native ASK volume is a separate candidate feature and must not be treated as the same unit.

Do not redefine local candle volume as raw `COUNT(*)`: the five discrepant candles showed that raw row count can differ from `SUM(raw vol)`.

## Five tick-volume discrepancies

| Local time | Local SUM(vol) | Duka ticks | Difference |
|---|---:|---:|---:|
| 2023-03-09 09:45 | 2173 | 3012 | -839 |
| 2023-07-27 12:45 | 1015 | 1233 | -218 |
| 2023-04-03 01:45 | 790 | 887 | -97 |
| 2023-04-03 02:45 | 976 | 1024 | -48 |
| 2023-07-06 04:45 | 1659 | 1695 | -36 |

All discrepancies have the same direction: the local source contains less tick activity.

A direct ASK-OHLC check of these five suspected partial outages found maximum selected-candle discrepancies of only 0.3–1.0 pip. The worst selected candle was March 9 at 1.0 pip; its high matched exactly. This does not justify adding a general partial-outage price-replacement rule.

## Source-hole findings

The 2023 local dataset contains structured multi-pair acquisition outages. Independent Dukascopy data recovers a large fraction of missing AUDCAD observations, but Dukascopy itself also contains source holes. Targeted redownloads of known Dukascopy holes returned no ticks, so repeated targeted retrieval should not be used as a repair strategy.

Cross-pair witness analysis showed that some residual AUDCAD gaps occurred while other major FX pairs were active. Witness counts must be described precisely: the prior label `NO_CROSS_PAIR_ACTIVITY` should be replaced by `NO_3PLUS_WITNESS_CONFIRMATION`, because many intervals had one or two active witness pairs.

## Frozen reconstruction policy

```text
PRICE
  local ASK OHLC
      else Dukascopy ASK OHLC
      else MISSING

ACTIVITY
  Dukascopy tick_count consistently across the reconstructed dataset

ADDITIONAL CANDIDATE FEATURE
  Dukascopy native askVolume

PROVENANCE
  preserve price source, activity source, and original local values
```

Do not overwrite `audcadrmp`. Build a persistent provenance-bearing reconstructed layer.

## Next phase

Prototype complete 2010–2025 reconstruction on AUDCAD before scaling to all FX pairs. Regenerate 15m/1h/4h/1d candles from the reconstructed layer and rerun the causal Fibonacci/feature/label pipeline end-to-end before bulk historical acquisition is generalized.
