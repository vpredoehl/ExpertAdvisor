# Phase Pocket 1 - source-grounded Pocket specification and causal-detector decision

## Status and evidence boundary

**Specification completed from the authoritative manual; detector not
implemented because material creation and causality rules are unspecified.**
This source-grounded re-run replaces the prior attempt's claim that the PDF
was absent.

### Authoritative source and citation convention

* Market Traders Institute, Inc., *Pockets* manual:
  `ResearchSources/Pockets/Pockets.pdf` (106 PDF pages; metadata creation
  2019-11-19).
* Citations give printed manual page then PDF page, e.g. `p. 48 (PDF p. 53)`.
* Definition diagrams on manual pp. 44-45 (PDF pp. 49-50) were visually
  inspected. Only their unambiguous labels are used; no OHLC formula is
  inferred from unlabelled graphic geometry.

The source is ignored locally by `/ResearchSources/`; it was read only and is
not changed or staged. No web, remembered, prior-report, or project claim is
Pocket authority.

## Classification vocabulary

| Classification | Meaning |
| --- | --- |
| **EXPLICIT** | Directly stated or unambiguously labelled/shown in the manual. |
| **DERIVED** | Deterministic consequence of explicit material. |
| **PROVISIONAL** | Earlier-project interpretation, not established by the manual. |
| **DISCRETIONARY** | Manual-supported trader judgement with no objective encoding. |
| **UNSPECIFIED** | Required for deterministic implementation but absent from the manual. |

## Pocket structure: source-grounded inventory

| Rule | Class | Traceability and bounded interpretation |
| --- | --- | --- |
| Pocket is a market inefficiency associated with aggressively bullish/bearish, abnormal-looking candles. | **EXPLICIT** | Definition, p. 43 (PDF p. 48). Qualitative, not a numerical spike predicate. |
| It may arise from a fundamental announcement or aggressive push. | **EXPLICIT** | Definition, p. 43 (PDF p. 48). Context, not an event-data requirement. |
| Pockets are valid on any timeframe. | **EXPLICIT** | Definition, p. 46 (PDF p. 51). |
| Higher-timeframe pockets take longer to confirm. | **EXPLICIT** | Definition, p. 46 (PDF p. 51). Latency not quantified. |
| Indicator adjusts to current chart timeframe. | **EXPLICIT** | Indicator, p. 48 (PDF p. 53). No repository aggregation contract follows. |
| Default indicator lookback is 15 candles. | **EXPLICIT** | Indicator, p. 48 (PDF p. 53). A default setting, not immutable structure. |
| Bullish indicator identifies a highest high; bearish identifies a lowest low in the lookback. | **EXPLICIT** | Indicator, p. 48 (PDF p. 53). Inclusion, ties, trigger, and relation to range endpoints are absent. |
| Bullish Pocket is breakout/buy direction; bearish Pocket is breakout/sell direction. | **EXPLICIT** | Breakout automation, p. 81 (PDF p. 86). Strategy terminology, not creation predicate. |
| Bullish graphic labels upper boundary `touch` and lower `close`; bearish reverses labels. | **EXPLICIT** | Visually inspected diagrams, pp. 44-45 (PDF pp. 49-50). No OHLC formula, inclusivity, or wick/close contact rule. |
| Direction, range endpoints, and a creation bar are calculable from those facts. | **UNSPECIFIED** | No complete boolean candle/window predicate, endpoint formula, or threshold for “aggressively.” |

### Creation, boundaries, confirmation, timeframe, lifecycle

The source supports only this narrow structural statement: its indicator
defaults to a 15-candle lookback, uses a highest-high reference for bullish
output and a lowest-low reference for bearish output, and diagrams a
two-boundary range with direction-specific touch/close labels.

It does **not** state which bars belong in the lookback; whether the event is a
close beyond, wick beyond, gap, or multi-candle displacement; how much
displacement qualifies; which OHLC values form the other boundary; or equal
extrema handling. “Pocket equals the extreme” and any exact range formula are
therefore not **DERIVED**.

Breakout text names “immediate confirmation” and says a breakout is confirmed
when the Pocket appears (pp. 52 and 81; PDF pp. 57 and 86). Confirmation is
**EXPLICIT** as a concept. The confirmation bar, completed-bar requirement,
future evidence, and latency are **UNSPECIFIED**.

The manual describes 1-hour, 4-hour, 8-hour, daily, and weekly pockets and
advises marking a larger-timeframe Pocket with ray/rectangle before moving to
a smaller chart (pp. 46 and 48; PDF pp. 51 and 53). This is **EXPLICIT**
chart-use guidance. Aggregation, session/time-zone alignment, missing-bar
policy, and mechanical cross-timeframe joining are **UNSPECIFIED**.

“Available” Pockets and touch/close targets are discussed, but no structural
expiry, replacement, or invalidation predicate is defined. A later trade,
touch, or close must not mutate the original event. Lifecycle is
**UNSPECIFIED**.

## Pocket structure is not execution

### Reversion strategy

Reversion trades back to a Pocket/inefficiency. Automation can wait for a
user-selected pip distance or ATR percentage; entry count is trader-selected;
profit options are total pips, close, or touch (pp. 60-61 and 77-79; PDF
pp. 65-66 and 82-84). Manual best practice says wait for trendline break,
noticeable price/oscillator divergence, topping/bottoming pattern, and
higher-timeframe support/resistance (p. 60; PDF p. 65).

These are **EXPLICIT** reversion concepts. Filters/manual interpretation are
**DISCRETIONARY**; pip distance, ATR percentage, entry count, target, and
tier-out are user-configured execution, never Pocket structure.

### Breakout strategy

Breakout is trend/momentum trading in Pocket direction. The manual permits two
entries per Pocket: immediate-confirmation entry plus one at/near close with
user-defined pip offset (p. 52; PDF p. 57). Automation repeats
bullish=buy/bearish=sell and discusses same-direction consecutive Pockets,
trailing stops, and profit management (pp. 81-84; PDF pp. 86-89).

Those are **EXPLICIT** strategy semantics, not structural creation. Exact
trigger/fill/order rules are **UNSPECIFIED** and out of scope.

### Related concepts

| Concept | Source-grounded role | Classification |
| --- | --- | --- |
| ATR | Reversion uses selected ATR percentage away from close; breakout can use Daily 20-period ATR percentage for trailing stop/profit (pp. 52, 60, 79, 81, 83-84; PDF pp. 57, 65, 84, 86, 88-89). | **EXPLICIT** execution/management; formula/bar construction and creation role **UNSPECIFIED**. |
| Fractals | Optional breakout trailing stop and manual trailing-stop choice (pp. 52-53, 68, 81, 83; PDF pp. 57-58, 73, 86, 88). | **EXPLICIT** management; not Pocket creation/confirmation. |
| Trendlines | Larger-TF lines connect wick highs/lows; reversion waits for a break (pp. 26, 60; PDF pp. 31, 65). | **EXPLICIT** discretionary filter; objective construction/break **UNSPECIFIED**; not structural. |
| Support/resistance | Higher-TF reversion context; larger TF said more significant (pp. 19, 60; PDF pp. 24, 65). | **EXPLICIT** discretionary context; level identity/tolerance/lifecycle **UNSPECIFIED**. |
| Divergence | Wait for noticeable price/momentum-oscillator divergence for manual reversion. | **EXPLICIT** discretionary filter (p. 60/PDF p. 65); oscillator/settings/pivots **UNSPECIFIED**. |
| Price patterns | Used to trade toward Pockets and in manual reversion (pp. 40-41, 60, 68-69; PDF pp. 45-46, 65, 73-74). | **EXPLICIT** discretionary context; no pattern is creation. |

## Prior Project Assumptions Audit

| Prior interpretation | Finding against manual | Classification |
| --- | --- | --- |
| 21-candle history | Manual default is 15, not 21 (p. 48/PDF p. 53). | **PROVISIONAL**; contradicted as asserted default. |
| Pocket = highest/lowest history point | Relevant extreme is identified, but graphic shows range and no formula makes Pocket equal to extreme. | Extreme **EXPLICIT**; equality **PROVISIONAL**. |
| Bullish/bearish spike semantics | Aggressive/abnormal candles stated; no measurable size/body/wick/ATR/news rule. | Concept **EXPLICIT**; exact spike **DISCRETIONARY/UNSPECIFIED**. |
| Fixed 30-pip levels | User-entered `X` pips and ATR percentages, no fixed 30-pip structural rule. | **PROVISIONAL**, unsupported. |
| Maximum ten levels | Reversion maximum trades user-selected (example five); breakout only two trades/Pocket. | **PROVISIONAL**, unsupported. |
| Exact upper/lower boundary | Graphics label directional touch/close boundary order, no OHLC endpoint formula. | Order **EXPLICIT**; formula **UNSPECIFIED**. |
| Exact touch definition | Touch is a labelled level/profit option; wick/bid/ask/intrabar/close absent. | Role **EXPLICIT**; detection **UNSPECIFIED**. |
| Exact close definition | Close is a labelled level/profit option; exact condition absent. | Role **EXPLICIT**; detection **UNSPECIFIED**. |
| Exact confirmation bar/time | Confirmation named and larger TF takes longer, no predicate/latency. | **UNSPECIFIED**. |
| Exact invalidation/closure | No structural invalidation, expiry, replacement rule. | **UNSPECIFIED**. |
| Higher-TF Pocket/lower-TF execution | Marking a larger Pocket and a daily-Pocket example exist, no mechanical mapping. | Guidance **EXPLICIT**; implementation **UNSPECIFIED**. |

Legacy `Database/indicators/pocket.plpgsql` is not source evidence. It uses
`max(high) over (rows 15 preceding)`, `lead(low,1)`, and unbounded later
scans for touch/close/excursions: future-data dependent, asymmetric, and not
causal.

## Causal domain model (design boundary only)

If a later authoritative decision supplies the predicate, the smallest
strategy-free representation should preserve event and confirmation separately:

```cpp
enum class PocketDirection { Bullish, Bearish };
struct PocketPriceRange { double lower; double upper; };
struct PocketObservation {
    PocketDirection direction;
    PocketPriceRange range;
    std::size_t eventBar;
    std::int64_t eventTimestamp;
    std::size_t confirmationBar;
    std::int64_t confirmationTimestamp;
    std::string sourceTimeframe;
};
```

This is **PROVISIONAL** design, not manual field nomenclature. It excludes
entries, grids, stops, ATR settings, targets, P&L, touch/fill state, and
invalidation. Required future invariant:

```text
eventBar <= confirmationBar
information cutoff == confirmationBar
emit only after completed confirmation bar
```

The manual does not resolve equality or latency. Centered calculations are
illegal unless their future-bar count is explicit and emission is delayed.

## Repository audit

| Component | Exact file/API | Reuse finding |
| --- | --- | --- |
| Completed-bar causal model | `Headers/CausalFractalTrendLineGeometry.hpp`: `EA::TG1A::Candle`, `ConfirmedFractal`, `AddCompletedBar`, `ValidateCandle`, `DetectLatestFractals` | Pattern for valid OHLC, increasing completed timestamps, deterministic replay, event/confirmation coordinates. Five-bar/radius-two fractal is TG1A-only. |
| Candlestick access | `Sources/MarketDataCore/MarketDataCore.hpp/.cpp`: `EA::MarketData::LoadCandlesticks`; `Database/forex/candlestick.plpgsql`: `candlestick`, `candlestick_cur`, `candlestick_mv` | OHLC/read patterns. `LoadCandlesticks` hard-codes 15-minute data, not timeframe-agnostic adapter; not exercised. |
| Database fractals | `Database/indicators/fractal.plpgsql`: `fractal`, `fractal_mv`, `current_high`, `current_low` | Centered two-before/two-after pivot needs future bars and has no confirmation timestamp. Do not alter or use as Pocket semantics. |
| Existing ATR | `Headers/CausalFractalTrendLineGeometry.hpp`: `UpdateAtr` | Causal recursive ATR exists but is no proof for Pocket creation/management. |
| Historical support/resistance | `Headers/CausalHistoricalLevelProximityFeatures.hpp`: `CausalHistoricalLevelProximity::AddCompletedBar` | Separate weekly corroborated scalar with its own 104-week/pivot policy; cannot substitute for judgement. |
| Empirical architecture | `Headers/TG4HistoricalEmpiricalEvaluation.hpp`: `EvaluationConfiguration`, `HistoricalEvaluator`; `Sources/TG4HistoricalMarketDataRepository.cpp`: `StreamCanonicalCandles` | Potentially later only after frozen contract and authority. No TG4 data/artifact/configuration opened, changed, or run. |

## Implementation Eligibility Decision

**Not eligible for a faithful structural detector.** The manual lacks: (1)
qualifying ordered bar/window predicate, lookback membership, and tie policy;
(2) both OHLC endpoint formulas; (3) completed-bar confirmation rule/latency;
and (4) wick/close contact and lifecycle semantics if touch/close state is
required.

No Pocket C++/SQL, database mutation, LSTM feature, target change, test,
historical run, or simulation was added. Minimum research decision: provide
an authoritative versioned rule fixing those four items. Phase Pocket 2 can
then implement a pure supplied-bar detector and source-backed synthetic tests
for directions, insufficient history, confirmation boundary, no premature
emission, boundaries, replay, consecutive events, and malformed input.
Reversion and breakout execution must remain separate.
