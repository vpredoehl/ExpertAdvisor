# Causal Fibonacci layout-9 incremental-information protocol

**Protocol identifier:** `causal-fibonacci-layout9-incremental-information-v1`

**Status:** frozen before execution

**Scope:** read-only research screen; not a training experiment or an LSTM ablation

## 1. Question and scope

This protocol asks a narrower question than the completed structural-transition
study:

> Does the frozen causal Fibonacci representation contain forward-outcome
> information beyond the existing layout-8 baseline representation, under
> fixed, out-of-time, low-capacity diagnostics?

It separately reports:

1. **Representational redundancy:** how much of each Fibonacci column the
   contemporaneous baseline can reconstruct.
2. **Unconditional association:** descriptive outcome composition by Fibonacci
   state, without conditioning on the baseline.
3. **Conditional/incremental information:** the paired change in a fixed
   baseline-only versus baseline-plus-Fibonacci directional diagnostic.
4. **Temporal robustness:** the same frozen diagnostics in separately reported
   chronological partitions.
5. **Cross-symbol robustness:** results remain per symbol; summaries do not
   pool rows or declare the symbols independent market trials.

This is not evidence of causation, profitability, LSTM accuracy, or a
controlled LSTM ablation.  It is a gate before deciding whether such an
experiment is warranted.

No ratio, anchor policy, feature, target, diagnostic hyperparameter, partition,
or decision interpretation may be changed after incremental results are
inspected under this identifier.  A changed choice requires a new protocol
identifier and a separately held-out confirmation period.

## 2. Frozen input identity

The candidate family is exactly the 23 layout-9 columns produced by
`Headers/CausalFibonacciStructuralFeatures.hpp`, with configuration name
`causal-fibonacci-layout9-symmetric-structural-v1` from
`Headers/CausalFibonacciStructuralFeatureConfiguration.hpp`:

- one `fib_recent_price_scale_valid` bit;
- direction-separated recent H1/H2 union, H1, H2, and overlap log-counts;
- direction-separated H1/H2 youngest-event ages;
- direction-separated median signed ATR-normalized distances to 1.272, 1.618,
  .382, .500, and .618.

Its exact persisted-family identity is the concrete
`FeatureAblation::kCausalFibonacciStructuralAblationMaskText` list (all 23
names, in that canonical order), whose existing ablation identity hash is
`fnv1a64:a3f595680caadb2e`.  That identity is used here only to pin the feature
family; this screen neither creates nor evaluates that experiment pair.

The source is the existing causal TG1A/TG3-derived structural producer.  It has
the established completed-bar, delayed-pivot, prefix-invariant timing contract;
no future-confirmed pivot is projected back to its pivot bar.  The `.500` level
is a conventional retracement level, not a Fibonacci ratio.  The representation
is not modified by this protocol.

`FeatureLayout.hpp` fixes layout 9 at 99 Tensor columns.  Its model input is
103 columns: the 99 Tensor columns plus four model-only return features.  The
baseline is **the exact layout-8 model-input family**, 80 columns: Tensor
columns 0--75 (through the three TG4 pulses) followed by the four return
features.  The candidate increment is precisely Tensor columns 76--98.

An artifact uses logical `baseline[80]` then `fibonacci[23]` ordering so the
two inputs can be compared directly.  This is deliberately different from the
physical layout-9 model-input ordering, where the return suffix follows the
Fibonacci Tensor columns.  Both logical names and original Tensor/model-input
coordinates are recorded in the schema manifest.

The baseline therefore includes, where defined by the current authoritative
feature semantics, OHLC return/body/range/wick measures, ATR and volatility,
EMA distances/slopes, Donchian information, 16-bar range position, 8-versus-32
range expansion, return/persistence/imbalance/autocorrelation measures,
historical weekly-level proximity, economic features, the three layout-8 TG4
pulses, and the 1/4/8/16-bar model-only returns.  These are competitors and
proxies, not a claim that any individual one exactly represents Fibonacci
structure.  The symmetric layout-9 producer is not interchangeable with the
layout-8, UpAB-only TG4 pulse.

## 3. Frozen population and chronological partitions

The entire population is exactly these canonical symbols:

```
audcadrmp  audusdrmp  eurusdrmp
gbpusdrmp  usdcadrmp  usdjpyrmp
```

CADCHF is excluded.  Rows retain symbol identity at every stage.  There is no
all-row model, no cross-symbol training, and no global winner or score.

All timestamps are the canonical completed 15-minute-candle timestamp used by
the established market-data/Tensor path.  The frozen partitions are half-open:

| Name | Interval | Role |
| --- | --- | --- |
| `development` | [2010-01-01, 2019-01-01) | Fit all scalers and fixed diagnostic parameters. |
| `validation` | [2019-01-01, 2022-01-01) | First out-of-time report; never used to tune. |
| `pre2025_lock_test` | [2022-01-01, 2025-01-01) | Second out-of-time report; never used to tune. |
| `confirmation_2025` | [2025-01-01, 2026-01-01) | One final confirmation execution for these new metrics. |

The earlier Fibonacci work inspected 2025 **structural-transition** summaries.
Thus 2025 is not globally unseen data.  It remains untouched only for this
protocol's redundancy and incremental-information metrics.  Before its one
confirmation execution, no selection, ratio change, preprocessing change,
metric choice, regularization choice, or interpretation rule may be learned
from 2025.  The development, validation, and pre-2025 lock-test reports are
also descriptive reports, not tuning rounds.

An observation is assigned by its decision timestamp.  It is eligible for a
partition only when the complete forward horizon required by its target is
also before that partition's end.  This avoids targets crossing partitions.

## 4. Observation populations and causality

The primary unit is a **completed-bar representation observation**: one
symbol, one decision timestamp, its point-in-time baseline vector, its
point-in-time 23-column Fibonacci vector, and a causal forward target.

The primary population keeps `fib_recent_price_scale_valid == 0` as an
observed state.  Invalid scale is not silently discarded or replaced by a
future-valid structure.  Rows are excluded only when a required predictor is
not finite/available under the authoritative feature path, the target cannot
be constructed, the target crosses the partition boundary, or a canonical row
identity is duplicated.  Each reason and count is reported separately by
symbol and partition.

A secondary, explicitly labelled **event-state bar subset** is also reported:

```
fib_recent_price_scale_valid == 1
and (fib_up_recent_union_count_log > 0 or fib_down_recent_union_count_log > 0)
```

It is a descriptive stratification of the primary bar population, not a
separate set of independent trades or events.  Consecutive bars and overlapping
structures remain dependent.  This protocol does not retroactively construct
unique event identities from the aggregate 23-column representation; a future
event-level protocol would need a separately frozen identity and deduplication
rule.

Predictors must be materialized no later than the completed decision bar.  The
existing Fibonacci prefix-invariance and delayed-pivot tests remain the causal
guarantee.  Target values are never admitted as predictors.

## 5. Frozen outcomes

The two co-primary outcomes are the project's existing directional labels at
H4 and H6, each with the existing `0.0008` log-return threshold.  They are
implemented by `TargetLabel::BuildLookaheadClassInfo` with
`evalWindowSize = 1`, `evalPredictionHorizon = 4` or `6`, and that threshold.
For a decision row `t`, this is the same target construction used for the
final row of a training window: the current close is `close[t]`; future highs
and lows are scanned at the next H available canonical rows; the first
threshold hit is the label; a simultaneous first hit is Up because
`upOffset <= downOffset`; and no hit uses the H-step terminal close.

Classes are exactly Down `0`, Neutral `1`, and Up `2`.  The artifact records
the selected future timestamp, terminal timestamp, first hit offsets where
applicable, terminal log return, and label for audit.  Terminal log return is
descriptive/audit output only, not an additional fitted gate outcome.

The market-data path defines how canonical rows are available; this protocol
does not manufacture bars across weekends or gaps.  A row with fewer than H
subsequent available canonical rows is censored.  Profitability is not an
outcome in this screen.

## 6. Redundancy diagnostics

All redundancy results are per Fibonacci column, symbol, and partition.  They
answer reconstruction and association questions, not whether the feature is
useful to an LSTM.

### Unconditional forward-association ledger

Before any baseline conditioning, every symbol x partition x H4/H6 report
contains the count and Down/Neutral/Up composition for these three exhaustive
structural states:

1. `scale_invalid` (`fib_recent_price_scale_valid == 0`);
2. `scale_valid_no_recent_event` (scale valid and both directional union counts
   equal zero); and
3. `scale_valid_recent_event` (the secondary event-state-bar subset from
   section 4).

For each state it also records mean and median terminal log return as audit
descriptions.  These summaries do not pool symbols, imply independent trials,
or form a selection gate.

The supplemental per-column ledger reports the same class composition for all
23 columns, never just columns that look favorable.  Scale validity is grouped
as 0/1.  For every other numeric Fibonacci column, five equal-frequency bins
are formed from that symbol's eligible `development` predictor values only;
the same frozen edges are used in every later partition.  Tied edges are
collapsed, and an insufficient number of distinct development values makes the
column's binned ledger unavailable with its reason.  Bins are based on the
predictor distribution, not the outcome, and no association in this ledger is
promoted to incremental evidence by itself.

1. **Pairwise association matrix.** Report Spearman rank association for every
   numeric baseline/Fibonacci column pair (ties retained), with the nearest
   absolute-association baseline proxy per Fibonacci column.  For the binary
   scale-validity column, report its point-biserial association with each
   continuous baseline column and prevalence by partition.  Full matrices are
   artifacts; the report must not select only attractive pairs.
2. **Time-separated reconstruction.** Fit one fixed baseline-to-Fibonacci
   reconstruction model per symbol and Fibonacci column on `development` only,
   then report validation, pre-2025 lock-test, and 2025 results without refit.
   Scale validity uses L2-regularized logistic regression and reports log loss
   and Brier score.  The remaining count, age, and signed-distance columns use
   L2-regularized linear ridge regression and report out-of-time raw-scale RMSE
   and R-squared.  A zero-variance holdout target makes R-squared unavailable,
   not zero.  A one-class development target makes the relevant logistic result
   unavailable, not a perfect reconstruction.
3. **Coverage and degeneracy.** Report scale-validity prevalence, nonzero
   count prevalence, finite-value counts, target variance, and unavailable
   reconstruction reasons.

The reconstructor is deliberately low capacity: intercept plus the 80 baseline
columns, no interactions, no nonlinear expansion, L2 coefficient penalty
`lambda = 1.0`, and an unpenalized intercept.  On development data only,
continuous baseline columns are standardized using that symbol's development
mean and standard deviation; categorical/binary columns remain 0/1; a
zero-variance development predictor is omitted from fitting but remains in the
artifact/schema.  Continuous reconstruction targets are standardized using
their development statistics for fitting and converted back for raw-scale
reporting.  The binary validity target is not standardized.

There is intentionally no universal correlation or R-squared cutoff.  Redundancy
is a spectrum and must be read beside the incremental screen.

## 7. Conditional incremental-information diagnostic

For each symbol and each H4/H6 target, fit exactly two fixed, three-class,
multinomial L2-logistic diagnostics on `development`:

| Diagnostic | Inputs |
| --- | --- |
| `baseline` | exact logical `baseline[80]` family |
| `baseline_plus_fibonacci` | the same `baseline[80]` plus exact `fibonacci[23]` family |

Both use the identical eligible development observations, class encoding,
preprocessing convention in section 6, natural class prevalence (no class
reweighting), unpenalized class intercepts, L2 coefficient penalty
`lambda = 1.0`, canonical schema order, zero initialization, and a deterministic
full-batch L-BFGS solver (`max_iterations=250`, infinity-norm gradient tolerance
`1e-8`, relative-objective tolerance `1e-12`).  No interactions, trees,
neural network, ratio selection, or hyperparameter search are permitted.  If a
development symbol/target lacks any of the three classes, its diagnostic and
every dependent holdout result are unavailable with that exact reason.

The fitted diagnostics are evaluated unchanged on validation, pre-2025
lock-test, and 2025 confirmation.  For each identical row set, report:

- multiclass log loss;
- multiclass Brier score (mean squared probability-vector error divided by 3);
- top-1 accuracy as secondary context only;
- paired deltas `baseline - baseline_plus_fibonacci` for every metric, so a
  positive loss/Brier delta favors the Fibonacci-augmented diagnostic;
- class prevalence, eligible/censored counts, and event-state subset results.

The primary screen is the paired out-of-time log-loss and Brier deltas, not
accuracy and not a pooled score.  A favorable result is predictive association
conditional on this baseline, not a causal effect and not proof of an LSTM
increment.

## 8. Dependence, aggregation, and reporting

Fifteen-minute rows, horizons, and structural states overlap.  The protocol
uses no IID standard errors, p-values, bootstrap intervals, or claims that six
FX symbols are independent trials.

For dependence-aware descriptive context, each holdout report also splits the
paired loss deltas into calendar-month blocks (based on the decision timestamp)
and gives the number of rows, mean delta, median delta, and 10th/90th percentile
of row deltas within each block.  These blocks are not inferential samples.

The main table is symbol x partition x horizon.  Any cross-symbol summary is
limited to an equal-symbol display of the six per-symbol values (ordered list,
median, minimum, maximum, and favorable/unfavorable/unavailable count).  It
must name every symbol and never pool raw observations, form a market-wide
effect, rank symbols, or announce a global winner.  H4 and H6 stay separate.

## 9. Interpretation contract

After all frozen reports are available, classify the screen qualitatively as
one of the following.  This is a multi-dimensional scientific review, not a
single numeric acceptance threshold.

- **`REDUNDANT_OR_NO_INCREMENT`** — reconstruction indicates that the active
  representation is largely available from the baseline and/or incremental
  loss/Brier deltas are neutral or adverse; an apparent benefit is isolated to
  one symbol/partition/horizon; coverage is too sparse; or alignment,
  causality, or censoring invalidates interpretation.
- **`MIXED_INCREMENTAL_EVIDENCE`** — some out-of-time positive deltas exist,
  but they vary materially by symbol, partition, or horizon, are plausibly
  explained by a small subset or regime, or redundancy/coverage leaves the
  practical increment unclear.
- **`ROBUST_INCREMENTAL_SCREENING_SIGNAL`** — no provenance/causality failure;
  usable coverage; favorable log-loss and Brier results in both H4 and H6 that
  persist through the predeclared partitions for a clear majority of frozen
  symbols (at least four, with none supplying the whole finding); no dependence
  on one column or one regime; and redundancy evidence does not make the entire
  active family a trivial baseline reconstruction.

The strongest state authorizes only a proposal for a future, separately frozen,
controlled layout-9 LSTM Fibonacci control/complete-family-ablation experiment.
It neither changes a current model nor authorizes a live experiment.  Mixed
evidence is a valid outcome and may warrant a new protocol, not tuning this
one.  No state is inferred from structural-transition association alone.

## 10. Future read-only execution artifact

The future harness should make a single chronological, read-only extraction per
symbol and reuse the result for all diagnostics.  It must reuse the
authoritative Tensor/ModelInput materialization path and `TargetLabel` rather
than reimplement baseline features or labels.  Full pre-window warmup remains
available to the existing stateful feature producers; only emitted decision rows
are partitioned.

The generated, uncommitted artifact directory is:

```
causal-fibonacci-layout9-incremental-information-v1/
  manifest.json
  feature_schema.csv
  rows.csv
  exclusions.csv
  sha256sums.txt
```

`rows.csv` is UTF-8 RFC4180 CSV, sorted by canonical symbol then decision
timestamp, with a strict schema hash.  It contains a stable row identity
`canonical_symbol|decision_timestamp|source_row_ordinal`, the partition,
80 named baseline values, 23 named Fibonacci values, both H4/H6 target audit
fields, and explicit eligibility/exclusion fields.  Duplicate symbol/timestamp
identity is an extraction error, never silently deduplicated.  Large artifacts
are not committed.

`manifest.json` records this protocol identifier and SHA-256, code commit,
feature configuration name, layout/tensor/model widths, both ordered feature
schemas and hashes, target identities, symbols, extraction and warmup ranges,
partition boundaries, canonical source price-domain/adapter identity, source
query identity, an immutable economic-calendar snapshot identity/content hash,
and artifact file hashes.  If the required point-in-time economic-calendar
snapshot cannot be identified, extraction is unavailable; it must not silently
use a mutable later calendar corpus.  The harness validates all hashes before
diagnostics run and writes a second result manifest containing the fixed solver
and preprocessing parameters.

The current source uses canonical 15-minute ask-OHLC candlesticks through the
established candlestick/Tensor path.  The execution manifest must record the
concrete adapter/source identity rather than relying on this prose description.

## 11. Required tests before execution

The eventual isolated harness must add focused tests for:

1. deterministic extraction and artifact/content hashes on a small fixture;
2. retained Fibonacci prefix invariance, delayed pivot availability, immutable
   historical output, and degenerate-range behavior;
3. exact baseline/Tensor/return-suffix parity with the authoritative
   ModelInput materializer and exactly 80 + 23 logical columns;
4. exact decision-row H4/H6 target parity with `BuildLookaheadClassInfo`,
   including simultaneous-hit tie behavior and terminal fallback;
5. predictor timestamp no later than decision timestamp and target timestamp
   strictly later than it;
6. partition-edge censoring, missing-target reasons, and no cross-partition
   target leakage;
7. symbol separation, canonical sorting, and duplicate row rejection;
8. scale-invalid state retention and explicit event-state-bar subset selection;
9. development-only scaler/model fitting, including zero variance and absent
   class unavailable states;
10. deterministic fixed-solver output and paired baseline/augmented row
    alignment; and
11. a confirmation-period guard proving that no 2025 row is used in fitting,
    preprocessing, or any parameter choice.

Execution remains prohibited until these tests pass and a separate request
authorizes the bounded, read-only historical extraction.  No scheduler, worker,
experiment, database row, registry, semantic layout, or production binary is
part of this protocol.
