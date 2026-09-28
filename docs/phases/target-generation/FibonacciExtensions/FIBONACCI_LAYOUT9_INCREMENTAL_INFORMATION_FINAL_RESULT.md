# Causal Fibonacci layout-9 incremental-information V2 — final result

## Study identity and completed holdout

This document finalizes the frozen read-only screen:

- protocol: `causal-fibonacci-layout9-incremental-information-v2`
- frozen protocol SHA-256: `9f39886dc46ede11e2322fd8c0afcef6b09505132bfdc243c85ae8b790352c53`

The 2025 confirmation holdout was executed exactly once, after explicit
authorization, using the dedicated confirmation runner. Its completion marker
was `FIBONACCI_INCREMENTAL_CONFIRMATION_2025_COMPLETE` and its completion
message bound the execution to this protocol and explicit authorization.

The `confirmation_2025` holdout is now consumed. It must never again be
treated as unseen confirmation data for this protocol, and it must not be used
to retune, select, reclassify, or otherwise adapt this completed screen.

## Frozen input and execution provenance

The immutable extraction artifact is
`ResearchArtifacts/causal-fibonacci-layout9-incremental-information-v1`:

| Item | SHA-256 |
| --- | --- |
| `manifest.json` | `864879e0a8ce444da684af402fc70dcb7622e29c4f602bd0983e83458a4abc90` |
| `rows.csv` | `e3644fc742f1c970d993e6410fb11cfdd228c916f8c044891cf9321d1ce2e39d` |
| `feature_schema.csv` | `e408074a8fe046a783a9d2d5fd54e74bafec27f57958c5ff652032bf73b49f6e` |
| `exclusions.csv` | `1be82083cb7433af12987683abab0f05ca042b718ecc39da52b2dd40d237c1d5` |

The frozen pre-2025 V2 result is
`ResearchArtifacts/causal-fibonacci-layout9-incremental-information-v1-pre2025-results-v2`.
Its manifest SHA-256 is
`4e05ac7137784a3eb57c415f97fa86197266013c5b707172871a506bdd5d3ea0`;
its recorded checksum-file SHA-256 is
`84c229faa734605ba0da582d55143137c06d30f7143335542ce789b3df11f267`.

The confirmation runner was committed as
`ad9df942eaf6da8008ea29917c47136c3e3fdf39`
(`Implement frozen Fibonacci 2025 confirmation runner`). The qualified Release
binary was
`DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`, with SHA-256
`a73e383c211713e0b706b8168a2916fd9b445b20d7fba8e42ce8353d914b658e`.

The one completed confirmation artifact is
`ResearchArtifacts/causal-fibonacci-layout9-incremental-information-v1-confirmation-2025-v2`.
Its manifest records the frozen protocol, extraction manifest/row identities,
frozen pre-2025 manifest/checksum identities, runner commit, asserted explicit
authorization, enforced no-refit state, `confirmation_2025` partition, and
the frozen `lambda=1`, 4000-iteration L-BFGS, `1e-8` gradient tolerance, and
`1e-12` relative-objective tolerance.

## Confirmation artifact checksum record

| Artifact | SHA-256 |
| --- | --- |
| `associations.csv` | `424bb21cef7dd51a627bc954c0088303caa33b6cbc749ad723ec4359ea3ebb78` |
| `conditional_incremental.csv` | `b5dd54b4cb2c4bfb49f25807d046355e275b54ee2a1463982ed9609f426b0154` |
| `completion.txt` | `3764bd8218fb8857d12ccfb6948b3b124de06a874964d237937a05c9e1e8f831` |
| `coverage_degeneracy.csv` | `7a39b98e35dab03657a902952919f94e628e84b208e5696feb3b074db42455e9` |
| `cross_symbol_equal_summary.csv` | `75d6191ec524fe073cb7433af79341801dd5c1b6f9ff7d7741ed55c45d9ac73b` |
| `fibonacci_ledgers.csv` | `acf98e5dffa36ef82952a2b75672de8225d5dae7a4b20dc21cfcec4f056e2063` |
| `manifest.json` | `b039c931daea9902f661af191df231cd641efa6f351b3a5ea083560ebd2293c6` |
| `monthly_deltas.csv` | `6a4c2298c4188f2ca883b4360c4a6aa591c0552e61c1d8418ab80fcacea5cf66` |
| `reconstruction.csv` | `cb807b3f047ec6fd16f51241f463738cad007147426989b11986c93c0bda47c7` |
| `structural_ledgers.csv` | `2f7113ba74cd0518b7cc8cecf76f170d61b7f7eb5e4aac0cfb1f79e45ef97aed` |
| `sha256sums.txt` | `3d51cef8ee683187eeb50bae03433ca2195766e6a61458975df6a91dcdcfa667` |

The confirmation directory and its recorded checksum file were independently
verified during finalization.

## Frozen-partition evidence

Paired deltas are baseline minus baseline-plus-Fibonacci, so positive log-loss
and Brier deltas favor the frozen 103-input Fibonacci-augmented diagnostic.
The primary evidence is log loss and Brier, not accuracy.

Pre-2025 V2 results were:

| Partition | H4 log loss | H4 Brier | H6 log loss | H6 Brier |
| --- | ---: | ---: | ---: | ---: |
| Validation | 6/6 favorable | 6/6 favorable | 6/6 favorable | 6/6 favorable |
| Pre-2025 lock test | 5/6 favorable | 5/6 favorable | 5/6 favorable | 5/6 favorable |

EURUSD was the sole contrary symbol in the pre-2025 lock partition. It was
independently favorable in the completed confirmation for both primary scores:

| Symbol | Horizon | Log-loss delta | Brier delta |
| --- | --- | ---: | ---: |
| EURUSD | H4 | +0.0017470511322259519 | +0.0003521039903723322 |
| EURUSD | H6 | +0.0019460659265666536 | +0.00036186200209420427 |

The completed 2025 equal-symbol summary was:

| Horizon and primary score | Favorable | Median delta | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| H4 log loss | 6/6 | +0.00093299463699780816 | +0.00038952515214640826 | +0.0017470511322259519 |
| H4 Brier | 6/6 | +0.00018538658012903775 | +0.000074487148959384575 | +0.0003521039903723322 |
| H6 log loss | 6/6 | +0.00069252188483454047 | +0.00051916269596141706 | +0.0019460659265666536 |
| H6 Brier | 6/6 | +0.00013210208231104192 | +0.00010148182162866837 | +0.00036186200209420427 |

The event-state subset was also favorable on both primary proper scores for
all six symbols at both H4 and H6. Accuracy remains secondary context under
the frozen protocol and moves in both directions; it is not a basis for
redefining the interpretation.

The deltas are small. The evidentiary strength is their cross-symbol,
cross-horizon, proper-score, and out-of-time persistence, not large magnitude.

## Frozen §9 classification

Section 9 permits `ROBUST_INCREMENTAL_SCREENING_SIGNAL` only where there is no
provenance or causality failure, coverage is usable, favorable log-loss and
Brier behavior persists through the predeclared partitions for at least four
symbols at both H4 and H6, no single symbol supplies the finding, the evidence
does not depend on one column or regime, and redundancy does not establish that
the active family is trivially reconstructable from the baseline.

**Final classification: `ROBUST_INCREMENTAL_SCREENING_SIGNAL`.**

This is a frozen-protocol finding of persistent incremental predictive
association conditional on the exact layout-8-equivalent baseline diagnostic.
It is not a profitability result, a production-readiness result, an LSTM
efficacy result, a trading-strategy efficacy result, or a causal economic
mechanism claim.

The sole authorized next scientific stage is a proposal for a separately
frozen controlled layout-9 LSTM experiment comparing Fibonacci-complete versus
the exact 23-column family ablation. This completed screen does not itself
design, queue, authorize, or execute that experiment.

No retuning or reclassification of this completed screen from the consumed
2025 holdout is permitted.
