# 2025 H1/H2 causal Fibonacci confirmation interpretation

## Status and provenance

This document records the first interpretation of the untouched 2025 causal
Fibonacci H1/H2 confirmation. It answers the seven questions frozen before the
holdout was opened in
[`PRE_2025_H1_H2_EMPIRICAL_INTERPRETATION_FREEZE.md`](PRE_2025_H1_H2_EMPIRICAL_INTERPRETATION_FREEZE.md).
These are descriptive answers to predeclared questions, not a new numeric
pass/fail system or a revision of the frozen policy.

- baseline methodology commit:
  `0440faa9a054d7c7c02df75e9162d3d85b33533d`
- pre-2025 interpretation freeze:
  `8ba1b81101697321e27fe5afc2e4fc4927685ced`
- frozen confirmation-boundary implementation:
  `4b8577c80474c1e8886fe8c0a464676c5d7554f8`
- first untouched 2025 confirmation artifact commit:
  `69317d92bd0dd41880b5104ee90211c403a47a4a`
- policy: `causal-fibonacci-extension-h1-h2-policy-frozen-v2`
- study: `causal-fibonacci-extension-2025-confirmation-v1`
- scoring interval: `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`
- outcome boundary: before `2026-01-01T00:00:00Z`
- input: completed 15-minute bars, read-only database access
- H3 status: `missing_authoritative_d_extension_contract`

The six predefined symbols were unchanged:
`audcadrmp`, `audusdrmp`, `eurusdrmp`, `gbpusdrmp`, `usdcadrmp`, and
`usdjpyrmp`.

No ratio, A/B construction, causal TG1/TG3 meaning, one-canonical-pip
tolerance, eligibility/rejection definition, 20-completed-bar horizon,
no-invalidation policy, target hierarchy, censoring rule, denominator rule, or
H3 behavior was changed for this confirmation.

## Aggregate comparison

| Endpoint | Pre-2025 | Untouched 2025 |
|---|---:|---:|
| H1 1.618 | 90.49402143% | 90.57833641% |
| H2 .382 | 77.32269775% | 78.25107463% |
| H2 .500 | 71.92549874% | 72.97234307% |
| H2 .618 | 66.54548428% | 67.57302672% |

| 2025 endpoint | Success | Failure | Resolved | Censored | Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| H1 1.618 | 34,879 | 3,628 | 38,507 | 2 | [0.90282496068458085, 0.9086608136113774] |
| H2 .382 | 30,219 | 8,399 | 38,618 | 16 | [0.77836825366347417, 0.78659703995839725] |
| H2 .500 | 28,179 | 10,437 | 38,616 | 18 | [0.72527131232057385, 0.73412984858056096] |
| H2 .618 | 26,094 | 12,522 | 38,616 | 18 | [0.67104419708612462, 0.68038137814497668] |

The H2 .382 external expert prior remains metadata only: `0.80`. The 2025
measured-minus-prior value is `-0.017489253715883835`. It was not an acceptance
threshold and did not select or tune the policy.

## Predefined timing comparisons

### Cumulative success probability, frozen resolved denominator

| Checkpoint | Pre-2025 H1 | Pre-2025 H2 .382 | 2025 H1 | 2025 H2 .382 |
|---|---:|---:|---:|---:|
| B1 | 73.317% | 45.930% | 72.831% | 47.214% |
| B2 | 78.060% | 52.867% | 77.698% | 54.343% |
| B4 | 82.490% | 60.550% | 82.284% | 61.948% |
| B8 | 86.344% | 68.100% | 86.304% | 69.336% |
| B12 | 88.242% | 72.295% | 88.205% | 73.435% |
| B16 | 89.527% | 75.148% | 89.467% | 76.094% |
| B20 | 90.494% | 77.323% | 90.578% | 78.251% |

### Share of eventual successes completed by bar

| Checkpoint | Pre-2025 H1 | Pre-2025 H2 .382 | 2025 H1 | 2025 H2 .382 |
|---|---:|---:|---:|---:|
| B1 | 81.018% | 59.400% | 80.407% | 60.336% |
| B2 | 86.260% | 68.372% | 85.779% | 69.446% |
| B4 | 91.155% | 78.308% | 90.843% | 79.165% |
| B8 | 95.414% | 88.072% | 95.281% | 88.607% |
| B12 | 97.511% | 93.497% | 97.380% | 93.845% |
| B16 | 98.931% | 97.187% | 98.773% | 97.243% |
| B20 | 100.000% | 100.000% | 100.000% | 100.000% |

### Conditional first-hit hazards

The first four conditional first-hit hazards directly address the frozen
time-decay question.

| Checkpoint | Pre-2025 H1 | Pre-2025 H2 .382 | 2025 H1 | 2025 H2 .382 |
|---|---:|---:|---:|---:|
| B1 | 73.317% | 45.930% | 72.831% | 47.214% |
| B2 | 17.776% | 12.829% | 17.912% | 13.505% |
| B3 | 12.205% | 9.469% | 12.320% | 9.562% |
| B4 | 9.095% | 7.547% | 9.402% | 7.845% |

Later-bar hazards have small local fluctuations, but the general strongly
front-loaded and decaying shape persists. This does not claim strict
monotonicity.

## Predefined H2 ordering

The predefined ordering was retained.

- Pre-2025: `.382 77.323% > .500 71.926% > .618 66.546%`
- 2025: `.382 78.251% > .500 72.972% > .618 67.573%`

## Predefined symbol/direction consistency

The `confirmation_2025` cohort rows show that all 12 H1 symbol/direction
cohorts remain high, with an approximate observed range of 88.72% to 91.73%.

H2 .382 has broader heterogeneity, with an approximate observed range of
72.85% to 81.71%. Examples include AUDCAD down at 81.44%, AUDUSD down at
81.71%, USDCAD down at 81.42%, EURUSD up at 75.95%, GBPUSD up at 75.59%, and
USDJPY up at 72.85%.

Thus aggregate H2 reproduced, while symbol/direction heterogeneity is retained
as a finding rather than hidden or used to retrofit the frozen hypothesis.

## Answers to the seven frozen confirmation questions

1. **H1 continuation probability:** H1 continued to show a high
   1.272-to-1.618 continuation probability: 90.57833641% in 2025 versus
   90.49402143% pre-2025.
2. **H1 first-completed-bar concentration:** H1 retained strong concentration
   in the first completed bar after eligibility: 72.831% of resolved 2025 H1
   events at B1 and 80.407% of eventual H1 successes completed by B1.
3. **H1 conditional time-decay:** H1 retained the general conditional
   first-hit time-decay pattern. The first four 2025 hazards were 72.831%,
   17.912%, 12.320%, and 9.402%; later bars show small local fluctuations but
   preserve the general front-loaded/decaying shape.
4. **H2 .382 aggregate probability:** H2 .382 remained broadly consistent
   with the pre-2025 77.323% estimate, measuring 78.25107463% in 2025.
5. **H2 .382 timing and time-decay:** H2 .382 retained front-loaded timing
   and the general conditional first-hit time-decay pattern: 47.214% at B1,
   61.948% cumulative by B4, and 60.336% of eventual successes completed by
   B1.
6. **H2 endpoint hierarchy:** The predefined `.382 > .500 > .618` ordering
   was retained in 2025.
7. **Symbol/direction consistency:** The findings were broadly present across
   the predefined symbols and directions, with materially greater H2
   heterogeneity than H1.

These descriptive answers do not create a new numerical acceptance threshold,
redefine an endpoint, or change the frozen study interpretation.

## Reconciliation and artifact notes

`RECONCILIATION_PASS` was obtained from the post-run observation analysis.

The generated `cohorts.csv` contains a small number of warmup-origin,
pre-2025-labeled rows with zero eligible observations. They do not enter the
2025 confirmation denominator. This is retained as an artifact/reporting note;
the generated artifact is not modified here.

`aggregate.csv`, `cohorts.csv`, `data_quality.csv`, and `observations.csv` are
generated/ignored artifacts and were not included in commit `69317d92`.
`manifest.json` and `summary.md` were included. This document records the
essential confirmation counts and timing evidence while the local generated
files remain the detailed reproducibility output.

## Limitations

- Overlapping structures share market paths, so observation rows are not
  independent Bernoulli trials.
- Wilson intervals do not model that dependence.
- These measurements establish target-reaching frequencies under the frozen
  event definitions, not trading profitability.
- The study establishes no transaction costs, execution/slippage, position
  sizing, or trade-entry policy.
- H3 remains unevaluated because its D-extension contract is undefined:
  `missing_authoritative_d_extension_contract`.
