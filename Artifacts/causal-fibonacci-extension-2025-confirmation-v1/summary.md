# 2025 causal Fibonacci extension confirmation study

Policy: `causal-fibonacci-extension-h1-h2-policy-frozen-v2`

Warmup: `[2010-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`; scoring: `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`; outcomes end before `2026-01-01T00:00:00Z`.

| Endpoint | Eligible | Success | Failure | Censored | Ineligible | Resolved | Probability | Wilson 95% | Bars to target (min/median/max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h1_1.618_after_1.272_beyond | 38509 | 34879 | 3628 | 2 | 860 | 38507 | 0.90578336406367677 | [0.90282496068458085, 0.9086608136113774] | 1/1/20 |
| h2_0.382_after_rejection | 38634 | 30219 | 8399 | 16 | 735 | 38618 | 0.78251074628411621 | [0.77836825366347417, 0.78659703995839725] | 1/1/20 |
| h2_descriptive_0.500_after_rejection | 38634 | 28179 | 10437 | 18 | 735 | 38616 | 0.72972343070229961 | [0.72527131232057385, 0.73412984858056096] | 1/1/20 |
| h2_secondary_0.618_after_rejection | 38634 | 26094 | 12522 | 18 | 735 | 38616 | 0.67573026724673713 | [0.67104419708612462, 0.68038137814497668] | 1/2/20 |

H2 primary expert prior: 0.80 (metadata only); measured minus prior: -0.017489253715883835.

Data quality: 2368834 usable completed bars, 9344 material timestamp gaps, 0 excluded symbols. Gaps are retained observed-market gaps; no synthetic bars were inserted.

H3 remains closed: `missing_authoritative_d_extension_contract`. Rows share market paths and overlapping structures; intervals do not model that dependence.
