# Frozen pre-2025 causal Fibonacci extension study

Policy: `causal-fibonacci-extension-h1-h2-policy-frozen-v2`

Boundary: `[2010-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`; 2025 bars were not loaded.

| Endpoint | Eligible | Success | Failure | Censored | Ineligible | Resolved | Probability | Wilson 95% | Bars to target (min/median/max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h1_1.618_after_1.272_beyond | 562429 | 508961 | 53464 | 4 | 5774 | 562425 | 0.90494021425078897 | [0.90417092493740525, 0.90570397198028774] | 1/1/20 |
| h2_0.382_after_rejection | 562523 | 434951 | 127563 | 9 | 5680 | 562514 | 0.77322697746189428 | [0.77213082829968382, 0.77431939486626444] | 1/1/20 |
| h2_descriptive_0.500_after_rejection | 562523 | 404591 | 157923 | 9 | 5680 | 562514 | 0.71925498743142391 | [0.71807919364930339, 0.72042778660984164] | 1/1/20 |
| h2_secondary_0.618_after_rejection | 562523 | 374327 | 188186 | 10 | 5680 | 562513 | 0.66545484282141032 | [0.66422070127867594, 0.66668672456348255] | 1/2/20 |

H2 primary expert prior: 0.80 (metadata only); measured minus prior: -0.026773022538105762.

Data quality: 2219451 usable completed bars, 9004 material timestamp gaps, 0 excluded symbols. Gaps are retained observed-market gaps; no synthetic bars were inserted.

H3 remains closed: `missing_authoritative_d_extension_contract`. Rows share market paths and overlapping structures; intervals do not model that dependence.
