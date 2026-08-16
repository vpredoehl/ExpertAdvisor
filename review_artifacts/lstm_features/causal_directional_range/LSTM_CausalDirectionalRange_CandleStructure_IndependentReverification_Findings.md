# LSTM Causal Directional Range / Candle Structure — Independent Reverification Findings

## Verdict

**SOURCE IMPLEMENTATION: PASS**

**FINAL PROVENANCE / COMMIT QUALIFICATION: NOT YET CLOSED**

The packaged source implementation is correct by independent inspection:

- column 39 is the prior completed bar's directional body/range ratio;
- the current bar is retained only after the current row's feature value is fixed;
- first-row behavior is deterministic zero;
- invalid/non-finite OHLC and nonpositive range emit zero;
- no arbitrary epsilon or clipping was introduced;
- `feature_size = 40`;
- current model width is `n_in = 44`;
- historical `n_in = 43` remains a 39-column tensor prefix and does not consume column 39;
- v11 is the new default semantic configuration while v3-v10 remain explicitly reconstructable.

However, the reverification package exposes a provenance defect: the directional-range implementation was still **uncommitted** when the package metadata was captured.

`git-status.txt` shows the implementation files as modified/untracked, while `commit.txt` points to:

`dec62009ed4a704aa7c650212563560145802c7f`

whose commit metadata is:

`Add causal 8x32 RMS volatility regime feature`

The package's `implementation-commit.patch` therefore contains no directional-range/v11 changes.

This does not invalidate the source implementation or the successful Release binary evidence, but it means the binary cannot yet be tied to a committed directional-range source revision through the package's commit provenance.

## Package integrity

The package checksum manifest verifies successfully for every packaged file.

Unlike the previous package, the manifest does not hash itself and therefore has no self-reference defect.

## 1. Feature definition

`CausalDirectionalRangeFeatures::PriorDirectionalBodyRange()` computes:

`(priorClose - priorOpen) / (priorHigh - priorLow)`

from exactly one retained prior completed bar.

It returns zero when:

- no prior bar exists;
- any prior OHLC component is non-finite;
- prior high-low range is non-finite;
- prior range is <= 0;
- the computed ratio is non-finite.

No clipping is applied. A malformed but finite bar whose open/close lie outside `[low, high]` can therefore produce a finite value outside `[-1, +1]`, exactly as requested.

## 2. Strict one-bar causality

`Tensor::Add()` performs the relevant operations in this order:

1. `directionalRange = causalDirectionalRange.PriorDirectionalBodyRange();`
2. `causalDirectionalRange.RetainCompletedBar(f.open, f.high, f.low, f.close);`
3. the already-fixed `directionalRange` is written to the row.

Therefore bar `t` cannot influence its own directional-range feature. Bar `t` becomes the state used by row `t+1`.

The first row has no prior retained bar and emits zero.

This satisfies the requested one-bar causal contract.

## 3. State minimality

The feature stores only:

- prior open;
- prior high;
- prior low;
- prior close;
- a boolean indicating whether prior state exists.

There is no multi-bar history or hidden lookback.

## 4. Tensor layout

The final packaged `FeatureLayout.hpp` establishes:

- legacy: columns 0..31
- Donchian: 32..33
- session phase: 34..35
- relative tick volume: 36
- causal return surprise: 37
- causal 8x32 volatility regime: 38
- causal directional range: 39
- `feature_size = 40`

The new feature is append-only. No historical column was renumbered.

## 5. Historical model-input compatibility

`ModelInputContract.hpp` now resolves:

- `n_in=36` -> 32 tensor features + 4 returns
- `n_in=38` -> 34 tensor features + 4 returns
- `n_in=40` -> 36 tensor features + 4 returns
- `n_in=41` -> 37 tensor features + 4 returns
- `n_in=42` -> 38 tensor features + 4 returns
- `n_in=43` -> 39 tensor features + 4 returns
- `n_in=44` -> 40 tensor features + 4 returns

Because projection copies exactly `contract.tensorFeatureCount` values, an existing `n_in=43` model ends at tensor column 38 and cannot consume column 39.

Only current `n_in=44` models consume the new feature.

This is the required compatibility behavior.

## 6. Focused test review

The focused `LSTMCausalDirectionalRangeTests.cpp` covers:

- no-prior-bar zero;
- exact bullish `+0.5`;
- exact bearish `-0.5`;
- doji zero;
- zero range;
- negative range;
- non-finite open/high/low/close cases;
- finite malformed OHLC without clipping;
- current-bar causality;
- one-bar state advancement;
- overwrite-only state behavior;
- Tensor placement at column 39;
- `feature_size=40`;
- historical `n_in=43` exclusion;
- current `n_in=44` inclusion.

No defect was found in those assertions or in the feature arithmetic.

## 7. Recommendation semantic v11 transition

The final packaged source explicitly defines `v11` and makes it the default for:

- effective configuration canonical text;
- recommendation candidate hash/identity;
- experiment invocation canonical text/hash/identity.

The canonical parser recognizes both v10 and v11 separately.

The canonical version-number mapping explicitly preserves:

- v3 -> 3
- v4 -> 4
- v5 -> 5
- v6 -> 6
- v7 -> 7
- v8 -> 8
- v9 -> 9
- v10 -> 10
- v11 -> 11

The materialization reconstruction paths include v11 in the same later-version persisted dimensions as v10.

The outcome-assessment source-invocation parser maps v10 to 10 and v11 to 11 rather than collapsing them.

`ExperimentRecommendationTests.cpp` also preserves an explicit v10 identity while checking v11 as current.

No semantic-version collapse or silent v10-to-v11 reconstruction was found.

## 8. Release-build evidence

The package contains build evidence for:

`DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`

identified as:

`Mach-O 64-bit executable arm64`

with SHA-256:

`e731fbde6ec02413366e02194547450339d697f0b1db20ea0fcdbd2833afff14`

This is useful evidence that a Release binary was produced from the working tree.

However, because the captured working tree was dirty and the recorded Git commit predates the directional-range implementation, the package does not prove that this binary corresponds to a committed directional-range revision.

## 9. Provenance defect requiring closure

This is the only blocking finding.

The package records:

- branch: `next-feature-increment`
- HEAD: `dec62009ed4a704aa7c650212563560145802c7f`
- HEAD subject: `Add causal 8x32 RMS volatility regime feature`

while `git-status.txt` shows all directional-range/v11 work as modified or untracked.

Consequences:

1. `implementation-commit.patch` is not the directional-range implementation patch.
2. The Release binary hash cannot be linked to a directional-range commit identity.
3. The feature should not yet be treated as fully archived/commit-qualified even though the source implementation itself passes reverification.

## Recommended closure

Do not change the feature implementation.

Instead:

1. stage the directional-range source, tests, and archived implementation artifacts;
2. commit them on `next-feature-increment`;
3. confirm `git status --short` is empty;
4. perform the clean Release build from that committed clean tree;
5. regenerate the reverification package so:
   - `commit.txt` is the directional-range commit;
   - `git-status.txt` is empty;
   - `implementation-commit.patch` contains the directional-range/v11 changes;
   - the Release binary evidence is captured from that clean committed revision.

At that point, the implementation can be considered fully closed and ready for merge.

## Final assessment

**Feature mathematics:** PASS
**One-bar causality:** PASS
**Degenerate handling:** PASS
**Tensor append-only layout:** PASS
**Historical model width compatibility:** PASS
**v11 semantic reconstruction:** PASS
**Package integrity:** PASS
**Release binary existence:** PASS
**Committed clean-tree provenance:** FAIL / NOT YET ESTABLISHED

No source correction is recommended. The remaining work is commit/provenance cleanup only.
