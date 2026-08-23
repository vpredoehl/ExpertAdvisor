---
title: "LSTM Profitability Phase 4 Training Objective Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4_TrainingObjective_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4 Training Objective Inspection

# Phase 4 profitability-aware training review

## 1. Executive summary

The current authoritative training objective for `UpNeutralDownReturn` is a three-class, class-weighted cross-entropy loss over first-hit Up/Neutral/Down labels. Training uses a shared LSTM core and a three-output direction head. The implementation already persists a separate scalar return head, but that head is inactive—and not explicitly initialized—in classification training.

The primary findings are:

- The existing target is determined from future highs and lows over offsets `1…H`, using first threshold hit. The exact terminal close at `t+H` is also already available within that same causal label window.
- Therefore, terminal-horizon log return
  \[
  r_t=\log(C_{t+H}/C_t)
  \]
  is a causally valid auxiliary training label. It requires no future information beyond what current supervised-label construction already consumes.
- Future-derived values may be used only as labels or training weights. They must never be added to LSTM input features.
- Profitability’s inference-side directional convention—Up uses \(r_t\), Down uses \(-r_t\), Neutral is nonactionable—is appropriate for validation and model selection, but it is not the right regression target. A regression head should predict the raw signed terminal return \(r_t\).
- Signed-profitability weighting of cross-entropy is unsafe. Negative weights make the loss ill-posed, while clamping or offsetting the weights silently changes the supervised target distribution.
- Magnitude-weighted cross-entropy is mathematically valid but changes calibration and overemphasizes volatile/outlier regimes. It is less interpretable than retaining classification and adding a separate return task.
- An expected-return objective based on \(p_{\mathrm{up}}-p_{\mathrm{down}}\) conflicts with the existing first-hit label whenever the first threshold hit and terminal return sign disagree. It also encourages probability saturation.
- The safest future gradient-based design is multi-task classification plus a robust auxiliary regression head. The existing classifier remains primary and unchanged; the scalar head predicts threshold-normalized terminal log return using Huber/Smooth L1 loss.
- That design should remain experimental and opt-in. The current production default should remain pure classification until chronological validation, loss provenance, resume compatibility, and controlled ablations are established.
- An experiment must never resume under a different objective, coefficient, normalization, clipping rule, robust-loss definition, head learning rate, or effective-gradient definition.
- No input-width change is needed.
- The historical profitability backfill does not block read-only design or isolated source/test development. It should block schema/model-format rollout, scheduler deployment, and experiment launches that might contend with active workers.

No repository, database, scheduler, experiment, model, generated artifact, or build-product changes were made.

---

## 2. Current training-loss architecture

### Authoritative files and functions

The principal training-loss path is:

1. Batch partitioning: [`Tensor.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp:129>), `Tensor::ForEachBatchFrom`
2. Epoch and batch driver: [`main.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:8012>)
3. Batch construction, labels, forward pass, loss, backward pass, and updates: [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:3173>), `LSTM::CalculateBatch`
4. Label construction: [`TargetLabel.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/TargetLabel.hpp:28>), `BuildLookaheadClassInfo`
5. Target and head definitions: [`LSTM.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/LSTM.hpp:101>)
6. Runtime defaults and learning-rate multipliers: [`Params.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/Params.hpp:62>)
7. Model and training-configuration persistence: [`PgModelIO.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp:166>)

### Target types

`TargetType` currently defines:

- `LogReturn`
- `PercentReturn`
- `UpNeutralDownReturn`

For `UpNeutralDownReturn`, the direction head has three logits:

| Index | Class |
|---:|---|
| 0 | Down |
| 1 | Neutral |
| 2 | Up |

The default prediction horizon is 6 bars, the default threshold is `0.0008`, and the default direction class weights are all 1.0. See [`Params.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/Params.hpp:62>).

### Batch-to-update call flow

The training loop in `main.cpp` calls:

```text
Tensor::ForEachBatchFrom
    -> LSTM::CalculateBatch(batchBegin, batchEnd)
       -> precompute causal feature rows
       -> build overlapping input windows
       -> BuildLookaheadClassInfo for each training example
       -> LSTM forward pass
       -> direction-head affine transform
       -> softmax
       -> weighted cross-entropy and dlogits
       -> direction-head gradient accumulation
       -> shared-core gradient seed
       -> cached-time-step BPTT
       -> gradient normalization and clipping
       -> SGD parameter updates
```

The outer tensor batch is normally 256 rows. Within it, `CalculateBatch` creates overlapping windows satisfying:

```text
start + window_size + prediction_horizon - 1 < batchRows
```

Those windows do not cross the enclosing batch boundary. See [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:3576>).

### Classification loss

The active three-class branch computes softmax probabilities and then, for example \(i\),

\[
\ell_i=a_{y_i}\left[-\log p_{i,y_i}\right],
\]

where \(a_y\) is the configured weight of the true class.

The active logit derivative is:

\[
\frac{\partial \ell_i}{\partial z_{i,k}}
=0.1\,a_{y_i}(p_{i,k}-\mathbf 1[k=y_i]).
\]

This is implemented in [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:4041>).

There is also a helper named `predictAndLoss3Class`, but the authoritative batched path is the inline classification branch inside `CalculateBatch`. The helper applies weights differently when class weights are nonunit, so it should not be treated as defining current batch semantics.

### Class weights and per-sample weights

Class weights enter as the weight of the true class. The current defaults are:

```text
Down    1.0
Neutral 1.0
Up      1.0
```

There are no independent, data-dependent per-sample weights in the active training path.

After accumulation, classification gradients are divided by the sum of true-class weights. With all weights equal to 1, this equals ordinary batch-mean cross-entropy.

One implementation detail should be resolved before introducing new weighting: the internally calculated classification loss is normalized by class-weight sum, but the final return from `CalculateBatch` is based on `sse / mseCount`. These coincide with current unit weights but may diverge under nonunit weighting. See [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:5471>) and [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:7233>).

### Gradient production and update

Direction-head gradients are accumulated and projected back into the hidden representation. The classification core seed is multiplied by `LSTM_CORE_GRAD_SCALE`, currently 4.0. The classifier’s logit gradient separately contains the 0.1 factor described above.

BPTT then propagates through the cached LSTM time steps. All accumulated gradients are mean-normalized and componentwise clipped to `[-10,10]`.

Despite an optimizer-related macro referring to Adam, the authoritative update path is plain SGD:

\[
\theta \leftarrow \theta-\eta g.
\]

Relevant code begins at [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:5553>) and performs updates around [`LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:6057>).

### Core and head learning rates

With the current base learning rate of approximately `0.000333333`:

- LSTM core: base × 120 ≈ `0.04`
- Direction-head weights: base × 25 ≈ `0.008333`
- Direction-head bias: base × 2.5 ≈ `0.000833`

These multipliers are applied during the final parameter update, not while forming the raw loss.

### Auxiliary losses

No simultaneous auxiliary loss currently exists.

The scalar return head supports alternative scalar target modes, but those modes replace classification; they do not run jointly with the three-class loss. For classification models, the scalar head is allocated and persisted, but it is not part of forward loss or backward propagation.

### Training versus validation semantics

There is no authoritative held-out validation-loss path that computes exactly the same cross-entropy as training.

`RunInferenceEvaluation` uses the same target-label construction and produces classifications, confusion statistics, accuracy, and profitability, but it does not compute the training cross-entropy. See [`main.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6520>).

The logged `EPOCH_3CLASS_ACCURACY` is derived from predictions made during the training batch stream. It is not a clean held-out validation measure.

Therefore:

- Training and inference share label semantics.
- Training and inference do not currently share a common reported loss.
- “Validation accuracy” must not be assumed to mean a chronologically isolated, held-out LSTM validation loss without tracing its exact caller.

This validation gap is an important blocker for safely selecting an auxiliary coefficient.

---

## 3. Causal target analysis

### Current first-hit label

`BuildLookaheadClassInfo` defines:

- Last input bar:
  \[
  t=\text{start}+\text{window size}-1
  \]
- Terminal target bar:
  \[
  t+H
  \]
- Decision price: `close_t`
- Terminal price: `targetClose = close[t+H]`
- Terminal log return:
  \[
  r_t=\log(C_{t+H}/C_t)
  \]

For each future offset \(k=1,\ldots,H\), it computes:

\[
u_k=\log(H_{t+k}/C_t),
\qquad
d_k=\log(L_{t+k}/C_t).
\]

Up is hit when \(u_k>\tau\). Down is hit when \(d_k<-\tau\). Comparisons are strict; equality with a threshold is not a hit.

The class is:

- Up if Up is first hit;
- Down if Down is first hit;
- Neutral if neither is hit by \(H\).

When Up and Down first hit on the same bar, Up wins because the implementation uses `upOffset <= downOffset`.

See [`TargetLabel.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/TargetLabel.hpp:28>).

### Terminal close versus first-hit target

The first-hit label uses future highs and lows across the entire horizon. It is not simply the sign of the terminal close return.

Consequently, these are valid and expected cases:

- first-hit Up but \(r_t<0\);
- first-hit Down but \(r_t>0\);
- Neutral with a nonzero terminal return that did not cross either intrahorizon threshold.

That distinction is central to the objective-design analysis. Any objective that assumes the terminal-return sign is identical to the classification label changes the meaning of the problem.

### Prediction horizon

The default horizon is 6 bars. With the repository’s 15-minute use cases, this corresponds to 90 minutes, although horizon is runtime configuration and historical experiments include other values.

### Is terminal-horizon return causally available?

Yes.

The terminal close is at exactly `t+H`, which lies inside the same future label window already required for current first-hit classification. An auxiliary terminal-return label would use no information beyond the existing supervised-label horizon.

This does not mean it may enter input features. It is valid only as a training or evaluation target.

### Information boundary

Allowed as training labels:

- Future highs and lows at offsets `1…H`
- Terminal close at `t+H`
- First-hit class and first-hit offset
- Terminal log return
- A deterministic, versioned transformation of terminal return used for a training-only target or weight

Available to the model at inference:

- Features computed from bars at or before the decision time \(t\)
- The same feature definitions, widths, normalization, ablations, and warmup rules used during training
- Model parameters and persisted inference configuration

Must never enter input features:

- `targetClose`
- Any future high or low
- Terminal return or its magnitude
- First-hit offsets
- Realized directional profitability
- Future-window volatility or normalization fitted using future bars
- A profitability-derived sample weight
- Any aggregate observation derived from later inference outcomes

Chronological split construction also needs an embargo so that training labels do not consume bars belonging to a validation or test period. At minimum, the split must purge the horizon overlap. A stricter experimental design should also document whether overlapping input windows across the boundary are allowed.

---

## 4. Existing profitability semantics

The canonical definition is documented in [`InferenceProfitability.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitability.hpp:11>) and implemented in [`InferenceProfitability.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitability.cpp:107>).

### Actionable prediction

A prediction is actionable only when:

- predicted class is Up or Down; and
- decision and terminal prices are finite and positive.

Neutral predictions are nonactionable. Invalid directional observations are also nonactionable.

Every evaluable inference window still contributes to the source-content hash, including Neutral or invalid-price rows.

### Directional treatment

Let:

\[
r=\log(C_{t+H}/C_t).
\]

Then:

\[
\pi =
\begin{cases}
r,&\text{predicted Up}\\
-r,&\text{predicted Down}\\
\text{not actionable},&\text{predicted Neutral}.
\end{cases}
\]

Thus, a correct Down prediction followed by a negative terminal return produces positive profitability.

A positive directional return is a win, a negative value is a loss, and zero is neither.

### Terminal-horizon semantics

Profitability uses the exact configured terminal horizon—not the first-hit price or first-hit offset. It deliberately has no transaction costs, sizing, leverage, overlap adjustment, or clipping.

It is therefore an inference utility/evidence measure, not complete trade P&L.

### Aggregate and average

For actionable predictions:

\[
\text{aggregate terminal log return}
=\sum_i \pi_i
\]

and

\[
\text{average per actionable prediction}
=\frac{\sum_i\pi_i}{N_{\mathrm{actionable}}}.
\]

The average is absent/null when there are no actionable predictions.

### Source-content hash

The source hash is an ordered FNV-1a hash over a version prefix and every inference record’s:

- ordinal position;
- predicted class;
- exact floating-point bits of decision close;
- exact floating-point bits of terminal close.

This proves which ordered source results produced the observation. See [`InferenceProfitability.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitability.cpp:78>).

### Metric-definition identity

The canonical metric definition is versioned and hashed independently from the source content. It covers actionability, sign, horizon, zero, cost, clipping, and related semantics.

Persistence is immutable and idempotent on exact provenance and canonical identity. Relevant repository code is in [`InferenceProfitabilityRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:154>) and the schema is in [`073_inference_profitability_observation.sql`](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:1>).

### Profitability distribution

Phase 3C’s primary value is average directional terminal return per actionable prediction. Cohorts are strictly defined by metric, scope, inference window, symbol, horizon, input width, and other semantic identities.

The empirical transform is sign-preserving and support-adjusted. It currently has zero direct Campaign Manager score weight and zero direct score contribution. See [`ProfitabilityDistribution.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityDistribution.hpp:12>) and [`ProfitabilityDistribution.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityDistribution.cpp:641>).

### Concepts reusable in training

Directly reusable:

- Raw terminal log return \(r=\log(C_{t+H}/C_t)\)
- Horizon identity
- Exact decision/terminal price validity requirements
- Metric-definition versioning discipline
- Immutable semantic provenance

Useful for validation but not as a regression target:

- Directional sign convention
- Aggregate and average profitability
- Actionable support
- Profitability-distribution cohorting and ranking

Not meaningful as per-example supervision:

- Source-content hash
- Aggregate observation identity
- Population percentile
- Support shrinkage
- Repository observation identity

A regression target should predict raw signed \(r\), not predicted-class-aligned profitability \(\pi\). The latter depends on the model’s decision and is therefore not a fixed ground-truth target.

---

## 5. Candidate objective designs

Let:

- \(y\in\{D,N,U\}\) be the first-hit class;
- \(p_k\) be the classifier probability for class \(k\);
- \(a_y\) be the existing class weight;
- \(r=\log(C_{t+H}/C_t)\);
- \(d_D=-1,d_N=0,d_U=1\);
- \(\tau\) be the configured threshold.

### A. Pure classification; profitability only for model selection

\[
L=L_{\mathrm{CE}}
=\frac{\sum_i a_{y_i}[-\log p_{i,y_i}]}
       {\sum_i a_{y_i}}.
\]

Required data: current class labels only for training; existing profitability observations for selection.

Causal validity: fully valid.

Optimization and calibration: preserves current behavior and probability interpretation.

Code impact: no model or loss change. A rigorous validation/model-selection workflow and provenance changes may still be needed.

Overfitting risk: profitability-based model selection can overfit the validation window, especially when actionable support is low.

Historical comparison: strongest compatibility with existing models.

Inference requirements: no change.

Provenance: selection policy, profitability metric, inference scope/window, support rule, and tie-breaking must be versioned.

Safeguards:

- Keep an untouched final test window.
- Require minimum actionable support.
- Compare profitability together with NLL/Brier/calibration and existing classification measures.
- Never tune and report on the same inference period.

This should remain the production baseline.

### B. Classification weighted by realized movement magnitude

Example:

\[
w_i=f(|r_i|/\tau),
\]

with bounded \(f\), and:

\[
L=
\frac{\sum_i a_{y_i}w_i[-\log p_{i,y_i}]}
     {\sum_i a_{y_i}w_i}.
\]

Causal validity: valid as a training-only weighting because \(r_i\) is inside the label horizon.

Optimization: gives larger gradient influence to large terminal moves.

Calibration: the classifier learns a movement-weighted posterior rather than the natural class posterior. Raw softmax probabilities cease to be naturally calibrated for the unweighted population.

Semantic risk: first-hit labels and terminal-magnitude information are different constructs. A first-hit Up example with a large negative terminal return would receive a large weight in favor of Up.

Overfitting risk: high, particularly around volatility spikes and outliers.

Code impact: label batches need terminal return and sample weights; denominators, reported loss, tests, and provenance all change.

Inference: no additional runtime value required.

Compatibility: new objective and new experiment identity; no resume from legacy classification.

Safeguards: bounded weights, no zero-weight class disappearance, train-only transform fitting, loss/gradient diagnostics, calibration evaluation, and strict objective versioning.

Assessment: valid but not preferred.

### C. Classification weighted by directional profitability

A tempting formulation is:

\[
w_i=d_{y_i}r_i.
\]

Using this directly is invalid because weights can be negative. Negative cross-entropy weights make confident error behavior ill-posed and can reward the model for increasing loss.

Replacing it with:

\[
w_i=\max(0,d_{y_i}r_i)
\]

is nonnegative, but it suppresses examples where first-hit class and terminal direction disagree. It thereby changes the effective target definition from first-hit classification toward terminal-direction classification.

Adding an arbitrary positive offset avoids zero weights but does not resolve the semantic ambiguity.

Calibration: severely distorted.

Optimization: asymmetric and potentially dominated by whichever class aligns more often with terminal sign.

Historical comparison: poor.

Inference: no new inputs, but the learned classifier’s meaning changes.

Assessment: reject.

### D. Auxiliary regression head for terminal return

Use the existing direction head for \(p\) and a separate scalar head for \(\hat u\), where:

\[
u=r/\tau.
\]

Then:

\[
L_{\mathrm{reg}}
=\frac1N\sum_i \operatorname{Huber}_{\delta=1}(\hat u_i-u_i)
\]

and:

\[
L=L_{\mathrm{CE}}+\lambda L_{\mathrm{reg}}.
\]

Causal validity: valid. It uses the same terminal bar already within the current first-hit horizon.

Architecture: separate scalar affine head from the shared final hidden state. The repository already stores a scalar head, so matrix dimensions need not expand, but its active semantics and initialization must be versioned.

Optimization: regression gives the shared core information about terminal return magnitude and sign without redefining the classifier label. Shared-core gradient conflict is possible and must be measured.

Calibration: classification remains directly supervised by cross-entropy, although shared-core changes can still affect calibration.

Overfitting: lower semantic risk than weighting, but the auxiliary task can overfit return noise or dominate the core if \(\lambda\) is too large.

Inference: classification inference need not consume or expose the scalar output. If the scalar prediction is exposed, it needs its own inference-output contract.

Compatibility: legacy checkpoints cannot resume as auxiliary models simply because a scalar matrix exists. The old head was inactive and may not be initialized with valid learned state.

Assessment: strongest auxiliary mechanism.

### E. Expected signed-return objective from class probabilities

Define:

\[
q=p_U-p_D.
\]

A direct utility term would be:

\[
L_{\mathrm{ER}}=-r q,
\qquad
L=L_{\mathrm{CE}}+\lambda L_{\mathrm{ER}}.
\]

Causal validity: the target return is valid.

Optimization: for \(r>0\), the utility term pushes toward \(p_U=1\); for \(r<0\), toward \(p_D=1\). It does not represent Neutral naturally and encourages saturated probabilities.

Semantic conflict: when first-hit Up ends with negative terminal return, cross-entropy pushes Up while the utility term pushes Down.

A squared form such as:

\[
(q-r/s)^2
\]

requires an arbitrary scaling \(s\), constrains a continuous return to `[-1,1]`, and still does not resolve Neutral semantics.

Calibration: poor unless heavily constrained and recalibrated.

Assessment: mathematically differentiable, but not coherent with the repository’s first-hit classification contract. Reject for the initial implementation.

### F. Direct profitability-aware classification loss

One example is:

\[
L=L_{\mathrm{CE}}(y,p)
+\beta m(r)L_{\mathrm{CE}}(y_{\mathrm{terminal}},p),
\]

where `y_terminal` is derived from terminal-return sign. Another is an expected cost:

\[
L=\sum_k p_k C(k,y,r).
\]

Unlike ordinary sample-weighted cross-entropy, which scales the same gradient toward \(y\), these losses change the direction or relative cost of errors depending on the predicted class and realized return.

That flexibility is also the danger: it creates a new cost-sensitive target definition, especially where first-hit and terminal-sign labels disagree.

Required provenance would include the complete cost matrix or target-construction function, magnitude transform, caps, thresholds, and loss version.

Calibration and historical comparison are weak. Tail-event overfitting is likely.

Assessment: defer unless a separately approved economic decision objective replaces first-hit classification.

### G. Multi-task classification and regression

This is the complete architecture built from D:

\[
L=
L_{\mathrm{first-hit\ classification}}
+\lambda L_{\mathrm{terminal-return\ regression}}.
\]

It is preferable to replacing or weighting classification because:

- the accepted first-hit class meaning remains explicit;
- raw terminal-return information has its own inspectable head;
- task conflict can be measured;
- \(\lambda=0\) supplies a direct implementation control;
- classifier output shape remains unchanged;
- each output can have separate metrics and calibration analysis.

Assessment: recommended future experimental design, with A retained as the production default and mandatory baseline.

---

## 6. Mathematical comparison

| Design | Gradient meaning | Preserves first-hit target? | Preserves natural calibration? | Main failure mode |
|---|---|---:|---:|---|
| A. Classification only | Estimate first-hit class posterior | Yes | Best | Validation-selection overfit |
| B. Magnitude weighting | Estimate magnitude-weighted posterior | Label IDs only | No | Volatility/outlier dominance |
| C. Directional weighting | Reward label/terminal-sign agreement | No | No | Negative or suppressed examples |
| D/G. CE + return regression | Learn class and continuous return jointly | Yes | Mostly; shared core can shift it | Gradient conflict or auxiliary dominance |
| E. Expected-return term | Push probabilities toward terminal sign | No | No | Saturation and Neutral ambiguity |
| F. Cost-sensitive classification | Learn a new economic cost target | Usually no | Generally no | Objective ambiguity |

The critical mathematical distinction is between scaling a fixed class gradient and changing its direction:

- Weighted CE still points toward the original one-hot target, although it changes the sampled population.
- Expected-return and direct profitability-aware losses can point toward a different class from the first-hit label.
- Multi-task regression keeps both facts—the first-hit event and the terminal return—rather than collapsing them into one ambiguous target.

---

## 7. Loss-scale and gradient analysis

### Classification scale

For three uniformly likely classes, initial cross-entropy is:

\[
-\log(1/3)\approx1.098612.
\]

Correct confident predictions approach zero; confident incorrect predictions can be much larger.

The current effective gradient scale is not simply the derivative of the logged cross-entropy because:

- classifier logit gradients are multiplied by 0.1;
- the shared-core seed is multiplied by 4.0;
- core and head use different update multipliers;
- gradients are clipped componentwise.

These effective-gradient conventions must be included in the loss-definition version before adding another branch.

### Terminal-return scale

At the default threshold:

\[
\tau=0.0008.
\]

That is an 8-basis-point log-return threshold. Raw terminal returns are plausibly around \(10^{-4}\) to \(10^{-3}\) for ordinary examples, with larger tails, but the repository inspection did not provide an authoritative target-return quantile distribution.

Existing persisted profitability summaries show average model-conditioned directional returns on the order of \(10^{-5}\) to \(10^{-4}\), but those averages are not substitutes for the per-example \(|r|\) distribution.

### Why raw-return MSE is poorly scaled

If a typical raw-return residual is \(10^{-3}\), then:

\[
\text{MSE}\sim10^{-6}.
\]

Matching its numerical loss scale to cross-entropy would suggest coefficients near \(10^6\), which is difficult to interpret and still does not account for gradient norms or learning-rate multipliers.

### Recommended normalization

Retain raw log return as the semantic target, but optimize:

\[
u=r/\tau.
\]

Benefits:

- reuses the repository’s threshold convention;
- produces values near order one around economically meaningful threshold moves;
- makes coefficients more comparable across experiments;
- avoids changing input features.

The normalization identity and exact threshold must be persisted.

### Robust loss choice

Recommended:

\[
\operatorname{Huber}_{\delta=1}(e)=
\begin{cases}
\frac12e^2,&|e|\le1\\
|e|-\frac12,&|e|>1.
\end{cases}
\]

Huber/Smooth L1 is preferable because it:

- is quadratic near zero;
- has bounded tail derivative;
- avoids MSE’s outlier-dominated gradients;
- is already a conventional, testable definition.

MSE is too sensitive to return tails. MAE has a nondifferentiable point and discards small-error curvature. Log-cosh would also bound tail gradients but adds a less familiar implementation without a clear repository-specific benefit.

### Clipping and winsorization

Initial recommendation: do not clip the target; use Huber to control gradient tails.

Persist:

```text
target_clipping_mode = none
```

If later experiments introduce clipping or winsorization, they require a new objective identity. Winsorization cutoffs must be fitted on training data only and persisted as exact values and methodology.

### Choosing the auxiliary coefficient

Do not choose \(\lambda\) by comparing raw loss values alone.

Recommended procedure:

1. Establish a chronological training/validation/test split with horizon purging.
2. On frozen representative batches, measure unscaled shared-core gradient norms for CE and regression separately.
3. Measure gradient cosine similarity and clipping frequency.
4. Select a log-spaced range of \(\lambda\) values that keeps the initial auxiliary shared-core contribution a predeclared minority of total gradient magnitude.
5. Include \(\lambda=0\) as a bit-for-bit control.
6. Use matched seeds and identical training data.
7. Evaluate:
   - classification NLL;
   - Brier score;
   - calibration error;
   - class recall/F1/confusion;
   - auxiliary Huber and MAE;
   - inference profitability using the exact canonical metric;
   - actionable support;
   - gradient norms and clip rates.
8. Choose using validation data only.
9. Report once on an untouched test period.

A 5–20% initial auxiliary gradient contribution might be a reasonable search guardrail, but it is not a recommended coefficient. Repository data must determine the actual value.

---

## 8. Architecture and persistence impact

### Output architecture

Recommended output structure:

```text
shared LSTM core
    ├── direction head: hidden -> 3 logits
    └── auxiliary return head: hidden -> 1 normalized terminal return
```

This is not a four-class output and must not be implemented as a single four-column softmax.

### Existing scalar head

`LSTM` already declares and persists:

- `returnHeadWeight`: hidden × 1
- `returnHeadBias`: 1 × 1
- direction head: hidden × 3 plus 1 × 3

See [`LSTM.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/LSTM.hpp:132>) and [`PgModelIO.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp:166>).

This means:

- No new matrix dimensions are strictly required.
- The scalar head must be explicitly initialized and activated.
- Its semantic role changes from inactive/alternative-target storage to a simultaneous auxiliary output.
- Model-format semantics still change even if physical shapes do not.

### Input width

No input-width change is needed or recommended.

The terminal-return value, its normalization, and regression-validity mask belong to training-label batches only.

### Serialization and identity

Current training metadata schema v1 contains 14 fields, including horizon, threshold, window, label rule, class weights, layers, normalization, epochs, and learning-rate multipliers. It does not describe auxiliary objectives. See [`PgModelIO.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp:445>).

A future implementation should persist at least:

- training-objective identifier;
- objective-definition version;
- classification-loss identifier;
- classification gradient-scale version;
- auxiliary mode;
- auxiliary coefficient;
- regression-target definition;
- target normalization identity;
- threshold used for normalization;
- robust-loss type and delta;
- clipping/winsorization mode and parameters;
- invalid-target policy;
- auxiliary-head weight/bias learning-rate multipliers;
- shared-core loss-combination rule;
- model/head semantic format version.

The training-config schema should be bumped so an older binary cannot silently ignore a separate auxiliary metadata record and resume the model as legacy classification.

### Checkpoint compatibility

Physical scalar-head presence does not make legacy checkpoints compatible with multi-task training. For classification checkpoints, the scalar head was not trained and may not be explicitly initialized under classification construction.

Compatibility rules should be:

- Legacy classification → legacy classification: allowed under current exact checks.
- Legacy classification → auxiliary objective: not resume-compatible.
- Auxiliary configuration X → exact same X: allowed.
- Auxiliary X → different coefficient, normalization, loss delta, clipping, LR, or gradient-combination rule: prohibited.
- Removing the auxiliary objective during resume: prohibited.
- Changing threshold when it is also the normalization denominator: prohibited.
- Warm-starting from a legacy core may be offered only as a distinct new experiment, with optimizer state reset and auxiliary head freshly initialized. It must not be called resume.

---

## 9. Historical-comparability requirements

The existing experiment uniqueness and effective-configuration canonical text do not include a profitability-aware training objective. The current experiment schema’s identity fields are insufficient for such models.

Relevant experiment configuration code is in [`ExperimentRecommendation.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp:869>).

Experiment and model identity must include:

- `training_objective_id`
- `loss_definition_version`
- `classification_loss_mode`
- `class_weights`
- `classification_gradient_scale`
- `auxiliary_loss_mode`
- `auxiliary_loss_coefficient`
- `regression_target_definition`
- `regression_normalization_definition`
- normalization parameters
- clipping/winsorization definition and parameters
- robust-loss type and delta
- invalid-target policy
- head and core learning-rate multipliers
- shared-gradient combination/normalization rule
- model semantic-format version

These fields should participate in:

- experiment uniqueness;
- effective experiment configuration hash/canonical text;
- model metadata;
- checkpoint metadata;
- scheduler retry identity;
- recommendation provenance;
- profitability-distribution cohort identity, either explicitly or through a guaranteed model-semantic identity.

Historical models remain `legacy_first_hit_weighted_ce_v1` or an equivalent explicit legacy identity. They must not be retroactively treated as having an auxiliary mode merely because their persisted matrices include a scalar head.

Comparisons should be labeled:

- same data, same seed, same nonobjective settings, different objective; or
- observational comparison only.

A profitability-aware model should not silently enter the same recommendation or distribution population as legacy models unless the policy explicitly permits cross-objective comparison.

---

## 10. Checkpoint and continuation implications

### Checkpoint inference

The existing three-class inference interface can remain unchanged. Checkpoint profitability can continue to use predicted direction and the canonical terminal-horizon metric.

If scalar-return predictions are recorded, they require separate output and persistence semantics. They should not modify the existing profitability metric.

### Checkpoint policy

Training-objective changes will alter model behavior and checkpoint metric distributions. Existing checkpoint policy thresholds should not automatically be retuned or reinterpreted.

Any profitability-based checkpoint-selection policy requires a new policy identity.

### Continuation policy

Continuation logic can continue consuming authoritative profitability observations. However, an experiment’s objective identity must be available when determining comparison or continuation compatibility.

A continuation that resumes training must require exact objective equality. A policy decision to start a new experiment from another model is a separate warm-start workflow.

### Campaign Manager and profitability distribution

Phase 3C profitability currently has zero direct score contribution. Profitability-aware training should not implicitly activate that score.

Distribution ranking will observe different model behavior. To avoid invalid cohort mixing, training-objective identity should be part of the population identity or an authoritative model-semantic identity already included in it.

### Resume rule

An experiment trained with one profitability-aware configuration may not safely resume under another.

That includes changing only \(\lambda\). The change modifies shared-core gradients and the meaning of optimizer progress, so it is a new experiment.

---

## 11. Train/infer parity requirements

The following invariants must remain unchanged:

1. Same causal feature definitions in training and inference.
2. Same input width and feature order.
3. Same normalization and clamping rules.
4. Same ablation and warmup behavior.
5. Same window endpoint: inputs end at decision time \(t\).
6. Same direction-head affine and softmax semantics.
7. Same direction index mapping: Down 0, Neutral 1, Up 2.
8. Same first-hit label rule for classification evaluation.
9. Same horizon and threshold identity.
10. Same deterministic threshold comparisons and same-bar tie behavior.
11. No terminal return, future close, future high/low, first-hit offset, or derived profitability value in inference features.
12. Same serialization interpretation between training, checkpoint loading, final inference, and resume.
13. Auxiliary training must be explicitly disabled during inference backward/update paths.
14. If the auxiliary scalar output is ignored at inference, ignoring it must not alter classifier probabilities.
15. A \(\lambda=0\) multi-task implementation must reproduce legacy classifier behavior within a predeclared numerical tolerance—preferably bit-for-bit before new head initialization consumes random-number state.

The last point requires care: initializing a new active scalar head using the same shared PRNG could change the subsequent direction-head initialization. Initialization order or independent deterministic seeds must preserve the legacy \(\lambda=0\) control.

---

## 12. Recommended Phase 4 design

The recommended future experimental architecture is multi-task classification plus auxiliary regression, while pure classification remains the production default and mandatory control.

### Exact target

Semantic target:

\[
r_t=\log(C_{t+H}/C_t).
\]

Optimization target:

\[
u_t=r_t/\tau.
\]

Use the terminal close at exactly the existing configured horizon.

Do not align the sign with the class. Do not use first-hit price as the regression target.

### Exact loss

Classification:

\[
L_{\mathrm{cls}}
=\frac{\sum_i a_{y_i}[-\log p_{i,y_i}]}
       {\sum_i a_{y_i}}.
\]

Regression:

\[
L_{\mathrm{reg}}
=\frac1{N_{\mathrm{valid}}}
 \sum_{i\in\mathrm{valid}}
 \operatorname{Huber}_{\delta=1}(\hat u_i-u_i).
\]

Combined:

\[
L=L_{\mathrm{cls}}+\lambda L_{\mathrm{reg}}.
\]

The implementation definition must additionally specify the existing 0.1 classifier-gradient scale, 4.0 core scale, independent denominators, clipping, and head/core update multipliers. Those are part of effective objective semantics.

### Head and output shape

- Direction head remains three logits.
- Scalar auxiliary head produces one value.
- No input-width change.
- No change to direction probability interpretation.
- Auxiliary output is training-only initially.

### Invalid prices

Recommended deterministic policy:

- Classification behavior remains unchanged.
- Auxiliary loss is applied only where `close_t` and `targetClose` are finite and positive.
- Invalid regression examples are excluded from the auxiliary denominator only.
- Counts of valid and excluded auxiliary targets are logged.
- Policy identity: for example, `exclude_aux_only_v1`.

### Persisted configuration

A possible canonical identity is:

```text
objective_id =
  und_first_hit_ce_plus_terminal_logreturn_huber_v1

classification_target =
  high_low_first_hit_strict_threshold_up_tie_v1

regression_target =
  terminal_close_log_return_v1

normalization =
  divide_by_direction_threshold_v1

robust_loss =
  huber_delta_1_v1

target_clipping =
  none

invalid_target_policy =
  exclude_aux_only_v1
```

Also persist \(\lambda\), all LR multipliers, effective-gradient scales, and model-format version.

### Compatibility

- Exact configuration only for resume.
- \(\lambda\) is immutable for a run.
- A coefficient schedule, if ever added, is part of objective identity and checkpoint state.
- Legacy models cannot resume into this mode.
- Warm starts are new experiments.
- Current inference profitability remains a validation/model-selection metric, not part of the per-batch loss.

---

## 13. Staged implementation roadmap

### Phase 4A — objective and provenance foundation

Likely files:

- `Headers/Params.hpp`
- `Headers/LSTM.hpp`
- `Headers/PgModelIO.hpp`
- `LSTM/main.cpp`
- `Sources/ExperimentRecommendation.hpp/.cpp`
- experiment/recommendation repositories
- scheduler configuration paths
- documentation

Work:

- Define canonical legacy and multi-task objective identities.
- Codify existing effective-gradient semantics.
- Add objective configuration and exact resume checks.
- Extend experiment identity and canonical text.
- Keep legacy classification as default.

Tests:

- Canonical identity stability
- Legacy default compatibility
- Exact objective mismatch rejection
- Older metadata rejection/fail-closed behavior
- Experiment uniqueness and recommendation provenance

Migration: likely yes for durable experiment/objective fields and uniqueness.

Model format: semantic version change, even before activating the head.

Scheduler behavior: no operational change if default remains legacy.

Backfill dependency: source/design work can proceed; schema rollout should wait.

### Phase 4B — deterministic target generation

Likely files:

- `Headers/TargetLabel.hpp`, or a dedicated training-target helper
- new focused target tests

Work:

- Produce terminal raw log return and normalized auxiliary target from the exact existing label window.
- Add a regression-validity mask.
- Avoid duplicating label-window arithmetic.

Tests:

- Exact `t+H` terminal close
- No access beyond `H`
- Strict threshold equality
- Up/Down first-hit ordering
- Same-bar tie → Up
- First-hit/terminal-sign disagreement cases
- Invalid price handling
- Threshold normalization
- Feature inputs remain unchanged

Migration: no.

Model format: no direct change.

Scheduler behavior: no.

Backfill dependency: none.

### Phase 4C — head and loss implementation

Likely files:

- `Headers/LSTM.hpp`
- `LSTM/LSTM.cpp`
- `LSTM/main.cpp`
- Xcode project/test registration as needed

Work:

- Explicitly initialize the scalar head.
- Add scalar forward path and Huber gradient.
- Combine auxiliary and classifier shared-core gradients.
- Preserve classifier update behavior at \(\lambda=0\).
- Add separate loss and gradient diagnostics.

Tests:

- Scalar-head forward result
- Huber boundary and tail behavior
- Finite-difference head gradients
- Finite-difference or focused shared-core gradient checks
- Valid-target denominator
- No effect from invalid auxiliary targets
- \(\lambda=0\) legacy-equivalence test
- Class-weight normalization test
- Deterministic initialization test
- Gradient clipping and finite-value tests

Migration: no direct DB migration.

Model format: activation of the scalar head requires new semantics.

Scheduler behavior: none until enabled.

Backfill dependency: none for isolated development.

### Phase 4D — serialization and resume compatibility

Likely files:

- `Headers/PgModelIO.hpp`
- `LSTM/main.cpp`
- model persistence and resume tests

Work:

- Bump training/model metadata schema.
- Persist objective canonical text/hash.
- Require exact scalar-head and objective state.
- Reject incompatible legacy and auxiliary resumes.

Tests:

- Exact round trip
- Legacy checkpoint rejection for multi-task resume
- Objective mismatch rejection
- Coefficient mismatch rejection
- Robust-loss/normalization mismatch rejection
- Missing scalar matrix or metadata rejection
- Warm-start versus resume distinction

Migration: possibly shared with 4A.

Model format: yes, semantic version change.

Scheduler behavior: no until deployed.

Backfill dependency: coding does not depend on it; rollout should wait.

### Phase 4E — controlled experiment support

Likely files:

- scheduler CLI/configuration
- experiment services/repositories
- recommendation configuration
- DB migration
- operator documentation

Work:

- Opt-in objective selection.
- Persist every objective parameter in experiment identity.
- Prevent mixed-objective retry/resume.
- Add chronological validation/test configuration.
- Add objective identity to comparison cohorts.

Tests:

- Scheduler command generation
- Retry identity
- No implicit production default change
- Exact campaign/recommendation provenance
- Cohort separation
- Rollback/default legacy behavior

Migration: yes.

Model format: uses the new format.

Production scheduler behavior: changes only when explicitly enabled.

Backfill dependency: complete/quiesced before deployment or launches.

### Phase 4F — independent verification and ablation

Campaign:

- Legacy CE baseline
- New implementation with \(\lambda=0\)
- Several predeclared \(\lambda\) candidates
- Matched seeds and identical data ranges
- Untouched final inference period
- Multiple symbols/horizons where supported

Verification:

- Classification metrics and calibration
- Auxiliary error
- Canonical profitability and support
- Gradient norms, cosine similarity, and clip rate
- Checkpoint and final-inference consistency
- Independent review of identity and split boundaries

Migration: none expected beyond 4A/4E.

Model format: already established.

Scheduler behavior: controlled experimental workload.

Backfill dependency: experiments should wait.

---

## 14. Safety assessment while profitability backfill is running

Safe now:

- Read-only source, test, documentation, migration, and schema inspection
- Architecture and mathematical design
- Offline review of already-written logs
- Planning deterministic tests
- Reviewing existing immutable profitability rows through approved read-only paths, if operationally permitted

Potentially safe while backfill runs, in a later implementation phase:

- Source-only edits in an isolated working tree
- Unit-test source creation
- Compilation into an isolated derived-data path, provided no active executable or generated worker artifact is replaced
- Tests that cannot start scheduler, training, inference, analysis, or database workflows

These activities would still require checking active processes before running anything capable of launching `LSTM_Release`.

Should wait until backfill completes or is deliberately quiesced:

- Database migrations
- Experiment identity or schema rollout
- Model-format deployment
- Replacement of the executable used by active workers
- Scheduler configuration or policy changes
- Continuation-policy deployment tied to new objective metadata
- Campaign/recommendation cohort changes
- Any experiment launch, training run, checkpoint inference, or final inference
- Empirical coefficient-selection campaigns
- Operations that read or rewrite active backfill artifacts

Historical backfill completion is not logically required for Phases 4A–4D source design and deterministic tests. It is required for safe operational rollout and useful population-level profitability calibration.

---

## 15. Open questions and blockers

1. There is no clearly authoritative held-out validation-loss workflow using the exact training CE semantics. That must be established before coefficient selection.
2. Chronological split and embargo rules need an explicit contract. Training labels must not reach into validation/test bars.
3. The empirical distributions of terminal return, normalized return, and first-hit/terminal-sign disagreement are not yet documented.
4. The active classification path has hidden effective-gradient scales—0.1 for logits and 4.0 into the core—that are not represented as objective provenance.
5. Loss reporting and gradient normalization may diverge when class weights are nonunit.
6. The inactive scalar head’s classification-mode initialization and legacy checkpoint meaning are not sufficient for resume compatibility.
7. Experiment and recommendation identity currently omit training-objective configuration.
8. The existing profitability metric excludes costs, sizing, leverage, and overlap effects. It must not be described as realized trading P&L.
9. The minimum actionable-support rule for model selection is not yet defined.
10. A policy is needed for whether profitability-distribution cohorts may compare models trained under different objectives.
11. The working branch is `lstm-feature-development`, while the repository instructions state that the current branch is `phase6`. That discrepancy should be resolved before implementation.
12. There appear to be no focused unit tests defining `CalculateBatch`’s classification-loss and gradient contract. Those should precede auxiliary changes.

---

## 16. Final recommendation

Do not change the production training objective yet.

For the future Phase 4 implementation, use multi-task first-hit classification plus a separate terminal-log-return regression head:

\[
L=L_{\mathrm{CE}}+\lambda
  \operatorname{Huber}_{\delta=1}
  \left(\hat u-\frac{\log(C_{t+H}/C_t)}{\tau}\right).
\]

Keep classification primary, keep all inference inputs unchanged, and keep pure classification as the production default and experimental baseline. Do not use signed-profitability sample weights or expected-return probability objectives.

Before activating the auxiliary objective:

- establish chronological validation and embargo rules;
- version the complete effective loss and gradient semantics;
- extend experiment/model identity;
- enforce exact resume compatibility;
- prove \(\lambda=0\) legacy equivalence;
- run matched-seed ablations on an untouched inference period.

### Inspection record

Files changed: none.

Behavioral changes: none.

Builds/tests run: none, as required by the inspection-only scope. No executable, scheduler, database, or worker-affecting command was run.

Read-only commands used: `rg`, `find`, `nl`, `sed`, `awk`, `wc`, and Git status/diff inspection.

`git status --short`:

```text
 M Database/backups/LSTM_latest.dump
 M Database/backups/LSTM_latest.dump.json
```

Those pre-existing or externally changing backup artifacts were not touched.

`git diff --stat` returned no output for the ordinary unstaged diff inspection.