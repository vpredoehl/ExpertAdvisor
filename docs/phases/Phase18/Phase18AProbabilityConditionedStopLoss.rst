Phase 18A Probability-Conditioned Stop-Loss Evaluation
=======================================================

Locked version-1 hypothesis
---------------------------

Phase 18A tests one mapping, identified as
``directional_probability_linear_bounded_v1``.  It was implemented without
examining a profitability result.  For an actionable Down or Up prediction,
``directional_probability`` is the probability assigned to that entered
direction.  With ``p`` denoting that value::

  normalized_directional_confidence = clamp((p - 1/3) / (2/3), 0, 1)
  stop_multiplier = min_multiplier
                  + (max_multiplier - min_multiplier)
                  * normalized_directional_confidence
  effective_stop_log_distance = base_stop_log_distance * stop_multiplier

The Phase 18A defaults are base logarithmic distance ``0.001``, minimum
multiplier ``0.75``, and maximum multiplier ``1.25``.  Thus the mathematical
probability values ``1/3``, ``2/3``, and ``1`` map to normalized confidence
``0``, ``0.5``, and ``1`` and to multipliers ``0.75``, ``1``, and ``1.25``.
The identity configuration sets both multipliers to ``1`` and is exactly
equivalent to ``FixedStopLossStrategy`` with the same base distance.

The hypothesis is that greater directional probability receives a wider
initial stop.  This is an experiment definition, not a claim that the mapping
is profitable or optimal.  There is no parameter sweep, probability
calibration, re-entry, trailing update, break-even update, take-profit,
profit-locking, Fibonacci target, grid, pyramiding, or multiple entry.

Authoritative decision-time inputs
----------------------------------

The production three-class head has Down, Neutral, and Up logits in indices
``0``, ``1``, and ``2``.  ``LSTM::PredictNextDirectionProbs`` applies the
existing production ``Softmax3`` and returns three ``float`` probabilities.
``ProcessBatchPredict`` captures those exact values in
``TensorInferenceDecision`` immediately after prediction.  The
``TensorMarketPathAdapter`` derives the predicted class with the production
strict-maximum rule: a unique Down or Up maximum is directional; a Neutral
maximum and every maximum tie are non-actionable Neutral.

There is no per-observation acceptance/rejection threshold.  The model
acceptance check is a later aggregate diagnostic over predicted-class
fractions (Neutral at most ``0.60``, Down and Up each at least ``0.15``); it
does not change an observation's class or entry.  The configured logarithmic
return threshold is used to construct evaluation labels, not to threshold the
model probabilities or strategy entries.

The decision timestamp and entry price are respectively the timestamp and
raw ask close at the last inference-window row.  The prediction horizon is
the configured number of subsequent 15-minute Tensor rows, ending at the
terminal row.  The adapter attaches the exact ordered raw ask OHLC rows from
decision row plus one through that terminal row.  Path construction requires
every probability to be finite and within ``[0,1]``, their double-precision
sum to be within ``1e-4`` of one, and their strict-maximum class to agree with
the stored class.  No separate Phase 18A softmax or calibration exists.

Causal and shared execution semantics
-------------------------------------

Direction, probability, confidence, multiplier, effective distance, and the
one initial stop are all fixed from the decision-time observation before any
future path point is inspected.  Phase 18A and Phase 17 fixed stop both call
the same ``EvaluateSingleInitialFixedStop`` primitive.  Long touch is
``low <= stop``; short touch is ``high >= stop``.  A gap beyond the stop fills
at the adverse open, a non-gap touch fills at the stop, the first triggering
bar exits the trade, and an untriggered trade exits at terminal close.

Controlled comparison and evidence
----------------------------------

``EvaluateControlledProbabilityConditionedStopExperiment`` evaluates exactly
three durable output identities on one immutable ``AuthoritativeMarketPath``:

* ``baseline_terminal_v1``
* ``fixed_stop_loss_v1``
* ``probability_conditioned_stop_loss_v1``

The fixed control uses exactly the probability strategy's base distance.
Pairing fails closed unless all results bind the same path hash and have the
same result count, ordinal, predicted direction, decision timestamp, and
actionable entry price.  Neutral observations must have no entry, stop, exit,
or return.  Because each policy receives the same immutable path, predicted
class, probabilities, decision timestamp, entry, and future OHLC are common;
only exit policy may differ.

Deterministic line records include experiment/configuration/semantic identity,
per-strategy aggregates, four predeclared confidence buckets
``[0,.25)``, ``[.25,.50)``, ``[.50,.75)``, and ``[.75,1]``, all three required
pairwise deltas, and per-observation decision and execution evidence.
Aggregates include counts, signed log-return sum and average, win/loss and
stop/terminal rates, mean and median holding time, maximum adverse and
favorable excursion, sequence maximum drawdown, and average effective stop
distance/multiplier.

Excursion uses all completed bars while a trade remains active.  On a
stop-triggering bar it uses the realized exit price only because OHLC cannot
establish whether another intrabar extreme happened before or after exit.
This prevents post-exit extremes from being presented as trade exposure.
Adverse excursion is reported as a nonnegative loss magnitude; favorable
excursion is reported as a nonnegative gain magnitude.
Drawdown is the peak-to-trough decline of cumulative directional log return in
stable observation-ordinal order; it is a deterministic comparison sequence,
not a portfolio-overlap simulation.

Read-only interface
-------------------

The standalone CLI option is::

  --infer --model=<model_id> \
    --probability-conditioned-stop-evaluation=<artifact_path> \
    <fromDate> <toDate>

It rejects scheduler context and ``--infer-all``, opens its database
transaction read-only, writes the deterministic artifact, commits the
read-only transaction, and skips inference persistence and model save.  The
core itself remains database-, Tensor-, scheduler-, and CLI-independent.
