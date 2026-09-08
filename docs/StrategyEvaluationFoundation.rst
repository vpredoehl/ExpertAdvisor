Strategy Evaluation Foundation
==============================

Phase 17 introduces a deterministic, downstream strategy-evaluation boundary.
It does not change model training identity, inference behavior, profitability
persistence, ranking, continuation policy, or production scheduling.

Current profitability data flow
-------------------------------

For every evaluable three-class inference window, ``LSTM/main.cpp`` computes
the down/neutral/up probabilities and selects a predicted class. It also reads
the raw close at the decision row and the raw close exactly at the configured
terminal prediction horizon. Those three values, in inference order, are the
complete input to ``InferenceProfitability::Accumulator``.

Neutral predictions are non-actionable. Up and down predictions are actionable
only when both prices are finite and positive. Up return is
``ln(terminal_close / decision_close)`` and down return is its negation. There
are no transaction costs, position sizing, leverage, overlap constraints, or
intrahorizon exits. The source-content hash binds ordinal, predicted class, and
the exact float bit patterns of the two closes.

The inference loop has transient access to all three probabilities and to the
tensor's timestamp and raw OHLC series across the horizon. The persisted
``inference_eval_result`` and immutable profitability observation contain only
aggregate inference/profitability data, the inference period, metric identity,
and terminal source-content hash. They do not persist per-window probabilities,
timestamps, or ordered intrahorizon OHLC paths.

Consequently, later stop, trailing, exit, and re-entry strategies will require
deterministic access to an immutable ordered market-data snapshot (or an
equivalently immutable per-window path artifact) plus per-window inference
outputs. That future input must bind symbol, bar interval, timestamps, ordering,
price-field semantics, inference identity, and a content hash. This phase does
not select or persist that representation.

Component and policy boundary
-----------------------------

``StrategyEvaluationCore`` is a static library with no database, scheduler,
CLI, or broker dependency. It depends on ``ProfitabilityCore`` so the explicit
``BaselineTerminalStrategy`` delegates to the authoritative accumulator. This
preserves exact arithmetic ordering, actionability, source-content hashing, and
metric identity.

``TradingStrategy`` is the intentionally small polymorphic policy boundary.
Evaluation inputs are value types. They carry the three historical terminal
fields and optional probabilities, timestamps, and market paths for future
policies. The baseline ignores optional fields. A future path-dependent policy
must fail closed when required inputs or provenance are absent or incompatible.
``PositionDirection`` establishes only the current flat/short/long mapping; no
position state machine or trading engine is introduced.

Deterministic identity
----------------------

A strategy identity contains a family, positive strategy version, canonical
configuration, configuration hash, full canonical identity, and identity hash.
Configuration schema version 1 treats entries as a logical map: keys are sorted
and duplicates are rejected. Canonical fields are length-prefixed, and hashes
use the repository's tagged deterministic FNV-1a-64 representation. Unknown
configuration schema versions and unsupported baseline parameters fail closed.

Strategy identity is separate from model/inference scientific identity. A
future strategy-evaluation identity must bind both identities plus symbol,
horizon, inference period, profitability metric definition, and immutable
market-data provenance. ``BuildStrategyEvaluationIdentity`` establishes that
contract without introducing persistence.

Historical compatibility
------------------------

Existing profitability observations remain the authoritative baseline record.
Their metric definition, source-content hash, and observation identity are not
changed or retrofitted with strategy identity. The new baseline identity is a
downstream interpretation: future strategy-result persistence can associate it
with an existing profitability observation without changing that observation's
scientific meaning. No database migration is part of this phase.
