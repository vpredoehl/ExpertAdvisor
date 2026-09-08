Phase 17C Authoritative Market Path and Fixed Stop Loss
======================================================

Gate A: authoritative path
--------------------------

The production market-data route remains unchanged::

  PostgreSQL candlestick(symbol, 15, 'minute', range) ORDER BY dt
      -> db_cursor_stream<Feature>
      -> Tensor raw OHLC/time arrays
      -> TensorMarketPathAdapter
      -> AuthoritativeMarketPath
      -> StrategyEvaluationCore

No repository, SQL query, or second historical-price loader was added. The
adapter reads only the raw values already retained by the exact ``Tensor`` used
by inference.

For a window beginning at source row ``w``, window size ``W``, and prediction
horizon ``H``:

* the model sees rows ``w`` through ``w + W - 1``;
* entry decision time and price are the timestamp and close of row
  ``d = w + W - 1``;
* execution path points are rows ``d + 1`` through ``d + H``, inclusive;
* the terminal timestamp and close are exactly row ``d + H``.

The adapter derives predicted class from the model's down/neutral/up
probabilities with the production strict-maximum rule. A unique down or up
maximum is directional; every maximum tie is neutral. Prediction, direction,
entry, and initial stop therefore use decision-time information only. Future
OHLC is attached only after those values are established and is used only for
execution simulation.

Timestamp and interval semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source ``dt`` is the start of the 15-minute candlestick bucket. Historical
source timestamps are naive ``America/New_York`` civil values. The existing
cursor parser applies its versioned U.S. Eastern DST rule and ``Tensor`` stores
the result at whole-second UTC precision. Paths require strictly increasing
timestamps and source-row order. They do not require a 900-second wall-clock
delta because market closures can create legitimate gaps. Duplicate or
descending timestamps fail closed.

The database candlestick definition aggregates the raw ``ask`` series.
Therefore all decision, OHLC, stop, and exit values in this phase are
single-domain ask prices. This is not broker-realistic two-sided execution:
long liquidation would ordinarily consume bid and short entry would ordinarily
sell bid. No spread, slippage, or synthetic bid is invented. ``price_domain``
is bound into path provenance so another domain cannot share the same identity.

Canonicalization and provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AuthoritativeMarketPath`` is immutable after construction and owns the
validated observations. Canonicalization version 1 uses explicit
length-prefixed fields, decimal integers, exact binary32 bit encodings for
OHLC/probabilities, explicit observation/path ordinals, and strict source-row
mapping. It binds:

* adapter family/version and canonicalization/schema versions;
* model ID and complete caller-supplied inference scientific identity/hash;
* symbol, window size, horizon, and evaluation period;
* bar duration, timestamp semantics, OHLC semantics, price domain, source
  relation, and ordering rule;
* evaluation metric canonical/hash;
* every decision class/probability/timestamp/price and every ordered OHLC path
  point.

The content hash is tagged deterministic FNV-1a-64, matching the existing
ProfitabilityCore convention. Construction rejects unsupported versions,
invalid identity hashes, absent semantic fields/timestamps, invalid classes or
probabilities, non-finite/non-positive prices, impossible OHLC relationships,
incorrect row/horizon mappings, unordered observations/points, and a terminal
point that does not exactly match the declared terminal timestamp/close.

The underlying ``candlestick`` function's fallback implementation should be
separately hardened in a future market-data phase: its ``first_value`` and
``last_value`` window expressions do not themselves declare an intra-bucket
window order. Phase 17C neither bypasses nor silently repairs that authoritative
source. Instead, the adapter binds the complete OHLC values actually supplied
to inference, so any changed materialization produces a different path hash.
Durable strategy-result persistence must retain the canonical path or an
immutable snapshot reference before cross-run reproduction is claimed.

Gate B: fixed stop loss version 1
---------------------------------

Gate A provides all information required for one fixed protective stop. There
is no hidden high/low ordering dependency: with one entry, one immutable stop,
no take-profit, no trailing update, and no re-entry, the first bar whose adverse
extreme touches the stop necessarily causes the sole exit. Favorable/adverse
ordering within that bar cannot change the outcome for either direction.

The sole parameter is a finite positive logarithmic distance ``delta``. It is
encoded by exact binary64 bits in strategy identity. For entry price ``E``::

  long stop  = E * exp(-delta)
  short stop = E * exp(+delta)

This is symmetric with the existing returns::

  long  =  ln(exit / entry)
  short = -ln(exit / entry)

Execution rule ``single_fixed_protective_stop_ohlc_v1`` is part of strategy
identity. Long exact touch is ``low <= stop``; short exact touch is
``high >= stop``. If that bar opens beyond the stop, the fill is the adverse
open (long ``open <= stop`` or short ``open >= stop``), never the more favorable
stop. Otherwise a touched stop fills at the stop price. If no point triggers,
the position closes at the terminal bar close. Neutral is ``no_action``.

Each result binds ``no_action``, ``terminal_exit``, or ``fixed_stop_exit`` plus
entry/exit timestamp and price, initial stop, return, and the triggering path
ordinal/source row when applicable. Result identity binds the path hash,
strategy/configuration/execution-rule identity, inference identity, evaluation
period, metric identity, and exit-reason-bearing canonical results.

Compatibility and persistence
-----------------------------

``BaselineTerminalStrategy`` still delegates directly to
``InferenceProfitability::Accumulator``. Its operation order, invalid-price
handling, statistics, metric definition/hash, and source-content hash are
unchanged. No historical profitability row or production schema was modified.

Future persistence should store strategy identity, inference/model identity,
market-path canonical/hash or immutable snapshot identity, evaluation period,
metric identity, result canonical/hash, and per-observation exit evidence. It
must not retrofit strategy identity into historical baseline observations.

Source ownership
----------------

``Sources/StrategyEvaluationCore`` owns the database/Tensor/scheduler-free
core and is compiled only by the ``StrategyEvaluationCore`` static-library
target. ``Sources/StrategyEvaluationAdapters`` owns the Tensor adapter and is
compiled once by each application target, where ``Tensor`` is available. Xcode
contains matching groups. Application targets link, but do not separately
compile, the core implementation.

Scheduler observation (Section 17A)
-----------------------------------

Read-only inspection confirmed experiments 616 and 617 were live stopped
workers with durable ``paused``, ``resume_requested=false``, and
``scheduler_resume_origin=none``. Current scheduler logs do not show a priority
preemption for either worker. In particular, 617 was launched normally and
later appeared stopped without a ``SCHEDULER_PRIORITY_PREEMPTED`` event; the
waiting experiment 618 had equal ``normal`` priority, which current policy
cannot use to preempt 617. The state is exactly the explicit pause contract,
and the single-experiment pause command does not use
``experiment_admin_worker_outcome``. The claimed scheduler-displacement origin
loss is therefore not confirmed by current code or evidence, and no scheduler
state or production row was changed in Phase 17C.
