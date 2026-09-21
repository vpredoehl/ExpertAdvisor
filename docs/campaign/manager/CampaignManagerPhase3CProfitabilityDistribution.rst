Campaign Manager Phase 3C profitability distribution
=====================================================

Scope and safety boundary
-------------------------

Phase 3C adds a pure deterministic advisory analysis layer in
``ProfitabilityDistribution.hpp``. It does not query or write PostgreSQL,
does not participate in recommendation evaluation or ranking, and has no
scheduler entry point. The authoritative persisted source remains
``inference_profitability_observation``. The generic advisory layer can
describe an exact FINAL or checkpoint scope, but Campaign Manager callers must
resolve an exact observation through the Phase 3A FINAL provenance path before
constructing a scoring-policy candidate. Checkpoint evidence is never a
substitute for FINAL evidence, and scopes cannot be mixed.

The Phase 3C API deliberately declares only these scoring values::

  profitability_weight = 0
  profitability_score_contribution = 0

They are compile-time constants, not configurable policy fields. The same
values are included in normalization-policy, summary, per-observation, and
whole-analysis canonical output. No Phase 3C type is a member of
``RecommendationScoringInput``, ``RecommendationScoreResult``,
``RecommendationEvaluationResult``, or ``RecommendationRankingMember``.

No database migration is part of Phase 3C. The observation schema, inference
persistence path, observation identities, and metric definition are unchanged.

Observation representation and validation
-----------------------------------------

The analytical observation carries the persisted observation/result/model/
experiment identifiers, all persisted counts and return sums, metric canonical
and hash, source-content hash, observation canonical and hash, inference scope
and window, exact symbol, horizon, model input width, feature-class identity,
inference-evaluation identity, and the Phase 3B scoring and evaluation semantic
identities.

Validation fails closed for:

* absent or non-positive authoritative identifiers;
* empty required canonical provenance;
* a malformed hash or canonical/hash disagreement;
* unsupported scope or inference dates that are not real Gregorian calendar
  dates in strict ``YYYY-MM-DD`` form, or a non-increasing inference window;
* empty symbol, non-positive horizon, or non-positive input width;
* empty or internally inconsistent semantic identities;
* non-finite values, invalid count relationships, invalid gross-return signs,
  an aggregate inconsistent with its gross parts, or an average inconsistent
  with aggregate/actionable count; and
* an average on a zero-actionable row, or no average on a nonzero-actionable
  row.

Zero actionable is available evidence, not missing evidence and not zero
profitability. Such a row remains in population membership and totals but its
primary distribution value and normalized candidate are unavailable.

Exact population identity
-------------------------

A requested distribution has one exact comparability identity containing:

* ``metric_definition_canonical`` and ``metric_definition_hash``;
* ``inference_scope``;
* exact ``inference_start`` and ``inference_end``;
* exact symbol;
* prediction horizon;
* model input width and feature-class semantic identity;
* inference/evaluation semantic identity for the persisted inference result;
* Phase 3B scoring semantic identity; and
* Phase 3B evaluation semantic identity.

Both canonical text and hash are verified; canonical equality is authoritative.
This implementation therefore normalizes by exact symbol+horizon+window and
the remaining semantic dimensions, not globally. No symbol-family grouping is
invented because this provenance path has no authoritative family identity.
The design is intentionally stricter
than a future empirical policy might ultimately require. Relaxing a dimension
requires evidence and an explicit later policy version; Phase 3C never coerces
or partitions an incompatible requested population.

``source_content_hash`` and ``observation_identity`` identify members. They are
not comparability grouping keys because different models are expected to
produce different predictions within one scientific cohort. Sorted observation
canonical identities form a separate membership canonical/hash, proving exactly
which rows were summarized and making database row order irrelevant.

Primary metric and statistics
-----------------------------

The primary distribution variable is
``average_terminal_horizon_log_return_per_actionable_prediction``. It removes
the first-order exposure-count scaling present in aggregate return and is the
least misleading existing measure of per-actionable directional quality. It is
still not portfolio P&L: transaction costs, sizing, leverage, capital limits,
and overlapping exposure are not modeled.

The summary reports total population and analyzable counts, zero-actionable
count, total predictions and actionables, negative/zero/positive counts,
minimum, maximum, arithmetic mean, median, configurable quantiles, population
standard deviation, and median absolute deviation. Quantiles use deterministic
linear interpolation at ``p * (n - 1)``. Values are sorted before accumulation,
and canonical numeric text is locale independent.

Normalization transform
-----------------------

Phase 3C implements ``signed_empirical_midrank_percentile_v1``. For a raw value
``x`` among ``n`` analyzable values::

  percentile = (count(value < x) + 0.5 * count(value = x)) / n

The bounded candidate is::

  x < 0: 0.5 * percentile
  x = 0: 0.5
  x > 0: 0.5 + 0.5 * percentile

The result is bounded in ``[0, 1]``, preserves the economic sign around neutral
``0.5``, uses deterministic midranks for ties, and limits outlier magnitude
without fabricating cross-cohort scale comparability. Unlike an ordinary
percentile, the best member of an all-negative population cannot appear as a
positive signal. Raw magnitude remains visible beside the candidate.

This transform is a candidate for later controlled policy experimentation, not
an approved scoring component. A later phase should compare it with robust
z-score and winsorized alternatives after the backfill provides adequate data.

Actionable support
------------------

Support is intentionally separate from profitability normalization::

  reliability = actionable_count /
                (actionable_count + support_half_saturation_count)

The default half-saturation count is 100, so 0 actionables produces 0 support
and 100 produces 0.5. This is a transparent monotone descriptive factor, not an
effective independent sample-size estimate: overlapping inference windows can
make raw actionable count overstate independence. Phase 3C never multiplies
reliability, the bounded candidate, or raw profitability into a recommendation
score.

Edge cases and failure behavior
-------------------------------

* Empty input is invalid.
* Any malformed or incompatible member invalidates the entire request; no row
  is silently discarded.
* A single analyzable member is explicitly insufficient. It receives midrank
  ``0.5`` for inspection, but its status remains insufficient.
* An all-equal population has zero standard deviation and MAD. Equal positive,
  zero, and negative values map to ``0.75``, ``0.5``, and ``0.25`` respectively.
* Negative raw values always map below ``0.5``; zero maps exactly to ``0.5``;
  positive values map above ``0.5``.
* Strong outliers cannot exceed the empirical midrank bound and raw magnitude
  remains visible.
* A zero-actionable member has no raw average, percentile, or bounded candidate.
* If analyzable membership is below the configured minimum, statistics and
  candidates are still exposed but every analyzable result is marked
  ``insufficient_population``.

Determinism and observability
-----------------------------

Equivalent input orders produce identical population membership, statistics,
per-observation ordering, canonical text, and hashes. There is no wall-clock,
random, locale, or database-row-order dependency.

The returned summary and per-observation normalization records expose raw
profitability, population identity and membership hashes, total and analyzable
population size, empirical percentile, bounded candidate, actionable support,
explicit unavailable/invalid reason, and the immutable zero weight and zero
score contribution. These records are in-memory/canonical advisory output only;
Phase 3C adds no persistence table or CLI mutation path.

Deferred decision-bearing work
------------------------------

A later explicitly approved phase must use the completed backfill to determine
cohort coverage, minimum stable sample size, effective support under overlapping
windows, symbol/horizon scale stability, regime/window treatment, costs, and
whether the signed percentile preserves economically useful magnitude. That
phase would also require versioned policy/schema design and regression proof
before adding any scoring component. None of that decision-bearing work is
implemented here.
