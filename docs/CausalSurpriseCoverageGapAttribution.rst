Causal Surprise Coverage-Gap Attribution
========================================

The Phase-6 diagnostic attributes causal first-release surprise coverage
without changing model features or persisted state. It reuses the Phase-5
``candlestick_feature_rows_after_feature_warmup_prefix`` population and the
same chronological ``EconomicEventFeatureEngine`` decision for every row.

Command
-------

::

  LSTM_Release --causal-surprise-coverage-gaps=EXPERIMENT_ID \
    --causal-surprise-coverage-gaps-scope=train|infer|combined

The default scope is ``combined``. The command uses repeatable-read,
``pqxx::read_transaction`` transactions for the LSTM and Forex databases. It
does not require a model, checkpoint, final inference, or scheduler worker.

Authority and identity
----------------------

First-release state and raw ``selection_reason`` come from migration 090's
``economic_event_first_release_actual`` assessment and its point-in-time
``economic_event_first_release_actual_at(T)`` API. Compatibility reasons are
reported by the same scalar compatibility decision used to create Tensor
columns 71 and 72. No independent eligibility or normalization policy exists.

The FNV attribution identity binds the Phase-5 upstream identity and coverage
identity, scope and ranges, warmup, width/layout, repository selection,
first-release, compatibility/normalization contracts, source/warmup row
counts, all attribution partitions, affected event identities, and the
Phase-6 semantic version. Experiment ID, downstream feature-ablation mask,
scheduler priority and ownership, process identity, and queue timestamps are
excluded.

Remediability rules
-------------------

The classification is deterministic and advisory only:

* ``no_relevant_event`` and ``not_yet_available`` are
  ``expected_by_contract``.
* ``provenance_unavailable`` with raw reason ``no_actual_observation`` is a
  ``potentially_remediable_data_gap``.
* Other ``provenance_unavailable`` reasons and all ``ambiguous`` rows are
  ``requires_manual_provenance_review``.
* ``missing_consensus`` is a ``potentially_remediable_data_gap``. Its exact
  current persisted state is ``no_selected_consensus_persisted``; the
  diagnostic does not invent a consensus or infer a provider artifact.
* ``incompatible`` is ``unsupported_semantics`` and retains the exact reason
  from the shared compatibility result.

Priority records are ordered by affected feature rows descending. Equal
impacts use raw reason, event family, agency, disposition, and remediability as
stable ascending tie-breakers. Both affected feature-row count and unique
affected-event count are emitted because one selected event can remain active
for many bars.

Time attribution uses the UTC calendar year of each feature bar. The
``no_relevant_event`` record also reports the first and last affected bar as
Unix seconds; year records retain the complete terminal-disposition split.
