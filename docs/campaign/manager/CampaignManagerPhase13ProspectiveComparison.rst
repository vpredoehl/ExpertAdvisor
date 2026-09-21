Campaign Manager Phase 13: Prospective Profitability Comparison
===============================================================

Phase 13 adds a deterministic, repeatable-read comparison of the frozen Phase
11 candidate and control selections.  It does not run inference, train a model,
write an outcome, alter a recommendation, or exercise scheduler authority.
The live profitability ranking weight remains exactly zero.

Frozen identities
-----------------

The comparison fails closed unless all of these identities match:

* cohort ``fnv1a64:fe7aee4a1aed8a5e``;
* Phase 11 artifact SHA-256
  ``8d2176ef26513f6b69d690fa5b550a870aab7190221bace24a18be9200a89bbc``;
* Phase 12 preparation artifact SHA-256
  ``441a1957ac2ffd53a0693e2332d2b9f7af1c1f9977ff688f9d8e043199f581e1``;
* Phase 12 preparation identity ``fnv1a64:efeec6ea26199cc7``;
* profitability metric ``fnv1a64:5894ecab3036bb93``; and
* outcome window ``2026-08-31`` through ``2026-09-30``.

Artifact, cohort, metric, and window mismatches have distinct fail-closed
readiness states.  Outcome rows are also revalidated against the frozen job,
feature, lineage, model-artifact, source-content, statistics, and outcome
identity hashes before they can contribute.

Aggregation semantics
---------------------

The production ranking contract selects recommendation records by ordinal; it
does not deduplicate source models before constructing Top-N.  Phase 12 only
deduplicates execution of the identical frozen-model inference job.  Phase 13
therefore uses recommendation selection slots as the economic weighting unit:
if two changed recommendation slots refer to one model, that model's outcome
contributes twice.  This measures the economic statistic represented by the
ranking slots; the underlying metric explicitly does not model transaction
costs, capital, position sizing, or portfolio P&L.

Statistical evidence is counted separately.  One source model remains one
evidence unit regardless of recommendation multiplicity, and output always
reports ``duplicate_recommendations_are_independent=false``.  Aggregate return,
prediction count, and actionable count are summed with recommendation-slot
weights.  Each side's per-actionable value is the summed aggregate return
divided by its summed actionable count.  No incremental per-actionable metric
is invented because subtracting two differently supported pooled averages is
not the committed metric.

Common recommendation members cancel from candidate-minus-control, so their
outcomes are not required for the incremental result.  Membership is still
rendered and identity-bound.  A final cutoff requires complete valid coverage
of every unique model in its entrant-or-exit recommendation set.  Full-cohort
coverage is reported separately and does not block an unaffected cutoff.

Frozen selection mappings
-------------------------

Top 5:

* control: ``359,404,416,361,360``;
* candidate: ``416,404,418,417,410``;
* retained: ``404,416``;
* entrants: ``410,417,418`` -> models ``1702,1660,1660``;
* exits: ``359,360,361`` -> model ``999`` three times; and
* changed-source coverage denominator: ``999,1660,1702`` (3).

Top 10:

* control: ``359,404,416,361,360,406,405,378,410,418``;
* candidate: ``416,404,418,417,410,406,405,359,407,411``;
* retained: ``359,404,405,406,410,416,418``;
* entrants: ``407,411,417`` -> models ``1658,1702,1660``;
* exits: ``360,361,378`` -> models ``999,999,1015``; and
* changed-source coverage denominator: ``999,1015,1658,1660,1702`` (5).

Top 20:

* control:
  ``359,404,416,361,360,406,405,378,410,418,417,368,380,379,411,412,370,369,362,407``;
* candidate:
  ``416,404,418,417,410,406,405,359,407,411,412,361,360,378,368,409,408,362,380,379``;
* retained:
  ``359,360,361,362,368,378,379,380,404,405,406,407,410,411,412,416,417,418``;
* entrants: ``408,409`` -> model ``1658`` twice;
* exits: ``369,370`` -> model ``1029`` twice; and
* changed-source coverage denominator: ``1029,1658`` (2).

Readiness and current result
----------------------------

The command is:

::

  PGOPTIONS='-c default_transaction_read_only=on' \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
    --compare-campaign-profitability-prospective=fnv1a64:fe7aee4a1aed8a5e

Exit code 0 means all three comparisons are final.  Exit code 4 means pending
or incomplete changed-selection coverage.  Identity, metric, window, or
changed-source compatibility failures return 3; tool/database failures return
2.

On the validation run's UTC date, 2026-08-30, all three cutoffs are
``pending_outcomes`` and ``final=false``.  No prospective outcome is present.
Full frozen-cohort coverage is 0/23 and changed-selection coverage is Top-5
0/3, Top-10 0/5, and Top-20 0/2.  All aggregate and incremental values remain
``PENDING``.  The deterministic status identity is
``fnv1a64:5eca089135fc4f89``; the complete five-line stdout artifact SHA-256 is
``48a633b5339f9e146f9c7d53913f080e3ff7c5c3aa6d1e07835a50e8d1073b05``.

Migration 085 is committed but is not currently installed in the production
database.  Before the window closes, an absent outcome table is safely treated
as zero outcomes; no migration is applied by the status command.

Model 499
---------

Source experiment 150 names model 499 as ``last_model_id``, but model 499 has
no persisted ``experiment_id``, its immutable name refers to experiment 108,
and no exact original FINAL inference row exists.  That evidence cannot prove
the exact lineage.  Phase 13 preserves ``model_lineage_ambiguous`` and makes no
provenance correction.  Model 499 is lower-ranked and is absent from all three
changed-selection source sets, so it limits full-cohort coverage without
blocking Top-5, Top-10, or Top-20.

Post-window procedure
---------------------

After 2026-09-30, first confirm market-data completeness without changing the
window.  Apply committed migration 085 through the repository's authorized
migration procedure if the table is still absent:

::

  ./migrate_lstm_db.sh

Then regenerate read-only job readiness:

::

  PGOPTIONS='-c default_transaction_read_only=on' \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
    --prepare-campaign-profitability-outcome-jobs=fnv1a64:fe7aee4a1aed8a5e \
    > Phase12_PostWindow_Readiness.txt

Execute only the 22 records reported as ``ready_to_execute`` using their exact
emitted commands, whose implemented syntax is:

::

  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
    --run-frozen-model-outcome-inference=COHORT_HASH,SOURCE_EXPERIMENT_ID,SOURCE_MODEL_ID,FROM_DATE,TO_DATE,JOB_HASH

Use the exact six values emitted for each ready job; do not execute model 499
and do not add overrides.  Once those immutable outcomes are persisted, run
the Phase 13 command above twice, compare the byte output, and commit the
machine-readable output plus its SHA-256 as the Phase 14/final evaluation
artifact.  A cutoff becomes final independently when its changed-selection
coverage reaches 100%; full-cohort coverage remains separately visible.
