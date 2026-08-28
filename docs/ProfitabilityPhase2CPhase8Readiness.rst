Profitability Phase 2C / Campaign Manager Phase 8 Readiness
============================================================

Phase 8 is a diagnostic and shadow-validation increment.  It does not change
recommendation scoring, the authoritative Campaign Manager rank, scheduler
policy, or any activation workflow.

Exact-FINAL evidence contract
-----------------------------

``--verify-profitability-evidence=EXPERIMENT_ID[,EXPERIMENT_ID...]`` reads one
declared experiment set in caller order under a repeatable-read, read-only
transaction.  IDs must be positive and unique.  The verifier resolves the
experiment's final model and the single exact completed inference result whose
scope is ``final`` and whose ``checkpoint_eval_id`` is NULL.  Checkpoint
evidence never substitutes for FINAL evidence.

The authoritative observation must bind the exact experiment, model,
inference-result ID, inference range, current metric canonical text and hash,
source-content hash, and reconstructed observation canonical identity/hash.
Counts, finite return values, aggregate decomposition, and the nullable average
shape are validated.  The metric is terminal-horizon directional log-return
evidence; it is not portfolio P&L and does not model costs, sizing, leverage, or
overlap capital constraints.

Evidence states are ``valid``, ``unavailable``, ``incomplete``, ``ambiguous``,
``invalid_provenance``, ``invalid_metric_definition``, and ``invalid_values``.
Missing evidence is unavailable, never zero.  A valid observation with zero
actionable predictions has aggregate zero and a NULL average.

The command emits ``PROFITABILITY_EVIDENCE`` and
``PROFITABILITY_VERIFICATION_SUMMARY`` records.  Exit 0 means all declared
evidence is valid; exit 4 means software success with unavailable or incomplete
scientific evidence; exit 3 means invalid or ambiguous evidence; exit 1 is an
argument/tool error other than PostgreSQL; and exit 2 is a PostgreSQL error.

Campaign Manager shadow contract
--------------------------------

``--campaign-profitability-readiness=RANKING_SNAPSHOT_ID`` reads one existing
immutable ranking snapshot.  For each member it verifies that the ranking
member and recommendation agree on source experiment and source model, resolves
the current exact-FINAL observation, and requires an exact match with the
Phase 3A profitability provenance frozen into the recommendation.  A later
observation cannot reinterpret a historical recommendation.  Missing legacy
frozen provenance, source-model mismatch, or frozen/current evidence mismatch
fails closed as invalid provenance.

The version-1 shadow policy is transparent and advisory.  It orders explicit
profitability signs as positive, zero, negative, zero-actionable, unavailable,
then invalid.  Within positive, zero, or negative evidence it orders average
terminal-horizon directional log return descending, aggregate return
descending, then actionable count descending.  Existing current rank and
ranking-member ID are deterministic tie-breakers.  The complete canonical
policy and its FNV-1a hash are emitted in
``CAMPAIGN_PROFITABILITY_SHADOW_POLICY``.

Every candidate record exposes current rank and score, exact evidence state and
identity, profitability sign and primitives, shadow rank, and rank delta.
Classification accuracy remains visible, so a high classification score cannot
conceal an explicitly negative profitability sign.  Missing evidence remains
separate from negative evidence.

Safety and readiness gate
-------------------------

Current persisted rank remains authoritative.  Phase 8 fixes live
profitability weight and live profitability score contribution at exactly zero,
reports ``current_rank_changed=false`` and ``live_ranking_changed=false``, and
always reports ``campaign_profitability_activation_performed=false``.

``CAMPAIGN_PROFITABILITY_READINESS_GATE`` separately reports software,
Campaign profitability contract, and shadow-ranking readiness.  Deterministic
actions include ``blocked_software_readiness``, ``blocked_evidence_contract``,
``needs_policy_decision``, and ``ready_for_shadow_validation``.  Passing this
gate authorizes shadow validation only.  A separately reviewed scientific
policy, validation evidence, and explicit future activation increment are
required before profitability can influence live Campaign Manager ranking.

Both Phase 8 commands are read-only.  Operational verification should also set
``PGOPTIONS='-c default_transaction_read_only=on'``.
