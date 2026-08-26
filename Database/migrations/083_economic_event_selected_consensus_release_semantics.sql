-- Expose the already-immutable actual semantics from the selected provider
-- observation through the provider-neutral selected-consensus abstraction.
-- No persisted consensus row is inserted, updated, or deleted by this
-- migration.  Feature code continues to read only the selected abstraction.

CREATE OR REPLACE VIEW economic_event_selected_consensus AS
SELECT
    c.economic_event_consensus_id,
    c.economic_event_id,
    c.forecast_canonical_value_low AS consensus_value_low,
    c.forecast_canonical_value_high AS consensus_value_high,
    c.forecast_value_kind AS consensus_value_kind,
    c.forecast_unit AS consensus_unit,
    c.forecast_scale AS consensus_scale,
    c.forecast_qualifier AS consensus_qualifier,
    c.consensus_source,
    c.source_report_id,
    c.source_event_id,
    c.source_observation_id,
    c.source_release_date,
    c.source_artifact_path,
    c.source_artifact_sha256,
    c.candidate_classification,
    c.match_rule,
    c.semantic_contract,
    c.provider_provenance,
    c.imported_at,
    c.actual_parse_status AS selected_actual_parse_status,
    c.actual_canonical_value_low AS selected_actual_value_low,
    c.actual_canonical_value_high AS selected_actual_value_high,
    c.actual_value_kind AS selected_actual_value_kind,
    c.actual_unit AS selected_actual_unit,
    c.actual_scale AS selected_actual_scale,
    c.actual_qualifier AS selected_actual_qualifier
FROM economic_event_consensus c
WHERE c.forecast_parse_status = 'parsed';

COMMENT ON VIEW economic_event_selected_consensus IS
    'The sole populated consensus observation for each authoritative economic event; selected immutable release-actual semantics and provider provenance remain explicit.';

COMMENT ON COLUMN economic_event_selected_consensus.selected_actual_parse_status IS
    'Whether the same immutable selected provider observation contains a parsed release actual; feature availability remains gated by the authoritative event timestamp.';

GRANT SELECT ON economic_event_selected_consensus TO pqxx;
