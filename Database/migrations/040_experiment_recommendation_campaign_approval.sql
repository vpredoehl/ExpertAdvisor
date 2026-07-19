-- Phase 4D Step 3: immutable operator decisions for one exact deterministic
-- campaign review. Approval is audit evidence only and never executes a
-- campaign or creates, queues, or modifies an experiment.

CREATE TABLE IF NOT EXISTS experiment_recommendation_campaign_approval (
    recommendation_campaign_approval_id bigserial PRIMARY KEY CHECK (
        recommendation_campaign_approval_id > 0),
    approval_contract_version integer NOT NULL CHECK (
        approval_contract_version = 1),
    recommendation_ranking_snapshot_id bigint NOT NULL REFERENCES
        experiment_recommendation_ranking_snapshot(
            recommendation_ranking_snapshot_id) ON DELETE RESTRICT CHECK (
                recommendation_ranking_snapshot_id > 0),
    ranking_snapshot_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(ranking_snapshot_identity_canonical) <> '' AND
        octet_length(ranking_snapshot_identity_canonical) <= 4194304),
    ranking_snapshot_identity_hash text NOT NULL CHECK (
        btrim(ranking_snapshot_identity_hash) <> '' AND
        octet_length(ranking_snapshot_identity_hash) <= 256),
    planning_policy_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(planning_policy_canonical) <> '' AND
        octet_length(planning_policy_canonical) <= 1048576),
    planning_policy_hash text NOT NULL CHECK (
        btrim(planning_policy_hash) <> '' AND
        octet_length(planning_policy_hash) <= 256),
    planning_scope_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(planning_scope_canonical) <> '' AND
        octet_length(planning_scope_canonical) <= 65536),
    campaign_plan_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(campaign_plan_identity_canonical) <> '' AND
        octet_length(campaign_plan_identity_canonical) <= 16777216),
    campaign_plan_identity_hash text NOT NULL CHECK (
        btrim(campaign_plan_identity_hash) <> '' AND
        octet_length(campaign_plan_identity_hash) <= 256),
    review_contract_version integer NOT NULL CHECK (
        review_contract_version = 1),
    campaign_review_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(campaign_review_identity_canonical) <> '' AND
        octet_length(campaign_review_identity_canonical) <= 33554432),
    campaign_review_identity_hash text NOT NULL CHECK (
        btrim(campaign_review_identity_hash) <> '' AND
        octet_length(campaign_review_identity_hash) <= 256),
    campaign_review_hash_collision_ordinal integer NOT NULL CHECK (
        campaign_review_hash_collision_ordinal >= 0),
    candidate_count integer NOT NULL CHECK (candidate_count >= 0),
    selected_count integer NOT NULL CHECK (selected_count >= 0),
    excluded_count integer NOT NULL CHECK (excluded_count >= 0),
    duplicate_group_count integer NOT NULL CHECK (duplicate_group_count >= 0),
    duplicate_candidate_count integer NOT NULL CHECK (
        duplicate_candidate_count >= 0 AND
        duplicate_candidate_count <= candidate_count),
    considered_family_count integer NOT NULL CHECK (
        considered_family_count >= 0),
    selected_family_count integer NOT NULL CHECK (
        selected_family_count >= 0 AND
        selected_family_count <= considered_family_count),
    considered_symbol_count integer NOT NULL CHECK (
        considered_symbol_count >= 0),
    selected_symbol_count integer NOT NULL CHECK (
        selected_symbol_count >= 0 AND
        selected_symbol_count <= considered_symbol_count),
    considered_horizon_count integer NOT NULL CHECK (
        considered_horizon_count >= 0),
    selected_horizon_count integer NOT NULL CHECK (
        selected_horizon_count >= 0 AND
        selected_horizon_count <= considered_horizon_count),
    deterministic_ordering_verified boolean NOT NULL CHECK (
        deterministic_ordering_verified),
    decision text NOT NULL CHECK (decision IN ('approved','rejected')),
    reviewer_identity text COLLATE "C" NOT NULL CHECK (
        reviewer_identity = btrim(reviewer_identity, E' \t\n\r\f\v') AND
        reviewer_identity <> '' AND octet_length(reviewer_identity) <= 200 AND
        reviewer_identity !~ '[[:cntrl:]]'),
    reason_text text NOT NULL CHECK (
        reason_text = btrim(reason_text, E' \t\n\r\f\v') AND
        reason_text <> '' AND octet_length(reason_text) <= 2000),
    approval_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(approval_identity_canonical) <> '' AND
        octet_length(approval_identity_canonical) <= 67108864),
    approval_identity_hash text NOT NULL CHECK (
        btrim(approval_identity_hash) <> '' AND
        octet_length(approval_identity_hash) <= 256),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_approval_review_hash_ordinal_uidx
        UNIQUE (
            campaign_review_identity_hash,
            campaign_review_hash_collision_ordinal),
    CONSTRAINT recommendation_campaign_approval_counts_check CHECK (
        candidate_count = selected_count + excluded_count),
    CONSTRAINT recommendation_campaign_approval_hash_format_check CHECK (
        ranking_snapshot_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
        planning_policy_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
        campaign_plan_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
        campaign_review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
        approval_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    CONSTRAINT recommendation_campaign_approval_zero_selection_check CHECK (
        decision = 'rejected' OR selected_count > 0)
);

CREATE INDEX IF NOT EXISTS recommendation_campaign_approval_review_hash_idx
    ON experiment_recommendation_campaign_approval(
        campaign_review_identity_hash);
-- Exact canonical equality remains authoritative. A PostgreSQL hash index
-- avoids btree tuple-size limits for reviews up to 32 MiB while retaining the
-- server's exact equality recheck for every lookup.
CREATE INDEX IF NOT EXISTS recommendation_campaign_approval_review_canonical_idx
    ON experiment_recommendation_campaign_approval USING hash(
        campaign_review_identity_canonical);
CREATE INDEX IF NOT EXISTS recommendation_campaign_approval_snapshot_idx
    ON experiment_recommendation_campaign_approval(
        recommendation_ranking_snapshot_id,
        recommendation_campaign_approval_id);
CREATE INDEX IF NOT EXISTS recommendation_campaign_approval_decision_idx
    ON experiment_recommendation_campaign_approval(
        decision, recommendation_campaign_approval_id);

COMMENT ON TABLE experiment_recommendation_campaign_approval IS
    'Immutable operator decision for one exact Phase 4D campaign review; never execution.';
COMMENT ON COLUMN
    experiment_recommendation_campaign_approval.campaign_review_identity_canonical IS
    'Authoritative exact reviewed identity; the hash is only an accelerator.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_approval FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_approval FROM pqxx;
GRANT SELECT ON experiment_recommendation_campaign_approval TO pqxx;
GRANT INSERT (
    approval_contract_version,
    recommendation_ranking_snapshot_id,
    ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,
    planning_policy_canonical,
    planning_policy_hash,
    planning_scope_canonical,
    campaign_plan_identity_canonical,
    campaign_plan_identity_hash,
    review_contract_version,
    campaign_review_identity_canonical,
    campaign_review_identity_hash,
    campaign_review_hash_collision_ordinal,
    candidate_count,
    selected_count,
    excluded_count,
    duplicate_group_count,
    duplicate_candidate_count,
    considered_family_count,
    selected_family_count,
    considered_symbol_count,
    selected_symbol_count,
    considered_horizon_count,
    selected_horizon_count,
    deterministic_ordering_verified,
    decision,
    reviewer_identity,
    reason_text,
    approval_identity_canonical,
    approval_identity_hash
) ON experiment_recommendation_campaign_approval TO pqxx;

DO $$
DECLARE
    sequence_name text := pg_get_serial_sequence(
        'experiment_recommendation_campaign_approval',
        'recommendation_campaign_approval_id');
BEGIN
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
