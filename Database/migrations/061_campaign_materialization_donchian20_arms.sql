-- Allow one ranked scientific candidate to materialize more than one explicit
-- Donchian arm. Proposal identity remains unique within a materialization and
-- carries the authoritative mode through conversion.
ALTER TABLE experiment_recommendation_campaign_materialization_member
    DROP CONSTRAINT IF EXISTS
        recommendation_campaign_materialization_member_rank_uidx;
