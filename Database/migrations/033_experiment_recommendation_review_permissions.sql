-- Phase 4A Step 5 hardening: review events are append-only for the normal
-- runtime role. Migration 032 granted DELETE to support fixture cleanup, but
-- cleanup must use an owner/test connection instead of weakening audit history.

REVOKE UPDATE, DELETE ON TABLE experiment_recommendation_review_event FROM pqxx;
GRANT SELECT, INSERT ON TABLE experiment_recommendation_review_event TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_recommendation_review_event_id_seq
    TO pqxx;
