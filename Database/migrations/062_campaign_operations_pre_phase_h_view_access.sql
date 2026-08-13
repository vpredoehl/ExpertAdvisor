-- Restore the explicit read needed by the owner of the Phase 2 status views.
-- Migration 055 seals campaign_operations_operational_request under the H1
-- boundary owner; that ownership transfer removes the former owner's
-- implicit access, while the views remain owned by campaign_operations_owner.
-- No LOGIN, capability membership, production privilege, or pqxx grant is
-- introduced here.
REVOKE ALL PRIVILEGES ON
    campaign_operations_budget_status_v1,
    campaign_operations_request_status_v1
    FROM pqxx;
REVOKE SELECT ON campaign_operations_operational_request
    FROM pqxx;
GRANT SELECT ON campaign_operations_operational_request
    TO campaign_operations_owner;
