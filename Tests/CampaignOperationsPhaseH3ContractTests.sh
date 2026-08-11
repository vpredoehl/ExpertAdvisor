#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="$repo_root/Sources/CampaignOperationsDispatchRepository.cpp"
service="$repo_root/Sources/CampaignOperationsManagerService.cpp"
migration_service="$repo_root/Sources/CampaignOperationsDispatchService.cpp"
migration="$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql"
help_source="$repo_root/Sources/ExperimentScheduler.cpp"

snapshot="$(sed -n '/SelectDispatchCandidatesForManager(/,/^}/p' "$repository")"
grep -q 'REPEATABLE READ, READ ONLY' <<<"$snapshot"
grep -q 'ORDER BY request.operational_request_id LIMIT \$1' <<<"$snapshot"
! grep -Eq 'FOR UPDATE|SKIP LOCKED|advisory_xact_lock|nextval|UPDATE |INSERT |DELETE ' <<<"$snapshot"

grep -q 'for (const auto& candidate : result.candidates)' "$service"
grep -q 'DispatchOneRequestForProductionManager' "$service"
! grep -Eq 'sleep_for|poll|daemon|autostart|supervis|continuous' "$service"
! grep -q 'DispatchTestHook' "$service"

grep -q 'source_canonical' "$migration"
grep -q 'ON DELETE RESTRICT' "$migration"
grep -q 'REVOKE ALL PRIVILEGES' "$migration"
grep -q 'CREATE CONSTRAINT TRIGGER campaign_operations_manager_attempt_complete' "$migration"
grep -q 'DEFERRABLE INITIALLY DEFERRED' "$migration"
grep -q 'source evidence incomplete' "$migration"
! grep -Eq 'CREATE ROLE|LOGIN|SKIP LOCKED|advisory_xact_lock|batch_uuid' "$migration"

grep -q 'RequireExactManagerOperationSourceEvidence' "$repository"
grep -q 'managerSourceCanonical' "$repository"
grep -q 'campaign_operations_manager_operation_key_reserved' "$migration_service"

grep -Fq 'continuous CLI/daemon mode. ADR-0020 accepts' "$help_source"
grep -Fq 'external deployment-owned H4 supervision' "$help_source"
grep -Fq 'no scheduler polling or database singleton, heartbeat,' "$help_source"

echo "Campaign Operations Phase H3 structural contract tests passed"
