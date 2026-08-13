#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="$repo_root/Sources/ExperimentScheduler.cpp"
generic_main="$repo_root/LSTM/main.cpp"
h4_env="$repo_root/Deployment/CampaignOperationsH4/campaign-operations-h4.connection.env.example"

# The generic builders are deliberately unchanged for ordinary LSTM runtime
# paths.  Production routing exists only in the Campaign Operations command
# dispatcher and requires explicit environment names with no pqxx fallback.
rg -q 'gssencmode=disable user=pqxx dbname=' "$source_file"
rg -q 'gssencmode=disable user=pqxx dbname=' "$generic_main"
rg -q 'CampaignOperationsProductionConnectionString' "$source_file"
rg -q 'principal environment variable' "$source_file"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER' "$source_file"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER' "$source_file"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_ENABLER_DB_USER' "$source_file"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_DISABLER_DB_USER' "$source_file"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=REPLACE_WITH_MANAGER_LOGIN' "$h4_env"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER=REPLACE_WITH_DISPATCH_SERVICE_LOGIN' "$h4_env"

command_body="$(sed -n '/int RunCampaignOperationsCommand/,/^int RunExperimentRecommendationCommand/p' "$source_file")"
manager_routes="$(printf '%s\n' "$command_body" | sed -n '1,/if (options.campaignOperationsProductionEnable)/p')"
enable_route="$(printf '%s\n' "$command_body" | sed -n '/if (options.campaignOperationsProductionEnable)/,/if (options.campaignOperationsProductionDisable)/p')"
disable_route="$(printf '%s\n' "$command_body" | sed -n '/if (options.campaignOperationsProductionDisable)/,/if (options.campaignOperationsProductionDispatchRequest)/p')"
dispatch_route="$(printf '%s\n' "$command_body" | sed -n '/if (options.campaignOperationsProductionDispatchRequest)/,/Pre-Phase-H Campaign Operations/p')"

rg -q 'campaignOperationsProductionReadiness' <<<"$manager_routes"
rg -q 'campaignOperationsProductionStatus' <<<"$manager_routes"
rg -q 'campaignOperationsManagerRunOnceLimit' <<<"$manager_routes"
[[ "$(rg -o 'CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER' <<<"$manager_routes" | wc -l | tr -d ' ')" == 3 ]]
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_ENABLER_DB_USER' <<<"$enable_route"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_DISABLER_DB_USER' <<<"$disable_route"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER' <<<"$dispatch_route"
rg -q 'CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER' <<<"$dispatch_route"
rg -q 'CampaignOperationsPrePhaseHConnectionString' <<<"$command_body"
rg -q 'ValidateCampaignOperationsPrePhaseHPrincipal' <<<"$command_body"
rg -q 'CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER' "$source_file"
rg -q 'campaignOperationsAdmitMaterializationId' "$source_file"
rg -q 'RunOperationalCampaignAdmissionCommand' "$source_file"
rg -q 'kCampaignOperationsCampaignCreatorRole' "$repo_root/Sources/CampaignOperationsRepository.cpp"
rg -q 'campaign_operations_pre_phase_h_principal_invalid' "$source_file"
rg -q 'missing required Campaign Operations pre-Phase-H principal' "$source_file"

# Migration 056 deliberately contains no LOGIN creation or pqxx capability
# grant.  The H2 deployment audit remains the runtime proof of its absence.
if rg -q 'CREATE ROLE .*LOGIN|GRANT .* TO pqxx' \
    "$repo_root/Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql"; then
  echo "migration 056 incorrectly creates a LOGIN or grants pqxx" >&2
  exit 1
fi

echo "Campaign Operations Phase H production connection routing tests passed"
