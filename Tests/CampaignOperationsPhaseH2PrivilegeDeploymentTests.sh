#!/usr/bin/env bash
set -euo pipefail

stage=""
host=""
port=""
user_name=""
database_name=""
while (($#)); do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --user) user_name="$2"; shift 2 ;;
    --database) database_name="$2"; shift 2 ;;
    *) exit 64 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$repo_root/Scripts/CampaignOperationsH2DeploymentAudit.sh" \
  --stage "$stage" --host "$host" --port "$port" --user "$user_name" --database "$database_name"

psql -X -qAt -v ON_ERROR_STOP=1 -h "$host" -p "$port" -U "$user_name" "$database_name" <<'SQL'
WITH checks(label, passed) AS (VALUES
 ('enabler-fixed-only',
   has_function_privilege('campaign_operations_production_enabler', 'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_enabler', 'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_enabler', 'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')),
 ('disabler-fixed-only',
   has_function_privilege('campaign_operations_production_disabler', 'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_disabler', 'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_disabler', 'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')),
 ('dispatcher-fixed-only',
   NOT has_function_privilege('campaign_operations_production_dispatcher', 'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatcher', 'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatcher', 'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatcher', 'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)', 'EXECUTE')),
 ('dispatch-service-minimum',
   has_function_privilege('campaign_operations_production_dispatch_service', 'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatch_service', 'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatch_service', 'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_dispatch_service', 'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)', 'EXECUTE')),
 ('phase5-helper-and-production-bind',
   has_function_privilege('campaign_operations_production_phase5_transactional', 'public.campaign_operations_scheduler_protocol_evidence_lock_v1()', 'EXECUTE')
   AND has_function_privilege('campaign_operations_production_phase5_transactional', 'public.transition_campaign_operations_request_bound_production_v2(bigint,integer,text,bigint,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_phase5_transactional', 'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_phase5_transactional', 'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)', 'EXECUTE')
   AND NOT has_function_privilege('campaign_operations_production_phase5_transactional', 'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)', 'EXECUTE')),
 ('no-direct-evidence-dml',
   NOT has_table_privilege('campaign_operations_production_dispatcher', 'public.campaign_operations_production_enablement_event', 'INSERT,UPDATE,DELETE')
   AND NOT has_table_privilege('campaign_operations_production_phase5_transactional', 'public.campaign_operations_production_enablement_event', 'INSERT,UPDATE,DELETE')),
 ('no-raw-scheduler-dml',
   NOT has_table_privilege('campaign_operations_production_enabler', 'public.experiment_scheduler_protocol', 'INSERT,UPDATE,DELETE')
   AND NOT has_table_privilege('campaign_operations_production_disabler', 'public.experiment_scheduler_protocol', 'INSERT,UPDATE,DELETE')
   AND NOT has_table_privilege('campaign_operations_production_dispatcher', 'public.experiment_scheduler_protocol', 'INSERT,UPDATE,DELETE')
   AND NOT has_table_privilege('campaign_operations_production_phase5_transactional', 'public.experiment_scheduler_protocol', 'INSERT,UPDATE,DELETE')),
 ('no-public-fixed-execute',
   NOT EXISTS (
     SELECT 1 FROM pg_catalog.pg_proc p
     CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl, pg_catalog.acldefault('f', p.proowner))) a
     WHERE p.oid IN (
       'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
       'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
       'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure,
       'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure)
       AND a.grantee=0 AND a.privilege_type='EXECUTE')),
 ('no-pqxx-fixed-execute',
   NOT EXISTS (
     SELECT 1 FROM pg_catalog.pg_proc p
     CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl, pg_catalog.acldefault('f', p.proowner))) a
     JOIN pg_catalog.pg_roles r ON r.oid=a.grantee
     WHERE p.oid IN (
       'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
       'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
       'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure,
       'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure)
       AND r.rolname='pqxx')))
SELECT CASE WHEN bool_and(passed) THEN 'H2_PRIVILEGE_MATRIX_OK'
            ELSE 'H2A003:privilege-matrix' END
FROM checks;
SQL
