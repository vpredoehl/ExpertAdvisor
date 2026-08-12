#!/usr/bin/env bash
set -euo pipefail

audit_version="h1-deployment-audit-v1"

audit_failure() {
    local sqlstate="$1"
    local diagnostic="$2"
    local object_identity="${3//$'\n'/;}"
    local failure_stage="$4"
    local detail="${5//$'\n'/;}"
    printf 'SQLSTATE=%s diagnostic=%s object=%s stage=%s audit_version=%s detail=%s\n' \
        "$sqlstate" "$diagnostic" "$object_identity" "$failure_stage" \
        "$audit_version" "$detail" >&2
    exit 1
}

usage() {
    echo "usage: $0 --stage STAGE --host HOST --port PORT --user USER --database DATABASE" >&2
    exit 64
}

stage=""
audit_host=""
audit_port=""
audit_user=""
audit_database=""
while (($#)); do
    case "$1" in
        --stage) stage="${2:-}"; shift 2 ;;
        --host) audit_host="${2:-}"; shift 2 ;;
        --port) audit_port="${2:-}"; shift 2 ;;
        --user) audit_user="${2:-}"; shift 2 ;;
        --database) audit_database="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$stage" && -n "$audit_host" && -n "$audit_port" &&
   -n "$audit_user" && -n "$audit_database" ]] || usage
case "$stage" in
    pre-upgrade|post-upgrade|pre-restore|post-role-recreation|\
    post-database-restore|pre-enablement) ;;
    *) audit_failure 55000 H1A011 "$stage" "$stage" \
         "unsupported-audit-stage" ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$repo_root/Scripts/CampaignOperationsH1ManifestValidator.sh" >/dev/null
migration="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
expected_checksum="$(shasum -a 256 "$migration" | awk '{print $1}')"
psql_target=(-X -qAt -v ON_ERROR_STOP=1 -v VERBOSITY=verbose -h "$audit_host" -p "$audit_port"
    -U "$audit_user" "$audit_database")

role_findings="$(psql "${psql_target[@]}" <<'SQL'
WITH RECURSIVE expected(role_name, must_super) AS (VALUES
 ('campaign_operations_h1_boundary_authority', true),
 ('campaign_operations_production_enabler', false),
 ('campaign_operations_production_disabler', false),
 ('campaign_operations_production_dispatcher', false),
 ('campaign_operations_production_phase5_transactional', false),
 ('campaign_operations_production_reader', false),
 ('campaign_operations_scheduler_protocol_evidence_owner', false),
 ('campaign_operations_scheduler_protocol_evidence_reader', false)),
attributes AS (
 SELECT expected.role_name,
        role.oid,
        role.rolcanlogin OR role.rolsuper <> expected.must_super OR
        NOT role.rolinherit OR role.rolcreatedb OR role.rolcreaterole OR
        role.rolreplication OR role.rolbypassrls OR role.rolconnlimit <> -1 OR
        role.rolpassword IS NOT NULL OR role.rolvaliduntil IS NOT NULL AS mismatch
 FROM expected
 LEFT JOIN pg_catalog.pg_authid role ON role.rolname = expected.role_name),
settings AS (
 SELECT DISTINCT attributes.role_name
 FROM attributes
 JOIN pg_catalog.pg_db_role_setting setting ON setting.setrole = attributes.oid),
raw_edges AS (
 SELECT granted.rolname::text AS granted_role,
        member_role.rolname::text AS member_role, membership.admin_option
 FROM pg_catalog.pg_auth_members membership
 JOIN pg_catalog.pg_roles granted ON granted.oid = membership.roleid
 JOIN pg_catalog.pg_roles member_role ON member_role.oid = membership.member),
member_to_h1 AS (
 SELECT edge.granted_role AS root_role, edge.member_role AS endpoint_role,
        ARRAY[edge.member_role, edge.granted_role]::text[] AS role_sequence,
        ARRAY[edge.admin_option]::boolean[] AS admin_sequence,
        ARRAY[edge.member_role, edge.granted_role]::text[] AS visited, 1 AS depth
 FROM raw_edges edge
 WHERE edge.granted_role IN (SELECT role_name FROM expected)
 UNION ALL
 SELECT path.root_role, edge.member_role,
        edge.member_role || path.role_sequence,
        edge.admin_option || path.admin_sequence,
        edge.member_role || path.visited, path.depth + 1
 FROM member_to_h1 path
 JOIN raw_edges edge ON edge.granted_role = path.endpoint_role
 WHERE NOT edge.member_role = ANY(path.visited)),
h1_to_granted AS (
 SELECT edge.member_role AS root_role, edge.granted_role AS endpoint_role,
        ARRAY[edge.member_role, edge.granted_role]::text[] AS role_sequence,
        ARRAY[edge.admin_option]::boolean[] AS admin_sequence,
        ARRAY[edge.member_role, edge.granted_role]::text[] AS visited, 1 AS depth
 FROM raw_edges edge
 WHERE edge.member_role IN (SELECT role_name FROM expected)
 UNION ALL
 SELECT path.root_role, edge.granted_role,
        path.role_sequence || edge.granted_role,
        path.admin_sequence || edge.admin_option,
        path.visited || edge.granted_role, path.depth + 1
 FROM h1_to_granted path
 JOIN raw_edges edge ON edge.member_role = path.endpoint_role
 WHERE NOT edge.granted_role = ANY(path.visited)),
shortest_paths AS (
 SELECT DISTINCT ON (direction, root_role, endpoint_role)
        direction, root_role, endpoint_role, role_sequence, admin_sequence, depth
 FROM (
   SELECT 'member_to_h1'::text AS direction, root_role, endpoint_role,
          role_sequence, admin_sequence, depth FROM member_to_h1
   UNION ALL
   SELECT 'h1_to_granted', root_role, endpoint_role,
          role_sequence, admin_sequence, depth FROM h1_to_granted
 ) candidate
 ORDER BY direction, root_role, endpoint_role, depth, role_sequence),
path_edges AS (
 SELECT path.*, edge_index,
        path.role_sequence[edge_index] AS edge_from,
        path.role_sequence[edge_index + 1] AS edge_to,
        path.admin_sequence[edge_index] AS admin_option
 FROM shortest_paths path
 CROSS JOIN LATERAL pg_catalog.generate_series(1, path.depth) edge_index)
SELECT 'missing:' || role_name FROM attributes WHERE oid IS NULL
UNION ALL
SELECT 'attributes:' || role_name FROM attributes WHERE coalesce(mismatch, false)
UNION ALL
SELECT 'settings:' || role_name FROM settings
UNION ALL
SELECT 'graph:direction=' || direction || ',root=' || root_role ||
       ',endpoint=' || endpoint_role || ',depth=' || depth ||
       ',roles=' || pg_catalog.array_to_string(role_sequence, '->') ||
       ',edge_depth=' || edge_index || ',edge=' || edge_from || '->' ||
       edge_to || ',admin_option=' || admin_option::text
FROM path_edges
ORDER BY 1;
SQL
)"

if [[ "$stage" == "pre-upgrade" ]]; then
    role_findings="$(printf '%s\n' "$role_findings" |
        awk '!/^missing:/' | sed '/^$/d')"
fi
if [[ "$stage" == "pre-enablement" ]]; then
    # H2 alone authorizes the direct deployment LOGIN-to-capability graph.
    # Retain H1's sealed-owner reachability check here; the H2 audit below
    # validates the exact accepted capability tuples and rejects every other
    # graph edge, including pqxx membership and ADMIN OPTION.
    role_findings="$(printf '%s\n' "$role_findings" |
        awk '!/^graph:/ || /campaign_operations_h1_boundary_authority/' |
        sed '/^$/d')"
fi
if [[ -n "$role_findings" ]]; then
    if rg -q '^graph:' <<<"$role_findings"; then
        audit_failure 42501 H1A003 "$role_findings" "$stage" \
            "prohibited-H1-role-graph-edge"
    elif rg -q '^missing:' <<<"$role_findings"; then
        audit_failure 42501 H1A001 "$role_findings" "$stage" \
            "required-H1-role-missing"
    else
        audit_failure 42501 H1A002 "$role_findings" "$stage" \
            "role-identity-mismatch"
    fi
fi

# H1's frozen in-database audit intentionally proves the exact 055 catalog.
# At pre-enablement after H2 is installed, it must not be asked to claim that
# the H2 readiness adapter existed at H1.  The H2 deployment audit is the
# separately versioned authority for that one extension (including migration
# ledger identity, ACL, owner, SECURITY DEFINER, volatility and search path).
# Keep the H1 manifest immutable and add only this checked-in, ledger-bound
# union at the post-H1 stage.  Any other boundary-owned function remains a
# closed-set mismatch.
if [[ "$stage" == "pre-enablement" ]]; then
    "$repo_root/Scripts/CampaignOperationsH2DeploymentAudit.sh" \
        --stage "$stage" --host "$audit_host" --port "$audit_port" \
        --user "$audit_user" --database "$audit_database"

    post_h1_function_findings="$({
        printf '%s\n' 'BEGIN TRANSACTION READ ONLY;'
        printf '%s\n' 'WITH expected_h1(signature) AS ('
        awk -F '\t' '
            NR > 1 && $3 == "function" &&
            $6 == "campaign_operations_h1_boundary_authority" {
                value = $5
                gsub(/\047/, "\047\047", value)
                if (count++) printf " UNION ALL\n"
                printf " SELECT \047%s\047::text", value
            }
            END { if (!count) exit 1; printf "\n" }
        ' "$repo_root/Database/manifests/055_campaign_operations_h1_object_inventory.tsv"
        printf '%s\n' '), accepted_post_h1(signature) AS ('
        printf '%s\n' " SELECT 'public.campaign_operations_production_readiness_snapshot_v1()'::text"
        printf '%s\n' " UNION ALL SELECT 'public.campaign_operations_production_dispatch_readiness_gate_v1(text)'::text"
        printf '%s\n' " UNION ALL SELECT 'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::text"
        printf '%s\n' '), expected(signature) AS ('
        printf '%s\n' ' SELECT signature FROM expected_h1 UNION ALL SELECT signature FROM accepted_post_h1'
        printf '%s\n' '), actual(signature) AS ('
        printf '%s\n' " SELECT namespace.nspname || '.' || function_row.oid::pg_catalog.regprocedure::text"
        printf '%s\n' ' FROM pg_catalog.pg_proc function_row'
        printf '%s\n' ' JOIN pg_catalog.pg_namespace namespace ON namespace.oid = function_row.pronamespace'
        printf '%s\n' " WHERE function_row.proowner = 'campaign_operations_h1_boundary_authority'::regrole"
        printf '%s\n' '), wrapper AS ('
        printf '%s\n' ' SELECT bool_and(function_row.proowner = '\''campaign_operations_h1_boundary_authority'\''::regrole'
        printf '%s\n' "        AND function_row.prosecdef AND function_row.prokind = 'f'"
        printf '%s\n' "        AND ((function_row.oid = 'public.campaign_operations_production_readiness_snapshot_v1()'::regprocedure AND function_row.provolatile = 's') OR (function_row.oid = 'public.campaign_operations_production_dispatch_readiness_gate_v1(text)'::regprocedure AND function_row.provolatile = 's') OR (function_row.oid = 'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure AND function_row.provolatile = 'v'))"
        printf '%s\n' "        AND function_row.proparallel = 'u'"
        printf '%s\n' '        AND NOT function_row.proleakproof AND function_row.pronargdefaults = 0'
        printf '%s\n' '        AND function_row.provariadic = 0'
        printf '%s\n' "        AND function_row.proconfig IS NOT DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[]) AS valid"
        printf '%s\n' ' FROM pg_catalog.pg_proc function_row'
        printf '%s\n' " WHERE function_row.oid IN ('public.campaign_operations_production_readiness_snapshot_v1()'::regprocedure, 'public.campaign_operations_production_dispatch_readiness_gate_v1(text)'::regprocedure, 'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure)"
        printf '%s\n' ')'
        printf '%s\n' "SELECT 'boundary-function-set:' || signature FROM ("
        printf '%s\n' ' SELECT signature FROM expected EXCEPT SELECT signature FROM actual'
        printf '%s\n' ' UNION ALL SELECT signature FROM actual EXCEPT SELECT signature FROM expected'
        printf '%s\n' ') mismatch'
        printf '%s\n' 'UNION ALL'
        printf '%s\n' "SELECT 'readiness-wrapper-properties' WHERE NOT coalesce((SELECT valid FROM wrapper), false);"
        printf '%s\n' 'COMMIT;'
    } | psql "${psql_target[@]}")"
    if [[ -n "$post_h1_function_findings" ]]; then
        audit_failure 42501 H1A004 "$post_h1_function_findings" "$stage" \
            "unaccepted-post-H1-boundary-function-evolution"
    fi

    echo "H1_DEPLOYMENT_AUDIT_V1_OK stage=$stage post_h1_evolution=059-direct-sql-boundary"
    exit 0
fi

if [[ "$stage" == "pre-upgrade" || "$stage" == "pre-restore" ]]; then
    preflight_findings="$(psql "${psql_target[@]}" <<'SQL'
WITH boundary AS (
 SELECT oid FROM pg_catalog.pg_roles
 WHERE rolname = 'campaign_operations_h1_boundary_authority'),
protected_names(name) AS (VALUES
 ('record_campaign_operations_production_enable_v1'),
 ('record_campaign_operations_production_disable_v1'),
 ('transition_campaign_operations_request_dispatch_production_v2'),
 ('campaign_operations_production_transition_context'))
SELECT 'owned-relation:' || namespace.nspname || '.' || relation.relname
FROM pg_catalog.pg_class relation
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace
WHERE relation.relowner = (SELECT oid FROM boundary)
  AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
  AND namespace.nspname <> 'information_schema'
UNION ALL
SELECT 'owned-schema:' || namespace.nspname
FROM pg_catalog.pg_namespace namespace
WHERE namespace.nspowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-type:' || namespace.nspname || '.' || type_row.typname
FROM pg_catalog.pg_type type_row
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = type_row.typnamespace
WHERE type_row.typowner = (SELECT oid FROM boundary)
  AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
  AND namespace.nspname <> 'information_schema'
UNION ALL
SELECT 'owned-operator:' || namespace.nspname || '.' || operator.oprname
FROM pg_catalog.pg_operator operator
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = operator.oprnamespace
WHERE operator.oprowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-event-trigger:' || trigger.evtname
FROM pg_catalog.pg_event_trigger trigger
WHERE trigger.evtowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-publication:' || publication.pubname
FROM pg_catalog.pg_publication publication
WHERE publication.pubowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-subscription:' || subscription.subname
FROM pg_catalog.pg_subscription subscription
WHERE subscription.subowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-large-object:' || large_object.oid::text
FROM pg_catalog.pg_largeobject_metadata large_object
WHERE large_object.lomowner = (SELECT oid FROM boundary)
UNION ALL
SELECT 'owned-function:' || namespace.nspname || '.' || function_row.proname
FROM pg_catalog.pg_proc function_row
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = function_row.pronamespace
WHERE function_row.proowner = (SELECT oid FROM boundary)
  AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
  AND namespace.nspname <> 'information_schema'
UNION ALL
SELECT 'protected-name:' || namespace.nspname || '.' || function_row.proname
FROM pg_catalog.pg_proc function_row
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = function_row.pronamespace
WHERE function_row.proname IN (SELECT name FROM protected_names)
UNION ALL
SELECT 'wrapper:' || namespace.nspname || '.' || function_row.proname
FROM pg_catalog.pg_proc function_row
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = function_row.pronamespace
WHERE namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
  AND namespace.nspname <> 'information_schema'
  AND function_row.prosrc ~
      '(record_campaign_operations_production_|transition_campaign_operations_request_dispatch_production_v2|campaign_operations_production_transition_context)'
UNION ALL
SELECT 'trigger-entry:' || relation_namespace.nspname || '.' ||
       relation.relname || '.' || trigger.tgname
FROM pg_catalog.pg_trigger trigger
JOIN pg_catalog.pg_proc function_row ON function_row.oid = trigger.tgfoid
JOIN pg_catalog.pg_class relation ON relation.oid = trigger.tgrelid
JOIN pg_catalog.pg_namespace relation_namespace
  ON relation_namespace.oid = relation.relnamespace
WHERE NOT trigger.tgisinternal
  AND function_row.prosrc ~
      '(record_campaign_operations_production_|transition_campaign_operations_request_dispatch_production_v2|campaign_operations_production_transition_context)'
UNION ALL
SELECT 'cast-entry:' || cast_row.oid::text
FROM pg_catalog.pg_cast cast_row
JOIN pg_catalog.pg_proc function_row ON function_row.oid = cast_row.castfunc
WHERE function_row.proname IN (SELECT name FROM protected_names)
UNION ALL
SELECT 'default-acl:' || owner_role.rolname || ':' ||
       default_acl.defaclobjtype::text || ':' || default_acl.defaclnamespace::text
FROM pg_catalog.pg_default_acl default_acl
JOIN pg_catalog.pg_roles owner_role ON owner_role.oid = default_acl.defaclrole
WHERE owner_role.rolname IN (
  'campaign_operations_h1_boundary_authority',
  'campaign_operations_owner',
  'campaign_operations_scheduler_protocol_evidence_owner')
  AND EXISTS (
    SELECT 1 FROM pg_catalog.aclexplode(default_acl.defaclacl) acl
    WHERE acl.grantee <> default_acl.defaclrole)
ORDER BY 1;
SQL
)"
    if [[ -n "$preflight_findings" ]]; then
        if rg -q '^(protected-name|wrapper|trigger-entry|cast-entry):' \
            <<<"$preflight_findings"; then
            audit_failure 42501 H1A005 "$preflight_findings" "$stage" \
                "protected-alternate-entry-point"
        elif rg -q '^default-acl:' <<<"$preflight_findings"; then
            audit_failure 42501 H1A007 "$preflight_findings" "$stage" \
                "unsafe-default-ACL"
        else
            audit_failure 42501 H1A004 "$preflight_findings" "$stage" \
                "unexpected-boundary-owned-object"
        fi
    fi
elif [[ "$stage" != "pre-restore" &&
        "$stage" != "post-role-recreation" ]]; then
    psql "${psql_target[@]}" -v audit_checksum="$expected_checksum" <<'SQL'
BEGIN TRANSACTION READ ONLY;
SELECT public.campaign_operations_h1_deployment_audit_v1(
    :'audit_checksum', true, false);
COMMIT;
SQL
    acl_findings="$(psql "${psql_target[@]}" \
        -f "$repo_root/Database/manifests/055_campaign_operations_h1_acl_manifest.sql")"
    if [[ -n "$acl_findings" ]]; then
        printf '%s\n' "$acl_findings" >&2
        if rg -q '^H1A007\|' <<<"$acl_findings"; then
            audit_failure 42501 H1A007 "pg_default_acl" "$stage" \
                "exact-default-ACL-tuple-matrix-mismatch"
        else
            acl_object="$(printf '%s\n' "$acl_findings" | head -n 1 | \
                awk -F '|' '{print $9}')"
            audit_failure 42501 H1A006 "${acl_object:-explicit-acl}" \
                "$stage" "exact-explicit-ACL-tuple-matrix-mismatch"
        fi
    fi
fi

echo "H1_DEPLOYMENT_AUDIT_V1_OK stage=$stage"
