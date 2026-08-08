#!/usr/bin/env bash
set -euo pipefail
usage() { echo "usage: $0 HOST PORT USER TEMPLATE_DATABASE OUTPUT" >&2; exit 64; }
[[ $# -eq 5 ]] || usage
host="$1" port="$2" user="$3" template_database="$4" output="$5"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixtures="$repo_root/Tests/fixtures/CampaignOperationsH1AclOriginFixtures.tsv"
target=(-h "$host" -p "$port" -U "$user")
run_id="${H1_EVIDENCE_RUN_ID:-h1-acl-origin-$(date -u +%Y%m%dT%H%M%SZ)-${$}}"
artifact_root="$(cd "$(dirname "$output")" && pwd)"
raw_root="$artifact_root/raw-acl-origin"
mkdir -p "$raw_root"
active_database=""
cleanup() {
  [[ -z "$active_database" ]] || dropdb "${target[@]}" --if-exists "$active_database" >/dev/null 2>&1 || true
}
trap cleanup EXIT

while IFS=$'\t' read -r fixture requirement class schema identity column \
    expected_origin actual_origin direction; do
  [[ "$fixture" == fixture_id ]] && continue
  active_database="h1_acl_origin_${$}_${fixture#H1AO}"
  createdb "${target[@]}" -T "$template_database" "$active_database"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$active_database" <<'SQL'
CREATE SCHEMA h1_acl_origin_schema;
CREATE TABLE public.h1_acl_origin_table(value integer);
CREATE TABLE public.h1_acl_origin_partitioned(value integer) PARTITION BY RANGE(value);
CREATE VIEW public.h1_acl_origin_view AS SELECT value FROM public.h1_acl_origin_table;
CREATE MATERIALIZED VIEW public.h1_acl_origin_matview AS SELECT value FROM public.h1_acl_origin_table;
CREATE FOREIGN DATA WRAPPER h1_acl_origin_fdw NO HANDLER;
CREATE SERVER h1_acl_origin_server FOREIGN DATA WRAPPER h1_acl_origin_fdw;
CREATE FOREIGN TABLE public.h1_acl_origin_foreign(value integer) SERVER h1_acl_origin_server;
CREATE SEQUENCE public.h1_acl_origin_sequence;
CREATE FUNCTION public.h1_acl_origin_function(integer) RETURNS integer LANGUAGE sql AS 'SELECT $1';
CREATE PROCEDURE public.h1_acl_origin_procedure(integer) LANGUAGE sql AS 'SELECT $1';
CREATE AGGREGATE public.h1_acl_origin_aggregate(integer) (SFUNC=int4pl, STYPE=integer, INITCOND='0');
CREATE TYPE public.h1_acl_origin_type AS (value integer);
CREATE DOMAIN public.h1_acl_origin_domain AS integer;
CREATE TABLE public.h1_acl_origin_columns(
  completion_insert integer, completion_audit_insert integer,
  request_owner_update integer, request_boundary_update integer,
  cancellation_audit_insert integer, recovery_audit_insert integer,
  scheduler_update integer);
SQL

  case "$class" in
    schema)
      locator="nspname='h1_acl_origin_schema'"; table=pg_namespace
      owner_expr=nspowner; acl_type=n; comment="COMMENT ON SCHEMA h1_acl_origin_schema" ;;
    table)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=r; comment="COMMENT ON TABLE ${identity}" ;;
    partitioned_table)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=r; comment="COMMENT ON TABLE ${identity}" ;;
    view)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=r; comment="COMMENT ON VIEW ${identity}" ;;
    materialized_view)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=r; comment="COMMENT ON MATERIALIZED VIEW ${identity}" ;;
    foreign_table)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=r; comment="COMMENT ON FOREIGN TABLE ${identity}" ;;
    sequence)
      relation="${identity#public.}"; locator="oid='public.${relation}'::regclass"
      table=pg_class; owner_expr=relowner; acl_type=S; comment="COMMENT ON SEQUENCE ${identity}" ;;
    function|procedure|aggregate)
      signature="${identity#public.}"; locator="oid='public.${signature}'::regprocedure"
      table=pg_proc; owner_expr=proowner; acl_type=f
      case "$class" in
        function) keyword=FUNCTION ;;
        procedure) keyword=PROCEDURE ;;
        aggregate) keyword=AGGREGATE ;;
      esac
      comment="COMMENT ON ${keyword} ${identity}" ;;
    type)
      type_name="${identity#public.}"; locator="oid='public.${type_name}'::regtype"
      table=pg_type; owner_expr=typowner; acl_type=T; comment="COMMENT ON TYPE ${identity}" ;;
    domain)
      type_name="${identity#public.}"; locator="oid='public.${type_name}'::regtype"
      table=pg_type; owner_expr=typowner; acl_type=T; comment="COMMENT ON DOMAIN ${identity}" ;;
    column)
      attribute="${identity##*.}"
      locator="attrelid='public.h1_acl_origin_columns'::regclass AND attname='${attribute}'"
      table=pg_attribute; owner_expr="(SELECT relowner FROM pg_class WHERE oid=attrelid)"
      acl_type=column; comment="COMMENT ON COLUMN ${identity}" ;;
    *) echo "unsupported ACL-origin class $class" >&2; exit 1 ;;
  esac
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$active_database" -c \
    "$comment IS 'h1-acl-origin:${expected_origin}'" >/dev/null
  if [[ "$actual_origin" == null ]]; then
    mutation="UPDATE ${table} SET ${column}=NULL WHERE ${locator}"
  elif [[ "$acl_type" == column ]]; then
    mutation="UPDATE ${table} SET ${column}=ARRAY[makeaclitem(${owner_expr},${owner_expr},'SELECT',false)] WHERE ${locator}"
  else
    mutation="UPDATE ${table} SET ${column}=acldefault('${acl_type}',${owner_expr}) WHERE ${locator}"
  fi
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$active_database" -c "$mutation" >/dev/null

  catalog="$(psql "${target[@]}" -X -At -F $'\t' "$active_database" -c \
    "SELECT (${column} IS NULL)::text,coalesce(${column}::text,'<NULL>'),
            pg_get_userbyid(${owner_expr}) FROM ${table} WHERE ${locator}")"
  IFS=$'\t' read -r raw_null raw_acl owner <<< "$catalog"
  [[ "$raw_null" == t || "$raw_null" == true ]] && raw_null=true || raw_null=false
  [[ "$raw_acl" != '<NULL>' ]] || raw_acl=""
  expansion_file="$(mktemp /tmp/h1-acl-expansion.XXXXXX)"
  if [[ "$acl_type" == column ]]; then
    if [[ "$raw_null" == true ]]; then
      : > "$expansion_file"
    else
      psql "${target[@]}" -X -At -F $'\t' "$active_database" -c \
        "SELECT CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
                acl.privilege_type,CASE WHEN acl.is_grantable THEN 'true' ELSE 'false' END
           FROM ${table},LATERAL aclexplode(${column}) acl
          WHERE ${locator} ORDER BY 1,2,3" > "$expansion_file"
    fi
    acldefault_type=column:none
  else
    psql "${target[@]}" -X -At -F $'\t' "$active_database" -c \
      "SELECT CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
              acl.privilege_type,CASE WHEN acl.is_grantable THEN 'true' ELSE 'false' END
         FROM ${table},LATERAL aclexplode(coalesce(${column},acldefault('${acl_type}',${owner_expr}))) acl
        WHERE ${locator} ORDER BY 1,2,3" > "$expansion_file"
    acldefault_type="$acl_type"
  fi

  log="$(mktemp /tmp/h1-acl-audit.XXXXXX)"
  if psql "${target[@]}" -X -q -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
      "$active_database" >"$log" 2>&1 -c \
      "BEGIN READ ONLY; SELECT public.campaign_operations_h1_deployment_audit_v1(NULL,false,false); COMMIT"; then
    cat "$log" >&2; echo "$fixture production audit unexpectedly passed" >&2; exit 1
  fi
  audit_line="$(rg -m1 'ERROR:  42501: H1A006 ' "$log" || true)"
  [[ -n "$audit_line" ]] || { cat "$log" >&2; echo "$fixture wrong production audit branch" >&2; exit 1; }
  diagnostic="${audit_line#*ERROR:  42501: }"
  [[ "$diagnostic" == *"object=${identity}"* && "$diagnostic" == *"stage=database-audit"* ]] || {
    cat "$log" >&2; echo "$fixture production audit identity/stage mismatch" >&2; exit 1;
  }
  dropdb "${target[@]}" "$active_database"; active_database=""
  raw_file="$raw_root/${fixture}.tsv"
  printf 'format\th1-acl-origin-raw-v3\n' > "$raw_file"
  printf 'meta\t%s\t%s\tGEN-ACL-ORIGIN\th1-generator-registry-v2\tScripts/CampaignOperationsH1AclEvidence.py\tgenerate\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t42501\t%s\t%s\tdatabase-audit\tPASS\n' \
    "$run_id" "ART-ACL-RAW-${fixture}" \
    "$fixture" "$requirement" "$class" "$schema" "$identity" "$column" \
    "$expected_origin" "$actual_origin" "$direction" "$raw_null" "$raw_acl" \
    "$owner" "$acldefault_type" "$diagnostic" "$identity" >> "$raw_file"
  awk -F '\t' 'BEGIN{OFS="\t"}{print "acl",$1,$2,$3}' "$expansion_file" >> "$raw_file"
done < "$fixtures"

python3 "$repo_root/Scripts/CampaignOperationsH1AclEvidence.py" generate \
  "$fixtures" "$raw_root" "$output" "$run_id"
echo "Campaign Operations H1 authentic ACL-origin production-audit tests passed rows=38"
