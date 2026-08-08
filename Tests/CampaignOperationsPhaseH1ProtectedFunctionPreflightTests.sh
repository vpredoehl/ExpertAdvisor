#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
cluster_root="$(mktemp -d /tmp/ea-h1-function-preflight.XXXXXX)"
cluster_data="$cluster_root/data"
cluster_socket="$cluster_root/s"
mkdir -p "$cluster_socket"

cleanup() {
  if [[ -f "$cluster_data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_data" -m immediate stop >/dev/null 2>&1 || true
  fi
  rm -rf -- "$cluster_root"
}
trap cleanup EXIT

extract_contract_array() {
  local source_file="$1" declaration="$2" occurrence="${3:-1}"
  awk -v declaration="$declaration" -v occurrence="$occurrence" '
    index($0, declaration " text[] := ARRAY[") {
      seen++
      if (seen == occurrence) {inside=1}
      next
    }
    inside && /];$/ {exit}
    inside {line=$0; sub(/^[[:space:]]*/, "", line); print line}
  ' "$source_file"
}

assert_duplicate_contracts() {
  local source_file="$1"
  diff -u \
    <(extract_contract_array "$source_file" protected_functions) \
    <(extract_contract_array "$source_file" allowed_functions) >/dev/null ||
    return 1
  diff -u \
    <(extract_contract_array "$source_file" protected_function_signatures) \
    <(extract_contract_array "$source_file" allowed_function_signatures 1) \
    >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" protected_function_signatures) \
    <(extract_contract_array "$source_file" allowed_function_signatures 2) \
    >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" protected_function_contracts) \
    <(extract_contract_array "$source_file" allowed_function_contracts 1) \
    >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" protected_function_contracts) \
    <(extract_contract_array "$source_file" allowed_function_contracts 2) \
    >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" \
       protected_function_return_contracts) \
    <(extract_contract_array "$source_file" \
       allowed_function_return_contracts) >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" \
       protected_function_identity_argument_contracts) \
    <(extract_contract_array "$source_file" \
       allowed_function_identity_argument_contracts) >/dev/null || return 1
  diff -u \
    <(extract_contract_array "$source_file" \
       protected_function_final_extra_acl_contracts) \
    <(extract_contract_array "$source_file" \
       allowed_function_final_extra_acl_contracts) >/dev/null || return 1
}

mutate_contract_literal() {
  local source_file="$1" output_file="$2" declaration="$3"
  local occurrence="${4:-1}"
  awk -v declaration="$declaration" -v occurrence="$occurrence" '
    index($0, declaration " text[] := ARRAY[") {
      seen++
      if (seen == occurrence) {inside=1}
    }
    inside && !mutated && /^[[:space:]]*'\''/ {
      sub(/'\'',?$/, "_hostile_literal_drift&")
      mutated=1
    }
    {print}
  ' "$source_file" >"$output_file"
}

# Preflight and defense-in-depth audit execute independently, but every
# duplicated declaration family is one exact contract.  Prove the validator
# rejects an omitted-family mutation as well as checking the production bytes.
assert_duplicate_contracts "$migration"
for mutation in \
  protected_functions \
  protected_function_signatures \
  protected_function_contracts \
  protected_function_return_contracts \
  protected_function_identity_argument_contracts \
  protected_function_final_extra_acl_contracts
do
  mutant="$cluster_root/${mutation}.sql"
  mutate_contract_literal "$migration" "$mutant" "$mutation"
  if assert_duplicate_contracts "$mutant"; then
    echo "${mutation}: duplicated contract drift escaped validation" >&2
    exit 1
  fi
done

[[ "$(rg -c -F \
  'SELECT acl.grantee, acl.privilege_type, acl.is_grantable' "$migration")" == 2 ]]
[[ "$(rg -c -F \
  "SELECT function_row.proowner, 'EXECUTE'::text, false" "$migration")" == 2 ]]
printf 'H1_PROTECTED_FUNCTION_PREFLIGHT duplicate_contract_families=ALL mutation_detection=PASS acl_tuple_components=PASS\n'

initdb -D "$cluster_data" -U campaign_manager_login \
  --auth=trust --no-instructions >/dev/null
pg_ctl -D "$cluster_data" -o "-F -h '' -k $cluster_socket" \
  -w start >/dev/null
target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)
createdb "${target[@]}" schema054

psql "${target[@]}" -q -v ON_ERROR_STOP=1 schema054 <<'SQL'
CREATE ROLE pqxx NOLOGIN;
CREATE ROLE vjp NOLOGIN;
CREATE ROLE campaign_operations_owner NOLOGIN;
CREATE ROLE campaign_operations_campaign_creator NOLOGIN;
CREATE ROLE campaign_operations_authorizer NOLOGIN;
CREATE ROLE campaign_operations_auditor NOLOGIN;
CREATE ROLE campaign_operations_reader NOLOGIN;
CREATE ROLE campaign_operations_budget_administrator NOLOGIN;
CREATE ROLE campaign_operations_request_acceptor NOLOGIN;
CREATE ROLE campaign_operations_dispatcher NOLOGIN;
CREATE ROLE campaign_operations_phase5_transactional NOLOGIN;
SQL

pg_restore --schema-only --no-privileges --file=- \
  "$repo_root/Database/backups/LSTM_latest.dump" |
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 schema054
for prerequisite_migration in \
  050_experiment_current_operation_canonicalization.sql \
  051_scheduler_ownership_and_worker_attempts.sql \
  052_scheduler_protocol_and_exact_attempt_hardening.sql \
  053_campaign_operations_controls_cancellation_reconciliation.sql \
  054_campaign_operations_completion_and_audit.sql
do
  psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 schema054 \
    -f "$repo_root/Database/migrations/$prerequisite_migration"
done

function_identity=
catalog_snapshot() {
  local database="$1"
  psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 "$database" -v identity="$function_identity" <<'SQL'
SELECT pg_catalog.jsonb_build_object(
  'oid', function_row.oid,
  'identity', pg_catalog.format('%I.%s', namespace.nspname,
    function_row.oid::pg_catalog.regprocedure::text),
  'owner', pg_catalog.pg_get_userbyid(function_row.proowner),
  'kind', function_row.prokind,
  'language', language.lanname,
  'security_definer', function_row.prosecdef,
  'volatility', function_row.provolatile,
  'parallel', function_row.proparallel,
  'leakproof', function_row.proleakproof,
  'defaults', function_row.pronargdefaults,
  'variadic', function_row.provariadic,
  'return_type', function_row.prorettype,
  'returns_set', function_row.proretset,
  'all_argument_types', function_row.proallargtypes,
  'argument_modes', function_row.proargmodes,
  'argument_names', function_row.proargnames,
  'config', function_row.proconfig,
  'acl_origin', CASE WHEN function_row.proacl IS NULL THEN 'null' ELSE 'explicit' END,
  'expanded_acl', (SELECT pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_array(
        CASE acl.grantee WHEN 0 THEN 'PUBLIC'
          ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
        acl.privilege_type, acl.is_grantable)
      ORDER BY CASE acl.grantee WHEN 0 THEN 'PUBLIC'
        ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
        acl.privilege_type, acl.is_grantable)
    FROM pg_catalog.aclexplode(coalesce(function_row.proacl,
      pg_catalog.acldefault('f', function_row.proowner))) acl),
  'definition', CASE WHEN function_row.prokind IN ('f', 'p')
    THEN pg_catalog.pg_get_functiondef(function_row.oid) ELSE NULL END
)::text
FROM pg_catalog.pg_proc function_row
JOIN pg_catalog.pg_namespace namespace
  ON namespace.oid = function_row.pronamespace
JOIN pg_catalog.pg_language language ON language.oid = function_row.prolang
WHERE pg_catalog.format('%I.%s', namespace.nspname,
        function_row.oid::pg_catalog.regprocedure::text) = :'identity';
SQL
}

run_negative_fixture() {
  local label="$1" setup_sql="$2" precondition_sql="$3"
  local expected_sqlstate="$4" expected_code="$5" expected_message="$6"
  local identity="$7" cleanup_sql="${8:-}"
  local diagnostic_identity="${9-$identity}"
  local database="h1_preflight_${label}_${$}"
  local fixture_log="$cluster_root/${label}.log"
  createdb "${target[@]}" -T schema054 "$database"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c "$setup_sql"
  if [[ "$(psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 \
      "$database" -c "$precondition_sql")" != t ]]; then
    echo "${label}: fixture precondition did not establish its sole intended mismatch" >&2
    exit 1
  fi

  function_identity="$identity"
  local before after
  before="$(catalog_snapshot "$database")"
  [[ -n "$before" ]] || {
    echo "${label}: protected object snapshot is empty" >&2
    exit 1
  }

  if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
      "$database" -f "$migration" >"$fixture_log" 2>&1; then
    echo "${label}: migration 055 unexpectedly normalized the incompatible object" >&2
    exit 1
  fi
  rg -Fq "ERROR:  ${expected_sqlstate}: ${expected_code} ${expected_message}" \
    "$fixture_log" || {
      tail -n 30 "$fixture_log" >&2
      echo "${label}: wrong SQLSTATE or H1A diagnostic" >&2
      exit 1
    }
  if [[ -n "$diagnostic_identity" ]] &&
     ! rg -Fq "$diagnostic_identity" "$fixture_log"; then
    tail -n 30 "$fixture_log" >&2
    echo "${label}: diagnostic omitted the exact protected signature" >&2
    exit 1
  fi
  rg -Fq 'PL/pgSQL function inline_code_block' "$fixture_log" || {
    echo "${label}: failure did not originate in migration 055 preflight" >&2
    exit 1
  }
  if rg -q 'cannot change return type|is not a function|is not a procedure' \
      "$fixture_log"; then
    echo "${label}: raw PostgreSQL DDL classification escaped preflight" >&2
    exit 1
  fi

  after="$(catalog_snapshot "$database")"
  [[ "$before" == "$after" ]] || {
    echo "${label}: rejected protected object was normalized despite rollback" >&2
    exit 1
  }
  [[ "$(psql "${target[@]}" -X -At "$database" -c \
    "SELECT pg_catalog.to_regclass(
       'public.campaign_operations_production_transition_context') IS NULL")" == t ]] || {
    echo "${label}: migration mutation preceded protected-function rejection" >&2
    exit 1
  }

  dropdb "${target[@]}" "$database"
  if [[ -n "$cleanup_sql" ]]; then
    psql "${target[@]}" -q -v ON_ERROR_STOP=1 postgres -c "$cleanup_sql"
  fi
  printf 'H1_PROTECTED_FUNCTION_PREFLIGHT fixture=%s sqlstate=%s diagnostic=%s object=%s rollback=PASS\n' \
    "$label" "$expected_sqlstate" "$expected_code" "$identity"
}

# The pristine, exact schema-054 phase remains the sole legacy normalization
# path.  Its result must also pass the final contract and direct SQL replay.
legacy_database="h1_preflight_legacy_${$}"
createdb "${target[@]}" -T schema054 "$legacy_database"
psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$legacy_database" \
  -f "$migration" >"$cluster_root/legacy-install.log" 2>&1
psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$legacy_database" \
  -f "$migration" >"$cluster_root/legacy-replay.log" 2>&1
[[ "$(psql "${target[@]}" -X -At "$legacy_database" -c \
  'SELECT campaign_operations_h1_deployment_audit_v1(NULL,false,false)')" == t ]]
printf 'H1_PROTECTED_FUNCTION_PREFLIGHT fixture=schema054_exact result=PASS replay=PASS audit=PASS\n'

transition_identity='public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)'

run_negative_fixture return_type '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
-- Recreate with the exact frozen input names; return identity is the only
-- catalog mismatch.
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
RETURNS text LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN $value$incompatible$value$; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT function_row.prokind='f' AND function_row.prorettype='text'::regtype
 AND function_row.proallargtypes IS NULL AND function_row.proargmodes IS NULL
 AND function_row.proacl IS NULL
 FROM pg_proc function_row
 WHERE function_row.oid='transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight return contract mismatch:' \
"$transition_identity"

run_negative_fixture object_kind '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE PROCEDURE transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN NULL; END $body$;
ALTER PROCEDURE transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT function_row.prokind='p' AND language.lanname='plpgsql'
 AND function_row.prosecdef AND function_row.provolatile='v'
 AND function_row.proparallel='u' AND NOT function_row.proleakproof
 AND function_row.pronargdefaults=0 AND function_row.provariadic=0
 AND function_row.proconfig=ARRAY['search_path=pg_catalog, public, pg_temp']::text[]
 FROM pg_proc function_row JOIN pg_language language ON language.oid=function_row.prolang
 WHERE function_row.proname='transition_campaign_operations_request_dispatching'
 AND pg_get_function_identity_arguments(function_row.oid)=
     'IN target_request_id bigint, IN expected_version_value integer, IN lease_digest_value text, IN dispatcher_value text'" \
55000 H1A008 'protected function preflight catalog mismatch:' \
"$transition_identity"

run_negative_fixture aggregate_kind '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION h1_hostile_aggregate_state(
  state_value bigint, target_request_id bigint,
  expected_version_value integer, lease_digest_value text,
  dispatcher_value text)
RETURNS bigint LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $body$ SELECT state_value $body$;
CREATE AGGREGATE transition_campaign_operations_request_dispatching(
  bigint, integer, text, text) (
  SFUNC=h1_hostile_aggregate_state, STYPE=bigint, INITCOND='0');
ALTER AGGREGATE transition_campaign_operations_request_dispatching(
  bigint,integer,text,text) OWNER TO campaign_operations_owner' \
"SELECT function_row.prokind='a' AND function_row.proacl IS NULL
 FROM pg_proc function_row
 WHERE function_row.oid=
   'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
"$transition_identity"

run_negative_fixture alternate_schema '
CREATE SCHEMA h1_hostile_alternate;
CREATE FUNCTION h1_hostile_alternate.transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN NULL; END $body$' \
"SELECT count(*)=1 FROM pg_proc function_row
 JOIN pg_namespace namespace ON namespace.oid=function_row.pronamespace
 WHERE namespace.nspname='h1_hostile_alternate'
 AND function_row.proname='transition_campaign_operations_request_dispatching'" \
42501 H1A005 'protected function alternate schema or overload' \
'h1_hostile_alternate.h1_hostile_alternate.transition_campaign_operations_request_dispatching(bigint,integer,text,text)' \
'' ''

run_negative_fixture overload '
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text, hostile_extra_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN NULL; END $body$' \
"SELECT count(*)=2 FROM pg_proc function_row
 JOIN pg_namespace namespace ON namespace.oid=function_row.pronamespace
 WHERE namespace.nspname='public'
 AND function_row.proname='transition_campaign_operations_request_dispatching'" \
42501 H1A005 'protected function alternate schema or overload' \
'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text,text)' \
'' ''

run_negative_fixture input_signature '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id integer, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN NULL; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(integer,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT count(*)=1 FROM pg_proc function_row
 JOIN pg_namespace namespace ON namespace.oid=function_row.pronamespace
 WHERE namespace.nspname='public'
 AND function_row.proname='transition_campaign_operations_request_dispatching'
 AND function_row.oid=
   'transition_campaign_operations_request_dispatching(integer,integer,text,text)'::regprocedure" \
42501 H1A005 'protected function signature mismatch' \
'public.transition_campaign_operations_request_dispatching(integer,integer,text,text)' \
'' ''

run_negative_fixture input_identity '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  hostile_target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN NULL; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT pg_get_function_identity_arguments(function_row.oid) LIKE
   'hostile_target_request_id bigint,%'
 FROM pg_proc function_row
 WHERE function_row.oid=
   'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight return contract mismatch:' \
"$transition_identity"

run_negative_fixture volatility '
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text) STABLE' \
"SELECT function_row.provolatile='s'
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture parallel_safety '
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text) PARALLEL SAFE' \
"SELECT function_row.proparallel='s'
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture security_invoker '
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text) SECURITY INVOKER' \
"SELECT NOT function_row.prosecdef
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture security_definer '
ALTER FUNCTION enforce_campaign_operations_complete_binding() SECURITY DEFINER' \
"SELECT function_row.prosecdef
 FROM pg_proc function_row
 WHERE function_row.oid='enforce_campaign_operations_complete_binding()'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.enforce_campaign_operations_complete_binding()'

run_negative_fixture leakproof '
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text) LEAKPROOF' \
"SELECT function_row.proleakproof AND function_row.prorettype='text'::regtype
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

# These are independently frozen pg_proc fields.  Each hostile setup changes
# the live catalog object and must reach its explicit protected-function
# preflight branch rather than an unrelated creation-time PostgreSQL failure.
run_negative_fixture language '
DROP FUNCTION campaign_operations_tagged_fnv1a64(text);
CREATE FUNCTION campaign_operations_tagged_fnv1a64(value text)
RETURNS text LANGUAGE sql IMMUTABLE STRICT PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ SELECT value $body$;
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
OWNER TO campaign_operations_owner' \
"SELECT language.lanname='sql' AND function_row.provolatile='i'
 AND function_row.proparallel='u' AND function_row.prosecdef
 AND NOT function_row.proleakproof AND function_row.pronargdefaults=0
 AND function_row.provariadic=0 AND function_row.proconfig=
 ARRAY['search_path=pg_catalog, public, pg_temp']::text[]
 FROM pg_proc function_row JOIN pg_language language
 ON language.oid=function_row.prolang
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture variadic "
-- PostgreSQL cannot ALTER an existing signature into VARIADIC.  In this
-- disposable superuser fixture mutate the frozen pg_proc field itself while
-- retaining the exact protected identity, so preflight reaches its explicit
-- variadic rejection instead of the unrelated overload branch.
UPDATE pg_proc SET provariadic='text'::regtype::oid
 WHERE oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
"SELECT function_row.provariadic='text'::regtype::oid
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
42501 H1A005 'protected function default or variadic mismatch' \
'public.campaign_operations_tagged_fnv1a64(text)' \
'' ''

run_negative_fixture configuration_search_path '
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
SET search_path=pg_catalog,pg_temp' \
"SELECT function_row.proconfig=
 ARRAY['search_path=pg_catalog, pg_temp']::text[]
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
55000 H1A008 'protected function preflight catalog mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture explicit_grantee '
CREATE ROLE h1_unexpected_function_grantee NOLOGIN;
GRANT EXECUTE ON FUNCTION campaign_operations_tagged_fnv1a64(text)
TO h1_unexpected_function_grantee' \
"SELECT count(*)=1 FROM pg_proc function_row,
 LATERAL aclexplode(function_row.proacl) acl
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure
 AND pg_get_userbyid(acl.grantee)='h1_unexpected_function_grantee'
 AND acl.privilege_type='EXECUTE' AND NOT acl.is_grantable" \
42501 H1A006 'protected function preflight ACL mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)' \
'DROP ROLE h1_unexpected_function_grantee'

run_negative_fixture public_execute '
GRANT EXECUTE ON FUNCTION campaign_operations_tagged_fnv1a64(text) TO PUBLIC' \
"SELECT count(*)=1 FROM pg_proc function_row,
 LATERAL aclexplode(function_row.proacl) acl
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure
 AND acl.grantee=0 AND acl.privilege_type='EXECUTE' AND NOT acl.is_grantable" \
42501 H1A006 'protected function preflight ACL mismatch:' \
'public.campaign_operations_tagged_fnv1a64(text)'

run_negative_fixture grant_option '
GRANT EXECUTE ON FUNCTION lock_campaign_operations_authorization_head(bigint,text)
TO campaign_operations_completion_writer WITH GRANT OPTION' \
"SELECT count(*)=1 FROM pg_proc function_row,
 LATERAL aclexplode(function_row.proacl) acl
 WHERE function_row.oid='lock_campaign_operations_authorization_head(bigint,text)'::regprocedure
 AND pg_get_userbyid(acl.grantee)='campaign_operations_completion_writer'
 AND acl.privilege_type='EXECUTE' AND acl.is_grantable" \
42501 H1A006 'protected function preflight ACL mismatch:' \
'public.lock_campaign_operations_authorization_head(bigint,text)'

run_negative_fixture missing_explicit_grant '
REVOKE EXECUTE ON FUNCTION lock_campaign_operations_authorization_head(bigint,text)
FROM campaign_operations_completion_writer' \
"SELECT count(*)=0 FROM pg_proc function_row,
 LATERAL aclexplode(function_row.proacl) acl
 WHERE function_row.oid='lock_campaign_operations_authorization_head(bigint,text)'::regprocedure
 AND pg_get_userbyid(acl.grantee)='campaign_operations_completion_writer'
 AND acl.privilege_type='EXECUTE' AND NOT acl.is_grantable" \
42501 H1A006 'protected function preflight ACL mismatch:' \
'public.lock_campaign_operations_authorization_head(bigint,text)'

run_negative_fixture explicit_acl_origin '
REVOKE EXECUTE ON FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
FROM PUBLIC;
GRANT EXECUTE ON FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
TO PUBLIC' \
"SELECT function_row.proacl IS NOT NULL AND NOT EXISTS (
   (SELECT acl.grantee, acl.privilege_type, acl.is_grantable
    FROM aclexplode(function_row.proacl) acl
    EXCEPT
    SELECT acl.grantee, acl.privilege_type, acl.is_grantable
    FROM aclexplode(acldefault('f', function_row.proowner)) acl)
   UNION ALL
   (SELECT acl.grantee, acl.privilege_type, acl.is_grantable
    FROM aclexplode(acldefault('f', function_row.proowner)) acl
    EXCEPT
    SELECT acl.grantee, acl.privilege_type, acl.is_grantable
    FROM aclexplode(function_row.proacl) acl))
 FROM pg_proc function_row
 WHERE function_row.oid=
   'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
42501 H1A006 'protected function preflight ACL mismatch:' \
"$transition_identity"

run_negative_fixture out_argument '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text,
  OUT dispatch_result campaign_operations_operational_request)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN dispatch_result := NULL; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT function_row.prorettype='campaign_operations_operational_request'::regtype
 AND function_row.proallargtypes IS NOT NULL
 AND array_to_string(function_row.proargmodes,',')='i,i,i,i,o'
 AND function_row.proargnames[5]='dispatch_result'
 AND function_row.proacl IS NULL
 FROM pg_proc function_row
 WHERE function_row.oid='transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight return contract mismatch:' \
"$transition_identity"

run_negative_fixture inout_argument '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, INOUT dispatcher_value text)
RETURNS text
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN dispatcher_value := dispatcher_value; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT function_row.proallargtypes IS NOT NULL
 AND array_to_string(function_row.proargmodes,',')='i,i,i,b'
 AND function_row.proargnames[4]='dispatcher_value'
 AND function_row.proacl IS NULL
 FROM pg_proc function_row
 WHERE function_row.oid=
   'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight return contract mismatch:' \
"$transition_identity"

run_negative_fixture table_return '
DROP FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text);
CREATE FUNCTION transition_campaign_operations_request_dispatching(
  target_request_id bigint, expected_version_value integer,
  lease_digest_value text, dispatcher_value text)
RETURNS TABLE(dispatch_result campaign_operations_operational_request)
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp
AS $body$ BEGIN RETURN; END $body$;
ALTER FUNCTION transition_campaign_operations_request_dispatching(bigint,integer,text,text)
OWNER TO campaign_operations_owner' \
"SELECT function_row.proretset
 AND function_row.proallargtypes IS NOT NULL
 AND array_to_string(function_row.proargmodes,',')='i,i,i,i,t'
 AND function_row.proargnames[5]='dispatch_result'
 AND function_row.proacl IS NULL
 FROM pg_proc function_row
 WHERE function_row.oid=
   'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure" \
55000 H1A008 'protected function preflight return contract mismatch:' \
"$transition_identity"

run_negative_fixture wrong_owner '
CREATE ROLE h1_wrong_function_owner NOLOGIN;
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
OWNER TO h1_wrong_function_owner' \
"SELECT pg_get_userbyid(function_row.proowner)='h1_wrong_function_owner'
 FROM pg_proc function_row
 WHERE function_row.oid='campaign_operations_tagged_fnv1a64(text)'::regprocedure" \
42501 H1A004 'incompatible pre-existing protected function owner:' \
'public.campaign_operations_tagged_fnv1a64(text)' \
'DROP ROLE h1_wrong_function_owner'

# Exercise the defense-in-depth audit independently of migration replay.
audit_missing_grant_database="h1_audit_missing_grant_${$}"
audit_missing_grant_log="$cluster_root/audit-missing-grant.log"
createdb "${target[@]}" -T "$legacy_database" "$audit_missing_grant_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$audit_missing_grant_database" \
  -c 'REVOKE EXECUTE ON FUNCTION
      lock_campaign_operations_authorization_head(bigint,text)
      FROM campaign_operations_completion_writer'
if psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$audit_missing_grant_database" -c \
    'SELECT campaign_operations_h1_deployment_audit_v1(NULL,false,false)' \
    >"$audit_missing_grant_log" 2>&1; then
  echo 'audit_missing_explicit_grant: deployment audit unexpectedly passed' >&2
  exit 1
fi
rg -Fq \
  'ERROR:  42501: H1A006 exact protected function ACL mismatch object=public.lock_campaign_operations_authorization_head(bigint,text) stage=database-audit' \
  "$audit_missing_grant_log"
dropdb "${target[@]}" "$audit_missing_grant_database"
printf 'H1_PROTECTED_FUNCTION_PREFLIGHT fixture=audit_missing_explicit_grant sqlstate=42501 diagnostic=H1A006 audit=PASS\n'

# A replay must not recreate an H1 role whose grants were removed.  Restore the
# exact role and its two grants afterward so the independent legacy-role probe
# below starts from a pristine installed contract.
missing_recreated_role_database="h1_missing_recreated_role_${$}"
missing_recreated_role_log="$cluster_root/missing-recreated-role.log"
createdb "${target[@]}" -T "$legacy_database" \
  "$missing_recreated_role_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$legacy_database" \
  -c 'DROP OWNED BY campaign_operations_scheduler_protocol_evidence_reader'
psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
  "$missing_recreated_role_database" <<'SQL'
DROP OWNED BY campaign_operations_scheduler_protocol_evidence_reader;
DROP ROLE campaign_operations_scheduler_protocol_evidence_reader;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$missing_recreated_role_database" -f "$migration" \
    >"$missing_recreated_role_log" 2>&1; then
  echo 'missing_recreated_role: migration replay unexpectedly passed' >&2
  exit 1
fi
rg -Fq \
  'ERROR:  42501: H1A006 missing expected protected function ACL grantee role: role=campaign_operations_scheduler_protocol_evidence_reader object=public.campaign_operations_scheduler_protocol_evidence_lock_v1() stage=preflight' \
  "$missing_recreated_role_log"
[[ "$(psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 \
  "$missing_recreated_role_database" -c \
  "SELECT NOT EXISTS (SELECT 1 FROM pg_roles
     WHERE rolname='campaign_operations_scheduler_protocol_evidence_reader')")" == t ]]
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$legacy_database" <<'SQL'
CREATE ROLE campaign_operations_scheduler_protocol_evidence_reader
  NOLOGIN NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
  NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL;
GRANT EXECUTE ON FUNCTION
  campaign_operations_scheduler_protocol_evidence_snapshot_v1(),
  campaign_operations_scheduler_protocol_evidence_lock_v1()
TO campaign_operations_scheduler_protocol_evidence_reader;
SQL
psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
  "$missing_recreated_role_database" \
  -c 'GRANT EXECUTE ON FUNCTION
      campaign_operations_scheduler_protocol_evidence_snapshot_v1(),
      campaign_operations_scheduler_protocol_evidence_lock_v1()
      TO campaign_operations_scheduler_protocol_evidence_reader'
[[ "$(psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 \
  "$legacy_database" -c \
  'SELECT campaign_operations_h1_deployment_audit_v1(NULL,false,false)')" == t ]]
dropdb "${target[@]}" "$missing_recreated_role_database"
printf 'H1_PROTECTED_FUNCTION_PREFLIGHT fixture=missing_recreated_expected_role sqlstate=42501 diagnostic=H1A006 preflight=PASS role_recreation=BLOCKED\n'

# Roles are cluster-wide, so run the missing-role probe last after removing all
# other cloned databases that retain ACL dependencies on the expected grantee.
missing_role_database="h1_missing_role_${$}"
missing_role_audit_log="$cluster_root/missing-role-audit.log"
missing_role_replay_log="$cluster_root/missing-role-replay.log"
createdb "${target[@]}" -T "$legacy_database" "$missing_role_database"
dropdb "${target[@]}" "$legacy_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 schema054 \
  -c 'DROP OWNED BY campaign_operations_completion_writer'
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$missing_role_database" <<'SQL'
DROP OWNED BY campaign_operations_completion_writer;
DROP ROLE campaign_operations_completion_writer;
SQL
[[ "$(psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 \
  "$missing_role_database" -c \
  "SELECT NOT EXISTS (SELECT 1 FROM pg_roles
     WHERE rolname='campaign_operations_completion_writer')")" == t ]]

function_identity='public.lock_campaign_operations_authorization_head(bigint,text)'
missing_role_before="$(catalog_snapshot "$missing_role_database")"
[[ -n "$missing_role_before" ]]

if psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$missing_role_database" -c \
    'SELECT campaign_operations_h1_deployment_audit_v1(NULL,false,false)' \
    >"$missing_role_audit_log" 2>&1; then
  echo 'missing_expected_role: deployment audit unexpectedly passed' >&2
  exit 1
fi
rg -Fq \
  'ERROR:  42501: H1A006 missing expected protected function ACL grantee role: role=campaign_operations_completion_writer object=public.lock_campaign_operations_authorization_head(bigint,text) stage=database-audit' \
  "$missing_role_audit_log"

if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$missing_role_database" -f "$migration" \
    >"$missing_role_replay_log" 2>&1; then
  echo 'missing_expected_role: migration replay unexpectedly passed' >&2
  exit 1
fi
rg -Fq \
  'ERROR:  42501: H1A006 missing expected protected function ACL grantee role: role=campaign_operations_completion_writer object=public.lock_campaign_operations_authorization_head(bigint,text) stage=preflight' \
  "$missing_role_replay_log"
[[ "$missing_role_before" == "$(catalog_snapshot "$missing_role_database")" ]]
[[ "$(psql "${target[@]}" -X -At -v ON_ERROR_STOP=1 \
  "$missing_role_database" -c \
  "SELECT NOT EXISTS (SELECT 1 FROM pg_roles
     WHERE rolname='campaign_operations_completion_writer')")" == t ]]
printf 'H1_PROTECTED_FUNCTION_PREFLIGHT fixture=missing_expected_role sqlstate=42501 diagnostic=H1A006 preflight=PASS audit=PASS rollback=PASS\n'

echo 'Campaign Operations Phase H1 complete protected-function preflight tests passed'
