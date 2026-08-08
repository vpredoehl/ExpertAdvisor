-- Migration 055 catalog, canonical, immutable-evidence, deferred-consistency,
-- owner-DML, and privilege checks.  The disposable harness supplies request
-- 7 at state version 3 and exact scheduler generation-52 evidence.

DO $$
DECLARE role_name text;
BEGIN
    IF to_regclass('campaign_operations_production_enablement_event') IS NULL OR
       to_regclass(
         'campaign_operations_production_enablement_audit_reference_event')
           IS NULL OR
       to_regclass('campaign_operations_request_production_admission') IS NULL OR
       to_regclass('campaign_operations_production_readiness_v1') IS NULL OR
       to_regclass('campaign_operations_production_status_v1') IS NULL OR
       to_regprocedure(
         'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)') IS NULL OR
       to_regprocedure(
         'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)') IS NULL OR
       to_regprocedure(
         'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)') IS NULL THEN
        RAISE EXCEPTION 'migration 055 object inventory incomplete';
    END IF;
    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_owner',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles
                       WHERE rolname = role_name AND NOT rolcanlogin AND
                             NOT rolsuper AND NOT rolcreatedb AND
                             NOT rolcreaterole AND NOT rolreplication AND
                             NOT rolbypassrls) OR
           pg_has_role('pqxx', role_name, 'MEMBER') THEN
            RAISE EXCEPTION 'migration 055 role mismatch %', role_name;
        END IF;
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                   WHERE rolname =
                     'campaign_operations_h1_boundary_authority'
                     AND NOT rolcanlogin AND rolsuper AND NOT rolcreatedb
                     AND NOT rolcreaterole AND NOT rolreplication) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members membership
               JOIN pg_catalog.pg_roles granted
                 ON granted.oid = membership.roleid
               JOIN pg_catalog.pg_roles member_role
                 ON member_role.oid = membership.member
               WHERE granted.rolname LIKE 'campaign_operations_production_%'
                  OR granted.rolname LIKE
                       'campaign_operations_scheduler_protocol_evidence_%'
                  OR granted.rolname =
                       'campaign_operations_h1_boundary_authority'
                  OR member_role.rolname =
                       'campaign_operations_h1_boundary_authority') THEN
        RAISE EXCEPTION 'migration 055 sealed role mismatch';
    END IF;
    IF NOT EXISTS (
         SELECT 1
         FROM pg_catalog.pg_proc function_row
         JOIN pg_catalog.pg_namespace namespace_row
           ON namespace_row.oid = function_row.pronamespace
         WHERE namespace_row.nspname = 'public'
           AND function_row.proname::text =
             'transition_campaign_operations_request_dispatch_production_v2'
           AND octet_length(function_row.proname::text) = 61) OR
       (SELECT count(*) FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace_row
          ON namespace_row.oid = function_row.pronamespace
        WHERE namespace_row.nspname = 'public'
          AND function_row.proname::text =
            'transition_campaign_operations_request_dispatch_production_v2') <> 1 OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_proc function_row
               JOIN pg_catalog.pg_namespace namespace_row
                 ON namespace_row.oid = function_row.pronamespace
               WHERE namespace_row.nspname = 'public'
                 AND function_row.proname::text =
                   'transition_campaign_operations_request_dispatching_production_v')
    THEN
        RAISE EXCEPTION 'exact acquisition catalog name mismatch';
    END IF;
    IF has_table_privilege(
          'campaign_operations_production_enabler',
          'campaign_operations_production_enablement_event', 'INSERT') OR
       has_table_privilege(
          'campaign_operations_owner',
          'campaign_operations_production_transition_context', 'INSERT') OR
       has_table_privilege(
          'campaign_operations_production_dispatcher',
          'campaign_operations_production_transition_context', 'INSERT') OR
       has_table_privilege(
          'campaign_operations_production_disabler',
          'campaign_operations_production_enablement_event', 'INSERT') OR
       has_table_privilege(
          'campaign_operations_production_dispatcher',
          'campaign_operations_request_production_admission', 'INSERT') OR
       has_table_privilege(
          'campaign_operations_production_reader',
          'campaign_operations_production_enablement_event', 'UPDATE') OR
       NOT has_table_privilege(
          'campaign_operations_production_reader',
          'campaign_operations_production_readiness_v1', 'SELECT') OR
       has_table_privilege(
          'campaign_operations_scheduler_protocol_evidence_owner',
          'experiment_scheduler_protocol', 'UPDATE') OR
       has_column_privilege(
          'campaign_operations_scheduler_protocol_evidence_owner',
          'experiment_scheduler_protocol', 'required_generation', 'UPDATE') OR
       has_column_privilege(
          'campaign_operations_scheduler_protocol_evidence_reader',
          'experiment_scheduler_protocol', 'required_generation', 'UPDATE') OR
       has_column_privilege(
          'campaign_operations_scheduler_protocol_evidence_owner',
          'experiment_scheduler_protocol', 'updated_at', 'SELECT') OR
       has_column_privilege(
          'campaign_operations_scheduler_protocol_evidence_owner',
          'experiment_scheduler_protocol', 'required_generation', 'SELECT')
    THEN
        RAISE EXCEPTION 'migration 055 privilege isolation mismatch';
    END IF;
    IF EXISTS (
         SELECT 1
         FROM pg_proc function_row,
              aclexplode(coalesce(function_row.proacl, acldefault(
                'f', function_row.proowner))) acl
         WHERE function_row.oid =
           'campaign_operations_scheduler_protocol_evidence_snapshot_v1()'::
             regprocedure
           AND acl.grantee = 0 AND acl.privilege_type = 'EXECUTE') OR
       NOT has_function_privilege(
          'campaign_operations_scheduler_protocol_evidence_reader',
          'campaign_operations_scheduler_protocol_evidence_snapshot_v1()',
          'EXECUTE') OR
       (SELECT proconfig FROM pg_proc WHERE oid =
          'campaign_operations_scheduler_protocol_evidence_snapshot_v1()'::
             regprocedure)::text NOT LIKE '%pg_catalog, public%' THEN
        RAISE EXCEPTION 'scheduler protocol evidence function ACL mismatch';
    END IF;
    IF EXISTS (
         SELECT 1
         FROM unnest(ARRAY[
           'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
           'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
           'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)'::regprocedure
         ]) fixed_function(function_oid)
         JOIN pg_proc function_row ON function_row.oid = function_oid
         JOIN pg_roles owner_role ON owner_role.oid = function_row.proowner
         WHERE owner_role.rolname <>
                 'campaign_operations_h1_boundary_authority' OR
               NOT function_row.prosecdef OR
               function_row.pronargdefaults <> 0 OR
               function_row.provariadic <> 0 OR
               function_row.proconfig::text NOT LIKE
                 '%search_path=pg_catalog, public%') OR
       EXISTS (
         SELECT 1 FROM pg_proc function_row
         JOIN pg_roles owner_role ON owner_role.oid=function_row.proowner
         WHERE function_row.oid =
           'campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)'::
             regprocedure
           AND (owner_role.rolname <>
                  'campaign_operations_h1_boundary_authority' OR
                function_row.prosecdef OR
                function_row.provolatile <> 'v' OR
                function_row.proparallel <> 'u' OR
                function_row.pronargdefaults <> 0 OR
                function_row.provariadic <> 0 OR
                function_row.proconfig IS DISTINCT FROM
                  ARRAY['search_path=pg_catalog, public']::text[])) OR
       EXISTS (
         SELECT 1
         FROM unnest(ARRAY[
           'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
           'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
           'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)'::regprocedure
         ]) fixed_function(function_oid)
         JOIN pg_proc function_row ON function_row.oid = function_oid,
              aclexplode(coalesce(function_row.proacl,
                acldefault('f', function_row.proowner))) acl
         LEFT JOIN pg_roles grantee_role ON grantee_role.oid = acl.grantee
         WHERE acl.privilege_type = 'EXECUTE' AND
               (acl.grantee = 0 OR grantee_role.rolcanlogin OR
                grantee_role.rolname IN (
                  'pqxx', 'campaign_operations_production_enabler',
                  'campaign_operations_production_disabler',
                  'campaign_operations_production_dispatcher',
                  'campaign_operations_production_phase5_transactional',
                  'campaign_operations_production_reader',
                  'campaign_operations_scheduler_protocol_evidence_reader')))
    THEN
        RAISE EXCEPTION 'fixed transition owner/search-path/ACL mismatch';
    END IF;
    IF pg_get_functiondef(
         'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)'::regprocedure)
         NOT LIKE '%max(existing.attempt_ordinal)%' OR
       NOT EXISTS (
         SELECT 1 FROM pg_constraint
         WHERE conrelid = 'campaign_operations_dispatch_attempt'::regclass
           AND contype = 'u'
           AND pg_get_constraintdef(oid) =
             'UNIQUE (operational_request_id, attempt_ordinal)') THEN
        RAISE EXCEPTION 'V1/V2 shared ordinal authority mismatch';
    END IF;
    IF EXISTS (SELECT 1 FROM campaign_operations_production_transition_context)
       OR EXISTS (SELECT 1 FROM campaign_operations_production_enablement_event)
       OR EXISTS (SELECT 1 FROM campaign_operations_request_production_admission)
       OR EXISTS (SELECT 1 FROM campaign_operations_operational_request
                  WHERE production_dispatch_enabled) THEN
        RAISE EXCEPTION 'H1 migration did not remain default-off';
    END IF;
END $$;

SET SESSION AUTHORIZATION campaign_operations_h1_boundary_authority;
CREATE FUNCTION public.campaign_operations_h1_future_function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE public.campaign_operations_h1_future_procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE public.campaign_operations_h1_future_table_probe(value integer);
CREATE SEQUENCE public.campaign_operations_h1_future_sequence_probe;
CREATE TYPE public.campaign_operations_h1_future_type_probe AS ENUM ('one');
CREATE DOMAIN public.campaign_operations_h1_future_domain_probe AS text;
CREATE SCHEMA campaign_operations_h1_future_schema_probe;
CREATE FUNCTION campaign_operations_h1_future_schema_probe.function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE campaign_operations_h1_future_schema_probe.procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE campaign_operations_h1_future_schema_probe.table_probe(value integer);
CREATE SEQUENCE campaign_operations_h1_future_schema_probe.sequence_probe;
CREATE TYPE campaign_operations_h1_future_schema_probe.type_probe AS ENUM ('one');
CREATE DOMAIN campaign_operations_h1_future_schema_probe.domain_probe AS text;
RESET SESSION AUTHORIZATION;
GRANT CREATE ON SCHEMA public TO campaign_operations_owner,
    campaign_operations_scheduler_protocol_evidence_owner;
SELECT format('GRANT CREATE ON DATABASE %I TO campaign_operations_owner, campaign_operations_scheduler_protocol_evidence_owner',
              current_database()) \gexec
SET SESSION AUTHORIZATION campaign_operations_owner;
CREATE FUNCTION public.campaign_operations_owner_future_function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE public.campaign_operations_owner_future_procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE public.campaign_operations_owner_future_table_probe(value integer);
CREATE SEQUENCE public.campaign_operations_owner_future_sequence_probe;
CREATE TYPE public.campaign_operations_owner_future_type_probe AS ENUM ('one');
CREATE DOMAIN public.campaign_operations_owner_future_domain_probe AS text;
CREATE SCHEMA campaign_operations_owner_future_schema_probe;
CREATE FUNCTION campaign_operations_owner_future_schema_probe.function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE campaign_operations_owner_future_schema_probe.procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE campaign_operations_owner_future_schema_probe.table_probe(value integer);
CREATE SEQUENCE campaign_operations_owner_future_schema_probe.sequence_probe;
CREATE TYPE campaign_operations_owner_future_schema_probe.type_probe AS ENUM ('one');
CREATE DOMAIN campaign_operations_owner_future_schema_probe.domain_probe AS text;
RESET SESSION AUTHORIZATION;
SET SESSION AUTHORIZATION
    campaign_operations_scheduler_protocol_evidence_owner;
CREATE FUNCTION public.campaign_operations_scheduler_future_function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE public.campaign_operations_scheduler_future_procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE public.campaign_operations_scheduler_future_table_probe(
    value integer);
CREATE SEQUENCE public.campaign_operations_scheduler_future_sequence_probe;
CREATE TYPE public.campaign_operations_scheduler_future_type_probe
    AS ENUM ('one');
CREATE DOMAIN public.campaign_operations_scheduler_future_domain_probe AS text;
CREATE SCHEMA campaign_operations_scheduler_future_schema_probe;
CREATE FUNCTION campaign_operations_scheduler_future_schema_probe.function_probe()
RETURNS integer LANGUAGE sql AS 'SELECT 1';
CREATE PROCEDURE campaign_operations_scheduler_future_schema_probe.procedure_probe()
LANGUAGE sql AS 'SELECT 1';
CREATE TABLE campaign_operations_scheduler_future_schema_probe.table_probe(value integer);
CREATE SEQUENCE campaign_operations_scheduler_future_schema_probe.sequence_probe;
CREATE TYPE campaign_operations_scheduler_future_schema_probe.type_probe AS ENUM ('one');
CREATE DOMAIN campaign_operations_scheduler_future_schema_probe.domain_probe AS text;
RESET SESSION AUTHORIZATION;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_proc function_row,
         LATERAL pg_catalog.aclexplode(coalesce(function_row.proacl,
           pg_catalog.acldefault('f',function_row.proowner))) acl
         WHERE function_row.oid IN (
           'public.campaign_operations_h1_future_function_probe()'::regprocedure,
           'public.campaign_operations_h1_future_procedure_probe()'::regprocedure,
           'public.campaign_operations_owner_future_function_probe()'::regprocedure,
           'public.campaign_operations_owner_future_procedure_probe()'::regprocedure,
           'public.campaign_operations_scheduler_future_function_probe()'::regprocedure,
           'public.campaign_operations_scheduler_future_procedure_probe()'::regprocedure,
           'campaign_operations_h1_future_schema_probe.function_probe()'::regprocedure,
           'campaign_operations_h1_future_schema_probe.procedure_probe()'::regprocedure,
           'campaign_operations_owner_future_schema_probe.function_probe()'::regprocedure,
           'campaign_operations_owner_future_schema_probe.procedure_probe()'::regprocedure,
           'campaign_operations_scheduler_future_schema_probe.function_probe()'::regprocedure,
           'campaign_operations_scheduler_future_schema_probe.procedure_probe()'::regprocedure)
           AND acl.grantee=0 AND acl.privilege_type='EXECUTE') THEN
        RAISE EXCEPTION 'future function default ACL mismatch';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_class relation_row,
         LATERAL pg_catalog.aclexplode(coalesce(relation_row.relacl,
           pg_catalog.acldefault(CASE relation_row.relkind
             WHEN 'S' THEN 'S'::"char" ELSE 'r'::"char" END,
             relation_row.relowner))) acl
         WHERE relation_row.oid IN (
           'public.campaign_operations_h1_future_table_probe'::regclass,
           'public.campaign_operations_h1_future_sequence_probe'::regclass,
           'public.campaign_operations_owner_future_table_probe'::regclass,
           'public.campaign_operations_owner_future_sequence_probe'::regclass,
           'public.campaign_operations_scheduler_future_table_probe'::regclass,
           'public.campaign_operations_scheduler_future_sequence_probe'::regclass,
           'campaign_operations_h1_future_schema_probe.table_probe'::regclass,
           'campaign_operations_h1_future_schema_probe.sequence_probe'::regclass,
           'campaign_operations_owner_future_schema_probe.table_probe'::regclass,
           'campaign_operations_owner_future_schema_probe.sequence_probe'::regclass,
           'campaign_operations_scheduler_future_schema_probe.table_probe'::regclass,
           'campaign_operations_scheduler_future_schema_probe.sequence_probe'::regclass)
           AND acl.grantee=0) THEN
        RAISE EXCEPTION 'future relation default ACL mismatch';
    END IF;
    IF EXISTS (
         SELECT 1 FROM pg_catalog.pg_type type_row,
         LATERAL pg_catalog.aclexplode(coalesce(type_row.typacl,
           pg_catalog.acldefault('T', type_row.typowner))) acl
         WHERE type_row.oid IN (
           'public.campaign_operations_h1_future_type_probe'::regtype,
           'public.campaign_operations_h1_future_domain_probe'::regtype,
           'public.campaign_operations_owner_future_type_probe'::regtype,
           'public.campaign_operations_owner_future_domain_probe'::regtype,
           'public.campaign_operations_scheduler_future_type_probe'::regtype,
           'public.campaign_operations_scheduler_future_domain_probe'::regtype,
           'campaign_operations_h1_future_schema_probe.type_probe'::regtype,
           'campaign_operations_h1_future_schema_probe.domain_probe'::regtype,
           'campaign_operations_owner_future_schema_probe.type_probe'::regtype,
           'campaign_operations_owner_future_schema_probe.domain_probe'::regtype,
           'campaign_operations_scheduler_future_schema_probe.type_probe'::regtype,
           'campaign_operations_scheduler_future_schema_probe.domain_probe'::regtype)
           AND acl.grantee=0) OR EXISTS (
         SELECT 1 FROM pg_catalog.pg_namespace namespace_row,
         LATERAL pg_catalog.aclexplode(coalesce(namespace_row.nspacl,
           pg_catalog.acldefault('n', namespace_row.nspowner))) acl
         WHERE namespace_row.nspname = ANY (ARRAY[
             'campaign_operations_h1_future_schema_probe',
             'campaign_operations_owner_future_schema_probe',
             'campaign_operations_scheduler_future_schema_probe'])
           AND acl.grantee=0) THEN
        RAISE EXCEPTION 'future type/domain/schema default ACL mismatch';
    END IF;
    IF EXISTS (
         SELECT 1
         FROM pg_catalog.pg_proc function_row
         JOIN pg_catalog.pg_roles owner_role
           ON owner_role.oid = function_row.proowner
         WHERE function_row.oid IN (
           'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)'::regprocedure,
           'transition_campaign_operations_request_ready_recovered(bigint,integer)'::regprocedure)
           AND owner_role.rolname =
                 'campaign_operations_h1_boundary_authority') THEN
        RAISE EXCEPTION 'unrelated lifecycle/recovery function was sealed';
    END IF;
    IF EXISTS (
         SELECT 1
         FROM pg_catalog.pg_trigger trigger_row
         JOIN pg_catalog.pg_proc function_row
           ON function_row.oid = trigger_row.tgfoid
         JOIN pg_catalog.pg_roles owner_role
           ON owner_role.oid = function_row.proowner
         WHERE NOT trigger_row.tgisinternal
           AND trigger_row.tgrelid IN (
             'public.campaign_operations_operational_request'::regclass,
             'public.campaign_operations_dispatch_attempt'::regclass,
             'public.campaign_operations_dispatch_audit_reference_event'::regclass,
             'public.campaign_operations_completion_event'::regclass,
             'public.campaign_operations_completion_audit_reference_event'::regclass,
             'public.campaign_operations_production_transition_context'::regclass,
             'public.campaign_operations_production_enablement_event'::regclass,
             'public.campaign_operations_production_enablement_audit_reference_event'::regclass,
             'public.campaign_operations_request_production_admission'::regclass)
           AND (owner_role.rolname <>
                  'campaign_operations_h1_boundary_authority' OR
                function_row.proconfig::text NOT LIKE
                  '%search_path=pg_catalog, public%' OR
                EXISTS (
                  SELECT 1
                  FROM pg_catalog.aclexplode(coalesce(function_row.proacl,
                    pg_catalog.acldefault('f', function_row.proowner))) acl
                  WHERE acl.grantee = 0
                    AND acl.privilege_type = 'EXECUTE'))) THEN
        RAISE EXCEPTION 'protected trigger function owner/search-path/ACL mismatch';
    END IF;
END $$;
DROP FUNCTION public.campaign_operations_h1_future_function_probe();
DROP PROCEDURE public.campaign_operations_h1_future_procedure_probe();
DROP TABLE public.campaign_operations_h1_future_table_probe;
DROP SEQUENCE public.campaign_operations_h1_future_sequence_probe;
DROP TYPE public.campaign_operations_h1_future_type_probe;
DROP DOMAIN public.campaign_operations_h1_future_domain_probe;
DROP SCHEMA campaign_operations_h1_future_schema_probe CASCADE;
DROP FUNCTION public.campaign_operations_owner_future_function_probe();
DROP PROCEDURE public.campaign_operations_owner_future_procedure_probe();
DROP TABLE public.campaign_operations_owner_future_table_probe;
DROP SEQUENCE public.campaign_operations_owner_future_sequence_probe;
DROP TYPE public.campaign_operations_owner_future_type_probe;
DROP DOMAIN public.campaign_operations_owner_future_domain_probe;
DROP SCHEMA campaign_operations_owner_future_schema_probe CASCADE;
DROP FUNCTION public.campaign_operations_scheduler_future_function_probe();
DROP PROCEDURE public.campaign_operations_scheduler_future_procedure_probe();
DROP TABLE public.campaign_operations_scheduler_future_table_probe;
DROP SEQUENCE public.campaign_operations_scheduler_future_sequence_probe;
DROP TYPE public.campaign_operations_scheduler_future_type_probe;
DROP DOMAIN public.campaign_operations_scheduler_future_domain_probe;
DROP SCHEMA campaign_operations_scheduler_future_schema_probe CASCADE;
REVOKE CREATE ON SCHEMA public FROM campaign_operations_owner,
    campaign_operations_scheduler_protocol_evidence_owner;
SELECT format('REVOKE CREATE ON DATABASE %I FROM campaign_operations_owner, campaign_operations_scheduler_protocol_evidence_owner',
              current_database()) \gexec
SELECT campaign_operations_h1_deployment_audit_v1(NULL, false, false);

-- H1 status and migration 053's authoritative Phase F transition must agree on
-- the positive case and on each exclusion predicate.
CREATE FUNCTION pg_temp.verify_campaign_operations_h1_recovery_alignment()
RETURNS void LANGUAGE plpgsql AS $$
DECLARE attempt_id bigint;
BEGIN
    SELECT dispatch_attempt_id INTO STRICT attempt_id
    FROM campaign_operations_dispatch_attempt
    WHERE operational_request_id = 7 AND attempt_contract_version = 2;
    SET LOCAL session_replication_role = replica;
    UPDATE campaign_operations_operational_request
       SET lease_expires_at = transaction_timestamp() - interval '1 minute'
     WHERE operational_request_id = 7;
    SET LOCAL session_replication_role = origin;
    IF NOT (SELECT phase_f_recovery_eligible
            FROM campaign_operations_production_status_v1
            WHERE operational_request_id = 7) THEN
        RAISE EXCEPTION 'Phase F positive status case not eligible';
    END IF;
    BEGIN
        SET LOCAL session_replication_role = replica;
        PERFORM transition_campaign_operations_request_ready_recovered(7, 4);
        SET LOCAL session_replication_role = origin;
        RAISE EXCEPTION 'rollback positive recovery proof' USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;

    BEGIN
        SET LOCAL session_replication_role = replica;
        INSERT INTO campaign_operations_request_binding(
          request_binding_id,operational_request_id,request_identity_canonical,
          recommendation_campaign_materialization_id,
          materialization_identity_canonical,
          recommendation_campaign_materialization_member_id,member_ordinal,
          selected_member_identity_canonical,selected_member_identity_hash,
          recommendation_conversion_proposal_id,proposal_identity_canonical,
          proposal_identity_hash,recommendation_conversion_review_decision_id,
          recommendation_conversion_execution_id,execution_identity_canonical,
          execution_identity_hash,recommendation_conversion_activation_id,
          activation_identity_canonical,activation_identity_hash,experiment_id,
          binding_disposition,execution_disposition,activation_disposition,
          binding_contract_version,binding_identity_canonical,
          binding_identity_hash)
        VALUES(7001,7,'request-canonical-v1',7,'materialization-canonical-v1',
          7001,1,'member','fnv1a64:1000000000000001',7001,'proposal',
          'fnv1a64:1000000000000002',7001,7001,'execution',
          'fnv1a64:1000000000000003',7001,'activation',
          'fnv1a64:1000000000000004',7001,'created','created','created',1,
          'binding','fnv1a64:1000000000000005');
        SET LOCAL session_replication_role = origin;
        IF (SELECT phase_f_recovery_eligible
            FROM campaign_operations_production_status_v1
            WHERE operational_request_id = 7) THEN
            RAISE EXCEPTION 'request binding did not block H1 recovery status';
        END IF;
        BEGIN
            SET LOCAL session_replication_role = replica;
            PERFORM transition_campaign_operations_request_ready_recovered(7,4);
            RAISE EXCEPTION 'binding did not block Phase F recovery';
        EXCEPTION WHEN serialization_failure THEN NULL;
        END;
        RAISE EXCEPTION 'rollback binding predicate proof' USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;

    BEGIN
        SET LOCAL session_replication_role = replica;
        INSERT INTO experiment_recommendation_campaign_materialization_member(
          recommendation_campaign_materialization_member_id,
          recommendation_campaign_materialization_id,member_ordinal,
          recommendation_ranking_member_id,recommendation_id,
          source_experiment_id,ranking_position,
          selected_member_identity_canonical,selected_member_identity_hash,
          recommendation_conversion_proposal_id,proposal_identity_canonical,
          proposal_identity_hash)
        VALUES(7002,7,1,7002,7002,7002,1,'member',
          'fnv1a64:2000000000000001',7002,'proposal',
          'fnv1a64:2000000000000002');
        INSERT INTO experiment_recommendation_conversion_execution(
          recommendation_conversion_execution_id,
          recommendation_conversion_proposal_id,
          recommendation_conversion_review_decision_id,experiment_id,
          execution_contract_version,authorization_decision,
          execution_identity_canonical,execution_identity_hash)
        VALUES(7002,7002,7002,7002,1,'approve','execution',
          'fnv1a64:2000000000000003');
        SET LOCAL session_replication_role = origin;
        IF (SELECT phase_f_recovery_eligible
            FROM campaign_operations_production_status_v1
            WHERE operational_request_id = 7) THEN
            RAISE EXCEPTION 'execution did not block H1 recovery status';
        END IF;
        BEGIN
            SET LOCAL session_replication_role = replica;
            PERFORM transition_campaign_operations_request_ready_recovered(7,4);
            RAISE EXCEPTION 'execution did not block Phase F recovery';
        EXCEPTION WHEN serialization_failure THEN NULL;
        END;
        RAISE EXCEPTION 'rollback execution predicate proof' USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;

    BEGIN
        SET LOCAL session_replication_role = replica;
        INSERT INTO campaign_operations_dispatch_attempt_outcome(
          dispatch_attempt_outcome_id,dispatch_attempt_id,
          attempt_identity_canonical,result_classification,
          downstream_evidence_classification,
          semantic_conflict_classification,
          uncertain_commit_recovery_classification,diagnostic_code,
          request_binding_set_identity_canonical,
          request_binding_set_identity_hash,expected_request_version,
          resulting_request_version,expected_reservation_version,
          resulting_reservation_version,outcome_contract_version,
          outcome_identity_canonical,outcome_identity_hash)
        VALUES(7003,attempt_id,'attempt','rejected','no_phase5_evidence',
          'lease_unavailable','proven_no_commit','lease_unavailable',NULL,NULL,
          3,4,1,1,1,'outcome','fnv1a64:3000000000000001');
        SET LOCAL session_replication_role = origin;
        IF (SELECT phase_f_recovery_eligible
            FROM campaign_operations_production_status_v1
            WHERE operational_request_id = 7) THEN
            RAISE EXCEPTION 'latest outcome did not block H1 recovery status';
        END IF;
        BEGIN
            SET LOCAL session_replication_role = replica;
            PERFORM transition_campaign_operations_request_ready_recovered(7,4);
            RAISE EXCEPTION 'latest outcome did not block Phase F recovery';
        EXCEPTION WHEN serialization_failure THEN NULL;
        END;
        RAISE EXCEPTION 'rollback outcome predicate proof' USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;
END $$;

DO $$
BEGIN
    IF EXISTS (
         SELECT 1 FROM unnest(ARRAY['.x','_x',':x','/x','-x','',
                                    'A' || repeat('-', 128), 'Aé']) value
         WHERE value ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$') OR
       EXISTS (
         SELECT 1 FROM unnest(ARRAY['A','0._:/-z',
                                    'A' || repeat('-', 127)]) value
         WHERE value !~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$') THEN
        RAISE EXCEPTION 'PostgreSQL operation-key vector mismatch';
    END IF;
END $$;

CREATE TEMP TABLE campaign_operations_h1_nested_trigger_spoof(value integer);
CREATE FUNCTION pg_temp.campaign_operations_h1_nested_trigger_spoof()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE public.campaign_operations_operational_request
       SET production_dispatch_enabled = true
     WHERE operational_request_id = 7;
    RETURN NEW;
END $$;
CREATE TRIGGER campaign_operations_h1_nested_trigger_spoof
AFTER INSERT ON campaign_operations_h1_nested_trigger_spoof
FOR EACH ROW EXECUTE FUNCTION
    pg_temp.campaign_operations_h1_nested_trigger_spoof();
DO $$
DECLARE diagnostic text;
BEGIN
    BEGIN
        INSERT INTO campaign_operations_h1_nested_trigger_spoof VALUES(1);
        RAISE EXCEPTION 'nested-trigger witness spoof unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production admission witness is immutable' THEN
            RAISE;
        END IF;
    END;
END $$;

-- The former ordinary table owner is no longer an owner and has no direct DML.
-- This proves its actual failure point is the expected ACL boundary.
SET ROLE campaign_operations_owner;
DO $$
DECLARE diagnostic text;
BEGIN
    BEGIN
        UPDATE campaign_operations_operational_request
           SET production_dispatch_enabled = true
         WHERE operational_request_id = 7;
        RAISE EXCEPTION 'table-owner false-to-true mutation succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic NOT LIKE '%permission denied for table%operational_request%'
        THEN RAISE; END IF;
    END;
    BEGIN
        INSERT INTO campaign_operations_production_transition_context(
          backend_pid,transaction_id,transition_kind,operational_request_id,
          operation_key)
        VALUES(pg_backend_pid(),txid_current(),'dispatch_v2',7,'forged-001');
        RAISE EXCEPTION 'former owner forged transition context';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic NOT LIKE '%permission denied for table%transition_context%'
        THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

GRANT campaign_operations_dispatcher TO campaign_manager_login;
DO $$
DECLARE transitioned campaign_operations_operational_request%ROWTYPE;
DECLARE attempt_id bigint;
DECLARE attempt_canonical text;
DECLARE stored_canonical text;
DECLARE stored_hash text;
BEGIN
    IF NOT campaign_operations_isolated_v1_authority_valid() THEN
        RAISE EXCEPTION 'positive Attempt V1 isolated authority not recognized';
    END IF;
    BEGIN
        transitioned := transition_campaign_operations_request_dispatching(
          7, 3, 'fnv1a64:0123456789abcdef', 'isolated.test');
        IF transitioned.request_state <> 'dispatching' OR
           transitioned.state_version <> 4 OR
           transitioned.production_dispatch_enabled THEN
            RAISE EXCEPTION 'positive Attempt V1 transition mismatch';
        END IF;
        attempt_canonical :=
          'campaign_operations_dispatch_attempt_v1' ||
          ';request_id=7' ||
          ';request_identity_canonical=20:request-canonical-v1' ||
          ';attempt_ordinal=1;expected_request_version=3' ||
          ';resulting_request_version=4' ||
          ';lease_token_digest=24:fnv1a64:0123456789abcdef' ||
          ';lease_expires_at=27:' || to_char(
            transitioned.lease_expires_at AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') ||
          ';dispatcher_identity=13:isolated.test';
        INSERT INTO campaign_operations_dispatch_attempt(
          operational_request_id,request_identity_canonical,attempt_ordinal,
          expected_request_version,resulting_request_version,
          lease_token_digest,lease_expires_at,dispatcher_identity,
          attempt_contract_version,attempt_identity_canonical,
          attempt_identity_hash)
        VALUES(7,'request-canonical-v1',1,3,4,
          'fnv1a64:0123456789abcdef',transitioned.lease_expires_at,
          'isolated.test',1,attempt_canonical,
          campaign_operations_tagged_fnv1a64(attempt_canonical))
        RETURNING dispatch_attempt_id INTO attempt_id;
        INSERT INTO campaign_operations_dispatch_audit_reference_event(
          operational_campaign_id,operational_request_id,dispatch_attempt_id,
          dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,
          prior_version,resulting_version,outcome,replay_disposition,
          diagnostic_code)
        VALUES(7,7,attempt_id,NULL,'dispatch_lease_acquired','isolated.test',
          'campaign_operations_dispatcher',3,4,'recorded','new_operation',
          'dispatch_lease_acquired');
        SET CONSTRAINTS ALL IMMEDIATE;
        SELECT candidate.attempt_identity_canonical,
               candidate.attempt_identity_hash
          INTO STRICT stored_canonical, stored_hash
        FROM campaign_operations_dispatch_attempt candidate
        WHERE candidate.dispatch_attempt_id = attempt_id;
        IF stored_canonical <> attempt_canonical OR
           stored_hash <> campaign_operations_tagged_fnv1a64(
             attempt_canonical) OR
           position(attempt_canonical IN
             campaign_operations_completion_evidence_text(7,'request')) = 0
        THEN
            RAISE EXCEPTION 'historical Attempt V1 byte preservation mismatch';
        END IF;
        RAISE EXCEPTION 'rollback positive Attempt V1 proof'
          USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;
    IF (SELECT request_state <> 'ready' OR state_version <> 3 OR
               production_dispatch_enabled
        FROM campaign_operations_operational_request
        WHERE operational_request_id = 7) THEN
        RAISE EXCEPTION 'positive Attempt V1 rollback mismatch';
    END IF;
END $$;
REVOKE campaign_operations_dispatcher FROM campaign_manager_login;

GRANT campaign_operations_dispatcher TO campaign_manager_login;
GRANT campaign_operations_production_reader TO campaign_manager_login;
DO $$
BEGIN
    IF campaign_operations_isolated_v1_authority_valid() THEN
        RAISE EXCEPTION 'direct production role did not block Attempt V1';
    END IF;
END $$;
REVOKE campaign_operations_production_reader FROM campaign_manager_login;
CREATE ROLE campaign_operations_h1_inherited_production NOLOGIN;
GRANT campaign_operations_production_dispatcher
TO campaign_operations_h1_inherited_production;
GRANT campaign_operations_h1_inherited_production TO campaign_manager_login;
DO $$
BEGIN
    IF campaign_operations_isolated_v1_authority_valid() THEN
        RAISE EXCEPTION 'inherited production role did not block Attempt V1';
    END IF;
END $$;
REVOKE campaign_operations_h1_inherited_production FROM campaign_manager_login;
REVOKE campaign_operations_production_dispatcher
FROM campaign_operations_h1_inherited_production;
DROP ROLE campaign_operations_h1_inherited_production;
REVOKE campaign_operations_dispatcher FROM campaign_manager_login;

CREATE ROLE campaign_operations_h1_nested_role_test NOLOGIN;
GRANT campaign_operations_production_reader
TO campaign_operations_h1_nested_role_test;
DO $$
BEGIN
    IF NOT campaign_operations_has_explicit_role_v1(
         'campaign_operations_h1_nested_role_test'::name,
         'campaign_operations_production_reader'::name) OR
       campaign_operations_has_explicit_role_v1(
         'campaign_operations_h1_nested_role_test'::name,
         'campaign_operations_production_dispatcher'::name) THEN
        RAISE EXCEPTION 'nested explicit role reconstruction mismatch';
    END IF;
END $$;
REVOKE campaign_operations_production_reader
FROM campaign_operations_h1_nested_role_test;
DROP ROLE campaign_operations_h1_nested_role_test;

DO $$
DECLARE scheduler_canonical text;
DECLARE build_canonical text;
DECLARE event campaign_operations_production_enablement_event%ROWTYPE;
DECLARE admission campaign_operations_request_production_admission%ROWTYPE;
DECLARE attempt campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE audit campaign_operations_dispatch_audit_reference_event%ROWTYPE;
DECLARE event_audit
    campaign_operations_production_enablement_audit_reference_event%ROWTYPE;
DECLARE diagnostic text;
DECLARE original_event_id bigint;
DECLARE original_attempt_id bigint;
DECLARE lease_expiry timestamptz;
BEGIN
    IF campaign_operations_isolated_v1_authority_valid() OR
       campaign_operations_has_explicit_role_v1(
         session_user::name,
         'campaign_operations_production_reader'::name) OR
       pg_get_functiondef(
         'campaign_operations_isolated_v1_authority_valid()'::regprocedure)
         ~* '\\mLIKE\\M' OR
       pg_get_functiondef(
         'campaign_operations_isolated_v1_authority_valid()'::regprocedure)
         NOT LIKE '%convert_to(current_database(), ''UTF8'')%' THEN
        RAISE EXCEPTION 'Attempt V1 literal-prefix/role isolation mismatch';
    END IF;
    BEGIN
        PERFORM transition_campaign_operations_request_dispatching(
          7, 3, 'fnv1a64:0123456789abcdef', 'isolated.test');
        RAISE EXCEPTION 'Attempt V1 isolation unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'campaign operations Attempt V1 isolated authority required' THEN
            RAISE;
        END IF;
    END;
    BEGIN
        UPDATE campaign_operations_operational_request
           SET production_dispatch_enabled = true
         WHERE operational_request_id = 7;
        RAISE EXCEPTION 'direct false-to-true owner mutation succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production admission witness is immutable' THEN
            RAISE;
        END IF;
    END;
    scheduler_canonical :=
      campaign_operations_scheduler_protocol_evidence_canonical_v1(
        52, 'complete', '2026-07-31T12:34:56.123456Z'::timestamptz,
        'scheduler.owner', '/Applications/LSTM_Release',
        'cutover-process-evidence');
    IF scheduler_canonical <>
       'campaign_operations_scheduler_protocol_evidence_v1' ||
       ';required_generation=52;cutover_state=complete' ||
       ';migration_contract=61:migration-052-scheduler-generation-52-exact-attempt-authority' ||
       ';protocol_contract=50:scheduler-generation-52-exact-attempt-authority-v1' ||
       ';cutover_completed_at=27:2026-07-31T12:34:56.123456Z' ||
       ';cutover_completed_by=15:scheduler.owner' ||
       ';cutover_executable_path=26:/Applications/LSTM_Release' ||
       ';cutover_process_evidence=24:cutover-process-evidence' OR
       campaign_operations_tagged_fnv1a64(scheduler_canonical) <>
           'fnv1a64:af614995e691e378' THEN
        RAISE EXCEPTION 'scheduler golden vector mismatch';
    END IF;
    build_canonical := campaign_operations_manager_build_canonical_v1(
      'campaign-operations-production-dispatch-and-manager-run-once-v1',
      repeat('a', 40), 'Apple clang 21.0.0',
      'sha256:' || repeat('b', 64));
    IF campaign_operations_tagged_fnv1a64(build_canonical) <>
           'fnv1a64:9afc5f65cbaaece3' THEN
        RAISE EXCEPTION 'Manager build golden vector mismatch';
    END IF;

    event.production_enablement_event_id := nextval(pg_get_serial_sequence(
      'campaign_operations_production_enablement_event',
      'production_enablement_event_id'));
    event.event_kind := 'enable';
    event.operation_key := 'enable-001';
    event.predecessor_event_id := NULL;
    event.predecessor_event_canonical := '';
    event.expected_prior_version := 0;
    event.resulting_version := 1;
    event.scheduler_protocol_evidence_canonical := scheduler_canonical;
    event.scheduler_protocol_evidence_hash :=
      campaign_operations_tagged_fnv1a64(scheduler_canonical);
    event.scheduler_required_generation := 52;
    event.scheduler_cutover_state := 'complete';
    event.scheduler_cutover_completed_at :=
      '2026-07-31T12:34:56.123456Z'::timestamptz;
    event.scheduler_cutover_completed_by := 'scheduler.owner';
    event.scheduler_cutover_executable_path :=
      '/Applications/LSTM_Release';
    event.scheduler_cutover_process_evidence := 'cutover-process-evidence';
    event.independent_verification_reference :=
      'cee://phase-h/generation-52';
    event.actor_identity := 'operator@example.test';
    event.capability := 'campaign_operations_production_enabler';
    event.manager_service_contract :=
      'campaign-operations-production-dispatch-and-manager-run-once-v1';
    event.approved_build_contract_canonical := build_canonical;
    event.approved_build_contract_hash :=
      campaign_operations_tagged_fnv1a64(build_canonical);
    event.approved_build_source_commit := repeat('a', 40);
    event.approved_build_compiler_contract := 'Apple clang 21.0.0';
    event.approved_build_executable_sha256 :=
      'sha256:' || repeat('b', 64);
    event.reason := 'Enable verified H1 evidence.';
    event.enablement_contract_version := 1;
    event.recorded_at := transaction_timestamp();
    event.enablement_identity_canonical :=
      campaign_operations_production_enablement_canonical_v1(event);
    event.enablement_identity_hash := campaign_operations_tagged_fnv1a64(
      event.enablement_identity_canonical);
    IF octet_length(event.enablement_identity_canonical) <> 1399 OR
       event.enablement_identity_hash <> 'fnv1a64:05e8f676db9ba009' THEN
        RAISE EXCEPTION 'enable event golden vector mismatch';
    END IF;
    BEGIN
        INSERT INTO campaign_operations_production_enablement_event
          SELECT event.*;
        RAISE EXCEPTION 'direct enablement insertion unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'fixed production enablement transition context required' THEN
            RAISE;
        END IF;
    END;
    PERFORM setval(pg_get_serial_sequence(
      'campaign_operations_production_enablement_event',
      'production_enablement_event_id'), 1, false);
    SELECT * INTO STRICT event
    FROM record_campaign_operations_production_enable_v1(
      'enable-001', 0, scheduler_canonical,
      'cee://phase-h/generation-52', 'operator@example.test',
      'campaign-operations-production-dispatch-and-manager-run-once-v1',
      build_canonical, repeat('a', 40), 'Apple clang 21.0.0',
      'sha256:' || repeat('b', 64), 'Enable verified H1 evidence.');
    original_event_id := event.production_enablement_event_id;
    SELECT * INTO STRICT event
    FROM record_campaign_operations_production_enable_v1(
      'enable-001', 0, scheduler_canonical,
      'cee://phase-h/generation-52', 'operator@example.test',
      'campaign-operations-production-dispatch-and-manager-run-once-v1',
      build_canonical, repeat('a', 40), 'Apple clang 21.0.0',
      'sha256:' || repeat('b', 64), 'Enable verified H1 evidence.');
    IF event.production_enablement_event_id <> original_event_id THEN
        RAISE EXCEPTION 'production enable exact replay changed identity';
    END IF;
    BEGIN
        PERFORM record_campaign_operations_production_enable_v1(
          'enable-001', 0, scheduler_canonical,
          'cee://phase-h/generation-52', 'different@example.test',
          'campaign-operations-production-dispatch-and-manager-run-once-v1',
          build_canonical, repeat('a', 40), 'Apple clang 21.0.0',
          'sha256:' || repeat('b', 64), 'Enable verified H1 evidence.');
        RAISE EXCEPTION 'production enable conflicting replay succeeded';
    EXCEPTION WHEN unique_violation THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production enable conflicting replay' THEN
            RAISE;
        END IF;
    END;
    BEGIN
        SET LOCAL session_replication_role = replica;
        UPDATE campaign_operations_production_enablement_event
           SET enablement_identity_canonical =
                 enablement_identity_canonical || 'changed'
         WHERE operation_key='enable-001';
        SET LOCAL session_replication_role = origin;
        PERFORM record_campaign_operations_production_enable_v1(
          'enable-001', 0, scheduler_canonical,
          'cee://phase-h/generation-52', 'operator@example.test',
          'campaign-operations-production-dispatch-and-manager-run-once-v1',
          build_canonical, repeat('a', 40), 'Apple clang 21.0.0',
          'sha256:' || repeat('b', 64), 'Enable verified H1 evidence.');
        RAISE EXCEPTION
          'production enable same-hash/different-canonical succeeded';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production enable replay evidence corrupt' THEN
            RAISE;
        END IF;
    END;

    -- A caller rollback after the fixed acquisition returns must erase the
    -- Boolean, admission, Attempt V2, audit, and private transaction context.
    BEGIN
        PERFORM
          transition_campaign_operations_request_dispatch_production_v2(
            7, 3, 'fnv1a64:0123456789abcdef',
            transaction_timestamp() + interval '5 minutes',
            'rollback-dispatch-001', 'manager@example.test', build_canonical);
        RAISE EXCEPTION 'force production dispatch rollback'
          USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;
    IF (SELECT request_state <> 'ready' OR state_version <> 3 OR
               production_dispatch_enabled OR lease_token_hash IS NOT NULL OR
               lease_expires_at IS NOT NULL OR dispatcher_identity IS NOT NULL
        FROM campaign_operations_operational_request
        WHERE operational_request_id = 7) OR
       EXISTS (SELECT 1 FROM
          campaign_operations_request_production_admission) OR
       EXISTS (SELECT 1 FROM campaign_operations_dispatch_attempt
               WHERE attempt_contract_version = 2) OR
       EXISTS (SELECT 1 FROM campaign_operations_dispatch_audit_reference_event
               WHERE request_production_admission_id IS NOT NULL) OR
       EXISTS (SELECT 1 FROM
          campaign_operations_production_transition_context) THEN
        RAISE EXCEPTION 'production dispatch rollback stability mismatch';
    END IF;

    admission.request_production_admission_id := nextval(
      pg_get_serial_sequence(
        'campaign_operations_request_production_admission',
        'request_production_admission_id'));
    admission.operational_request_id := 7;
    admission.request_identity_canonical := 'request-canonical-v1';
    admission.expected_request_version := 3;
    admission.dispatch_operation_key := 'dispatch-001';
    admission.production_enablement_event_id :=
      event.production_enablement_event_id;
    admission.enable_event_canonical :=
      event.enablement_identity_canonical;
    admission.requesting_actor := 'manager@example.test';
    admission.original_executing_service_principal :=
      'campaign_manager_login';
    admission.approved_build_contract_canonical := build_canonical;
    admission.approved_build_contract_hash :=
      campaign_operations_tagged_fnv1a64(build_canonical);
    admission.capability := 'campaign_operations_production_dispatcher';
    admission.admission_contract_version := 1;
    admission.admitted_at := transaction_timestamp();
    admission.admission_identity_canonical :=
      campaign_operations_request_production_admission_canonical_v1(
        admission);
    admission.admission_identity_hash :=
      campaign_operations_tagged_fnv1a64(
        admission.admission_identity_canonical);
    IF octet_length(admission.admission_identity_canonical) <> 2244 OR
       admission.admission_identity_hash <> 'fnv1a64:674a534f93f43c48'
    THEN
        RAISE EXCEPTION 'request admission golden vector mismatch length=% hash=% event=%',
          octet_length(admission.admission_identity_canonical),
          admission.admission_identity_hash,
          event.production_enablement_event_id;
    END IF;
    BEGIN
        INSERT INTO campaign_operations_request_production_admission
          SELECT admission.*;
        RAISE EXCEPTION 'direct admission insertion unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'fixed production acquisition transition context required' THEN
            RAISE;
        END IF;
    END;

    attempt.dispatch_attempt_id := nextval(pg_get_serial_sequence(
      'campaign_operations_dispatch_attempt', 'dispatch_attempt_id'));
    attempt.operational_request_id := 7;
    attempt.request_identity_canonical := 'request-canonical-v1';
    attempt.attempt_ordinal := 1;
    attempt.expected_request_version := 3;
    attempt.resulting_request_version := 4;
    attempt.lease_token_digest := 'fnv1a64:0123456789abcdef';
    attempt.lease_expires_at :=
      '2026-07-31T12:39:56.123456Z'::timestamptz;
    attempt.dispatcher_identity := 'manager@example.test';
    attempt.attempt_contract_version := 2;
    attempt.acquired_at := transaction_timestamp();
    attempt.request_production_admission_id :=
      admission.request_production_admission_id;
    attempt.request_production_admission_canonical :=
      admission.admission_identity_canonical;
    attempt.request_production_admission_hash :=
      admission.admission_identity_hash;
    attempt.production_enablement_event_id :=
      event.production_enablement_event_id;
    attempt.production_enablement_event_canonical :=
      event.enablement_identity_canonical;
    attempt.production_enablement_event_hash :=
      event.enablement_identity_hash;
    attempt.operation_key := 'dispatch-001';
    attempt.requesting_actor := 'manager@example.test';
    attempt.original_executing_service_principal :=
      'campaign_manager_login';
    attempt.approved_build_contract_canonical := build_canonical;
    attempt.approved_build_contract_hash :=
      campaign_operations_tagged_fnv1a64(build_canonical);
    attempt.production_capability :=
      'campaign_operations_production_dispatcher';
    attempt.attempt_identity_canonical :=
      campaign_operations_dispatch_attempt_v2_canonical(attempt);
    attempt.attempt_identity_hash := campaign_operations_tagged_fnv1a64(
      attempt.attempt_identity_canonical);
    IF attempt.attempt_identity_hash <> 'fnv1a64:17a81d69b45dc93a'
    THEN
        RAISE EXCEPTION 'Attempt V2 golden vector mismatch';
    END IF;
    BEGIN
        INSERT INTO campaign_operations_dispatch_attempt SELECT attempt.*;
        RAISE EXCEPTION 'direct Attempt V2 insertion unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'fixed production acquisition transition context required' THEN
            RAISE;
        END IF;
    END;
    lease_expiry := transaction_timestamp() + interval '5 minutes';
    SELECT * INTO STRICT attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      7, 3, 'fnv1a64:0123456789abcdef',
      lease_expiry, 'dispatch-001',
      'manager@example.test', build_canonical);
    original_attempt_id := attempt.dispatch_attempt_id;
    SELECT * INTO STRICT attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      7, 3, 'fnv1a64:0123456789abcdef', lease_expiry, 'dispatch-001',
      'manager@example.test', build_canonical);
    IF attempt.dispatch_attempt_id <> original_attempt_id THEN
        RAISE EXCEPTION 'production acquisition exact replay changed identity';
    END IF;
    BEGIN
        PERFORM transition_campaign_operations_request_dispatch_production_v2(
          7, 3, 'fnv1a64:ffffffffffffffff', lease_expiry, 'dispatch-001',
          'manager@example.test', build_canonical);
        RAISE EXCEPTION 'production acquisition conflicting replay succeeded';
    EXCEPTION WHEN unique_violation THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production acquisition conflicting replay' THEN
            RAISE;
        END IF;
    END;
    BEGIN
        SET LOCAL session_replication_role = replica;
        UPDATE campaign_operations_request_production_admission
           SET admission_identity_canonical =
                 admission_identity_canonical || 'changed'
         WHERE operational_request_id=7;
        SET LOCAL session_replication_role = origin;
        PERFORM transition_campaign_operations_request_dispatch_production_v2(
          7, 3, 'fnv1a64:0123456789abcdef', lease_expiry, 'dispatch-001',
          'manager@example.test', build_canonical);
        RAISE EXCEPTION
          'production admission same-hash/different-canonical succeeded';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production acquisition replay evidence corrupt' THEN
            RAISE;
        END IF;
    END;
    BEGIN
        SET LOCAL session_replication_role = replica;
        UPDATE campaign_operations_dispatch_attempt
           SET attempt_identity_canonical =
                 attempt_identity_canonical || 'changed'
         WHERE dispatch_attempt_id=original_attempt_id;
        SET LOCAL session_replication_role = origin;
        PERFORM transition_campaign_operations_request_dispatch_production_v2(
          7, 3, 'fnv1a64:0123456789abcdef', lease_expiry, 'dispatch-001',
          'manager@example.test', build_canonical);
        RAISE EXCEPTION
          'Attempt V2 same-hash/different-canonical succeeded';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production acquisition replay evidence corrupt' THEN
            RAISE;
        END IF;
    END;
    SELECT * INTO STRICT admission
    FROM campaign_operations_request_production_admission
    WHERE operational_request_id = 7;
    SELECT * INTO STRICT audit
    FROM campaign_operations_dispatch_audit_reference_event
    WHERE dispatch_attempt_id = attempt.dispatch_attempt_id
      AND cause_kind = 'dispatch_lease_acquired';

    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    IF NOT (SELECT production_dispatch_enabled
            FROM campaign_operations_operational_request
            WHERE operational_request_id = 7) OR
       campaign_operations_production_enablement_canonical_v1(
         (SELECT candidate FROM
           campaign_operations_production_enablement_event candidate
          WHERE production_enablement_event_id =
                event.production_enablement_event_id)) <>
           event.enablement_identity_canonical OR
       campaign_operations_request_production_admission_canonical_v1(
         (SELECT candidate FROM
           campaign_operations_request_production_admission candidate
          WHERE request_production_admission_id =
                admission.request_production_admission_id)) <>
           admission.admission_identity_canonical OR
       campaign_operations_dispatch_attempt_v2_canonical(
         (SELECT candidate FROM campaign_operations_dispatch_attempt candidate
          WHERE dispatch_attempt_id = attempt.dispatch_attempt_id)) <>
           attempt.attempt_identity_canonical OR
       position(attempt.attempt_identity_canonical IN
         campaign_operations_completion_evidence_text(7, 'request')) = 0 THEN
        RAISE EXCEPTION 'PostgreSQL canonical reconstruction mismatch';
    END IF;

    BEGIN
        UPDATE campaign_operations_operational_request
           SET production_dispatch_enabled = false
         WHERE operational_request_id = 7;
        RAISE EXCEPTION 'true-to-false owner mutation unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'production admission witness is immutable' THEN
            RAISE;
        END IF;
    END;
    BEGIN
        UPDATE campaign_operations_production_enablement_event
           SET reason = reason || ' changed'
         WHERE production_enablement_event_id =
               event.production_enablement_event_id;
        RAISE EXCEPTION 'enablement owner mutation unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'Campaign Operations production evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        UPDATE campaign_operations_request_production_admission
           SET requesting_actor = 'changed'
         WHERE request_production_admission_id =
               admission.request_production_admission_id;
        RAISE EXCEPTION 'admission owner mutation unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'Campaign Operations production evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        DELETE FROM campaign_operations_production_enablement_event
         WHERE production_enablement_event_id =
               event.production_enablement_event_id;
        RAISE EXCEPTION 'enablement owner delete unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'Campaign Operations production evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        TRUNCATE campaign_operations_request_production_admission CASCADE;
        RAISE EXCEPTION 'admission owner truncate unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <> 'Campaign Operations production evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        UPDATE campaign_operations_dispatch_attempt
           SET attempt_identity_hash = 'fnv1a64:ffffffffffffffff'
         WHERE dispatch_attempt_id = attempt.dispatch_attempt_id;
        RAISE EXCEPTION 'Attempt V2 owner mutation unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'Campaign Operations production attempt evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        DELETE FROM campaign_operations_dispatch_attempt
         WHERE dispatch_attempt_id = attempt.dispatch_attempt_id;
        RAISE EXCEPTION 'Attempt V2 owner delete unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'Campaign Operations production attempt evidence is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        UPDATE campaign_operations_dispatch_audit_reference_event
           SET diagnostic_code = 'changed'
         WHERE dispatch_audit_reference_event_id =
               audit.dispatch_audit_reference_event_id;
        RAISE EXCEPTION 'production audit owner mutation unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'Campaign Operations production dispatch audit is immutable'
        THEN RAISE; END IF;
    END;
    BEGIN
        DELETE FROM campaign_operations_dispatch_audit_reference_event
         WHERE dispatch_audit_reference_event_id =
               audit.dispatch_audit_reference_event_id;
        RAISE EXCEPTION 'production audit owner delete unexpectedly succeeded';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'Campaign Operations production dispatch audit is immutable'
        THEN RAISE; END IF;
    END;

    -- Every production child must reach migration 054's completed-campaign
    -- gate, while the Boolean remains independently one-way immutable.  The
    -- disposable boundary and false witness exist only in this subtransaction.
    BEGIN
        SET LOCAL session_replication_role = replica;
        UPDATE campaign_operations_campaign
           SET completion_boundary_closed = true
         WHERE operational_campaign_id = 7;
        UPDATE campaign_operations_operational_request
           SET production_dispatch_enabled = false
         WHERE operational_request_id = 7;
        SET LOCAL session_replication_role = origin;
        BEGIN
            UPDATE campaign_operations_operational_request
               SET production_dispatch_enabled = true
             WHERE operational_request_id = 7;
            RAISE EXCEPTION 'post-completion Boolean transition succeeded';
        EXCEPTION WHEN object_not_in_prerequisite_state THEN
            GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
            IF diagnostic <> 'production admission witness is immutable' THEN
                RAISE;
            END IF;
        END;
        BEGIN
            INSERT INTO campaign_operations_request_production_admission
              SELECT admission.*;
            RAISE EXCEPTION 'post-completion admission succeeded';
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
            IF diagnostic NOT LIKE '%completed campaign%' THEN RAISE; END IF;
        END;
        BEGIN
            INSERT INTO campaign_operations_dispatch_attempt SELECT attempt.*;
            RAISE EXCEPTION 'post-completion Attempt V2 succeeded';
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
            IF diagnostic NOT LIKE '%completed campaign%' THEN RAISE; END IF;
        END;
        BEGIN
            INSERT INTO campaign_operations_dispatch_audit_reference_event
              SELECT audit.*;
            RAISE EXCEPTION 'post-completion production audit succeeded';
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
            IF diagnostic NOT LIKE '%completed campaign%' THEN RAISE; END IF;
        END;
        RAISE EXCEPTION 'rollback post-completion fixture'
          USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;
    BEGIN
        event.production_enablement_event_id := nextval(
          pg_get_serial_sequence(
            'campaign_operations_production_enablement_event',
            'production_enablement_event_id'));
        event.operation_key := 'malformed-001';
        event.enablement_identity_canonical :=
          event.enablement_identity_canonical || 'changed';
        INSERT INTO campaign_operations_production_enablement_event
          SELECT event.*;
        RAISE EXCEPTION 'same-hash changed canonical unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic <>
             'fixed production enablement transition context required' THEN
            RAISE;
        END IF;
    END;
END $$;

-- Every nullable V2-only mirror is required as one shape.  Replica mode is
-- used only to bypass immutability triggers so the named CHECK, rather than an
-- earlier unrelated guard, is the observed rejection.
DO $$
DECLARE field_name text;
DECLARE violation_name text;
BEGIN
    SET LOCAL session_replication_role = replica;
    FOREACH field_name IN ARRAY ARRAY[
      'request_production_admission_id',
      'request_production_admission_canonical',
      'request_production_admission_hash',
      'production_enablement_event_id',
      'production_enablement_event_canonical',
      'production_enablement_event_hash', 'operation_key',
      'requesting_actor', 'original_executing_service_principal',
      'approved_build_contract_canonical', 'approved_build_contract_hash',
      'production_capability']
    LOOP
        BEGIN
            EXECUTE format(
              'UPDATE campaign_operations_dispatch_attempt SET %I=NULL '
              'WHERE attempt_contract_version=2', field_name);
            RAISE EXCEPTION 'incomplete V2 field % unexpectedly accepted',
              field_name;
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS violation_name = CONSTRAINT_NAME;
            IF violation_name <>
                 'campaign_operations_dispatch_attempt_v1_v2_shape' THEN
                RAISE;
            END IF;
        END;
    END LOOP;
    BEGIN
        UPDATE campaign_operations_dispatch_attempt
           SET attempt_contract_version = 1
         WHERE attempt_contract_version = 2;
        RAISE EXCEPTION 'mixed V1/V2 shape unexpectedly accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS violation_name = CONSTRAINT_NAME;
        IF violation_name <>
             'campaign_operations_dispatch_attempt_v1_v2_shape' THEN
            RAISE;
        END IF;
    END;
    SET LOCAL session_replication_role = origin;
END $$;

-- A complete fixed disable transition rolled back by its caller leaves the
-- enablement chain and matching audit unchanged.
DO $$
DECLARE predecessor campaign_operations_production_enablement_event%ROWTYPE;
DECLARE disabled campaign_operations_production_enablement_event%ROWTYPE;
DECLARE replayed campaign_operations_production_enablement_event%ROWTYPE;
DECLARE diagnostic text;
BEGIN
    SELECT * INTO STRICT predecessor
    FROM campaign_operations_production_enablement_event
    ORDER BY resulting_version DESC LIMIT 1;
    BEGIN
        SELECT * INTO STRICT disabled
        FROM record_campaign_operations_production_disable_v1(
          'rollback-disable-001', predecessor.production_enablement_event_id,
          predecessor.enablement_identity_canonical,
          predecessor.resulting_version, 'operator@example.test',
          'Prove fixed-transition rollback stability.');
        SELECT * INTO STRICT replayed
        FROM record_campaign_operations_production_disable_v1(
          'rollback-disable-001', predecessor.production_enablement_event_id,
          predecessor.enablement_identity_canonical,
          predecessor.resulting_version, 'operator@example.test',
          'Prove fixed-transition rollback stability.');
        IF replayed.production_enablement_event_id <>
             disabled.production_enablement_event_id THEN
            RAISE EXCEPTION 'production disable replay changed identity';
        END IF;
        BEGIN
            PERFORM record_campaign_operations_production_disable_v1(
              'rollback-disable-001',
              predecessor.production_enablement_event_id,
              predecessor.enablement_identity_canonical,
              predecessor.resulting_version, 'different@example.test',
              'Prove fixed-transition rollback stability.');
            RAISE EXCEPTION 'production disable conflicting replay succeeded';
        EXCEPTION WHEN unique_violation THEN
            GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
            IF diagnostic <> 'production disable conflicting replay' THEN
                RAISE;
            END IF;
        END;
        RAISE EXCEPTION 'force fixed transition rollback' USING ERRCODE='P1234';
    EXCEPTION WHEN SQLSTATE 'P1234' THEN NULL;
    END;
END $$;

SELECT pg_temp.verify_campaign_operations_h1_recovery_alignment();

DO $$
BEGIN
    IF (SELECT count(*) FROM
          campaign_operations_production_enablement_event) <> 1 OR
       (SELECT count(*) FROM
          campaign_operations_request_production_admission) <> 1 OR
       (SELECT count(*) FROM campaign_operations_dispatch_attempt
          WHERE attempt_contract_version = 2) <> 1 OR
       (SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event
          WHERE request_production_admission_id IS NOT NULL) <> 1 THEN
        RAISE EXCEPTION 'H1 rollback/replay stability mismatch';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM campaign_operations_production_readiness_v1
        WHERE scheduler_required_generation = 52 AND
              scheduler_cutover_state = 'complete' AND
              scheduler_evidence_complete AND
              completion_nested_v2_proof_valid) THEN
        RAISE EXCEPTION 'H1 readiness reconstruction mismatch';
    END IF;
END $$;

-- Recovery lookup is identity-independent: immutable acquisition arguments
-- are hydrated from the stored Attempt V2 while PostgreSQL validates the
-- current recovery principal through the separately granted dispatcher
-- capability.  This temporary grant models the later accepted deployment
-- grant and is revoked before the post-upgrade ACL audit.
CREATE TEMP TABLE pg_temp.h1_cross_principal_recovery_before AS
SELECT to_jsonb(attempt) AS attempt_row,
       to_jsonb(admission) AS admission_row,
       attempt.operational_request_id, attempt.expected_request_version,
       attempt.lease_token_digest, attempt.lease_expires_at,
       attempt.operation_key, attempt.requesting_actor,
       attempt.approved_build_contract_canonical,
       convert_to(attempt.attempt_identity_canonical, 'UTF8')
         AS attempt_canonical_bytes,
       convert_to(admission.admission_identity_canonical, 'UTF8')
         AS admission_canonical_bytes,
       request.request_state, request.state_version,
       (SELECT count(*) FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_contract_version=2)
         AS attempt_count,
       (SELECT count(*)
        FROM campaign_operations_request_production_admission
        WHERE operational_request_id=7) AS admission_count,
       (SELECT count(*)
        FROM campaign_operations_dispatch_audit_reference_event
        WHERE operational_request_id=7 AND
              request_production_admission_id IS NOT NULL) AS audit_count
FROM campaign_operations_dispatch_attempt attempt
JOIN campaign_operations_request_production_admission admission
  ON admission.request_production_admission_id =
     attempt.request_production_admission_id
JOIN campaign_operations_operational_request request
  ON request.operational_request_id = attempt.operational_request_id
WHERE attempt.operational_request_id=7 AND
      attempt.operation_key='dispatch-001';

CREATE ROLE campaign_operations_h1_recovery_authorized
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
CREATE ROLE campaign_operations_h1_recovery_unauthorized
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
GRANT campaign_operations_production_dispatcher,
      campaign_operations_production_reader
TO campaign_operations_h1_recovery_authorized;
GRANT SELECT ON pg_temp.h1_cross_principal_recovery_before
TO campaign_operations_h1_recovery_authorized;
GRANT EXECUTE ON FUNCTION
  transition_campaign_operations_request_dispatch_production_v2(
    bigint, integer, text, timestamptz, text, text, text)
TO campaign_operations_production_dispatcher;

SET SESSION AUTHORIZATION campaign_operations_h1_recovery_authorized;
DO $$
DECLARE recovered_attempt campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE replayed_attempt campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE before_row record;
BEGIN
    IF session_user <> 'campaign_operations_h1_recovery_authorized' OR
       current_user <> 'campaign_operations_h1_recovery_authorized' OR
       NOT pg_has_role(session_user,
             'campaign_operations_production_dispatcher', 'MEMBER') OR
       (SELECT rolsuper FROM pg_roles WHERE rolname=session_user) THEN
        RAISE EXCEPTION
          'cross-principal recovery authority prerequisite mismatch';
    END IF;
    SELECT * INTO STRICT before_row
    FROM pg_temp.h1_cross_principal_recovery_before;
    SELECT * INTO STRICT recovered_attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      before_row.operational_request_id, before_row.expected_request_version,
      before_row.lease_token_digest, before_row.lease_expires_at,
      before_row.operation_key, before_row.requesting_actor,
      'current-recovery-build-is-not-historical-input');
    SELECT * INTO STRICT replayed_attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      before_row.operational_request_id, before_row.expected_request_version,
      before_row.lease_token_digest, before_row.lease_expires_at,
      before_row.operation_key, before_row.requesting_actor,
      'second-current-recovery-build-is-not-historical-input');
    IF to_jsonb(recovered_attempt) IS DISTINCT FROM before_row.attempt_row OR
       to_jsonb(replayed_attempt) IS DISTINCT FROM before_row.attempt_row OR
       convert_to(recovered_attempt.attempt_identity_canonical, 'UTF8') <>
         before_row.attempt_canonical_bytes OR
       recovered_attempt.original_executing_service_principal = session_user OR
       recovered_attempt.original_executing_service_principal <>
         'campaign_manager_login' OR
       recovered_attempt.approved_build_contract_canonical IN (
         'current-recovery-build-is-not-historical-input',
         'second-current-recovery-build-is-not-historical-input') THEN
        RAISE EXCEPTION 'cross-principal recovery rewrote historical identity';
    END IF;
END $$;
RESET SESSION AUTHORIZATION;

SET SESSION AUTHORIZATION campaign_operations_h1_recovery_unauthorized;
DO $$
DECLARE diagnostic text;
BEGIN
    IF session_user <> 'campaign_operations_h1_recovery_unauthorized' OR
       pg_has_role(session_user,
         'campaign_operations_production_dispatcher', 'MEMBER') THEN
        RAISE EXCEPTION
          'unauthorized recovery authority prerequisite mismatch';
    END IF;
    BEGIN
        PERFORM transition_campaign_operations_request_dispatch_production_v2(
          7, 3, 'fnv1a64:0123456789abcdef', transaction_timestamp(),
          'dispatch-001', 'manager@example.test', 'unauthorized-build');
        RAISE EXCEPTION 'unauthorized cross-principal recovery succeeded';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS diagnostic = MESSAGE_TEXT;
        IF diagnostic NOT LIKE 'permission denied for function %' THEN
            RAISE;
        END IF;
    END;
END $$;
RESET SESSION AUTHORIZATION;

-- The original principal remains able to replay the exact stored operation.
DO $$
DECLARE replayed_attempt campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE before_row record;
BEGIN
    SELECT * INTO STRICT before_row
    FROM pg_temp.h1_cross_principal_recovery_before;
    SELECT * INTO STRICT replayed_attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      before_row.operational_request_id, before_row.expected_request_version,
      before_row.lease_token_digest, before_row.lease_expires_at,
      before_row.operation_key, before_row.requesting_actor,
      before_row.approved_build_contract_canonical);
    IF to_jsonb(replayed_attempt) IS DISTINCT FROM before_row.attempt_row THEN
        RAISE EXCEPTION 'same-principal recovery changed identity';
    END IF;
END $$;

REVOKE EXECUTE ON FUNCTION
  transition_campaign_operations_request_dispatch_production_v2(
    bigint, integer, text, timestamptz, text, text, text)
FROM campaign_operations_production_dispatcher;
REVOKE campaign_operations_production_dispatcher,
       campaign_operations_production_reader
FROM campaign_operations_h1_recovery_authorized;
REVOKE SELECT ON pg_temp.h1_cross_principal_recovery_before
FROM campaign_operations_h1_recovery_authorized;
DROP ROLE campaign_operations_h1_recovery_authorized;
DROP ROLE campaign_operations_h1_recovery_unauthorized;

DO $$
DECLARE before_row record;
BEGIN
    SELECT * INTO STRICT before_row
    FROM pg_temp.h1_cross_principal_recovery_before;
    IF (SELECT to_jsonb(attempt)
        FROM campaign_operations_dispatch_attempt attempt
        WHERE attempt.operational_request_id=7 AND
              attempt.operation_key='dispatch-001') IS DISTINCT FROM
         before_row.attempt_row OR
       (SELECT to_jsonb(admission)
        FROM campaign_operations_request_production_admission admission
        WHERE admission.operational_request_id=7) IS DISTINCT FROM
         before_row.admission_row OR
       (SELECT convert_to(attempt_identity_canonical, 'UTF8')
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND operation_key='dispatch-001') <>
         before_row.attempt_canonical_bytes OR
       (SELECT convert_to(admission_identity_canonical, 'UTF8')
        FROM campaign_operations_request_production_admission
        WHERE operational_request_id=7) <>
         before_row.admission_canonical_bytes OR
       (SELECT request_state FROM campaign_operations_operational_request
        WHERE operational_request_id=7) <> before_row.request_state OR
       (SELECT state_version FROM campaign_operations_operational_request
        WHERE operational_request_id=7) <> before_row.state_version OR
       (SELECT count(*) FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_contract_version=2) <>
         before_row.attempt_count OR
       (SELECT count(*)
        FROM campaign_operations_request_production_admission
        WHERE operational_request_id=7) <> before_row.admission_count OR
       (SELECT count(*)
        FROM campaign_operations_dispatch_audit_reference_event
        WHERE operational_request_id=7 AND
              request_production_admission_id IS NOT NULL) <>
         before_row.audit_count OR
       has_function_privilege('campaign_operations_production_dispatcher',
         'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)',
         'EXECUTE') THEN
        RAISE EXCEPTION
          'cross-principal recovery changed evidence or retained authority';
    END IF;
END $$;
DROP TABLE pg_temp.h1_cross_principal_recovery_before;

-- Commit the accepted Phase F recovery and reacquire using the advanced
-- request version.  The first admission is retained byte-for-byte while a new
-- Attempt V2/current acquisition audit is appended.
CREATE FUNCTION pg_temp.verify_completed_recovery_for_reacquisition()
RETURNS void LANGUAGE plpgsql AS $$
DECLARE attempt_id bigint;
DECLARE attempt_canonical text;
DECLARE outcome_id bigint;
DECLARE observation_id bigint;
DECLARE cursor_id bigint;
BEGIN
    SET LOCAL session_replication_role=replica;
    UPDATE campaign_operations_operational_request
       SET lease_expires_at=transaction_timestamp()-interval '1 minute'
     WHERE operational_request_id=7;
    SET LOCAL session_replication_role=origin;
    PERFORM transition_campaign_operations_request_ready_recovered(7,4);
    SELECT dispatch_attempt_id,attempt_identity_canonical
      INTO STRICT attempt_id,attempt_canonical
    FROM campaign_operations_dispatch_attempt
    WHERE operational_request_id=7 AND attempt_ordinal=1;
    -- The surrounding H1 fixture deliberately omits the Phase F service's
    -- unrelated source rows. Materialize the exact recovery witness shape
    -- under replica mode, then restore every H1 guard before reacquisition.
    SET LOCAL session_replication_role=replica;
    INSERT INTO campaign_operations_reconciliation_cursor_event(
      run_key,prior_target_id,last_target_id,requested_limit,selected_count)
    VALUES('h1-post-recovery-reacquisition',0,7,1,1)
    RETURNING reconciliation_cursor_event_id INTO cursor_id;
    INSERT INTO campaign_operations_reconciliation_observation(
      reconciliation_cursor_event_id,run_key,operational_campaign_id,
      operational_request_id,
      request_identity_canonical,expected_request_state,
      expected_request_version,reason_code,evidence_identity_canonical,
      evidence_identity_hash,recommended_service,recommended_action,
      diagnostic_code,observation_contract_version,
      observation_identity_canonical,observation_identity_hash)
    VALUES(cursor_id,'h1-post-recovery-reacquisition',7,7,
      'request-canonical-v1',
      'dispatching',4,'dispatch_lease_expired_no_downstream_evidence',
      'recovery-evidence','fnv1a64:b111111111111111',
      'campaign_operations_dispatch_recovery','clear_stale_dispatch_lease',
      'dispatch_lease_expired_no_downstream_evidence',1,
      'recovery-observation','fnv1a64:b222222222222222')
    RETURNING reconciliation_observation_id INTO observation_id;
    INSERT INTO campaign_operations_dispatch_attempt_outcome(
      dispatch_attempt_id,attempt_identity_canonical,result_classification,
      downstream_evidence_classification,semantic_conflict_classification,
      uncertain_commit_recovery_classification,diagnostic_code,
      expected_request_version,resulting_request_version,
      expected_reservation_version,resulting_reservation_version,
      outcome_contract_version,outcome_identity_canonical,
      outcome_identity_hash)
    VALUES(attempt_id,attempt_canonical,'rejected','no_phase5_evidence',
      'none','proven_no_commit',
      'dispatch_lease_expired_no_downstream_evidence',4,5,1,1,1,
      'recovery-outcome','fnv1a64:b333333333333333')
    RETURNING dispatch_attempt_outcome_id INTO outcome_id;
    INSERT INTO campaign_operations_dispatch_audit_reference_event(
      operational_campaign_id,operational_request_id,dispatch_attempt_id,
      dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,
      prior_version,resulting_version,outcome,replay_disposition,
      diagnostic_code)
    VALUES(7,7,attempt_id,outcome_id,'dispatch_lease_recovered',
      'campaign_operations_recovery','campaign_operations_recovery',4,5,
      'recorded','proven_absent',
      'dispatch_lease_expired_no_downstream_evidence');
    INSERT INTO campaign_operations_reconciliation_resolution(
      reconciliation_observation_id,observation_identity_canonical,
      owning_service,owning_capability,transition_identity_canonical,
      transition_identity_hash,dispatch_attempt_outcome_id,
      resolution_disposition,resolution_contract_version,
      resolution_identity_canonical,resolution_identity_hash)
    VALUES(observation_id,'recovery-observation',
      'campaign_operations_dispatch_recovery','campaign_operations_recovery',
      'recovery-outcome','fnv1a64:b333333333333333',outcome_id,
      'request_returned_ready',1,'recovery-resolution',
      'fnv1a64:b444444444444444');
    SET LOCAL session_replication_role=origin;
    IF NOT EXISTS (
        SELECT 1 FROM campaign_operations_operational_request
        WHERE operational_request_id=7 AND request_state='ready'
          AND state_version=5 AND production_dispatch_enabled
          AND lease_token_hash IS NULL AND lease_expires_at IS NULL
          AND dispatcher_identity IS NULL) THEN
        RAISE EXCEPTION 'committed Phase F recovery state mismatch';
    END IF;
    SET CONSTRAINTS ALL IMMEDIATE;
END $$;
SELECT pg_temp.verify_completed_recovery_for_reacquisition();

CREATE FUNCTION pg_temp.verify_post_recovery_reacquisition()
RETURNS void LANGUAGE plpgsql AS $$
DECLARE admission_id bigint;
DECLARE admission_canonical text;
DECLARE build_canonical text;
DECLARE new_attempt campaign_operations_dispatch_attempt%ROWTYPE;
BEGIN
    SELECT request_production_admission_id,admission_identity_canonical
      INTO STRICT admission_id,admission_canonical
    FROM campaign_operations_request_production_admission
    WHERE operational_request_id=7;
    SELECT approved_build_contract_canonical INTO STRICT build_canonical
    FROM campaign_operations_production_enablement_event
    WHERE event_kind='enable';
    SELECT candidate.* INTO STRICT new_attempt
    FROM transition_campaign_operations_request_dispatch_production_v2(
      7,5,'fnv1a64:fedcba9876543210',
      transaction_timestamp()+interval '5 minutes','dispatch-002',
      'later.manager@example.test',build_canonical) candidate;
    IF new_attempt.attempt_ordinal <> 2 OR
       new_attempt.expected_request_version <> 5 OR
       new_attempt.resulting_request_version <> 6 OR
       new_attempt.request_production_admission_id <> admission_id OR
       new_attempt.request_production_admission_canonical <>
         admission_canonical OR
       (SELECT count(*) FROM campaign_operations_request_production_admission
        WHERE operational_request_id=7) <> 1 OR
       (SELECT count(*) FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_contract_version=2) <> 2 OR
       (SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event
        WHERE operational_request_id=7 AND cause_kind='dispatch_lease_acquired'
          AND request_production_admission_id=admission_id) <> 2 THEN
        RAISE EXCEPTION 'post-recovery reacquisition evidence mismatch';
    END IF;
    SET CONSTRAINTS ALL IMMEDIATE;
END $$;
SELECT pg_temp.verify_post_recovery_reacquisition();

-- A replay of Attempt #2 must close over the immutable Attempt #1 admitted by
-- Admission A.  Each corruption is synthesized inside a nested disposable
-- subtransaction, proves Attempt #2 stayed byte-for-byte intact, requires the
-- exact corruption classification, proves replay performed no repair, and is
-- then restored only by rolling the subtransaction back.
CREATE FUNCTION pg_temp.verify_first_attempt_corruption_rejected(
    mutation_name text, setup_sql text, mutation_sql text, probe_sql text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE first_attempt_id bigint;
DECLARE second_attempt_id bigint;
DECLARE before_first_evidence jsonb;
DECLARE before_second_attempt jsonb;
DECLARE mutated_first_evidence jsonb;
DECLARE after_second_attempt jsonb;
DECLARE probe_reached boolean;
DECLARE replay_row_count integer;
DECLARE diagnostic text;
DECLARE error_state text;
BEGIN
    SELECT dispatch_attempt_id INTO STRICT first_attempt_id
    FROM campaign_operations_dispatch_attempt
    WHERE operational_request_id=7 AND attempt_ordinal=1;
    SELECT dispatch_attempt_id INTO STRICT second_attempt_id
    FROM campaign_operations_dispatch_attempt
    WHERE operational_request_id=7 AND operation_key='dispatch-002';
    SELECT jsonb_build_object(
      'attempts',coalesce((SELECT jsonb_agg(to_jsonb(candidate) ORDER BY
                    candidate.dispatch_attempt_id)
                 FROM campaign_operations_dispatch_attempt candidate
                 WHERE candidate.request_production_admission_id=(SELECT
                   request_production_admission_id
                   FROM campaign_operations_dispatch_attempt
                   WHERE dispatch_attempt_id=first_attempt_id)
                   AND candidate.attempt_ordinal=1),'[]'::jsonb),
      'audits',coalesce((SELECT jsonb_agg(to_jsonb(audit) ORDER BY
                    audit.dispatch_audit_reference_event_id)
                 FROM campaign_operations_dispatch_audit_reference_event audit
                 WHERE audit.dispatch_attempt_id=first_attempt_id),'[]'::jsonb))
      INTO STRICT before_first_evidence;
    SELECT to_jsonb(candidate) INTO STRICT before_second_attempt
    FROM campaign_operations_dispatch_attempt candidate
    WHERE candidate.dispatch_attempt_id=second_attempt_id;

    BEGIN
        PERFORM set_config('session_replication_role','replica',true);
        IF setup_sql <> '' THEN EXECUTE setup_sql; END IF;
        EXECUTE mutation_sql;
        PERFORM set_config('session_replication_role','origin',true);
        EXECUTE probe_sql INTO STRICT probe_reached;
        IF NOT probe_reached THEN
            RAISE EXCEPTION '% mutation did not reach intended branch',
              mutation_name;
        END IF;
        SELECT jsonb_build_object(
          'attempts',coalesce((SELECT jsonb_agg(to_jsonb(candidate) ORDER BY
                        candidate.dispatch_attempt_id)
                     FROM campaign_operations_dispatch_attempt candidate
                     WHERE candidate.request_production_admission_id=(SELECT
                       request_production_admission_id
                       FROM campaign_operations_dispatch_attempt
                       WHERE dispatch_attempt_id=first_attempt_id)
                       AND candidate.attempt_ordinal=1),'[]'::jsonb),
          'audits',coalesce((SELECT jsonb_agg(to_jsonb(audit) ORDER BY
                        audit.dispatch_audit_reference_event_id)
                     FROM campaign_operations_dispatch_audit_reference_event audit
                     WHERE audit.dispatch_attempt_id=first_attempt_id),
                    '[]'::jsonb))
          INTO STRICT mutated_first_evidence;
        SELECT to_jsonb(candidate) INTO STRICT after_second_attempt
        FROM campaign_operations_dispatch_attempt candidate
        WHERE candidate.dispatch_attempt_id=second_attempt_id;
        IF mutated_first_evidence IS NOT DISTINCT FROM before_first_evidence OR
           after_second_attempt IS DISTINCT FROM before_second_attempt THEN
            RAISE EXCEPTION '% mutation authenticity mismatch', mutation_name;
        END IF;
        BEGIN
            PERFORM candidate.dispatch_attempt_id
            FROM campaign_operations_dispatch_attempt current_attempt,
                 LATERAL campaign_operations_production_acquire_replay_v2(
                   current_attempt.operational_request_id,
                   current_attempt.expected_request_version,
                   current_attempt.lease_token_digest,
                   current_attempt.lease_expires_at,
                   current_attempt.operation_key,
                   current_attempt.requesting_actor,
                   current_attempt.approved_build_contract_canonical) candidate
            WHERE current_attempt.dispatch_attempt_id=second_attempt_id;
            GET DIAGNOSTICS replay_row_count = ROW_COUNT;
            RAISE EXCEPTION
              '% corrupt first attempt returned existing_identical rows=%',
              mutation_name,replay_row_count;
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS error_state = RETURNED_SQLSTATE,
              diagnostic = MESSAGE_TEXT;
            IF error_state <> '23514' OR diagnostic <>
                 'production acquisition replay evidence corrupt' THEN
                RAISE EXCEPTION '% wrong replay rejection % %',
                  mutation_name,error_state,diagnostic;
            END IF;
        END;
        EXECUTE probe_sql INTO STRICT probe_reached;
        SELECT to_jsonb(candidate) INTO STRICT after_second_attempt
        FROM campaign_operations_dispatch_attempt candidate
        WHERE candidate.dispatch_attempt_id=second_attempt_id;
        IF NOT probe_reached OR
           after_second_attempt IS DISTINCT FROM before_second_attempt THEN
            RAISE EXCEPTION '% replay repaired or changed evidence', mutation_name;
        END IF;
        RAISE EXCEPTION 'rollback synthesized first-attempt corruption'
          USING ERRCODE='ZH101';
    EXCEPTION WHEN SQLSTATE 'ZH101' THEN
        NULL;
    END;

    SELECT jsonb_build_object(
      'attempts',coalesce((SELECT jsonb_agg(to_jsonb(candidate) ORDER BY
                    candidate.dispatch_attempt_id)
                 FROM campaign_operations_dispatch_attempt candidate
                 WHERE candidate.request_production_admission_id=(SELECT
                   request_production_admission_id
                   FROM campaign_operations_dispatch_attempt
                   WHERE dispatch_attempt_id=first_attempt_id)
                   AND candidate.attempt_ordinal=1),'[]'::jsonb),
      'audits',coalesce((SELECT jsonb_agg(to_jsonb(audit) ORDER BY
                    audit.dispatch_audit_reference_event_id)
                 FROM campaign_operations_dispatch_audit_reference_event audit
                 WHERE audit.dispatch_attempt_id=first_attempt_id),'[]'::jsonb))
      INTO STRICT mutated_first_evidence;
    SELECT to_jsonb(candidate) INTO STRICT after_second_attempt
    FROM campaign_operations_dispatch_attempt candidate
    WHERE candidate.dispatch_attempt_id=second_attempt_id;
    IF mutated_first_evidence IS DISTINCT FROM before_first_evidence OR
       after_second_attempt IS DISTINCT FROM before_second_attempt OR
       current_setting('session_replication_role') <> 'origin' THEN
        RAISE EXCEPTION '% fixture rollback mismatch', mutation_name;
    END IF;
END $$;

SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first canonical changed/stored hash retained','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET attempt_identity_canonical=attempt_identity_canonical||'x'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT attempt_identity_canonical LIKE '%x'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first hash changed/canonical retained','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET attempt_identity_hash='fnv1a64:ffffffffffffffff'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT attempt_identity_hash='fnv1a64:ffffffffffffffff'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first typed request mirror changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET request_identity_canonical='corrupt-first-request'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT request_identity_canonical='corrupt-first-request'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first admission canonical relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET request_production_admission_canonical=
             request_production_admission_canonical||'x'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT request_production_admission_canonical LIKE '%x'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first admission hash relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET request_production_admission_hash='fnv1a64:ffffffffffffffff'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT request_production_admission_hash=
                  'fnv1a64:ffffffffffffffff'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first admission relationship id changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET request_production_admission_id=999999
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT request_production_admission_id=999999
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first enablement canonical relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET production_enablement_event_canonical=
             production_enablement_event_canonical||'x'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT production_enablement_event_canonical LIKE '%x'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first enablement hash relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET production_enablement_event_hash='fnv1a64:ffffffffffffffff'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT production_enablement_event_hash=
                  'fnv1a64:ffffffffffffffff'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first enablement relationship id changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET production_enablement_event_id=999999
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT production_enablement_event_id=999999
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first approved-build relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET approved_build_contract_canonical=
             approved_build_contract_canonical||'x'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT approved_build_contract_canonical LIKE '%x'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first approved-build hash relationship changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET approved_build_contract_hash='fnv1a64:ffffffffffffffff'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT approved_build_contract_hash='fnv1a64:ffffffffffffffff'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'duplicate first attempt authority',
  $setup$DO $block$ DECLARE constraint_name name; BEGIN
      SELECT conname INTO STRICT constraint_name FROM pg_constraint
       WHERE conrelid='campaign_operations_dispatch_attempt'::regclass
         AND contype='u' AND pg_get_constraintdef(oid)=
           'UNIQUE (operational_request_id, attempt_ordinal)';
      EXECUTE format('ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT %I',
                     constraint_name);
    END $block$;
    DROP INDEX campaign_operations_dispatch_attempt_v2_operation_uidx;
    ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT
      campaign_operations_dispatch_attempt_request_version_uidx$setup$,
  $mutation$INSERT INTO campaign_operations_dispatch_attempt
    SELECT (jsonb_populate_record(NULL::campaign_operations_dispatch_attempt,
      jsonb_set(to_jsonb(candidate),'{dispatch_attempt_id}',
        to_jsonb(nextval(pg_get_serial_sequence(
          'campaign_operations_dispatch_attempt','dispatch_attempt_id')))))).*
    FROM campaign_operations_dispatch_attempt candidate
    WHERE candidate.operational_request_id=7 AND candidate.attempt_ordinal=1$mutation$,
  $probe$SELECT count(*)=2 FROM campaign_operations_dispatch_attempt candidate
    WHERE candidate.request_production_admission_id=(SELECT
      request_production_admission_id FROM campaign_operations_dispatch_attempt
      WHERE dispatch_attempt_id=(SELECT min(dispatch_attempt_id)
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1))
      AND candidate.attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first operation key changed','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET operation_key='corrupt-first-operation'
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT operation_key='corrupt-first-operation'
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first request versions contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_attempt
       SET expected_request_version=8,resulting_request_version=9
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT expected_request_version=8 AND resulting_request_version=9
           FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first ordinal contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_attempt SET attempt_ordinal=9
     WHERE operational_request_id=7 AND attempt_ordinal=1$mutation$,
  $probe$SELECT count(*)=0 FROM campaign_operations_dispatch_attempt
          WHERE operational_request_id=7 AND attempt_ordinal=1$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first acquisition audit missing','',
  $mutation$DELETE FROM campaign_operations_dispatch_audit_reference_event
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT count(*)=0
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first acquisition audit duplicated',
  'DROP INDEX campaign_operations_dispatch_audit_acquisition_uidx',
  $mutation$INSERT INTO campaign_operations_dispatch_audit_reference_event(
      operational_campaign_id,operational_request_id,dispatch_attempt_id,
      dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,
      prior_version,resulting_version,outcome,replay_disposition,
      diagnostic_code,request_production_admission_id,
      production_enablement_event_id)
    SELECT operational_campaign_id,operational_request_id,dispatch_attempt_id,
      dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,
      prior_version,resulting_version,outcome,replay_disposition,
      diagnostic_code,request_production_admission_id,
      production_enablement_event_id
    FROM campaign_operations_dispatch_audit_reference_event
    WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
      FROM campaign_operations_dispatch_attempt
      WHERE operational_request_id=7 AND attempt_ordinal=1)
      AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT count(*)=2
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first audit actor contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_audit_reference_event
       SET actor_identity='corrupt.audit.actor'
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT actor_identity='corrupt.audit.actor'
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first audit capability contradicted',
  'ALTER TABLE campaign_operations_dispatch_audit_reference_event DROP CONSTRAINT campaign_operations_dispatch_audit_cause_shape',
  $mutation$UPDATE campaign_operations_dispatch_audit_reference_event
       SET capability='campaign_operations_production_phase5_transactional'
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT capability=
              'campaign_operations_production_phase5_transactional'
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first audit versions contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_audit_reference_event
       SET prior_version=8,resulting_version=9
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT prior_version=8 AND resulting_version=9
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first audit admission mirror contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_audit_reference_event
       SET request_production_admission_id=999999
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT request_production_admission_id=999999
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);
SELECT pg_temp.verify_first_attempt_corruption_rejected(
  'first audit enablement mirror contradicted','',
  $mutation$UPDATE campaign_operations_dispatch_audit_reference_event
       SET production_enablement_event_id=999999
     WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
       FROM campaign_operations_dispatch_attempt
       WHERE operational_request_id=7 AND attempt_ordinal=1)
       AND cause_kind='dispatch_lease_acquired'$mutation$,
  $probe$SELECT production_enablement_event_id=999999
       FROM campaign_operations_dispatch_audit_reference_event
      WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
        FROM campaign_operations_dispatch_attempt
        WHERE operational_request_id=7 AND attempt_ordinal=1)
        AND cause_kind='dispatch_lease_acquired'$probe$);

-- Construct a rollback-scoped non-genesis authorization graph.  The real
-- Attempt #2 replay below therefore closes over v3 -> v2 -> v1 rather than
-- the genesis-only production fixture used by the earlier corruption cases.
DO $$
DECLARE v1 campaign_operations_production_enablement_event%ROWTYPE;
DECLARE v2 campaign_operations_production_enablement_event%ROWTYPE;
DECLARE v3 campaign_operations_production_enablement_event%ROWTYPE;
DECLARE before_v2 jsonb;
DECLARE before_attempt jsonb;
DECLARE after_attempt jsonb;
DECLARE replay_count integer;
DECLARE diagnostic text;
DECLARE error_state text;
BEGIN
  BEGIN
    SELECT * INTO STRICT v1
    FROM campaign_operations_production_enablement_event
    WHERE operation_key='enable-001';
    SELECT * INTO STRICT v2
    FROM record_campaign_operations_production_disable_v1(
      'non-genesis-disable-v2',v1.production_enablement_event_id,
      v1.enablement_identity_canonical,v1.resulting_version,
      'operator@example.test','Non-genesis regression disable.') ;
    SELECT * INTO STRICT v3
    FROM record_campaign_operations_production_enable_v1(
      'non-genesis-enable-v3',v2.resulting_version,
      v1.scheduler_protocol_evidence_canonical,
      'cee://phase-h/generation-52-v3','operator@example.test',
      v1.manager_service_contract,v1.approved_build_contract_canonical,
      v1.approved_build_source_commit,v1.approved_build_compiler_contract,
      v1.approved_build_executable_sha256,'Non-genesis regression enable.');
    IF NOT campaign_operations_production_enablement_history_valid_v1(
      v3.production_enablement_event_id) THEN
      RAISE EXCEPTION 'non-genesis lineage was not valid before corruption';
    END IF;
    SET LOCAL session_replication_role=replica;
    UPDATE campaign_operations_request_production_admission
       SET production_enablement_event_id=v3.production_enablement_event_id,
           enable_event_canonical=v3.enablement_identity_canonical,
           approved_build_contract_canonical=v3.approved_build_contract_canonical,
           approved_build_contract_hash=v3.approved_build_contract_hash
     WHERE operational_request_id=7;
    UPDATE campaign_operations_request_production_admission admission
       SET admission_identity_canonical=
             campaign_operations_request_production_admission_canonical_v1(admission);
    UPDATE campaign_operations_request_production_admission
       SET admission_identity_hash=
             campaign_operations_tagged_fnv1a64(admission_identity_canonical)
     WHERE operational_request_id=7;
    UPDATE campaign_operations_dispatch_attempt
       SET production_enablement_event_id=v3.production_enablement_event_id,
           production_enablement_event_canonical=v3.enablement_identity_canonical,
           production_enablement_event_hash=v3.enablement_identity_hash,
           approved_build_contract_canonical=v3.approved_build_contract_canonical,
           approved_build_contract_hash=v3.approved_build_contract_hash
     WHERE operational_request_id=7 AND attempt_contract_version=2;
    UPDATE campaign_operations_dispatch_attempt attempt
       SET request_production_admission_canonical=
             admission.admission_identity_canonical,
           request_production_admission_hash=admission.admission_identity_hash
      FROM campaign_operations_request_production_admission admission
     WHERE attempt.operational_request_id=admission.operational_request_id
       AND attempt.operational_request_id=7
       AND attempt.attempt_contract_version=2;
    UPDATE campaign_operations_dispatch_attempt attempt
       SET attempt_identity_canonical=
             campaign_operations_dispatch_attempt_v2_canonical(attempt);
    UPDATE campaign_operations_dispatch_attempt
       SET attempt_identity_hash=
             campaign_operations_tagged_fnv1a64(attempt_identity_canonical)
     WHERE operational_request_id=7 AND attempt_contract_version=2;
    UPDATE campaign_operations_dispatch_audit_reference_event audit
       SET production_enablement_event_id=v3.production_enablement_event_id
      FROM campaign_operations_dispatch_attempt attempt
     WHERE audit.dispatch_attempt_id=attempt.dispatch_attempt_id
       AND attempt.operational_request_id=7
       AND attempt.attempt_contract_version=2
       AND audit.cause_kind='dispatch_lease_acquired';
    SET LOCAL session_replication_role=origin;
    PERFORM candidate.dispatch_attempt_id
    FROM campaign_operations_dispatch_attempt attempt,
         LATERAL campaign_operations_production_acquire_replay_v2(
           attempt.operational_request_id,attempt.expected_request_version,
           attempt.lease_token_digest,attempt.lease_expires_at,
           attempt.operation_key,attempt.requesting_actor,
           attempt.approved_build_contract_canonical) candidate
    WHERE attempt.operational_request_id=7 AND attempt.operation_key='dispatch-002';
    GET DIAGNOSTICS replay_count=ROW_COUNT;
    IF replay_count <> 1 THEN
      RAISE EXCEPTION 'valid non-genesis Attempt #2 replay did not succeed';
    END IF;
    SELECT to_jsonb(v2) INTO STRICT before_v2;
    SELECT to_jsonb(attempt) INTO STRICT before_attempt
    FROM campaign_operations_dispatch_attempt attempt
    WHERE operational_request_id=7 AND operation_key='dispatch-002';
    SET LOCAL session_replication_role=replica;
    UPDATE campaign_operations_production_enablement_event
       SET enablement_identity_hash='fnv1a64:ffffffffffffffff'
     WHERE production_enablement_event_id=v2.production_enablement_event_id;
    SET LOCAL session_replication_role=origin;
    IF (SELECT enablement_identity_hash='fnv1a64:ffffffffffffffff'
        FROM campaign_operations_production_enablement_event
        WHERE production_enablement_event_id=v2.production_enablement_event_id) IS NOT TRUE OR
       (SELECT to_jsonb(attempt) FROM campaign_operations_dispatch_attempt attempt
        WHERE operational_request_id=7 AND operation_key='dispatch-002')
          IS DISTINCT FROM before_attempt THEN
      RAISE EXCEPTION 'non-genesis adversarial mutation authenticity mismatch';
    END IF;
    BEGIN
      PERFORM candidate.dispatch_attempt_id
      FROM campaign_operations_dispatch_attempt attempt,
           LATERAL campaign_operations_production_acquire_replay_v2(
             attempt.operational_request_id,attempt.expected_request_version,
             attempt.lease_token_digest,attempt.lease_expires_at,
             attempt.operation_key,attempt.requesting_actor,
             attempt.approved_build_contract_canonical) candidate
      WHERE attempt.operational_request_id=7 AND attempt.operation_key='dispatch-002';
      RAISE EXCEPTION 'v2 hash-only corruption replay unexpectedly succeeded';
    EXCEPTION WHEN check_violation THEN
      GET STACKED DIAGNOSTICS error_state=RETURNED_SQLSTATE,
        diagnostic=MESSAGE_TEXT;
      IF error_state <> '23514' OR diagnostic <>
           'production acquisition replay evidence corrupt' THEN RAISE; END IF;
    END;
    SELECT to_jsonb(attempt) INTO STRICT after_attempt
    FROM campaign_operations_dispatch_attempt attempt
    WHERE operational_request_id=7 AND operation_key='dispatch-002';
    IF after_attempt IS DISTINCT FROM before_attempt OR
       (SELECT to_jsonb(event) FROM campaign_operations_production_enablement_event event
        WHERE production_enablement_event_id=v2.production_enablement_event_id)
          IS NOT DISTINCT FROM before_v2 THEN
      RAISE EXCEPTION 'non-genesis replay repaired evidence unexpectedly';
    END IF;
    RAISE EXCEPTION 'rollback non-genesis fixture' USING ERRCODE='ZH102';
  EXCEPTION WHEN SQLSTATE 'ZH102' THEN NULL;
  END;
  IF NOT campaign_operations_production_enablement_history_valid_v1(
    (SELECT production_enablement_event_id FROM campaign_operations_dispatch_attempt
     WHERE operational_request_id=7 AND attempt_ordinal=1)) THEN
    RAISE EXCEPTION 'non-genesis fixture did not restore the baseline graph';
  END IF;
END $$;

DO $$
DECLARE before_attempt jsonb;
DECLARE replayed campaign_operations_dispatch_attempt%ROWTYPE;
BEGIN
    SELECT to_jsonb(candidate) INTO STRICT before_attempt
    FROM campaign_operations_dispatch_attempt candidate
    WHERE operational_request_id=7 AND operation_key='dispatch-002';
    SELECT candidate.* INTO STRICT replayed
    FROM campaign_operations_dispatch_attempt current_attempt,
         LATERAL campaign_operations_production_acquire_replay_v2(
           current_attempt.operational_request_id,
           current_attempt.expected_request_version,
           current_attempt.lease_token_digest,current_attempt.lease_expires_at,
           current_attempt.operation_key,current_attempt.requesting_actor,
           current_attempt.approved_build_contract_canonical) candidate
    WHERE current_attempt.operational_request_id=7
      AND current_attempt.operation_key='dispatch-002';
    IF to_jsonb(replayed) IS DISTINCT FROM before_attempt THEN
        RAISE EXCEPTION 'valid Attempt #2 replay changed evidence';
    END IF;
END $$;

SELECT 'Campaign Operations Phase H1 migration tests passed' AS result;
