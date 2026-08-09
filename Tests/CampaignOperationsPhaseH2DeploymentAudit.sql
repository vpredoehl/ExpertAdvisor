-- H2 deployment audit SQL.  It is read-only and emits zero rows on success.
SELECT 'H2A004:migration-055'
WHERE NOT EXISTS (
  SELECT 1 FROM public.schema_migrations
  WHERE version='055'
    AND filename='055_campaign_operations_production_admission_foundation.sql'
    AND checksum='86a35844edd3cc233e8f72ff985c339474dc09d3cd79d354fcb3adeb902aa66f');
SELECT 'H2A004:migration-056'
WHERE NOT EXISTS (
  SELECT 1 FROM public.schema_migrations
  WHERE version='056'
    AND filename='056_campaign_operations_h2_privilege_deployment_contract.sql'
    AND checksum=:'h2_sum');
SELECT 'H2A004:migration-057'
WHERE NOT EXISTS (
  SELECT 1 FROM public.schema_migrations
  WHERE version='057'
    AND filename='057_campaign_operations_h2_production_bind_state_contract.sql'
    AND checksum=:'h2_bind_sum');

WITH expected(object_identity,grantee) AS (VALUES
 ('record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)','campaign_operations_h1_boundary_authority'),
 ('record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)','campaign_operations_production_enabler'),
 ('record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)','campaign_operations_h1_boundary_authority'),
 ('record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)','campaign_operations_production_disabler'),
 ('transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)','campaign_operations_h1_boundary_authority'),
 ('transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)','campaign_operations_production_dispatcher'),
 ('transition_campaign_operations_request_bound_production_v2(bigint,integer,text,bigint,text,text)','campaign_operations_owner'),
 ('transition_campaign_operations_request_bound_production_v2(bigint,integer,text,bigint,text,text)','campaign_operations_production_phase5_transactional')),
actual AS (
 SELECT p.oid::regprocedure::text AS object_identity,
        CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END AS grantee,
        a.is_grantable
 FROM pg_catalog.pg_proc p
 CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl,pg_catalog.acldefault('f',p.proowner))) a
 WHERE p.oid IN (SELECT to_regprocedure(object_identity) FROM expected)),
diff AS (
 SELECT 'H2A002:'||object_identity||':'||grantee FROM expected EXCEPT SELECT 'H2A002:'||object_identity||':'||grantee FROM actual
 UNION ALL
 SELECT 'H2A003:'||object_identity||':'||grantee FROM actual EXCEPT SELECT 'H2A003:'||object_identity||':'||grantee FROM expected
 UNION ALL SELECT 'H2A003:'||object_identity||':grant-option' FROM actual WHERE is_grantable)
SELECT * FROM diff LIMIT 1;

SELECT 'H2A005:'||p.oid::regprocedure::text
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_roles r ON r.oid=p.proowner
WHERE p.oid IN (
 'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
 'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
 'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure)
AND (r.rolname<>'campaign_operations_h1_boundary_authority' OR NOT p.prosecdef OR p.prokind<>'f' OR p.proleakproof OR p.provolatile<>'v' OR p.proparallel<>'u' OR p.pronargdefaults<>0 OR p.provariadic<>0 OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[])
LIMIT 1;

SELECT 'H2A005:production-bind-transition'
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_roles r ON r.oid=p.proowner
WHERE p.oid='transition_campaign_operations_request_bound_production_v2(bigint,integer,text,bigint,text,text)'::regprocedure
AND (r.rolname<>'campaign_operations_owner' OR NOT p.prosecdef OR p.prokind<>'f'
     OR p.proleakproof OR p.provolatile<>'v' OR p.proparallel<>'u'
     OR p.pronargdefaults<>0 OR p.provariadic<>0
     OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[])
LIMIT 1;

WITH expected(grantee) AS (VALUES
 ('campaign_operations_h1_boundary_authority'),
 ('campaign_operations_production_reader')),
actual AS (
 SELECT CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END AS grantee,
        a.is_grantable
 FROM pg_catalog.pg_proc p
 CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl,pg_catalog.acldefault('f',p.proowner))) a
 WHERE p.oid='campaign_operations_production_readiness_snapshot_v1()'::regprocedure),
diff AS (
 SELECT 'H2A002:readiness-wrapper:'||grantee FROM expected
 EXCEPT SELECT 'H2A002:readiness-wrapper:'||grantee FROM actual
 UNION ALL
 SELECT 'H2A003:readiness-wrapper:'||grantee FROM actual
 EXCEPT SELECT 'H2A003:readiness-wrapper:'||grantee FROM expected
 UNION ALL
 SELECT 'H2A003:readiness-wrapper:grant-option' FROM actual WHERE is_grantable)
SELECT * FROM diff LIMIT 1;

SELECT 'H2A005:readiness-wrapper'
FROM pg_catalog.pg_proc p
WHERE p.oid='campaign_operations_production_readiness_snapshot_v1()'::regprocedure
  AND (p.proowner <> 'campaign_operations_h1_boundary_authority'::regrole
       OR NOT p.prosecdef OR p.prokind <> 'f' OR p.proleakproof
       OR p.provolatile <> 's' OR p.proparallel <> 'u'
       OR p.pronargdefaults <> 0 OR p.provariadic <> 0
       OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[])
LIMIT 1;

WITH expected(object_identity,grantee) AS (VALUES
 ('campaign_operations_scheduler_protocol_evidence_lock_v1()','campaign_operations_h1_boundary_authority'),
 ('campaign_operations_scheduler_protocol_evidence_lock_v1()','campaign_operations_scheduler_protocol_evidence_reader'),
 ('campaign_operations_scheduler_protocol_evidence_lock_v1()','campaign_operations_production_phase5_transactional')),
actual AS (
 SELECT p.oid::regprocedure::text AS object_identity,
        CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END AS grantee,
        a.is_grantable
 FROM pg_catalog.pg_proc p
 CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl,pg_catalog.acldefault('f',p.proowner))) a
 WHERE p.oid='campaign_operations_scheduler_protocol_evidence_lock_v1()'::regprocedure),
diff AS (
 SELECT 'H2A002:helper:'||object_identity||':'||grantee FROM expected EXCEPT SELECT 'H2A002:helper:'||object_identity||':'||grantee FROM actual
 UNION ALL SELECT 'H2A003:helper:'||object_identity||':'||grantee FROM actual EXCEPT SELECT 'H2A003:helper:'||object_identity||':'||grantee FROM expected
 UNION ALL SELECT 'H2A003:helper:grant-option' FROM actual WHERE is_grantable)
SELECT * FROM diff LIMIT 1;

WITH expected(object_identity,grantee) AS (VALUES
 ('public.campaign_operations_production_enablement_event','campaign_operations_h1_boundary_authority'),
 ('public.campaign_operations_production_enablement_event','campaign_operations_owner'),
 ('public.campaign_operations_production_enablement_event','campaign_operations_production_reader'),
 ('public.campaign_operations_production_enablement_event','campaign_operations_production_dispatcher'),
 ('public.campaign_operations_production_enablement_event','campaign_operations_production_phase5_transactional'),
 ('public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_h1_boundary_authority'),
 ('public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_owner'),
 ('public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_production_reader'),
 ('public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_production_dispatcher'),
 ('public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_production_phase5_transactional'),
 ('public.campaign_operations_request_production_admission','campaign_operations_h1_boundary_authority'),
 ('public.campaign_operations_request_production_admission','campaign_operations_owner'),
 ('public.campaign_operations_request_production_admission','campaign_operations_production_reader'),
 ('public.campaign_operations_request_production_admission','campaign_operations_production_dispatcher'),
 ('public.campaign_operations_request_production_admission','campaign_operations_production_phase5_transactional')),
actual AS (
 SELECT n.nspname||'.'||c.relname AS object_identity,
        CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END AS grantee,
        a.is_grantable,a.privilege_type
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault('r',c.relowner))) a
 WHERE n.nspname='public' AND c.relname IN ('campaign_operations_production_enablement_event','campaign_operations_production_enablement_audit_reference_event','campaign_operations_request_production_admission')
   AND a.privilege_type='SELECT'),
diff AS (
 SELECT 'H2A002:evidence:'||object_identity||':'||grantee FROM expected EXCEPT SELECT 'H2A002:evidence:'||object_identity||':'||grantee FROM actual
 UNION ALL SELECT 'H2A003:evidence:'||object_identity||':'||grantee FROM actual EXCEPT SELECT 'H2A003:evidence:'||object_identity||':'||grantee FROM expected
 UNION ALL SELECT 'H2A003:evidence:grant-option' FROM actual WHERE is_grantable
 UNION ALL SELECT 'H2A003:evidence:privilege:'||object_identity||':'||grantee||':'||privilege_type FROM actual WHERE grantee IN ('campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional') AND privilege_type<>'SELECT')
SELECT * FROM diff LIMIT 1;

SELECT 'H2A001:role:'||expected.role_name
FROM (VALUES
 ('campaign_operations_h1_boundary_authority',true),
 ('campaign_operations_production_enabler',false),
 ('campaign_operations_production_disabler',false),
 ('campaign_operations_production_dispatcher',false),
 ('campaign_operations_production_phase5_transactional',false),
 ('campaign_operations_production_reader',false),
 ('campaign_operations_scheduler_protocol_evidence_reader',false)) expected(role_name,must_super)
LEFT JOIN pg_catalog.pg_authid r ON r.rolname=expected.role_name
WHERE r.oid IS NULL OR r.rolcanlogin OR r.rolsuper<>expected.must_super OR NOT r.rolinherit OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls OR r.rolconnlimit<>-1 OR r.rolpassword IS NOT NULL OR r.rolvaliduntil IS NOT NULL
LIMIT 1;

SELECT 'H2A002:sealed-graph:'||g.rolname||'->'||m.rolname
FROM pg_catalog.pg_auth_members x
JOIN pg_catalog.pg_roles g ON g.oid=x.roleid
JOIN pg_catalog.pg_roles m ON m.oid=x.member
WHERE g.rolname='campaign_operations_h1_boundary_authority'
   OR m.rolname='campaign_operations_h1_boundary_authority'
   OR x.admin_option
LIMIT 1;

SELECT 'H2A002:capability-graph:'||g.rolname||'->'||m.rolname
FROM pg_catalog.pg_auth_members x
JOIN pg_catalog.pg_roles g ON g.oid=x.roleid
JOIN pg_catalog.pg_roles m ON m.oid=x.member
WHERE g.rolname IN ('campaign_operations_production_enabler','campaign_operations_production_disabler','campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional','campaign_operations_production_reader','campaign_operations_scheduler_protocol_evidence_reader')
  AND (NOT m.rolcanlogin OR x.admin_option)
LIMIT 1;

WITH RECURSIVE role_reach(login_oid,reached_oid,path,depth) AS (
 SELECT member,roleid,ARRAY[member,roleid]::oid[],1
 FROM pg_catalog.pg_auth_members membership
 JOIN pg_catalog.pg_roles login ON login.oid=membership.member
 WHERE login.rolcanlogin AND login.rolinherit
 UNION ALL
 SELECT reach.login_oid,membership.roleid,
        reach.path||membership.roleid,reach.depth+1
 FROM role_reach reach
 JOIN pg_catalog.pg_auth_members membership
   ON membership.member=reach.reached_oid
 WHERE reach.depth < 1024
   AND NOT membership.roleid=ANY(reach.path)
), effective AS (
 SELECT DISTINCT login.rolname AS login_name,reached.rolname AS reached_name
 FROM role_reach reach
 JOIN pg_catalog.pg_roles login ON login.oid=reach.login_oid
 JOIN pg_catalog.pg_roles reached ON reached.oid=reach.reached_oid
), combos AS (
 SELECT login_name,pg_catalog.string_agg(reached_name,',' ORDER BY reached_name) AS combo
 FROM effective
 WHERE reached_name IN ('campaign_operations_production_enabler','campaign_operations_production_disabler','campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional','campaign_operations_production_reader','campaign_operations_scheduler_protocol_evidence_reader')
 GROUP BY login_name)
SELECT 'H2A006:login:'||login_name||':'||combo
FROM combos
WHERE combo NOT IN (
 'campaign_operations_production_reader',
 'campaign_operations_scheduler_protocol_evidence_reader',
 'campaign_operations_production_reader,campaign_operations_scheduler_protocol_evidence_reader',
 'campaign_operations_production_enabler,campaign_operations_production_reader,campaign_operations_scheduler_protocol_evidence_reader',
 'campaign_operations_production_disabler,campaign_operations_production_reader',
 'campaign_operations_production_dispatcher,campaign_operations_production_phase5_transactional,campaign_operations_production_reader,campaign_operations_scheduler_protocol_evidence_reader')
LIMIT 1;

WITH RECURSIVE role_reach(login_oid,reached_oid,path,depth) AS (
 SELECT member,roleid,ARRAY[member,roleid]::oid[],1
 FROM pg_catalog.pg_auth_members membership
 JOIN pg_catalog.pg_roles login ON login.oid=membership.member
 WHERE login.rolcanlogin AND login.rolinherit
 UNION ALL
 SELECT reach.login_oid,membership.roleid,
        reach.path||membership.roleid,reach.depth+1
 FROM role_reach reach
 JOIN pg_catalog.pg_auth_members membership
   ON membership.member=reach.reached_oid
 WHERE reach.depth < 1024
   AND NOT membership.roleid=ANY(reach.path)
), effective AS (
 SELECT DISTINCT login.rolname AS login_name,reached.rolname AS reached_name
 FROM role_reach reach
 JOIN pg_catalog.pg_roles login ON login.oid=reach.login_oid
 JOIN pg_catalog.pg_roles reached ON reached.oid=reach.reached_oid
)
SELECT 'H2A006:forbidden-effective-role:'||login_name||':'||reached_name
FROM effective
WHERE reached_name IN (
 'campaign_operations_h1_boundary_authority','campaign_operations_owner',
 'campaign_operations_scheduler_protocol_evidence_owner',
 'campaign_operations_dispatcher','campaign_operations_phase5_transactional',
 'campaign_operations_recovery','campaign_operations_cancellation_coordinator',
 'campaign_operations_completion_writer','pqxx')
  AND login_name IN (
   SELECT login_name FROM effective
   WHERE reached_name IN ('campaign_operations_production_dispatcher',
     'campaign_operations_production_phase5_transactional',
     'campaign_operations_production_enabler',
     'campaign_operations_production_disabler'))
LIMIT 1;

SELECT 'H2A003:raw-scheduler-privilege:'||pg_get_userbyid(a.grantee)
FROM pg_catalog.pg_class c
CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault('r',c.relowner))) a
WHERE c.relname='experiment_scheduler_protocol'
  AND pg_get_userbyid(a.grantee) IN ('campaign_operations_production_enabler','campaign_operations_production_disabler','campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional')
LIMIT 1;

SELECT 'H2A003:evidence-dml:'||c.relname||':'||pg_get_userbyid(a.grantee)||':'||a.privilege_type
FROM pg_catalog.pg_class c
CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault('r',c.relowner))) a
WHERE c.relname IN ('campaign_operations_production_enablement_event','campaign_operations_production_enablement_audit_reference_event','campaign_operations_request_production_admission')
  AND pg_get_userbyid(a.grantee) IN ('campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional')
  AND a.privilege_type <> 'SELECT'
LIMIT 1;

SELECT 'H2A003:protected-table-dml:'||n.nspname||'.'||c.relname||':'||pg_get_userbyid(a.grantee)||':'||a.privilege_type
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
JOIN pg_catalog.pg_roles owner_role ON owner_role.oid=c.relowner
CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault('r',c.relowner))) a
WHERE owner_role.rolname='campaign_operations_h1_boundary_authority'
  AND pg_get_userbyid(a.grantee) IN ('campaign_operations_production_enabler','campaign_operations_production_disabler','campaign_operations_production_dispatcher','campaign_operations_production_phase5_transactional')
  AND a.privilege_type IN ('INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER')
LIMIT 1;
