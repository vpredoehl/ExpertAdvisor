-- ADR-0019B executable ACL manifest, version h1-acl-manifest-v1.
-- Read-only psql input.  Success emits no rows.  Every emitted row is a
-- stable, complete expected-minus-actual or actual-minus-expected mismatch.
-- The deployment audit and migration suite invoke this same file.

WITH
object_manifest(object_class,schema_name,object_name,object_owner,
                acldefault_type,acl_is_null,requirement_id) AS (VALUES
 ('schema','public','public','pg_database_owner','n'::"char",false,'H1-ACL-SCHEMA-PUBLIC'),
 ('table','public','public.schema_migrations','vjp','r'::"char",false,'H1-ACL-LEDGER'),
 ('table','public','public.experiment_scheduler_protocol','vjp','r'::"char",false,'H1-ACL-SCHEDULER-EVIDENCE'),
 ('table','public','public.campaign_operations_production_transition_context','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-CONTEXT'),
 ('table','public','public.campaign_operations_production_enablement_event','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-ENABLEMENT'),
 ('table','public','public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-ENABLEMENT-AUDIT'),
 ('table','public','public.campaign_operations_request_production_admission','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-ADMISSION'),
 ('table','public','public.campaign_operations_operational_request','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-REQUEST'),
 ('table','public','public.campaign_operations_dispatch_attempt','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-ATTEMPT'),
 ('table','public','public.campaign_operations_dispatch_audit_reference_event','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-DISPATCH-AUDIT'),
 ('table','public','public.campaign_operations_completion_event','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-COMPLETION'),
 ('table','public','public.campaign_operations_completion_audit_reference_event','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-COMPLETION-AUDIT'),
 ('view','public','public.campaign_operations_production_readiness_v1','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-READINESS'),
 ('view','public','public.campaign_operations_production_status_v1','campaign_operations_h1_boundary_authority','r'::"char",false,'H1-ACL-STATUS'),
 ('sequence','public','public.campaign_operations_completio_completion_audit_reference_ev_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_completion_event_completion_event_id_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_dispatch__dispatch_audit_reference_even_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_operational_requ_operational_request_id_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_productio_production_enablement_audit_r_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_productio_production_enablement_event_i_seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('sequence','public','public.campaign_operations_request_p_request_production_admission__seq','campaign_operations_h1_boundary_authority','S'::"char",false,'H1-ACL-SEQUENCES'),
 ('function','public','public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-DEPLOYMENT-AUDIT'),
 ('function','public','public.campaign_operations_has_explicit_role_v1(name,name)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.campaign_operations_isolated_v1_authority_valid()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.campaign_operations_manager_build_canonical_v1(text,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CANONICAL'),
 ('function','public','public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-REPLAY'),
 ('function','public','public.campaign_operations_production_enablement_history_valid_v1(bigint)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-REPLAY'),
 ('function','public','public.campaign_operations_production_context_valid_v1(text,bigint,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CONTEXT'),
 ('function','public','public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-REPLAY'),
 ('function','public','public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-REPLAY'),
 ('function','public','public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CANONICAL'),
 ('function','public','public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CANONICAL'),
 ('function','public','public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public','public.campaign_operations_scheduler_protocol_evidence_lock_v1()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public','public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public','public.campaign_operations_tagged_fnv1a64(text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CANONICAL'),
 ('function','public','public.enforce_campaign_operations_complete_binding()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_completion_audit_complete()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_completion_event()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_dispatch_acquisition_complete()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_phase4_request_transition_complete()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_production_admission_consistent()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_production_context_empty_v1()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-CONTEXT'),
 ('function','public','public.enforce_campaign_operations_request()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_operations_request_acquisition_complete()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.enforce_campaign_ops_enablement_audit_complete()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_attempt_v1_isolation()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_completed_campaign()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_completion_v2_evidence()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_dispatch_control()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_production_admission_witness()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_production_attempt_mutation()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_production_dispatch_audit_insert()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_production_dispatch_audit_mutation()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.guard_campaign_operations_production_post_completion()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.lock_campaign_operations_authorization_head(bigint,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-LOCK-1'),
 ('function','public','public.lock_campaign_operations_budget_head(bigint)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-LOCK-2'),
 ('function','public','public.lock_campaign_operations_campaign(bigint)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-LOCK-3'),
 ('function','public','public.lock_campaign_operations_request(bigint)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-LOCK-5'),
 ('function','public','public.lock_campaign_operations_reservation(bigint)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-LOCK-4'),
 ('function','public','public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-FIXED-TRANSITIONS'),
 ('function','public','public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-FIXED-TRANSITIONS'),
 ('function','public','public.reject_campaign_operations_completion_mutation()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.reject_campaign_operations_production_mutation()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-FIXED-TRANSITIONS'),
 ('function','public','public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-ATTEMPT-V1'),
 ('function','public','public.validate_campaign_operations_dispatch_attempt_v2_insert()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.validate_campaign_operations_production_enablement_insert()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS'),
 ('function','public','public.validate_campaign_ops_production_admission_insert()','campaign_operations_h1_boundary_authority','f'::"char",false,'H1-ACL-SUPPORTING-DEFINERS')
),
coupled_types AS (
 SELECT CASE WHEN t.typtype='d' THEN 'domain' ELSE 'type' END AS object_class,
        n.nspname AS schema_name,
        quote_ident(n.nspname)||'.'||quote_ident(t.typname) AS object_name,
        'campaign_operations_h1_boundary_authority'::text AS object_owner,
        'T'::"char" AS acldefault_type,true AS acl_is_null,
        'H1-ACL-COUPLED-TYPES'::text AS requirement_id
 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
 JOIN pg_roles owner_role ON owner_role.oid=t.typowner
 WHERE owner_role.rolname='campaign_operations_h1_boundary_authority'
   AND n.nspname='public'
   AND t.typname IN (
     '_campaign_operations_completion_audit_reference_event',
     '_campaign_operations_completion_event',
     '_campaign_operations_dispatch_attempt',
     '_campaign_operations_dispatch_audit_reference_event',
     '_campaign_operations_operational_request',
     '_campaign_operations_production_enablement_audit_reference_even',
     '_campaign_operations_production_enablement_event',
     '_campaign_operations_production_readiness_v1',
     '_campaign_operations_production_status_v1',
     '_campaign_operations_production_transition_context',
     '_campaign_operations_request_production_admission',
     'campaign_operations_completion_audit_reference_event',
     'campaign_operations_completion_event',
     'campaign_operations_dispatch_attempt',
     'campaign_operations_dispatch_audit_reference_event',
     'campaign_operations_operational_request',
     'campaign_operations_production_enablement_audit_reference_event',
     'campaign_operations_production_enablement_event',
     'campaign_operations_production_readiness_v1',
     'campaign_operations_production_status_v1',
     'campaign_operations_production_transition_context',
     'campaign_operations_request_production_admission')
   AND EXISTS (SELECT 1 FROM pg_depend d
               WHERE (d.objid=t.oid OR d.refobjid=t.oid)
                 AND (d.classid='pg_type'::regclass OR
                      d.refclassid='pg_type'::regclass))
),
all_objects AS (SELECT * FROM object_manifest UNION ALL SELECT * FROM coupled_types),
owner_expected AS (
 SELECT object_class,current_database() AS database_name,schema_name,object_name,
        object_owner,CASE acl.grantee WHEN 0 THEN 'PUBLIC'
          ELSE pg_get_userbyid(acl.grantee) END AS grantee,
        acl.privilege_type,acl.is_grantable,acl_is_null AS derived_from_null_acl,
        acldefault_type,requirement_id
 FROM all_objects o JOIN pg_roles owner_role ON owner_role.rolname=o.object_owner
 CROSS JOIN LATERAL aclexplode(acldefault(o.acldefault_type,owner_role.oid)) acl
 WHERE o.acl_is_null OR acl.grantee=owner_role.oid
),
extra_acl(object_class,object_name,grantee,privilege_type,requirement_id) AS (VALUES
 ('schema','public','PUBLIC','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_auditor','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_cancellation_coordinator','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_completion_writer','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_controller','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_production_reader','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_reader','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_reconciler','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_recovery','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','campaign_operations_scheduler_protocol_evidence_reader','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','experiment_lifecycle_cancellation','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('schema','public','experiment_lifecycle_cancellation_owner','USAGE','H1-ACL-SCHEMA-PUBLIC'),
 ('table','public.schema_migrations','campaign_operations_owner','SELECT','H1-ACL-LEDGER'),
 ('table','public.experiment_scheduler_protocol','pqxx','SELECT','H1-ACL-SCHEDULER-EVIDENCE'),
 ('table','public.campaign_operations_production_enablement_event','campaign_operations_owner','SELECT','H1-ACL-ENABLEMENT'),
 ('table','public.campaign_operations_production_enablement_event','campaign_operations_production_reader','SELECT','H1-ACL-ENABLEMENT'),
 ('table','public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_owner','SELECT','H1-ACL-ENABLEMENT-AUDIT'),
 ('table','public.campaign_operations_production_enablement_audit_reference_event','campaign_operations_production_reader','SELECT','H1-ACL-ENABLEMENT-AUDIT'),
 ('table','public.campaign_operations_request_production_admission','campaign_operations_owner','SELECT','H1-ACL-ADMISSION'),
 ('table','public.campaign_operations_request_production_admission','campaign_operations_production_reader','SELECT','H1-ACL-ADMISSION'),
 ('view','public.campaign_operations_production_readiness_v1','campaign_operations_production_reader','SELECT','H1-ACL-READINESS'),
 ('view','public.campaign_operations_production_status_v1','campaign_operations_production_reader','SELECT','H1-ACL-STATUS'),
 ('table','public.campaign_operations_operational_request','campaign_operations_owner','SELECT','H1-ACL-REQUEST'),
 ('table','public.campaign_operations_operational_request','campaign_operations_cancellation_coordinator','SELECT','H1-ACL-REQUEST'),
 ('table','public.campaign_operations_operational_request','campaign_operations_completion_writer','SELECT','H1-ACL-REQUEST'),
 ('table','public.campaign_operations_operational_request','campaign_operations_reconciler','SELECT','H1-ACL-REQUEST'),
 ('table','public.campaign_operations_operational_request','campaign_operations_recovery','SELECT','H1-ACL-REQUEST'),
 ('table','public.campaign_operations_dispatch_attempt','campaign_operations_owner','SELECT','H1-ACL-ATTEMPT'),
 ('table','public.campaign_operations_dispatch_attempt','campaign_operations_cancellation_coordinator','SELECT','H1-ACL-ATTEMPT'),
 ('table','public.campaign_operations_dispatch_attempt','campaign_operations_completion_writer','SELECT','H1-ACL-ATTEMPT'),
 ('table','public.campaign_operations_dispatch_attempt','campaign_operations_reconciler','SELECT','H1-ACL-ATTEMPT'),
 ('table','public.campaign_operations_dispatch_attempt','campaign_operations_recovery','SELECT','H1-ACL-ATTEMPT'),
 ('table','public.campaign_operations_dispatch_audit_reference_event','campaign_operations_owner','SELECT','H1-ACL-DISPATCH-AUDIT'),
 ('table','public.campaign_operations_dispatch_audit_reference_event','campaign_operations_recovery','SELECT','H1-ACL-DISPATCH-AUDIT'),
 ('table','public.campaign_operations_completion_event','campaign_operations_owner','SELECT','H1-ACL-COMPLETION'),
 ('table','public.campaign_operations_completion_event','campaign_operations_auditor','SELECT','H1-ACL-COMPLETION'),
 ('table','public.campaign_operations_completion_event','campaign_operations_completion_writer','SELECT','H1-ACL-COMPLETION'),
 ('table','public.campaign_operations_completion_event','campaign_operations_reader','SELECT','H1-ACL-COMPLETION'),
 ('table','public.campaign_operations_completion_audit_reference_event','campaign_operations_owner','SELECT','H1-ACL-COMPLETION-AUDIT'),
 ('table','public.campaign_operations_completion_audit_reference_event','campaign_operations_auditor','SELECT','H1-ACL-COMPLETION-AUDIT'),
 ('table','public.campaign_operations_completion_audit_reference_event','campaign_operations_completion_writer','SELECT','H1-ACL-COMPLETION-AUDIT'),
 ('table','public.campaign_operations_completion_audit_reference_event','campaign_operations_reader','SELECT','H1-ACL-COMPLETION-AUDIT'),
 ('sequence','public.campaign_operations_completion_event_completion_event_id_seq','campaign_operations_completion_writer','USAGE','H1-ACL-SEQUENCES'),
 ('sequence','public.campaign_operations_completio_completion_audit_reference_ev_seq','campaign_operations_completion_writer','USAGE','H1-ACL-SEQUENCES'),
 ('sequence','public.campaign_operations_dispatch__dispatch_audit_reference_even_seq','campaign_operations_cancellation_coordinator','USAGE','H1-ACL-SEQUENCES'),
 ('sequence','public.campaign_operations_dispatch__dispatch_audit_reference_even_seq','campaign_operations_recovery','USAGE','H1-ACL-SEQUENCES'),
 ('function','public.campaign_operations_scheduler_protocol_evidence_lock_v1()','campaign_operations_scheduler_protocol_evidence_reader','EXECUTE','H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()','campaign_operations_scheduler_protocol_evidence_reader','EXECUTE','H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()','campaign_operations_owner','EXECUTE','H1-ACL-SCHEDULER-EVIDENCE'),
 ('function','public.campaign_operations_tagged_fnv1a64(text)','campaign_operations_owner','EXECUTE','H1-ACL-CANONICAL'),
 ('function','public.lock_campaign_operations_authorization_head(bigint,text)','campaign_operations_owner','EXECUTE','H1-ACL-LOCK-1'),
 ('function','public.lock_campaign_operations_authorization_head(bigint,text)','campaign_operations_completion_writer','EXECUTE','H1-ACL-LOCK-1'),
 ('function','public.lock_campaign_operations_budget_head(bigint)','campaign_operations_owner','EXECUTE','H1-ACL-LOCK-2'),
 ('function','public.lock_campaign_operations_budget_head(bigint)','campaign_operations_cancellation_coordinator','EXECUTE','H1-ACL-LOCK-2'),
 ('function','public.lock_campaign_operations_budget_head(bigint)','campaign_operations_completion_writer','EXECUTE','H1-ACL-LOCK-2'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_owner','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_cancellation_coordinator','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_completion_writer','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_controller','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_reconciler','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_campaign(bigint)','campaign_operations_recovery','EXECUTE','H1-ACL-LOCK-3'),
 ('function','public.lock_campaign_operations_reservation(bigint)','campaign_operations_owner','EXECUTE','H1-ACL-LOCK-4'),
 ('function','public.lock_campaign_operations_reservation(bigint)','campaign_operations_cancellation_coordinator','EXECUTE','H1-ACL-LOCK-4'),
 ('function','public.lock_campaign_operations_reservation(bigint)','campaign_operations_completion_writer','EXECUTE','H1-ACL-LOCK-4'),
 ('function','public.lock_campaign_operations_reservation(bigint)','campaign_operations_recovery','EXECUTE','H1-ACL-LOCK-4'),
 ('function','public.lock_campaign_operations_request(bigint)','campaign_operations_owner','EXECUTE','H1-ACL-LOCK-5'),
 ('function','public.lock_campaign_operations_request(bigint)','campaign_operations_cancellation_coordinator','EXECUTE','H1-ACL-LOCK-5'),
 ('function','public.lock_campaign_operations_request(bigint)','campaign_operations_completion_writer','EXECUTE','H1-ACL-LOCK-5'),
 ('function','public.lock_campaign_operations_request(bigint)','campaign_operations_reconciler','EXECUTE','H1-ACL-LOCK-5'),
 ('function','public.lock_campaign_operations_request(bigint)','campaign_operations_recovery','EXECUTE','H1-ACL-LOCK-5')
),
extra_expected AS (
 SELECT e.object_class,current_database(),o.schema_name,e.object_name,o.object_owner,
        e.grantee,e.privilege_type,false,false,o.acldefault_type,e.requirement_id
 FROM extra_acl e JOIN all_objects o USING(object_class,object_name)
),
all_column_grants(table_name,grantee,privilege_type,column_names,requirement_id) AS (VALUES
 ('campaign_operations_completion_event','campaign_operations_completion_writer','INSERT',NULL::text[],'H1-ACL-COMPLETION-COLUMNS'),
 ('campaign_operations_completion_audit_reference_event','campaign_operations_completion_writer','INSERT',NULL::text[],'H1-ACL-COMPLETION-AUDIT-COLUMNS'),
 ('campaign_operations_operational_request','campaign_operations_owner','UPDATE',ARRAY['request_state','state_version','lease_token_hash','lease_expires_at','dispatcher_identity','updated_at'],'H1-ACL-REQUEST-COLUMNS'),
 ('campaign_operations_operational_request','campaign_operations_h1_boundary_authority','UPDATE',ARRAY['request_state','state_version','lease_token_hash','lease_expires_at','dispatcher_identity','updated_at'],'H1-ACL-REQUEST-COLUMNS'),
 ('campaign_operations_dispatch_audit_reference_event','campaign_operations_cancellation_coordinator','INSERT',ARRAY['operational_campaign_id','operational_request_id','dispatch_attempt_id','dispatch_attempt_outcome_id','prior_version','resulting_version','actor_identity','capability','cause_kind','outcome','replay_disposition','diagnostic_code'],'H1-ACL-DISPATCH-AUDIT-COLUMNS'),
 ('campaign_operations_dispatch_audit_reference_event','campaign_operations_recovery','INSERT',ARRAY['operational_campaign_id','operational_request_id','dispatch_attempt_id','dispatch_attempt_outcome_id','prior_version','resulting_version','actor_identity','capability','cause_kind','outcome','replay_disposition','diagnostic_code'],'H1-ACL-DISPATCH-AUDIT-COLUMNS'),
 ('experiment_scheduler_protocol','pqxx','UPDATE',ARRAY['cutover_state','cutover_completed_at','cutover_completed_by','cutover_executable_path','cutover_process_evidence','failure_diagnostic','updated_at'],'H1-ACL-SCHEDULER-COLUMNS')
),
column_expected AS (
 SELECT 'column'::text,current_database(),'public'::text,
        'public.'||quote_ident(g.table_name)||'.'||quote_ident(a.attname),
        pg_get_userbyid(c.relowner),g.grantee,g.privilege_type,false,false,
        NULL::"char",g.requirement_id
 FROM all_column_grants g JOIN pg_class c ON c.oid=('public.'||g.table_name)::regclass
 JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
 WHERE (g.column_names IS NULL AND
        NOT (g.table_name='campaign_operations_completion_event' AND
             a.attname IN ('completion_event_id','recorded_at')) AND
        NOT (g.table_name='campaign_operations_completion_audit_reference_event' AND
             a.attname IN ('completion_audit_reference_event_id','created_at')))
    OR a.attname=ANY(g.column_names)
),
expected AS (SELECT * FROM owner_expected UNION ALL SELECT * FROM extra_expected
             UNION ALL SELECT * FROM column_expected),
actual_objects AS (
 SELECT 'schema'::text AS object_class,current_database() AS database_name,
        n.nspname AS schema_name,quote_ident(n.nspname) AS object_name,
        pg_get_userbyid(n.nspowner) AS object_owner,n.nspacl AS acl,
        'n'::"char" AS acldefault_type
 FROM pg_namespace n WHERE n.nspname='public'
 UNION ALL
 SELECT CASE c.relkind WHEN 'S' THEN 'sequence' WHEN 'v' THEN 'view'
          WHEN 'm' THEN 'materialized_view' WHEN 'p' THEN 'partitioned_table'
          WHEN 'f' THEN 'foreign_table' ELSE 'table' END,current_database(),
        n.nspname,quote_ident(n.nspname)||'.'||quote_ident(c.relname),
        pg_get_userbyid(c.relowner),c.relacl,
        CASE WHEN c.relkind='S' THEN 'S'::"char" ELSE 'r'::"char" END
 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname='public'
   AND c.relkind IN ('r','p','S','v','m','f')
   AND (c.relowner=(SELECT oid FROM pg_roles
                    WHERE rolname='campaign_operations_h1_boundary_authority')
        OR c.relname IN ('schema_migrations','experiment_scheduler_protocol'))
 UNION ALL
 SELECT CASE p.prokind WHEN 'p' THEN 'procedure' WHEN 'a' THEN 'aggregate'
          ELSE 'function' END,current_database(),n.nspname,
        quote_ident(n.nspname)||'.'||p.oid::regprocedure::text,
        pg_get_userbyid(p.proowner),p.proacl,'f'::"char"
 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 WHERE p.proowner=(SELECT oid FROM pg_roles
                   WHERE rolname='campaign_operations_h1_boundary_authority')
 UNION ALL
 SELECT CASE WHEN t.typtype='d' THEN 'domain' ELSE 'type' END,current_database(),
        n.nspname,quote_ident(n.nspname)||'.'||quote_ident(t.typname),
        pg_get_userbyid(t.typowner),t.typacl,'T'::"char"
 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
 JOIN pg_roles owner_role ON owner_role.oid=t.typowner
 WHERE owner_role.rolname='campaign_operations_h1_boundary_authority'
   AND n.nspname='public'
),
actual_object_acl AS (
 SELECT o.object_class,o.database_name,o.schema_name,o.object_name,o.object_owner,
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
        acl.privilege_type,acl.is_grantable,
        o.acl IS NULL,
        o.acldefault_type,
        coalesce(m.requirement_id,'H1-ACL-UNEXPECTED-OBJECT')
 FROM actual_objects o LEFT JOIN all_objects m USING(object_class,object_name)
 JOIN pg_roles owner_role ON owner_role.rolname=o.object_owner
 CROSS JOIN LATERAL aclexplode(coalesce(o.acl,acldefault(o.acldefault_type,owner_role.oid))) acl
),
actual_columns AS (
 SELECT 'column'::text,current_database(),n.nspname,
        quote_ident(n.nspname)||'.'||quote_ident(c.relname)||'.'||quote_ident(a.attname),
        pg_get_userbyid(c.relowner),CASE acl.grantee WHEN 0 THEN 'PUBLIC'
          ELSE pg_get_userbyid(acl.grantee) END,acl.privilege_type,
        acl.is_grantable,false,NULL::"char",
        CASE c.relname
          WHEN 'campaign_operations_completion_event' THEN 'H1-ACL-COMPLETION-COLUMNS'
          WHEN 'campaign_operations_completion_audit_reference_event' THEN 'H1-ACL-COMPLETION-AUDIT-COLUMNS'
          WHEN 'campaign_operations_dispatch_audit_reference_event' THEN 'H1-ACL-DISPATCH-AUDIT-COLUMNS'
          WHEN 'campaign_operations_operational_request' THEN 'H1-ACL-REQUEST-COLUMNS'
          WHEN 'experiment_scheduler_protocol' THEN 'H1-ACL-SCHEDULER-COLUMNS'
          ELSE 'H1-ACL-COLUMNS' END
 FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
 JOIN pg_namespace n ON n.oid=c.relnamespace
 CROSS JOIN LATERAL aclexplode(a.attacl) acl
 WHERE n.nspname='public' AND a.attnum>0 AND NOT a.attisdropped
   AND (c.relowner=(SELECT oid FROM pg_roles
                    WHERE rolname='campaign_operations_h1_boundary_authority')
        OR c.relname IN ('experiment_scheduler_protocol'))
),
actual AS (SELECT * FROM actual_object_acl UNION ALL SELECT * FROM actual_columns),
mismatches AS (
 (SELECT 'expected_minus_actual' AS difference,e.* FROM expected e
  EXCEPT ALL SELECT 'expected_minus_actual',a.* FROM actual a)
 UNION ALL
 (SELECT 'actual_minus_expected',a.* FROM actual a
  EXCEPT ALL SELECT 'actual_minus_expected',e.* FROM expected e)
)
SELECT concat_ws('|','H1A006','explicit_acl',difference,requirement_id,
       object_class,database_name,schema_name,object_name,object_owner,grantee,
       privilege_type,is_grantable,derived_from_null_acl,acldefault_type)
FROM mismatches ORDER BY difference,object_class,object_name,grantee,privilege_type;

-- Column ACL origin is a catalog-state contract even where the owner already
-- has the same effective privilege implicitly.  These sentinels cover both
-- directions without deriving the scan universe from the grant tuples.
WITH column_origin_manifest(table_name,column_name,acl_is_null,requirement_id) AS (VALUES
 ('campaign_operations_production_transition_context','backend_pid',true,
  'H1-ACL-CONTEXT-COLUMN-ORIGIN'),
 ('campaign_operations_operational_request','request_state',false,
  'H1-ACL-REQUEST-COLUMNS')
), actual AS (
 SELECT m.*,a.attacl IS NULL AS actual_acl_is_null
 FROM column_origin_manifest m
 LEFT JOIN pg_class c ON c.oid=('public.'||m.table_name)::regclass
 LEFT JOIN pg_attribute a ON a.attrelid=c.oid AND a.attname=m.column_name
)
SELECT concat_ws('|','H1A006','column_acl_origin','origin_mismatch',
       requirement_id,'column',current_database(),'public',
       'public.'||quote_ident(table_name)||'.'||quote_ident(column_name),
       acl_is_null,actual_acl_is_null)
FROM actual
WHERE actual_acl_is_null IS DISTINCT FROM acl_is_null
ORDER BY table_name,column_name;

WITH
owners(owner_role) AS (VALUES
 ('campaign_operations_h1_boundary_authority'),('campaign_operations_owner'),
 ('campaign_operations_scheduler_protocol_evidence_owner')),
state_manifest(scope_name,defaclobjtype,row_state,requirement_id) AS (VALUES
 ('<global>','f'::"char",'explicit','H1-DEFAULT-GLOBAL-FUNCTION'),
 ('<global>','r'::"char",'null','H1-DEFAULT-GLOBAL-TABLE'),
 ('<global>','S'::"char",'null','H1-DEFAULT-GLOBAL-SEQUENCE'),
 ('<global>','T'::"char",'explicit','H1-DEFAULT-GLOBAL-TYPE'),
 ('<global>','n'::"char",'null','H1-DEFAULT-GLOBAL-SCHEMA'),
 ('public','f'::"char",'null','H1-DEFAULT-PUBLIC-FUNCTION'),
 ('public','r'::"char",'null','H1-DEFAULT-PUBLIC-TABLE'),
 ('public','S'::"char",'null','H1-DEFAULT-PUBLIC-SEQUENCE'),
 ('public','T'::"char",'null','H1-DEFAULT-PUBLIC-TYPE')),
states AS (SELECT * FROM owners CROSS JOIN state_manifest),
expected AS (
 SELECT s.owner_role,s.scope_name,s.defaclobjtype,
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END grantee,
        acl.privilege_type,acl.is_grantable,s.row_state,
        acldefault(s.defaclobjtype,owner_role.oid)::text AS acldefault_expansion,
        s.requirement_id
 FROM states s JOIN pg_roles owner_role ON owner_role.rolname=s.owner_role
 CROSS JOIN LATERAL aclexplode(CASE WHEN s.row_state='explicit' THEN
   CASE s.defaclobjtype WHEN 'f' THEN ARRAY[(quote_ident(s.owner_role)||'=X/'||quote_ident(s.owner_role))::aclitem]
        WHEN 'T' THEN ARRAY[(quote_ident(s.owner_role)||'=U/'||quote_ident(s.owner_role))::aclitem] END
   ELSE acldefault(s.defaclobjtype,owner_role.oid) END) acl
),
actual_known_states AS (
 SELECT s.owner_role,s.scope_name,s.defaclobjtype,
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
        acl.privilege_type,acl.is_grantable,
        CASE WHEN d.oid IS NULL THEN 'null' ELSE 'explicit' END,
        acldefault(s.defaclobjtype,owner_role.oid)::text,s.requirement_id
 FROM states s JOIN pg_roles owner_role ON owner_role.rolname=s.owner_role
 LEFT JOIN pg_namespace n ON n.nspname=s.scope_name AND s.scope_name<>'<global>'
 LEFT JOIN pg_default_acl d ON d.defaclrole=owner_role.oid
  AND d.defaclnamespace=CASE WHEN s.scope_name='<global>' THEN 0 ELSE n.oid END
  AND d.defaclobjtype=s.defaclobjtype
 CROSS JOIN LATERAL aclexplode(coalesce(d.defaclacl,
   acldefault(s.defaclobjtype,owner_role.oid))) acl
),
unexpected_states AS (
 SELECT owner_role.rolname,coalesce(n.nspname,'<global>'),d.defaclobjtype,
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
        acl.privilege_type,acl.is_grantable,'explicit',
        acldefault(d.defaclobjtype,d.defaclrole)::text,'H1-DEFAULT-UNEXPECTED-ROW'
 FROM pg_default_acl d JOIN pg_roles owner_role ON owner_role.oid=d.defaclrole
 LEFT JOIN pg_namespace n ON n.oid=d.defaclnamespace
 CROSS JOIN LATERAL aclexplode(d.defaclacl) acl
 WHERE owner_role.rolname IN (SELECT owner_role FROM owners)
   AND NOT EXISTS (SELECT 1 FROM states s WHERE s.owner_role=owner_role.rolname
     AND s.scope_name=coalesce(n.nspname,'<global>')
     AND s.defaclobjtype=d.defaclobjtype)
),
actual AS (SELECT * FROM actual_known_states UNION ALL SELECT * FROM unexpected_states),
mismatches AS (
 (SELECT 'expected_minus_actual' AS difference,e.* FROM expected e
  EXCEPT ALL SELECT 'expected_minus_actual',a.* FROM actual a)
 UNION ALL
 (SELECT 'actual_minus_expected',a.* FROM actual a
  EXCEPT ALL SELECT 'actual_minus_expected',e.* FROM expected e))
SELECT concat_ws('|','H1A007','default_acl',difference,requirement_id,
       owner_role,scope_name,defaclobjtype,grantee,privilege_type,is_grantable,
       row_state,acldefault_expansion)
FROM mismatches ORDER BY difference,owner_role,scope_name,defaclobjtype,grantee,privilege_type;
