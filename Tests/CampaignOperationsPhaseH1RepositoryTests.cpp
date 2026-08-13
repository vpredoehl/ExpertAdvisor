#include "../Sources/CampaignOperationsProductionAdmissionRepository.hpp"
#include "../Sources/CampaignOperationsProductionAdmissionService.hpp"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

using namespace EA::CampaignOperations;

namespace
{

enum class HydrationTarget { enablement, admission, attempt };

std::string AttemptRowJson(pqxx::transaction_base& transaction,
    long long attemptId)
{
    return transaction.exec(
        "SELECT coalesce((SELECT to_jsonb(attempt)::text FROM "
        "campaign_operations_dispatch_attempt attempt WHERE "
        "dispatch_attempt_id=$1),'missing');",
        pqxx::params{attemptId}).one_row()[0].as<std::string>();
}

std::string AttemptEvidenceJson(pqxx::transaction_base& transaction,
    long long attemptId)
{
    return transaction.exec(
        "SELECT jsonb_build_object('attempt',(SELECT to_jsonb(attempt) FROM "
        "campaign_operations_dispatch_attempt attempt WHERE "
        "dispatch_attempt_id=$1),'audits',coalesce((SELECT jsonb_agg("
        "to_jsonb(audit) ORDER BY audit.dispatch_audit_reference_event_id) "
        "FROM campaign_operations_dispatch_audit_reference_event audit "
        "WHERE audit.dispatch_attempt_id=$1),'[]'::jsonb))::text;",
        pqxx::params{attemptId}).one_row()[0].as<std::string>();
}

std::string ReadinessEvidenceJson(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT jsonb_build_object("
        "'migration',(SELECT to_jsonb(migration) FROM schema_migrations "
        "migration WHERE version='055'),"
        "'scheduler',(SELECT to_jsonb(scheduler) FROM "
        "experiment_scheduler_protocol scheduler WHERE scheduler.singleton),"
        "'enablement',coalesce((SELECT jsonb_agg(to_jsonb(enablement) "
        "ORDER BY enablement.production_enablement_event_id) FROM "
        "campaign_operations_production_enablement_event enablement),"
        "'[]'::jsonb),"
        "'enablement_audit',coalesce((SELECT jsonb_agg(to_jsonb(audit) "
        "ORDER BY audit.production_enablement_audit_reference_event_id) FROM "
        "campaign_operations_production_enablement_audit_reference_event audit),"
        "'[]'::jsonb),"
        "'admission',coalesce((SELECT jsonb_agg(to_jsonb(admission) "
        "ORDER BY admission.request_production_admission_id) FROM "
        "campaign_operations_request_production_admission admission),"
        "'[]'::jsonb),"
        "'attempt',coalesce((SELECT jsonb_agg(to_jsonb(attempt) "
        "ORDER BY attempt.dispatch_attempt_id) FROM "
        "campaign_operations_dispatch_attempt attempt),"
        "'[]'::jsonb),"
        "'audit',coalesce((SELECT jsonb_agg(to_jsonb(audit) "
        "ORDER BY audit.dispatch_audit_reference_event_id) FROM "
        "campaign_operations_dispatch_audit_reference_event audit),"
        "'[]'::jsonb),"
        "'request',coalesce((SELECT jsonb_agg(to_jsonb(request) "
        "ORDER BY request.operational_request_id) FROM "
        "campaign_operations_operational_request request),"
        "'[]'::jsonb),"
        "'completion',coalesce((SELECT jsonb_agg(to_jsonb(completion) "
        "ORDER BY completion.completion_event_id) FROM "
        "campaign_operations_completion_event completion),'[]'::jsonb),"
        "'completion_audit',coalesce((SELECT jsonb_agg(to_jsonb(audit) "
        "ORDER BY audit.completion_audit_reference_event_id) FROM "
        "campaign_operations_completion_audit_reference_event audit),"
        "'[]'::jsonb),"
        "'reconciliation_observation',coalesce((SELECT jsonb_agg("
        "to_jsonb(observation) ORDER BY observation.reconciliation_observation_id) "
        "FROM campaign_operations_reconciliation_observation observation),"
        "'[]'::jsonb),"
        "'reconciliation_resolution',coalesce((SELECT jsonb_agg("
        "to_jsonb(resolution) ORDER BY resolution.reconciliation_resolution_id) "
        "FROM campaign_operations_reconciliation_resolution resolution),"
        "'[]'::jsonb),"
        // The readiness view directly derives Completion proof-version evidence
        // from matching check constraints.  Keep the semantic identity and
        // normalized definition, not OIDs or physical catalog details.
        "'catalog',(WITH RECURSIVE readiness_role_closure(role_oid) AS ("
        "SELECT role.oid FROM pg_catalog.pg_roles role "
        "WHERE role.rolname=session_user::name UNION SELECT membership.roleid "
        "FROM pg_catalog.pg_auth_members membership JOIN "
        "readiness_role_closure inherited ON "
        "inherited.role_oid=membership.member) SELECT jsonb_build_object("
        "'completion_constraints',coalesce((SELECT jsonb_agg("
        "jsonb_build_object('schema',namespace.nspname,'relation',"
        "relation.relname,'constraint_name',constraint_row.conname,"
        "'constraint_type',constraint_row.contype::text,'definition',"
        "pg_get_constraintdef(constraint_row.oid)) ORDER BY "
        "namespace.nspname,relation.relname,constraint_row.conname) "
        "FROM pg_catalog.pg_constraint constraint_row JOIN "
        "pg_catalog.pg_class relation ON relation.oid=constraint_row.conrelid "
        "JOIN pg_catalog.pg_namespace namespace ON "
        "namespace.oid=relation.relnamespace WHERE namespace.nspname='public' "
        "AND relation.relname='campaign_operations_completion_event' "
        "AND constraint_row.contype='c' AND "
        "pg_get_constraintdef(constraint_row.oid) LIKE "
        "'%completion_contract_version = %'),'[]'::jsonb),"
        // campaign_operations_has_explicit_role_v1 reads only role identity
        // and recursive membership edges; role attributes and membership
        // option flags do not participate in its result.
        "'roles',coalesce((SELECT jsonb_agg(jsonb_build_object("
        "'role',role_row.rolname) ORDER BY role_row.rolname) "
        "FROM pg_catalog.pg_roles role_row JOIN readiness_role_closure closure "
        "ON closure.role_oid=role_row.oid),'[]'::jsonb),"
        "'role_memberships',coalesce((SELECT jsonb_agg(jsonb_build_object("
        "'member_role',member_role.rolname,'granted_role',granted_role.rolname) "
        "ORDER BY member_role.rolname,granted_role.rolname) "
        "FROM pg_catalog.pg_auth_members membership JOIN "
        "readiness_role_closure closure ON closure.role_oid=membership.member "
        "JOIN pg_catalog.pg_roles member_role ON member_role.oid=membership.member "
        "JOIN pg_catalog.pg_roles granted_role ON granted_role.oid=membership.roleid),"
        "'[]'::jsonb))))::text;")
        .one_row()[0].as<std::string>();
}

void AssertRoleHelperCatalogSnapshotDependency(pqxx::connection& connection)
{
    pqxx::work transaction{connection};
    const auto before = ReadinessEvidenceJson(transaction);
    const auto beforeReadiness = LoadProductionReadinessSnapshot(transaction);
    assert(!beforeReadiness.readerMember);
    transaction.exec(
        "CREATE ROLE campaign_operations_h1_readiness_snapshot_role NOLOGIN;"
        "GRANT campaign_operations_production_reader TO "
        "campaign_operations_h1_readiness_snapshot_role;"
        "GRANT campaign_operations_h1_readiness_snapshot_role TO "
        "campaign_manager_login;");
    const auto after = ReadinessEvidenceJson(transaction);
    const auto afterReadiness = LoadProductionReadinessSnapshot(transaction);
    assert(after != before);
    assert(afterReadiness.readerMember);
    assert(transaction.exec(
        "SELECT ($1::jsonb->'catalog'->'role_memberships') @> $2::jsonb;",
        pqxx::params{after,
            "[{\"member_role\":\"campaign_manager_login\","
            "\"granted_role\":\"campaign_operations_h1_readiness_snapshot_role\"},"
            "{\"member_role\":\"campaign_operations_h1_readiness_snapshot_role\","
            "\"granted_role\":\"campaign_operations_production_reader\"}]"})
        .one_row()[0].as<bool>());
    // A membership edge outside session_user's recursive closure is not a
    // readiness dependency.  The relevant nested reader path above remains
    // present, while this genuinely unrelated edge must not enter the
    // catalog snapshot or alter the evaluated role proof.
    transaction.exec(
        "CREATE ROLE campaign_operations_h1_readiness_unrelated_member NOLOGIN;"
        "CREATE ROLE campaign_operations_h1_readiness_unrelated_granted NOLOGIN;"
        "GRANT campaign_operations_h1_readiness_unrelated_granted TO "
        "campaign_operations_h1_readiness_unrelated_member;");
    const auto afterUnrelated = ReadinessEvidenceJson(transaction);
    const auto afterUnrelatedReadiness = LoadProductionReadinessSnapshot(
        transaction);
    assert(afterUnrelated == after);
    assert(afterUnrelatedReadiness.readerMember);
    assert(afterUnrelatedReadiness.dispatcherMember ==
        afterReadiness.dispatcherMember);
    transaction.abort();
}

void AssertCompletionCatalogSnapshotNegativeControl(pqxx::connection& connection)
{
    pqxx::work transaction{connection};
    const auto before = ReadinessEvidenceJson(transaction);
    const auto beforeReadiness = LoadProductionReadinessSnapshot(transaction);
    // The collector must retain the actual Completion contract proof while
    // excluding an unrelated check constraint on the same relation.
    assert(transaction.exec(
        "SELECT jsonb_array_length($1::jsonb->'catalog'->"
        "'completion_constraints') > 0;", pqxx::params{before})
        .one_row()[0].as<bool>());
    transaction.exec(
        "ALTER TABLE campaign_operations_completion_event ADD CONSTRAINT "
        "h1_unrelated_completion_constraint_test CHECK "
        "(completion_event_id > 0);");
    assert(transaction.exec(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint WHERE "
        "conrelid='campaign_operations_completion_event'::regclass AND "
        "conname='h1_unrelated_completion_constraint_test');")
        .one_row()[0].as<bool>());
    const auto after = ReadinessEvidenceJson(transaction);
    const auto afterReadiness = LoadProductionReadinessSnapshot(transaction);
    assert(after == before);
    assert(afterReadiness.completionNestedV2ProofVersion ==
        beforeReadiness.completionNestedV2ProofVersion);
    assert(afterReadiness.completionNestedV2ProofValid ==
        beforeReadiness.completionNestedV2ProofValid);
    transaction.abort();
}

void AssertLoadedReadinessBlocks(pqxx::connection& connection,
    const std::string& mutationSql, const std::string& expectedOutput,
    bool expectCatalogSnapshotChange = false,
    const std::string& expectedCatalogConstraintName = {})
{
    pqxx::work transaction{connection};
    const auto before = expectCatalogSnapshotChange
        ? ReadinessEvidenceJson(transaction) : std::string{};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    transaction.exec(mutationSql);
    if (expectCatalogSnapshotChange)
    {
        assert(ReadinessEvidenceJson(transaction) != before);
        const auto catalogDefinition = transaction.exec(
            "SELECT constraint_row->>'definition' FROM jsonb_array_elements("
            "$1::jsonb->'catalog'->'completion_constraints') constraint_row "
            "WHERE constraint_row->>'constraint_name'=$2;",
            pqxx::params{ReadinessEvidenceJson(transaction),
                expectedCatalogConstraintName}).one_row()[0].as<std::string>();
        const auto expectedDefinition = transaction.exec(
            "SELECT pg_get_constraintdef(oid) FROM pg_catalog.pg_constraint "
            "WHERE conrelid='public.campaign_operations_completion_event'::regclass "
            "AND conname=$1;", pqxx::params{expectedCatalogConstraintName})
            .one_row()[0].as<std::string>();
        assert(catalogDefinition == expectedDefinition);
    }
    const auto snapshot = LoadProductionReadinessSnapshot(transaction);
    const auto evaluation = EvaluateProductionReadiness(snapshot, std::nullopt);
    const auto output = RenderProductionReadiness(evaluation);
    assert(!evaluation.ready);
    assert(output.find("ready=false") != std::string::npos);
    if (output.find(expectedOutput) == std::string::npos)
    {
        std::cerr << "missing readiness expectation=" << expectedOutput
                  << " output=" << output;
        assert(false);
    }
    assert(output.find("blockers=") != std::string::npos);
    transaction.abort();
}

void AssertLoadedReadableVersionBlock(pqxx::connection& connection,
    const std::string& mutationSql, const std::string& expectedField)
{
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    transaction.exec(mutationSql);
    transaction.exec("SET LOCAL session_replication_role=origin;");
    const auto snapshot = LoadProductionReadinessSnapshot(transaction);
    const auto evaluation = EvaluateProductionReadiness(snapshot, std::nullopt);
    const auto output = RenderProductionReadiness(evaluation);
    assert(!evaluation.ready);
    assert(output.find(expectedField) != std::string::npos);
    assert(output.find("blockers=canonical_contract_versions") !=
        std::string::npos);
    assert(snapshot.observedEnablementHead);
    assert(snapshot.observedEnablementHead->approvedBuildIdentity);
    assert(output.find("manager_service_contract=none") == std::string::npos);
    assert(output.find("enablement_canonical=none") == std::string::npos);
    assert(output.find("enablement_hash=none") == std::string::npos);
    assert(output.find("approved_build_canonical=none") == std::string::npos);
    assert(output.find("approved_build_hash=none") == std::string::npos);
    transaction.abort();
}

void AssertReadinessIntegrityFailure(pqxx::connection& connection,
    const std::string& mutationSql, const std::string& expectedDiagnostic)
{
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    transaction.exec(mutationSql);
    transaction.exec("SET LOCAL session_replication_role=origin;");
    bool rejected = false;
    try { (void)LoadProductionReadinessSnapshot(transaction); }
    catch (const Error& error)
    {
        rejected = error.code() == ErrorCode::persistenceCorruption &&
            std::string(error.what()) == expectedDiagnostic;
    }
    assert(rejected);
    transaction.abort();
}

void AssertHydrationRejects(pqxx::connection& connection,
    const std::string& corruptionSql, HydrationTarget target,
    long long attemptId)
{
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    transaction.exec(corruptionSql);
    bool rejected = false;
    try
    {
        if (target == HydrationTarget::enablement)
            (void)FindCurrentProductionEnablementHead(transaction);
        else if (target == HydrationTarget::admission)
            (void)FindRequestProductionAdmission(
                transaction, OperationalRequestId(7));
        else
            (void)FindProductionDispatchAttemptV2(
                transaction, DispatchAttemptId(attemptId));
    }
    catch (const Error& error)
    {
        rejected = error.code() == ErrorCode::persistenceCorruption ||
            error.code() == ErrorCode::invalidCanonicalText ||
            error.code() == ErrorCode::invalidCanonicalHash;
    }
    assert(rejected);
    transaction.abort();
}

void AssertHistoricalFirstAttemptHydrationRejects(
    pqxx::connection& connection, const std::string& corruptionSql,
    HydrationTarget target, long long firstAttemptId, long long secondAttemptId)
{
    {
        pqxx::work transaction{connection};
        const auto validSecond = FindProductionDispatchAttemptV2(
            transaction, DispatchAttemptId(secondAttemptId));
        assert(validSecond && validSecond->attempt.attemptOrdinal == 2);
        const auto firstBefore = AttemptEvidenceJson(
            transaction, firstAttemptId);
        const auto secondBefore = AttemptRowJson(transaction, secondAttemptId);
        transaction.exec("SET LOCAL session_replication_role=replica;");
        transaction.exec(corruptionSql);
        transaction.exec("SET LOCAL session_replication_role=origin;");
        assert(AttemptEvidenceJson(transaction, firstAttemptId) != firstBefore);
        assert(AttemptRowJson(transaction, secondAttemptId) == secondBefore);
        bool rejected = false;
        try
        {
            if (target == HydrationTarget::admission)
                (void)FindRequestProductionAdmission(
                    transaction, OperationalRequestId(7));
            else
                (void)FindProductionDispatchAttemptV2(
                    transaction, DispatchAttemptId(secondAttemptId));
        }
        catch (const Error& error)
        {
            rejected = error.code() == ErrorCode::persistenceCorruption ||
                error.code() == ErrorCode::invalidCanonicalText ||
                error.code() == ErrorCode::invalidCanonicalHash ||
                error.code() == ErrorCode::invalidEnumText ||
                error.code() == ErrorCode::unsupportedContractVersion;
        }
        assert(rejected);
        assert(AttemptRowJson(transaction, secondAttemptId) == secondBefore);
        transaction.abort();
    }
    pqxx::read_transaction transaction{connection};
    const auto restoredSecond = FindProductionDispatchAttemptV2(
        transaction, DispatchAttemptId(secondAttemptId));
    assert(restoredSecond && restoredSecond->attempt.attemptOrdinal == 2);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: CampaignOperationsPhaseH1RepositoryTests CONNECTION\n";
        return 2;
    }
    const std::string connectionString = argv[1];
    pqxx::connection connection{connectionString};
    {
        pqxx::read_transaction transaction{connection};
        transaction.exec(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        assert(ProductionAdmissionSchemaExists(transaction));
        const auto scheduler =
            LoadSchedulerProtocolEvidenceSnapshot(transaction);
        assert(scheduler);
        assert(scheduler->requiredGeneration == 52);
        assert(scheduler->identity.hash() ==
            "fnv1a64:af614995e691e378");
        const auto head = FindCurrentProductionEnablementHead(transaction);
        assert(head);
        assert(head->kind == ProductionEnablementEventKind::enable);
        assert(head->identity.hash() == "fnv1a64:05e8f676db9ba009");
        assert(head->auditEvidence.eventId == head->eventId);
        assert(head->auditEvidence.replayDisposition == "new_operation");
        const auto eventByKey = FindProductionEnablementEventByOperationKey(
            transaction, "enable-001");
        assert(eventByKey);
        assert(eventByKey->eventId == head->eventId);
        const auto admission = FindRequestProductionAdmission(
            transaction, OperationalRequestId(7));
        assert(admission);
        assert(admission->admission.identity.hash() ==
            "fnv1a64:674a534f93f43c48");
        assert(admission->authorizingEnablement.eventId ==
            admission->admission.enableEventId);
        assert(admission->firstAttempt.acquisitionAudit.admissionId ==
            admission->admissionId);
        assert(admission->firstAttempt.attemptId.value() > 0);
        assert(admission->firstAttempt.attempt.attemptOrdinal == 1);
        assert(admission->firstAttempt.attempt.identity.canonicalText() ==
            transaction.exec(
                "SELECT attempt_identity_canonical FROM "
                "campaign_operations_dispatch_attempt WHERE "
                "operational_request_id=7 AND attempt_ordinal=1;")
                .one_row()[0].as<std::string>());
        const long long attemptId = transaction.exec(
            "SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt "
            "WHERE attempt_contract_version=2 AND "
            "operation_key='dispatch-001';").one_row()[0].as<long long>();
        const auto attempt = FindProductionDispatchAttemptV2(
            transaction, DispatchAttemptId(attemptId));
        assert(attempt);
        assert(!attempt->attempt.identity.hash().empty());
        assert(attempt->attempt.admission.identity.canonicalText() ==
            admission->admission.identity.canonicalText());
        assert(attempt->admissionEvidence.admissionId ==
            admission->admissionId);
        assert(attempt->authorizingEnablement.eventId ==
            attempt->attempt.enableEventId);
        assert(attempt->acquisitionAudit.attemptId == attempt->attemptId);
        const auto attemptByKey = FindProductionDispatchAttemptV2(
            transaction, OperationalRequestId(7), "dispatch-001");
        assert(attemptByKey);
        assert(attemptByKey->attemptId == attempt->attemptId);
        const auto readiness = LoadProductionReadinessSnapshot(transaction);
        assert(readiness.migrationVersion == "055");
        assert(readiness.migrationFilename ==
            kProductionAdmissionMigrationFilename);
        assert(readiness.migrationChecksum);
        assert(*readiness.migrationChecksum ==
            kProductionAdmissionMigrationChecksum);
        assert(readiness.schedulerEvidenceContractVersion == "1");
        assert(readiness.managerBuildContractVersion == "1");
        assert(readiness.enablementContractVersion == "1");
        assert(readiness.admissionContractVersion == "1");
        assert(readiness.productionAttemptContractVersion == "2");
        assert(readiness.admissionEvidenceCount == 1);
        assert(readiness.productionAttemptEvidenceCount >= 1);
        assert(readiness.readyAdmittedRequestCount == 0);
        assert(readiness.completionNestedV2ProofVersion == "1");
        assert(readiness.completionNestedV2ProofValid);
        const auto status = LoadProductionStatusSnapshot(transaction);
        assert(status.size() == 1U);
        assert(status[0].requestId == OperationalRequestId(7));
        assert(status[0].productionDispatchEnabled);
    }
    {
        pqxx::work transaction{connection};
        const auto scheduler = LockSchedulerProtocolEvidence(transaction);
        assert(scheduler.identity.hash() ==
            "fnv1a64:af614995e691e378");
        transaction.abort();
    }
    const auto evaluated = LoadProductionReadiness(connection);
    assert(!evaluated.ready);
    assert(!evaluated.blockers.empty());

    // Exercise the actual repository snapshot with an enabled, otherwise
    // intact disposable state whose production evidence families are empty.
    // The transaction is rolled back and never touches a live database.
    {
        pqxx::work transaction{connection};
        transaction.exec("SET LOCAL session_replication_role=replica;");
        transaction.exec(
            "DELETE FROM campaign_operations_dispatch_audit_reference_event "
            "WHERE dispatch_attempt_id IS NOT NULL OR "
            "request_production_admission_id IS NOT NULL;"
            "DELETE FROM campaign_operations_dispatch_attempt_outcome "
            "WHERE dispatch_attempt_id IN (SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt WHERE "
            "request_production_admission_id IS NOT NULL OR "
            "production_enablement_event_id IS NOT NULL OR "
            "attempt_contract_version=2);"
            "DELETE FROM campaign_operations_dispatch_attempt WHERE "
            "request_production_admission_id IS NOT NULL OR "
            "production_enablement_event_id IS NOT NULL OR "
            "attempt_contract_version=2;"
            "DELETE FROM campaign_operations_request_production_admission;"
            "UPDATE campaign_operations_operational_request SET "
            "request_state='ready',state_version=3,lease_token_hash=NULL,"
            "lease_expires_at=NULL,dispatcher_identity=NULL,"
            "production_dispatch_enabled=false;");
        transaction.exec("SET LOCAL session_replication_role=origin;");
        const auto snapshot = LoadProductionReadinessSnapshot(transaction);
        assert(snapshot.admissionEvidenceCount == 0);
        assert(snapshot.productionAttemptEvidenceCount == 0);
        assert(!snapshot.admissionContractVersion);
        assert(!snapshot.productionAttemptContractVersion);
        const auto genesisBuild = [&]() -> std::optional<ManagerBuildContract>
        {
            const auto head = FindCurrentProductionEnablementHead(transaction);
            assert(head && head->approvedBuildContract);
            return head->approvedBuildContract;
        }();
        const auto genesis = EvaluateProductionReadiness(snapshot,
            genesisBuild);
        assert(std::find(genesis.blockers.begin(), genesis.blockers.end(),
            "canonical_contract_versions") == genesis.blockers.end());
        assert(RenderProductionReadiness(genesis).find(
            "admission_contract_version=genesis-empty") != std::string::npos);
        assert(RenderProductionReadiness(genesis).find(
            "production_attempt_contract_version=genesis-empty") !=
            std::string::npos);
        transaction.abort();
    }
    AssertRoleHelperCatalogSnapshotDependency(connection);
    const auto status = LoadProductionStatus(connection);
    assert(status.size() == 1U);
    const long long attemptId = [&]
    {
        pqxx::read_transaction transaction{connection};
        return transaction.exec(
            "SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt WHERE "
            "attempt_contract_version=2 AND "
            "operation_key='dispatch-001';").one_row()[0].as<long long>();
    }();
    const long long secondAttemptId = [&]
    {
        pqxx::read_transaction transaction{connection};
        return transaction.exec(
            "SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt WHERE "
            "attempt_contract_version=2 AND operation_key='dispatch-002';")
            .one_row()[0].as<long long>();
    }();
    for (const std::string& sql : std::vector<std::string>{
        "DELETE FROM campaign_operations_dispatch_audit_reference_event "
        "WHERE cause_kind='dispatch_lease_acquired' AND "
        "request_production_admission_id IS NOT NULL;",
        "UPDATE campaign_operations_dispatch_audit_reference_event SET "
        "diagnostic_code='corrupt_audit' WHERE "
        "cause_kind='dispatch_lease_acquired' AND "
        "request_production_admission_id IS NOT NULL;",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "request_identity_canonical='request-corrupt' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "request_production_admission_canonical="
        "request_production_admission_canonical||'x' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "request_production_admission_hash='fnv1a64:ffffffffffffffff' "
        "WHERE attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "production_enablement_event_canonical="
        "production_enablement_event_canonical||'x' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "production_enablement_event_hash='fnv1a64:ffffffffffffffff' "
        "WHERE attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "operation_key='corrupt-operation' WHERE attempt_contract_version=2 "
        "AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET attempt_ordinal=99 "
        "WHERE attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "lease_token_digest='fnv1a64:ffffffffffffffff' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "lease_expires_at=lease_expires_at+interval '1 second' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "original_executing_service_principal='corrupt_principal' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "approved_build_contract_canonical="
        "approved_build_contract_canonical||'x' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "approved_build_contract_hash='fnv1a64:ffffffffffffffff' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "attempt_identity_canonical=attempt_identity_canonical||'x' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "attempt_identity_hash='fnv1a64:ffffffffffffffff' WHERE "
        "attempt_contract_version=2 AND operation_key='dispatch-001';"})
        AssertHydrationRejects(
            connection, sql, HydrationTarget::attempt, attemptId);
    for (const std::string& sql : std::vector<std::string>{
        "DELETE FROM campaign_operations_dispatch_audit_reference_event "
        "WHERE cause_kind='dispatch_lease_acquired' AND "
        "request_production_admission_id IS NOT NULL;",
        "UPDATE campaign_operations_dispatch_audit_reference_event SET "
        "actor_identity='different.actor' WHERE "
        "cause_kind='dispatch_lease_acquired' AND "
        "request_production_admission_id IS NOT NULL;",
        "UPDATE campaign_operations_request_production_admission SET "
        "request_identity_canonical='request-corrupt';",
        "UPDATE campaign_operations_request_production_admission SET "
        "enable_event_canonical=enable_event_canonical||'x';",
        "UPDATE campaign_operations_request_production_admission SET "
        "approved_build_contract_canonical="
        "approved_build_contract_canonical||'x';",
        "UPDATE campaign_operations_request_production_admission SET "
        "approved_build_contract_hash='fnv1a64:ffffffffffffffff';",
        "UPDATE campaign_operations_request_production_admission SET "
        "admission_identity_canonical=admission_identity_canonical||'x';",
        "UPDATE campaign_operations_request_production_admission SET "
        "admission_identity_hash='fnv1a64:ffffffffffffffff';"})
        AssertHydrationRejects(
            connection, sql, HydrationTarget::admission, attemptId);
    for (const std::string& sql : std::vector<std::string>{
        "DELETE FROM "
        "campaign_operations_production_enablement_audit_reference_event;",
        "UPDATE campaign_operations_production_enablement_audit_reference_event "
        "SET actor_identity='corrupt.actor';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "scheduler_protocol_evidence_canonical="
        "scheduler_protocol_evidence_canonical||'x';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "scheduler_protocol_evidence_hash='fnv1a64:ffffffffffffffff';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "approved_build_compiler_contract='corrupt compiler';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "approved_build_contract_canonical="
        "approved_build_contract_canonical||'x';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "enablement_identity_canonical=enablement_identity_canonical||'x';",
        "UPDATE campaign_operations_production_enablement_event SET "
        "enablement_identity_hash='fnv1a64:ffffffffffffffff';"})
        AssertHydrationRejects(
            connection, sql, HydrationTarget::enablement, attemptId);
    const std::vector<std::string> historicalFirstAttemptCorruptions{
        "UPDATE campaign_operations_dispatch_attempt SET "
        "attempt_identity_canonical=attempt_identity_canonical||'x' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "attempt_identity_hash='fnv1a64:ffffffffffffffff' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "request_identity_canonical='corrupt-first-request' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "request_production_admission_canonical="
        "request_production_admission_canonical||'x' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "production_enablement_event_canonical="
        "production_enablement_event_canonical||'x' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "approved_build_contract_canonical="
        "approved_build_contract_canonical||'x' WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "operation_key='corrupt-first-operation' WHERE dispatch_attempt_id=" +
        std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET "
        "expected_request_version=8,resulting_request_version=9 WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "UPDATE campaign_operations_dispatch_attempt SET attempt_ordinal=9 "
        "WHERE dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT "
        "campaign_operations_dispatch_attempt_contract_version_check, DROP "
        "CONSTRAINT campaign_operations_dispatch_attempt_v1_v2_shape; UPDATE "
        "campaign_operations_dispatch_attempt SET attempt_contract_version=1 "
        "WHERE dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "DELETE FROM campaign_operations_dispatch_attempt WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + ";",
        "DELETE FROM campaign_operations_dispatch_audit_reference_event WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + " AND "
        "cause_kind='dispatch_lease_acquired';",
        "UPDATE campaign_operations_dispatch_audit_reference_event SET "
        "actor_identity='corrupt.audit.actor' WHERE dispatch_attempt_id=" +
        std::to_string(attemptId) + " AND "
        "cause_kind='dispatch_lease_acquired';",
        "DROP INDEX campaign_operations_dispatch_audit_acquisition_uidx; "
        "INSERT INTO campaign_operations_dispatch_audit_reference_event("
        "operational_campaign_id,operational_request_id,dispatch_attempt_id,"
        "dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,"
        "prior_version,resulting_version,outcome,replay_disposition,"
        "diagnostic_code,request_production_admission_id,"
        "production_enablement_event_id) SELECT operational_campaign_id,"
        "operational_request_id,dispatch_attempt_id,"
        "dispatch_attempt_outcome_id,cause_kind,actor_identity,capability,"
        "prior_version,resulting_version,outcome,replay_disposition,"
        "diagnostic_code,request_production_admission_id,"
        "production_enablement_event_id FROM "
        "campaign_operations_dispatch_audit_reference_event WHERE "
        "dispatch_attempt_id=" + std::to_string(attemptId) + " AND "
        "cause_kind='dispatch_lease_acquired';"};
    for (const auto target : {HydrationTarget::admission,
             HydrationTarget::attempt})
        for (const auto& sql : historicalFirstAttemptCorruptions)
            AssertHistoricalFirstAttemptHydrationRejects(connection, sql,
                target, attemptId, secondAttemptId);
    for (const std::string& sql : std::vector<std::string>{
        "ALTER TABLE campaign_operations_production_enablement_event DROP "
        "CONSTRAINT campaign_operations_production_enablement_shape; UPDATE "
        "campaign_operations_production_enablement_event SET "
        "capability='campaign_operations_production_disabler';",
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'campaign_operations_production_enablement_event'::regclass AND "
        "contype='c' AND pg_get_constraintdef(oid) LIKE "
        "'%enablement_contract_version = 1%'; EXECUTE format('ALTER TABLE "
        "campaign_operations_production_enablement_event DROP CONSTRAINT "
        "%I',constraint_name); END $block$; UPDATE "
        "campaign_operations_production_enablement_event SET "
        "enablement_contract_version=2;"})
    AssertHydrationRejects(
            connection, sql, HydrationTarget::enablement, attemptId);
    AssertLoadedReadableVersionBlock(connection,
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'experiment_scheduler_protocol'::regclass AND contype='c' AND "
        "pg_get_constraintdef(oid) LIKE "
        "'%scheduler_evidence_contract_version = 1%'; EXECUTE format("
        "'ALTER TABLE experiment_scheduler_protocol DROP CONSTRAINT %I',"
        "constraint_name); END $block$; UPDATE experiment_scheduler_protocol SET "
        "scheduler_evidence_contract_version=2 WHERE singleton;",
        "scheduler_contract_version=2");
    AssertReadinessIntegrityFailure(connection,
        "UPDATE campaign_operations_production_enablement_event SET "
        "approved_build_contract_canonical=regexp_replace("
        "approved_build_contract_canonical,'build_contract_version=1$',"
        "'build_contract_version=2') WHERE event_kind='enable';",
        "campaign_operations_manager_build_corrupt");
    AssertLoadedReadableVersionBlock(connection,
        "UPDATE campaign_operations_production_enablement_event SET "
        "approved_build_contract_canonical=regexp_replace("
        "approved_build_contract_canonical,'build_contract_version=1$',"
        "'build_contract_version=2'),approved_build_contract_hash="
        "campaign_operations_tagged_fnv1a64(regexp_replace("
        "approved_build_contract_canonical,'build_contract_version=1$',"
        "'build_contract_version=2')) WHERE event_kind='enable';"
        "UPDATE campaign_operations_production_enablement_event event SET "
        "enablement_identity_canonical="
        "campaign_operations_production_enablement_canonical_v1(event),"
        "enablement_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_production_enablement_canonical_v1(event)) "
        "WHERE event_kind='enable';"
        "UPDATE campaign_operations_request_production_admission admission SET "
        "enable_event_canonical=event.enablement_identity_canonical,"
        "approved_build_contract_canonical=event.approved_build_contract_canonical,"
        "approved_build_contract_hash=event.approved_build_contract_hash FROM "
        "campaign_operations_production_enablement_event event WHERE "
        "event.production_enablement_event_id=admission.production_enablement_event_id;"
        "UPDATE campaign_operations_request_production_admission admission SET "
        "admission_identity_canonical="
        "campaign_operations_request_production_admission_canonical_v1(admission),"
        "admission_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_request_production_admission_canonical_v1(admission));"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "request_production_admission_canonical=admission.admission_identity_canonical,"
        "request_production_admission_hash=admission.admission_identity_hash,"
        "production_enablement_event_canonical=event.enablement_identity_canonical,"
        "production_enablement_event_hash=event.enablement_identity_hash,"
        "approved_build_contract_canonical=event.approved_build_contract_canonical,"
        "approved_build_contract_hash=event.approved_build_contract_hash FROM "
        "campaign_operations_request_production_admission admission JOIN "
        "campaign_operations_production_enablement_event event ON "
        "event.production_enablement_event_id=admission.production_enablement_event_id "
        "WHERE attempt.request_production_admission_id="
        "admission.request_production_admission_id;"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "attempt_identity_canonical=campaign_operations_dispatch_attempt_v2_canonical(attempt),"
        "attempt_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_dispatch_attempt_v2_canonical(attempt)) WHERE "
        "attempt_contract_version=2;",
        "manager_build_contract_version=2");
    AssertReadinessIntegrityFailure(connection,
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'campaign_operations_production_enablement_event'::regclass AND "
        "contype='c' AND pg_get_constraintdef(oid) LIKE "
        "'%enablement_contract_version = 1%'; EXECUTE format('ALTER TABLE "
        "campaign_operations_production_enablement_event DROP CONSTRAINT "
        "%I',constraint_name); END $block$; UPDATE "
        "campaign_operations_production_enablement_event SET "
        "enablement_contract_version=2;",
        "campaign_operations_enablement_head_corrupt");
    AssertLoadedReadableVersionBlock(connection,
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'campaign_operations_production_enablement_event'::regclass AND "
        "contype='c' AND pg_get_constraintdef(oid) LIKE "
        "'%enablement_contract_version = 1%'; EXECUTE format('ALTER TABLE "
        "campaign_operations_production_enablement_event DROP CONSTRAINT "
        "%I',constraint_name); END $block$; UPDATE "
        "campaign_operations_production_enablement_event SET "
        "enablement_contract_version=2,enablement_identity_canonical="
        "regexp_replace(enablement_identity_canonical,"
        "'enablement_contract_version=1$',"
        "'enablement_contract_version=2'),enablement_identity_hash="
        "campaign_operations_tagged_fnv1a64(regexp_replace("
        "enablement_identity_canonical,'enablement_contract_version=1$',"
        "'enablement_contract_version=2'));"
        "UPDATE campaign_operations_request_production_admission admission SET "
        "enable_event_canonical=event.enablement_identity_canonical FROM "
        "campaign_operations_production_enablement_event event WHERE "
        "event.production_enablement_event_id=admission.production_enablement_event_id;"
        "UPDATE campaign_operations_request_production_admission admission SET "
        "admission_identity_canonical="
        "campaign_operations_request_production_admission_canonical_v1(admission),"
        "admission_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_request_production_admission_canonical_v1(admission));"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "request_production_admission_canonical=admission.admission_identity_canonical,"
        "request_production_admission_hash=admission.admission_identity_hash,"
        "production_enablement_event_canonical=event.enablement_identity_canonical,"
        "production_enablement_event_hash=event.enablement_identity_hash FROM "
        "campaign_operations_request_production_admission admission JOIN "
        "campaign_operations_production_enablement_event event ON "
        "event.production_enablement_event_id=admission.production_enablement_event_id "
        "WHERE attempt.request_production_admission_id="
        "admission.request_production_admission_id;"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "attempt_identity_canonical=campaign_operations_dispatch_attempt_v2_canonical(attempt),"
        "attempt_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_dispatch_attempt_v2_canonical(attempt)) WHERE "
        "attempt_contract_version=2;",
        "enablement_contract_version=2");
    AssertLoadedReadinessBlocks(connection,
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'campaign_operations_request_production_admission'::regclass AND "
        "contype='c' AND pg_get_constraintdef(oid) LIKE "
        "'%admission_contract_version = 1%'; EXECUTE format('ALTER TABLE "
        "campaign_operations_request_production_admission DROP CONSTRAINT "
        "%I',constraint_name); END $block$; UPDATE "
        "campaign_operations_request_production_admission SET "
        "admission_contract_version=2,admission_identity_canonical="
        "regexp_replace(admission_identity_canonical,"
        "'admission_contract_version=1$',"
        "'admission_contract_version=2'),admission_identity_hash="
        "campaign_operations_tagged_fnv1a64(regexp_replace("
        "admission_identity_canonical,'admission_contract_version=1$',"
        "'admission_contract_version=2'));"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "request_production_admission_canonical=admission.admission_identity_canonical,"
        "request_production_admission_hash=admission.admission_identity_hash FROM "
        "campaign_operations_request_production_admission admission WHERE "
        "attempt.request_production_admission_id="
        "admission.request_production_admission_id;"
        "UPDATE campaign_operations_dispatch_attempt attempt SET "
        "attempt_identity_canonical=campaign_operations_dispatch_attempt_v2_canonical(attempt),"
        "attempt_identity_hash=campaign_operations_tagged_fnv1a64("
        "campaign_operations_dispatch_attempt_v2_canonical(attempt)) WHERE "
        "attempt_contract_version=2;",
        "admission_contract_version=2");
    AssertLoadedReadableVersionBlock(connection,
        "ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT "
        "campaign_operations_dispatch_attempt_contract_version_check, DROP "
        "CONSTRAINT campaign_operations_dispatch_attempt_v1_v2_shape; UPDATE "
        "campaign_operations_dispatch_attempt SET attempt_contract_version=3,"
        "attempt_identity_canonical=regexp_replace(attempt_identity_canonical,"
        "'attempt_contract_version=2$',"
        "'attempt_contract_version=3'),attempt_identity_hash="
        "campaign_operations_tagged_fnv1a64(regexp_replace("
        "attempt_identity_canonical,'attempt_contract_version=2$',"
        "'attempt_contract_version=3')) "
        "WHERE request_production_admission_id IS NOT NULL;",
        "production_attempt_contract_version=3");
    AssertLoadedReadableVersionBlock(connection,
        "ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT "
        "campaign_operations_dispatch_attempt_contract_version_check, DROP "
        "CONSTRAINT campaign_operations_dispatch_attempt_v1_v2_shape; UPDATE "
        "campaign_operations_dispatch_attempt SET attempt_contract_version=3,"
        "attempt_identity_canonical=regexp_replace(attempt_identity_canonical,"
        "'attempt_contract_version=2$',"
        "'attempt_contract_version=3'),attempt_identity_hash="
        "campaign_operations_tagged_fnv1a64(regexp_replace("
        "attempt_identity_canonical,'attempt_contract_version=2$',"
        "'attempt_contract_version=3')) WHERE operation_key='dispatch-002';",
        "production_attempt_contract_version=2|3");
    for (const auto& [sql, diagnostic] : std::vector<std::pair<std::string,
             std::string>>{
        {"UPDATE campaign_operations_request_production_admission SET "
         "admission_identity_canonical=admission_identity_canonical||'x';",
         "campaign_operations_admission_corrupt"},
        {"UPDATE campaign_operations_request_production_admission SET "
         "admission_identity_hash='fnv1a64:ffffffffffffffff';",
         "campaign_operations_admission_corrupt"},
        {"DELETE FROM campaign_operations_dispatch_audit_reference_event WHERE "
         "cause_kind='dispatch_lease_acquired' AND "
         "request_production_admission_id IS NOT NULL;",
         "campaign_operations_acquisition_audit_missing"},
        {"UPDATE campaign_operations_dispatch_audit_reference_event SET "
         "diagnostic_code='corrupt_audit' WHERE "
         "cause_kind='dispatch_lease_acquired' AND "
         "request_production_admission_id IS NOT NULL;",
         "campaign_operations_acquisition_audit_corrupt"},
        {"UPDATE campaign_operations_dispatch_attempt SET "
         "attempt_identity_canonical=attempt_identity_canonical||'x' WHERE "
         "attempt_contract_version=2;",
         "campaign_operations_attempt_v2_corrupt"},
        {"UPDATE campaign_operations_dispatch_attempt SET "
         "attempt_identity_hash='fnv1a64:ffffffffffffffff' WHERE "
         "attempt_contract_version=2;",
         "campaign_operations_attempt_v2_corrupt"},
        {"ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT "
         "campaign_operations_dispatch_attempt_v1_v2_shape; UPDATE "
         "campaign_operations_dispatch_attempt SET "
         "production_enablement_event_id=NULL WHERE attempt_contract_version=2;",
         "campaign_operations_attempt_v2_typed_mirror_corrupt"},
        {"ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT "
         "campaign_operations_dispatch_attempt_v1_v2_shape; UPDATE "
         "campaign_operations_dispatch_attempt SET "
         "request_production_admission_id=NULL WHERE attempt_contract_version=2;",
         "campaign_operations_attempt_v2_typed_mirror_corrupt"}})
        AssertReadinessIntegrityFailure(connection, sql, diagnostic);
    AssertLoadedReadinessBlocks(connection,
        "DO $block$ DECLARE constraint_name name; BEGIN SELECT conname INTO "
        "STRICT constraint_name FROM pg_constraint WHERE conrelid="
        "'campaign_operations_completion_event'::regclass AND contype='c' "
        "AND pg_get_constraintdef(oid) LIKE '%completion_contract_version = 1%'; "
        "EXECUTE format('ALTER TABLE campaign_operations_completion_event "
        "DROP CONSTRAINT %I',constraint_name); END $block$; ALTER TABLE "
        "campaign_operations_completion_event ADD CONSTRAINT "
        "h1_completion_contract_version_test CHECK (completion_contract_version=2);",
        "completion_nested_v2_proof_version=2", true,
        "h1_completion_contract_version_test");
    AssertLoadedReadinessBlocks(connection,
        "ALTER TABLE experiment_scheduler_protocol DROP CONSTRAINT "
        "experiment_scheduler_protocol_required_generation_check; ALTER TABLE "
        "experiment_scheduler_protocol ALTER COLUMN required_generation DROP "
        "NOT NULL; UPDATE "
        "experiment_scheduler_protocol SET required_generation=NULL WHERE singleton;",
        "scheduler_generation=missing");
    AssertLoadedReadinessBlocks(connection,
        "ALTER TABLE experiment_scheduler_protocol DROP CONSTRAINT "
        "experiment_scheduler_protocol_cutover_state_check, DROP CONSTRAINT "
        "experiment_scheduler_protocol_check; ALTER TABLE "
        "experiment_scheduler_protocol ALTER COLUMN cutover_state DROP NOT NULL; "
        "UPDATE experiment_scheduler_protocol "
        "SET cutover_state=NULL WHERE singleton;",
        "scheduler_cutover_state=missing");
    AssertReadinessIntegrityFailure(connection,
        "DELETE FROM campaign_operations_production_enablement_event;",
        "campaign_operations_enablement_missing");
    AssertReadinessIntegrityFailure(connection,
        "DELETE FROM campaign_operations_request_production_admission;",
        "campaign_operations_admission_missing");
    AssertReadinessIntegrityFailure(connection,
        "DELETE FROM campaign_operations_production_enablement_audit_reference_event;",
        "campaign_operations_enablement_audit_missing");
    AssertCompletionCatalogSnapshotNegativeControl(connection);
    std::ostringstream output;
    std::ostringstream errors;
    const auto readinessSnapshot = [&]
    {
        pqxx::read_transaction transaction{connection};
        return LoadProductionReadinessSnapshot(transaction);
    }();
    const auto actualBuild = [&]() -> std::optional<ManagerBuildContract>
    {
        pqxx::read_transaction transaction{connection};
        const auto head = FindCurrentProductionEnablementHead(transaction);
        assert(head && head->approvedBuildContract);
        return head->approvedBuildContract;
    }();
    const auto readinessEvidenceBefore = [&]
    {
        pqxx::read_transaction transaction{connection};
        return ReadinessEvidenceJson(transaction);
    }();
    assert(RunProductionReadinessCommand(
        connectionString, output, errors, actualBuild) == 2);
    const auto readinessEvidenceAfter = [&]
    {
        pqxx::read_transaction transaction{connection};
        return ReadinessEvidenceJson(transaction);
    }();
    assert(readinessEvidenceAfter == readinessEvidenceBefore);
    assert(errors.str().empty());
    assert(output.str().find("ready=false") != std::string::npos);
    assert(output.str().find("migration_version=" +
        readinessSnapshot.migrationVersion) != std::string::npos);
    assert(output.str().find("migration_checksum=" +
        *readinessSnapshot.migrationChecksum) != std::string::npos);
    assert(output.str().find("scheduler_contract_version=" +
        *readinessSnapshot.schedulerEvidenceContractVersion) !=
        std::string::npos);
    assert(output.str().find("scheduler_generation=" +
        std::to_string(*readinessSnapshot.schedulerGeneration)) !=
        std::string::npos);
    assert(output.str().find("scheduler_evidence_canonical=") !=
        std::string::npos);
    assert(output.str().find("scheduler_evidence_hash=fnv1a64:") !=
        std::string::npos);
    assert(output.str().find("independent_verification_reference=") !=
        std::string::npos);
    assert(output.str().find("enablement_contract_version=" +
        *readinessSnapshot.enablementContractVersion) !=
        std::string::npos);
    assert(output.str().find("manager_build_contract_version=" +
        *readinessSnapshot.managerBuildContractVersion) !=
        std::string::npos);
    assert(readinessSnapshot.enablementHead &&
        readinessSnapshot.enablementHead->approvedBuildContract);
    assert(output.str().find("manager_service_contract=" +
        readinessSnapshot.enablementHead->approvedBuildContract->
            managerServiceContract) != std::string::npos);
    assert(output.str().find("admission_contract_version=" +
        *readinessSnapshot.admissionContractVersion) !=
        std::string::npos);
    assert(output.str().find("production_attempt_contract_version=" +
        *readinessSnapshot.productionAttemptContractVersion) !=
        std::string::npos);
    assert(output.str().find("enablement_canonical=") != std::string::npos);
    assert(output.str().find("enablement_hash=fnv1a64:") !=
        std::string::npos);
    assert(output.str().find("approved_build_canonical=") !=
        std::string::npos);
    assert(output.str().find("actual_running_build_canonical=") !=
        std::string::npos);
    assert(output.str().find("build_comparison=match") != std::string::npos);
    assert(output.str().find("completion_nested_v2_proof_version=" +
        *readinessSnapshot.completionNestedV2ProofVersion) !=
        std::string::npos);
    assert(output.str().find("completion_nested_v2_proof=valid") !=
        std::string::npos);
    assert(output.str().find("old_event_blocked_leases=" +
        std::to_string(readinessSnapshot.oldEventBlockedLeaseCount)) !=
        std::string::npos);
    assert(output.str().find("reconciliation_required=" +
        std::to_string(readinessSnapshot.reconciliationRequiredCount)) !=
        std::string::npos);
    assert(output.str().find("blockers=") != std::string::npos);
    output.str({});
    assert(RunProductionStatusCommand(
        connectionString, output, errors) == 0);
    assert(output.str().find(
        "CAMPAIGN_OPERATIONS_PRODUCTION_STATUS_COMPLETE,count=1") !=
        std::string::npos);
    assert(output.str().find("lease_expires_at=") != std::string::npos);
    assert(output.str().find("phase_f_recovery_eligible=") !=
        std::string::npos);
    assert(output.str().find("reconciliation_required=") !=
        std::string::npos);
    assert(output.str().find("lease_expires_at=" +
        status[0].leaseExpiresAt.value_or("none")) != std::string::npos);
    assert(output.str().find("phase_f_recovery_eligible=" + std::string(
        status[0].phaseFRecoveryEligible ? "true" : "false")) !=
        std::string::npos);
    assert(output.str().find("reconciliation_required=" + std::string(
        status[0].reconciliationRequired ? "true" : "false")) !=
        std::string::npos);
}
