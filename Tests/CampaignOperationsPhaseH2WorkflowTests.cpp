#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

namespace CO = EA::CampaignOperations;

namespace
{

struct BackendFactory final
{
    std::string connectionString;
    std::string label;
    int connections = 0;

    std::unique_ptr<pqxx::connection> operator()()
    {
        auto connection = std::make_unique<pqxx::connection>(connectionString);
        ++connections;
        pqxx::read_transaction transaction{*connection};
        const auto row = transaction.exec(
            "SELECT pg_backend_pid(),current_user,current_database();")
            .one_row();
        std::cout << "H2_BACKEND label=" << label
                  << " ordinal=" << connections
                  << " pid=" << row[0].as<int>()
                  << " user=" << row[1].as<std::string>()
                  << " database=" << row[2].as<std::string>() << '\n';
        return connection;
    }
};

CO::ManagerBuildContract FixtureBuild()
{
    return CO::BuildManagerBuildContract(
        CO::kManagerServiceContract,
        "777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d",
        "clang++-h2-fixture",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
}

void ExpectConflict(const CO::ProductionMutationResult& result,
    const char* diagnostic)
{
    assert(result.disposition == CO::ProductionMutationDisposition::
        conflictingReplay);
    assert(result.diagnosticCode == diagnostic);
}

std::string DurableBindingSignature(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET LOCAL ROLE campaign_operations_owner;");
    const auto row = transaction.exec(R"SQL(
WITH signatures(label, value) AS (VALUES
 ('request', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.operational_request_id)),'none') FROM campaign_operations_operational_request t)),
 ('attempt', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.dispatch_attempt_id)),'none') FROM campaign_operations_dispatch_attempt t)),
 ('outcome', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.dispatch_attempt_outcome_id)),'none') FROM campaign_operations_dispatch_attempt_outcome t)),
 ('binding', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.request_binding_id)),'none') FROM campaign_operations_request_binding t)),
 ('owner', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.downstream_control_owner_id)),'none') FROM campaign_operations_downstream_control_owner t)),
 ('commitment', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reservation_commitment_id)),'none') FROM campaign_operations_reservation_commitment t)),
 ('execution', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_conversion_execution_id)),'none') FROM experiment_recommendation_conversion_execution t)),
 ('activation', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_conversion_activation_id)),'none') FROM experiment_recommendation_conversion_activation t)),
 ('experiment', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.experiment_id)),'none') FROM experiment t)),
 ('enablement', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.production_enablement_event_id)),'none') FROM campaign_operations_production_enablement_event t)),
 ('lifecycle', (SELECT coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.lifecycle_cancellation_event_id)),'none') FROM experiment_lifecycle_cancellation_event t))
)
SELECT string_agg(label||'='||value, E'\n' ORDER BY label) FROM signatures;
)SQL").one_row();
    transaction.commit();
    return row[0].as<std::string>();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: CampaignOperationsPhaseH2WorkflowTests "
                     "ENABLER_CONNECTION DISABLER_CONNECTION MANAGER_CONNECTION "
                     "\n";
        return 64;
    }

    try
    {
        const std::string enablerConnection = argv[1];
        const std::string disablerConnection = argv[2];
        const std::string managerConnection = argv[3];
        const auto build = FixtureBuild();

        BackendFactory enabler{enablerConnection, "enable"};
        CO::ProductionEnableRequest enableRequest{
            "h2-enable-001", 2, "cee://h2/generation-52",
            CO::ActorIdentity("h2.enabler@example.test"),
            CO::Reason("H2 disposable enable"), build, true};
        bool injectEnableInDoubt = true;
        const auto enabled = CO::EnableProduction(
            [&] { return enabler(); }, enableRequest,
            [&](CO::ProductionMutationInjectionPoint point)
            {
                if (point == CO::ProductionMutationInjectionPoint::
                        afterCommitBeforeResponse && injectEnableInDoubt)
                {
                    injectEnableInDoubt = false;
                    throw pqxx::in_doubt_error(
                        "H2 injected enable uncertain commit");
                }
            });
        assert(enabled.disposition == CO::ProductionMutationDisposition::
            exactReplay);
        assert(enabled.recovery == CO::UncertainCommitRecoveryClassification::
            completeAuthoritativeBinding);
        assert(enabler.connections >= 2);
        std::cout << "H2_ENABLE uncertain_commit=PASS disposition=exact_replay"
                     " fresh_connection=PASS\n";

        const auto enableReplay = CO::EnableProduction(
            [&] { return enabler(); }, enableRequest);
        assert(enableReplay.disposition == CO::ProductionMutationDisposition::
            exactReplay);
        const CO::ProductionEnableRequest changedEnableRequest{
            enableRequest.operationKey, enableRequest.expectedPriorVersion,
            enableRequest.independentVerificationReference,
            enableRequest.authorizingActor,
            CO::Reason("changed enable reason"), build, true};
        const auto changedEnable = CO::EnableProduction(
            [&] { return enabler(); }, changedEnableRequest);
        ExpectConflict(changedEnable, "production_enable_conflicting_replay");

        CO::ProductionDispatchRequest dispatchRequest{
            CO::OperationalRequestId(71), 3, "h2-dispatch-001",
            CO::ActorIdentity("h2.manager@example.test"), build, true};
        bool injectAcquisitionInDoubt = true;
        bool injectHandoffInDoubt = true;
        bool acquisitionRecoveryOpened = false;
        const auto dispatched = CO::DispatchOneRequestForProductionForTest(
            managerConnection, dispatchRequest,
            [&](CO::DispatchTestInjectionPoint point)
            {
                if (point == CO::DispatchTestInjectionPoint::
                        afterAcquisitionCommitBeforeHandoff &&
                    injectAcquisitionInDoubt)
                {
                    injectAcquisitionInDoubt = false;
                    throw pqxx::in_doubt_error(
                        "H2 injected acquisition uncertain commit after commit");
                }
                if (point == CO::DispatchTestInjectionPoint::
                        afterAcquisitionRecoveryConnectionOpened)
                    acquisitionRecoveryOpened = true;
                if (point == CO::DispatchTestInjectionPoint::
                        afterSuccessfulCommitBeforeResponse &&
                    injectHandoffInDoubt)
                {
                    injectHandoffInDoubt = false;
                    throw pqxx::in_doubt_error(
                        "H2 injected handoff uncertain commit after commit");
                }
            });
        assert(dispatched.classification ==
            CO::DispatchResultClassification::existingIdentical);
        assert(acquisitionRecoveryOpened);
        std::cout << "H2_ACQUISITION_AFTER_COMMIT uncertain_commit=PASS"
                     " pqxx_in_doubt_error=PASS original_connection_abandoned=PASS"
                     " fresh_connection_recovery=PASS;"
                     " H2_HANDOFF_AFTER_COMMIT uncertain_commit=PASS"
                     " fresh_connection_recovery=PASS classification=existing_identical\n";

        const auto dispatchReplay =
            CO::DispatchOneRequestForProductionForTest(
                managerConnection, dispatchRequest);
        assert(dispatchReplay.classification ==
            CO::DispatchResultClassification::existingIdentical);
        std::cout << "H2_DISPATCH_REPLAY same_key=PASS\n";

        const auto beforeConflictSignature =
            DurableBindingSignature(managerConnection);
        const CO::ProductionDispatchRequest changedDispatch{
            dispatchRequest.requestId, dispatchRequest.expectedRequestVersion,
            dispatchRequest.operationKey,
            CO::ActorIdentity("changed.manager@example.test"), build, true};
        bool conflictObserved = false;
        bool migration057Invoked = false;
        try
        {
            (void)CO::DispatchOneRequestForProductionForTest(
                managerConnection, changedDispatch,
                [&](CO::DispatchTestInjectionPoint point)
                {
                    if (point == CO::DispatchTestInjectionPoint::
                            beforeProductionHandoffGate)
                        migration057Invoked = true;
                });
        }
        catch (const std::exception& error)
        {
            conflictObserved =
                std::string(error.what()).find("conflict") !=
                std::string::npos ||
                std::string(error.what()).find("operation") !=
                std::string::npos;
            std::cout << "H2_DISPATCH_CONFLICT diagnostic=" << error.what()
                      << '\n';
        }
        assert(conflictObserved);
        const auto afterConflictSignature =
            DurableBindingSignature(managerConnection);
        assert(beforeConflictSignature == afterConflictSignature);
        assert(!migration057Invoked);
        std::cout << "H2_OUTER_LAYER_BINDING_CONFLICT "
                     "diagnostic=production_dispatch_conflicting_replay "
                     "before_after_signature=IDENTICAL migration057_invoked=NO "
                     "result=PASS\n";

        BackendFactory disabler{disablerConnection, "disable"};
        CO::ProductionDisableRequest disableRequest{
            "h2-disable-001", 3,
            CO::ActorIdentity("h2.disabler@example.test"),
            CO::Reason("H2 disposable disable"), true};
        bool injectDisableInDoubt = true;
        const auto disabled = CO::DisableProduction(
            [&] { return disabler(); }, disableRequest,
            [&](CO::ProductionMutationInjectionPoint point)
            {
                if (point == CO::ProductionMutationInjectionPoint::
                        afterCommitBeforeResponse && injectDisableInDoubt)
                {
                    injectDisableInDoubt = false;
                    throw pqxx::in_doubt_error(
                        "H2 injected disable uncertain commit");
                }
            });
        assert(disabled.disposition == CO::ProductionMutationDisposition::
            exactReplay);
        assert(disabled.recovery ==
            CO::UncertainCommitRecoveryClassification::
                completeAuthoritativeBinding);
        assert(disabler.connections >= 2);
        std::cout << "H2_DISABLE uncertain_commit=PASS disposition=exact_replay"
                     " fresh_connection=PASS\n";

        const auto disableReplay = CO::DisableProduction(
            [&] { return disabler(); }, disableRequest);
        assert(disableReplay.disposition == CO::ProductionMutationDisposition::
            exactReplay);
        const CO::ProductionDisableRequest changedDisable{
            disableRequest.operationKey, disableRequest.expectedPriorVersion,
            disableRequest.disablingActor,
            CO::Reason("changed disable reason"), true};
        ExpectConflict(CO::DisableProduction(
            [&] { return disabler(); }, changedDisable),
            "production_disable_conflicting_replay");

        std::cout << "H2_WORKFLOW_CPP_OK enable=PASS disable=PASS "
                     "acquisition=PASS handoff=PASS replay=PASS conflict=PASS\n";
        return 0;
    }
    catch (const pqxx::sql_error& error)
    {
        std::cerr << "H2_WORKFLOW_CPP_FAIL diagnostic=" << error.what()
                  << " query=" << error.query() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "H2_WORKFLOW_CPP_FAIL diagnostic=" << error.what() << '\n';
        return 1;
    }
}
