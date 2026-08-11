#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsManager.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace CO = EA::CampaignOperations;

namespace
{

CO::ManagerBuildContract FixtureBuild()
{
    return CO::BuildManagerBuildContract(
        CO::kManagerServiceContract,
        "777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d",
        "clang++-h3-compatibility-fixture",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
}

CO::ProductionDispatchRequest Request(long long id, std::string key,
    const CO::ActorIdentity& actor, const CO::ManagerBuildContract& build)
{
    return {CO::OperationalRequestId(id), 3, std::move(key), actor, build, true};
}

void RequireReserved(const std::string& connection,
    const CO::ProductionDispatchRequest& request)
{
    bool rejected = false;
    try
    {
        (void)CO::DispatchOneRequestForProductionForTest(connection, request);
    }
    catch (const std::invalid_argument& error)
    {
        rejected = std::string(error.what()) ==
            "campaign_operations_manager_operation_key_reserved";
    }
    assert(rejected);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: CampaignOperationsPhaseH3CompatibilityTests "
                     "pre|post ENABLER_CONNECTION MANAGER_CONNECTION\n";
        return 64;
    }
    try
    {
        const std::string phase = argv[1];
        const std::string enablerConnection = argv[2];
        const std::string managerConnection = argv[3];
        const auto build = FixtureBuild();
        const CO::ActorIdentity h2Actor("h2.manager@example.test");
        const auto exact = Request(71, "mgr-v1:legacy-h2-exact", h2Actor, build);
        const auto recoverable = Request(72, "mgr-v1:legacy-h2-recoverable",
            h2Actor, build);

        if (phase == "pre")
        {
            CO::ProductionEnableRequest enable{
                "h3-compat-enable", 2, "cee://h3-compat/generation-52",
                CO::ActorIdentity("h2.enabler@example.test"),
                CO::Reason("H3 compatibility disposable enable"), build, true};
            const auto enabled = CO::EnableProduction(
                [&] { return std::make_unique<pqxx::connection>(enablerConnection); },
                enable);
            assert(enabled.disposition == CO::ProductionMutationDisposition::newOperation);

            const auto first = CO::DispatchOneRequestForProductionForTest(
                managerConnection, exact);
            assert(first.classification == CO::DispatchResultClassification::createdAndBound);
            const auto replay = CO::DispatchOneRequestForProductionForTest(
                managerConnection, exact);
            assert(replay.classification ==
                CO::DispatchResultClassification::existingIdentical);

            bool stoppedAfterAcquisition = false;
            try
            {
                (void)CO::DispatchOneRequestForProductionForTest(
                    managerConnection, recoverable,
                    [](CO::DispatchTestInjectionPoint point)
                    {
                        if (point == CO::DispatchTestInjectionPoint::
                                afterAcquisitionCommitBeforeHandoff)
                            throw std::runtime_error("h3_compatibility_stop_after_acquisition");
                    });
            }
            catch (const std::runtime_error& error)
            {
                stoppedAfterAcquisition = std::string(error.what()) ==
                    "h3_compatibility_stop_after_acquisition";
            }
            assert(stoppedAfterAcquisition);
            std::cout << "H3_COMPAT_PRE058 exact_h2_replay=PASS "
                         "recoverable_h2_attempt=PASS\n";
            return 0;
        }

        if (phase != "post") throw std::invalid_argument("h3_compatibility_phase");

        const auto replay = CO::DispatchOneRequestForProductionForTest(
            managerConnection, exact);
        assert(replay.classification == CO::DispatchResultClassification::existingIdentical);

        const auto conflicting = Request(71, exact.operationKey,
            CO::ActorIdentity("changed.manager@example.test"), build);
        bool conflictObserved = false;
        try
        {
            (void)CO::DispatchOneRequestForProductionForTest(
                managerConnection, conflicting);
        }
        catch (const std::exception& error)
        {
            conflictObserved = std::string(error.what()).find("conflict") !=
                std::string::npos;
        }
        assert(conflictObserved);

        const auto recovered = CO::DispatchOneRequestForProductionForTest(
            managerConnection, recoverable);
        assert(recovered.classification == CO::DispatchResultClassification::createdAndBound ||
            recovered.classification == CO::DispatchResultClassification::existingIdentical);

        RequireReserved(managerConnection,
            Request(73, "mgr-v1:post058-new", h2Actor, build));
        const auto ordinary = CO::DispatchOneRequestForProductionForTest(
            managerConnection, Request(73, "h2-post058-ordinary", h2Actor, build));
        assert(ordinary.classification == CO::DispatchResultClassification::createdAndBound);

        pqxx::connection lookup{managerConnection};
        pqxx::read_transaction transaction{lookup};
        transaction.exec("SET LOCAL ROLE campaign_operations_production_phase5_transactional;");
        const auto source = transaction.exec(
            "SELECT request_identity_canonical FROM "
            "campaign_operations_operational_request "
            "WHERE operational_request_id=74;").one_row()[0].as<std::string>();
        transaction.commit();
        const auto managerIdentity = CO::BuildManagerRequestOperationIdentity(source, 3);
        const auto managerRequest = Request(74, managerIdentity.operationKey,
            CO::ActorIdentity("campaign_operations_manager"), build);
        const auto manager = CO::DispatchOneRequestForProductionManagerWithFixture(
            managerConnection, managerRequest, managerIdentity.source.canonicalText(),
            build);
        assert(manager.classification == CO::DispatchResultClassification::createdAndBound ||
            manager.classification == CO::DispatchResultClassification::existingIdentical);
        RequireReserved(managerConnection, managerRequest);

        std::cout << "H3_COMPAT_POST058 historical_exact=PASS "
                     "historical_conflict=PASS historical_recovery=PASS "
                     "new_prefix_rejected=PASS ordinary_h2=PASS "
                     "manager_only=PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "H3_COMPAT_FAIL diagnostic=" << error.what() << '\n';
        return 1;
    }
}
