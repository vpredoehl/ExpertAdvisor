#include "CampaignOperationsControlService.hpp"
#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <pqxx/pqxx>
#include <string>
#include <stdexcept>
#include <thread>

namespace CO = EA::CampaignOperations;

namespace
{

CO::ManagerBuildContract FixtureBuild()
{
    return CO::BuildManagerBuildContract(
        CO::kManagerServiceContract,
        "777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d",
        "clang++-h2-fixture",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
}

void ExpireAcquiredLeaseForDisposableFixture(
    const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    // The acquisition itself is produced by the real production service.
    // This disposable-only timing seam shortens its otherwise five-minute
    // lease so the authoritative reconciliation workflow can be exercised.
    transaction.exec("SET LOCAL session_replication_role=replica;");
    transaction.exec(
        "UPDATE campaign_operations_operational_request SET "
        "lease_expires_at=now()-interval '1 second' "
        "WHERE operational_request_id=71;");
    transaction.commit();
}

std::string RequestState(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT request_state || '|' || state_version "
        "FROM campaign_operations_operational_request "
        "WHERE operational_request_id=71;").one_row()[0].as<std::string>();
}

void PrintReplayPredecessor(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    const auto row = transaction.exec(
        "SELECT request.request_state,request.state_version,"
        "request.production_dispatch_enabled,request.lease_token_hash IS NULL,"
        "request.lease_expires_at IS NULL,reservation.reservation_state,"
        "reservation.expires_at IS NULL,"
        "(SELECT count(*) FROM campaign_operations_request_production_admission "
        "WHERE operational_request_id=71) "
        "FROM campaign_operations_operational_request request JOIN "
        "campaign_operations_reservation reservation USING (reservation_id) "
        "WHERE request.operational_request_id=71;").one_row();
    std::cerr << "H2_REPLAY_PREDECESSOR state=" << row[0].as<std::string>()
              << "|" << row[1].as<int>() << " enabled=" << row[2].as<bool>()
              << " lease_clear=" << row[3].as<bool>() << "/" << row[4].as<bool>()
              << " reservation=" << row[5].as<std::string>()
              << " reservation_expiry_clear=" << row[6].as<bool>()
              << " admission_count=" << row[7].as<int>() << '\n';
}

void AcquireK2WithAuthoritativeTransition(const std::string& connectionString,
    const std::string& buildCanonical)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    try
    {
        transaction.exec(
            "SELECT dispatch_attempt_id FROM "
            "transition_campaign_operations_request_dispatch_production_v2("
            "$1,$2,$3,now()+interval '300 seconds',$4,$5,$6);",
            pqxx::params{71, 5, "fnv1a64:1111111111111111", "h2-replay-k2",
                "h2.manager@example.test", buildCanonical});
        transaction.commit();
        std::cerr << "H2_REPLAY_PROGRESS k2_acquisition=PASS\n";
    }
    catch (const pqxx::sql_error& error)
    {
        std::cerr << "H2_REPLAY_PROGRESS k2_acquisition_sqlstate="
                  << error.sqlstate() << " diagnostic=" << error.what() << '\n';
        throw;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: CampaignOperationsPhaseH2ReplayAssociationTests "
                     "ENABLER_CONNECTION MANAGER_CONNECTION RECOVERY_CONNECTION\n";
        return 64;
    }

    try
    {
        const std::string enablerConnection = argv[1];
        const std::string managerConnection = argv[2];
        const std::string recoveryConnection = argv[3];
        const auto build = FixtureBuild();
        std::cerr << "H2_REPLAY_PROGRESS enable\n";

        CO::ProductionEnableRequest enableRequest{
            "h2-replay-enable-001", 0, "cee://h2/replay-reacquisition",
            CO::ActorIdentity("h2.enabler@example.test"),
            CO::Reason("H2 exact producing attempt fixture"), build, true};
        (void)CO::EnableProduction(
            [&] { return std::make_unique<pqxx::connection>(enablerConnection); },
            enableRequest);

        bool acquisitionStoppedAfterCommit = false;
        try
        {
            (void)CO::DispatchOneRequestForProductionForTest(
                managerConnection,
                CO::ProductionDispatchRequest{
                    CO::OperationalRequestId(71), 3, "h2-replay-k1",
                    CO::ActorIdentity("h2.manager@example.test"), build, true},
                [&](CO::DispatchTestInjectionPoint point)
                {
                    if (point == CO::DispatchTestInjectionPoint::
                            afterAcquisitionCommitBeforeHandoff)
                        throw std::runtime_error(
                            "H2 fixture stopped after committed K1 acquisition");
                });
        }
        catch (const std::exception& error)
        {
            acquisitionStoppedAfterCommit = std::string(error.what()).find(
                "committed K1 acquisition") != std::string::npos;
        }
        assert(acquisitionStoppedAfterCommit);
        ExpireAcquiredLeaseForDisposableFixture(recoveryConnection);
        std::cout << "H2_REPLAY_FIXTURE attempt_v2_1=PASS key=K1 "
                     "acquisition_commit=PASS\n";
        std::cerr << "H2_REPLAY_PROGRESS recover\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        const auto recovery = CO::ObserveAndRecoverCampaignOperations(
            recoveryConnection,
            CO::ReconciliationObserveRequest{
                "h2-replay-reacquisition-recovery-001", 0, 100, true});
        std::cerr << "H2_REPLAY_PROGRESS recovered\n";
        assert(recovery.resolutionCount >= 1);
        const std::string recoveredState = RequestState(recoveryConnection);
        assert(recoveredState.rfind("ready|", 0) == 0);
        const int k2ExpectedVersion = std::stoi(
            recoveredState.substr(recoveredState.find('|') + 1));
        PrintReplayPredecessor(recoveryConnection);
        AcquireK2WithAuthoritativeTransition(managerConnection,
            build.identity.canonicalText());
        std::cout << "H2_REPLAY_FIXTURE recovery=PASS state="
                  << recoveredState << " resolution_count="
                  << recovery.resolutionCount << '\n';

        const CO::ProductionDispatchRequest k2Request{
            CO::OperationalRequestId(71), k2ExpectedVersion, "h2-replay-k2",
            CO::ActorIdentity("h2.manager@example.test"), build, true};
        const auto firstK2 =
            CO::DispatchOneRequestForProductionForTest(
                managerConnection, k2Request);
        std::cerr << "H2_REPLAY_PROGRESS k2_classification="
                  << CO::ToText(firstK2.classification)
                  << " diagnostic=" << firstK2.diagnosticCode << '\n';
        assert(firstK2.classification ==
            CO::DispatchResultClassification::createdAndBound);
        const auto replayK2 =
            CO::DispatchOneRequestForProductionForTest(
                managerConnection, k2Request);
        assert(replayK2.classification ==
            CO::DispatchResultClassification::existingIdentical);
        assert(replayK2.bindingSetIdentityHash ==
            firstK2.bindingSetIdentityHash);
        std::cout << "H2_REPLAY_FIXTURE attempt_v2_2=PASS key=K2 "
                     "completed_bind=PASS k2_exact_replay=PASS\n";

        bool k1Rejected = false;
        try
        {
            (void)CO::DispatchOneRequestForProductionForTest(
                managerConnection,
                CO::ProductionDispatchRequest{
                    CO::OperationalRequestId(71), 3, "h2-replay-k1",
                    CO::ActorIdentity("h2.manager@example.test"), build, true});
        }
        catch (const std::exception& error)
        {
            k1Rejected = std::string(error.what()).find("conflicting") !=
                std::string::npos;
            std::cout << "H2_REPLAY_FIXTURE k1_replay_diagnostic="
                      << error.what() << '\n';
        }
        assert(k1Rejected);

        pqxx::connection connection{managerConnection};
        pqxx::read_transaction transaction{connection};
        const auto counts = transaction.exec(
            "SELECT (SELECT count(*) FROM "
            "campaign_operations_request_production_admission WHERE "
            "operational_request_id=71),"
            "(SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=71 AND attempt_contract_version=2),"
            "(SELECT count(*) FROM campaign_operations_request_binding "
            "WHERE operational_request_id=71),"
            "(SELECT count(*) FROM campaign_operations_dispatch_attempt_outcome "
            "outcome JOIN campaign_operations_dispatch_attempt attempt USING "
            "(dispatch_attempt_id) WHERE attempt.operational_request_id=71),"
            "(SELECT attempt.operation_key FROM "
            "campaign_operations_dispatch_attempt_outcome outcome JOIN "
            "campaign_operations_dispatch_attempt attempt USING "
            "(dispatch_attempt_id) WHERE outcome.result_classification="
            "'created_and_bound' "
            "AND attempt.operational_request_id=71 ORDER BY dispatch_attempt_id DESC "
            "LIMIT 1);").one_row();
        assert(counts[0].as<int>() == 1);
        assert(counts[1].as<int>() == 2);
        assert(counts[2].as<int>() == 1);
        assert(counts[3].as<int>() == 2);
        assert(counts[4].as<std::string>() == "h2-replay-k2");
        std::cout << "H2_REPLAY_ASSOCIATION_OK producing_attempt=K2 "
                     "k1_alias=REJECTED durable_counts=admission:1,attempt_v2:2,"
                     "binding:1,outcome:2 exact_identity=PASS\n";
        return 0;
    }
    catch (const pqxx::sql_error& error)
    {
        std::cerr << "H2_REPLAY_ASSOCIATION_FAIL diagnostic=" << error.what()
                  << " query=" << error.query() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "H2_REPLAY_ASSOCIATION_FAIL diagnostic=" << error.what()
                  << '\n';
        return 1;
    }
}
