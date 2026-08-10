#include "CampaignOperationsManagerService.hpp"
#include "CampaignOperationsManager.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <pqxx/pqxx>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace CO = EA::CampaignOperations;

namespace
{

struct Arguments final
{
    std::string socket;
    std::vector<std::string> databases;
};

std::string Connection(const Arguments& arguments, const std::string& database,
    const std::string& user, const std::string& application)
{
    return "host=" + arguments.socket + " port=5432 dbname=" + database +
        " user=" + user + " application_name=" + application;
}

const CO::ManagerBuildContract& Build()
{
    static const auto build = CO::BuildManagerBuildContract(
        CO::kManagerServiceContract,
        "777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d",
        "clang++-h3-disposable-fixture",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    return build;
}

void Require(bool condition, const std::string& diagnostic)
{
    if (!condition) throw std::runtime_error(diagnostic);
}

void Execute(const std::string& connectionString, const std::string& sql)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec(sql);
    transaction.commit();
}

std::string Scalar(const std::string& connectionString,
    const std::string& sql)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    const auto result = transaction.exec(sql);
    Require(result.size() == 1U && result.one_row().size() == 1U,
        "h3_scalar_shape_invalid");
    const auto value = result.one_row()[0].as<std::string>();
    transaction.commit();
    return value;
}

void ExecuteFixtureMutation(const std::string& connectionString,
    const std::string& mutation)
{
    Execute(connectionString,
        "SET session_replication_role=replica;" + mutation +
        " SET session_replication_role=origin;");
}

void Enable(const Arguments& arguments, const std::string& database,
    const std::string& label)
{
    const auto connection = Connection(arguments, database,
        "h2_enabler_login", "h3-" + label + "-enabler");
    const CO::ProductionEnableRequest request{
        "h3-" + label + "-enable", 0, "cee://h3/generation-52",
        CO::ActorIdentity("h3.enabler@example.test"),
        CO::Reason("H3 disposable runtime enable"), Build(), true};
    const auto result = CO::EnableProduction(
        [&] { return std::make_unique<pqxx::connection>(connection); }, request);
    Require(result.disposition == CO::ProductionMutationDisposition::newOperation,
        "h3_enable_not_new_operation_" + label);
}

void Disable(const Arguments& arguments, const std::string& database,
    const std::string& label)
{
    const auto connection = Connection(arguments, database,
        "h2_disabler_login", "h3-" + label + "-disabler");
    const CO::ProductionDisableRequest request{
        "h3-" + label + "-disable", 1,
        CO::ActorIdentity("h3.disabler@example.test"),
        CO::Reason("H3 disposable runtime disable"), true};
    const auto result = CO::DisableProduction(
        [&] { return std::make_unique<pqxx::connection>(connection); }, request);
    Require(result.disposition == CO::ProductionMutationDisposition::newOperation,
        "h3_disable_not_new_operation_" + label);
}

CO::ManagerRunOnceResult Run(const Arguments& arguments,
    const std::string& database, const std::string& application, int limit,
    const CO::ManagerTestHook& hook)
{
    return CO::RunCampaignOperationsManagerOnceForTest(
        Connection(arguments, database, "h2_manager_login", application),
        limit, Build(), hook);
}

std::string CandidateIds(const CO::ManagerRunOnceResult& result)
{
    std::string ids;
    for (const auto& candidate : result.candidates)
    {
        if (!ids.empty()) ids += ':';
        ids += std::to_string(candidate.requestId.value());
    }
    return ids;
}

std::string CountSignature(const Arguments& arguments,
    const std::string& database, long long requestId)
{
    return Scalar(Connection(arguments, database, "campaign_manager_login",
        "h3-observer"),
        "SELECT (SELECT count(*) FROM campaign_operations_request_production_admission "
        "WHERE operational_request_id=" + std::to_string(requestId) + ")::text||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=" + std::to_string(requestId) +
        " AND attempt_contract_version=2)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event "
        "WHERE operational_request_id=" + std::to_string(requestId) +
        " AND dispatch_attempt_id IS NOT NULL)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_binding "
        "WHERE operational_request_id=" + std::to_string(requestId) + ")||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_manager_operation m "
        "JOIN campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=" + std::to_string(requestId) + ");");
}

std::string RequestState(const Arguments& arguments, const std::string& database,
    long long requestId)
{
    return Scalar(Connection(arguments, database, "campaign_manager_login",
        "h3-state-observer"),
        "SELECT request_state||'|'||state_version::text||'|'||"
        "production_dispatch_enabled::text FROM campaign_operations_operational_request "
        "WHERE operational_request_id=" + std::to_string(requestId) + ";");
}

std::string IdentitySignature(const Arguments& arguments,
    const std::string& database)
{
    return Scalar(Connection(arguments, database, "campaign_manager_login",
        "h3-identity-observer"),
        "SELECT (SELECT count(*) FROM campaign_operations_production_enablement_event)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_production_admission)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE attempt_contract_version=2)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event WHERE dispatch_attempt_id IS NOT NULL)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_binding)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_manager_operation);");
}

std::string HistoricalAttemptSignature(const Arguments& arguments,
    const std::string& database)
{
    return Scalar(Connection(arguments, database, "campaign_manager_login",
        "h3-historical-attempt-observer"),
        "SELECT coalesce(string_agg(dispatch_attempt_id::text||':'||"
        "request_identity_canonical||':'||attempt_identity_canonical||':'||"
        "attempt_identity_hash, ',' ORDER BY dispatch_attempt_id),'none') "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE attempt_contract_version=1;");
}

void RequireCandidateSet(const CO::ManagerRunOnceResult& result,
    const std::string& expected, const std::string& label)
{
    Require(CandidateIds(result) == expected,
        label + " candidate snapshot changed: " + CandidateIds(result));
    for (const auto& candidate : result.candidates)
        Require(candidate.expectedRequestVersion == 3,
            label + " candidate version was not the original snapshot version");
}

void RunLocalContinuation(const Arguments& arguments,
    const std::string& database)
{
    Enable(arguments, database, "local");
    std::vector<long long> before;
    std::vector<long long> after;
    const auto admin = Connection(arguments, database,
        "campaign_manager_login", "h3-local-fixture-mutator");
    const auto result = Run(arguments, database, "h3-local-manager", 6,
        [&](CO::ManagerTestInjectionPoint point, int, CO::OperationalRequestId id)
        {
            if (point == CO::ManagerTestInjectionPoint::afterCandidateSnapshot)
                return;
            if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing)
            {
                before.push_back(id.value());
                if (id.value() == 72)
                    ExecuteFixtureMutation(admin,
                        "UPDATE campaign_operations_operational_request "
                        "SET state_version=state_version+1 WHERE operational_request_id=72;");
                if (id.value() == 73)
                    ExecuteFixtureMutation(admin,
                        "UPDATE campaign_operations_operational_request "
                        "SET request_state='dispatching' WHERE operational_request_id=73;");
                if (id.value() == 74)
                    ExecuteFixtureMutation(admin,
                        "UPDATE campaign_operations_operational_request "
                        "SET production_dispatch_enabled=true WHERE operational_request_id=74;");
            }
            else after.push_back(id.value());
        });
    RequireCandidateSet(result, "71:72:73:74:75:76", "A");
    Require(!result.stoppedEarly && result.requests.size() == 6U,
        "A did not continue through all selected candidates processed=" +
        std::to_string(result.requests.size()) + " stopped=" +
        (result.stoppedEarly ? "true" : "false") + " reason=" +
        CO::ToText(result.stopReason) + " diagnostic=" + result.stopDiagnostic);
    Require(before == std::vector<long long>({71,72,73,74,75,76}) &&
            after == before,
        "A processing trace was not bounded and sequential");
    Require(result.requests[0].outcome == CO::ManagerRequestOutcomeClassification::dispatchResult &&
            result.requests[0].newlyCommitted,
        "A prior successful request did not commit outcome=" +
        CO::ToText(result.requests[0].outcome) + " dispatch=" +
        result.requests[0].dispatchClassification + " replay=" +
        result.requests[0].replayDisposition + " diagnostic=" +
        result.requests[0].diagnosticCode);
    for (std::size_t index = 1; index <= 3; ++index)
        Require(result.requests[index].outcome ==
                CO::ManagerRequestOutcomeClassification::requestLocalSemanticFailure,
            "A local failure was not classified request-local index=" +
                std::to_string(index) + " request=" +
                std::to_string(result.requests[index].requestId.value()) +
                " outcome=" + CO::ToText(result.requests[index].outcome) +
                " diagnostic=" + result.requests[index].diagnosticCode);
    Require(result.requests[4].outcome == CO::ManagerRequestOutcomeClassification::dispatchResult &&
            result.requests[4].newlyCommitted,
        "A did not continue after local failures");
    for (long long requestId : {71LL, 75LL, 76LL})
    {
        const auto signature = CountSignature(arguments, database, requestId);
        Require(signature == "1|1|2|1|1",
            "A successful evidence incomplete for request " +
                std::to_string(requestId) + " signature=" + signature);
    }
    for (long long requestId : {72LL, 73LL, 74LL})
        Require(CountSignature(arguments, database, requestId) == "0|0|0|0|0",
            "A failed request has partial evidence " + std::to_string(requestId));
    const auto identitySignature = IdentitySignature(arguments, database);
    Require(identitySignature == "1|3|3|6|3|3",
        "A immutable evidence count changed unexpectedly signature=" +
            identitySignature);
    std::cout << "H3_A_LOCAL_CONTINUATION snapshot=PASS sequence=PASS "
                 "three_distinct_local_failures=PASS prior_commit=PASS "
                 "later_success=PASS no_partial_evidence=PASS\n";
}

void RunStopScenario(const Arguments& arguments, const std::string& database,
    const std::string& label, CO::ManagerGlobalStopReason reason,
    const CO::ManagerTestHook& hook)
{
    Enable(arguments, database, label);
    const auto result = Run(arguments, database, "h3-" + label + "-manager", 6,
        hook);
    RequireCandidateSet(result, "71:72:73:74:75:76", label);
    Require(result.stoppedEarly && result.stopReason == reason,
        label + " stop classification was not stable actual=" +
            CO::ToText(result.stopReason) + " diagnostic=" +
            result.stopDiagnostic + " processed=" +
            std::to_string(result.requests.size()) +
            (result.requests.size() > 1U
                ? " request2=" + result.requests[1].diagnosticCode : ""));
    Require(result.requests.size() == 1U && result.requests[0].requestId.value() == 71,
        label + " did not stop immediately after the first committed request");
    Require(result.requests[0].newlyCommitted,
        label + " prior request was not committed");
    for (long long requestId : {72LL, 73LL, 74LL, 75LL, 76LL})
        Require(CountSignature(arguments, database, requestId) == "0|0|0|0|0",
            label + " processed a candidate after stop");
    Require(CountSignature(arguments, database, 71) == "1|1|2|1|1",
        label + " rewrote prior committed evidence");
    std::cout << "H3_" << label << "_STOP classification=PASS immediate_stop=PASS "
                 "prior_commit=PASS later_untouched=PASS immutable_evidence=PASS\n";
}

void RunGlobalDisable(const Arguments& arguments, const std::string& database)
{
    const auto disabler = Connection(arguments, database,
        "h2_disabler_login", "h3-disable-disabler");
    RunStopScenario(arguments, database, "GLOBAL_DISABLE",
        CO::ManagerGlobalStopReason::productionDisabled,
        [&](CO::ManagerTestInjectionPoint point, int,
            CO::OperationalRequestId id)
        {
            if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing &&
                id.value() == 72)
                Disable(arguments, database, "global");
            (void)disabler;
        });
}

void RunSchedulerStop(const Arguments& arguments, const std::string& database)
{
    const auto admin = Connection(arguments, database,
        "campaign_manager_login", "h3-scheduler-mutator");
    RunStopScenario(arguments, database, "SCHEDULER_PROTOCOL",
        CO::ManagerGlobalStopReason::schedulerProtocolIneffective,
        [&](CO::ManagerTestInjectionPoint point, int,
            CO::OperationalRequestId id)
        {
            if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing &&
                id.value() == 72)
                ExecuteFixtureMutation(admin,
                    "UPDATE experiment_scheduler_protocol SET "
                    "cutover_state='failed', "
                    "cutover_completed_at=NULL, cutover_completed_by=NULL, "
                    "cutover_executable_path=NULL, cutover_process_evidence=NULL, "
                    "failure_diagnostic='H3 disposable mismatch', "
                    "updated_at=transaction_timestamp() WHERE singleton;");
        });
}

void RunPrivilegeStop(const Arguments& arguments, const std::string& database)
{
    const auto admin = Connection(arguments, database,
        "campaign_manager_login", "h3-privilege-mutator");
    RunStopScenario(arguments, database, "PRIVILEGE",
        CO::ManagerGlobalStopReason::privilegeFailure,
        [&](CO::ManagerTestInjectionPoint point, int,
            CO::OperationalRequestId id)
        {
            if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing &&
                id.value() == 72)
                Execute(admin, "REVOKE INSERT ON "
                              "campaign_operations_dispatch_manager_operation "
                              "FROM campaign_operations_production_phase5_transactional;");
        });
}

void RunDatabaseStop(const Arguments& arguments, const std::string& database)
{
    RunStopScenario(arguments, database, "DATABASE",
        CO::ManagerGlobalStopReason::databaseFailure,
        [&](CO::ManagerTestInjectionPoint point, int,
            CO::OperationalRequestId id)
        {
            if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing &&
                id.value() == 72)
                throw pqxx::broken_connection(
                    "H3 disposable database-wide connection failure");
        });
}

struct SnapshotBarrier final
{
    std::mutex mutex;
    std::condition_variable condition;
    int arrived = 0;
    bool released = false;
    const int target;

    explicit SnapshotBarrier(int targetValue) : target(targetValue) {}

    void ArriveAndWait()
    {
        std::unique_lock lock(mutex);
        ++arrived;
        condition.notify_all();
        condition.wait(lock, [&] { return released; });
    }

    void WaitUntilAllArrived()
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return arrived == target; });
    }

    void Release()
    {
        std::lock_guard lock(mutex);
        released = true;
        condition.notify_all();
    }
};

template <typename Value>
struct Async final
{
    std::optional<Value> value;
    std::exception_ptr error;
    std::mutex mutex;
    std::condition_variable condition;
    bool done = false;
};

template <typename Value, typename Function>
std::thread Start(Async<Value>& result, Function&& function)
{
    return std::thread([&result, function = std::forward<Function>(function)]
    {
        try { result.value.emplace(function()); }
        catch (...) { result.error = std::current_exception(); }
        {
            std::lock_guard lock(result.mutex);
            result.done = true;
        }
        result.condition.notify_all();
    });
}

void WaitDone(Async<CO::ManagerRunOnceResult>& result)
{
    std::unique_lock lock(result.mutex);
    result.condition.wait(lock, [&] { return result.done; });
}

struct BlockingEvidence final
{
    long waiterPid = 0;
    long blockerPid = 0;
    std::string pids;
};

std::optional<BlockingEvidence> FindBlocking(
    const std::string& observerConnection, const std::string& waiter,
    const std::string& blocker)
{
    pqxx::connection connection{observerConnection};
    pqxx::read_transaction transaction{connection};
    const auto rows = transaction.exec(
        "SELECT waiter.pid, blocker.pid, "
        "array_to_string(pg_blocking_pids(waiter.pid), ',') "
        "FROM pg_stat_activity waiter JOIN pg_stat_activity blocker "
        "ON blocker.pid=ANY(pg_blocking_pids(waiter.pid)) "
        "WHERE waiter.application_name=$1 AND blocker.application_name=$2 "
        "ORDER BY waiter.pid LIMIT 1;", pqxx::params{waiter, blocker});
    if (rows.empty())
    {
        transaction.commit();
        return std::nullopt;
    }
    const auto row = rows.one_row();
    const BlockingEvidence result{row[0].as<long>(), row[1].as<long>(),
        row[2].as<std::string>()};
    transaction.commit();
    return result;
}

BlockingEvidence WaitForBlocking(const std::string& observerConnection,
    const std::string& waiter, const std::string& blocker)
{
    for (int attempt = 0; attempt != 100000; ++attempt)
    {
        if (const auto evidence = FindBlocking(observerConnection, waiter, blocker))
            return *evidence;
        std::this_thread::yield();
    }
    throw std::runtime_error("H3 blocking evidence was not observed");
}

BlockingEvidence WaitForEitherManagerBlocking(
    const std::string& observerConnection, const std::string& firstWaiter,
    const std::string& secondWaiter, const std::string& blocker)
{
    for (int attempt = 0; attempt != 100000; ++attempt)
    {
        pqxx::connection connection{observerConnection};
        pqxx::read_transaction transaction{connection};
        const auto rows = transaction.exec(
            "SELECT waiter.pid, blocker.pid, "
            "array_to_string(pg_blocking_pids(waiter.pid), ',') "
            "FROM pg_stat_activity waiter JOIN pg_stat_activity blocker "
            "ON blocker.pid=ANY(pg_blocking_pids(waiter.pid)) "
            "WHERE waiter.application_name IN ($1,$2) "
            "AND blocker.application_name=$3 ORDER BY waiter.pid LIMIT 1;",
            pqxx::params{firstWaiter, secondWaiter, blocker});
        if (!rows.empty())
        {
            const auto row = rows.one_row();
            const BlockingEvidence result{row[0].as<long>(), row[1].as<long>(),
                row[2].as<std::string>()};
            transaction.commit();
            return result;
        }
        transaction.commit();
        std::this_thread::yield();
    }
    throw std::runtime_error("H3 Manager blocking evidence was not observed");
}

void CheckAsync(const Async<CO::ManagerRunOnceResult>& result,
    const std::string& label)
{
    if (result.error)
    {
        try { std::rethrow_exception(result.error); }
        catch (const std::exception& error)
        { throw std::runtime_error(label + ": " + error.what()); }
    }
    Require(result.value.has_value(), label + " returned no result");
}

void RunOverlappingManagers(const Arguments& arguments,
    const std::string& database)
{
    Enable(arguments, database, "overlap");
    SnapshotBarrier snapshots(2);
    const CO::ManagerTestHook hook = [&](CO::ManagerTestInjectionPoint point,
        int, CO::OperationalRequestId)
    {
        if (point == CO::ManagerTestInjectionPoint::afterCandidateSnapshot)
            snapshots.ArriveAndWait();
    };
    Async<CO::ManagerRunOnceResult> first;
    Async<CO::ManagerRunOnceResult> second;
    auto firstThread = Start(first, [&]
    {
        return Run(arguments, database, "h3-overlap-a", 1, hook);
    });
    auto secondThread = Start(second, [&]
    {
        return Run(arguments, database, "h3-overlap-b", 1, hook);
    });
    snapshots.WaitUntilAllArrived();
    pqxx::connection blockerConnection{Connection(arguments, database,
        "campaign_manager_login", "h3-overlap-blocker")};
    pqxx::work blocker{blockerConnection};
    blocker.exec("SELECT operational_request_id FROM "
                 "campaign_operations_operational_request WHERE "
                 "operational_request_id=71 FOR UPDATE;");
    snapshots.Release();
    const auto blocking = WaitForEitherManagerBlocking(
        Connection(arguments, database, "campaign_manager_login",
            "h3-overlap-observer"),
        "h3-overlap-a", "h3-overlap-b", "h3-overlap-blocker");
    blocker.commit();
    firstThread.join();
    secondThread.join();
    CheckAsync(first, "F first Manager");
    CheckAsync(second, "F second Manager");
    const auto& a = *first.value;
    const auto& b = *second.value;
    RequireCandidateSet(a, "71", "F-A");
    RequireCandidateSet(b, "71", "F-B");
    Require(a.candidates[0].requestIdentityCanonical ==
            b.candidates[0].requestIdentityCanonical,
        "F Managers did not snapshot the same source canonical");
    Require(CO::BuildManagerRequestOperationIdentity(
                a.candidates[0].requestIdentityCanonical,
                a.candidates[0].expectedRequestVersion).operationKey ==
            CO::BuildManagerRequestOperationIdentity(
                b.candidates[0].requestIdentityCanonical,
                b.candidates[0].expectedRequestVersion).operationKey,
        "F Managers did not derive the same deterministic operation key");
    Require(a.requests.size() == 1U && b.requests.size() == 1U,
        "F Manager request result shape changed");
    Require((a.requests[0].newlyCommitted && b.requests[0].exactReplay) ||
            (b.requests[0].newlyCommitted && a.requests[0].exactReplay),
        "F did not resolve through one new operation and one exact replay");
    Require(CountSignature(arguments, database, 71) == "1|1|2|1|1",
        "F duplicate admission/Attempt/binding/source evidence exists");
    std::cout << "H3_F_OVERLAPPING_SNAPSHOT same_request_version=PASS "
                 "same_source_canonical=PASS deterministic_key=PASS "
                 "one_operation_one_replay=PASS no_duplicate_evidence=PASS "
                 "waiter_pid=" << blocking.waiterPid << " blocker_pid="
              << blocking.blockerPid << " pg_blocking_pids=" << blocking.pids
              << "\n";
}

void RunUnrelatedConcurrency(const Arguments& arguments,
    const std::string& database)
{
    Enable(arguments, database, "unrelated");
    SnapshotBarrier snapshots(2);
    const CO::ManagerTestHook firstHook = [&](CO::ManagerTestInjectionPoint point,
        int, CO::OperationalRequestId)
    {
        if (point == CO::ManagerTestInjectionPoint::afterCandidateSnapshot)
            snapshots.ArriveAndWait();
    };
    const CO::ManagerTestHook secondHook = [&](CO::ManagerTestInjectionPoint point,
        int, CO::OperationalRequestId id)
    {
        if (point == CO::ManagerTestInjectionPoint::afterCandidateSnapshot)
            snapshots.ArriveAndWait();
        if (point == CO::ManagerTestInjectionPoint::beforeCandidateProcessing &&
            id.value() == 71)
            throw std::runtime_error("h3_test_request_local_skip_for_unrelated_race");
    };
    Async<CO::ManagerRunOnceResult> first;
    Async<CO::ManagerRunOnceResult> second;
    auto firstThread = Start(first, [&]
    { return Run(arguments, database, "h3-unrelated-a", 1, firstHook); });
    auto secondThread = Start(second, [&]
    { return Run(arguments, database, "h3-unrelated-b", 2, secondHook); });
    snapshots.WaitUntilAllArrived();
    pqxx::connection blockerConnection{Connection(arguments, database,
        "campaign_manager_login", "h3-unrelated-blocker")};
    pqxx::work blocker{blockerConnection};
    blocker.exec("SELECT operational_request_id FROM "
                 "campaign_operations_operational_request WHERE "
                 "operational_request_id=71 FOR UPDATE;");
    snapshots.Release();
    WaitDone(second);
    const auto blocking = WaitForBlocking(Connection(arguments, database,
        "campaign_manager_login", "h3-unrelated-observer"),
        "h3-unrelated-a", "h3-unrelated-blocker");
    secondThread.join();
    CheckAsync(second, "G unrelated Manager");
    blocker.commit();
    firstThread.join();
    CheckAsync(first, "G first Manager");
    Require(second.value->requests.size() == 2U &&
            second.value->requests[0].outcome ==
                CO::ManagerRequestOutcomeClassification::requestLocalSemanticFailure &&
            second.value->requests[1].requestId.value() == 72 &&
            second.value->requests[1].newlyCommitted,
        "G unrelated request did not progress while X was blocked");
    Require(first.value->requests.size() == 1U &&
            first.value->requests[0].newlyCommitted,
        "G blocked request X did not complete after its lock was released");
    std::cout << "H3_G_UNRELATED_CONCURRENCY independent_progress=PASS "
                 "shared_authority_domain_only=PASS no_manager_mutex=PASS "
                 "waiter_pid=" << blocking.waiterPid << " blocker_pid="
              << blocking.blockerPid << " pg_blocking_pids=" << blocking.pids
              << "\n";
}

struct CandidateIdentity final
{
    std::string requestCanonical;
    CO::ManagerRequestOperationIdentity operation;
    CO::ProductionDispatchRequest request;
};

CandidateIdentity IdentityFor(const Arguments& arguments,
    const std::string& database, long long requestId)
{
    pqxx::connection connection{Connection(arguments, database,
        "campaign_manager_login", "h3-evidence-reader")};
    pqxx::read_transaction transaction{connection};
    const auto row = transaction.exec(
        "SELECT request_identity_canonical,state_version FROM "
        "campaign_operations_operational_request WHERE operational_request_id=$1;",
        pqxx::params{requestId}).one_row();
    const std::string canonical = row[0].as<std::string>();
    (void)row[1].as<int>();
    const auto operation = CO::BuildManagerRequestOperationIdentity(canonical, 3);
    transaction.commit();
    return {canonical, std::move(operation),
        {CO::OperationalRequestId(requestId), 3,
         CO::BuildManagerRequestOperationIdentity(canonical, 3).operationKey,
         CO::ActorIdentity(CO::kCampaignOperationsManagerActor), Build(), true}};
}

void ExpectManagerReplayFailure(const Arguments& arguments,
    const std::string& database, const CandidateIdentity& identity,
    const std::string& source, const std::string& label)
{
    bool failed = false;
    try
    {
        (void)CO::DispatchOneRequestForProductionManagerWithFixture(
            Connection(arguments, database, "h2_manager_login", "h3-" + label),
            identity.request, source, Build());
    }
    catch (const std::exception& error)
    {
        failed = std::string(error.what()).find("conflict") != std::string::npos ||
            std::string(error.what()).find("evidence") != std::string::npos ||
            std::string(error.what()).find("missing") != std::string::npos ||
            std::string(error.what()).find("invalid") != std::string::npos ||
            std::string(error.what()).find("corrupt") != std::string::npos;
    }
    Require(failed, label + " did not fail closed");
}

void AcquireManagerShapedAttemptWithoutSource(const Arguments& arguments,
    const std::string& database, const CandidateIdentity& identity,
    const std::string& label, bool bypassConstraints)
{
    pqxx::connection connection{Connection(arguments, database,
        "campaign_manager_login", "h3-rv001-" + label)};
    pqxx::work transaction{connection};
    if (bypassConstraints)
        transaction.exec("SET LOCAL session_replication_role=replica;");
    const auto expiry = transaction.exec(
        "SELECT (transaction_timestamp()+make_interval(secs=>300))::text;")
        .one_row()[0].as<std::string>();
    const auto lease = CO::LeaseTokenDigest::Derive(
        std::string(32U, 'h') + "-h3-rv001-" + label + "-lease");
    const auto rows = transaction.exec(
        "SELECT dispatch_attempt_id FROM "
        "transition_campaign_operations_request_dispatch_production_v2("
        "$1,$2,$3,$4::timestamptz,$5,$6,$7);",
        pqxx::params{identity.request.requestId.value(),
            identity.request.expectedRequestVersion, lease.value(), expiry,
            identity.request.operationKey,
            identity.request.requestingActor.value(),
            identity.request.executingBuild.identity.canonicalText()});
    Require(rows.size() == 1U,
        "H3-RV-001 raw acquisition returned no Attempt V2");
    transaction.commit();
}

void RunSourceEvidence(const Arguments& arguments, const std::string& database)
{
    const auto historicalBefore = HistoricalAttemptSignature(arguments, database);
    Enable(arguments, database, "source");
    const auto one = IdentityFor(arguments, database, 71);
    const auto two = IdentityFor(arguments, database, 72);
    const auto three = IdentityFor(arguments, database, 73);
    const auto four = IdentityFor(arguments, database, 74);
    const auto five = IdentityFor(arguments, database, 75);
    const auto six = IdentityFor(arguments, database, 76);

    bool callerKeyRejected = false;
    try
    {
        (void)CO::DispatchOneRequestForProductionForTest(
            Connection(arguments, database, "h2_manager_login",
                "h3-rv001-caller-key"), five.request);
    }
    catch (const std::invalid_argument& error)
    {
        callerKeyRejected = std::string(error.what()).find("reserved") !=
            std::string::npos;
    }
    Require(callerKeyRejected,
        "H3-RV-001 caller-keyed H2 Manager namespace was not rejected");
    Require(CountSignature(arguments, database, 75) == "0|0|0|0|0" &&
            RequestState(arguments, database, 75) == "ready|3|false",
        "H3-RV-001 rejected caller path changed durable state");

    bool databaseCommitRejected = false;
    try
    {
        AcquireManagerShapedAttemptWithoutSource(arguments, database, five,
            "normal", false);
    }
    catch (const pqxx::sql_error& error)
    {
        databaseCommitRejected = error.sqlstate() == "23514";
    }
    Require(databaseCommitRejected,
        "H3-RV-001 COMMIT accepted Manager Attempt V2 without source");
    Require(CountSignature(arguments, database, 75) == "0|0|0|0|0" &&
            RequestState(arguments, database, 75) == "ready|3|false",
        "H3-RV-001 database rejection left an incomplete Attempt V2");

    const auto rolled = CO::DispatchOneRequestForProductionManagerWithFixture(
        Connection(arguments, database, "h2_manager_login", "h3-source-rollback"),
        four.request, four.operation.source.canonicalText(), Build(),
        [](CO::DispatchTestInjectionPoint point)
        {
            if (point == CO::DispatchTestInjectionPoint::beforeAcquisitionCommit)
                throw CO::DispatchTestSqlState("40001");
        });
    Require(rolled.failure == CO::DispatchServiceFailureClassification::
                transientDatabaseRetryExhausted,
        "H rollback injection did not exhaust without guessing success");
    Require(CountSignature(arguments, database, 74) == "0|0|0|0|0" &&
            RequestState(arguments, database, 74) == "ready|3|false",
        "H rollback did not remove Attempt V2/source atomically");

    bool interruptedAfterAcquisition = false;
    try
    {
        (void)CO::DispatchOneRequestForProductionManagerWithFixture(
            Connection(arguments, database, "h2_manager_login",
                "h3-rv001-boundary-interrupt"), six.request,
            six.operation.source.canonicalText(), Build(),
            [](CO::DispatchTestInjectionPoint point)
            {
                if (point == CO::DispatchTestInjectionPoint::
                        afterAcquisitionCommitBeforeHandoff)
                    throw std::runtime_error("h3-rv001-interrupted-after-acquisition");
            });
    }
    catch (const std::runtime_error& error)
    {
        interruptedAfterAcquisition = std::string(error.what()).find(
            "h3-rv001-interrupted") != std::string::npos;
    }
    Require(interruptedAfterAcquisition &&
            CountSignature(arguments, database, 76) == "1|1|1|0|1",
        "H3-RV-001 acquisition boundary did not retain complete source evidence");
    const auto boundaryRecovery =
        CO::DispatchOneRequestForProductionManagerWithFixture(
            Connection(arguments, database, "h2_manager_login",
                "h3-rv001-boundary-recovery"), six.request,
            six.operation.source.canonicalText(), Build());
    Require(boundaryRecovery.replayDisposition ==
                CO::ExactReplayDisposition::newOperation &&
            CountSignature(arguments, database, 76) == "1|1|2|1|1",
        "H3-RV-001 complete boundary recovery did not use exact source evidence");

    const auto result = Run(arguments, database, "h3-rv001-manager", 5,
        [&](CO::ManagerTestInjectionPoint point, int,
            CO::OperationalRequestId)
        {
            if (point == CO::ManagerTestInjectionPoint::afterCandidateSnapshot)
                AcquireManagerShapedAttemptWithoutSource(arguments, database,
                    five, "legacy", true);
        });
    Require(result.stoppedEarly &&
            result.stopReason == CO::ManagerGlobalStopReason::databaseFailure &&
            result.requests.size() == 4U,
        "H3-RV-001 Manager run-once adopted missing-source recovery");
    Require(CountSignature(arguments, database, 75) == "1|1|1|0|0" &&
            RequestState(arguments, database, 75) == "dispatching|4|true",
        "H3-RV-001 legacy missing-source state was bound or backfilled");

    const auto exact = CO::DispatchOneRequestForProductionManagerWithFixture(
        Connection(arguments, database, "h2_manager_login", "h3-source-replay"),
        one.request, one.operation.source.canonicalText(), Build());
    Require(exact.replayDisposition ==
            CO::ExactReplayDisposition::authoritativeExisting,
        "H exact Manager replay did not full-compare successfully");
    Require(Scalar(Connection(arguments, database, "campaign_manager_login",
        "h3-source-proof"),
        "SELECT count(*)::text FROM campaign_operations_dispatch_manager_operation "
        "m JOIN campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=71 AND m.operation_key='" +
        one.operation.operationKey + "' AND m.expected_request_version=3 AND "
        "m.source_canonical='" + one.operation.source.canonicalText() + "';") == "1",
        "H source evidence canonical/key/version association mismatch");
    ExpectManagerReplayFailure(arguments, database, one,
        one.operation.source.canonicalText() + ";altered", "H-altered-canonical");
    ExecuteFixtureMutation(Connection(arguments, database, "campaign_manager_login",
        "h3-source-mutation"),
        "UPDATE campaign_operations_dispatch_manager_operation SET "
        "source_canonical=source_canonical||';same-hash-altered' "
        "WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id FROM "
        "campaign_operations_dispatch_manager_operation m JOIN "
        "campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=71);" );
    ExpectManagerReplayFailure(arguments, database, one,
        one.operation.source.canonicalText(), "H-same-hash-different-canonical");
    ExecuteFixtureMutation(Connection(arguments, database, "campaign_manager_login",
        "h3-source-missing"),
        "DELETE FROM campaign_operations_dispatch_manager_operation WHERE "
        "dispatch_attempt_id=(SELECT dispatch_attempt_id FROM "
        "campaign_operations_dispatch_manager_operation m JOIN "
        "campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=72);" );
    ExpectManagerReplayFailure(arguments, database, two,
        two.operation.source.canonicalText(), "H-missing-evidence");
    bool duplicateRejected = false;
    try
    {
        Execute(Connection(arguments, database, "campaign_manager_login",
            "h3-source-duplicate"),
            "INSERT INTO campaign_operations_dispatch_manager_operation "
            "SELECT * FROM campaign_operations_dispatch_manager_operation "
            "WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_manager_operation m JOIN "
            "campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
            "WHERE a.operational_request_id=73);");
    }
    catch (const pqxx::sql_error&) { duplicateRejected = true; }
    Require(duplicateRejected, "H duplicate source evidence was not rejected");
    ExecuteFixtureMutation(Connection(arguments, database, "campaign_manager_login",
        "h3-source-mismatch"),
        "DELETE FROM campaign_operations_dispatch_manager_operation WHERE "
        "dispatch_attempt_id=(SELECT dispatch_attempt_id FROM "
        "campaign_operations_dispatch_manager_operation m JOIN "
        "campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=74);" );
    ExecuteFixtureMutation(Connection(arguments, database, "campaign_manager_login",
        "h3-source-mismatch-insert"),
        "INSERT INTO campaign_operations_dispatch_manager_operation "
        "SELECT (SELECT dispatch_attempt_id FROM campaign_operations_dispatch_attempt a4 "
        "WHERE a4.operational_request_id=74 AND a4.attempt_contract_version=2),"
        "74,m.operation_key,m.request_identity_canonical,"
        "m.expected_request_version,m.source_canonical,m.source_hash,m.contract_version,m.created_at "
        "FROM campaign_operations_dispatch_manager_operation m JOIN "
        "campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) "
        "WHERE a.operational_request_id=73;" );
    ExpectManagerReplayFailure(arguments, database, four,
        four.operation.source.canonicalText(), "H-mismatched-attempt-association");
    Require(HistoricalAttemptSignature(arguments, database) == historicalBefore,
        "H historical Attempt V1 canonical bytes changed");
    std::cout << "H3_H3RV001 caller_namespace=PASS commit_completeness=PASS "
                 "boundary_recovery=PASS legacy_missing_source=PASS no_backfill=PASS "
                 "H3_H_MIGRATION058 source_exact=PASS attempt_association=PASS "
                 "expected_version=PASS deterministic_key=PASS full_replay=PASS "
                 "missing=PASS duplicate=PASS mismatch=PASS altered=PASS "
                 "same_hash_different_canonical=PASS rollback_atomic=PASS "
                 "historical_attempt_bytes_unchanged=PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        std::cerr << "usage: CampaignOperationsPhaseH3RuntimeConcurrencyTests "
                     "SOCKET LOCAL DISABLE SCHEDULER PRIVILEGE DATABASE "
                     "OVERLAP UNRELATED SOURCE\n";
        return 64;
    }
    try
    {
        Arguments arguments{argv[1], {argv[2], argv[3], argv[4], argv[5],
            argv[6], argv[7], argv[8], argv[9]}};
        RunLocalContinuation(arguments, arguments.databases[0]);
        RunGlobalDisable(arguments, arguments.databases[1]);
        RunSchedulerStop(arguments, arguments.databases[2]);
        RunPrivilegeStop(arguments, arguments.databases[3]);
        RunDatabaseStop(arguments, arguments.databases[4]);
        RunOverlappingManagers(arguments, arguments.databases[5]);
        RunUnrelatedConcurrency(arguments, arguments.databases[6]);
        RunSourceEvidence(arguments, arguments.databases[7]);
        std::cout << "H3_RUNTIME_HARNESS_OK A=PASS B=PASS C=PASS D=PASS E=PASS "
                     "F=PASS G=PASS H=PASS I=PASS J=PASS\n";
        return 0;
    }
    catch (const pqxx::sql_error& error)
    {
        std::cerr << "H3_RUNTIME_HARNESS_FAIL sqlstate=" << error.sqlstate()
                  << " diagnostic=" << error.what() << " query="
                  << error.query() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "H3_RUNTIME_HARNESS_FAIL diagnostic=" << error.what() << '\n';
        return 1;
    }
}
