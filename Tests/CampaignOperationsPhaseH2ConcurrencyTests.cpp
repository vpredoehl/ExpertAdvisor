#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <pqxx/pqxx>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace CO = EA::CampaignOperations;

namespace
{

using namespace std::chrono_literals;

const CO::ManagerBuildContract& Build()
{
    static const auto build = CO::BuildManagerBuildContract(
        CO::kManagerServiceContract,
        "777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d",
        "clang++-h2-fixture",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    return build;
}

std::string Conn(const std::string& socket, const std::string& database,
    const std::string& user, const std::string& application)
{
    return "host=" + socket + " port=5432 dbname=" + database +
        " user=" + user + " application_name=" + application;
}

struct HoldPoint final
{
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool released = false;

    template <typename Point>
    void Hold(Point point, Point wanted)
    {
        if (point != wanted) return;
        std::unique_lock lock(mutex);
        entered = true;
        condition.notify_all();
        condition.wait(lock, [&] { return released; });
    }

    void WaitUntilEntered(const std::string& label)
    {
        std::unique_lock lock(mutex);
        if (!condition.wait_for(lock, 8s, [&] { return entered; }))
            throw std::runtime_error(label + " did not reach held lock point");
    }

    void Release()
    {
        std::lock_guard lock(mutex);
        released = true;
        condition.notify_all();
    }
};

template <typename Value>
struct AsyncResult final
{
    std::optional<Value> value;
    std::exception_ptr error;
};

template <typename Function, typename Value>
std::thread Start(AsyncResult<Value>& result, Function&& function)
{
    return std::thread([&result, function = std::forward<Function>(function)]
    {
        try { result.value.emplace(function()); }
        catch (...) { result.error = std::current_exception(); }
    });
}

void Rethrow(const std::exception_ptr& error, const std::string& label)
{
    if (!error) return;
    try { std::rethrow_exception(error); }
    catch (const std::exception& exception)
    {
        throw std::runtime_error(label + ": " + exception.what());
    }
}

CO::ProductionMutationResult Enable(const std::string& connection)
{
    const CO::ProductionEnableRequest request{
        "h2-enable", 0, "cee://h2/generation-52",
        CO::ActorIdentity("h2.enabler@example.test"),
        CO::Reason("H2 concurrency fixture enable"), Build(), true};
    return CO::EnableProduction(
        [&]
        {
            return std::make_unique<pqxx::connection>(connection);
        }, request);
}

CO::ProductionMutationResult Disable(const std::string& connection,
    CO::ProductionMutationTestHook testHook = {})
{
    const CO::ProductionDisableRequest request{
        "h2-disable", 1, CO::ActorIdentity("h2.disabler@example.test"),
        CO::Reason("H2 concurrency fixture disable"), true};
    bool pidPrinted = false;
    return CO::DisableProduction(
        [&]
        {
            auto result = std::make_unique<pqxx::connection>(connection);
            if (!pidPrinted)
            {
                pqxx::read_transaction transaction{*result};
                std::cout << "H2_ACTOR_PID label=disable pid="
                          << transaction.exec("SELECT pg_backend_pid();")
                                 .one_row()[0].as<long>() << '\n';
                transaction.commit();
                pidPrinted = true;
            }
            return result;
        }, request, std::move(testHook));
}

CO::ProductionDispatchRequest DispatchRequest(const std::string& key)
{
    return {CO::OperationalRequestId(71), 3, key,
        CO::ActorIdentity("h2.manager@example.test"), Build(), true};
}

struct BlockingEvidence final
{
    long waiterPid = 0;
    long blockerPid = 0;
    std::string waiterApplication;
    std::string blockerApplication;
    std::string waitEvent;
    std::string lockCatalog;
};

std::optional<BlockingEvidence> FindBlocking(
    pqxx::connection& observer, const std::string& waiterApplication,
    const std::string& blockerApplication)
{
    pqxx::read_transaction transaction{observer};
    const auto rows = transaction.exec(
        "SELECT waiter.pid, blocker.pid, waiter.application_name, "
        "blocker.application_name, coalesce(waiter.wait_event,''), "
        "array_to_string(pg_blocking_pids(waiter.pid), ',') "
        "FROM pg_stat_activity waiter "
        "JOIN pg_stat_activity blocker "
        "  ON blocker.pid = ANY(pg_blocking_pids(waiter.pid)) "
        "WHERE waiter.application_name=$1 "
        "  AND blocker.application_name=$2 "
        "  AND blocker.pid <> waiter.pid "
        "ORDER BY waiter.pid LIMIT 1;",
        pqxx::params{waiterApplication, blockerApplication});
    if (rows.empty())
    {
        transaction.commit();
        return std::nullopt;
    }
    const auto row = rows.one_row();
    const auto locks = transaction.exec(
        "SELECT locktype||':'||mode||':'||granted::text||':'||"
        "coalesce(classid::text,'')||':'||coalesce(objid::text,'') "
        "FROM pg_locks WHERE pid IN ($1,$2) "
        "ORDER BY pid,locktype,mode,granted;",
        pqxx::params{row[0].as<long>(), row[1].as<long>()});
    std::string lockCatalog;
    for (const auto& lock : locks)
    {
        if (!lockCatalog.empty()) lockCatalog += ',';
        lockCatalog += lock[0].as<std::string>();
    }
    transaction.commit();
    return BlockingEvidence{row[0].as<long>(), row[1].as<long>(),
        row[2].as<std::string>(), row[3].as<std::string>(),
        row[4].as<std::string>(), lockCatalog};
}

BlockingEvidence WaitForBlocking(const std::string& observerConnection,
    const std::string& waiterApplication, const std::string& blockerApplication,
    const std::string& label)
{
    pqxx::connection observer{observerConnection};
    const auto deadline = std::chrono::steady_clock::now() + 8s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (const auto evidence = FindBlocking(observer, waiterApplication,
                blockerApplication))
        {
            std::cout << "H2_BLOCKING_PIDS case=" << label
                      << " waiter_pid=" << evidence->waiterPid
                      << " blocker_pid=" << evidence->blockerPid
                      << " waiter_app=" << evidence->waiterApplication
                      << " blocker_app=" << evidence->blockerApplication
                      << " wait_event=" << evidence->waitEvent
                      << " locks=" << evidence->lockCatalog << '\n';
            return *evidence;
        }
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error(label + " did not produce catalog blocking evidence");
}

std::string Snapshot(pqxx::connection& connection, const std::string& label)
{
    pqxx::read_transaction transaction{connection};
    const auto row = transaction.exec(
        "SELECT r.request_state||'|'||r.state_version::text||'|'||"
        "r.production_dispatch_enabled::text||'|'||"
        "(r.lease_token_hash IS NULL AND r.lease_expires_at IS NULL AND "
        " r.dispatcher_identity IS NULL)::text||'|'||"
        "(SELECT count(*) FROM campaign_operations_production_enablement_event)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_production_admission "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt "
        " WHERE operational_request_id=71 AND attempt_contract_version=2)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_binding "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_downstream_control_owner "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_reservation_commitment "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt_outcome outcome "
        " JOIN campaign_operations_dispatch_attempt attempt "
        "   ON attempt.dispatch_attempt_id=outcome.dispatch_attempt_id "
        " WHERE attempt.operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM experiment_recommendation_conversion_execution execution "
        " JOIN campaign_operations_request_binding binding "
        "   ON binding.recommendation_conversion_execution_id="
        "      execution.recommendation_conversion_execution_id "
        " WHERE binding.operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM experiment_recommendation_conversion_activation activation "
        " JOIN campaign_operations_request_binding binding "
        "   ON binding.recommendation_conversion_activation_id="
        "      activation.recommendation_conversion_activation_id "
        " WHERE binding.operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_binding "
        " WHERE operational_request_id=71)||'|'||"
        "coalesce((SELECT string_agg(dispatch_attempt_id::text||':'||operation_key, ',' "
        "ORDER BY dispatch_attempt_id) FROM campaign_operations_dispatch_attempt "
        " WHERE operational_request_id=71 AND attempt_contract_version=2),'none')||'|'||"
        "coalesce((SELECT string_agg(request_binding_id::text||':'||binding_identity_hash, ',' "
        "ORDER BY request_binding_id) FROM campaign_operations_request_binding "
        " WHERE operational_request_id=71),'none') "
        "FROM campaign_operations_operational_request r "
        "WHERE r.operational_request_id=71;").one_row();
    transaction.commit();
    const std::string result = row[0].as<std::string>();
    std::cout << "H2_FINAL_STATE case=" << label << " fields=" << result << '\n';
    return result;
}

void PrintIdentities(pqxx::connection& connection, const std::string& label)
{
    pqxx::read_transaction transaction{connection};
    const auto row = transaction.exec(
        "SELECT "
        "coalesce((SELECT string_agg(dispatch_attempt_id::text||':'||operation_key, ',' "
        "ORDER BY dispatch_attempt_id) FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=71 AND attempt_contract_version=2),'none'),"
        "coalesce((SELECT string_agg(request_binding_id::text||':'||binding_identity_hash, ',' "
        "ORDER BY request_binding_id) FROM campaign_operations_request_binding "
        "WHERE operational_request_id=71),'none'),"
        "coalesce((SELECT string_agg(downstream_control_owner_id::text||':'||experiment_id::text, ',' "
        "ORDER BY downstream_control_owner_id) FROM campaign_operations_downstream_control_owner "
        "WHERE operational_request_id=71),'none'),"
        "coalesce((SELECT string_agg(reservation_commitment_id::text||':'||operational_request_id::text, ',' "
        "ORDER BY reservation_commitment_id) FROM campaign_operations_reservation_commitment "
        "WHERE operational_request_id=71),'none'),"
        "coalesce((SELECT string_agg(outcome.dispatch_attempt_outcome_id::text||':'||outcome.outcome_identity_hash, ',' "
        "ORDER BY outcome.dispatch_attempt_outcome_id) "
        "FROM campaign_operations_dispatch_attempt_outcome outcome "
        "JOIN campaign_operations_dispatch_attempt attempt "
        " ON attempt.dispatch_attempt_id=outcome.dispatch_attempt_id "
        "WHERE attempt.operational_request_id=71),'none'),"
        "coalesce((SELECT string_agg(execution.recommendation_conversion_execution_id::text||':'||execution.execution_identity_hash, ',' "
        "ORDER BY execution.recommendation_conversion_execution_id) "
        "FROM experiment_recommendation_conversion_execution execution "
        "JOIN campaign_operations_request_binding binding "
        " ON binding.recommendation_conversion_execution_id=execution.recommendation_conversion_execution_id "
        "WHERE binding.operational_request_id=71),'none'),"
        "coalesce((SELECT string_agg(activation.recommendation_conversion_activation_id::text||':'||activation.experiment_id::text, ',' "
        "ORDER BY activation.recommendation_conversion_activation_id) "
        "FROM experiment_recommendation_conversion_activation activation "
        "JOIN campaign_operations_request_binding binding "
        " ON binding.recommendation_conversion_activation_id=activation.recommendation_conversion_activation_id "
        "WHERE binding.operational_request_id=71),'none');").one_row();
    transaction.commit();
    std::cout << "H2_IDENTITIES case=" << label
              << " attempt=" << row[0].as<std::string>()
              << " binding=" << row[1].as<std::string>()
              << " owner_experiment=" << row[2].as<std::string>()
              << " commitment=" << row[3].as<std::string>()
              << " outcome=" << row[4].as<std::string>()
              << " execution=" << row[5].as<std::string>()
              << " activation_experiment=" << row[6].as<std::string>() << '\n';
}

std::vector<std::string> Fields(const std::string& value)
{
    std::vector<std::string> fields;
    std::stringstream stream(value);
    std::string field;
    while (std::getline(stream, field, '|')) fields.push_back(field);
    return fields;
}

void RequireCounts(const std::string& snapshot, const std::string& label,
    const std::vector<std::string>& expected)
{
    const auto fields = Fields(snapshot);
    if (fields.size() != 17U)
        throw std::runtime_error(label + " snapshot shape changed");
    for (std::size_t index = 4; index <= 13; ++index)
        if (fields[index] != expected[index - 4])
            throw std::runtime_error(label + " count mismatch at field " +
                std::to_string(index) + " actual=" + fields[index] +
                " expected=" + expected[index - 4]);
}

void RequireNoError(const AsyncResult<CO::DispatchServiceResult>& result,
    const std::string& label)
{
    Rethrow(result.error, label);
    if (!result.value) throw std::runtime_error(label + " returned no result");
}

void RequireNoError(const AsyncResult<CO::ProductionMutationResult>& result,
    const std::string& label)
{
    Rethrow(result.error, label);
    if (!result.value) throw std::runtime_error(label + " returned no result");
}

void RunDisableAcquisition(const std::string& socket,
    const std::string& database)
{
    const auto admin = Conn(socket, database, "campaign_manager_login", "h2-observer-c1");
    const auto enabler = Conn(socket, database, "h2_enabler_login", "h2-c1-enabler");
    const auto disabler = Conn(socket, database, "h2_disabler_login", "h2-c1-disable");
    const auto manager = Conn(socket, database, "h2_manager_login", "h2-c1-acquisition");
    (void)Enable(enabler);
    HoldPoint hold;
    AsyncResult<CO::DispatchServiceResult> acquisition;
    auto acquisitionThread = Start(acquisition, [&]
    {
        return CO::DispatchOneRequestForProductionForTest(
            manager, DispatchRequest("h2-c1-dispatch"), [&](CO::DispatchTestInjectionPoint point)
            { hold.Hold(point, CO::DispatchTestInjectionPoint::beforeAcquisitionCommit); });
    });
    hold.WaitUntilEntered("C1 acquisition");
    AsyncResult<CO::ProductionMutationResult> disable;
    auto disableThread = Start(disable, [&] { return Disable(disabler); });
    (void)WaitForBlocking(admin, "h2-c1-disable", "h2-c1-acquisition", "C1");
    hold.Release();
    acquisitionThread.join();
    disableThread.join();
    RequireNoError(disable, "C1 disable");
    if (acquisition.error)
        std::cout << "H2_RACE_RESULT case=C1 acquisition=fail_closed\n";
    else
        std::cout << "H2_RACE_RESULT case=C1 acquisition="
                  << CO::ToText(acquisition.value->classification) << '\n';
    auto observer = std::make_unique<pqxx::connection>(admin);
    const auto snapshot = Snapshot(*observer, "C1");
    PrintIdentities(*observer, "C1");
    const auto fields = Fields(snapshot);
    if (fields[0] != "dispatching" || fields[1] != "4" || fields[2] != "true" ||
        fields[3] != "false")
        throw std::runtime_error("C1 stale authorization/state result");
    RequireCounts(snapshot, "C1", {"2", "1", "1", "1", "0", "0", "0", "0", "0", "0"});
    std::cout << "H2_RACE_PASS case=C1 acquisition_won_gate=PASS disable_blocked=PASS "
                 "disable_committed=PASS no_downstream_duplicate=PASS\n";
}

void RunDisableFirst(const std::string& socket, const std::string& database,
    const std::string& label)
{
    const auto admin = Conn(socket, database, "campaign_manager_login", "h2-observer-" + label);
    const auto enabler = Conn(socket, database, "h2_enabler_login", "h2-" + label + "-enabler");
    const auto disabler = Conn(socket, database, "h2_disabler_login", "h2-" + label + "-disable");
    const auto manager = Conn(socket, database, "h2_manager_login", "h2-" + label + "-handoff");
    (void)Enable(enabler);
    const auto disabled = Disable(disabler);
    if (disabled.disposition != CO::ProductionMutationDisposition::newOperation)
        throw std::runtime_error(label + " disable did not win first");
    bool rejected = false;
    try
    {
        (void)CO::DispatchOneRequestForProductionForTest(
            manager, DispatchRequest("h2-" + label + "-dispatch"));
    }
    catch (const pqxx::sql_error& error)
    {
        rejected = error.sqlstate() == "23514";
        std::cout << "H2_IMMEDIATE_FAIL case=" << label
                  << " sqlstate=" << error.sqlstate()
                  << " diagnostic=" << error.what() << '\n';
    }
    if (!rejected) throw std::runtime_error(label + " acquisition did not fail closed");
    auto observer = std::make_unique<pqxx::connection>(admin);
    const auto snapshot = Snapshot(*observer, label);
    PrintIdentities(*observer, label);
    const auto fields = Fields(snapshot);
    if (fields[0] != "ready" || fields[1] != "3" || fields[2] != "false" ||
        fields[3] != "true")
        throw std::runtime_error(label + " stale authorization/state result");
    RequireCounts(snapshot, label, {"2", "0", "0", "0", "0", "0", "0", "0", "0", "0"});
    std::cout << "H2_RACE_PASS case=" << label << " disable_won=PASS "
                 "immediate_fail_closed=PASS no_production_rows=PASS\n";
}

void RunHandoffFirst(const std::string& socket, const std::string& database)
{
    const auto admin = Conn(socket, database, "campaign_manager_login", "h2-observer-d1");
    const auto enabler = Conn(socket, database, "h2_enabler_login", "h2-d1-enabler");
    const auto disabler = Conn(socket, database, "h2_disabler_login", "h2-d1-disable");
    const auto manager = Conn(socket, database, "h2_manager_login", "h2-d1-handoff");
    (void)Enable(enabler);
    HoldPoint hold;
    AsyncResult<CO::DispatchServiceResult> handoff;
    auto handoffThread = Start(handoff, [&]
    {
        return CO::DispatchOneRequestForProductionForTest(
            manager, DispatchRequest("h2-d1-dispatch"), [&](CO::DispatchTestInjectionPoint point)
            { hold.Hold(point, CO::DispatchTestInjectionPoint::beforeHandoffCommit); });
    });
    hold.WaitUntilEntered("D1 handoff");
    AsyncResult<CO::ProductionMutationResult> disable;
    auto disableThread = Start(disable, [&] { return Disable(disabler); });
    (void)WaitForBlocking(admin, "h2-d1-disable", "h2-d1-handoff", "D1");
    hold.Release();
    handoffThread.join();
    disableThread.join();
    RequireNoError(handoff, "D1 handoff");
    RequireNoError(disable, "D1 disable");
    if (handoff.value->classification != CO::DispatchResultClassification::createdAndBound)
        throw std::runtime_error("D1 handoff did not win");
    auto observer = std::make_unique<pqxx::connection>(admin);
    const auto snapshot = Snapshot(*observer, "D1");
    PrintIdentities(*observer, "D1");
    const auto fields = Fields(snapshot);
    if (fields[0] != "bound" || fields[1] != "5" || fields[2] != "true" ||
        fields[3] != "true")
        throw std::runtime_error("D1 bound state changed by disable");
    RequireCounts(snapshot, "D1", {"2", "1", "1", "2", "1", "1", "1", "1", "1", "1"});
    std::cout << "H2_RACE_PASS case=D1 handoff_won_gate=PASS disable_blocked=PASS "
                 "bound_state_preserved=PASS exact_downstream_counts=PASS\n";
}

void RunDisableFirstAfterAcquisition(const std::string& socket,
    const std::string& database)
{
    const auto admin = Conn(socket, database, "campaign_manager_login",
        "h2-observer-d2");
    const auto enabler = Conn(socket, database, "h2_enabler_login",
        "h2-d2-enabler");
    const auto disabler = Conn(socket, database, "h2_disabler_login",
        "h2-d2-disable");
    const auto manager = Conn(socket, database, "h2_manager_login",
        "h2-d2-handoff");
    (void)Enable(enabler);
    HoldPoint handoffGate;
    AsyncResult<CO::DispatchServiceResult> handoff;
    auto handoffThread = Start(handoff, [&]
    {
        return CO::DispatchOneRequestForProductionForTest(
            manager, DispatchRequest("h2-d2-dispatch"),
            [&](CO::DispatchTestInjectionPoint point)
            {
                handoffGate.Hold(point,
                    CO::DispatchTestInjectionPoint::
                        beforeProductionHandoffGate);
            });
    });
    handoffGate.WaitUntilEntered("D2 actual handoff gate");

    pqxx::connection observer{admin};
    const auto preRace = Snapshot(observer, "D2-pre-race");
    const auto preRaceFields = Fields(preRace);
    if (preRaceFields[0] != "dispatching" || preRaceFields[1] != "4" ||
        preRaceFields[2] != "true" || preRaceFields[3] != "false" ||
        preRaceFields[4] != "1" || preRaceFields[5] != "1" ||
        preRaceFields[6] != "1" || preRaceFields[7] != "1")
        throw std::runtime_error("D2 pre-race predecessor is not durably acquired");
    pqxx::read_transaction preRaceTransaction{observer};
    const auto preRaceEvidence = preRaceTransaction.exec(
        "SELECT r.request_state||'|'||r.state_version||'|'||"
        "r.production_dispatch_enabled||'|'||r.lease_token_hash||'|'||"
        "r.lease_expires_at::text||'|'||a.operation_key||'|'||"
        "a.dispatch_attempt_id||'|'||ad.request_production_admission_id||'|'||"
        "(SELECT count(*) FROM campaign_operations_request_binding "
        " WHERE operational_request_id=71)||'|'||"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt_outcome o "
        " JOIN campaign_operations_dispatch_attempt x "
        " ON x.dispatch_attempt_id=o.dispatch_attempt_id "
        " WHERE x.operational_request_id=71) "
        "FROM campaign_operations_operational_request r "
        "JOIN campaign_operations_dispatch_attempt a "
        " ON a.operational_request_id=r.operational_request_id "
        "AND a.attempt_contract_version=2 "
        "JOIN campaign_operations_request_production_admission ad "
        " ON ad.operational_request_id=r.operational_request_id "
        "WHERE r.operational_request_id=71;").one_row()[0].as<std::string>();
    std::cout << "H2_D2_COMMITTED_ACQUISITION request_state_version_enabled_lease_key="
              << preRaceEvidence << '\n';
    const auto handoffPid = preRaceTransaction.exec(
        "SELECT pid FROM pg_stat_activity "
        "WHERE application_name='h2-d2-handoff' AND pid <> pg_backend_pid() "
        "ORDER BY pid DESC LIMIT 1;").one_row()[0].as<long>();
    preRaceTransaction.commit();
    std::cout << "H2_ACTOR_PID label=handoff pid=" << handoffPid << '\n';

    HoldPoint disableBeforeCommit;
    AsyncResult<CO::ProductionMutationResult> disable;
    auto disableThread = Start(disable, [&]
    {
        return Disable(disabler, [&](CO::ProductionMutationInjectionPoint point)
        {
            disableBeforeCommit.Hold(point,
                CO::ProductionMutationInjectionPoint::afterTransitionBeforeCommit);
        });
    });
    disableBeforeCommit.WaitUntilEntered("D2 disable exclusive gate");
    handoffGate.Release();
    const auto blocking = WaitForBlocking(admin, "h2-d2-handoff",
        "h2-d2-disable", "D2");
    if (blocking.waiterPid != handoffPid || blocking.blockerPid == handoffPid ||
        blocking.lockCatalog.find("advisory:ShareLock:false:19055:1") ==
            std::string::npos ||
        blocking.lockCatalog.find("advisory:ExclusiveLock:true:19055:1") ==
            std::string::npos)
        throw std::runtime_error("D2 advisory lock/PID evidence is not exact");
    std::cout << "H2_D2_DISABLE_FIRST_ORDERING handoff_pid=" << handoffPid
              << " disable_pid=" << blocking.blockerPid
              << " handoff_waited_on_disable=PASS advisory_key=(19055,1)"
                 " waiter=ShareLock blocker=ExclusiveLock\n";
    disableBeforeCommit.Release();
    disableThread.join();
    RequireNoError(disable, "D2 disable");
    if (disable.value->disposition !=
            CO::ProductionMutationDisposition::newOperation)
        throw std::runtime_error("D2 disable did not commit first");
    handoffThread.join();
    if (!handoff.error)
        throw std::runtime_error("D2 stale handoff unexpectedly succeeded");
    try
    {
        std::rethrow_exception(handoff.error);
    }
    catch (const std::exception& error)
    {
        if (std::string(error.what()).find("authority") == std::string::npos &&
            std::string(error.what()).find("enablement") == std::string::npos)
            throw std::runtime_error("D2 wrong rejection: " +
                std::string(error.what()));
        std::cout << "H2_D2_HANDOFF_REJECTED diagnostic=" << error.what() << '\n';
    }
    const auto final = Snapshot(observer, "D2");
    RequireCounts(final, "D2", {"2", "1", "1", "1", "0", "0", "0", "0", "0", "0"});
    const auto finalFields = Fields(final);
    if (finalFields[0] != "dispatching" || finalFields[1] != "4" ||
        finalFields[2] != "true" || finalFields[3] != "false")
        throw std::runtime_error("D2 stale handoff changed request authority");
    PrintIdentities(observer, "D2");
    std::cout << "H2_RACE_PASS case=D2 disable_won_after_acquisition=PASS "
                 "stale_handoff_rejected=PASS no_partial_downstream=PASS\n";
}

void RunConcurrentDispatch(const std::string& socket,
    const std::string& database, bool differentKey)
{
    const std::string label = differentKey ? "F" : "E";
    const auto admin = Conn(socket, database, "campaign_manager_login", "h2-observer-" + label);
    const auto enabler = Conn(socket, database, "h2_enabler_login", "h2-" + label + "-enabler");
    const auto first = Conn(socket, database, "h2_manager_login", "h2-" + label + "-first");
    const auto second = Conn(socket, database, "h2_manager_login", "h2-" + label + "-second");
    (void)Enable(enabler);
    const std::string firstKey = "h2-" + label + "-key-a";
    const std::string secondKey = differentKey ? "h2-" + label + "-key-b" : firstKey;
    HoldPoint hold;
    AsyncResult<CO::DispatchServiceResult> firstResult;
    auto firstThread = Start(firstResult, [&]
    {
        return CO::DispatchOneRequestForProductionForTest(
            first, DispatchRequest(firstKey), [&](CO::DispatchTestInjectionPoint point)
            { hold.Hold(point, CO::DispatchTestInjectionPoint::beforeAcquisitionCommit); });
    });
    hold.WaitUntilEntered(label + " first dispatch");
    AsyncResult<CO::DispatchServiceResult> secondResult;
    auto secondThread = Start(secondResult, [&]
    {
        return CO::DispatchOneRequestForProductionForTest(
            second, DispatchRequest(secondKey));
    });
    (void)WaitForBlocking(admin, "h2-" + label + "-second",
        "h2-" + label + "-first", label);
    hold.Release();
    firstThread.join();
    secondThread.join();
    RequireNoError(firstResult, label + " first dispatch");
    if (differentKey)
    {
        if (!secondResult.error) throw std::runtime_error("F loser did not conflict");
        try { std::rethrow_exception(secondResult.error); }
        catch (const std::exception& error)
        {
            if (std::string(error.what()).find("conflicting") == std::string::npos &&
                std::string(error.what()).find("operation") == std::string::npos)
                throw std::runtime_error("F loser diagnostic was not conflict: " +
                    std::string(error.what()));
            std::cout << "H2_CONFLICT case=F diagnostic=" << error.what() << '\n';
        }
    }
    else
    {
        RequireNoError(secondResult, "E second dispatch");
        if (firstResult.value->bindingSetIdentityHash !=
            secondResult.value->bindingSetIdentityHash)
            throw std::runtime_error("E successful results do not share exact binding identity");
    }
    auto observer = std::make_unique<pqxx::connection>(admin);
    const auto snapshot = Snapshot(*observer, label);
    PrintIdentities(*observer, label);
    RequireCounts(snapshot, label, {"1", "1", "1", "2", "1", "1", "1", "1", "1", "1"});
    const auto fields = Fields(snapshot);
    if (fields[15].find(firstKey) == std::string::npos ||
        fields[16].find("fnv1a64:") == std::string::npos)
        throw std::runtime_error(label + " exact winning identity missing");
    if (differentKey && fields[15].find(secondKey) != std::string::npos)
        throw std::runtime_error("F losing operation key became durable");
    std::cout << "H2_RACE_PASS case=" << label
              << (differentKey ? " winner_one_conflict_one=PASS" :
                  " same_key_exact_identity=PASS")
              << " no_duplicate_rows=PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 9)
    {
        std::cerr << "usage: CampaignOperationsPhaseH2ConcurrencyTests "
                     "SOCKET C1_DB C2_DB D1_DB D2_DB SAME_DB DIFFERENT_DB "
                     "ADMIN_USER\n";
        return 64;
    }
    try
    {
        const std::string socket = argv[1];
        std::cout << "H2_CONCURRENCY_SESSIONS independent=PASS advisory=(19055,1)\n";
        RunDisableAcquisition(socket, argv[2]);
        RunDisableFirst(socket, argv[3], "C2");
        RunHandoffFirst(socket, argv[4]);
        RunDisableFirstAfterAcquisition(socket, argv[5]);
        RunConcurrentDispatch(socket, argv[6], false);
        RunConcurrentDispatch(socket, argv[7], true);
        (void)argv[8];
        std::cout << "H2_FOUR_RACE_SUITE_OK C1=PASS C2=PASS D1=PASS D2=PASS "
                     "E_SAME_KEY=PASS F_DIFFERENT_KEY=PASS\n";
        return 0;
    }
    catch (const pqxx::sql_error& error)
    {
        std::cerr << "H2_FOUR_RACE_SUITE_FAIL sqlstate=" << error.sqlstate()
                  << " diagnostic=" << error.what() << " query=" << error.query() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "H2_FOUR_RACE_SUITE_FAIL diagnostic=" << error.what() << '\n';
        return 1;
    }
}
