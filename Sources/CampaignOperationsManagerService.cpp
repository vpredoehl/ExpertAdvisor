#include "CampaignOperationsManagerService.hpp"

#include "CampaignOperationsManager.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <pqxx/pqxx>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

bool Contains(const std::string& value, const char* needle)
{
    return value.find(needle) != std::string::npos;
}

ManagerGlobalStopReason GlobalReason(const std::exception& error)
{
    const std::string diagnostic = error.what();
    if (Contains(diagnostic, "scheduler_protocol_evidence") ||
        Contains(diagnostic, "scheduler_not_ready") ||
        Contains(diagnostic, "scheduler evidence incomplete"))
        return ManagerGlobalStopReason::schedulerProtocolIneffective;
    if (Contains(diagnostic, "enablement_ineffective") ||
        Contains(diagnostic, "production dispatch enablement is ineffective") ||
        Contains(diagnostic, "enablement_absent"))
        return ManagerGlobalStopReason::productionDisabled;
    if (Contains(diagnostic, "production_dispatch_build_mismatch") ||
        Contains(diagnostic, "manager_build_contract") ||
        Contains(diagnostic, "production_readiness_blocked"))
        return ManagerGlobalStopReason::managerBuildNotReady;
    if (Contains(diagnostic, "permission denied") ||
        Contains(diagnostic, "42501"))
        return ManagerGlobalStopReason::privilegeFailure;
    if (dynamic_cast<const pqxx::broken_connection*>(&error) != nullptr ||
        dynamic_cast<const pqxx::in_doubt_error*>(&error) != nullptr)
        return ManagerGlobalStopReason::databaseFailure;
    if (const auto* sqlError = dynamic_cast<const pqxx::sql_error*>(&error))
    {
        const auto& state = sqlError->sqlstate();
        if (state == "42501" || state.rfind("08", 0U) == 0U ||
            state.rfind("42", 0U) == 0U ||
            state.rfind("53", 0U) == 0U || state.rfind("57", 0U) == 0U ||
            state.rfind("58", 0U) == 0U)
            return state == "42501"
                ? ManagerGlobalStopReason::privilegeFailure
                : ManagerGlobalStopReason::databaseFailure;
    }
    if (const auto* campaignError = dynamic_cast<const Error*>(&error);
        campaignError &&
        (campaignError->code() == ErrorCode::persistenceCorruption ||
         campaignError->code() == ErrorCode::invalidCanonicalText ||
         campaignError->code() == ErrorCode::invalidOperationalRequest))
        return ManagerGlobalStopReason::databaseFailure;
    return ManagerGlobalStopReason::none;
}

void RenderRequest(std::ostream& output, const ManagerRequestResult& result)
{
    output << "CAMPAIGN_OPERATIONS_MANAGER_REQUEST"
           << ",request_id=" << result.requestId.value()
           << ",expected_request_version=" << result.expectedRequestVersion
           << ",operation_key=" << result.operationKey
           << ",outcome=" << ToText(result.outcome)
           << ",dispatch_classification="
           << (result.dispatchClassification.empty()
                   ? "none" : result.dispatchClassification)
           << ",replay_disposition="
           << (result.replayDisposition.empty()
                   ? "none" : result.replayDisposition)
           << ",newly_committed="
           << (result.newlyCommitted ? "true" : "false")
           << ",exact_replay=" << (result.exactReplay ? "true" : "false")
           << ",diagnostic_code="
           << (result.diagnosticCode.empty() ? "none" : result.diagnosticCode)
           << '\n';
}

} // namespace

std::string ToText(ManagerRequestOutcomeClassification value)
{
    switch (value)
    {
        case ManagerRequestOutcomeClassification::dispatchResult:
            return "dispatch_result";
        case ManagerRequestOutcomeClassification::requestLocalSemanticFailure:
            return "request_local_semantic_failure";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_manager_request_outcome_invalid");
}

std::string ToText(ManagerGlobalStopReason value)
{
    switch (value)
    {
        case ManagerGlobalStopReason::none: return "none";
        case ManagerGlobalStopReason::productionDisabled:
            return "production_disabled";
        case ManagerGlobalStopReason::schedulerProtocolIneffective:
            return "scheduler_protocol_ineffective";
        case ManagerGlobalStopReason::managerBuildNotReady:
            return "manager_build_not_ready";
        case ManagerGlobalStopReason::privilegeFailure:
            return "privilege_failure";
        case ManagerGlobalStopReason::databaseFailure:
            return "database_failure";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_manager_stop_reason_invalid");
}

namespace
{

using ManagerCandidateHook = std::function<void(
    int, int, OperationalRequestId)>;

ManagerRunOnceResult RunManagerOnceInternal(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString, int dispatchLimit,
    const std::string& executablePath,
    const std::optional<ManagerBuildContract>& fixtureBuild,
    const ManagerCandidateHook& hook)
{
    if (dispatchLimit <= 0 ||
        dispatchLimit > kCampaignOperationsManagerMaximumRunOnceLimit)
        throw std::invalid_argument(
            "campaign_operations_manager_run_once_limit_invalid");
    const auto actualBuild = fixtureBuild
        ? fixtureBuild
        : CaptureActualManagerBuildContract(executablePath);
    if (!actualBuild) throw std::runtime_error(
        "campaign_operations_manager_build_contract_unavailable");

    ManagerRunOnceResult result;
    result.dispatchLimit = dispatchLimit;
    {
        pqxx::connection connection{managerConnectionString};
        result.candidates = SelectDispatchCandidatesForManager(
            connection, dispatchLimit);
    }
    if (hook) hook(0, -1,
        result.candidates.empty()
            ? OperationalRequestId(1)
            : result.candidates.front().requestId);

    std::size_t index = 0;
    for (const auto& candidate : result.candidates)
    {
        ManagerRequestResult requestResult{
            candidate.requestId, candidate.expectedRequestVersion,
            {},
            ManagerRequestOutcomeClassification::dispatchResult,
            {}, {}, false, false, {}};
        try
        {
            if (hook) hook(1, static_cast<int>(index), candidate.requestId);
            const auto operation = BuildManagerRequestOperationIdentity(
                candidate.requestIdentityCanonical,
                candidate.expectedRequestVersion);
            requestResult.operationKey = operation.operationKey;
            const ProductionDispatchRequest request{
                candidate.requestId, candidate.expectedRequestVersion,
                operation.operationKey,
                ActorIdentity(kCampaignOperationsManagerActor), *actualBuild,
                true};
            const auto dispatched = fixtureBuild
#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
                ? DispatchOneRequestForProductionManagerWithFixture(
                    managerConnectionString, dispatchServiceConnectionString,
                    request,
                    operation.source.canonicalText(), *fixtureBuild)
#else
                ? throw std::logic_error("H3 fixture adapter unavailable")
#endif
                : DispatchOneRequestForProductionManager(
                    managerConnectionString, dispatchServiceConnectionString,
                    request, operation.source.canonicalText(), executablePath);
            if (dispatched.failure ==
                DispatchServiceFailureClassification::commitOutcomeUnknown)
            {
                result.stoppedEarly = true;
                result.stopReason = ManagerGlobalStopReason::databaseFailure;
                result.stopDiagnostic = dispatched.diagnosticCode;
                break;
            }
            requestResult.dispatchClassification =
                ToText(dispatched.classification);
            requestResult.replayDisposition =
                ToText(dispatched.replayDisposition);
            requestResult.newlyCommitted =
                dispatched.replayDisposition == ExactReplayDisposition::newOperation;
            requestResult.exactReplay =
                dispatched.replayDisposition ==
                    ExactReplayDisposition::authoritativeExisting &&
                dispatched.classification ==
                    DispatchResultClassification::existingIdentical;
            if (dispatched.failure ==
                DispatchServiceFailureClassification::transientDatabaseRetryExhausted)
                requestResult.outcome =
                    ManagerRequestOutcomeClassification::requestLocalSemanticFailure;
            requestResult.diagnosticCode = dispatched.diagnosticCode;
        }
        catch (const std::exception& error)
        {
            const auto globalReason = GlobalReason(error);
            if (globalReason != ManagerGlobalStopReason::none)
            {
                result.stoppedEarly = true;
                result.stopReason = globalReason;
                result.stopDiagnostic = error.what();
                break;
            }
            requestResult.outcome =
                ManagerRequestOutcomeClassification::requestLocalSemanticFailure;
            requestResult.diagnosticCode = error.what();
        }
        if (hook) hook(2, static_cast<int>(index), candidate.requestId);
        result.requests.push_back(std::move(requestResult));
        ++index;
    }
    return result;
}

} // namespace

ManagerRunOnceResult RunCampaignOperationsManagerOnce(
    const std::string& connectionString, int dispatchLimit,
    const std::string& executablePath)
{
    return RunManagerOnceInternal(connectionString, connectionString, dispatchLimit,
        executablePath, std::nullopt, {});
}

ManagerRunOnceResult RunCampaignOperationsManagerOnce(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString, int dispatchLimit,
    const std::string& executablePath)
{
    return RunManagerOnceInternal(managerConnectionString,
        dispatchServiceConnectionString, dispatchLimit, executablePath,
        std::nullopt, {});
}

#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
ManagerRunOnceResult RunCampaignOperationsManagerOnceForTest(
    const std::string& connectionString, int dispatchLimit,
    const ManagerBuildContract& fixtureBuild, const ManagerTestHook& testHook)
{
    const ManagerCandidateHook hook = [&](int point, int index,
        OperationalRequestId requestId)
    {
        if (!testHook) return;
        const auto injection = point == 0
            ? ManagerTestInjectionPoint::afterCandidateSnapshot
            : point == 1
                ? ManagerTestInjectionPoint::beforeCandidateProcessing
                : ManagerTestInjectionPoint::afterCandidateProcessing;
        testHook(injection, index, requestId);
    };
    return RunManagerOnceInternal(connectionString, connectionString,
        dispatchLimit, {},
        fixtureBuild, hook);
}

ManagerRunOnceResult RunCampaignOperationsManagerOnceForTest(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString, int dispatchLimit,
    const ManagerBuildContract& fixtureBuild, const ManagerTestHook& testHook)
{
    const ManagerCandidateHook hook = [&](int point, int index,
        OperationalRequestId requestId)
    {
        if (!testHook) return;
        const auto injection = point == 0
            ? ManagerTestInjectionPoint::afterCandidateSnapshot
            : point == 1
                ? ManagerTestInjectionPoint::beforeCandidateProcessing
                : ManagerTestInjectionPoint::afterCandidateProcessing;
        testHook(injection, index, requestId);
    };
    return RunManagerOnceInternal(managerConnectionString,
        dispatchServiceConnectionString, dispatchLimit, {}, fixtureBuild, hook);
}
#endif

int RunCampaignOperationsManagerOnceCommand(
    const std::string& connectionString, int dispatchLimit,
    const std::string& executablePath, std::ostream& output,
    std::ostream& errors)
{
    const auto result = RunCampaignOperationsManagerOnce(
        connectionString, dispatchLimit, executablePath);
    output << "CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE"
           << ",dispatch_limit=" << result.dispatchLimit
           << ",candidates_selected=" << result.candidates.size()
           << ",processed=" << result.requests.size()
           << ",stopped_early=" << (result.stoppedEarly ? "true" : "false")
           << ",stop_reason=" << ToText(result.stopReason)
           << ",candidate_request_ids=";
    for (std::size_t index = 0; index < result.candidates.size(); ++index)
    {
        if (index != 0U) output << ':';
        output << result.candidates[index].requestId.value();
    }
    if (result.candidates.empty()) output << "none";
    output << '\n';
    for (const auto& request : result.requests)
        RenderRequest(output, request);
    if (result.stoppedEarly)
    {
        errors << "CAMPAIGN_OPERATIONS_MANAGER_STOPPED"
               << ",reason=" << ToText(result.stopReason)
               << ",diagnostic=" << result.stopDiagnostic << '\n';
        return 2;
    }
    return 0;
}

int RunCampaignOperationsManagerOnceCommand(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString, int dispatchLimit,
    const std::string& executablePath, std::ostream& output,
    std::ostream& errors)
{
    const auto result = RunCampaignOperationsManagerOnce(
        managerConnectionString, dispatchServiceConnectionString,
        dispatchLimit, executablePath);
    output << "CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE"
           << ",dispatch_limit=" << result.dispatchLimit
           << ",candidates_selected=" << result.candidates.size()
           << ",processed=" << result.requests.size()
           << ",stopped_early=" << (result.stoppedEarly ? "true" : "false")
           << ",stop_reason=" << ToText(result.stopReason)
           << ",candidate_request_ids=";
    for (std::size_t index = 0; index < result.candidates.size(); ++index)
    {
        if (index != 0U) output << ':';
        output << result.candidates[index].requestId.value();
    }
    if (result.candidates.empty()) output << "none";
    output << '\n';
    for (const auto& request : result.requests)
        RenderRequest(output, request);
    if (result.stoppedEarly)
    {
        errors << "CAMPAIGN_OPERATIONS_MANAGER_STOPPED"
               << ",reason=" << ToText(result.stopReason)
               << ",diagnostic=" << result.stopDiagnostic << '\n';
        return 2;
    }
    return 0;
}

} // namespace EA::CampaignOperations
