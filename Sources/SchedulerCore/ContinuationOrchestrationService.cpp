#include "ContinuationOrchestrationService.hpp"

#include "SchedulerAuthorityService.hpp"

#include <algorithm>
#include <exception>
#include <ostream>
#include <utility>

namespace EA::SchedulerCore
{
namespace
{

using EA::ExperimentScheduler::ContinuationOptionalDoubleText;
using EA::ExperimentScheduler::ContinuationOptionalIntText;
using EA::ExperimentScheduler::ContinuationProfitabilityEvidenceLogFields;
using EA::ExperimentScheduler::ContinuationProfitabilityPolicyLogFields;

std::string OptionalIdentity(long long value)
{
    return value >= 0 ? std::to_string(value) : "NULL";
}

bool EvaluationIsError(
    const EA::ExperimentScheduler::ContinuationEvaluation& evaluation)
{
    return evaluation.reason == "migration_required" ||
           evaluation.reason == "source_experiment_not_found" ||
           evaluation.reason.rfind("invalid_configuration:", 0) == 0;
}

} // namespace

bool BetterContinuationAutomationCandidate(
    const ContinuationAutomationCandidate& lhs,
    const ContinuationAutomationCandidate& rhs)
{
    if (lhs.evaluation.rankValue.has_value() !=
        rhs.evaluation.rankValue.has_value())
    {
        return lhs.evaluation.rankValue.has_value();
    }
    if (lhs.evaluation.rankValue.has_value() &&
        *lhs.evaluation.rankValue != *rhs.evaluation.rankValue)
    {
        return *lhs.evaluation.rankValue < *rhs.evaluation.rankValue;
    }
    if (lhs.evaluation.selected.leaderScore.has_value() !=
        rhs.evaluation.selected.leaderScore.has_value())
    {
        return lhs.evaluation.selected.leaderScore.has_value();
    }
    if (lhs.evaluation.selected.leaderScore.has_value() &&
        *lhs.evaluation.selected.leaderScore !=
            *rhs.evaluation.selected.leaderScore)
    {
        return *lhs.evaluation.selected.leaderScore >
               *rhs.evaluation.selected.leaderScore;
    }
    if (lhs.evaluation.selected.inferAccuracy.has_value() !=
        rhs.evaluation.selected.inferAccuracy.has_value())
    {
        return lhs.evaluation.selected.inferAccuracy.has_value();
    }
    if (lhs.evaluation.selected.inferAccuracy.has_value() &&
        *lhs.evaluation.selected.inferAccuracy !=
            *rhs.evaluation.selected.inferAccuracy)
    {
        return *lhs.evaluation.selected.inferAccuracy >
               *rhs.evaluation.selected.inferAccuracy;
    }
    if (lhs.evaluation.selected.completedEpoch !=
        rhs.evaluation.selected.completedEpoch)
    {
        return lhs.evaluation.selected.completedEpoch >
               rhs.evaluation.selected.completedEpoch;
    }
    if (lhs.sourceExperimentId != rhs.sourceExperimentId)
        return lhs.sourceExperimentId < rhs.sourceExperimentId;
    return lhs.evaluation.selected.modelId < rhs.evaluation.selected.modelId;
}

ContinuationOrchestrationService::ContinuationOrchestrationService(
    ContinuationOrchestrationPort& port,
    std::ostream& output,
    std::ostream& errors)
    : port_(port), output_(output), errors_(errors)
{
}

void ContinuationOrchestrationService::printEvaluation(
    const std::string& marker,
    const ContinuationAutomationCandidate& candidate,
    bool dryRun,
    const std::optional<std::string>& action)
{
    const auto& config = candidate.config;
    const auto& evaluation = candidate.evaluation;
    output_ << marker
            << ",source_experiment_id=" << config.sourceExperimentId
            << ",source_model_id=" << OptionalIdentity(evaluation.selected.modelId)
            << ",source_epoch="
            << (evaluation.selected.completedEpoch > 0
                    ? std::to_string(evaluation.selected.completedEpoch)
                    : "NULL")
            << ",target_epochs=" << ContinuationOptionalIntText(config.targetEpochs)
            << ",decision_id=" << OptionalIdentity(evaluation.decisionId)
            << ",decision=" << evaluation.decision
            << ",reason=" << evaluation.reason
            << ",rank=" << ContinuationOptionalIntText(evaluation.rankValue)
            << ",leader_score="
            << ContinuationOptionalDoubleText(evaluation.selected.leaderScore)
            << ",infer_accuracy="
            << ContinuationOptionalDoubleText(evaluation.selected.inferAccuracy)
            << ",evidence_count=" << evaluation.evidenceCount
            << ",policy_revision=" << config.policyRevision
            << ",policy_hash="
            << (evaluation.policyHash.empty() ? "NULL" : evaluation.policyHash)
            << ",current_policy_hash="
            << (evaluation.currentPolicyHash.empty()
                    ? "NULL" : evaluation.currentPolicyHash)
            << ",persisted_decision_policy_hash="
            << (evaluation.persistedDecisionPolicyHash.empty()
                    ? "NULL" : evaluation.persistedDecisionPolicyHash)
            << ",evidence_watermark="
            << (evaluation.evidenceWatermark.empty()
                    ? "NULL" : evaluation.evidenceWatermark)
            << ",queued_experiment_id="
            << (evaluation.queuedExperimentId.has_value()
                    ? std::to_string(*evaluation.queuedExperimentId)
                    : "NULL")
            << ContinuationProfitabilityEvidenceLogFields(evaluation.selected)
            << ContinuationProfitabilityPolicyLogFields(
                   config,
                   evaluation.profitabilityGate)
            << ",dry_run=" << (dryRun ? "1" : "0");
    if (action.has_value())
        output_ << ",action=" << *action;
    output_ << std::endl;
}

ContinuationAutomationCounts
ContinuationOrchestrationService::runAutomaticScan(
    const ContinuationAutomationOptions& options)
{
    ContinuationAutomationCounts counts;
    if (!port_.automationAllowed())
    {
        output_ << "CONTINUATION_AUTO_SCAN_SKIPPED"
                << ",reason=global_experiment_control"
                << std::endl;
        return counts;
    }

    output_ << "CONTINUATION_AUTO_SCAN_STARTED"
            << ",auto_evaluate=1"
            << ",auto_queue=" << (options.autoQueue ? "1" : "0")
            << ",scan_seconds=" << options.scanSeconds
            << ",max_queues_per_scan=" << options.maxQueuesPerScan
            << ",dry_run=" << (options.dryRun ? "1" : "0")
            << std::endl;

    std::vector<long long> candidateIds;
    try
    {
        candidateIds = port_.loadCandidateIds();
    }
    catch (const std::exception& error)
    {
        ++counts.errors;
        errors_ << "CONTINUATION_AUTO_ERROR"
                << ",source_experiment_id=NULL"
                << ",reason=" << error.what()
                << ",dry_run=" << (options.dryRun ? "1" : "0")
                << std::endl;
    }
    counts.candidates = static_cast<int>(candidateIds.size());

    std::vector<ContinuationAutomationCandidate> eligible;
    eligible.reserve(candidateIds.size());
    for (const long long sourceExperimentId : candidateIds)
    {
        if (!port_.refreshAuthority())
            throw SchedulerAuthorityLost("continuation_scan_ownership_lost");
        output_ << "CONTINUATION_AUTO_SCAN_CANDIDATE"
                << ",source_experiment_id=" << sourceExperimentId
                << ",dry_run=" << (options.dryRun ? "1" : "0")
                << std::endl;
        try
        {
            const ContinuationAutomationPreflight preflight =
                port_.preflight(sourceExperimentId, options.dryRun);
            if (preflight.alreadySatisfied)
            {
                ++counts.alreadySatisfied;
                output_ << "CONTINUATION_AUTO_ALREADY_SATISFIED"
                        << preflight.satisfiedDiagnosticFields
                        << std::endl;
                continue;
            }

            ContinuationAutomationCandidate candidate =
                port_.evaluate(sourceExperimentId, !options.dryRun);
            ++counts.evaluated;
            printEvaluation(
                "CONTINUATION_AUTO_EVALUATED",
                candidate,
                options.dryRun);
            if (options.dryRun)
            {
                printEvaluation(
                    "CONTINUATION_AUTO_DRY_RUN",
                    candidate,
                    true,
                    "evaluate");
            }

            if (candidate.evaluation.decision == "eligible" &&
                !candidate.evaluation.alreadyQueued &&
                !candidate.evaluation.queuedExperimentId.has_value())
            {
                ++counts.eligible;
                printEvaluation(
                    "CONTINUATION_AUTO_ELIGIBLE",
                    candidate,
                    options.dryRun);
                eligible.push_back(std::move(candidate));
            }
            else if (EvaluationIsError(candidate.evaluation))
            {
                ++counts.errors;
                printEvaluation(
                    "CONTINUATION_AUTO_ERROR",
                    candidate,
                    options.dryRun);
            }
            else
            {
                printEvaluation(
                    "CONTINUATION_AUTO_SKIPPED",
                    candidate,
                    options.dryRun);
            }
        }
        catch (const SchedulerAuthorityLost&)
        {
            throw;
        }
        catch (const std::exception& error)
        {
            ++counts.errors;
            errors_ << "CONTINUATION_AUTO_ERROR"
                    << ",source_experiment_id=" << sourceExperimentId
                    << ",reason=" << error.what()
                    << ",dry_run=" << (options.dryRun ? "1" : "0")
                    << std::endl;
        }
    }

    std::sort(
        eligible.begin(),
        eligible.end(),
        BetterContinuationAutomationCandidate);
    if (options.autoQueue)
    {
        int selectedCount = 0;
        for (ContinuationAutomationCandidate& candidate : eligible)
        {
            if (selectedCount >= options.maxQueuesPerScan)
                break;
            ++selectedCount;
            printEvaluation(
                "CONTINUATION_AUTO_QUEUE_SELECTED",
                candidate,
                options.dryRun);
            if (options.dryRun)
            {
                printEvaluation(
                    "CONTINUATION_AUTO_DRY_RUN",
                    candidate,
                    true,
                    "queue");
                continue;
            }

            const ContinuationQueueResult queueResult =
                port_.queue(candidate.sourceExperimentId);
            if (queueResult == ContinuationQueueResult::Queued)
            {
                ++counts.queued;
                candidate.evaluation.decision = "continuation_queued";
                candidate.evaluation.reason =
                    "continuation_experiment_created";
                port_.refreshQueuedIdentity(candidate);
                printEvaluation(
                    "CONTINUATION_AUTO_QUEUED",
                    candidate,
                    false);
            }
            else if (queueResult == ContinuationQueueResult::AlreadyQueued)
            {
                candidate.evaluation.decision = "already_continued";
                candidate.evaluation.reason = "continuation_already_queued";
                port_.refreshQueuedIdentity(candidate);
                printEvaluation(
                    "CONTINUATION_AUTO_SKIPPED",
                    candidate,
                    false);
            }
            else
            {
                ++counts.errors;
                candidate.evaluation.reason = "automatic_queue_failed";
                printEvaluation(
                    "CONTINUATION_AUTO_ERROR",
                    candidate,
                    false);
            }
        }
    }

    output_ << "CONTINUATION_AUTO_SCAN_COMPLETED"
            << ",scan_candidates=" << counts.candidates
            << ",scan_evaluated=" << counts.evaluated
            << ",scan_already_satisfied=" << counts.alreadySatisfied
            << ",scan_eligible=" << counts.eligible
            << ",scan_queued=" << counts.queued
            << ",scan_errors=" << counts.errors
            << ",dry_run=" << (options.dryRun ? "1" : "0")
            << std::endl;
    return counts;
}

} // namespace EA::SchedulerCore
