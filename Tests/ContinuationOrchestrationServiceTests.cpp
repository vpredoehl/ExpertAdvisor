#include "SchedulerCore/ContinuationOrchestrationService.hpp"
#include "SchedulerCore/SchedulerAuthorityService.hpp"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using namespace EA::SchedulerCore;
using EA::ExperimentScheduler::ContinuationEvaluation;
using EA::ExperimentScheduler::ContinuationPolicyConfig;

ContinuationAutomationCandidate Candidate(
    long long experimentId,
    std::string decision,
    std::string reason,
    std::optional<int> rank,
    std::optional<double> leaderScore,
    std::optional<double> inferAccuracy,
    int epoch,
    long long modelId)
{
    ContinuationAutomationCandidate candidate;
    candidate.sourceExperimentId = experimentId;
    candidate.config.sourceExperimentId = experimentId;
    candidate.config.targetEpochs = 80;
    candidate.evaluation.decision = std::move(decision);
    candidate.evaluation.reason = std::move(reason);
    candidate.evaluation.rankValue = rank;
    candidate.evaluation.selected.leaderScore = leaderScore;
    candidate.evaluation.selected.inferAccuracy = inferAccuracy;
    candidate.evaluation.selected.completedEpoch = epoch;
    candidate.evaluation.selected.modelId = modelId;
    return candidate;
}

class MemoryPort final : public ContinuationOrchestrationPort
{
public:
    bool allowed = true;
    bool authorityCurrent = true;
    bool throwLoading = false;
    std::vector<long long> ids;
    std::unordered_map<long long, ContinuationAutomationPreflight> preflights;
    std::unordered_map<long long, ContinuationAutomationCandidate> evaluations;
    std::unordered_map<long long, ContinuationQueueResult> queueResults;
    std::vector<long long> evaluatedIds;
    std::vector<bool> persistValues;
    std::vector<long long> queuedIds;
    std::vector<long long> refreshedIds;

    bool automationAllowed() override { return allowed; }
    bool refreshAuthority() override { return authorityCurrent; }

    std::vector<long long> loadCandidateIds() override
    {
        if (throwLoading)
            throw std::runtime_error("candidate_load_failed");
        return ids;
    }

    ContinuationAutomationPreflight preflight(
        long long sourceExperimentId,
        bool) override
    {
        return preflights[sourceExperimentId];
    }

    ContinuationAutomationCandidate evaluate(
        long long sourceExperimentId,
        bool persist) override
    {
        evaluatedIds.push_back(sourceExperimentId);
        persistValues.push_back(persist);
        const auto found = evaluations.find(sourceExperimentId);
        if (found == evaluations.end())
            throw std::runtime_error("evaluation_failed");
        return found->second;
    }

    ContinuationQueueResult queue(long long sourceExperimentId) override
    {
        queuedIds.push_back(sourceExperimentId);
        return queueResults[sourceExperimentId];
    }

    void refreshQueuedIdentity(
        ContinuationAutomationCandidate& candidate) override
    {
        refreshedIds.push_back(candidate.sourceExperimentId);
        candidate.evaluation.queuedExperimentId =
            candidate.sourceExperimentId + 1000;
    }
};

} // namespace

int main()
{
    {
        MemoryPort port;
        port.allowed = false;
        std::ostringstream output;
        std::ostringstream errors;
        ContinuationOrchestrationService service{port, output, errors};
        const auto counts = service.runAutomaticScan({true, 45, 2, false});
        assert(counts.candidates == 0);
        assert(output.str() ==
               "CONTINUATION_AUTO_SCAN_SKIPPED,reason=global_experiment_control\n");
        assert(errors.str().empty());
    }

    {
        MemoryPort port;
        port.ids = {10, 20, 30, 40, 50, 55, 60};
        port.preflights[10] = {
            true,
            ",source_experiment_id=10,dry_run=0"};
        port.evaluations[20] = Candidate(
            20, "eligible", "policy_passed", 3, 0.8, 0.7, 40, 220);
        port.evaluations[30] = Candidate(
            30, "eligible", "policy_passed", 1, 0.6, 0.9, 20, 230);
        port.evaluations[40] = Candidate(
            40, "skipped", "invalid_configuration:target", {}, {}, {}, 0, -1);
        port.evaluations[50] = Candidate(
            50, "not_eligible", "minimum_score_not_met", {}, 0.2, {}, 20, 250);
        port.evaluations[55] = Candidate(
            55, "eligible", "policy_passed", 2, 0.7, 0.8, 30, 255);
        port.queueResults[30] = ContinuationQueueResult::Queued;
        port.queueResults[55] = ContinuationQueueResult::Failed;
        port.queueResults[20] = ContinuationQueueResult::AlreadyQueued;

        std::ostringstream output;
        std::ostringstream errors;
        ContinuationOrchestrationService service{port, output, errors};
        const auto counts = service.runAutomaticScan({true, 30, 3, false});
        assert(counts.candidates == 7);
        assert(counts.evaluated == 5);
        assert(counts.alreadySatisfied == 1);
        assert(counts.eligible == 3);
        assert(counts.queued == 1);
        assert(counts.errors == 3);
        assert((port.queuedIds == std::vector<long long>{30, 55, 20}));
        assert((port.refreshedIds == std::vector<long long>{30, 20}));
        assert(output.str().find("CONTINUATION_AUTO_ALREADY_SATISFIED") !=
               std::string::npos);
        assert(output.str().find("scan_errors=3,dry_run=0") !=
               std::string::npos);
        assert(errors.str().find(
                   "source_experiment_id=60,reason=evaluation_failed") !=
               std::string::npos);
    }

    {
        MemoryPort port;
        port.ids = {70, 80};
        port.evaluations[70] = Candidate(
            70, "eligible", "policy_passed", {}, 0.5, 0.8, 20, 270);
        port.evaluations[80] = Candidate(
            80, "eligible", "policy_passed", {}, 0.7, 0.7, 20, 280);
        std::ostringstream output;
        std::ostringstream errors;
        ContinuationOrchestrationService service{port, output, errors};
        const auto counts = service.runAutomaticScan({true, 60, 1, true});
        assert(counts.eligible == 2);
        assert(counts.queued == 0);
        assert(port.queuedIds.empty());
        assert((port.persistValues == std::vector<bool>{false, false}));
        assert(output.str().find(
                   "source_experiment_id=80") != std::string::npos);
        assert(output.str().find(
                   "CONTINUATION_AUTO_DRY_RUN") != std::string::npos);
    }

    {
        MemoryPort port;
        port.ids = {90};
        port.authorityCurrent = false;
        std::ostringstream output;
        std::ostringstream errors;
        ContinuationOrchestrationService service{port, output, errors};
        bool lost = false;
        try
        {
            (void)service.runAutomaticScan({});
        }
        catch (const SchedulerAuthorityLost& error)
        {
            lost = std::string{error.what()} ==
                   "continuation_scan_ownership_lost";
        }
        assert(lost);
    }

    {
        MemoryPort port;
        port.throwLoading = true;
        std::ostringstream output;
        std::ostringstream errors;
        ContinuationOrchestrationService service{port, output, errors};
        const auto counts = service.runAutomaticScan({});
        assert(counts.errors == 1);
        assert(errors.str().find("candidate_load_failed") != std::string::npos);
    }

    return 0;
}
