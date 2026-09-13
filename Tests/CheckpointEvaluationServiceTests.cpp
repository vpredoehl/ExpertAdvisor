#include "SchedulerCore/CheckpointEvaluationService.hpp"

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace EA::ExperimentScheduler;
using namespace EA::SchedulerCore;

CheckpointEvaluationRecord Evaluation()
{
    return {88, 41, 700, 20, "EURUSD", 4};
}

CheckpointPolicyConfig Policy()
{
    CheckpointPolicyConfig config;
    config.enabled = true;
    config.checkpointInferEnabled = true;
    config.minLeaderScore = 0.75;
    config.scope = "symbol_horizon";
    config.stopMode = "next_checkpoint";
    config.graceEvals = 1;
    config.checkpointInterval = 20;
    config.targetEpochs = 100;
    config.currentEpoch = 20;
    config.status = "running";
    config.phase = "train";
    config.policyRevision = 3;
    config.persistedPolicyHash = CheckpointPolicySemanticHash(config);
    config.activeTrainingAttemptId = 901;
    return config;
}

ValidatedCheckpointPolicyEvidence Evidence()
{
    return {0.80, 0.70, 501, 601, "2025-01-01", "2025-02-01"};
}

class MemoryOperations
{
public:
    bool schema = true;
    std::optional<CheckpointPolicyConfig> policy = Policy();
    CheckpointPolicyEvidenceLoadResult evidence{Evidence(), {}};
    CheckpointPolicyPopulation completed{2, std::nullopt, "completed-watermark"};
    CheckpointPolicyPopulation rank{3, 1, "rank-watermark"};
    PersistedCheckpointPolicyDecision persisted{77, "active", false};
    std::string stopResult = "stop_request_applied";
    std::vector<std::string> calls;
    std::optional<CheckpointPolicyEvidenceIdentity> persistedIdentity;
    std::optional<CheckpointPolicyDecision> persistedDecision;

    CheckpointEvaluationOperations operations()
    {
        return {
            [this] {
                calls.push_back("schema");
                return schema;
            },
            [this](long long parentExperimentId) {
                assert(parentExperimentId == 41);
                calls.push_back("load_policy");
                return policy;
            },
            [this](const CheckpointEvaluationRecord& evaluation,
                   CheckpointPolicyConfig config) {
                assert(evaluation.checkpointEvalId == 88);
                calls.push_back("reconcile_policy_identity");
                return config;
            },
            [this](const CheckpointEvaluationRecord&) {
                calls.push_back("load_evidence");
                return evidence;
            },
            [this](long long parentExperimentId) {
                assert(parentExperimentId == 41);
                calls.push_back("load_completed_population");
                return completed;
            },
            [this](const CheckpointEvaluationRecord&,
                   const CheckpointPolicyConfig&) {
                calls.push_back("load_rank_population");
                return rank;
            },
            [this](const CheckpointEvaluationRecord&,
                   const CheckpointPolicyConfig&,
                   const CheckpointPolicyDecision& decision,
                   const ValidatedCheckpointPolicyEvidence&,
                   const CheckpointPolicyEvidenceIdentity& identity) {
                calls.push_back("persist_decision");
                persistedDecision = decision;
                persistedIdentity = identity;
                return persisted;
            },
            [this](const CheckpointEvaluationRecord&,
                   const CheckpointPolicyConfig&,
                   const CheckpointPolicyDecision&,
                   const PersistedCheckpointPolicyDecision& decision,
                   const std::string& evidenceWatermark) {
                assert(decision.decisionId == 77);
                assert(!evidenceWatermark.empty());
                calls.push_back("apply_stop_request");
                return stopResult;
            }};
    }
};

} // namespace

int main()
{
    {
        MemoryOperations port;
        port.schema = false;
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(!result.evaluated);
        assert(result.reason == "migration_required");
        assert((port.calls == std::vector<std::string>{"schema"}));
        assert(output.str() ==
               "CHECKPOINT_POLICY_SKIPPED,parent_experiment_id=41,"
               "checkpoint_eval_id=88,checkpoint_epoch=20,"
               "checkpoint_model_id=700,symbol=EURUSD,"
               "prediction_horizon=4,reason=migration_required\n");
    }

    {
        MemoryOperations port;
        port.policy->enabled = false;
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(!result.evaluated && result.reason == "disabled");
        assert((port.calls ==
                std::vector<std::string>{"schema", "load_policy"}));
    }

    {
        MemoryOperations port;
        port.policy->checkpointInferEnabled = false;
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(!result.evaluated &&
               result.reason == "checkpoint_infer_not_enabled");
        assert((port.calls == std::vector<std::string>{
            "schema", "load_policy", "reconcile_policy_identity"}));
    }

    {
        MemoryOperations port;
        port.evidence = {std::nullopt, "checkpoint_analysis_not_linked"};
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(!result.evaluated &&
               result.reason == "checkpoint_analysis_not_linked");
        assert(port.calls.back() == "load_evidence");
    }

    {
        MemoryOperations port;
        port.policy->minLeaderScore = 0.90;
        port.policy->graceEvals = 1;
        port.policy->persistedPolicyHash =
            CheckpointPolicySemanticHash(*port.policy);
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(result.evaluated);
        assert(result.decision == "stop_requested");
        assert(port.persistedDecision->requestedStopEpoch == 40);
        assert(port.persistedIdentity->analysisId == 501);
        assert(port.persistedIdentity->inferenceEvalResultId == 601);
        assert(port.persistedIdentity->completedEvalCount == 2);
        assert(port.persistedIdentity->completedPopulationWatermark ==
               "completed-watermark");
        assert(port.persistedIdentity->rankPopulationWatermark ==
               "not_decision_bearing");
        assert((port.calls == std::vector<std::string>{
            "schema",
            "load_policy",
            "reconcile_policy_identity",
            "load_evidence",
            "load_completed_population",
            "load_rank_population",
            "persist_decision",
            "apply_stop_request"}));
        assert(output.str().find("CHECKPOINT_POLICY_EVALUATING") !=
               std::string::npos);
        assert(output.str().find("CHECKPOINT_POLICY_STOP_REQUESTED") !=
               std::string::npos);
        assert(output.str().find("stop_action_result=stop_request_applied") !=
               std::string::npos);
    }

    {
        MemoryOperations port;
        port.policy->minLeaderScore.reset();
        port.policy->topN = 2;
        port.policy->persistedPolicyHash =
            CheckpointPolicySemanticHash(*port.policy);
        std::ostringstream output;
        CheckpointEvaluationService service{port.operations(), output};
        const auto result = service.evaluate(Evaluation());
        assert(result.evaluated && result.decision == "continue");
        assert(port.persistedIdentity->rankValue == 1);
        assert(port.persistedIdentity->rankScope == "symbol_horizon");
        assert(port.persistedIdentity->rankPopulationWatermark ==
               "rank-watermark");
    }

    {
        CheckpointPolicyConfig config = Policy();
        config.scope = "unsupported";
        assert(CheckpointPolicyConfigurationError(config, true) ==
               "invalid_scope");
        config.scope = "global";
        config.minLeaderScore.reset();
        assert(CheckpointPolicyConfigurationError(config, true) ==
               "at_least_one_continue_rule_required");
    }

    {
        MemoryOperations port;
        std::ostringstream output;
        bool rejected = false;
        try
        {
            CheckpointEvaluationOperations incomplete = port.operations();
            incomplete.persistDecision = {};
            CheckpointEvaluationService service{
                std::move(incomplete), output};
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                "complete checkpoint evaluation operations required";
        }
        assert(rejected);
    }

    return 0;
}
