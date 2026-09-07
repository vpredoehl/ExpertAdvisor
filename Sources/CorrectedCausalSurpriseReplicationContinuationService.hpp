#pragma once

#include "CorrectedCausalSurpriseReplicationContinuation.hpp"
#include "FeatureAblationPairEvaluation.hpp"
#include "FeatureAblationReplicationEvaluationService.hpp"

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <pqxx/pqxx>

namespace EA::CorrectedCausalSurpriseReplicationContinuation
{

struct Command
{
    std::vector<std::pair<long long, long long>> evidencePairs;
    std::pair<long long, long long> anchorPair;
};

struct Assessment
{
    Replication::ReplicationEvaluation replication;
    FeatureAblationPairEvaluation::ArmEvidence anchorControl;
    FeatureAblationPairEvaluation::ArmEvidence anchorTreatment;
    FeatureAblationPairEvaluation::ComparisonResult anchorComparison;
    Plan plan;
    Gate gate;
};

Assessment EvaluateCommand(pqxx::transaction_base& transaction,
                           const Command& command);
std::string RenderAssessment(const Assessment& assessment);

int RunStatusCommand(const std::string& connectionString,
                     const Command& command,
                     std::ostream& output,
                     std::ostream& errors);

} // namespace EA::CorrectedCausalSurpriseReplicationContinuation
