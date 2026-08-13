#pragma once

#include "ExperimentRecommendationRankingRepository.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <utility>

namespace EA::ExperimentRecommendation
{

struct RecommendationRankingCommandRequest
{
    RecommendationRankingPolicy policy;
    RecommendationRankingScope scope;
    int limit = 100;
    bool dryRun = false;
};

int RunRankExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    const RecommendationRankingCommandRequest& request,
    std::ostream& output,
    std::ostream& errors);
int RunListExperimentRecommendationRankingSnapshotsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output);
int RunExperimentRecommendationRankingStatusCommand(
    const std::string& connectionString,
    long long snapshotId,
    std::ostream& output);
int RunListExperimentRecommendationRankingMembersCommand(
    const std::string& connectionString,
    long long snapshotId,
    std::optional<RecommendationRankingBucket> bucket,
    int limit,
    std::ostream& output);
int RunExperimentRecommendationRankingMemberStatusCommand(
    const std::string& connectionString,
    long long memberId,
    std::ostream& output);
int RunCompareExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    std::pair<long long, long long> evaluationIds,
    std::ostream& output);
int RunCompareExperimentRecommendationRankingMembersCommand(
    const std::string& connectionString,
    std::pair<long long, long long> memberIds,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
