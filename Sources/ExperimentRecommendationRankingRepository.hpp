#pragma once

#include "ExperimentRecommendationRanking.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct RecommendationRankingSnapshotRequest
{
    RecommendationRankingPolicy policy;
    RecommendationRankingScope scope;
    int limit = 100;
    std::string snapshotIdentityCanonical;
    std::string snapshotIdentityHash;
    std::string membershipCanonical;
    std::string membershipHash;
};

struct RecommendationRankingSnapshotBeginResult
{
    long long snapshotId = -1;
    bool created = false;
    std::string status;
};

struct RecommendationRankingCounts
{
    int memberCount = 0;
    int advisoryReadyCount = 0;
    int blockedCount = 0;
    int nonActionableCount = 0;
};

struct PersistedRecommendationRankingSnapshot
{
    long long snapshotId = -1;
    std::string status;
    std::string snapshotIdentityCanonical;
    std::string snapshotIdentityHash;
    std::string rankingPolicyCanonical;
    std::string rankingPolicyHash;
    int rankingVersion = 0;
    RecommendationRankingScope scope;
    std::string scopeCanonical;
    std::string scopeHash;
    int requestedLimit = 0;
    std::string membershipCanonical;
    std::string membershipHash;
    RecommendationRankingCounts counts;
    std::string startedAt;
    std::optional<std::string> completedAt;
    std::optional<std::string> errorMessage;
};

struct PersistedRecommendationRankingMember
{
    long long memberId = -1;
    long long snapshotId = -1;
    RecommendationRankingMember member;
    std::string createdAt;
};

bool RecommendationRankingSchemaExists(pqxx::connection& connection);
std::vector<RecommendationRankingEvaluation> LoadEvaluationsForRanking(
    pqxx::connection& connection,
    const RecommendationRankingScope& scope);
RecommendationRankingSnapshotBeginResult BeginOrFindRecommendationRankingSnapshot(
    pqxx::connection& connection,
    const RecommendationRankingSnapshotRequest& request);
std::vector<PersistedRecommendationRankingMember> PersistRecommendationRankingMembers(
    pqxx::connection& connection,
    long long snapshotId,
    const std::vector<RecommendationRankingMember>& members);
void CompleteRecommendationRankingSnapshot(
    pqxx::connection& connection,
    long long snapshotId,
    const RecommendationRankingCounts& counts);
void FailRecommendationRankingSnapshot(
    pqxx::connection& connection,
    long long snapshotId,
    const std::string& errorMessage);
std::vector<PersistedRecommendationRankingSnapshot>
ListRecommendationRankingSnapshots(pqxx::connection& connection, int limit);
std::optional<PersistedRecommendationRankingSnapshot>
FindRecommendationRankingSnapshot(pqxx::connection& connection,
                                  long long snapshotId);
std::vector<PersistedRecommendationRankingMember>
ListRecommendationRankingMembers(
    pqxx::connection& connection,
    long long snapshotId,
    std::optional<RecommendationRankingBucket> bucket,
    int limit);
std::optional<PersistedRecommendationRankingMember>
FindRecommendationRankingMember(pqxx::connection& connection,
                                long long memberId);
std::optional<RecommendationRankingEvaluation> FindRankingEvaluation(
    pqxx::connection& connection,
    long long evaluationResultId);

} // namespace EA::ExperimentRecommendation
