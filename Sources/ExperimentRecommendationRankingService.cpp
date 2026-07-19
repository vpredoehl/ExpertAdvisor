#include "ExperimentRecommendationRankingService.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationService.hpp"

#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalScore(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "NULL";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

void PrintSafety(std::ostream& output)
{
    output << "experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false";
}

RecommendationRankingCounts CountMembers(
    const std::vector<RecommendationRankingMember>& members)
{
    RecommendationRankingCounts counts;
    counts.memberCount = static_cast<int>(members.size());
    for (const auto& member : members)
    {
        switch (member.bucket)
        {
            case RecommendationRankingBucket::advisoryReady:
                ++counts.advisoryReadyCount; break;
            case RecommendationRankingBucket::blocked:
                ++counts.blockedCount; break;
            case RecommendationRankingBucket::nonActionable:
                ++counts.nonActionableCount; break;
        }
    }
    return counts;
}

void PrintSnapshot(std::ostream& output,
                   const PersistedRecommendationRankingSnapshot& snapshot)
{
    output << "EXPERIMENT_RECOMMENDATION_RANKING_SNAPSHOT"
           << ",ranking_snapshot_id=" << snapshot.snapshotId
           << ",status=" << RecommendationMachineText(snapshot.status)
           << ",ranking_snapshot_identity_hash="
           << RecommendationMachineText(snapshot.snapshotIdentityHash)
           << ",ranking_policy_version=" << snapshot.rankingVersion
           << ",ranking_policy_hash="
           << RecommendationMachineText(snapshot.rankingPolicyHash)
           << ",scope_type="
           << RecommendationRankingScopeTypeText(snapshot.scope.type)
           << ",scope_value="
           << RecommendationMachineText(
                  RecommendationRankingScopeValueText(snapshot.scope))
           << ",source_evaluation_run_id="
           << OptionalNumber(snapshot.scope.evaluationRunId)
           << ",member_count=" << snapshot.counts.memberCount
           << ",advisory_ready_count=" << snapshot.counts.advisoryReadyCount
           << ",blocked_count=" << snapshot.counts.blockedCount
           << ",non_actionable_count=" << snapshot.counts.nonActionableCount
           << ",persisted=true,";
    PrintSafety(output);
    output << '\n';
}

void PrintMember(std::ostream& output,
                 const RecommendationRankingMember& member,
                 std::optional<long long> snapshotId,
                 std::optional<long long> memberId,
                 bool persisted)
{
    const auto& value = member.evaluation;
    output << "EXPERIMENT_RECOMMENDATION_RANKING_MEMBER"
           << ",ranking_snapshot_id=" << OptionalNumber(snapshotId)
           << ",ranking_member_id=" << OptionalNumber(memberId)
           << ",evaluation_result_id=" << value.evaluationResultId
           << ",recommendation_id=" << value.recommendationId
           << ",recommendation_semantic_hash="
           << RecommendationMachineText(value.recommendationSemanticHash)
           << ",evaluation_identity_hash="
           << RecommendationMachineText(value.evaluationIdentityHash)
           << ",evaluation_policy_hash="
           << RecommendationMachineText(value.evaluationPolicyHash)
           << ",evaluation_policy_version=" << value.evaluationVersion
           << ",evaluator_version=" << value.evaluatorVersion
           << ",scoring_policy_hash="
           << RecommendationMachineText(value.scoringPolicyHash)
           << ",scoring_policy_version=" << value.scoringVersion
           << ",bucket=" << RecommendationRankingBucketText(member.bucket)
           << ",bucket_rank=" << member.bucketRank
           << ",global_ordinal=" << member.globalOrdinal
           << ",final_score=" << OptionalScore(value.finalScore)
           << ",component_count=" << value.componentCount
           << ",disposition="
           << RecommendationEvaluationDispositionText(value.disposition)
           << ",tie_break_1="
           << RecommendationMachineText(member.tieBreakPrimary)
           << ",tie_break_2="
           << RecommendationMachineText(member.tieBreakSemanticHash)
           << ",tie_break_3="
           << RecommendationMachineText(member.tieBreakEvaluationHash)
           << ",inclusion_reason="
           << RecommendationMachineText(member.inclusionReason)
           << ",block_reason=" << OptionalText(member.blockReason)
           << ",top_positive_component="
           << OptionalText(member.topPositiveComponent)
           << ",top_penalty_component="
           << OptionalText(member.topPenaltyComponent)
           << ",symbol=" << RecommendationMachineText(value.symbol)
           << ",horizon=" << value.horizon
           << ",family=" << RecommendationMachineText(value.family)
           << ",source_value="
           << RecommendationMachineText(value.sourceValueCanonical)
           << ",proposed_value="
           << RecommendationMachineText(value.proposedValueCanonical)
           << ",persisted=" << (persisted ? "true" : "false") << ',';
    PrintSafety(output);
    output << '\n';
}

void PrintHumanRanking(std::ostream& output,
                       const RecommendationRankingScope& scope,
                       const std::vector<RecommendationRankingMember>& members,
                       std::optional<long long> snapshotId)
{
    output << "\nAdvisory recommendation ranking "
           << (snapshotId ? std::to_string(*snapshotId) : "(dry run)")
           << "\nScope: "
           << RecommendationHumanText(
                  RecommendationRankingScopeTypeText(scope.type) + "=" +
                  RecommendationRankingScopeValueText(scope)) << "\n";
    const RecommendationRankingBucket buckets[] = {
        RecommendationRankingBucket::advisoryReady,
        RecommendationRankingBucket::blocked,
        RecommendationRankingBucket::nonActionable};
    for (const RecommendationRankingBucket bucket : buckets)
    {
        output << "\n" << RecommendationHumanText(
            RecommendationRankingBucketText(bucket)) << ":\n";
        bool any = false;
        for (const auto& member : members)
        {
            if (member.bucket != bucket) continue;
            any = true;
            const auto& value = member.evaluation;
            output << "  " << member.bucketRank << ". recommendation "
                   << value.recommendationId << " | "
                   << RecommendationHumanText(value.symbol) << " | horizon "
                   << value.horizon << " | "
                   << RecommendationHumanText(value.family) << " | "
                   << RecommendationHumanText(value.sourceValueCanonical)
                   << " -> "
                   << RecommendationHumanText(value.proposedValueCanonical)
                   << " | score " << OptionalScore(value.finalScore)
                   << " | "
                   << RecommendationEvaluationDispositionText(value.disposition)
                   << " | positive "
                   << (member.topPositiveComponent
                           ? RecommendationHumanText(*member.topPositiveComponent)
                           : "NULL")
                   << " | penalty "
                   << (member.topPenaltyComponent
                           ? RecommendationHumanText(*member.topPenaltyComponent)
                           : "NULL")
                   << " | " << RecommendationHumanText(value.explanation)
                   << "\n";
        }
        if (!any) output << "  none\n";
    }
    output << "\nAdvisory ranking only. No experiment was created or queued; "
              "scheduler state was not changed.\n";
}

void PrintComparison(std::ostream& output,
                     const RecommendationComparisonResult& result)
{
    output << "EXPERIMENT_RECOMMENDATION_RANKING_COMPARISON"
           << ",left_evaluation_result_id=" << result.leftEvaluationResultId
           << ",right_evaluation_result_id=" << result.rightEvaluationResultId
           << ",left_recommendation_semantic_hash="
           << RecommendationMachineText(result.leftRecommendationSemanticHash)
           << ",right_recommendation_semantic_hash="
           << RecommendationMachineText(result.rightRecommendationSemanticHash)
           << ",left_evaluation_policy_hash="
           << RecommendationMachineText(result.leftEvaluationPolicyHash)
           << ",right_evaluation_policy_hash="
           << RecommendationMachineText(result.rightEvaluationPolicyHash)
           << ",left_scoring_policy_hash="
           << RecommendationMachineText(result.leftScoringPolicyHash)
           << ",right_scoring_policy_hash="
           << RecommendationMachineText(result.rightScoringPolicyHash)
           << ",left_scoring_policy_version=" << result.leftScoringVersion
           << ",right_scoring_policy_version=" << result.rightScoringVersion
           << ",left_evaluator_version=" << result.leftEvaluatorVersion
           << ",right_evaluator_version=" << result.rightEvaluatorVersion
           << ",comparability="
           << RecommendationComparisonStateText(result.state)
           << ",left_score=" << OptionalScore(result.leftScore)
           << ",right_score=" << OptionalScore(result.rightScore)
           << ",score_delta=" << OptionalScore(result.scoreDelta)
           << ",left_rank=" << OptionalNumber(result.leftRank)
           << ",right_rank=" << OptionalNumber(result.rightRank)
           << ",rank_delta=" << OptionalNumber(result.rankDelta)
           << ",largest_positive_difference="
           << OptionalText(result.largestPositiveDifference)
           << ",largest_penalty_difference="
           << OptionalText(result.largestPenaltyDifference)
           << ",explanation="
           << RecommendationMachineText(result.explanation) << ',';
    PrintSafety(output);
    output << '\n';
    for (const auto& difference : result.componentDifferences)
        output << "EXPERIMENT_RECOMMENDATION_RANKING_COMPONENT_DIFFERENCE"
               << ",left_evaluation_result_id=" << result.leftEvaluationResultId
               << ",right_evaluation_result_id=" << result.rightEvaluationResultId
               << ",component_name="
               << RecommendationMachineText(difference.componentName)
               << ",left_contribution="
               << OptionalScore(difference.leftContribution)
               << ",right_contribution="
               << OptionalScore(difference.rightContribution)
               << ",contribution_delta="
               << OptionalScore(difference.contributionDelta)
               << ",is_penalty=" << (difference.penalty ? "true" : "false")
               << '\n';
    output << "\nComparison is advisory only. No experiment was created or queued; "
              "scheduler state was not changed.\n";
}

} // namespace

int RunRankExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    const RecommendationRankingCommandRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    if (const auto error = ValidateRecommendationRankingPolicy(request.policy))
        throw std::invalid_argument(*error);
    if (const auto error = ValidateRecommendationRankingScope(request.scope))
        throw std::invalid_argument(*error);
    if (request.limit <= 0 || request.limit > kMaximumRecommendationRankingMembers)
        throw std::invalid_argument("recommendation_ranking_limit_invalid");
    pqxx::connection connection{connectionString};
    if (!request.dryRun && !RecommendationRankingSchemaExists(connection))
        throw std::runtime_error("recommendation_ranking_schema_unavailable");
    const auto evaluations = LoadEvaluationsForRanking(connection, request.scope);
    const auto members = RankRecommendationEvaluationEvidence(
        request.policy, evaluations, request.limit);
    const std::string membership =
        RecommendationRankingMembershipCanonicalText(evaluations);
    const std::string identity = RecommendationRankingSnapshotIdentityCanonicalText(
        request.policy, request.scope, request.limit, evaluations);
    const std::string identityHash = RecommendationRankingCanonicalHash(identity);
    const RecommendationRankingCounts counts = CountMembers(members);
    output << "EXPERIMENT_RECOMMENDATION_RANKING_START"
           << ",ranking_snapshot_id=NULL"
           << ",ranking_snapshot_identity_hash="
           << RecommendationMachineText(identityHash)
           << ",ranking_policy_version=" << request.policy.rankingVersion
           << ",ranking_policy_hash="
           << RecommendationRankingCanonicalHash(
                  RecommendationRankingPolicyCanonicalText(request.policy))
           << ",scope_type="
           << RecommendationRankingScopeTypeText(request.scope.type)
           << ",scope_value="
           << RecommendationMachineText(
                  RecommendationRankingScopeValueText(request.scope))
           << ",source_evaluation_run_id="
           << OptionalNumber(request.scope.evaluationRunId)
           << ",member_count=" << counts.memberCount
           << ",advisory_ready_count=" << counts.advisoryReadyCount
           << ",blocked_count=" << counts.blockedCount
           << ",non_actionable_count=" << counts.nonActionableCount
           << ",persisted=false,";
    PrintSafety(output);
    output << '\n';
    if (request.dryRun)
    {
        for (const auto& member : members)
            PrintMember(output, member, std::nullopt, std::nullopt, false);
        PrintHumanRanking(output, request.scope, members, std::nullopt);
        output << "EXPERIMENT_RECOMMENDATION_RANKING_COMPLETE"
               << ",ranking_snapshot_id=NULL,member_count=" << members.size()
               << ",persisted=false,";
        PrintSafety(output);
        output << '\n';
        return 0;
    }

    RecommendationRankingSnapshotBeginResult snapshot;
    try
    {
        snapshot = BeginOrFindRecommendationRankingSnapshot(connection, {
            request.policy, request.scope, request.limit, identity, identityHash,
            membership, RecommendationRankingCanonicalHash(membership)});
        if (snapshot.status == "failed")
            throw std::runtime_error("recommendation_ranking_snapshot_failed");
        const auto persisted = PersistRecommendationRankingMembers(
            connection, snapshot.snapshotId, members);
        CompleteRecommendationRankingSnapshot(
            connection, snapshot.snapshotId, counts);
        for (const auto& member : persisted)
            PrintMember(output, member.member, member.snapshotId,
                        member.memberId, true);
        PrintHumanRanking(output, request.scope, members, snapshot.snapshotId);
        output << "EXPERIMENT_RECOMMENDATION_RANKING_COMPLETE"
               << ",ranking_snapshot_id=" << snapshot.snapshotId
               << ",member_count=" << members.size()
               << ",persisted=true,";
        PrintSafety(output);
        output << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        if (snapshot.created && snapshot.snapshotId > 0)
        {
            try { FailRecommendationRankingSnapshot(
                connection, snapshot.snapshotId, error.what()); }
            catch (...) {}
        }
        errors << "EXPERIMENT_RECOMMENDATION_RANKING_FAILED"
               << ",ranking_snapshot_id="
               << (snapshot.snapshotId > 0 ? std::to_string(snapshot.snapshotId)
                                           : "NULL")
               << ",reason=" << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

int RunListExperimentRecommendationRankingSnapshotsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto snapshots = ListRecommendationRankingSnapshots(connection, limit);
    for (const auto& snapshot : snapshots) PrintSnapshot(output, snapshot);
    output << "EXPERIMENT_RECOMMENDATION_RANKING_SNAPSHOT_LIST_COMPLETE,count="
           << snapshots.size() << '\n';
    return 0;
}

int RunExperimentRecommendationRankingStatusCommand(
    const std::string& connectionString,
    long long snapshotId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto snapshot = FindRecommendationRankingSnapshot(connection, snapshotId);
    if (!snapshot) return 3;
    PrintSnapshot(output, *snapshot);
    return 0;
}

int RunListExperimentRecommendationRankingMembersCommand(
    const std::string& connectionString,
    long long snapshotId,
    std::optional<RecommendationRankingBucket> bucket,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    if (!FindRecommendationRankingSnapshot(connection, snapshotId)) return 3;
    const auto members = ListRecommendationRankingMembers(
        connection, snapshotId, bucket, limit);
    for (const auto& member : members)
        PrintMember(output, member.member, member.snapshotId, member.memberId, true);
    output << "EXPERIMENT_RECOMMENDATION_RANKING_MEMBER_LIST_COMPLETE"
           << ",ranking_snapshot_id=" << snapshotId
           << ",count=" << members.size() << '\n';
    return 0;
}

int RunExperimentRecommendationRankingMemberStatusCommand(
    const std::string& connectionString,
    long long memberId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto member = FindRecommendationRankingMember(connection, memberId);
    if (!member) return 3;
    PrintMember(output, member->member, member->snapshotId, member->memberId, true);
    int ordinal = 0;
    for (const auto& component : member->member.evaluation.components)
        output << "EXPERIMENT_RECOMMENDATION_RANKING_MEMBER_COMPONENT"
               << ",ranking_snapshot_id=" << member->snapshotId
               << ",ranking_member_id=" << member->memberId
               << ",evaluation_result_id="
               << member->member.evaluation.evaluationResultId
               << ",component_ordinal=" << ++ordinal
               << ",component_name="
               << RecommendationMachineText(component.componentName)
               << ",normalized_value="
               << CanonicalRecommendationDouble(component.normalizedValue)
               << ",weight=" << CanonicalRecommendationDouble(component.weight)
               << ",contribution="
               << CanonicalRecommendationDouble(component.weightedContribution)
               << ",is_penalty=" << (component.penalty ? "true" : "false")
               << ",reason_code="
               << RecommendationMachineText(component.reasonCode)
               << ",explanation="
               << RecommendationMachineText(component.explanation) << '\n';
    return 0;
}

int RunCompareExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    std::pair<long long, long long> evaluationIds,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto left = FindRankingEvaluation(connection, evaluationIds.first);
    const auto right = FindRankingEvaluation(connection, evaluationIds.second);
    if (!left || !right) return 3;
    PrintComparison(output, CompareRecommendationEvaluations(*left, *right));
    return 0;
}

int RunCompareExperimentRecommendationRankingMembersCommand(
    const std::string& connectionString,
    std::pair<long long, long long> memberIds,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto left = FindRecommendationRankingMember(connection, memberIds.first);
    const auto right = FindRecommendationRankingMember(connection, memberIds.second);
    if (!left || !right) return 3;
    const bool sameScope = left->snapshotId == right->snapshotId;
    PrintComparison(output, CompareRecommendationEvaluations(
        left->member.evaluation, right->member.evaluation,
        left->member.globalOrdinal, right->member.globalOrdinal, sameScope));
    return 0;
}

} // namespace EA::ExperimentRecommendation
