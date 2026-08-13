#include "../Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace EA::ExperimentRecommendation;

namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using Classification =
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification;
using Consistency =
    RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Diagnostic = RecommendationCampaignOutcomeAssessmentDiagnosticCode;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using ScientificMember =
    RecommendationCampaignOutcomeAssessmentScientificMemberEvidence;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;

class ThrowingStreamBuffer : public std::streambuf
{
protected:
    int_type overflow(int_type) override
    {
        throw std::runtime_error("expected_stream_failure");
    }
};

RecommendationCampaignStatusMemberInput StatusMember(
    int ordinal,
    RecommendationCampaignStatusTerminalResult terminal =
        RecommendationCampaignStatusTerminalResult::succeeded,
    std::optional<long long> experimentId = std::nullopt)
{
    RecommendationCampaignStatusMemberInput member;
    member.memberOrdinal = ordinal;
    member.materializationMemberId = 100 + ordinal;
    member.rankingMemberId = 200 + ordinal;
    member.recommendationId = 300 + ordinal;
    member.sourceExperimentId = 400 + ordinal;
    member.proposalId = 500 + ordinal;
    member.currentReviewDecisionId = 600 + ordinal;
    member.currentReviewDecision = "approve";
    if (terminal == RecommendationCampaignStatusTerminalResult::notTerminal)
        return member;

    member.authorizationReviewDecisionId = 600 + ordinal;
    member.executionId = 700 + ordinal;
    member.activationId = 800 + ordinal;
    member.experimentId = experimentId.value_or(900 + ordinal);
    member.executionCount = 1;
    member.activationCount = 1;
    RecommendationCampaignStatusExperimentEvidence experiment;
    experiment.experimentId = *member.experimentId;
    experiment.symbol = "eurusd";
    experiment.predictionHorizon = 12;
    experiment.targetEpochs = 120;
    experiment.invocationMode = "recommendation_conversion";
    experiment.invocationIdentityCanonical = "integration-invocation";
    experiment.invocationProvenanceValid = true;
    if (terminal == RecommendationCampaignStatusTerminalResult::succeeded)
    {
        experiment.status = "completed";
        experiment.phase = "done";
        experiment.completedAt = "2026-07-19 12:00:00+00";
        experiment.modelId = 1000 + ordinal;
        experiment.modelLinkCount = 1;
        experiment.inferStartPresent = true;
        experiment.inferEndPresent = true;
        experiment.inferConfigured = true;
        experiment.inferenceResultCount = 1;
        experiment.completedInferenceResultCount = 1;
        experiment.analysisResultCount = 1;
        experiment.completedAnalysisResultCount = 1;
    }
    else if (terminal == RecommendationCampaignStatusTerminalResult::failed)
    {
        experiment.status = "failed";
        experiment.phase = "train";
        experiment.completedAt = "2026-07-19 12:00:00+00";
        experiment.errorMessage = "worker_failed";
    }
    else if (terminal == RecommendationCampaignStatusTerminalResult::cancelled)
    {
        experiment.status = "cancelled";
        experiment.phase = "train";
        experiment.completedAt = "2026-07-19 12:00:00+00";
    }
    else
    {
        experiment.status = "malformed";
        experiment.phase = "unknown";
    }
    member.experiment = std::move(experiment);
    return member;
}

RecommendationCampaignStatusSnapshot StatusSnapshot(
    std::vector<RecommendationCampaignStatusMemberInput> members)
{
    RecommendationCampaignStatusInput input;
    input.materializationId = 71;
    input.materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    input.materializationIdentityCanonical =
        "integration-materialization-canonical";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = static_cast<int>(members.size());
    input.observedAt = "2026-07-19 12:00:00+00";
    input.members = std::move(members);
    return BuildRecommendationCampaignStatusSnapshot({71}, std::move(input));
}

Context ComparisonContext(
    std::string symbol = "eurusd",
    double threshold = 0.001,
    int windowSize = 64,
    std::string label = "label_rule_id=1;target_type=2",
    std::string rangeStart = "2025-01-01",
    std::string rangeEnd = "2026-01-01")
{
    return {std::move(symbol), 12, threshold, windowSize,
        std::move(label), std::move(rangeStart), std::move(rangeEnd)};
}

Metric Numeric(std::string identity, std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

SourceEvidence Source(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparisonContext())
{
    return {400 + ordinal, 1100 + ordinal, 1200 + ordinal,
        std::move(context), MetricCollection(std::move(metrics))};
}

ResultEvidence Result(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparisonContext(),
    std::optional<long long> experimentId = std::nullopt,
    std::optional<long long> modelId = std::nullopt,
    std::vector<ResultIdentity> identities = {})
{
    if (identities.empty())
        identities.emplace_back("experiment_analysis_result", 1300 + ordinal);
    return {experimentId.value_or(900 + ordinal),
        modelId ? modelId : std::optional<long long>{1000 + ordinal},
        std::move(identities), std::move(context),
        MetricCollection(std::move(metrics))};
}

std::vector<Metric> SourceMetrics()
{
    return {Numeric("inference_accuracy", 0.70),
        Numeric("leader_score", 0.80)};
}

std::vector<Metric> ResultMetrics()
{
    return {Numeric("inference_accuracy", 0.75),
        Numeric("leader_score", 0.78)};
}

ScientificMember Science(
    int ordinal,
    std::optional<SourceEvidence> source,
    std::optional<ResultEvidence> result,
    Consistency consistency = Consistency::Consistent)
{
    return {ordinal, consistency, std::move(source), std::move(result)};
}

RecommendationCampaignOutcomeAssessmentEvidenceSnapshot Evidence(
    RecommendationCampaignStatusSnapshot status,
    std::vector<ScientificMember> science)
{
    const std::string campaignCanonical =
        "integration-campaign-approval-canonical";
    const std::string materializationCanonical =
        "integration-materialization-canonical";
    const std::string campaignHash =
        RecommendationCanonicalHash(campaignCanonical);
    return {
        RecommendationCampaignOutcomeAssessmentCampaignIdentity(
            41, campaignCanonical, campaignHash),
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
            71, 41, campaignHash,
            kRecommendationCampaignMaterializationContractVersion,
            static_cast<int>(status.members.size()),
            materializationCanonical,
            RecommendationCanonicalHash(materializationCanonical)),
        std::move(status), std::move(science)};
}

Assessment Assess(
    std::vector<RecommendationCampaignStatusMemberInput> statusMembers,
    std::vector<ScientificMember> science)
{
    return BuildRecommendationCampaignOutcomeAssessmentFromEvidence(
        Evidence(StatusSnapshot(std::move(statusMembers)),
            std::move(science)));
}

bool HasDiagnostic(
    const RecommendationCampaignOutcomeAssessmentMember& member,
    Diagnostic diagnostic)
{
    return std::find(member.diagnostics.begin(), member.diagnostics.end(),
               diagnostic) != member.diagnostics.end();
}

const RecommendationCampaignOutcomeAssessmentMetricComparison& Comparison(
    const RecommendationCampaignOutcomeAssessmentMember& member,
    const std::string& identity)
{
    const auto found = std::find_if(member.comparisons.begin(),
        member.comparisons.end(), [&](const auto& comparison)
    {
        return comparison.metricIdentity == identity;
    });
    assert(found != member.comparisons.end());
    return *found;
}

} // namespace

int main()
{
    for (const auto [status, expected] : {
             std::pair{RecommendationCampaignStatusTerminalResult::notTerminal,
                 Lifecycle::NotTerminal},
             std::pair{RecommendationCampaignStatusTerminalResult::succeeded,
                 Lifecycle::Succeeded},
             std::pair{RecommendationCampaignStatusTerminalResult::failed,
                 Lifecycle::Failed},
             std::pair{RecommendationCampaignStatusTerminalResult::cancelled,
                 Lifecycle::Cancelled},
             std::pair{RecommendationCampaignStatusTerminalResult::unknown,
                 Lifecycle::Unknown}})
        assert(RecommendationCampaignOutcomeAssessmentLifecycleFromCampaignStatus(
                   status) == expected);

    const auto comparable = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, ResultMetrics()))});
    assert(comparable.campaignIdentity.campaignApprovalId == 41);
    assert(comparable.materializationIdentity.campaignApprovalId == 41);
    assert(comparable.materializationIdentity.campaignIdentityHash ==
        comparable.campaignIdentity.identityHash);
    assert(comparable.members[0].identity.memberOrdinal == 1);
    assert(comparable.members[0].identity.materializationMemberId == 101);
    assert(comparable.members[0].identity.rankingMemberId == 201);
    assert(comparable.members[0].identity.recommendationId == 301);
    assert(comparable.members[0].identity.sourceExperimentId == 401);
    assert(comparable.members[0].identity.proposalId == 501);
    assert(comparable.members[0].identity.expectedExperimentId ==
        std::optional<long long>{901});
    assert(comparable.members[0].lifecycle == Lifecycle::Succeeded);
    assert(comparable.members[0].consistency == Consistency::Consistent);
    assert(comparable.summary.comparableMetricCount == 2);
    assert(Comparison(comparable.members[0], "inference_accuracy").delta &&
        *Comparison(comparable.members[0], "inference_accuracy").delta >
            0.049 &&
        *Comparison(comparable.members[0], "inference_accuracy").delta <
            0.051);

    const auto contextChanged = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, ResultMetrics(), ComparisonContext("gbpusd")))});
    assert(contextChanged.summary.contextChangedMetricCount == 2);
    assert(Comparison(contextChanged.members[0], "leader_score")
               .classification == Classification::ContextChanged);
    assert(!Comparison(contextChanged.members[0], "leader_score").delta);

    const auto missingSource = Assess({StatusMember(1)},
        {Science(1,
            Source(1, {Numeric("leader_score", 0.80)}),
            Result(1, ResultMetrics()))});
    assert(Comparison(missingSource.members[0], "inference_accuracy")
               .classification == Classification::MissingSourceMetric);

    const auto missingResult = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, {Numeric("inference_accuracy", 0.75)}))});
    assert(Comparison(missingResult.members[0], "leader_score")
               .classification == Classification::MissingResultMetric);

    const ResultEvidence missingModel(901, std::nullopt,
        {ResultIdentity("experiment_analysis_result", 1301)},
        ComparisonContext(), MetricCollection(ResultMetrics()));
    const auto withoutModel = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()), missingModel)});
    assert(HasDiagnostic(withoutModel.members[0],
        Diagnostic::ResultModelIdentityMissing));

    const ResultEvidence missingAnalysis(901, 1001, {},
        ComparisonContext(), MetricCollection(ResultMetrics()));
    const auto withoutAnalysis = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()), missingAnalysis)});
    assert(HasDiagnostic(withoutAnalysis.members[0],
        Diagnostic::ResultIdentityMissing));

    const auto mismatchedResult = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, ResultMetrics(), ComparisonContext(), 9999))});
    assert(HasDiagnostic(mismatchedResult.members[0],
        Diagnostic::ResultExperimentIdentityMismatch));

    auto inconsistentStatus = StatusMember(1);
    inconsistentStatus.workflowConsistent = false;
    inconsistentStatus.workflowDiagnostics = {
        "campaign_status_test_upstream_inconsistency"};
    const auto upstreamInconsistent = Assess({inconsistentStatus},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, ResultMetrics()))});
    assert(upstreamInconsistent.members[0].lifecycle == Lifecycle::Unknown);
    assert(upstreamInconsistent.members[0].consistency ==
        Consistency::Inconsistent);
    assert(HasDiagnostic(upstreamInconsistent.members[0],
        Diagnostic::InputEvidenceInconsistent));

    const auto scientificInconsistent = Assess({StatusMember(1)},
        {Science(1, Source(1, SourceMetrics()),
            Result(1, ResultMetrics()), Consistency::Inconsistent)});
    assert(HasDiagnostic(scientificInconsistent.members[0],
        Diagnostic::InputEvidenceInconsistent));

    const auto proposalOnly = Assess(
        {StatusMember(1,
            RecommendationCampaignStatusTerminalResult::notTerminal)},
        {Science(1, std::nullopt, std::nullopt)});
    assert(proposalOnly.members[0].lifecycle == Lifecycle::NotTerminal);
    assert(proposalOnly.summary.notReadyMemberCount == 1);

    const auto failed = Assess(
        {StatusMember(1,
            RecommendationCampaignStatusTerminalResult::failed)},
        {Science(1, Source(1, SourceMetrics()), std::nullopt)});
    assert(failed.members[0].lifecycle == Lifecycle::Failed);
    assert(failed.summary.terminalFailedMemberCount == 1);

    const auto cancelled = Assess(
        {StatusMember(1,
            RecommendationCampaignStatusTerminalResult::cancelled)},
        {Science(1, Source(1, SourceMetrics()), std::nullopt)});
    assert(cancelled.members[0].lifecycle == Lifecycle::Cancelled);
    assert(cancelled.summary.terminalCancelledMemberCount == 1);

    auto unknown = StatusMember(1,
        RecommendationCampaignStatusTerminalResult::unknown);
    const auto malformed = Assess({unknown},
        {Science(1, Source(1, SourceMetrics()), std::nullopt)});
    assert(malformed.members[0].lifecycle == Lifecycle::Unknown);
    assert(malformed.members[0].consistency == Consistency::Inconsistent);

    auto duplicateExpectedFirst = StatusMember(1,
        RecommendationCampaignStatusTerminalResult::succeeded, 9901);
    auto duplicateExpectedSecond = StatusMember(2,
        RecommendationCampaignStatusTerminalResult::succeeded, 9901);
    const auto duplicateExpected = Assess(
        {duplicateExpectedSecond, duplicateExpectedFirst},
        {Science(2, Source(2, SourceMetrics()),
             Result(2, ResultMetrics(), ComparisonContext(), 9901)),
         Science(1, Source(1, SourceMetrics()),
             Result(1, ResultMetrics(), ComparisonContext(), 9901))});
    assert(duplicateExpected.members[0].identity.memberOrdinal == 1);
    assert(duplicateExpected.members[1].identity.memberOrdinal == 2);
    assert(HasDiagnostic(duplicateExpected.members[0],
        Diagnostic::ExpectedExperimentIdentityReused));
    assert(HasDiagnostic(duplicateExpected.members[1],
        Diagnostic::ExpectedExperimentIdentityReused));

    const std::vector<ResultIdentity> sharedIdentity{
        ResultIdentity("experiment_analysis_result", 7777)};
    const auto duplicateResults = Assess(
        {StatusMember(2), StatusMember(1)},
        {Science(2, Source(2, SourceMetrics()),
             Result(2, ResultMetrics(), ComparisonContext(), 7770, 8888,
                 sharedIdentity)),
         Science(1, Source(1, SourceMetrics()),
             Result(1, ResultMetrics(), ComparisonContext(), 7770, 8888,
                 sharedIdentity))});
    for (const auto& member : duplicateResults.members)
    {
        assert(HasDiagnostic(member, Diagnostic::ResultExperimentReused));
        assert(HasDiagnostic(member, Diagnostic::ResultModelIdentityReused));
        assert(HasDiagnostic(member, Diagnostic::ResultIdentityReused));
    }

    const auto ordered = Assess({StatusMember(2), StatusMember(1)},
        {Science(2, Source(2, SourceMetrics()), Result(2, ResultMetrics())),
         Science(1, Source(1, SourceMetrics()), Result(1, ResultMetrics()))});
    const auto reordered = Assess({StatusMember(1), StatusMember(2)},
        {Science(1, Source(1, SourceMetrics()), Result(1, ResultMetrics())),
         Science(2, Source(2, SourceMetrics()), Result(2, ResultMetrics()))});
    assert(ordered.members == reordered.members);
    assert(ordered.identity == reordered.identity);

    std::ostringstream firstOutput;
    std::ostringstream secondOutput;
    const auto encoded = Assess({StatusMember(1)},
        {Science(1,
            Source(1, SourceMetrics(), ComparisonContext("eurusd", 0.001,
                64, "label definition,1")),
            Result(1, ResultMetrics(), ComparisonContext("eurusd", 0.001,
                64, "label definition,1")))});
    WriteRecommendationCampaignOutcomeAssessment(firstOutput, encoded);
    WriteRecommendationCampaignOutcomeAssessment(secondOutput, encoded);
    assert(firstOutput.str() == secondOutput.str());

    std::ostringstream flaggedOutput;
    flaggedOutput.setf(std::ios::hex, std::ios::basefield);
    flaggedOutput.setf(std::ios::showbase);
    flaggedOutput.precision(2);
    const auto flaggedFlags = flaggedOutput.flags();
    const auto flaggedPrecision = flaggedOutput.precision();
    WriteRecommendationCampaignOutcomeAssessment(flaggedOutput, encoded);
    assert(flaggedOutput.str() == firstOutput.str());
    assert(flaggedOutput.flags() == flaggedFlags);
    assert(flaggedOutput.precision() == flaggedPrecision);

    ThrowingStreamBuffer throwingBuffer;
    std::ostream throwingOutput{&throwingBuffer};
    throwingOutput.setf(std::ios::hex, std::ios::basefield);
    throwingOutput.precision(3);
    const auto throwingFlags = throwingOutput.flags();
    const auto throwingPrecision = throwingOutput.precision();
    throwingOutput.exceptions(std::ios::badbit | std::ios::failbit);
    bool outputFailureObserved = false;
    try
    {
        WriteRecommendationCampaignOutcomeAssessment(
            throwingOutput, encoded);
    }
    catch (const std::exception&)
    {
        outputFailureObserved = true;
    }
    assert(outputFailureObserved);
    assert(throwingOutput.flags() == throwingFlags);
    assert(throwingOutput.precision() == throwingPrecision);

    assert(firstOutput.str().find(
        "RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT") == 0);
    assert(firstOutput.str().find("label%20definition%2C1") !=
        std::string::npos);
    assert(firstOutput.str().find("result_model_id=1001") !=
        std::string::npos);
    assert(firstOutput.str().find(
        "materialization_campaign_approval_identity_hash=" +
        comparable.materializationIdentity.campaignIdentityHash) !=
        std::string::npos);
    assert(firstOutput.str().find(
        "campaign_approval_contract_version=1") != std::string::npos);
    assert(firstOutput.str().find(
        "campaign_status_contract_version=1") != std::string::npos);
    assert(firstOutput.str().find(
        "input_consistency=consistent,final_consistency=consistent") !=
        std::string::npos);
    assert(firstOutput.str().find("source_evidence_present=true") !=
        std::string::npos);
    assert(firstOutput.str().find("result_evidence_present=true") !=
        std::string::npos);
    assert(firstOutput.str().find("result_context_present=true") !=
        std::string::npos);
    assert(firstOutput.str().find("transaction_read_only=true") !=
        std::string::npos);
    assert(firstOutput.str().find("assessment_persisted=false") !=
        std::string::npos);
    assert(firstOutput.str().find("automatic_follow_up=false") !=
        std::string::npos);
    assert(firstOutput.str().find("campaign_success_declared=false") !=
        std::string::npos);
    assert(firstOutput.str().find("follow_up_authorized=false") !=
        std::string::npos);

    std::ostringstream inconsistentOutput;
    WriteRecommendationCampaignOutcomeAssessment(
        inconsistentOutput, scientificInconsistent);
    assert(inconsistentOutput.str().find(
        "input_consistency=inconsistent,final_consistency=inconsistent") !=
        std::string::npos);

    std::ostringstream proposalOnlyOutput;
    WriteRecommendationCampaignOutcomeAssessment(
        proposalOnlyOutput, proposalOnly);
    assert(proposalOnlyOutput.str().find("source_evidence_present=false") !=
        std::string::npos);
    assert(proposalOnlyOutput.str().find("result_evidence_present=false") !=
        std::string::npos);
    assert(proposalOnlyOutput.str().find("result_context_present=false") !=
        std::string::npos);

    return 0;
}
