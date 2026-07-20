#include "ExperimentRecommendationCampaignOutcomeAssessment.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace EA::ExperimentRecommendation;

namespace
{

using AggregateOutcome =
    RecommendationCampaignOutcomeAssessmentAggregateOutcome;
using Assessment = RecommendationCampaignOutcomeAssessment;
using CampaignIdentity =
    RecommendationCampaignOutcomeAssessmentCampaignIdentity;
using Classification =
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification;
using Consistency = RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Diagnostic = RecommendationCampaignOutcomeAssessmentDiagnosticCode;
using Difference = RecommendationCampaignOutcomeAssessmentContextDifference;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using MaterializationIdentity =
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity;
using Member = RecommendationCampaignOutcomeAssessmentMember;
using MemberEvidence = RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MemberIdentity =
    RecommendationCampaignOutcomeAssessmentMemberIdentity;
using MemberOutcome = RecommendationCampaignOutcomeAssessmentMemberOutcome;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricComparison =
    RecommendationCampaignOutcomeAssessmentMetricComparison;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;
using Summary = RecommendationCampaignOutcomeAssessmentSummary;

static_assert(Assessment::readOnly);
static_assert(Assessment::pointInTime);
static_assert(!Assessment::persistent);
static_assert(!Assessment::authoritative);
static_assert(!Assessment::declaresCampaignSuccess);
static_assert(!std::is_default_constructible_v<Assessment>);
static_assert(!std::is_default_constructible_v<Member>);
static_assert(!std::is_default_constructible_v<Summary>);
static_assert(!std::is_default_constructible_v<MetricComparison>);
static_assert(!std::is_aggregate_v<Assessment>);
static_assert(!std::is_aggregate_v<Member>);
static_assert(!std::is_aggregate_v<Summary>);
static_assert(!std::is_aggregate_v<MetricComparison>);
static_assert(!std::is_copy_assignable_v<Assessment>);
static_assert(!std::is_copy_assignable_v<Member>);
static_assert(!std::is_copy_assignable_v<Summary>);
static_assert(!std::is_copy_assignable_v<MetricComparison>);
static_assert(!std::is_constructible_v<MetricComparison, std::string,
    Classification, std::optional<double>, std::optional<double>,
    std::optional<double>, std::vector<Difference>>);
static_assert(!std::is_constructible_v<Member, MemberIdentity, Lifecycle,
    Consistency, MemberOutcome, std::vector<Diagnostic>,
    std::optional<SourceEvidence>, std::optional<ResultEvidence>,
    std::vector<MetricComparison>>);

template <std::size_t... Indices>
consteval bool SummaryConstructorIsPublic(std::index_sequence<Indices...>)
{
    return std::is_constructible_v<Summary, AggregateOutcome,
        decltype((void)Indices, int{})...>;
}

static_assert(!SummaryConstructorIsPublic(std::make_index_sequence<25>{}));

template <typename Function>
void AssertInvalidArgument(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
}

CampaignIdentity Campaign()
{
    const std::string canonical = "campaign-approval-canonical";
    return {41, canonical, RecommendationCanonicalHash(canonical)};
}

MaterializationIdentity Materialization(int memberCount)
{
    const std::string canonical =
        "materialization-canonical:" + std::to_string(memberCount);
    return {71, 41, Campaign().identityHash,
        kRecommendationCampaignMaterializationContractVersion, memberCount,
        canonical, RecommendationCanonicalHash(canonical)};
}

Context ComparableContext(
    std::string symbol = "eurusd",
    int horizon = 12,
    double threshold = 0.0008,
    int windowSize = 96,
    std::string labelDefinition = "three_class_forward_return_v1",
    std::string rangeStart = "2026-01-01T00:00:00Z",
    std::string rangeEnd = "2026-03-31T23:59:59Z")
{
    return {std::move(symbol), horizon, threshold, windowSize,
        std::move(labelDefinition), std::move(rangeStart),
        std::move(rangeEnd)};
}

Metric NumericMetric(
    std::string identity,
    std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

Metric UnsupportedMetric(
    std::string identity,
    std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::Unsupported};
}

MetricCollection Metrics(std::vector<Metric> metrics)
{
    return MetricCollection(std::move(metrics));
}

long long SourceExperiment(int ordinal)
{
    return 500 + ordinal;
}

long long ResultExperiment(int ordinal)
{
    return 900 + ordinal;
}

long long ResultRecord(int ordinal)
{
    return 2000 + ordinal;
}

MemberIdentity Identity(
    int ordinal,
    std::optional<long long> expectedExperimentId)
{
    return {ordinal, 100 + ordinal, 200 + ordinal, 300 + ordinal,
        SourceExperiment(ordinal), 400 + ordinal, expectedExperimentId};
}

SourceEvidence Source(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparableContext(),
    std::optional<long long> sourceExperimentOverride = std::nullopt)
{
    return {sourceExperimentOverride.value_or(SourceExperiment(ordinal)),
        600 + ordinal, 700 + ordinal, std::move(context),
        Metrics(std::move(metrics))};
}

ResultEvidence Result(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparableContext(),
    std::optional<long long> experimentOverride = std::nullopt,
    std::optional<long long> resultRecordOverride = std::nullopt)
{
    return {experimentOverride.value_or(ResultExperiment(ordinal)),
        1000 + ordinal,
        {ResultIdentity("scientific_result",
            resultRecordOverride.value_or(ResultRecord(ordinal)))},
        std::move(context), Metrics(std::move(metrics))};
}

std::vector<Metric> BaseSourceMetrics()
{
    return {NumericMetric("inference_accuracy", 0.70),
        NumericMetric("leader_score", 0.80)};
}

std::vector<Metric> BaseResultMetrics()
{
    return {NumericMetric("inference_accuracy", 0.75),
        NumericMetric("leader_score", 0.78)};
}

MemberEvidence Succeeded(
    int ordinal,
    std::vector<Metric> sourceMetrics = BaseSourceMetrics(),
    std::vector<Metric> resultMetrics = BaseResultMetrics(),
    Context sourceContext = ComparableContext(),
    Context resultContext = ComparableContext())
{
    return {Identity(ordinal, ResultExperiment(ordinal)),
        Lifecycle::Succeeded,
        Source(ordinal, std::move(sourceMetrics), std::move(sourceContext)),
        Result(ordinal, std::move(resultMetrics), std::move(resultContext))};
}

MemberEvidence LifecycleOnly(
    int ordinal,
    Lifecycle lifecycle,
    std::optional<long long> expectedExperimentId = std::nullopt)
{
    return {Identity(ordinal, expectedExperimentId), lifecycle, std::nullopt,
        std::nullopt};
}

Assessment Assess(
    std::vector<MemberEvidence> members,
    std::string observedAt = "2026-07-19T22:00:00Z")
{
    return BuildRecommendationCampaignOutcomeAssessment(Campaign(),
        Materialization(static_cast<int>(members.size())), observedAt,
        members);
}

const MetricComparison& ComparisonFor(
    const Member& member,
    const std::string& identity)
{
    const auto found = std::find_if(member.comparisons.begin(),
        member.comparisons.end(), [&](const MetricComparison& comparison)
    {
        return comparison.metricIdentity == identity;
    });
    assert(found != member.comparisons.end());
    return *found;
}

bool HasDiagnostic(const Member& member, Diagnostic diagnostic)
{
    return std::find(member.diagnostics.begin(), member.diagnostics.end(),
               diagnostic) != member.diagnostics.end();
}

} // namespace

int main()
{
    // Multiple metrics, including a future metric, are canonicalized by metric
    // identity and compared independently.
    const std::vector<Metric> sourceMetrics{
        NumericMetric("leader_score", 0.80),
        NumericMetric("future_calibration", 0.50),
        NumericMetric("inference_accuracy", 0.70)};
    const std::vector<Metric> resultMetrics{
        NumericMetric("inference_accuracy", 0.75),
        NumericMetric("leader_score", 0.78),
        NumericMetric("future_calibration", 0.55)};
    const auto multiMetric = Assess(
        {Succeeded(1, sourceMetrics, resultMetrics)});
    const auto& multiMember = multiMetric.members[0];
    assert(multiMember.outcome == MemberOutcome::SucceededComparable);
    assert(multiMember.consistency == Consistency::Consistent);
    assert(multiMember.sourceEvidence->metrics.metrics.size() == 3);
    assert(multiMember.resultEvidence->metrics.metrics.size() == 3);
    assert(multiMember.comparisons.size() == 3);
    assert(multiMember.sourceEvidence->metrics.metrics[0].identity ==
        "future_calibration");
    assert(multiMember.sourceEvidence->metrics.metrics[1].identity ==
        "inference_accuracy");
    assert(multiMember.sourceEvidence->metrics.metrics[2].identity ==
        "leader_score");
    assert(multiMember.comparisons[0].metricIdentity == "future_calibration");
    assert(multiMember.comparisons[1].metricIdentity == "inference_accuracy");
    assert(multiMember.comparisons[2].metricIdentity == "leader_score");
    assert(ComparisonFor(multiMember, "inference_accuracy").delta &&
        *ComparisonFor(multiMember, "inference_accuracy").delta > 0.0);
    assert(ComparisonFor(multiMember, "leader_score").delta &&
        *ComparisonFor(multiMember, "leader_score").delta < 0.0);
    assert(ComparisonFor(multiMember, "future_calibration").delta &&
        *ComparisonFor(multiMember, "future_calibration").delta > 0.0);

    const auto reorderedMetrics = Assess({Succeeded(1,
        {sourceMetrics[2], sourceMetrics[0], sourceMetrics[1]},
        {resultMetrics[2], resultMetrics[1], resultMetrics[0]})});
    assert(reorderedMetrics.members == multiMetric.members);
    assert(reorderedMetrics.summary == multiMetric.summary);
    assert(reorderedMetrics.identity == multiMetric.identity);

    AssertInvalidArgument([&]
    {
        (void)Metrics({NumericMetric("leader_score", 0.1),
            NumericMetric("leader_score", 0.2)});
    });
    AssertInvalidArgument([]
    {
        (void)ResultEvidence(901, 1001,
            {ResultIdentity("analysis", 1),
                ResultIdentity("analysis", 1)},
            ComparableContext(), Metrics({}));
    });

    // Successful members distinguish comparable, changed-context, and metric
    // gap outcomes without mixing those scientific results into lifecycle.
    const auto comparable = Assess({Succeeded(1)});
    assert(comparable.members[0].lifecycle == Lifecycle::Succeeded);
    assert(comparable.members[0].outcome ==
        MemberOutcome::SucceededComparable);
    assert(comparable.summary.succeededComparableMemberCount == 1);
    assert(comparable.summary.comparableMetricCount == 2);
    assert(comparable.summary.positiveDeltaCount == 1);
    assert(comparable.summary.negativeDeltaCount == 1);

    const auto changedContext = Assess({Succeeded(1, BaseSourceMetrics(),
        BaseResultMetrics(), ComparableContext(), ComparableContext("gbpusd"))});
    assert(changedContext.members[0].outcome ==
        MemberOutcome::SucceededContextChanged);
    assert(changedContext.members[0].lifecycle == Lifecycle::Succeeded);
    assert(changedContext.members[0].consistency == Consistency::Consistent);
    for (const auto& comparison : changedContext.members[0].comparisons)
    {
        assert(comparison.classification == Classification::ContextChanged);
        assert(comparison.contextDifferences ==
            std::vector<Difference>{Difference::Symbol});
        assert(!comparison.delta);
    }

    const auto allContextDifferences = Assess({Succeeded(1,
        BaseSourceMetrics(), BaseResultMetrics(), ComparableContext(),
        ComparableContext("gbpusd", 24, 0.001, 192,
            "binary_forward_return_v2", "2026-02-01T00:00:00Z",
            "2026-04-30T23:59:59Z"))});
    assert(allContextDifferences.members[0].comparisons[0].contextDifferences ==
        std::vector<Difference>({Difference::Symbol,
            Difference::PredictionHorizon, Difference::Threshold,
            Difference::WindowSize, Difference::LabelDefinition,
            Difference::InferenceRange}));

    const auto metricGap = Assess({Succeeded(1,
        {NumericMetric("inference_accuracy", 0.70),
            NumericMetric("leader_score", 0.80)},
        {NumericMetric("future_calibration", 0.55),
            NumericMetric("inference_accuracy", 0.75)})});
    assert(metricGap.members[0].outcome == MemberOutcome::SucceededMetricGap);
    assert(ComparisonFor(metricGap.members[0], "future_calibration")
               .classification == Classification::MissingSourceMetric);
    assert(ComparisonFor(metricGap.members[0], "leader_score")
               .classification == Classification::MissingResultMetric);
    assert(metricGap.summary.succeededMetricGapMemberCount == 1);
    assert(metricGap.summary.metricComparisonCount == 3);
    assert(metricGap.summary.comparableMetricCount == 1);
    assert(metricGap.summary.missingSourceMetricCount == 1);
    assert(metricGap.summary.missingResultMetricCount == 1);

    const auto missingValue = Assess({Succeeded(1,
        {NumericMetric("inference_accuracy", std::nullopt)},
        {NumericMetric("inference_accuracy", 0.75)})});
    assert(missingValue.members[0].outcome ==
        MemberOutcome::SucceededMetricGap);
    assert(missingValue.members[0].comparisons[0].classification ==
        Classification::MetricValueUnavailable);
    const auto unsupported = Assess({Succeeded(1,
        {UnsupportedMetric("inference_accuracy", 0.70)},
        {NumericMetric("inference_accuracy", 0.75)})});
    assert(unsupported.members[0].comparisons[0].classification ==
        Classification::Unsupported);
    const auto supportedSingleMetric = Assess({Succeeded(1,
        {NumericMetric("inference_accuracy", 0.70)},
        {NumericMetric("inference_accuracy", 0.75)})});
    assert(supportedSingleMetric.members[0].comparisons[0].classification ==
        Classification::Comparable);
    assert(supportedSingleMetric.identity.hash != unsupported.identity.hash);

    const auto missingSourceEvidence = Assess({MemberEvidence(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded, std::nullopt,
        Result(1, BaseResultMetrics()))});
    assert(missingSourceEvidence.members[0].consistency ==
        Consistency::Consistent);
    assert(missingSourceEvidence.members[0].outcome ==
        MemberOutcome::SucceededMetricGap);
    assert(missingSourceEvidence.members[0].comparisons.size() == 2);
    assert(missingSourceEvidence.summary.missingSourceMetricCount == 2);

    // Lifecycle states do not require result payload. Proposal-only is simply
    // not ready, never a missing-result scientific classification.
    const auto failed = Assess(
        {LifecycleOnly(1, Lifecycle::Failed, ResultExperiment(1))});
    assert(failed.members[0].outcome == MemberOutcome::TerminalFailed);
    assert(failed.members[0].consistency == Consistency::Consistent);
    assert(!failed.members[0].resultEvidence);
    assert(failed.members[0].comparisons.empty());

    const auto cancelled = Assess(
        {LifecycleOnly(1, Lifecycle::Cancelled, ResultExperiment(1))});
    assert(cancelled.members[0].outcome == MemberOutcome::TerminalCancelled);
    assert(!cancelled.members[0].resultEvidence);

    const auto nonterminal = Assess(
        {LifecycleOnly(1, Lifecycle::NotTerminal, ResultExperiment(1))});
    assert(nonterminal.members[0].outcome == MemberOutcome::NotReady);
    assert(nonterminal.members[0].diagnostics.empty());
    assert(!nonterminal.members[0].resultEvidence);

    const auto unknown = Assess({LifecycleOnly(1, Lifecycle::Unknown)});
    assert(unknown.members[0].lifecycle == Lifecycle::Unknown);
    assert(unknown.members[0].outcome == MemberOutcome::NotReady);

    const auto proposalOnly = Assess(
        {LifecycleOnly(1, Lifecycle::NotTerminal)});
    assert(!proposalOnly.members[0].identity.expectedExperimentId);
    assert(!proposalOnly.members[0].sourceEvidence);
    assert(!proposalOnly.members[0].resultEvidence);
    assert(proposalOnly.members[0].outcome == MemberOutcome::NotReady);
    assert(proposalOnly.members[0].diagnostics.empty());
    assert(proposalOnly.identity.canonicalText.find("result_evidence_missing") ==
        std::string::npos);

    const auto unexpectedFailedResult = Assess({MemberEvidence(
        Identity(1, ResultExperiment(1)), Lifecycle::Failed, std::nullopt,
        Result(1, BaseResultMetrics()))});
    assert(unexpectedFailedResult.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ResultEvidenceUnexpectedForLifecycle});
    assert(unexpectedFailedResult.members[0].resultEvidence);
    assert(unexpectedFailedResult.members[0].outcome ==
        MemberOutcome::Inconsistent);

    const auto failedWithoutExperiment =
        Assess({LifecycleOnly(1, Lifecycle::Failed)});
    assert(failedWithoutExperiment.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ExpectedExperimentIdentityMissing});
    const auto cancelledWithoutExperiment =
        Assess({LifecycleOnly(1, Lifecycle::Cancelled)});
    assert(cancelledWithoutExperiment.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ExpectedExperimentIdentityMissing});

    const auto inputInconsistent = Assess({MemberEvidence(Identity(1,
        std::nullopt), Lifecycle::Unknown, std::nullopt, std::nullopt,
        Consistency::Inconsistent)});
    assert(inputInconsistent.members[0].lifecycle == Lifecycle::Unknown);
    assert(inputInconsistent.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::InputEvidenceInconsistent});
    assert(inputInconsistent.members[0].consistency ==
        Consistency::Inconsistent);
    assert(inputInconsistent.summary.outcome == AggregateOutcome::Inconsistent);

    // Malformed downstream evidence is retained when member identity remains
    // trustworthy, with fixed-order diagnostics and no scientific comparison.
    const MemberEvidence incompleteResult(
        Identity(1, std::nullopt), Lifecycle::Succeeded,
        Source(1, {}, ComparableContext(), 999),
        ResultEvidence(901, std::nullopt, {}, std::nullopt, Metrics({})));
    const auto inconsistent = Assess({incompleteResult});
    assert(inconsistent.members[0].lifecycle == Lifecycle::Succeeded);
    assert(inconsistent.members[0].consistency == Consistency::Inconsistent);
    assert(inconsistent.members[0].outcome == MemberOutcome::Inconsistent);
    assert(inconsistent.members[0].resultEvidence);
    assert(inconsistent.members[0].comparisons.empty());
    assert(inconsistent.members[0].diagnostics ==
        std::vector<Diagnostic>({
            Diagnostic::SourceExperimentIdentityMismatch,
            Diagnostic::ExpectedExperimentIdentityMissing,
            Diagnostic::ResultModelIdentityMissing,
            Diagnostic::ResultIdentityMissing,
            Diagnostic::ResultContextMissing}));
    assert(inconsistent.summary.inconsistentMemberCount == 1);
    assert(inconsistent.summary.metricComparisonCount == 0);

    const MemberEvidence noResult(Identity(1, ResultExperiment(1)),
        Lifecycle::Succeeded, Source(1, BaseSourceMetrics()), std::nullopt);
    const auto noResultAssessment = Assess({noResult});
    assert(noResultAssessment.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultEvidenceMissing});
    assert(noResultAssessment.members[0].outcome ==
        MemberOutcome::Inconsistent);

    const MemberEvidence missingResultExperiment(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        ResultEvidence(std::nullopt, 1001,
            {ResultIdentity("scientific_result", ResultRecord(1))},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const auto missingExperiment = Assess({missingResultExperiment});
    assert(missingExperiment.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ResultExperimentIdentityMissing});

    const MemberEvidence provenanceMismatch(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        Result(1, BaseResultMetrics(), ComparableContext(), 999));
    const auto malformedProvenance = Assess({provenanceMismatch});
    assert(malformedProvenance.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ResultExperimentIdentityMismatch});
    assert(malformedProvenance.members[0].resultEvidence->experimentId == 999);
    assert(malformedProvenance.members[0].consistency ==
        Consistency::Inconsistent);

    const MemberEvidence missingModel(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        ResultEvidence(ResultExperiment(1), std::nullopt,
            {ResultIdentity("scientific_result", ResultRecord(1))},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const auto missingModelAssessment = Assess({missingModel});
    assert(missingModelAssessment.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultModelIdentityMissing});
    assert(missingModelAssessment.members[0].consistency ==
        Consistency::Inconsistent);
    assert(missingModelAssessment.identity.hash !=
        malformedProvenance.identity.hash);

    // Cross-member experiment reuse marks every affected member, and the
    // mismatched member additionally retains its exact mismatch diagnostic.
    const MemberEvidence reusedExperimentOne(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()), Result(1, BaseResultMetrics()));
    const MemberEvidence reusedExperimentTwo(
        Identity(2, ResultExperiment(2)), Lifecycle::Succeeded,
        Source(2, BaseSourceMetrics()), Result(2, BaseResultMetrics(),
            ComparableContext(), ResultExperiment(1)));
    const auto experimentReuse =
        Assess({reusedExperimentTwo, reusedExperimentOne});
    assert(HasDiagnostic(experimentReuse.members[0],
        Diagnostic::ResultExperimentReused));
    assert(HasDiagnostic(experimentReuse.members[1],
        Diagnostic::ResultExperimentReused));
    assert(!HasDiagnostic(experimentReuse.members[0],
        Diagnostic::ResultExperimentIdentityMismatch));
    assert(HasDiagnostic(experimentReuse.members[1],
        Diagnostic::ResultExperimentIdentityMismatch));
    assert(experimentReuse.members[0].outcome == MemberOutcome::Inconsistent);
    assert(experimentReuse.members[1].outcome == MemberOutcome::Inconsistent);

    const MemberEvidence reusedResultOne(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()), Result(1, BaseResultMetrics(),
            ComparableContext(), std::nullopt, 8888));
    const MemberEvidence reusedResultTwo(
        Identity(2, ResultExperiment(2)), Lifecycle::Succeeded,
        Source(2, BaseSourceMetrics()), Result(2, BaseResultMetrics(),
            ComparableContext(), std::nullopt, 8888));
    const auto resultReuse = Assess({reusedResultTwo, reusedResultOne});
    assert(resultReuse.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultIdentityReused});
    assert(resultReuse.members[1].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultIdentityReused});

    const MemberEvidence reusedModelOne(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        ResultEvidence(ResultExperiment(1), 7777,
            {ResultIdentity("scientific_result", ResultRecord(1))},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const MemberEvidence reusedModelTwo(
        Identity(2, ResultExperiment(2)), Lifecycle::Succeeded,
        Source(2, BaseSourceMetrics()),
        ResultEvidence(ResultExperiment(2), 7777,
            {ResultIdentity("scientific_result", ResultRecord(2))},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const auto modelReuse = Assess({reusedModelTwo, reusedModelOne});
    assert(modelReuse.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultModelIdentityReused});
    assert(modelReuse.members[1].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::ResultModelIdentityReused});

    // Aggregate precedence is deterministic: inconsistency, failure,
    // cancellation, not-ready, then successful context/gap/comparable states.
    const auto allPrecedence = Assess({Succeeded(5),
        LifecycleOnly(2, Lifecycle::Failed, ResultExperiment(2)),
        provenanceMismatch,
        LifecycleOnly(4, Lifecycle::NotTerminal, ResultExperiment(4)),
        LifecycleOnly(3, Lifecycle::Cancelled, ResultExperiment(3))});
    assert(allPrecedence.summary.outcome == AggregateOutcome::Inconsistent);
    assert(Assess({Succeeded(3),
               LifecycleOnly(1, Lifecycle::Failed, ResultExperiment(1)),
               LifecycleOnly(2, Lifecycle::Cancelled, ResultExperiment(2))})
               .summary.outcome == AggregateOutcome::HasFailures);
    assert(Assess({Succeeded(2),
               LifecycleOnly(1, Lifecycle::Cancelled, ResultExperiment(1))})
               .summary.outcome == AggregateOutcome::HasCancellations);
    assert(Assess({Succeeded(2),
               LifecycleOnly(1, Lifecycle::NotTerminal,
                   ResultExperiment(1))})
               .summary.outcome == AggregateOutcome::NotReady);
    assert(Assess({Succeeded(1, BaseSourceMetrics(), BaseResultMetrics(),
                       ComparableContext(), ComparableContext("gbpusd")),
               Succeeded(2, {NumericMetric("a", 0.1)}, {})})
               .summary.outcome ==
        AggregateOutcome::SucceededContextChanged);
    assert(Assess({Succeeded(1, {NumericMetric("a", 0.1)}, {}),
               Succeeded(2)})
               .summary.outcome == AggregateOutcome::SucceededMetricGap);
    assert(comparable.summary.outcome ==
        AggregateOutcome::SucceededComparable);

    // Member outcomes and per-metric/delta counts remain separate.
    assert(multiMetric.summary.memberCount == 1);
    assert(multiMetric.summary.succeededLifecycleCount == 1);
    assert(multiMetric.summary.succeededComparableMemberCount == 1);
    assert(multiMetric.summary.metricComparisonCount == 3);
    assert(multiMetric.summary.comparableMetricCount == 3);
    assert(multiMetric.summary.positiveDeltaCount == 2);
    assert(multiMetric.summary.negativeDeltaCount == 1);
    assert(multiMetric.summary.zeroDeltaCount == 0);

    const auto zeroDelta = Assess({Succeeded(1,
        {NumericMetric("a", -0.0)}, {NumericMetric("a", 0.0)})});
    assert(zeroDelta.members[0].comparisons[0].delta);
    assert(*zeroDelta.members[0].comparisons[0].delta == 0.0);
    assert(!std::signbit(*zeroDelta.members[0].comparisons[0].delta));
    assert(zeroDelta.summary.zeroDeltaCount == 1);

    // Every contract dimension is bound into identity. observed_at alone is
    // intentionally excluded.
    const auto laterObservation = Assess({Succeeded(1)},
        "2026-07-19T22:05:00Z");
    assert(laterObservation.observedAt != comparable.observedAt);
    assert(laterObservation.identity == comparable.identity);

    const auto lifecycleNotTerminal =
        Assess({LifecycleOnly(1, Lifecycle::NotTerminal)});
    const auto lifecycleUnknown =
        Assess({LifecycleOnly(1, Lifecycle::Unknown)});
    assert(lifecycleNotTerminal.members[0].outcome ==
        lifecycleUnknown.members[0].outcome);
    assert(lifecycleNotTerminal.identity.hash !=
        lifecycleUnknown.identity.hash);

    assert(comparable.identity.hash != malformedProvenance.identity.hash);
    assert(malformedProvenance.identity.hash !=
        noResultAssessment.identity.hash);
    const auto changedMetricValue = Assess({Succeeded(1,
        BaseSourceMetrics(),
        {NumericMetric("inference_accuracy", 0.76),
            NumericMetric("leader_score", 0.78)})});
    assert(changedMetricValue.identity.hash != comparable.identity.hash);
    assert(metricGap.identity.hash != comparable.identity.hash);
    assert(changedContext.identity.hash != comparable.identity.hash);
    assert(unsupported.identity.hash != supportedSingleMetric.identity.hash);

    const MemberEvidence changedProvenance(
        Identity(1, 1901), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        ResultEvidence(1901, 1001,
            {ResultIdentity("scientific_result", 2901)},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const auto changedProvenanceAssessment = Assess({changedProvenance});
    assert(changedProvenanceAssessment.members[0].outcome ==
        MemberOutcome::SucceededComparable);
    assert(changedProvenanceAssessment.identity.hash != comparable.identity.hash);
    assert(comparable.identity.contractVersion ==
        kRecommendationCampaignOutcomeAssessmentContractVersion);
    assert(comparable.identity.canonicalText.starts_with(
        "experiment_recommendation_campaign_outcome_assessment_v2;"
        "assessment_contract_version=2;"));
    assert(comparable.identity.hash == RecommendationCanonicalHash(
        comparable.identity.canonicalText));
    assert(comparable.identity.canonicalText.find(
               "campaign_identity=27:campaign-approval-canonical") !=
        std::string::npos);
    assert(comparable.identity.canonicalText.find(
               "materialization_identity=27:materialization-canonical:1") !=
        std::string::npos);
    assert(comparable.identity.canonicalText.find("lifecycle=succeeded") !=
        std::string::npos);
    assert(malformedProvenance.identity.canonicalText.find(
               "result_experiment_identity_mismatch") != std::string::npos);

    // Member order, result identity order, diagnostics, metrics, comparisons,
    // canonical text, and hash are deterministic.
    const MemberEvidence twoResultIdentities(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, BaseSourceMetrics()),
        ResultEvidence(ResultExperiment(1), 1001,
            {ResultIdentity("z_future", 12),
                ResultIdentity("analysis", 11)},
            ComparableContext(), Metrics(BaseResultMetrics())));
    const MemberEvidence twoResultIdentitiesReordered(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, {BaseSourceMetrics()[1], BaseSourceMetrics()[0]}),
        ResultEvidence(ResultExperiment(1), 1001,
            {ResultIdentity("analysis", 11),
                ResultIdentity("z_future", 12)},
            ComparableContext(),
            Metrics({BaseResultMetrics()[1], BaseResultMetrics()[0]})));
    const auto canonicalOne = Assess({twoResultIdentities});
    const auto canonicalTwo = Assess({twoResultIdentitiesReordered});
    assert(canonicalOne.members == canonicalTwo.members);
    assert(canonicalOne.identity == canonicalTwo.identity);
    assert(canonicalOne.members[0].resultEvidence->resultIdentities[0].kind ==
        "analysis");
    assert(canonicalOne.members[0].resultEvidence->resultIdentities[1].kind ==
        "z_future");

    const auto orderedMembers = Assess({Succeeded(1), Succeeded(2)});
    const auto shuffledMembers = Assess({Succeeded(2), Succeeded(1)});
    assert(orderedMembers.members == shuffledMembers.members);
    assert(orderedMembers.summary == shuffledMembers.summary);
    assert(orderedMembers.identity == shuffledMembers.identity);
    assert(shuffledMembers.members[0].identity.memberOrdinal == 1);
    assert(shuffledMembers.members[1].identity.memberOrdinal == 2);

    // Invalid scalar input and untrustworthy top-level identity fail fast.
    AssertInvalidArgument([]
    {
        (void)Metric("inference_accuracy",
            std::numeric_limits<double>::quiet_NaN(),
            MetricSupport::NumericDelta);
    });
    AssertInvalidArgument([]
    {
        (void)ComparableContext("eurusd", 12,
            std::numeric_limits<double>::infinity());
    });
    AssertInvalidArgument([]
    {
        (void)ResultIdentity("analysis", 0);
    });
    AssertInvalidArgument([]
    {
        (void)Identity(1, 0);
    });
    AssertInvalidArgument([]
    {
        (void)Metric(std::string("bad\0identity", 12), 0.1,
            MetricSupport::NumericDelta);
    });
    AssertInvalidArgument([]
    {
        (void)BuildRecommendationCampaignOutcomeAssessment(Campaign(),
            Materialization(2), "2026-07-19T22:00:00Z",
            {Succeeded(1), Succeeded(1)});
    });
    AssertInvalidArgument([]
    {
        const std::string canonical = "materialization-canonical:1";
        const MaterializationIdentity mismatched(71, 41,
            RecommendationCanonicalHash("different-campaign-identity"),
            kRecommendationCampaignMaterializationContractVersion, 1,
            canonical, RecommendationCanonicalHash(canonical));
        (void)BuildRecommendationCampaignOutcomeAssessment(Campaign(),
            mismatched, "2026-07-19T22:00:00Z", {Succeeded(1)});
    });
    const MemberEvidence reusedExpectedOne(Identity(1, 9901),
        Lifecycle::NotTerminal, std::nullopt, std::nullopt);
    const MemberEvidence reusedExpectedTwo(Identity(2, 9901),
        Lifecycle::NotTerminal, std::nullopt, std::nullopt);
    const auto expectedReuse =
        Assess({reusedExpectedTwo, reusedExpectedOne});
    assert(expectedReuse.members[0].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ExpectedExperimentIdentityReused});
    assert(expectedReuse.members[1].diagnostics ==
        std::vector<Diagnostic>{
            Diagnostic::ExpectedExperimentIdentityReused});
    assert(expectedReuse.summary.outcome == AggregateOutcome::Inconsistent);
    AssertInvalidArgument([]
    {
        (void)BuildRecommendationCampaignOutcomeAssessment(Campaign(),
            Materialization(1), "", {Succeeded(1)});
    });
    AssertInvalidArgument([]
    {
        (void)MemberEvidence(Identity(1, std::nullopt), Lifecycle::Unknown,
            std::nullopt, std::nullopt,
            static_cast<Consistency>(999));
    });

    const MemberEvidence overflowingDelta(
        Identity(1, ResultExperiment(1)), Lifecycle::Succeeded,
        Source(1, {NumericMetric("wide",-
            std::numeric_limits<double>::max())}),
        Result(1, {NumericMetric("wide",
            std::numeric_limits<double>::max())}));
    const auto overflowAssessment = Assess({overflowingDelta});
    assert(overflowAssessment.members[0].diagnostics ==
        std::vector<Diagnostic>{Diagnostic::NumericDeltaNonFinite});
    assert(overflowAssessment.members[0].comparisons.empty());
    assert(overflowAssessment.members[0].outcome == MemberOutcome::Inconsistent);

    assert(RecommendationCampaignOutcomeAssessmentLifecycleStateText(
               Lifecycle::NotTerminal) == "not_terminal");
    assert(RecommendationCampaignOutcomeAssessmentLifecycleStateText(
               Lifecycle::Succeeded) == "succeeded");
    assert(RecommendationCampaignOutcomeAssessmentLifecycleStateText(
               Lifecycle::Failed) == "failed");
    assert(RecommendationCampaignOutcomeAssessmentLifecycleStateText(
               Lifecycle::Cancelled) == "cancelled");
    assert(RecommendationCampaignOutcomeAssessmentLifecycleStateText(
               Lifecycle::Unknown) == "unknown");
    assert(RecommendationCampaignOutcomeAssessmentConsistencyStateText(
               Consistency::Consistent) == "consistent");
    assert(RecommendationCampaignOutcomeAssessmentConsistencyStateText(
               Consistency::Inconsistent) == "inconsistent");
    assert(RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
               MemberOutcome::SucceededMetricGap) == "succeeded_metric_gap");
    assert(RecommendationCampaignOutcomeAssessmentAggregateOutcomeText(
               AggregateOutcome::HasFailures) == "has_failures");

    return 0;
}
