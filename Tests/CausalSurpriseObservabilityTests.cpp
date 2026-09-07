#include "CausalSurpriseObservability.hpp"

#include "FeatureAblation.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace Observability = EA::CausalSurpriseObservability;
namespace Economic = EA::EconomicCalendar;

namespace
{

constexpr std::int64_t kBaseMicros = 1'700'000'000'000'000LL;

PriceTP Bar(std::int64_t index)
{
    return PriceTP{std::chrono::seconds{
        kBaseMicros / 1'000'000LL + index * 15 * 60}};
}

Economic::EconomicEvent ConsensusEvent(double forecast,
                                       double actual,
                                       std::int64_t availableBar,
                                       std::string provider = "fixture_consensus")
{
    Economic::EconomicEvent event;
    event.economicEventId = 10;
    event.currency = "USD";
    event.eventFamily = "CPI";
    event.eventTimestampUnixMicros = kBaseMicros + 15 * 60 * 1'000'000LL;
    event.eventTimestampUtc = "fixture";
    event.sourceAgency = "BLS";
    event.sourceUrl = "fixture";
    event.eventImportance = 3;
    event.historicalTimeConfidence = "exact";

    Economic::EconomicEventSelectedConsensus selected;
    selected.forecast.valueKind = "scalar";
    selected.forecast.canonicalValueLow = forecast;
    selected.forecast.unit = "percent";
    selected.forecast.scale = 1.0;
    selected.provider = std::move(provider);
    event.selectedConsensus = std::move(selected);

    event.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::provenFirstRelease;
    event.firstReleaseActualSelectionReason =
        "unique_authoritative_initial_at_earliest_source_publication";
    Economic::EconomicEventFirstReleaseActual firstRelease;
    firstRelease.actual.valueKind = "scalar";
    firstRelease.actual.canonicalValueLow = actual;
    firstRelease.actual.unit = "percent";
    firstRelease.actual.scale = 1.0;
    firstRelease.provenAvailableAtUnixMicros =
        kBaseMicros + availableBar * 15 * 60 * 1'000'000LL;
    firstRelease.sourceName = "BLS";
    firstRelease.sourceObservationId = "fixture-observation";
    firstRelease.evidenceKey = "fixture-evidence";
    event.firstReleaseActual = std::move(firstRelease);
    return event;
}

Economic::CausalSurpriseObservation ObservationAt(
    Economic::EconomicEvent event,
    std::int64_t barIndex)
{
    Economic::EconomicEventFeatureEngine engine{{std::move(event)}};
    Economic::CausalSurpriseObservation observation;
    for (std::int64_t index = 0; index <= barIndex; ++index)
    {
        (void)engine.AdvanceCompletedBar(Bar(index));
        observation = engine.LastCausalSurpriseObservation();
    }
    return observation;
}

Observability::ExperimentContext Context(long long experimentId)
{
    Observability::ExperimentContext context;
    context.experimentId = experimentId;
    context.symbol = "eurusdrmp";
    context.predictionHorizon = 4;
    context.trainStart = "2010-01-01 00:00:00+00";
    context.trainEnd = "2025-01-01 00:00:00+00";
    context.inferStart = "2025-01-01 00:00:00+00";
    context.inferEnd = "2026-01-01 00:00:00+00";
    context.modelInputWidth = static_cast<int>(
        EA::kCausalEconomicEventSurpriseModelInputWidth);
    context.modelInputSemanticLayoutVersion =
        EA::kModelInputSemanticLayoutVersion;
    context.featureWarmupScope = EA::FeatureWarmupScope::FullHistoryWarmup;
    context.donchian20Mode = Donchian20Mode::Enabled;
    context.donchianLookback = 20;
    return context;
}

Observability::Result EmptyResult(
    const Observability::ExperimentContext& context,
    Observability::Scope scope = Observability::Scope::train)
{
    const auto ranges = Observability::ResolveRanges(context, scope);
    std::vector<Observability::SegmentInput> segments;
    for (const auto& range : ranges)
        segments.push_back({range, {}, 0, {}});
    return Observability::Evaluate(context, scope, std::move(segments));
}

bool HasReason(const Observability::ParityResult& result,
               const std::string& reason)
{
    return std::find(result.reasons.begin(), result.reasons.end(), reason) !=
        result.reasons.end();
}

void TestAuthoritativeDispositionsAndBoundary()
{
    const auto positive = ObservationAt(ConsensusEvent(0.2, 0.5, 2), 2);
    assert(positive.disposition ==
           Economic::CausalSurpriseDisposition::available);
    assert(std::abs(positive.surprise - 0.03F) < 1e-7F);
    assert(positive.firstReleaseSource == "BLS");
    assert(positive.consensusSource == "fixture_consensus");
    assert(positive.eventFamily == "CPI");

    const auto negative = ObservationAt(ConsensusEvent(0.7, 0.2, 2), 2);
    assert(negative.disposition ==
           Economic::CausalSurpriseDisposition::available);
    assert(negative.surprise < 0.0F);

    const auto zero = ObservationAt(ConsensusEvent(0.2, 0.2, 2), 2);
    assert(zero.disposition ==
           Economic::CausalSurpriseDisposition::available);
    assert(zero.surprise == 0.0F);

    const auto before = ObservationAt(ConsensusEvent(0.2, 0.5, 2), 0);
    assert(before.disposition ==
           Economic::CausalSurpriseDisposition::notYetAvailable);
    assert(before.surprise == 0.0F);

    // The completed cutoff for Bar(1) is exactly Bar(2). Completed feature
    // rows represent [barStart, informationCutoff), so evidence published
    // exactly at that cutoff is unavailable until the following bar.
    const auto exactly = ObservationAt(ConsensusEvent(0.2, 0.5, 2), 1);
    const auto after = ObservationAt(ConsensusEvent(0.2, 0.5, 2), 2);
    assert(exactly.disposition ==
           Economic::CausalSurpriseDisposition::notYetAvailable);
    assert(exactly.surprise == 0.0F);
    assert(after.disposition ==
           Economic::CausalSurpriseDisposition::available);
    assert(std::abs(after.surprise - 0.03F) < 1e-7F);

    auto provenance = ConsensusEvent(0.2, 0.5, 2);
    provenance.firstReleaseActual.reset();
    provenance.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::
            provenanceUnavailable;
    assert(ObservationAt(provenance, 1).disposition ==
           Economic::CausalSurpriseDisposition::provenanceUnavailable);

    auto ambiguous = provenance;
    ambiguous.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::ambiguous;
    assert(ObservationAt(ambiguous, 1).disposition ==
           Economic::CausalSurpriseDisposition::ambiguous);

    auto repositoryNotYet = provenance;
    repositoryNotYet.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::notYetAvailable;
    assert(ObservationAt(repositoryNotYet, 1).disposition ==
           Economic::CausalSurpriseDisposition::notYetAvailable);

    auto missingConsensus = ConsensusEvent(0.2, 0.5, 2);
    missingConsensus.selectedConsensus.reset();
    assert(ObservationAt(missingConsensus, 2).disposition ==
           Economic::CausalSurpriseDisposition::missingForecast);

    auto incompatible = ConsensusEvent(0.2, 0.5, 2);
    incompatible.firstReleaseActual->actual.scale = 100.0;
    assert(ObservationAt(incompatible, 2).disposition ==
           Economic::CausalSurpriseDisposition::incompatible);

    Economic::EconomicEventFeatureEngine empty{
        std::vector<Economic::EconomicEvent>{}};
    const auto emptyValues = empty.AdvanceCompletedBar(Bar(0));
    assert(emptyValues.causalFirstReleaseSurpriseAvailable == 0.0F);
    assert(emptyValues.causalFirstReleaseSurprise == 0.0F);
    assert(empty.LastCausalSurpriseObservation().disposition ==
           Economic::CausalSurpriseDisposition::noRelevantEvent);
}

void TestCoveragePartitionsStatisticsAndClamp()
{
    Observability::Coverage coverage;
    const auto add = [&coverage](Economic::CausalSurpriseObservation value)
    {
        Observability::Observe(coverage, value);
    };
    const auto unavailable = [](Economic::CausalSurpriseDisposition value)
    {
        Economic::CausalSurpriseObservation observation;
        observation.disposition = value;
        return observation;
    };
    add(unavailable(Economic::CausalSurpriseDisposition::noRelevantEvent));
    add(unavailable(
        Economic::CausalSurpriseDisposition::provenanceUnavailable));
    add(unavailable(Economic::CausalSurpriseDisposition::ambiguous));
    add(unavailable(Economic::CausalSurpriseDisposition::notYetAvailable));
    add(unavailable(Economic::CausalSurpriseDisposition::missingForecast));
    add(unavailable(Economic::CausalSurpriseDisposition::incompatible));

    auto available = [](float surprise)
    {
        Economic::CausalSurpriseObservation value;
        value.disposition = Economic::CausalSurpriseDisposition::available;
        value.surprise = surprise;
        value.eventFamily = "CPI";
        value.firstReleaseSource = "BLS";
        value.consensusSource = "fixture_consensus";
        return value;
    };
    add(available(-2.0F));
    add(available(0.0F));
    add(available(4.0F));

    assert(coverage.totalFeatureRows == 9);
    assert(coverage.surpriseAvailableCount == 3);
    assert(Observability::SurpriseUnavailableCount(coverage) == 6);
    assert(Observability::SurpriseAvailableRate(coverage) == 1.0 / 3.0);
    assert(Observability::DispositionPartitionCount(coverage) == 9);
    assert(coverage.validZeroSurpriseCount == 1);
    assert(coverage.nonzeroSurpriseCount == 2);
    assert(coverage.negativeSurpriseCount == 1);
    assert(coverage.positiveSurpriseCount == 1);
    assert(coverage.negativeSurpriseCount +
               coverage.validZeroSurpriseCount +
               coverage.positiveSurpriseCount ==
           coverage.surpriseAvailableCount);
    assert(coverage.minimumSurprise == -2.0F);
    assert(coverage.maximumSurprise == 4.0F);
    assert(Observability::MeanSurprise(coverage) == 2.0 / 3.0);
    // Six unavailable placeholder zeros are excluded from valid-zero and
    // descriptive statistics.
    assert(coverage.validZeroSurpriseCount == 1);
    assert(coverage.firstReleaseSourceCounts.at("BLS") == 3);
    assert(coverage.consensusSourceCounts.at("fixture_consensus") == 3);
    assert(coverage.eventFamilyCounts.at("CPI") == 3);

    const auto upper = ObservationAt(ConsensusEvent(0.0, 200.0, 2), 2);
    const auto lower = ObservationAt(ConsensusEvent(0.0, -200.0, 2), 2);
    assert(upper.upperClamped && !upper.lowerClamped &&
           upper.surprise == 10.0F);
    assert(lower.lowerClamped && !lower.upperClamped &&
           lower.surprise == -10.0F);
    add(upper);
    add(lower);
    assert(coverage.lowerClampedCount == 1);
    assert(coverage.upperClampedCount == 1);
}

void TestWarmupDenominatorAndIdentity()
{
    auto context = Context(619);
    const auto ranges = Observability::ResolveRanges(
        context, Observability::Scope::train);
    std::vector<PriceTP> bars{Bar(0), Bar(1), Bar(2)};
    auto result = Observability::Evaluate(
        context,
        Observability::Scope::train,
        {{ranges.front(), bars, 1, {ConsensusEvent(0.2, 0.5, 2)}}});
    assert(result.sourceRowCount == 3);
    assert(result.warmupRowCount == 1);
    assert(result.coverage.totalFeatureRows == 2);
    assert(result.coverage.surpriseAvailableCount == 1);
    assert(result.experiment.modelInputWidth == 77);
    assert(result.experiment.modelInputSemanticLayoutVersion == 7);
    assert(!result.upstreamFeatureIdentity.empty());
    assert(!result.coverageIdentity.empty());
    assert(!result.diagnosticIdentity.empty());

    auto same = Observability::Evaluate(
        context,
        Observability::Scope::train,
        {{ranges.front(), bars, 1, {ConsensusEvent(0.2, 0.5, 2)}}});
    assert(result.upstreamFeatureIdentity == same.upstreamFeatureIdentity);
    assert(result.coverageIdentity == same.coverageIdentity);
    assert(result.diagnosticIdentity == same.diagnosticIdentity);

    auto historical = context;
    historical.experimentId = 500;
    historical.modelInputWidth = static_cast<int>(
        EA::kEconomicEventReleaseActualModelInputWidth);
    historical.modelInputSemanticLayoutVersion = 5;
    const auto historicalResult = EmptyResult(historical);
    assert(historicalResult.experiment.modelInputWidth == 75);
    assert(historicalResult.experiment.modelInputSemanticLayoutVersion == 5);
}

void TestControlledPairParity()
{
    auto control = Context(619);
    auto ablation = Context(620);
    ablation.featureAblationMask = std::string{
        EA::kCausalEconomicEventSurpriseAblationMaskText};
    const auto left = EmptyResult(control);
    const auto right = EmptyResult(ablation);
    const auto parity = Observability::CompareUpstream(left, right);
    assert(parity.comparable);
    assert(parity.coverageMatches);
    assert(left.upstreamFeatureIdentity == right.upstreamFeatureIdentity);
    assert(left.coverageIdentity == right.coverageIdentity);
    assert(left.attributionIdentity == right.attributionIdentity);
    assert(left.gapAttribution == right.gapAttribution);
    assert(left.diagnosticIdentity != right.diagnosticIdentity);

    auto symbol = ablation;
    symbol.symbol = "gbpusdrmp";
    assert(HasReason(
        Observability::CompareUpstream(left, EmptyResult(symbol)),
        "symbol_mismatch"));

    auto horizon = ablation;
    horizon.predictionHorizon = 8;
    assert(HasReason(
        Observability::CompareUpstream(left, EmptyResult(horizon)),
        "prediction_horizon_mismatch"));

    auto dates = ablation;
    dates.trainEnd = "2024-01-01 00:00:00+00";
    assert(HasReason(
        Observability::CompareUpstream(left, EmptyResult(dates)),
        "scope_or_date_range_mismatch"));

    auto width = ablation;
    width.modelInputWidth = 75;
    width.modelInputSemanticLayoutVersion = 5;
    const auto widthParity = Observability::CompareUpstream(
        left, EmptyResult(width));
    assert(HasReason(widthParity, "model_input_width_mismatch"));
    assert(HasReason(
        widthParity, "model_input_semantic_layout_version_mismatch"));

    auto unrelatedMask = ablation;
    unrelatedMask.featureAblationMask = "donchian_up";
    assert(HasReason(
        Observability::CompareUpstream(left, EmptyResult(unrelatedMask)),
        "unexpected_feature_ablation_mask_difference"));
}

void TestExactCompatibilityReasons()
{
    auto event = ConsensusEvent(0.2, 0.5, 2);
    const auto assess = [](const Economic::EconomicEvent& value)
    {
        return Economic::AssessCausalScalarSurpriseCompatibility(
            *value.selectedConsensus, *value.firstReleaseActual,
            value.eventFamily);
    };

    auto forecastRange = event;
    forecastRange.selectedConsensus->forecast.valueKind = "range";
    forecastRange.selectedConsensus->forecast.canonicalValueHigh = 0.3;
    assert(assess(forecastRange) == Economic::
        CausalSurpriseIncompatibilityReason::forecastNotScalar);

    auto actualRange = event;
    actualRange.firstReleaseActual->actual.valueKind = "range";
    actualRange.firstReleaseActual->actual.canonicalValueHigh = 0.6;
    assert(assess(actualRange) == Economic::
        CausalSurpriseIncompatibilityReason::actualNotScalar);

    auto unit = event;
    unit.firstReleaseActual->actual.unit = "count";
    assert(ObservationAt(unit, 2).incompatibilityReason == Economic::
        CausalSurpriseIncompatibilityReason::unitMismatch);

    auto scale = event;
    scale.firstReleaseActual->actual.scale = 100.0;
    assert(ObservationAt(scale, 2).incompatibilityReason == Economic::
        CausalSurpriseIncompatibilityReason::scaleMismatch);

    auto qualifier = event;
    qualifier.firstReleaseActual->actual.qualifier = "seasonally_adjusted";
    assert(ObservationAt(qualifier, 2).incompatibilityReason == Economic::
        CausalSurpriseIncompatibilityReason::qualifierMismatch);

    auto invalidForecast = event;
    invalidForecast.selectedConsensus->forecast.canonicalValueLow =
        std::numeric_limits<double>::quiet_NaN();
    assert(assess(invalidForecast) == Economic::
        CausalSurpriseIncompatibilityReason::invalidForecastValueShape);

    auto invalidActual = event;
    invalidActual.firstReleaseActual->actual.scale = 0.0;
    assert(assess(invalidActual) == Economic::
        CausalSurpriseIncompatibilityReason::invalidActualValueShape);
    bool invalidShapeFailedClosed = false;
    try
    {
        (void)ObservationAt(invalidActual, 2);
    }
    catch (const std::invalid_argument& error)
    {
        invalidShapeFailedClosed = std::string{error.what()}.find(
            "first_release_actual_pit_invalid") != std::string::npos;
    }
    assert(invalidShapeFailedClosed);

    auto unsupported = event;
    unsupported.eventFamily = "UNSUPPORTED";
    assert(assess(unsupported) == Economic::
        CausalSurpriseIncompatibilityReason::
            unsupportedNormalizationFamilyOrUnit);

    assert(std::string{Economic::CausalSurpriseIncompatibilityReasonText(
        Economic::CausalSurpriseIncompatibilityReason::scaleMismatch)} ==
        "scale_mismatch");
}

void TestGapAttributionReconciliationAndUniqueEvents()
{
    auto context = Context(619);
    const auto range = Observability::ResolveRanges(
        context, Observability::Scope::train).front();
    auto event = ConsensusEvent(0.2, 0.5, 2);
    event.firstReleaseActual.reset();
    event.firstReleaseActualState = Economic::
        EconomicEventFirstReleaseActualState::provenanceUnavailable;
    event.firstReleaseActualSelectionReason = "no_actual_observation";
    const std::vector<PriceTP> bars{Bar(0), Bar(1), Bar(2), Bar(3)};
    const auto result = Observability::Evaluate(
        context, Observability::Scope::train,
        {{range, bars, 1, {event}}});

    // Phase 6 consumes exactly the Phase-5 denominator after its prefix.
    assert(result.sourceRowCount - result.warmupRowCount ==
           result.coverage.totalFeatureRows);
    assert(result.coverage.totalFeatureRows == 3);
    assert(result.coverage.provenanceUnavailableCount == 3);
    assert(result.gapAttribution.provenanceReasonCounts.at(
        "no_actual_observation") == 3);
    assert(result.gapAttribution.familyCounts.at("CPI").totalRows == 3);
    assert(result.gapAttribution.agencyCounts.at("BLS").totalRows == 3);
    assert(result.gapAttribution.yearCounts.at("2023").totalRows == 3);
    assert(result.gapAttribution.reasonCounts.size() == 1);
    assert(result.gapAttribution.reasonCounts.front().affectedFeatureRows == 3);
    assert(result.gapAttribution.reasonCounts.front().affectedEventIds.size() ==
           1);
    assert(result.gapAttribution.reasonCounts.front().remediability ==
           Observability::RemediabilityClass::
               potentiallyRemediableDataGap);
    assert(!result.attributionIdentity.empty());

    const auto repeated = Observability::Evaluate(
        context, Observability::Scope::train,
        {{range, bars, 1, {event}}});
    assert(result.attributionIdentity == repeated.attributionIdentity);
    assert(result.attributionCanonical == repeated.attributionCanonical);
}

void TestAllTerminalGapStatesAndClassifications()
{
    auto context = Context(619);
    const auto range = Observability::ResolveRanges(
        context, Observability::Scope::train).front();
    const auto evaluate = [&](Economic::EconomicEvent event)
    {
        return Observability::Evaluate(
            context, Observability::Scope::train,
            {{range, {Bar(1)}, 0, {std::move(event)}}});
    };

    auto ambiguous = ConsensusEvent(0.2, 0.5, 2);
    ambiguous.firstReleaseActual.reset();
    ambiguous.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::ambiguous;
    ambiguous.firstReleaseActualSelectionReason =
        "conflicting_authoritative_initials_at_same_earliest_publication";
    const auto ambiguousResult = evaluate(ambiguous);
    assert(ambiguousResult.coverage.ambiguousCount == 1);
    assert(ambiguousResult.gapAttribution.reasonCounts.front().remediability ==
           Observability::RemediabilityClass::
               requiresManualProvenanceReview);

    auto unproved = ambiguous;
    unproved.firstReleaseActualState = Economic::
        EconomicEventFirstReleaseActualState::provenanceUnavailable;
    unproved.firstReleaseActualSelectionReason =
        "no_authoritative_initial_with_exact_source_publication";
    const auto unprovedResult = evaluate(unproved);
    assert(unprovedResult.gapAttribution.provenanceReasonCounts.at(
        "no_authoritative_initial_with_exact_source_publication") == 1);
    assert(unprovedResult.gapAttribution.reasonCounts.front().rawReason ==
           "no_authoritative_initial_with_exact_source_publication");
    assert(unprovedResult.gapAttribution.reasonCounts.front().remediability ==
           Observability::RemediabilityClass::
               requiresManualProvenanceReview);

    auto notYet = ConsensusEvent(0.2, 0.5, 2);
    notYet.firstReleaseActual.reset();
    notYet.firstReleaseActualState =
        Economic::EconomicEventFirstReleaseActualState::notYetAvailable;
    const auto notYetResult = evaluate(notYet);
    assert(notYetResult.coverage.notYetAvailableCount == 1);
    assert(notYetResult.gapAttribution.reasonCounts.front().remediability ==
           Observability::RemediabilityClass::expectedByContract);

    auto missing = ConsensusEvent(0.2, 0.5, 1);
    missing.selectedConsensus.reset();
    const auto missingResult = evaluate(missing);
    assert(missingResult.coverage.missingConsensusCount == 1);
    assert(missingResult.gapAttribution.missingConsensusReasonCounts.at(
        "no_selected_consensus_persisted") == 1);
    assert(missingResult.gapAttribution.familyCounts.at("CPI")
               .dispositionCounts.at("missing_consensus") == 1);
    assert(missingResult.gapAttribution.agencyCounts.at("BLS")
               .dispositionCounts.at("missing_consensus") == 1);
    assert(missingResult.gapAttribution.yearCounts.at("2023")
               .dispositionCounts.at("missing_consensus") == 1);

    auto incompatible = ConsensusEvent(0.2, 0.5, 1);
    incompatible.firstReleaseActual->actual.scale = 100.0;
    const auto incompatibleResult = evaluate(incompatible);
    assert(incompatibleResult.coverage.incompatibleCount == 1);
    assert(incompatibleResult.gapAttribution.incompatibilityReasonCounts.at(
        "scale_mismatch") == 1);

    const auto noEvent = Observability::Evaluate(
        context, Observability::Scope::train,
        {{range, {Bar(0), Bar(1)}, 0, {}}});
    assert(noEvent.coverage.noRelevantEventCount == 2);
    assert(noEvent.gapAttribution.familyCounts.at("NONE").totalRows == 2);
    assert(noEvent.gapAttribution.firstNoRelevantBarUnixSeconds);
    assert(noEvent.gapAttribution.lastNoRelevantBarUnixSeconds);
}

void TestPriorityOrderingAndControlledAttributionParity()
{
    auto control = Context(619);
    auto ablation = Context(620);
    ablation.featureAblationMask = std::string{
        EA::kCausalEconomicEventSurpriseAblationMaskText};
    const auto ranges = Observability::ResolveRanges(
        control, Observability::Scope::combined);

    auto provenance = ConsensusEvent(0.2, 0.5, 1);
    provenance.firstReleaseActual.reset();
    provenance.firstReleaseActualState = Economic::
        EconomicEventFirstReleaseActualState::provenanceUnavailable;
    provenance.firstReleaseActualSelectionReason = "no_actual_observation";
    auto missing = ConsensusEvent(0.2, 0.5, 1);
    missing.selectedConsensus.reset();

    const auto evaluate = [&](const Observability::ExperimentContext& value)
    {
        return Observability::Evaluate(
            value, Observability::Scope::combined,
            {{ranges[0], {Bar(0)}, 0, {provenance}},
             {ranges[1], {Bar(1)}, 0, {missing}}});
    };
    const auto left = evaluate(control);
    const auto right = evaluate(ablation);
    assert(left.gapAttribution.priorities.size() == 2);
    assert(left.gapAttribution.priorities[0].affectedFeatureRows == 1);
    assert(left.gapAttribution.priorities[1].affectedFeatureRows == 1);
    assert(left.gapAttribution.priorities[0].rawReason ==
           "no_actual_observation");
    assert(left.gapAttribution.priorities[1].rawReason ==
           "no_selected_consensus_persisted");
    assert(left.attributionIdentity == right.attributionIdentity);
    assert(Observability::CompareUpstream(left, right).coverageMatches);
    // Scheduler priority is intentionally absent from ExperimentContext and
    // therefore cannot perturb either upstream identity.
}

} // namespace

int main()
{
    TestAuthoritativeDispositionsAndBoundary();
    TestCoveragePartitionsStatisticsAndClamp();
    TestWarmupDenominatorAndIdentity();
    TestControlledPairParity();
    TestExactCompatibilityReasons();
    TestGapAttributionReconciliationAndUniqueEvents();
    TestAllTerminalGapStatesAndClassifications();
    TestPriorityOrderingAndControlledAttributionParity();
    std::cout << "Causal surprise observability tests passed\n";
    return 0;
}
