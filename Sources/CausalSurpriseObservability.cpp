#include "CausalSurpriseObservability.hpp"

#include "EconomicEventFeatureLayout.hpp"
#include "FeatureAblation.hpp"
#include "FeatureLayout.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace EA::CausalSurpriseObservability
{
namespace
{

std::string OptionalInteger(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string Number(long double value)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(21) << value;
    return output.str();
}

std::string OptionalFloat(const std::optional<float>& value)
{
    if (!value) return "NULL";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(9) << *value;
    return output.str();
}

void AppendFramed(std::ostringstream& output,
                  std::string_view name,
                  std::string_view value)
{
    output << ';' << name << '=' << value.size() << ':' << value;
}

void AppendMap(std::ostringstream& output,
               std::string_view name,
               const std::map<std::string, std::size_t>& values)
{
    output << ';' << name << "_count=" << values.size();
    std::size_t index = 0;
    for (const auto& [key, count] : values)
    {
        output << ';' << name << '[' << index++ << "]="
               << key.size() << ':' << key << ':' << count;
    }
}

std::size_t MapCount(const auto& values)
{
    std::size_t total = 0;
    for (const auto& [key, count] : values)
    {
        (void)key;
        total += count;
    }
    return total;
}

std::string FeatureYear(PriceTP barStart)
{
    const std::chrono::year_month_day date{
        std::chrono::floor<std::chrono::days>(barStart)};
    return std::to_string(static_cast<int>(date.year()));
}

std::string RawGapReason(
    const EconomicCalendar::CausalSurpriseObservation& observation)
{
    using Disposition = EconomicCalendar::CausalSurpriseDisposition;
    switch (observation.disposition)
    {
        case Disposition::noRelevantEvent:
            return "no_relevant_event_before_information_cutoff";
        case Disposition::provenanceUnavailable:
        case Disposition::ambiguous:
        case Disposition::notYetAvailable:
            return observation.firstReleaseSelectionReason.empty()
                ? "selection_reason_unavailable"
                : observation.firstReleaseSelectionReason;
        case Disposition::missingForecast:
            return "no_selected_consensus_persisted";
        case Disposition::incompatible:
            return EconomicCalendar::
                CausalSurpriseIncompatibilityReasonText(
                    observation.incompatibilityReason);
        case Disposition::available:
            return "available";
    }
    return "unknown";
}

RemediabilityClass Classify(
    EconomicCalendar::CausalSurpriseDisposition disposition,
    std::string_view rawReason)
{
    using Disposition = EconomicCalendar::CausalSurpriseDisposition;
    switch (disposition)
    {
        case Disposition::noRelevantEvent:
        case Disposition::notYetAvailable:
            return RemediabilityClass::expectedByContract;
        case Disposition::provenanceUnavailable:
            return rawReason == "no_actual_observation"
                ? RemediabilityClass::potentiallyRemediableDataGap
                : RemediabilityClass::requiresManualProvenanceReview;
        case Disposition::ambiguous:
            return RemediabilityClass::requiresManualProvenanceReview;
        case Disposition::missingForecast:
            return RemediabilityClass::potentiallyRemediableDataGap;
        case Disposition::incompatible:
            return RemediabilityClass::unsupportedSemantics;
        case Disposition::available:
            return RemediabilityClass::expectedByContract;
    }
    return RemediabilityClass::requiresManualProvenanceReview;
}

void AddAggregate(
    AttributionAggregate& aggregate,
    const EconomicCalendar::CausalSurpriseObservation& observation)
{
    ++aggregate.totalRows;
    const std::string disposition = DispositionText(observation.disposition);
    ++aggregate.dispositionCounts[disposition];
    if (observation.disposition ==
        EconomicCalendar::CausalSurpriseDisposition::available)
    {
        ++aggregate.availableRows;
    }
    else
    {
        ++aggregate.unavailableRows;
        if (observation.economicEventId > 0)
            aggregate.affectedEventIds.insert(observation.economicEventId);
    }
}

void ObserveAttribution(
    GapAttribution& attribution,
    const EconomicCalendar::CausalSurpriseObservation& observation,
    PriceTP barStart)
{
    const std::string family = observation.eventFamily.empty()
        ? "NONE" : observation.eventFamily;
    const std::string agency = observation.sourceAgency.empty()
        ? "NONE" : observation.sourceAgency;
    const std::string year = FeatureYear(barStart);
    AddAggregate(attribution.familyCounts[family], observation);
    AddAggregate(attribution.agencyCounts[agency], observation);
    AddAggregate(attribution.yearCounts[year], observation);

    using Disposition = EconomicCalendar::CausalSurpriseDisposition;
    if (observation.disposition == Disposition::available) return;

    const std::string rawReason = RawGapReason(observation);
    const RemediabilityClass remediability =
        Classify(observation.disposition, rawReason);
    if (observation.disposition == Disposition::provenanceUnavailable)
        ++attribution.provenanceReasonCounts[rawReason];
    if (observation.disposition == Disposition::missingForecast)
        ++attribution.missingConsensusReasonCounts[rawReason];
    if (observation.disposition == Disposition::incompatible)
        ++attribution.incompatibilityReasonCounts[rawReason];

    const std::string disposition = DispositionText(observation.disposition);
    auto reason = std::find_if(
        attribution.reasonCounts.begin(), attribution.reasonCounts.end(),
        [&](const GapReasonSummary& value)
        {
            return value.disposition == disposition &&
                value.rawReason == rawReason &&
                value.remediability == remediability;
        });
    if (reason == attribution.reasonCounts.end())
    {
        attribution.reasonCounts.push_back(
            {disposition, rawReason, remediability, 0, {}});
        reason = std::prev(attribution.reasonCounts.end());
    }
    ++reason->affectedFeatureRows;
    if (observation.economicEventId > 0)
        reason->affectedEventIds.insert(observation.economicEventId);

    auto priority = std::find_if(
        attribution.priorities.begin(), attribution.priorities.end(),
        [&](const GapPriorityEntry& value)
        {
            return value.disposition == disposition &&
                value.rawReason == rawReason &&
                value.eventFamily == family &&
                value.sourceAgency == agency &&
                value.remediability == remediability;
        });
    if (priority == attribution.priorities.end())
    {
        attribution.priorities.push_back(
            {disposition, rawReason, family, agency, remediability, 0, {}});
        priority = std::prev(attribution.priorities.end());
    }
    ++priority->affectedFeatureRows;
    if (observation.economicEventId > 0)
        priority->affectedEventIds.insert(observation.economicEventId);

    if (observation.disposition == Disposition::noRelevantEvent)
    {
        const std::int64_t seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                barStart.time_since_epoch()).count();
        if (!attribution.firstNoRelevantBarUnixSeconds)
            attribution.firstNoRelevantBarUnixSeconds = seconds;
        attribution.lastNoRelevantBarUnixSeconds = seconds;
    }
}

void MergeAggregate(AttributionAggregate& destination,
                    const AttributionAggregate& source)
{
    destination.totalRows += source.totalRows;
    destination.availableRows += source.availableRows;
    destination.unavailableRows += source.unavailableRows;
    for (const auto& [key, count] : source.dispositionCounts)
        destination.dispositionCounts[key] += count;
    destination.affectedEventIds.insert(
        source.affectedEventIds.begin(), source.affectedEventIds.end());
}

void MergeAttribution(GapAttribution& destination,
                      const GapAttribution& source)
{
    const auto mergeMap = [](auto& target, const auto& values)
    {
        for (const auto& [key, count] : values) target[key] += count;
    };
    mergeMap(destination.provenanceReasonCounts,
             source.provenanceReasonCounts);
    mergeMap(destination.missingConsensusReasonCounts,
             source.missingConsensusReasonCounts);
    mergeMap(destination.incompatibilityReasonCounts,
             source.incompatibilityReasonCounts);
    const auto mergeAggregates = [](auto& target, const auto& values)
    {
        for (const auto& [key, aggregate] : values)
            MergeAggregate(target[key], aggregate);
    };
    mergeAggregates(destination.familyCounts, source.familyCounts);
    mergeAggregates(destination.agencyCounts, source.agencyCounts);
    mergeAggregates(destination.yearCounts, source.yearCounts);
    for (const GapReasonSummary& value : source.reasonCounts)
    {
        auto existing = std::find_if(
            destination.reasonCounts.begin(), destination.reasonCounts.end(),
            [&](const GapReasonSummary& candidate)
            {
                return candidate.disposition == value.disposition &&
                    candidate.rawReason == value.rawReason &&
                    candidate.remediability == value.remediability;
            });
        if (existing == destination.reasonCounts.end())
            destination.reasonCounts.push_back(value);
        else
        {
            existing->affectedFeatureRows += value.affectedFeatureRows;
            existing->affectedEventIds.insert(
                value.affectedEventIds.begin(), value.affectedEventIds.end());
        }
    }
    for (const GapPriorityEntry& value : source.priorities)
    {
        auto existing = std::find_if(
            destination.priorities.begin(), destination.priorities.end(),
            [&](const GapPriorityEntry& candidate)
            {
                return candidate.disposition == value.disposition &&
                    candidate.rawReason == value.rawReason &&
                    candidate.eventFamily == value.eventFamily &&
                    candidate.sourceAgency == value.sourceAgency &&
                    candidate.remediability == value.remediability;
            });
        if (existing == destination.priorities.end())
            destination.priorities.push_back(value);
        else
        {
            existing->affectedFeatureRows += value.affectedFeatureRows;
            existing->affectedEventIds.insert(
                value.affectedEventIds.begin(), value.affectedEventIds.end());
        }
    }
    if (source.firstNoRelevantBarUnixSeconds)
    {
        if (!destination.firstNoRelevantBarUnixSeconds ||
            *source.firstNoRelevantBarUnixSeconds <
                *destination.firstNoRelevantBarUnixSeconds)
        {
            destination.firstNoRelevantBarUnixSeconds =
                source.firstNoRelevantBarUnixSeconds;
        }
    }
    if (source.lastNoRelevantBarUnixSeconds)
    {
        if (!destination.lastNoRelevantBarUnixSeconds ||
            *source.lastNoRelevantBarUnixSeconds >
                *destination.lastNoRelevantBarUnixSeconds)
        {
            destination.lastNoRelevantBarUnixSeconds =
                source.lastNoRelevantBarUnixSeconds;
        }
    }
}

void SortAttribution(GapAttribution& attribution)
{
    std::sort(
        attribution.reasonCounts.begin(), attribution.reasonCounts.end(),
        [](const GapReasonSummary& left, const GapReasonSummary& right)
        {
            return std::tie(left.disposition, left.rawReason,
                            left.remediability) <
                std::tie(right.disposition, right.rawReason,
                         right.remediability);
        });
    std::sort(
        attribution.priorities.begin(), attribution.priorities.end(),
        [](const GapPriorityEntry& left, const GapPriorityEntry& right)
        {
            if (left.affectedFeatureRows != right.affectedFeatureRows)
                return left.affectedFeatureRows > right.affectedFeatureRows;
            return std::tie(left.rawReason, left.eventFamily,
                            left.sourceAgency, left.disposition,
                            left.remediability) <
                std::tie(right.rawReason, right.eventFamily,
                         right.sourceAgency, right.disposition,
                         right.remediability);
        });
}

std::string RangesCanonical(const std::vector<DateRange>& ranges)
{
    std::ostringstream output;
    output << "range_count=" << ranges.size();
    for (std::size_t index = 0; index < ranges.size(); ++index)
    {
        AppendFramed(output, "range_role_" + std::to_string(index),
                     ranges[index].role);
        AppendFramed(output, "range_start_" + std::to_string(index),
                     ranges[index].start);
        AppendFramed(output, "range_end_" + std::to_string(index),
                     ranges[index].end);
    }
    return output.str();
}

std::string UpstreamCanonicalText(
    const ExperimentContext& experiment,
    Scope scope,
    const std::vector<DateRange>& ranges)
{
    std::ostringstream output;
    output << "causal_surprise_upstream_feature_identity_v1";
    AppendFramed(output, "symbol", experiment.symbol);
    output << ";prediction_horizon=" << experiment.predictionHorizon;
    AppendFramed(output, "scope", ScopeText(scope));
    AppendFramed(output, "ranges", RangesCanonical(ranges));
    AppendFramed(output, "feature_warmup_scope",
                 FeatureWarmupScopeText(experiment.featureWarmupScope));
    AppendFramed(output, "donchian20_mode",
                 Donchian20ModeText(experiment.donchian20Mode));
    output << ";donchian_lookback=" << experiment.donchianLookback
           << ";model_input_width="
           << OptionalInteger(experiment.modelInputWidth)
           << ";model_input_semantic_layout_version="
           << OptionalInteger(
                  experiment.modelInputSemanticLayoutVersion)
           << ";physical_tensor_feature_count=" << feature_size
           << ";availability_tensor_column="
           << causalFirstReleaseSurpriseAvailableCol
           << ";surprise_tensor_column="
           << causalFirstReleaseSurpriseCol;
    AppendFramed(output, "availability_feature_name",
                 EconomicCalendar::kEconomicEventFeatureNames[
                     static_cast<std::size_t>(EconomicCalendar::
                         EconomicEventFeatureIndex::
                             causalFirstReleaseSurpriseAvailable)]);
    AppendFramed(output, "surprise_feature_name",
                 EconomicCalendar::kEconomicEventFeatureNames[
                     static_cast<std::size_t>(EconomicCalendar::
                         EconomicEventFeatureIndex::
                             causalFirstReleaseSurprise)]);
    AppendFramed(output, "first_release_pit_contract",
                 kFirstReleasePitContract);
    AppendFramed(output, "normalization_contract",
                 kNormalizationContract);
    AppendFramed(output, "repository_selection_contract",
                 kRepositorySelectionContract);
    output << ";diagnostic_semantic_version="
           << kDiagnosticSemanticVersion;
    return output.str();
}

void Merge(Coverage& destination, const Coverage& source)
{
    destination.totalFeatureRows += source.totalFeatureRows;
    destination.noRelevantEventCount += source.noRelevantEventCount;
    destination.provenanceUnavailableCount +=
        source.provenanceUnavailableCount;
    destination.ambiguousCount += source.ambiguousCount;
    destination.notYetAvailableCount += source.notYetAvailableCount;
    destination.missingConsensusCount += source.missingConsensusCount;
    destination.incompatibleCount += source.incompatibleCount;
    destination.surpriseAvailableCount += source.surpriseAvailableCount;
    destination.validZeroSurpriseCount += source.validZeroSurpriseCount;
    destination.nonzeroSurpriseCount += source.nonzeroSurpriseCount;
    destination.negativeSurpriseCount += source.negativeSurpriseCount;
    destination.positiveSurpriseCount += source.positiveSurpriseCount;
    destination.lowerClampedCount += source.lowerClampedCount;
    destination.upperClampedCount += source.upperClampedCount;
    destination.surpriseSum += source.surpriseSum;
    if (source.minimumSurprise)
    {
        destination.minimumSurprise = destination.minimumSurprise
            ? std::min(*destination.minimumSurprise,
                       *source.minimumSurprise)
            : source.minimumSurprise;
        destination.maximumSurprise = destination.maximumSurprise
            ? std::max(*destination.maximumSurprise,
                       *source.maximumSurprise)
            : source.maximumSurprise;
    }
    const auto mergeMap = [](auto& target, const auto& values)
    {
        for (const auto& [key, count] : values)
            target[key] += count;
    };
    mergeMap(destination.firstReleaseSourceCounts,
             source.firstReleaseSourceCounts);
    mergeMap(destination.consensusSourceCounts,
             source.consensusSourceCounts);
    mergeMap(destination.eventFamilyCounts, source.eventFamilyCounts);
}

struct SegmentEvaluation
{
    Coverage coverage;
    GapAttribution attribution;
};

SegmentEvaluation EvaluateSegment(SegmentInput segment)
{
    if (segment.warmupRowCount > segment.sourceBarStarts.size())
        throw std::invalid_argument(
            "causal_surprise_observability_warmup_exceeds_source_rows");

    EconomicCalendar::EconomicEventFeatureEngine engine{
        std::move(segment.economicEvents)};
    SegmentEvaluation evaluation;
    for (std::size_t index = 0;
         index < segment.sourceBarStarts.size(); ++index)
    {
        (void)engine.AdvanceCompletedBar(segment.sourceBarStarts[index]);
        if (index < segment.warmupRowCount) continue;
        const auto& observation = engine.LastCausalSurpriseObservation();
        Observe(evaluation.coverage, observation);
        ObserveAttribution(
            evaluation.attribution, observation,
            segment.sourceBarStarts[index]);
    }
    return evaluation;
}

void AddReason(ParityResult& result, std::string reason)
{
    result.reasons.push_back(std::move(reason));
}

} // namespace

const char* RemediabilityClassText(RemediabilityClass value) noexcept
{
    switch (value)
    {
        case RemediabilityClass::expectedByContract:
            return "expected_by_contract";
        case RemediabilityClass::potentiallyRemediableDataGap:
            return "potentially_remediable_data_gap";
        case RemediabilityClass::requiresManualProvenanceReview:
            return "requires_manual_provenance_review";
        case RemediabilityClass::unsupportedSemantics:
            return "unsupported_semantics";
    }
    return "unknown";
}

const char* DispositionText(
    EconomicCalendar::CausalSurpriseDisposition value) noexcept
{
    using Disposition = EconomicCalendar::CausalSurpriseDisposition;
    switch (value)
    {
        case Disposition::noRelevantEvent: return "no_relevant_event";
        case Disposition::provenanceUnavailable:
            return "provenance_unavailable";
        case Disposition::ambiguous: return "ambiguous";
        case Disposition::notYetAvailable: return "not_yet_available";
        case Disposition::missingForecast: return "missing_consensus";
        case Disposition::incompatible: return "incompatible";
        case Disposition::available: return "available";
    }
    return "unknown";
}

const char* ScopeText(Scope scope)
{
    switch (scope)
    {
        case Scope::train: return "train";
        case Scope::infer: return "infer";
        case Scope::combined: return "combined";
    }
    throw std::invalid_argument("unsupported_causal_surprise_scope");
}

Scope ParseScope(const std::string& text)
{
    if (text == "train") return Scope::train;
    if (text == "infer") return Scope::infer;
    if (text == "combined") return Scope::combined;
    throw std::invalid_argument(
        "invalid causal-surprise scope value; expected train, infer, or "
        "combined");
}

std::vector<DateRange> ResolveRanges(
    const ExperimentContext& experiment,
    Scope scope)
{
    if (experiment.experimentId <= 0 || experiment.symbol.empty() ||
        experiment.predictionHorizon <= 0 ||
        experiment.trainStart.empty() || experiment.trainEnd.empty())
    {
        throw std::invalid_argument(
            "causal_surprise_observability_invalid_experiment_context");
    }
    if (experiment.trainStart >= experiment.trainEnd)
        throw std::invalid_argument(
            "causal_surprise_observability_invalid_train_range");

    const DateRange train{"train", experiment.trainStart,
                          experiment.trainEnd};
    if (scope == Scope::train) return {train};

    if (!experiment.inferStart || !experiment.inferEnd)
        throw std::invalid_argument(
            "causal_surprise_observability_infer_range_unavailable");
    if (*experiment.inferStart >= *experiment.inferEnd)
        throw std::invalid_argument(
            "causal_surprise_observability_invalid_infer_range");
    const DateRange infer{"infer", *experiment.inferStart,
                          *experiment.inferEnd};
    if (scope == Scope::infer) return {infer};
    return {train, infer};
}

void Observe(
    Coverage& coverage,
    const EconomicCalendar::CausalSurpriseObservation& observation)
{
    ++coverage.totalFeatureRows;
    using Disposition = EconomicCalendar::CausalSurpriseDisposition;
    switch (observation.disposition)
    {
        case Disposition::noRelevantEvent:
            ++coverage.noRelevantEventCount;
            return;
        case Disposition::provenanceUnavailable:
            ++coverage.provenanceUnavailableCount;
            return;
        case Disposition::ambiguous:
            ++coverage.ambiguousCount;
            return;
        case Disposition::notYetAvailable:
            ++coverage.notYetAvailableCount;
            return;
        case Disposition::missingForecast:
            ++coverage.missingConsensusCount;
            return;
        case Disposition::incompatible:
            ++coverage.incompatibleCount;
            return;
        case Disposition::available:
            break;
    }

    ++coverage.surpriseAvailableCount;
    const float surprise = observation.surprise;
    coverage.surpriseSum += static_cast<long double>(surprise);
    coverage.minimumSurprise = coverage.minimumSurprise
        ? std::min(*coverage.minimumSurprise, surprise)
        : std::optional<float>{surprise};
    coverage.maximumSurprise = coverage.maximumSurprise
        ? std::max(*coverage.maximumSurprise, surprise)
        : std::optional<float>{surprise};
    if (surprise < 0.0F)
        ++coverage.negativeSurpriseCount;
    else if (surprise > 0.0F)
        ++coverage.positiveSurpriseCount;
    else
        ++coverage.validZeroSurpriseCount;
    coverage.nonzeroSurpriseCount =
        coverage.negativeSurpriseCount + coverage.positiveSurpriseCount;
    if (observation.lowerClamped) ++coverage.lowerClampedCount;
    if (observation.upperClamped) ++coverage.upperClampedCount;
    if (observation.firstReleaseSource.empty() ||
        observation.consensusSource.empty() ||
        observation.eventFamily.empty())
    {
        throw std::logic_error(
            "causal_surprise_observability_available_source_missing");
    }
    ++coverage.firstReleaseSourceCounts[observation.firstReleaseSource];
    ++coverage.consensusSourceCounts[observation.consensusSource];
    ++coverage.eventFamilyCounts[observation.eventFamily];
}

std::size_t DispositionPartitionCount(const Coverage& coverage) noexcept
{
    return coverage.noRelevantEventCount +
        coverage.provenanceUnavailableCount + coverage.ambiguousCount +
        coverage.notYetAvailableCount + coverage.missingConsensusCount +
        coverage.incompatibleCount + coverage.surpriseAvailableCount;
}

std::size_t SurpriseUnavailableCount(const Coverage& coverage) noexcept
{
    return coverage.totalFeatureRows - coverage.surpriseAvailableCount;
}

double SurpriseAvailableRate(const Coverage& coverage) noexcept
{
    return coverage.totalFeatureRows == 0
        ? 0.0
        : static_cast<double>(coverage.surpriseAvailableCount) /
              static_cast<double>(coverage.totalFeatureRows);
}

std::optional<double> MeanSurprise(const Coverage& coverage) noexcept
{
    if (coverage.surpriseAvailableCount == 0) return std::nullopt;
    return static_cast<double>(coverage.surpriseSum /
        static_cast<long double>(coverage.surpriseAvailableCount));
}

std::string CoverageCanonicalText(const Coverage& coverage)
{
    std::ostringstream output;
    output << "causal_surprise_coverage_v1"
           << ";total_feature_rows=" << coverage.totalFeatureRows
           << ";no_relevant_event=" << coverage.noRelevantEventCount
           << ";provenance_unavailable="
           << coverage.provenanceUnavailableCount
           << ";ambiguous=" << coverage.ambiguousCount
           << ";not_yet_available=" << coverage.notYetAvailableCount
           << ";missing_consensus=" << coverage.missingConsensusCount
           << ";incompatible=" << coverage.incompatibleCount
           << ";surprise_available=" << coverage.surpriseAvailableCount
           << ";valid_zero=" << coverage.validZeroSurpriseCount
           << ";nonzero=" << coverage.nonzeroSurpriseCount
           << ";negative=" << coverage.negativeSurpriseCount
           << ";positive=" << coverage.positiveSurpriseCount
           << ";lower_clamped=" << coverage.lowerClampedCount
           << ";upper_clamped=" << coverage.upperClampedCount
           << ";surprise_sum=" << Number(coverage.surpriseSum)
           << ";minimum=" << OptionalFloat(coverage.minimumSurprise)
           << ";maximum=" << OptionalFloat(coverage.maximumSurprise);
    AppendMap(output, "first_release_source",
              coverage.firstReleaseSourceCounts);
    AppendMap(output, "consensus_source", coverage.consensusSourceCounts);
    AppendMap(output, "event_family", coverage.eventFamilyCounts);
    return output.str();
}

std::string GapAttributionCanonicalText(const GapAttribution& attribution)
{
    const auto appendSet = [](std::ostringstream& output,
                              const std::set<long long>& values)
    {
        output << values.size();
        for (const long long value : values) output << ':' << value;
    };
    const auto appendAggregates = [&](std::ostringstream& output,
                                      std::string_view name,
                                      const auto& values)
    {
        output << ';' << name << "_count=" << values.size();
        std::size_t index = 0;
        for (const auto& [key, aggregate] : values)
        {
            output << ';' << name << '[' << index++ << "]="
                   << key.size() << ':' << key
                   << ':' << aggregate.totalRows
                   << ':' << aggregate.availableRows
                   << ':' << aggregate.unavailableRows
                   << ":dispositions=" << aggregate.dispositionCounts.size();
            for (const auto& [disposition, count] :
                 aggregate.dispositionCounts)
            {
                output << ':' << disposition.size() << ':' << disposition
                       << ':' << count;
            }
            output << ":events=";
            appendSet(output, aggregate.affectedEventIds);
        }
    };

    std::ostringstream output;
    output << "causal_surprise_gap_attribution_v1";
    AppendMap(output, "provenance_reason",
              attribution.provenanceReasonCounts);
    AppendMap(output, "missing_consensus_reason",
              attribution.missingConsensusReasonCounts);
    AppendMap(output, "incompatibility_reason",
              attribution.incompatibilityReasonCounts);
    appendAggregates(output, "family", attribution.familyCounts);
    appendAggregates(output, "agency", attribution.agencyCounts);
    appendAggregates(output, "year", attribution.yearCounts);
    output << ";reason_count=" << attribution.reasonCounts.size();
    for (std::size_t index = 0;
         index < attribution.reasonCounts.size(); ++index)
    {
        const auto& value = attribution.reasonCounts[index];
        output << ";reason[" << index << "]="
               << value.disposition.size() << ':' << value.disposition
               << ':' << value.rawReason.size() << ':' << value.rawReason
               << ':' << RemediabilityClassText(value.remediability)
               << ':' << value.affectedFeatureRows << ":events=";
        appendSet(output, value.affectedEventIds);
    }
    output << ";priority_count=" << attribution.priorities.size();
    for (std::size_t index = 0;
         index < attribution.priorities.size(); ++index)
    {
        const auto& value = attribution.priorities[index];
        output << ";priority[" << index << "]="
               << value.disposition.size() << ':' << value.disposition
               << ':' << value.rawReason.size() << ':' << value.rawReason
               << ':' << value.eventFamily.size() << ':' << value.eventFamily
               << ':' << value.sourceAgency.size() << ':' << value.sourceAgency
               << ':' << RemediabilityClassText(value.remediability)
               << ':' << value.affectedFeatureRows << ":events=";
        appendSet(output, value.affectedEventIds);
    }
    output << ";first_no_relevant_bar_unix_seconds="
           << (attribution.firstNoRelevantBarUnixSeconds
                   ? std::to_string(
                         *attribution.firstNoRelevantBarUnixSeconds)
                   : "NULL")
           << ";last_no_relevant_bar_unix_seconds="
           << (attribution.lastNoRelevantBarUnixSeconds
                   ? std::to_string(
                         *attribution.lastNoRelevantBarUnixSeconds)
                   : "NULL");
    return output.str();
}

Result Evaluate(const ExperimentContext& experiment,
                Scope scope,
                std::vector<SegmentInput> segments)
{
    const std::vector<DateRange> ranges = ResolveRanges(experiment, scope);
    if (segments.size() != ranges.size())
        throw std::invalid_argument(
            "causal_surprise_observability_scope_segment_count_mismatch");

    Result result;
    result.experiment = experiment;
    result.scope = scope;
    result.evaluatedRanges = ranges;
    for (std::size_t index = 0; index < segments.size(); ++index)
    {
        if (segments[index].range != ranges[index])
            throw std::invalid_argument(
                "causal_surprise_observability_scope_segment_range_mismatch");
        result.sourceRowCount += segments[index].sourceBarStarts.size();
        result.warmupRowCount += segments[index].warmupRowCount;
        SegmentEvaluation evaluation =
            EvaluateSegment(std::move(segments[index]));
        Merge(result.coverage, evaluation.coverage);
        MergeAttribution(result.gapAttribution, evaluation.attribution);
    }
    SortAttribution(result.gapAttribution);
    const auto aggregateRows = [](const auto& values)
    {
        std::size_t total = 0;
        for (const auto& [key, aggregate] : values)
        {
            (void)key;
            if (aggregate.totalRows !=
                    aggregate.availableRows + aggregate.unavailableRows ||
                MapCount(aggregate.dispositionCounts) != aggregate.totalRows)
            {
                throw std::logic_error(
                    "causal_surprise_gap_aggregate_invariant_failed");
            }
            total += aggregate.totalRows;
        }
        return total;
    };
    const auto vectorRows = [](const auto& values)
    {
        std::size_t total = 0;
        for (const auto& value : values)
            total += value.affectedFeatureRows;
        return total;
    };
    if (result.sourceRowCount - result.warmupRowCount !=
            result.coverage.totalFeatureRows ||
        DispositionPartitionCount(result.coverage) !=
            result.coverage.totalFeatureRows ||
        result.coverage.negativeSurpriseCount +
                result.coverage.validZeroSurpriseCount +
                result.coverage.positiveSurpriseCount !=
            result.coverage.surpriseAvailableCount ||
        result.coverage.nonzeroSurpriseCount !=
            result.coverage.negativeSurpriseCount +
                result.coverage.positiveSurpriseCount ||
        result.coverage.lowerClampedCount +
                result.coverage.upperClampedCount >
            result.coverage.surpriseAvailableCount ||
        MapCount(result.coverage.firstReleaseSourceCounts) !=
            result.coverage.surpriseAvailableCount ||
        MapCount(result.coverage.consensusSourceCounts) !=
            result.coverage.surpriseAvailableCount ||
        MapCount(result.coverage.eventFamilyCounts) !=
            result.coverage.surpriseAvailableCount ||
        MapCount(result.gapAttribution.provenanceReasonCounts) !=
            result.coverage.provenanceUnavailableCount ||
        MapCount(result.gapAttribution.missingConsensusReasonCounts) !=
            result.coverage.missingConsensusCount ||
        MapCount(result.gapAttribution.incompatibilityReasonCounts) !=
            result.coverage.incompatibleCount ||
        aggregateRows(result.gapAttribution.familyCounts) !=
            result.coverage.totalFeatureRows ||
        aggregateRows(result.gapAttribution.agencyCounts) !=
            result.coverage.totalFeatureRows ||
        aggregateRows(result.gapAttribution.yearCounts) !=
            result.coverage.totalFeatureRows ||
        vectorRows(result.gapAttribution.reasonCounts) !=
            SurpriseUnavailableCount(result.coverage) ||
        vectorRows(result.gapAttribution.priorities) !=
            SurpriseUnavailableCount(result.coverage))
    {
        throw std::logic_error(
            "causal_surprise_observability_partition_invariant_failed");
    }

    result.upstreamFeatureCanonical =
        UpstreamCanonicalText(experiment, scope, ranges);
    result.upstreamFeatureIdentity = TrainingObjective::DeterministicHash(
        result.upstreamFeatureCanonical);
    const std::string coverageCanonical =
        result.upstreamFeatureCanonical +
        ";source_row_count=" + std::to_string(result.sourceRowCount) +
        ";warmup_row_count=" + std::to_string(result.warmupRowCount) +
        ";coverage=" +
        CoverageCanonicalText(result.coverage);
    result.coverageIdentity =
        TrainingObjective::DeterministicHash(coverageCanonical);
    const std::string diagnosticCanonical =
        "causal_surprise_diagnostic_v1;experiment_id=" +
        std::to_string(experiment.experimentId) +
        ";feature_ablation_mask=" +
        std::to_string(experiment.featureAblationMask.size()) + ':' +
        experiment.featureAblationMask +
        ";coverage_identity=" + result.coverageIdentity;
    result.diagnosticIdentity =
        TrainingObjective::DeterministicHash(diagnosticCanonical);
    result.attributionCanonical =
        "causal_surprise_gap_attribution_identity_v1;semantic_version=" +
        std::to_string(kGapAttributionSemanticVersion) +
        ";upstream_feature_identity=" + result.upstreamFeatureIdentity +
        ";coverage_identity=" + result.coverageIdentity +
        ";compatibility_contract=" + kCompatibilityContract +
        ";normalization_contract=" + kNormalizationContract +
        ";source_row_count=" + std::to_string(result.sourceRowCount) +
        ";warmup_row_count=" + std::to_string(result.warmupRowCount) +
        ";attribution=" +
        GapAttributionCanonicalText(result.gapAttribution);
    result.attributionIdentity = TrainingObjective::DeterministicHash(
        result.attributionCanonical);
    return result;
}

ParityResult CompareUpstream(const Result& left, const Result& right)
{
    ParityResult result;
    if (left.experiment.symbol != right.experiment.symbol)
        AddReason(result, "symbol_mismatch");
    if (left.experiment.predictionHorizon !=
        right.experiment.predictionHorizon)
        AddReason(result, "prediction_horizon_mismatch");
    if (left.scope != right.scope ||
        left.evaluatedRanges != right.evaluatedRanges)
        AddReason(result, "scope_or_date_range_mismatch");
    if (left.experiment.featureWarmupScope !=
        right.experiment.featureWarmupScope)
        AddReason(result, "feature_warmup_scope_mismatch");
    if (left.experiment.donchian20Mode !=
        right.experiment.donchian20Mode ||
        left.experiment.donchianLookback !=
        right.experiment.donchianLookback)
        AddReason(result, "feature_eligibility_contract_mismatch");
    if (left.experiment.modelInputWidth !=
        right.experiment.modelInputWidth)
        AddReason(result, "model_input_width_mismatch");
    if (left.experiment.modelInputSemanticLayoutVersion !=
        right.experiment.modelInputSemanticLayoutVersion)
        AddReason(result, "model_input_semantic_layout_version_mismatch");

    const std::string& leftMask = left.experiment.featureAblationMask;
    const std::string& rightMask = right.experiment.featureAblationMask;
    if (leftMask != rightMask)
    {
        const std::string surpriseMask{
            kCausalEconomicEventSurpriseAblationMaskText};
        const bool controlledDifference =
            (leftMask.empty() && rightMask == surpriseMask) ||
            (rightMask.empty() && leftMask == surpriseMask);
        if (!controlledDifference)
            AddReason(result,
                      "unexpected_feature_ablation_mask_difference");
    }

    if (left.upstreamFeatureIdentity != right.upstreamFeatureIdentity)
        AddReason(result, "upstream_feature_identity_mismatch");
    result.comparable = result.reasons.empty();
    result.coverageMatches = result.comparable &&
        left.coverageIdentity == right.coverageIdentity &&
        left.coverage == right.coverage &&
        left.attributionIdentity == right.attributionIdentity &&
        left.gapAttribution == right.gapAttribution;
    if (result.comparable && !result.coverageMatches)
        AddReason(result, "upstream_coverage_mismatch");
    return result;
}

} // namespace EA::CausalSurpriseObservability
