#include "CausalSurpriseObservability.hpp"

#include "EconomicEventFeatureLayout.hpp"
#include "FeatureAblation.hpp"
#include "FeatureLayout.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>

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

Coverage EvaluateSegment(SegmentInput segment)
{
    if (segment.warmupRowCount > segment.sourceBarStarts.size())
        throw std::invalid_argument(
            "causal_surprise_observability_warmup_exceeds_source_rows");

    EconomicCalendar::EconomicEventFeatureEngine engine{
        std::move(segment.economicEvents)};
    Coverage coverage;
    for (std::size_t index = 0;
         index < segment.sourceBarStarts.size(); ++index)
    {
        (void)engine.AdvanceCompletedBar(segment.sourceBarStarts[index]);
        if (index < segment.warmupRowCount) continue;
        Observe(coverage, engine.LastCausalSurpriseObservation());
    }
    return coverage;
}

void AddReason(ParityResult& result, std::string reason)
{
    result.reasons.push_back(std::move(reason));
}

} // namespace

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
        "invalid --causal-surprise-observability-scope value; expected "
        "train, infer, or combined");
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
        Merge(result.coverage, EvaluateSegment(std::move(segments[index])));
    }
    const auto mapCount = [](const auto& values)
    {
        std::size_t total = 0;
        for (const auto& [key, count] : values)
        {
            (void)key;
            total += count;
        }
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
        mapCount(result.coverage.firstReleaseSourceCounts) !=
            result.coverage.surpriseAvailableCount ||
        mapCount(result.coverage.consensusSourceCounts) !=
            result.coverage.surpriseAvailableCount ||
        mapCount(result.coverage.eventFamilyCounts) !=
            result.coverage.surpriseAvailableCount)
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
        left.coverage == right.coverage;
    if (result.comparable && !result.coverageMatches)
        AddReason(result, "upstream_coverage_mismatch");
    return result;
}

} // namespace EA::CausalSurpriseObservability
