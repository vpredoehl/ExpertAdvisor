#include "CausalSurpriseObservabilityService.hpp"

#include "CausalSurpriseObservabilityRepository.hpp"
#include "EconomicEventRepository.hpp"
#include "FeatureLayout.hpp"
#include "FeatureWarmupScope.hpp"
#include "ModelInputExpansion.hpp"

#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <pqxx/pqxx>

namespace EA::CausalSurpriseObservability
{
namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.' ||
            character == ',' || character == ':' || character == '+';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

std::string OptionalInteger(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalNumber(const std::optional<double>& value)
{
    if (!value) return "NULL";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << *value;
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

std::string Number(double value)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << value;
    return output.str();
}

void RenderSourceMap(
    std::ostringstream& output,
    std::string_view dimension,
    const std::map<std::string, std::size_t>& values)
{
    if (values.empty())
    {
        output << "CAUSAL_SURPRISE_OBSERVABILITY_SOURCE"
               << ",dimension=" << dimension
               << ",value=NONE,count=0\n";
        return;
    }
    for (const auto& [value, count] : values)
    {
        output << "CAUSAL_SURPRISE_OBSERVABILITY_SOURCE"
               << ",dimension=" << dimension
               << ",value=" << MachineText(value)
               << ",count=" << count << '\n';
    }
}

bool ChannelsProjected(const ExperimentContext& experiment)
{
    return experiment.modelInputWidth &&
        experiment.modelInputSemanticLayoutVersion &&
        *experiment.modelInputWidth >=
            static_cast<int>(kCausalEconomicEventSurpriseModelInputWidth) &&
        *experiment.modelInputSemanticLayoutVersion >=
            kModelInputSemanticLayoutVersion;
}

std::size_t CountDisposition(const AttributionAggregate& aggregate,
                             std::string_view disposition)
{
    const auto found = aggregate.dispositionCounts.find(
        std::string{disposition});
    return found == aggregate.dispositionCounts.end() ? 0 : found->second;
}

void RenderAttributionAggregate(
    std::ostringstream& output,
    std::string_view record,
    std::string_view dimension,
    const std::map<std::string, AttributionAggregate>& values)
{
    for (const auto& [value, aggregate] : values)
    {
        output << record
               << ",dimension=" << dimension
               << ",value=" << MachineText(value)
               << ",total_rows=" << aggregate.totalRows
               << ",available_rows=" << aggregate.availableRows
               << ",unavailable_rows=" << aggregate.unavailableRows
               << ",availability_rate="
               << Number(aggregate.totalRows == 0 ? 0.0 :
                   static_cast<double>(aggregate.availableRows) /
                       static_cast<double>(aggregate.totalRows))
               << ",no_relevant_event="
               << CountDisposition(aggregate, "no_relevant_event")
               << ",provenance_unavailable="
               << CountDisposition(aggregate, "provenance_unavailable")
               << ",ambiguous="
               << CountDisposition(aggregate, "ambiguous")
               << ",not_yet_available="
               << CountDisposition(aggregate, "not_yet_available")
               << ",missing_consensus="
               << CountDisposition(aggregate, "missing_consensus")
               << ",incompatible="
               << CountDisposition(aggregate, "incompatible")
               << ",available="
               << CountDisposition(aggregate, "available")
               << ",unique_affected_events="
               << aggregate.affectedEventIds.size() << '\n';
    }
}

Result EvaluateFromRepositories(
    const std::string& lstmConnectionString,
    const std::string& forexConnectionString,
    long long experimentId,
    Scope scope)
{
    pqxx::connection lstmConnection{lstmConnectionString};
    pqxx::read_transaction lstmRead{lstmConnection};
    lstmRead.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const ExperimentContext experiment =
        LoadExperimentContext(lstmRead, experimentId);
    const auto economicCalendarSnapshot =
        EconomicCalendar::LoadExperimentEconomicCalendarSnapshot(
            lstmRead, experimentId);
    const std::vector<DateRange> ranges =
        ResolveRanges(experiment, scope);

    pqxx::connection forexConnection{forexConnectionString};
    pqxx::read_transaction forexRead{forexConnection};
    forexRead.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");

    std::vector<SegmentInput> segments;
    segments.reserve(ranges.size());
    for (const DateRange& range : ranges)
    {
        const bool fullHistoryWarmup =
            experiment.featureWarmupScope ==
                FeatureWarmupScope::FullHistoryWarmup;
        const std::string sourceStart = fullHistoryWarmup
            ? kTensorFeatureHistoryQueryStart : range.start;
        BarPopulation bars = LoadBarPopulation(
            forexRead, experiment.symbol, sourceStart,
            range.start, range.end);
        std::vector<EconomicCalendar::EconomicEvent> events =
            EconomicCalendar::LoadEconomicEventsForFeatureRange(
                lstmRead,
                std::string{EconomicCalendar::
                    kEconomicEventFeatureCurrency},
                sourceStart,
                range.end,
                economicCalendarSnapshot);
        segments.push_back(SegmentInput{
            range,
            std::move(bars.sourceBarStarts),
            bars.warmupRowCount,
            std::move(events)});
    }
    return Evaluate(experiment, scope, std::move(segments));
}

} // namespace

std::string Render(const Result& result)
{
    const auto& experiment = result.experiment;
    const auto& coverage = result.coverage;
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "CAUSAL_SURPRISE_OBSERVABILITY"
           << ",version=" << kDiagnosticSemanticVersion
           << ",experiment_id=" << experiment.experimentId
           << ",scope=" << ScopeText(result.scope)
           << ",read_only=true\n";
    output << "CAUSAL_SURPRISE_OBSERVABILITY_CONTEXT"
           << ",symbol=" << MachineText(experiment.symbol)
           << ",prediction_horizon=" << experiment.predictionHorizon
           << ",train_start=" << MachineText(experiment.trainStart)
           << ",train_end=" << MachineText(experiment.trainEnd)
           << ",infer_start="
           << (experiment.inferStart
                   ? MachineText(*experiment.inferStart) : "NULL")
           << ",infer_end="
           << (experiment.inferEnd
                   ? MachineText(*experiment.inferEnd) : "NULL")
           << ",model_input_width="
           << OptionalInteger(experiment.modelInputWidth)
           << ",model_input_semantic_layout_version="
           << OptionalInteger(
                  experiment.modelInputSemanticLayoutVersion)
           << ",physical_tensor_feature_count=" << feature_size
           << ",feature_warmup_scope="
           << FeatureWarmupScopeText(experiment.featureWarmupScope)
           << ",donchian20_mode="
           << Donchian20ModeText(experiment.donchian20Mode)
           << ",donchian_lookback=" << experiment.donchianLookback
           << ",feature_ablation_mask="
           << (experiment.featureAblationMask.empty()
                   ? "EMPTY" : experiment.featureAblationMask)
           << ",surprise_channels_projected_to_model_input="
           << (ChannelsProjected(experiment) ? "true" : "false")
           << ",feature_ablation_is_downstream=true\n";
    for (const DateRange& range : result.evaluatedRanges)
    {
        output << "CAUSAL_SURPRISE_OBSERVABILITY_RANGE"
               << ",role=" << range.role
               << ",start=" << MachineText(range.start)
               << ",end=" << MachineText(range.end)
               << ",boundary=half_open\n";
    }
    output << "CAUSAL_SURPRISE_OBSERVABILITY_CONTRACT"
           << ",availability_feature="
           << "causal_first_release_surprise_available"
           << ",surprise_feature=causal_first_release_surprise"
           << ",availability_tensor_column="
           << causalFirstReleaseSurpriseAvailableCol
           << ",surprise_tensor_column="
           << causalFirstReleaseSurpriseCol
           << ",first_release_pit_contract=" << kFirstReleasePitContract
           << ",normalization_contract=" << kNormalizationContract
           << ",repository_selection_contract="
           << kRepositorySelectionContract
           << ",information_cutoff=completed_bar_end"
           << ",proven_availability_boundary=inclusive"
           << ",clamp_observability=authoritative_preclamp_hook\n";
    output << "CAUSAL_SURPRISE_OBSERVABILITY_IDENTITY"
           << ",upstream_feature_identity="
           << result.upstreamFeatureIdentity
           << ",coverage_identity=" << result.coverageIdentity
           << ",diagnostic_identity=" << result.diagnosticIdentity
           << ",experiment_id_in_upstream_identity=false"
           << ",feature_ablation_mask_in_upstream_identity=false"
           << ",operational_metadata_in_upstream_identity=false\n";
    output << "CAUSAL_SURPRISE_OBSERVABILITY_COVERAGE"
           << ",denominator_contract="
           << "candlestick_feature_rows_after_feature_warmup_prefix"
           << ",source_rows=" << result.sourceRowCount
           << ",warmup_rows_excluded=" << result.warmupRowCount
           << ",total_feature_rows=" << coverage.totalFeatureRows
           << ",surprise_available_count="
           << coverage.surpriseAvailableCount
           << ",surprise_unavailable_count="
           << SurpriseUnavailableCount(coverage)
           << ",surprise_available_rate="
           << Number(SurpriseAvailableRate(coverage)) << '\n';
    output << "CAUSAL_SURPRISE_OBSERVABILITY_DISPOSITIONS"
           << ",no_relevant_event=" << coverage.noRelevantEventCount
           << ",provenance_unavailable="
           << coverage.provenanceUnavailableCount
           << ",ambiguous=" << coverage.ambiguousCount
           << ",not_yet_available=" << coverage.notYetAvailableCount
           << ",missing_consensus=" << coverage.missingConsensusCount
           << ",incompatible=" << coverage.incompatibleCount
           << ",available=" << coverage.surpriseAvailableCount
           << ",partition_sum=" << DispositionPartitionCount(coverage)
           << ",partition_matches_total="
           << (DispositionPartitionCount(coverage) ==
                       coverage.totalFeatureRows
                   ? "true" : "false") << '\n';
    output << "CAUSAL_SURPRISE_OBSERVABILITY_AVAILABLE_VALUES"
           << ",valid_zero_surprise_count="
           << coverage.validZeroSurpriseCount
           << ",nonzero_surprise_count="
           << coverage.nonzeroSurpriseCount
           << ",negative_surprise_count="
           << coverage.negativeSurpriseCount
           << ",positive_surprise_count="
           << coverage.positiveSurpriseCount
           << ",sign_partition_sum="
           << coverage.negativeSurpriseCount +
                  coverage.validZeroSurpriseCount +
                  coverage.positiveSurpriseCount
           << ",unavailable_placeholder_zero_in_valid_zero=false\n";
    output << "CAUSAL_SURPRISE_OBSERVABILITY_DISTRIBUTION"
           << ",count=" << coverage.surpriseAvailableCount
           << ",minimum=" << OptionalFloat(coverage.minimumSurprise)
           << ",maximum=" << OptionalFloat(coverage.maximumSurprise)
           << ",mean=" << OptionalNumber(MeanSurprise(coverage))
           << ",unavailable_rows_excluded=true\n";
    output << "CAUSAL_SURPRISE_OBSERVABILITY_CLAMP"
           << ",lower_clamped_count=" << coverage.lowerClampedCount
           << ",upper_clamped_count=" << coverage.upperClampedCount
           << ",total_clamped_count="
           << coverage.lowerClampedCount + coverage.upperClampedCount
           << ",evidence=authoritative_preclamp_normalized_value\n";
    RenderSourceMap(output, "first_release_actual_source",
                    coverage.firstReleaseSourceCounts);
    RenderSourceMap(output, "consensus_source",
                    coverage.consensusSourceCounts);
    RenderSourceMap(output, "event_family", coverage.eventFamilyCounts);
    output << "CAUSAL_SURPRISE_OBSERVABILITY_RESULT"
           << ",experiment_id=" << experiment.experimentId
           << ",scope=" << ScopeText(result.scope)
           << ",read_only=true,software_success=true\n";
    return output.str();
}

std::string RenderGapAttribution(const Result& result)
{
    const auto& experiment = result.experiment;
    const auto& coverage = result.coverage;
    const auto& attribution = result.gapAttribution;
    const std::size_t unavailable = SurpriseUnavailableCount(coverage);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "CAUSAL_SURPRISE_GAP_ATTRIBUTION"
           << ",version=" << kGapAttributionSemanticVersion
           << ",experiment_id=" << experiment.experimentId
           << ",scope=" << ScopeText(result.scope)
           << ",read_only=true"
           << ",denominator_contract="
           << "candlestick_feature_rows_after_feature_warmup_prefix\n";
    for (const DateRange& range : result.evaluatedRanges)
    {
        output << "CAUSAL_SURPRISE_GAP_RANGE"
               << ",role=" << range.role
               << ",start=" << MachineText(range.start)
               << ",end=" << MachineText(range.end)
               << ",boundary=half_open\n";
    }
    output << "CAUSAL_SURPRISE_GAP_CONTRACT"
           << ",first_release_pit_contract=" << kFirstReleasePitContract
           << ",compatibility_contract=" << kCompatibilityContract
           << ",compatibility_normalization_contract="
           << kNormalizationContract
           << ",repository_selection_contract="
           << kRepositorySelectionContract
           << ",information_cutoff=completed_bar_end"
           << ",feature_ablation_is_downstream=true\n";
    output << "CAUSAL_SURPRISE_GAP_IDENTITY"
           << ",upstream_feature_identity="
           << result.upstreamFeatureIdentity
           << ",coverage_identity=" << result.coverageIdentity
           << ",attribution_identity=" << result.attributionIdentity
           << ",experiment_id_in_upstream_attribution_identity=false"
           << ",feature_ablation_mask_in_upstream_attribution_identity=false"
           << ",operational_metadata_in_upstream_attribution_identity=false\n";
    output << "CAUSAL_SURPRISE_GAP_RECONCILIATION"
           << ",source_rows=" << result.sourceRowCount
           << ",warmup_rows_excluded=" << result.warmupRowCount
           << ",total_feature_rows=" << coverage.totalFeatureRows
           << ",total_available=" << coverage.surpriseAvailableCount
           << ",total_unavailable=" << unavailable
           << ",terminal_partition_sum="
           << DispositionPartitionCount(coverage)
           << ",terminal_partition_matches_total="
           << (DispositionPartitionCount(coverage) ==
                       coverage.totalFeatureRows ? "true" : "false")
           << '\n';

    for (const GapReasonSummary& reason : attribution.reasonCounts)
    {
        output << "CAUSAL_SURPRISE_GAP_REASON"
               << ",disposition=" << reason.disposition
               << ",raw_reason=" << MachineText(reason.rawReason)
               << ",remediability_class="
               << RemediabilityClassText(reason.remediability)
               << ",affected_feature_rows="
               << reason.affectedFeatureRows
               << ",unique_affected_events="
               << reason.affectedEventIds.size() << '\n';
    }
    RenderAttributionAggregate(
        output, "CAUSAL_SURPRISE_GAP_FAMILY", "event_family",
        attribution.familyCounts);
    RenderAttributionAggregate(
        output, "CAUSAL_SURPRISE_GAP_AGENCY", "source_agency",
        attribution.agencyCounts);
    RenderAttributionAggregate(
        output, "CAUSAL_SURPRISE_GAP_YEAR", "calendar_year_utc",
        attribution.yearCounts);

    for (const auto& [reason, count] :
         attribution.provenanceReasonCounts)
    {
        output << "CAUSAL_SURPRISE_GAP_PROVENANCE"
               << ",raw_selection_reason=" << MachineText(reason)
               << ",affected_feature_rows=" << count << '\n';
    }
    for (const auto& [reason, count] :
         attribution.missingConsensusReasonCounts)
    {
        output << "CAUSAL_SURPRISE_GAP_MISSING_CONSENSUS"
               << ",raw_reason=" << MachineText(reason)
               << ",affected_feature_rows=" << count << '\n';
    }
    for (const auto& [reason, count] :
         attribution.incompatibilityReasonCounts)
    {
        output << "CAUSAL_SURPRISE_GAP_INCOMPATIBILITY"
               << ",exact_reason=" << MachineText(reason)
               << ",affected_feature_rows=" << count << '\n';
    }
    output << "CAUSAL_SURPRISE_GAP_NO_RELEVANT_EVENT"
           << ",affected_feature_rows=" << coverage.noRelevantEventCount
           << ",first_bar_unix_seconds="
           << (attribution.firstNoRelevantBarUnixSeconds
                   ? std::to_string(
                         *attribution.firstNoRelevantBarUnixSeconds)
                   : "NULL")
           << ",last_bar_unix_seconds="
           << (attribution.lastNoRelevantBarUnixSeconds
                   ? std::to_string(
                         *attribution.lastNoRelevantBarUnixSeconds)
                   : "NULL")
           << ",year_detail_in=CAUSAL_SURPRISE_GAP_YEAR\n";

    for (std::size_t index = 0;
         index < attribution.priorities.size(); ++index)
    {
        const GapPriorityEntry& priority = attribution.priorities[index];
        output << "CAUSAL_SURPRISE_GAP_PRIORITY"
               << ",rank=" << index + 1
               << ",disposition=" << priority.disposition
               << ",raw_reason=" << MachineText(priority.rawReason)
               << ",event_family=" << MachineText(priority.eventFamily)
               << ",source_agency=" << MachineText(priority.sourceAgency)
               << ",affected_feature_rows="
               << priority.affectedFeatureRows
               << ",percent_of_total_feature_rows="
               << Number(coverage.totalFeatureRows == 0 ? 0.0 :
                   static_cast<double>(priority.affectedFeatureRows) /
                       static_cast<double>(coverage.totalFeatureRows))
               << ",percent_of_unavailable_rows="
               << Number(unavailable == 0 ? 0.0 :
                   static_cast<double>(priority.affectedFeatureRows) /
                       static_cast<double>(unavailable))
               << ",remediability_class="
               << RemediabilityClassText(priority.remediability)
               << ",unique_affected_events="
               << priority.affectedEventIds.size() << '\n';
    }
    output << "CAUSAL_SURPRISE_GAP_RESULT"
           << ",experiment_id=" << experiment.experimentId
           << ",scope=" << ScopeText(result.scope)
           << ",read_only=true,software_success=true\n";
    return output.str();
}

int RunCommand(const std::string& lstmConnectionString,
               const std::string& forexConnectionString,
               long long experimentId,
               Scope scope,
               std::ostream& output,
               std::ostream& errors)
{
    try
    {
        const Result result = EvaluateFromRepositories(
            lstmConnectionString, forexConnectionString,
            experimentId, scope);
        output << Render(result);
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CAUSAL_SURPRISE_OBSERVABILITY_FAILED"
               << ",experiment_id=" << experimentId
               << ",scope=" << ScopeText(scope)
               << ",reason=" << MachineText(error.what())
               << ",read_only=true,software_success=false\n";
        return 3;
    }
}

int RunGapAttributionCommand(
    const std::string& lstmConnectionString,
    const std::string& forexConnectionString,
    long long experimentId,
    Scope scope,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const Result result = EvaluateFromRepositories(
            lstmConnectionString, forexConnectionString,
            experimentId, scope);
        output << RenderGapAttribution(result);
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CAUSAL_SURPRISE_GAP_ATTRIBUTION_FAILED"
               << ",experiment_id=" << experimentId
               << ",scope=" << ScopeText(scope)
               << ",reason=" << MachineText(error.what())
               << ",read_only=true,software_success=false\n";
        return 3;
    }
}

} // namespace EA::CausalSurpriseObservability
