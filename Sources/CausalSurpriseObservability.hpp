#pragma once

#include "EconomicEventFeatures.hpp"

#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "FeatureWarmupScope.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace EA::CausalSurpriseObservability
{

inline constexpr int kDiagnosticSemanticVersion = 1;
inline constexpr int kGapAttributionSemanticVersion = 1;
inline constexpr const char* kFirstReleasePitContract =
    "economic_event_first_release_actual_at_v1_migration090";
inline constexpr const char* kNormalizationContract =
    "economic_event_family_unit_fixed_scale_clamp_10_v1";
inline constexpr const char* kCompatibilityContract =
    "causal_scalar_valid_shape_kind_unit_scale_qualifier_v1";
inline constexpr const char* kRepositorySelectionContract =
    "economic_event_feature_range_relevant_event_selected_consensus_v1";

enum class Scope
{
    train,
    infer,
    combined,
};

const char* ScopeText(Scope scope);
Scope ParseScope(const std::string& text);

struct DateRange
{
    std::string role;
    std::string start;
    std::string end;
    bool operator==(const DateRange&) const = default;
};

struct ExperimentContext
{
    long long experimentId = 0;
    std::string symbol;
    int predictionHorizon = 0;
    std::string trainStart;
    std::string trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<int> modelInputWidth;
    std::optional<int> modelInputSemanticLayoutVersion;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    FeatureWarmupScope featureWarmupScope = kDefaultFeatureWarmupScope;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    std::string featureAblationMask;
    bool operator==(const ExperimentContext&) const = default;
};

struct Coverage
{
    std::size_t totalFeatureRows = 0;
    std::size_t noRelevantEventCount = 0;
    std::size_t provenanceUnavailableCount = 0;
    std::size_t ambiguousCount = 0;
    std::size_t notYetAvailableCount = 0;
    std::size_t missingConsensusCount = 0;
    std::size_t incompatibleCount = 0;
    std::size_t surpriseAvailableCount = 0;
    std::size_t validZeroSurpriseCount = 0;
    std::size_t nonzeroSurpriseCount = 0;
    std::size_t negativeSurpriseCount = 0;
    std::size_t positiveSurpriseCount = 0;
    std::size_t lowerClampedCount = 0;
    std::size_t upperClampedCount = 0;
    long double surpriseSum = 0.0L;
    std::optional<float> minimumSurprise;
    std::optional<float> maximumSurprise;
    std::map<std::string, std::size_t> firstReleaseSourceCounts;
    std::map<std::string, std::size_t> consensusSourceCounts;
    std::map<std::string, std::size_t> eventFamilyCounts;
    bool operator==(const Coverage&) const = default;
};

struct SegmentInput
{
    DateRange range;
    std::vector<PriceTP> sourceBarStarts;
    std::size_t warmupRowCount = 0;
    std::vector<EconomicCalendar::EconomicEvent> economicEvents;
};

enum class RemediabilityClass
{
    expectedByContract,
    potentiallyRemediableDataGap,
    requiresManualProvenanceReview,
    unsupportedSemantics,
};

const char* RemediabilityClassText(RemediabilityClass value) noexcept;
const char* DispositionText(
    EconomicCalendar::CausalSurpriseDisposition value) noexcept;

struct AttributionAggregate
{
    std::size_t totalRows = 0;
    std::size_t availableRows = 0;
    std::size_t unavailableRows = 0;
    std::map<std::string, std::size_t> dispositionCounts;
    std::set<long long> affectedEventIds;
    bool operator==(const AttributionAggregate&) const = default;
};

struct GapReasonSummary
{
    std::string disposition;
    std::string rawReason;
    RemediabilityClass remediability =
        RemediabilityClass::expectedByContract;
    std::size_t affectedFeatureRows = 0;
    std::set<long long> affectedEventIds;
    bool operator==(const GapReasonSummary&) const = default;
};

struct GapPriorityEntry
{
    std::string disposition;
    std::string rawReason;
    std::string eventFamily;
    std::string sourceAgency;
    RemediabilityClass remediability =
        RemediabilityClass::expectedByContract;
    std::size_t affectedFeatureRows = 0;
    std::set<long long> affectedEventIds;
    bool operator==(const GapPriorityEntry&) const = default;
};

struct GapAttribution
{
    std::map<std::string, std::size_t> provenanceReasonCounts;
    std::map<std::string, std::size_t> missingConsensusReasonCounts;
    std::map<std::string, std::size_t> incompatibilityReasonCounts;
    std::map<std::string, AttributionAggregate> familyCounts;
    std::map<std::string, AttributionAggregate> agencyCounts;
    std::map<std::string, AttributionAggregate> yearCounts;
    std::vector<GapReasonSummary> reasonCounts;
    std::vector<GapPriorityEntry> priorities;
    std::optional<std::int64_t> firstNoRelevantBarUnixSeconds;
    std::optional<std::int64_t> lastNoRelevantBarUnixSeconds;
    bool operator==(const GapAttribution&) const = default;
};

struct Result
{
    ExperimentContext experiment;
    Scope scope = Scope::combined;
    std::vector<DateRange> evaluatedRanges;
    std::size_t sourceRowCount = 0;
    std::size_t warmupRowCount = 0;
    Coverage coverage;
    std::string upstreamFeatureCanonical;
    std::string upstreamFeatureIdentity;
    std::string coverageIdentity;
    std::string diagnosticIdentity;
    GapAttribution gapAttribution;
    std::string attributionCanonical;
    std::string attributionIdentity;
};

struct ParityResult
{
    bool comparable = false;
    bool coverageMatches = false;
    std::vector<std::string> reasons;
};

std::vector<DateRange> ResolveRanges(
    const ExperimentContext& experiment,
    Scope scope);

void Observe(Coverage& coverage,
             const EconomicCalendar::CausalSurpriseObservation& observation);

std::size_t DispositionPartitionCount(const Coverage& coverage) noexcept;
std::size_t SurpriseUnavailableCount(const Coverage& coverage) noexcept;
double SurpriseAvailableRate(const Coverage& coverage) noexcept;
std::optional<double> MeanSurprise(const Coverage& coverage) noexcept;

Result Evaluate(const ExperimentContext& experiment,
                Scope scope,
                std::vector<SegmentInput> segments);

ParityResult CompareUpstream(const Result& left, const Result& right);

std::string CoverageCanonicalText(const Coverage& coverage);
std::string GapAttributionCanonicalText(const GapAttribution& attribution);

} // namespace EA::CausalSurpriseObservability
