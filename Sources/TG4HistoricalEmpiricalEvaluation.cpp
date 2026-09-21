#include "TG4HistoricalEmpiricalEvaluation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace EA::TG4
{
namespace
{

constexpr std::int64_t kExploratoryStart = 1'262'304'000; // 2010-01-01 UTC
constexpr std::int64_t kCalibrationStart = 1'577'836'800; // 2020-01-01 UTC
constexpr std::int64_t kValidationStart = 1'672'531'200; // 2023-01-01 UTC
constexpr std::int64_t kConfirmationStart = 1'735'689'600; // 2025-01-01 UTC
constexpr std::int64_t kNamedStudyEnd = 1'767'225'600; // 2026-01-01 UTC

std::string Trim(std::string value)
{
    const auto notSpace = [](unsigned char value)
    {
        return value != ' ' && value != '\t' && value != '\r' && value != '\n';
    };
    const auto begin = std::find_if(value.begin(), value.end(), notSpace);
    const auto end = std::find_if(value.rbegin(), value.rend(), notSpace).base();
    return begin < end ? std::string(begin, end) : std::string{};
}

std::string Require(const std::map<std::string, std::string>& values,
                    const std::string& key)
{
    const auto found = values.find(key);
    if (found == values.end() || found->second.empty())
        throw std::invalid_argument("TG4 configuration requires '" + key + "'");
    return found->second;
}

template<typename Integer>
Integer ParseInteger(const std::map<std::string, std::string>& values,
                     const std::string& key)
{
    const std::string text = Require(values, key);
    Integer parsed{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(),
                                        parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("TG4 configuration '" + key +
                                    "' must be an integer");
    return parsed;
}

double ParseDouble(const std::map<std::string, std::string>& values,
                   const std::string& key)
{
    const std::string text = Require(values, key);
    std::size_t consumed = 0;
    const double parsed = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed))
        throw std::invalid_argument("TG4 configuration '" + key +
                                    "' must be finite numeric text");
    return parsed;
}

std::vector<double> ParseRatios(const std::string& text)
{
    std::vector<double> ratios;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t comma = text.find(',', begin);
        const std::string token = Trim(text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin));
        if (token.empty())
            throw std::invalid_argument(
                "TG4 Fibonacci ratios must be an explicit comma-separated list");
        std::size_t consumed = 0;
        const double ratio = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(ratio))
            throw std::invalid_argument("TG4 Fibonacci ratio is not finite numeric text");
        ratios.push_back(ratio);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return ratios;
}

void RequireValue(const std::map<std::string, std::string>& values,
                  const std::string& key,
                  const std::string& expected)
{
    const std::string actual = Require(values, key);
    if (actual != expected)
        throw std::invalid_argument("TG4 configuration '" + key +
                                    "' must be '" + expected + "'");
}

std::int64_t DaysFromCivil(int year, unsigned month, unsigned day)
{
    year -= month <= 2 ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yearOfEra = static_cast<unsigned>(year - era * 400);
    const unsigned shiftedMonth = static_cast<unsigned>(
        static_cast<int>(month) + (month > 2 ? -3 : 9));
    const unsigned dayOfYear =
        (153 * shiftedMonth + 2) / 5 + day - 1;
    const unsigned dayOfEra = yearOfEra * 365 + yearOfEra / 4 -
        yearOfEra / 100 + dayOfYear;
    return static_cast<std::int64_t>(era) * 146097 +
        static_cast<std::int64_t>(dayOfEra) - 719468;
}

bool LeapYear(int year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int DaysInMonth(int year, int month)
{
    static constexpr std::array<int, 12> days{
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return month == 2 && LeapYear(year) ? 29 : days.at(
        static_cast<std::size_t>(month - 1));
}

std::string JsonEscape(std::string_view value)
{
    std::ostringstream output;
    for (const unsigned char character : value)
    {
        switch (character)
        {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20)
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec << std::setfill(' ');
                else
                    output << static_cast<char>(character);
        }
    }
    return output.str();
}

std::string Csv(std::string_view value)
{
    bool quote = false;
    for (char character : value)
        quote = quote || character == ',' || character == '"' ||
            character == '\n' || character == '\r';
    if (!quote) return std::string(value);
    std::string result{"\""};
    for (char character : value)
    {
        if (character == '"') result += "\"\"";
        else result += character;
    }
    result += '"';
    return result;
}

std::string JoinCsv(const std::vector<std::string>& fields)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0) output << ',';
        output << Csv(fields[index]);
    }
    output << '\n';
    return output.str();
}

template<typename T>
std::string Number(T value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

template<typename T>
std::string OptionalNumber(const std::optional<T>& value)
{
    return value.has_value() ? Number(*value) : std::string{};
}

std::string OptionalTimestamp(const std::optional<std::int64_t>& value)
{
    return value.has_value() ? FormatUtcTimestamp(*value) : std::string{};
}

std::string DirectionName(TG1A::TrendLineDirection direction)
{
    return direction == TG1A::TrendLineDirection::UTL ? "UTL" : "DTL";
}

std::string ClassificationName(TG1B::SteepnessClassification classification)
{
    switch (classification)
    {
        case TG1B::SteepnessClassification::LongTerm: return "LongTerm";
        case TG1B::SteepnessClassification::Outer: return "Outer";
        case TG1B::SteepnessClassification::Inner: return "Inner";
        case TG1B::SteepnessClassification::Unclassified: return "Unclassified";
    }
    return "Unclassified";
}

std::string ResolutionName(TG2::ResolutionState state)
{
    switch (state)
    {
        case TG2::ResolutionState::Pending: return "pending";
        case TG2::ResolutionState::Succeeded: return "succeeded";
        case TG2::ResolutionState::Failed: return "failed";
        case TG2::ResolutionState::Censored: return "censored";
        case TG2::ResolutionState::StructurallyIneligible:
            return "structurally_ineligible";
    }
    return "pending";
}

std::string CensorReasonName(TG2::CensorReason reason)
{
    switch (reason)
    {
        case TG2::CensorReason::None: return "none";
        case TG2::CensorReason::EndOfInput: return "end_of_input";
        case TG2::CensorReason::CapacityEviction: return "capacity_eviction";
    }
    return "none";
}

std::string ConfluenceName(TG3::ConfluenceState state)
{
    switch (state)
    {
        case TG3::ConfluenceState::Confluence: return "confluent";
        case TG3::ConfluenceState::NoConfluence: return "non_confluent";
        case TG3::ConfluenceState::StructurallyIneligible:
            return "structurally_ineligible";
    }
    return "structurally_ineligible";
}

std::string IneligibleReasonName(TG3::IneligibleReason reason)
{
    switch (reason)
    {
        case TG3::IneligibleReason::None: return "none";
        case TG3::IneligibleReason::NoTG2PairedOuter:
            return "no_tg2_paired_outer";
        case TG3::IneligibleReason::DirectionUnsupportedByStudy:
            return "direction_unsupported_by_source_study";
        case TG3::IneligibleReason::NoCausallyEligibleAB:
            return "no_causally_eligible_ab";
    }
    return "none";
}

std::string BreakPolicyName(TG2::BreakPolicy policy)
{
    return policy == TG2::BreakPolicy::CompletedCloseBeyondLine
        ? "completed_close_beyond_line" : "completed_wick_beyond_line";
}

std::string DirectionalPolicyName(TG3::DirectionalStudyPolicy policy)
{
    return policy == TG3::DirectionalStudyPolicy::SourceUTLUpABOnly
        ? "source_utl_up_ab_only"
        : "symmetric_directional_diagnostic_hypothesis";
}

std::string ABDirectionName(TG3::ABDirection direction)
{
    return direction == TG3::ABDirection::UpAB ? "UpAB" : "DownAB";
}

std::string RatiosText(const std::vector<double>& ratios)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < ratios.size(); ++index)
    {
        if (index != 0) output << ';';
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << ratios[index];
    }
    return output.str();
}

std::string LevelsText(const std::vector<TG3::LevelMeasurement>& levels)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < levels.size(); ++index)
    {
        if (index != 0) output << ';';
        const auto& level = levels[index];
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << "ratio=" << level.ratio
               << "|price=" << level.levelPrice
               << "|lower=" << level.zoneLowerPrice
               << "|upper=" << level.zoneUpperPrice
               << "|distance=" << level.rawPriceDistance
               << "|matched=" << (level.matched ? "true" : "false");
    }
    return output.str();
}

std::string CandidateIdentityText(const TG2::CandidateIdentity& identity)
{
    std::ostringstream output;
    output << DirectionName(identity.direction) << ':'
           << identity.anchor1Timestamp << ':' << identity.anchor2Timestamp
           << ':' << identity.creationTimestamp << ':'
           << identity.anchor1Bar << ':' << identity.anchor2Bar << ':'
           << identity.creationBar;
    return output.str();
}

std::uint64_t Fnv1a64(std::string_view text)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (unsigned char character : text)
    {
        value ^= character;
        value *= 1099511628211ULL;
    }
    return value;
}

std::string HashIdentity(std::string_view canonical)
{
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
           << Fnv1a64(canonical);
    return output.str();
}

std::string EventIdentity(const std::string& symbol,
                          const std::string& timeframe,
                          const TG2::BreakEvent& event)
{
    return HashIdentity(std::string(kObservationSchemaVersion) + "|" + symbol +
        "|" + timeframe + "|" + CandidateIdentityText(event.candidate) + "|" +
        std::to_string(event.timestamp) + "|" +
        std::to_string(event.eventSequence));
}

bool SameCandidate(const TG2::CandidateIdentity& identity,
                   const TG1B::ClassifiedTrendLineCandidate& candidate)
{
    return identity.direction == candidate.direction &&
        identity.anchor1Bar == candidate.anchor1Bar &&
        identity.anchor2Bar == candidate.anchor2Bar &&
        identity.creationBar == candidate.creationBar &&
        identity.anchor1Timestamp == candidate.anchor1Timestamp &&
        identity.anchor2Timestamp == candidate.anchor2Timestamp &&
        identity.creationTimestamp == candidate.creationTimestamp;
}

std::optional<TG1B::ClassifiedTrendLineCandidate> FindCandidate(
    const std::vector<TG1B::ClassifiedTrendLineCandidate>& candidates,
    const TG2::CandidateIdentity& identity)
{
    const auto found = std::find_if(candidates.begin(), candidates.end(),
        [&identity](const auto& candidate)
        {
            return SameCandidate(identity, candidate);
        });
    return found == candidates.end() ? std::nullopt
                                     : std::optional(*found);
}

bool Terminal(const TG2::OutcomeResolution& outcome)
{
    return outcome.state != TG2::ResolutionState::Pending;
}

bool Terminal(const ObservationRecord& record)
{
    return Terminal(record.behavior.retest) &&
        Terminal(record.behavior.outerTarget) &&
        Terminal(record.behavior.outerTargetAfterRetest);
}

void CensorMissing(TG2::OutcomeResolution& outcome,
                   std::size_t bar,
                   std::int64_t timestamp,
                   std::size_t breakBar)
{
    if (outcome.state != TG2::ResolutionState::Pending) return;
    outcome.state = TG2::ResolutionState::Censored;
    outcome.censorReason = TG2::CensorReason::CapacityEviction;
    outcome.resolutionBar = bar;
    outcome.resolutionTimestamp = timestamp;
    if (bar >= breakBar) outcome.latencyBars = bar - breakBar;
}

void AddOutcome(OutcomeTally& tally, const TG2::OutcomeResolution& outcome)
{
    if (outcome.state == TG2::ResolutionState::StructurallyIneligible)
    {
        ++tally.structurallyIneligible;
        return;
    }
    ++tally.structurallyEligible;
    switch (outcome.state)
    {
        case TG2::ResolutionState::Pending: ++tally.pending; break;
        case TG2::ResolutionState::Censored: ++tally.censored; break;
        case TG2::ResolutionState::Succeeded:
            ++tally.resolved;
            ++tally.successes;
            break;
        case TG2::ResolutionState::Failed:
            ++tally.resolved;
            ++tally.failures;
            break;
        case TG2::ResolutionState::StructurallyIneligible: break;
    }
}

void AddTally(CohortTally& tally, const ObservationRecord& record)
{
    ++tally.totalObservations;
    if (record.behavior.pairedOuter.has_value()) ++tally.paired;
    else ++tally.unpaired;
    switch (record.confluence.confluenceState)
    {
        case TG3::ConfluenceState::Confluence: ++tally.confluent; break;
        case TG3::ConfluenceState::NoConfluence: ++tally.nonConfluent; break;
        case TG3::ConfluenceState::StructurallyIneligible:
            ++tally.confluenceIneligible;
            break;
    }
    AddOutcome(tally.retest, record.behavior.retest);
    AddOutcome(tally.outerTarget, record.behavior.outerTarget);
    AddOutcome(tally.outerTargetAfterRetest,
               record.behavior.outerTargetAfterRetest);
}

std::vector<std::string> CohortsFor(const ObservationRecord& record)
{
    std::vector<std::string> result{"all"};
    result.push_back(record.behavior.pairedOuter.has_value()
        ? "pairing=paired" : "pairing=unpaired");
    const std::string retest = "retest=" + ResolutionName(
        record.behavior.retest.state);
    result.push_back(retest);
    const std::string confluence = "confluence=" + ConfluenceName(
        record.confluence.confluenceState);
    result.push_back(confluence);
    result.push_back(retest + "+" + confluence);
    result.push_back("inner_class=" + ClassificationName(
        record.behavior.breakEvent.frozenClassification));
    result.push_back("outer_class=" + (record.frozenOuter.has_value()
        ? ClassificationName(record.frozenOuter->classification)
        : std::string("unpaired")));
    return result;
}

std::vector<std::tuple<std::string, std::string, std::string>> ContextsFor(
    const ObservationRecord& record)
{
    const std::string partition = TemporalPartitionName(record.partition);
    const std::string direction = DirectionName(
        record.behavior.breakEvent.candidate.direction);
    return {
        {record.symbol, partition, direction},
        {record.symbol, partition, "all"},
        {record.symbol, "all", "all"},
        {"__event_weighted__", partition, direction},
        {"__event_weighted__", partition, "all"},
        {"__event_weighted__", "all", "all"}};
}

std::string OutcomeFields(const OutcomeTally& value)
{
    const WilsonInterval interval = Wilson95(value.successes, value.failures);
    std::ostringstream output;
    output << value.structurallyEligible << ','
           << value.structurallyIneligible << ',' << value.pending << ','
           << value.censored << ',' << value.resolved << ','
           << value.successes << ',' << value.failures << ','
           << interval.denominator << ','
           << OptionalNumber(interval.rate) << ','
           << OptionalNumber(interval.lower95) << ','
           << OptionalNumber(interval.upper95);
    return output.str();
}

std::string FinalRecordState(const ObservationRecord& record)
{
    const std::array<TG2::OutcomeResolution, 3> outcomes{
        record.behavior.retest, record.behavior.outerTarget,
        record.behavior.outerTargetAfterRetest};
    if (std::any_of(outcomes.begin(), outcomes.end(), [](const auto& value)
        { return value.state == TG2::ResolutionState::Pending; }))
        return "pending";
    if (std::any_of(outcomes.begin(), outcomes.end(), [](const auto& value)
        { return value.state == TG2::ResolutionState::Censored; }))
        return "censored";
    if (!record.behavior.pairedOuter.has_value())
        return "resolved_with_structural_ineligibility";
    return "resolved";
}

std::string NearestRatio(const TG3::ConfluenceObservation& observation)
{
    if (observation.levels.empty()) return {};
    const auto found = std::min_element(
        observation.levels.begin(), observation.levels.end(),
        [](const auto& left, const auto& right)
        {
            if (left.rawPriceDistance != right.rawPriceDistance)
                return left.rawPriceDistance < right.rawPriceDistance;
            return left.ratio < right.ratio;
        });
    return Number(found->ratio);
}

std::string Bool(bool value) { return value ? "true" : "false"; }

void RenameComplete(const std::filesystem::path& temporary,
                    const std::filesystem::path& final)
{
    std::error_code error;
    std::filesystem::rename(temporary, final, error);
    if (error)
        throw std::runtime_error("TG4 could not finalize artifact '" +
                                 final.string() + "': " + error.message());
}

} // namespace

bool CohortKey::operator<(const CohortKey& other) const
{
    return std::tie(symbol, partition, direction, cohort) <
        std::tie(other.symbol, other.partition, other.direction, other.cohort);
}

void AngleMoments::Add(double value)
{
    if (!std::isfinite(value)) return;
    if (count == 0)
    {
        minimum = value;
        maximum = value;
        mean = value;
        count = 1;
        return;
    }
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (value - mean);
}

std::optional<double> AngleMoments::SampleStandardDeviation() const
{
    return count < 2 ? std::nullopt
                     : std::optional(std::sqrt(m2 /
                           static_cast<double>(count - 1)));
}

void AggregateAccumulator::Add(const ObservationRecord& record)
{
    const auto contexts = ContextsFor(record);
    const auto cohorts = CohortsFor(record);
    for (const auto& [symbol, partition, direction] : contexts)
    {
        for (const std::string& cohort : cohorts)
        {
            const CohortKey key{symbol, partition, direction, cohort};
            AddTally(cohorts_[key], record);
            AngleTally& angles = angles_[key];
            if (record.frozenInner.has_value() &&
                record.frozenInner->calibratedAngleMagnitudeDegrees.has_value())
                angles.inner.Add(
                    *record.frozenInner->calibratedAngleMagnitudeDegrees);
            if (record.frozenOuter.has_value() &&
                record.frozenOuter->calibratedAngleMagnitudeDegrees.has_value())
                angles.outer.Add(
                    *record.frozenOuter->calibratedAngleMagnitudeDegrees);
        }

        const auto outer = record.behavior.outerTarget.state;
        const bool resolvedOuter = outer == TG2::ResolutionState::Succeeded ||
            outer == TG2::ResolutionState::Failed;
        const bool success = outer == TG2::ResolutionState::Succeeded;
        const auto retest = record.behavior.retest.state;
        const bool resolvedRetest = retest == TG2::ResolutionState::Succeeded ||
            retest == TG2::ResolutionState::Failed;
        if (record.behavior.pairedOuter.has_value() && resolvedOuter &&
            resolvedRetest)
        {
            ComparisonCounts& counts = comparisons_[
                {symbol, partition, direction, "retest_vs_no_retest"}];
            if (retest == TG2::ResolutionState::Succeeded)
                success ? ++counts.groupASuccesses : ++counts.groupAFailures;
            else
                success ? ++counts.groupBSuccesses : ++counts.groupBFailures;
        }
        const auto confluence = record.confluence.confluenceState;
        const bool confluenceComparable =
            confluence == TG3::ConfluenceState::Confluence ||
            confluence == TG3::ConfluenceState::NoConfluence;
        if (record.behavior.pairedOuter.has_value() && resolvedOuter &&
            confluenceComparable)
        {
            ComparisonCounts& counts = comparisons_[
                {symbol, partition, direction, "confluence_vs_no_confluence"}];
            if (confluence == TG3::ConfluenceState::Confluence)
                success ? ++counts.groupASuccesses : ++counts.groupAFailures;
            else
                success ? ++counts.groupBSuccesses : ++counts.groupBFailures;
        }
        if (record.behavior.pairedOuter.has_value() && resolvedOuter &&
            resolvedRetest && confluenceComparable)
        {
            ComparisonCounts& counts = comparisons_[
                {symbol, partition, direction,
                 "retest_and_confluence_vs_other_comparable"}];
            const bool groupA = retest == TG2::ResolutionState::Succeeded &&
                confluence == TG3::ConfluenceState::Confluence;
            if (groupA)
                success ? ++counts.groupASuccesses : ++counts.groupAFailures;
            else
                success ? ++counts.groupBSuccesses : ++counts.groupBFailures;
        }
    }
}

const std::map<CohortKey, CohortTally>& AggregateAccumulator::Cohorts() const
{
    return cohorts_;
}

const std::map<CohortKey, AngleTally>& AggregateAccumulator::Angles() const
{
    return angles_;
}

const std::map<CohortKey, ComparisonCounts>&
AggregateAccumulator::Comparisons() const
{
    return comparisons_;
}

HistoricalEvaluator::HistoricalEvaluator(
    std::string symbol,
    EvaluationConfiguration configuration,
    TemporalRange range,
    RecordSink sink)
    : symbol_(std::move(symbol)),
      configuration_(std::move(configuration)),
      range_(range),
      sink_(std::move(sink)),
      integration_(
          TG1B::CalibrationConfiguration(configuration_.referenceBarScale),
          configuration_.fibonacci,
          configuration_.geometry,
          configuration_.behavior,
          {symbol_, configuration_.timeframe})
{
    ValidateConfiguration(configuration_);
    ValidateTemporalRange(range_);
    if (!sink_) throw std::invalid_argument("TG4 requires a record sink");
    dataQuality_.symbol = symbol_;
}

void HistoricalEvaluator::UpdateDataQuality(const TG1A::Candle& candle)
{
    ++dataQuality_.sourceRows;
    if (lastTimestamp_.has_value())
    {
        if (candle.timestamp == *lastTimestamp_)
        {
            ++dataQuality_.duplicateTimestamps;
            throw std::invalid_argument(
                "TG4 rejects duplicate completed-bar timestamps for " + symbol_);
        }
        if (candle.timestamp < *lastTimestamp_)
        {
            ++dataQuality_.outOfOrderTimestamps;
            throw std::invalid_argument(
                "TG4 rejects out-of-order completed-bar timestamps for " + symbol_);
        }
        const std::int64_t gap = candle.timestamp - *lastTimestamp_;
        const double threshold =
            static_cast<double>(configuration_.expectedIntervalSeconds) *
            configuration_.materialGapMultiple;
        if (static_cast<double>(gap) > threshold)
        {
            const bool scored = candle.timestamp >= range_.scoreStart &&
                *lastTimestamp_ < range_.scoreEnd;
            dataQuality_.materialGaps.push_back(
                {*lastTimestamp_, candle.timestamp, gap, scored});
        }
    }
    lastTimestamp_ = candle.timestamp;
    ++dataQuality_.usableRows;
    if (!dataQuality_.firstUsableTimestamp.has_value())
        dataQuality_.firstUsableTimestamp = candle.timestamp;
    dataQuality_.lastUsableTimestamp = candle.timestamp;
    if (candle.timestamp < range_.scoreStart) ++dataQuality_.warmupRows;
    else if (candle.timestamp < range_.scoreEnd) ++dataQuality_.scoredRows;
    else ++dataQuality_.outcomeOnlyRows;
    ++dataQuality_.partitionRows[PartitionForTimestamp(candle.timestamp)];
}

void HistoricalEvaluator::AddCompletedBar(const TG1A::Candle& candle)
{
    if (finalized_)
        throw std::logic_error("TG4 cannot accept bars after finalization");
    if (candle.timestamp < range_.warmupStart ||
        candle.timestamp >= range_.outcomeEnd)
        throw std::invalid_argument("TG4 bar is outside the declared query range");
    UpdateDataQuality(candle);
    const TG3::Update update = integration_.AddCompletedBar(candle);
    SynchronizePending(update.bar, candle.timestamp);
    CaptureNewRecords(update);
    FlushTerminalPrefix();
}

void HistoricalEvaluator::SynchronizePending(
    std::size_t currentBar, std::int64_t currentTimestamp)
{
    const auto& behavior = integration_.Behavior().Observations();
    const auto& confluence = integration_.Observations();
    for (ObservationRecord& record : pending_)
    {
        const std::uint64_t sequence =
            record.behavior.breakEvent.eventSequence;
        const auto behaviorFound = std::find_if(
            behavior.begin(), behavior.end(), [sequence](const auto& value)
            {
                return value.breakEvent.eventSequence == sequence;
            });
        if (behaviorFound != behavior.end())
        {
            record.behavior.retest = behaviorFound->retest;
            record.behavior.outerTarget = behaviorFound->outerTarget;
            record.behavior.outerTargetAfterRetest =
                behaviorFound->outerTargetAfterRetest;
        }
        else
        {
            const std::size_t breakBar = record.behavior.breakEvent.bar;
            CensorMissing(record.behavior.retest, currentBar,
                          currentTimestamp, breakBar);
            CensorMissing(record.behavior.outerTarget, currentBar,
                          currentTimestamp, breakBar);
            CensorMissing(record.behavior.outerTargetAfterRetest, currentBar,
                          currentTimestamp, breakBar);
        }
        const auto confluenceFound = std::find_if(
            confluence.begin(), confluence.end(), [sequence](const auto& value)
            {
                return value.breakEventSequence == sequence;
            });
        if (confluenceFound != confluence.end())
        {
            record.confluence.retest = confluenceFound->retest;
            record.confluence.outerTarget = confluenceFound->outerTarget;
            record.confluence.outerTargetAfterRetest =
                confluenceFound->outerTargetAfterRetest;
        }
    }
}

void HistoricalEvaluator::CaptureNewRecords(const TG3::Update& update)
{
    const auto& behavior = integration_.Behavior().Observations();
    const auto& confluence = integration_.Observations();
    const auto& candidates = integration_.Behavior().Classification()
        .ClassifiedCandidates();
    for (const std::uint64_t sequence : update.newConfluenceObservations)
    {
        const auto behaviorFound = std::find_if(
            behavior.begin(), behavior.end(), [sequence](const auto& value)
            {
                return value.breakEvent.eventSequence == sequence;
            });
        const auto confluenceFound = std::find_if(
            confluence.begin(), confluence.end(), [sequence](const auto& value)
            {
                return value.breakEventSequence == sequence;
            });
        if (behaviorFound == behavior.end() ||
            confluenceFound == confluence.end())
            throw std::logic_error("TG4 cannot locate a newly emitted TG3 record");
        const std::int64_t eventTimestamp =
            behaviorFound->breakEvent.timestamp;
        if (eventTimestamp < range_.scoreStart ||
            eventTimestamp >= range_.scoreEnd)
            continue;

        ObservationRecord record;
        record.symbol = symbol_;
        record.timeframe = configuration_.timeframe;
        record.partition = PartitionForTimestamp(eventTimestamp);
        record.behavior = *behaviorFound;
        record.confluence = *confluenceFound;
        record.frozenInner = FindCandidate(
            candidates, behaviorFound->breakEvent.candidate);
        if (!record.frozenInner.has_value())
            throw std::logic_error(
                "TG4 cannot freeze the Inner classification at break time");
        if (behaviorFound->pairedOuter.has_value())
        {
            record.frozenOuter = FindCandidate(
                candidates, behaviorFound->pairedOuter->candidate);
            if (!record.frozenOuter.has_value())
                throw std::logic_error(
                    "TG4 cannot freeze the paired Outer classification at break time");
            record.outerPairingEligibilityReason =
                "eligible_and_paired_by_tg2_policy";
        }
        else
        {
            record.outerPairingEligibilityReason =
                "no_eligible_coexisting_outer_beyond_break_candle";
        }
        record.eventIdentity = EventIdentity(
            symbol_, configuration_.timeframe, behaviorFound->breakEvent);
        pending_.push_back(std::move(record));
        peakPendingRecordCount_ = std::max(
            peakPendingRecordCount_, pending_.size());
        if (pending_.size() > configuration_.maxPendingTG4Records)
            throw std::runtime_error(
                "TG4 pending-record bound exceeded; refusing to discard evidence");
    }
}

void HistoricalEvaluator::FlushTerminalPrefix()
{
    while (!pending_.empty() && Terminal(pending_.front()))
    {
        sink_(pending_.front());
        pending_.pop_front();
        ++emittedRecordCount_;
    }
}

void HistoricalEvaluator::Finalize()
{
    if (finalized_) return;
    integration_.Finalize();
    const std::size_t currentBar = integration_.Behavior().Classification()
        .Geometry().CompletedBarCount() == 0
        ? 0
        : integration_.Behavior().Classification().Geometry()
              .CompletedBarCount() - 1;
    const std::int64_t timestamp = lastTimestamp_.value_or(0);
    SynchronizePending(currentBar, timestamp);
    FlushTerminalPrefix();
    if (!pending_.empty())
        throw std::logic_error(
            "TG4 finalization left a pending observation unexpectedly");
    finalized_ = true;
}

const DataQualityAudit& HistoricalEvaluator::DataQuality() const
{
    return dataQuality_;
}

const std::deque<ObservationRecord>& HistoricalEvaluator::PendingRecords() const
{
    return pending_;
}

std::size_t HistoricalEvaluator::EmittedRecordCount() const
{
    return emittedRecordCount_;
}

std::size_t HistoricalEvaluator::PeakPendingRecordCount() const
{
    return peakPendingRecordCount_;
}

bool HistoricalEvaluator::IsFinalized() const { return finalized_; }

EvaluationConfiguration LoadConfigurationFile(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("TG4 cannot open configuration: " +
                                 path.string());
    std::map<std::string, std::string> values;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        line = Trim(line);
        if (line.empty() || line.front() == '#') continue;
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos)
            throw std::invalid_argument("TG4 configuration line " +
                std::to_string(lineNumber) + " lacks '='");
        const std::string key = Trim(line.substr(0, equals));
        const std::string value = Trim(line.substr(equals + 1));
        if (key.empty() || !values.emplace(key, value).second)
            throw std::invalid_argument("TG4 configuration has an empty or duplicate key");
    }

    static const std::set<std::string> allowed{
        "configuration_schema", "name", "provenance", "timeframe",
        "candle_period", "candle_unit", "expected_interval_seconds",
        "material_gap_multiple", "minimum_human_report_resolved_n",
        "max_pending_tg4_records", "tg1_fractal_semantics",
        "tg1_intervening_price_tolerance", "tg1_touch_price_tolerance",
        "tg1_max_fractal_anchor_lookback_bars",
        "tg1_max_confirmed_fractals_per_kind",
        "tg1_max_candidate_age_bars", "tg1_max_candidates",
        "tg1_atr_period", "tg1b_reference_bar_scale", "tg1b_angle_bands",
        "tg2_break_policy", "tg2_rearm_policy", "tg2_retest_contact_policy",
        "tg2_outer_pairing_policy", "tg2_break_price_tolerance",
        "tg2_retest_price_tolerance", "tg2_outer_target_price_tolerance",
        "tg2_retest_horizon_bars", "tg2_outer_target_horizon_bars",
        "tg2_max_active_break_observations",
        "tg2_max_retained_break_observations", "tg3_ab_policy",
        "tg3_retracement_ratios", "tg3_ratio_provenance",
        "tg3_absolute_price_tolerance", "tg3_confluence_policy",
        "tg3_directional_study_policy", "tg3_max_confirmed_fractals_per_kind",
        "tg3_max_ab_age_bars", "tg3_max_active_ab_structures",
        "tg3_max_active_confluence_observations",
        "tg3_max_retained_confluence_observations"};
    for (const auto& [key, value] : values)
    {
        (void)value;
        if (!allowed.contains(key))
            throw std::invalid_argument("TG4 configuration has unknown key '" +
                                        key + "'");
    }

    EvaluationConfiguration result;
    result.configurationSchema = Require(values, "configuration_schema");
    result.name = Require(values, "name");
    result.provenance = Require(values, "provenance");
    result.timeframe = Require(values, "timeframe");
    result.candlePeriod = ParseInteger<int>(values, "candle_period");
    result.candleUnit = Require(values, "candle_unit");
    result.expectedIntervalSeconds = ParseInteger<std::int64_t>(
        values, "expected_interval_seconds");
    result.materialGapMultiple = ParseDouble(values, "material_gap_multiple");
    result.minimumHumanReportResolvedN = ParseInteger<std::size_t>(
        values, "minimum_human_report_resolved_n");
    result.maxPendingTG4Records = ParseInteger<std::size_t>(
        values, "max_pending_tg4_records");

    RequireValue(values, "tg1_fractal_semantics",
                 "strict_five_completed_candles_radius_2");
    result.geometry.interveningPriceTolerance = ParseDouble(
        values, "tg1_intervening_price_tolerance");
    result.geometry.touchPriceTolerance = ParseDouble(
        values, "tg1_touch_price_tolerance");
    result.geometry.maxFractalAnchorLookbackBars = ParseInteger<std::size_t>(
        values, "tg1_max_fractal_anchor_lookback_bars");
    result.geometry.maxConfirmedFractalsPerKind = ParseInteger<std::size_t>(
        values, "tg1_max_confirmed_fractals_per_kind");
    result.geometry.maxCandidateAgeBars = ParseInteger<std::size_t>(
        values, "tg1_max_candidate_age_bars");
    result.geometry.maxCandidates = ParseInteger<std::size_t>(
        values, "tg1_max_candidates");
    result.geometry.atrPeriod = ParseInteger<std::size_t>(values, "tg1_atr_period");
    result.referenceBarScale = ParseDouble(values, "tg1b_reference_bar_scale");
    RequireValue(values, "tg1b_angle_bands",
        "long_term_12_20_outer_25_40_inner_45_85_degrees_inclusive");

    const std::string breakPolicy = Require(values, "tg2_break_policy");
    if (breakPolicy == "completed_close_beyond_line")
        result.behavior.breakPolicy = TG2::BreakPolicy::CompletedCloseBeyondLine;
    else if (breakPolicy == "completed_wick_beyond_line")
        result.behavior.breakPolicy = TG2::BreakPolicy::CompletedWickBeyondLine;
    else
        throw std::invalid_argument("TG4 configuration has unknown TG2 break policy");
    RequireValue(values, "tg2_rearm_policy",
                 "completed_bar_returns_to_valid_side");
    RequireValue(values, "tg2_retest_contact_policy",
                 "wick_reaches_projected_line_from_broken_side");
    RequireValue(values, "tg2_outer_pairing_policy",
                 "nearest_coexisting_outer_beyond_break_candle");
    result.behavior.breakPriceTolerance = ParseDouble(
        values, "tg2_break_price_tolerance");
    result.behavior.retestPriceTolerance = ParseDouble(
        values, "tg2_retest_price_tolerance");
    result.behavior.outerTargetPriceTolerance = ParseDouble(
        values, "tg2_outer_target_price_tolerance");
    result.behavior.retestHorizonBars = ParseInteger<std::size_t>(
        values, "tg2_retest_horizon_bars");
    result.behavior.outerTargetHorizonBars = ParseInteger<std::size_t>(
        values, "tg2_outer_target_horizon_bars");
    result.behavior.maxActiveBreakObservations = ParseInteger<std::size_t>(
        values, "tg2_max_active_break_observations");
    result.behavior.maxRetainedBreakObservations = ParseInteger<std::size_t>(
        values, "tg2_max_retained_break_observations");

    RequireValue(values, "tg3_ab_policy",
                 "most_recent_prior_opposite_confirmed_fractal");
    result.fibonacci.retracementRatios = ParseRatios(
        Require(values, "tg3_retracement_ratios"));
    const std::string ratioProvenance = Require(values, "tg3_ratio_provenance");
    if (ratioProvenance.find("experimental") == std::string::npos)
        throw std::invalid_argument(
            "TG4 ratio provenance must explicitly contain 'experimental'");
    result.provenance += "; fibonacci_ratios=" + ratioProvenance;
    result.fibonacci.absolutePriceTolerance = ParseDouble(
        values, "tg3_absolute_price_tolerance");
    RequireValue(values, "tg3_confluence_policy",
                 "absolute_price_tolerance_around_exact_retracement_level");
    const std::string directional = Require(
        values, "tg3_directional_study_policy");
    if (directional == "source_utl_up_ab_only")
        result.fibonacci.directionalStudyPolicy =
            TG3::DirectionalStudyPolicy::SourceUTLUpABOnly;
    else if (directional == "symmetric_directional_diagnostic_hypothesis")
        result.fibonacci.directionalStudyPolicy =
            TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    else
        throw std::invalid_argument(
            "TG4 configuration has unknown TG3 directional policy");
    result.fibonacci.maxConfirmedFractalsPerKind = ParseInteger<std::size_t>(
        values, "tg3_max_confirmed_fractals_per_kind");
    result.fibonacci.maxABAgeBars = ParseInteger<std::size_t>(
        values, "tg3_max_ab_age_bars");
    result.fibonacci.maxActiveABStructures = ParseInteger<std::size_t>(
        values, "tg3_max_active_ab_structures");
    result.fibonacci.maxActiveConfluenceObservations = ParseInteger<std::size_t>(
        values, "tg3_max_active_confluence_observations");
    result.fibonacci.maxRetainedConfluenceObservations = ParseInteger<std::size_t>(
        values, "tg3_max_retained_confluence_observations");
    ValidateConfiguration(result);
    return result;
}

void ValidateConfiguration(const EvaluationConfiguration& value)
{
    if (value.configurationSchema != "tg4-analysis-configuration-v1")
        throw std::invalid_argument("TG4 configuration schema is unsupported");
    if (value.name.empty() || value.provenance.empty() || value.timeframe.empty())
        throw std::invalid_argument("TG4 configuration identity is incomplete");
    if (value.provenance.find("experimental") == std::string::npos)
        throw std::invalid_argument(
            "TG4 configuration provenance must explicitly contain 'experimental'");
    if (value.candlePeriod <= 0 || value.candleUnit.empty() ||
        value.expectedIntervalSeconds <= 0 ||
        !std::isfinite(value.materialGapMultiple) ||
        value.materialGapMultiple <= 1.0 ||
        value.minimumHumanReportResolvedN == 0 ||
        value.maxPendingTG4Records == 0 ||
        !std::isfinite(value.referenceBarScale) ||
        value.referenceBarScale <= 0.0)
        throw std::invalid_argument("TG4 configuration bounds are invalid");
    if (value.maxPendingTG4Records <
        value.behavior.maxActiveBreakObservations)
        throw std::invalid_argument(
            "TG4 pending bound must cover the TG2 active-observation bound");
    (void)TG1B::CalibrationConfiguration(value.referenceBarScale);
    (void)TG1A::CausalFractalTrendLineGeometry(value.geometry);
    (void)TG2::TrendLineBehaviorTracker(value.behavior);
    (void)TG3::FibonacciConfluenceTracker(value.fibonacci);
}

void ValidateTemporalRange(const TemporalRange& range)
{
    if (!(range.warmupStart <= range.scoreStart &&
          range.scoreStart < range.scoreEnd &&
          range.scoreEnd <= range.outcomeEnd))
        throw std::invalid_argument(
            "TG4 requires warmup_start <= score_start < score_end <= outcome_end");
    if (range.scoreEnd > kNamedStudyEnd)
        throw std::invalid_argument(
            "TG4 v1 refuses scored observations after 2025-12-31");
}

TemporalPartition PartitionForTimestamp(std::int64_t timestamp)
{
    if (timestamp >= kExploratoryStart && timestamp < kCalibrationStart)
        return TemporalPartition::Exploratory2010To2019;
    if (timestamp >= kCalibrationStart && timestamp < kValidationStart)
        return TemporalPartition::Calibration2020To2022;
    if (timestamp >= kValidationStart && timestamp < kConfirmationStart)
        return TemporalPartition::Validation2023To2024;
    if (timestamp >= kConfirmationStart && timestamp < kNamedStudyEnd)
        return TemporalPartition::Confirmation2025;
    return TemporalPartition::OutsideNamedStudy;
}

std::string TemporalPartitionName(TemporalPartition partition)
{
    switch (partition)
    {
        case TemporalPartition::Exploratory2010To2019:
            return "exploratory_2010_2019";
        case TemporalPartition::Calibration2020To2022:
            return "calibration_2020_2022";
        case TemporalPartition::Validation2023To2024:
            return "validation_2023_2024";
        case TemporalPartition::Confirmation2025:
            return "confirmation_2025";
        case TemporalPartition::OutsideNamedStudy:
            return "outside_named_study";
    }
    return "outside_named_study";
}

std::int64_t ParseUtcDateOrTimestamp(std::string_view text)
{
    if (text.size() != 10 && text.size() != 20)
        throw std::invalid_argument(
            "TG4 timestamps must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ");
    const auto parse = [text](std::size_t offset, std::size_t length)
    {
        int value = 0;
        const auto result = std::from_chars(
            text.data() + static_cast<std::ptrdiff_t>(offset),
            text.data() + static_cast<std::ptrdiff_t>(offset + length), value);
        if (result.ec != std::errc{} ||
            result.ptr != text.data() + static_cast<std::ptrdiff_t>(offset + length))
            throw std::invalid_argument("TG4 timestamp contains invalid digits");
        return value;
    };
    if (text[4] != '-' || text[7] != '-')
        throw std::invalid_argument("TG4 timestamp has an invalid date shape");
    const int year = parse(0, 4);
    const int month = parse(5, 2);
    const int day = parse(8, 2);
    if (month < 1 || month > 12 || day < 1 || day > DaysInMonth(year, month))
        throw std::invalid_argument("TG4 timestamp has an invalid date");
    int hour = 0;
    int minute = 0;
    int second = 0;
    if (text.size() == 20)
    {
        if (text[10] != 'T' || text[13] != ':' || text[16] != ':' ||
            text[19] != 'Z')
            throw std::invalid_argument("TG4 timestamp must be UTC with Z suffix");
        hour = parse(11, 2);
        minute = parse(14, 2);
        second = parse(17, 2);
        if (hour > 23 || minute > 59 || second > 59)
            throw std::invalid_argument("TG4 timestamp has an invalid time");
    }
    return DaysFromCivil(year, static_cast<unsigned>(month),
                         static_cast<unsigned>(day)) * 86400 +
        hour * 3600 + minute * 60 + second;
}

std::string FormatUtcTimestamp(std::int64_t timestamp)
{
    const std::time_t raw = static_cast<std::time_t>(timestamp);
    std::tm utc{};
    if (gmtime_r(&raw, &utc) == nullptr) return std::to_string(timestamp);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

WilsonInterval Wilson95(std::size_t successes, std::size_t failures)
{
    WilsonInterval result;
    result.numerator = successes;
    result.denominator = successes + failures;
    if (result.denominator == 0) return result;
    constexpr double z = 1.959963984540054;
    const double n = static_cast<double>(result.denominator);
    const double p = static_cast<double>(successes) / n;
    const double z2 = z * z;
    const double denominator = 1.0 + z2 / n;
    const double center = (p + z2 / (2.0 * n)) / denominator;
    const double half = z * std::sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) /
        denominator;
    result.rate = p;
    result.lower95 = std::max(0.0, center - half);
    result.upper95 = std::min(1.0, center + half);
    return result;
}

std::string ObservationCsvHeader()
{
    return JoinCsv({
        "schema_version", "event_identity", "symbol", "timeframe",
        "temporal_partition", "event_sequence", "inner_candidate_identity",
        "outer_candidate_identity", "direction", "inner_frozen_class",
        "inner_frozen_angle_degrees", "inner_creation_atr",
        "inner_creation_atr_normalized_slope", "outer_frozen_class",
        "outer_frozen_angle_degrees", "outer_creation_atr",
        "outer_creation_atr_normalized_slope", "inner_anchor1_bar",
        "inner_anchor1_timestamp", "inner_anchor1_price", "inner_anchor2_bar",
        "inner_anchor2_timestamp", "inner_anchor2_price", "inner_creation_bar",
        "inner_creation_timestamp", "break_bar", "break_timestamp",
        "break_policy", "break_tolerance", "break_projected_line_price",
        "break_observed_component", "break_penetration",
        "break_directional_distance", "break_open", "break_high", "break_low",
        "break_close", "retest_policy", "retest_tolerance",
        "retest_horizon_bars", "retest_state", "retest_resolution_bar",
        "retest_resolution_timestamp", "retest_latency_bars",
        "retest_projected_line_price", "retest_censor_reason",
        "outer_pairing_policy", "outer_pairing_eligible", "paired_outer",
        "outer_pairing_eligibility_reason", "outer_projection_at_break",
        "outer_target_tolerance", "outer_target_horizon_bars",
        "outer_target_state", "outer_target_resolution_bar",
        "outer_target_resolution_timestamp", "outer_target_latency_bars",
        "outer_target_projected_line_price", "outer_target_censor_reason",
        "retest_then_outer_state", "retest_then_outer_resolution_bar",
        "retest_then_outer_resolution_timestamp", "retest_then_outer_latency_bars",
        "retest_then_outer_projected_line_price",
        "retest_then_outer_censor_reason", "tg3_ab_policy",
        "tg3_ab_direction", "tg3_a_bar", "tg3_a_timestamp", "tg3_a_price",
        "tg3_a_confirmation_bar", "tg3_a_confirmation_timestamp", "tg3_b_bar",
        "tg3_b_timestamp", "tg3_b_price", "tg3_b_confirmation_bar",
        "tg3_b_confirmation_timestamp", "tg3_ab_availability_bar",
        "tg3_ab_availability_timestamp", "tg3_ratio_set",
        "tg3_ratio_configuration_provenance", "tg3_directional_policy",
        "tg3_confluence_policy", "tg3_absolute_price_tolerance",
        "tg3_confluence_state", "tg3_ineligible_reason", "tg3_matched_ratios",
        "tg3_nearest_ratio", "tg3_minimum_raw_price_distance",
        "tg3_observation_atr", "tg3_minimum_atr_normalized_distance",
        "tg3_level_diagnostics", "final_record_state"});
}

std::string ObservationCsvRow(
    const ObservationRecord& record,
    const EvaluationConfiguration& configuration)
{
    const auto& event = record.behavior.breakEvent;
    const auto& candidate = event.candidate;
    const auto& retest = record.behavior.retest;
    const auto& outerTarget = record.behavior.outerTarget;
    const auto& conditioned = record.behavior.outerTargetAfterRetest;
    const auto& confluence = record.confluence;
    const std::optional<TG2::PairedOuterLine>& outer =
        record.behavior.pairedOuter;
    const std::optional<TG3::ABStructure>& ab = confluence.selectedAB;
    const double innerAnchor2Price = record.behavior.innerAnchor1Price +
        record.behavior.innerRawPriceSlopePerBar *
        (static_cast<double>(candidate.anchor2Bar) -
         static_cast<double>(candidate.anchor1Bar));

    std::vector<std::string> fields;
    fields.reserve(100);
    fields.insert(fields.end(), {
        std::string(kObservationSchemaVersion), record.eventIdentity,
        record.symbol, record.timeframe, TemporalPartitionName(record.partition),
        std::to_string(event.eventSequence), CandidateIdentityText(candidate),
        outer.has_value() ? CandidateIdentityText(outer->candidate) : "",
        DirectionName(candidate.direction), ClassificationName(event.frozenClassification),
        record.frozenInner.has_value()
            ? OptionalNumber(record.frozenInner->calibratedAngleMagnitudeDegrees) : "",
        record.frozenInner.has_value()
            ? OptionalNumber(record.frozenInner->creationAtr) : "",
        record.frozenInner.has_value()
            ? OptionalNumber(record.frozenInner->creationAtrNormalizedSlope) : "",
        record.frozenOuter.has_value()
            ? ClassificationName(record.frozenOuter->classification) : "",
        record.frozenOuter.has_value()
            ? OptionalNumber(record.frozenOuter->calibratedAngleMagnitudeDegrees) : "",
        record.frozenOuter.has_value()
            ? OptionalNumber(record.frozenOuter->creationAtr) : "",
        record.frozenOuter.has_value()
            ? OptionalNumber(record.frozenOuter->creationAtrNormalizedSlope) : "",
        std::to_string(candidate.anchor1Bar), FormatUtcTimestamp(candidate.anchor1Timestamp),
        Number(record.behavior.innerAnchor1Price), std::to_string(candidate.anchor2Bar),
        FormatUtcTimestamp(candidate.anchor2Timestamp), Number(innerAnchor2Price),
        std::to_string(candidate.creationBar), FormatUtcTimestamp(candidate.creationTimestamp),
        std::to_string(event.bar), FormatUtcTimestamp(event.timestamp),
        BreakPolicyName(event.policy), Number(configuration.behavior.breakPriceTolerance),
        Number(event.projectedLinePrice), Number(event.observedBreakComponent),
        Number(event.penetration), Number(event.directionalDistance),
        Number(event.open), Number(event.high), Number(event.low), Number(event.close),
        "wick_reaches_projected_line_from_broken_side",
        Number(configuration.behavior.retestPriceTolerance),
        std::to_string(configuration.behavior.retestHorizonBars),
        ResolutionName(retest.state), OptionalNumber(retest.resolutionBar),
        OptionalTimestamp(retest.resolutionTimestamp), OptionalNumber(retest.latencyBars),
        OptionalNumber(retest.projectedLinePrice), CensorReasonName(retest.censorReason),
        "nearest_coexisting_outer_beyond_break_candle", Bool(outer.has_value()),
        Bool(outer.has_value()), record.outerPairingEligibilityReason,
        outer.has_value() ? Number(outer->projectedPriceAtBreak) : "",
        Number(configuration.behavior.outerTargetPriceTolerance),
        std::to_string(configuration.behavior.outerTargetHorizonBars),
        ResolutionName(outerTarget.state), OptionalNumber(outerTarget.resolutionBar),
        OptionalTimestamp(outerTarget.resolutionTimestamp),
        OptionalNumber(outerTarget.latencyBars),
        OptionalNumber(outerTarget.projectedLinePrice),
        CensorReasonName(outerTarget.censorReason), ResolutionName(conditioned.state),
        OptionalNumber(conditioned.resolutionBar),
        OptionalTimestamp(conditioned.resolutionTimestamp),
        OptionalNumber(conditioned.latencyBars),
        OptionalNumber(conditioned.projectedLinePrice),
        CensorReasonName(conditioned.censorReason),
        "most_recent_prior_opposite_confirmed_fractal",
        ab.has_value() ? ABDirectionName(ab->identity.direction) : "",
        ab.has_value() ? std::to_string(ab->identity.aBar) : "",
        ab.has_value() ? FormatUtcTimestamp(ab->identity.aTimestamp) : "",
        ab.has_value() ? Number(ab->aPrice) : "",
        ab.has_value() ? std::to_string(ab->aConfirmationBar) : "",
        ab.has_value() ? FormatUtcTimestamp(ab->aConfirmationTimestamp) : "",
        ab.has_value() ? std::to_string(ab->identity.bBar) : "",
        ab.has_value() ? FormatUtcTimestamp(ab->identity.bTimestamp) : "",
        ab.has_value() ? Number(ab->bPrice) : "",
        ab.has_value() ? std::to_string(ab->bConfirmationBar) : "",
        ab.has_value() ? FormatUtcTimestamp(ab->bConfirmationTimestamp) : "",
        ab.has_value() ? std::to_string(ab->identity.availabilityBar) : "",
        ab.has_value() ? FormatUtcTimestamp(ab->identity.availabilityTimestamp) : "",
        RatiosText(configuration.fibonacci.retracementRatios), configuration.provenance,
        DirectionalPolicyName(confluence.directionalStudyPolicy),
        "absolute_price_tolerance_around_exact_retracement_level",
        Number(confluence.absolutePriceTolerance), ConfluenceName(confluence.confluenceState),
        IneligibleReasonName(confluence.ineligibleReason),
        RatiosText(confluence.matchedRatios), NearestRatio(confluence),
        OptionalNumber(confluence.minimumRawPriceDistance),
        OptionalNumber(confluence.observationAtr),
        OptionalNumber(confluence.minimumAtrNormalizedDistance),
        LevelsText(confluence.levels), FinalRecordState(record)});
    return JoinCsv(fields);
}

std::string FrozenCausalSnapshot(const ObservationRecord& record)
{
    std::ostringstream output;
    output << record.eventIdentity << '|'
           << CandidateIdentityText(record.behavior.breakEvent.candidate) << '|'
           << ClassificationName(record.behavior.breakEvent.frozenClassification)
           << '|'
           << (record.frozenInner.has_value()
               ? OptionalNumber(record.frozenInner->calibratedAngleMagnitudeDegrees)
               : "")
           << '|'
           << (record.behavior.pairedOuter.has_value()
               ? CandidateIdentityText(record.behavior.pairedOuter->candidate) : "")
           << '|'
           << (record.frozenOuter.has_value()
               ? OptionalNumber(record.frozenOuter->calibratedAngleMagnitudeDegrees)
               : "")
           << '|' << ConfluenceName(record.confluence.confluenceState)
           << '|' << IneligibleReasonName(record.confluence.ineligibleReason)
           << '|' << RatiosText(record.confluence.matchedRatios)
           << '|' << LevelsText(record.confluence.levels);
    return output.str();
}

std::string EffectiveConfigurationJson(
    const EvaluationConfiguration& configuration,
    const TemporalRange& range,
    std::string_view baselineCommit)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n"
           << "  \"schema_version\": \"tg4-study-metadata-v1\",\n"
           << "  \"study_contract_version\": \"" << kStudyContractVersion << "\",\n"
           << "  \"baseline_commit\": \"" << JsonEscape(baselineCommit) << "\",\n"
           << "  \"configuration_name\": \"" << JsonEscape(configuration.name) << "\",\n"
           << "  \"configuration_provenance\": \"" << JsonEscape(configuration.provenance) << "\",\n"
           << "  \"database_access\": \"read_only_repeatable_read\",\n"
           << "  \"warmup_policy\": \"feed all canonical bars from warmup_start; score only break timestamps in [score_start,score_end)\",\n"
           << "  \"warmup_start\": \"" << FormatUtcTimestamp(range.warmupStart) << "\",\n"
           << "  \"score_start\": \"" << FormatUtcTimestamp(range.scoreStart) << "\",\n"
           << "  \"score_end_exclusive\": \"" << FormatUtcTimestamp(range.scoreEnd) << "\",\n"
           << "  \"outcome_end_exclusive\": \"" << FormatUtcTimestamp(range.outcomeEnd) << "\",\n"
           << "  \"timeframe\": \"" << JsonEscape(configuration.timeframe) << "\",\n"
           << "  \"candle_period\": " << configuration.candlePeriod << ",\n"
           << "  \"candle_unit\": \"" << JsonEscape(configuration.candleUnit) << "\",\n"
           << "  \"expected_interval_seconds\": " << configuration.expectedIntervalSeconds << ",\n"
           << "  \"material_gap_multiple\": " << configuration.materialGapMultiple << ",\n"
           << "  \"minimum_human_report_resolved_n\": " << configuration.minimumHumanReportResolvedN << ",\n"
           << "  \"max_pending_tg4_records\": " << configuration.maxPendingTG4Records << ",\n"
           << "  \"tg1\": {\"fractal_semantics\":\"strict_five_completed_candles_radius_2\","
           << "\"intervening_price_tolerance\":" << configuration.geometry.interveningPriceTolerance
           << ",\"touch_price_tolerance\":" << configuration.geometry.touchPriceTolerance
           << ",\"max_fractal_anchor_lookback_bars\":" << configuration.geometry.maxFractalAnchorLookbackBars
           << ",\"max_confirmed_fractals_per_kind\":" << configuration.geometry.maxConfirmedFractalsPerKind
           << ",\"max_candidate_age_bars\":" << configuration.geometry.maxCandidateAgeBars
           << ",\"max_candidates\":" << configuration.geometry.maxCandidates
           << ",\"atr_period\":" << configuration.geometry.atrPeriod << "},\n"
           << "  \"tg1b\": {\"reference_bar_scale\":" << configuration.referenceBarScale
           << ",\"angle_bands\":\"LongTerm=[12,20],Outer=[25,40],Inner=[45,85] degrees\"},\n"
           << "  \"tg2\": {\"break_policy\":\"" << BreakPolicyName(configuration.behavior.breakPolicy)
           << "\",\"rearm_policy\":\"completed_bar_returns_to_valid_side\""
           << ",\"break_tolerance\":" << configuration.behavior.breakPriceTolerance
           << ",\"retest_policy\":\"wick_reaches_projected_line_from_broken_side\""
           << ",\"retest_tolerance\":" << configuration.behavior.retestPriceTolerance
           << ",\"retest_horizon_bars\":" << configuration.behavior.retestHorizonBars
           << ",\"outer_pairing_policy\":\"nearest_coexisting_outer_beyond_break_candle\""
           << ",\"outer_target_tolerance\":" << configuration.behavior.outerTargetPriceTolerance
           << ",\"outer_target_horizon_bars\":" << configuration.behavior.outerTargetHorizonBars
           << ",\"max_active_break_observations\":" << configuration.behavior.maxActiveBreakObservations
           << ",\"max_retained_break_observations\":" << configuration.behavior.maxRetainedBreakObservations << "},\n"
           << "  \"tg3\": {\"ab_policy\":\"most_recent_prior_opposite_confirmed_fractal\""
           << ",\"ratio_set\":\"" << RatiosText(configuration.fibonacci.retracementRatios) << "\""
           << ",\"ratio_provenance\":\"explicit experimental caller configuration\""
           << ",\"absolute_price_tolerance\":" << configuration.fibonacci.absolutePriceTolerance
           << ",\"confluence_policy\":\"absolute_price_tolerance_around_exact_retracement_level\""
           << ",\"directional_policy\":\"" << DirectionalPolicyName(configuration.fibonacci.directionalStudyPolicy) << "\""
           << ",\"max_confirmed_fractals_per_kind\":" << configuration.fibonacci.maxConfirmedFractalsPerKind
           << ",\"max_ab_age_bars\":" << configuration.fibonacci.maxABAgeBars
           << ",\"max_active_ab_structures\":" << configuration.fibonacci.maxActiveABStructures
           << ",\"max_active_confluence_observations\":" << configuration.fibonacci.maxActiveConfluenceObservations
           << ",\"max_retained_confluence_observations\":" << configuration.fibonacci.maxRetainedConfluenceObservations << "},\n"
           << "  \"model_semantics_changed\": false,\n"
           << "  \"parameter_optimization_performed\": false,\n"
           << "  \"confirmation_period_used_for_selection\": false\n"
           << "}\n";
    return output.str();
}

class StudyArtifactWriter::Implementation
{
public:
    Implementation(std::filesystem::path directory,
                   EvaluationConfiguration config,
                   TemporalRange temporalRange,
                   std::string baseline)
        : outputDirectory(std::move(directory)),
          configuration(std::move(config)),
          range(temporalRange),
          baselineCommit(std::move(baseline))
    {
        std::filesystem::create_directories(outputDirectory);
        observationsFinal = outputDirectory / "observations.csv";
        observationsTemporary = outputDirectory / "observations.csv.tmp";
        if (std::filesystem::exists(observationsFinal) ||
            std::filesystem::exists(observationsTemporary))
            throw std::runtime_error(
                "TG4 output already exists; choose a new output directory");
        observations.open(observationsTemporary, std::ios::binary);
        if (!observations)
            throw std::runtime_error("TG4 cannot create observations artifact");
        observations << ObservationCsvHeader();
    }

    std::filesystem::path outputDirectory;
    std::filesystem::path observationsFinal;
    std::filesystem::path observationsTemporary;
    EvaluationConfiguration configuration;
    TemporalRange range;
    std::string baselineCommit;
    std::ofstream observations;
    AggregateAccumulator aggregates;
    std::vector<DataQualityAudit> audits;
    std::size_t observationCount = 0;
    bool complete = false;

    void WriteText(const std::string& name, const std::string& contents)
    {
        const auto final = outputDirectory / name;
        const auto temporary = outputDirectory / (name + ".tmp");
        if (std::filesystem::exists(final) || std::filesystem::exists(temporary))
            throw std::runtime_error("TG4 output artifact already exists: " +
                                     final.string());
        std::ofstream output(temporary, std::ios::binary);
        if (!output) throw std::runtime_error("TG4 cannot create " + final.string());
        output << contents;
        output.close();
        if (!output) throw std::runtime_error("TG4 failed writing " + final.string());
        RenameComplete(temporary, final);
    }

    void WriteCohorts()
    {
        std::ostringstream output;
        output << "schema_version,symbol,temporal_partition,direction,cohort,"
                  "total_observations,paired,unpaired,confluent,non_confluent,"
                  "confluence_ineligible,"
                  "retest_structurally_eligible,retest_structurally_ineligible,"
                  "retest_pending,retest_censored,retest_resolved,retest_successes,"
                  "retest_failures,retest_denominator,retest_rate,retest_wilson95_lower,"
                  "retest_wilson95_upper,outer_structurally_eligible,"
                  "outer_structurally_ineligible,outer_pending,outer_censored,"
                  "outer_resolved,outer_successes,outer_failures,outer_denominator,"
                  "outer_rate,outer_wilson95_lower,outer_wilson95_upper,"
                  "conditioned_structurally_eligible,conditioned_structurally_ineligible,"
                  "conditioned_pending,conditioned_censored,conditioned_resolved,"
                  "conditioned_successes,conditioned_failures,conditioned_denominator,"
                  "conditioned_rate,conditioned_wilson95_lower,conditioned_wilson95_upper\n";
        for (const auto& [key, value] : aggregates.Cohorts())
        {
            output << kAggregateSchemaVersion << ',' << Csv(key.symbol) << ','
                   << Csv(key.partition) << ',' << Csv(key.direction) << ','
                   << Csv(key.cohort) << ',' << value.totalObservations << ','
                   << value.paired << ',' << value.unpaired << ','
                   << value.confluent << ',' << value.nonConfluent << ','
                   << value.confluenceIneligible << ','
                   << OutcomeFields(value.retest) << ','
                   << OutcomeFields(value.outerTarget) << ','
                   << OutcomeFields(value.outerTargetAfterRetest) << '\n';
        }
        WriteText("cohorts.csv", output.str());
    }

    void WriteAngles()
    {
        std::ostringstream output;
        output << "schema_version,symbol,temporal_partition,direction,cohort,"
                  "inner_angle_n,inner_angle_min,inner_angle_max,inner_angle_mean,"
                  "inner_angle_sample_sd,outer_angle_n,outer_angle_min,outer_angle_max,"
                  "outer_angle_mean,outer_angle_sample_sd\n";
        for (const auto& [key, value] : aggregates.Angles())
        {
            const auto fields = [](const AngleMoments& moments)
            {
                if (moments.count == 0) return std::string("0,,,,");
                return std::to_string(moments.count) + ',' +
                    Number(moments.minimum) + ',' + Number(moments.maximum) + ',' +
                    Number(moments.mean) + ',' +
                    OptionalNumber(moments.SampleStandardDeviation());
            };
            output << kAggregateSchemaVersion << ',' << Csv(key.symbol) << ','
                   << Csv(key.partition) << ',' << Csv(key.direction) << ','
                   << Csv(key.cohort) << ',' << fields(value.inner) << ','
                   << fields(value.outer) << '\n';
        }
        WriteText("angle_distributions.csv", output.str());
    }

    void WriteComparisons()
    {
        std::ostringstream output;
        output << "schema_version,symbol,temporal_partition,direction,comparison,"
                  "group_a_successes,group_a_failures,group_a_denominator,group_a_rate,"
                  "group_a_wilson95_lower,group_a_wilson95_upper,group_b_successes,"
                  "group_b_failures,group_b_denominator,group_b_rate,"
                  "group_b_wilson95_lower,group_b_wilson95_upper,absolute_rate_difference\n";
        for (const auto& [key, value] : aggregates.Comparisons())
        {
            const WilsonInterval a = Wilson95(
                value.groupASuccesses, value.groupAFailures);
            const WilsonInterval b = Wilson95(
                value.groupBSuccesses, value.groupBFailures);
            const std::optional<double> difference = a.rate.has_value() && b.rate.has_value()
                ? std::optional(*a.rate - *b.rate) : std::nullopt;
            output << kAggregateSchemaVersion << ',' << Csv(key.symbol) << ','
                   << Csv(key.partition) << ',' << Csv(key.direction) << ','
                   << Csv(key.cohort) << ',' << value.groupASuccesses << ','
                   << value.groupAFailures << ',' << a.denominator << ','
                   << OptionalNumber(a.rate) << ',' << OptionalNumber(a.lower95) << ','
                   << OptionalNumber(a.upper95) << ',' << value.groupBSuccesses << ','
                   << value.groupBFailures << ',' << b.denominator << ','
                   << OptionalNumber(b.rate) << ',' << OptionalNumber(b.lower95) << ','
                   << OptionalNumber(b.upper95) << ',' << OptionalNumber(difference)
                   << '\n';
        }
        WriteText("comparisons.csv", output.str());
    }

    void WriteEqualSymbol()
    {
        struct EqualRates
        {
            AngleMoments retest;
            AngleMoments outer;
            AngleMoments conditioned;
        };
        std::map<std::tuple<std::string, std::string, std::string>, EqualRates> rows;
        for (const auto& [key, value] : aggregates.Cohorts())
        {
            if (key.symbol == "__event_weighted__") continue;
            auto& row = rows[{key.partition, key.direction, key.cohort}];
            const WilsonInterval retest = Wilson95(
                value.retest.successes, value.retest.failures);
            const WilsonInterval outer = Wilson95(
                value.outerTarget.successes, value.outerTarget.failures);
            const WilsonInterval conditioned = Wilson95(
                value.outerTargetAfterRetest.successes,
                value.outerTargetAfterRetest.failures);
            if (retest.rate.has_value()) row.retest.Add(*retest.rate);
            if (outer.rate.has_value()) row.outer.Add(*outer.rate);
            if (conditioned.rate.has_value()) row.conditioned.Add(*conditioned.rate);
        }
        std::ostringstream output;
        output << "schema_version,temporal_partition,direction,cohort,outcome,"
                  "symbols_with_nonzero_denominator,equal_symbol_mean_rate,"
                  "minimum_symbol_rate,maximum_symbol_rate,sample_sd_across_symbols\n";
        for (const auto& [key, rates] : rows)
        {
            const auto emit = [&](std::string_view outcome,
                                  const AngleMoments& moments)
            {
                output << kAggregateSchemaVersion << ','
                       << Csv(std::get<0>(key)) << ',' << Csv(std::get<1>(key))
                       << ',' << Csv(std::get<2>(key)) << ',' << outcome << ','
                       << moments.count << ',';
                if (moments.count == 0) output << ",,,,\n";
                else output << Number(moments.mean) << ','
                            << Number(moments.minimum) << ','
                            << Number(moments.maximum) << ','
                            << OptionalNumber(moments.SampleStandardDeviation())
                            << '\n';
            };
            emit("retest", rates.retest);
            emit("outer_target", rates.outer);
            emit("retest_then_outer", rates.conditioned);
        }
        WriteText("equal_symbol_rates.csv", output.str());
    }

    void WriteDataQuality()
    {
        std::ostringstream summary;
        summary << "schema_version,symbol,excluded,exclusion_reason,source_rows,"
                   "usable_rows,warmup_rows,scored_rows,outcome_only_rows,"
                   "duplicate_timestamps,out_of_order_timestamps,first_usable_timestamp,"
                   "last_usable_timestamp,material_gap_count,scored_material_gap_count,"
                   "exploratory_rows,calibration_rows,validation_rows,confirmation_rows,"
                   "outside_named_study_rows\n";
        std::ostringstream gaps;
        gaps << "schema_version,symbol,previous_timestamp,next_timestamp,gap_seconds,"
                "intersects_scored_range\n";
        for (const DataQualityAudit& audit : audits)
        {
            const auto partition = [&audit](TemporalPartition value)
            {
                const auto found = audit.partitionRows.find(value);
                return found == audit.partitionRows.end() ? std::size_t{0}
                                                          : found->second;
            };
            const std::size_t scoredGaps = static_cast<std::size_t>(std::count_if(
                audit.materialGaps.begin(), audit.materialGaps.end(),
                [](const GapObservation& value)
                {
                    return value.intersectsScoredRange;
                }));
            summary << "tg4-data-quality-v1," << Csv(audit.symbol) << ','
                    << Bool(audit.excluded) << ',' << Csv(audit.exclusionReason) << ','
                    << audit.sourceRows << ',' << audit.usableRows << ','
                    << audit.warmupRows << ',' << audit.scoredRows << ','
                    << audit.outcomeOnlyRows << ',' << audit.duplicateTimestamps << ','
                    << audit.outOfOrderTimestamps << ','
                    << (audit.firstUsableTimestamp.has_value()
                        ? FormatUtcTimestamp(*audit.firstUsableTimestamp) : "") << ','
                    << (audit.lastUsableTimestamp.has_value()
                        ? FormatUtcTimestamp(*audit.lastUsableTimestamp) : "") << ','
                    << audit.materialGaps.size() << ',' << scoredGaps << ','
                    << partition(TemporalPartition::Exploratory2010To2019) << ','
                    << partition(TemporalPartition::Calibration2020To2022) << ','
                    << partition(TemporalPartition::Validation2023To2024) << ','
                    << partition(TemporalPartition::Confirmation2025) << ','
                    << partition(TemporalPartition::OutsideNamedStudy) << '\n';
            for (const GapObservation& gap : audit.materialGaps)
                gaps << "tg4-data-gap-v1," << Csv(audit.symbol) << ','
                     << FormatUtcTimestamp(gap.previousTimestamp) << ','
                     << FormatUtcTimestamp(gap.nextTimestamp) << ','
                     << gap.gapSeconds << ',' << Bool(gap.intersectsScoredRange)
                     << '\n';
        }
        WriteText("data_quality.csv", summary.str());
        WriteText("data_gaps.csv", gaps.str());
    }

    void WriteReport()
    {
        std::ostringstream output;
        output << "# TG4 historical empirical study report\n\n"
               << "This report is descriptive evidence, not proof of market edge, "
                  "profitability, or observation independence. Overlapping events can "
                  "share trend lines, time, and market episodes.\n\n"
               << "- Baseline commit: `" << baselineCommit << "`\n"
               << "- Configuration: `" << configuration.name << "`\n"
               << "- Configuration provenance: " << configuration.provenance << "\n"
               << "- Observation schema: `" << kObservationSchemaVersion << "`\n"
               << "- Inner-break observations: " << observationCount << "\n"
               << "- Rate denominator: successes + failures only\n"
               << "- Interval: 95% Wilson score interval\n"
               << "- Tiny-cell display rule: resolved N below "
               << configuration.minimumHumanReportResolvedN
               << " is retained in CSV but omitted from the rate table below\n\n"
               << "## Data quality\n\n"
               << "| Symbol | Usable bars | First | Last | Duplicates | Material gaps | Excluded |\n"
               << "|---|---:|---|---|---:|---:|---|\n";
        for (const auto& audit : audits)
            output << '|' << audit.symbol << '|' << audit.usableRows << '|'
                   << (audit.firstUsableTimestamp.has_value()
                       ? FormatUtcTimestamp(*audit.firstUsableTimestamp) : "") << '|'
                   << (audit.lastUsableTimestamp.has_value()
                       ? FormatUtcTimestamp(*audit.lastUsableTimestamp) : "") << '|'
                   << audit.duplicateTimestamps << '|' << audit.materialGaps.size()
                   << '|' << (audit.excluded ? audit.exclusionReason : "no") << "|\n";
        output << "\n## Event-weighted outer-target rates (all cohort)\n\n"
               << "| Partition | Direction | Successes | Failures | N | Rate | Wilson 95% |\n"
               << "|---|---|---:|---:|---:|---:|---|\n";
        for (const auto& [key, value] : aggregates.Cohorts())
        {
            if (key.symbol != "__event_weighted__" || key.cohort != "all" ||
                key.partition == "all") continue;
            const WilsonInterval interval = Wilson95(
                value.outerTarget.successes, value.outerTarget.failures);
            if (interval.denominator < configuration.minimumHumanReportResolvedN)
                continue;
            output << '|' << key.partition << '|' << key.direction << '|'
                   << value.outerTarget.successes << '|'
                   << value.outerTarget.failures << '|' << interval.denominator
                   << '|' << OptionalNumber(interval.rate) << "|["
                   << OptionalNumber(interval.lower95) << ", "
                   << OptionalNumber(interval.upper95) << "]|\n";
        }
        output << "\nMachine-readable `equal_symbol_rates.csv` must be reviewed alongside "
                  "event-weighted totals so high-event-count symbols do not masquerade "
                  "as cross-symbol consistency. `comparisons.csv` contains the underlying "
                  "2x2 counts. Pending, censored, unpaired, and structurally ineligible "
                  "records are never counted as failures.\n\n"
               << "No parameter search or automatic calibration was performed. The 2025 "
                  "confirmation partition was measured only under the predeclared supplied "
                  "configuration and was not used to select ratios, tolerances, angle "
                  "scales, bands, or horizons.\n";
        WriteText("report.md", output.str());
    }
};

StudyArtifactWriter::StudyArtifactWriter(
    std::filesystem::path outputDirectory,
    EvaluationConfiguration configuration,
    TemporalRange range,
    std::string baselineCommit)
    : implementation_(new Implementation(
          std::move(outputDirectory), std::move(configuration), range,
          std::move(baselineCommit)))
{
}

StudyArtifactWriter::~StudyArtifactWriter()
{
    delete implementation_;
}

void StudyArtifactWriter::Write(const ObservationRecord& record)
{
    implementation_->observations << ObservationCsvRow(
        record, implementation_->configuration);
    if (!implementation_->observations)
        throw std::runtime_error("TG4 failed writing observations.csv");
    implementation_->aggregates.Add(record);
    ++implementation_->observationCount;
}

void StudyArtifactWriter::AddDataQuality(DataQualityAudit audit)
{
    implementation_->audits.push_back(std::move(audit));
    std::sort(implementation_->audits.begin(), implementation_->audits.end(),
        [](const DataQualityAudit& left, const DataQualityAudit& right)
        {
            return left.symbol < right.symbol;
        });
}

void StudyArtifactWriter::Complete()
{
    if (implementation_->complete) return;
    implementation_->observations.close();
    if (!implementation_->observations)
        throw std::runtime_error("TG4 failed closing observations.csv");
    RenameComplete(implementation_->observationsTemporary,
                   implementation_->observationsFinal);
    implementation_->WriteText("metadata.json", EffectiveConfigurationJson(
        implementation_->configuration, implementation_->range,
        implementation_->baselineCommit));
    implementation_->WriteCohorts();
    implementation_->WriteAngles();
    implementation_->WriteComparisons();
    implementation_->WriteEqualSymbol();
    implementation_->WriteDataQuality();
    implementation_->WriteReport();
    implementation_->complete = true;
}

const AggregateAccumulator& StudyArtifactWriter::Aggregates() const
{
    return implementation_->aggregates;
}

std::size_t StudyArtifactWriter::ObservationCount() const
{
    return implementation_->observationCount;
}

} // namespace EA::TG4
