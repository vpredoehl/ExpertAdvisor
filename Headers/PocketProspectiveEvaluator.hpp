#pragma once

// Phase Pocket 4 implements the frozen Phase Pocket 3 *project-defined*
// prospective evaluator.  It deliberately does not claim to be a mathematical
// transcription of the MTI manual, and it never supplies future bars to the
// causal detector.

#include "CausalPocketDetector.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace EA::Pocket::Prospective
{
inline constexpr std::string_view kStudyId =
    "pocket-prospective-preconfirmation-v1";
inline constexpr std::string_view kProtocolId =
    "phase-pocket-3-prospective-empirical-evaluation-protocol-v1";
inline constexpr std::string_view kDetectorId =
    "causal-pocket-detector-phase2-v1";
inline constexpr std::string_view kConfigurationSchema =
    "phase-pocket-4-canonical-run-configuration-v1";
inline constexpr std::string_view kProtocolDocument =
    "docs/phases/pockets/PhasePocket3/PocketProspectiveEmpiricalEvaluationProtocolFreeze.md";
inline constexpr std::string_view kProtocolDocumentSha256 =
    "e2852555026098e7c9fcd378874fbc075732f23db5ebb9183bf04a6590f067e1";
inline constexpr std::int64_t kCadenceSeconds = 900;
inline constexpr std::int64_t kPreconfirmationEnd = 1735689600; // 2025-01-01T00:00:00Z
inline constexpr std::array<std::size_t, 3> kLookbacks{{10, 15, 20}};
inline constexpr std::array<std::size_t, 3> kHorizons{{4, 16, 64}};

inline std::string Sha256(std::string_view content)
{
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(content.data(), static_cast<CC_LONG>(content.size()), digest.data());
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) out << std::setw(2) << static_cast<int>(byte);
    return out.str();
}

inline std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("POCKET_FILE_UNREADABLE:" + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

inline std::string FileSha256(const std::filesystem::path& path)
{
    return Sha256(ReadTextFile(path));
}

inline bool IsHexSha256(std::string_view value)
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

struct Partition final
{
    std::string name;
    std::int64_t start = 0;
    std::int64_t end = 0;
};

struct SymbolSource final
{
    std::string symbol;
    std::string table;
    double pipSize = 0.0;
};

struct RunConfiguration final
{
    std::string schema;
    std::string study;
    std::string protocol;
    std::string protocolDocumentSha256;
    std::string detector;
    std::string detectorBaseline;
    std::string sourceAdapter;
    std::string sourceIdentity;
    std::string timeframe;
    std::int64_t cadenceSeconds = 0;
    std::array<std::size_t, 3> lookbacks{};
    std::array<std::size_t, 3> horizons{};
    std::array<SymbolSource, 6> symbols{};
    std::array<Partition, 3> partitions{};
    std::int64_t resolutionEnd = 0;
    std::size_t warmupBars = 0;
    std::size_t bootstrapReplicates = 0;
    double bootstrapConfidence = 0.0;
    std::string bootstrapBlock;
    std::string ordering;
    std::string outcomeContract;
    std::string censoringContract;
    std::string aggregationContract;
    std::string configurationSha256;
};

inline const std::vector<std::pair<std::string, std::string>>& FrozenFields()
{
    // Sorted immutable rendering.  Every value is a direct Phase Pocket 3
    // contract value; no run-time option can amend one of these choices.
    static const std::vector<std::pair<std::string, std::string>> fields{
        {"aggregation", "event_weighted_and_equal_symbol_separate"},
        {"bootstrap_block", "utc_calendar_week_confirmation_blocks"},
        {"bootstrap_confidence", "0.95"},
        {"bootstrap_replicates", "2000"},
        {"cadence_seconds", "900"},
        {"censoring", "right_censor_first_unusable_bar_boundary_tail_gap_invalid"},
        {"configuration_schema", std::string(kConfigurationSchema)},
        {"detector", std::string(kDetectorId)},
        {"detector_baseline", "41800f4"},
        {"horizons", "4,16,64"},
        {"lookbacks", "10,15,20"},
        {"ordering", "symbol,lookback,confirmation_timestamp,observation_identity"},
        {"outcomes", "inclusive_touch_and_close;bounded_continuation_and_race"},
        {"partitions", "exploratory:1262304000:1577836800,calibration:1577836800:1672531200,validation:1672531200:1735689600"},
        {"protocol", std::string(kProtocolId)},
        {"protocol_document_sha256", std::string(kProtocolDocumentSha256)},
        {"resolution_end", "1735689600"},
        {"source_adapter", "postgresql-candlestick-canonical-absolute-half-open-v1"},
        {"source_identity", "canonical_15m_completed_candlestick"},
        {"study", std::string(kStudyId)},
        {"symbols", "AUDCAD:audcadrmp:0.0001,AUDUSD:audusdrmp:0.0001,EURUSD:eurusdrmp:0.0001,GBPUSD:gbpusdrmp:0.0001,USDCAD:usdcadrmp:0.0001,USDJPY:usdjpyrmp:0.01"},
        {"timeframe", "15m_completed"},
        {"warmup_bars", "21"},
    };
    return fields;
}

inline std::string CanonicalPayload()
{
    std::ostringstream output;
    for (const auto& [key, value] : FrozenFields()) output << key << '=' << value << '\n';
    return output.str();
}

inline std::string CanonicalConfigurationText()
{
    return CanonicalPayload() + "configuration_sha256=" + Sha256(CanonicalPayload()) + '\n';
}

inline RunConfiguration FrozenConfiguration()
{
    RunConfiguration config;
    config.schema = std::string(kConfigurationSchema);
    config.study = std::string(kStudyId);
    config.protocol = std::string(kProtocolId);
    config.protocolDocumentSha256 = std::string(kProtocolDocumentSha256);
    config.detector = std::string(kDetectorId);
    config.detectorBaseline = "41800f4";
    config.sourceAdapter = "postgresql-candlestick-canonical-absolute-half-open-v1";
    config.sourceIdentity = "canonical_15m_completed_candlestick";
    config.timeframe = "15m_completed";
    config.cadenceSeconds = kCadenceSeconds;
    config.lookbacks = kLookbacks;
    config.horizons = kHorizons;
    config.symbols = {{{"AUDCAD", "audcadrmp", 0.0001}, {"AUDUSD", "audusdrmp", 0.0001},
        {"EURUSD", "eurusdrmp", 0.0001}, {"GBPUSD", "gbpusdrmp", 0.0001},
        {"USDCAD", "usdcadrmp", 0.0001}, {"USDJPY", "usdjpyrmp", 0.01}}};
    config.partitions = {{{"exploratory", 1262304000, 1577836800},
        {"calibration", 1577836800, 1672531200}, {"validation", 1672531200, 1735689600}}};
    config.resolutionEnd = kPreconfirmationEnd;
    config.warmupBars = 21;
    config.bootstrapReplicates = 2000;
    config.bootstrapConfidence = 0.95;
    config.bootstrapBlock = "utc_calendar_week_confirmation_blocks";
    config.ordering = "symbol,lookback,confirmation_timestamp,observation_identity";
    config.outcomeContract = "inclusive_touch_and_close;bounded_continuation_and_race";
    config.censoringContract = "right_censor_first_unusable_bar_boundary_tail_gap_invalid";
    config.aggregationContract = "event_weighted_and_equal_symbol_separate";
    config.configurationSha256 = Sha256(CanonicalPayload());
    return config;
}

inline RunConfiguration LoadAndValidateConfiguration(const std::filesystem::path& path)
{
    const std::string text = ReadTextFile(path);
    if (text != CanonicalConfigurationText())
        throw std::invalid_argument("POCKET_CONFIGURATION_NONCANONICAL_OR_UNKNOWN_FIELD");
    const RunConfiguration configuration = FrozenConfiguration();
    if (configuration.schema != kConfigurationSchema || configuration.study != kStudyId ||
        configuration.protocol != kProtocolId || configuration.detector != kDetectorId ||
        configuration.lookbacks != kLookbacks || configuration.horizons != kHorizons ||
        configuration.timeframe != "15m_completed" || configuration.cadenceSeconds != kCadenceSeconds ||
        configuration.resolutionEnd != kPreconfirmationEnd ||
        !IsHexSha256(configuration.configurationSha256) ||
        (!configuration.protocolDocumentSha256.empty() &&
         !IsHexSha256(configuration.protocolDocumentSha256)))
        throw std::invalid_argument("POCKET_CONFIGURATION_FROZEN_CONTRACT_MISMATCH");
    return configuration;
}

inline void VerifyFrozenProtocolDocument()
{
    if (kProtocolDocumentSha256.empty())
        throw std::logic_error("POCKET_PROTOCOL_DOCUMENT_HASH_NOT_INTEGRATED");
    if (FileSha256(std::filesystem::path(kProtocolDocument)) != kProtocolDocumentSha256)
        throw std::invalid_argument("POCKET_PROTOCOL_DOCUMENT_HASH_MISMATCH");
}

inline const Partition* PartitionFor(const RunConfiguration& configuration,
    std::int64_t confirmationTimestamp)
{
    for (const Partition& partition : configuration.partitions)
        if (confirmationTimestamp >= partition.start && confirmationTimestamp < partition.end)
            return &partition;
    return nullptr;
}

inline const SymbolSource& SourceFor(const RunConfiguration& configuration,
    std::string_view symbol)
{
    const auto found = std::find_if(configuration.symbols.begin(), configuration.symbols.end(),
        [symbol](const SymbolSource& candidate) { return candidate.symbol == symbol; });
    if (found == configuration.symbols.end())
        throw std::invalid_argument("POCKET_UNKNOWN_SYMBOL");
    return *found;
}

inline void ValidateCompletedBar(const CompletedBar& bar)
{
    if (!std::isfinite(bar.open) || !std::isfinite(bar.high) || !std::isfinite(bar.low) ||
        !std::isfinite(bar.close) || bar.high < bar.low || bar.open < bar.low ||
        bar.open > bar.high || bar.close < bar.low || bar.close > bar.high)
        throw std::invalid_argument("POCKET_MALFORMED_OHLC");
}

struct SourceAudit final
{
    std::string sourceDatabaseIdentity;
    std::string symbol;
    std::string table;
    std::size_t rowCount = 0;
    std::int64_t firstTimestamp = 0;
    std::int64_t lastTimestamp = 0;
    std::size_t cadenceGaps = 0;
};

inline SourceAudit PreflightCompletedBars(const RunConfiguration& configuration,
    std::string_view symbol, std::string_view table, std::string sourceDatabaseIdentity,
    const std::vector<CompletedBar>& bars)
{
    const SymbolSource& expected = SourceFor(configuration, symbol);
    if (table != expected.table) throw std::invalid_argument("POCKET_SOURCE_TABLE_MISMATCH");
    if (sourceDatabaseIdentity.empty()) throw std::invalid_argument("POCKET_SOURCE_PROVENANCE_MISSING");
    if (bars.empty()) throw std::invalid_argument("POCKET_SOURCE_EMPTY");
    SourceAudit audit{std::move(sourceDatabaseIdentity), expected.symbol, expected.table};
    audit.rowCount = bars.size(); audit.firstTimestamp = bars.front().timestamp;
    audit.lastTimestamp = bars.back().timestamp;
    std::optional<std::int64_t> previous;
    for (const CompletedBar& bar : bars)
    {
        ValidateCompletedBar(bar);
        if (bar.timestamp >= configuration.resolutionEnd)
            throw std::invalid_argument("POCKET_PRECONFIRMATION_FIREWALL_SOURCE_BOUNDARY");
        if (previous.has_value())
        {
            if (bar.timestamp <= *previous) throw std::invalid_argument("POCKET_DUPLICATE_OR_NONMONOTONIC_TIMESTAMP");
            if (bar.timestamp - *previous != configuration.cadenceSeconds) ++audit.cadenceGaps;
        }
        previous = bar.timestamp;
    }
    return audit;
}

enum class CensorReason { None, Gap, Boundary, Tail, InvalidInput };
inline std::string_view CensorReasonName(CensorReason reason)
{
    switch (reason) {
        case CensorReason::None: return "none"; case CensorReason::Gap: return "gap";
        case CensorReason::Boundary: return "boundary"; case CensorReason::Tail: return "tail";
        case CensorReason::InvalidInput: return "invalid_input";
    }
    return "invalid_input";
}

struct OutcomeLabel final
{
    std::size_t horizon = 0;
    std::size_t validFutureBars = 0;
    bool complete = false;
    CensorReason censor = CensorReason::None;
    std::optional<std::size_t> touchAt;
    std::optional<std::size_t> closeAt;
    double mfe = 0.0;
    double mae = 0.0;
    double directionalCloseReturn = 0.0;
    std::string race = "not_applicable";
};

// Explicitly bounded retrospective labeler.  The caller provides only the
// confirmation index and this routine reads at most horizon future bars.
inline OutcomeLabel EvaluateBoundedOutcome(const PocketObservation& observation,
    const std::vector<CompletedBar>& bars, std::size_t horizon,
    std::int64_t authorizedResolutionEnd)
{
    if (horizon == 0 || observation.confirmationBar >= bars.size())
        throw std::invalid_argument("POCKET_OUTCOME_INVALID_COORDINATE");
    OutcomeLabel label; label.horizon = horizon;
    const std::size_t c = observation.confirmationBar;
    const double p0 = bars[c].close;
    const double width = observation.range.upper - observation.range.lower;
    if (!(width > 0.0)) throw std::invalid_argument("POCKET_OUTCOME_NONPOSITIVE_WIDTH");
    double high = p0, low = p0;
    std::optional<std::size_t> continuationAt;
    for (std::size_t offset = 1; offset <= horizon; ++offset)
    {
        const std::size_t index = c + offset;
        if (index >= bars.size()) { label.censor = CensorReason::Tail; break; }
        const CompletedBar& bar = bars[index];
        if (bar.timestamp >= authorizedResolutionEnd) { label.censor = CensorReason::Boundary; break; }
        try { ValidateCompletedBar(bar); }
        catch (const std::invalid_argument&) { label.censor = CensorReason::InvalidInput; break; }
        const CompletedBar& previous = bars[index - 1];
        if (bar.timestamp - previous.timestamp != kCadenceSeconds) { label.censor = CensorReason::Gap; break; }
        ++label.validFutureBars;
        high = std::max(high, bar.high); low = std::min(low, bar.low);
        const bool touch = observation.direction == PocketDirection::Bullish
            ? bar.low <= observation.TouchPrice() : bar.high >= observation.TouchPrice();
        const bool close = observation.direction == PocketDirection::Bullish
            ? bar.close <= observation.ClosePrice() : bar.close >= observation.ClosePrice();
        const bool continuation = observation.direction == PocketDirection::Bullish
            ? bar.high >= p0 + width : bar.low <= p0 - width;
        if (touch && !label.touchAt) label.touchAt = offset;
        if (close && !label.closeAt) label.closeAt = offset;
        if (continuation && !continuationAt) continuationAt = offset;
    }
    label.complete = label.validFutureBars == horizon;
    if (label.complete)
    {
        label.censor = CensorReason::None;
        label.mfe = observation.direction == PocketDirection::Bullish
            ? std::max(0.0, high - p0) : std::max(0.0, p0 - low);
        label.mae = observation.direction == PocketDirection::Bullish
            ? std::max(0.0, p0 - low) : std::max(0.0, high - p0);
        label.directionalCloseReturn = observation.direction == PocketDirection::Bullish
            ? bars[c + horizon].close - p0 : p0 - bars[c + horizon].close;
        if (horizon == 64) {
            if (continuationAt && label.touchAt && *continuationAt == *label.touchAt) label.race = "same_bar_intrabar_order_indeterminate";
            else if (continuationAt && (!label.touchAt || *continuationAt < *label.touchAt)) label.race = "continuation_first";
            else if (label.touchAt) label.race = "revisit_first";
            else label.race = "neither";
        }
    }
    return label;
}

struct EvaluatedObservation final
{
    std::string identity;
    std::string symbol;
    std::string partition;
    std::size_t lookback = 0;
    PocketObservation observation;
    std::array<OutcomeLabel, 3> outcomes;
};

inline std::string ObservationIdentity(const RunConfiguration& configuration,
    std::string_view symbol, std::size_t lookback, const PocketObservation& observation)
{
    return configuration.study + ":" + configuration.configurationSha256 + ":" +
        std::string(symbol) + ":15m:" + std::to_string(lookback) + ":" +
        std::to_string(observation.eventBar) + ":" + std::to_string(observation.confirmationBar) +
        ":" + std::to_string(observation.confirmationTimestamp);
}

inline std::vector<EvaluatedObservation> ReplayCausallyAndLabelBounded(
    const RunConfiguration& configuration, std::string_view symbol,
    const std::vector<CompletedBar>& bars)
{
    (void)SourceFor(configuration, symbol);
    std::vector<EvaluatedObservation> output;
    for (const std::size_t lookback : configuration.lookbacks)
    {
        CausalPocketDetector detector("15m_completed", lookback);
        for (const CompletedBar& bar : bars)
        {
            const auto observation = detector.AddCompletedBar(bar);
            if (!observation) continue;
            const Partition* partition = PartitionFor(configuration, observation->confirmationTimestamp);
            if (!partition) continue;
            // Outcomes are separate labels: no future bar can influence detector state.
            EvaluatedObservation record{ObservationIdentity(configuration, symbol, lookback, *observation),
                std::string(symbol), partition->name, lookback, *observation, {}};
            for (std::size_t index = 0; index < kHorizons.size(); ++index)
                record.outcomes[index] = EvaluateBoundedOutcome(*observation, bars, kHorizons[index], configuration.resolutionEnd);
            output.push_back(std::move(record));
        }
    }
    std::sort(output.begin(), output.end(), [](const EvaluatedObservation& a, const EvaluatedObservation& b) {
        return std::tie(a.symbol, a.lookback, a.observation.confirmationTimestamp, a.identity) <
            std::tie(b.symbol, b.lookback, b.observation.confirmationTimestamp, b.identity);
    });
    return output;
}

struct Metric final
{
    std::size_t eligible = 0, complete = 0, touches = 0, closes = 0, censored = 0;
    std::optional<double> touchRate, closeRate, touchLower95, touchUpper95;
};

inline std::uint64_t DerivedBootstrapSeed(std::string_view configurationHash)
{
    if (!IsHexSha256(configurationHash)) throw std::invalid_argument("POCKET_BOOTSTRAP_HASH_INVALID");
    return std::stoull(std::string(configurationHash.substr(0, 16)), nullptr, 16);
}

inline std::pair<std::optional<double>, std::optional<double>> BootstrapTouchInterval(
    const std::vector<const EvaluatedObservation*>& records, std::size_t horizonIndex,
    std::uint64_t seed, std::size_t replicates)
{
    std::map<std::int64_t, std::vector<const EvaluatedObservation*>> weeks;
    for (const auto* record : records)
        weeks[record->observation.confirmationTimestamp / (7 * 24 * 60 * 60)].push_back(record);
    if (weeks.empty() || replicates == 0) return {};
    std::vector<const std::vector<const EvaluatedObservation*>*> blocks;
    for (const auto& [week, values] : weeks) { (void)week; blocks.push_back(&values); }
    std::mt19937_64 random(seed); std::uniform_int_distribution<std::size_t> choose(0, blocks.size() - 1);
    std::vector<double> values;
    for (std::size_t replicate = 0; replicate < replicates; ++replicate) {
        std::size_t denominator = 0, numerator = 0;
        for (std::size_t draw = 0; draw < blocks.size(); ++draw) for (const auto* record : *blocks[choose(random)]) {
            const OutcomeLabel& label = record->outcomes[horizonIndex];
            if (!label.complete) continue;
            ++denominator; if (label.touchAt) ++numerator;
        }
        if (denominator != 0) values.push_back(static_cast<double>(numerator) / denominator);
    }
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    return {values[(values.size() - 1) * 25 / 1000], values[(values.size() - 1) * 975 / 1000]};
}

inline Metric AggregateMetric(const std::vector<const EvaluatedObservation*>& records,
    std::size_t horizonIndex, std::uint64_t seed, std::size_t replicates)
{
    Metric metric; metric.eligible = records.size();
    for (const auto* record : records) {
        const OutcomeLabel& label = record->outcomes[horizonIndex];
        if (!label.complete) { ++metric.censored; continue; }
        ++metric.complete; if (label.touchAt) ++metric.touches; if (label.closeAt) ++metric.closes;
    }
    if (metric.complete != 0) {
        metric.touchRate = static_cast<double>(metric.touches) / metric.complete;
        metric.closeRate = static_cast<double>(metric.closes) / metric.complete;
        std::tie(metric.touchLower95, metric.touchUpper95) =
            BootstrapTouchInterval(records, horizonIndex, seed, replicates);
    }
    return metric;
}

inline std::pair<std::optional<double>, std::optional<double>> BootstrapEqualSymbolTouchInterval(
    const std::vector<const std::vector<const EvaluatedObservation*>*>& symbolCohorts,
    std::size_t horizonIndex, std::uint64_t seed, std::size_t replicates)
{
    // The frozen equal-symbol bootstrap samples six symbols with replacement,
    // then samples that selected symbol's UTC-week confirmation blocks.
    if (symbolCohorts.size() != 6 || replicates == 0) return {};
    std::vector<std::vector<std::vector<const EvaluatedObservation*>>> allBlocks;
    for (const auto* cohort : symbolCohorts) {
        std::map<std::int64_t, std::vector<const EvaluatedObservation*>> weeks;
        for (const auto* record : *cohort)
            weeks[record->observation.confirmationTimestamp / (7 * 24 * 60 * 60)].push_back(record);
        if (weeks.empty()) return {};
        std::vector<std::vector<const EvaluatedObservation*>> blocks;
        for (const auto& [week, values] : weeks) { (void)week; blocks.push_back(values); }
        allBlocks.push_back(std::move(blocks));
    }
    std::mt19937_64 random(seed); std::uniform_int_distribution<std::size_t> symbolPick(0, 5);
    std::vector<double> values;
    for (std::size_t replicate = 0; replicate < replicates; ++replicate) {
        double mean = 0.0; bool valid = true;
        for (std::size_t draw = 0; draw < 6; ++draw) {
            const auto& blocks = allBlocks[symbolPick(random)];
            std::uniform_int_distribution<std::size_t> blockPick(0, blocks.size() - 1);
            std::size_t denominator = 0, numerator = 0;
            for (std::size_t blockDraw = 0; blockDraw < blocks.size(); ++blockDraw)
                for (const auto* record : blocks[blockPick(random)]) {
                    const OutcomeLabel& label = record->outcomes[horizonIndex];
                    if (!label.complete) continue;
                    ++denominator; if (label.touchAt) ++numerator;
                }
            if (denominator == 0) { valid = false; break; }
            mean += static_cast<double>(numerator) / denominator;
        }
        if (valid) values.push_back(mean / 6.0);
    }
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    return {values[(values.size() - 1) * 25 / 1000], values[(values.size() - 1) * 975 / 1000]};
}

inline std::vector<const EvaluatedObservation*> GreedyTemporalThin(
    const std::vector<EvaluatedObservation>& records)
{
    std::vector<const EvaluatedObservation*> result;
    std::map<std::pair<std::string, std::size_t>, std::vector<const EvaluatedObservation*>> cohorts;
    for (const auto& record : records) cohorts[{record.symbol, record.lookback}].push_back(&record);
    for (auto& [key, cohort] : cohorts) {
        (void)key;
        std::sort(cohort.begin(), cohort.end(), [](const auto* a, const auto* b) {
            return std::tie(a->observation.confirmationBar, a->identity) < std::tie(b->observation.confirmationBar, b->identity);
        });
        std::optional<std::size_t> last;
        for (const auto* record : cohort)
            if (!last || record->observation.confirmationBar >= *last + 64) {
                result.push_back(record); last = record->observation.confirmationBar;
            }
    }
    return result;
}

inline std::string CsvNumber(double value)
{
    std::ostringstream output; output.imbue(std::locale::classic());
    output << std::setprecision(17) << value; return output.str();
}

inline std::string ObservationsCsv(const std::vector<EvaluatedObservation>& records)
{
    std::ostringstream out;
    out << "identity,symbol,partition,lookback,direction,event_timestamp,confirmation_timestamp,lower,upper,horizon,complete,censor,valid_future_bars,touch_at,close_at,mfe,mae,directional_close_return,race\n";
    for (const auto& record : records) for (const auto& label : record.outcomes)
        out << record.identity << ',' << record.symbol << ',' << record.partition << ',' << record.lookback << ','
            << (record.observation.direction == PocketDirection::Bullish ? "bullish" : "bearish") << ','
            << record.observation.eventTimestamp << ',' << record.observation.confirmationTimestamp << ','
            << CsvNumber(record.observation.range.lower) << ',' << CsvNumber(record.observation.range.upper) << ','
            << label.horizon << ',' << (label.complete ? "true" : "false") << ',' << CensorReasonName(label.censor) << ','
            << label.validFutureBars << ',' << (label.touchAt ? std::to_string(*label.touchAt) : "") << ','
            << (label.closeAt ? std::to_string(*label.closeAt) : "") << ',' << CsvNumber(label.mfe) << ','
            << CsvNumber(label.mae) << ',' << CsvNumber(label.directionalCloseReturn) << ',' << label.race << '\n';
    return out.str();
}

inline std::string MetricCell(const std::optional<double>& value)
{
    return value ? CsvNumber(*value) : "undefined";
}

inline std::string AggregatesCsv(const RunConfiguration& configuration,
    const std::vector<EvaluatedObservation>& records)
{
    using Key = std::tuple<std::string, std::size_t, std::string, std::string>;
    std::map<Key, std::vector<const EvaluatedObservation*>> cohorts;
    for (const auto& record : records)
        cohorts[{record.symbol, record.lookback, record.partition,
            record.observation.direction == PocketDirection::Bullish ? "bullish" : "bearish"}].push_back(&record);
    std::ostringstream out;
    out << "aggregation,symbol,lookback,partition,direction,horizon,contributors,eligible,complete,touches,closes,censored,touch_rate,close_rate,touch_lower95,touch_upper95\n";
    for (const auto& [key, cohort] : cohorts) {
        const auto& [symbol, lookback, partition, direction] = key;
        for (std::size_t horizon = 0; horizon < kHorizons.size(); ++horizon) {
            const Metric metric = AggregateMetric(cohort, horizon,
                DerivedBootstrapSeed(configuration.configurationSha256) + horizon, configuration.bootstrapReplicates);
            out << "event_weighted," << symbol << ',' << lookback << ',' << partition << ',' << direction << ','
                << kHorizons[horizon] << ",1," << metric.eligible << ',' << metric.complete << ',' << metric.touches << ','
                << metric.closes << ',' << metric.censored << ',' << MetricCell(metric.touchRate) << ','
                << MetricCell(metric.closeRate) << ',' << MetricCell(metric.touchLower95) << ',' << MetricCell(metric.touchUpper95) << '\n';
        }
    }
    // Equal-symbol rows deliberately average only defined per-symbol rates and
    // disclose their contributor count; they do not replace event weighting.
    std::map<std::tuple<std::size_t, std::string, std::string>, std::vector<const std::vector<const EvaluatedObservation*>*>> equal;
    for (const auto& [key, cohort] : cohorts) {
        const auto& [symbol, lookback, partition, direction] = key; (void)symbol;
        equal[{lookback, partition, direction}].push_back(&cohort);
    }
    for (const auto& [key, symbolCohorts] : equal) {
        const auto& [lookback, partition, direction] = key;
        for (std::size_t horizon = 0; horizon < kHorizons.size(); ++horizon) {
            std::size_t contributors = 0; double touchSum = 0.0, closeSum = 0.0;
            for (const auto* cohort : symbolCohorts) {
                const Metric metric = AggregateMetric(*cohort, horizon,
                    DerivedBootstrapSeed(configuration.configurationSha256), 0);
                if (!metric.touchRate || !metric.closeRate) continue;
                ++contributors; touchSum += *metric.touchRate; closeSum += *metric.closeRate;
            }
            const std::optional<double> touch = contributors ? std::optional<double>(touchSum / contributors) : std::nullopt;
            const std::optional<double> close = contributors ? std::optional<double>(closeSum / contributors) : std::nullopt;
            const auto [lower, upper] = BootstrapEqualSymbolTouchInterval(symbolCohorts, horizon,
                DerivedBootstrapSeed(configuration.configurationSha256) + horizon, configuration.bootstrapReplicates);
            out << "equal_symbol,ALL," << lookback << ',' << partition << ',' << direction << ',' << kHorizons[horizon]
                << ',' << contributors << ",undefined,undefined,undefined,undefined,undefined," << MetricCell(touch)
                << ',' << MetricCell(close) << ',' << MetricCell(lower) << ',' << MetricCell(upper) << '\n';
        }
    }
    return out.str();
}

class ImmutableArtifactWriter final
{
public:
    explicit ImmutableArtifactWriter(std::filesystem::path target)
        : target_(std::move(target)), staging_(target_.string() + ".tmp." + std::to_string(::getpid()))
    {
        if (target_.empty() || std::filesystem::exists(target_))
            throw std::invalid_argument("POCKET_OUTPUT_TARGET_EXISTS_OR_EMPTY");
        if (std::filesystem::exists(staging_)) throw std::invalid_argument("POCKET_STAGING_TARGET_EXISTS");
        std::filesystem::create_directories(staging_);
    }
    ~ImmutableArtifactWriter() { if (!published_) { std::error_code ignored; std::filesystem::remove_all(staging_, ignored); } }
    ImmutableArtifactWriter(const ImmutableArtifactWriter&) = delete;

    void Publish(const RunConfiguration& configuration, std::string_view provenance,
        const std::vector<EvaluatedObservation>& records,
        std::string_view configurationText = CanonicalConfigurationText(),
        std::string_view artifactSchema = {})
    {
        Write("configuration.conf", configurationText);
        Write("observations.csv", ObservationsCsv(records));
        Write("aggregates.csv", AggregatesCsv(configuration, records));
        const std::string configurationHash = FileSha256(staging_ / "configuration.conf");
        const std::string observationsHash = FileSha256(staging_ / "observations.csv");
        const std::string aggregatesHash = FileSha256(staging_ / "aggregates.csv");
        const std::string manifest = "study=" + configuration.study + "\nprotocol=" + configuration.protocol +
            "\nprotocol_document_sha256=" + configuration.protocolDocumentSha256 + "\ndetector=" + configuration.detector +
            "\ndetector_baseline=" + configuration.detectorBaseline + "\nconfiguration_sha256=" + configuration.configurationSha256 +
            "\nconfiguration_file_sha256=" + configurationHash + "\nobservations_sha256=" + observationsHash +
            "\naggregates_sha256=" + aggregatesHash +
            (artifactSchema.empty() ? "" : "\nartifact_schema=" + std::string(artifactSchema)) +
            "\nprovenance=" + std::string(provenance) + "\n";
        Write("manifest.txt", manifest);
        VerifyDirectoryContract(staging_, configuration, configurationText, artifactSchema);
        std::filesystem::rename(staging_, target_);
        published_ = true;
    }

    static void VerifyDirectory(const std::filesystem::path& directory)
    {
        VerifyDirectoryContract(directory, FrozenConfiguration(), CanonicalConfigurationText());
    }
    static void VerifyDirectoryContract(const std::filesystem::path& directory,
        const RunConfiguration& expected, std::string_view configurationText,
        std::string_view artifactSchema = {})
    {
        if (!std::filesystem::is_directory(directory)) throw std::invalid_argument("POCKET_ARTIFACT_DIRECTORY_MISSING");
        const std::string manifest = ReadTextFile(directory / "manifest.txt");
        const auto field = [&manifest](std::string_view key) {
            const std::string prefix = std::string(key) + "=";
            const auto start = manifest.find(prefix); if (start == std::string::npos) throw std::invalid_argument("POCKET_MANIFEST_FIELD_MISSING");
            const auto end = manifest.find('\n', start); return manifest.substr(start + prefix.size(), end - start - prefix.size());
        };
        if (field("study") != expected.study || field("protocol") != expected.protocol || field("detector") != expected.detector ||
            field("protocol_document_sha256") != expected.protocolDocumentSha256 ||
            field("configuration_file_sha256") != FileSha256(directory / "configuration.conf") ||
            field("observations_sha256") != FileSha256(directory / "observations.csv") ||
            field("aggregates_sha256") != FileSha256(directory / "aggregates.csv"))
            throw std::invalid_argument("POCKET_ARTIFACT_TAMPER_OR_INCOMPLETE");
        if (ReadTextFile(directory / "configuration.conf") != configurationText ||
            field("configuration_sha256") != expected.configurationSha256 ||
            (!artifactSchema.empty() && field("artifact_schema") != artifactSchema))
            throw std::invalid_argument("POCKET_ARTIFACT_CONFIGURATION_MISMATCH");
    }
private:
    void Write(std::string_view name, std::string_view content)
    {
        std::ofstream out(staging_ / std::string(name), std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("POCKET_ARTIFACT_WRITE_FAILED");
        out << content; out.close(); if (!out) throw std::runtime_error("POCKET_ARTIFACT_WRITE_FAILED");
    }
    std::filesystem::path target_, staging_; bool published_ = false;
};
} // namespace EA::Pocket::Prospective
