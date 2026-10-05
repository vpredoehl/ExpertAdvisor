#include "CausalFibonacciRetracementLifecycleHistoricalArtifact.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace EA::FibonacciResearch::RetracementLifecycle
{
namespace
{
std::string Csv(std::string_view value)
{
    std::string result{"\""};
    for (const char character : value)
    {
        if (character == '\"') result += "\"\"";
        else result += character;
    }
    return result + '\"';
}

std::string Number(double value)
{
    std::ostringstream result;
    result << std::setprecision(17) << value;
    return result.str();
}

template<typename T>
std::string Optional(const std::optional<T>& value)
{
    if (!value.has_value()) return {};
    if constexpr (std::is_same_v<T, double>) return Number(*value);
    return std::to_string(*value);
}

void AddOccurrence(std::vector<std::string>& values,
                   const std::optional<Occurrence>& occurrence)
{
    values.push_back(occurrence ? std::to_string(occurrence->bar) : "");
    values.push_back(occurrence ? std::to_string(occurrence->timestamp) : "");
    values.push_back(occurrence ? Number(occurrence->price) : "");
}

void AddLevel(std::vector<std::string>& values, const LevelState& level)
{
    values.push_back(Number(level.price));
    AddOccurrence(values, level.reached);
    values.push_back(Optional(level.barsBToReached));
    AddOccurrence(values, level.firstDirectionalClose);
    AddOccurrence(values, level.firstDirectionalBreak);
    AddOccurrence(values, level.firstCloseBackThroughLevel);
}

void AddEntry(std::vector<std::string>& values, const EntryExcursion& entry)
{
    AddOccurrence(values, entry.entry);
    values.push_back(Optional(entry.barsBToEntry));
    values.push_back(Optional(entry.barsRetracementToEntry));
    values.push_back(Optional(entry.barsEntryToD));
    values.push_back(Optional(entry.barsEntryToAPenetration));
    values.push_back(Optional(entry.barsEntryToACloseBeyond));
    values.push_back(Optional(entry.highestHigh));
    values.push_back(Optional(entry.lowestLow));
    values.push_back(Optional(entry.maximumFavorableExcursion));
    values.push_back(Optional(entry.maximumAdverseExcursion));
    values.push_back(Optional(entry.maximumFavorableExcursionABRanges));
    values.push_back(Optional(entry.maximumAdverseExcursionABRanges));
    values.push_back(Optional(entry.preDHighestHigh));
    values.push_back(Optional(entry.preDLowestLow));
    values.push_back(Optional(entry.preDMaximumFavorableExcursion));
    values.push_back(Optional(entry.preDMaximumAdverseExcursion));
    values.push_back(Optional(entry.preDMaximumFavorableExcursionABRanges));
    values.push_back(Optional(entry.preDMaximumAdverseExcursionABRanges));
}

void AddLevelHeader(std::vector<std::string>& values, std::string_view name)
{
    const std::string prefix{name};
    for (const auto& suffix : {"price", "reach_bar", "reach_timestamp",
                               "reach_price", "bars_b_to_reach",
                               "directional_close_bar", "directional_close_timestamp",
                               "directional_close_price", "directional_break_bar",
                               "directional_break_timestamp", "directional_break_price",
                               "close_back_bar", "close_back_timestamp", "close_back_price"})
        values.push_back(prefix + "_" + suffix);
}

void AddEntryHeader(std::vector<std::string>& values, std::string_view name)
{
    const std::string prefix{name};
    for (const auto& suffix : {"entry_bar", "entry_timestamp", "entry_price",
                               "bars_b_to_entry", "bars_retracement_to_entry",
                               "bars_entry_to_d", "bars_entry_to_a_penetration",
                               "bars_entry_to_a_close_beyond", "highest_high", "lowest_low",
                               "mfe_raw", "mae_raw", "mfe_ab_ranges", "mae_ab_ranges", "pre_d_highest_high", "pre_d_lowest_low", "pre_d_mfe_raw", "pre_d_mae_raw", "pre_d_mfe_ab_ranges", "pre_d_mae_ab_ranges"})
        values.push_back(prefix + "_" + suffix);
}

std::string Join(std::vector<std::string> values)
{
    std::ostringstream result;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index) result << ',';
        result << Csv(values[index]);
    }
    return result.str();
}

std::string Direction(TG3::ABDirection direction)
{
    return direction == TG3::ABDirection::UpAB ? "up_ab" : "down_ab";
}

std::string Hypothesis(DTargetHypothesis hypothesis)
{
    return hypothesis == DTargetHypothesis::Extension1272 ? "1.272" : "1.618";
}

std::string AnchorSelectionPolicy(TG3::AnchorSelectionPolicy)
{
    return "most_recent_prior_opposite_confirmed_fractal";
}

std::string Json(std::string_view value)
{
    std::string result;
    for (const char character : value)
    {
        if (character == '\\' || character == '\"') result += '\\';
        result += character;
    }
    return result;
}
} // namespace

std::string LifecycleObservationCsvHeader()
{
    std::vector<std::string> values{
        "protocol_version", "symbol", "timeframe", "d_hypothesis", "direction",
        "a_bar", "a_timestamp", "a_price", "a_confirmation_bar",
        "a_confirmation_timestamp", "b_bar", "b_timestamp", "b_price",
        "b_confirmation_bar", "b_confirmation_timestamp", "ab_availability_bar",
        "ab_availability_timestamp", "ab_price_range", "anchor_selection_policy"};
    AddLevelHeader(values, "retracement_0382");
    AddLevelHeader(values, "retracement_0500");
    AddLevelHeader(values, "retracement_0618");
    values.insert(values.end(), {"d_price", "a_penetration_bar", "a_penetration_timestamp",
        "a_penetration_price", "a_close_beyond_bar", "a_close_beyond_timestamp",
        "a_close_beyond_price", "d_reached_bar", "d_reached_timestamp", "d_reached_price",
        "bars_b_to_d", "bars_b_to_a_penetration", "bars_b_to_a_close_beyond",
        "right_censored"});
    for (const auto& level : {"0382", "0500", "0618"})
        for (const auto& candidate : {"directional_close", "directional_break",
                                      "close_back_through"})
            AddEntryHeader(values, std::string("retracement_") + level + "_" + candidate);
    return Join(std::move(values));
}

std::string LifecycleObservationCsvRow(const HistoricalRecord& historical,
                                       std::string_view symbol,
                                       std::string_view timeframe)
{
    const Record& record = historical.lifecycle;
    const TG3::ABStructure& ab = record.sourceAB;
    std::vector<std::string> values{
        std::string(kProtocolVersion), std::string(symbol), std::string(timeframe),
        Hypothesis(historical.dTarget), Direction(ab.identity.direction),
        std::to_string(ab.identity.aBar), std::to_string(ab.identity.aTimestamp),
        Number(ab.aPrice), std::to_string(ab.aConfirmationBar),
        std::to_string(ab.aConfirmationTimestamp), std::to_string(ab.identity.bBar),
        std::to_string(ab.identity.bTimestamp), Number(ab.bPrice),
        std::to_string(ab.bConfirmationBar), std::to_string(ab.bConfirmationTimestamp),
        std::to_string(ab.identity.availabilityBar),
        std::to_string(ab.identity.availabilityTimestamp), Number(ab.priceRange),
        AnchorSelectionPolicy(ab.anchorSelectionPolicy)};
    AddLevel(values, record.retracement0382);
    AddLevel(values, record.retracement0500);
    AddLevel(values, record.retracement0618);
    values.push_back(Number(record.dPrice));
    AddOccurrence(values, record.firstAPenetration);
    AddOccurrence(values, record.firstCloseBeyondA);
    AddOccurrence(values, record.firstDReached);
    values.push_back(Optional(record.barsBToD));
    values.push_back(Optional(record.barsBToAPenetration));
    values.push_back(Optional(record.barsBToACloseBeyond));
    values.push_back(record.rightCensored ? "true" : "false");
    for (const EntryExcursion* entry : {&record.directionalClose0382,
             &record.directionalBreak0382, &record.closeBackThrough0382,
             &record.directionalClose0500, &record.directionalBreak0500,
             &record.closeBackThrough0500, &record.directionalClose0618,
             &record.directionalBreak0618, &record.closeBackThrough0618})
        AddEntry(values, *entry);
    return Join(std::move(values));
}

class HistoricalArtifactWriter::Implementation
{
public:
    Implementation(std::filesystem::path outputDirectory,
                   HistoricalArtifactProvenance provenance)
        : outputDirectory_(std::move(outputDirectory)), provenance_(std::move(provenance))
    {
        if (provenance_.symbol.empty() || provenance_.timeframe.empty())
            throw std::invalid_argument("retracement lifecycle provenance is incomplete");
    }

    void Complete()
    {
        if (complete_) return;
        std::sort(records_.begin(), records_.end(), [](const HistoricalRecord& left,
                                                        const HistoricalRecord& right)
        {
            const auto& a = left.lifecycle.sourceAB.identity;
            const auto& b = right.lifecycle.sourceAB.identity;
            return std::tie(a.availabilityBar, a.bBar, a.direction, a.aTimestamp,
                            left.dTarget) < std::tie(b.availabilityBar, b.bBar,
                            b.direction, b.aTimestamp, right.dTarget);
        });
        std::filesystem::create_directories(outputDirectory_);
        std::ofstream observations(outputDirectory_ / "observations.csv",
                                  std::ios::binary | std::ios::trunc);
        if (!observations) throw std::runtime_error("cannot write lifecycle observations");
        observations << LifecycleObservationCsvHeader() << '\n';
        for (const HistoricalRecord& record : records_)
            observations << LifecycleObservationCsvRow(record, provenance_.symbol,
                                                        provenance_.timeframe) << '\n';
        std::ofstream manifest(outputDirectory_ / "manifest.json",
                               std::ios::binary | std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write lifecycle manifest");
        manifest << "{\n"
                 << "  \"protocol_version\": \"" << Json(kProtocolVersion) << "\",\n"
                 << "  \"symbol\": \"" << Json(provenance_.symbol) << "\",\n"
                 << "  \"timeframe\": \"" << Json(provenance_.timeframe) << "\",\n"
                 << "  \"configuration_path\": \"" << Json(provenance_.configurationPath.string()) << "\",\n"
                 << "  \"configuration_fingerprint\": \"" << Json(provenance_.configurationFingerprint) << "\",\n"
                 << "  \"baseline_commit\": \"" << Json(provenance_.baselineCommit) << "\",\n"
                 << "  \"requested_range\": \"" << Json(provenance_.requestedRange) << "\",\n"
                 << "  \"database_access\": \"read_only_repeatable_read\",\n"
                 << "  \"preflight_rows\": " << quality_.preflightRows << ",\n"
                 << "  \"streamed_rows\": " << quality_.streamedRows << ",\n"
                 << "  \"duplicate_timestamp_count\": " << quality_.duplicateTimestampCount << ",\n"
                 << "  \"observation_count\": " << records_.size() << ",\n"
                 << "  \"reproduction_command\": \"" << Json(provenance_.reproductionCommand) << "\"\n}\n";
        complete_ = true;
    }

    std::filesystem::path outputDirectory_;
    HistoricalArtifactProvenance provenance_;
    HistoricalArtifactDataQuality quality_;
    std::vector<HistoricalRecord> records_;
    bool complete_ = false;
};

HistoricalArtifactWriter::HistoricalArtifactWriter(
    std::filesystem::path outputDirectory, HistoricalArtifactProvenance provenance)
    : implementation_(std::make_unique<Implementation>(std::move(outputDirectory),
                                                        std::move(provenance))) {}
HistoricalArtifactWriter::~HistoricalArtifactWriter() = default;
void HistoricalArtifactWriter::AddRecord(HistoricalRecord record)
{ implementation_->records_.push_back(std::move(record)); }
void HistoricalArtifactWriter::SetDataQuality(HistoricalArtifactDataQuality quality)
{ implementation_->quality_ = std::move(quality); }
void HistoricalArtifactWriter::Complete() { implementation_->Complete(); }
std::size_t HistoricalArtifactWriter::ObservationCount() const
{ return implementation_->records_.size(); }

} // namespace EA::FibonacciResearch::RetracementLifecycle
