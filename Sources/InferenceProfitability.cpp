#include "InferenceProfitability.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace EA::InferenceProfitability
{
namespace
{

void AppendField(std::string& output,
                 std::string_view name,
                 const std::string& value)
{
    output.append(name);
    output.push_back('=');
    output.append(std::to_string(value.size()));
    output.push_back(':');
    output.append(value);
    output.push_back(';');
}

std::string CanonicalDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("inference_profitability_nonfinite_statistic");
    if (value == 0.0) value = 0.0;
    std::array<char, 128> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general);
    if (result.ec != std::errc{})
        throw std::runtime_error("inference_profitability_number_format_failed");
    return std::string(buffer.data(), result.ptr);
}

std::string Hex64(std::uint64_t value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - 1 - index) * 4);
        encoded[index] = digits[(value >> shift) & 0x0fU];
    }
    return std::string(encoded.begin(), encoded.end());
}

std::string Hex32(std::uint32_t value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - 1 - index) * 4);
        encoded[index] = digits[(value >> shift) & 0x0fU];
    }
    return std::string(encoded.begin(), encoded.end());
}

} // namespace

std::optional<double>
Statistics::AverageTerminalHorizonLogReturnPerActionablePrediction() const
{
    if (actionableCount == 0)
        return std::nullopt;
    return aggregateTerminalHorizonLogReturnSum /
        static_cast<double>(actionableCount);
}

Accumulator::Accumulator()
{
    constexpr std::string_view prefix =
        "inference_profitability_source_content_v1;";
    HashText(prefix.data(), prefix.size());
}

void Accumulator::HashText(const char* data, std::size_t size)
{
    for (std::size_t index = 0; index < size; ++index)
    {
        sourceHash_ ^= static_cast<unsigned char>(data[index]);
        sourceHash_ *= 1099511628211ULL;
    }
}

void Accumulator::Observe(int predictedClass,
                          float decisionClose,
                          float terminalClose)
{
    const std::string sourceRecord =
        "ordinal=" + std::to_string(statistics_.predictionCount) +
        ";predicted_class=" + std::to_string(predictedClass) +
        ";decision_close_f32=" +
        Hex32(std::bit_cast<std::uint32_t>(decisionClose)) +
        ";terminal_close_f32=" +
        Hex32(std::bit_cast<std::uint32_t>(terminalClose)) + ";";
    HashText(sourceRecord.data(), sourceRecord.size());

    ++statistics_.predictionCount;
    if (predictedClass == kNeutralClass)
        return;
    if (predictedClass != kDownClass && predictedClass != kUpClass)
        throw std::invalid_argument("inference_profitability_invalid_predicted_class");
    if (!std::isfinite(decisionClose) || !std::isfinite(terminalClose) ||
        decisionClose <= 0.0f || terminalClose <= 0.0f)
        return;

    const double terminalLogReturn =
        std::log(static_cast<double>(terminalClose) /
                 static_cast<double>(decisionClose));
    const double directionalLogReturn = predictedClass == kUpClass
        ? terminalLogReturn
        : -terminalLogReturn;

    ++statistics_.actionableCount;
    statistics_.aggregateTerminalHorizonLogReturnSum += directionalLogReturn;
    if (predictedClass == kUpClass)
    {
        ++statistics_.upActionableCount;
        statistics_.upTerminalHorizonLogReturnSum += directionalLogReturn;
    }
    else
    {
        ++statistics_.downActionableCount;
        statistics_.downTerminalHorizonLogReturnSum += directionalLogReturn;
    }

    if (directionalLogReturn > 0.0)
    {
        ++statistics_.winningActionableCount;
        statistics_.grossPositiveTerminalHorizonLogReturnSum +=
            directionalLogReturn;
        if (predictedClass == kUpClass)
        {
            ++statistics_.upWinningActionableCount;
            statistics_.upGrossPositiveTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
        else
        {
            ++statistics_.downWinningActionableCount;
            statistics_.downGrossPositiveTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
    }
    else if (directionalLogReturn < 0.0)
    {
        ++statistics_.losingActionableCount;
        statistics_.grossNegativeTerminalHorizonLogReturnSum +=
            directionalLogReturn;
        if (predictedClass == kUpClass)
        {
            ++statistics_.upLosingActionableCount;
            statistics_.upGrossNegativeTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
        else
        {
            ++statistics_.downLosingActionableCount;
            statistics_.downGrossNegativeTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
    }
}

std::string Accumulator::SourceContentHash() const
{
    return "fnv1a64:" + Hex64(sourceHash_);
}

std::string DeterministicHash(const std::string& canonicalText)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonicalText)
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return "fnv1a64:" + Hex64(hash);
}

std::string MetricDefinitionHash()
{
    return DeterministicHash(kMetricDefinitionCanonical);
}

std::string StatisticsCanonicalText(const Statistics& statistics)
{
    std::string canonical = "inference_profitability_statistics_v1;";
    AppendField(canonical, "prediction_count",
                std::to_string(statistics.predictionCount));
    AppendField(canonical, "actionable_count",
                std::to_string(statistics.actionableCount));
    AppendField(canonical, "winning_actionable_count",
                std::to_string(statistics.winningActionableCount));
    AppendField(canonical, "losing_actionable_count",
                std::to_string(statistics.losingActionableCount));
    AppendField(canonical, "gross_positive_terminal_horizon_log_return_sum",
                CanonicalDouble(
                    statistics.grossPositiveTerminalHorizonLogReturnSum));
    AppendField(canonical, "gross_negative_terminal_horizon_log_return_sum",
                CanonicalDouble(
                    statistics.grossNegativeTerminalHorizonLogReturnSum));
    AppendField(canonical, "aggregate_terminal_horizon_log_return_sum",
                CanonicalDouble(
                    statistics.aggregateTerminalHorizonLogReturnSum));
    const auto average =
        statistics.AverageTerminalHorizonLogReturnPerActionablePrediction();
    AppendField(canonical,
                "average_terminal_horizon_log_return_per_actionable_prediction",
                average ? CanonicalDouble(*average) : "NULL");
    return canonical;
}

} // namespace EA::InferenceProfitability
