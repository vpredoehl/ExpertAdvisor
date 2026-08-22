#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace EA::InferenceProfitability
{

inline constexpr int kDownClass = 0;
inline constexpr int kNeutralClass = 1;
inline constexpr int kUpClass = 2;

// This definition names the executable semantics of the observation. It is
// deliberately explicit that the statistic is not portfolio P&L.
inline constexpr const char* kMetricDefinitionCanonical =
    "inference_terminal_horizon_directional_log_return_v1;"
    "prediction=one_evaluable_inference_window;"
    "actionable=predicted_up_or_down_with_finite_positive_terminal_prices;"
    "up_return=ln(terminal_close/decision_close);"
    "down_return=-ln(terminal_close/decision_close);"
    "neutral=non_actionable;"
    "horizon=configured_terminal_prediction_horizon;"
    "win=actionable_return_gt_zero;"
    "loss=actionable_return_lt_zero;"
    "zero_return=actionable_neither_win_nor_loss;"
    "transaction_costs=not_modeled;"
    "position_sizing=not_modeled;"
    "leverage=not_modeled;"
    "overlap_capital_constraints=not_modeled;"
    "return_clipping=none";

struct Statistics
{
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t winningActionableCount = 0;
    std::uint64_t losingActionableCount = 0;
    double grossPositiveTerminalHorizonLogReturnSum = 0.0;
    double grossNegativeTerminalHorizonLogReturnSum = 0.0;
    double aggregateTerminalHorizonLogReturnSum = 0.0;

    std::uint64_t upActionableCount = 0;
    std::uint64_t downActionableCount = 0;
    std::uint64_t upWinningActionableCount = 0;
    std::uint64_t upLosingActionableCount = 0;
    std::uint64_t downWinningActionableCount = 0;
    std::uint64_t downLosingActionableCount = 0;
    double upTerminalHorizonLogReturnSum = 0.0;
    double downTerminalHorizonLogReturnSum = 0.0;
    double upGrossPositiveTerminalHorizonLogReturnSum = 0.0;
    double upGrossNegativeTerminalHorizonLogReturnSum = 0.0;
    double downGrossPositiveTerminalHorizonLogReturnSum = 0.0;
    double downGrossNegativeTerminalHorizonLogReturnSum = 0.0;

    std::optional<double> AverageTerminalHorizonLogReturnPerActionablePrediction() const;
};

class Accumulator
{
public:
    Accumulator();

    // Every evaluated classification window is observed, including Neutral
    // predictions and directional predictions with invalid terminal prices.
    void Observe(int predictedClass, float decisionClose, float terminalClose);

    const Statistics& statistics() const { return statistics_; }
    std::string SourceContentHash() const;

private:
    void HashText(const char* data, std::size_t size);

    Statistics statistics_;
    std::uint64_t sourceHash_ = 14695981039346656037ULL;
};

std::string DeterministicHash(const std::string& canonicalText);
std::string MetricDefinitionHash();
std::string StatisticsCanonicalText(const Statistics& statistics);

} // namespace EA::InferenceProfitability
