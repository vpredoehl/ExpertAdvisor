#pragma once

#include "InferenceProfitability.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::StrategyEvaluation
{

inline constexpr int kStrategyConfigurationSchemaVersion = 1;
inline constexpr const char* kBaselineTerminalStrategyFamily =
    "baseline_terminal_directional_log_return";
inline constexpr int kBaselineTerminalStrategyVersion = 1;

struct ConfigurationEntry
{
    std::string key;
    std::string value;

    bool operator==(const ConfigurationEntry&) const = default;
};

// Entries are a logical map. Canonicalization sorts by key and rejects
// duplicate keys so caller insertion order cannot affect identity.
struct StrategyConfiguration
{
    int schemaVersion = kStrategyConfigurationSchemaVersion;
    std::vector<ConfigurationEntry> entries;
};

std::string CanonicalConfiguration(
    const StrategyConfiguration& configuration);
std::string ConfigurationHash(
    const StrategyConfiguration& configuration);

struct StrategyIdentity
{
    std::string family;
    int version = 0;
    std::string canonicalConfiguration;
    std::string configurationHash;
    std::string canonical;
    std::string hash;

    bool operator==(const StrategyIdentity&) const = default;
};

StrategyIdentity BuildStrategyIdentity(
    const std::string& family,
    int version,
    const StrategyConfiguration& configuration);

enum class PositionDirection
{
    flat,
    shortPosition,
    longPosition
};

PositionDirection DirectionForPredictedClass(int predictedClass);

struct PredictionProbabilities
{
    std::array<float, 3> downNeutralUp{};
};

// The first three fields are the complete input to the historical
// profitability calculation. Optional prediction and market-path fields make
// the data boundary explicit without changing baseline semantics. A future
// path-dependent strategy must require and validate the fields it consumes.
struct MarketPathPoint
{
    std::string timestamp;
    float open = 0.0f;
    float high = 0.0f;
    float low = 0.0f;
    float close = 0.0f;
};

struct StrategyEvaluationObservation
{
    StrategyEvaluationObservation() = default;
    StrategyEvaluationObservation(int predictedClassValue,
                                  float decisionCloseValue,
                                  float terminalCloseValue)
        : predictedClass(predictedClassValue),
          decisionClose(decisionCloseValue),
          terminalClose(terminalCloseValue)
    {
    }

    int predictedClass = InferenceProfitability::kNeutralClass;
    float decisionClose = 0.0f;
    float terminalClose = 0.0f;
    std::optional<PredictionProbabilities> probabilities;
    std::string decisionTimestamp;
    std::string terminalTimestamp;
    std::vector<MarketPathPoint> marketPath;
};

struct StrategyEvaluationInput
{
    std::vector<StrategyEvaluationObservation> observations;
};

struct StrategyEvaluationResult
{
    StrategyIdentity strategyIdentity;
    InferenceProfitability::Statistics statistics;
    std::string sourceContentHash;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
};

// Scientific inference identity is supplied independently and is never
// modified by strategy configuration. Market-data provenance is explicit
// because terminal source content alone will be insufficient for future
// intrahorizon policies.
struct StrategyEvaluationProvenance
{
    std::string inferenceScientificIdentityCanonical;
    std::string inferenceScientificIdentityHash;
    std::string symbol;
    std::uint64_t predictionHorizon = 0;
    std::string inferenceStart;
    std::string inferenceEnd;
    std::string metricDefinitionHash;
    std::string marketDataProvenanceHash;
};

struct StrategyEvaluationIdentity
{
    std::string canonical;
    std::string hash;

    bool operator==(const StrategyEvaluationIdentity&) const = default;
};

StrategyEvaluationIdentity BuildStrategyEvaluationIdentity(
    const StrategyIdentity& strategyIdentity,
    const StrategyEvaluationProvenance& provenance);

class TradingStrategy
{
public:
    virtual ~TradingStrategy() = default;

    virtual const StrategyIdentity& Identity() const noexcept = 0;
    virtual StrategyEvaluationResult Evaluate(
        const StrategyEvaluationInput& input) const = 0;
};

// This is the exact historical terminal-return policy. It deliberately
// delegates to InferenceProfitability::Accumulator so operation order,
// actionability, hashes, and metric identity cannot drift independently.
class BaselineTerminalStrategy final : public TradingStrategy
{
public:
    explicit BaselineTerminalStrategy(
        StrategyConfiguration configuration = {});

    const StrategyIdentity& Identity() const noexcept override;
    StrategyEvaluationResult Evaluate(
        const StrategyEvaluationInput& input) const override;

private:
    StrategyIdentity identity_;
};

StrategyEvaluationResult EvaluateStrategy(
    const TradingStrategy& strategy,
    const StrategyEvaluationInput& input);

} // namespace EA::StrategyEvaluation
