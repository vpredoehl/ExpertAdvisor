#pragma once

#include "../InferenceProfitability.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace EA::StrategyEvaluation
{

inline constexpr int kStrategyConfigurationSchemaVersion = 1;
inline constexpr const char* kBaselineTerminalStrategyFamily =
    "baseline_terminal_directional_log_return";
inline constexpr int kBaselineTerminalStrategyVersion = 1;

inline constexpr int kMarketPathSchemaVersion = 1;
inline constexpr int kMarketPathCanonicalizationVersion = 1;

inline constexpr const char* kFixedStopLossStrategyFamily =
    "fixed_stop_loss";
inline constexpr int kFixedStopLossStrategyVersion = 1;
inline constexpr int kFixedStopLossConfigurationSchemaVersion = 1;
inline constexpr const char* kFixedStopExecutionRule =
    "single_fixed_protective_stop_ohlc_v1";
inline constexpr int kFixedStopExecutionRuleVersion = 1;
inline constexpr const char* kProbabilityConditionedStopLossStrategyFamily =
    "probability_conditioned_stop_loss";
inline constexpr int kProbabilityConditionedStopLossStrategyVersion = 1;
inline constexpr int
    kProbabilityConditionedStopLossConfigurationSchemaVersion = 1;
inline constexpr const char* kDirectionalProbabilityMappingIdentity =
    "directional_probability_linear_bounded_v1";
inline constexpr double kPhase18ADefaultBaseStopLogarithmicDistance = 0.0010;
inline constexpr double kPhase18ADefaultMinimumStopMultiplier = 0.75;
inline constexpr double kPhase18ADefaultMaximumStopMultiplier = 1.25;
inline constexpr const char* kBaselineTerminalOutputIdentity =
    "baseline_terminal_v1";
inline constexpr const char* kFixedStopLossOutputIdentity =
    "fixed_stop_loss_v1";
inline constexpr const char* kProbabilityConditionedStopLossOutputIdentity =
    "probability_conditioned_stop_loss_v1";
inline constexpr std::array<double, 3> kControlledFixedStopDistanceGrid{{
    0.0005, 0.0010, 0.0020}};
inline constexpr const char* kFixedStopMetricDefinitionCanonical =
    "fixed_stop_directional_log_return_v1;"
    "entry=decision_bar_close;"
    "direction=predicted_class_down_or_up;"
    "neutral=no_action;"
    "initial_stop=entry_times_exp(direction_sign_times_negative_log_distance);"
    "stop=immutable;"
    "path=ordered_subsequent_bars_through_terminal_horizon;"
    "long_touch=low_less_than_or_equal_stop;"
    "short_touch=high_greater_than_or_equal_stop;"
    "gap=bar_open_beyond_stop_fills_at_open;"
    "non_gap_stop_fill=stop_price;"
    "no_stop=terminal_bar_close;"
    "take_profit=none;trailing=none;reentry=none;"
    "transaction_costs=not_modeled;slippage=not_modeled;"
    "price_domain=authoritative_path_price_domain;"
    "up_return=ln(exit/entry);down_return=-ln(exit/entry)";

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

    bool operator==(const PredictionProbabilities&) const = default;
};

// This reproduces the current inference selection rule exactly: strict unique
// down/up maxima are directional and every tie involving the maximum is
// neutral. The function intentionally depends only on model output.
int PredictedClassForProbabilities(
    const PredictionProbabilities& probabilities);

struct MarketPathPoint
{
    std::uint64_t sourceRow = 0;
    std::optional<std::int64_t> timestampUnixSeconds;
    float open = 0.0f;
    float high = 0.0f;
    float low = 0.0f;
    float close = 0.0f;

    bool operator==(const MarketPathPoint&) const = default;
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

    std::uint64_t observationOrdinal = 0;
    std::uint64_t inferenceWindowStartRow = 0;
    std::uint64_t decisionRow = 0;
    std::uint64_t terminalRow = 0;
    int predictedClass = InferenceProfitability::kNeutralClass;
    float decisionClose = 0.0f;
    float terminalClose = 0.0f;
    std::optional<PredictionProbabilities> probabilities;
    std::optional<std::int64_t> decisionTimestampUnixSeconds;
    std::optional<std::int64_t> terminalTimestampUnixSeconds;
    std::vector<MarketPathPoint> marketPath;

    bool operator==(const StrategyEvaluationObservation&) const = default;
};

// All strings here are semantic identifiers, not display labels. The adapter
// supplies them from the authoritative inference/Tensor path, and path
// canonicalization binds every field.
struct MarketPathProvenance
{
    int schemaVersion = kMarketPathSchemaVersion;
    int canonicalizationVersion = kMarketPathCanonicalizationVersion;
    std::string adapterFamily;
    int adapterVersion = 0;
    long long modelId = 0;
    std::string inferenceScientificIdentityCanonical;
    std::string inferenceScientificIdentityHash;
    std::string symbol;
    std::uint64_t inferenceWindowSize = 0;
    std::uint64_t predictionHorizon = 0;
    std::string evaluationStart;
    std::string evaluationEnd;
    std::uint64_t barIntervalSeconds = 0;
    std::string timestampSemantics;
    std::string ohlcIntervalSemantics;
    std::string priceDomain;
    std::string marketDataSource;
    std::string marketDataSourceRelation;
    std::string pathOrdering;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;

    bool operator==(const MarketPathProvenance&) const = default;
};

// This value is immutable after validated construction. It owns the exact
// ordered observations and their complete canonical representation.
class AuthoritativeMarketPath final
{
public:
    AuthoritativeMarketPath(const AuthoritativeMarketPath&) = default;
    AuthoritativeMarketPath(AuthoritativeMarketPath&&) noexcept = default;
    AuthoritativeMarketPath& operator=(const AuthoritativeMarketPath&) = delete;
    AuthoritativeMarketPath& operator=(AuthoritativeMarketPath&&) = delete;

    const MarketPathProvenance& Provenance() const noexcept;
    const std::vector<StrategyEvaluationObservation>& Observations() const
        noexcept;
    const std::string& Canonical() const noexcept;
    const std::string& Hash() const noexcept;

private:
    friend AuthoritativeMarketPath BuildAuthoritativeMarketPath(
        MarketPathProvenance,
        std::vector<StrategyEvaluationObservation>);

    AuthoritativeMarketPath(
        MarketPathProvenance provenance,
        std::vector<StrategyEvaluationObservation> observations,
        std::string canonical,
        std::string hash);

    MarketPathProvenance provenance_;
    std::vector<StrategyEvaluationObservation> observations_;
    std::string canonical_;
    std::string hash_;
};

AuthoritativeMarketPath BuildAuthoritativeMarketPath(
    MarketPathProvenance provenance,
    std::vector<StrategyEvaluationObservation> observations);

class StrategyEvaluationInput final
{
public:
    // Legacy/baseline-only input preserves the exact historical contract.
    explicit StrategyEvaluationInput(
        std::vector<StrategyEvaluationObservation> observations);
    explicit StrategyEvaluationInput(AuthoritativeMarketPath marketPath);

    const std::vector<StrategyEvaluationObservation>& Observations() const
        noexcept;
    const AuthoritativeMarketPath* MarketPath() const noexcept;

private:
    std::vector<StrategyEvaluationObservation> observations_;
    std::shared_ptr<const AuthoritativeMarketPath> marketPath_;
};

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

StrategyEvaluationProvenance BuildStrategyEvaluationProvenance(
    const AuthoritativeMarketPath& marketPath);

struct StrategyEvaluationIdentity
{
    std::string canonical;
    std::string hash;

    bool operator==(const StrategyEvaluationIdentity&) const = default;
};

StrategyEvaluationIdentity BuildStrategyEvaluationIdentity(
    const StrategyIdentity& strategyIdentity,
    const StrategyEvaluationProvenance& provenance);

enum class StrategyExitReason
{
    noAction,
    terminalExit,
    fixedStopExit
};

const char* StrategyExitReasonText(StrategyExitReason reason) noexcept;

struct StrategyExecutionResult
{
    std::uint64_t observationOrdinal = 0;
    PositionDirection direction = PositionDirection::flat;
    StrategyExitReason reason = StrategyExitReason::noAction;
    std::optional<std::int64_t> entryTimestampUnixSeconds;
    std::optional<std::int64_t> exitTimestampUnixSeconds;
    std::optional<double> entryPrice;
    std::optional<double> initialStopPrice;
    std::optional<double> exitPrice;
    std::optional<double> directionalLogReturn;
    std::optional<std::uint64_t> triggeringPathPointOrdinal;
    std::optional<std::uint64_t> triggeringSourceRow;

    bool operator==(const StrategyExecutionResult&) const = default;
};

struct StrategyEvaluationResult
{
    StrategyIdentity strategyIdentity;
    std::optional<StrategyEvaluationIdentity> evaluationIdentity;
    InferenceProfitability::Statistics statistics;
    std::string sourceContentHash;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::vector<StrategyExecutionResult> executionResults;
    std::string resultCanonical;
    std::string resultHash;
};

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

struct FixedStopLossConfiguration
{
    int schemaVersion = kFixedStopLossConfigurationSchemaVersion;
    double logarithmicDistance = 0.0;
};

class FixedStopLossStrategy final : public TradingStrategy
{
public:
    explicit FixedStopLossStrategy(FixedStopLossConfiguration configuration);

    const StrategyIdentity& Identity() const noexcept override;
    double InitialStopPrice(int predictedClass, float entryPrice) const;
    StrategyEvaluationResult Evaluate(
        const StrategyEvaluationInput& input) const override;

private:
    FixedStopLossConfiguration configuration_;
    StrategyIdentity identity_;
};

struct ProbabilityConditionedStopLossConfiguration
{
    int schemaVersion =
        kProbabilityConditionedStopLossConfigurationSchemaVersion;
    double baseStopLogarithmicDistance =
        kPhase18ADefaultBaseStopLogarithmicDistance;
    double minimumStopMultiplier =
        kPhase18ADefaultMinimumStopMultiplier;
    double maximumStopMultiplier =
        kPhase18ADefaultMaximumStopMultiplier;
};

struct ProbabilityConditionedStopDecision
{
    double directionalProbability = 0.0;
    double normalizedDirectionalConfidence = 0.0;
    double stopMultiplier = 0.0;
    double effectiveStopLogarithmicDistance = 0.0;
    double initialStopPrice = 0.0;

    bool operator==(const ProbabilityConditionedStopDecision&) const = default;
};

// Phase 18A v1 conditions only the immutable initial stop. Its probability is
// the decision-time probability assigned to the entered Down/Up direction.
class ProbabilityConditionedStopLossStrategy final : public TradingStrategy
{
public:
    explicit ProbabilityConditionedStopLossStrategy(
        ProbabilityConditionedStopLossConfiguration configuration = {});

    const StrategyIdentity& Identity() const noexcept override;
    const ProbabilityConditionedStopLossConfiguration& Configuration() const
        noexcept;
    double NormalizedDirectionalConfidence(
        double directionalProbability) const;
    double StopMultiplierForNormalizedConfidence(
        double normalizedDirectionalConfidence) const;
    ProbabilityConditionedStopDecision StopDecision(
        const StrategyEvaluationObservation& observation) const;
    StrategyEvaluationResult Evaluate(
        const StrategyEvaluationInput& input) const override;

private:
    ProbabilityConditionedStopLossConfiguration configuration_;
    StrategyIdentity identity_;
};

std::string FixedStopMetricDefinitionHash();

struct ControlledStrategyVariantResult
{
    std::optional<double> fixedStopLogarithmicDistance;
    StrategyEvaluationResult evaluation;
    std::uint64_t noActionCount = 0;
    std::uint64_t fixedStopExitCount = 0;
    std::uint64_t terminalExitCount = 0;
    std::uint64_t zeroOutcomeCount = 0;
    double aggregateReturnDeltaVersusBaseline = 0.0;
    double averageActionableReturnDeltaVersusBaseline = 0.0;
    std::string profitabilityComparisonVersusBaseline;
};

struct ControlledFixedStopExperimentResult
{
    std::string experimentIdentityCanonical;
    std::string experimentIdentityHash;
    std::string comparisonMetricDefinitionCanonical;
    std::string comparisonMetricDefinitionHash;
    std::string canonicalCsv;
    std::string resultHash;
    std::vector<ControlledStrategyVariantResult> variants;
};

// Evaluates the baseline followed by the pre-registered, ordered fixed-stop
// grid against one immutable path. Invalid, duplicate, or altered distances
// and every population/accounting mismatch fail closed.
ControlledFixedStopExperimentResult EvaluateControlledFixedStopExperiment(
    const AuthoritativeMarketPath& marketPath,
    const std::vector<double>& fixedStopLogarithmicDistances = {
        kControlledFixedStopDistanceGrid.begin(),
        kControlledFixedStopDistanceGrid.end()});

struct Phase18AConfidenceBucketMetrics
{
    double lowerInclusive = 0.0;
    double upperExclusive = 0.0;
    bool includesUpperBound = false;
    std::uint64_t actionableCount = 0;
    std::uint64_t winningCount = 0;
    std::uint64_t losingCount = 0;
    std::uint64_t zeroOutcomeCount = 0;
    std::uint64_t stopHitCount = 0;
    std::uint64_t terminalExitCount = 0;
    double aggregateDirectionalLogReturn = 0.0;
    std::optional<double> averageDirectionalLogReturn;
};

struct Phase18AStrategyMetrics
{
    std::uint64_t observationCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t nonActionableCount = 0;
    std::uint64_t winningCount = 0;
    std::uint64_t losingCount = 0;
    std::uint64_t zeroOutcomeCount = 0;
    std::uint64_t stopHitCount = 0;
    std::uint64_t terminalExitCount = 0;
    double aggregateDirectionalLogReturn = 0.0;
    std::optional<double> averageDirectionalLogReturn;
    std::optional<double> winningRate;
    std::optional<double> losingRate;
    std::optional<double> stopHitRate;
    std::optional<double> terminalExitRate;
    std::optional<double> averageHoldingDurationSeconds;
    std::optional<double> medianHoldingDurationSeconds;
    std::optional<double> maximumAdverseExcursion;
    std::optional<double> maximumFavorableExcursion;
    double maximumDrawdown = 0.0;
    std::optional<double> averageEffectiveStopLogarithmicDistance;
    std::optional<double> averageStopMultiplier;
};

struct Phase18AObservationEvidence
{
    std::string strategyOutputIdentity;
    std::uint64_t observationOrdinal = 0;
    int predictedClass = InferenceProfitability::kNeutralClass;
    PositionDirection direction = PositionDirection::flat;
    bool actionable = false;
    std::int64_t decisionTimestampUnixSeconds = 0;
    double decisionPrice = 0.0;
    PredictionProbabilities probabilities;
    std::optional<double> directionalProbability;
    std::optional<double> normalizedDirectionalConfidence;
    std::optional<double> baseStopLogarithmicDistance;
    std::optional<double> appliedStopMultiplier;
    std::optional<double> effectiveStopLogarithmicDistance;
    StrategyExecutionResult execution;
    std::optional<double> holdingDurationSeconds;
    std::optional<double> maximumAdverseExcursion;
    std::optional<double> maximumFavorableExcursion;
};

struct Phase18AStrategyVariantResult
{
    std::string strategyOutputIdentity;
    StrategyEvaluationResult evaluation;
    Phase18AStrategyMetrics metrics;
    std::array<Phase18AConfidenceBucketMetrics, 4> confidenceBuckets;
};

struct Phase18APairwiseDelta
{
    std::string candidateStrategyOutputIdentity;
    std::string referenceStrategyOutputIdentity;
    double aggregateDirectionalLogReturnDelta = 0.0;
    double averageDirectionalLogReturnDelta = 0.0;
    double winningRateDelta = 0.0;
    double losingRateDelta = 0.0;
    double stopHitRateDelta = 0.0;
    double terminalExitRateDelta = 0.0;
    double averageHoldingDurationSecondsDelta = 0.0;
    double maximumDrawdownDelta = 0.0;
};

struct ControlledProbabilityConditionedStopExperimentResult
{
    std::string experimentIdentityCanonical;
    std::string experimentIdentityHash;
    ProbabilityConditionedStopLossConfiguration configuration;
    std::vector<Phase18AStrategyVariantResult> variants;
    std::vector<Phase18APairwiseDelta> pairwiseDeltas;
    std::vector<Phase18AObservationEvidence> observationEvidence;
    std::string canonicalLines;
    std::string resultHash;
};

// Runs baseline, the exact same-base-distance fixed-stop control, and the
// predeclared Phase 18A v1 probability-conditioned strategy on one immutable
// observation population. It performs no persistence or scheduler mutation.
ControlledProbabilityConditionedStopExperimentResult
EvaluateControlledProbabilityConditionedStopExperiment(
    const AuthoritativeMarketPath& marketPath,
    ProbabilityConditionedStopLossConfiguration configuration = {});

StrategyEvaluationResult EvaluateStrategy(
    const TradingStrategy& strategy,
    const StrategyEvaluationInput& input);

} // namespace EA::StrategyEvaluation
