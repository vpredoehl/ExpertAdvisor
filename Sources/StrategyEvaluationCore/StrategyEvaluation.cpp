#include "StrategyEvaluation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace EA::StrategyEvaluation
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

std::string CanonicalFloat(float value)
{
    return "f32:" + Hex32(std::bit_cast<std::uint32_t>(value));
}

std::string CanonicalDouble(double value)
{
    return "f64:" + Hex64(std::bit_cast<std::uint64_t>(value));
}

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    return std::all_of(value.begin() + 8, value.end(), [](unsigned char byte) {
        return (byte >= '0' && byte <= '9') ||
            (byte >= 'a' && byte <= 'f');
    });
}

void ValidateStrategyIdentity(const StrategyIdentity& identity)
{
    if (identity.family.empty() || identity.version <= 0 ||
        identity.canonicalConfiguration.empty() ||
        !TaggedHash(identity.configurationHash) ||
        InferenceProfitability::DeterministicHash(
            identity.canonicalConfiguration) != identity.configurationHash ||
        identity.canonical.empty() || !TaggedHash(identity.hash) ||
        InferenceProfitability::DeterministicHash(identity.canonical) !=
            identity.hash)
    {
        throw std::invalid_argument("invalid_strategy_identity");
    }
}

void RequireSemanticText(const std::string& value, const char* error)
{
    if (value.empty()) throw std::invalid_argument(error);
}

void ValidateProbabilityVector(const StrategyEvaluationObservation& observation)
{
    if (!observation.probabilities)
        return;

    double sum = 0.0;
    for (const float probability :
         observation.probabilities->downNeutralUp)
    {
        if (!std::isfinite(probability) || probability < 0.0f ||
            probability > 1.0f)
        {
            throw std::invalid_argument(
                "invalid_market_path_prediction_probability");
        }
        sum += static_cast<double>(probability);
    }
    if (std::fabs(sum - 1.0) > 1.0e-4)
        throw std::invalid_argument(
            "invalid_market_path_prediction_probability_sum");
    if (PredictedClassForProbabilities(*observation.probabilities) !=
        observation.predictedClass)
    {
        throw std::invalid_argument(
            "market_path_predicted_class_probability_mismatch");
    }
}

void ValidateOhlc(const MarketPathPoint& point)
{
    if (!std::isfinite(point.open) || !std::isfinite(point.high) ||
        !std::isfinite(point.low) || !std::isfinite(point.close) ||
        point.open <= 0.0f || point.high <= 0.0f || point.low <= 0.0f ||
        point.close <= 0.0f)
    {
        throw std::invalid_argument("invalid_market_path_ohlc_price");
    }
    if (point.high < point.low || point.high < point.open ||
        point.high < point.close || point.low > point.open ||
        point.low > point.close)
    {
        throw std::invalid_argument("impossible_market_path_ohlc");
    }
}

void ValidateMarketPathProvenance(const MarketPathProvenance& provenance)
{
    if (provenance.schemaVersion != kMarketPathSchemaVersion)
        throw std::invalid_argument("unsupported_market_path_schema_version");
    if (provenance.canonicalizationVersion !=
        kMarketPathCanonicalizationVersion)
    {
        throw std::invalid_argument(
            "unsupported_market_path_canonicalization_version");
    }
    if (provenance.adapterVersion <= 0 || provenance.modelId <= 0 ||
        provenance.inferenceWindowSize == 0 ||
        provenance.predictionHorizon == 0 ||
        provenance.barIntervalSeconds == 0)
    {
        throw std::invalid_argument("invalid_market_path_provenance_number");
    }
    RequireSemanticText(provenance.adapterFamily,
                        "missing_market_path_adapter_family");
    RequireSemanticText(provenance.inferenceScientificIdentityCanonical,
                        "missing_market_path_inference_identity");
    RequireSemanticText(provenance.symbol, "missing_market_path_symbol");
    RequireSemanticText(provenance.evaluationStart,
                        "missing_market_path_evaluation_start");
    RequireSemanticText(provenance.evaluationEnd,
                        "missing_market_path_evaluation_end");
    RequireSemanticText(provenance.timestampSemantics,
                        "missing_market_path_timestamp_semantics");
    RequireSemanticText(provenance.ohlcIntervalSemantics,
                        "missing_market_path_ohlc_semantics");
    RequireSemanticText(provenance.priceDomain,
                        "missing_market_path_price_domain");
    RequireSemanticText(provenance.marketDataSource,
                        "missing_market_path_data_source");
    RequireSemanticText(provenance.marketDataSourceRelation,
                        "missing_market_path_data_relation");
    RequireSemanticText(provenance.pathOrdering,
                        "missing_market_path_ordering");
    RequireSemanticText(provenance.metricDefinitionCanonical,
                        "missing_market_path_metric_definition");
    if (!TaggedHash(provenance.inferenceScientificIdentityHash) ||
        InferenceProfitability::DeterministicHash(
            provenance.inferenceScientificIdentityCanonical) !=
            provenance.inferenceScientificIdentityHash)
    {
        throw std::invalid_argument("invalid_market_path_inference_identity");
    }
    if (!TaggedHash(provenance.metricDefinitionHash) ||
        InferenceProfitability::DeterministicHash(
            provenance.metricDefinitionCanonical) !=
            provenance.metricDefinitionHash)
    {
        throw std::invalid_argument("invalid_market_path_metric_identity");
    }
}

void ValidateObservation(
    const MarketPathProvenance& provenance,
    const StrategyEvaluationObservation& observation,
    std::uint64_t expectedOrdinal,
    const StrategyEvaluationObservation* previous)
{
    if (observation.observationOrdinal != expectedOrdinal)
        throw std::invalid_argument("invalid_market_path_observation_ordinal");
    (void)DirectionForPredictedClass(observation.predictedClass);
    ValidateProbabilityVector(observation);
    if (!std::isfinite(observation.decisionClose) ||
        !std::isfinite(observation.terminalClose) ||
        observation.decisionClose <= 0.0f ||
        observation.terminalClose <= 0.0f)
    {
        throw std::invalid_argument("invalid_market_path_decision_price");
    }
    if (!observation.decisionTimestampUnixSeconds ||
        !observation.terminalTimestampUnixSeconds)
    {
        throw std::invalid_argument("missing_market_path_timestamp");
    }

    const std::uint64_t windowTail = provenance.inferenceWindowSize - 1;
    if (observation.inferenceWindowStartRow >
        std::numeric_limits<std::uint64_t>::max() - windowTail ||
        observation.decisionRow !=
            observation.inferenceWindowStartRow + windowTail)
    {
        throw std::invalid_argument("invalid_market_path_decision_row_mapping");
    }
    if (observation.decisionRow >
        std::numeric_limits<std::uint64_t>::max() -
            provenance.predictionHorizon ||
        observation.terminalRow !=
            observation.decisionRow + provenance.predictionHorizon)
    {
        throw std::invalid_argument("invalid_market_path_terminal_row_mapping");
    }
    if (observation.marketPath.size() != provenance.predictionHorizon)
        throw std::invalid_argument("invalid_market_path_horizon_length");

    if (previous != nullptr &&
        (observation.decisionRow <= previous->decisionRow ||
         *observation.decisionTimestampUnixSeconds <=
             *previous->decisionTimestampUnixSeconds))
    {
        throw std::invalid_argument("unordered_market_path_observations");
    }

    std::int64_t previousTimestamp =
        *observation.decisionTimestampUnixSeconds;
    for (std::size_t pathOrdinal = 0;
         pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
    {
        const MarketPathPoint& point = observation.marketPath[pathOrdinal];
        if (!point.timestampUnixSeconds)
            throw std::invalid_argument("missing_market_path_timestamp");
        if (*point.timestampUnixSeconds <= previousTimestamp)
            throw std::invalid_argument("unordered_market_path_points");
        previousTimestamp = *point.timestampUnixSeconds;
        if (point.sourceRow != observation.decisionRow + pathOrdinal + 1)
            throw std::invalid_argument("invalid_market_path_source_row");
        ValidateOhlc(point);
    }

    const MarketPathPoint& terminal = observation.marketPath.back();
    if (terminal.sourceRow != observation.terminalRow ||
        terminal.timestampUnixSeconds !=
            observation.terminalTimestampUnixSeconds ||
        std::bit_cast<std::uint32_t>(terminal.close) !=
            std::bit_cast<std::uint32_t>(observation.terminalClose))
    {
        throw std::invalid_argument("market_path_terminal_mismatch");
    }
}

std::string CanonicalMarketPath(
    const MarketPathProvenance& provenance,
    const std::vector<StrategyEvaluationObservation>& observations)
{
    std::string canonical = "authoritative_market_path_v1;";
    AppendField(canonical, "schema_version",
                std::to_string(provenance.schemaVersion));
    AppendField(canonical, "canonicalization_version",
                std::to_string(provenance.canonicalizationVersion));
    AppendField(canonical, "adapter_family", provenance.adapterFamily);
    AppendField(canonical, "adapter_version",
                std::to_string(provenance.adapterVersion));
    AppendField(canonical, "model_id", std::to_string(provenance.modelId));
    AppendField(canonical, "inference_scientific_identity_hash",
                provenance.inferenceScientificIdentityHash);
    AppendField(canonical, "inference_scientific_identity_canonical",
                provenance.inferenceScientificIdentityCanonical);
    AppendField(canonical, "symbol", provenance.symbol);
    AppendField(canonical, "inference_window_size",
                std::to_string(provenance.inferenceWindowSize));
    AppendField(canonical, "prediction_horizon",
                std::to_string(provenance.predictionHorizon));
    AppendField(canonical, "evaluation_start", provenance.evaluationStart);
    AppendField(canonical, "evaluation_end", provenance.evaluationEnd);
    AppendField(canonical, "bar_interval_seconds",
                std::to_string(provenance.barIntervalSeconds));
    AppendField(canonical, "timestamp_semantics",
                provenance.timestampSemantics);
    AppendField(canonical, "ohlc_interval_semantics",
                provenance.ohlcIntervalSemantics);
    AppendField(canonical, "price_domain", provenance.priceDomain);
    AppendField(canonical, "market_data_source",
                provenance.marketDataSource);
    AppendField(canonical, "market_data_source_relation",
                provenance.marketDataSourceRelation);
    AppendField(canonical, "path_ordering", provenance.pathOrdering);
    AppendField(canonical, "metric_definition_hash",
                provenance.metricDefinitionHash);
    AppendField(canonical, "metric_definition_canonical",
                provenance.metricDefinitionCanonical);
    AppendField(canonical, "observation_count",
                std::to_string(observations.size()));

    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        const auto& observation = observations[index];
        const std::string prefix = "observation[" + std::to_string(index) + "].";
        AppendField(canonical, prefix + "ordinal",
                    std::to_string(observation.observationOrdinal));
        AppendField(canonical, prefix + "window_start_row",
                    std::to_string(observation.inferenceWindowStartRow));
        AppendField(canonical, prefix + "decision_row",
                    std::to_string(observation.decisionRow));
        AppendField(canonical, prefix + "terminal_row",
                    std::to_string(observation.terminalRow));
        AppendField(canonical, prefix + "predicted_class",
                    std::to_string(observation.predictedClass));
        AppendField(canonical, prefix + "decision_close",
                    CanonicalFloat(observation.decisionClose));
        AppendField(canonical, prefix + "terminal_close",
                    CanonicalFloat(observation.terminalClose));
        AppendField(canonical, prefix + "decision_timestamp_unix_seconds",
                    std::to_string(*observation.decisionTimestampUnixSeconds));
        AppendField(canonical, prefix + "terminal_timestamp_unix_seconds",
                    std::to_string(*observation.terminalTimestampUnixSeconds));
        AppendField(canonical, prefix + "probabilities_available",
                    observation.probabilities ? "1" : "0");
        if (observation.probabilities)
        {
            AppendField(canonical, prefix + "probability_down",
                        CanonicalFloat(
                            observation.probabilities->downNeutralUp[0]));
            AppendField(canonical, prefix + "probability_neutral",
                        CanonicalFloat(
                            observation.probabilities->downNeutralUp[1]));
            AppendField(canonical, prefix + "probability_up",
                        CanonicalFloat(
                            observation.probabilities->downNeutralUp[2]));
        }
        AppendField(canonical, prefix + "path_point_count",
                    std::to_string(observation.marketPath.size()));
        for (std::size_t pathOrdinal = 0;
             pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
        {
            const auto& point = observation.marketPath[pathOrdinal];
            const std::string pathPrefix = prefix + "path[" +
                std::to_string(pathOrdinal) + "].";
            AppendField(canonical, pathPrefix + "source_row",
                        std::to_string(point.sourceRow));
            AppendField(canonical, pathPrefix + "timestamp_unix_seconds",
                        std::to_string(*point.timestampUnixSeconds));
            AppendField(canonical, pathPrefix + "open",
                        CanonicalFloat(point.open));
            AppendField(canonical, pathPrefix + "high",
                        CanonicalFloat(point.high));
            AppendField(canonical, pathPrefix + "low",
                        CanonicalFloat(point.low));
            AppendField(canonical, pathPrefix + "close",
                        CanonicalFloat(point.close));
        }
    }
    return canonical;
}

const char* DirectionText(PositionDirection direction)
{
    switch (direction)
    {
        case PositionDirection::flat: return "flat";
        case PositionDirection::shortPosition: return "short";
        case PositionDirection::longPosition: return "long";
    }
    return "invalid";
}

std::string OptionalTimestampText(
    const std::optional<std::int64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalDoubleText(const std::optional<double>& value)
{
    return value ? CanonicalDouble(*value) : "NULL";
}

std::string OptionalUnsignedText(const std::optional<std::uint64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

void ObserveDirectionalResult(InferenceProfitability::Statistics& statistics,
                              int predictedClass,
                              double directionalLogReturn)
{
    ++statistics.predictionCount;
    if (predictedClass == InferenceProfitability::kNeutralClass)
        return;

    ++statistics.actionableCount;
    statistics.aggregateTerminalHorizonLogReturnSum += directionalLogReturn;
    if (predictedClass == InferenceProfitability::kUpClass)
    {
        ++statistics.upActionableCount;
        statistics.upTerminalHorizonLogReturnSum += directionalLogReturn;
    }
    else
    {
        ++statistics.downActionableCount;
        statistics.downTerminalHorizonLogReturnSum += directionalLogReturn;
    }

    if (directionalLogReturn > 0.0)
    {
        ++statistics.winningActionableCount;
        statistics.grossPositiveTerminalHorizonLogReturnSum +=
            directionalLogReturn;
        if (predictedClass == InferenceProfitability::kUpClass)
        {
            ++statistics.upWinningActionableCount;
            statistics.upGrossPositiveTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
        else
        {
            ++statistics.downWinningActionableCount;
            statistics.downGrossPositiveTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
    }
    else if (directionalLogReturn < 0.0)
    {
        ++statistics.losingActionableCount;
        statistics.grossNegativeTerminalHorizonLogReturnSum +=
            directionalLogReturn;
        if (predictedClass == InferenceProfitability::kUpClass)
        {
            ++statistics.upLosingActionableCount;
            statistics.upGrossNegativeTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
        else
        {
            ++statistics.downLosingActionableCount;
            statistics.downGrossNegativeTerminalHorizonLogReturnSum +=
                directionalLogReturn;
        }
    }
}

std::string CanonicalExecutionResults(
    const StrategyIdentity& strategyIdentity,
    const StrategyEvaluationIdentity& evaluationIdentity,
    const AuthoritativeMarketPath& marketPath,
    const std::string& metricDefinitionHash,
    const std::vector<StrategyExecutionResult>& results)
{
    std::string canonical = "strategy_execution_result_v1;";
    AppendField(canonical, "strategy_identity_hash", strategyIdentity.hash);
    AppendField(canonical, "evaluation_identity_hash",
                evaluationIdentity.hash);
    AppendField(canonical, "market_path_hash", marketPath.Hash());
    AppendField(canonical, "metric_definition_hash", metricDefinitionHash);
    AppendField(canonical, "result_count", std::to_string(results.size()));
    for (std::size_t index = 0; index < results.size(); ++index)
    {
        const auto& result = results[index];
        const std::string prefix = "result[" + std::to_string(index) + "].";
        AppendField(canonical, prefix + "observation_ordinal",
                    std::to_string(result.observationOrdinal));
        AppendField(canonical, prefix + "direction",
                    DirectionText(result.direction));
        AppendField(canonical, prefix + "reason",
                    StrategyExitReasonText(result.reason));
        AppendField(canonical, prefix + "entry_timestamp_unix_seconds",
                    OptionalTimestampText(
                        result.entryTimestampUnixSeconds));
        AppendField(canonical, prefix + "exit_timestamp_unix_seconds",
                    OptionalTimestampText(result.exitTimestampUnixSeconds));
        AppendField(canonical, prefix + "entry_price",
                    OptionalDoubleText(result.entryPrice));
        AppendField(canonical, prefix + "initial_stop_price",
                    OptionalDoubleText(result.initialStopPrice));
        AppendField(canonical, prefix + "exit_price",
                    OptionalDoubleText(result.exitPrice));
        AppendField(canonical, prefix + "directional_log_return",
                    OptionalDoubleText(result.directionalLogReturn));
        AppendField(canonical, prefix + "triggering_path_point_ordinal",
                    OptionalUnsignedText(
                        result.triggeringPathPointOrdinal));
        AppendField(canonical, prefix + "triggering_source_row",
                    OptionalUnsignedText(result.triggeringSourceRow));
    }
    return canonical;
}

} // namespace

std::string CanonicalConfiguration(
    const StrategyConfiguration& configuration)
{
    if (configuration.schemaVersion != kStrategyConfigurationSchemaVersion)
        throw std::invalid_argument(
            "unsupported_strategy_configuration_schema_version");

    std::vector<ConfigurationEntry> entries = configuration.entries;
    std::sort(entries.begin(), entries.end(), [](const auto& left,
                                                 const auto& right) {
        return left.key < right.key;
    });

    std::string canonical = "strategy_configuration_v1;";
    AppendField(canonical, "schema_version",
                std::to_string(configuration.schemaVersion));
    AppendField(canonical, "entry_count", std::to_string(entries.size()));
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        if (entries[index].key.empty())
            throw std::invalid_argument("empty_strategy_configuration_key");
        if (index > 0 && entries[index - 1].key == entries[index].key)
            throw std::invalid_argument("duplicate_strategy_configuration_key");
        AppendField(canonical,
                    "entry[" + std::to_string(index) + "].key",
                    entries[index].key);
        AppendField(canonical,
                    "entry[" + std::to_string(index) + "].value",
                    entries[index].value);
    }
    return canonical;
}

std::string ConfigurationHash(
    const StrategyConfiguration& configuration)
{
    return InferenceProfitability::DeterministicHash(
        CanonicalConfiguration(configuration));
}

StrategyIdentity BuildStrategyIdentity(
    const std::string& family,
    int version,
    const StrategyConfiguration& configuration)
{
    if (family.empty())
        throw std::invalid_argument("empty_strategy_family");
    if (version <= 0)
        throw std::invalid_argument("invalid_strategy_version");

    StrategyIdentity identity;
    identity.family = family;
    identity.version = version;
    identity.canonicalConfiguration = CanonicalConfiguration(configuration);
    identity.configurationHash =
        InferenceProfitability::DeterministicHash(
            identity.canonicalConfiguration);
    identity.canonical = "strategy_identity_v1;";
    AppendField(identity.canonical, "family", identity.family);
    AppendField(identity.canonical, "version", std::to_string(identity.version));
    AppendField(identity.canonical, "configuration_hash",
                identity.configurationHash);
    AppendField(identity.canonical, "canonical_configuration",
                identity.canonicalConfiguration);
    identity.hash =
        InferenceProfitability::DeterministicHash(identity.canonical);
    return identity;
}

PositionDirection DirectionForPredictedClass(int predictedClass)
{
    switch (predictedClass)
    {
        case InferenceProfitability::kDownClass:
            return PositionDirection::shortPosition;
        case InferenceProfitability::kNeutralClass:
            return PositionDirection::flat;
        case InferenceProfitability::kUpClass:
            return PositionDirection::longPosition;
    }
    throw std::invalid_argument("invalid_strategy_predicted_class");
}

int PredictedClassForProbabilities(
    const PredictionProbabilities& probabilities)
{
    const auto& values = probabilities.downNeutralUp;
    return (values[0] > values[1] && values[0] > values[2])
        ? InferenceProfitability::kDownClass
        : ((values[2] > values[1] && values[2] > values[0])
               ? InferenceProfitability::kUpClass
               : InferenceProfitability::kNeutralClass);
}

AuthoritativeMarketPath::AuthoritativeMarketPath(
    MarketPathProvenance provenance,
    std::vector<StrategyEvaluationObservation> observations,
    std::string canonical,
    std::string hash)
    : provenance_(std::move(provenance)),
      observations_(std::move(observations)),
      canonical_(std::move(canonical)),
      hash_(std::move(hash))
{
}

const MarketPathProvenance& AuthoritativeMarketPath::Provenance() const noexcept
{
    return provenance_;
}

const std::vector<StrategyEvaluationObservation>&
AuthoritativeMarketPath::Observations() const noexcept
{
    return observations_;
}

const std::string& AuthoritativeMarketPath::Canonical() const noexcept
{
    return canonical_;
}

const std::string& AuthoritativeMarketPath::Hash() const noexcept
{
    return hash_;
}

AuthoritativeMarketPath BuildAuthoritativeMarketPath(
    MarketPathProvenance provenance,
    std::vector<StrategyEvaluationObservation> observations)
{
    ValidateMarketPathProvenance(provenance);
    const StrategyEvaluationObservation* previous = nullptr;
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        ValidateObservation(provenance, observations[index], index, previous);
        previous = &observations[index];
    }
    std::string canonical = CanonicalMarketPath(provenance, observations);
    std::string hash =
        InferenceProfitability::DeterministicHash(canonical);
    return AuthoritativeMarketPath(
        std::move(provenance), std::move(observations),
        std::move(canonical), std::move(hash));
}

StrategyEvaluationInput::StrategyEvaluationInput(
    std::vector<StrategyEvaluationObservation> observations)
    : observations_(std::move(observations))
{
}

StrategyEvaluationInput::StrategyEvaluationInput(
    AuthoritativeMarketPath marketPath)
    : marketPath_(
          std::make_shared<AuthoritativeMarketPath>(std::move(marketPath)))
{
}

const std::vector<StrategyEvaluationObservation>&
StrategyEvaluationInput::Observations() const noexcept
{
    return marketPath_ ? marketPath_->Observations() : observations_;
}

const AuthoritativeMarketPath* StrategyEvaluationInput::MarketPath() const
    noexcept
{
    return marketPath_.get();
}

StrategyEvaluationProvenance BuildStrategyEvaluationProvenance(
    const AuthoritativeMarketPath& marketPath)
{
    const auto& path = marketPath.Provenance();
    StrategyEvaluationProvenance provenance;
    provenance.inferenceScientificIdentityCanonical =
        path.inferenceScientificIdentityCanonical;
    provenance.inferenceScientificIdentityHash =
        path.inferenceScientificIdentityHash;
    provenance.symbol = path.symbol;
    provenance.predictionHorizon = path.predictionHorizon;
    provenance.inferenceStart = path.evaluationStart;
    provenance.inferenceEnd = path.evaluationEnd;
    provenance.metricDefinitionHash = path.metricDefinitionHash;
    provenance.marketDataProvenanceHash = marketPath.Hash();
    return provenance;
}

StrategyEvaluationIdentity BuildStrategyEvaluationIdentity(
    const StrategyIdentity& strategyIdentity,
    const StrategyEvaluationProvenance& provenance)
{
    ValidateStrategyIdentity(strategyIdentity);
    if (provenance.inferenceScientificIdentityCanonical.empty() ||
        !TaggedHash(provenance.inferenceScientificIdentityHash) ||
        InferenceProfitability::DeterministicHash(
            provenance.inferenceScientificIdentityCanonical) !=
            provenance.inferenceScientificIdentityHash ||
        provenance.symbol.empty() || provenance.predictionHorizon == 0 ||
        provenance.inferenceStart.empty() || provenance.inferenceEnd.empty() ||
        !TaggedHash(provenance.metricDefinitionHash) ||
        !TaggedHash(provenance.marketDataProvenanceHash))
    {
        throw std::invalid_argument("invalid_strategy_evaluation_provenance");
    }

    StrategyEvaluationIdentity identity;
    identity.canonical = "strategy_evaluation_identity_v1;";
    AppendField(identity.canonical, "strategy_identity_hash",
                strategyIdentity.hash);
    AppendField(identity.canonical, "strategy_identity_canonical",
                strategyIdentity.canonical);
    AppendField(identity.canonical, "inference_scientific_identity_hash",
                provenance.inferenceScientificIdentityHash);
    AppendField(identity.canonical, "inference_scientific_identity_canonical",
                provenance.inferenceScientificIdentityCanonical);
    AppendField(identity.canonical, "symbol", provenance.symbol);
    AppendField(identity.canonical, "prediction_horizon",
                std::to_string(provenance.predictionHorizon));
    AppendField(identity.canonical, "inference_start",
                provenance.inferenceStart);
    AppendField(identity.canonical, "inference_end",
                provenance.inferenceEnd);
    AppendField(identity.canonical, "metric_definition_hash",
                provenance.metricDefinitionHash);
    AppendField(identity.canonical, "market_data_provenance_hash",
                provenance.marketDataProvenanceHash);
    identity.hash =
        InferenceProfitability::DeterministicHash(identity.canonical);
    return identity;
}

const char* StrategyExitReasonText(StrategyExitReason reason) noexcept
{
    switch (reason)
    {
        case StrategyExitReason::noAction: return "no_action";
        case StrategyExitReason::terminalExit: return "terminal_exit";
        case StrategyExitReason::fixedStopExit: return "fixed_stop_exit";
    }
    return "invalid";
}

BaselineTerminalStrategy::BaselineTerminalStrategy(
    StrategyConfiguration configuration)
{
    if (configuration.schemaVersion != kStrategyConfigurationSchemaVersion ||
        !configuration.entries.empty())
    {
        throw std::invalid_argument(
            "baseline_terminal_strategy_unsupported_configuration");
    }
    identity_ = BuildStrategyIdentity(
        kBaselineTerminalStrategyFamily,
        kBaselineTerminalStrategyVersion,
        configuration);
}

const StrategyIdentity& BaselineTerminalStrategy::Identity() const noexcept
{
    return identity_;
}

StrategyEvaluationResult BaselineTerminalStrategy::Evaluate(
    const StrategyEvaluationInput& input) const
{
    InferenceProfitability::Accumulator profitability;
    for (const auto& observation : input.Observations())
    {
        profitability.Observe(observation.predictedClass,
                              observation.decisionClose,
                              observation.terminalClose);
    }

    StrategyEvaluationResult result;
    result.strategyIdentity = identity_;
    result.statistics = profitability.statistics();
    result.sourceContentHash = profitability.SourceContentHash();
    result.metricDefinitionCanonical =
        InferenceProfitability::kMetricDefinitionCanonical;
    result.metricDefinitionHash =
        InferenceProfitability::MetricDefinitionHash();
    return result;
}

FixedStopLossStrategy::FixedStopLossStrategy(
    FixedStopLossConfiguration configuration)
    : configuration_(configuration)
{
    if (configuration_.schemaVersion !=
        kFixedStopLossConfigurationSchemaVersion)
    {
        throw std::invalid_argument(
            "unsupported_fixed_stop_configuration_schema_version");
    }
    if (!std::isfinite(configuration_.logarithmicDistance) ||
        configuration_.logarithmicDistance <= 0.0 ||
        !std::isfinite(std::exp(configuration_.logarithmicDistance)) ||
        std::exp(-configuration_.logarithmicDistance) <= 0.0)
    {
        throw std::invalid_argument("invalid_fixed_stop_logarithmic_distance");
    }

    StrategyConfiguration identityConfiguration;
    identityConfiguration.entries = {
        {"configuration_schema_version",
         std::to_string(configuration_.schemaVersion)},
        {"execution_rule", kFixedStopExecutionRule},
        {"execution_rule_version",
         std::to_string(kFixedStopExecutionRuleVersion)},
        {"logarithmic_distance_binary64",
         CanonicalDouble(configuration_.logarithmicDistance)}};
    identity_ = BuildStrategyIdentity(
        kFixedStopLossStrategyFamily,
        kFixedStopLossStrategyVersion,
        identityConfiguration);
}

const StrategyIdentity& FixedStopLossStrategy::Identity() const noexcept
{
    return identity_;
}

double FixedStopLossStrategy::InitialStopPrice(int predictedClass,
                                               float entryPrice) const
{
    const PositionDirection direction =
        DirectionForPredictedClass(predictedClass);
    if (direction == PositionDirection::flat)
        throw std::invalid_argument("fixed_stop_no_action_has_no_stop");
    if (!std::isfinite(entryPrice) || entryPrice <= 0.0f)
        throw std::invalid_argument("invalid_fixed_stop_entry_price");

    const double exponent = direction == PositionDirection::longPosition
        ? -configuration_.logarithmicDistance
        : configuration_.logarithmicDistance;
    const double stop = static_cast<double>(entryPrice) * std::exp(exponent);
    if (!std::isfinite(stop) || stop <= 0.0)
        throw std::invalid_argument("invalid_fixed_stop_price");
    return stop;
}

std::string FixedStopMetricDefinitionHash()
{
    return InferenceProfitability::DeterministicHash(
        kFixedStopMetricDefinitionCanonical);
}

StrategyEvaluationResult FixedStopLossStrategy::Evaluate(
    const StrategyEvaluationInput& input) const
{
    const AuthoritativeMarketPath* marketPath = input.MarketPath();
    if (marketPath == nullptr)
        throw std::invalid_argument("fixed_stop_requires_authoritative_market_path");
    if (marketPath->Provenance().metricDefinitionCanonical !=
            kFixedStopMetricDefinitionCanonical ||
        marketPath->Provenance().metricDefinitionHash !=
            FixedStopMetricDefinitionHash())
    {
        throw std::invalid_argument("fixed_stop_metric_identity_mismatch");
    }

    StrategyEvaluationResult result;
    result.strategyIdentity = identity_;
    result.sourceContentHash = marketPath->Hash();
    result.metricDefinitionCanonical = kFixedStopMetricDefinitionCanonical;
    result.metricDefinitionHash = FixedStopMetricDefinitionHash();
    result.evaluationIdentity = BuildStrategyEvaluationIdentity(
        identity_, BuildStrategyEvaluationProvenance(*marketPath));
    result.executionResults.reserve(input.Observations().size());

    for (const auto& observation : input.Observations())
    {
        StrategyExecutionResult execution;
        execution.observationOrdinal = observation.observationOrdinal;
        execution.direction =
            DirectionForPredictedClass(observation.predictedClass);
        if (execution.direction == PositionDirection::flat)
        {
            ObserveDirectionalResult(
                result.statistics, observation.predictedClass, 0.0);
            result.executionResults.push_back(std::move(execution));
            continue;
        }

        execution.entryTimestampUnixSeconds =
            observation.decisionTimestampUnixSeconds;
        execution.entryPrice = static_cast<double>(observation.decisionClose);
        execution.initialStopPrice = InitialStopPrice(
            observation.predictedClass, observation.decisionClose);

        for (std::size_t pathOrdinal = 0;
             pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
        {
            const auto& point = observation.marketPath[pathOrdinal];
            const bool gapBeyondStop =
                execution.direction == PositionDirection::longPosition
                ? static_cast<double>(point.open) <=
                    *execution.initialStopPrice
                : static_cast<double>(point.open) >=
                    *execution.initialStopPrice;
            const bool touchedStop =
                execution.direction == PositionDirection::longPosition
                ? static_cast<double>(point.low) <=
                    *execution.initialStopPrice
                : static_cast<double>(point.high) >=
                    *execution.initialStopPrice;
            if (!touchedStop)
                continue;

            execution.reason = StrategyExitReason::fixedStopExit;
            execution.exitTimestampUnixSeconds = point.timestampUnixSeconds;
            execution.exitPrice = gapBeyondStop
                ? static_cast<double>(point.open)
                : *execution.initialStopPrice;
            execution.triggeringPathPointOrdinal = pathOrdinal;
            execution.triggeringSourceRow = point.sourceRow;
            break;
        }

        if (!execution.exitPrice)
        {
            execution.reason = StrategyExitReason::terminalExit;
            execution.exitTimestampUnixSeconds =
                observation.terminalTimestampUnixSeconds;
            execution.exitPrice =
                static_cast<double>(observation.terminalClose);
        }

        const double terminalLogReturn =
            std::log(*execution.exitPrice / *execution.entryPrice);
        execution.directionalLogReturn =
            execution.direction == PositionDirection::longPosition
            ? terminalLogReturn
            : -terminalLogReturn;
        if (!std::isfinite(*execution.directionalLogReturn))
            throw std::invalid_argument("nonfinite_fixed_stop_result");
        ObserveDirectionalResult(result.statistics,
                                 observation.predictedClass,
                                 *execution.directionalLogReturn);
        result.executionResults.push_back(std::move(execution));
    }

    result.resultCanonical = CanonicalExecutionResults(
        result.strategyIdentity, *result.evaluationIdentity, *marketPath,
        result.metricDefinitionHash, result.executionResults);
    result.resultHash = InferenceProfitability::DeterministicHash(
        result.resultCanonical);
    return result;
}

StrategyEvaluationResult EvaluateStrategy(
    const TradingStrategy& strategy,
    const StrategyEvaluationInput& input)
{
    ValidateStrategyIdentity(strategy.Identity());
    StrategyEvaluationResult result = strategy.Evaluate(input);
    if (result.strategyIdentity != strategy.Identity())
        throw std::runtime_error("strategy_evaluation_identity_mismatch");
    return result;
}

} // namespace EA::StrategyEvaluation
