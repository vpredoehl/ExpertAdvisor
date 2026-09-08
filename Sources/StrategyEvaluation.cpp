#include "StrategyEvaluation.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

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
    for (const auto& observation : input.observations)
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
