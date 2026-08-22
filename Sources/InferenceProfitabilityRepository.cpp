#include "InferenceProfitabilityRepository.hpp"

#include <cmath>
#include <regex>
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

bool IsTaggedHash(const std::string& value)
{
    static const std::regex pattern{"^fnv1a64:[0-9a-f]{16}$"};
    return std::regex_match(value, pattern);
}

void ValidateRequest(const ObservationRequest& request)
{
    const auto& provenance = request.provenance;
    const auto& statistics = request.statistics;
    if ((provenance.experimentId.has_value() &&
         *provenance.experimentId <= 0) ||
        provenance.modelId <= 0 || provenance.inferenceEvalResultId <= 0)
        throw std::invalid_argument("invalid_inference_profitability_provenance_id");
    if (provenance.scope == Scope::finalInference &&
        provenance.checkpointEvalId.has_value())
        throw std::invalid_argument("final_profitability_has_checkpoint_identity");
    if (provenance.scope == Scope::checkpointInference &&
        (!provenance.checkpointEvalId.has_value() ||
         *provenance.checkpointEvalId <= 0 ||
         !provenance.experimentId.has_value()))
        throw std::invalid_argument("checkpoint_profitability_missing_checkpoint_identity");
    if (provenance.inferenceStart.empty() || provenance.inferenceEnd.empty())
        throw std::invalid_argument("missing_inference_profitability_range");
    if (statistics.actionableCount > statistics.predictionCount ||
        statistics.winningActionableCount > statistics.actionableCount ||
        statistics.losingActionableCount > statistics.actionableCount ||
        statistics.winningActionableCount + statistics.losingActionableCount >
            statistics.actionableCount)
        throw std::invalid_argument("invalid_inference_profitability_counts");
    if (statistics.grossPositiveTerminalHorizonLogReturnSum < 0.0 ||
        statistics.grossNegativeTerminalHorizonLogReturnSum > 0.0 ||
        !std::isfinite(
            statistics.grossPositiveTerminalHorizonLogReturnSum) ||
        !std::isfinite(
            statistics.grossNegativeTerminalHorizonLogReturnSum) ||
        !std::isfinite(statistics.aggregateTerminalHorizonLogReturnSum))
        throw std::invalid_argument("invalid_inference_profitability_returns");
    if (request.metricDefinitionCanonical.empty())
        throw std::invalid_argument("missing_profitability_metric_definition");
    if (!IsTaggedHash(request.sourceContentHash))
        throw std::invalid_argument("invalid_profitability_source_content_hash");
}

Observation MapObservation(const pqxx::row& row)
{
    Observation observation;
    observation.observationId =
        row["profitability_observation_id"].as<long long>();
    if (!row["experiment_id"].is_null())
        observation.provenance.experimentId =
            row["experiment_id"].as<long long>();
    observation.provenance.modelId = row["model_id"].as<long long>();
    observation.provenance.inferenceEvalResultId =
        row["inference_eval_result_id"].as<long long>();
    const std::string scope = row["inference_scope"].as<std::string>();
    if (scope == "final")
        observation.provenance.scope = Scope::finalInference;
    else if (scope == "checkpoint")
        observation.provenance.scope = Scope::checkpointInference;
    else
        throw std::runtime_error("unknown_inference_profitability_scope");
    if (!row["checkpoint_eval_id"].is_null())
        observation.provenance.checkpointEvalId =
            row["checkpoint_eval_id"].as<long long>();
    observation.provenance.inferenceStart =
        row["inference_start"].as<std::string>();
    observation.provenance.inferenceEnd =
        row["inference_end"].as<std::string>();
    observation.statistics.predictionCount =
        row["prediction_count"].as<std::uint64_t>();
    observation.statistics.actionableCount =
        row["actionable_count"].as<std::uint64_t>();
    observation.statistics.winningActionableCount =
        row["winning_actionable_count"].as<std::uint64_t>();
    observation.statistics.losingActionableCount =
        row["losing_actionable_count"].as<std::uint64_t>();
    observation.statistics.grossPositiveTerminalHorizonLogReturnSum =
        row["gross_positive_terminal_horizon_log_return_sum"].as<double>();
    observation.statistics.grossNegativeTerminalHorizonLogReturnSum =
        row["gross_negative_terminal_horizon_log_return_sum"].as<double>();
    observation.statistics.aggregateTerminalHorizonLogReturnSum =
        row["aggregate_terminal_horizon_log_return_sum"].as<double>();
    if (!row["average_terminal_horizon_log_return_per_actionable_prediction"].is_null())
        observation.averageTerminalHorizonLogReturnPerActionablePrediction =
            row["average_terminal_horizon_log_return_per_actionable_prediction"]
                .as<double>();
    observation.metricDefinitionCanonical =
        row["metric_definition_canonical"].as<std::string>();
    observation.metricDefinitionHash =
        row["metric_definition_hash"].as<std::string>();
    observation.sourceContentHash =
        row["source_content_hash"].as<std::string>();
    observation.observationIdentityCanonical =
        row["observation_identity_canonical"].as<std::string>();
    observation.observationIdentityHash =
        row["observation_identity_hash"].as<std::string>();
    observation.createdAt = row["created_at"].as<std::string>();
    return observation;
}

const char* kSelectColumns =
    "profitability_observation_id,experiment_id,model_id,"
    "inference_eval_result_id,inference_scope,checkpoint_eval_id,"
    "inference_start,inference_end,prediction_count,actionable_count,"
    "winning_actionable_count,losing_actionable_count,"
    "gross_positive_terminal_horizon_log_return_sum,"
    "gross_negative_terminal_horizon_log_return_sum,"
    "aggregate_terminal_horizon_log_return_sum,"
    "average_terminal_horizon_log_return_per_actionable_prediction,"
    "metric_definition_canonical,metric_definition_hash,source_content_hash,"
    "observation_identity_canonical,observation_identity_hash,"
    "created_at::text AS created_at";

} // namespace

std::string ScopeText(Scope scope)
{
    return scope == Scope::finalInference ? "final" : "checkpoint";
}

bool SchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('inference_profitability_observation') "
        "IS NOT NULL AS exists;")
        .one_row()["exists"].as<bool>();
}

std::string BuildObservationIdentityCanonical(
    const ObservationRequest& request)
{
    ValidateRequest(request);
    std::string canonical = "inference_profitability_observation_v1;";
    AppendField(canonical, "experiment_id",
                request.provenance.experimentId
                    ? std::to_string(*request.provenance.experimentId)
                    : "NULL");
    AppendField(canonical, "model_id",
                std::to_string(request.provenance.modelId));
    AppendField(canonical, "inference_eval_result_id",
                std::to_string(request.provenance.inferenceEvalResultId));
    AppendField(canonical, "inference_scope",
                ScopeText(request.provenance.scope));
    AppendField(canonical, "checkpoint_eval_id",
                request.provenance.checkpointEvalId
                    ? std::to_string(*request.provenance.checkpointEvalId)
                    : "NULL");
    AppendField(canonical, "inference_start",
                request.provenance.inferenceStart);
    AppendField(canonical, "inference_end",
                request.provenance.inferenceEnd);
    AppendField(canonical, "metric_definition_hash",
                DeterministicHash(request.metricDefinitionCanonical));
    AppendField(canonical, "metric_definition_canonical",
                request.metricDefinitionCanonical);
    AppendField(canonical, "source_content_hash", request.sourceContentHash);
    AppendField(canonical, "statistics",
                StatisticsCanonicalText(request.statistics));
    return canonical;
}

PersistResult PersistObservationIdempotently(
    pqxx::transaction_base& transaction,
    const ObservationRequest& request)
{
    const std::string identityCanonical =
        BuildObservationIdentityCanonical(request);
    const std::string identityHash = DeterministicHash(identityCanonical);
    const std::string metricHash =
        DeterministicHash(request.metricDefinitionCanonical);
    const auto average =
        request.statistics
            .AverageTerminalHorizonLogReturnPerActionablePrediction();

    pqxx::result rows = transaction.exec(
        "INSERT INTO inference_profitability_observation ("
        "experiment_id,model_id,inference_eval_result_id,inference_scope,"
        "checkpoint_eval_id,inference_start,inference_end,prediction_count,"
        "actionable_count,winning_actionable_count,losing_actionable_count,"
        "gross_positive_terminal_horizon_log_return_sum,"
        "gross_negative_terminal_horizon_log_return_sum,"
        "aggregate_terminal_horizon_log_return_sum,"
        "average_terminal_horizon_log_return_per_actionable_prediction,"
        "metric_definition_canonical,metric_definition_hash,source_content_hash,"
        "observation_identity_canonical,observation_identity_hash) VALUES ("
        "$1,$2,$3,$4,NULLIF($5::bigint,-1),$6,$7,$8,$9,$10,$11,"
        "$12,$13,$14,$15,$16,$17,$18,$19,$20) "
        "ON CONFLICT (observation_identity_canonical) DO NOTHING RETURNING " +
        std::string{kSelectColumns} + ";",
        pqxx::params{
            request.provenance.experimentId,
            request.provenance.modelId,
            request.provenance.inferenceEvalResultId,
            ScopeText(request.provenance.scope),
            request.provenance.checkpointEvalId.value_or(-1),
            request.provenance.inferenceStart,
            request.provenance.inferenceEnd,
            request.statistics.predictionCount,
            request.statistics.actionableCount,
            request.statistics.winningActionableCount,
            request.statistics.losingActionableCount,
            request.statistics.grossPositiveTerminalHorizonLogReturnSum,
            request.statistics.grossNegativeTerminalHorizonLogReturnSum,
            request.statistics.aggregateTerminalHorizonLogReturnSum,
            average,
            request.metricDefinitionCanonical,
            metricHash,
            request.sourceContentHash,
            identityCanonical,
            identityHash});

    PersistResult result;
    result.created = !rows.empty();
    if (rows.empty())
    {
        rows = transaction.exec(
            "SELECT " + std::string{kSelectColumns} +
            " FROM inference_profitability_observation "
            "WHERE observation_identity_canonical=$1;",
            pqxx::params{identityCanonical});
    }
    if (rows.size() != 1)
        throw std::runtime_error("profitability_observation_idempotency_failure");
    result.observation = MapObservation(rows.one_row());
    if (result.observation.observationIdentityHash != identityHash ||
        result.observation.metricDefinitionCanonical !=
            request.metricDefinitionCanonical ||
        result.observation.metricDefinitionHash != metricHash ||
        result.observation.sourceContentHash != request.sourceContentHash)
        throw std::runtime_error("profitability_observation_identity_mismatch");
    return result;
}

std::vector<Observation> LoadObservations(
    pqxx::transaction_base& transaction,
    const ObservationSelector& selector)
{
    if (selector.inferenceEvalResultId <= 0 ||
        selector.metricDefinitionHash.empty() ||
        selector.sourceContentHash.empty())
        throw std::invalid_argument("invalid_profitability_observation_selector");
    if ((selector.scope == Scope::finalInference) !=
        !selector.checkpointEvalId.has_value())
        throw std::invalid_argument("profitability_selector_scope_mismatch");

    const pqxx::result rows = transaction.exec(
        "SELECT " + std::string{kSelectColumns} +
        " FROM inference_profitability_observation "
        "WHERE inference_eval_result_id=$1 AND inference_scope=$2 "
        "AND checkpoint_eval_id IS NOT DISTINCT FROM NULLIF($3::bigint,-1) "
        "AND metric_definition_hash=$4 AND source_content_hash=$5 "
        "ORDER BY profitability_observation_id;",
        pqxx::params{
            selector.inferenceEvalResultId,
            ScopeText(selector.scope),
            selector.checkpointEvalId.value_or(-1),
            selector.metricDefinitionHash,
            selector.sourceContentHash});
    std::vector<Observation> observations;
    observations.reserve(rows.size());
    for (const auto& row : rows)
        observations.push_back(MapObservation(row));
    return observations;
}

std::optional<Observation> LoadObservationById(
    pqxx::transaction_base& transaction,
    long long observationId)
{
    if (observationId <= 0)
        throw std::invalid_argument("invalid_profitability_observation_id");
    const pqxx::result rows = transaction.exec(
        "SELECT " + std::string{kSelectColumns} +
        " FROM inference_profitability_observation "
        "WHERE profitability_observation_id=$1;",
        pqxx::params{observationId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error("duplicate_profitability_observation_id");
    return MapObservation(rows.one_row());
}

} // namespace EA::InferenceProfitability
