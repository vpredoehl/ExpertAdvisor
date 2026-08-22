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

std::string AuthoritativeObservationStatusText(
    AuthoritativeObservationStatus status)
{
    switch (status)
    {
        case AuthoritativeObservationStatus::available:
            return "available";
        case AuthoritativeObservationStatus::noObservation:
            return "no_profitability_observation";
        case AuthoritativeObservationStatus::ambiguousObservation:
            return "ambiguous_profitability_observation";
        case AuthoritativeObservationStatus::metricDefinitionMismatch:
            return "metric_definition_mismatch";
        case AuthoritativeObservationStatus::provenanceMismatch:
            return "provenance_mismatch";
    }
    throw std::logic_error("unknown_authoritative_profitability_status");
}

AuthoritativeObservationSelection SelectAuthoritativeObservation(
    pqxx::transaction_base& transaction,
    const AuthoritativeObservationSelector& selector)
{
    if ((selector.experimentId.has_value() && *selector.experimentId <= 0) ||
        selector.modelId <= 0 || selector.inferenceEvalResultId <= 0 ||
        (selector.checkpointEvalId.has_value() &&
         *selector.checkpointEvalId <= 0) ||
        selector.metricDefinitionCanonical.empty() ||
        !IsTaggedHash(selector.metricDefinitionHash))
    {
        throw std::invalid_argument(
            "invalid_authoritative_profitability_selector");
    }
    if ((selector.scope == Scope::finalInference) !=
        !selector.checkpointEvalId.has_value() ||
        (selector.scope == Scope::checkpointInference &&
         !selector.experimentId.has_value()))
    {
        throw std::invalid_argument(
            "authoritative_profitability_selector_scope_mismatch");
    }

    const pqxx::result rows = transaction.exec(
        "SELECT " + std::string{kSelectColumns} +
        " FROM inference_profitability_observation "
        "WHERE inference_eval_result_id=$1 "
        "ORDER BY profitability_observation_id;",
        pqxx::params{selector.inferenceEvalResultId});

    AuthoritativeObservationSelection selection;
    if (rows.empty())
        return selection;

    std::vector<Observation> exactProvenance;
    for (const pqxx::row& row : rows)
    {
        Observation observation = MapObservation(row);
        if (observation.provenance.experimentId == selector.experimentId &&
            observation.provenance.modelId == selector.modelId &&
            observation.provenance.inferenceEvalResultId ==
                selector.inferenceEvalResultId &&
            observation.provenance.scope == selector.scope &&
            observation.provenance.checkpointEvalId ==
                selector.checkpointEvalId)
        {
            exactProvenance.push_back(std::move(observation));
        }
    }
    if (exactProvenance.empty())
    {
        selection.status =
            AuthoritativeObservationStatus::provenanceMismatch;
        return selection;
    }

    std::vector<Observation> exactMetric;
    for (Observation& observation : exactProvenance)
    {
        if (observation.metricDefinitionCanonical ==
                selector.metricDefinitionCanonical &&
            observation.metricDefinitionHash ==
                selector.metricDefinitionHash)
        {
            exactMetric.push_back(std::move(observation));
        }
    }
    if (exactMetric.empty())
    {
        selection.status =
            AuthoritativeObservationStatus::metricDefinitionMismatch;
        return selection;
    }
    if (exactMetric.size() != 1)
    {
        selection.status =
            AuthoritativeObservationStatus::ambiguousObservation;
        return selection;
    }

    selection.status = AuthoritativeObservationStatus::available;
    selection.observation = std::move(exactMetric.front());
    return selection;
}

std::string ExactFinalInferenceResultStatusText(
    ExactFinalInferenceResultStatus status)
{
    switch (status)
    {
        case ExactFinalInferenceResultStatus::available:
            return "available";
        case ExactFinalInferenceResultStatus::noExactFinalInferenceResult:
            return "no_exact_final_inference_result";
        case ExactFinalInferenceResultStatus::ambiguousFinalInferenceResult:
            return "ambiguous_final_inference_result";
        case ExactFinalInferenceResultStatus::finalInferenceContextMismatch:
            return "final_inference_context_mismatch";
    }
    throw std::logic_error("unknown_exact_final_inference_result_status");
}

ExactFinalInferenceResultSelection ResolveExactFinalInferenceResult(
    pqxx::transaction_base& transaction,
    long long sourceExperimentId,
    long long finalModelId)
{
    if (sourceExperimentId <= 0 || finalModelId <= 0)
        throw std::invalid_argument("invalid_exact_final_inference_selector");

    const pqxx::row row = transaction.exec(
        "WITH cfg AS ("
        "  SELECT model_id,"
        "    round(max(value) FILTER (WHERE col_idx=1))::bigint AS prediction_horizon,"
        "    max(value) FILTER (WHERE col_idx=2) AS threshold_logret,"
        "    round(max(value) FILTER (WHERE col_idx=3))::bigint AS window_size,"
        "    round(max(value) FILTER (WHERE col_idx=4))::integer AS label_rule_id,"
        "    round(max(value) FILTER (WHERE col_idx=10))::bigint AS completed_epochs "
        "  FROM matrix WHERE param_name='train_config_meta' AND row_idx=0 "
        "  GROUP BY model_id "
        "  HAVING count(DISTINCT col_idx) FILTER "
        "    (WHERE col_idx BETWEEN 0 AND 13) >= 14"
        "), target AS ("
        "  SELECT model_id,"
        "    round(max(value) FILTER (WHERE col_idx=0))::integer AS target_type "
        "  FROM matrix WHERE param_name='target_meta' AND row_idx=0 "
        "  GROUP BY model_id"
        "), context AS ("
        "  SELECT e.experiment_id, e.last_model_id AS model_id, e.symbol,"
        "    cfg.prediction_horizon, cfg.threshold_logret, cfg.window_size,"
        "    cfg.label_rule_id, COALESCE(target.target_type,1) AS target_type,"
        "    e.infer_start::date::text AS inference_start,"
        "    e.infer_end::date::text AS inference_end, cfg.completed_epochs "
        "  FROM experiment e "
        "  JOIN model m ON m.model_id=e.last_model_id "
        "    AND m.experiment_id=e.experiment_id "
        "  JOIN cfg ON cfg.model_id=m.model_id "
        "  LEFT JOIN target ON target.model_id=m.model_id "
        "  WHERE e.experiment_id=$1 AND e.last_model_id=$2 "
        "    AND e.infer_start IS NOT NULL AND e.infer_end IS NOT NULL "
        "    AND cfg.prediction_horizon=e.prediction_horizon "
        "    AND abs(cfg.threshold_logret-e.c_next_threshold)<=1e-7"
        "), exact_result AS ("
        "  SELECT ier.id, ier.accept_model "
        "  FROM context c "
        "  JOIN inference_eval_result ier ON ier.model_id=c.model_id "
        "    AND ier.symbol=c.symbol "
        "    AND ier.prediction_horizon=c.prediction_horizon "
        "    AND ier.threshold_logret=c.threshold_logret "
        "    AND ier.window_size=c.window_size "
        "    AND ier.label_rule_id=c.label_rule_id "
        "    AND ier.target_type=c.target_type "
        "    AND ier.from_date=c.inference_start "
        "    AND ier.to_date=c.inference_end "
        "    AND ier.completed_epochs=c.completed_epochs "
        "    AND ier.status='completed' AND ier.inference_scope='final' "
        "    AND ier.checkpoint_eval_id IS NULL "
        "    AND ier.parent_experiment_id IS NULL"
        ") SELECT EXISTS(SELECT 1 FROM context) AS context_matches,"
        "  count(exact_result.id) AS exact_count,"
        "  min(exact_result.id) AS exact_id,"
        "  (array_agg(exact_result.accept_model ORDER BY exact_result.id) "
        "    FILTER (WHERE exact_result.id IS NOT NULL))[1] AS accept_model "
        "FROM exact_result;",
        pqxx::params{sourceExperimentId, finalModelId}).one_row();

    ExactFinalInferenceResultSelection selection;
    if (!row["context_matches"].as<bool>())
    {
        selection.status =
            ExactFinalInferenceResultStatus::finalInferenceContextMismatch;
        return selection;
    }

    const long long exactCount = row["exact_count"].as<long long>();
    if (exactCount == 0)
        return selection;
    if (exactCount != 1)
    {
        selection.status =
            ExactFinalInferenceResultStatus::ambiguousFinalInferenceResult;
        return selection;
    }

    selection.status = ExactFinalInferenceResultStatus::available;
    selection.inferenceEvalResultId = row["exact_id"].as<long long>();
    if (!row["accept_model"].is_null())
        selection.acceptModel = row["accept_model"].as<bool>();
    return selection;
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
