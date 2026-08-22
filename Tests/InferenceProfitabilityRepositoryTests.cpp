#include <cassert>
#include <cstdlib>
#include <string>

#include <pqxx/pqxx>

#include "../Sources/InferenceProfitabilityRepository.hpp"

using namespace EA::InferenceProfitability;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

ObservationRequest Request(long long inferenceResultId,
                           Scope scope,
                           const std::optional<long long>& checkpointEvalId,
                           const Accumulator& accumulator)
{
    ObservationRequest request;
    request.provenance.experimentId = 1;
    request.provenance.modelId = 10;
    request.provenance.inferenceEvalResultId = inferenceResultId;
    request.provenance.scope = scope;
    request.provenance.checkpointEvalId = checkpointEvalId;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2026-01-01";
    request.statistics = accumulator.statistics();
    request.sourceContentHash = accumulator.SourceContentHash();
    return request;
}

} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    const std::string schema =
        EnvironmentOr("LSTM_PROFITABILITY_TEST_SCHEMA", "");
    assert(!schema.empty());

    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ", public;");
    assert(SchemaExists(transaction));

    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,recommendation_score_guard,campaign_guard,"
        "continuation_policy_guard,checkpoint_policy_guard) "
        "VALUES(1,0.75,'unchanged','unchanged','unchanged');"
        "INSERT INTO model(model_id,experiment_id) VALUES(10,1),(11,NULL);"
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES(200,1,10);"
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,from_date,to_date) VALUES"
        "(99,10,'completed','final',NULL,NULL,'2024-01-01','2024-12-31'),"
        "(100,10,'completed','final',NULL,NULL,'2025-01-01','2026-01-01'),"
        "(101,10,'completed','checkpoint',200,1,'2025-01-01','2026-01-01'),"
        "(102,11,'completed','final',NULL,NULL,'2025-01-01','2026-01-01');");

    Accumulator calculation;
    calculation.Observe(kNeutralClass, 100.0f, 120.0f);
    calculation.Observe(kUpClass, 100.0f, 110.0f);
    calculation.Observe(kDownClass, 100.0f, 110.0f);

    const ObservationRequest finalRequest =
        Request(100, Scope::finalInference, std::nullopt, calculation);
    const PersistResult finalFirst =
        PersistObservationIdempotently(transaction, finalRequest);
    const PersistResult finalSecond =
        PersistObservationIdempotently(transaction, finalRequest);
    assert(finalFirst.created);
    assert(!finalSecond.created);
    assert(finalFirst.observation.observationId ==
           finalSecond.observation.observationId);
    assert(finalFirst.observation.provenance.scope == Scope::finalInference);
    assert(!finalFirst.observation.provenance.checkpointEvalId);
    assert(finalFirst.observation.statistics.predictionCount == 3);
    assert(finalFirst.observation.statistics.actionableCount == 2);

    ObservationRequest legacyFinalRequest = finalRequest;
    legacyFinalRequest.provenance.experimentId.reset();
    legacyFinalRequest.provenance.modelId = 11;
    legacyFinalRequest.provenance.inferenceEvalResultId = 102;
    const PersistResult legacyFinal =
        PersistObservationIdempotently(transaction, legacyFinalRequest);
    assert(legacyFinal.created);
    assert(!legacyFinal.observation.provenance.experimentId);

    ObservationSelector finalSelector;
    finalSelector.inferenceEvalResultId = 100;
    finalSelector.scope = Scope::finalInference;
    finalSelector.metricDefinitionHash = MetricDefinitionHash();
    finalSelector.sourceContentHash = calculation.SourceContentHash();
    const auto finalLoaded = LoadObservations(transaction, finalSelector);
    assert(finalLoaded.size() == 1);
    const auto finalById = LoadObservationById(
        transaction, finalFirst.observation.observationId);
    assert(finalById);
    assert(finalById->observationIdentityCanonical ==
           finalFirst.observation.observationIdentityCanonical);

    const ObservationRequest checkpointRequest =
        Request(101, Scope::checkpointInference, 200, calculation);
    const PersistResult checkpoint =
        PersistObservationIdempotently(transaction, checkpointRequest);
    assert(checkpoint.created);
    assert(checkpoint.observation.provenance.scope ==
           Scope::checkpointInference);
    assert(checkpoint.observation.provenance.checkpointEvalId == 200);
    assert(checkpoint.observation.observationId !=
           finalFirst.observation.observationId);

    bool finalCheckpointRejected = false;
    try
    {
        pqxx::subtransaction invalid{transaction, "final_checkpoint_scope"};
        invalid.exec(
            "INSERT INTO inference_profitability_observation ("
            "experiment_id,model_id,inference_eval_result_id,inference_scope,"
            "checkpoint_eval_id,inference_start,inference_end,prediction_count,"
            "actionable_count,winning_actionable_count,losing_actionable_count,"
            "gross_positive_terminal_horizon_log_return_sum,"
            "gross_negative_terminal_horizon_log_return_sum,"
            "aggregate_terminal_horizon_log_return_sum,"
            "average_terminal_horizon_log_return_per_actionable_prediction,"
            "metric_definition_canonical,metric_definition_hash,source_content_hash,"
            "observation_identity_canonical,observation_identity_hash) "
            "SELECT experiment_id,model_id,inference_eval_result_id,'final',200,"
            "inference_start,inference_end,prediction_count,actionable_count,"
            "winning_actionable_count,losing_actionable_count,"
            "gross_positive_terminal_horizon_log_return_sum,"
            "gross_negative_terminal_horizon_log_return_sum,"
            "aggregate_terminal_horizon_log_return_sum,"
            "average_terminal_horizon_log_return_per_actionable_prediction,"
            "metric_definition_canonical,metric_definition_hash,source_content_hash,"
            "'invalid-final-checkpoint','fnv1a64:0000000000000000' "
            "FROM inference_profitability_observation "
            "WHERE profitability_observation_id=$1;",
            pqxx::params{finalFirst.observation.observationId});
        invalid.commit();
    }
    catch (const pqxx::sql_error&)
    {
        finalCheckpointRejected = true;
    }
    assert(finalCheckpointRejected);

    bool checkpointWithoutIdentityRejected = false;
    try
    {
        pqxx::subtransaction invalid{transaction, "checkpoint_without_identity"};
        invalid.exec(
            "INSERT INTO inference_profitability_observation ("
            "experiment_id,model_id,inference_eval_result_id,inference_scope,"
            "checkpoint_eval_id,inference_start,inference_end,prediction_count,"
            "actionable_count,winning_actionable_count,losing_actionable_count,"
            "gross_positive_terminal_horizon_log_return_sum,"
            "gross_negative_terminal_horizon_log_return_sum,"
            "aggregate_terminal_horizon_log_return_sum,"
            "average_terminal_horizon_log_return_per_actionable_prediction,"
            "metric_definition_canonical,metric_definition_hash,source_content_hash,"
            "observation_identity_canonical,observation_identity_hash) "
            "SELECT experiment_id,model_id,inference_eval_result_id,'checkpoint',NULL,"
            "inference_start,inference_end,prediction_count,actionable_count,"
            "winning_actionable_count,losing_actionable_count,"
            "gross_positive_terminal_horizon_log_return_sum,"
            "gross_negative_terminal_horizon_log_return_sum,"
            "aggregate_terminal_horizon_log_return_sum,"
            "average_terminal_horizon_log_return_per_actionable_prediction,"
            "metric_definition_canonical,metric_definition_hash,source_content_hash,"
            "'invalid-checkpoint-null','fnv1a64:0000000000000000' "
            "FROM inference_profitability_observation "
            "WHERE profitability_observation_id=$1;",
            pqxx::params{checkpoint.observation.observationId});
        invalid.commit();
    }
    catch (const pqxx::sql_error&)
    {
        checkpointWithoutIdentityRejected = true;
    }
    assert(checkpointWithoutIdentityRejected);

    Accumulator changedCalculation;
    changedCalculation.Observe(kNeutralClass, 100.0f, 121.0f);
    changedCalculation.Observe(kUpClass, 100.0f, 110.0f);
    changedCalculation.Observe(kDownClass, 100.0f, 110.0f);
    const PersistResult changedSource = PersistObservationIdempotently(
        transaction,
        Request(100, Scope::finalInference, std::nullopt,
                changedCalculation));
    assert(changedSource.created);
    assert(changedSource.observation.observationId !=
           finalFirst.observation.observationId);

    ObservationRequest changedMetric = finalRequest;
    changedMetric.metricDefinitionCanonical += ";fixture_revision=2";
    const PersistResult changedDefinition =
        PersistObservationIdempotently(transaction, changedMetric);
    assert(changedDefinition.created);
    assert(changedDefinition.observation.observationId !=
           finalFirst.observation.observationId);
    assert(changedDefinition.observation.metricDefinitionHash !=
           finalFirst.observation.metricDefinitionHash);

    assert(transaction.exec(
        "SELECT count(*) FROM inference_profitability_observation "
        "WHERE inference_eval_result_id=99;")
        .one_row()[0].as<int>() == 0);
    const pqxx::row policyGuards = transaction.exec(
        "SELECT recommendation_score_guard,campaign_guard,"
        "continuation_policy_guard,checkpoint_policy_guard "
        "FROM experiment WHERE experiment_id=1;").one_row();
    assert(policyGuards[0].as<double>() == 0.75);
    assert(policyGuards[1].as<std::string>() == "unchanged");
    assert(policyGuards[2].as<std::string>() == "unchanged");
    assert(policyGuards[3].as<std::string>() == "unchanged");

    bool immutable = false;
    try
    {
        pqxx::subtransaction invalid{transaction, "immutable_observation"};
        invalid.exec(
            "UPDATE inference_profitability_observation SET actionable_count=0 "
            "WHERE profitability_observation_id=$1;",
            pqxx::params{finalFirst.observation.observationId});
        invalid.commit();
    }
    catch (const pqxx::sql_error&)
    {
        immutable = true;
    }
    assert(immutable);

    // The surrounding transaction rolls back all fixtures. The repository
    // never writes experiment, recommendation/campaign, continuation, or
    // checkpoint-policy state.
    transaction.abort();
    return 0;
}
