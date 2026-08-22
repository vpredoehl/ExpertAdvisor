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
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "infer_start,infer_end,last_model_id,"
        "recommendation_score_guard,campaign_guard,"
        "continuation_policy_guard,checkpoint_policy_guard) "
        "VALUES(1,'USDJPYRMP',5,0.001,'2025-01-01','2026-01-01',10,"
        "0.75,'unchanged','unchanged','unchanged');"
        "INSERT INTO model(model_id,experiment_id) VALUES(10,1),(11,NULL);"
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT 10,'train_config_meta',0,ordinality-1,value "
        "FROM unnest(ARRAY[1,5,0.001,64,1,1,1,1,1,1,100,1,1,1]"
        "::double precision[]) WITH ORDINALITY AS config(value,ordinality);"
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES(10,'target_meta',0,0,1);"
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES(200,1,10),(201,1,10);"
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accept_model,completed_at) VALUES"
        "(99,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,64,1,1,"
        "'2024-01-01','2024-12-31',100,false,'2026-01-01'),"
        "(100,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2026-01-01',100,true,'2026-01-02'),"
        "(101,10,'completed','checkpoint',200,1,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2026-01-01',100,true,'2026-01-03'),"
        "(102,11,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2026-01-01',100,true,'2026-01-04'),"
        "(103,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,65,1,1,"
        "'2025-01-01','2026-01-01',100,true,'2026-01-05'),"
        "(104,10,'completed','checkpoint',201,1,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2026-01-01',100,true,'2026-01-06'),"
        "(105,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-02-01','2026-02-01',100,false,'2026-08-01'),"
        "(106,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.002,64,1,1,"
        "'2025-01-01','2026-01-01',100,false,'2026-08-02');");

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

    const auto exactFinalInference = ResolveExactFinalInferenceResult(
        transaction, 1, 10);
    assert(exactFinalInference.status ==
           ExactFinalInferenceResultStatus::available);
    assert(exactFinalInference.inferenceEvalResultId == 100);
    assert(exactFinalInference.acceptModel.has_value() &&
           *exactFinalInference.acceptModel);
    assert(ExactFinalInferenceResultStatusText(
               exactFinalInference.status) == "available");

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

    AuthoritativeObservationSelector exactFinalSelector;
    exactFinalSelector.experimentId = 1;
    exactFinalSelector.modelId = 10;
    exactFinalSelector.inferenceEvalResultId = 100;
    exactFinalSelector.scope = Scope::finalInference;
    exactFinalSelector.metricDefinitionCanonical =
        kMetricDefinitionCanonical;
    exactFinalSelector.metricDefinitionHash = MetricDefinitionHash();
    const auto exactFinal = SelectAuthoritativeObservation(
        transaction,
        exactFinalSelector);
    assert(exactFinal.status == AuthoritativeObservationStatus::available);
    assert(exactFinal.observation.has_value());
    assert(exactFinal.observation->observationId ==
           finalFirst.observation.observationId);
    assert(exactFinal.observation->provenance.inferenceEvalResultId ==
           *exactFinalInference.inferenceEvalResultId);

    transaction.exec(
        "UPDATE experiment SET infer_start='2025-03-01',"
        "infer_end='2026-03-01' WHERE experiment_id=1;");
    const auto noExactFinalInference = ResolveExactFinalInferenceResult(
        transaction, 1, 10);
    assert(noExactFinalInference.status ==
           ExactFinalInferenceResultStatus::noExactFinalInferenceResult);
    assert(!noExactFinalInference.inferenceEvalResultId.has_value());
    assert(ExactFinalInferenceResultStatusText(
               noExactFinalInference.status) ==
           "no_exact_final_inference_result");

    transaction.exec(
        "UPDATE experiment SET infer_start='2025-01-01',"
        "infer_end='2026-01-01',c_next_threshold=0.01 "
        "WHERE experiment_id=1;");
    const auto mismatchedFinalContext = ResolveExactFinalInferenceResult(
        transaction, 1, 10);
    assert(mismatchedFinalContext.status ==
           ExactFinalInferenceResultStatus::finalInferenceContextMismatch);
    assert(!mismatchedFinalContext.inferenceEvalResultId.has_value());
    assert(ExactFinalInferenceResultStatusText(
               mismatchedFinalContext.status) ==
           "final_inference_context_mismatch");
    transaction.exec(
        "UPDATE experiment SET c_next_threshold=0.001 "
        "WHERE experiment_id=1;");

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

    AuthoritativeObservationSelector exactCheckpointSelector;
    exactCheckpointSelector.experimentId = 1;
    exactCheckpointSelector.modelId = 10;
    exactCheckpointSelector.inferenceEvalResultId = 101;
    exactCheckpointSelector.scope = Scope::checkpointInference;
    exactCheckpointSelector.checkpointEvalId = 200;
    exactCheckpointSelector.metricDefinitionCanonical =
        kMetricDefinitionCanonical;
    exactCheckpointSelector.metricDefinitionHash = MetricDefinitionHash();
    const auto exactCheckpoint = SelectAuthoritativeObservation(
        transaction,
        exactCheckpointSelector);
    assert(exactCheckpoint.status ==
           AuthoritativeObservationStatus::available);
    assert(exactCheckpoint.observation.has_value());
    assert(exactCheckpoint.observation->observationId ==
           checkpoint.observation.observationId);

    AuthoritativeObservationSelector finalCannotUseCheckpoint =
        exactFinalSelector;
    finalCannotUseCheckpoint.inferenceEvalResultId = 101;
    assert(SelectAuthoritativeObservation(
               transaction,
               finalCannotUseCheckpoint).status ==
           AuthoritativeObservationStatus::provenanceMismatch);

    AuthoritativeObservationSelector checkpointCannotUseFinal =
        exactCheckpointSelector;
    checkpointCannotUseFinal.inferenceEvalResultId = 100;
    assert(SelectAuthoritativeObservation(
               transaction,
               checkpointCannotUseFinal).status ==
           AuthoritativeObservationStatus::provenanceMismatch);

    AuthoritativeObservationSelector mismatchedProvenance =
        exactFinalSelector;
    mismatchedProvenance.modelId = 11;
    assert(SelectAuthoritativeObservation(
               transaction,
               mismatchedProvenance).status ==
           AuthoritativeObservationStatus::provenanceMismatch);

    AuthoritativeObservationSelector mismatchedMetric = exactFinalSelector;
    mismatchedMetric.metricDefinitionHash = "fnv1a64:0000000000000000";
    assert(SelectAuthoritativeObservation(
               transaction,
               mismatchedMetric).status ==
           AuthoritativeObservationStatus::metricDefinitionMismatch);
    mismatchedMetric = exactFinalSelector;
    mismatchedMetric.metricDefinitionCanonical += ";future_revision";
    assert(SelectAuthoritativeObservation(
               transaction,
               mismatchedMetric).status ==
           AuthoritativeObservationStatus::metricDefinitionMismatch);

    AuthoritativeObservationSelector historicalSelector = exactFinalSelector;
    historicalSelector.inferenceEvalResultId = 99;
    assert(SelectAuthoritativeObservation(
               transaction,
               historicalSelector).status ==
           AuthoritativeObservationStatus::noObservation);

    Accumulator zeroActionableCalculation;
    zeroActionableCalculation.Observe(kNeutralClass, 100.0f, 110.0f);
    const PersistResult zeroActionable = PersistObservationIdempotently(
        transaction,
        Request(103, Scope::finalInference, std::nullopt,
                zeroActionableCalculation));
    AuthoritativeObservationSelector zeroActionableSelector =
        exactFinalSelector;
    zeroActionableSelector.inferenceEvalResultId = 103;
    const auto selectedZeroActionable = SelectAuthoritativeObservation(
        transaction,
        zeroActionableSelector);
    assert(selectedZeroActionable.status ==
           AuthoritativeObservationStatus::available);
    assert(selectedZeroActionable.observation->statistics.actionableCount ==
           0);
    assert(!selectedZeroActionable.observation
                ->averageTerminalHorizonLogReturnPerActionablePrediction
                .has_value());

    const PersistResult laterCheckpoint = PersistObservationIdempotently(
        transaction,
        Request(104, Scope::checkpointInference, 201, calculation));
    AuthoritativeObservationSelector laterCheckpointSelector =
        exactCheckpointSelector;
    laterCheckpointSelector.inferenceEvalResultId = 104;
    laterCheckpointSelector.checkpointEvalId = 201;
    const auto selectedLaterCheckpoint = SelectAuthoritativeObservation(
        transaction,
        laterCheckpointSelector);
    assert(selectedLaterCheckpoint.status ==
           AuthoritativeObservationStatus::available);
    assert(selectedLaterCheckpoint.observation->observationId ==
           laterCheckpoint.observation.observationId);
    assert(selectedLaterCheckpoint.observation->observationId !=
           exactCheckpoint.observation->observationId);

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
    const auto ambiguousFinal = SelectAuthoritativeObservation(
        transaction,
        exactFinalSelector);
    assert(ambiguousFinal.status ==
           AuthoritativeObservationStatus::ambiguousObservation);
    assert(!ambiguousFinal.observation.has_value());

    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accept_model,completed_at) VALUES("
        "107,10,'completed','final',NULL,NULL,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2026-01-01',100,false,'2026-08-03');");
    const auto ambiguousFinalInference = ResolveExactFinalInferenceResult(
        transaction, 1, 10);
    assert(ambiguousFinalInference.status ==
           ExactFinalInferenceResultStatus::ambiguousFinalInferenceResult);
    assert(!ambiguousFinalInference.inferenceEvalResultId.has_value());
    assert(ExactFinalInferenceResultStatusText(
               ambiguousFinalInference.status) ==
           "ambiguous_final_inference_result");

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
