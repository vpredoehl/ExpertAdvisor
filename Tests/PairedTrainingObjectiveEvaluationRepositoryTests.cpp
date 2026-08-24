#include "InferenceProfitabilityRepository.hpp"
#include "PairedTrainingObjectiveEvaluation.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace Pair = EA::PairedTrainingObjectiveEvaluation;
namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

constexpr double kExperimentThreshold = 0.0008;
const double kModelThreshold =
    static_cast<double>(static_cast<float>(kExperimentThreshold));

struct FixtureIds
{
    long long experiment;
    long long model;
    long long inference;
    long long analysis;
};

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::runtime_error(std::string{"missing_environment:"} + name);
    return value;
}

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row,
                                   const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

void InsertMatrixRow(pqxx::transaction_base& transaction,
                     long long modelId,
                     const std::string& name,
                     const std::vector<double>& values)
{
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        transaction.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
            "VALUES($1,$2,0,$3,$4);",
            pqxx::params{modelId, name, static_cast<int>(index), values[index]});
    }
}

void InsertAscii(pqxx::transaction_base& transaction,
                 long long modelId,
                 const std::string& name,
                 const std::string& value)
{
    std::vector<double> encoded;
    encoded.reserve(value.size());
    for (const unsigned char character : value)
        encoded.push_back(static_cast<double>(character));
    InsertMatrixRow(transaction, modelId, name, encoded);
}

void ReplaceAscii(pqxx::transaction_base& transaction,
                  long long modelId,
                  const std::string& name,
                  const std::string& value)
{
    transaction.exec(
        "DELETE FROM matrix WHERE model_id=$1 AND param_name=$2;",
        pqxx::params{modelId, name});
    InsertAscii(transaction, modelId, name, value);
}

void InsertExperiment(pqxx::transaction_base& transaction,
                      const FixtureIds& ids,
                      const Objective::Configuration& objective)
{
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::DeterministicHash(canonical);
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,last_model_id,"
        "resume_model_id,duplicate_nonce,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,"
        "schema_version,scheduler_version,binary_name,training_objective_id,"
        "training_objective_version,loss_definition_version,"
        "training_objective_canonical,training_objective_hash,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "regression_target_definition,regression_normalization_identity,"
        "robust_loss_definition,robust_loss_delta,target_clipping_definition,"
        "objective_normalization_identity) VALUES("
        "$1,'synthetic_phase4d_symbol',6,$2,120,25,80,20,"
        "'2020-01-01','2025-01-01','2025-01-02','2026-01-01',"
        "'completed','done',$3,NULL,$1,'enabled',20,'full_history_warmup','',"
        "false,'syntheticcommit','phase6',false,'Release','AppleClang-test',"
        "'079','scheduler-test','LSTM_Release',$4,$5,$6,$7,$8,$9,$10,"
        "$11,$12,$13,$14,$15,$16);",
        pqxx::params{
            ids.experiment,
            kExperimentThreshold,
            ids.model,
            objective.objectiveIdentifier,
            objective.objectiveVersion,
            objective.lossDefinitionVersion,
            canonical,
            hash,
            std::string{Objective::AuxiliaryLossModeText(
                objective.auxiliaryLossMode)},
            objective.auxiliaryLossCoefficient,
            objective.regressionTargetDefinition,
            objective.regressionNormalizationIdentity,
            objective.robustLossDefinition,
            objective.robustLossDelta,
            objective.targetClippingDefinition,
            std::string{"weighted_loss_sum_by_weight_sum_gradients__"
                        "calculate_batch_return_by_example_count_v1"}});

    transaction.exec(
        "INSERT INTO model(model_id,experiment_id) VALUES($1,$2);",
        pqxx::params{ids.model, ids.experiment});

    InsertMatrixRow(transaction, ids.model, "train_config_meta", {
        1.0, 6.0, kModelThreshold, 64.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 80.0, 120.0, 25.0, 2.5});
    InsertMatrixRow(transaction, ids.model, "target_meta",
                    {1.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    InsertMatrixRow(transaction, ids.model, "model_meta", {1.0, 50.0, 64.0});
    InsertMatrixRow(transaction, ids.model, "model_input_semantics_meta",
                    {1.0, 1.0});
    InsertAscii(transaction, ids.model, "training_objective_canonical_meta",
                canonical);
    InsertAscii(transaction, ids.model, "training_objective_hash_meta", hash);
    InsertAscii(transaction, ids.model, "feature_warmup_scope_meta",
                "full_history_warmup");
    InsertAscii(transaction, ids.model, "donchian20_mode_meta", "enabled");
    InsertAscii(transaction, ids.model, "donchian_lookback_meta", "20");

    transaction.exec(
        "INSERT INTO phase4d_runtime_objective_event("
        "experiment_id,event_name,objective_identifier,objective_hash) "
        "VALUES($1,'TRAINING_OBJECTIVE_ACTIVE',$2,$3);",
        pqxx::params{ids.experiment, objective.objectiveIdentifier, hash});
}

void InsertInference(pqxx::transaction_base& transaction,
                     const FixtureIds& ids,
                     long long inferenceId,
                     const std::string& scope = "final",
                     const std::optional<long long>& checkpointId = std::nullopt,
                     const std::string& fromDate = "2025-01-02",
                     const std::string& toDate = "2026-01-01")
{
    const std::optional<long long> parent =
        scope == "checkpoint"
            ? std::optional<long long>{ids.experiment}
            : std::nullopt;
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accuracy,accept_model,pred_down,pred_neutral,pred_up) "
        "VALUES($1,$2,'completed',$3,$4,$5,'synthetic_phase4d_symbol',6,$6,"
        "64,1,1,$7,$8,80,0.70,true,0.20,0.30,0.50);",
        pqxx::params{inferenceId, ids.model, scope, checkpointId, parent,
                     kModelThreshold, fromDate, toDate});
}

void InsertAnalysis(pqxx::transaction_base& transaction,
                    const FixtureIds& ids)
{
    transaction.exec(
        "INSERT INTO experiment_analysis_result("
        "analysis_id,experiment_id,model_id,analysis_scope,analysis_status,"
        "infer_accuracy,accept_accuracy,accept_rate,leader_score) "
        "VALUES($1,$2,$3,'final','completed',0.70,0.75,0.70,0.63);",
        pqxx::params{ids.analysis, ids.experiment, ids.model});
}

Profitability::Observation PersistObservation(
    pqxx::transaction_base& transaction,
    const FixtureIds& ids,
    long long inferenceId,
    Profitability::Scope scope,
    const std::optional<long long>& checkpointId,
    const Profitability::Accumulator& accumulator,
    const std::string& fromDate = "2025-01-02",
    const std::string& toDate = "2026-01-01")
{
    Profitability::ObservationRequest request;
    request.provenance.experimentId = ids.experiment;
    request.provenance.modelId = ids.model;
    request.provenance.inferenceEvalResultId = inferenceId;
    request.provenance.scope = scope;
    request.provenance.checkpointEvalId = checkpointId;
    request.provenance.inferenceStart = fromDate;
    request.provenance.inferenceEnd = toDate;
    request.statistics = accumulator.statistics();
    request.sourceContentHash = accumulator.SourceContentHash();
    return Profitability::PersistObservationIdempotently(transaction, request)
        .observation;
}

double MatrixValue(pqxx::transaction_base& transaction,
                   long long modelId,
                   const std::string& name,
                   int column)
{
    return transaction.exec(
        "SELECT value FROM matrix WHERE model_id=$1 AND param_name=$2 "
        "AND row_idx=0 AND col_idx=$3;",
        pqxx::params{modelId, name, column}).one_row()[0].as<double>();
}

Pair::ArmEvidence LoadArm(pqxx::transaction_base& transaction,
                          long long experimentId)
{
    Pair::ArmEvidence arm;
    const pqxx::row row = transaction.exec(
        "SELECT experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start::date::text AS train_start,"
        "train_end::date::text AS train_end,"
        "infer_start::date::text AS infer_start,"
        "infer_end::date::text AS infer_end,status,phase,last_model_id,"
        "resume_model_id,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,"
        "schema_version,scheduler_version,binary_name,"
        "training_objective_canonical,training_objective_hash "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId}).one_row();

    auto& configuration = arm.configuration;
    configuration.experimentId = row["experiment_id"].as<long long>();
    configuration.symbol = row["symbol"].as<std::string>();
    configuration.predictionHorizon = row["prediction_horizon"].as<int>();
    configuration.threshold = row["c_next_threshold"].as<double>();
    configuration.coreLearningRateMultiplier =
        OptionalValue<double>(row, "core_lr_mult");
    configuration.headLearningRateMultiplier =
        OptionalValue<double>(row, "head_lr_mult");
    configuration.targetEpochs = row["target_epochs"].as<int>();
    configuration.checkpointInterval = row["checkpoint_interval"].as<int>();
    configuration.trainStart = row["train_start"].as<std::string>();
    configuration.trainEnd = row["train_end"].as<std::string>();
    configuration.inferenceStart = row["infer_start"].as<std::string>();
    configuration.inferenceEnd = row["infer_end"].as<std::string>();
    configuration.featureWarmupScope =
        row["feature_warmup_scope"].as<std::string>();
    configuration.donchianMode = row["donchian20_mode"].as<std::string>();
    configuration.donchianLookback = row["donchian_lookback"].as<int>();
    configuration.featureAblationMask =
        row["feature_ablation_mask"].as<std::string>();
    configuration.resumeModelId =
        OptionalValue<long long>(row, "resume_model_id");
    configuration.resumeExpandInputWidth =
        row["resume_expand_input_width"].as<bool>();
    configuration.experimentObjective = {
        row["training_objective_canonical"].as<std::string>(),
        row["training_objective_hash"].as<std::string>()};
    configuration.runProvenance = {
        row["git_commit"].as<std::string>(),
        row["git_branch"].as<std::string>(),
        row["git_dirty"].as<bool>(),
        row["build_config"].as<std::string>(),
        row["compiler_version"].as<std::string>(),
        row["schema_version"].as<std::string>(),
        row["scheduler_version"].as<std::string>(),
        row["binary_name"].as<std::string>()};
    arm.experimentStatus = row["status"].as<std::string>();
    arm.experimentPhase = row["phase"].as<std::string>();
    arm.finalModelId = OptionalValue<long long>(row, "last_model_id");

    if (!arm.finalModelId) return arm;
    const long long finalModelId = *arm.finalModelId;
    configuration.inputWidth = static_cast<int>(
        std::llround(MatrixValue(transaction, finalModelId, "model_meta", 1)));
    configuration.hiddenSize = static_cast<int>(
        std::llround(MatrixValue(transaction, finalModelId, "model_meta", 2)));
    configuration.layerCount = static_cast<int>(std::llround(
        MatrixValue(transaction, finalModelId, "train_config_meta", 8)));
    configuration.windowSize = static_cast<int>(std::llround(
        MatrixValue(transaction, finalModelId, "train_config_meta", 3)));
    configuration.labelRuleId = static_cast<int>(std::llround(
        MatrixValue(transaction, finalModelId, "train_config_meta", 4)));
    configuration.targetType = static_cast<int>(std::llround(
        MatrixValue(transaction, finalModelId, "target_meta", 0)));
    const int normalization = static_cast<int>(std::llround(
        MatrixValue(transaction, finalModelId, "train_config_meta", 9)));
    const double modelCore =
        MatrixValue(transaction, finalModelId, "train_config_meta", 11);
    const double modelHeadWeight =
        MatrixValue(transaction, finalModelId, "train_config_meta", 12);
    const double modelHeadBias =
        MatrixValue(transaction, finalModelId, "train_config_meta", 13);
    configuration.architectureCanonical =
        "lstm_v1;layers=" + std::to_string(configuration.layerCount) +
        ";hidden=" + std::to_string(configuration.hiddenSize) +
        ";heads=3class+scalar";
    configuration.featureConfigurationCanonical =
        "feature_layout_v1;width=" + std::to_string(configuration.inputWidth) +
        ";normalization=" + std::to_string(normalization);
    configuration.optimizerConfigurationCanonical = "sgd_v1;moments=0";
    configuration.learningRateConfigurationCanonical =
        "base=runtime_default;core=" + Objective::CanonicalDouble(modelCore) +
        ";head_weight=" + Objective::CanonicalDouble(modelHeadWeight) +
        ";head_bias=" + Objective::CanonicalDouble(modelHeadBias);
    configuration.labelRuleCanonical =
        "up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1";
    configuration.targetSemanticsCanonical = "up_neutral_down_return_v1";
    configuration.modelInputProvenanceCanonical =
        "model_input_semantics_meta_v1;layout=" +
        std::to_string(static_cast<int>(std::llround(MatrixValue(
            transaction, finalModelId, "model_input_semantics_meta", 1)))) +
        ";width=" + std::to_string(configuration.inputWidth);
    configuration.initializationCanonical =
        "fresh_model_constant_initialization_contract_v1";

    const pqxx::result objectives = transaction.exec(
        "SELECT m.model_id,"
        "(SELECT string_agg(chr(round(value)::integer),'' ORDER BY col_idx) "
        " FROM matrix WHERE model_id=m.model_id "
        " AND param_name='training_objective_canonical_meta' AND row_idx=0) "
        " AS canonical,"
        "(SELECT string_agg(chr(round(value)::integer),'' ORDER BY col_idx) "
        " FROM matrix WHERE model_id=m.model_id "
        " AND param_name='training_objective_hash_meta' AND row_idx=0) AS hash "
        "FROM model m WHERE m.experiment_id=$1 ORDER BY m.model_id;",
        pqxx::params{experimentId});
    for (const pqxx::row& objectiveRow : objectives)
    {
        const long long modelId = objectiveRow["model_id"].as<long long>();
        arm.materializedModelObjectives.push_back({
            modelId,
            modelId == finalModelId,
            {objectiveRow["canonical"].is_null()
                 ? std::string{}
                 : objectiveRow["canonical"].as<std::string>(),
             objectiveRow["hash"].is_null()
                 ? std::string{}
                 : objectiveRow["hash"].as<std::string>()}});
    }

    const pqxx::result runtime = transaction.exec(
        "SELECT event_name,objective_identifier,objective_hash "
        "FROM phase4d_runtime_objective_event WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (runtime.size() == 1)
    {
        arm.runtimeObjective = Pair::RuntimeObjectiveEvidence{
            runtime[0]["event_name"].as<std::string>(),
            runtime[0]["objective_identifier"].as<std::string>(),
            runtime[0]["objective_hash"].as<std::string>()};
    }

    const auto exact = Profitability::ResolveExactFinalInferenceResult(
        transaction, experimentId, finalModelId);
    if (exact.status !=
            Profitability::ExactFinalInferenceResultStatus::available ||
        !exact.inferenceEvalResultId)
        return arm;

    const pqxx::result inferenceRows = transaction.exec(
        "SELECT id,model_id,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,status,symbol,prediction_horizon,"
        "threshold_logret,window_size,label_rule_id,target_type,from_date,"
        "to_date,completed_epochs,accuracy,pred_down,pred_neutral,pred_up "
        "FROM inference_eval_result WHERE id=$1;",
        pqxx::params{*exact.inferenceEvalResultId});
    const pqxx::result analysisRows = transaction.exec(
        "SELECT analysis_id,experiment_id,model_id,analysis_scope,"
        "analysis_status,infer_accuracy,accept_accuracy,accept_rate,"
        "leader_score FROM experiment_analysis_result "
        "WHERE experiment_id=$1 AND model_id=$2 AND analysis_scope='final';",
        pqxx::params{experimentId, finalModelId});
    if (inferenceRows.size() != 1 || analysisRows.size() != 1)
        return arm;

    const pqxx::row inference = inferenceRows.one_row();
    const pqxx::row analysis = analysisRows.one_row();
    Pair::ClassificationEvidence classification;
    classification.inferenceResultId = inference["id"].as<long long>();
    classification.modelId = inference["model_id"].as<long long>();
    classification.inferenceScope =
        inference["inference_scope"].as<std::string>();
    classification.checkpointEvalId =
        OptionalValue<long long>(inference, "checkpoint_eval_id");
    classification.parentExperimentId =
        OptionalValue<long long>(inference, "parent_experiment_id");
    classification.status = inference["status"].as<std::string>();
    classification.symbol = inference["symbol"].as<std::string>();
    classification.predictionHorizon =
        inference["prediction_horizon"].as<int>();
    classification.threshold = inference["threshold_logret"].as<double>();
    classification.windowSize = inference["window_size"].as<int>();
    classification.labelRuleId = inference["label_rule_id"].as<int>();
    classification.targetType = inference["target_type"].as<int>();
    classification.inferenceStart = inference["from_date"].as<std::string>();
    classification.inferenceEnd = inference["to_date"].as<std::string>();
    classification.completedEpochs = inference["completed_epochs"].as<int>();
    classification.accuracy = OptionalValue<double>(inference, "accuracy");
    classification.predictedDownProportion =
        OptionalValue<double>(inference, "pred_down");
    classification.predictedNeutralProportion =
        OptionalValue<double>(inference, "pred_neutral");
    classification.predictedUpProportion =
        OptionalValue<double>(inference, "pred_up");
    classification.analysisId = analysis["analysis_id"].as<long long>();
    classification.analysisExperimentId =
        analysis["experiment_id"].as<long long>();
    classification.analysisModelId = analysis["model_id"].as<long long>();
    classification.analysisScope =
        analysis["analysis_scope"].as<std::string>();
    classification.analysisStatus =
        analysis["analysis_status"].as<std::string>();
    classification.inferenceAccuracy =
        OptionalValue<double>(analysis, "infer_accuracy");
    classification.acceptAccuracy =
        OptionalValue<double>(analysis, "accept_accuracy");
    classification.acceptRate = OptionalValue<double>(analysis, "accept_rate");
    classification.leaderScore = OptionalValue<double>(analysis, "leader_score");
    arm.classification = classification;

    Profitability::AuthoritativeObservationSelector selector;
    selector.experimentId = experimentId;
    selector.modelId = finalModelId;
    selector.inferenceEvalResultId = *exact.inferenceEvalResultId;
    selector.scope = Profitability::Scope::finalInference;
    selector.metricDefinitionCanonical =
        Profitability::kMetricDefinitionCanonical;
    selector.metricDefinitionHash = Profitability::MetricDefinitionHash();
    const auto selected =
        Profitability::SelectAuthoritativeObservation(transaction, selector);
    if (selected.status !=
            Profitability::AuthoritativeObservationStatus::available ||
        !selected.observation)
        return arm;

    const auto& observation = *selected.observation;
    Pair::ProfitabilityEvidence evidence;
    evidence.observationId = observation.observationId;
    evidence.experimentId = *observation.provenance.experimentId;
    evidence.modelId = observation.provenance.modelId;
    evidence.inferenceResultId =
        observation.provenance.inferenceEvalResultId;
    evidence.inferenceScope =
        Profitability::ScopeText(observation.provenance.scope);
    evidence.checkpointEvalId = observation.provenance.checkpointEvalId;
    evidence.inferenceStart = observation.provenance.inferenceStart;
    evidence.inferenceEnd = observation.provenance.inferenceEnd;
    evidence.predictionCount = observation.statistics.predictionCount;
    evidence.actionableCount = observation.statistics.actionableCount;
    evidence.aggregateTerminalHorizonLogReturnSum =
        observation.statistics.aggregateTerminalHorizonLogReturnSum;
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction =
        observation.averageTerminalHorizonLogReturnPerActionablePrediction;
    evidence.metricDefinitionCanonical =
        observation.metricDefinitionCanonical;
    evidence.metricDefinitionHash = observation.metricDefinitionHash;
    evidence.sourceContentHash = observation.sourceContentHash;
    evidence.observationIdentityCanonical =
        observation.observationIdentityCanonical;
    evidence.observationIdentityHash = observation.observationIdentityHash;
    arm.profitability = evidence;
    return arm;
}

Pair::MaterialityPolicy Policy()
{
    Pair::MaterialityPolicy policy;
    policy.minimumProfitabilityImprovement = 0.01;
    policy.maximumProfitabilityWorsening = 0.01;
    policy.classification = Pair::ClassificationDegradationPolicy{
        0.02, 0.02, 0.05, 0.02, 0.05};
    return policy;
}

Pair::ComparisonResult Compare(const Pair::ArmEvidence& control,
                               const Pair::ArmEvidence& treatment)
{
    return Pair::Compare(control, treatment, Policy());
}

} // namespace

int main()
{
    const std::string databaseName =
        RequiredEnvironment("EA_PHASE4D_VERIFY_DB_NAME");
    assert(std::regex_match(
        databaseName,
        std::regex{"^ea_phase4d_verify_[A-Za-z0-9_]+$"}));
    const std::string connectionString =
        "host=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_HOST") +
        " port=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_PORT") +
        " user=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_USER") +
        " dbname=" + databaseName;
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};

    const FixtureIds controlIds{990001, 1990001, 2990001, 4990001};
    const FixtureIds treatmentIds{990002, 1990002, 2990002, 4990002};
    InsertExperiment(transaction, controlIds, Objective::Legacy());
    InsertExperiment(
        transaction, treatmentIds, Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, controlIds, controlIds.inference);
    InsertInference(transaction, treatmentIds, treatmentIds.inference);
    InsertAnalysis(transaction, controlIds);
    InsertAnalysis(transaction, treatmentIds);

    Profitability::Accumulator controlReturns;
    controlReturns.Observe(Profitability::kNeutralClass, 100.0f, 110.0f);
    controlReturns.Observe(Profitability::kUpClass, 100.0f, 105.0f);
    Profitability::Accumulator treatmentReturns;
    treatmentReturns.Observe(Profitability::kNeutralClass, 100.0f, 110.0f);
    treatmentReturns.Observe(Profitability::kUpClass, 100.0f, 120.0f);
    const auto controlObservation = PersistObservation(
        transaction, controlIds, controlIds.inference,
        Profitability::Scope::finalInference, std::nullopt, controlReturns);
    const auto treatmentObservation = PersistObservation(
        transaction, treatmentIds, treatmentIds.inference,
        Profitability::Scope::finalInference, std::nullopt, treatmentReturns);

    const Pair::ArmEvidence control = LoadArm(transaction, controlIds.experiment);
    const Pair::ArmEvidence treatment =
        LoadArm(transaction, treatmentIds.experiment);
    assert(control.finalModelId == controlIds.model);
    assert(treatment.finalModelId == treatmentIds.model);
    assert(control.materializedModelObjectives.size() == 1);
    assert(treatment.materializedModelObjectives.size() == 1);
    assert(control.materializedModelObjectives[0].isFinalModel);
    assert(treatment.materializedModelObjectives[0].isFinalModel);
    assert(control.configuration.threshold == kExperimentThreshold);
    assert(treatment.configuration.threshold == kExperimentThreshold);
    assert(treatment.classification);
    assert(treatment.classification->threshold == kModelThreshold);
    assert(treatment.classification->threshold !=
           treatment.configuration.threshold);
    assert(std::fabs(treatment.classification->threshold -
                     treatment.configuration.threshold) <= 1.0e-7);
    assert(treatment.profitability);
    assert(treatment.profitability->observationId ==
           treatmentObservation.observationId);
    assert(treatment.profitability->inferenceResultId == treatmentIds.inference);
    assert(treatment.profitability->inferenceScope == "final");
    assert(!treatment.profitability->checkpointEvalId);
    assert(treatment.profitability->actionableCount == 1);
    assert(treatment.profitability
               ->averageTerminalHorizonLogReturnPerActionablePrediction);
    assert(treatment.classification->inferenceAccuracy == 0.70);
    assert(treatment.classification->acceptAccuracy == 0.75);
    assert(treatment.classification->acceptRate == 0.70);
    assert(treatment.classification->leaderScore == 0.63);
    assert(control.profitability->observationId == controlObservation.observationId);
    const auto positive = Compare(control, treatment);
    assert(positive.disposition == Pair::Disposition::Promising);
    assert(positive.invalidReasons.empty());
    assert(positive.incompleteReasons.empty());

    // Checkpoint profitability cannot fill a missing final observation.
    const FixtureIds checkpointOnlyIds{990011, 1990011, 2990011, 4990011};
    InsertExperiment(transaction, checkpointOnlyIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, checkpointOnlyIds, checkpointOnlyIds.inference);
    InsertAnalysis(transaction, checkpointOnlyIds);
    const long long checkpointId = 5990011;
    const long long checkpointInferenceId = 2990111;
    transaction.exec(
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{checkpointId, checkpointOnlyIds.experiment,
                     checkpointOnlyIds.model});
    InsertInference(transaction, checkpointOnlyIds, checkpointInferenceId,
                    "checkpoint", checkpointId);
    PersistObservation(transaction, checkpointOnlyIds, checkpointInferenceId,
                       Profitability::Scope::checkpointInference, checkpointId,
                       treatmentReturns);
    const auto checkpointOnly =
        LoadArm(transaction, checkpointOnlyIds.experiment);
    assert(checkpointOnly.classification);
    assert(!checkpointOnly.profitability);
    const auto checkpointOnlyResult = Compare(control, checkpointOnly);
    assert(checkpointOnlyResult.disposition == Pair::Disposition::Incomplete);
    assert(Has(checkpointOnlyResult.incompleteReasons,
               "treatment_final_profitability_missing"));

    // A final inference row plus only a differently linked final observation
    // remains incomplete; there is no recency or model-only fallback.
    const FixtureIds noProfitIds{990012, 1990012, 2990012, 4990012};
    InsertExperiment(transaction, noProfitIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, noProfitIds, noProfitIds.inference);
    InsertAnalysis(transaction, noProfitIds);
    const long long wrongContextInferenceId = 2991012;
    InsertInference(transaction, noProfitIds, wrongContextInferenceId,
                    "final", std::nullopt, "2025-02-01", "2026-02-01");
    PersistObservation(transaction, noProfitIds, wrongContextInferenceId,
                       Profitability::Scope::finalInference, std::nullopt,
                       treatmentReturns, "2025-02-01", "2026-02-01");
    const auto noProfit = LoadArm(transaction, noProfitIds.experiment);
    assert(noProfit.classification);
    assert(!noProfit.profitability);
    assert(Compare(control, noProfit).disposition ==
           Pair::Disposition::Incomplete);

    // More than one exact final inference row fails closed without choosing
    // the newest or smallest identity.
    const FixtureIds ambiguousIds{990013, 1990013, 2990013, 4990013};
    InsertExperiment(transaction, ambiguousIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, ambiguousIds, ambiguousIds.inference);
    InsertInference(transaction, ambiguousIds, 2991013);
    InsertAnalysis(transaction, ambiguousIds);
    const auto ambiguous = LoadArm(transaction, ambiguousIds.experiment);
    assert(!ambiguous.classification);
    assert(!ambiguous.profitability);
    assert(Compare(control, ambiguous).disposition ==
           Pair::Disposition::Incomplete);

    // Zero actionable predictions preserve aggregate zero and NULL average.
    const FixtureIds zeroIds{990014, 1990014, 2990014, 4990014};
    InsertExperiment(transaction, zeroIds, Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, zeroIds, zeroIds.inference);
    InsertAnalysis(transaction, zeroIds);
    Profitability::Accumulator zeroReturns;
    zeroReturns.Observe(Profitability::kNeutralClass, 100.0f, 120.0f);
    PersistObservation(transaction, zeroIds, zeroIds.inference,
                       Profitability::Scope::finalInference, std::nullopt,
                       zeroReturns);
    const auto zero = LoadArm(transaction, zeroIds.experiment);
    assert(zero.profitability);
    assert(zero.profitability->actionableCount == 0);
    assert(zero.profitability->aggregateTerminalHorizonLogReturnSum == 0.0);
    assert(!zero.profitability
                ->averageTerminalHorizonLogReturnPerActionablePrediction);
    const auto zeroResult = Compare(control, zero);
    assert(zeroResult.invalidReasons.empty());
    assert(zeroResult.incompleteReasons.empty());

    // Hostile persisted experiment hash is rejected by the evaluator.  The
    // migration constraint is deliberately dropped only inside this test
    // transaction; the transaction is aborted below.
    transaction.exec(
        "ALTER TABLE experiment DROP CONSTRAINT "
        "experiment_training_objective_provenance_check;");
    transaction.exec(
        "UPDATE experiment SET training_objective_hash="
        "'fnv1a64:0000000000000000' WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    const auto badExperimentHash =
        LoadArm(transaction, treatmentIds.experiment);
    const auto badExperimentHashResult = Compare(control, badExperimentHash);
    assert(badExperimentHashResult.disposition ==
           Pair::Disposition::InvalidComparison);
    assert(Has(badExperimentHashResult.invalidReasons,
               "treatment_experiment_objective_invalid"));
    const std::string auxiliaryCanonical =
        Objective::CanonicalText(Objective::ProfitabilityAuxiliary());
    const std::string auxiliaryHash =
        Objective::DeterministicHash(auxiliaryCanonical);
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, auxiliaryCanonical, auxiliaryHash});

    // A numerically close but noncanonical objective text remains invalid even
    // with its matching hash; canonical representation is authoritative.
    std::string closeCanonical = auxiliaryCanonical;
    const std::string needle = "auxiliary_loss_coefficient=0.1;";
    const std::size_t coefficient = closeCanonical.find(needle);
    assert(coefficient != std::string::npos);
    closeCanonical.replace(coefficient, needle.size(),
                           "auxiliary_loss_coefficient=0.10000000000000001;");
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, closeCanonical,
                     Objective::DeterministicHash(closeCanonical)});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "treatment_experiment_objective_invalid"));
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, auxiliaryCanonical, auxiliaryHash});

    const std::string legacyCanonical =
        Objective::CanonicalText(Objective::Legacy());
    const std::string legacyHash = Objective::DeterministicHash(legacyCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_canonical_meta", legacyCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_hash_meta", legacyHash);
    const auto modelMismatch = LoadArm(transaction, treatmentIds.experiment);
    const auto modelMismatchResult = Compare(control, modelMismatch);
    assert(modelMismatchResult.disposition == Pair::Disposition::InvalidComparison);
    assert(Has(modelMismatchResult.invalidReasons,
               "treatment_experiment_model_objective_mismatch"));
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_canonical_meta", auxiliaryCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_hash_meta", auxiliaryHash);

    transaction.exec(
        "UPDATE experiment SET symbol='synthetic_other_symbol' "
        "WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "symbol_mismatch"));
    transaction.exec(
        "UPDATE experiment SET symbol='synthetic_phase4d_symbol' "
        "WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});

    transaction.exec(
        "UPDATE experiment SET prediction_horizon=12 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "prediction_horizon_mismatch"));
    transaction.exec(
        "UPDATE experiment SET prediction_horizon=6 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});

    transaction.exec(
        "UPDATE experiment SET c_next_threshold=0.001 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "threshold_mismatch"));
    transaction.exec(
        "UPDATE experiment SET c_next_threshold=$2 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, kExperimentThreshold});

    transaction.exec(
        "UPDATE experiment SET core_lr_mult=121 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "core_lr_mismatch"));
    transaction.exec(
        "UPDATE experiment SET core_lr_mult=120 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});

    transaction.exec(
        "UPDATE matrix SET value=2.5001 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND row_idx=0 AND col_idx=13;",
        pqxx::params{treatmentIds.model});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "learning_rate_configuration_mismatch"));
    transaction.exec(
        "UPDATE matrix SET value=2.5 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND row_idx=0 AND col_idx=13;",
        pqxx::params{treatmentIds.model});

    // The Phase 1 trigger rejects a profitability row whose declared
    // experiment/model provenance does not own its inference result.
    bool wrongInferenceLinkRejected = false;
    try
    {
        pqxx::subtransaction hostile{transaction, "wrong_inference_link"};
        Profitability::ObservationRequest request;
        request.provenance.experimentId = treatmentIds.experiment;
        request.provenance.modelId = treatmentIds.model;
        request.provenance.inferenceEvalResultId = controlIds.inference;
        request.provenance.scope = Profitability::Scope::finalInference;
        request.provenance.inferenceStart = "2025-01-02";
        request.provenance.inferenceEnd = "2026-01-01";
        request.statistics = treatmentReturns.statistics();
        request.sourceContentHash = "fnv1a64:0000000000000001";
        (void)Profitability::PersistObservationIdempotently(hostile, request);
        hostile.commit();
    }
    catch (const pqxx::sql_error&)
    {
        wrongInferenceLinkRejected = true;
    }
    assert(wrongInferenceLinkRejected);

    // Everything above is synthetic and remains confined to the disposable
    // database.  Rollback is an additional fixture cleanup layer; the shell
    // drops the entire database afterward.
    transaction.abort();
    std::cout << "paired_training_objective_repository_tests_passed\n";
    return 0;
}
