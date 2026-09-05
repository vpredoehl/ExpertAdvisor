#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include "InferenceProfitabilityRepository.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "TrainingObjective.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace EA::PairedTrainingObjectiveEvaluation
{
namespace
{

namespace Profitability = EA::InferenceProfitability;
namespace Objective = EA::TrainingObjective;

[[noreturn]] void ContractFailure(long long experimentId,
                                  std::string reason)
{
    throw EvidenceLoadError(
        EvidenceLoadErrorKind::ProvenanceContractFailure,
        "experiment_" + std::to_string(experimentId) + '_' +
            std::move(reason));
}

[[noreturn]] void Ambiguous(long long experimentId, std::string reason)
{
    throw EvidenceLoadError(
        EvidenceLoadErrorKind::AmbiguousEvidence,
        "experiment_" + std::to_string(experimentId) + '_' +
            std::move(reason));
}

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row,
                                   const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

std::string TextOrEmpty(const pqxx::row& row, const char* column)
{
    return row[column].is_null() ? std::string{} :
        row[column].as<std::string>();
}

struct MatrixMetadata
{
    int rows = 0;
    int columns = 0;
    std::vector<double> values;
};

MatrixMetadata LoadMatrixMetadata(pqxx::transaction_base& transaction,
                                  long long experimentId,
                                  long long modelId,
                                  const std::string& name,
                                  bool required = true)
{
    const pqxx::result records = transaction.exec(
        "SELECT row_idx,col_idx,n_rows,n_cols,value FROM matrix "
        "WHERE model_id=$1 AND param_name=$2 ORDER BY row_idx,col_idx;",
        pqxx::params{modelId, name});
    if (records.empty())
    {
        if (!required) return {};
        ContractFailure(experimentId, "missing_" + name);
    }
    MatrixMetadata metadata;
    metadata.rows = records[0]["n_rows"].as<int>();
    metadata.columns = records[0]["n_cols"].as<int>();
    if (metadata.rows <= 0 || metadata.columns <= 0)
        ContractFailure(experimentId, name + "_shape_invalid");
    metadata.values.reserve(records.size());
    int expectedRow = 0;
    int expectedColumn = 0;
    for (const pqxx::row& record : records)
    {
        if (record["n_rows"].as<int>() != metadata.rows ||
            record["n_cols"].as<int>() != metadata.columns ||
            record["row_idx"].as<int>() != expectedRow ||
            record["col_idx"].as<int>() != expectedColumn)
            ContractFailure(experimentId, name + "_layout_invalid");
        metadata.values.push_back(record["value"].as<double>());
        if (++expectedColumn == metadata.columns)
        {
            expectedColumn = 0;
            ++expectedRow;
        }
    }
    if (expectedRow != metadata.rows || expectedColumn != 0 ||
        metadata.values.size() !=
            static_cast<std::size_t>(metadata.rows) *
                static_cast<std::size_t>(metadata.columns))
        ContractFailure(experimentId, name + "_value_count_invalid");
    return metadata;
}

std::pair<int, int> LoadMatrixShape(pqxx::transaction_base& transaction,
                                    long long experimentId,
                                    long long modelId,
                                    const std::string& name)
{
    const pqxx::result rows = transaction.exec(
        "SELECT DISTINCT n_rows,n_cols FROM matrix "
        "WHERE model_id=$1 AND param_name=$2;",
        pqxx::params{modelId, name});
    if (rows.size() != 1)
        ContractFailure(experimentId, name + "_shape_not_unique");
    return {rows[0][0].as<int>(), rows[0][1].as<int>()};
}

long long ExactInteger(long long experimentId,
                       double value,
                       const std::string& field,
                       long long minimum = 0)
{
    if (!std::isfinite(value) || std::trunc(value) != value ||
        value < static_cast<double>(minimum) ||
        value > static_cast<double>(std::numeric_limits<int>::max()))
        ContractFailure(experimentId, field + "_invalid");
    return static_cast<long long>(value);
}

std::string DecodeAscii(pqxx::transaction_base& transaction,
                        long long experimentId,
                        long long modelId,
                        const std::string& name,
                        bool required = true)
{
    const MatrixMetadata metadata = LoadMatrixMetadata(
        transaction, experimentId, modelId, name, required);
    if (metadata.values.empty() && !required) return {};
    if (metadata.rows != 1 || metadata.columns <= 0)
        ContractFailure(experimentId, name + "_ascii_shape_invalid");
    std::string value;
    value.reserve(metadata.values.size());
    for (double encoded : metadata.values)
    {
        const long long character = ExactInteger(
            experimentId, encoded, name + "_character", 1);
        if (character > 255)
            ContractFailure(experimentId, name + "_character_invalid");
        value.push_back(static_cast<char>(character));
    }
    return value;
}

void LoadFinalModelConfiguration(pqxx::transaction_base& transaction,
                                 ArmEvidence& arm)
{
    const long long experimentId = arm.configuration.experimentId;
    const long long modelId = *arm.finalModelId;
    const pqxx::result ownership = transaction.exec(
        "SELECT experiment_id,parent_model_id FROM model WHERE model_id=$1;",
        pqxx::params{modelId});
    if (ownership.size() != 1 || ownership[0]["experiment_id"].is_null() ||
        ownership[0]["experiment_id"].as<long long>() != experimentId)
        ContractFailure(experimentId, "final_model_ownership_mismatch");
    const std::optional<long long> finalParent = OptionalValue<long long>(
        ownership.one_row(), "parent_model_id");
    if (finalParent != arm.configuration.resumeModelId)
        ContractFailure(experimentId,
                        "final_model_initialization_lineage_mismatch");

    auto& configuration = arm.configuration;
    const MatrixMetadata model = LoadMatrixMetadata(
        transaction, experimentId, modelId, "model_meta");
    if (model.rows != 1 || model.columns != 3)
        ContractFailure(experimentId, "model_meta_shape_invalid");
    configuration.modelMetadataSchemaVersion = static_cast<int>(
        ExactInteger(experimentId, model.values[0], "model_meta_schema", 1));
    configuration.inputWidth = static_cast<int>(
        ExactInteger(experimentId, model.values[1], "model_input_width", 1));
    configuration.hiddenSize = static_cast<int>(
        ExactInteger(experimentId, model.values[2], "model_hidden_size", 1));
    (void)EA::ContractForModelInputWidth(
        static_cast<std::size_t>(configuration.inputWidth));
    const auto [parameterRows, parameterColumns] = LoadMatrixShape(
        transaction, experimentId, modelId, "param");
    if (parameterRows != configuration.inputWidth + configuration.hiddenSize ||
        parameterColumns != 4 * configuration.hiddenSize)
        ContractFailure(experimentId, "model_parameter_shape_mismatch");

    const MatrixMetadata train = LoadMatrixMetadata(
        transaction, experimentId, modelId, "train_config_meta");
    if (train.rows != 1 || train.columns < 14)
        ContractFailure(experimentId, "train_config_meta_shape_invalid");
    configuration.trainConfigurationSchemaVersion = static_cast<int>(
        ExactInteger(experimentId, train.values[0], "train_config_schema", 1));
    const int modelHorizon = static_cast<int>(ExactInteger(
        experimentId, train.values[1], "model_prediction_horizon", 1));
    configuration.windowSize = static_cast<int>(ExactInteger(
        experimentId, train.values[3], "model_window_size", 1));
    configuration.labelRuleId = static_cast<int>(ExactInteger(
        experimentId, train.values[4], "model_label_rule", 1));
    configuration.classWeightDown = train.values[5];
    configuration.classWeightNeutral = train.values[6];
    configuration.classWeightUp = train.values[7];
    configuration.layerCount = static_cast<int>(ExactInteger(
        experimentId, train.values[8], "model_layer_count", 1));
    configuration.normalizationVersion = static_cast<int>(ExactInteger(
        experimentId, train.values[9], "normalization_version", 0));
    const int completedEpochs = static_cast<int>(ExactInteger(
        experimentId, train.values[10], "model_completed_epochs", 0));
    configuration.persistedCoreLearningRateMultiplier = train.values[11];
    configuration.persistedHeadWeightLearningRateMultiplier = train.values[12];
    configuration.persistedHeadBiasLearningRateMultiplier = train.values[13];
    if (modelHorizon != configuration.predictionHorizon ||
        completedEpochs != configuration.targetEpochs ||
        !std::isfinite(train.values[2]) ||
        std::fabs(train.values[2] - configuration.threshold) > 1.0e-7)
        ContractFailure(experimentId, "final_model_training_context_mismatch");

    const MatrixMetadata target = LoadMatrixMetadata(
        transaction, experimentId, modelId, "target_meta");
    if (target.rows != 1 || target.columns != 6)
        ContractFailure(experimentId, "target_meta_shape_invalid");
    configuration.targetType = static_cast<int>(ExactInteger(
        experimentId, target.values[0], "target_type", 0));
    configuration.targetScale = target.values[1];
    configuration.targetBias = target.values[2];
    const long long useZScore = ExactInteger(
        experimentId, target.values[3], "target_use_zscore", 0);
    if (useZScore > 1)
        ContractFailure(experimentId, "target_use_zscore_invalid");
    configuration.targetUseZScore = useZScore == 1;
    configuration.targetMean = target.values[4];
    configuration.targetStandardDeviation = target.values[5];

    const MatrixMetadata optimizer = LoadMatrixMetadata(
        transaction, experimentId, modelId, "optimizer_meta");
    if (optimizer.rows != 1 || optimizer.columns < 5)
        ContractFailure(experimentId, "optimizer_meta_shape_invalid");
    configuration.optimizerMetadataSchemaVersion = static_cast<int>(
        ExactInteger(experimentId, optimizer.values[0],
                     "optimizer_meta_schema", 1));
    configuration.optimizerType = static_cast<int>(ExactInteger(
        experimentId, optimizer.values[1], "optimizer_type", 1));
    configuration.optimizerUpdateCount = static_cast<std::uint64_t>(
        ExactInteger(experimentId, optimizer.values[2],
                     "optimizer_update_count", 0));
    configuration.optimizerFirstMomentBufferCount = static_cast<int>(
        ExactInteger(experimentId, optimizer.values[3],
                     "optimizer_first_moment_count", 0));
    configuration.optimizerSecondMomentBufferCount = static_cast<int>(
        ExactInteger(experimentId, optimizer.values[4],
                     "optimizer_second_moment_count", 0));

    const MatrixMetadata input = LoadMatrixMetadata(
        transaction, experimentId, modelId, "model_input_semantics_meta");
    const EA::ModelInputSemanticMetadata inputMetadata =
        EA::ParseModelInputSemanticMetadata(
            static_cast<std::size_t>(input.rows),
            static_cast<std::size_t>(input.columns), input.values);
    if (!EA::IsModelInputSemanticLayoutWidthCompatible(
            inputMetadata.layoutVersion,
            static_cast<std::size_t>(configuration.inputWidth),
            EA::kModelInputSemanticLayoutRegistry,
            EA::kRegisteredModelInputWidths,
            EA::kModelInputSemanticLayoutVersion,
            EA::kCurrentModelInputWidth, false))
        ContractFailure(experimentId, "model_input_semantics_incompatible");
    configuration.modelInputMetadataSchemaVersion =
        inputMetadata.schemaVersion;
    configuration.modelInputLayoutVersion = inputMetadata.layoutVersion;

    configuration.persistedTrainingSymbol = DecodeAscii(
        transaction, experimentId, modelId, "train_symbol_meta");
    const std::string range = DecodeAscii(
        transaction, experimentId, modelId, "train_range_meta");
    const std::size_t separator = range.find('|');
    if (separator == std::string::npos ||
        range.find('|', separator + 1) != std::string::npos)
        ContractFailure(experimentId, "train_range_meta_invalid");
    configuration.persistedTrainingStart = range.substr(0, separator);
    configuration.persistedTrainingEnd = range.substr(separator + 1);

    const std::string expansion = DecodeAscii(
        transaction, experimentId, modelId, "input_width_expansion_meta", false);
    if (!expansion.empty())
    {
        const EA::InputWidthExpansionProvenance provenance =
            EA::ParseInputWidthExpansionProvenance(expansion);
        if (!configuration.resumeModelId ||
            provenance.sourceModelId != *configuration.resumeModelId ||
            provenance.expandedInputWidth !=
                static_cast<std::size_t>(configuration.inputWidth))
            ContractFailure(experimentId,
                            "input_width_expansion_context_mismatch");
        configuration.inputWidthExpansionCanonical = expansion;
    }
}

void LoadMaterializedObjectives(pqxx::transaction_base& transaction,
                                ArmEvidence& arm)
{
    const long long experimentId = arm.configuration.experimentId;
    const long long finalModelId = *arm.finalModelId;
    const pqxx::result models = transaction.exec(
        "SELECT model_id FROM model WHERE experiment_id=$1 ORDER BY model_id;",
        pqxx::params{experimentId});
    if (models.empty()) ContractFailure(experimentId, "model_lineage_missing");
    int finalCount = 0;
    for (const pqxx::row& model : models)
    {
        const long long modelId = model[0].as<long long>();
        const std::string canonical = DecodeAscii(
            transaction, experimentId, modelId,
            "training_objective_canonical_meta");
        const std::string hash = DecodeAscii(
            transaction, experimentId, modelId,
            "training_objective_hash_meta");
        // Parse now so malformed persisted text is a loader contract failure,
        // while the pure evaluator remains the objective-equality authority.
        (void)Objective::ParseSupportedCanonicalText(canonical);
        if (Objective::DeterministicHash(canonical) != hash)
            ContractFailure(experimentId, "model_objective_hash_mismatch");
        const bool isFinal = modelId == finalModelId;
        finalCount += isFinal ? 1 : 0;
        arm.materializedModelObjectives.push_back(
            {modelId, isFinal, {canonical, hash}});
    }
    if (finalCount != 1)
        ContractFailure(experimentId, "final_model_lineage_ambiguous");
}

void LoadFinalClassificationAndProfitability(
    pqxx::transaction_base& transaction,
    ArmEvidence& arm)
{
    const long long experimentId = arm.configuration.experimentId;
    const long long modelId = *arm.finalModelId;
    const auto exact = Profitability::ResolveExactFinalInferenceResult(
        transaction, experimentId, modelId);
    using ExactStatus = Profitability::ExactFinalInferenceResultStatus;
    if (exact.status == ExactStatus::ambiguousFinalInferenceResult)
        Ambiguous(experimentId, "final_inference_ambiguous");
    if (exact.status == ExactStatus::finalInferenceContextMismatch)
        ContractFailure(experimentId, "final_inference_context_mismatch");
    if (exact.status == ExactStatus::noExactFinalInferenceResult ||
        !exact.inferenceEvalResultId)
        return;

    const pqxx::result inferenceRows = transaction.exec(
        "SELECT id,model_id,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,status,symbol,prediction_horizon,"
        "threshold_logret,window_size,label_rule_id,target_type,from_date,"
        "to_date,completed_epochs,accuracy,pred_down,pred_neutral,pred_up "
        "FROM inference_eval_result WHERE id=$1;",
        pqxx::params{*exact.inferenceEvalResultId});
    if (inferenceRows.size() != 1)
        ContractFailure(experimentId, "final_inference_identity_missing");

    const pqxx::result analysisRows = transaction.exec(
        "SELECT analysis_id,experiment_id,model_id,analysis_scope,"
        "checkpoint_eval_id,parent_experiment_id,analysis_status,"
        "infer_accuracy,accept_accuracy,accept_rate,leader_score,"
        "pred_down_count,pred_neutral_count,pred_up_count,accept_count "
        "FROM experiment_analysis_result WHERE experiment_id=$1 "
        "AND model_id=$2 AND analysis_scope='final' ORDER BY analysis_id;",
        pqxx::params{experimentId, modelId});
    if (analysisRows.size() > 1)
        Ambiguous(experimentId, "final_analysis_ambiguous");
    if (analysisRows.empty()) return;

    const pqxx::row inference = inferenceRows.one_row();
    const pqxx::row analysis = analysisRows.one_row();
    ClassificationEvidence classification;
    classification.inferenceResultId = inference["id"].as<long long>();
    classification.modelId = inference["model_id"].as<long long>();
    classification.inferenceScope =
        inference["inference_scope"].as<std::string>();
    classification.checkpointEvalId = OptionalValue<long long>(
        inference, "checkpoint_eval_id");
    classification.parentExperimentId = OptionalValue<long long>(
        inference, "parent_experiment_id");
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
    classification.completedEpochs =
        inference["completed_epochs"].is_null() ? 0 :
            inference["completed_epochs"].as<int>();
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
    classification.analysisModelId = analysis["model_id"].is_null() ? 0 :
        analysis["model_id"].as<long long>();
    classification.analysisScope =
        analysis["analysis_scope"].as<std::string>();
    classification.analysisCheckpointEvalId = OptionalValue<long long>(
        analysis, "checkpoint_eval_id");
    classification.analysisParentExperimentId = OptionalValue<long long>(
        analysis, "parent_experiment_id");
    classification.analysisStatus =
        analysis["analysis_status"].as<std::string>();
    classification.inferenceAccuracy =
        OptionalValue<double>(analysis, "infer_accuracy");
    classification.acceptAccuracy =
        OptionalValue<double>(analysis, "accept_accuracy");
    classification.acceptRate = OptionalValue<double>(analysis, "accept_rate");
    classification.leaderScore =
        OptionalValue<double>(analysis, "leader_score");
    classification.predictedDownCount = OptionalValue<std::uint64_t>(
        analysis, "pred_down_count");
    classification.predictedNeutralCount = OptionalValue<std::uint64_t>(
        analysis, "pred_neutral_count");
    classification.predictedUpCount = OptionalValue<std::uint64_t>(
        analysis, "pred_up_count");
    classification.acceptedPredictionCount = OptionalValue<std::uint64_t>(
        analysis, "accept_count");
    arm.classification = classification;

    Profitability::AuthoritativeObservationSelector selector;
    selector.experimentId = experimentId;
    selector.modelId = modelId;
    selector.inferenceEvalResultId = classification.inferenceResultId;
    selector.scope = Profitability::Scope::finalInference;
    selector.metricDefinitionCanonical =
        Profitability::kMetricDefinitionCanonical;
    selector.metricDefinitionHash = Profitability::MetricDefinitionHash();
    const auto selected = Profitability::SelectAuthoritativeObservation(
        transaction, selector);
    using ObservationStatus = Profitability::AuthoritativeObservationStatus;
    if (selected.status == ObservationStatus::ambiguousObservation)
        Ambiguous(experimentId, "final_profitability_ambiguous");
    if (selected.status == ObservationStatus::metricDefinitionMismatch ||
        selected.status == ObservationStatus::provenanceMismatch)
        ContractFailure(experimentId, "final_profitability_" +
            Profitability::AuthoritativeObservationStatusText(selected.status));
    if (selected.status == ObservationStatus::noObservation ||
        !selected.observation)
        return;

    const auto& observation = *selected.observation;
    if (!observation.provenance.experimentId)
        ContractFailure(experimentId,
                        "final_profitability_experiment_missing");
    ProfitabilityEvidence evidence;
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
}

} // namespace

EvidenceLoadError::EvidenceLoadError(EvidenceLoadErrorKind kind,
                                     std::string reason)
    : std::runtime_error(reason), kind_(kind), reason_(std::move(reason))
{
}

ArmEvidence LoadAuthoritativeArmEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    if (experimentId <= 0)
        throw std::invalid_argument("invalid_experiment_id");
    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start::date::text AS train_start,"
        "train_end::date::text AS train_end,"
        "infer_start::date::text AS infer_start,"
        "infer_end::date::text AS infer_end,status,phase,last_model_id,"
        "resume_model_id,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,"
        "schema_version,scheduler_version,binary_name,training_objective_id,"
        "training_objective_canonical,training_objective_hash "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.empty())
        throw EvidenceLoadError(EvidenceLoadErrorKind::ExperimentNotFound,
                                "experiment_" +
                                    std::to_string(experimentId) +
                                    "_not_found");
    if (rows.size() != 1) Ambiguous(experimentId, "identity_ambiguous");

    const pqxx::row row = rows.one_row();
    ArmEvidence arm;
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
    configuration.trainStart = TextOrEmpty(row, "train_start");
    configuration.trainEnd = TextOrEmpty(row, "train_end");
    configuration.inferenceStart = TextOrEmpty(row, "infer_start");
    configuration.inferenceEnd = TextOrEmpty(row, "infer_end");
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
    const auto parsedObjective = Objective::ParseSupportedCanonicalText(
        configuration.experimentObjective.canonical);
    if (parsedObjective.objectiveIdentifier !=
            row["training_objective_id"].as<std::string>() ||
        Objective::DeterministicHash(
            configuration.experimentObjective.canonical) !=
            configuration.experimentObjective.hash)
        ContractFailure(experimentId,
                        "experiment_objective_provenance_mismatch");
    configuration.runProvenance = {
        TextOrEmpty(row, "git_commit"),
        TextOrEmpty(row, "git_branch"),
        OptionalValue<bool>(row, "git_dirty"),
        TextOrEmpty(row, "build_config"),
        TextOrEmpty(row, "compiler_version"),
        TextOrEmpty(row, "schema_version"),
        TextOrEmpty(row, "scheduler_version"),
        TextOrEmpty(row, "binary_name")};
    arm.experimentStatus = row["status"].as<std::string>();
    arm.experimentPhase = row["phase"].as<std::string>();

    // last_model_id is only an authoritative final-model identity after the
    // workflow itself has reached completed/done.
    if (arm.experimentStatus != "completed" || arm.experimentPhase != "done")
        return arm;
    arm.finalModelId = OptionalValue<long long>(row, "last_model_id");
    if (!arm.finalModelId) return arm;

    LoadFinalModelConfiguration(transaction, arm);
    LoadMaterializedObjectives(transaction, arm);
    LoadFinalClassificationAndProfitability(transaction, arm);
    return arm;
}

} // namespace EA::PairedTrainingObjectiveEvaluation
