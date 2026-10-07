#include "ModelRuntimeValidation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "CanonicalSymbol.hpp"
#include "Donchian20Mode.hpp"
#include "ModelInputContract.hpp"

namespace EA::ModelRuntimeValidation
{
namespace
{
Diagnostics gDiagnostics;
bool LogSummary() { return gDiagnostics.summaryEnabled && gDiagnostics.summaryEnabled(); }
std::ostream& DiagnosticOut()
{
    if (!gDiagnostics.diagnosticOut) throw std::logic_error("model_runtime_validation_diagnostics_unconfigured");
    return gDiagnostics.diagnosticOut();
}
using PersistedModelRuntimeConfig::ParseTrainConfigMeta;
using PersistedModelRuntimeConfig::TrainConfigMeta;
constexpr size_t BaselineReturnFeatureCount = kModelReturnFeatureCount;
} // namespace

void SetDiagnostics(Diagnostics diagnostics) { gDiagnostics = diagnostics; }

const char* TargetTypeName(EA::LSTM::TargetType targetType)
{
    switch (targetType)
    {
        case EA::LSTM::TargetType::LogReturn: return "LogReturn";
        case EA::LSTM::TargetType::PercentReturn: return "PercentReturn";
        case EA::LSTM::TargetType::UpNeutralDownReturn: return "UpNeutralDownReturn";
    }
    return "Unknown";
}

const char* DirectionLabelRuleName()
{
    return "lookahead_high_low_first_hit";
}

int DirectionLabelRuleId()
{
    return DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
}

const char* TrainConfigMetaFieldMapping()
{
    return "schema_version,prediction_horizon,threshold_logret,window_size,label_rule_id,class_weight_down,class_weight_neutral,class_weight_up,num_layers,normalization_version,epochs_trained,core_lr_mult,head_weight_lr_mult,head_bias_lr_mult";
}

size_t RuntimeTensorFeatureWidth(const Tensor& tensor)
{
    return (tensor.begin() != tensor.end())
        ? static_cast<size_t>((*tensor.begin()).Shape()[1])
        : 0;
}

size_t RuntimeModelInputWidth(
    const Tensor& tensor,
    std::optional<std::size_t> persistedModelInputWidth)
{
    const size_t baseFeatureCount = RuntimeTensorFeatureWidth(tensor);
    if (persistedModelInputWidth.has_value())
        return EA::ResolveModelInputContract(*persistedModelInputWidth,
                                             baseFeatureCount).modelInputWidth;
    return baseFeatureCount + BaselineReturnFeatureCount;
}

std::string JoinStrings(const std::vector<std::string>& values, const char* separator)
{
    if (values.empty())
        return "none";

    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i) oss << separator;
        oss << values[i];
    }
    return oss.str();
}

ModelConfigValidationResult PrintModelConfigValidation(pqxx::work& w,
                                                       long long modelId,
                                                       EA::LSTM::TargetType requestedTargetType,
                                                       const Tensor& tensor,
                                                       const std::string& runtimeSymbol)
{
    ModelConfigValidationResult result;
    std::vector<std::string> persistedParams;
    std::vector<std::string> persistedMetadata;
    bool hasTargetMeta = false;
    bool hasModelMeta = false;
    bool hasTrainSymbolMeta = false;
    bool hasTrainConfigMeta = false;
    bool hasTrainConfigNumLayers = false;
    bool hasTrainConfigNormalizationVersion = false;
    bool hasTrainConfigCoreLrMult = false;
    bool hasTrainConfigHeadWeightLrMult = false;
    bool hasTrainConfigHeadBiasLrMult = false;

    DiagnosticOut() << "MODEL_TRAIN_CONFIG_META_FIELDS,"
                    << TrainConfigMetaFieldMapping()
                    << std::endl;

    try
    {
        pqxx::result params = w.exec_params(
            "SELECT DISTINCT param_name FROM matrix WHERE model_id = $1 ORDER BY param_name;",
            modelId);
        for (const auto& row : params)
        {
            const std::string paramName = row[0].as<std::string>();
            persistedParams.push_back(paramName);
            if (paramName == "target_meta")
            {
                hasTargetMeta = true;
                persistedMetadata.push_back("target_meta(type;scale;bias;use_zscore;mean;std)");
            }
            else if (paramName == "model_meta")
            {
                hasModelMeta = true;
                persistedMetadata.push_back("model_meta(schemaVersion;n_in;hidden_size)");
            }
            else if (paramName == "train_config_meta")
            {
                hasTrainConfigMeta = true;
                persistedMetadata.push_back("train_config_meta(schema_version;prediction_horizon;threshold_logret;window_size;label_rule_id;class_weight_down;class_weight_neutral;class_weight_up;num_layers;normalization_version;epochs_trained;core_lr_mult;head_weight_lr_mult;head_bias_lr_mult)");
            }
            else if (paramName == "train_symbol_meta")
            {
                hasTrainSymbolMeta = true;
                persistedMetadata.push_back("train_symbol_meta(ascii_table_name)");
            }
            else if (paramName == "train_range_meta")
            {
                persistedMetadata.push_back("train_range_meta(fromDate;toDate)");
            }
            else if (paramName == "donchian20_mode_meta")
            {
                persistedMetadata.push_back("donchian20_mode_meta(ascii_mode)");
            }
            else if (paramName == "optimizer_meta")
            {
                persistedMetadata.push_back("optimizer_meta(schema_version;optimizer_type;update_count;first_moment_buffer_count;second_moment_buffer_count)");
            }
            else if (paramName == "training_objective_canonical_meta")
            {
                persistedMetadata.push_back(
                    "training_objective_canonical_meta(ascii_canonical_contract)");
            }
            else if (paramName == "training_objective_hash_meta")
            {
                persistedMetadata.push_back(
                    "training_objective_hash_meta(ascii_fnv1a64_identity)");
            }
        }
    }
    catch (const std::exception& e)
    {
        DiagnosticOut() << "MODEL_METADATA_READ_FAIL"
                        << ",model_id=" << modelId
                        << ",error=" << e.what()
                        << std::endl;
    }

    DiagnosticOut() << "MODEL_METADATA_PERSISTED"
                    << ",model_id=" << modelId
                    << ",param_names=" << JoinStrings(persistedParams, ";")
                    << ",metadata=" << JoinStrings(persistedMetadata, ";")
                    << std::endl;

    bool targetMetaMatches = false;
    bool modelMetaMatches = false;
    bool trainSymbolMetaMatches = true;
    bool trainConfigMetaMatches = false;
    bool mismatch = false;
    std::optional<std::string> decodedModelSymbol;

    auto printMismatch = [&](const char* field, const auto& modelValue, const auto& runtimeValue)
    {
        mismatch = true;
        if (LogSummary())
            std::cout << "MODEL_CONFIG_MISMATCH"
                      << ",field=" << field
                      << ",model=" << modelValue
                      << ",runtime=" << runtimeValue
                      << std::endl;
    };

    auto compareIntField = [&](const char* field, double modelValue, long long runtimeValue) -> bool
    {
        const long long roundedModelValue = static_cast<long long>(std::llround(modelValue));
        if (roundedModelValue != runtimeValue)
        {
            printMismatch(field, roundedModelValue, runtimeValue);
            return false;
        }
        return true;
    };

    auto compareFloatField = [&](const char* field, double modelValue, double runtimeValue) -> bool
    {
        constexpr double kFloatCompareTolerance = 1e-7;
        if (std::fabs(modelValue - runtimeValue) > kFloatCompareTolerance)
        {
            printMismatch(field, modelValue, runtimeValue);
            return false;
        }
        return true;
    };

    auto compareStringField = [&](const char* field,
                                  const std::string& modelValue,
                                  const std::string& runtimeValue) -> bool
    {
        if (modelValue != runtimeValue)
        {
            printMismatch(field, modelValue, runtimeValue);
            return false;
        }
        return true;
    };

    try
    {
        const std::string modelSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
        decodedModelSymbol = modelSymbol;
        PrintDatabaseModelSymbol(modelId, modelSymbol);
        if (LogSummary())
            std::cout << "MODEL_TRAIN_SYMBOL_META"
                      << ",model_id=" << modelId
                      << ",symbol=" << modelSymbol
                      << std::endl;
        if (result.trainConfigMeta.has_value())
            result.trainConfigMeta->symbol = modelSymbol;
        trainSymbolMetaMatches = compareStringField("symbol",
                                                    EA::CanonicalSymbol::Normalize(modelSymbol),
                                                    EA::CanonicalSymbol::Normalize(runtimeSymbol));
    }
    catch (const std::exception& e)
    {
        if (hasTrainSymbolMeta)
        {
            printMismatch("train_symbol_meta", e.what(), runtimeSymbol);
            trainSymbolMetaMatches = false;
        }
        else
        {
            PrintMissingModelSymbol(modelId);
            DiagnosticOut() << "MODEL_CONFIG_WARN"
                            << ",model_id=" << modelId
                            << ",field=symbol"
                            << ",model=missing_legacy_train_symbol_meta"
                            << ",runtime=" << runtimeSymbol
                            << std::endl;
        }
    }
    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "target_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "target_meta");
        if (dims.n_rows != 1 || dims.n_cols != 6 || vals.size() != 6)
        {
            printMismatch("target_meta_shape", std::to_string(dims.n_rows) + "x" + std::to_string(dims.n_cols), "1x6");
        }
        else
        {
            bool sectionMatches = true;
            const auto modelTargetType = static_cast<EA::LSTM::TargetType>(static_cast<int>(vals[0]));
            if (static_cast<int>(modelTargetType) != static_cast<int>(requestedTargetType))
            {
                printMismatch("target_type", TargetTypeName(modelTargetType), TargetTypeName(requestedTargetType));
                sectionMatches = false;
            }
            targetMetaMatches = sectionMatches;
        }
    }
    catch (const std::exception&)
    {
        // Older models may not have target_meta.
    }

    try
    {
        const auto modelMeta =
            DBIO::PgModelIO::loadRequiredModelMeta(w, modelId);
        const int modelInputWidth = static_cast<int>(modelMeta.inputWidth);
        const int modelHiddenSize = static_cast<int>(modelMeta.hiddenSize);
        const int runtimeInputWidth = static_cast<int>(RuntimeModelInputWidth(
            tensor, modelMeta.inputWidth));
        const int runtimeHiddenSize = static_cast<int>(hidden_size);
        bool sectionMatches = true;
        if (modelMeta.schemaVersion != 1)
        {
            printMismatch("model_meta_schema_version", modelMeta.schemaVersion, 1);
            sectionMatches = false;
        }
        if (modelInputWidth != runtimeInputWidth)
        {
            printMismatch("feature_count", modelInputWidth, runtimeInputWidth);
            sectionMatches = false;
        }
        if (modelHiddenSize != runtimeHiddenSize)
        {
            printMismatch("hidden_size", modelHiddenSize, runtimeHiddenSize);
            sectionMatches = false;
        }
        modelMetaMatches = sectionMatches;
    }
    catch (const std::exception& error)
    {
        if (hasModelMeta)
            printMismatch("model_meta", error.what(), "supported persisted model");
    }

    try
    {
        const Donchian20Mode modelMode =
            DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
        if (modelMode != tensor.GetDonchian20Mode())
            printMismatch("donchian20_mode",
                          Donchian20ModeText(modelMode),
                          Donchian20ModeText(tensor.GetDonchian20Mode()));
    }
    catch (const std::exception& error)
    {
        printMismatch("donchian20_mode", error.what(),
                      Donchian20ModeText(tensor.GetDonchian20Mode()));
    }

    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
        if (dims.n_rows != 1 ||
            dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaFieldCount ||
            vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaFieldCount))
        {
            printMismatch("train_config_meta_shape",
                          std::to_string(dims.n_rows) + "x" + std::to_string(dims.n_cols),
                          "1x>=8");
        }
        else
        {
            TrainConfigMeta trainConfigMeta;
            trainConfigMeta.schemaVersion = static_cast<int>(std::llround(vals[0]));
            trainConfigMeta.predictionHorizon = static_cast<size_t>(std::llround(vals[1]));
            trainConfigMeta.thresholdLogret = static_cast<float>(vals[2]);
            trainConfigMeta.windowSize = static_cast<size_t>(std::llround(vals[3]));
            trainConfigMeta.labelRuleId = static_cast<int>(std::llround(vals[4]));
            trainConfigMeta.classWeightDown = static_cast<float>(vals[5]);
            trainConfigMeta.classWeightNeutral = static_cast<float>(vals[6]);
            trainConfigMeta.classWeightUp = static_cast<float>(vals[7]);
            if (vals.size() >= 9)
            {
                trainConfigMeta.numLayers = static_cast<size_t>(std::llround(vals[8]));
                hasTrainConfigNumLayers = true;
            }
            if (vals.size() >= 10)
            {
                trainConfigMeta.normalizationVersion = static_cast<int>(std::llround(vals[9]));
                hasTrainConfigNormalizationVersion = true;
            }
            if (vals.size() >= 11)
            {
                trainConfigMeta.epochsTrained = static_cast<size_t>(std::llround(vals[10]));
            }
            if (vals.size() >= 12)
            {
                trainConfigMeta.coreLrMult = static_cast<float>(vals[11]);
                hasTrainConfigCoreLrMult = true;
            }
            if (vals.size() >= 13)
            {
                trainConfigMeta.headWeightLrMult = static_cast<float>(vals[12]);
                hasTrainConfigHeadWeightLrMult = true;
            }
            if (vals.size() >= 14)
            {
                trainConfigMeta.headBiasLrMult = static_cast<float>(vals[13]);
                hasTrainConfigHeadBiasLrMult = true;
            }
            if (decodedModelSymbol.has_value())
                trainConfigMeta.symbol = *decodedModelSymbol;
            result.trainConfigMeta = trainConfigMeta;

            if (LogSummary())
            {
                std::cout << "MODEL_TRAIN_CONFIG_META"
                          << ",model_id=" << modelId
                          << ",schema_version=" << trainConfigMeta.schemaVersion
                          << ",prediction_horizon=" << trainConfigMeta.predictionHorizon
                          << ",threshold_logret=" << trainConfigMeta.thresholdLogret
                          << ",window_size=" << trainConfigMeta.windowSize
                          << ",label_rule_id=" << trainConfigMeta.labelRuleId
                          << ",class_weight_down=" << trainConfigMeta.classWeightDown
                          << ",class_weight_neutral=" << trainConfigMeta.classWeightNeutral
                          << ",class_weight_up=" << trainConfigMeta.classWeightUp
                          << ",num_layers=" << trainConfigMeta.numLayers
                          << ",normalization_version=" << trainConfigMeta.normalizationVersion
                          << ",epochs_trained=";
                if (trainConfigMeta.epochsTrained.has_value())
                    std::cout << *trainConfigMeta.epochsTrained;
                else
                    std::cout << "missing";
                std::cout << ",core_lr_mult=";
                if (trainConfigMeta.coreLrMult.has_value())
                    std::cout << *trainConfigMeta.coreLrMult;
                else
                    std::cout << "missing";
                std::cout << ",head_weight_lr_mult=";
                if (trainConfigMeta.headWeightLrMult.has_value())
                    std::cout << *trainConfigMeta.headWeightLrMult;
                else
                    std::cout << "missing";
                std::cout << ",head_bias_lr_mult=";
                if (trainConfigMeta.headBiasLrMult.has_value())
                    std::cout << *trainConfigMeta.headBiasLrMult;
                else
                    std::cout << "missing";
                std::cout << std::endl;
            }

            bool sectionMatches = true;
            sectionMatches = compareIntField("schema_version",
                                             vals[0],
                                             DBIO::PgModelIO::kTrainConfigMetaSchemaVersion) && sectionMatches;
            sectionMatches = compareIntField("prediction_horizon",
                                             vals[1],
                                             prediction_horizon) && sectionMatches;
            sectionMatches = compareFloatField("threshold_logret",
                                               vals[2],
                                               c_next_threshold) && sectionMatches;
            sectionMatches = compareIntField("window_size",
                                             vals[3],
                                             window_size) && sectionMatches;
            sectionMatches = compareIntField("label_rule_id",
                                             vals[4],
                                             DirectionLabelRuleId()) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_down",
                                               vals[5],
                                               kClassWeightDown) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_neutral",
                                               vals[6],
                                               kClassWeightNeutral) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_up",
                                               vals[7],
                                               kClassWeightUp) && sectionMatches;
            if (vals.size() >= 9)
                sectionMatches = compareIntField("num_layers",
                                                 vals[8],
                                                 num_layers) && sectionMatches;
            if (vals.size() >= 10)
                sectionMatches = compareIntField("normalization_version",
                                                 vals[9],
                                                 normalization_version) && sectionMatches;
            if (vals.size() >= 12)
                sectionMatches = compareFloatField("core_lr_mult",
                                                   vals[11],
                                                   EA::LSTM::CoreLrMultForTarget(requestedTargetType)) && sectionMatches;
            if (vals.size() >= 13)
                sectionMatches = compareFloatField("head_weight_lr_mult",
                                                   vals[12],
                                                   head_weight_lr_mult) && sectionMatches;
            if (vals.size() >= 14)
                sectionMatches = compareFloatField("head_bias_lr_mult",
                                                   vals[13],
                                                   head_bias_lr_mult) && sectionMatches;
            trainConfigMetaMatches = sectionMatches;
        }
    }
    catch (const std::exception&)
    {
        // Older models may not have train_config_meta.
    }

    std::vector<std::string> missingMinimum;

    if (!hasTrainConfigMeta)
    {
        missingMinimum.push_back("prediction_horizon");
        missingMinimum.push_back("threshold_logret");
        missingMinimum.push_back("window_size");
        missingMinimum.push_back("label_rule");
        missingMinimum.push_back("class_weight_down");
        missingMinimum.push_back("class_weight_neutral");
        missingMinimum.push_back("class_weight_up");
        missingMinimum.push_back("num_layers");
        missingMinimum.push_back("normalization_version");
        missingMinimum.push_back("core_lr_mult");
        missingMinimum.push_back("head_weight_lr_mult");
        missingMinimum.push_back("head_bias_lr_mult");
    }
    else
    {
        if (!hasTrainConfigNumLayers)
            missingMinimum.push_back("num_layers");
        if (!hasTrainConfigNormalizationVersion)
            missingMinimum.push_back("normalization_version");
        if (!hasTrainConfigCoreLrMult)
            missingMinimum.push_back("core_lr_mult");
        if (!hasTrainConfigHeadWeightLrMult)
            missingMinimum.push_back("head_weight_lr_mult");
        if (!hasTrainConfigHeadBiasLrMult)
            missingMinimum.push_back("head_bias_lr_mult");
    }
    if (!hasTargetMeta)
        missingMinimum.push_back("target_type");
    if (!hasTrainSymbolMeta)
        missingMinimum.push_back("symbol");
    if (!hasModelMeta)
    {
        missingMinimum.push_back("feature_count");
        missingMinimum.push_back("hidden_size");
    }

    result.hasMismatch = mismatch;
    result.metadataGap = !missingMinimum.empty();
    result.configMatch =
        targetMetaMatches &&
        modelMetaMatches &&
        trainSymbolMetaMatches &&
        trainConfigMetaMatches &&
        !mismatch &&
        missingMinimum.empty();

    if (result.configMatch)
    {
        if (LogSummary())
            std::cout << "MODEL_CONFIG_MATCH=1" << std::endl;
    }

    if (!missingMinimum.empty())
    {
        if (LogSummary())
            std::cout << "MODEL_CONFIG_METADATA_GAP"
                      << ",model_id=" << modelId
                      << ",missing=" << JoinStrings(missingMinimum, ";")
                      << ",recommend_minimum_additions=" << JoinStrings(missingMinimum, ";")
                      << std::endl;
    }

    return result;
}

ModelConfigValidationResult PrintMaterializedModelConfigValidation(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted,
    EA::LSTM::TargetType requestedTargetType,
    const Tensor& tensor,
    const std::string& runtimeSymbol)
{
    ModelConfigValidationResult result;
    const long long modelId = persisted.identity.modelId;
    std::vector<std::string> names{"bias", "model_meta", "param"};
    if (persisted.targetMeta) names.push_back("target_meta");
    if (persisted.trainConfigMeta) names.push_back("train_config_meta");
    if (persisted.trainSymbol) names.push_back("train_symbol_meta");
    if (persisted.trainRange) names.push_back("train_range_meta");
    if (persisted.optimizerMeta) names.push_back("optimizer_meta");
    if (persisted.returnHeadWeight) names.push_back("returnHeadWeight");
    if (persisted.returnHeadBias) names.push_back("returnHeadBias");
    if (persisted.returnHeadDirWeight) names.push_back("returnHeadDirWeight");
    if (persisted.returnHeadDirBias) names.push_back("returnHeadDirBias");
    if (persisted.trainingObjectiveCanonical)
        names.push_back("training_objective_canonical_meta");
    if (persisted.trainingObjectiveHash)
        names.push_back("training_objective_hash_meta");
    std::sort(names.begin(), names.end());
    DiagnosticOut() << "MODEL_TRAIN_CONFIG_META_FIELDS,"
                    << TrainConfigMetaFieldMapping() << std::endl;
    DiagnosticOut() << "MODEL_METADATA_PERSISTED"
                    << ",model_id=" << modelId
                    << ",param_names=" << JoinStrings(names, ";")
                    << ",metadata=detached_materialization_snapshot"
                    << std::endl;

    bool mismatch = false;
    const auto report = [&](const char* field, const auto& modelValue,
                            const auto& runtimeValue)
    {
        mismatch = true;
        if (LogSummary())
            std::cout << "MODEL_CONFIG_MISMATCH,field=" << field
                      << ",model=" << modelValue << ",runtime="
                      << runtimeValue << std::endl;
    };
    bool symbolMatches = true;
    if (persisted.trainSymbol)
    {
        PrintDatabaseModelSymbol(modelId, *persisted.trainSymbol);
        symbolMatches = EA::CanonicalSymbol::Normalize(*persisted.trainSymbol) ==
            EA::CanonicalSymbol::Normalize(runtimeSymbol);
        if (!symbolMatches) report("symbol", *persisted.trainSymbol, runtimeSymbol);
    }
    else
        PrintMissingModelSymbol(modelId);

    bool targetMatches = false;
    if (persisted.targetMeta)
    {
        targetMatches = persisted.targetMeta->targetType == requestedTargetType;
        if (!targetMatches)
            report("target_type", TargetTypeName(persisted.targetMeta->targetType),
                   TargetTypeName(requestedTargetType));
    }

    bool modelMatches = persisted.modelMeta.schemaVersion == 1;
    const int runtimeInput = static_cast<int>(RuntimeModelInputWidth(
        tensor, persisted.modelMeta.inputWidth));
    if (static_cast<int>(persisted.modelMeta.inputWidth) != runtimeInput)
    {
        report("feature_count", persisted.modelMeta.inputWidth, runtimeInput);
        modelMatches = false;
    }
    if (persisted.modelMeta.hiddenSize != static_cast<std::size_t>(hidden_size))
    {
        report("hidden_size", persisted.modelMeta.hiddenSize, hidden_size);
        modelMatches = false;
    }
    if (persisted.donchian20Mode != tensor.GetDonchian20Mode())
        report("donchian20_mode", Donchian20ModeText(persisted.donchian20Mode),
               Donchian20ModeText(tensor.GetDonchian20Mode()));

    bool trainMatches = false;
    if (persisted.trainConfigMeta)
    {
        try
        {
            TrainConfigMeta meta = ParseTrainConfigMeta(*persisted.trainConfigMeta,
                false, "train_config_meta missing required 1x8 inference fields");
            if (persisted.trainSymbol) meta.symbol = *persisted.trainSymbol;
            result.trainConfigMeta = meta;
            const auto& values = persisted.trainConfigMeta->values;
            const auto sameInt = [](double a, long long b) {
                return std::llround(a) == b;
            };
            const auto sameFloat = [](double a, double b) {
                return std::fabs(a - b) <= 1e-7;
            };
            trainMatches = sameInt(values[0], DBIO::PgModelIO::kTrainConfigMetaSchemaVersion) &&
                sameInt(values[1], prediction_horizon) &&
                sameFloat(values[2], c_next_threshold) &&
                sameInt(values[3], window_size) &&
                sameInt(values[4], DirectionLabelRuleId()) &&
                sameFloat(values[5], kClassWeightDown) &&
                sameFloat(values[6], kClassWeightNeutral) &&
                sameFloat(values[7], kClassWeightUp);
            if (values.size() >= 9) trainMatches = trainMatches && sameInt(values[8], num_layers);
            if (values.size() >= 10) trainMatches = trainMatches && sameInt(values[9], normalization_version);
            if (values.size() >= 12) trainMatches = trainMatches && sameFloat(values[11], EA::LSTM::CoreLrMultForTarget(requestedTargetType));
            if (values.size() >= 13) trainMatches = trainMatches && sameFloat(values[12], head_weight_lr_mult);
            if (values.size() >= 14) trainMatches = trainMatches && sameFloat(values[13], head_bias_lr_mult);
            if (!trainMatches) report("train_config_meta", "persisted", "runtime");
        }
        catch (const std::exception& error)
        {
            report("train_config_meta", error.what(), "supported persisted model");
        }
    }
    std::vector<std::string> missing;
    if (!persisted.targetMeta) missing.push_back("target_type");
    if (!persisted.trainSymbol) missing.push_back("symbol");
    if (!persisted.trainConfigMeta) missing.push_back("train_config_meta");
    result.hasMismatch = mismatch;
    result.metadataGap = !missing.empty();
    result.configMatch = targetMatches && modelMatches && symbolMatches &&
        trainMatches && !mismatch && missing.empty();
    if (result.configMatch && LogSummary()) std::cout << "MODEL_CONFIG_MATCH=1" << std::endl;
    if (!missing.empty() && LogSummary())
        std::cout << "MODEL_CONFIG_METADATA_GAP,model_id=" << modelId
                  << ",missing=" << JoinStrings(missing, ";")
                  << ",recommend_minimum_additions=" << JoinStrings(missing, ";")
                  << std::endl;
    return result;
}

void PrintDatabaseModelSymbol(long long modelId, const std::string& symbol)
{
    if (!LogSummary()) return;
    std::cout << "MODEL_SYMBOL" << ",source=database" << ",model_id="
              << modelId << ",symbol=" << symbol << std::endl;
}

void PrintLegacyModelSymbol(long long modelId, const std::string& symbol)
{
    if (!LogSummary()) return;
    std::cout << "MODEL_SYMBOL" << ",source=legacy" << ",model_id="
              << modelId << ",symbol=" << symbol
              << ",warning=missing_metadata" << std::endl;
}

void PrintMissingModelSymbol(long long modelId)
{
    if (!LogSummary()) return;
    std::cout << "MODEL_SYMBOL_MISSING" << ",model_id=" << modelId
              << std::endl;
}

void ValidateRuntimeSymbolMatchesModel(
    const std::optional<std::string>& runtimeSymbol,
    const std::string& modelSymbol)
{
    if (!runtimeSymbol.has_value()) return;
    const std::string canonicalRuntimeSymbol =
        EA::CanonicalSymbol::Normalize(*runtimeSymbol);
    const std::string canonicalModelSymbol =
        EA::CanonicalSymbol::Normalize(modelSymbol);
    if (canonicalRuntimeSymbol != canonicalModelSymbol)
    {
        std::cerr << "MODEL_SYMBOL_MISMATCH" << ",runtime="
                  << canonicalRuntimeSymbol << ",model=" << canonicalModelSymbol
                  << std::endl;
        throw std::runtime_error("MODEL_SYMBOL_MISMATCH");
    }
    DiagnosticOut() << "INFERENCE_CLI_ARG_REDUNDANT"
                    << ",param=--symbol" << ",value=" << canonicalRuntimeSymbol
                    << ",source=persisted_model" << std::endl;
}

std::optional<std::string> ResolveLegacySymbolFromModelName(const std::string& modelName,
                                                            const std::vector<std::string>& availableSymbols)
{
    std::optional<std::string> bestMatch;
    for (const auto& symbol : availableSymbols)
    {
        const bool exact = (modelName == symbol);
        const bool prefixWithDash = modelName.rfind(symbol + "-", 0) == 0;
        const bool prefixWithUnderscore = modelName.rfind(symbol + "_", 0) == 0;
        if (exact || prefixWithDash || prefixWithUnderscore)
        {
            if (!bestMatch.has_value() || symbol.size() > bestMatch->size())
                bestMatch = symbol;
        }
    }
    return bestMatch;
}

std::string ResolveLegacyModelSymbol(long long modelId,
                                     const std::string& modelName,
                                     const std::optional<std::string>& runtimeSymbol,
                                     const std::vector<std::string>& availableSymbols)
{
    PrintMissingModelSymbol(modelId);
    const auto modelNameSymbol = ResolveLegacySymbolFromModelName(modelName, availableSymbols);
    if (modelNameSymbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*modelNameSymbol);
        if (runtimeSymbol.has_value())
            ValidateRuntimeSymbolMatchesModel(runtimeSymbol, legacySymbol);
        PrintLegacyModelSymbol(modelId, legacySymbol);
        return legacySymbol;
    }

    if (runtimeSymbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*runtimeSymbol);
        PrintLegacyModelSymbol(modelId, legacySymbol);
        return legacySymbol;
    }

    throw std::runtime_error("unable to resolve legacy model symbol; train_symbol_meta is missing");
}

void ValidateLoadedModelSymbolForSelectedTable(pqxx::work& w,
                                               long long modelId,
                                               const std::string& selectedSymbol)
{
    std::optional<std::string> databaseSymbol;
    try
    {
        databaseSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    }
    catch (const std::exception&)
    {
    }

    if (databaseSymbol.has_value())
    {
        PrintDatabaseModelSymbol(modelId, *databaseSymbol);
        ValidateRuntimeSymbolMatchesModel(EA::CanonicalSymbol::Normalize(selectedSymbol), *databaseSymbol);
        return;
    }

    PrintMissingModelSymbol(modelId);
    PrintLegacyModelSymbol(modelId, EA::CanonicalSymbol::Normalize(selectedSymbol));
}

} // namespace EA::ModelRuntimeValidation
