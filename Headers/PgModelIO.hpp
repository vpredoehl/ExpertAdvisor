// PgModelIO.hpp - header-only helpers to persist/load EA::LSTM parameters with PostgreSQL (libpqxx)
#pragma once

#include <pqxx/pqxx>
#include <MetaNN/meta_nn.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <optional>
#include <utility>

#include "LSTM.hpp"
#include "CanonicalSymbol.hpp"
#include "Donchian20Mode.hpp"
#include "ModelInputContract.hpp"


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace DBIO
{

// Convenience alias
template <typename T>
using MatGPU = MetaNN::Matrix<T, MetaNN::DeviceTags::Metal>;

// Flatten to row-major vector<double>
template <typename T>
inline std::vector<double> flattenRowMajor(const MatGPU<T>& m)
{
    const size_t rows = m.Shape()[0];
    const size_t cols = m.Shape()[1];
    std::vector<double> out;
    out.resize(rows * cols);

    auto eval = MetaNN::Evaluate(m);
    auto low  = MetaNN::LowerAccess(eval);
    const T* src = low.RawMemory();
    for (size_t i = 0; i < rows * cols; ++i)    out[i] = static_cast<double>(src[i]);
    return out;
}

// Reconstruct from row-major vector<double> into Matrix<T>
template <typename T = float>
inline MatGPU<T> fromFlatRowMajor(const std::vector<double>& vals, size_t rows, size_t cols)
{
    if (vals.size() != rows * cols)
        throw std::runtime_error("fromFlatRowMajor: size mismatch with rows*cols");

    MatGPU<T> m(rows, cols);
    auto low = MetaNN::LowerAccess(m);
    T* p = low.MutableRawMemory();
    const size_t N = rows * cols;
    for (size_t i = 0; i < N; ++i) p[i] = static_cast<T>(vals[i]);
    return m;
}

// Build a PostgreSQL array literal like: "{1.0,2.0,3.0}"
inline std::string toPgArrayLiteral(const std::vector<double>& vals)
{
    std::ostringstream oss;
    oss.precision(17);
    oss << "{";
    for (size_t i = 0; i < vals.size(); ++i)
    {
        if (i) oss << ",";
        oss << vals[i];
    }
    oss << "}";
    return oss.str();
}

// Save a MetaNN matrix into the `matrix` table via replace_parameter
class PgModelIO {
public:

    struct PersistedModelMeta
    {
        int schemaVersion = 0;
        std::size_t inputWidth = 0;
        std::size_t hiddenSize = 0;
    };

    struct PersistedTargetMeta
    {
        EA::LSTM::TargetType targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
        float targetScale = 1.0f;
        float targetBias = 0.0f;
        bool targetUseZScore = false;
        float targetMean = 0.0f;
        float targetStd = 1.0f;
    };
    static constexpr int kTrainConfigMetaSchemaVersion = 1;
    static constexpr int kLookaheadHighLowFirstHitLabelRuleId = 1;
    static constexpr int kTrainConfigMetaFieldCount = 8;
    static constexpr int kTrainConfigMetaExtendedFieldCount = 14;
    static constexpr int kOptimizerMetaSchemaVersion = 1;
    static constexpr int kOptimizerTypeSgd = 1;
    static constexpr int kOptimizerMetaFieldCount = 5;

    // Create a new row in `model` table and return model_id
    static long long createModel(pqxx::work& w,
                                 const std::string& name,
                                 const std::string& comment,
                                 std::optional<long long> experimentId = std::nullopt,
                                 std::optional<long long> parentModelId = std::nullopt)
    {
        pqxx::result r;
        if (experimentId.has_value() && parentModelId.has_value())
        {
            r = w.exec_params(
                "INSERT INTO model (name, comment, experiment_id, parent_model_id) VALUES ($1, $2, $3, $4) RETURNING model_id;",
                name,
                comment,
                *experimentId,
                *parentModelId);
        }
        else if (experimentId.has_value())
        {
            r = w.exec_params(
                "INSERT INTO model (name, comment, experiment_id) VALUES ($1, $2, $3) RETURNING model_id;",
                name,
                comment,
                *experimentId);
        }
        else if (parentModelId.has_value())
        {
            r = w.exec_params(
                "INSERT INTO model (name, comment, parent_model_id) VALUES ($1, $2, $3) RETURNING model_id;",
                name,
                comment,
                *parentModelId);
        }
        else
        {
            r = w.exec_params(
                "INSERT INTO model (name, comment) VALUES ($1, $2) RETURNING model_id;",
                name, comment);
        }
        if (r.empty()) throw std::runtime_error("createModel failed to return model_id");
        return r[0][0].as<long long>();
    }

    template <typename T>
    static void saveParameter(pqxx::work& w,
                              long long modelId,
                              const std::string& paramName,
                              const MatGPU<T>& mat)
    {
        const int n_rows = static_cast<int>(mat.Shape()[0]);
        const int n_cols = static_cast<int>(mat.Shape()[1]);
        auto flat = flattenRowMajor(mat);
        auto arr  = toPgArrayLiteral(flat);
        w.exec_params(
            "CALL replace_parameter($1, $2, $3, $4, $5::double precision[]);",
            modelId, paramName, n_rows, n_cols, arr
        );
    }

    // Save all LSTM learnable parameters
    static void saveAll(pqxx::work& w,
                        long long modelId,
                        const EA::LSTM& lstm,
                        const std::string& symbol = {},
                        const std::string& fromDate = {},
                        const std::string& toDate = {},
                        Donchian20Mode donchian20Mode = kDefaultDonchian20Mode)
    {
        saveParameter(w, modelId, "param",            lstm.param);
        saveParameter(w, modelId, "bias",             lstm.bias);
        saveParameter(w, modelId, "returnHeadWeight", lstm.returnHeadWeight);
        saveParameter(w, modelId, "returnHeadBias",   lstm.returnHeadBias);
        saveParameter(w, modelId, "returnHeadDirWeight", lstm.returnHeadDirWeight);
        saveParameter(w, modelId, "returnHeadDirBias",   lstm.returnHeadDirBias);
        saveTargetMeta(w, modelId, lstm);
        saveModelMeta(w, modelId, lstm);
        saveTrainConfigMeta(w, modelId, lstm);
        saveOptimizerMeta(w, modelId, lstm);
        saveDonchian20ModeMeta(w, modelId, donchian20Mode);
        if (symbol.empty())
            throw std::runtime_error("saveAll requires a canonical training symbol");
        saveTrainSymbolMeta(w, modelId, symbol);
        if (!fromDate.empty() || !toDate.empty())
            saveTrainRangeMeta(w, modelId, fromDate, toDate);
    }

    static void saveDonchian20ModeMeta(pqxx::work& w,
                                       long long modelId,
                                       Donchian20Mode mode)
    {
        saveAsciiMeta(w, modelId, "donchian20_mode_meta", Donchian20ModeText(mode));
    }

    static Donchian20Mode loadDonchian20ModeMeta(pqxx::work& w,
                                                 long long modelId)
    {
        try
        {
            return ParseDonchian20Mode(
                decodeAsciiMeta(w, modelId, "donchian20_mode_meta"));
        }
        catch (const std::exception& error)
        {
            if (std::string{error.what()}.find("No entries for parameter") !=
                std::string::npos)
                return kDefaultDonchian20Mode;
            throw;
        }
    }

    // Save target mapping metadata as a 1x6 matrix in order:
    // [type(int), scale, bias, useZ(0/1), mean, std]
    static void saveTargetMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        MatGPU<float> meta(1, 6);
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            p[0] = static_cast<float>(static_cast<int>(lstm.targetType));
            p[1] = lstm.targetScale;
            p[2] = lstm.targetBias;
            p[3] = lstm.targetUseZScore ? 1.0f : 0.0f;
            p[4] = lstm.targetMean;
            p[5] = lstm.targetStd;
        }
        saveParameter(w, modelId, "target_meta", meta);
    }

    // Save minimal model metadata as a 1x3 matrix: [schemaVersion, n_in, hidden_size]
    static void saveModelMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        // Derive hidden_size and n_in from param shape to avoid accessing private members
        const size_t rows = lstm.param.Shape()[0];
        const size_t cols = lstm.param.Shape()[1];
        const size_t hidden_size = cols / 4;
        const size_t n_in = rows - hidden_size;

        MatGPU<float> meta(1, 3);
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            p[0] = 1.0f; // schemaVersion
            p[1] = static_cast<float>(n_in);
            p[2] = static_cast<float>(hidden_size);
        }
        saveParameter(w, modelId, "model_meta", meta);
    }

    // Save training/evaluation compatibility metadata in order:
    // [schema_version, prediction_horizon, threshold_logret, window_size,
    //  label_rule_id, class_weight_down, class_weight_neutral, class_weight_up,
    //  num_layers, normalization_version, epochs_trained, core_lr_mult,
    //  head_weight_lr_mult, head_bias_lr_mult]
    static void saveTrainConfigMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        MatGPU<float> meta(1, kTrainConfigMetaExtendedFieldCount);
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            p[0] = static_cast<float>(kTrainConfigMetaSchemaVersion);
            p[1] = static_cast<float>(prediction_horizon);
            p[2] = static_cast<float>(c_next_threshold);
            p[3] = static_cast<float>(window_size);
            p[4] = static_cast<float>(kLookaheadHighLowFirstHitLabelRuleId);
            p[5] = kClassWeightDown;
            p[6] = kClassWeightNeutral;
            p[7] = kClassWeightUp;
            p[8] = static_cast<float>(num_layers);
            p[9] = static_cast<float>(normalization_version);
            p[10] = static_cast<float>(lstm.completedEpochs > 0 ? lstm.completedEpochs : static_cast<size_t>(epoch_count));
            p[11] = EA::LSTM::CoreLrMultForTarget(lstm.targetType);
            p[12] = head_weight_lr_mult;
            p[13] = head_bias_lr_mult;
        }
        saveParameter(w, modelId, "train_config_meta", meta);
    }

    // Current optimizer is SGD, so there are no moment/variance buffers.
    // Layout: [schema_version, optimizer_type, update_count, first_moment_buffer_count, second_moment_buffer_count]
    static void saveOptimizerMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        MatGPU<float> meta(1, kOptimizerMetaFieldCount);
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            p[0] = static_cast<float>(kOptimizerMetaSchemaVersion);
            p[1] = static_cast<float>(kOptimizerTypeSgd);
            p[2] = static_cast<float>(lstm.optimizerUpdateCount);
            p[3] = 0.0f;
            p[4] = 0.0f;
        }
        saveParameter(w, modelId, "optimizer_meta", meta);
    }

    // Save the source symbol/table as ASCII codepoints in a 1xN matrix.
    // This keeps metadata in the existing matrix persistence mechanism.
    static void saveTrainSymbolMeta(pqxx::work& w, long long modelId, const std::string& symbol)
    {
        const std::string canonicalSymbol = EA::CanonicalSymbol::Normalize(symbol);
        MatGPU<float> meta(1, canonicalSymbol.size());
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            for (size_t i = 0; i < canonicalSymbol.size(); ++i)
                p[i] = static_cast<float>(static_cast<unsigned char>(canonicalSymbol[i]));
        }
        saveParameter(w, modelId, "train_symbol_meta", meta);
    }

    static std::string decodeTrainSymbolMeta(pqxx::work& w, long long modelId)
    {
        auto dims = loadParameterDims(w, modelId, "train_symbol_meta");
        auto vals = loadParameterValues(w, modelId, "train_symbol_meta");
        if (dims.n_rows != 1 || dims.n_cols <= 0 ||
            vals.size() != static_cast<size_t>(dims.n_cols))
            throw std::runtime_error("train_symbol_meta has invalid shape");

        std::string symbol;
        symbol.reserve(vals.size());
        for (double value : vals)
        {
            const long long code = static_cast<long long>(std::llround(value));
            if (code <= 0 || code > 255)
                throw std::runtime_error("train_symbol_meta contains invalid character code");
            symbol.push_back(static_cast<char>(code));
        }
        return EA::CanonicalSymbol::Normalize(symbol);
    }

    static void saveTrainRangeMeta(pqxx::work& w,
                                   long long modelId,
                                   const std::string& fromDate,
                                   const std::string& toDate)
    {
        saveAsciiMeta(w, modelId, "train_range_meta", fromDate + "|" + toDate);
    }

    static std::pair<std::string, std::string> decodeTrainRangeMeta(pqxx::work& w, long long modelId)
    {
        const std::string encoded = decodeAsciiMeta(w, modelId, "train_range_meta");
        const size_t sep = encoded.find('|');
        if (sep == std::string::npos)
            throw std::runtime_error("train_range_meta missing separator");
        return { encoded.substr(0, sep), encoded.substr(sep + 1) };
    }

    static void loadOptimizerMeta(pqxx::work& w, long long modelId, EA::LSTM& lstm)
    {
        auto dims = loadParameterDims(w, modelId, "optimizer_meta");
        auto vals = loadParameterValues(w, modelId, "optimizer_meta");
        if (dims.n_rows != 1 ||
            dims.n_cols < kOptimizerMetaFieldCount ||
            vals.size() < static_cast<size_t>(kOptimizerMetaFieldCount))
            throw std::runtime_error("optimizer_meta has invalid shape");

        const int schemaVersion = static_cast<int>(std::llround(vals[0]));
        const int optimizerType = static_cast<int>(std::llround(vals[1]));
        const int firstMomentBuffers = static_cast<int>(std::llround(vals[3]));
        const int secondMomentBuffers = static_cast<int>(std::llround(vals[4]));

        if (schemaVersion != kOptimizerMetaSchemaVersion)
            throw std::runtime_error("optimizer_meta unsupported schema_version");
        if (optimizerType != kOptimizerTypeSgd)
            throw std::runtime_error("optimizer_meta optimizer_type is not supported by this binary");
        if (firstMomentBuffers != 0 || secondMomentBuffers != 0)
            throw std::runtime_error("optimizer_meta declares moment buffers unsupported by current SGD optimizer");

        lstm.optimizerUpdateCount = static_cast<size_t>(std::llround(vals[2]));
    }

private:
    static void saveAsciiMeta(pqxx::work& w,
                              long long modelId,
                              const std::string& paramName,
                              const std::string& value)
    {
        MatGPU<float> meta(1, value.size());
        {
            auto low = MetaNN::LowerAccess(meta);
            float* p = low.MutableRawMemory();
            for (size_t i = 0; i < value.size(); ++i)
                p[i] = static_cast<float>(static_cast<unsigned char>(value[i]));
        }
        saveParameter(w, modelId, paramName, meta);
    }

    static std::string decodeAsciiMeta(pqxx::work& w,
                                       long long modelId,
                                       const std::string& paramName)
    {
        auto dims = loadParameterDims(w, modelId, paramName);
        auto vals = loadParameterValues(w, modelId, paramName);
        if (dims.n_rows != 1 || dims.n_cols <= 0 ||
            vals.size() != static_cast<size_t>(dims.n_cols))
            throw std::runtime_error(paramName + " has invalid shape");

        std::string value;
        value.reserve(vals.size());
        for (double v : vals)
        {
            const long long code = static_cast<long long>(std::llround(v));
            if (code <= 0 || code > 255)
                throw std::runtime_error(paramName + " contains invalid character code");
            value.push_back(static_cast<char>(code));
        }
        return value;
    }

public:

    struct ParamDims { int n_rows; int n_cols; };

    // Load model_meta and prove that its structural width agrees with the
    // persisted input parameter matrix.  model_meta is written from that
    // matrix by saveModelMeta, so this is the persisted model contract.
    static PersistedModelMeta loadRequiredModelMeta(pqxx::work& w,
                                                    long long modelId)
    {
        const auto dims = loadParameterDims(w, modelId, "model_meta");
        const auto vals = loadParameterValues(w, modelId, "model_meta");
        if (dims.n_rows != 1 || dims.n_cols != 3 || vals.size() != 3)
            throw std::runtime_error("model_meta has invalid shape; expected 1x3");

        const int schemaVersion = static_cast<int>(std::llround(vals[0]));
        const double inputWidthValue = vals[1];
        const double hiddenSizeValue = vals[2];
        if (schemaVersion != 1 ||
            !std::isfinite(inputWidthValue) ||
            !std::isfinite(hiddenSizeValue) ||
            std::llround(inputWidthValue) != inputWidthValue ||
            std::llround(hiddenSizeValue) != hiddenSizeValue ||
            inputWidthValue <= 0.0 || hiddenSizeValue <= 0.0)
            throw std::runtime_error("model_meta has invalid schema or dimensions");

        const std::size_t inputWidth = static_cast<std::size_t>(inputWidthValue);
        const std::size_t hiddenSize = static_cast<std::size_t>(hiddenSizeValue);
        (void)EA::ContractForModelInputWidth(inputWidth);

        const auto paramDims = loadParameterDims(w, modelId, "param");
        if (paramDims.n_rows <= 0 || paramDims.n_cols <= 0 ||
            paramDims.n_cols % 4 != 0 ||
            static_cast<std::size_t>(paramDims.n_cols / 4) != hiddenSize ||
            paramDims.n_rows <= paramDims.n_cols / 4)
            throw std::runtime_error("param has invalid LSTM gate-matrix shape");

        const std::size_t parameterInputWidth =
            static_cast<std::size_t>(paramDims.n_rows - paramDims.n_cols / 4);
        if (parameterInputWidth != inputWidth)
            throw std::runtime_error(
                "MODEL_META_PARAMETER_SHAPE_MISMATCH,model_meta_n_in=" +
                std::to_string(inputWidth) + ",param_n_in=" +
                std::to_string(parameterInputWidth));

        return {schemaVersion, inputWidth, hiddenSize};
    }

    // Try to load minimal model metadata and validate against current parameter shapes
    static bool tryLoadModelMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        try {
            const auto meta = loadRequiredModelMeta(w, modelId);
            const size_t rows = lstm.param.Shape()[0];
            const size_t cols = lstm.param.Shape()[1];
            if (cols % 4 != 0 || rows <= cols / 4) return false;
            const size_t hidden_size = cols / 4;
            const size_t n_in = rows - hidden_size;
            return meta.inputWidth == n_in && meta.hiddenSize == hidden_size;
        }
        catch (...) { return false;   }
    }

    static ParamDims loadParameterDims(pqxx::work& w,
                                       long long modelId,
                                       const std::string& paramName)
    {
        pqxx::result r = w.exec_params(
            "SELECT DISTINCT n_rows, n_cols FROM matrix WHERE model_id = $1 AND param_name = $2;",
            modelId, paramName
        );
        if (r.empty()) throw std::runtime_error("No entries for parameter: " + paramName);
        return { r[0][0].as<int>(), r[0][1].as<int>() };
    }

    static std::vector<double> loadParameterValues(pqxx::work& w,
                                                   long long modelId,
                                                   const std::string& paramName)
    {
        std::vector<double> vals;
        pqxx::result r = w.exec_params(
            "SELECT value FROM matrix WHERE model_id = $1 AND param_name = $2 ORDER BY row_idx, col_idx;",
            modelId, paramName
        );
        vals.reserve(r.size());
        for (const auto& row : r) vals.push_back(row[0].as<double>());
        return vals;
    }

    template <typename T = float>
    static MatGPU<T> loadParameterMatrix(pqxx::work& w,
                                         long long modelId,
                                         const std::string& paramName)
    {
        auto dims = loadParameterDims(w, modelId, paramName);
        auto vals = loadParameterValues(w, modelId, paramName);
        return fromFlatRowMajor<T>(vals, static_cast<size_t>(dims.n_rows), static_cast<size_t>(dims.n_cols));
    }

    // Decode the persisted target contract used to choose the real training
    // path.  Type 3 remains accepted as the legacy spelling of the current
    // three-class direction target.
    static PersistedTargetMeta loadRequiredTargetMeta(pqxx::work& w,
                                                      long long modelId)
    {
        const auto dims = loadParameterDims(w, modelId, "target_meta");
        const auto vals = loadParameterValues(w, modelId, "target_meta");
        if (dims.n_rows != 1 || dims.n_cols != 6 || vals.size() != 6)
            throw std::runtime_error("target_meta has invalid shape");

        const double typeValue = vals[0];
        if (!std::isfinite(typeValue) || std::llround(typeValue) != typeValue)
            throw std::runtime_error("target_meta has invalid target type");

        PersistedTargetMeta meta;
        switch (std::llround(typeValue))
        {
            case 0: meta.targetType = EA::LSTM::TargetType::LogReturn; break;
            case 1: meta.targetType = EA::LSTM::TargetType::PercentReturn; break;
            case 2:
            case 3: // legacy spelling of the three-class direction target
                meta.targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
                break;
            default: throw std::runtime_error("target_meta has unsupported target type");
        }
        meta.targetScale = static_cast<float>(vals[1]);
        meta.targetBias = static_cast<float>(vals[2]);
        meta.targetUseZScore = (vals[3] != 0.0);
        meta.targetMean = static_cast<float>(vals[4]);
        meta.targetStd = static_cast<float>(vals[5]);
        return meta;
    }

    static void applyTargetMeta(const PersistedTargetMeta& meta, EA::LSTM& lstm)
    {
        lstm.targetType = meta.targetType;
        lstm.targetScale = meta.targetScale;
        lstm.targetBias = meta.targetBias;
        lstm.targetUseZScore = meta.targetUseZScore;
        lstm.targetMean = meta.targetMean;
        lstm.targetStd = meta.targetStd;
    }

    static MatGPU<float> loadRequiredDirectionHeadMatrix(
        pqxx::work& w,
        long long modelId,
        const std::string& paramName,
        std::size_t expectedRows,
        std::size_t expectedCols)
    {
        auto matrix = loadParameterMatrix<float>(w, modelId, paramName);
        if (matrix.Shape()[0] != expectedRows || matrix.Shape()[1] != expectedCols)
            throw std::runtime_error(
                paramName + " has invalid shape; expected " +
                std::to_string(expectedRows) + "x" +
                std::to_string(expectedCols));
        return matrix;
    }

    // Validate the persisted state that the training resume path loads after
    // its configuration metadata has been accepted.  Keep this in sync with
    // loadAll/loadOptimizerMeta so callers can reject a checkpoint before it
    // is made a resume source.
    static void validateTrainingResumeState(pqxx::work& w, long long modelId)
    {
        const auto modelMeta = loadRequiredModelMeta(w, modelId);
        (void)loadParameterMatrix<float>(w, modelId, "param");
        (void)loadParameterMatrix<float>(w, modelId, "bias");
        (void)loadParameterMatrix<float>(w, modelId, "returnHeadWeight");
        (void)loadParameterMatrix<float>(w, modelId, "returnHeadBias");

        const auto targetMeta = loadRequiredTargetMeta(w, modelId);
        if (targetMeta.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
        {
            (void)loadRequiredDirectionHeadMatrix(
                w,
                modelId,
                "returnHeadDirWeight",
                modelMeta.hiddenSize,
                static_cast<std::size_t>(direction_output_size));
            (void)loadRequiredDirectionHeadMatrix(
                w,
                modelId,
                "returnHeadDirBias",
                1,
                static_cast<std::size_t>(direction_output_size));
        }

        const auto optimizerDims = loadParameterDims(w, modelId, "optimizer_meta");
        const auto optimizerValues =
            loadParameterValues(w, modelId, "optimizer_meta");
        if (optimizerDims.n_rows != 1 ||
            optimizerDims.n_cols < kOptimizerMetaFieldCount ||
            optimizerValues.size() < static_cast<size_t>(kOptimizerMetaFieldCount))
            throw std::runtime_error("resume requires valid optimizer_meta");

        const int optimizerSchema =
            static_cast<int>(std::llround(optimizerValues[0]));
        const int optimizerType =
            static_cast<int>(std::llround(optimizerValues[1]));
        const int firstMomentBuffers =
            static_cast<int>(std::llround(optimizerValues[3]));
        const int secondMomentBuffers =
            static_cast<int>(std::llround(optimizerValues[4]));
        if (optimizerSchema != kOptimizerMetaSchemaVersion ||
            optimizerType != kOptimizerTypeSgd ||
            firstMomentBuffers != 0 || secondMomentBuffers != 0)
            throw std::runtime_error("resume optimizer_meta is not supported by this binary");
    }

    static bool hasTrainingResumeState(pqxx::work& w, long long modelId)
    {
        try
        {
            validateTrainingResumeState(w, modelId);
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    // Load all parameters into an existing LSTM instance
    static void loadAll(pqxx::work& w, long long modelId, EA::LSTM& lstm)
    {
        lstm.param            = loadParameterMatrix<float>(w, modelId, "param");
        lstm.bias             = loadParameterMatrix<float>(w, modelId, "bias");
        lstm.returnHeadWeight = loadParameterMatrix<float>(w, modelId, "returnHeadWeight");
        lstm.returnHeadBias   = loadParameterMatrix<float>(w, modelId, "returnHeadBias");

        const std::optional<PersistedTargetMeta> targetMeta =
            tryLoadTargetMeta(w, modelId, lstm);
        if (targetMeta.has_value() &&
            targetMeta->targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
        {
            const std::size_t loadedHiddenSize = lstm.param.Shape()[1] / 4;
            lstm.returnHeadDirWeight = loadRequiredDirectionHeadMatrix(
                w,
                modelId,
                "returnHeadDirWeight",
                loadedHiddenSize,
                static_cast<std::size_t>(direction_output_size));
            lstm.returnHeadDirBias = loadRequiredDirectionHeadMatrix(
                w,
                modelId,
                "returnHeadDirBias",
                1,
                static_cast<std::size_t>(direction_output_size));
        }
        else
        {
            // Directional tensors were not consumed by these target paths in
            // historical checkpoints, so preserve their optional loading.
            try { lstm.returnHeadDirWeight = loadParameterMatrix<float>(w, modelId, "returnHeadDirWeight"); } catch (...) { /* keep defaults */ }
            try { lstm.returnHeadDirBias   = loadParameterMatrix<float>(w, modelId, "returnHeadDirBias"); } catch (...) { /* keep defaults */ }
        }

        const size_t rows = lstm.param.Shape()[0];
        const size_t cols = lstm.param.Shape()[1];
        if (cols == 0 || cols % 4 != 0 || rows <= cols / 4 ||
            rows - cols / 4 != static_cast<size_t>(lstm.InputFeatureCount()))
            throw std::runtime_error(
                "MODEL_PARAMETER_SHAPE_MISMATCH,loaded_n_in=" +
                std::to_string(rows > cols / 4 ? rows - cols / 4 : 0) +
                ",runtime_n_in=" + std::to_string(lstm.InputFeatureCount()));

        try
        {
            (void)loadRequiredModelMeta(w, modelId);
        }
        catch (const std::exception& error)
        {
            if (std::string{error.what()}.find("No entries for parameter: model_meta") ==
                std::string::npos)
                throw;
        }
    }

    static std::optional<PersistedTargetMeta> tryLoadTargetMeta(pqxx::work& w,
                                                                 long long modelId,
                                                                 EA::LSTM& lstm)
    {
        try
        {
            const PersistedTargetMeta meta = loadRequiredTargetMeta(w, modelId);
            applyTargetMeta(meta, lstm);
            return meta;
        }
        catch (...) { return std::nullopt; }
    }
};

} // namespace DBIO
#pragma clang diagnostic pop
