// PgModelIO.hpp - header-only helpers to persist/load EA::LSTM parameters with PostgreSQL (libpqxx)
#pragma once

#include <pqxx/pqxx>
#include <MetaNN/meta_nn.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include "LSTM.hpp"


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
    // Create a new row in `model` table and return model_id
    static long long createModel(pqxx::work& w, const std::string& name, const std::string& comment)
    {
        pqxx::result r = w.exec_params(
            "INSERT INTO model (name, comment) VALUES ($1, $2) RETURNING model_id;",
            name, comment
        );
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
    static void saveAll(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        saveParameter(w, modelId, "param",            lstm.param);
        saveParameter(w, modelId, "bias",             lstm.bias);
        saveParameter(w, modelId, "returnHeadWeight", lstm.returnHeadWeight);
        saveParameter(w, modelId, "returnHeadBias",   lstm.returnHeadBias);
        saveParameter(w, modelId, "returnHeadDirWeight", lstm.returnHeadDirWeight);
        saveParameter(w, modelId, "returnHeadDirBias",   lstm.returnHeadDirBias);
        saveTargetMeta(w, modelId, lstm);
        saveModelMeta(w, modelId, lstm);
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

    // Try to load minimal model metadata and validate against current parameter shapes
    static bool tryLoadModelMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        try {
            auto dims = loadParameterDims(w, modelId, "model_meta");
            if (dims.n_rows != 1 || dims.n_cols != 3) return false;
            auto vals = loadParameterValues(w, modelId, "model_meta");
            if (vals.size() != 3) return false;

            const int schemaVersion = static_cast<int>(vals[0]);
            if (schemaVersion != 1) throw std::runtime_error("model_meta: unsupported schemaVersion");

            // Derive from current param
            const size_t rows = lstm.param.Shape()[0];
            const size_t cols = lstm.param.Shape()[1];
            const size_t hidden_size = cols / 4;
            const size_t n_in = rows - hidden_size;

            const int n_in_db = static_cast<int>(vals[1]);
            const int h_db    = static_cast<int>(vals[2]);
            if (n_in_db != static_cast<int>(n_in) || h_db != static_cast<int>(hidden_size))
                throw std::runtime_error("model_meta mismatch: n_in/hidden_size differ from persisted values");
            return true;
        }
        catch (...) { return false;   }
    }

    struct ParamDims { int n_rows; int n_cols; };

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

    // Load all parameters into an existing LSTM instance
    static void loadAll(pqxx::work& w, long long modelId, EA::LSTM& lstm)
    {
        lstm.param            = loadParameterMatrix<float>(w, modelId, "param");
        lstm.bias             = loadParameterMatrix<float>(w, modelId, "bias");
        lstm.returnHeadWeight = loadParameterMatrix<float>(w, modelId, "returnHeadWeight");
        lstm.returnHeadBias   = loadParameterMatrix<float>(w, modelId, "returnHeadBias");
        // Try to load binary classification head if present (backward compatible)
        try { lstm.returnHeadDirWeight = loadParameterMatrix<float>(w, modelId, "returnHeadDirWeight"); } catch (...) { /* keep defaults */ }
        try { lstm.returnHeadDirBias   = loadParameterMatrix<float>(w, modelId, "returnHeadDirBias"); } catch (...) { /* keep defaults */ }
        (void)tryLoadTargetMeta(w, modelId, lstm);
        (void)tryLoadModelMeta(w, modelId, lstm);
    }

    static bool tryLoadTargetMeta(pqxx::work& w, long long modelId, EA::LSTM& lstm)
    {
        try {
            auto dims = loadParameterDims(w, modelId, "target_meta");
            if (dims.n_rows != 1 || dims.n_cols != 6) return false;
            auto vals = loadParameterValues(w, modelId, "target_meta");
            if (vals.size() != 6) return false;
            int typeInt = static_cast<int>(vals[0]);
            switch (typeInt) {
                case 0: lstm.targetType = EA::LSTM::TargetType::LogReturn; break;
                case 1: lstm.targetType = EA::LSTM::TargetType::PercentReturn; break;
                case 2: lstm.targetType = EA::LSTM::TargetType::BinaryReturn; break;
                default: return false; // unknown type, fail to load meta
            }
            lstm.targetScale = static_cast<float>(vals[1]);
            lstm.targetBias  = static_cast<float>(vals[2]);
            lstm.targetUseZScore = (vals[3] != 0.0);
            lstm.targetMean  = static_cast<float>(vals[4]);
            lstm.targetStd   = static_cast<float>(vals[5]);
            return true;
        }
        catch (...) { return false;   }
    }
};

} // namespace DBIO
#pragma clang diagnostic pop

