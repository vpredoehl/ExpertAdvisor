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
using MatCPU = MetaNN::Matrix<T, MetaNN::DeviceTags::CPU>;

// Flatten to row-major vector<double>
template <typename T>
inline std::vector<double> flattenRowMajor(const MatCPU<T>& m)
{
    const size_t rows = m.Shape()[0];
    const size_t cols = m.Shape()[1];
    std::vector<double> out;
    out.reserve(rows * cols);

    auto mm = MetaNN::Evaluate(m);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            out.push_back(static_cast<double>(mm(r, c)));
    return out;
}

// Reconstruct from row-major vector<double> into Matrix<T>
template <typename T = float>
inline MatCPU<T> fromFlatRowMajor(const std::vector<double>& vals, size_t rows, size_t cols)
{
    if (vals.size() != rows * cols)
        throw std::runtime_error("fromFlatRowMajor: size mismatch with rows*cols");

    MatCPU<T> m(rows, cols);
    size_t idx = 0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m.SetValue(r, c, static_cast<T>(vals[idx++]));
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
                              const MatCPU<T>& mat)
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
    }

    // Save target mapping metadata as a 1x6 matrix in order:
    // [type(int), scale, bias, useZ(0/1), mean, std]
    static void saveTargetMeta(pqxx::work& w, long long modelId, const EA::LSTM& lstm)
    {
        MatCPU<float> meta(1, 6);
        meta.SetValue(0, 0, static_cast<float>(static_cast<int>(lstm.targetType)));
        meta.SetValue(0, 1, lstm.targetScale);
        meta.SetValue(0, 2, lstm.targetBias);
        meta.SetValue(0, 3, lstm.targetUseZScore ? 1.0f : 0.0f);
        meta.SetValue(0, 4, lstm.targetMean);
        meta.SetValue(0, 5, lstm.targetStd);
        saveParameter(w, modelId, "target_meta", meta);
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
    static MatCPU<T> loadParameterMatrix(pqxx::work& w,
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
        } catch (...) {
            return false;
        }
    }
};

} // namespace DBIO
#pragma clang diagnostic pop

