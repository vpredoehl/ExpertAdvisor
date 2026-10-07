#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <string>

#include <pqxx/pqxx>

#include "LSTM.hpp"
#include "PersistedModelRuntimeConfig.hpp"
#include "PgModelIO.hpp"
#include "Tensor.hpp"

namespace EA::ModelRuntimeValidation
{

using TrainConfigMeta = PersistedModelRuntimeConfig::TrainConfigMeta;

struct ModelConfigValidationResult
{
    std::optional<TrainConfigMeta> trainConfigMeta;
    bool configMatch = false;
    bool hasMismatch = false;
    bool metadataGap = false;
};

// The host supplies the established output gates/stream, preserving the
// executable's current logging behavior without importing its application layer.
struct Diagnostics
{
    bool (*summaryEnabled)() = nullptr;
    std::ostream& (*diagnosticOut)() = nullptr;
};

void SetDiagnostics(Diagnostics diagnostics);
const char* TargetTypeName(LSTM::TargetType targetType);
const char* DirectionLabelRuleName();
int DirectionLabelRuleId();
size_t RuntimeTensorFeatureWidth(const Tensor& tensor);
size_t RuntimeModelInputWidth(const Tensor& tensor, std::optional<std::size_t> persistedModelInputWidth = std::nullopt);
ModelConfigValidationResult PrintModelConfigValidation(pqxx::work& w, long long modelId, LSTM::TargetType requestedTargetType, const Tensor& tensor, const std::string& runtimeSymbol);
ModelConfigValidationResult PrintMaterializedModelConfigValidation(const DBIO::PgModelIO::PersistedModelMaterialization& persisted, LSTM::TargetType requestedTargetType, const Tensor& tensor, const std::string& runtimeSymbol);
void PrintDatabaseModelSymbol(long long modelId, const std::string& symbol);
void PrintLegacyModelSymbol(long long modelId, const std::string& symbol);
void PrintMissingModelSymbol(long long modelId);
void ValidateRuntimeSymbolMatchesModel(const std::optional<std::string>& runtimeSymbol, const std::string& modelSymbol);
std::optional<std::string> ResolveLegacySymbolFromModelName(const std::string& modelName, const std::vector<std::string>& availableSymbols);
std::string ResolveLegacyModelSymbol(long long modelId, const std::string& modelName, const std::optional<std::string>& runtimeSymbol, const std::vector<std::string>& availableSymbols);
void ValidateLoadedModelSymbolForSelectedTable(pqxx::work& w, long long modelId, const std::string& selectedSymbol);

} // namespace EA::ModelRuntimeValidation
