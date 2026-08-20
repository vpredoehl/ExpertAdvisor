#ifndef ModelInputExpansion_hpp
#define ModelInputExpansion_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ModelInputContract.hpp"

namespace EA
{

inline constexpr int kModelInputSemanticMetaSchemaVersion = 1;
inline constexpr int kModelInputSemanticLayoutVersion = 1;
inline constexpr int kInputWidthExpansionProvenanceSchemaVersion = 1;
inline constexpr std::string_view kInputWidthExpansionInitializationPolicy =
    "zero";

struct ModelInputSemanticLayoutRegistryEntry
{
    int layoutVersion;
    std::size_t maximumInputWidth;
    int appendOnlyPredecessorVersion;
};

// A semantic-layout version identifies the complete registered feature layout
// at the time a model is saved.  Append-only additions add a new entry whose
// predecessor is the prior current version; old entries and their fixed
// maximum widths must never be changed.
inline constexpr std::array<ModelInputSemanticLayoutRegistryEntry, 1>
    kModelInputSemanticLayoutRegistry{{
        {1, kHistoricalLevelProximityModelInputWidth, 0},
    }};

static_assert(kModelInputSemanticLayoutRegistry.back().layoutVersion ==
              kModelInputSemanticLayoutVersion);
static_assert(kModelInputSemanticLayoutRegistry.back().maximumInputWidth ==
              kCurrentModelInputWidth);

struct ModelInputSemanticMetadata
{
    int schemaVersion;
    int layoutVersion;
};

[[noreturn]] inline void ThrowModelInputSemanticMetadataIncompatible()
{
    throw std::runtime_error(
        "MODEL_INPUT_EXPANSION_SEMANTIC_METADATA_INCOMPATIBLE");
}

inline ModelInputSemanticMetadata ParseModelInputSemanticMetadata(
    std::size_t rows,
    std::size_t columns,
    std::span<const double> values)
{
    if (rows != 1 || columns != 2 || values.size() != 2)
        ThrowModelInputSemanticMetadataIncompatible();
    for (const double value : values)
    {
        if (!std::isfinite(value) || std::trunc(value) != value ||
            value < static_cast<double>(std::numeric_limits<int>::min()) ||
            value > static_cast<double>(std::numeric_limits<int>::max()))
        {
            ThrowModelInputSemanticMetadataIncompatible();
        }
    }
    return {static_cast<int>(values[0]), static_cast<int>(values[1])};
}

inline bool IsModelInputSemanticLayoutWidthCompatible(
    int layoutVersion,
    std::size_t inputWidth,
    std::span<const ModelInputSemanticLayoutRegistryEntry> layoutRegistry,
    std::span<const std::size_t> registeredInputWidths,
    int currentLayoutVersion,
    std::size_t currentInputWidth,
    bool requireLayoutMaximumWidth)
{
    if (std::count(registeredInputWidths.begin(), registeredInputWidths.end(),
                   inputWidth) != 1 ||
        std::count(registeredInputWidths.begin(), registeredInputWidths.end(),
                   currentInputWidth) != 1)
    {
        return false;
    }

    const auto findUniqueLayout =
        [layoutRegistry](int version)
        -> const ModelInputSemanticLayoutRegistryEntry*
    {
        const ModelInputSemanticLayoutRegistryEntry* result = nullptr;
        for (const auto& entry : layoutRegistry)
        {
            if (entry.layoutVersion != version) continue;
            if (result != nullptr) return nullptr;
            result = &entry;
        }
        return result;
    };

    const auto* persistedLayout = findUniqueLayout(layoutVersion);
    const auto* currentLayout = findUniqueLayout(currentLayoutVersion);
    if (persistedLayout == nullptr || currentLayout == nullptr ||
        std::count(registeredInputWidths.begin(),
                   registeredInputWidths.end(),
                   persistedLayout->maximumInputWidth) != 1 ||
        currentLayout->maximumInputWidth != currentInputWidth ||
        (requireLayoutMaximumWidth
             ? inputWidth != persistedLayout->maximumInputWidth
             : inputWidth > persistedLayout->maximumInputWidth))
    {
        return false;
    }

    const ModelInputSemanticLayoutRegistryEntry* cursor = currentLayout;
    for (std::size_t visited = 0; visited <= layoutRegistry.size(); ++visited)
    {
        if (cursor->layoutVersion == persistedLayout->layoutVersion) return true;
        if (cursor->appendOnlyPredecessorVersion == 0) return false;
        const auto* predecessor =
            findUniqueLayout(cursor->appendOnlyPredecessorVersion);
        if (predecessor == nullptr ||
            std::count(registeredInputWidths.begin(),
                       registeredInputWidths.end(),
                       predecessor->maximumInputWidth) != 1 ||
            predecessor->maximumInputWidth >= cursor->maximumInputWidth)
        {
            return false;
        }
        cursor = predecessor;
    }
    return false;
}

inline void ValidateModelInputSemanticMetadataForExpansion(
    int schemaVersion,
    int layoutVersion,
    std::size_t persistedInputWidth,
    std::span<const ModelInputSemanticLayoutRegistryEntry> layoutRegistry,
    std::span<const std::size_t> registeredInputWidths,
    int currentLayoutVersion,
    std::size_t currentInputWidth)
{
    if (schemaVersion != kModelInputSemanticMetaSchemaVersion ||
        !IsModelInputSemanticLayoutWidthCompatible(
            layoutVersion, persistedInputWidth, layoutRegistry,
            registeredInputWidths, currentLayoutVersion, currentInputWidth,
            false))
    {
        ThrowModelInputSemanticMetadataIncompatible();
    }
}

inline void ValidateModelInputSemanticMetadataForExpansion(
    int schemaVersion,
    int layoutVersion,
    std::size_t persistedInputWidth)
{
    ValidateModelInputSemanticMetadataForExpansion(
        schemaVersion, layoutVersion, persistedInputWidth,
        kModelInputSemanticLayoutRegistry, kRegisteredModelInputWidths,
        kModelInputSemanticLayoutVersion, kCurrentModelInputWidth);
}

struct AppendedTensorFeatureSemantic
{
    std::size_t column;
    std::string_view name;
};

// FeatureLayout.hpp is the structural authority.  This registry supplies the
// corresponding persisted semantic names for the append-only portion that can
// be introduced by an expansion from any supported historical width.
inline constexpr std::array<AppendedTensorFeatureSemantic, 16>
    kAppendedTensorFeatureSemantics{{
        {donchianUpCol, "donchian_up"},
        {donchianDownCol, "donchian_down"},
        {sessionPhaseSinCol, "session_phase_sin"},
        {sessionPhaseCosCol, "session_phase_cos"},
        {relativeTickVolumeCol, "relative_tick_volume"},
        {causalReturnSurpriseCol, "rms_return_surprise"},
        {causalVolatilityRegimeCol, "volatility_regime"},
        {causalDirectionalRangeCol, "directional_range"},
        {causalCloseLocationCol, "close_location"},
        {causalDirectionalPersistenceCol, "directional_efficiency"},
        {causalReturnSignPersistenceCol, "return_sign_persistence"},
        {causalReturnDirectionImbalanceCol, "return_direction_imbalance"},
        {causalDirectionalAdverseExcursionCol,
         "directional_adverse_excursion"},
        {causalMultiBarRangePressureCol, "multi_bar_range_pressure"},
        {causalRollingRangeExpansionCol, "rolling_range_expansion"},
        {historicalLevelProximityCol, "historical_level_proximity"},
    }};

struct InputWidthExpansionPlan
{
    std::size_t sourceInputWidth = 0;
    std::size_t expandedInputWidth = 0;
    std::size_t sourceTensorFeatureCount = 0;
    std::size_t expandedTensorFeatureCount = 0;
    std::size_t returnFeatureCount = kModelReturnFeatureCount;
    std::vector<std::string> newlyIntroducedTensorFeatures;
};

inline InputWidthExpansionPlan BuildRegisteredInputWidthExpansionPlan(
    std::size_t sourceInputWidth,
    std::size_t expandedInputWidth,
    std::span<const std::size_t> registeredInputWidths,
    std::span<const AppendedTensorFeatureSemantic> appendedFeatureSemantics)
{
    if (sourceInputWidth > expandedInputWidth)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_SOURCE_WIDER_THAN_TARGET,source_n_in=" +
            std::to_string(sourceInputWidth) + ",target_n_in=" +
            std::to_string(expandedInputWidth));
    if (sourceInputWidth == expandedInputWidth)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_NOT_REQUIRED,source_n_in=" +
            std::to_string(sourceInputWidth) + ",target_n_in=" +
            std::to_string(expandedInputWidth));
    if (std::count(registeredInputWidths.begin(), registeredInputWidths.end(),
                   sourceInputWidth) != 1 ||
        std::count(registeredInputWidths.begin(), registeredInputWidths.end(),
                   expandedInputWidth) != 1 ||
        sourceInputWidth <= kModelReturnFeatureCount ||
        expandedInputWidth <= kModelReturnFeatureCount)
    {
        throw std::runtime_error(
            "MODEL_INPUT_WIDTH_UNSUPPORTED,n_in=" +
            std::to_string(std::count(registeredInputWidths.begin(),
                                      registeredInputWidths.end(),
                                      sourceInputWidth) == 1
                               ? expandedInputWidth
                               : sourceInputWidth));
    }

    InputWidthExpansionPlan plan;
    plan.sourceInputWidth = sourceInputWidth;
    plan.expandedInputWidth = expandedInputWidth;
    plan.sourceTensorFeatureCount =
        sourceInputWidth - kModelReturnFeatureCount;
    plan.expandedTensorFeatureCount =
        expandedInputWidth - kModelReturnFeatureCount;
    for (std::size_t column = plan.sourceTensorFeatureCount;
         column < plan.expandedTensorFeatureCount;
         ++column)
    {
        const auto semantic = std::find_if(
            appendedFeatureSemantics.begin(), appendedFeatureSemantics.end(),
            [column](const AppendedTensorFeatureSemantic& candidate)
            {
                return candidate.column == column;
            });
        if (semantic == appendedFeatureSemantics.end())
            throw std::runtime_error(
                "MODEL_INPUT_EXPANSION_SEMANTIC_COLUMN_UNKNOWN,column=" +
                std::to_string(column));
        plan.newlyIntroducedTensorFeatures.emplace_back(semantic->name);
    }
    return plan;
}

inline InputWidthExpansionPlan BuildInputWidthExpansionPlan(
    std::size_t sourceInputWidth,
    std::size_t expandedInputWidth = kCurrentModelInputWidth)
{
    if (sourceInputWidth > expandedInputWidth)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_SOURCE_WIDER_THAN_TARGET,source_n_in=" +
            std::to_string(sourceInputWidth) + ",target_n_in=" +
            std::to_string(expandedInputWidth));
    if (sourceInputWidth == expandedInputWidth)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_NOT_REQUIRED,source_n_in=" +
            std::to_string(sourceInputWidth) + ",target_n_in=" +
            std::to_string(expandedInputWidth));

    if (expandedInputWidth != kCurrentModelInputWidth)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_TARGET_NOT_CURRENT,target_n_in=" +
            std::to_string(expandedInputWidth) + ",current_n_in=" +
            std::to_string(kCurrentModelInputWidth));
    return BuildRegisteredInputWidthExpansionPlan(
        sourceInputWidth, expandedInputWidth, kRegisteredModelInputWidths,
        kAppendedTensorFeatureSemantics);
}

template <typename T>
inline std::vector<T> ExpandFusedLstmParameterRowMajor(
    const std::vector<T>& source,
    std::size_t hiddenSize,
    const InputWidthExpansionPlan& plan)
{
    if (hiddenSize == 0 ||
        hiddenSize > std::numeric_limits<std::size_t>::max() / 4)
        throw std::runtime_error("MODEL_INPUT_EXPANSION_INVALID_HIDDEN_SIZE");
    if (plan.sourceTensorFeatureCount + plan.returnFeatureCount !=
            plan.sourceInputWidth ||
        plan.expandedTensorFeatureCount + plan.returnFeatureCount !=
            plan.expandedInputWidth ||
        plan.sourceTensorFeatureCount >= plan.expandedTensorFeatureCount)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PLAN_SHAPE_INCONSISTENT");
    }
    const std::size_t gateColumns = 4 * hiddenSize;
    if (plan.sourceInputWidth >
            std::numeric_limits<std::size_t>::max() - hiddenSize ||
        plan.expandedInputWidth >
            std::numeric_limits<std::size_t>::max() - hiddenSize)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PARAMETER_SIZE_OVERFLOW");
    }
    const std::size_t sourceRows = plan.sourceInputWidth + hiddenSize;
    const std::size_t expandedRows = plan.expandedInputWidth + hiddenSize;
    if (sourceRows > std::numeric_limits<std::size_t>::max() / gateColumns ||
        expandedRows > std::numeric_limits<std::size_t>::max() / gateColumns ||
        gateColumns > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PARAMETER_SIZE_OVERFLOW");
    }
    if (source.size() != sourceRows * gateColumns)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PARAMETER_SHAPE_MISMATCH,source_values=" +
            std::to_string(source.size()) + ",expected_values=" +
            std::to_string(sourceRows * gateColumns));

    std::vector<T> expanded(expandedRows * gateColumns, static_cast<T>(0));
    const std::size_t rowBytes = gateColumns * sizeof(T);

    // Stable Tensor prefix remains at the same row indices.
    std::memcpy(expanded.data(), source.data(),
                plan.sourceTensorFeatureCount * rowBytes);

    // The four return features are a stable semantic suffix of the input
    // block.  Newly appended Tensor columns are inserted before that suffix,
    // so relocate it rather than reinterpreting its old row indices.
    std::memcpy(
        expanded.data() + plan.expandedTensorFeatureCount * gateColumns,
        source.data() + plan.sourceTensorFeatureCount * gateColumns,
        plan.returnFeatureCount * rowBytes);

    // Recurrent hidden-to-hidden rows follow the complete input block.
    std::memcpy(
        expanded.data() + plan.expandedInputWidth * gateColumns,
        source.data() + plan.sourceInputWidth * gateColumns,
        hiddenSize * rowBytes);
    return expanded;
}

struct InputWidthExpansionProvenance
{
    long long sourceModelId = -1;
    std::size_t sourceInputWidth = 0;
    std::size_t expandedInputWidth = 0;
    std::size_t firstNewTensorColumn = 0;
    std::size_t newTensorColumnEnd = 0;
    std::vector<std::string> newlyIntroducedTensorFeatures;
    std::string initializationPolicy;
    int semanticLayoutVersion = 0;

    std::string CanonicalText() const
    {
        std::ostringstream out;
        out << "schema=" << kInputWidthExpansionProvenanceSchemaVersion
            << ";source_model_id=" << sourceModelId
            << ";source_input_width=" << sourceInputWidth
            << ";expanded_input_width=" << expandedInputWidth
            << ";new_tensor_columns=" << firstNewTensorColumn << ":"
            << newTensorColumnEnd
            << ";new_tensor_features=";
        for (std::size_t i = 0; i < newlyIntroducedTensorFeatures.size(); ++i)
        {
            if (i != 0) out << '|';
            out << newlyIntroducedTensorFeatures[i];
        }
        out << ";initialization=" << initializationPolicy
            << ";semantic_layout=" << semanticLayoutVersion;
        return out.str();
    }
};

inline InputWidthExpansionProvenance MakeInputWidthExpansionProvenance(
    long long sourceModelId,
    const InputWidthExpansionPlan& plan,
    int semanticLayoutVersion,
    std::span<const ModelInputSemanticLayoutRegistryEntry> layoutRegistry,
    std::span<const std::size_t> registeredInputWidths,
    std::span<const AppendedTensorFeatureSemantic> appendedFeatureSemantics,
    int currentLayoutVersion,
    std::size_t currentInputWidth)
{
    if (sourceModelId <= 0)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_INVALID_SOURCE_MODEL_ID");
    const InputWidthExpansionPlan expected =
        BuildRegisteredInputWidthExpansionPlan(
            plan.sourceInputWidth, plan.expandedInputWidth,
            registeredInputWidths, appendedFeatureSemantics);
    if (!IsModelInputSemanticLayoutWidthCompatible(
            semanticLayoutVersion, plan.expandedInputWidth, layoutRegistry,
            registeredInputWidths, currentLayoutVersion, currentInputWidth,
            true) ||
        plan.sourceTensorFeatureCount != expected.sourceTensorFeatureCount ||
        plan.expandedTensorFeatureCount !=
            expected.expandedTensorFeatureCount ||
        plan.returnFeatureCount != expected.returnFeatureCount ||
        plan.newlyIntroducedTensorFeatures !=
            expected.newlyIntroducedTensorFeatures)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INCOMPATIBLE");
    }
    return {
        sourceModelId,
        plan.sourceInputWidth,
        plan.expandedInputWidth,
        plan.sourceTensorFeatureCount,
        plan.expandedTensorFeatureCount,
        plan.newlyIntroducedTensorFeatures,
        std::string{kInputWidthExpansionInitializationPolicy},
        semanticLayoutVersion};
}

inline InputWidthExpansionProvenance MakeInputWidthExpansionProvenance(
    long long sourceModelId,
    const InputWidthExpansionPlan& plan)
{
    return MakeInputWidthExpansionProvenance(
        sourceModelId, plan, kModelInputSemanticLayoutVersion,
        kModelInputSemanticLayoutRegistry, kRegisteredModelInputWidths,
        kAppendedTensorFeatureSemantics, kModelInputSemanticLayoutVersion,
        kCurrentModelInputWidth);
}

inline std::string InputExpansionField(const std::string& text,
                                       const std::string& key)
{
    const std::string prefix = key + "=";
    std::size_t begin = text.find(prefix);
    while (begin != std::string::npos && begin != 0 && text[begin - 1] != ';')
        begin = text.find(prefix, begin + 1);
    if (begin == std::string::npos)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_MISSING_FIELD:" + key);
    const std::size_t valueBegin = begin + prefix.size();
    const std::size_t end = text.find(';', valueBegin);
    return text.substr(valueBegin, end == std::string::npos
                                       ? std::string::npos
                                       : end - valueBegin);
}

inline long long ParseExpansionLongLong(const std::string& text,
                                        const std::string& field)
{
    const std::string value = InputExpansionField(text, field);
    std::size_t consumed = 0;
    long long result = 0;
    try
    {
        result = std::stoll(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:" + field);
    }
    if (consumed != value.size())
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:" + field);
    return result;
}

inline std::size_t ParseExpansionSize(const std::string& value,
                                      const std::string& field)
{
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try
    {
        if (value.empty() || value.front() == '-')
            throw std::invalid_argument("negative");
        parsed = std::stoull(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:" + field);
    }
    if (consumed != value.size() ||
        parsed > std::numeric_limits<std::size_t>::max())
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:" + field);
    }
    return static_cast<std::size_t>(parsed);
}

inline InputWidthExpansionProvenance ParseInputWidthExpansionProvenance(
    const std::string& text,
    std::span<const ModelInputSemanticLayoutRegistryEntry> layoutRegistry,
    std::span<const std::size_t> registeredInputWidths,
    std::span<const AppendedTensorFeatureSemantic> appendedFeatureSemantics,
    int currentLayoutVersion,
    std::size_t currentInputWidth)
{
    const long long schema = ParseExpansionLongLong(text, "schema");
    if (schema != kInputWidthExpansionProvenanceSchemaVersion)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_SCHEMA_UNSUPPORTED");

    InputWidthExpansionProvenance result;
    result.sourceModelId = ParseExpansionLongLong(text, "source_model_id");
    result.sourceInputWidth = ParseExpansionSize(
        InputExpansionField(text, "source_input_width"),
        "source_input_width");
    result.expandedInputWidth = ParseExpansionSize(
        InputExpansionField(text, "expanded_input_width"),
        "expanded_input_width");
    const std::string columns = InputExpansionField(text, "new_tensor_columns");
    const std::size_t separator = columns.find(':');
    if (separator == std::string::npos)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:new_tensor_columns");
    result.firstNewTensorColumn = ParseExpansionSize(
        columns.substr(0, separator), "new_tensor_columns");
    result.newTensorColumnEnd = ParseExpansionSize(
        columns.substr(separator + 1), "new_tensor_columns");

    const std::string features =
        InputExpansionField(text, "new_tensor_features");
    std::size_t start = 0;
    while (start < features.size())
    {
        const std::size_t end = features.find('|', start);
        result.newlyIntroducedTensorFeatures.push_back(features.substr(
            start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    result.initializationPolicy =
        InputExpansionField(text, "initialization");
    const long long semanticLayout =
        ParseExpansionLongLong(text, "semantic_layout");
    if (semanticLayout < std::numeric_limits<int>::min() ||
        semanticLayout > std::numeric_limits<int>::max())
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INVALID_FIELD:semantic_layout");
    }
    result.semanticLayoutVersion = static_cast<int>(semanticLayout);

    InputWidthExpansionPlan expected;
    try
    {
        expected = BuildRegisteredInputWidthExpansionPlan(
            result.sourceInputWidth, result.expandedInputWidth,
            registeredInputWidths, appendedFeatureSemantics);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INCOMPATIBLE");
    }
    if (result.sourceModelId <= 0 ||
        result.firstNewTensorColumn != expected.sourceTensorFeatureCount ||
        result.newTensorColumnEnd != expected.expandedTensorFeatureCount ||
        result.newlyIntroducedTensorFeatures !=
            expected.newlyIntroducedTensorFeatures ||
        result.initializationPolicy !=
            kInputWidthExpansionInitializationPolicy ||
        !IsModelInputSemanticLayoutWidthCompatible(
            result.semanticLayoutVersion, result.expandedInputWidth,
            layoutRegistry, registeredInputWidths, currentLayoutVersion,
            currentInputWidth, true))
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_INCOMPATIBLE");
    }
    if (result.CanonicalText() != text)
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_PROVENANCE_NOT_CANONICAL");
    return result;
}

inline InputWidthExpansionProvenance ParseInputWidthExpansionProvenance(
    const std::string& text)
{
    return ParseInputWidthExpansionProvenance(
        text, kModelInputSemanticLayoutRegistry, kRegisteredModelInputWidths,
        kAppendedTensorFeatureSemantics, kModelInputSemanticLayoutVersion,
        kCurrentModelInputWidth);
}

} // namespace EA

#endif /* ModelInputExpansion_hpp */
