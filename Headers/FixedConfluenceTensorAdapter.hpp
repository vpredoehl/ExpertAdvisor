#pragma once

#include "MarketStructureProductionObservationAdapter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>

// This is intentionally a narrow model boundary, not another detector.  It
// consumes only the two frozen Phase-2B production definitions and reduces
// their already-bounded descriptive outputs to two availability bits.
namespace EA::MarketStructure::TensorProjection
{

inline constexpr std::string_view kSupportFeatureName =
    "confluence_tg4_structural_fibonacci_retracement_support_available";
inline constexpr std::string_view kContradictionFeatureName =
    "confluence_tg4_structural_fibonacci_retracement_contradiction_available";

class FixedConfluenceTensorAdapter final
{
public:
    FixedConfluenceTensorAdapter()
        : definitions_(Production::FrozenConfluenceDefinitions()),
          supportIdentity_(CanonicalConfluenceDefinitionIdentity(definitions_[0])),
          contradictionIdentity_(CanonicalConfluenceDefinitionIdentity(definitions_[1]))
    {
    }

    std::array<float, 2> Adapt(
        const Production::ProductionConfluenceDescription& description,
        std::chrono::sys_seconds decisionTime) const
    {
        std::array<float, 2> result{};
        for (const ConfluenceObservation& output : description.outputs)
        {
            if (output.decisionTime != decisionTime ||
                output.availableAt > decisionTime || output.components.empty())
            {
                throw std::invalid_argument(
                    "FIXED_CONFLUENCE_TENSOR_OUTPUT_CAUSALITY_INVALID");
            }

            const auto latestComponent = std::max_element(
                output.components.begin(), output.components.end(),
                [](const Observation& left, const Observation& right) {
                    return left.availableAt < right.availableAt;
                })->availableAt;
            if (latestComponent != output.availableAt)
                throw std::invalid_argument(
                    "FIXED_CONFLUENCE_TENSOR_OUTPUT_AVAILABILITY_INVALID");

            if (output.definitionIdentity == supportIdentity_)
            {
                ValidateOutput(output, definitions_[0], RelationKind::support);
                SetOnce(result[0]);
            }
            else if (output.definitionIdentity == contradictionIdentity_)
            {
                ValidateOutput(output, definitions_[1], RelationKind::contradiction);
                SetOnce(result[1]);
            }
            else
            {
                // A future definition requires its own Tensor design gate; it
                // must never silently acquire a meaning in layout 11.
                throw std::invalid_argument(
                    "FIXED_CONFLUENCE_TENSOR_DEFINITION_UNSUPPORTED");
            }
        }
        return result;
    }

private:
    static void SetOnce(float& target)
    {
        if (target != 0.0f)
            throw std::invalid_argument("FIXED_CONFLUENCE_TENSOR_OUTPUT_DUPLICATE");
        target = 1.0f;
    }

    static void ValidateOutput(const ConfluenceObservation& output,
                               const ConfluenceDefinition& definition,
                               RelationKind expectedRelation)
    {
        if (output.definitionId != definition.definitionId ||
            output.definitionVersion != definition.definitionVersion ||
            output.descriptor.relation != expectedRelation ||
            output.descriptor.leftComponentCount == 0 ||
            output.descriptor.rightComponentCount == 0)
        {
            throw std::invalid_argument(
                "FIXED_CONFLUENCE_TENSOR_OUTPUT_SEMANTICS_INVALID");
        }
    }

    std::array<ConfluenceDefinition, 2> definitions_;
    std::string supportIdentity_;
    std::string contradictionIdentity_;
};

} // namespace EA::MarketStructure::TensorProjection
