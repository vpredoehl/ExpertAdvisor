#include "ExperimentPairComparisonService.hpp"

#include <charconv>
#include <ostream>
#include <string>

namespace EA::ExperimentPairComparison
{
namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

long long ParsePositiveExperimentId(std::string_view value)
{
    long long parsed = 0;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        parsed <= 0)
        throw std::invalid_argument(
            "experiment-pair IDs must be positive integers");
    return parsed;
}

} // namespace

EvidenceUnavailableError::EvidenceUnavailableError(std::string reason)
    : std::runtime_error(reason), reason_(std::move(reason))
{
}

std::pair<long long, long long> ParseExperimentIdPair(std::string_view text)
{
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 >= text.size() ||
        text.find(':', separator + 1) != std::string_view::npos)
        throw std::invalid_argument(
            "--compare-experiment-pair requires EXPERIMENT_A_ID:EXPERIMENT_B_ID");
    const auto result = std::pair{
        ParsePositiveExperimentId(text.substr(0, separator)),
        ParsePositiveExperimentId(text.substr(separator + 1))};
    if (result.first == result.second)
        throw std::invalid_argument(
            "--compare-experiment-pair requires distinct experiment IDs");
    return result;
}

Request MakeComparisonRequest(
    const FeatureAblationPairEvaluation::ArmEvidence& armA,
    const FeatureAblationPairEvaluation::ArmEvidence& armB)
{
    Request request;
    const std::string& maskA =
        armA.authoritative.configuration.featureAblationMask;
    const std::string& maskB =
        armB.authoritative.configuration.featureAblationMask;
    if (maskA != maskB && (maskA.empty() != maskB.empty()))
        request.intentionalDifferenceFields = {"feature_ablation_mask"};
    return request;
}

int RunComparisonCommand(const ComparisonCommand& command,
                         const EvidenceSource& source,
                         std::ostream& output,
    std::ostream& errors)
{
    try
    {
        if (command.experimentIds.first <= 0 ||
            command.experimentIds.second <= 0)
            throw std::invalid_argument(
                "experiment-pair IDs must be positive integers");
        if (command.experimentIds.first == command.experimentIds.second)
            throw std::invalid_argument(
                "--compare-experiment-pair requires distinct experiment IDs");
        const auto armAEvidence = source.Load(command.experimentIds.first);
        const auto armBEvidence = source.Load(command.experimentIds.second);
        const Request request = MakeComparisonRequest(
            armAEvidence, armBEvidence);
        const ComparisonResult result = Compare(
            MakeArmResultSet(armAEvidence), MakeArmResultSet(armBEvidence),
            request);
        output << (command.summary ? RenderSummary(result) : Render(result));
        return 0;
    }
    catch (const EvidenceUnavailableError& error)
    {
        errors << "EXPERIMENT_PAIR_COMPARISON_LOAD_FAILED"
               << ",experiment_a_id=" << command.experimentIds.first
               << ",experiment_b_id=" << command.experimentIds.second
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "EXPERIMENT_PAIR_COMPARISON_LOAD_FAILED"
               << ",experiment_a_id=" << command.experimentIds.first
               << ",experiment_b_id=" << command.experimentIds.second
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::ExperimentPairComparison
