#include "ExperimentReplicationComparisonService.hpp"

#include <ostream>
#include <set>
#include <stdexcept>

namespace EA::ExperimentReplicationComparison
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

void ValidatePairs(
    const std::vector<std::pair<long long, long long>>& pairs)
{
    if (pairs.size() < 2)
        throw std::invalid_argument(
            "--compare-experiment-replications requires at least two pairs");
    std::set<long long> experimentIds;
    for (const auto& [armA, armB] : pairs)
    {
        if (armA <= 0 || armB <= 0)
            throw std::invalid_argument(
                "experiment-replication IDs must be positive integers");
        if (armA == armB)
            throw std::invalid_argument(
                "experiment-replication pairs require distinct IDs");
        if (!experimentIds.insert(armA).second ||
            !experimentIds.insert(armB).second)
            throw std::invalid_argument(
                "experiment-replication IDs must be globally unique");
    }
}

} // namespace

std::vector<std::pair<long long, long long>> ParseExperimentIdPairs(
    std::string_view text)
{
    if (text.empty())
        throw std::invalid_argument(
            "--compare-experiment-replications requires at least two A:B pairs");
    std::vector<std::pair<long long, long long>> result;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t end = text.find(',', start);
        const std::string_view token = text.substr(
            start, end == std::string_view::npos
                ? std::string_view::npos : end - start);
        if (token.empty())
            throw std::invalid_argument(
                "experiment-replication membership contains an empty pair");
        result.push_back(ExperimentPairComparison::ParseExperimentIdPair(
            token));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    ValidatePairs(result);
    return result;
}

int RunComparisonCommand(
    const ComparisonCommand& command,
    const ExperimentPairComparison::EvidenceSource& source,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        ValidatePairs(command.experimentPairs);
        std::vector<ExperimentPairComparison::ComparisonResult> pairs;
        pairs.reserve(command.experimentPairs.size());
        for (const auto& [armAId, armBId] : command.experimentPairs)
        {
            const auto armAEvidence = source.Load(armAId);
            const auto armBEvidence = source.Load(armBId);
            const auto request = ExperimentPairComparison::MakeComparisonRequest(
                armAEvidence, armBEvidence);
            pairs.push_back(ExperimentPairComparison::Compare(
                ExperimentPairComparison::MakeArmResultSet(armAEvidence),
                ExperimentPairComparison::MakeArmResultSet(armBEvidence),
                request));
        }
        output << Render(Compare(std::move(pairs)));
        return 0;
    }
    catch (const ExperimentPairComparison::EvidenceUnavailableError& error)
    {
        errors << "EXPERIMENT_REPLICATION_COMPARISON_LOAD_FAILED"
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "EXPERIMENT_REPLICATION_COMPARISON_LOAD_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::ExperimentReplicationComparison
