#include <cassert>
#include <string>
#include <vector>

#include "../Sources/ExperimentRecommendationService.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationSource Source(long long id,
                            const char* symbol,
                            int horizon,
                            double leader,
                            double accuracy,
                            long long evidence)
{
    RecommendationSource source;
    source.experimentId = id;
    source.invocation.configuration.symbol = symbol;
    source.invocation.configuration.predictionHorizon = horizon;
    source.leaderScore = leader;
    source.inferenceAccuracy = accuracy;
    source.evidenceCount = evidence;
    return source;
}

std::vector<long long> SelectedIds(
    const RecommendationSourceSelectionResult& selection)
{
    std::vector<long long> ids;
    for (const auto& record : selection.selected)
        ids.push_back(record.source.experimentId);
    return ids;
}

} // namespace

int main()
{
    assert(RecommendationMachineText("").empty());
    assert(RecommendationMachineText("NULL") == "%4E%55%4C%4C");
    assert(RecommendationMachineText("a,b=c%\n\r") ==
           "a%2Cb%3Dc%25%0A%0D");
    assert(RecommendationMachineText(std::string{"\xC3\xA9", 2}) ==
           "%C3%A9");

    RecommendationPolicy policy;
    policy.topSourcesPerScope = 2;

    const std::vector<RecommendationSource> sources{
        Source(8, "eurusd", 12, 0.7, 0.8, 100),
        Source(4, "eurusd", 12, 0.7, 0.8, 100),
        Source(3, "eurusd", 12, 0.7, 0.8, 90),
        Source(9, "eurusd", 24, 0.9, 0.7, 100),
        Source(5, "gbpusd", 12, 0.8, 0.9, 100)};

    policy.sourceScope = RecommendationSourceScope::symbolHorizon;
    const auto bySymbolHorizon = SelectRecommendationSources(policy, sources);
    assert((SelectedIds(bySymbolHorizon) ==
            std::vector<long long>{4, 8, 9, 5}));
    assert(bySymbolHorizon.skipped.size() == 1);
    assert(bySymbolHorizon.skipped[0].first == 3);
    assert(bySymbolHorizon.skipped[0].second ==
           "outside_top_sources_per_scope");

    policy.sourceScope = RecommendationSourceScope::symbol;
    const auto bySymbol = SelectRecommendationSources(policy, sources);
    assert((SelectedIds(bySymbol) == std::vector<long long>{9, 4, 5}));
    assert(bySymbol.skipped.size() == 2);

    policy.sourceScope = RecommendationSourceScope::global;
    const auto global = SelectRecommendationSources(policy, sources);
    assert((SelectedIds(global) == std::vector<long long>{9, 5}));
    assert(global.skipped.size() == 3);

    // Input order cannot affect grouping, ranking, or overflow selection.
    const std::vector<RecommendationSource> reversed(sources.rbegin(),
                                                       sources.rend());
    const auto globalReversed = SelectRecommendationSources(policy, reversed);
    assert(SelectedIds(global) == SelectedIds(globalReversed));
    assert(global.skipped == globalReversed.skipped);

    return 0;
}
