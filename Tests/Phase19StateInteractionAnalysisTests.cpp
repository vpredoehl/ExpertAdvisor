#include "../Sources/StrategyEvaluationCore/Phase19StateInteractionAnalysis.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

std::string Error(const auto& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

Strategy::MarketPathProvenance Provenance(
    long long modelId = 1745,
    std::string symbol = "audchfrmp",
    std::uint64_t horizon = 4,
    std::string start = "2025-01-01",
    std::string end = "2026-01-01")
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "phase19_fixture";
    value.adapterVersion = 1;
    value.modelId = modelId;
    value.inferenceScientificIdentityCanonical =
        "phase19_fixture_inference_v1;";
    value.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            value.inferenceScientificIdentityCanonical);
    value.symbol = std::move(symbol);
    value.inferenceWindowSize = 3;
    value.predictionHorizon = horizon;
    value.evaluationStart = std::move(start);
    value.evaluationEnd = std::move(end);
    value.barIntervalSeconds = 900;
    value.timestampSemantics = "fixture_utc_v1";
    value.ohlcIntervalSemantics = "fixture_ask_ohlc_v1";
    value.priceDomain = "ask";
    value.marketDataSource = "immutable_fixture_v1";
    value.marketDataSourceRelation = "fixture";
    value.pathOrdering = "source_row_ascending_v1";
    value.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    value.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();
    return value;
}

Strategy::StrategyEvaluationObservation Observation(
    std::uint64_t ordinal, float directionalProbability, float futureLow,
    float terminalClose, int predictedClass = Profitability::kUpClass)
{
    const std::uint64_t row = ordinal * 10;
    const std::int64_t timestamp =
        1'700'000'000 + static_cast<std::int64_t>(ordinal) * 10'000;
    const float other = (1.0f - directionalProbability) / 2.0f;
    Strategy::PredictionProbabilities probabilities{{
        other, other, directionalProbability}};
    if (predictedClass == Profitability::kNeutralClass)
        probabilities = {{0.2f, 0.6f, 0.2f}};
    Strategy::StrategyEvaluationObservation value;
    value.observationOrdinal = ordinal;
    value.inferenceWindowStartRow = row;
    value.decisionRow = row + 2;
    value.terminalRow = row + 6;
    value.predictedClass = predictedClass;
    value.decisionClose = 100.0f;
    value.terminalClose = terminalClose;
    value.probabilities = probabilities;
    value.decisionTimestampUnixSeconds = timestamp;
    value.terminalTimestampUnixSeconds = timestamp + 3600;
    value.marketPath = {
        {row + 3, timestamp + 900, 100.0f, 100.2f, futureLow, 100.05f},
        {row + 4, timestamp + 1800, 100.05f, 100.2f, 99.95f, 100.05f},
        {row + 5, timestamp + 2700, 100.05f, 100.2f, 99.95f, 100.05f},
        {row + 6, timestamp + 3600, 100.05f, 120.0f, 99.95f,
         terminalClose}};
    return value;
}

Strategy::AuthoritativeMarketPath Path(
    Strategy::MarketPathProvenance provenance = Provenance())
{
    std::vector<Strategy::StrategyEvaluationObservation> observations;
    // One inactive actionable observation, eight activated observations, and
    // one non-actionable observation. Some paths distinguish widened from
    // fixed stops while retaining immutable Phase 18B execution semantics.
    observations.push_back(Observation(0, 0.60f, 99.80f, 100.10f));
    for (std::uint64_t ordinal = 1; ordinal <= 8; ++ordinal)
    {
        observations.push_back(Observation(
            ordinal, static_cast<float>(0.70 + 0.03 * ordinal),
            ordinal % 2 == 0 ? 99.89f : 99.70f,
            ordinal % 3 == 0 ? 99.95f : 100.15f));
    }
    observations.push_back(Observation(
        9, 0.60f, 90.0f, 110.0f, Profitability::kNeutralClass));
    return Strategy::BuildAuthoritativeMarketPath(
        std::move(provenance), std::move(observations));
}

std::vector<Strategy::Phase19EntryState> States()
{
    std::vector<Strategy::Phase19EntryState> result;
    for (std::uint64_t ordinal = 0; ordinal < 10; ++ordinal)
    {
        Strategy::Phase19EntryState state;
        state.observationOrdinal = ordinal;
        for (std::size_t feature = 0;
             feature < Strategy::kPhase19FeatureCount; ++feature)
            state.values[feature] = static_cast<double>(ordinal + feature);
        // Stable tied values at multiple quartile boundaries.
        state.values[0] = ordinal < 4 ? 1.0 :
            (ordinal < 7 ? 2.0 : 3.0);
        result.push_back(state);
    }
    return result;
}

} // namespace

int main()
{
    // Frozen Phase 18B parameters and identities are part of Phase 19's
    // compile-time contract and cannot be supplied by the CLI.
    static_assert(Strategy::kPhase18BDefaultBaseStopLogarithmicDistance ==
                  0.001);
    static_assert(Strategy::kPhase18BDefaultActivationConfidence == 0.50);
    static_assert(Strategy::kPhase18BMinimumStopMultiplier == 1.00);
    static_assert(Strategy::kPhase18BDefaultMaximumStopMultiplier == 1.25);

    const auto boundaries = Strategy::ComputePhase19QuartileBoundaries(
        {4, 1, 3, 2, 8, 7, 6, 5});
    assert(boundaries.q1 == 3.0);
    assert(boundaries.q2 == 5.0);
    assert(boundaries.q3 == 7.0);
    assert(Strategy::Phase19QuartileStratum(2.0, boundaries) == 0);
    assert(Strategy::Phase19QuartileStratum(3.0, boundaries) == 1);
    assert(Strategy::Phase19QuartileStratum(5.0, boundaries) == 2);
    assert(Strategy::Phase19QuartileStratum(7.0, boundaries) == 3);
    const auto tied = Strategy::ComputePhase19QuartileBoundaries(
        {1, 1, 1, 1, 2, 2, 2, 2});
    assert(tied.q1 == 1.0 && tied.q2 == 2.0 && tied.q3 == 2.0);
    assert(Strategy::Phase19QuartileStratum(1.0, tied) == 1);
    assert(Strategy::Phase19QuartileStratum(2.0, tied) == 3);

    const auto groups = Strategy::ComputePhase19CategoricalGroups(
        {"high", "low", "high", "medium"});
    assert((groups == std::vector<std::string>{"high", "low", "medium"}));
    assert(Strategy::Phase19CategoricalStratum("low", groups) == 1);

    const auto path = Path();
    const auto phase18B =
        Strategy::EvaluateControlledOneSidedStopExtensionExperiment(path);
    const auto first = Strategy::EvaluatePhase19StateInteractionAnalysis(
        path, States(), {true});
    const auto repeated = Strategy::EvaluatePhase19StateInteractionAnalysis(
        path, States(), {true});
    assert(first.resultHash == repeated.resultHash);
    assert(first.canonicalLines == repeated.canonicalLines);
    assert(first.cohortRole == Strategy::Phase19CohortRole::primary);
    assert(first.windowLabel ==
           Strategy::Phase19WindowLabel::discoveryHistory2025);
    assert(first.modelAcceptanceQualified);
    assert(first.requiresReadOnlyTransaction);
    assert(!first.productionRowsModified);
    assert(first.populations.size() == 4);
    assert(first.populations[0].fixed.actionableCount == 9);
    assert(first.populations[0].fixed.activatedCount == 8);
    assert(first.populations[1].fixed.actionableCount == 8);
    assert(first.populations[2].fixed.actionableCount +
               first.populations[3].fixed.actionableCount == 8);
    assert(first.stateStrata.size() == Strategy::kPhase19FeatureCount * 4);
    for (std::size_t feature = 0;
         feature < Strategy::kPhase19FeatureCount; ++feature)
    {
        std::uint64_t count = 0;
        for (std::size_t stratum = 0; stratum < 4; ++stratum)
        {
            const auto& item = first.stateStrata[feature * 4 + stratum];
            count += item.activatedCount;
            assert(item.fixed.actionableCount == item.activatedCount);
            assert(item.extension.actionableCount == item.activatedCount);
            assert(item.extensionMinusFixed.aggregateReturnDelta ==
                item.extension.aggregateDirectionalLogReturn -
                    item.fixed.aggregateDirectionalLogReturn);
            assert(item.extensionMinusFixed.stopHitRateDelta ==
                item.extension.stopHitRate.value_or(0.0) -
                    item.fixed.stopHitRate.value_or(0.0));
        }
        assert(count == 8);
    }
    assert(first.phase18BStrategyConfigurationHash ==
           phase18B.variants[3].evaluation.strategyIdentity.configurationHash);
    assert(first.canonicalLines.find("production_rows_modified=false") !=
           std::string::npos);
    assert(first.canonicalLines.find("outcomes_only_gating_prohibited") !=
           std::string::npos);
    assert(first.canonicalLines.find("maximum_adverse_excursion_outcome") !=
           std::string::npos);

    const auto later = Strategy::EvaluatePhase19StateInteractionAnalysis(
        Path(Provenance(1745, "audchfrmp", 4,
                        "2026-01-01", "2026-09-01")),
        States());
    assert(later.windowLabel ==
           Strategy::Phase19WindowLabel::temporalValidationHistory2026);
    assert(first.canonicalLines.find("window_label=discovery_history_2025") !=
           std::string::npos);
    assert(later.canonicalLines.find(
        "window_label=temporal_validation_history_2026") != std::string::npos);

    const auto diagnostic = Strategy::EvaluatePhase19StateInteractionAnalysis(
        Path(Provenance(1805, "eurusdrmp", 4)), States());
    assert(diagnostic.cohortRole == Strategy::Phase19CohortRole::diagnostic);
    assert(diagnostic.canonicalLines.find("cohort_role=diagnostic") !=
           std::string::npos);

    auto invalidStates = States();
    invalidStates.pop_back();
    assert(Error([&] {
        Strategy::EvaluatePhase19StateInteractionAnalysis(path, invalidStates);
    }) == "phase19_entry_state_population_mismatch");
    assert(Error([&] {
        Strategy::EvaluatePhase19StateInteractionAnalysis(
            Path(Provenance(9999, "audchfrmp", 4)), States());
    }) == "phase19_model_not_in_predeclared_cohort");
    assert(Error([&] {
        Strategy::EvaluatePhase19StateInteractionAnalysis(
            Path(Provenance(1745, "audchfrmp", 4,
                            "2025-02-01", "2026-01-01")), States());
    }) == "phase19_date_range_not_predeclared");

    Strategy::ValidatePhase19Invocation(
        {true, true, false, false, false, false, false});
    assert(Error([] {
        Strategy::ValidatePhase19Invocation(
            {true, true, true, false, false, false, false});
    }).find("standalone --infer") != std::string::npos);
    assert(Error([] {
        Strategy::ValidatePhase19Invocation(
            {true, true, false, true, false, false, false});
    }).find("no scheduler context") != std::string::npos);
    assert(Error([] {
        Strategy::ValidatePhase19Invocation(
            {true, true, false, false, true, false, false});
    }).find("no scheduler context") != std::string::npos);
    assert(Error([] {
        Strategy::ValidatePhase19Invocation(
            {true, true, false, false, false, true, false});
    }).find("no scheduler context") != std::string::npos);
    return 0;
}
