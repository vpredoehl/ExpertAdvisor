#include "../Sources/StrategyEvaluationCore/Phase19BPostEntryPathMechanismExtractor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

Strategy::PredictionProbabilities Probabilities(int predictedClass,
                                                float directional = 1.0f)
{
    if (predictedClass == Profitability::kNeutralClass)
        return {{0.2f, 0.6f, 0.2f}};
    const float remainder = (1.0f - directional) / 2.0f;
    return predictedClass == Profitability::kDownClass
        ? Strategy::PredictionProbabilities{{directional, remainder, remainder}}
        : Strategy::PredictionProbabilities{{remainder, remainder, directional}};
}

Strategy::MarketPathPoint Point(float open, float high, float low, float close)
{
    return {0, std::nullopt, open, high, low, close};
}

Strategy::MarketPathProvenance Provenance(long long modelId,
                                          std::string symbol,
                                          std::size_t horizon,
                                          std::string start = "2025-01-01",
                                          std::string end = "2026-01-01")
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "phase19b_synthetic_fixture";
    value.adapterVersion = 1;
    value.modelId = modelId;
    value.inferenceScientificIdentityCanonical =
        "phase19b_test_inference_v1;model=" + std::to_string(modelId) + ';';
    value.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            value.inferenceScientificIdentityCanonical);
    value.symbol = std::move(symbol);
    value.inferenceWindowSize = 3;
    value.predictionHorizon = horizon;
    value.evaluationStart = std::move(start);
    value.evaluationEnd = std::move(end);
    value.barIntervalSeconds = 900;
    value.timestampSemantics = "synthetic_utc_v1";
    value.ohlcIntervalSemantics = "synthetic_ask_ohlc_v1";
    value.priceDomain = "ask";
    value.marketDataSource = "immutable_synthetic_fixture_v1";
    value.marketDataSourceRelation = "fixture";
    value.pathOrdering = "source_row_ascending_v1";
    value.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    value.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();
    return value;
}

Strategy::StrategyEvaluationObservation Observation(
    std::uint64_t ordinal,
    int predictedClass,
    std::vector<Strategy::MarketPathPoint> points,
    float directionalProbability = 1.0f)
{
    const std::uint64_t startRow = ordinal * 100;
    const std::int64_t startTime =
        1'700'000'000 + static_cast<std::int64_t>(ordinal) * 100'000;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        points[index].sourceRow = startRow + 3 + index;
        points[index].timestampUnixSeconds = startTime +
            static_cast<std::int64_t>((index + 1) * 900);
    }
    Strategy::StrategyEvaluationObservation value;
    value.observationOrdinal = ordinal;
    value.inferenceWindowStartRow = startRow;
    value.decisionRow = startRow + 2;
    value.terminalRow = startRow + 2 + points.size();
    value.predictedClass = predictedClass;
    value.decisionClose = 100.0f;
    value.terminalClose = points.back().close;
    value.probabilities = Probabilities(
        predictedClass, directionalProbability);
    value.decisionTimestampUnixSeconds = startTime;
    value.terminalTimestampUnixSeconds =
        points.back().timestampUnixSeconds;
    value.marketPath = std::move(points);
    return value;
}

Strategy::AuthoritativeMarketPath H4Fixture(long long modelId = 1745,
                                             std::string symbol = "audchfrmp")
{
    std::vector<Strategy::StrategyEvaluationObservation> observations;
    // Saved: fixed is touched, extension survives, then the path recovers.
    observations.push_back(Observation(0, Profitability::kUpClass, {
        Point(100.0f, 100.0f, 99.89f, 99.90f),
        Point(99.90f, 100.0f, 99.88f, 100.0f),
        Point(100.0f, 100.11f, 100.0f, 100.10f),
        Point(100.10f, 100.21f, 100.05f, 100.20f)}));
    // Harmed short; both stops touch on the first bar using the existing
    // direction-specific OHLC touch convention.
    observations.push_back(Observation(1, Profitability::kDownClass, {
        Point(100.0f, 100.14f, 100.0f, 100.13f),
        Point(100.13f, 100.18f, 100.05f, 100.12f),
        Point(100.12f, 100.20f, 100.02f, 100.11f),
        Point(100.11f, 100.16f, 100.01f, 100.10f)}));
    // Unchanged and no fixed breach. Entry equality is an exact boundary and
    // does not invent an adverse-extremum timestamp.
    observations.push_back(Observation(2, Profitability::kUpClass, {
        Point(100.0f, 100.04f, 100.0f, 100.02f),
        Point(100.02f, 100.06f, 100.0f, 100.04f),
        Point(100.04f, 100.08f, 100.0f, 100.02f),
        Point(100.02f, 100.05f, 100.0f, 100.0f)}));
    // Fixed stop on bar 1, extension on bar 2, entry recovery on bar 3.
    observations.push_back(Observation(3, Profitability::kUpClass, {
        Point(100.0f, 100.0f, 99.89f, 99.92f),
        Point(99.92f, 99.95f, 99.86f, 99.90f),
        Point(99.90f, 100.0f, 99.88f, 99.99f),
        Point(99.99f, 100.03f, 99.97f, 100.01f)}));
    // Actionable but below the frozen activation boundary; it must not be
    // emitted even when its post-entry path is volatile.
    observations.push_back(Observation(4, Profitability::kUpClass, {
        Point(100.0f, 101.0f, 99.0f, 100.0f),
        Point(100.0f, 101.0f, 99.0f, 100.0f),
        Point(100.0f, 101.0f, 99.0f, 100.0f),
        Point(100.0f, 101.0f, 99.0f, 100.0f)}, 0.60f));
    return Strategy::BuildAuthoritativeMarketPath(
        Provenance(modelId, std::move(symbol), 4),
        std::move(observations));
}

Strategy::AuthoritativeMarketPath H14Fixture()
{
    std::vector<Strategy::MarketPathPoint> points;
    points.reserve(14);
    for (std::size_t index = 0; index < 14; ++index)
    {
        const float high = index == 6 ? 100.08f : 100.04f;
        points.push_back(Point(100.0f, high, 100.0f, 100.01f));
    }
    return Strategy::BuildAuthoritativeMarketPath(
        Provenance(1692, "AUDCAD", 14),
        {Observation(0, Profitability::kUpClass, std::move(points))});
}

Strategy::Phase19BExtractionContext Context()
{
    return {42, true};
}

std::string Error(const auto& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

void TestOutcomeClassesAndPairing()
{
    const auto path = H4Fixture();
    const auto result =
        Strategy::ExtractPhase19BPostEntryPathMechanism(path, Context());
    assert(result.totalObservationCount == 5);
    assert(result.totalActionableCount == 5);
    assert(result.activatedCount == 4);
    assert(result.savedCount == 1);
    assert(result.harmedCount == 2);
    assert(result.unchangedCount == 1);
    assert(result.savedCount + result.harmedCount + result.unchangedCount ==
           result.activatedCount);
    assert(result.observations[0].outcomeClass ==
           Strategy::Phase19BOutcomeClass::saved);
    assert(result.observations[1].outcomeClass ==
           Strategy::Phase19BOutcomeClass::harmed);
    assert(result.observations[2].outcomeClass ==
           Strategy::Phase19BOutcomeClass::unchanged);
    assert(result.observations[0].extensionMinusFixedReturn > 0.0);
    assert(result.observations[1].extensionMinusFixedReturn < 0.0);
    assert(result.observations[2].extensionMinusFixedReturn == 0.0);
    assert(result.observations[0].activated);
    assert(result.observations[0].normalizedDirectionalConfidence > 0.5);
}

void TestExcursionsStopsAndRecovery()
{
    const auto result = Strategy::ExtractPhase19BPostEntryPathMechanism(
        H4Fixture(), Context());
    const auto& saved = result.observations[0];
    assert(saved.maximumAdverseExcursion > 0.001);
    assert(saved.maximumFavorableExcursion > 0.001);
    assert(saved.maximumAdversePathPointOrdinal == 1);
    assert(saved.maximumFavorablePathPointOrdinal == 3);
    assert(saved.fixedStopHit);
    assert(!saved.extensionStopHit);
    assert(saved.fixedStopPathPointOrdinal == 0);
    assert(!saved.extensionStopPathPointOrdinal);
    assert(saved.barsFromEntryToFixedStop == 1);
    assert(saved.fixedStopHorizonFraction == 0.25);
    assert(saved.barsRemainingAfterFixedStop == 3);
    assert(saved.recoveredToEntry);
    assert(saved.recoveryToEntryPathPointOrdinal == 1);
    assert(saved.barsFromFixedStopToRecovery == 1);
    assert(saved.recoveryToEntryHorizonFraction == 0.5);
    assert(saved.achievedFavorableBaseStopAfterFixedBreach);
    assert(saved.favorableBaseStopPathPointOrdinal == 2);
    assert(saved.requiredExtraRoomBeforeRecovery);
    assert(*saved.requiredExtraRoomBeforeRecovery > 0.0);
    assert(saved.adverseExcursionBeyondFixedStop > 0.0);

    const auto& shortHarmed = result.observations[1];
    assert(shortHarmed.maximumAdverseExcursion > 0.0);
    assert(shortHarmed.maximumFavorableExcursion == 0.0);
    assert(shortHarmed.maximumAdversePathPointOrdinal == 2);
    assert(!shortHarmed.maximumFavorablePathPointOrdinal);
    assert(shortHarmed.fixedStopPathPointOrdinal == 0);
    assert(shortHarmed.extensionStopPathPointOrdinal == 0);
    assert(!shortHarmed.recoveredToEntry);
    assert(!shortHarmed.recoveryToEntryPathPointOrdinal);
    assert(!shortHarmed.requiredExtraRoomBeforeRecovery);

    const auto& unchanged = result.observations[2];
    assert(!unchanged.fixedStopHit);
    assert(!unchanged.extensionStopHit);
    assert(!unchanged.fixedStopPathPointOrdinal);
    assert(!unchanged.fixedStopHorizonFraction);
    assert(!unchanged.barsRemainingAfterFixedStop);
    assert(!unchanged.recoveredToEntry);
    assert(!unchanged.recoveryToEntryPathPointOrdinal);
    assert(!unchanged.requiredExtraRoomBeforeRecovery);
    assert(unchanged.maximumAdverseExcursion == 0.0);
    assert(!unchanged.maximumAdversePathPointOrdinal);
    assert(unchanged.terminalDirectionalReturn == 0.0);
    assert(unchanged.terminalDirectionalClass == "neutral");

    const auto& later = result.observations[3];
    assert(later.fixedStopPathPointOrdinal == 0);
    assert(later.extensionStopPathPointOrdinal == 1);
    assert(later.barsFromEntryToExtensionStop == 2);
    assert(later.extensionStopHorizonFraction == 0.5);
    assert(later.recoveredToEntry);
    assert(later.recoveryToEntryPathPointOrdinal == 2);
}

void TestHorizonNormalizationAndTrace()
{
    const auto h4 = Strategy::ExtractPhase19BPostEntryPathMechanism(
        H4Fixture(), Context());
    assert(h4.pathTrace.size() == 16);
    assert(h4.pathTrace[0].pathPointOrdinal == 0);
    assert(h4.pathTrace[0].horizonFraction == 0.25);
    assert(h4.pathTrace[3].horizonFraction == 1.0);
    assert(h4.pathTrace[4].directionalAdverseExtremeLogReturn < 0.0);

    const auto h14 = Strategy::ExtractPhase19BPostEntryPathMechanism(
        H14Fixture(), Context());
    assert(h14.predictionHorizon == 14);
    assert(h14.pathTrace.size() == 14);
    assert(h14.pathTrace[6].horizonFraction == 0.5);
    assert(h14.observations[0].maximumFavorablePathPointOrdinal == 6);
    assert(h14.observations[0].maximumFavorableHorizonFraction == 0.5);
    assert(!h14.observations[0].fixedStopHorizonFraction);
}

void TestReadOnlyFrozenIdentityAndDiagnosticTag()
{
    Strategy::ValidatePhase19BInvocation(
        {true, true, false, false, false, false, false});
    assert(Error([] {
        Strategy::ValidatePhase19BInvocation(
            {true, true, false, true, false, false, false});
    }) ==
        "phase19b_path_mechanism_requires_standalone_single_model_inference");
    assert(Error([] {
        Strategy::ExtractPhase19BPostEntryPathMechanism(
            H4Fixture(), {42, false});
    }) == "phase19b_read_only_transaction_not_enforced");
    assert(Error([] {
        Strategy::ExtractPhase19BPostEntryPathMechanism(
            H4Fixture(1745, "EURCHF"), Context());
    }) == "phase19b_frozen_model_symbol_mismatch");

    const auto diagnostic =
        Strategy::ExtractPhase19BPostEntryPathMechanism(
            H4Fixture(1805, "eurusdrmp"), Context());
    assert(diagnostic.cohortRole == "diagnostic");
    assert(diagnostic.metadataTsv.find(
        "\tdiagnostic_model_pooled\tfalse\n") != std::string::npos);
}

void TestDeterministicArtifactsAndNullEncoding()
{
    const auto path = H4Fixture();
    const auto first =
        Strategy::ExtractPhase19BPostEntryPathMechanism(path, Context());
    const auto repeated =
        Strategy::ExtractPhase19BPostEntryPathMechanism(path, Context());
    assert(first.resultHash == repeated.resultHash);
    assert(first.metadataTsv == repeated.metadataTsv);
    assert(first.observationsTsv == repeated.observationsTsv);
    assert(first.pathTraceTsv == repeated.pathTraceTsv);
    assert(first.metadataTsv.find("\trequires_read_only_transaction\ttrue\n") !=
           std::string::npos);
    assert(first.metadataTsv.find("\tproduction_rows_modified\tfalse\n") !=
           std::string::npos);
    assert(first.observationsTsv.find(
        "phase19b_post_entry_path_mechanism_v1\t1\t") !=
           std::string::npos);
    // Undefined optional timing values serialize as adjacent delimiters,
    // never as a substituted numeric zero.
    assert(first.observationsTsv.find("\t\t") != std::string::npos);
    assert(first.pathTraceTsv.find(first.observations[0].observationId) !=
           std::string::npos);
}

} // namespace

int main()
{
    TestOutcomeClassesAndPairing();
    TestExcursionsStopsAndRecovery();
    TestHorizonNormalizationAndTrace();
    TestReadOnlyFrozenIdentityAndDiagnosticTag();
    TestDeterministicArtifactsAndNullEncoding();
}
