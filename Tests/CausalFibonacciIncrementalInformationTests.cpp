#include "CausalFibonacciIncrementalInformation.hpp"
#include "CanonicalMarketDataRange.hpp"

#include <cassert>
#include <iostream>

namespace F = EA::CausalFibonacciIncrementalInformation;

namespace {
struct SqlQuoter {
    std::string quote(const std::string& value) const { return "'" + value + "'"; }
    std::string quote(int value) const { return std::to_string(value); }
};
}

F::Row MakeRow(std::int64_t timestamp, std::uint64_t ordinal)
{
    F::Row row;
    row.symbol = "audcadrmp";
    row.decisionTimestamp = timestamp;
    row.sourceRowOrdinal = ordinal;
    for (std::size_t i = 0; i < row.baseline.size(); ++i) row.baseline[i] = static_cast<float>(i + ordinal);
    row.fibonacci[0] = 1.0f;
    row.fibonacci[1] = ordinal == 1 ? 0.0f : 1.0f;
    row.h4 = {2, timestamp + 900, timestamp + 3600, .01f, true, ""};
    row.h6 = {1, timestamp + 900, timestamp + 5400, .0f, true, ""};
    return row;
}

int main()
{
    F::VerifyFrozenProtocolDocument("docs/phases/target-generation/FibonacciExtensions/FIBONACCI_LAYOUT9_INCREMENTAL_INFORMATION_PROTOCOL.md");
    SqlQuoter quoter;
    const auto fullHistory = EA::CanonicalMarketData::CanonicalFullHistoryThroughCandlestickCte(
        quoter, "audcadrmp", PriceTP{std::chrono::seconds{F::kProtocolEnd}});
    assert(fullHistory.find("'-infinity'::timestamp") != std::string::npos);
    assert(fullHistory.find("2026-01-01 00:00:00+00") != std::string::npos);
    bool mismatch = false;
    try { F::VerifyFrozenProtocolDocument("Tests/CausalFibonacciIncrementalInformationTests.cpp"); }
    catch (const std::runtime_error& error) { mismatch = std::string(error.what()).find("expected=") != std::string::npos; }
    assert(mismatch);

    const auto schema = F::FrozenFeatureSchema();
    assert(schema.baseline.size() == 80 && schema.fibonacci.size() == 23);
    assert(schema.fibonacci.front().name == "fib_recent_price_scale_valid");

    auto first = MakeRow(F::kDevelopmentStart + 900, 1);
    auto second = MakeRow(F::kDevelopmentStart + 1800, 2);
    F::ValidateRows({first, second});
    assert(F::StateFor(first) == F::StructuralState::ScaleValidNoRecentEvent);
    assert(F::StateFor(second) == F::StructuralState::ScaleValidRecentEvent);
    first.fibonacci[0] = 0.0f;
    assert(F::StateFor(first) == F::StructuralState::ScaleInvalid);

    bool duplicate = false;
    try { F::ValidateRows({second, second}); }
    catch (const std::invalid_argument&) { duplicate = true; }
    assert(duplicate);
    auto edge = MakeRow(F::kValidationStart - 900, 9);
    edge.h4.terminalTimestamp = F::kValidationStart;
    bool crossingTarget = false;
    try { F::ValidateRows({edge}); } catch (const std::invalid_argument&) { crossingTarget = true; }
    assert(crossingTarget);

    F::ArtifactProvenance provenance = F::FixtureArtifactProvenance();
    F::ValidateArtifactProvenance(provenance);
    provenance.economicCalendarSnapshotId.clear();
    bool missingCalendar = false;
    try { F::ValidateArtifactProvenance(provenance); }
    catch (const std::invalid_argument&) { missingCalendar = true; }
    assert(missingCalendar);

    first = MakeRow(F::kDevelopmentStart + 900, 1);
    second = MakeRow(F::kDevelopmentStart + 1800, 2);
    const auto scale = F::FitDevelopmentBaselineScale("audcadrmp", {first, second}, schema);
    const auto scaled = F::ApplyBaselineScale(first, scale, schema);
    assert(!scaled.empty() && scale.fittedRows == 2 && scale.symbol == "audcadrmp");
    const auto artifact = std::filesystem::temp_directory_path() / "ea_fibonacci_incremental_fixture";
    provenance.economicCalendarSnapshotId = "snapshot";
    F::WriteFixtureArtifact(artifact, provenance, {first, second});
    const auto firstRowsHash = F::FileSha256(artifact / "rows.csv");
    F::WriteFixtureArtifact(artifact, provenance, {first, second});
    assert(firstRowsHash == F::FileSha256(artifact / "rows.csv"));
    assert(std::filesystem::exists(artifact / "manifest.json"));
    F::VerifyArtifactDirectory(artifact);
    { std::ofstream corrupt(artifact / "rows.csv", std::ios::app); corrupt << "corruption\n"; }
    bool corruptRejected = false; try { F::VerifyArtifactDirectory(artifact); } catch (const std::invalid_argument&) { corruptRejected = true; }
    assert(corruptRejected);
    F::WriteFixtureArtifact(artifact, provenance, {first, second});
    std::array<float, 103> physical{};
    for (std::size_t i = 0; i < physical.size(); ++i) physical[i] = static_cast<float>(i);
    const auto projected = F::ProjectAuthoritativeLayout9Row("audcadrmp", F::kDevelopmentStart + 900, 1, physical);
    assert(projected.baseline[0] == 0.0f && projected.baseline[75] == 75.0f && projected.baseline[76] == 99.0f && projected.fibonacci[0] == 76.0f && projected.fibonacci[22] == 98.0f);
    const auto ridge = F::EvaluateRidgeReconstruction({{0.0}, {1.0}, {2.0}}, {1.0, 3.0, 5.0},
                                                        {{3.0}, {4.0}}, {7.0, 9.0});
    assert(ridge.available);
    assert(std::abs(ridge.rmse - std::sqrt(26.0) / 3.0) < 1e-12);
    assert(std::abs(ridge.rSquared + 17.0 / 9.0) < 1e-12);
    // The unnormalised objective has beta=(5/3,4/3).  A row-normalised
    // penalty convention would not produce this frozen value.
    const auto ridgeConvention = F::EvaluateRidgeReconstruction(
        {{0.0}, {1.0}, {2.0}}, {1.0, 3.0, 5.0}, {{3.0}}, {17.0 / 3.0});
    assert(std::abs(ridgeConvention.rmse) < 1e-12);
    const auto constant = F::EvaluateRidgeReconstruction({{0.0}, {1.0}}, {0.0, 1.0}, {{2.0}}, {1.0});
    assert(constant.available && !std::isfinite(constant.rSquared) &&
           constant.unavailableReason == "zero_variance_holdout_target");
    const auto optimized = F::OptimizeLbfgs(std::vector<double>{0.0},
        [](const std::vector<double>& x) { return std::pair{(x[0]-3.0)*(x[0]-3.0), std::vector<double>{2.0*(x[0]-3.0)}}; });
    assert(optimized.converged && std::abs(optimized.parameters[0] - 3.0) < 1e-10);

    const std::vector<std::vector<double>> binaryX{{-2.0}, {-1.0}, {1.0}, {2.0}};
    const std::vector<int> binaryY{0, 0, 1, 1};
    const auto binary = F::FitBinaryLogistic(binaryX, binaryY);
    assert(binary.available);
    const auto binaryAgain = F::FitBinaryLogistic(binaryX, binaryY);
    assert(binaryAgain.available && binary.coefficients == binaryAgain.coefficients);
    const auto binaryMetrics = F::EvaluateBinaryLogistic(binary, binaryX, binaryY);
    assert(binaryMetrics.available && std::isfinite(binaryMetrics.logLoss) && std::isfinite(binaryMetrics.brier));
    const auto oneClass = F::FitBinaryLogistic({{1.0}, {2.0}}, {1, 1});
    assert(!oneClass.available && oneClass.unavailableReason == "development_one_class_binary_target");

    // The augmented feature identifies the class while the baseline is a
    // constant.  The paired sign must therefore favor augmentation.
    const std::vector<std::vector<double>> baselineX{{0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}};
    const std::vector<std::vector<double>> augmentedX{{-2.0}, {-1.0}, {0.0}, {0.0}, {1.0}, {2.0}};
    const std::vector<int> multiclassY{0, 0, 1, 1, 2, 2};
    const auto baselineModel = F::FitMultinomialDiagnostic(baselineX, multiclassY);
    const auto augmentedModel = F::FitMultinomialDiagnostic(augmentedX, multiclassY);
    assert(baselineModel.available && augmentedModel.available);
    const auto baselineMetrics = F::EvaluateMultinomialDiagnostic(baselineModel, baselineX, multiclassY);
    const auto augmentedMetrics = F::EvaluateMultinomialDiagnostic(augmentedModel, augmentedX, multiclassY);
    assert(baselineMetrics.available && augmentedMetrics.available);
    for (const auto& probability : augmentedMetrics.probabilities) {
        assert(std::isfinite(probability[0]) && std::isfinite(probability[1]) && std::isfinite(probability[2]));
        assert(std::abs(probability[0] + probability[1] + probability[2] - 1.0) < 1e-12);
    }
    const auto paired = F::BaselineMinusAugmented(baselineMetrics, augmentedMetrics);
    assert(paired.available && paired.logLoss > 0.0 && paired.brier > 0.0);
    const auto absentClass = F::FitMultinomialDiagnostic({{0.0}, {1.0}}, {0, 1});
    assert(!absentClass.available && absentClass.unavailableReason == "development_missing_directional_class");
    // Frozen Brier is the mean of the three squared probability errors.
    F::MultinomialModel fixed; fixed.available = true; fixed.featureCount = 0;
    fixed.parameters = {std::log(.2), std::log(.3), std::log(.5)};
    const auto fixedMetrics = F::EvaluateMultinomialDiagnostic(fixed, {{}}, {2});
    assert(std::abs(fixedMetrics.brier - ((.2*.2 + .3*.3 + .5*.5) / 3.0)) < 1e-12);

    auto bins = F::FitDevelopmentQuantileBins({first, second}, "audcadrmp", 1);
    assert(bins.available && F::QuantileBinFor(bins, 0.0));
    auto tiedA = MakeRow(F::kDevelopmentStart + 2700, 3); tiedA.fibonacci[1] = 7.0f;
    auto tiedB = MakeRow(F::kDevelopmentStart + 3600, 4); tiedB.fibonacci[1] = 7.0f;
    const auto tied = F::FitDevelopmentQuantileBins({tiedA, tiedB}, "audcadrmp", 1);
    assert(!tied.available && tied.unavailableReason.find("tied_edge") != std::string::npos);
    const auto ledger = F::StructuralLedger({first, second}, "audcadrmp", F::Partition::Development, false);
    assert(ledger[1].rows == 1 && ledger[2].rows == 1);
    const auto matrix = F::CompleteAssociationMatrix({first, second}, "audcadrmp", F::Partition::Development);
    assert(std::isfinite(matrix.scaleValidityPointBiserial[0]) || std::isnan(matrix.scaleValidityPointBiserial[0]));
    const auto monthly = F::MonthlyDependenceAwareDeltas({F::kValidationStart, F::kValidationStart + 900}, {.1, -.1});
    assert(monthly.size() == 1 && monthly[0].rows == 2 && std::abs(monthly[0].mean) < 1e-12);

    F::ConfirmationAuthorization confirmation;
    confirmation.artifactDirectory = artifact; confirmation.developmentFitProvenance = provenance;
    bool unauthorized = false; try { F::RequireConfirmation2025(confirmation); } catch (const std::invalid_argument&) { unauthorized = true; }
    assert(unauthorized);
    confirmation.explicitlyAuthorized = true; confirmation.frozenPre2025ResultAvailable = true; confirmation.noRefit = false;
    bool refitRejected = false; try { F::RequireConfirmation2025(confirmation); } catch (const std::invalid_argument& error) { refitRejected = std::string(error.what()).find("refit") != std::string::npos; }
    assert(refitRejected);
    confirmation.noRefit = true; F::RequireConfirmation2025(confirmation);
    std::filesystem::remove_all(artifact);
    std::cout << "CausalFibonacciIncrementalInformationTests passed\n";
}
