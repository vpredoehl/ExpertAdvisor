#include "../Sources/CausalFibonacciIncrementalInformationAnalysis.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

namespace A = EA::CausalFibonacciIncrementalInformation::Analysis;
namespace F = EA::CausalFibonacciIncrementalInformation;

int main()
{
    // Frozen development-bin boundaries remain fixed for later partitions.
    {
        const std::vector<double> edges{1.0, 2.0, 3.0};

        assert(A::FrozenBinIndex(edges, -100.0) == 0);
        assert(A::FrozenBinIndex(edges, 1.0) == 0);
        assert(A::FrozenBinIndex(edges, 2.0) == 1);
        assert(A::FrozenBinIndex(edges, 3.0) == 2);
        assert(A::FrozenBinIndex(edges, 100.0) == 2);

        bool emptyRejected = false;
        try { (void)A::FrozenBinIndex({}, 1.0); }
        catch (const std::invalid_argument&) { emptyRejected = true; }
        assert(emptyRejected);
    }

    // L-BFGS termination diagnostics distinguish every optimizer exit that
    // can make an otherwise valid multinomial fit unavailable.
    {
        const auto gradient = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                return std::pair<double, std::vector<double>>{
                    (x[0] - 3.0) * (x[0] - 3.0), {2.0 * (x[0] - 3.0)}};
            });
        assert(gradient.converged);
        assert(gradient.terminationReason == F::LbfgsTerminationReason::GradientInfinityTolerance);
        assert(gradient.iterations == 1);
        assert(gradient.initialObjective == 9.0 && gradient.finalObjective == 0.0);
        assert(gradient.finalGradientInfinityNorm == 0.0);
        assert(gradient.finalRelativeObjectiveChange == 1.0);
        assert(gradient.finalAcceptedStepSize == 0.5);
        assert(gradient.terminatingLineSearchAttempts == 0);

        const auto relative = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                return std::pair<double, std::vector<double>>{1e12 + 1e-6 * x[0], {1e-6}};
            });
        assert(relative.converged);
        assert(relative.terminationReason == F::LbfgsTerminationReason::RelativeObjectiveTolerance);
        assert(relative.iterations == 1);
        assert(relative.finalGradientInfinityNorm == 1e-6);
        assert(relative.finalRelativeObjectiveChange == 0.0);
        assert(relative.finalAcceptedStepSize == 1.0);
        assert(relative.terminatingLineSearchAttempts == 1);

        const auto maximum = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                return std::pair<double, std::vector<double>>{
                    (x[0] - 3.0) * (x[0] - 3.0), {2.0 * (x[0] - 3.0)}};
            }, {1, 1e-8, 1e-12, true});
        assert(!maximum.converged);
        assert(maximum.terminationReason == F::LbfgsTerminationReason::MaximumIterations);
        assert(maximum.iterations == 1);
        assert(maximum.finalObjective == 0.0 && maximum.finalGradientInfinityNorm == 0.0);
        assert(maximum.finalAcceptedStepSize == 0.5);
        assert(maximum.terminatingLineSearchAttempts == 2);
        assert(maximum.trajectory.size() == 2);
        assert(maximum.trajectory[0].iterations == 0 && maximum.trajectory[1].iterations == 1);
        assert(maximum.trajectory[1].objective == 0.0 && maximum.trajectory[1].historySize == 1);

        const auto lineSearch = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                return std::pair<double, std::vector<double>>{x[0] == 0.0 ? 0.0 : 1.0, {1.0}};
            });
        assert(!lineSearch.converged);
        assert(lineSearch.terminationReason == F::LbfgsTerminationReason::ArmijoLineSearchFailure);
        assert(lineSearch.iterations == 0 && lineSearch.initialObjective == 0.0 && lineSearch.finalObjective == 0.0);
        assert(lineSearch.finalGradientInfinityNorm == 1.0);
        assert(!std::isfinite(lineSearch.finalAcceptedStepSize));
        assert(lineSearch.terminatingLineSearchAttempts > 0);

        const auto nonFinite = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                if (x[0] == 0.0) return std::pair<double, std::vector<double>>{0.0, {1.0}};
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::pair<double, std::vector<double>>{nan, {nan}};
            });
        assert(!nonFinite.converged);
        assert(nonFinite.terminationReason == F::LbfgsTerminationReason::NonFiniteObjectiveOrGradient);
        assert(nonFinite.iterations == 0 && nonFinite.initialObjective == 0.0 && nonFinite.finalObjective == 0.0);
        assert(nonFinite.finalGradientInfinityNorm == 1.0);
        assert(!std::isfinite(nonFinite.finalAcceptedStepSize));
        assert(nonFinite.terminatingLineSearchAttempts > 0);

        // Continuation trajectory capture is diagnostic-only and extends at
        // deterministic sparse points without changing optimizer decisions.
        assert(F::LbfgsTrajectoryCadence(300));
        assert(F::LbfgsTrajectoryCadence(400));
        assert(F::LbfgsTrajectoryCadence(500));
        assert(F::LbfgsTrajectoryCadence(750));
        assert(F::LbfgsTrajectoryCadence(1000));
        assert(F::LbfgsTrajectoryCadence(5000));
        const auto continuation = F::OptimizeLbfgs(std::vector<double>{0.0},
            [](const std::vector<double>& x) {
                return std::pair<double, std::vector<double>>{1e6 - x[0], {-1.0}};
            }, {500, 1e-8, 1e-12, true});
        assert(!continuation.converged);
        assert(continuation.terminationReason == F::LbfgsTerminationReason::MaximumIterations);
        assert(continuation.iterations == 500);
        assert(continuation.finalHistorySize == 0);
        assert(continuation.finalDirectionalDerivative == -1.0);
        assert(std::any_of(continuation.trajectory.begin(), continuation.trajectory.end(),
                           [](const auto& point) { return point.iterations == 300; }));
        assert(std::any_of(continuation.trajectory.begin(), continuation.trajectory.end(),
                           [](const auto& point) { return point.iterations == 400; }));
        assert(std::any_of(continuation.trajectory.begin(), continuation.trajectory.end(),
                           [](const auto& point) { return point.iterations == 500; }));
    }

    // The exact FitMultiRows objective has the expected three-class gradient,
    // regularizes coefficients but not class intercepts, and is invariant to
    // a common intercept shift.
    {
        std::vector<A::ParsedRow> rows(6);
        const std::array<double, 6> values{{-2.0, -1.0, 0.5, 1.5, 2.0, 3.0}};
        const std::array<int, 6> classes{{0, 1, 2, 0, 1, 2}};
        for (std::size_t i = 0; i < rows.size(); ++i) {
            rows[i].baseline[0] = static_cast<float>(values[i]);
            rows[i].h4.assignedClass = classes[i];
            rows[i].h4.eligible = true;
        }
        const std::vector<std::size_t> selected{0, 1, 2, 3, 4, 5};
        const std::vector<A::FeatureRef> features{{false, 0, 0.0, 1.0, false}};
        const auto label = [](const A::ParsedRow& row) -> const A::ParsedTarget& { return row.h4; };
        const std::vector<double> beta{0.25, -0.4, -0.35, 0.3, 0.1, 0.2};
        const auto evaluated = A::MultiRowsObjective(rows, selected, features, label, beta);

        double maximumDiscrepancy = 0.0;
        constexpr double epsilon = 1e-6;
        for (std::size_t i = 0; i < beta.size(); ++i) {
            auto plus = beta, minus = beta;
            plus[i] += epsilon;
            minus[i] -= epsilon;
            const double finiteDifference =
                (A::MultiRowsObjective(rows, selected, features, label, plus).first -
                 A::MultiRowsObjective(rows, selected, features, label, minus).first) /
                (2.0 * epsilon);
            maximumDiscrepancy = std::max(maximumDiscrepancy, std::abs(finiteDifference - evaluated.second[i]));
        }
        assert(maximumDiscrepancy < 1e-6);
        std::cout << "FitMultiRows finite-difference maximum discrepancy=" << maximumDiscrepancy << '\n';

        double dataLoss = 0.0;
        std::vector<double> dataGradient(beta.size());
        for (const auto index : selected) {
            std::array<double, 3> probability{};
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < 3; ++c) {
                probability[c] = beta[c * 2] + beta[c * 2 + 1] * values[index];
                maximum = std::max(maximum, probability[c]);
            }
            double sum = 0.0;
            for (auto& value : probability) { value = std::exp(value - maximum); sum += value; }
            for (auto& value : probability) value /= sum;
            dataLoss -= std::log(probability[classes[index]]);
            for (std::size_t c = 0; c < 3; ++c) {
                const double error = probability[c] - (classes[index] == static_cast<int>(c));
                dataGradient[c * 2] += error;
                dataGradient[c * 2 + 1] += error * values[index];
            }
        }
        double expectedLoss = dataLoss;
        for (std::size_t c = 0; c < 3; ++c) {
            expectedLoss += beta[c * 2 + 1] * beta[c * 2 + 1];
            dataGradient[c * 2 + 1] += 2.0 * beta[c * 2 + 1];
        }
        assert(std::abs(evaluated.first - expectedLoss) < 1e-12);
        for (std::size_t i = 0; i < beta.size(); ++i)
            assert(std::abs(evaluated.second[i] - dataGradient[i]) < 1e-12);
        assert(std::abs(evaluated.second[0] + evaluated.second[2] + evaluated.second[4]) < 1e-12);

        auto shifted = beta;
        shifted[0] += 7.0; shifted[2] += 7.0; shifted[4] += 7.0;
        const auto shiftedObjective = A::MultiRowsObjective(rows, selected, features, label, shifted);
        assert(std::abs(shiftedObjective.first - evaluated.first) < 1e-12);
    }

    // Numeric Fibonacci bin edges use only eligible development targets for
    // their own horizon; later-partition values must not affect them.
    {
        std::vector<A::ParsedRow> rows(8);
        for (std::size_t i = 0; i < 5; ++i) {
            rows[i].partition = EA::CausalFibonacciIncrementalInformation::Partition::Development;
            rows[i].fibonacci[1] = static_cast<float>(i + 1);
            rows[i].h4.eligible = true;
            rows[i].h6.eligible = true;
        }
        rows[5].partition = EA::CausalFibonacciIncrementalInformation::Partition::Development;
        rows[5].fibonacci[1] = 100.0f;
        rows[5].h4.eligible = false;
        rows[5].h6.eligible = true;
        rows[6].partition = EA::CausalFibonacciIncrementalInformation::Partition::Validation;
        rows[6].fibonacci[1] = 200.0f;
        rows[6].h4.eligible = true;
        rows[6].h6.eligible = true;
        rows[7].partition = EA::CausalFibonacciIncrementalInformation::Partition::Pre2025LockTest;
        rows[7].fibonacci[1] = 300.0f;
        rows[7].h4.eligible = true;
        rows[7].h6.eligible = true;

        assert((A::FrozenDevelopmentBinEdges(rows, 0, false) == std::vector<double>{0.0, 1.0}));
        assert((A::FrozenDevelopmentBinEdges(rows, 1, false) == std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
        assert((A::FrozenDevelopmentBinEdges(rows, 1, true) == std::vector<double>{2.0, 3.0, 4.0, 5.0, 100.0}));

        rows[0].fibonacci[1] = 1.0f;
        rows[1].fibonacci[1] = 1.0f;
        rows[2].fibonacci[1] = 1.0f;
        rows[3].fibonacci[1] = 1.0f;
        rows[4].fibonacci[1] = 1.0f;
        assert(A::FrozenDevelopmentBinEdges(rows, 1, false).empty());
        assert((A::FrozenDevelopmentBinEdges(rows, 1, true) == std::vector<double>{1.0, 100.0}));

        const auto ledgerPath = std::filesystem::temp_directory_path() / "ea_fibonacci_horizon_bin_ledger.csv";
        std::filesystem::remove(ledgerPath);
        std::ofstream ledger(ledgerPath);
        A::WriteFibonacciLedgers(ledger, "fixture", rows);
        ledger.close();
        std::ifstream ledgerInput(ledgerPath);
        const std::string ledgerContents((std::istreambuf_iterator<char>(ledgerInput)), {});
        assert(ledgerContents.find("fixture,development,H4,1,,0,0,0,0,,,insufficient_distinct_development_values_after_tied_edge_collapse\n") != std::string::npos);
        assert(ledgerContents.find("fixture,development,H6,1,0,1,") != std::string::npos);
        std::filesystem::remove(ledgerPath);
    }

    // Ridge target standardization uses the exact development population SD.
    {
        std::vector<A::ParsedRow> rows(3);
        rows[0].fibonacci[1] = 1.0f;
        rows[1].fibonacci[1] = 2.0f;
        rows[2].fibonacci[1] = 3.0f;

        const std::vector<std::size_t> selected{0, 1, 2};
        const std::vector<A::FeatureRef> noFeatures;

        const auto model = A::FitRidge(rows, selected, noFeatures, 1);

        assert(model.available);
        assert(std::abs(model.targetMean - 2.0) < 1e-12);
        assert(std::abs(
            model.targetStandardDeviation - std::sqrt(2.0 / 3.0)
        ) < 1e-12);
    }

    const std::string confirmation =
        "ignored,audcadrmp,1735689600,1,confirmation_2025,99,0,0,nan,0,,99,0,0,nan,0,";
    assert(!A::ParsePre2025Row(confirmation).has_value());
    assert(!A::ParseDevelopmentOnlyRow(confirmation).has_value());

    const std::string validation =
        "ignored,audcadrmp,1546301700,1,validation,99,0,0,nan,0,,99,0,0,nan,0,";
    assert(!A::ParseDevelopmentOnlyRow(validation).has_value());

    const std::string development =
        "id,audcadrmp,1262304900,1,development,2,1262305800,1262308500,0.01,1,,1,1262305800,1262310300,0,1,";
    bool shortRejected = false;
    try { (void)A::ParsePre2025Row(development); }
    catch (const std::invalid_argument&) { shortRejected = true; }
    assert(shortRejected);

    const auto root = std::filesystem::temp_directory_path() / "ea_fibonacci_pre2025_analysis_fixture";
    const auto artifact = root / "artifact", output = root / "output";
    std::filesystem::remove_all(root);
    std::vector<EA::CausalFibonacciIncrementalInformation::Row> rows;
    const std::array<std::string, 6> symbols{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};
    for (const auto& symbol : symbols) {
        std::uint64_t ordinal = 0;
        for (const auto [start, count] : std::array<std::pair<std::int64_t, int>, 4>{{
            {EA::CausalFibonacciIncrementalInformation::kDevelopmentStart + 900, 6},
            {EA::CausalFibonacciIncrementalInformation::kValidationStart + 900, 3},
            {EA::CausalFibonacciIncrementalInformation::kPre2025Start + 900, 3},
            {EA::CausalFibonacciIncrementalInformation::kConfirmationStart + 900, 1}}}) {
            for (int n = 0; n < count; ++n, ++ordinal) {
                EA::CausalFibonacciIncrementalInformation::Row row;
                row.symbol = symbol; row.decisionTimestamp = start + n * 900; row.sourceRowOrdinal = ordinal;
                for (std::size_t i = 0; i < row.baseline.size(); ++i) row.baseline[i] = static_cast<float>(ordinal + i * .01);
                for (std::size_t i = 0; i < row.fibonacci.size(); ++i) row.fibonacci[i] = static_cast<float>((ordinal + i) % 5);
                row.fibonacci[0] = ordinal % 2;
                row.fibonacci[22] = 0.0f;
                const bool confirmation = start == EA::CausalFibonacciIncrementalInformation::kConfirmationStart + 900;
                row.h4 = {confirmation ? 99 : static_cast<int>(ordinal % 3), row.decisionTimestamp + 900, row.decisionTimestamp + 900, .001f, !confirmation, {}};
                row.h6 = {confirmation ? 99 : static_cast<int>((ordinal + 1) % 3), row.decisionTimestamp + 900, row.decisionTimestamp + 900, -.001f, !confirmation, {}};
                rows.push_back(row);
            }
        }
    }
    EA::CausalFibonacciIncrementalInformation::WriteArtifact(artifact,
        EA::CausalFibonacciIncrementalInformation::FixtureArtifactProvenance(), rows);

    // Scientific FitMultiRows continues to use frozen defaults.  The
    // qualification-only options are the separate 5,000-iteration path.
    const auto frozenOptions = A::AudcadH4BaselineMultinomialDiagnosticOptions(
        A::AudcadH4BaselineMultinomialDiagnosticMode::Frozen250);
    const auto continuationOptions = A::AudcadH4BaselineMultinomialDiagnosticOptions(
        A::AudcadH4BaselineMultinomialDiagnosticMode::Continuation5000);
    assert(F::LbfgsOptions{}.maxIterations == 250);
    assert(frozenOptions.maxIterations == 250);
    assert(continuationOptions.maxIterations == 5000);
    const auto qualificationOptions = A::SolverBudgetQualificationOptions();
    assert(qualificationOptions.maxIterations == 5000);
    assert(qualificationOptions.gradientInfinityTolerance == 1e-8);
    assert(qualificationOptions.relativeObjectiveTolerance == 1e-12);
    assert(qualificationOptions.captureTrajectory);
    assert(frozenOptions.gradientInfinityTolerance == continuationOptions.gradientInfinityTolerance);
    assert(frozenOptions.relativeObjectiveTolerance == continuationOptions.relativeObjectiveTolerance);
    assert(frozenOptions.captureTrajectory && continuationOptions.captureTrajectory);

    const auto grouped = A::LoadPre2025Rows(artifact);
    const auto& audcadRows = grouped.at("audcadrmp");
    const auto transform = A::FitDevelopmentInputTransform(audcadRows);
    const auto developmentRows = A::EligibleRows(audcadRows,
        EA::CausalFibonacciIncrementalInformation::Partition::Development,
        [](const auto& row) -> const A::ParsedTarget& { return row.h4; });
    const auto label = [](const A::ParsedRow& row) -> const A::ParsedTarget& { return row.h4; };
    const auto scientificModel = A::FitMultiRows(audcadRows, developmentRows, transform.baseline, label);
    const auto frozenDiagnosticModel = A::FitMultiRowsDiagnostic(
        audcadRows, developmentRows, transform.baseline, label, frozenOptions);
    assert(scientificModel.optimizer && frozenDiagnosticModel.optimizer);
    assert(scientificModel.optimizer->iterations <= 250);
    assert(scientificModel.optimizer->terminationReason == frozenDiagnosticModel.optimizer->terminationReason);
    assert(scientificModel.optimizer->iterations == frozenDiagnosticModel.optimizer->iterations);
    assert(scientificModel.optimizer->initialObjective == frozenDiagnosticModel.optimizer->initialObjective);
    assert(scientificModel.optimizer->finalObjective == frozenDiagnosticModel.optimizer->finalObjective);
    assert(scientificModel.optimizer->finalGradientInfinityNorm == frozenDiagnosticModel.optimizer->finalGradientInfinityNorm);
    assert(scientificModel.available == frozenDiagnosticModel.available);
    assert(scientificModel.unavailableReason == frozenDiagnosticModel.unavailableReason);

    std::ostringstream diagnostic;
    A::RunAudcadH4BaselineMultinomialDiagnostic(artifact, diagnostic);
    const std::string diagnosticContents = diagnostic.str();
    assert(diagnosticContents.find("symbol=audcadrmp,horizon=H4,feature_set=baseline") != std::string::npos);
    assert(diagnosticContents.find("diagnostic_iteration_ceiling=250") != std::string::npos);
    assert(diagnosticContents.find("feature_count=") != std::string::npos);
    assert(diagnosticContents.find("termination_reason=") != std::string::npos);
    assert(diagnosticContents.find("iterations=") != std::string::npos);
    assert(diagnosticContents.find("initial_objective=") != std::string::npos);
    assert(diagnosticContents.find("final_objective=") != std::string::npos);
    assert(diagnosticContents.find("final_gradient_infinity_norm=") != std::string::npos);
    assert(diagnosticContents.find("final_relative_objective_change=") != std::string::npos);
    assert(diagnosticContents.find("final_accepted_step_size=") != std::string::npos);
    assert(diagnosticContents.find("terminating_line_search_attempts=") != std::string::npos);
    assert(diagnosticContents.find("FIBONACCI_INCREMENTAL_MULTINOMIAL_CONDITIONING") != std::string::npos);
    assert(diagnosticContents.find("FIBONACCI_INCREMENTAL_MULTINOMIAL_TRAJECTORY symbol=audcadrmp,horizon=H4,feature_set=baseline,iterations=0,") != std::string::npos);
    assert(diagnosticContents.find("confirmation_2025=sealed") != std::string::npos);
    std::ostringstream continuationDiagnostic;
    A::RunAudcadH4BaselineMultinomialDiagnostic(
        artifact, continuationDiagnostic,
        A::AudcadH4BaselineMultinomialDiagnosticMode::Continuation5000);
    const std::string continuationContents = continuationDiagnostic.str();
    assert(continuationContents.find("diagnostic_iteration_ceiling=5000") != std::string::npos);
    assert(continuationContents.find("final_history_size=") != std::string::npos);
    assert(continuationContents.find("final_directional_derivative=") != std::string::npos);
    assert(continuationContents.find("convergence_criterion=") != std::string::npos);
    assert(continuationContents.find("objective_at_iteration_250=") != std::string::npos);
    assert(continuationContents.find("objective_improvement_from_iteration_250=") != std::string::npos);
    assert(continuationContents.find("confirmation_2025=sealed") != std::string::npos);

    // Qualification is exactly the canonical 6 x 2 x 2 development-only
    // population. It emits only deterministic solver diagnostics and no
    // predictive partitions, metrics, deltas, or confirmation content.
    const auto developmentOnly = A::LoadDevelopmentOnlyRows(artifact);
    assert(developmentOnly.size() == 6);
    for (const auto& [symbol, developmentRowsOnly] : developmentOnly) {
        assert(std::find(symbols.begin(), symbols.end(), symbol) != symbols.end());
        assert(developmentRowsOnly.size() == 6);
        for (const auto& row : developmentRowsOnly)
            assert(row.partition == F::Partition::Development);
    }
    std::ostringstream qualification, repeatedQualification;
    A::RunSolverBudgetQualification(artifact, qualification);
    A::RunSolverBudgetQualification(artifact, repeatedQualification);
    const std::string qualificationContents = qualification.str();
    assert(qualificationContents == repeatedQualification.str());
    constexpr std::string_view fitPrefix = "FIBONACCI_INCREMENTAL_SOLVER_BUDGET_FIT ";
    std::size_t fitCount = 0, offset = 0;
    while ((offset = qualificationContents.find(fitPrefix, offset)) != std::string::npos) {
        ++fitCount;
        offset += fitPrefix.size();
    }
    assert(fitCount == 24);
    for (const auto& symbol : symbols)
        for (const std::string_view horizon : {"H4", "H6"})
            for (const std::string_view featureSet : {"baseline", "augmented"})
                assert(qualificationContents.find(
                    "symbol=" + symbol + ",horizon=" + std::string(horizon) +
                    ",feature_set=" + std::string(featureSet) +
                    ",development_row_count=6,") != std::string::npos);
    for (const std::string_view field : {
             "retained_feature_count=", "converged=", "termination_reason=",
             "iteration_count=", "initial_objective=", "final_objective=",
             "final_gradient_infinity_norm=", "final_relative_objective_change=",
             "final_accepted_step=", "terminating_line_search_attempts=",
             "history_size=", "final_directional_derivative=",
             "objective_at_iteration_250=",
             "objective_improvement_after_iteration_250="})
        assert(qualificationContents.find(field) != std::string::npos);
    assert(qualificationContents.find("FIBONACCI_INCREMENTAL_SOLVER_BUDGET_SUMMARY converged=") != std::string::npos);
    assert(qualificationContents.find("reaching_5000=") != std::string::npos);
    assert(qualificationContents.find("termination_reason_counts=") != std::string::npos);
    for (const std::string_view forbidden : {
             "validation", "pre2025_lock_test", "confirmation_2025", "log_loss",
             "brier", "accuracy", "monthly", "cross_symbol", "delta"})
        assert(qualificationContents.find(forbidden) == std::string::npos);
    A::Run({artifact, output, "fixture-analysis"});
    std::ifstream conditional(output / "conditional_incremental.csv");
    const std::string contents((std::istreambuf_iterator<char>(conditional)), {});
    assert(contents.find("confirmation_2025") == std::string::npos);
    assert(std::filesystem::exists(output / "cross_symbol_equal_summary.csv"));
    std::ifstream coverage(output / "coverage_degeneracy.csv");
    const std::string coverageContents((std::istreambuf_iterator<char>(coverage)), {});
    assert(coverageContents.starts_with("symbol,partition,fibonacci_column,scale_valid_numerator,scale_valid_denominator,scale_valid_prevalence,nonzero_count_numerator,nonzero_count_denominator,nonzero_count_prevalence,finite_value_count,finite_value_denominator,target_variance,reconstruction_unavailable_reason\n"));
    assert(coverageContents.find("audcadrmp,development,0,3,6,0.5,,,,6,6,,\n") != std::string::npos);
    assert(coverageContents.find("audcadrmp,development,1,,,,5,6,0.83333333333333337,6,6,1.805555555555556,\n") != std::string::npos);
    assert(coverageContents.find("audcadrmp,validation,1,,,,3,3,1,3,3,0.66666666666666663,\n") != std::string::npos);
    assert(coverageContents.find("audcadrmp,development,22,,,,,,,6,6,0,zero_variance_development_target\n") != std::string::npos);
    assert(coverageContents.find("confirmation_2025") == std::string::npos);
    std::ifstream reconstruction(output / "reconstruction.csv");
    const std::string reconstructionContents((std::istreambuf_iterator<char>(reconstruction)), {});
    assert(reconstructionContents.find("audcadrmp,development,22,ridge,0,zero_variance_development_target,,,") != std::string::npos);
    std::ifstream sums(output / "sha256sums.txt");
    const std::string sumsContents((std::istreambuf_iterator<char>(sums)), {});
    assert(sumsContents.find(EA::CausalFibonacciIncrementalInformation::FileSha256(output / "coverage_degeneracy.csv") + "  coverage_degeneracy.csv\n") != std::string::npos);
    std::filesystem::remove_all(root);
}
