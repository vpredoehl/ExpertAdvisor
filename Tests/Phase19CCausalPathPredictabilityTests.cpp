#include "../Sources/InferenceProfitability.hpp"
#include "../Sources/StrategyEvaluationCore/Phase19CCausalPathPredictability.hpp"

#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Strategy = EA::StrategyEvaluation;

namespace
{

const std::string kHeader =
    "schema_identity\tschema_version\tmodel_id\texperiment_id\tsymbol\tcohort_role\tprediction_horizon\twindow_start\twindow_end\tobservation_id\tobservation_ordinal\tentry_source_row\tentry_timestamp_unix_seconds\tpredicted_class\tdirection\tdirectional_probability\tnormalized_directional_confidence\tactivation_status\tfixed_base_stop_log_distance\textension_minus_fixed_return\toutcome_class\tfixed_stop_hit\tmaximum_adverse_excursion\tterminal_directional_return\tcomparison_semantics\n";

std::string Observations()
{
    return kHeader +
        "phase19b_post_entry_path_mechanism_v1\t1\t1745\t610\tAUDCHF\tprimary\t4\t2025-01-01\t2026-01-01\ta\t0\t9\t100\t2\tlong\t0.80000001192092896\t0.70000001788139343\ttrue\t0.001\t0.1\tSAVED\ttrue\t0.0011\t0.2\texact_binary64_delta_sign_v1\n"
        "phase19b_post_entry_path_mechanism_v1\t1\t1745\t610\tAUDCHF\tprimary\t4\t2025-01-01\t2026-01-01\tb\t1\t10\t101\t0\tshort\t0.89999997615814209\t0.84999996423721313\ttrue\t0.001\t-0.1\tHARMED\ttrue\t0.0013\t-0.2\texact_binary64_delta_sign_v1\n"
        "phase19b_post_entry_path_mechanism_v1\t1\t1745\t610\tAUDCHF\tprimary\t4\t2025-01-01\t2026-01-01\tc\t2\t11\t102\t2\tlong\t0.75\t0.625\ttrue\t0.001\t0\tUNCHANGED\tfalse\t0.0004\t0\texact_binary64_delta_sign_v1\n";
}

std::string Metadata(const std::string& observations)
{
    const auto hash = EA::InferenceProfitability::DeterministicHash(observations);
    const auto line = [](const std::string& key, const std::string& value) {
        return "phase19b_post_entry_path_mechanism_v1\t1\t" + key + '\t' + value + '\n';
    };
    return "schema_identity\tschema_version\tkey\tvalue\n" +
        line("strategy_identity", "probability_conditioned_stop_extension_v1") +
        line("mapping_identity", "directional_probability_one_sided_stop_extension_v1") +
        line("comparison_semantics", "exact_binary64_delta_sign_v1") +
        line("base_stop_logarithmic_distance", "0.001") +
        line("activation_confidence", "0.5") +
        line("minimum_stop_multiplier", "1") +
        line("maximum_stop_multiplier", "1.25") +
        line("model_id", "1745") + line("experiment_id", "610") +
        line("symbol", "AUDCHF") + line("cohort_role", "primary") +
        line("prediction_horizon", "4") +
        line("window_start", "2025-01-01") +
        line("window_end", "2026-01-01") +
        line("activated_count", "3") + line("saved_count", "1") +
        line("harmed_count", "1") + line("unchanged_count", "1") +
        line("observations_content_hash", hash) +
        line("result_hash", "fnv1a64:source") +
        line("requires_read_only_transaction", "true") +
        line("production_rows_modified", "false");
}

std::vector<Strategy::Phase19CPredictorIdentity> Predictors()
{
    return {{"causal_one", "authoritative_model_input_tensor_decision_row",
             "completed_decision_row_at_or_before_entry", 0, false},
            {"causal_two", "authoritative_model_input_return_suffix",
             "computed_from_completed_closes_ending_at_decision_row", 1,
             false}};
}

std::vector<Strategy::Phase19CCausalEntryState> States()
{
    return {
        {0, 9, 100, 2, "long", 0.80000001192092896,
         0.70000001788139343, {1.0, 2.0}},
        {1, 10, 101, 0, "short", 0.89999997615814209,
         0.84999996423721313, {3.0, 4.0}},
        {2, 11, 102, 2, "long", 0.75, 0.625, {5.0, 6.0}},
    };
}

Strategy::Phase19CCausalDatasetContext Context()
{
    return {1745, 610, "AUDCHF", "primary", 4, "2025-01-01",
            "2026-01-01", 2, 5, "", true};
}

std::string Error(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

void TestTargetPreservationAndLeakageSeparation()
{
    const auto observations = Observations();
    const auto result = Strategy::BuildPhase19CCausalDataset(
        observations, Metadata(observations), States(), Predictors(), Context());
    assert(result.activatedCount == 3);
    assert(result.savedCount == 1);
    assert(result.harmedCount == 1);
    assert(result.unchangedCount == 1);
    assert(result.joinedCount == 3);
    assert(result.joinFailureCount == 0);
    assert(result.datasetTsv.find("predictor__causal_one") != std::string::npos);
    assert(result.datasetTsv.find("outcome__maximum_adverse_excursion") !=
           std::string::npos);
    assert(result.datasetTsv.find("predictor__maximum_adverse_excursion") ==
           std::string::npos);
    assert(result.metadataTsv.find("\tleakage_audit_passed\ttrue\n") !=
           std::string::npos);
    assert(result.metadataTsv.find("\tunchanged_retained_not_classified\ttrue\n") !=
           std::string::npos);
}

void TestHardJoinFailures()
{
    const auto observations = Observations();
    auto missing = States();
    missing.pop_back();
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), missing, Predictors(), Context());
    }) == "phase19c_missing_entry_state_join");

    auto duplicate = States();
    duplicate.push_back(duplicate.front());
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), duplicate, Predictors(), Context());
    }) == "phase19c_duplicate_entry_state_ordinal");

    auto mismatch = States();
    mismatch[0].entryTimestampUnixSeconds = 101;
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), mismatch, Predictors(), Context());
    }) == "phase19c_nonunique_or_mismatched_entry_state_join");
}

void TestSemanticLayoutAndLeakageFailClosed()
{
    const auto observations = Observations();
    auto duplicate = Predictors();
    duplicate[1].name = duplicate[0].name;
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), States(), duplicate, Context());
    }) == "phase19c_feature_layout_identity_ambiguous");

    auto leaked = Predictors();
    leaked[1].name = "maximum_adverse_excursion";
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), States(), leaked, Context());
    }) == "phase19c_forbidden_predictor_leakage");

    auto unknownSource = Predictors();
    unknownSource[1].source = "unregistered_source";
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), States(), unknownSource,
            Context());
    }) == "phase19c_forbidden_predictor_leakage");

    auto invalidDecision = States();
    invalidDecision[0].direction = "short";
    assert(Error([&] {
        Strategy::BuildPhase19CCausalDataset(
            observations, Metadata(observations), invalidDecision,
            Predictors(), Context());
    }) == "phase19c_invalid_entry_decision_state");
}

void TestDeterministicRoundTripShape()
{
    const auto observations = Observations();
    const auto first = Strategy::BuildPhase19CCausalDataset(
        observations, Metadata(observations), States(), Predictors(), Context());
    const auto second = Strategy::BuildPhase19CCausalDataset(
        observations, Metadata(observations), States(), Predictors(), Context());
    assert(first.datasetHash == second.datasetHash);
    assert(first.resultHash == second.resultHash);
    assert(first.datasetTsv == second.datasetTsv);
    const auto headerEnd = first.datasetTsv.find('\n');
    const auto rowEnd = first.datasetTsv.find('\n', headerEnd + 1);
    const auto count = [](std::string_view row) {
        return 1 + static_cast<std::size_t>(
            std::count(row.begin(), row.end(), '\t'));
    };
    assert(count(std::string_view{first.datasetTsv}.substr(0, headerEnd)) ==
           count(std::string_view{first.datasetTsv}.substr(
               headerEnd + 1, rowEnd - headerEnd - 1)));
}

} // namespace

int main()
{
    Strategy::ValidatePhase19CInvocation(true, true, false, false, true);
    assert(Error([] {
        Strategy::ValidatePhase19CInvocation(true, true, false, false, false);
    }) == "phase19c_requires_explicit_phase19b_artifact_directory");
    TestTargetPreservationAndLeakageSeparation();
    TestHardJoinFailures();
    TestSemanticLayoutAndLeakageFailClosed();
    TestDeterministicRoundTripShape();
}
