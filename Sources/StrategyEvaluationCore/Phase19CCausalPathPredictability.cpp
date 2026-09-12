#include "Phase19CCausalPathPredictability.hpp"

#include "../InferenceProfitability.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace EA::StrategyEvaluation
{
namespace
{

using Row = std::vector<std::string>;

std::vector<std::string> Split(std::string_view line, char delimiter)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    do
    {
        const std::size_t end = line.find(delimiter, begin);
        fields.emplace_back(line.substr(
            begin, end == std::string_view::npos ? line.size() - begin
                                                  : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    } while (begin <= line.size());
    return fields;
}

std::vector<Row> ParseTsv(const std::string& content,
                          const char* emptyError,
                          const char* fieldError)
{
    std::vector<Row> rows;
    std::size_t begin = 0;
    while (begin < content.size())
    {
        const std::size_t end = content.find('\n', begin);
        std::string_view line{content.data() + begin,
                              (end == std::string::npos ? content.size() : end) - begin};
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (!line.empty()) rows.push_back(Split(line, '\t'));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (rows.empty()) throw std::runtime_error(emptyError);
    const std::size_t width = rows.front().size();
    if (width == 0) throw std::runtime_error(emptyError);
    for (const auto& row : rows)
        if (row.size() != width) throw std::runtime_error(fieldError);
    return rows;
}

std::unordered_map<std::string, std::size_t> HeaderMap(const Row& header)
{
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t index = 0; index < header.size(); ++index)
        if (!result.emplace(header[index], index).second)
            throw std::runtime_error("phase19c_duplicate_phase19b_column");
    return result;
}

const std::string& Require(const Row& row,
                           const std::unordered_map<std::string, std::size_t>& header,
                           const char* name)
{
    const auto found = header.find(name);
    if (found == header.end())
        throw std::runtime_error(
            std::string{"phase19c_required_phase19b_column_missing:"} + name);
    return row[found->second];
}

template <typename Integer>
Integer IntegerValue(const std::string& text, const char* field)
{
    Integer value{};
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::runtime_error(
            std::string{"phase19c_invalid_integer:"} + field);
    return value;
}

double DoubleValue(const std::string& text, const char* field)
{
    std::size_t consumed = 0;
    double value = 0.0;
    try { value = std::stod(text, &consumed); }
    catch (...) {
        throw std::runtime_error(
            std::string{"phase19c_invalid_double:"} + field);
    }
    if (consumed != text.size() || !std::isfinite(value))
        throw std::runtime_error(
            std::string{"phase19c_invalid_double:"} + field);
    return value;
}

std::string Decimal(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

std::map<std::string, std::string> Metadata(const std::string& content)
{
    const auto rows = ParseTsv(
        content, "phase19c_phase19b_metadata_empty",
        "phase19c_phase19b_metadata_field_count_mismatch");
    const auto header = HeaderMap(rows.front());
    std::map<std::string, std::string> result;
    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        if (Require(rows[index], header, "schema_identity") !=
                "phase19b_post_entry_path_mechanism_v1" ||
            Require(rows[index], header, "schema_version") != "1")
            throw std::runtime_error("phase19c_phase19b_metadata_schema_mismatch");
        const auto& key = Require(rows[index], header, "key");
        if (!result.emplace(key, Require(rows[index], header, "value")).second)
            throw std::runtime_error("phase19c_duplicate_phase19b_metadata_key");
    }
    return result;
}

const std::string& Meta(const std::map<std::string, std::string>& values,
                        const char* key)
{
    const auto found = values.find(key);
    if (found == values.end())
        throw std::runtime_error(
            std::string{"phase19c_required_phase19b_metadata_missing:"} + key);
    return found->second;
}

void RequireSafe(std::string_view value)
{
    if (value.find_first_of("\t\r\n") != std::string_view::npos)
        throw std::runtime_error("phase19c_tsv_unsafe_text");
}

bool IsAllowedPredictorProvenance(const Phase19CPredictorIdentity& predictor)
{
    return (predictor.source ==
                "authoritative_model_input_tensor_decision_row" &&
            predictor.causalTiming ==
                "completed_decision_row_at_or_before_entry") ||
        (predictor.source == "authoritative_model_input_return_suffix" &&
         predictor.causalTiming ==
             "computed_from_completed_closes_ending_at_decision_row");
}

void ValidateDecisionState(const Phase19CCausalEntryState& state)
{
    const bool isShort =
        state.predictedClass == InferenceProfitability::kDownClass &&
        state.direction == "short";
    const bool isLong =
        state.predictedClass == InferenceProfitability::kUpClass &&
        state.direction == "long";
    if (!isShort && !isLong)
        throw std::runtime_error("phase19c_invalid_entry_decision_state");
    if (!std::isfinite(state.directionalProbability) ||
        !std::isfinite(state.normalizedDirectionalConfidence) ||
        state.directionalProbability < 0.0 ||
        state.directionalProbability > 1.0 ||
        state.normalizedDirectionalConfidence < 0.0 ||
        state.normalizedDirectionalConfidence > 1.0)
        throw std::runtime_error("phase19c_invalid_entry_decision_state");
}

std::string LayoutCanonical(
    const std::vector<Phase19CPredictorIdentity>& predictors,
    const Phase19CCausalDatasetContext& context)
{
    std::ostringstream output;
    output << "phase19c_model_input_feature_layout_v1;model_input_width="
           << context.modelInputWidth << ";semantic_layout_version="
           << context.semanticLayoutVersion << ";feature_ablation_identity="
           << context.featureAblationIdentity << ';';
    for (const auto& predictor : predictors)
        output << predictor.modelInputColumn << ':' << predictor.name << ':'
               << predictor.source << ':' << predictor.causalTiming << ':'
               << (predictor.categorical ? "categorical" : "numeric") << ';';
    return output.str();
}

bool SameDouble(double left, double right)
{
    return std::bit_cast<std::uint64_t>(left) ==
        std::bit_cast<std::uint64_t>(right);
}

void RequireMetadataMatch(const std::map<std::string, std::string>& metadata,
                          const Phase19CCausalDatasetContext& context,
                          const std::string& observationsHash)
{
    if (Meta(metadata, "strategy_identity") !=
            "probability_conditioned_stop_extension_v1" ||
        Meta(metadata, "mapping_identity") !=
            "directional_probability_one_sided_stop_extension_v1" ||
        Meta(metadata, "comparison_semantics") !=
            "exact_binary64_delta_sign_v1" ||
        Meta(metadata, "base_stop_logarithmic_distance") != "0.001" ||
        Meta(metadata, "activation_confidence") != "0.5" ||
        Meta(metadata, "minimum_stop_multiplier") != "1" ||
        Meta(metadata, "maximum_stop_multiplier") != "1.25")
        throw std::runtime_error("phase19c_frozen_phase19b_contract_mismatch");
    if (IntegerValue<long long>(Meta(metadata, "model_id"), "model_id") !=
            context.modelId ||
        Meta(metadata, "symbol") != context.symbol ||
        Meta(metadata, "cohort_role") != context.cohortRole ||
        IntegerValue<std::uint64_t>(Meta(metadata, "prediction_horizon"),
                                    "prediction_horizon") !=
            context.predictionHorizon ||
        Meta(metadata, "window_start") != context.windowStart ||
        Meta(metadata, "window_end") != context.windowEnd)
        throw std::runtime_error("phase19c_phase19b_scope_mismatch");
    const std::string expectedExperimentId = context.experimentId
        ? std::to_string(*context.experimentId) : "NULL";
    if (Meta(metadata, "experiment_id") != expectedExperimentId)
        throw std::runtime_error("phase19c_phase19b_experiment_mismatch");
    if (Meta(metadata, "observations_content_hash") != observationsHash)
        throw std::runtime_error("phase19c_phase19b_observations_hash_mismatch");
    if (Meta(metadata, "requires_read_only_transaction") != "true" ||
        Meta(metadata, "production_rows_modified") != "false")
        throw std::runtime_error("phase19c_phase19b_read_only_contract_mismatch");
}

} // namespace

void ValidatePhase19CInvocation(bool inferenceMode,
                                bool hasExplicitModel,
                                bool inferAll,
                                bool schedulerContext,
                                bool hasArtifactDirectory)
{
    if (!inferenceMode || !hasExplicitModel || inferAll || schedulerContext)
        throw std::invalid_argument(
            "phase19c_requires_standalone_single_model_inference");
    if (!hasArtifactDirectory)
        throw std::invalid_argument(
            "phase19c_requires_explicit_phase19b_artifact_directory");
}

std::vector<Phase19CCausalEntryState> ParsePhase19BCausalJoinRequests(
    const std::string& phase19BObservationsTsv)
{
    const auto rows = ParseTsv(
        phase19BObservationsTsv, "phase19c_phase19b_observations_empty",
        "phase19c_phase19b_observation_field_count_mismatch");
    const auto header = HeaderMap(rows.front());
    std::vector<Phase19CCausalEntryState> result;
    result.reserve(rows.size() - 1);
    std::set<std::uint64_t> ordinals;
    std::set<std::string> identities;
    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        const Row& row = rows[index];
        if (Require(row, header, "schema_identity") !=
                "phase19b_post_entry_path_mechanism_v1" ||
            Require(row, header, "schema_version") != "1")
            throw std::runtime_error("phase19c_phase19b_observation_schema_mismatch");
        Phase19CCausalEntryState state;
        state.observationOrdinal = IntegerValue<std::uint64_t>(
            Require(row, header, "observation_ordinal"),
            "observation_ordinal");
        state.entrySourceRow = IntegerValue<std::uint64_t>(
            Require(row, header, "entry_source_row"), "entry_source_row");
        state.entryTimestampUnixSeconds = IntegerValue<std::int64_t>(
            Require(row, header, "entry_timestamp_unix_seconds"),
            "entry_timestamp_unix_seconds");
        state.predictedClass = IntegerValue<int>(
            Require(row, header, "predicted_class"), "predicted_class");
        state.direction = Require(row, header, "direction");
        state.directionalProbability = DoubleValue(
            Require(row, header, "directional_probability"),
            "directional_probability");
        state.normalizedDirectionalConfidence = DoubleValue(
            Require(row, header, "normalized_directional_confidence"),
            "normalized_directional_confidence");
        if (Require(row, header, "activation_status") != "true")
            throw std::runtime_error("phase19c_nonactivated_phase19b_observation");
        ValidateDecisionState(state);
        if (!ordinals.emplace(state.observationOrdinal).second ||
            !identities.emplace(std::to_string(state.entrySourceRow) + ':' +
                                std::to_string(state.entryTimestampUnixSeconds)).second)
            throw std::runtime_error("phase19c_duplicate_observation_identity");
        result.push_back(std::move(state));
    }
    return result;
}

Phase19CCausalDatasetResult BuildPhase19CCausalDataset(
    const std::string& phase19BObservationsTsv,
    const std::string& phase19BMetadataTsv,
    const std::vector<Phase19CCausalEntryState>& entryStates,
    const std::vector<Phase19CPredictorIdentity>& predictors,
    const Phase19CCausalDatasetContext& context)
{
    if (!context.readOnlyTransactionEnforced)
        throw std::runtime_error("phase19c_read_only_transaction_not_enforced");
    if (predictors.empty() || predictors.size() != context.modelInputWidth)
        throw std::runtime_error("phase19c_predictor_layout_count_mismatch");
    std::set<std::string> predictorNames;
    for (std::size_t index = 0; index < predictors.size(); ++index)
    {
        const auto& predictor = predictors[index];
        RequireSafe(predictor.name);
        RequireSafe(predictor.source);
        RequireSafe(predictor.causalTiming);
        if (predictor.modelInputColumn != index ||
            !predictorNames.emplace(predictor.name).second)
            throw std::runtime_error("phase19c_feature_layout_identity_ambiguous");
        if (!IsAllowedPredictorProvenance(predictor) ||
            predictor.source.find("post_entry") != std::string::npos ||
            predictor.causalTiming.find("post_entry") != std::string::npos ||
            predictor.name.find("maximum_adverse_excursion") != std::string::npos ||
            predictor.name.find("maximum_favorable_excursion") != std::string::npos ||
            predictor.name.find("stop_hit") != std::string::npos ||
            predictor.name.find("terminal") != std::string::npos ||
            predictor.name.find("recovery") != std::string::npos)
            throw std::runtime_error("phase19c_forbidden_predictor_leakage");
    }

    const std::string observationsHash =
        InferenceProfitability::DeterministicHash(phase19BObservationsTsv);
    const auto metadata = Metadata(phase19BMetadataTsv);
    RequireMetadataMatch(metadata, context, observationsHash);

    const auto rows = ParseTsv(
        phase19BObservationsTsv, "phase19c_phase19b_observations_empty",
        "phase19c_phase19b_observation_field_count_mismatch");
    const Row& sourceHeader = rows.front();
    const auto header = HeaderMap(sourceHeader);
    if (Require(sourceHeader, header, "schema_identity") != "schema_identity")
        throw std::runtime_error("phase19c_phase19b_header_invalid");

    std::map<std::uint64_t, const Phase19CCausalEntryState*> states;
    std::set<std::string> stateIdentities;
    for (const auto& state : entryStates)
    {
        ValidateDecisionState(state);
        if (state.predictorValues.size() != predictors.size())
            throw std::runtime_error("phase19c_entry_state_feature_count_mismatch");
        if (!states.emplace(state.observationOrdinal, &state).second)
            throw std::runtime_error("phase19c_duplicate_entry_state_ordinal");
        const std::string identity = std::to_string(state.entrySourceRow) + ':' +
            std::to_string(state.entryTimestampUnixSeconds);
        if (!stateIdentities.emplace(identity).second)
            throw std::runtime_error("phase19c_duplicate_entry_state_identity");
        for (double value : state.predictorValues)
            if (!std::isfinite(value))
                throw std::runtime_error("phase19c_nonfinite_predictor");
    }

    const std::string layoutCanonical = LayoutCanonical(predictors, context);
    const std::string layoutHash =
        InferenceProfitability::DeterministicHash(layoutCanonical);
    const std::string layoutIdentity =
        "model_input_semantic_layout_" +
        std::to_string(context.semanticLayoutVersion) + "_width_" +
        std::to_string(context.modelInputWidth);

    std::ostringstream dataset;
    dataset << "schema_identity\tschema_version\tmodel_id\texperiment_id\tsymbol"
            << "\tcohort_role\tprediction_horizon\twindow_start\twindow_end"
            << "\tobservation_id\tobservation_ordinal\tentry_source_row"
            << "\tentry_timestamp_unix_seconds\tfeature_layout_identity"
            << "\tfeature_layout_hash\tsource_phase19b_observations_hash"
            << "\tsource_phase19b_result_hash\tpredictor__predicted_class"
            << "\tpredictor__direction\tpredictor__directional_probability"
            << "\tpredictor__normalized_directional_confidence";
    for (const auto& predictor : predictors)
        dataset << "\tpredictor__" << predictor.name;

    const std::size_t firstOutcome = header.at("fixed_base_stop_log_distance");
    for (std::size_t index = firstOutcome; index < sourceHeader.size(); ++index)
        dataset << "\toutcome__" << sourceHeader[index];
    dataset << '\n';

    Phase19CCausalDatasetResult result;
    result.schemaIdentity = kPhase19CCausalDatasetSchemaIdentity;
    result.predictorCount = predictors.size() + 4;
    result.featureLayoutIdentity = layoutIdentity;
    result.featureLayoutHash = layoutHash;
    result.sourcePhase19BObservationsHash = observationsHash;
    result.sourcePhase19BResultHash = Meta(metadata, "result_hash");
    std::set<std::string> observationIds;
    std::set<std::uint64_t> observationOrdinals;
    for (std::size_t rowIndex = 1; rowIndex < rows.size(); ++rowIndex)
    {
        const Row& row = rows[rowIndex];
        if (Require(row, header, "schema_identity") !=
                "phase19b_post_entry_path_mechanism_v1" ||
            Require(row, header, "schema_version") != "1")
            throw std::runtime_error("phase19c_phase19b_observation_schema_mismatch");
        if (Require(row, header, "activation_status") != "true")
            throw std::runtime_error("phase19c_nonactivated_phase19b_observation");
        const auto ordinal = IntegerValue<std::uint64_t>(
            Require(row, header, "observation_ordinal"),
            "observation_ordinal");
        const auto found = states.find(ordinal);
        if (found == states.end())
            throw std::runtime_error("phase19c_missing_entry_state_join");
        const auto& state = *found->second;
        const std::string& observationId =
            Require(row, header, "observation_id");
        if (!observationIds.emplace(observationId).second ||
            !observationOrdinals.emplace(ordinal).second)
            throw std::runtime_error("phase19c_duplicate_observation_identity");
        const auto sourceRow = IntegerValue<std::uint64_t>(
            Require(row, header, "entry_source_row"), "entry_source_row");
        const auto timestamp = IntegerValue<std::int64_t>(
            Require(row, header, "entry_timestamp_unix_seconds"),
            "entry_timestamp_unix_seconds");
        const int predictedClass = IntegerValue<int>(
            Require(row, header, "predicted_class"), "predicted_class");
        const double probability = DoubleValue(
            Require(row, header, "directional_probability"),
            "directional_probability");
        const double confidence = DoubleValue(
            Require(row, header, "normalized_directional_confidence"),
            "normalized_directional_confidence");
        if (sourceRow != state.entrySourceRow || timestamp !=
                state.entryTimestampUnixSeconds || predictedClass !=
                state.predictedClass ||
            Require(row, header, "direction") != state.direction ||
            !SameDouble(probability, state.directionalProbability) ||
            !SameDouble(confidence, state.normalizedDirectionalConfidence))
            throw std::runtime_error("phase19c_nonunique_or_mismatched_entry_state_join");
        if (timestamp > state.entryTimestampUnixSeconds)
            throw std::runtime_error("phase19c_predictor_timestamp_after_entry");

        const std::string& outcome = Require(row, header, "outcome_class");
        if (outcome == "SAVED") ++result.savedCount;
        else if (outcome == "HARMED") ++result.harmedCount;
        else if (outcome == "UNCHANGED") ++result.unchangedCount;
        else throw std::runtime_error("phase19c_invalid_phase19b_outcome_class");

        dataset << result.schemaIdentity << '\t' << result.schemaVersion
                << '\t' << context.modelId << '\t'
                << (context.experimentId ? std::to_string(*context.experimentId)
                                         : "")
                << '\t' << context.symbol << '\t' << context.cohortRole
                << '\t' << context.predictionHorizon << '\t'
                << context.windowStart << '\t' << context.windowEnd << '\t'
                << observationId << '\t' << ordinal << '\t' << sourceRow
                << '\t' << timestamp << '\t' << layoutIdentity << '\t'
                << layoutHash << '\t' << observationsHash << '\t'
                << result.sourcePhase19BResultHash << '\t' << predictedClass
                << '\t' << state.direction << '\t' << Decimal(probability)
                << '\t' << Decimal(confidence);
        for (double value : state.predictorValues)
            dataset << '\t' << Decimal(value);
        for (std::size_t index = firstOutcome; index < row.size(); ++index)
            dataset << '\t' << row[index];
        dataset << '\n';
        ++result.joinedCount;
    }
    result.activatedCount = rows.size() - 1;
    if (result.savedCount + result.harmedCount + result.unchangedCount !=
            result.activatedCount || result.joinedCount != result.activatedCount)
        throw std::runtime_error("phase19c_outcome_or_join_accounting_mismatch");
    if (IntegerValue<std::uint64_t>(Meta(metadata, "activated_count"),
                                    "activated_count") != result.activatedCount ||
        IntegerValue<std::uint64_t>(Meta(metadata, "saved_count"),
                                    "saved_count") != result.savedCount ||
        IntegerValue<std::uint64_t>(Meta(metadata, "harmed_count"),
                                    "harmed_count") != result.harmedCount ||
        IntegerValue<std::uint64_t>(Meta(metadata, "unchanged_count"),
                                    "unchanged_count") != result.unchangedCount)
        throw std::runtime_error("phase19c_phase19b_count_mismatch");

    result.datasetTsv = dataset.str();
    result.datasetHash =
        InferenceProfitability::DeterministicHash(result.datasetTsv);

    std::ostringstream manifest;
    manifest << "schema_identity\tschema_version\tpredictor_name\tsource_category"
             << "\tcausal_timing\tmodel_input_column\tvalue_type"
             << "\tincluded_in_full_model\n";
    manifest << result.schemaIdentity << '\t' << result.schemaVersion
             << "\tpredicted_class\tfrozen_decision_state\tat_entry_decision\t"
             << "\tcategorical\ttrue\n"
             << result.schemaIdentity << '\t' << result.schemaVersion
             << "\tdirection\tfrozen_decision_state\tat_entry_decision\t"
             << "\tcategorical\ttrue\n"
             << result.schemaIdentity << '\t' << result.schemaVersion
             << "\tdirectional_probability\tfrozen_decision_state\tat_entry_decision\t"
             << "\tnumeric\ttrue\n"
             << result.schemaIdentity << '\t' << result.schemaVersion
             << "\tnormalized_directional_confidence\tfrozen_decision_state"
             << "\tat_entry_decision\t\tnumeric\ttrue\n";
    for (const auto& predictor : predictors)
        manifest << result.schemaIdentity << '\t' << result.schemaVersion
                 << '\t' << predictor.name << '\t' << predictor.source << '\t'
                 << predictor.causalTiming << '\t'
                 << predictor.modelInputColumn << '\t'
                 << (predictor.categorical ? "categorical" : "numeric")
                 << "\ttrue\n";
    result.predictorsTsv = manifest.str();

    std::ostringstream resultCanonical;
    resultCanonical << "phase19c_causal_dataset_result_v1;dataset_hash="
                    << result.datasetHash << ";feature_layout_hash="
                    << layoutHash << ";source_phase19b_result_hash="
                    << result.sourcePhase19BResultHash << ';';
    result.resultHash =
        InferenceProfitability::DeterministicHash(resultCanonical.str());

    std::ostringstream meta;
    meta << "schema_identity\tschema_version\tkey\tvalue\n";
    const auto add = [&](const char* key, const auto& value) {
        meta << result.schemaIdentity << '\t' << result.schemaVersion << '\t'
             << key << '\t' << value << '\n';
    };
    add("analysis_plan_identity", kPhase19CAnalysisPlanIdentity);
    add("model_id", context.modelId);
    add("experiment_id", context.experimentId
        ? std::to_string(*context.experimentId) : "NULL");
    add("symbol", context.symbol);
    add("cohort_role", context.cohortRole);
    add("prediction_horizon", context.predictionHorizon);
    add("window_start", context.windowStart);
    add("window_end", context.windowEnd);
    add("activated_count", result.activatedCount);
    add("saved_count", result.savedCount);
    add("harmed_count", result.harmedCount);
    add("unchanged_count", result.unchangedCount);
    add("joined_count", result.joinedCount);
    add("join_failure_count", result.joinFailureCount);
    add("usable_predictor_count", result.predictorCount);
    add("model_input_width", context.modelInputWidth);
    add("semantic_layout_version", context.semanticLayoutVersion);
    add("feature_layout_identity", layoutIdentity);
    add("feature_layout_hash", layoutHash);
    add("feature_ablation_identity", context.featureAblationIdentity);
    add("source_phase19b_observations_hash", observationsHash);
    add("source_phase19b_result_hash", result.sourcePhase19BResultHash);
    add("dataset_hash", result.datasetHash);
    add("result_hash", result.resultHash);
    add("target_comparison_semantics", "exact_binary64_delta_sign_v1");
    add("primary_target_population", "SAVED_vs_HARMED_only");
    add("unchanged_retained_not_classified", "true");
    add("predictor_state_semantics", "completed_entry_decision_row_and_history_only");
    add("post_entry_predictors_included", "false");
    add("leakage_audit_passed", "true");
    add("requires_read_only_transaction", "true");
    add("production_rows_modified", "false");
    result.metadataTsv = meta.str();
    return result;
}

} // namespace EA::StrategyEvaluation
