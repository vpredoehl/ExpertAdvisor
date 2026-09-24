#include "ExperimentReplicationComparison.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentReplicationComparison
{
namespace
{

using IdentityMap =
    std::map<std::string, std::optional<std::string>, std::less<>>;

std::string Number(double value)
{
    char buffer[128] {};
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
        throw std::runtime_error("experiment_replication_number_format_failed");
    return std::string(buffer, converted.ptr);
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

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

std::string Reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index != 0) output << '|';
        output << MachineText(reasons[index]);
    }
    return output.str();
}

void AddReason(std::vector<std::string>& reasons, std::string reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(std::move(reason));
}

IdentityMap MakeIdentityMap(const std::vector<Pair::IdentityField>& fields)
{
    IdentityMap result;
    for (const auto& field : fields)
    {
        if (!result.emplace(field.name, field.value).second)
            throw std::invalid_argument(
                "replication_pair_scientific_identity_duplicated");
    }
    return result;
}

bool PairScientificallyComparable(const Pair::ComparisonResult& pair)
{
    return pair.invalidReasons.empty() &&
        pair.unexpectedDifferences.empty() &&
        (pair.status == Pair::Status::ComparableComplete ||
         pair.status == Pair::Status::ComparableIncomplete);
}

bool IsMaterializationField(std::string_view field)
{
    static const std::set<std::string, std::less<>> fields {
        "model_input_width",
        "hidden_size",
        "layer_count",
        "window_size",
        "model_metadata_schema_version",
        "train_configuration_schema_version",
        "normalization_version",
        "class_weight_down",
        "class_weight_neutral",
        "class_weight_up",
        "optimizer_metadata_schema_version",
        "optimizer_type",
        "optimizer_first_moment_buffer_count",
        "optimizer_second_moment_buffer_count",
        "persisted_core_lr_mult",
        "persisted_head_weight_lr_mult",
        "persisted_head_bias_lr_mult",
        "label_rule_id",
        "target_type",
        "target_scale",
        "target_bias",
        "target_use_zscore",
        "target_mean",
        "target_standard_deviation",
        "model_input_metadata_schema_version",
        "model_input_semantic_layout_version",
        "persisted_training_symbol",
        "persisted_training_start",
        "persisted_training_end",
        "input_width_expansion_canonical",
        "training_execution_identity",
        "inference_execution_identity",
    };
    return fields.contains(field);
}

std::optional<std::string> RequiredConfiguredValue(
    const IdentityMap& identity,
    std::string_view field)
{
    const auto found = identity.find(field);
    if (found == identity.end() || !found->second ||
        *found->second == "NULL")
        return std::nullopt;
    return found->second;
}

void CompareIntervention(const Pair::ComparisonResult& reference,
                         const Pair::ComparisonResult& candidate,
                         std::size_t ordinal,
                         Result& result)
{
    if (reference.intentionalDifferences == candidate.intentionalDifferences)
        return;
    AddReason(result.compatibilityReasons,
              "pair_" + std::to_string(ordinal) +
                  "_intentional_intervention_mismatch");
    result.compatibility = Compatibility::Incompatible;
}

void CompareRoleIdentities(
    std::string_view role,
    const std::vector<IdentityMap>& identities,
    Result& result)
{
    std::set<std::string, std::less<>> names;
    for (const auto& identity : identities)
        for (const auto& [name, unused] : identity)
        {
            (void)unused;
            names.insert(name);
        }

    for (const std::string& name : names)
    {
        if (name == "fresh_initialization_seed") continue;
        if (name == "configured_model_input_width" ||
            name == "configured_model_input_semantic_layout_version")
            continue;

        std::optional<std::string> expected;
        bool unavailable = false;
        for (const auto& identity : identities)
        {
            const auto found = identity.find(name);
            if (found == identity.end() || !found->second)
            {
                unavailable = true;
                continue;
            }
            if (!expected) expected = *found->second;
            else if (*expected != *found->second)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_mismatch:" + name);
                result.compatibility = Compatibility::Incompatible;
            }
        }
        if (unavailable && !IsMaterializationField(name))
        {
            AddReason(result.compatibilityReasons,
                      std::string(role) + "_identity_unavailable:" + name);
            if (result.compatibility != Compatibility::Incompatible)
                result.compatibility =
                    Compatibility::UndeterminedDueToMissingEvidence;
        }
    }

    // These immutable experiment columns are authoritative before a final
    // model exists. A legacy NULL may fall back to final-model metadata, but
    // if neither source exists compatibility is genuinely undetermined.
    for (const auto& [configured, materialized] : {
             std::pair<std::string_view, std::string_view>{
                 "configured_model_input_width", "model_input_width"},
             {"configured_model_input_semantic_layout_version",
              "model_input_semantic_layout_version"}})
    {
        std::optional<std::string> expected;
        for (const auto& identity : identities)
        {
            auto value = RequiredConfiguredValue(identity, configured);
            if (!value)
            {
                const auto fallback = identity.find(materialized);
                if (fallback != identity.end()) value = fallback->second;
            }
            if (!value)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_unavailable:" +
                              std::string(configured));
                if (result.compatibility != Compatibility::Incompatible)
                    result.compatibility =
                        Compatibility::UndeterminedDueToMissingEvidence;
                continue;
            }
            if (!expected) expected = *value;
            else if (*expected != *value)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_mismatch:" +
                              std::string(configured));
                result.compatibility = Compatibility::Incompatible;
            }
        }
    }
}

std::optional<std::string> Seed(const IdentityMap& identity)
{
    const auto found = identity.find("fresh_initialization_seed");
    if (found == identity.end() || !found->second ||
        *found->second == "NULL")
        return std::nullopt;
    return found->second;
}

MetricAggregate AggregateMetric(
    const std::vector<Pair::ComparisonResult>& pairs,
    const Pair::MetricDefinition& definition)
{
    MetricAggregate result;
    result.name = definition.name;
    result.pairCount = pairs.size();
    result.pairDeltas.reserve(pairs.size());
    double sum = 0.0;
    for (const auto& pair : pairs)
    {
        const auto& delta = pair.*definition.member;
        result.pairDeltas.push_back(delta.armBMinusArmA);
        if (!delta.armBMinusArmA) continue;
        const double value = *delta.armBMinusArmA;
        if (!std::isfinite(value))
            throw std::invalid_argument("replication_metric_nonfinite");
        sum += value;
        if (!std::isfinite(sum))
            throw std::invalid_argument("replication_metric_sum_nonfinite");
        ++result.availableCount;
        if (!result.minimum || value < *result.minimum) result.minimum = value;
        if (!result.maximum || value > *result.maximum) result.maximum = value;
        if (value > 0.0) ++result.positiveCount;
        else if (value < 0.0) ++result.negativeCount;
        else ++result.zeroCount;
    }
    if (result.availableCount != 0)
        result.descriptiveMean =
            sum / static_cast<double>(result.availableCount);
    return result;
}

std::string Deltas(const std::vector<std::optional<double>>& values)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << OptionalNumber(values[index]);
    }
    return output.str();
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? MachineText(*value) : "NULL";
}

} // namespace

Result Compare(std::vector<Pair::ComparisonResult> pairs)
{
    if (pairs.size() < 2)
        throw std::invalid_argument(
            "experiment replication comparison requires at least two pairs");

    Result result;
    result.pairs = std::move(pairs);
    result.compatibility = Compatibility::Compatible;

    std::set<long long> experimentIds;
    std::vector<IdentityMap> armAIdentities;
    std::vector<IdentityMap> armBIdentities;
    armAIdentities.reserve(result.pairs.size());
    armBIdentities.reserve(result.pairs.size());

    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        const auto& pair = result.pairs[index];
        const std::size_t ordinal = index + 1;
        if (pair.experimentAId <= 0 || pair.experimentBId <= 0 ||
            pair.experimentAId == pair.experimentBId ||
            !experimentIds.insert(pair.experimentAId).second ||
            !experimentIds.insert(pair.experimentBId).second)
            throw std::invalid_argument(
                "replication experiment IDs must be globally unique");
        if (!PairScientificallyComparable(pair))
        {
            AddReason(result.compatibilityReasons,
                      "pair_" + std::to_string(ordinal) +
                          "_not_scientifically_comparable");
            result.compatibility = Compatibility::Incompatible;
        }
        if (index != 0)
            CompareIntervention(result.pairs.front(), pair, ordinal, result);
        armAIdentities.push_back(MakeIdentityMap(
            pair.armAScientificIdentity));
        armBIdentities.push_back(MakeIdentityMap(
            pair.armBScientificIdentity));
    }

    CompareRoleIdentities("arm_a", armAIdentities, result);
    CompareRoleIdentities("arm_b", armBIdentities, result);

    std::set<std::string> seeds;
    bool everySeedAvailable = true;
    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        ReplicationDimension dimension;
        dimension.pairOrdinal = index + 1;
        dimension.armASeed = Seed(armAIdentities[index]);
        dimension.armBSeed = Seed(armBIdentities[index]);
        if (!dimension.armASeed || !dimension.armBSeed)
            everySeedAvailable = false;
        else
        {
            seeds.insert(*dimension.armASeed);
            if (*dimension.armASeed != *dimension.armBSeed)
            {
                AddReason(result.compatibilityReasons,
                          "pair_" + std::to_string(index + 1) +
                              "_seed_mismatch");
                result.compatibility = Compatibility::Incompatible;
            }
        }
        result.replicationDimensions.push_back(std::move(dimension));
    }
    if (everySeedAvailable)
        result.seedReplicationMode = seeds.size() > 1
            ? SeedReplicationMode::DifferentSeeds
            : SeedReplicationMode::SameSeedRepeatedPairs;

    if (result.compatibility == Compatibility::Compatible)
        for (const auto& definition : Pair::MetricDefinitions())
            result.metrics.push_back(AggregateMetric(result.pairs, definition));
    return result;
}

std::string Render(const Result& result)
{
    std::ostringstream output;
    output << "EXPERIMENT_REPLICATION_COMPARISON"
           << ",version=1"
           << ",pair_count=" << result.pairs.size()
           << ",pair_order=argument_order"
           << ",arm_order=argument_order"
           << ",delta_sign_convention=arm_b_minus_arm_a"
           << ",aggregation=unweighted_descriptive_available_pair_deltas"
           << '\n';
    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        output << "EXPERIMENT_REPLICATION_PAIR"
               << ",ordinal=" << index + 1
               << ",experiment_a_id=" << result.pairs[index].experimentAId
               << ",experiment_b_id=" << result.pairs[index].experimentBId
               << ",status=" << Pair::StatusText(result.pairs[index].status)
               << '\n';
        output << Pair::RenderSummary(result.pairs[index]);
    }
    for (const auto& dimension : result.replicationDimensions)
        output << "EXPERIMENT_REPLICATION_DIMENSION"
               << ",pair_ordinal=" << dimension.pairOrdinal
               << ",field=fresh_initialization_seed"
               << ",arm_a=" << OptionalText(dimension.armASeed)
               << ",arm_b=" << OptionalText(dimension.armBSeed)
               << '\n';
    output << "EXPERIMENT_REPLICATION_COMPATIBILITY"
           << ",state=" << CompatibilityText(result.compatibility)
           << ",reasons=" << Reasons(result.compatibilityReasons)
           << ",seed_replication_mode="
           << SeedReplicationModeText(result.seedReplicationMode)
           << ",statistical_independence=not_inferred"
           << '\n';
    if (result.compatibility == Compatibility::Compatible)
    {
        for (const auto& metric : result.metrics)
            output << "EXPERIMENT_REPLICATION_METRIC"
                   << ",metric=" << metric.name
                   << ",n_pairs=" << metric.pairCount
                   << ",n_available=" << metric.availableCount
                   << ",pair_deltas=" << Deltas(metric.pairDeltas)
                   << ",descriptive_mean="
                   << OptionalNumber(metric.descriptiveMean)
                   << ",minimum=" << OptionalNumber(metric.minimum)
                   << ",maximum=" << OptionalNumber(metric.maximum)
                   << ",count_positive=" << metric.positiveCount
                   << ",count_zero=" << metric.zeroCount
                   << ",count_negative=" << metric.negativeCount
                   << '\n';
    }
    else
        output << "EXPERIMENT_REPLICATION_AGGREGATE"
               << ",suppressed=true"
               << ",reason=replication_compatibility_not_established\n";
    output << "EXPERIMENT_REPLICATION_RESULT"
           << ",compatibility=" << CompatibilityText(result.compatibility)
           << ",subjective_winner=NONE"
           << ",read_only=true\n";
    return output.str();
}

std::string CompatibilityText(Compatibility value)
{
    switch (value)
    {
        case Compatibility::Compatible: return "compatible";
        case Compatibility::Incompatible: return "incompatible";
        case Compatibility::UndeterminedDueToMissingEvidence:
            return "undetermined_due_to_missing_evidence";
    }
    throw std::invalid_argument("unknown replication compatibility");
}

std::string SeedReplicationModeText(SeedReplicationMode value)
{
    switch (value)
    {
        case SeedReplicationMode::DifferentSeeds:
            return "different_seed_replications";
        case SeedReplicationMode::SameSeedRepeatedPairs:
            return "same_seed_repeated_pairs";
        case SeedReplicationMode::UnavailableOrNotApplicable:
            return "unavailable_or_not_applicable";
    }
    throw std::invalid_argument("unknown seed replication mode");
}

} // namespace EA::ExperimentReplicationComparison
