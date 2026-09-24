#include "ExperimentReplicationPlanning.hpp"

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentReplicationPlanning
{
namespace
{

using IdentityMap = std::map<std::string, std::optional<std::string>>;

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

void AddReason(std::vector<std::string>& reasons, std::string reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(std::move(reason));
}

IdentityMap Identities(const Pair::ArmResultSet& arm,
                       std::vector<std::string>* errors = nullptr)
{
    IdentityMap result;
    for (const Pair::IdentityField& field : arm.scientificIdentity)
    {
        if (field.name.empty())
        {
            if (errors) AddReason(*errors, "scientific_identity_name_empty");
            continue;
        }
        if (!result.emplace(field.name, field.value).second && errors)
            AddReason(*errors,
                      "scientific_identity_field_duplicated:" + field.name);
    }
    return result;
}

Pair::ArmResultSet ConfigurationOnly(Pair::ArmResultSet arm)
{
    arm.finalInferenceAvailable = false;
    arm.finalAnalysisAvailable = false;
    arm.profitabilityObservationAvailable = false;
    arm.metrics = {};
    return arm;
}

bool IsRequiredPlanningField(std::string_view name)
{
    // Producing-executable identity is evidence about a completed execution,
    // not an input accepted by experiment construction. Every other field in
    // ExperimentPairComparison's identity catalog is retained and required.
    return name != "training_execution_identity" &&
        name != "inference_execution_identity";
}

void ValidateRequiredIdentity(const Pair::ArmResultSet& arm,
                              std::string_view role,
                              std::vector<std::string>& reasons,
                              bool& missing)
{
    std::vector<std::string> errors;
    const IdentityMap identity = Identities(arm, &errors);
    for (const std::string& error : errors)
        AddReason(reasons, std::string(role) + '_' + error);
    if (!errors.empty()) missing = true;

    for (const auto& [name, value] : identity)
    {
        if (!IsRequiredPlanningField(name)) continue;
        if (!value)
        {
            AddReason(reasons, std::string(role) +
                      "_identity_unavailable:" + name);
            missing = true;
        }
    }
    for (const std::string_view required : {
             "symbol", "prediction_horizon", "target_epochs", "threshold",
             "train_start", "train_end", "inference_start", "inference_end",
             "feature_ablation_mask", "configured_model_input_width",
             "configured_model_input_semantic_layout_version",
             "training_objective_canonical", "training_objective_hash",
             "fresh_initialization_seed"})
    {
        const auto found = identity.find(std::string{required});
        if (found == identity.end() || !found->second)
        {
            AddReason(reasons, std::string(role) +
                      "_required_identity_unavailable:" +
                      std::string{required});
            missing = true;
        }
    }
}

std::optional<unsigned int> Seed(const Pair::ArmResultSet& arm)
{
    const IdentityMap identity = Identities(arm);
    const auto found = identity.find("fresh_initialization_seed");
    if (found == identity.end() || !found->second ||
        *found->second == "NULL")
        return std::nullopt;
    unsigned long long parsed = 0;
    const std::string& text = *found->second;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() ||
        parsed == 0 || parsed > std::numeric_limits<unsigned int>::max())
        return std::nullopt;
    return static_cast<unsigned int>(parsed);
}

void SetSeed(Pair::ArmResultSet& arm, unsigned int seed)
{
    std::size_t matches = 0;
    for (Pair::IdentityField& field : arm.scientificIdentity)
    {
        if (field.name != "fresh_initialization_seed") continue;
        field.value = std::to_string(seed);
        ++matches;
    }
    if (matches != 1)
        throw std::invalid_argument(
            "fresh_initialization_seed_identity_missing_or_duplicated");
}

std::vector<Pair::IdentityDifference> Differences(
    const Pair::ArmResultSet& source,
    const Pair::ArmResultSet& proposed)
{
    const IdentityMap left = Identities(source);
    const IdentityMap right = Identities(proposed);
    std::set<std::string> names;
    for (const auto& [name, unused] : left)
    {
        (void)unused;
        names.insert(name);
    }
    for (const auto& [name, unused] : right)
    {
        (void)unused;
        names.insert(name);
    }
    std::vector<Pair::IdentityDifference> result;
    for (const std::string& name : names)
    {
        const auto lhs = left.find(name);
        const auto rhs = right.find(name);
        const std::string lhsValue = lhs == left.end()
            ? "<MISSING>" : lhs->second.value_or("<UNAVAILABLE>");
        const std::string rhsValue = rhs == right.end()
            ? "<MISSING>" : rhs->second.value_or("<UNAVAILABLE>");
        if (lhsValue != rhsValue)
            result.push_back({name, lhsValue, rhsValue,
                              name == "fresh_initialization_seed"});
    }
    return result;
}

bool SourceComparisonInvalid(const Pair::ComparisonResult& comparison)
{
    return comparison.status == Pair::Status::InvalidEvidence ||
        comparison.status == Pair::Status::IncompatibleScientificIdentity ||
        comparison.intentionalDifferences.empty() ||
        !comparison.unexpectedDifferences.empty();
}

void RenderIdentityValue(std::ostringstream& output,
                         const std::optional<std::string>& value)
{
    if (!value)
        output << "UNAVAILABLE";
    else if (*value == "NULL")
        output << "NULL";
    else
        output << std::quoted(*value);
}

std::string Reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index) output << '|';
        output << MachineText(reasons[index]);
    }
    return output.str();
}

std::string Ids(const std::vector<long long>& ids)
{
    if (ids.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        if (index) output << '|';
        output << ids[index];
    }
    return output.str();
}

} // namespace

std::vector<unsigned int> ParseReplicationSeeds(std::string_view text)
{
    if (text.empty())
        throw std::invalid_argument(
            "--replication-seeds requires at least one seed");
    std::vector<unsigned int> result;
    std::set<unsigned int> unique;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t end = text.find(',', start);
        const std::string_view token = text.substr(
            start, end == std::string_view::npos
                ? std::string_view::npos : end - start);
        if (token.empty() || token.front() == '+' || token.front() == '-' ||
            (token.size() > 1 && token.front() == '0'))
            throw std::invalid_argument(
                "--replication-seeds requires canonical positive uint32 values");
        unsigned long long parsed = 0;
        const auto [parsedEnd, error] = std::from_chars(
            token.data(), token.data() + token.size(), parsed);
        if (error != std::errc{} || parsedEnd != token.data() + token.size() ||
            parsed == 0 || parsed > std::numeric_limits<unsigned int>::max())
            throw std::invalid_argument(
                "--replication-seeds requires canonical positive uint32 values");
        const auto seed = static_cast<unsigned int>(parsed);
        if (!unique.insert(seed).second)
            throw std::invalid_argument(
                "--replication-seeds contains a duplicate seed");
        result.push_back(seed);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return result;
}

Plan MakePlan(const Pair::ArmResultSet& sourceArmA,
              const Pair::ArmResultSet& sourceArmB,
              const Pair::Request& comparisonRequest,
              const std::vector<unsigned int>& requestedSeeds,
              const EquivalentExperimentSource* equivalents)
{
    if (requestedSeeds.empty())
        throw std::invalid_argument(
            "replication planning requires at least one seed");
    std::set<unsigned int> unique;
    for (const unsigned int seed : requestedSeeds)
        if (seed == 0 || !unique.insert(seed).second)
            throw std::invalid_argument(
                "replication planning seeds must be positive and unique");

    Plan plan;
    plan.sourceExperimentAId = sourceArmA.experimentId;
    plan.sourceExperimentBId = sourceArmB.experimentId;
    plan.requestedSeeds = requestedSeeds;
    plan.sourceArmA = ConfigurationOnly(sourceArmA);
    plan.sourceArmB = ConfigurationOnly(sourceArmB);
    plan.comparisonRequest = comparisonRequest;

    plan.pairs.reserve(requestedSeeds.size());
    for (std::size_t index = 0; index < requestedSeeds.size(); ++index)
    {
        PlannedPair pair;
        pair.ordinal = index + 1;
        pair.requestedSeed = requestedSeeds[index];
        pair.armA.role = "arm_a";
        pair.armA.sourceExperimentId = sourceArmA.experimentId;
        pair.armA.proposed = plan.sourceArmA;
        SetSeed(pair.armA.proposed, pair.requestedSeed);
        pair.armB.role = "arm_b";
        pair.armB.sourceExperimentId = sourceArmB.experimentId;
        pair.armB.proposed = plan.sourceArmB;
        SetSeed(pair.armB.proposed, pair.requestedSeed);
        plan.pairs.push_back(std::move(pair));
    }
    RecomputePreflight(plan);

    if (equivalents)
        for (PlannedPair& pair : plan.pairs)
        {
            pair.armA.equivalent =
                equivalents->FindEquivalent(pair.armA.proposed);
            pair.armB.equivalent =
                equivalents->FindEquivalent(pair.armB.proposed);
        }
    return plan;
}

void RecomputePreflight(Plan& plan)
{
    plan.reasons.clear();
    plan.sourceIntentionalDifferences.clear();
    plan.sourceUnexpectedDifferences.clear();
    plan.state = PlanState::Valid;
    plan.sourceSeedAvailable = false;

    const Pair::ComparisonResult sourceComparison = Pair::Compare(
        ConfigurationOnly(plan.sourceArmA), ConfigurationOnly(plan.sourceArmB),
        plan.comparisonRequest);
    plan.sourceIntentionalDifferences = sourceComparison.intentionalDifferences;
    plan.sourceUnexpectedDifferences = sourceComparison.unexpectedDifferences;
    if (SourceComparisonInvalid(sourceComparison))
    {
        plan.state = PlanState::Invalid;
        AddReason(plan.reasons, "source_pair_not_scientifically_comparable");
        for (const std::string& reason : sourceComparison.invalidReasons)
            AddReason(plan.reasons, "source_pair_" + reason);
        for (const auto& difference : sourceComparison.unexpectedDifferences)
            AddReason(plan.reasons,
                      "source_pair_unexpected_difference:" + difference.field);
        if (sourceComparison.intentionalDifferences.empty())
            AddReason(plan.reasons,
                      "source_pair_intentional_intervention_unavailable");
    }

    bool sourceMissing = false;
    ValidateRequiredIdentity(plan.sourceArmA, "source_arm_a", plan.reasons,
                             sourceMissing);
    ValidateRequiredIdentity(plan.sourceArmB, "source_arm_b", plan.reasons,
                             sourceMissing);
    const auto seedA = Seed(plan.sourceArmA);
    const auto seedB = Seed(plan.sourceArmB);
    if (!seedA || !seedB)
    {
        sourceMissing = true;
        AddReason(plan.reasons, "source_seed_unavailable");
    }
    else if (*seedA != *seedB)
    {
        plan.state = PlanState::Invalid;
        AddReason(plan.reasons, "source_pair_fresh_initialization_seed_mismatch");
    }
    else
    {
        plan.sourceSeed = *seedA;
        plan.sourceSeedAvailable = true;
    }
    if (sourceMissing && plan.state != PlanState::Invalid)
        plan.state = PlanState::UndeterminedDueToMissingEvidence;

    const PlanState sourceState = plan.state;
    const std::vector<std::string> sourceReasons = plan.reasons;
    for (PlannedPair& pair : plan.pairs)
    {
        pair.reasons.clear();
        pair.intentionalDifferences.clear();
        pair.unexpectedDifferences.clear();
        pair.armA.changedFromSource = Differences(
            plan.sourceArmA, pair.armA.proposed);
        pair.armB.changedFromSource = Differences(
            plan.sourceArmB, pair.armB.proposed);
        pair.sourceSeedReuse = plan.sourceSeedAvailable &&
            pair.requestedSeed == plan.sourceSeed;
        pair.replicationClassification =
            ExperimentReplicationComparison::SeedReplicationModeText(
                !plan.sourceSeedAvailable
                    ? ExperimentReplicationComparison::SeedReplicationMode::
                          UnavailableOrNotApplicable
                    : (pair.sourceSeedReuse
                           ? ExperimentReplicationComparison::
                                 SeedReplicationMode::SameSeedRepeatedPairs
                           : ExperimentReplicationComparison::
                                 SeedReplicationMode::DifferentSeeds));
        pair.preflightState = sourceState;
        pair.reasons = sourceReasons;

        const auto validateSourceChange = [&](const ArmPlan& arm,
                                              std::string_view role)
        {
            for (const auto& difference : arm.changedFromSource)
                if (difference.field != "fresh_initialization_seed")
                {
                    pair.preflightState = PlanState::Invalid;
                    AddReason(pair.reasons, std::string(role) +
                              "_unexpected_change_from_source:" +
                              difference.field);
                }
            const auto actualSeed = Seed(arm.proposed);
            if (!actualSeed || *actualSeed != pair.requestedSeed)
            {
                pair.preflightState = PlanState::Invalid;
                AddReason(pair.reasons, std::string(role) +
                          "_requested_seed_assignment_mismatch");
            }
        };
        validateSourceChange(pair.armA, "arm_a");
        validateSourceChange(pair.armB, "arm_b");

        const Pair::ComparisonResult proposedComparison = Pair::Compare(
            ConfigurationOnly(pair.armA.proposed),
            ConfigurationOnly(pair.armB.proposed), plan.comparisonRequest);
        pair.intentionalDifferences =
            proposedComparison.intentionalDifferences;
        pair.unexpectedDifferences = proposedComparison.unexpectedDifferences;
        if (SourceComparisonInvalid(proposedComparison))
        {
            pair.preflightState = PlanState::Invalid;
            AddReason(pair.reasons,
                      "proposed_pair_not_scientifically_comparable");
            for (const auto& difference : proposedComparison.unexpectedDifferences)
                AddReason(pair.reasons,
                          "proposed_pair_unexpected_difference:" +
                          difference.field);
        }
        if (pair.intentionalDifferences != plan.sourceIntentionalDifferences)
        {
            pair.preflightState = PlanState::Invalid;
            AddReason(pair.reasons, "source_intervention_not_preserved_exactly");
        }

        bool proposedMissing = false;
        ValidateRequiredIdentity(pair.armA.proposed, "arm_a", pair.reasons,
                                 proposedMissing);
        ValidateRequiredIdentity(pair.armB.proposed, "arm_b", pair.reasons,
                                 proposedMissing);
        if (proposedMissing && pair.preflightState != PlanState::Invalid)
            pair.preflightState = PlanState::UndeterminedDueToMissingEvidence;

        if (pair.preflightState == PlanState::Invalid)
        {
            plan.state = PlanState::Invalid;
            for (const std::string& reason : pair.reasons)
                AddReason(plan.reasons,
                          "pair_" + std::to_string(pair.ordinal) + ':' +
                              reason);
        }
        else if (pair.preflightState ==
                     PlanState::UndeterminedDueToMissingEvidence &&
                 plan.state != PlanState::Invalid)
        {
            plan.state = PlanState::UndeterminedDueToMissingEvidence;
            for (const std::string& reason : pair.reasons)
                AddReason(plan.reasons,
                          "pair_" + std::to_string(pair.ordinal) + ':' +
                              reason);
        }
    }
}

std::string Render(const Plan& plan)
{
    std::ostringstream output;
    output << "CONTROLLED_REPLICATION_WAVE_PLAN"
           << ",planner=controlled_replication_wave_planner"
           << ",version=" << plan.plannerVersion
           << ",source_experiment_a_id=" << plan.sourceExperimentAId
           << ",source_experiment_b_id=" << plan.sourceExperimentBId
           << ",source_seed=";
    if (plan.sourceSeedAvailable) output << plan.sourceSeed;
    else output << "UNAVAILABLE";
    output << ",requested_seeds=";
    for (std::size_t index = 0; index < plan.requestedSeeds.size(); ++index)
    {
        if (index) output << '|';
        output << plan.requestedSeeds[index];
    }
    output << ",replication_dimension=fresh_initialization_seed"
           << ",equivalence_semantics=experiment_pair_configured_identity_v1"
           << ",statistical_independence=not_inferred"
           << ",read_only=true"
           << ",state=" << PlanStateText(plan.state) << '\n';

    for (const auto& difference : plan.sourceIntentionalDifferences)
        output << "CONTROLLED_REPLICATION_SOURCE_INTERVENTION"
               << ",field=" << MachineText(difference.field)
               << ",arm_a=" << std::quoted(difference.armA)
               << ",arm_b=" << std::quoted(difference.armB) << '\n';
    if (plan.sourceIntentionalDifferences.empty())
        output << "CONTROLLED_REPLICATION_SOURCE_INTERVENTION,field=UNAVAILABLE\n";

    for (const PlannedPair& pair : plan.pairs)
    {
        output << "CONTROLLED_REPLICATION_PLANNED_PAIR"
               << ",ordinal=" << pair.ordinal
               << ",requested_seed=" << pair.requestedSeed
               << ",replication_classification="
               << pair.replicationClassification
               << ",source_seed_reuse="
               << (pair.sourceSeedReuse ? "true" : "false")
               << ",scientific_preflight_state="
               << PlanStateText(pair.preflightState)
               << ",statistical_independence=not_inferred"
               << ",reasons=" << Reasons(pair.reasons) << '\n';

        const auto renderArm = [&](const ArmPlan& arm)
        {
            for (const Pair::IdentityField& field : arm.proposed.scientificIdentity)
            {
                output << "CONTROLLED_REPLICATION_PROPOSED_ARM_IDENTITY"
                       << ",pair_ordinal=" << pair.ordinal
                       << ",role=" << arm.role
                       << ",source_experiment_id=" << arm.sourceExperimentId
                       << ",field=" << MachineText(field.name)
                       << ",value=";
                RenderIdentityValue(output, field.value);
                output << '\n';
            }
            if (arm.changedFromSource.empty())
                output << "CONTROLLED_REPLICATION_CHANGED_FIELD"
                       << ",pair_ordinal=" << pair.ordinal
                       << ",role=" << arm.role << ",field=NONE\n";
            for (const auto& difference : arm.changedFromSource)
                output << "CONTROLLED_REPLICATION_CHANGED_FIELD"
                       << ",pair_ordinal=" << pair.ordinal
                       << ",role=" << arm.role
                       << ",field=" << MachineText(difference.field)
                       << ",source=" << std::quoted(difference.armA)
                       << ",proposed=" << std::quoted(difference.armB) << '\n';
            output << "CONTROLLED_REPLICATION_EQUIVALENCE"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",role=" << arm.role
                   << ",state="
                   << EquivalentExperimentStateText(arm.equivalent.state);
            if (arm.equivalent.state ==
                EquivalentExperimentState::EquivalentExperimentFound)
                output << ",equivalent_experiment_found="
                       << (arm.equivalent.experimentIds.empty()
                               ? "UNAVAILABLE"
                               : std::to_string(
                                     arm.equivalent.experimentIds.front()));
            else if (arm.equivalent.state ==
                     EquivalentExperimentState::EquivalentExperimentAmbiguous)
                output << ",equivalent_experiment_ambiguous="
                       << Ids(arm.equivalent.experimentIds);
            else
                output << ",no_equivalent_experiment_found=true";
            output << ",experiment_ids=" << Ids(arm.equivalent.experimentIds)
                   << ",reason="
                   << (arm.equivalent.reason.empty()
                           ? "NONE" : MachineText(arm.equivalent.reason))
                   << '\n';
        };
        renderArm(pair.armA);
        renderArm(pair.armB);

        if (pair.intentionalDifferences.empty())
            output << "CONTROLLED_REPLICATION_PAIR_DIFFERENCE"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",kind=intentional,field=NONE\n";
        for (const auto& difference : pair.intentionalDifferences)
            output << "CONTROLLED_REPLICATION_PAIR_DIFFERENCE"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",kind=intentional"
                   << ",field=" << MachineText(difference.field)
                   << ",arm_a=" << std::quoted(difference.armA)
                   << ",arm_b=" << std::quoted(difference.armB) << '\n';
        if (pair.unexpectedDifferences.empty())
            output << "CONTROLLED_REPLICATION_PAIR_DIFFERENCE"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",kind=unexpected,field=NONE\n";
        for (const auto& difference : pair.unexpectedDifferences)
            output << "CONTROLLED_REPLICATION_PAIR_DIFFERENCE"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",kind=unexpected"
                   << ",field=" << MachineText(difference.field)
                   << ",arm_a=" << std::quoted(difference.armA)
                   << ",arm_b=" << std::quoted(difference.armB) << '\n';
    }

    output << "CONTROLLED_REPLICATION_WAVE_RESULT"
           << ",state=" << PlanStateText(plan.state)
           << ",reasons=" << Reasons(plan.reasons)
           << ",pair_count=" << plan.pairs.size()
           << ",materialized=false"
           << ",read_only=true\n";
    return output.str();
}

std::string PlanStateText(PlanState state)
{
    switch (state)
    {
        case PlanState::Valid: return "valid";
        case PlanState::Invalid: return "invalid";
        case PlanState::UndeterminedDueToMissingEvidence:
            return "undetermined_due_to_missing_evidence";
    }
    throw std::invalid_argument("unknown_replication_plan_state");
}

std::string EquivalentExperimentStateText(EquivalentExperimentState state)
{
    switch (state)
    {
        case EquivalentExperimentState::NoEquivalentExperimentFound:
            return "no_equivalent_experiment_found";
        case EquivalentExperimentState::EquivalentExperimentFound:
            return "equivalent_experiment_found";
        case EquivalentExperimentState::EquivalentExperimentAmbiguous:
            return "equivalent_experiment_ambiguous";
    }
    throw std::invalid_argument("unknown_equivalent_experiment_state");
}

} // namespace EA::ExperimentReplicationPlanning
