#include "ExperimentReplicationMaterialization.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentReplicationMaterialization
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

std::string Seeds(const std::vector<unsigned int>& seeds)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < seeds.size(); ++index)
    {
        if (index) output << '|';
        output << seeds[index];
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

void Validate(const MaterializationCommand& command)
{
    if (command.sourceExperimentIds.first <= 0 ||
        command.sourceExperimentIds.second <= 0)
        throw std::invalid_argument(
            "replication materialization source IDs must be positive integers");
    if (command.sourceExperimentIds.first ==
        command.sourceExperimentIds.second)
        throw std::invalid_argument(
            "replication materialization requires distinct source IDs");
    if (command.requestedSeeds.empty())
        throw std::invalid_argument(
            "replication materialization requires at least one seed");
}

void RenderHeader(std::ostream& output,
                  const MaterializationCommand& command,
                  const Planning::Plan* plan)
{
    output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION"
           << ",materializer=controlled_replication_wave_materializer"
           << ",version=" << kMaterializerVersion
           << ",source_experiment_a_id=" << command.sourceExperimentIds.first
           << ",source_experiment_b_id=" << command.sourceExperimentIds.second
           << ",source_seed=";
    if (plan && plan->sourceSeedAvailable) output << plan->sourceSeed;
    else output << "UNAVAILABLE";
    output << ",requested_seeds=" << Seeds(command.requestedSeeds)
           << ",replication_dimension=fresh_initialization_seed"
           << ",statistical_independence=not_inferred"
           << ",transaction=single_postgresql_write_transaction"
           << ",atomicity=all_or_nothing"
           << ",serialization=experiment_table_share_row_exclusive_lock"
           << ",queued=false,started=false\n";
}

void RenderPlanEvidence(std::ostream& output, const Planning::Plan& plan)
{
    output << "CONTROLLED_REPLICATION_MATERIALIZATION_PREFLIGHT"
           << ",state=" << Planning::PlanStateText(plan.state)
           << ",reasons=" << Reasons(plan.reasons) << '\n';
    for (const auto& difference : plan.sourceIntentionalDifferences)
        output << "CONTROLLED_REPLICATION_MATERIALIZATION_INTERVENTION"
               << ",field=" << MachineText(difference.field)
               << ",arm_a=" << std::quoted(difference.armA)
               << ",arm_b=" << std::quoted(difference.armB) << '\n';
    if (plan.sourceIntentionalDifferences.empty())
        output << "CONTROLLED_REPLICATION_MATERIALIZATION_INTERVENTION"
               << ",field=UNAVAILABLE\n";

    const auto renderArm = [&](const Planning::PlannedPair& pair,
                               const Planning::ArmPlan& arm)
    {
        output << "CONTROLLED_REPLICATION_MATERIALIZATION_EQUIVALENCE"
               << ",pair_ordinal=" << pair.ordinal
               << ",requested_seed=" << pair.requestedSeed
               << ",role=" << arm.role
               << ",state="
               << Planning::EquivalentExperimentStateText(
                      arm.equivalent.state);
        if (arm.equivalent.state ==
            Planning::EquivalentExperimentState::NoEquivalentExperimentFound)
            output << ",no_equivalent_experiment_found=true";
        else if (arm.equivalent.state ==
                 Planning::EquivalentExperimentState::EquivalentExperimentFound)
            output << ",equivalent_experiment_found="
                   << (arm.equivalent.experimentIds.empty()
                           ? "UNAVAILABLE"
                           : std::to_string(
                                 arm.equivalent.experimentIds.front()));
        else
            output << ",equivalent_experiment_ambiguous="
                   << Ids(arm.equivalent.experimentIds);
        output << ",experiment_ids=" << Ids(arm.equivalent.experimentIds)
               << ",reason="
               << (arm.equivalent.reason.empty()
                       ? "NONE" : MachineText(arm.equivalent.reason))
               << '\n';
    };
    for (const Planning::PlannedPair& pair : plan.pairs)
    {
        renderArm(pair, pair.armA);
        renderArm(pair, pair.armB);
    }
}

bool HasEquivalenceConflict(const Planning::Plan& plan)
{
    for (const Planning::PlannedPair& pair : plan.pairs)
        for (const Planning::ArmPlan* arm : {&pair.armA, &pair.armB})
            if (arm->equivalent.state !=
                Planning::EquivalentExperimentState::
                    NoEquivalentExperimentFound)
                return true;
    return false;
}

} // namespace

int RunMaterializationInTransaction(
    const MaterializationCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    ExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        Validate(command);
        const auto armAEvidence = evidence.Load(
            command.sourceExperimentIds.first);
        const auto armBEvidence = evidence.Load(
            command.sourceExperimentIds.second);
        const auto request = ExperimentPairComparison::MakeComparisonRequest(
            armAEvidence, armBEvidence);
        const Planning::Plan plan = Planning::MakePlan(
            ExperimentPairComparison::MakeArmResultSet(armAEvidence),
            ExperimentPairComparison::MakeArmResultSet(armBEvidence), request,
            command.requestedSeeds, &equivalents);

        RenderHeader(output, command, &plan);
        RenderPlanEvidence(output, plan);
        if (plan.state != Planning::PlanState::Valid)
        {
            output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
                   << ",state=not_materialized"
                   << ",reason=scientific_preflight_"
                   << Planning::PlanStateText(plan.state)
                   << ",pair_count=0,experiment_count=0"
                   << ",transaction=rolled_back,queued=false,started=false"
                   << ",exit_code=3\n";
            return 3;
        }
        if (HasEquivalenceConflict(plan))
        {
            output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
                   << ",state=not_materialized"
                   << ",reason=equivalence_conflict"
                   << ",pair_count=0,experiment_count=0"
                   << ",transaction=rolled_back,queued=false,started=false"
                   << ",exit_code=3\n";
            return 3;
        }

        struct Created
        {
            const Planning::PlannedPair* pair = nullptr;
            long long armAId = 0;
            long long armBId = 0;
        };
        std::vector<Created> created;
        created.reserve(plan.pairs.size());
        for (const Planning::PlannedPair& pair : plan.pairs)
        {
            Created value;
            value.pair = &pair;
            value.armAId = inserter.InsertFreshPausedExperiment(
                pair.armA.proposed);
            value.armBId = inserter.InsertFreshPausedExperiment(
                pair.armB.proposed);
            if (value.armAId <= 0 || value.armBId <= 0)
                throw std::runtime_error(
                    "materialized_experiment_id_invalid");
            created.push_back(value);
        }

        for (const Created& value : created)
        {
            output << "CONTROLLED_REPLICATION_MATERIALIZED_PAIR"
                   << ",ordinal=" << value.pair->ordinal
                   << ",requested_seed=" << value.pair->requestedSeed
                   << ",arm_a_experiment_id=" << value.armAId
                   << ",arm_b_experiment_id=" << value.armBId
                   << ",source_experiment_a_id="
                   << command.sourceExperimentIds.first
                   << ",source_experiment_b_id="
                   << command.sourceExperimentIds.second;
            if (value.pair->intentionalDifferences.empty())
                output << ",intentional_intervention=UNAVAILABLE";
            else
            {
                const auto& difference =
                    value.pair->intentionalDifferences.front();
                output << ",intentional_intervention_field="
                       << MachineText(difference.field)
                       << ",intentional_intervention_arm_a="
                       << std::quoted(difference.armA)
                       << ",intentional_intervention_arm_b="
                       << std::quoted(difference.armB);
            }
            output << ",queued=false,started=false\n";
        }
        output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
               << ",state=materialized"
               << ",pair_count=" << created.size()
               << ",experiment_count=" << created.size() * 2
               << ",transaction=committed,queued=false,started=false\n";
        return 0;
    }
    catch (const ExperimentPairComparison::EvidenceUnavailableError& error)
    {
        RenderHeader(errors, command, nullptr);
        errors << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
               << ",state=not_materialized,reason="
               << MachineText(error.reason())
               << ",pair_count=0,experiment_count=0"
               << ",transaction=rolled_back,queued=false,started=false"
               << ",exit_code=3\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        RenderHeader(errors, command, nullptr);
        errors << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
               << ",state=not_materialized,reason="
               << MachineText(error.what())
               << ",pair_count=0,experiment_count=0"
               << ",transaction=rolled_back,queued=false,started=false"
               << ",exit_code=3\n";
        return 3;
    }
}

} // namespace EA::ExperimentReplicationMaterialization
