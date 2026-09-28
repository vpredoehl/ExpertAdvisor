#include "ExperimentReplicationMaterialization.hpp"

#include "SchedulerCore/SemanticWorkerRegistry.hpp"

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

bool HasEquivalentFound(const Planning::Plan& plan)
{
    for (const auto& pair : plan.pairs)
        for (const Planning::ArmPlan* arm : {&pair.armA, &pair.armB})
            if (arm->equivalent.state ==
                Planning::EquivalentExperimentState::EquivalentExperimentFound)
                return true;
    return false;
}

bool HasEquivalentAmbiguous(const Planning::Plan& plan)
{
    for (const auto& pair : plan.pairs)
        for (const Planning::ArmPlan* arm : {&pair.armA, &pair.armB})
            if (arm->equivalent.state ==
                Planning::EquivalentExperimentState::EquivalentExperimentAmbiguous)
                return true;
    return false;
}

std::string ExistingEquivalentAuthorization(
    const MaterializationCommand& command, const Planning::Plan& plan)
{
    if (HasEquivalentAmbiguous(plan)) return "ambiguous_not_authorized";
    if (!HasEquivalentFound(plan))
        return command.allowExistingEquivalent
            ? "authorized_no_equivalent" : "not_applicable";
    return command.allowExistingEquivalent ? "authorized" : "not_requested";
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
           << ",allow_existing_equivalent="
           << (command.allowExistingEquivalent ? "true" : "false")
           << ",existing_equivalent_authorization="
           << (plan ? ExistingEquivalentAuthorization(command, *plan)
                    : (command.allowExistingEquivalent
                           ? "requested_but_not_verified" : "not_requested"))
           << ",statistical_independence=not_inferred"
           << ",train_worker_routing_state="
           << (plan && plan->trainWorkerRoutingEvaluated
                   ? Planning::TrainWorkerRoutingStateText(
                         plan->trainWorkerRoutingState)
                   : "not_evaluated")
           << ",wave_train_execution_identity_homogeneous="
           << (plan && plan->waveTrainExecutionIdentityHomogeneous
                   ? "true" : "false")
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
        for (const Planning::ArmPlan* arm : {&pair.armA, &pair.armB})
        {
            const auto& routing = arm->trainWorkerRouting;
            output << "CONTROLLED_REPLICATION_MATERIALIZATION_TRAIN_WORKER_ROUTING"
                   << ",pair_ordinal=" << pair.ordinal
                   << ",requested_seed=" << pair.requestedSeed
                   << ",role=" << arm->role
                   << ",source_experiment_id=" << arm->sourceExperimentId
                   << ",semantic_layout="
                   << (routing.semanticLayoutVersion > 0
                           ? std::to_string(routing.semanticLayoutVersion)
                           : "UNAVAILABLE")
                   << ",model_input_width="
                   << (routing.modelInputWidth > 0
                           ? std::to_string(routing.modelInputWidth)
                           : "UNAVAILABLE")
                   << ",effective_required_train_capabilities=";
            if (routing.effectiveRequiredCapabilities.empty()) output << "NONE";
            else
                for (std::size_t index = 0;
                     index < routing.effectiveRequiredCapabilities.size(); ++index)
                {
                    if (index) output << '|';
                    output << MachineText(
                        routing.effectiveRequiredCapabilities[index]);
                }
            output << ",selected_worker_role=" << routing.selectedWorkerRole
                   << ",selected_worker_rule="
                   << (routing.selectedWorkerRule.empty()
                           ? "UNAVAILABLE" :
                              MachineText(routing.selectedWorkerRule))
                   << ",selection_priority="
                   << (routing.state == Planning::TrainWorkerRoutingState::Selected
                           ? std::to_string(routing.selectionPriority)
                           : "UNAVAILABLE")
                   << ",source_commit="
                   << (routing.sourceCommit.empty()
                           ? "UNAVAILABLE" : routing.sourceCommit)
                   << ",executable_sha256="
                   << (routing.executableSha256.empty()
                           ? "UNAVAILABLE" : routing.executableSha256)
                   << ",runtime_identity="
                   << (routing.runtimeIdentity.empty()
                           ? "UNAVAILABLE" : routing.runtimeIdentity)
                   << ",canonical_executable_path="
                   << (routing.canonicalExecutablePath.empty()
                           ? "UNAVAILABLE" :
                              MachineText(routing.canonicalExecutablePath))
                   << ",canonical_manifest_path="
                   << (routing.canonicalManifestPath.empty()
                           ? "UNAVAILABLE" :
                              MachineText(routing.canonicalManifestPath))
                   << ",canonical_train_execution_identity="
                   << (routing.canonicalTrainExecutionIdentity.empty()
                           ? "UNAVAILABLE" :
                              MachineText(routing.canonicalTrainExecutionIdentity))
                   << ",selection_state="
                   << Planning::TrainWorkerRoutingStateText(routing.state)
                   << ",reason="
                   << (routing.reason.empty() ? "NONE" :
                       MachineText(routing.reason))
                   << ",registry_schema_version="
                   << (routing.registrySchemaVersion > 0
                           ? std::to_string(routing.registrySchemaVersion)
                           : "UNAVAILABLE")
                   << ",registry_path="
                   << (routing.registryPath.empty()
                           ? "UNAVAILABLE" : MachineText(routing.registryPath))
                   << ",statistical_independence=not_inferred\n";
        }
        output << "CONTROLLED_REPLICATION_MATERIALIZATION_PAIR_TRAIN_WORKER_ROUTING"
               << ",pair_ordinal=" << pair.ordinal
               << ",requested_seed=" << pair.requestedSeed
               << ",pair_train_execution_identity_homogeneous="
               << (pair.pairTrainExecutionIdentityHomogeneous
                       ? "true" : "false")
               << ",statistical_independence=not_inferred\n";
    }
    output << "CONTROLLED_REPLICATION_MATERIALIZATION_WAVE_TRAIN_WORKER_ROUTING"
           << ",train_worker_routing_state="
           << (plan.trainWorkerRoutingEvaluated
                   ? Planning::TrainWorkerRoutingStateText(
                         plan.trainWorkerRoutingState)
                   : "not_evaluated")
           << ",every_proposed_arm_has_deterministic_train_worker="
           << (plan.everyProposedArmHasDeterministicTrainWorker
                   ? "true" : "false")
           << ",distinct_selected_train_execution_identity_count="
           << plan.distinctSelectedTrainExecutionIdentities.size()
           << ",wave_train_execution_identity_homogeneous="
           << (plan.waveTrainExecutionIdentityHomogeneous ? "true" : "false")
           << ",registry_schema_version="
           << (plan.trainWorkerRoutingEvaluated
                   ? std::to_string(plan.trainWorkerRegistrySchemaVersion)
                   : "UNAVAILABLE")
           << ",registry_path="
           << (plan.trainWorkerRoutingEvaluated
                   ? MachineText(plan.trainWorkerRegistryPath)
                   : "UNAVAILABLE")
           << ",statistical_independence=not_inferred\n";
}

bool HasEquivalenceConflict(const Planning::Plan& plan,
                           bool allowExistingEquivalent)
{
    for (const Planning::PlannedPair& pair : plan.pairs)
    {
        for (const Planning::ArmPlan* arm : {&pair.armA, &pair.armB})
        {
            if (arm->equivalent.state ==
                Planning::EquivalentExperimentState::EquivalentExperimentAmbiguous)
                return true;
            if (arm->equivalent.state ==
                    Planning::EquivalentExperimentState::EquivalentExperimentFound &&
                !allowExistingEquivalent)
                return true;
        }
    }
    return false;
}

} // namespace

int RunMaterializationInTransaction(
    const MaterializationCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    ExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors,
    const EA::Scheduler::SemanticWorkerRegistry* registry)
{
    try
    {
        Validate(command);
        if (registry == nullptr)
        {
            RenderHeader(output, command, nullptr);
            output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
                   << ",state=not_materialized"
                   << ",reason=train_worker_routing_registry_unavailable"
                   << ",pair_count=0,experiment_count=0"
                   << ",transaction=rolled_back,queued=false,started=false"
                   << ",exit_code=3\n";
            return 3;
        }
        const auto armAEvidence = evidence.Load(
            command.sourceExperimentIds.first);
        const auto armBEvidence = evidence.Load(
            command.sourceExperimentIds.second);
        const auto request = ExperimentPairComparison::MakeComparisonRequest(
            armAEvidence, armBEvidence);
        Planning::Plan plan = Planning::MakePlan(
            ExperimentPairComparison::MakeArmResultSet(armAEvidence),
            ExperimentPairComparison::MakeArmResultSet(armBEvidence), request,
            command.requestedSeeds, &equivalents);
        if (registry != nullptr)
            Planning::AttachTrainWorkerRouting(plan, *registry);

        RenderHeader(output, command, &plan);
        RenderPlanEvidence(output, plan);
        if (plan.state != Planning::PlanState::Valid)
        {
            output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
                   << ",state=not_materialized"
                   << ",reason="
                   << (plan.trainWorkerRoutingEvaluated &&
                               !plan.everyProposedArmHasDeterministicTrainWorker
                           ? "train_worker_routing_not_admissible"
                           : "scientific_preflight_" +
                                 Planning::PlanStateText(plan.state))
                   << ",pair_count=0,experiment_count=0"
                   << ",transaction=rolled_back,queued=false,started=false"
                   << ",exit_code=3\n";
            return 3;
        }
        if (HasEquivalenceConflict(plan, command.allowExistingEquivalent))
        {
            output << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
                   << ",state=not_materialized"
                   << ",reason="
                   << (HasEquivalentAmbiguous(plan)
                           ? "equivalence_ambiguous_not_authorized"
                           : "equivalence_conflict")
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
            output << ",existing_equivalent_arm_a_ids="
                   << Ids(value.pair->armA.equivalent.experimentIds)
                   << ",existing_equivalent_arm_b_ids="
                   << Ids(value.pair->armB.equivalent.experimentIds)
                   << ",existing_equivalent_authorization="
                   << (command.allowExistingEquivalent &&
                               (value.pair->armA.equivalent.state ==
                                    Planning::EquivalentExperimentState::
                                        EquivalentExperimentFound ||
                                value.pair->armB.equivalent.state ==
                                    Planning::EquivalentExperimentState::
                                        EquivalentExperimentFound)
                           ? "authorized" : "not_applicable")
                   << ",materialization_kind=fresh_execution_replication"
                   << ",configured_identity_repeat=true"
                   << ",existing_equivalent_reused=false"
                   << ",old_equivalent_modified=false"
                   << ",materialized_status=paused"
                   << ",materialized_phase=train";
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
