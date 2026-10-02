#include "ExperimentReplicationMaterialization.hpp"

#include "FeatureAblation.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "SchedulerCore/SemanticWorkerRegistry.hpp"
#include "SchedulerCore/TrainingWorkerSelection.hpp"

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

constexpr std::string_view kConfluenceMask =
    "confluence_tg4_structural_fibonacci_retracement_support_available,"
    "confluence_tg4_structural_fibonacci_retracement_contradiction_available";
static_assert(kModelInputSemanticLayoutVersion == 11);
static_assert(kFixedConfluenceTensorModelInputWidth == 116);

void SetIdentity(ExperimentPairComparison::ArmResultSet& arm,
                 std::string_view name, std::string value)
{
    std::size_t matches = 0;
    for (auto& field : arm.scientificIdentity)
        if (field.name == name)
        {
            field.value = std::move(value);
            ++matches;
        }
    if (matches != 1)
        throw std::invalid_argument("layout11_confluence_identity_missing_or_duplicated:" +
                                    std::string{name});
}

std::string Identity(const ExperimentPairComparison::ArmResultSet& arm,
                     std::string_view name)
{
    const auto found = std::find_if(arm.scientificIdentity.begin(),
                                    arm.scientificIdentity.end(),
        [name](const auto& field) { return field.name == name; });
    if (found == arm.scientificIdentity.end() || !found->value)
        throw std::invalid_argument("layout11_confluence_identity_unavailable:" +
                                    std::string{name});
    return *found->value;
}

void ValidateTemplate(const FeatureAblationPairEvaluation::ArmEvidence& evidence)
{
    const auto& configuration = evidence.authoritative.configuration;
    const auto& extended = evidence.extended;
    if (configuration.predictionHorizon != 4 || configuration.targetEpochs != 20 ||
        configuration.checkpointInterval != 20 ||
        std::abs(configuration.threshold - 0.0008) > 1e-12 ||
        !configuration.coreLearningRateMultiplier ||
        !configuration.headLearningRateMultiplier ||
        std::abs(*configuration.coreLearningRateMultiplier - 120.0) > 1e-12 ||
        std::abs(*configuration.headLearningRateMultiplier - 25.0) > 1e-12 ||
        configuration.featureWarmupScope != "full_history_warmup" ||
        configuration.featureAblationMask != "" ||
        !extended.configuredModelInputWidth ||
        !extended.configuredModelInputLayoutVersion ||
        *extended.configuredModelInputWidth != 114 ||
        *extended.configuredModelInputLayoutVersion != 10 ||
        !extended.economicCalendarSnapshotId ||
        *extended.economicCalendarSnapshotId != 1 ||
        !extended.economicCalendarSnapshotHash ||
        *extended.economicCalendarSnapshotHash != "fnv1a64:67610f94f5c8e7cc" ||
        !extended.freshInitializationSeed ||
        configuration.resumeModelId || configuration.resumeExpandInputWidth ||
        extended.continuationPolicyEnabled)
        throw std::invalid_argument("layout11_confluence_template_baseline_mismatch");
    const auto date = [](const std::string& value)
    { return value.size() >= 10 ? value.substr(0, 10) : value; };
    if (date(configuration.trainStart) != "2010-01-01" ||
        date(configuration.trainEnd) != "2025-01-01" ||
        date(configuration.inferenceStart) != "2025-01-01" ||
        date(configuration.inferenceEnd) != "2026-01-01" ||
        configuration.experimentObjective.canonical.empty() ||
        configuration.experimentObjective.hash.empty())
        throw std::invalid_argument("layout11_confluence_template_identity_mismatch");
}

Layout11ConfluenceArm MakeConfluenceArm(
    const ExperimentPairComparison::ArmResultSet& templateArm,
    long long templateExperimentId, unsigned int seed, std::string role,
    std::string mask)
{
    Layout11ConfluenceArm arm;
    arm.templateExperimentId = templateExperimentId;
    arm.freshInitializationSeed = seed;
    arm.role = std::move(role);
    arm.featureAblationMask = std::move(mask);
    arm.proposed = templateArm;
    SetIdentity(arm.proposed, "fresh_initialization_seed", std::to_string(seed));
    SetIdentity(arm.proposed, "feature_ablation_mask", arm.featureAblationMask);
    SetIdentity(arm.proposed, "configured_model_input_width",
                std::to_string(kFixedConfluenceTensorModelInputWidth));
    SetIdentity(arm.proposed, "configured_model_input_semantic_layout_version",
                std::to_string(kModelInputSemanticLayoutVersion));
    // The template's completed model metadata describes its historical
    // Layout-10 producer.  A fresh planned arm has no model, so its projected
    // model identity must agree with the configured Layout-11 contract.
    SetIdentity(arm.proposed, "model_input_width",
                std::to_string(kFixedConfluenceTensorModelInputWidth));
    SetIdentity(arm.proposed, "model_input_semantic_layout_version",
                std::to_string(kModelInputSemanticLayoutVersion));
    return arm;
}

void ValidatePair(const Layout11ConfluenceArm& control,
                  const Layout11ConfluenceArm& treatment)
{
    if (control.featureAblationMask != "" || treatment.featureAblationMask != kConfluenceMask)
        throw std::invalid_argument("layout11_confluence_pair_mask_invalid");
    if (EA::FeatureAblationMask::ParseForSemanticLayout(
            treatment.featureAblationMask, 11).CanonicalText() != kConfluenceMask)
        throw std::invalid_argument("layout11_confluence_pair_mask_not_canonical");
    if (Identity(control.proposed, "configured_model_input_width") != "116" ||
        Identity(treatment.proposed, "configured_model_input_width") != "116")
        throw std::invalid_argument("layout11_confluence_pair_width_invalid");
    if (Identity(control.proposed, "configured_model_input_semantic_layout_version") != "11" ||
        Identity(treatment.proposed, "configured_model_input_semantic_layout_version") != "11")
        throw std::invalid_argument("layout11_confluence_pair_layout_invalid");
    if (control.freshInitializationSeed != treatment.freshInitializationSeed)
        throw std::invalid_argument("layout11_confluence_pair_seed_invalid");
    std::map<std::string, std::optional<std::string>> controlIdentity;
    std::map<std::string, std::optional<std::string>> treatmentIdentity;
    for (const auto& field : control.proposed.scientificIdentity)
        if (!controlIdentity.emplace(field.name, field.value).second)
            throw std::invalid_argument("layout11_confluence_pair_identity_duplicated");
    for (const auto& field : treatment.proposed.scientificIdentity)
        if (!treatmentIdentity.emplace(field.name, field.value).second)
            throw std::invalid_argument("layout11_confluence_pair_identity_duplicated");
    if (controlIdentity.size() != treatmentIdentity.size())
        throw std::invalid_argument("layout11_confluence_pair_identity_mismatch");
    for (const auto& [name, value] : controlIdentity)
    {
        const auto treatmentValue = treatmentIdentity.find(name);
        if (treatmentValue == treatmentIdentity.end() ||
            (name != "feature_ablation_mask" && treatmentValue->second != value))
            throw std::invalid_argument("layout11_confluence_pair_identity_mismatch:" + name);
    }
}

void ValidateRouting(const Layout11ConfluenceArm& control,
                     const Layout11ConfluenceArm& treatment,
                     const EA::Scheduler::SemanticWorkerRegistry& registry,
                     std::ostream& output)
{
    const EA::Scheduler::PersistedWorkerSemanticIdentity identity{
        kFixedConfluenceTensorModelInputWidth, kModelInputSemanticLayoutVersion, false};
    const auto controlWorker = registry.selectTrainingReferenceWorker(
        identity, EA::Scheduler::RequiredTrainingWorkerCapabilities(control.featureAblationMask));
    const auto treatmentWorker = registry.selectTrainingReferenceWorker(
        identity, EA::Scheduler::RequiredTrainingWorkerCapabilities(treatment.featureAblationMask));
    if (!controlWorker.selected || !treatmentWorker.selected ||
        controlWorker.canonicalExecutablePath != treatmentWorker.canonicalExecutablePath)
        throw std::invalid_argument("layout11_confluence_train_worker_contract_unresolved");
    const auto* worker = registry.findByCanonicalExecutable(
        treatmentWorker.canonicalExecutablePath);
    if (worker == nullptr || std::find(worker->capabilities.begin(), worker->capabilities.end(),
        "train_feature_ablation_v1") == worker->capabilities.end())
        throw std::invalid_argument("layout11_confluence_train_feature_ablation_capability_missing");
    output << "LAYOUT11_CONFLUENCE_REPLICATION_TRAIN_WORKER"
           << ",semantic_layout=11,model_input_width=116"
           << ",canonical_executable_path=" << worker->canonicalExecutablePath
           << ",executable_sha256=" << worker->sha256
           << ",source_commit=" << worker->sourceCommit
           << ",runtime_identity=" << worker->runtimeIdentity
           << ",capability=train_feature_ablation_v1\n";
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

int RunLayout11ConfluenceMaterializationInTransaction(
    const Layout11ConfluenceCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    Layout11ConfluenceExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors,
    const EA::Scheduler::SemanticWorkerRegistry* registry,
    bool apply)
{
    try
    {
        if (command.templateExperimentId <= 0 || command.requestedSeeds.empty())
            throw std::invalid_argument("layout11_confluence_command_invalid");
        if (registry == nullptr)
            throw std::invalid_argument("layout11_confluence_worker_registry_unavailable");
        std::set<unsigned int> seeds;
        for (const unsigned int seed : command.requestedSeeds)
            if (seed == 0 || !seeds.insert(seed).second)
                throw std::invalid_argument("layout11_confluence_seeds_invalid");

        const auto templateEvidence = evidence.Load(command.templateExperimentId);
        ValidateTemplate(templateEvidence);
        const auto templateArm = ExperimentPairComparison::MakeArmResultSet(templateEvidence);
        output << "LAYOUT11_CONFLUENCE_REPLICATION_PLAN"
               << ",template_experiment_id=" << command.templateExperimentId
               << ",template_semantic_layout=10,template_model_input_width=114"
               << ",semantic_layout=11,model_input_width=116"
               << ",treatment_mask=" << kConfluenceMask
               << ",mutations=0,queued=false,started=false\n";
        for (const unsigned int seed : command.requestedSeeds)
        {
            const auto control = MakeConfluenceArm(templateArm, command.templateExperimentId,
                seed, "control", "");
            const auto treatment = MakeConfluenceArm(templateArm, command.templateExperimentId,
                seed, "treatment", std::string{kConfluenceMask});
            ValidatePair(control, treatment);
            ValidateRouting(control, treatment, *registry, output);
            const auto controlEquivalent = equivalents.FindEquivalent(control.proposed);
            const auto treatmentEquivalent = equivalents.FindEquivalent(treatment.proposed);
            for (const Layout11ConfluenceArm* arm : {&control, &treatment})
                for (const auto& field : arm->proposed.scientificIdentity)
                    output << "LAYOUT11_CONFLUENCE_REPLICATION_CONFIGURATION"
                           << ",seed=" << seed << ",role=" << arm->role
                           << ",template_experiment_id=" << command.templateExperimentId
                           << ",field=" << MachineText(field.name)
                           << ",value=" << std::quoted(field.value.value_or("UNAVAILABLE"))
                           << '\n';
            output << "LAYOUT11_CONFLUENCE_REPLICATION_ARM"
                   << ",seed=" << seed << ",role=control"
                   << ",symbol=" << Identity(control.proposed, "symbol")
                   << ",prediction_horizon=" << Identity(control.proposed, "prediction_horizon")
                   << ",target_epochs=" << Identity(control.proposed, "target_epochs")
                   << ",feature_ablation_mask=EMPTY,model_input_width=116"
                   << ",semantic_layout=11,threshold=" << Identity(control.proposed, "threshold")
                   << ",core_lr_mult=" << Identity(control.proposed, "core_lr_mult")
                   << ",head_lr_mult=" << Identity(control.proposed, "head_lr_mult")
                   << ",checkpoint_interval=" << Identity(control.proposed, "checkpoint_interval")
                   << ",train_start=" << Identity(control.proposed, "train_start")
                   << ",train_end=" << Identity(control.proposed, "train_end")
                   << ",inference_start=" << Identity(control.proposed, "inference_start")
                   << ",inference_end=" << Identity(control.proposed, "inference_end")
                   << ",feature_warmup_scope=" << Identity(control.proposed, "feature_warmup_scope")
                   << ",training_objective=" << Identity(control.proposed, "training_objective_canonical")
                   << ",economic_calendar_snapshot_id=" << Identity(control.proposed, "economic_calendar_snapshot_id")
                   << ",economic_calendar_snapshot_hash=" << Identity(control.proposed, "economic_calendar_snapshot_hash")
                   << ",equivalence=" << Planning::EquivalentExperimentStateText(controlEquivalent.state) << '\n';
            output << "LAYOUT11_CONFLUENCE_REPLICATION_ARM"
                   << ",seed=" << seed << ",role=treatment"
                   << ",symbol=" << Identity(treatment.proposed, "symbol")
                   << ",prediction_horizon=" << Identity(treatment.proposed, "prediction_horizon")
                   << ",target_epochs=" << Identity(treatment.proposed, "target_epochs")
                   << ",feature_ablation_mask=" << kConfluenceMask
                   << ",model_input_width=116,semantic_layout=11"
                   << ",training_objective=" << Identity(treatment.proposed, "training_objective_canonical")
                   << ",equivalence=" << Planning::EquivalentExperimentStateText(treatmentEquivalent.state) << '\n';
            if (controlEquivalent.state != Planning::EquivalentExperimentState::NoEquivalentExperimentFound ||
                treatmentEquivalent.state != Planning::EquivalentExperimentState::NoEquivalentExperimentFound)
                throw std::invalid_argument("layout11_confluence_equivalence_conflict");
            if (apply)
            {
                const long long controlId = inserter.InsertFreshPausedLayout11ConfluenceExperiment(control);
                const long long treatmentId = inserter.InsertFreshPausedLayout11ConfluenceExperiment(treatment);
                output << "LAYOUT11_CONFLUENCE_REPLICATION_MATERIALIZED_PAIR"
                       << ",seed=" << seed << ",control_experiment_id=" << controlId
                       << ",treatment_experiment_id=" << treatmentId
                       << ",queued=false,started=false\n";
            }
        }
        output << "LAYOUT11_CONFLUENCE_REPLICATION_RESULT,state="
               << (apply ? "materialized" : "planned")
               << ",pair_count=" << command.requestedSeeds.size()
               << ",experiment_count=" << command.requestedSeeds.size() * 2
               << ",mutations=" << (apply ? command.requestedSeeds.size() * 2 : 0)
               << ",queued=false,started=false\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "LAYOUT11_CONFLUENCE_REPLICATION_RESULT,state=not_materialized,reason="
               << MachineText(error.what())
               << ",mutations=0,queued=false,started=false,exit_code=3\n";
        return 3;
    }
}

} // namespace EA::ExperimentReplicationMaterialization
