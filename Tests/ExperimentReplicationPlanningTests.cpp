#include "ExperimentReplicationPlanningService.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Planning = EA::ExperimentReplicationPlanning;
namespace Pair = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Shared = EA::PairedTrainingObjectiveEvaluation;

namespace
{

constexpr std::string_view kMask =
    "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,"
    "tg4_source_tg3_confluent";

Feature::ArmEvidence Arm(long long experimentId,
                         std::string mask,
                         unsigned int seed)
{
    Feature::ArmEvidence result;
    auto& shared = result.authoritative;
    auto& configuration = shared.configuration;
    configuration.experimentId = experimentId;
    configuration.symbol = "eurusdrmp";
    configuration.predictionHorizon = 4;
    configuration.trainStart = "2010-01-01";
    configuration.trainEnd = "2025-01-01";
    configuration.inferenceStart = "2025-01-01";
    configuration.inferenceEnd = "2026-01-01";
    configuration.targetEpochs = 80;
    configuration.threshold = 0.0008;
    configuration.coreLearningRateMultiplier = 120.0;
    configuration.headLearningRateMultiplier = 25.0;
    configuration.checkpointInterval = 20;
    configuration.inputWidth = 80;
    configuration.hiddenSize = 64;
    configuration.layerCount = 1;
    configuration.windowSize = 64;
    configuration.modelMetadataSchemaVersion = 1;
    configuration.trainConfigurationSchemaVersion = 1;
    configuration.normalizationVersion = 1;
    configuration.classWeightDown = 1.0;
    configuration.classWeightNeutral = 1.0;
    configuration.classWeightUp = 1.0;
    configuration.featureWarmupScope = "full_history_warmup";
    configuration.donchianMode = "enabled";
    configuration.donchianLookback = 20;
    configuration.featureAblationMask = std::move(mask);
    configuration.optimizerMetadataSchemaVersion = 1;
    configuration.optimizerType = 1;
    configuration.optimizerFirstMomentBufferCount = 4;
    configuration.optimizerSecondMomentBufferCount = 4;
    configuration.persistedCoreLearningRateMultiplier = 120.0;
    configuration.persistedHeadWeightLearningRateMultiplier = 25.0;
    configuration.persistedHeadBiasLearningRateMultiplier = 25.0;
    configuration.labelRuleId = 1;
    configuration.targetType = 0;
    configuration.targetScale = 1.0;
    configuration.targetStandardDeviation = 1.0;
    configuration.modelInputMetadataSchemaVersion = 1;
    configuration.modelInputLayoutVersion = 8;
    configuration.persistedTrainingSymbol = configuration.symbol;
    configuration.persistedTrainingStart = configuration.trainStart;
    configuration.persistedTrainingEnd = configuration.trainEnd;
    configuration.experimentObjective = {
        "objective_id=legacy_first_hit_weighted_ce_v1;",
        "fnv1a64:0000000000000001"};

    result.extended.configuredModelInputWidth = 80;
    result.extended.configuredModelInputLayoutVersion = 8;
    result.extended.economicCalendarSnapshotId = 50;
    result.extended.economicCalendarSnapshotHash =
        "fnv1a64:0000000000000002";
    result.extended.freshInitializationSeed = seed;
    result.extended.trainingObjectiveVersion = 1;
    result.extended.lossDefinitionVersion = 1;
    result.extended.auxiliaryLossMode = "disabled";
    result.extended.targetClippingDefinition = "none";
    result.extended.objectiveNormalizationIdentity = "fixture_v1";
    result.extended.checkpointPolicyScope = "symbol_horizon";
    result.extended.checkpointPolicyStopMode = "next_checkpoint";
    result.extended.checkpointPolicyGraceEvaluations = 1;
    result.extended.checkpointPolicyRevision = 1;

    shared.experimentStatus = "completed";
    shared.experimentPhase = "done";
    shared.finalModelId = experimentId + 1000;
    shared.trainingExecution = {
        experimentId * 10 + 1, "train", 8, 80, "train", "source_commit",
        "train_sha256", "LSTM_Release", "train_runtime_identity", ""};
    shared.inferenceExecution = {
        experimentId * 10 + 2, "infer", 8, 80, "infer", "source_commit",
        "infer_sha256", "lstm-infer-worker", "infer_runtime_identity", ""};
    return result;
}

const Pair::IdentityField& Identity(const Pair::ArmResultSet& arm,
                                    std::string_view name)
{
    const auto found = std::find_if(
        arm.scientificIdentity.begin(), arm.scientificIdentity.end(),
        [name](const auto& field) { return field.name == name; });
    assert(found != arm.scientificIdentity.end());
    return *found;
}

Pair::IdentityField& Identity(Pair::ArmResultSet& arm, std::string_view name)
{
    return const_cast<Pair::IdentityField&>(Identity(
        static_cast<const Pair::ArmResultSet&>(arm), name));
}

void ExpectSeedParseFailure(std::string_view value)
{
    try
    {
        (void)Planning::ParseReplicationSeeds(value);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
    }
}

class FixtureEquivalents final : public Planning::EquivalentExperimentSource
{
public:
    std::map<std::pair<unsigned int, std::string>,
             Planning::EquivalentExperimentResult> results;
    mutable std::size_t calls = 0;

    Planning::EquivalentExperimentResult FindEquivalent(
        const Pair::ArmResultSet& arm) const override
    {
        ++calls;
        const unsigned int seed = static_cast<unsigned int>(
            std::stoul(*Identity(arm, "fresh_initialization_seed").value));
        const std::string mask = *Identity(arm, "feature_ablation_mask").value;
        const auto found = results.find({seed, mask});
        return found == results.end()
            ? Planning::EquivalentExperimentResult{}
            : found->second;
    }
};

Planning::Plan PlanFor(const Feature::ArmEvidence& armA,
                       const Feature::ArmEvidence& armB,
                       const std::vector<unsigned int>& seeds,
                       const Planning::EquivalentExperimentSource* equivalents =
                           nullptr)
{
    return Planning::MakePlan(
        Pair::MakeArmResultSet(armA), Pair::MakeArmResultSet(armB),
        Pair::MakeComparisonRequest(armA, armB), seeds, equivalents);
}

class FixtureEvidence final : public Pair::EvidenceSource
{
public:
    std::map<long long, Feature::ArmEvidence> arms;
    mutable std::vector<long long> loads;

    Feature::ArmEvidence Load(long long id) const override
    {
        loads.push_back(id);
        const auto found = arms.find(id);
        if (found == arms.end())
            throw Pair::EvidenceUnavailableError("fixture_not_found");
        return found->second;
    }
};

} // namespace

int main()
{
    assert((Pair::ParseExperimentIdPair("656:657") ==
            std::pair<long long, long long>{656, 657}));
    for (std::string_view malformed : {"", "0:2", "1:1", "1:2:3"})
    {
        try
        {
            (void)Pair::ParseExperimentIdPair(malformed);
            assert(false);
        }
        catch (const std::invalid_argument&)
        {
        }
    }

    assert(Planning::ParseReplicationSeeds("44,45,46") ==
           std::vector<unsigned int>({44, 45, 46}));
    for (std::string_view malformed : {
             "", ",", "44,", ",44", "44,,45", "44,44", "0", "-1",
             "+1", "01", " 44", "44 ", "4294967296", "x"})
        ExpectSeedParseFailure(malformed);

    const auto sourceA = Arm(656, "", 42);
    const auto sourceB = Arm(657, std::string{kMask}, 42);
    auto plan = PlanFor(sourceA, sourceB, {44, 45, 46});
    assert(plan.state == Planning::PlanState::Valid);
    assert(plan.requestedSeeds == std::vector<unsigned int>({44, 45, 46}));
    assert(plan.pairs.size() == 3);
    for (std::size_t index = 0; index < plan.pairs.size(); ++index)
    {
        const auto& pair = plan.pairs[index];
        assert(pair.ordinal == index + 1);
        assert(pair.requestedSeed == 44 + index);
        assert(Identity(pair.armA.proposed, "fresh_initialization_seed").value ==
               std::to_string(pair.requestedSeed));
        assert(Identity(pair.armB.proposed, "fresh_initialization_seed").value ==
               std::to_string(pair.requestedSeed));
        assert(pair.armA.changedFromSource.size() == 1);
        assert(pair.armB.changedFromSource.size() == 1);
        assert(pair.armA.changedFromSource[0].field ==
               "fresh_initialization_seed");
        assert(pair.armB.changedFromSource[0].field ==
               "fresh_initialization_seed");
        assert(pair.intentionalDifferences.size() == 1);
        assert(pair.intentionalDifferences[0].field ==
               "feature_ablation_mask");
        assert(pair.intentionalDifferences[0].armA.empty());
        assert(pair.intentionalDifferences[0].armB == kMask);
        assert(pair.unexpectedDifferences.empty());
        assert(pair.replicationClassification ==
               "different_seed_replications");
    }

    // Reusing the source seed is accepted but explicitly classified as a
    // same-seed repetition and reports no changed field.
    const auto sameSeed = PlanFor(sourceA, sourceB, {42});
    assert(sameSeed.state == Planning::PlanState::Valid);
    assert(sameSeed.pairs[0].sourceSeedReuse);
    assert(sameSeed.pairs[0].replicationClassification ==
           "same_seed_repeated_pairs");
    assert(sameSeed.pairs[0].armA.changedFromSource.empty());
    assert(sameSeed.pairs[0].armB.changedFromSource.empty());

    auto mismatchedSeedB = sourceB;
    mismatchedSeedB.extended.freshInitializationSeed = 43;
    const auto seedMismatch = PlanFor(sourceA, mismatchedSeedB, {44});
    assert(seedMismatch.state == Planning::PlanState::Invalid);

    auto incompatibleB = sourceB;
    incompatibleB.authoritative.configuration.predictionHorizon = 6;
    const auto incompatible = PlanFor(sourceA, incompatibleB, {44});
    assert(incompatible.state == Planning::PlanState::Invalid);

    // Any non-seed proposal mutation fails exact source-relative preflight.
    plan.pairs[0].armA.proposed.scientificIdentity.front().value = "gbpusdrmp";
    Planning::RecomputePreflight(plan);
    assert(plan.pairs[0].preflightState == Planning::PlanState::Invalid);

    // An unavailable required identity is undetermined, never converted to a
    // zero, empty value, or mismatch.
    auto missing = Pair::MakeArmResultSet(sourceA);
    Identity(missing, "model_input_width").value.reset();
    const auto missingPlan = Planning::MakePlan(
        missing, Pair::MakeArmResultSet(sourceB),
        Pair::MakeComparisonRequest(sourceA, sourceB), {44});
    assert(missingPlan.state ==
           Planning::PlanState::UndeterminedDueToMissingEvidence);

    FixtureEquivalents equivalents;
    equivalents.results[{44, ""}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentFound,
        {700}, ""};
    equivalents.results[{44, std::string{kMask}}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentAmbiguous,
        {701, 702}, "multiple_exact_equivalents"};
    const auto equivalentPlan = PlanFor(sourceA, sourceB, {44, 45},
                                        &equivalents);
    assert(equivalents.calls == 4);
    assert(equivalentPlan.pairs[0].armA.equivalent.state ==
           Planning::EquivalentExperimentState::EquivalentExperimentFound);
    assert(equivalentPlan.pairs[0].armB.equivalent.state ==
           Planning::EquivalentExperimentState::EquivalentExperimentAmbiguous);
    assert(equivalentPlan.pairs[1].armA.equivalent.state ==
           Planning::EquivalentExperimentState::NoEquivalentExperimentFound);
    assert(equivalentPlan.state == Planning::PlanState::Valid);

    const std::string rendered = Planning::Render(equivalentPlan);
    assert(rendered == Planning::Render(equivalentPlan));
    assert(rendered.find("replication_dimension=fresh_initialization_seed") !=
           std::string::npos);
    assert(rendered.find("statistical_independence=not_inferred") !=
           std::string::npos);
    assert(rendered.find("read_only=true") != std::string::npos);
    assert(rendered.find("equivalent_experiment_found") != std::string::npos);
    assert(rendered.find("equivalent_experiment_ambiguous") !=
           std::string::npos);
    assert(rendered.find("winner") == std::string::npos);
    assert(rendered.find("ranking") == std::string::npos);
    assert(rendered.find("recommendation") == std::string::npos);
    assert(rendered.find("queue_decision") == std::string::npos);
    assert(rendered.find("queued=") == std::string::npos);

    FixtureEvidence evidence;
    evidence.arms.emplace(656, sourceA);
    evidence.arms.emplace(657, sourceB);
    std::ostringstream output;
    std::ostringstream errors;
    Planning::PlanningCommand command{{656, 657}, {44, 45}};
    assert(Planning::RunPlanningCommand(
               command, evidence, equivalents, output, errors) == 0);
    assert((evidence.loads == std::vector<long long>{656, 657}));
    assert(errors.str().empty());
    assert(output.str().find("pair_count=2") != std::string::npos);

    Planning::PlanningCommand missingCommand{{656, 999}, {44}};
    output.str({});
    errors.str({});
    assert(Planning::RunPlanningCommand(
               missingCommand, evidence, equivalents, output, errors) == 3);
    assert(errors.str().find("exit_code=3") != std::string::npos);

    std::cout << "ExperimentReplicationPlanningTests passed\n";
}
