#include "ExperimentReplicationMaterialization.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace Materialization = EA::ExperimentReplicationMaterialization;
namespace Planning = EA::ExperimentReplicationPlanning;
namespace Pair = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;

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

class Evidence final : public Pair::EvidenceSource
{
public:
    std::map<long long, Feature::ArmEvidence> arms;

    Feature::ArmEvidence Load(long long id) const override
    {
        const auto found = arms.find(id);
        if (found == arms.end())
            throw Pair::EvidenceUnavailableError("fixture_missing");
        return found->second;
    }
};

class Equivalents final : public Planning::EquivalentExperimentSource
{
public:
    std::map<std::pair<unsigned int, std::string>,
             Planning::EquivalentExperimentResult> values;

    Planning::EquivalentExperimentResult FindEquivalent(
        const Pair::ArmResultSet& arm) const override
    {
        const auto seed = static_cast<unsigned int>(std::stoul(
            *Identity(arm, "fresh_initialization_seed").value));
        const auto mask = *Identity(arm, "feature_ablation_mask").value;
        const auto found = values.find({seed, mask});
        return found == values.end()
            ? Planning::EquivalentExperimentResult{} : found->second;
    }
};

class Inserter final : public Materialization::ExperimentInserter
{
public:
    std::vector<Planning::ProposedExperimentSpecification> specifications;
    std::size_t failOnCall = 0;
    long long nextId = 1000;

    long long InsertFreshPausedExperiment(
        const Planning::ProposedExperimentSpecification& specification)
        override
    {
        specifications.push_back(specification);
        if (failOnCall != 0 && specifications.size() == failOnCall)
            throw std::runtime_error("fixture_insert_failure");
        return nextId++;
    }
};

int Run(const std::vector<unsigned int>& seeds,
        Evidence& evidence,
        Equivalents& equivalents,
        Inserter& inserter,
        std::string& output,
        std::string& errors)
{
    std::ostringstream out;
    std::ostringstream err;
    Materialization::MaterializationCommand command;
    command.sourceExperimentIds = {101, 102};
    command.requestedSeeds = seeds;
    const int result = Materialization::RunMaterializationInTransaction(
        command, evidence, equivalents, inserter, out, err);
    output = out.str();
    errors = err.str();
    return result;
}

} // namespace

int main()
{
    Evidence evidence;
    evidence.arms.emplace(101, Arm(101, "", 43));
    evidence.arms.emplace(102, Arm(102, std::string{kMask}, 43));
    Equivalents equivalents;
    Inserter inserter;
    std::string output;
    std::string errors;

    assert(Run({44}, evidence, equivalents, inserter, output, errors) == 0);
    assert(inserter.specifications.size() == 2);
    assert(inserter.specifications[0].authoritativeSourceExperimentId == 101);
    assert(inserter.specifications[1].authoritativeSourceExperimentId == 102);
    assert(inserter.specifications[0].freshInitializationSeed == 44);
    assert(inserter.specifications[1].freshInitializationSeed == 44);
    assert(Identity(inserter.specifications[0], "feature_ablation_mask").value ==
           std::optional<std::string>{""});
    assert(Identity(inserter.specifications[1], "feature_ablation_mask").value ==
           std::optional<std::string>{std::string{kMask}});
    assert(output.find("arm_a_experiment_id=1000,arm_b_experiment_id=1001") !=
           std::string::npos);
    assert(output.find("state=materialized,pair_count=1,experiment_count=2") !=
           std::string::npos);
    assert(output.find("queued=false,started=false") != std::string::npos);
    assert(output.find("statistical_independence=not_inferred") !=
           std::string::npos);
    assert(output.find("winner") == std::string::npos);
    assert(output.find("ranking") == std::string::npos);
    assert(output.find("recommendation") == std::string::npos);

    const std::string deterministicOutput = output;
    inserter = {};
    assert(Run({44}, evidence, equivalents, inserter, output, errors) == 0);
    assert(output == deterministicOutput);

    inserter = {};
    assert(Run({46, 44, 45}, evidence, equivalents, inserter,
               output, errors) == 0);
    assert(inserter.specifications.size() == 6);
    assert((std::vector<unsigned int>{
        inserter.specifications[0].freshInitializationSeed,
        inserter.specifications[1].freshInitializationSeed,
        inserter.specifications[2].freshInitializationSeed,
        inserter.specifications[3].freshInitializationSeed,
        inserter.specifications[4].freshInitializationSeed,
        inserter.specifications[5].freshInitializationSeed} ==
        std::vector<unsigned int>{46, 46, 44, 44, 45, 45}));

    equivalents.values[{44, ""}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentFound,
        {700}, ""};
    inserter = {};
    assert(Run({44, 45}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("reason=equivalence_conflict") != std::string::npos);

    equivalents.values.clear();
    equivalents.values[{44, std::string{kMask}}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentAmbiguous,
        {701, 702}, "multiple_exact_equivalents"};
    inserter = {};
    assert(Run({44}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("equivalent_experiment_ambiguous=701|702") !=
           std::string::npos);

    equivalents.values.clear();
    evidence.arms[102].extended.freshInitializationSeed = 42;
    inserter = {};
    assert(Run({44}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("scientific_preflight_invalid") != std::string::npos);
    evidence.arms[102].extended.freshInitializationSeed = 43;

    evidence.arms[102].authoritative.configuration.checkpointInterval = 21;
    inserter = {};
    assert(Run({44}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("source_pair_unexpected_difference") !=
           std::string::npos);
    evidence.arms[102].authoritative.configuration.checkpointInterval = 20;

    evidence.arms[102].extended.configuredModelInputWidth.reset();
    inserter = {};
    assert(Run({44}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("scientific_preflight_") != std::string::npos);
    evidence.arms[102].extended.configuredModelInputWidth = 80;

    // A successful first wave followed by exact equivalents is idempotent:
    // the second attempt creates no rows and never reuses existing IDs.
    inserter = {};
    assert(Run({47}, evidence, equivalents, inserter,
               output, errors) == 0);
    assert(inserter.specifications.size() == 2);
    equivalents.values[{47, ""}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentFound,
        {inserter.nextId - 2}, ""};
    equivalents.values[{47, std::string{kMask}}] = {
        Planning::EquivalentExperimentState::EquivalentExperimentFound,
        {inserter.nextId - 1}, ""};
    inserter = {};
    assert(Run({47}, evidence, equivalents, inserter,
               output, errors) == 3);
    assert(inserter.specifications.empty());
    assert(output.find("state=not_materialized") != std::string::npos);
    equivalents.values.clear();

    inserter = {};
    inserter.failOnCall = 4;
    std::ostringstream failedOutput;
    std::ostringstream failedErrors;
    Materialization::MaterializationCommand failedCommand;
    failedCommand.sourceExperimentIds = {101, 102};
    failedCommand.requestedSeeds = {44, 45, 46};
    try
    {
        (void)Materialization::RunMaterializationInTransaction(
            failedCommand, evidence, equivalents, inserter,
            failedOutput, failedErrors);
        assert(false);
    }
    catch (const std::runtime_error& error)
    {
        assert(std::string_view{error.what()} == "fixture_insert_failure");
    }
    assert(failedOutput.str().find("CONTROLLED_REPLICATION_MATERIALIZED_PAIR") ==
           std::string::npos);
    assert(failedOutput.str().find("state=materialized") == std::string::npos);

    return 0;
}
