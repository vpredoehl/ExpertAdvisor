#include "FeatureAblationFactorialComparison.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

namespace Factorial = EA::FeatureAblationFactorialComparison;
namespace Pair = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Shared = EA::PairedTrainingObjectiveEvaluation;

namespace
{

constexpr const char* kMask11 = "tg4_inner_break_any";
constexpr const char* kMask01 = "tg4_source_tg3_structurally_eligible";
constexpr const char* kMask10 = "tg4_source_tg3_confluent";

Feature::ArmEvidence Arm(long long id, std::string mask, unsigned int seed,
                         double accuracy, double aggregate)
{
    Feature::ArmEvidence result;
    auto& c = result.authoritative.configuration;
    c.experimentId = id;
    c.symbol = "audcad";
    c.predictionHorizon = 4;
    c.trainStart = "2020-01-01";
    c.trainEnd = "2025-01-01";
    c.inferenceStart = "2025-01-01";
    c.inferenceEnd = "2026-01-01";
    c.targetEpochs = 80;
    c.threshold = 0.0008;
    c.coreLearningRateMultiplier = 120.0;
    c.headLearningRateMultiplier = 25.0;
    c.checkpointInterval = 20;
    c.inputWidth = 171;
    c.hiddenSize = 64;
    c.layerCount = 1;
    c.windowSize = 64;
    c.modelMetadataSchemaVersion = 1;
    c.trainConfigurationSchemaVersion = 1;
    c.normalizationVersion = 1;
    c.classWeightDown = 1.0;
    c.classWeightNeutral = 1.0;
    c.classWeightUp = 1.0;
    c.featureWarmupScope = "full_history_warmup";
    c.donchianMode = "enabled";
    c.donchianLookback = 20;
    c.featureAblationMask = std::move(mask);
    c.optimizerMetadataSchemaVersion = 1;
    c.optimizerType = 1;
    c.optimizerFirstMomentBufferCount = 4;
    c.optimizerSecondMomentBufferCount = 4;
    c.persistedCoreLearningRateMultiplier = 120.0;
    c.persistedHeadWeightLearningRateMultiplier = 25.0;
    c.persistedHeadBiasLearningRateMultiplier = 25.0;
    c.labelRuleId = 1;
    c.targetType = 0;
    c.targetScale = 1.0;
    c.targetStandardDeviation = 1.0;
    c.modelInputMetadataSchemaVersion = 1;
    c.modelInputLayoutVersion = 13;
    c.persistedTrainingSymbol = c.symbol;
    c.persistedTrainingStart = c.trainStart;
    c.persistedTrainingEnd = c.trainEnd;
    c.experimentObjective = {"objective_id=test;", "fnv1a64:0000000000000001"};

    result.extended.configuredModelInputWidth = 171;
    result.extended.configuredModelInputLayoutVersion = 13;
    result.extended.economicCalendarSnapshotId = 50;
    result.extended.economicCalendarSnapshotHash = "fnv1a64:0000000000000002";
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

    auto& shared = result.authoritative;
    shared.experimentStatus = "completed";
    shared.experimentPhase = "done";
    shared.finalModelId = id + 1000;
    shared.trainingExecution = {id + 1, "train", 13, 171, "train", "commit",
                                "sha", "LSTM_Release", "runtime", ""};
    shared.inferenceExecution = {id + 2, "infer", 13, 171, "infer", "commit",
                                 "sha", "lstm-infer-worker", "runtime", ""};
    result.exactFinalInferenceResultId = id + 2000;
    Shared::ClassificationEvidence classification;
    classification.inferenceResultId = id + 2000;
    classification.modelId = id + 1000;
    classification.inferenceScope = "final";
    classification.status = "completed";
    classification.inferenceAccuracy = accuracy;
    classification.acceptRate = 0.5;
    classification.acceptAccuracy = 0.6;
    classification.leaderScore = 0.7;
    shared.classification = classification;
    Shared::ProfitabilityEvidence profitability;
    profitability.predictionCount = 100;
    profitability.actionableCount = 40;
    profitability.winningActionableCount = 20;
    profitability.losingActionableCount = 20;
    profitability.grossPositiveTerminalHorizonLogReturnSum = aggregate + 0.2;
    profitability.grossNegativeTerminalHorizonLogReturnSum = -0.2;
    profitability.aggregateTerminalHorizonLogReturnSum = aggregate;
    profitability.averageTerminalHorizonLogReturnPerActionablePrediction = aggregate / 40.0;
    shared.profitability = profitability;
    return result;
}

class Source final : public Pair::EvidenceSource
{
public:
    std::map<long long, Feature::ArmEvidence> values;
    Feature::ArmEvidence Load(long long id) const override
    {
        const auto found = values.find(id);
        if (found == values.end()) throw Pair::EvidenceUnavailableError("missing");
        return found->second;
    }
};

Factorial::Command Command()
{
    return Factorial::ParseCommand(
        "101:102:103:104;105:106:107:108@" + std::string(kMask11) + "/" +
        kMask01 + "/" + kMask10 + "/");
}

Factorial::Command CustomLabelCommand()
{
    return Factorial::ParseCommand(
        "retracement_ages:a_boundary_ages|101:102:103:104;105:106:107:108@" +
        std::string(kMask11) + "/" + kMask01 + "/" + kMask10 + "/");
}

Source ValidSource()
{
    Source source;
    // Seed one: Y11=10, Y01=6, Y10=7, Y00=1.
    source.values.emplace(101, Arm(101, kMask11, 1002, 0.90, 10.0));
    source.values.emplace(102, Arm(102, kMask01, 1002, 0.70, 6.0));
    source.values.emplace(103, Arm(103, kMask10, 1002, 0.75, 7.0));
    source.values.emplace(104, Arm(104, "", 1002, 0.50, 1.0));
    // Seed two doubles each effect: Y11=20, Y01=12, Y10=14, Y00=2.
    source.values.emplace(105, Arm(105, kMask11, 1003, 0.90, 20.0));
    source.values.emplace(106, Arm(106, kMask01, 1003, 0.70, 12.0));
    source.values.emplace(107, Arm(107, kMask10, 1003, 0.75, 14.0));
    source.values.emplace(108, Arm(108, "", 1003, 0.50, 2.0));
    return source;
}

const Factorial::MetricSummary& Metric(const Factorial::Report& report,
                                       std::string_view name)
{
    const auto found = std::find_if(report.metrics.begin(), report.metrics.end(),
        [name](const auto& metric) { return metric.name == name; });
    assert(found != report.metrics.end());
    return *found;
}

} // namespace

int main()
{
    const auto command = Command();
    assert(command.seedCells.size() == 2);
    assert(command.design.expectedMasks[0] == kMask11);
    assert(command.design.expectedMasks[3].empty());
    assert(command.design.factorALabel == "counts");
    assert(command.design.factorBLabel == "ages");

    const auto source = ValidSource();
    const auto report = Factorial::Evaluate(command, source);
    assert(report.declaredSeedCount == 2);
    assert(report.eligibleSeedCount == 2);
    const auto& aggregate = Metric(report, "aggregate_return");
    assert(aggregate.eligibleSeedCount == 2);
    assert(aggregate.availableSeedCount == 2);
    assert(std::fabs(*report.seeds[0].effects[12].factorAMainEffect - 5.0) < 1e-12);
    assert(std::fabs(*report.seeds[0].effects[12].factorBMainEffect - 4.0) < 1e-12);
    assert(std::fabs(*report.seeds[0].effects[12].interaction + 2.0) < 1e-12);
    assert(std::fabs(*aggregate.descriptiveFactorAMainEffectMean - 7.5) < 1e-12);
    assert(std::fabs(*aggregate.descriptiveFactorBMainEffectMean - 6.0) < 1e-12);
    assert(std::fabs(*aggregate.descriptiveInteractionMean + 3.0) < 1e-12);

    // Caller-declared labels are durable report identity, not arithmetic
    // inputs. They allow a subsequent factorial to name different features.
    const auto customLabels = Factorial::Evaluate(CustomLabelCommand(), source);
    assert(customLabels.design.factorALabel == "retracement_ages");
    assert(customLabels.design.factorBLabel == "a_boundary_ages");
    const auto& customAggregate = Metric(customLabels, "aggregate_return");
    assert(customAggregate.descriptiveFactorAMainEffectMean ==
           aggregate.descriptiveFactorAMainEffectMean);
    assert(customAggregate.descriptiveFactorBMainEffectMean ==
           aggregate.descriptiveFactorBMainEffectMean);
    assert(customAggregate.descriptiveInteractionMean ==
           aggregate.descriptiveInteractionMean);
    const std::string customRendered = Factorial::Render(customLabels);
    assert(customRendered.find("factor_a=retracement_ages") != std::string::npos);
    assert(customRendered.find("factor_b=a_boundary_ages") != std::string::npos);
    // All three baseline comparisons intentionally permit the declared,
    // nested non-empty ablation contrasts only in this factorial path.
    assert(report.seeds[0].comparisons[0].intentionalDifferences.size() == 1);
    assert(report.seeds[0].comparisons[1].intentionalDifferences.size() == 1);
    assert(report.seeds[0].comparisons[2].intentionalDifferences.size() == 1);
    assert(Factorial::Render(report) == Factorial::Render(report));

    // The ordinary pair/replication policy remains unchanged: two non-empty
    // masks do not become intentional merely because they differ.
    const auto ordinary = Pair::MakeComparisonRequest(
        source.values.at(101), source.values.at(102));
    assert(ordinary.intentionalDifferenceFields.empty());

    auto maskMismatch = ValidSource();
    maskMismatch.values.at(102).authoritative.configuration.featureAblationMask = kMask10;
    const auto mismatchReport = Factorial::Evaluate(command, maskMismatch);
    assert(!mismatchReport.seeds[0].eligible);

    auto identityMismatch = ValidSource();
    identityMismatch.values.at(103).authoritative.configuration.predictionHorizon = 8;
    const auto identityReport = Factorial::Evaluate(command, identityMismatch);
    assert(!identityReport.seeds[0].eligible);

    auto provenanceMismatch = ValidSource();
    provenanceMismatch.values.at(104).authoritative.trainingExecution->executableSha256 = "other";
    const auto provenanceReport = Factorial::Evaluate(command, provenanceMismatch);
    assert(!provenanceReport.seeds[0].eligible);

    auto seedMismatch = ValidSource();
    seedMismatch.values.at(104).extended.freshInitializationSeed = 1003;
    const auto seedReport = Factorial::Evaluate(command, seedMismatch);
    assert(!seedReport.seeds[0].eligible);

    auto missingMetric = ValidSource();
    missingMetric.values.at(104).authoritative.classification->acceptAccuracy.reset();
    const auto incomplete = Factorial::Evaluate(command, missingMetric);
    const auto& acceptAccuracy = Metric(incomplete, "accept_accuracy");
    assert(incomplete.eligibleSeedCount == 2);
    assert(acceptAccuracy.availableSeedCount == 1);
    assert(!incomplete.seeds[0].effects[2].factorAMainEffect);
    assert(incomplete.seeds[1].effects[2].factorAMainEffect);

    for (std::string_view invalid : {"", "101:102:103@a/b/c/d",
                                     "101:102:103:104@a/b/c",
                                     "101:102:103:104@a/a/b/c",
                                     "Retracement:ages|101:102:103:104@a/b/c/d",
                                     "retracement-ages:ages|101:102:103:104@a/b/c/d",
                                     "ages:ages|101:102:103:104@a/b/c/d",
                                     "retracement:ages:extra|101:102:103:104@a/b/c/d"})
    {
        try { (void)Factorial::ParseCommand(invalid); assert(false); }
        catch (const std::invalid_argument&) {}
    }

    std::ostringstream output, errors;
    assert(Factorial::RunCommand(command, source, output, errors) == 0);
    assert(errors.str().empty());
    assert(output.str().find("statistical_significance=NOT_INFERRED") != std::string::npos);
    assert(output.str().find("subjective_winner=NONE") != std::string::npos);

    std::cout << "FeatureAblationFactorialComparisonTests passed\n";
}
