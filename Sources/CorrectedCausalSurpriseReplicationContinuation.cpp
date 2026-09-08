#include "CorrectedCausalSurpriseReplicationContinuation.hpp"

#include "CanonicalSymbol.hpp"
#include "FeatureAblation.hpp"
#include "SupportedSymbols.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cmath>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::CorrectedCausalSurpriseReplicationContinuation
{
namespace
{

constexpr std::string_view kPolicyName =
    "corrected_causal_surprise_outcome_blind_replication_policy_v1";
constexpr std::string_view kIndependence = "symbol_and_prediction_horizon";

bool TaggedHash(std::string_view value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    return std::all_of(value.begin() + 8, value.end(), [](char character)
    {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
    });
}

std::string OptionalDouble(const std::optional<double>& value)
{
    return value ? TrainingObjective::CanonicalDouble(*value) : "NULL";
}

std::string OptionalInt(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalHash(const std::optional<std::string>& value)
{
    return value.value_or("NULL");
}

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

bool IsoDate(std::string_view value)
{
    if (value.size() < 10 || value[4] != '-' || value[7] != '-') return false;
    for (std::size_t index = 0; index < 10; ++index)
        if (index != 4 && index != 7 &&
            (value[index] < '0' || value[index] > '9'))
            return false;
    return true;
}

std::string DateOnly(std::string_view value)
{
    if (!IsoDate(value))
        throw std::invalid_argument("corrected_replication_date_invalid");
    return std::string{value.substr(0, 10)};
}

bool SupportedSymbol(std::string_view value)
{
    const auto& symbols = SupportedSymbols::TrainingSymbols();
    return std::find(symbols.begin(), symbols.end(), value) != symbols.end();
}

std::string ConfigurationCanonical(const ScientificConfiguration& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "initial_symbol=" << value.initialSymbol << ';'
        << "initial_prediction_horizon="
        << value.initialPredictionHorizon << ';'
        << "target_epochs=" << value.targetEpochs << ';'
        << "threshold=" << TrainingObjective::CanonicalDouble(value.threshold)
        << ';'
        << "core_lr_multiplier="
        << OptionalDouble(value.coreLearningRateMultiplier) << ';'
        << "head_lr_multiplier="
        << OptionalDouble(value.headLearningRateMultiplier) << ';'
        << "checkpoint_interval=" << value.checkpointInterval << ';'
        << "training_objective_canonical="
        << value.trainingObjectiveCanonical << ';'
        << "training_objective_hash=" << value.trainingObjectiveHash << ';'
        << "train_start=" << DateOnly(value.trainStart) << ';'
        << "train_end=" << DateOnly(value.trainEnd) << ';'
        << "inference_start=" << DateOnly(value.inferenceStart) << ';'
        << "inference_end=" << DateOnly(value.inferenceEnd) << ';'
        << "donchian_mode=" << value.donchianMode << ';'
        << "feature_warmup_scope=" << value.featureWarmupScope << ';'
        << "donchian_lookback=" << value.donchianLookback << ';'
        << "model_input_width=" << value.modelInputWidth << ';'
        << "semantic_layout_version=" << value.semanticLayoutVersion << ';'
        << "economic_calendar_snapshot_id="
        << value.economicCalendarSnapshotId << ';'
        << "economic_calendar_snapshot_hash="
        << value.economicCalendarSnapshotHash << ';'
        << "base_learning_rate="
        << TrainingObjective::CanonicalDouble(value.baseLearningRate) << ';'
        << "batch_size=" << value.batchSize << ';'
        << "fresh_initialization_seed="
        << (value.freshInitializationSeed
                ? std::to_string(*value.freshInitializationSeed) : "NULL")
        << ';'
        << "checkpoint_inference_enabled="
        << Boolean(value.checkpointInferenceEnabled) << ';'
        << "checkpoint_inference_minimum_epoch="
        << OptionalInt(value.checkpointInferenceMinimumEpoch) << ';'
        << "checkpoint_inference_interval="
        << OptionalInt(value.checkpointInferenceInterval) << ';'
        << "checkpoint_policy_enabled="
        << Boolean(value.checkpointPolicyEnabled) << ';'
        << "checkpoint_policy_minimum_leader_score="
        << OptionalDouble(value.checkpointPolicyMinimumLeaderScore) << ';'
        << "checkpoint_policy_minimum_inference_accuracy="
        << OptionalDouble(value.checkpointPolicyMinimumInferenceAccuracy) << ';'
        << "checkpoint_policy_top_n="
        << OptionalInt(value.checkpointPolicyTopN) << ';'
        << "checkpoint_policy_scope=" << value.checkpointPolicyScope << ';'
        << "checkpoint_policy_stop_mode="
        << value.checkpointPolicyStopMode << ';'
        << "checkpoint_policy_grace_evaluations="
        << value.checkpointPolicyGraceEvaluations << ';'
        << "checkpoint_policy_revision="
        << value.checkpointPolicyRevision << ';'
        << "checkpoint_policy_hash="
        << OptionalHash(value.checkpointPolicyHash) << ';'
        << "continuation_policy_enabled="
        << Boolean(value.continuationPolicyEnabled) << ';'
        << "continuation_policy_scientific_identity="
        << value.continuationPolicyScientificIdentity << ';'
        << "scheduler_priority=" << value.schedulerPriority << ';';
    return out.str();
}

std::string ArmCanonical(const Pair& pair,
                         std::string_view role,
                         std::string_view mask,
                         std::string_view configurationHash)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "corrected_causal_surprise_replication_arm_v1;"
        << "pair_ordinal=" << pair.ordinal << ';'
        << "role=" << role << ';'
        << "symbol=" << pair.symbol << ';'
        << "prediction_horizon=" << pair.predictionHorizon << ';'
        << "feature_ablation_mask=" << mask << ';'
        << "configuration_hash=" << configurationHash << ';';
    return out.str();
}

void Add(std::vector<std::string>& reasons, std::string reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(std::move(reason));
}

[[noreturn]] void PlannedArmMismatch(std::string_view field)
{
    throw std::invalid_argument(
        "corrected_replication_planned_arm_" + std::string{field} +
        "_mismatch");
}

template <typename Left, typename Right>
void RequireEqual(const Left& actual,
                  const Right& expected,
                  std::string_view field)
{
    if (actual != expected) PlannedArmMismatch(field);
}

} // namespace

Plan MakePlan(const ScientificConfiguration& input)
{
    Plan plan;
    plan.configuration = input;
    plan.configuration.initialSymbol =
        CanonicalSymbol::Normalize(input.initialSymbol);
    plan.configuration.trainStart = DateOnly(input.trainStart);
    plan.configuration.trainEnd = DateOnly(input.trainEnd);
    plan.configuration.inferenceStart = DateOnly(input.inferenceStart);
    plan.configuration.inferenceEnd = DateOnly(input.inferenceEnd);

    const std::string configurationCanonical =
        ConfigurationCanonical(plan.configuration);
    const std::string configurationHash =
        TrainingObjective::DeterministicHash(configurationCanonical);

    // This order is predeclared. It is the smallest subset of the established
    // supported-symbol replication ladder that adds two symbols and a second
    // horizon after the EURUSD/h4 anchor. No observed metric is an input.
    const std::pair<std::string_view, int> fixedUnits[] = {
        {"gbpusdrmp", 6},
        {"usdcadrmp", 6}
    };
    for (std::size_t index = 0; index < std::size(fixedUnits); ++index)
    {
        Pair pair;
        pair.ordinal = index + 1;
        pair.symbol = std::string{fixedUnits[index].first};
        pair.predictionHorizon = fixedUnits[index].second;
        pair.independentDimension = std::string{kIndependence};
        std::ostringstream unit;
        unit << "corrected_causal_surprise_replication_unit_v1;"
             << "symbol=" << pair.symbol << ';'
             << "prediction_horizon=" << pair.predictionHorizon << ';'
             << "configuration_hash=" << configurationHash << ';'
             << "fresh_initialization_seed="
             << (plan.configuration.freshInitializationSeed
                     ? std::to_string(
                           *plan.configuration.freshInitializationSeed)
                     : "NULL") << ';';
        pair.replicationUnitCanonical = unit.str();
        pair.replicationUnitHash = TrainingObjective::DeterministicHash(
            pair.replicationUnitCanonical);
        pair.control.role = "control";
        pair.control.featureAblationMask.clear();
        pair.control.scientificIdentityCanonical = ArmCanonical(
            pair, pair.control.role, pair.control.featureAblationMask,
            configurationHash);
        pair.control.scientificIdentityHash =
            TrainingObjective::DeterministicHash(
                pair.control.scientificIdentityCanonical);
        pair.treatment.role = "treatment";
        pair.treatment.featureAblationMask =
            std::string{kCausalEconomicEventSurpriseAblationMaskText};
        pair.treatment.scientificIdentityCanonical = ArmCanonical(
            pair, pair.treatment.role, pair.treatment.featureAblationMask,
            configurationHash);
        pair.treatment.scientificIdentityHash =
            TrainingObjective::DeterministicHash(
                pair.treatment.scientificIdentityCanonical);
        plan.pairs.push_back(std::move(pair));
    }

    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "corrected_causal_surprise_replication_plan_v1;"
              << "plan_semantic_version=" << plan.semanticVersion << ';'
              << "scientific_policy=" << kPolicyName << ';'
              << "scientific_policy_version="
              << plan.scientificPolicyVersion << ';'
              << "minimum_valid_replications="
              << Replication::kMinimumValidReplications << ';'
              << "independent_dimension=" << kIndependence << ';'
              << "configuration=" << configurationCanonical
              << "configuration_hash=" << configurationHash << ';'
              << "follow_on_pair_count=" << plan.pairs.size() << ';';
    for (const Pair& pair : plan.pairs)
        canonical << "pair_ordinal=" << pair.ordinal
                  << ",replication_unit_hash=" << pair.replicationUnitHash
                  << ",control_identity_hash="
                  << pair.control.scientificIdentityHash
                  << ",treatment_identity_hash="
                  << pair.treatment.scientificIdentityHash << ';';
    plan.canonical = canonical.str();
    plan.hash = TrainingObjective::DeterministicHash(plan.canonical);
    ValidatePlan(plan);
    return plan;
}

void ValidatePlan(const Plan& plan)
{
    const ScientificConfiguration& value = plan.configuration;
    if (plan.semanticVersion != kPlanSemanticVersion ||
        plan.scientificPolicyVersion != kScientificPolicyVersion)
        throw std::invalid_argument("corrected_replication_plan_version_invalid");
    if (value.initialSymbol != "eurusdrmp" ||
        value.initialPredictionHorizon != 4)
        throw std::invalid_argument("corrected_replication_anchor_unit_invalid");
    if (value.targetEpochs <= 0 || value.checkpointInterval <= 0 ||
        value.donchianLookback <= 0 || value.batchSize <= 0 ||
        !std::isfinite(value.threshold) ||
        !value.coreLearningRateMultiplier ||
        !std::isfinite(*value.coreLearningRateMultiplier) ||
        !value.headLearningRateMultiplier ||
        !std::isfinite(*value.headLearningRateMultiplier) ||
        !std::isfinite(value.baseLearningRate))
        throw std::invalid_argument("corrected_replication_configuration_invalid");
    if (DateOnly(value.trainStart) >= DateOnly(value.trainEnd) ||
        DateOnly(value.inferenceStart) >= DateOnly(value.inferenceEnd))
        throw std::invalid_argument("corrected_replication_date_range_invalid");
    if (value.modelInputWidth !=
            FeatureAblationPairEvaluation::kCausalSurpriseModelInputWidth ||
        value.semanticLayoutVersion !=
            FeatureAblationPairEvaluation::
                kCorrectedCausalSurpriseSemanticLayoutVersion)
        throw std::invalid_argument("corrected_replication_input_identity_invalid");
    if (value.economicCalendarSnapshotId <= 0 ||
        !TaggedHash(value.economicCalendarSnapshotHash))
        throw std::invalid_argument("corrected_replication_snapshot_invalid");
    if (!TaggedHash(value.trainingObjectiveHash) ||
        TrainingObjective::DeterministicHash(
            value.trainingObjectiveCanonical) != value.trainingObjectiveHash)
        throw std::invalid_argument("corrected_replication_objective_invalid");
    try
    {
        (void)TrainingObjective::ParseSupportedCanonicalText(
            value.trainingObjectiveCanonical);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("corrected_replication_objective_unsupported");
    }
    if (value.donchianMode.empty() || value.featureWarmupScope.empty() ||
        value.checkpointPolicyScope.empty() ||
        value.checkpointPolicyStopMode.empty() ||
        value.checkpointPolicyGraceEvaluations <= 0 ||
        value.checkpointPolicyRevision <= 0 ||
        value.continuationPolicyScientificIdentity.empty() ||
        value.schedulerPriority != "high" ||
        value.freshInitializationSeed != std::optional<unsigned int>{42U})
        throw std::invalid_argument("corrected_replication_runtime_policy_invalid");
    if (value.checkpointInferenceEnabled || value.checkpointPolicyEnabled ||
        value.continuationPolicyEnabled)
        throw std::invalid_argument(
            "corrected_replication_unsupported_active_runtime_policy");
    if (plan.pairs.size() != kFollowOnPairCount)
        throw std::invalid_argument("corrected_replication_pair_count_invalid");

    const std::string configurationCanonical = ConfigurationCanonical(value);
    const std::string configurationHash =
        TrainingObjective::DeterministicHash(configurationCanonical);
    std::set<std::string> units;
    std::set<std::string> arms;
    for (std::size_t index = 0; index < plan.pairs.size(); ++index)
    {
        const Pair& pair = plan.pairs[index];
        if (pair.ordinal != index + 1 || !SupportedSymbol(pair.symbol) ||
            pair.predictionHorizon <= 0 ||
            pair.independentDimension != kIndependence ||
            (pair.symbol == value.initialSymbol &&
             pair.predictionHorizon == value.initialPredictionHorizon) ||
            pair.replicationUnitCanonical.empty() ||
            !TaggedHash(pair.replicationUnitHash) ||
            TrainingObjective::DeterministicHash(
                pair.replicationUnitCanonical) != pair.replicationUnitHash ||
            !units.insert(pair.replicationUnitHash).second)
            throw std::invalid_argument("corrected_replication_unit_invalid");
        std::ostringstream expectedUnit;
        expectedUnit.imbue(std::locale::classic());
        expectedUnit
            << "corrected_causal_surprise_replication_unit_v1;"
            << "symbol=" << pair.symbol << ';'
            << "prediction_horizon=" << pair.predictionHorizon << ';'
            << "configuration_hash=" << configurationHash << ';'
            << "fresh_initialization_seed="
            << (value.freshInitializationSeed
                    ? std::to_string(*value.freshInitializationSeed)
                    : "NULL") << ';';
        if (pair.replicationUnitCanonical != expectedUnit.str())
            throw std::invalid_argument(
                "corrected_replication_unit_canonical_mismatch");
        if (pair.control.role != "control" ||
            !pair.control.featureAblationMask.empty() ||
            pair.treatment.role != "treatment" ||
            FeatureAblationMask::Parse(
                pair.treatment.featureAblationMask).CanonicalText() !=
                kCausalEconomicEventSurpriseAblationMaskText)
            throw std::invalid_argument("corrected_replication_arm_mask_invalid");
        for (const Arm* arm : {&pair.control, &pair.treatment})
        {
            const std::string expectedCanonical = ArmCanonical(
                pair, arm->role, arm->featureAblationMask,
                configurationHash);
            if (arm->scientificIdentityCanonical.empty() ||
                arm->scientificIdentityCanonical != expectedCanonical ||
                !TaggedHash(arm->scientificIdentityHash) ||
                TrainingObjective::DeterministicHash(
                    arm->scientificIdentityCanonical) !=
                    arm->scientificIdentityHash ||
                !arms.insert(arm->scientificIdentityHash).second)
                throw std::invalid_argument("corrected_replication_arm_identity_invalid");
        }
    }
    if (plan.pairs[0].symbol != "gbpusdrmp" ||
        plan.pairs[0].predictionHorizon != 6 ||
        plan.pairs[1].symbol != "usdcadrmp" ||
        plan.pairs[1].predictionHorizon != 6)
        throw std::invalid_argument("corrected_replication_fixed_order_invalid");
    std::ostringstream expectedPlanCanonical;
    expectedPlanCanonical.imbue(std::locale::classic());
    expectedPlanCanonical
        << "corrected_causal_surprise_replication_plan_v1;"
        << "plan_semantic_version=" << plan.semanticVersion << ';'
        << "scientific_policy=" << kPolicyName << ';'
        << "scientific_policy_version=" << plan.scientificPolicyVersion << ';'
        << "minimum_valid_replications="
        << Replication::kMinimumValidReplications << ';'
        << "independent_dimension=" << kIndependence << ';'
        << "configuration=" << configurationCanonical
        << "configuration_hash=" << configurationHash << ';'
        << "follow_on_pair_count=" << plan.pairs.size() << ';';
    for (const Pair& pair : plan.pairs)
        expectedPlanCanonical
            << "pair_ordinal=" << pair.ordinal
            << ",replication_unit_hash=" << pair.replicationUnitHash
            << ",control_identity_hash="
            << pair.control.scientificIdentityHash
            << ",treatment_identity_hash="
            << pair.treatment.scientificIdentityHash << ';';
    if (plan.canonical != expectedPlanCanonical.str())
        throw std::invalid_argument(
            "corrected_replication_plan_canonical_mismatch");
    if (plan.canonical.empty() || !TaggedHash(plan.hash) ||
        TrainingObjective::DeterministicHash(plan.canonical) != plan.hash)
        throw std::invalid_argument("corrected_replication_plan_identity_invalid");
    if (plan.hash != kPredeclaredPlanHash ||
        plan.pairs[0].replicationUnitHash != kFirstReplicationUnitHash ||
        plan.pairs[0].control.scientificIdentityHash !=
            kFirstControlIdentityHash ||
        plan.pairs[0].treatment.scientificIdentityHash !=
            kFirstTreatmentIdentityHash ||
        plan.pairs[1].replicationUnitHash != kSecondReplicationUnitHash ||
        plan.pairs[1].control.scientificIdentityHash !=
            kSecondControlIdentityHash ||
        plan.pairs[1].treatment.scientificIdentityHash !=
            kSecondTreatmentIdentityHash)
        throw std::invalid_argument(
            "corrected_replication_predeclared_plan_mismatch");
}

std::string MaterializationProvenance(const Plan& plan,
                                      const Pair& pair,
                                      const Arm& arm)
{
    ValidatePlan(plan);
    if (arm != pair.control && arm != pair.treatment)
        throw std::invalid_argument(
            "corrected_replication_arm_not_owned_by_pair");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "corrected_causal_surprise_replication_materialization_v1;"
        << "plan_hash=" << plan.hash << ';'
        << "plan_semantic_version=" << plan.semanticVersion << ';'
        << "scientific_policy_version=" << plan.scientificPolicyVersion << ';'
        << "outcome_blind=true;"
        << "pair_ordinal=" << pair.ordinal << ';'
        << "replication_unit_hash=" << pair.replicationUnitHash << ';'
        << "arm_role=" << arm.role << ';'
        << "arm_identity_hash=" << arm.scientificIdentityHash << ';';
    return out.str();
}

void ValidatePlannedArmEvidence(
    const Plan& plan,
    const Pair& pair,
    const Arm& arm,
    const FeatureAblationPairEvaluation::ArmEvidence& evidence)
{
    ValidatePlan(plan);
    const auto& expected = plan.configuration;
    const auto& actual = evidence.authoritative.configuration;
    const auto& extended = evidence.extended;
    RequireEqual(actual.symbol, pair.symbol, "symbol");
    RequireEqual(actual.predictionHorizon, pair.predictionHorizon, "horizon");
    RequireEqual(actual.targetEpochs, expected.targetEpochs, "target_epochs");
    RequireEqual(actual.threshold, expected.threshold, "threshold");
    RequireEqual(actual.coreLearningRateMultiplier,
                 expected.coreLearningRateMultiplier, "core_lr");
    RequireEqual(actual.headLearningRateMultiplier,
                 expected.headLearningRateMultiplier, "head_lr");
    RequireEqual(actual.checkpointInterval, expected.checkpointInterval,
                 "checkpoint_interval");
    RequireEqual(actual.trainStart, expected.trainStart, "train_start");
    RequireEqual(actual.trainEnd, expected.trainEnd, "train_end");
    RequireEqual(actual.inferenceStart, expected.inferenceStart,
                 "inference_start");
    RequireEqual(actual.inferenceEnd, expected.inferenceEnd, "inference_end");
    RequireEqual(actual.donchianMode, expected.donchianMode, "donchian_mode");
    RequireEqual(actual.featureWarmupScope, expected.featureWarmupScope,
                 "feature_warmup_scope");
    RequireEqual(actual.donchianLookback, expected.donchianLookback,
                 "donchian_lookback");
    RequireEqual(actual.featureAblationMask, arm.featureAblationMask, "mask");
    RequireEqual(actual.resumeModelId, std::optional<long long>{},
                 "resume_model_id");
    RequireEqual(actual.resumeExpandInputWidth, false,
                 "resume_expand_input_width");
    RequireEqual(actual.experimentObjective.canonical,
                 expected.trainingObjectiveCanonical, "objective_canonical");
    RequireEqual(actual.experimentObjective.hash,
                 expected.trainingObjectiveHash, "objective_hash");
    RequireEqual(extended.configuredModelInputWidth,
                 std::optional<int>{expected.modelInputWidth}, "input_width");
    RequireEqual(extended.configuredModelInputLayoutVersion,
                 std::optional<int>{expected.semanticLayoutVersion}, "layout");
    RequireEqual(extended.economicCalendarSnapshotId,
                 std::optional<long long>{expected.economicCalendarSnapshotId},
                 "snapshot_id");
    RequireEqual(extended.economicCalendarSnapshotHash,
                 std::optional<std::string>{expected.economicCalendarSnapshotHash},
                 "snapshot_hash");
    RequireEqual(extended.baseLearningRate, expected.baseLearningRate,
                 "base_learning_rate");
    RequireEqual(extended.batchSize, expected.batchSize, "batch_size");
    RequireEqual(extended.freshInitializationSeed,
                 expected.freshInitializationSeed, "seed");
    RequireEqual(extended.checkpointInferenceEnabled,
                 expected.checkpointInferenceEnabled,
                 "checkpoint_inference_enabled");
    RequireEqual(extended.checkpointInferenceMinimumEpoch,
                 expected.checkpointInferenceMinimumEpoch,
                 "checkpoint_inference_minimum_epoch");
    RequireEqual(extended.checkpointInferenceInterval,
                 expected.checkpointInferenceInterval,
                 "checkpoint_inference_interval");
    RequireEqual(extended.checkpointPolicyEnabled,
                 expected.checkpointPolicyEnabled,
                 "checkpoint_policy_enabled");
    RequireEqual(extended.checkpointPolicyMinimumLeaderScore,
                 expected.checkpointPolicyMinimumLeaderScore,
                 "checkpoint_policy_minimum_leader_score");
    RequireEqual(extended.checkpointPolicyMinimumInferenceAccuracy,
                 expected.checkpointPolicyMinimumInferenceAccuracy,
                 "checkpoint_policy_minimum_inference_accuracy");
    RequireEqual(extended.checkpointPolicyTopN,
                 expected.checkpointPolicyTopN, "checkpoint_policy_top_n");
    RequireEqual(extended.checkpointPolicyScope,
                 expected.checkpointPolicyScope, "checkpoint_policy_scope");
    RequireEqual(extended.checkpointPolicyStopMode,
                 expected.checkpointPolicyStopMode,
                 "checkpoint_policy_stop_mode");
    RequireEqual(extended.checkpointPolicyGraceEvaluations,
                 expected.checkpointPolicyGraceEvaluations,
                 "checkpoint_policy_grace_evaluations");
    RequireEqual(extended.checkpointPolicyRevision,
                 expected.checkpointPolicyRevision,
                 "checkpoint_policy_revision");
    RequireEqual(extended.checkpointPolicyHash,
                 expected.checkpointPolicyHash, "checkpoint_policy_hash");
    RequireEqual(extended.continuationPolicyEnabled,
                 expected.continuationPolicyEnabled,
                 "continuation_policy_enabled");
    RequireEqual(extended.continuationPolicyScientificIdentity,
                 expected.continuationPolicyScientificIdentity,
                 "continuation_policy_identity");
    RequireEqual(evidence.operational.schedulerPriority,
                 expected.schedulerPriority, "scheduler_priority");
}

Gate EvaluateGate(const Replication::ReplicationEvaluation& evaluation,
                  const Replication::MemberEvaluation& anchorPair)
{
    Gate gate;
    gate.correctedValidPairCount =
        evaluation.population.correctedValidPairCount;
    gate.historicalPreFixPairCount =
        evaluation.population.historicalPreFixPairCount;
    gate.invalidOrIncompatiblePairCount =
        evaluation.population.invalidPairCount +
        evaluation.population.missingEvidencePairCount;
    gate.correctedReplicationMinimumSatisfied =
        gate.correctedValidPairCount >=
        static_cast<std::size_t>(gate.minimumValidReplications);
    gate.anchorPairComplete =
        anchorPair.evidenceState == Replication::MemberEvidenceState::Complete;
    gate.anchorPairValidCorrected = gate.anchorPairComplete &&
        anchorPair.scientificallyValidComplete &&
        anchorPair.evidenceClassification ==
            FeatureAblationPairEvaluation::EvidenceClassification::
                CorrectedCausalSurprisePairEvidence;

    if (anchorPair.evidenceState ==
            Replication::MemberEvidenceState::Invalid ||
        anchorPair.evidenceState ==
            Replication::MemberEvidenceState::MissingEvidence ||
        anchorPair.evidenceState ==
            Replication::MemberEvidenceState::ProfitabilityUnavailable ||
        anchorPair.evidenceState ==
            Replication::MemberEvidenceState::HistoricalPreFix)
    {
        gate.nextAction = NextAction::InvalidPairRequiresReview;
        Add(gate.reasons, "anchor_pair_not_valid_corrected_evidence");
    }
    else if (!gate.anchorPairComplete)
    {
        gate.nextAction = NextAction::AwaitPairCompletion;
        Add(gate.reasons, "anchor_pair_incomplete");
    }
    else if (!gate.anchorPairValidCorrected ||
             evaluation.evidenceIntegrityFailure)
    {
        gate.nextAction = NextAction::InvalidPairRequiresReview;
        Add(gate.reasons, "corrected_replication_integrity_failure");
    }
    else if (gate.correctedReplicationMinimumSatisfied)
    {
        gate.nextAction = NextAction::ReplicationThresholdSatisfied;
        Add(gate.reasons, "minimum_valid_replications_satisfied");
    }
    else
    {
        gate.nextAction = NextAction::PrepareAdditionalReplications;
        Add(gate.reasons, "additional_corrected_replication_required");
        const std::size_t offset = gate.correctedValidPairCount - 1;
        if (offset < kFollowOnPairCount)
            gate.nextPlanPairOrdinal = offset + 1;
        else
        {
            gate.nextAction = NextAction::InvalidPairRequiresReview;
            gate.nextPlanPairOrdinal.reset();
            Add(gate.reasons, "predeclared_plan_exhausted_before_threshold");
        }
    }
    return gate;
}

std::string NextActionText(NextAction value)
{
    switch (value)
    {
        case NextAction::AwaitPairCompletion: return "await_pair_completion";
        case NextAction::InvalidPairRequiresReview:
            return "invalid_pair_requires_review";
        case NextAction::PrepareAdditionalReplications:
            return "prepare_additional_replications";
        case NextAction::ReplicationThresholdSatisfied:
            return "replication_threshold_satisfied";
    }
    throw std::invalid_argument("unknown_corrected_replication_next_action");
}

std::string RenderPlan(const Plan& plan)
{
    ValidatePlan(plan);
    std::ostringstream output;
    output << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_PLAN"
           << ",plan_semantic_version=" << plan.semanticVersion
           << ",scientific_policy_version=" << plan.scientificPolicyVersion
           << ",plan_hash=" << plan.hash
           << ",minimum_valid_replications="
           << Replication::kMinimumValidReplications
           << ",outcome_metrics_in_plan=false"
           << ",materialized=false\n";
    for (const Pair& pair : plan.pairs)
        output << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_PLAN_PAIR"
               << ",ordinal=" << pair.ordinal
               << ",symbol=" << pair.symbol
               << ",prediction_horizon=" << pair.predictionHorizon
               << ",independent_dimension=" << pair.independentDimension
               << ",replication_unit_hash=" << pair.replicationUnitHash
               << ",control_role=" << pair.control.role
               << ",control_mask=EMPTY"
               << ",control_identity_hash="
               << pair.control.scientificIdentityHash
               << ",treatment_role=" << pair.treatment.role
               << ",treatment_mask="
               << pair.treatment.featureAblationMask
               << ",treatment_identity_hash="
               << pair.treatment.scientificIdentityHash << '\n';
    output << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_PLAN_CANONICAL="
           << plan.canonical << '\n';
    return output.str();
}

std::string RenderGate(const Gate& gate)
{
    std::ostringstream output;
    output << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_STATUS"
           << ",corrected_valid_pair_count="
           << gate.correctedValidPairCount
           << ",historical_pre_fix_pair_count="
           << gate.historicalPreFixPairCount
           << ",invalid_or_incompatible_pair_count="
           << gate.invalidOrIncompatiblePairCount
           << ",minimum_valid_replications="
           << gate.minimumValidReplications
           << ",corrected_replication_minimum_satisfied="
           << Boolean(gate.correctedReplicationMinimumSatisfied)
           << ",anchor_pair_complete=" << Boolean(gate.anchorPairComplete)
           << ",anchor_pair_valid_corrected="
           << Boolean(gate.anchorPairValidCorrected)
           << ",next_action=" << NextActionText(gate.nextAction)
           << ",next_plan_pair_ordinal="
           << (gate.nextPlanPairOrdinal
                   ? std::to_string(*gate.nextPlanPairOrdinal) : "NULL")
           << ",reasons=";
    if (gate.reasons.empty()) output << "NONE";
    for (std::size_t index = 0; index < gate.reasons.size(); ++index)
    {
        if (index != 0) output << '|';
        output << gate.reasons[index];
    }
    output << '\n';
    return output.str();
}

} // namespace EA::CorrectedCausalSurpriseReplicationContinuation
