#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <signal.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

#include <pqxx/pqxx>

#include "ExperimentScheduler.hpp"
#include "GlobalExperimentControl.hpp"
#include "CanonicalSymbol.hpp"
#include "ExperimentCurrentOperation.hpp"
#include "CheckpointPolicy.hpp"
#include "ContinuationPolicy.hpp"
#include "ContinuationPolicyInheritance.hpp"
#include "ContinuationPolicyPersistence.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "ProfitabilityVerificationService.hpp"
#include "ExperimentRecommendationService.hpp"
#include "ExperimentRecommendationEvaluationService.hpp"
#include "ExperimentRecommendationRankingService.hpp"
#include "ExperimentRecommendationConversionProposalReviewService.hpp"
#include "ExperimentRecommendationConversionExecutionService.hpp"
#include "ExperimentRecommendationConversionActivationService.hpp"
#include "ExperimentRecommendationConversionWorkflowService.hpp"
#include "ExperimentRecommendationCampaignPlanningService.hpp"
#include "ExperimentRecommendationCampaignReviewService.hpp"
#include "ExperimentRecommendationCampaignApprovalService.hpp"
#include "ExperimentRecommendationCampaignMaterializationService.hpp"
#include "ExperimentRecommendationCampaignHandoffService.hpp"
#include "ExperimentRecommendationCampaignProposalReviewService.hpp"
#include "ExperimentRecommendationCampaignExecutionService.hpp"
#include "ExperimentRecommendationCampaignActivationService.hpp"
#include "ExperimentRecommendationCampaignLaunchService.hpp"
#include "ExperimentRecommendationCampaignStatusService.hpp"
#include "ExperimentRecommendationCampaignOutcomeAssessmentService.hpp"
#include "CampaignOperationsService.hpp"
#include "CampaignOperationsControlService.hpp"
#include "CampaignOperationsCompletionService.hpp"
#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsManagerService.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"
#include "PgModelIO.hpp"
#include "Params.hpp"
#include "RunMetadata.hpp"
#include "SchedulerChildStatus.hpp"
#include "SchedulerExecutablePath.hpp"
#include "SchedulerOwnershipPolicy.hpp"
#include "SchedulerOwnershipRepository.hpp"
#include "SchedulerCore/CheckpointAnalysisOrchestrationService.hpp"
#include "SchedulerCore/CheckpointEvaluationService.hpp"
#include "SchedulerCore/ContinuationOrchestrationService.hpp"
#include "SchedulerCore/ExperimentTransitionService.hpp"
#include "SchedulerCore/FinalExperimentDispatchService.hpp"
#include "SchedulerCore/InferenceWorkerSelection.hpp"
#include "SchedulerCore/PostgresSchedulerRepository.hpp"
#include "SchedulerCore/ReconciliationService.hpp"
#include "SchedulerCore/SchedulerAdmissionService.hpp"
#include "SchedulerCore/SchedulerAuthorityService.hpp"
#include "SchedulerCore/SchedulerChildCompletionService.hpp"
#include "SchedulerCore/SchedulerCycleService.hpp"
#include "SchedulerCore/SchedulerEngine.hpp"
#include "SchedulerCore/SchedulerPolicy.hpp"
#include "SchedulerCore/SchedulerRuntimeContext.hpp"
#include "SchedulerCore/SchedulerWorkerRegistration.hpp"
#include "SchedulerCore/WorkerControlService.hpp"
#include "SchedulerCore/WorkerAttemptLifecycleService.hpp"
#include "SchedulerCore/WorkerProcessController.hpp"
#include "SchedulerCore/SchedulerSemanticAdmission.hpp"
#include "SupportedSymbols.hpp"
#include "WorkerLifecycleDiagnostics.hpp"
#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "FeatureWarmupScope.hpp"
#include "FeatureAblation.hpp"
#include "TrainingObjective.hpp"
#include "PairedTrainingObjectiveEvaluationService.hpp"
#include "FeatureAblationPairEvaluationService.hpp"
#include "FeatureAblationPairEvaluationRepository.hpp"
#include "FeatureAblationReplicationEvaluationService.hpp"
#include "CorrectedCausalSurpriseReplicationContinuationService.hpp"
#include "CausalSurpriseObservabilityService.hpp"
#include "EconomicEventRepository.hpp"

#include "SchedulerCore/ProductionSchedulerDaemon.hpp"
#include "SchedulerCore/ProductionSchedulerRuntimeInternal.hpp"

namespace EA::SchedulerCore::ProductionRuntimeDetail
{

using namespace EA::ExperimentScheduler;
using EA::SchedulerCore::SchedulerAuthorityLost;
using EA::SchedulerCore::kSchedulerLeaseSeconds;
using EA::SchedulerCore::CheckpointPolicyConfigurationError;
using EA::SchedulerCore::CheckpointPolicyDecisionContext;
using EA::SchedulerCore::CheckpointPolicyEvaluationResult;
using EA::SchedulerCore::CheckpointPolicyPopulation;
using EA::SchedulerCore::MakeCheckpointPolicyEvidenceIdentity;
using EA::SchedulerCore::PersistedCheckpointPolicyDecision;
using EA::SchedulerCore::PlanCheckpointPolicyDecision;
using EA::SchedulerCore::ValidatedCheckpointPolicyEvidence;

using EA::SchedulerCore::ReservedWorkerAttempt;

using EA::SchedulerCore::SchedulerChildCompletionEvidence;

using EA::SchedulerCore::SchedulerOwnedChild;

volatile sig_atomic_t gSchedulerStopRequested = 0;

EA::SchedulerCore::SchedulerRuntimeContext& SchedulerRuntime(
    const SchedulerOptions& options)
{
    if (options.runtimeContext == nullptr)
        throw std::logic_error("scheduler_runtime_context_missing");
    return *options.runtimeContext;
}

constexpr int kSchedulerLaunchRecoveryGraceSeconds = 10;

struct PhaseSchedulingStats
{
    std::string phase;
    int examined = 0;
    int skipped = 0;
    int launched = 0;
    int freeSlots = 0;
};

struct SchedulerEventLogState
{
    std::optional<std::string> previousQueueKey;
    std::map<std::string, std::string> previousPhaseKeys;
    std::set<std::string> previousSkipKeys;
    std::set<std::string> currentSkipKeys;
    std::set<std::string> previousWorkerSelectionKeys;
    std::set<std::string> currentWorkerSelectionKeys;
    std::set<std::string> previousRunningPresentKeys;
    std::set<std::string> currentRunningPresentKeys;
};

std::string GetEnvOrDefault(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return (value && *value) ? std::string{value} : std::string{fallback};
}

std::string LstmDbConnectionString()
{
    return "hostaddr=" + GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1") +
           " gssencmode=disable user=pqxx dbname=" + GetEnvOrDefault("LSTM_DB_NAME", "LSTM");
}

std::string SqlNullable(pqxx::work& w, const std::optional<std::string>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

std::string SqlNullable(pqxx::work& w, const std::optional<double>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

std::string SqlNullable(pqxx::work& w, const std::optional<int>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

std::string SqlNullable(pqxx::work& w, const std::optional<long long>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

std::string SqlContinuationTargetSequence(
    const std::optional<std::vector<int>>& sequence)
{
    if (!sequence.has_value())
        return "NULL";
    std::ostringstream out;
    out << "ARRAY[";
    for (size_t index = 0; index < sequence->size(); ++index)
    {
        if (index != 0)
            out << ',';
        out << (*sequence)[index];
    }
    out << "]::integer[]";
    return out.str();
}

std::string FormatDouble(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

void ValidateCheckpointPolicyConfig(const SchedulerOptions& options)
{
    CheckpointPolicyConfig config;
    config.minLeaderScore = options.checkpointPolicyMinLeaderScore;
    config.minInferAccuracy = options.checkpointPolicyMinInferAccuracy;
    config.topN = options.checkpointPolicyTopN;
    config.scope = options.checkpointPolicyScope;
    config.stopMode = options.checkpointPolicyStopMode;
    config.graceEvals = options.checkpointPolicyGraceEvals;
    const std::optional<std::string> error = CheckpointPolicyConfigurationError(
        config, options.queueCheckpointPolicy);
    if (error.has_value())
        throw std::invalid_argument("invalid checkpoint policy configuration: " + *error);
}

bool TableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !r.empty();
}

bool ColumnExists(pqxx::work& w, const std::string& tableName, const std::string& columnName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2 LIMIT 1;",
        tableName,
        columnName);
    return !r.empty();
}

bool ModelExists(pqxx::work& w, long long modelId)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM model WHERE model_id = $1 LIMIT 1;",
        modelId);
    return !r.empty();
}

std::string LoadModelFeatureAblationMask(pqxx::work& w, long long modelId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT m.experiment_id,e.feature_ablation_mask FROM model m "
        "LEFT JOIN experiment e ON e.experiment_id=m.experiment_id "
        "WHERE m.model_id=$1;",
        modelId);
    if (rows.empty())
        throw std::runtime_error("resume model_id not found");
    // A model without experiment provenance predates formal ablation and is
    // intentionally interpreted as the historical no-ablation contract.
    if (rows[0][0].is_null()) return {};
    if (rows[0][1].is_null())
        throw std::runtime_error("resume_model_feature_ablation_lineage_missing");
    return EA::FeatureAblationMask::Parse(rows[0][1].as<std::string>()).CanonicalText();
}

std::string DateOnly(const std::string& value)
{
    return value.size() >= 10 ? value.substr(0, 10) : value;
}

bool SameDate(const std::string& lhs, const std::string& rhs)
{
    return DateOnly(lhs) == DateOnly(rhs);
}

QueueResumeMeta LoadQueueResumeMeta(pqxx::work& w, long long modelId)
{
    if (!ModelExists(w, modelId))
        throw std::runtime_error("resume model_id not found");

    auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
    auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
    if (dims.n_rows != 1 ||
        dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount ||
        vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount))
        throw std::runtime_error("resume requires complete train_config_meta with 14 fields");

    QueueResumeMeta meta;
    meta.modelId = modelId;
    meta.featureWarmupScope = DBIO::PgModelIO::loadFeatureWarmupScopeMeta(w, modelId);
    meta.donchian20Mode = DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
    meta.donchianLookback = DBIO::PgModelIO::loadDonchianLookbackMeta(w, modelId);
    meta.featureAblationMask = LoadModelFeatureAblationMask(w, modelId);
    meta.modelInputWidth =
        DBIO::PgModelIO::loadRequiredModelMeta(w, modelId).inputWidth;
    meta.trainingObjective =
        DBIO::PgModelIO::loadTrainingObjectiveMeta(w, modelId);
    meta.symbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    meta.predictionHorizon = static_cast<int>(std::llround(vals[1]));
    meta.threshold = vals[2];
    meta.completedEpochs = static_cast<int>(std::llround(vals[10]));
    meta.coreLrMult = vals[11];
    meta.headLrMult = vals[12];
    const auto range = DBIO::PgModelIO::decodeTrainRangeMeta(w, modelId);
    meta.trainStart = DateOnly(range.first);
    meta.trainEnd = DateOnly(range.second);
    return meta;
}

void PrintQueueResumeMeta(const char* marker,
                                 const QueueResumeMeta& meta,
                                 int targetEpochs)
{
    std::cout << marker
              << ",resume_model_id=" << meta.modelId
              << ",symbol=" << meta.symbol
              << ",prediction_horizon=" << meta.predictionHorizon
              << ",threshold=" << FormatDouble(meta.threshold)
              << ",donchian20_mode=" << Donchian20ModeText(meta.donchian20Mode)
              << ",feature_warmup_scope=" << EA::FeatureWarmupScopeText(meta.featureWarmupScope)
              << ",donchian_lookback=" << meta.donchianLookback
              << ",feature_ablation_mask=" << meta.featureAblationMask
              << ",model_input_width=" << meta.modelInputWidth
              << ",training_objective_hash="
              << EA::TrainingObjective::Identity(meta.trainingObjective)
              << ",completed_epochs=" << meta.completedEpochs
              << ",target_epochs=" << targetEpochs
              << ",train_start=" << meta.trainStart
              << ",train_end=" << meta.trainEnd
              << std::endl;
}

void ThrowQueueResumeInvalid(const std::string& reason,
                                    long long modelId,
                                    const std::string& detail)
{
    std::cout << "QUEUE_RESUME_INVALID"
              << ",resume_model_id=" << modelId
              << ",reason=" << reason;
    if (!detail.empty())
        std::cout << ",detail=" << detail;
    std::cout << std::endl;
    throw std::invalid_argument("QUEUE_RESUME_INVALID:" + reason);
}

std::optional<std::string> QueueResumeConfigurationFailure(
    const QueueResumeMeta& meta,
    const QueueResumeCompatibilityRequirements& requirements);

std::optional<std::string> QueueResumeCompatibilityFailure(
    const QueueResumeMeta& meta,
    const QueueResumeCompatibilityRequirements& requirements)
{
    if (requirements.targetEpochs <= meta.completedEpochs)
        return "target_epochs_not_greater_than_completed_epoch";
    return QueueResumeConfigurationFailure(meta, requirements);
}

std::optional<std::string> QueueResumeConfigurationFailure(
    const QueueResumeMeta& meta,
    const QueueResumeCompatibilityRequirements& requirements)
{
    if (requirements.symbol.has_value() &&
        EA::CanonicalSymbol::Normalize(*requirements.symbol) != meta.symbol)
        return "symbol_mismatch";
    if (requirements.predictionHorizon.has_value() &&
        *requirements.predictionHorizon != meta.predictionHorizon)
        return "prediction_horizon_mismatch";
    if (requirements.threshold.has_value() &&
        std::fabs(*requirements.threshold - meta.threshold) > 1e-7)
        return "threshold_mismatch";
    if (requirements.trainStart.has_value() &&
        !SameDate(*requirements.trainStart, meta.trainStart))
        return "train_start_mismatch";
    if (requirements.trainEnd.has_value() &&
        !SameDate(*requirements.trainEnd, meta.trainEnd))
        return "train_end_mismatch";
    if (requirements.coreLrMult.has_value() &&
        (!meta.coreLrMult.has_value() ||
         std::fabs(*requirements.coreLrMult - *meta.coreLrMult) > 1e-7))
        return "core_lr_mismatch";
    if (requirements.headLrMult.has_value() &&
        (!meta.headLrMult.has_value() ||
         std::fabs(*requirements.headLrMult - *meta.headLrMult) > 1e-7))
        return "head_lr_mismatch";
    if (requirements.donchian20Mode.has_value() &&
        *requirements.donchian20Mode != meta.donchian20Mode)
        return "donchian20_mode_mismatch";
    if (requirements.featureWarmupScope.has_value() &&
        *requirements.featureWarmupScope != meta.featureWarmupScope)
        return "feature_warmup_scope_mismatch";
    if (requirements.donchianLookback.has_value() &&
        *requirements.donchianLookback != meta.donchianLookback)
        return "donchian_lookback_mismatch";
    if (requirements.featureAblationMask.has_value() &&
        *requirements.featureAblationMask != meta.featureAblationMask)
        return "feature_ablation_mask_mismatch";
    if (!EA::TrainingObjective::ResumeCompatible(
            meta.trainingObjective, requirements.trainingObjective))
        return "training_objective_mismatch";
    return std::nullopt;
}

void MergeResumeMetaIntoQueueOptions(SchedulerOptions& options,
                                            const QueueResumeMeta& meta)
{
    if (!options.targetEpochs.has_value())
        ThrowQueueResumeInvalid("missing_target_epochs", meta.modelId);
    const QueueResumeCompatibilityRequirements requirements{
        *options.targetEpochs,
        options.symbol,
        options.predictionHorizon,
        options.cNextThreshold,
        options.trainStart,
        options.trainEnd,
        std::nullopt,
        std::nullopt,
        options.donchian20Mode,
        options.featureWarmupScopeSpecified
            ? std::optional<EA::FeatureWarmupScope>{options.featureWarmupScope}
            : std::nullopt,
        options.donchianLookbackSpecified
            ? std::optional<std::size_t>{options.donchianLookback}
            : std::nullopt,
        options.resumeExpandInputWidth
            ? std::nullopt
            : std::optional<std::string>{options.featureAblationMask},
        options.trainingObjectiveSpecified
            ? options.trainingObjective
            : meta.trainingObjective
    };
    if (const std::optional<std::string> failure =
            QueueResumeCompatibilityFailure(meta, requirements);
        failure.has_value())
    {
        ThrowQueueResumeInvalid(*failure, meta.modelId);
    }

    options.symbol = meta.symbol;
    options.predictionHorizon = meta.predictionHorizon;
    options.cNextThreshold = meta.threshold;
    options.trainStart = meta.trainStart;
    options.trainEnd = meta.trainEnd;
    options.coreLrMult = meta.coreLrMult;
    options.headLrMult = meta.headLrMult;
    options.donchian20Mode = meta.donchian20Mode;
    options.featureWarmupScope = meta.featureWarmupScope;
    options.donchianLookback = meta.donchianLookback;
    options.trainingObjective = meta.trainingObjective;

    if (options.resumeExpandInputWidth)
    {
        const EA::InputWidthExpansionPlan plan =
            EA::BuildInputWidthExpansionPlan(meta.modelInputWidth);
        if (!options.featureAblationMaskSpecified)
        {
            options.featureAblationMask = meta.featureAblationMask;
        }
        else
        {
            const EA::FeatureAblationMask sourceMask =
                EA::FeatureAblationMask::Parse(meta.featureAblationMask);
            const EA::FeatureAblationMask requestedMask =
                EA::FeatureAblationMask::Parse(options.featureAblationMask);
            const auto contains = [](const std::vector<std::size_t>& values,
                                     std::size_t value)
            {
                return std::find(values.begin(), values.end(), value) !=
                    values.end();
            };
            for (const std::size_t sourceColumn : sourceMask.tensorColumns())
            {
                if (!contains(requestedMask.tensorColumns(), sourceColumn))
                    ThrowQueueResumeInvalid(
                        "expansion_ablation_removes_source_ablation",
                        meta.modelId);
            }
            for (const std::size_t requestedColumn :
                 requestedMask.tensorColumns())
            {
                if (!contains(sourceMask.tensorColumns(), requestedColumn) &&
                    requestedColumn < plan.sourceTensorFeatureCount)
                {
                    ThrowQueueResumeInvalid(
                        "expansion_ablation_changes_historical_feature",
                        meta.modelId);
                }
            }
        }
    }

    PrintQueueResumeMeta("QUEUE_RESUME_MODEL", meta, *options.targetEpochs);
}

std::optional<QueueResumeMeta> TryLoadRecoverableModelMeta(pqxx::work& w, long long modelId)
{
    try
    {
        return LoadQueueResumeMeta(w, modelId);
    }
    catch (const std::exception& e)
    {
        std::cout << "SCHEDULER_ORPHAN_MODEL_UNUSABLE"
                  << ",model_id=" << modelId
                  << ",reason=" << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

QueueResumeCompatibilityRequirements TrainingCheckpointRequirements(
    const ExperimentRow& experiment)
{
    return QueueResumeCompatibilityRequirements{
        experiment.targetEpochs,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.trainStart,
        experiment.trainEnd,
        experiment.coreLrMult.value_or(default_core_lr_mult),
        experiment.headLrMult.value_or(default_head_weight_lr_mult),
        experiment.donchian20Mode,
        experiment.featureWarmupScope,
        experiment.donchianLookback,
        experiment.featureAblationMask,
        experiment.trainingObjective
    };
}

void PrintTrainingCheckpointCandidateRejected(
    long long experimentId,
    long long modelId,
    const std::optional<int>& completedEpoch,
    const std::string& reason)
{
    std::cout << "SCHEDULER_CHECKPOINT_CANDIDATE_REJECTED"
              << ",experiment_id=" << experimentId
              << ",model_id=" << modelId
              << ",completed_epoch="
              << (completedEpoch.has_value()
                      ? std::to_string(*completedEpoch)
                      : "NULL")
              << ",reason=" << reason
              << std::endl;
}

TrainingCheckpointSelection SelectUsableTrainingCheckpoint(
    pqxx::work& w,
    const ExperimentRow& experiment,
    const std::optional<double>& createdAtOrAfter,
    bool allowFinalModel)
{
    std::string sql =
        "SELECT m.model_id,cfg.value,"
        "COALESCE(m.comment,'') ILIKE '%periodic training checkpoint%',"
        "e.model_input_width "
        "FROM model m JOIN experiment e "
        "ON e.experiment_id=m.experiment_id "
        "LEFT JOIN matrix cfg ON cfg.model_id=m.model_id "
        "AND cfg.param_name='train_config_meta' "
        "AND cfg.row_idx=0 AND cfg.col_idx=10 "
        "WHERE m.experiment_id=$1 ";
    pqxx::result rows;
    if (createdAtOrAfter.has_value())
    {
        sql += "AND m.created_at>=to_timestamp($2) ";
        sql +=
            "ORDER BY cfg.value DESC NULLS LAST,"
            "(COALESCE(m.comment,'') ILIKE "
            "'%periodic training checkpoint%') ASC,"
            "m.model_id DESC;";
        rows = w.exec_params(
            sql, experiment.experimentId, *createdAtOrAfter);
    }
    else
    {
        sql +=
            "ORDER BY cfg.value DESC NULLS LAST,"
            "(COALESCE(m.comment,'') ILIKE "
            "'%periodic training checkpoint%') ASC,"
            "m.model_id DESC;";
        rows = w.exec_params(sql, experiment.experimentId);
    }

    std::vector<TrainingCheckpointCandidate> candidates;
    candidates.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        TrainingCheckpointCandidate candidate;
        candidate.modelId = row[0].as<long long>();
        candidate.periodicCheckpoint = row[2].as<bool>();
        if (!row[3].is_null())
            candidate.experimentInputWidth = row[3].as<std::size_t>();
        if (!row[1].is_null())
        {
            const double persistedEpoch = row[1].as<double>();
            const double roundedEpoch = std::round(persistedEpoch);
            if (std::isfinite(persistedEpoch) &&
                std::fabs(persistedEpoch - roundedEpoch) <= 1e-7 &&
                roundedEpoch >= 0.0 &&
                roundedEpoch <=
                    static_cast<double>(std::numeric_limits<int>::max()))
            {
                candidate.completedEpoch =
                    static_cast<int>(roundedEpoch);
            }
        }
        candidates.push_back(candidate);
    }

    const QueueResumeCompatibilityRequirements requirements =
        TrainingCheckpointRequirements(experiment);
    for (size_t index = 0; index < candidates.size();)
    {
        const TrainingCheckpointCandidate& candidate = candidates[index];
        if (!candidate.completedEpoch.has_value())
        {
            PrintTrainingCheckpointCandidateRejected(
                experiment.experimentId,
                candidate.modelId,
                std::nullopt,
                "missing_or_invalid_completed_epoch");
            ++index;
            continue;
        }

        size_t groupEnd = index + 1;
        while (groupEnd < candidates.size() &&
               candidates[groupEnd].completedEpoch ==
                   candidate.completedEpoch)
        {
            ++groupEnd;
        }
        bool resolvedTerminalGroup = false;
        if (groupEnd - index > 1)
        {
            const bool terminalEpoch =
                allowFinalModel &&
                *candidate.completedEpoch >= experiment.targetEpochs;

            if (terminalEpoch)
            {
                size_t nonPeriodicCount = 0;
                for (size_t candidateIndex = index;
                     candidateIndex < groupEnd;
                     ++candidateIndex)
                {
                    if (!candidates[candidateIndex].periodicCheckpoint)
                        ++nonPeriodicCount;
                }

                if (nonPeriodicCount == 1 &&
                    !candidate.periodicCheckpoint)
                {
                    resolvedTerminalGroup = true;
                    std::cout
                        << "SCHEDULER_CHECKPOINT_FINAL_CANDIDATE_RESOLVED"
                        << ",experiment_id=" << experiment.experimentId
                        << ",completed_epoch=" << *candidate.completedEpoch
                        << ",candidate_count=" << (groupEnd - index)
                        << ",model_id=" << candidate.modelId
                        << ",reason=unique_non_periodic_target_model"
                        << std::endl;
                }
                else
                {
                    std::cout
                        << "SCHEDULER_CHECKPOINT_CANDIDATE_AMBIGUOUS"
                        << ",experiment_id=" << experiment.experimentId
                        << ",completed_epoch=" << *candidate.completedEpoch
                        << ",candidate_count=" << (groupEnd - index)
                        << ",non_periodic_count=" << nonPeriodicCount
                        << ",result=terminal_epoch_not_regressed"
                        << std::endl;
                    std::cout
                        << "SCHEDULER_CHECKPOINT_SELECTION_FAILED"
                        << ",experiment_id=" << experiment.experimentId
                        << ",candidate_count=" << candidates.size()
                        << ",reason=terminal_model_ambiguous"
                        << std::endl;
                    return TrainingCheckpointSelection{
                        std::nullopt, "terminal_model_ambiguous"};
                }
            }
            else
            {
                std::cout << "SCHEDULER_CHECKPOINT_CANDIDATE_TIE_BREAK"
                          << ",experiment_id=" << experiment.experimentId
                          << ",completed_epoch=" << *candidate.completedEpoch
                          << ",candidate_count=" << (groupEnd - index)
                          << ",candidate_order=periodic_flag_asc_model_id_desc"
                          << ",selected_model_id=" << candidate.modelId
                          << std::endl;
            }
        }

        if (*candidate.completedEpoch > experiment.targetEpochs)
        {
            PrintTrainingCheckpointCandidateRejected(
                experiment.experimentId,
                candidate.modelId,
                candidate.completedEpoch,
                "completed_epoch_exceeds_target");
            ++index;
            continue;
        }
        if (!allowFinalModel &&
            *candidate.completedEpoch >= experiment.targetEpochs)
        {
            PrintTrainingCheckpointCandidateRejected(
                experiment.experimentId,
                candidate.modelId,
                candidate.completedEpoch,
                "no_remaining_training");
            ++index;
            continue;
        }
        if (*candidate.completedEpoch < experiment.targetEpochs &&
            !candidate.periodicCheckpoint)
        {
            PrintTrainingCheckpointCandidateRejected(
                experiment.experimentId,
                candidate.modelId,
                candidate.completedEpoch,
                "intermediate_model_not_checkpoint");
            ++index;
            continue;
        }

        try
        {
            DBIO::PgModelIO::validateTrainingResumeState(
                w, candidate.modelId);
            (void)DBIO::PgModelIO::validateModelInputSemanticsForLoad(
                w, candidate.modelId);
            const QueueResumeMeta meta =
                LoadQueueResumeMeta(w, candidate.modelId);
            if (meta.completedEpochs != *candidate.completedEpoch)
                throw std::runtime_error("completed_epoch_mismatch");
            if (const auto failure =
                    QueueResumeConfigurationFailure(meta, requirements);
                failure.has_value())
            {
                throw std::runtime_error(*failure);
            }
            if (candidate.experimentInputWidth.has_value() &&
                meta.modelInputWidth != *candidate.experimentInputWidth)
            {
                throw std::runtime_error("model_input_width_mismatch");
            }

            std::cout << "SCHEDULER_CHECKPOINT_SELECTED"
                      << ",experiment_id=" << experiment.experimentId
                      << ",model_id=" << meta.modelId
                      << ",completed_epoch=" << meta.completedEpochs
                      << ",result="
                      << (meta.completedEpochs >= experiment.targetEpochs
                              ? "final_model"
                              : "resume_checkpoint")
                      << std::endl;
            return TrainingCheckpointSelection{meta, "selected"};
        }
        catch (const std::exception& error)
        {
            PrintTrainingCheckpointCandidateRejected(
                experiment.experimentId,
                candidate.modelId,
                candidate.completedEpoch,
                error.what());
            // Candidate qualification is side-effect free. For a uniquely
            // resolved but invalid final artifact, skip the remaining target-
            // epoch periodic sibling and continue at the next lower epoch.
            if (resolvedTerminalGroup)
                index = groupEnd;
            else
                ++index;
            continue;
        }
    }

    std::cout << "SCHEDULER_CHECKPOINT_SELECTION_FAILED"
              << ",experiment_id=" << experiment.experimentId
              << ",candidate_count=" << candidates.size()
              << ",reason=no_valid_checkpoint"
              << std::endl;
    return TrainingCheckpointSelection{
        std::nullopt, "no_valid_checkpoint"};
}

std::optional<long long> FindLatestModelForExperiment(pqxx::work& w, long long experimentId)
{
    if (!ColumnExists(w, "model", "experiment_id"))
        return std::nullopt;

    pqxx::result rows = w.exec_params(
        "WITH cfg AS ("
        "  SELECT model_id, max(value) FILTER (WHERE col_idx = 10) AS completed_epochs "
        "  FROM matrix "
        "  WHERE param_name = 'train_config_meta' AND row_idx = 0 "
        "  GROUP BY model_id"
        ") "
        "SELECT m.model_id "
        "FROM model m "
        "LEFT JOIN cfg ON cfg.model_id = m.model_id "
        "WHERE m.experiment_id = $1 "
        "ORDER BY cfg.completed_epochs DESC NULLS LAST, m.model_id DESC "
        "LIMIT 1;",
        experimentId);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

std::optional<long long> FindLatestModelForExperimentSince(
    pqxx::work& w,
    long long experimentId,
    double attemptStartedEpoch)
{
    if (!ColumnExists(w, "model", "experiment_id"))
        return std::nullopt;

    pqxx::result rows = w.exec_params(
        "WITH cfg AS ("
        "  SELECT model_id, max(value) FILTER (WHERE col_idx = 10) AS completed_epochs "
        "  FROM matrix "
        "  WHERE param_name = 'train_config_meta' AND row_idx = 0 "
        "  GROUP BY model_id"
        ") "
        "SELECT m.model_id "
        "FROM model m "
        "LEFT JOIN cfg ON cfg.model_id = m.model_id "
        "WHERE m.experiment_id = $1 "
        "AND m.created_at >= to_timestamp($2) "
        "ORDER BY cfg.completed_epochs DESC NULLS LAST, m.model_id DESC "
        "LIMIT 1;",
        experimentId,
        attemptStartedEpoch);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

bool RequireSchedulerTables(pqxx::work& w)
{
    std::vector<std::string> missing;
    if (!TableExists(w, "experiment"))
        missing.push_back("experiment");
    if (!TableExists(w, "experiment_analysis_result"))
        missing.push_back("experiment_analysis_result");
    if (TableExists(w, "experiment") &&
        !ColumnExists(w, "experiment", "worker_started_at"))
        missing.push_back("experiment.worker_started_at");
    if (!TableExists(w, "experiment_scheduler_invocation"))
        missing.push_back("experiment_scheduler_invocation");
    if (!TableExists(w, "experiment_scheduler_lease"))
        missing.push_back("experiment_scheduler_lease");
    if (!TableExists(w, "experiment_scheduler_worker_attempt"))
        missing.push_back("experiment_scheduler_worker_attempt");
    if (!TableExists(w, "experiment_scheduler_protocol"))
        missing.push_back("experiment_scheduler_protocol");
    if (TableExists(w, "experiment") &&
        !ColumnExists(
            w, "experiment", "active_scheduler_worker_attempt_id"))
    {
        missing.push_back(
            "experiment.active_scheduler_worker_attempt_id");
    }
    if (TableExists(w, "experiment") &&
        !ColumnExists(
            w,
            "experiment",
            "operator_forced_final_inference_rerun_requested"))
    {
        missing.push_back(
            "experiment.operator_forced_final_inference_rerun_requested");
    }
    if (TableExists(w, "experiment") &&
        !ColumnExists(w, "experiment", "scheduler_priority"))
        missing.push_back("experiment.scheduler_priority");
    if (TableExists(w, "experiment") &&
        !ColumnExists(w, "experiment", "resume_requested"))
        missing.push_back("experiment.resume_requested");
    if (TableExists(w, "experiment") &&
        !ColumnExists(w, "experiment", "scheduler_resume_origin"))
        missing.push_back("experiment.scheduler_resume_origin");
    if (TableExists(w, "experiment_checkpoint_eval") &&
        !ColumnExists(
            w,
            "experiment_checkpoint_eval",
            "active_scheduler_worker_attempt_id"))
    {
        missing.push_back(
            "experiment_checkpoint_eval.active_scheduler_worker_attempt_id");
    }

    if (missing.empty())
        return true;

    std::ostringstream oss;
    for (size_t i = 0; i < missing.size(); ++i)
    {
        if (i)
            oss << "|";
        oss << missing[i];
    }
    std::cerr << "DATABASE_MIGRATION_REQUIRED"
              << ",missing=" << oss.str()
              << ",command=./migrate_lstm_db.sh"
              << std::endl;
    return false;
}

std::optional<EA::GlobalExperimentControl::ControlSnapshot>
LoadLockedGlobalControl(pqxx::work& w)
{
    if (!TableExists(w, "experiment_global_control") ||
        !TableExists(w, "experiment_admin_request"))
    {
        std::cerr
            << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,"
            << "missing=global_experiment_control"
            << std::endl;
        return std::nullopt;
    }
    EA::GlobalExperimentControl::AcquireCoordinationLock(w);
    return EA::GlobalExperimentControl::LoadControlSnapshot(w);
}

bool SchedulerLaunchAllowed(
    pqxx::work& w,
    const std::string& phase,
    bool cancellationInference = false,
    bool cancellationCheckpointTrain = false)
{
    const auto snapshot = LoadLockedGlobalControl(w);
    if (!snapshot)
        return false;
    const bool allowed =
        EA::GlobalExperimentControl::NormalSchedulingAllowed(*snapshot) ||
        (cancellationInference &&
         EA::GlobalExperimentControl::CancellationInferenceAllowed(*snapshot)) ||
        (cancellationCheckpointTrain &&
         EA::GlobalExperimentControl::CancellationCheckpointTrainAllowed(
             *snapshot));
    if (!allowed)
    {
        std::cout << "SCHEDULER_GLOBAL_CONTROL_BLOCK"
                  << ",phase=" << phase
                  << ",desired_state=" << snapshot->desiredState
                  << ",active_request_id="
                  << (snapshot->activeRequestId
                          ? std::to_string(*snapshot->activeRequestId)
                          : "NULL")
                  << std::endl;
    }
    return allowed;
}

void SetTransactionReadWrite(pqxx::work& w)
{
    w.exec("SET TRANSACTION READ WRITE;");
    EA::SchedulerOwnership::SetCorrectedSchedulerProtocolSession(w);
}

void SetTransactionReadOnly(pqxx::work& w)
{
    w.exec("SET TRANSACTION READ ONLY;");
}

std::string CanonicalizeObservedExecutable(
    const std::string& executable)
{
    if (executable.empty() || executable.front() != '/')
        return {};

    errno = 0;
    char* resolved = ::realpath(executable.c_str(), nullptr);
    if (resolved != nullptr)
    {
        std::string canonical{resolved};
        std::free(resolved);
        return canonical;
    }

    // A live process may outlive the executable's directory entry.
    // Process observation has already obtained an absolute executable path
    // from the live process. Preserve that identity only when the pathname
    // disappeared from the filesystem.
    if (errno == ENOENT)
        return executable;

    return {};
}

SchedulerOwnerProcessEvidence InspectSchedulerOwnerProcess(
    int pid,
    int processGroupId,
    const std::string& processStartIdentity,
    const std::string& canonicalExecutable,
    EA::GlobalExperimentControl::ProcessObservation* observed)
{
    std::unique_ptr<EA::GlobalExperimentControl::ProcessOperations>
        processes =
            EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const EA::GlobalExperimentControl::ProcessObservation observation =
        processes->Observe(pid);
    if (observed != nullptr)
        *observed = observation;
    return EA::SchedulerCore::EvaluateSchedulerOwnerProcess(
        pid,
        processGroupId,
        processStartIdentity,
        canonicalExecutable,
        {observation.exists,
         observation.inspectionSucceeded,
         observation.pid,
         observation.processGroupId,
         observation.processStartIdentity,
         observation.executable,
         observation.commandLine});
}

SchedulerProcessAbsenceEvidence
InspectAllSchedulerDispatchProcesses()
{
    SchedulerProcessAbsenceEvidence evidence;
    FILE* pipe = ::popen("ps -axo pid=,command=", "r");
    if (pipe == nullptr)
        return evidence;
    char buffer[32768] = {};
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        std::istringstream row{buffer};
        int pid = -1;
        if (!(row >> pid))
            continue;
        std::string command;
        std::getline(row, command);
        const size_t first = command.find_first_not_of(" \t");
        if (first != std::string::npos)
            command.erase(0, first);
        if (command.find("--schedule-experiments") !=
            std::string::npos)
        {
            evidence.schedulers.emplace_back(
                pid, std::move(command));
        }
    }
    evidence.inspectionSucceeded = ::pclose(pipe) == 0;
    return evidence;
}

bool AcquireSchedulerAuthority(SchedulerOptions& options)
{
    const int pid = static_cast<int>(::getpid());
    const int processGroupId = static_cast<int>(::getpgrp());
    const std::optional<std::string> processStartIdentity =
        EA::GlobalExperimentControl::ReadProcessStartIdentity(pid);
    if (!processStartIdentity)
        throw std::runtime_error(
            "scheduler_process_start_identity_unavailable");

    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    if (!RequireSchedulerTables(transaction))
        return false;
    SchedulerServiceComposition services{transaction};
    const auto acquisition = services.authority.acquire({
        pid,
        processGroupId,
        *processStartIdentity,
        options.schedulerExecutablePath,
        options.invocationCommandLine});
    options.schedulerAuthority = acquisition.authority;
    if (!acquisition.protocolAccepted)
    {
        transaction.commit();
        std::cout << "SCHEDULER_PROTOCOL_BARRIER_REJECTED"
                  << ",required_generation="
                  << EA::SchedulerCore::kSchedulerProtocolGeneration
                  << ",database_generation="
                  << (acquisition.databaseCutoverState != "missing"
                          ? std::to_string(
                                acquisition.databaseProtocolGeneration)
                          : "NULL")
                  << ",cutover_state="
                  << acquisition.databaseCutoverState
                  << ",reason="
                  << acquisition.protocolFailureReason
                  << ",mutations=0"
                  << std::endl;
        return false;
    }
    if (!acquisition.acquired())
    {
        transaction.commit();
        std::cout << "SCHEDULER_OWNERSHIP_REJECTED"
                  << ",scheduler_invocation_id="
                  << options.schedulerAuthority.schedulerInvocationId
                  << ",lease_owner="
                  << acquisition.previousLeaseOwner.value_or("NULL")
                  << ",fencing_token="
                  << acquisition.previousFencingToken
                  << ",reason="
                  << EA::SchedulerCore::SchedulerTakeoverDecisionText(
                         acquisition.decision)
                  << ",mutations=0"
                  << std::endl;
        return false;
    }
    transaction.commit();
    std::cout << "SCHEDULER_OWNERSHIP_ACQUIRED"
              << ",scheduler_invocation_id="
              << options.schedulerAuthority.schedulerInvocationId
              << ",fencing_token="
              << options.schedulerAuthority.fencingToken
              << ",lease_seconds=" << kSchedulerLeaseSeconds
              << ",reason="
              << EA::SchedulerCore::SchedulerTakeoverDecisionText(
                     acquisition.decision)
              << ",canonical_executable_path=" << options.schedulerExecutablePath
              << std::endl;
    return true;
}

void RequireAndRefreshSchedulerAuthority(
    pqxx::work& transaction,
    const SchedulerOptions& options)
{
    SchedulerServiceComposition services{transaction};
    services.authority.requireAndRenew(options.schedulerAuthority);
}

bool SchedulerAuthorityTestFailpointEnabled(
    const std::string& boundary)
{
    const char* enabled =
        std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_ENABLE");
    const char* selected =
        std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY");
    const std::string database =
        GetEnvOrDefault("LSTM_DB_NAME", "LSTM");
    return enabled && std::string{enabled} == "1" &&
           selected && std::string{selected} == boundary &&
           (database.rfind("ea_scheduler_", 0) == 0 ||
            database.rfind("ea_global_control_test_", 0) == 0);
}

void InjectSchedulerAuthorityLossForTest(
    pqxx::work& transaction,
    const SchedulerOptions& options,
    const std::string& boundary)
{
    if (!SchedulerAuthorityTestFailpointEnabled(boundary))
        return;
    const char* foreign =
        std::getenv(
            "EA_SCHEDULER_OWNERSHIP_TEST_FOREIGN_INVOCATION_ID");
    if (!foreign || !*foreign)
        throw std::runtime_error(
            "scheduler_authority_test_foreign_invocation_missing");
    SchedulerServiceComposition services{transaction};
    services.authority.displaceForTest(
        options.schedulerAuthority, foreign, boundary);
}

void PersistSchedulerAuthorityLossForTest(
    const SchedulerOptions& options,
    const std::string& boundary)
{
    if (!SchedulerAuthorityTestFailpointEnabled(boundary))
        return;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    InjectSchedulerAuthorityLossForTest(
        transaction, options, boundary);
    transaction.commit();
}

bool RefreshSchedulerAuthority(const SchedulerOptions& options)
{
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        RequireAndRefreshSchedulerAuthority(transaction, options);
        transaction.commit();
        return true;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_OWNERSHIP_LOST"
                  << ",scheduler_invocation_id="
                  << options.schedulerAuthority.schedulerInvocationId
                  << ",fencing_token="
                  << options.schedulerAuthority.fencingToken
                  << ",reason=" << error.what()
                  << std::endl;
        return false;
    }
}

void ReleaseSchedulerAuthority(
    const SchedulerOptions& options,
    const std::string& reason)
{
    if (!options.schedulerAuthority.held)
        return;
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        SchedulerServiceComposition services{transaction};
        const bool released = services.authority.release(
            options.schedulerAuthority, reason);
        transaction.commit();
        std::cout << "SCHEDULER_OWNERSHIP_RELEASED"
                  << ",scheduler_invocation_id="
                  << options.schedulerAuthority.schedulerInvocationId
                  << ",fencing_token="
                  << options.schedulerAuthority.fencingToken
                  << ",released=" << (released ? 1 : 0)
                  << ",reason=" << reason
                  << std::endl;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_OWNERSHIP_RELEASE_FAILED"
                  << ",scheduler_invocation_id="
                  << options.schedulerAuthority.schedulerInvocationId
                  << ",fencing_token="
                  << options.schedulerAuthority.fencingToken
                  << ",reason=" << error.what()
                  << std::endl;
    }
}

void SchedulerStopSignalHandler(int)
{
    gSchedulerStopRequested = 1;
}

void InstallSchedulerSignalHandlers()
{
    struct sigaction action {};
    action.sa_handler = SchedulerStopSignalHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    if (::sigaction(SIGINT, &action, nullptr) != 0 ||
        ::sigaction(SIGTERM, &action, nullptr) != 0)
    {
        throw std::runtime_error(
            "scheduler_signal_handler_install_failed");
    }

    struct sigaction pipeAction {};
    pipeAction.sa_handler = SIG_IGN;
    sigemptyset(&pipeAction.sa_mask);
    pipeAction.sa_flags = 0;
    if (::sigaction(SIGPIPE, &pipeAction, nullptr) != 0)
        throw std::runtime_error(
            "scheduler_sigpipe_handler_install_failed");
}

class SchedulerAuthorityReleaseGuard
{
public:
    explicit SchedulerAuthorityReleaseGuard(
        const SchedulerOptions& options)
        : options_(options)
    {
    }

    ~SchedulerAuthorityReleaseGuard()
    {
        ReleaseSchedulerAuthority(
            options_,
            gSchedulerStopRequested != 0
                ? "graceful_signal_shutdown"
                : "scheduler_exit");
    }

private:
    const SchedulerOptions& options_;
};

void PrintModelSymbolMissing(long long modelId)
{
    std::cout << "MODEL_SYMBOL_MISSING"
              << ",model_id=" << modelId
              << std::endl;
}

std::optional<std::string> TryLoadPersistedCanonicalSymbol(pqxx::work& w,
                                                                  long long modelId)
{
    try
    {
        return DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

void ResolveEconomicCalendarSnapshotForQueue(
    pqxx::work& transaction,
    SchedulerOptions& options,
    const std::string& creationMode)
{
    if (options.economicCalendarSnapshot) return;
    if (options.resumeModelId)
    {
        options.economicCalendarSnapshot =
            EA::EconomicCalendar::LoadModelEconomicCalendarSnapshot(
                transaction, *options.resumeModelId);
        return;
    }
    const auto report =
        EA::EconomicCalendar::CreateOrReuseEconomicCalendarSnapshot(
            transaction, creationMode, std::nullopt, false);
    if (!report.snapshotId)
        throw std::runtime_error(
            "economic_calendar_snapshot_queue_resolution_failed");
    options.economicCalendarSnapshot =
        EA::EconomicCalendar::EconomicCalendarSnapshotIdentity{
            *report.snapshotId, report.contentHash};
}

QueuedModelInputIdentity ResolveQueuedModelInputIdentity(
    pqxx::work& w, const SchedulerOptions& options)
{
    if (!ColumnExists(w, "experiment", "model_input_width") ||
        !ColumnExists(w, "experiment",
                      "model_input_semantic_layout_version"))
    {
        throw std::runtime_error(
            "model input identity migration required; run ./migrate_lstm_db.sh");
    }

    QueuedModelInputIdentity identity;
    if (options.resumeModelId.has_value())
    {
        (void)DBIO::PgModelIO::validateModelInputSemanticsForLoad(
            w, *options.resumeModelId);
        const std::size_t sourceWidth =
            DBIO::PgModelIO::loadRequiredModelMeta(
                w, *options.resumeModelId).inputWidth;
        identity.width = options.resumeExpandInputWidth
            ? EA::kCurrentModelInputWidth : sourceWidth;
    }
    (void)EA::ContractForModelInputWidth(identity.width);
    if (!EA::IsModelInputSemanticLayoutWidthCompatible(
            identity.semanticLayoutVersion, identity.width,
            EA::kModelInputSemanticLayoutRegistry,
            EA::kRegisteredModelInputWidths,
            EA::kModelInputSemanticLayoutVersion,
            EA::kCurrentModelInputWidth, false))
    {
        throw std::runtime_error(
            "queued model input identity is incompatible with this binary");
    }
    return identity;
}

long long InsertExperimentRecord(pqxx::work& w,
                                        const SchedulerOptions& options,
                                        const std::string& canonicalSymbol,
                                        long long duplicateNonce)
{
    const QueuedModelInputIdentity inputIdentity =
        ResolveQueuedModelInputIdentity(w, options);
    const bool includeRunMetadata = EA::RunMetadata::ExperimentRunMetadataColumnsExist(w);
    const bool hasDonchian20Mode = ColumnExists(w, "experiment", "donchian20_mode");
    if (!hasDonchian20Mode)
        throw std::runtime_error("Donchian-20 mode migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "feature_warmup_scope"))
        throw std::runtime_error("feature warmup scope migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "donchian_lookback"))
        throw std::runtime_error("donchian lookback migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "feature_ablation_mask"))
        throw std::runtime_error("feature ablation migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "fresh_initialization_seed"))
        throw std::runtime_error("fresh initialization seed migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "resume_expand_input_width"))
        throw std::runtime_error(
            "input width expansion migration required; run ./migrate_lstm_db.sh");
    if (!ColumnExists(w, "experiment", "training_objective_canonical") ||
        !ColumnExists(w, "experiment", "training_objective_hash"))
        throw std::runtime_error(
            "training objective provenance migration required; run ./migrate_lstm_db.sh");
    const bool hasCheckpointInferEnabled = ColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = ColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointInferMinEpoch = ColumnExists(w, "experiment", "checkpoint_infer_min_epoch");
    const bool hasCheckpointInferInterval = ColumnExists(w, "experiment", "checkpoint_infer_interval");
    const bool hasCheckpointPolicyEnabled = ColumnExists(w, "experiment", "checkpoint_policy_enabled");
    const bool hasCheckpointPolicyMinLeaderScore = ColumnExists(w, "experiment", "checkpoint_policy_min_leader_score");
    const bool hasCheckpointPolicyMinInferAccuracy = ColumnExists(w, "experiment", "checkpoint_policy_min_infer_accuracy");
    const bool hasCheckpointPolicyTopN = ColumnExists(w, "experiment", "checkpoint_policy_top_n");
    const bool hasCheckpointPolicyScope = ColumnExists(w, "experiment", "checkpoint_policy_scope");
    const bool hasCheckpointPolicyStopMode = ColumnExists(w, "experiment", "checkpoint_policy_stop_mode");
    const bool hasCheckpointPolicyGraceEvals = ColumnExists(w, "experiment", "checkpoint_policy_grace_evals");
    const bool hasContinuationCandidateExcluded =
        ColumnExists(w, "experiment", "continuation_candidate_excluded");
    if (options.queueCheckpointPolicy &&
        (!hasCheckpointPolicyEnabled ||
         !hasCheckpointPolicyScope ||
         !hasCheckpointPolicyStopMode ||
         !hasCheckpointPolicyGraceEvals))
    {
        throw std::runtime_error("checkpoint policy migration required; run ./migrate_lstm_db.sh");
    }
    if (options.queueContinuationCandidateExcluded && !hasContinuationCandidateExcluded)
        throw std::runtime_error("continuation policy migration required; run ./migrate_lstm_db.sh");
    const std::string invocationMode = options.queueInvocationMode.value_or(
        options.queueSweep
            ? "queue_sweep"
            : (options.queueExperiment
                   ? "queue_experiment" : "enqueue_experiment"));
    const EA::RunMetadata::Snapshot runMetadata =
        EA::RunMetadata::Capture(options.schedulerExecutablePath, invocationMode);
    const std::string schemaVersion = EA::RunMetadata::CurrentSchemaVersion(w);

    std::ostringstream sql;
    sql << "INSERT INTO experiment ("
        << "symbol, prediction_horizon, c_next_threshold, core_lr_mult, head_lr_mult, "
        << "target_epochs, checkpoint_interval, train_start, train_end, infer_start, infer_end, "
        << "resume_model_id, duplicate_nonce, status, phase, updated_at";
    sql << ", donchian20_mode, feature_warmup_scope, donchian_lookback, feature_ablation_mask, fresh_initialization_seed, resume_expand_input_width, training_objective_canonical, training_objective_hash, training_objective_id, training_objective_version, loss_definition_version, auxiliary_loss_mode, auxiliary_loss_coefficient, regression_target_definition, regression_normalization_identity, robust_loss_definition, robust_loss_delta, target_clipping_definition, objective_normalization_identity, model_input_width, model_input_semantic_layout_version, economic_calendar_snapshot_id, economic_calendar_snapshot_hash";
    if (hasCheckpointInferEnabled)
        sql << ", checkpoint_infer_enabled";
    if (hasOpportunisticCheckpointInfer)
        sql << ", opportunistic_checkpoint_infer";
    if (hasCheckpointInferMinEpoch)
        sql << ", checkpoint_infer_min_epoch";
    if (hasCheckpointInferInterval)
        sql << ", checkpoint_infer_interval";
    if (hasCheckpointPolicyEnabled)
        sql << ", checkpoint_policy_enabled";
    if (hasCheckpointPolicyMinLeaderScore)
        sql << ", checkpoint_policy_min_leader_score";
    if (hasCheckpointPolicyMinInferAccuracy)
        sql << ", checkpoint_policy_min_infer_accuracy";
    if (hasCheckpointPolicyTopN)
        sql << ", checkpoint_policy_top_n";
    if (hasCheckpointPolicyScope)
        sql << ", checkpoint_policy_scope";
    if (hasCheckpointPolicyStopMode)
        sql << ", checkpoint_policy_stop_mode";
    if (hasCheckpointPolicyGraceEvals)
        sql << ", checkpoint_policy_grace_evals";
    if (hasContinuationCandidateExcluded)
        sql << ", continuation_candidate_excluded";
    if (includeRunMetadata)
        EA::RunMetadata::AppendRunMetadataColumns(sql);
    sql << ") VALUES ("
        << w.quote(canonicalSymbol) << ","
        << *options.predictionHorizon << ","
        << FormatDouble(*options.cNextThreshold) << ","
        << SqlNullable(w, options.coreLrMult) << ","
        << SqlNullable(w, options.headLrMult) << ","
        << *options.targetEpochs << ","
        << options.checkpointInterval << ","
        << w.quote(*options.trainStart) << "::timestamptz,"
        << w.quote(*options.trainEnd) << "::timestamptz,"
        << SqlNullable(w, options.inferStart) << "::timestamptz,"
        << SqlNullable(w, options.inferEnd) << "::timestamptz,"
        << SqlNullable(w, options.resumeModelId) << ","
        << duplicateNonce << ","
        << "'pending','train',now(),"
        << w.quote(Donchian20ModeText(options.donchian20Mode.value_or(kDefaultDonchian20Mode)))
        << "," << w.quote(EA::FeatureWarmupScopeText(options.featureWarmupScope))
        << "," << DonchianLookbackDatabaseValue(options.donchianLookback)
        << "," << w.quote(options.featureAblationMask)
        << "," << options.freshInitializationSeed
        << "," << (options.resumeExpandInputWidth ? "true" : "false")
        << "," << w.quote(EA::TrainingObjective::CanonicalText(
            options.trainingObjective))
        << "," << w.quote(EA::TrainingObjective::Identity(
            options.trainingObjective))
        << "," << w.quote(options.trainingObjective.objectiveIdentifier)
        << "," << options.trainingObjective.objectiveVersion
        << "," << options.trainingObjective.lossDefinitionVersion
        << "," << w.quote(EA::TrainingObjective::AuxiliaryLossModeText(
            options.trainingObjective.auxiliaryLossMode))
        << "," << EA::TrainingObjective::CanonicalDouble(
            options.trainingObjective.auxiliaryLossCoefficient)
        << "," << (options.trainingObjective.regressionTargetDefinition
            ? w.quote(*options.trainingObjective.regressionTargetDefinition)
            : "NULL")
        << "," << (options.trainingObjective.regressionNormalizationIdentity
            ? w.quote(*options.trainingObjective.regressionNormalizationIdentity)
            : "NULL")
        << "," << (options.trainingObjective.robustLossDefinition
            ? w.quote(*options.trainingObjective.robustLossDefinition)
            : "NULL")
        << "," << (options.trainingObjective.robustLossDelta
            ? EA::TrainingObjective::CanonicalDouble(
                  *options.trainingObjective.robustLossDelta)
            : "NULL")
        << "," << w.quote(options.trainingObjective.targetClippingDefinition)
        << "," << w.quote(
            EA::TrainingObjective::AuxiliaryEnabled(options.trainingObjective)
                ? "classification_weighted_ce_plus_unweighted_coefficient_huber__loss_and_gradients_by_true_class_weight_sum__calculate_batch_return_by_example_count_v1"
                : "weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1")
        << "," << inputIdentity.width
        << "," << inputIdentity.semanticLayoutVersion
        << "," << (options.economicCalendarSnapshot
                ? std::to_string(options.economicCalendarSnapshot->snapshotId)
                : "NULL")
        << "," << (options.economicCalendarSnapshot
                ? w.quote(options.economicCalendarSnapshot->contentHash)
                : "NULL");
    if (hasCheckpointInferEnabled)
        sql << "," << (options.queueCheckpointInfer ? "true" : "false");
    if (hasOpportunisticCheckpointInfer)
        sql << "," << (options.queueCheckpointInfer ? "true" : "false");
    if (hasCheckpointInferMinEpoch)
        sql << "," << SqlNullable(w, options.queueCheckpointInferMinEpoch);
    if (hasCheckpointInferInterval)
        sql << "," << SqlNullable(w, options.queueCheckpointInferInterval);
    if (hasCheckpointPolicyEnabled)
        sql << "," << (options.queueCheckpointPolicy ? "true" : "false");
    if (hasCheckpointPolicyMinLeaderScore)
        sql << "," << SqlNullable(w, options.checkpointPolicyMinLeaderScore);
    if (hasCheckpointPolicyMinInferAccuracy)
        sql << "," << SqlNullable(w, options.checkpointPolicyMinInferAccuracy);
    if (hasCheckpointPolicyTopN)
        sql << "," << SqlNullable(w, options.checkpointPolicyTopN);
    if (hasCheckpointPolicyScope)
        sql << "," << w.quote(options.checkpointPolicyScope);
    if (hasCheckpointPolicyStopMode)
        sql << "," << w.quote(options.checkpointPolicyStopMode);
    if (hasCheckpointPolicyGraceEvals)
        sql << "," << options.checkpointPolicyGraceEvals;
    if (hasContinuationCandidateExcluded)
        sql << "," << (options.queueContinuationCandidateExcluded ? "true" : "false");
    if (includeRunMetadata)
        EA::RunMetadata::AppendRunMetadataValues(sql, w, runMetadata, schemaVersion);
    sql << ") RETURNING experiment_id;";

    pqxx::result inserted = w.exec(sql.str());
    return inserted[0][0].as<long long>();
}

void EnsureRequiredQueueOptions(const SchedulerOptions& options)
{
    if (options.epochs.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep use --target-epochs as the absolute final epoch; --epochs is not supported");
    if (options.autoResume && !options.queueExperiment)
        throw std::invalid_argument("--auto-resume is supported only with --queue-experiment");
    if (options.autoResume && options.resumeModelId.has_value())
        throw std::invalid_argument("--auto-resume cannot be combined with --resume-model-id");
    if (options.queueSweep && options.resumeModelId.has_value())
        throw std::invalid_argument("--queue-sweep does not support --resume-model-id");
    if (options.queueSweep && options.autoResume)
        throw std::invalid_argument("--queue-sweep does not support --auto-resume");
    if (options.resumeExpandInputWidth && !options.resumeModelId.has_value())
        throw std::invalid_argument(
            "--resume-expand-input-width requires --resume-model-id");
    if (!options.targetEpochs.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep require --target-epochs");
    if (!options.resumeModelId.has_value() && !options.autoResume && !options.predictionHorizon.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep require --prediction-horizon");
    if (options.queueExperiment && !options.symbol.has_value())
    {
        if (!options.resumeModelId.has_value())
            throw std::invalid_argument("--queue-experiment requires --symbol unless --resume-model-id is supplied");
    }
    if (options.queueSweep && options.symbol.has_value())
        throw std::invalid_argument("--queue-sweep queues all supported symbols; do not pass --symbol");
    if (!options.queueCheckpointInfer &&
        (options.queueCheckpointInferMinEpoch.has_value() ||
         options.queueCheckpointInferInterval.has_value()))
    {
        throw std::invalid_argument("--checkpoint-infer-min-epoch/--checkpoint-infer-interval require --checkpoint-infer when queueing experiments");
    }
    if (options.queueCheckpointPolicy && !options.queueCheckpointInfer)
        throw std::invalid_argument("--checkpoint-policy requires --checkpoint-infer");
    ValidateCheckpointPolicyConfig(options);
}

std::optional<double> OptionalDoubleCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<double>();
}

std::optional<long long> OptionalLongLongCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<long long>();
}

std::optional<std::string> OptionalStringCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<std::string>();
}

ExperimentRow RowToExperiment(const pqxx::row& row)
{
    ExperimentRow experiment;
    experiment.experimentId = row[0].as<long long>();
    experiment.symbol = EA::CanonicalSymbol::Normalize(row[1].as<std::string>());
    experiment.predictionHorizon = row[2].as<int>();
    experiment.cNextThreshold = row[3].as<double>();
    experiment.coreLrMult = OptionalDoubleCell(row, 4);
    experiment.headLrMult = OptionalDoubleCell(row, 5);
    experiment.targetEpochs = row[6].as<int>();
    experiment.checkpointInterval = row[7].as<int>();
    experiment.trainStart = row[8].as<std::string>();
    experiment.trainEnd = row[9].as<std::string>();
    experiment.inferStart = OptionalStringCell(row, 10);
    experiment.inferEnd = OptionalStringCell(row, 11);
    experiment.lastModelId = OptionalLongLongCell(row, 12);
    experiment.resumeModelId = OptionalLongLongCell(row, 13);
    experiment.trainLogPath = OptionalStringCell(row, 14);
    experiment.inferLogPath = OptionalStringCell(row, 15);
    experiment.analysisLogPath = OptionalStringCell(row, 16);
    experiment.donchian20Mode = ParseDonchian20Mode(row[17].as<std::string>());
    experiment.featureWarmupScope = EA::ParseFeatureWarmupScope(
        row[18].as<std::string>());
    experiment.donchianLookback = ParseDonchianLookback(row[19].as<std::string>());
    experiment.featureAblationMask = EA::FeatureAblationMask::Parse(
        row[20].as<std::string>()).CanonicalText();
    experiment.freshInitializationSeed = row[21].as<unsigned int>();
    experiment.resumeExpandInputWidth = row[22].as<bool>();
    experiment.trainingObjective = EA::TrainingObjective::ResolvePersisted(
        row[23].as<std::string>(), row[24].as<std::string>());
    return experiment;
}

std::optional<ExperimentRow> LoadExperimentCheckpointIdentity(
    pqxx::work& w,
    long long experimentId)
{
    const pqxx::result rows = w.exec(
        "SELECT experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start::text,train_end::text,infer_start::text,infer_end::text,"
        "last_model_id,resume_model_id,train_log_path,infer_log_path,"
        "analysis_log_path,donchian20_mode,feature_warmup_scope,"
        "donchian_lookback,feature_ablation_mask,fresh_initialization_seed,resume_expand_input_width,"
        "training_objective_canonical,training_objective_hash "
        "FROM experiment WHERE experiment_id=" +
        std::to_string(experimentId) + ";");
    if (rows.size() != 1)
        return std::nullopt;
    return RowToExperiment(rows[0]);
}

ExperimentRow RepositoryRecordToExperiment(
    const EA::SchedulerCore::SchedulerExperimentRecord& record)
{
    ExperimentRow experiment;
    experiment.experimentId = record.experimentId;
    experiment.symbol = EA::CanonicalSymbol::Normalize(record.symbol);
    experiment.predictionHorizon = record.predictionHorizon;
    experiment.cNextThreshold = record.cNextThreshold;
    experiment.coreLrMult = record.coreLrMult;
    experiment.headLrMult = record.headLrMult;
    experiment.targetEpochs = record.targetEpochs;
    experiment.checkpointInterval = record.checkpointInterval;
    experiment.trainStart = record.trainStart;
    experiment.trainEnd = record.trainEnd;
    experiment.inferStart = record.inferStart;
    experiment.inferEnd = record.inferEnd;
    experiment.lastModelId = record.lastModelId;
    experiment.resumeModelId = record.resumeModelId;
    experiment.trainLogPath = record.trainLogPath;
    experiment.inferLogPath = record.inferLogPath;
    experiment.analysisLogPath = record.analysisLogPath;
    experiment.donchian20Mode =
        ParseDonchian20Mode(record.donchian20Mode);
    experiment.featureWarmupScope =
        EA::ParseFeatureWarmupScope(record.featureWarmupScope);
    experiment.donchianLookback =
        ParseDonchianLookback(record.donchianLookback);
    experiment.featureAblationMask = EA::FeatureAblationMask::Parse(
        record.featureAblationMask).CanonicalText();
    experiment.freshInitializationSeed = record.freshInitializationSeed;
    experiment.resumeExpandInputWidth = record.resumeExpandInputWidth;
    experiment.trainingObjective = EA::TrainingObjective::ResolvePersisted(
        record.trainingObjectiveCanonical,
        record.trainingObjectiveHash);
    return experiment;
}

std::vector<ExperimentRow> LoadPendingExperiments(
    EA::SchedulerCore::SchedulerAdmissionService& admission,
    const std::string& phase,
    bool cancellationOnly = false)
{
    const auto records =
        admission.loadCandidates(phase, cancellationOnly);
    std::vector<ExperimentRow> experiments;
    experiments.reserve(records.size());
    for (const auto& record : records)
    {
        ExperimentRow experiment =
            RepositoryRecordToExperiment(record.experiment);
        experiment.schedulerPriority = record.schedulerPriority;
        experiment.resumeRequested = record.resumeRequested;
        experiment.schedulerResumeOrigin = record.schedulerResumeOrigin;
        experiment.activeWorkerAttemptId = record.activeWorkerAttemptId;
        experiments.push_back(std::move(experiment));
    }
    return experiments;
}

int RecoverOrphanedRunningExperiments(pqxx::work& w,
                                      const SchedulerOptions& options,
                                      SchedulerEventLogState* logState,
                                      bool verbose);

void AdvanceCheckpointEvalToAnalyze(
    pqxx::work& w,
    const CheckpointEvalRow& eval,
    long long inferenceResultId,
    bool hasInferCompletedAt,
    const std::optional<long long>&
        workerAttemptId = std::nullopt);

QueueSnapshot LoadQueueSnapshot(
    EA::SchedulerCore::SchedulerAdmissionService& admission)
{
    const auto record = admission.loadQueueSnapshot();
    return QueueSnapshot{
        record.pendingTrain,
        record.pendingInfer,
        record.pendingAnalyze,
        record.runningTrain,
        record.runningInfer,
        record.runningAnalyze};
}

std::string QueueSnapshotKey(const QueueSnapshot& snapshot)
{
    std::ostringstream key;
    key << snapshot.pendingTrain << "|"
        << snapshot.pendingInfer << "|"
        << snapshot.pendingAnalyze << "|"
        << snapshot.runningTrain << "|"
        << snapshot.runningInfer << "|"
        << snapshot.runningAnalyze;
    return key.str();
}

void PrintQueueSnapshot(const QueueSnapshot& snapshot,
                        SchedulerEventLogState* logState,
                        bool verbose)
{
    const std::string key = QueueSnapshotKey(snapshot);
    if (!verbose && logState != nullptr && logState->previousQueueKey == key)
        return;

    std::cout << "SCHEDULER_QUEUE"
              << ",pending_train=" << snapshot.pendingTrain
              << ",pending_infer=" << snapshot.pendingInfer
              << ",pending_analyze=" << snapshot.pendingAnalyze
              << ",running_train=" << snapshot.runningTrain
              << ",running_infer=" << snapshot.runningInfer
              << ",running_analyze=" << snapshot.runningAnalyze
              << std::endl;
    if (logState != nullptr)
        logState->previousQueueKey = key;
}

void LogSkip(const std::string& phase,
             long long experimentId,
             const std::string& reason,
             SchedulerEventLogState* logState,
             bool verbose)
{
    const std::string key = phase + "|" + std::to_string(experimentId) + "|" + reason;
    if (logState != nullptr)
        logState->currentSkipKeys.insert(key);
    if (!verbose &&
        logState != nullptr &&
        logState->previousSkipKeys.find(key) != logState->previousSkipKeys.end())
    {
        return;
    }

    std::cout << "SCHEDULER_SKIP_" << (phase == "train" ? "TRAIN" : phase == "infer" ? "INFER" : "ANALYZE")
              << ",experiment_id=" << experimentId
              << ",reason=" << reason
              << std::endl;
}

EA::Scheduler::SemanticAdmissionDecision LoadSemanticWorkerAdmission(
    pqxx::transaction_base& transaction,
    long long experimentId,
    const std::string& phase,
    EA::Scheduler::PersistedWorkerSemanticIdentity* loaded = nullptr)
{
    const pqxx::result rows = transaction.exec_params(
        "SELECT model_input_width,model_input_semantic_layout_version,"
        "last_model_id,resume_model_id "
        "FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.size() != 1)
        return {false, "semantic_worker_identity_unavailable"};

    EA::Scheduler::PersistedWorkerSemanticIdentity persisted;
    if (!rows[0][0].is_null())
        persisted.inputWidth = rows[0][0].as<std::size_t>();
    if (!rows[0][1].is_null())
        persisted.layoutVersion = rows[0][1].as<int>();

    const std::optional<long long> lastModelId =
        rows[0][2].is_null()
            ? std::nullopt
            : std::optional<long long>{rows[0][2].as<long long>()};
    const std::optional<long long> resumeModelId =
        rows[0][3].is_null()
            ? std::nullopt
            : std::optional<long long>{rows[0][3].as<long long>()};

    persisted.modelIdentityExpected =
        lastModelId.has_value() || resumeModelId.has_value();

    // Migration 089 intentionally preserves historical experiments as
    // NULL/NULL.  If an explicit experiment identity exists, it remains
    // authoritative and immutable.  Only recover identity from a model for
    // the intentional legacy NULL/NULL case.
    if (!persisted.inputWidth && !persisted.layoutVersion)
    {
        std::optional<long long> authoritativeModelId;
        if (phase == "infer")
        {
            authoritativeModelId = lastModelId;
        }
        else if (phase == "train")
        {
            authoritativeModelId =
                resumeModelId.has_value() ? resumeModelId : lastModelId;
        }

        if (authoritativeModelId.has_value())
        {
            try
            {
                const auto modelMeta =
                    DBIO::PgModelIO::loadRequiredModelMeta(
                        transaction, *authoritativeModelId);
                persisted.inputWidth = modelMeta.inputWidth;

                const auto semanticMetadata =
                    DBIO::PgModelIO::loadModelInputSemanticMetadata(
                        transaction, *authoritativeModelId);

                if (!semanticMetadata.has_value())
                {
                    if (loaded != nullptr) *loaded = persisted;
                    return EA::Scheduler::
                        EvaluateLegacyMarkerlessModelAdmission(
                            modelMeta.inputWidth);
                }

                if (semanticMetadata->schemaVersion !=
                    EA::kModelInputSemanticMetaSchemaVersion)
                {
                    if (loaded != nullptr) *loaded = persisted;
                    return {false, "semantic_worker_model_identity_invalid"};
                }

                persisted.layoutVersion =
                    semanticMetadata->layoutVersion;
            }
            catch (const std::exception&)
            {
                if (loaded != nullptr) *loaded = persisted;
                return {false, "semantic_worker_model_identity_invalid"};
            }
        }
    }

    if (loaded != nullptr) *loaded = persisted;
    return EA::Scheduler::EvaluateSemanticWorkerAdmission(phase, persisted);
}

const EA::Scheduler::SemanticWorkerRegistry&
SemanticWorkerRegistryFor(const SchedulerOptions& options)
{
    if (!options.semanticWorkerRegistry)
        throw std::logic_error("semantic_worker_registry_not_initialized");
    return *options.semanticWorkerRegistry;
}

EA::Scheduler::InferenceWorkerSelection LoadInferenceWorkerSelection(
    pqxx::transaction_base& transaction,
    const SchedulerOptions& options,
    long long experimentId,
    EA::Scheduler::PersistedWorkerSemanticIdentity* loaded = nullptr)
{
    EA::Scheduler::PersistedWorkerSemanticIdentity persisted;
    (void)LoadSemanticWorkerAdmission(
        transaction, experimentId, "infer", &persisted);
    if (loaded != nullptr) *loaded = persisted;
    return EA::Scheduler::SelectInferenceWorker(
        persisted, SemanticWorkerRegistryFor(options));
}

bool SemanticWorkerPreflight(
    const SchedulerOptions& options,
    long long experimentId,
    const std::optional<long long>& modelId,
    const std::string& phase,
    SchedulerEventLogState* logState,
    bool verbose)
{
    if (phase == "analyze") return true;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::read_transaction transaction{connection};
    EA::Scheduler::PersistedWorkerSemanticIdentity persisted;
    EA::Scheduler::SemanticAdmissionDecision decision;
    std::optional<EA::Scheduler::InferenceWorkerSelection> selection;
    if (phase == "infer")
    {
        selection = LoadInferenceWorkerSelection(
            transaction, options, experimentId, &persisted);
        decision = {selection->selected, selection->diagnostic};
    }
    else
    {
        decision = LoadSemanticWorkerAdmission(
            transaction, experimentId, phase, &persisted);
    }
    if (decision.admissible)
    {
        const std::string& executable = selection
            ? selection->canonicalExecutablePath
            : options.currentWorkerExecutablePath;
        const auto runtime = SemanticWorkerRegistryFor(options)
            .validateRuntimeForExecutable(executable);
        if (!runtime.ready)
        {
            LogSkip(phase, experimentId, runtime.diagnostic, logState, verbose);
            std::cout << "SCHEDULER_SEMANTIC_WORKER_RUNTIME_UNAVAILABLE"
                      << ",experiment_id=" << experimentId
                      << ",phase=" << phase
                      << ",worker_executable=" << executable
                      << ",diagnostic=" << runtime.diagnostic
                      << ",capacity_consumed=0,child_launched=0,"
                         "experiment_status_changed=false"
                      << std::endl;
            return false;
        }
    }
    if (decision.admissible)
    {
        if (selection)
        {
            std::ostringstream key;
            key << experimentId << '|' << modelId.value_or(-1) << '|'
                << persisted.inputWidth.value_or(0) << '|'
                << persisted.layoutVersion.value_or(0) << '|'
                << selection->semanticLayoutVersion << '|'
                << selection->canonicalExecutablePath;
            const bool alreadyLogged =
                !verbose && logState != nullptr &&
                logState->previousWorkerSelectionKeys.contains(key.str());
            if (logState != nullptr)
                logState->currentWorkerSelectionKeys.insert(key.str());
            if (!alreadyLogged)
            {
                std::cout << "SCHEDULER_INFER_WORKER_SELECTED"
                          << ",experiment_id=" << experimentId
                          << ",model_id="
                          << (modelId ? std::to_string(*modelId) : "NULL")
                          << ",model_input_width="
                          << *persisted.inputWidth
                          << ",model_input_semantic_layout_version="
                          << *persisted.layoutVersion
                          << ",worker_semantic_layout_version="
                          << selection->semanticLayoutVersion
                          << ",worker_executable="
                          << selection->canonicalExecutablePath
                          << ",reason=" << selection->reason
                          << std::endl;
            }
        }
        return true;
    }

    const std::string reason = decision.diagnostic;
    LogSkip(phase, experimentId, reason, logState, verbose);
    if (verbose || logState == nullptr ||
        logState->previousSkipKeys.find(
            phase + "|" + std::to_string(experimentId) + "|" + reason) ==
            logState->previousSkipKeys.end())
    {
        std::cout << "SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE"
                  << ",experiment_id=" << experimentId
                  << ",phase=" << phase
                  << ",model_input_width="
                  << (persisted.inputWidth
                          ? std::to_string(*persisted.inputWidth) : "NULL")
                  << ",model_input_semantic_layout_version="
                  << (persisted.layoutVersion
                          ? std::to_string(*persisted.layoutVersion) : "NULL")
                  << ",worker_maximum_input_width="
                  << (selection && selection->maximumInputWidth != 0
                          ? selection->maximumInputWidth
                          : EA::kCurrentModelInputWidth)
                  << ",worker_semantic_layout_version="
                  << (selection && selection->semanticLayoutVersion != 0
                          ? selection->semanticLayoutVersion
                          : EA::kModelInputSemanticLayoutVersion)
                  << ",diagnostic=" << reason
                  << ",capacity_consumed=0,child_launched=0,"
                     "experiment_status_changed=false"
                  << std::endl;
    }
    return false;
}

std::string PhaseSchedulingStatsKey(const PhaseSchedulingStats& stats)
{
    std::ostringstream key;
    key << stats.examined << "|"
        << stats.skipped << "|"
        << stats.launched << "|"
        << stats.freeSlots;
    return key.str();
}

void PrintPhaseSchedulingStats(const PhaseSchedulingStats& stats,
                               SchedulerEventLogState* logState,
                               bool verbose)
{
    const std::string key = PhaseSchedulingStatsKey(stats);
    if (!verbose &&
        logState != nullptr &&
        logState->previousPhaseKeys[stats.phase] == key)
    {
        return;
    }

    std::cout << "SCHEDULER_QUEUE_PHASE"
              << ",phase=" << stats.phase
              << ",examined=" << stats.examined
              << ",skipped=" << stats.skipped
              << ",launched=" << stats.launched
              << ",free_slots=" << stats.freeSlots
              << std::endl;
    if (logState != nullptr)
        logState->previousPhaseKeys[stats.phase] = key;
}

int FailInvalidSchedulerPhases(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT experiment_id,phase,status,"
        "active_scheduler_worker_attempt_id "
        "FROM experiment "
        "WHERE status IN ('pending', 'running') "
        "AND phase NOT IN ('train', 'infer', 'analyze', 'done') "
        "ORDER BY updated_at ASC, experiment_id ASC;");

    for (const auto& row : rows)
    {
        const long long experimentId = row[0].as<long long>();
        const std::string phase = row[1].as<std::string>();
        if (row[2].as<std::string>() == "running" ||
            !row[3].is_null())
        {
            std::cout << "EXPERIMENT_INVALID_PHASE_DEFERRED"
                      << ",experiment_id=" << experimentId
                      << ",phase=" << phase
                      << ",reason=exact_active_attempt_required"
                      << std::endl;
            continue;
        }
        pqxx::result failed = w.exec_params(
            "UPDATE experiment "
            "SET status = 'failed', "
            "exit_code = -1, "
            "error_message = $1, "
            "completed_at = now(), "
            "updated_at = now() "
            "WHERE experiment_id = $2 "
            "AND status='pending' "
            "AND active_scheduler_worker_attempt_id IS NULL "
            "RETURNING experiment_id;",
            "unknown_scheduler_phase:" + phase,
            experimentId);
        if (failed.size() != 1)
            continue;
        std::cout << "EXPERIMENT_FAILED"
                  << ",experiment_id=" << experimentId
                  << ",phase=" << phase
                  << ",reason=unknown_scheduler_phase"
                  << std::endl;
    }

    return rows.empty() ? 0 : 1;
}

void EnsureLogDir(const std::string& logDir)
{
    std::filesystem::create_directories(logDir);
}

std::string LogPathFor(const SchedulerOptions& options,
                              const ExperimentRow& experiment,
                              const std::string& phase)
{
    std::ostringstream oss;
    oss << options.schedulerLogDir
        << "/experiment_" << experiment.experimentId
        << "_" << EA::CanonicalSymbol::Normalize(experiment.symbol)
        << "_" << phase << ".log";
    return oss.str();
}

std::string BaseModelName(const ExperimentRow& experiment)
{
    std::ostringstream oss;
    oss << EA::CanonicalSymbol::Normalize(experiment.symbol)
        << "-experiment" << experiment.experimentId
        << "_h" << experiment.predictionHorizon
        << "_e" << experiment.targetEpochs;
    return oss.str();
}

std::string ShellDisplayQuote(const std::string& value)
{
    if (value.find_first_of(" \t\n\"'\\$`") == std::string::npos)
        return value;
    std::string quoted = "'";
    for (char c : value)
    {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted.push_back(c);
    }
    quoted.push_back('\'');
    return quoted;
}

std::string CommandForDisplay(const std::vector<std::string>& argv)
{
    std::ostringstream oss;
    for (size_t i = 0; i < argv.size(); ++i)
    {
        if (i)
            oss << ' ';
        oss << ShellDisplayQuote(argv[i]);
    }
    return oss.str();
}

std::string CommandForProcessObservation(
    const std::vector<std::string>& argv)
{
    return EA::SchedulerCore::WorkerCommandLine(argv);
}

void AddCliFlag(std::vector<std::string>& argv, const std::string& optionName)
{
    argv.push_back(optionName);
}

void AddCliOption(std::vector<std::string>& argv,
                         const std::string& optionName,
                         const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty value");
    argv.push_back(optionName + "=" + value);
}

void AddLstmProfileOptions(std::vector<std::string>& argv,
                           const SchedulerOptions& options)
{
    if (!options.lstmProfileHotspots)
        return;
    AddCliFlag(argv, "--lstm-profile-hotspots");
    if (options.lstmProfileOutputPath.has_value())
        AddCliOption(argv, "--lstm-profile-output", *options.lstmProfileOutputPath);
}

void AddCliPositional(std::vector<std::string>& argv, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("positional argument must not be empty");
    argv.push_back(value);
}

void PrintSchedulerExec(const std::vector<std::string>& argv)
{
    std::cout << "SCHEDULER_EXEC: "
              << CommandForDisplay(argv)
              << std::endl;
}

class SchedulerChildLaunchError : public std::runtime_error
{
public:
    SchedulerChildLaunchError(int errorNumber, const std::string& message)
        : std::runtime_error(message), errorNumber_(errorNumber)
    {
    }

    int ErrorNumber() const { return errorNumber_; }

private:
    int errorNumber_ = 0;
};

std::string RequireProcessStartIdentity(pid_t pid);

bool OperatorForcedFinalInferenceRerunRequested(
    pqxx::work& w,
    long long experimentId);

bool HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
    pqxx::work& w,
    const ExperimentRow& experiment,
    long long workerAttemptId,
    bool useThresholdTolerance);

EA::GlobalExperimentControl::ManagedWorker ManagedWorkerFromExactAttempt(
    const EA::SchedulerOwnership::ExactAttemptSnapshot& exact)
{
    EA::GlobalExperimentControl::ManagedWorker worker;
    worker.workerAttemptId = exact.workerAttemptId;
    worker.workerKind = exact.workerKind;
    worker.capacityClass = exact.capacityClass;
    worker.attemptLifecycleState = exact.lifecycleState;
    worker.launchAttemptIdentity = exact.launchAttemptIdentity;
    worker.experimentId = exact.experimentId;
    worker.phase = exact.lifecyclePhase;
    worker.lifecycleStatus = exact.lifecycleStatus;
    worker.pid = *exact.workerPid;
    worker.processGroupId = exact.processGroupId;
    worker.executable = exact.canonicalExecutablePath;
    worker.commandLine = exact.commandLine;
    worker.processStartIdentity = exact.processStartIdentity;
    return worker;
}

void CompensateSchedulerPreemptionRollback(
    const EA::GlobalExperimentControl::ManagedWorker& worker,
    EA::GlobalExperimentControl::ProcessOperations& processes)
{
    const auto resumed =
        EA::GlobalExperimentControl::ResumeWorker(worker, processes);
    std::cerr << "SCHEDULER_PREEMPTION_ROLLBACK_COMPENSATION"
              << ",experiment_id=" << worker.experimentId
              << ",worker_attempt_id="
              << worker.workerAttemptId.value_or(-1)
              << ",result=" << resumed.result
              << ",restored=" << (resumed.success ? 1 : 0)
              << ",detail=" << resumed.detail
              << std::endl;
}

bool PreemptOneLowerPriorityWorker(
    const SchedulerOptions& options,
    const ExperimentRow& candidate,
    const std::string& phase,
    int maximumCapacity)
{
    if (phase != "train" && phase != "infer")
        return false;

    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    if (!SchedulerLaunchAllowed(transaction, phase))
    {
        transaction.commit();
        return false;
    }

    const pqxx::result pending = transaction.exec(
        "SELECT status,phase,scheduler_priority,scheduler_resume_origin,"
        "cancellation_request_id,active_scheduler_worker_attempt_id "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{candidate.experimentId});
    if (pending.size() != 1 ||
        pending[0][0].as<std::string>() != "pending" ||
        pending[0][1].as<std::string>() != phase ||
        pending[0][2].as<std::string>() != candidate.schedulerPriority ||
        pending[0][3].as<std::string>() !=
            candidate.schedulerResumeOrigin ||
        !pending[0][4].is_null())
    {
        transaction.commit();
        return false;
    }
    if (services.admission.capacityUsed(phase) < maximumCapacity)
    {
        transaction.commit();
        return false;
    }

    const auto victim = services.admission.selectPreemptionVictim(
        phase, candidate.schedulerPriority);
    if (!victim)
    {
        std::cout << "SCHEDULER_PREEMPTION_DEFERRED"
                  << ",candidate_experiment_id="
                  << candidate.experimentId
                  << ",candidate_priority="
                  << candidate.schedulerPriority
                  << ",phase=" << phase
                  << ",reason=no_strictly_lower_pause_safe_victim"
                  << std::endl;
        transaction.commit();
        return false;
    }

    const long long victimExperimentId =
        victim->experimentId;
    const std::string victimPriority =
        victim->priority;
    const long long workerAttemptId =
        victim->workerAttemptId;
    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = workerAttemptId;
    expected.experimentId = victimExperimentId;
    expected.workerKind = "experiment";
    expected.lifecyclePhase = phase;
    expected.capacityClass = phase;
    expected.requireSignalable = true;
    expected.requireCompleteProcessIdentity = true;
    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
    if (!exact || exact->lifecycleStatus != "running" ||
        exact->lifecycleRowPhase != phase ||
        exact->lifecycleState == "stopped")
    {
        std::cout << "SCHEDULER_PREEMPTION_DEFERRED"
                  << ",candidate_experiment_id="
                  << candidate.experimentId
                  << ",victim_experiment_id="
                  << victimExperimentId
                  << ",worker_attempt_id=" << workerAttemptId
                  << ",phase=" << phase
                  << ",reason=exact_attempt_verification_failed"
                  << std::endl;
        transaction.commit();
        return false;
    }

    const pqxx::result reverified = transaction.exec(
        "SELECT status,phase,scheduler_priority,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "worker_global_pause_request_id,worker_control_state "
        "FROM experiment WHERE experiment_id=$1 "
        "AND active_scheduler_worker_attempt_id=$2;",
        pqxx::params{victimExperimentId, workerAttemptId});
    if (reverified.size() != 1 ||
        reverified[0][0].as<std::string>() != "running" ||
        reverified[0][1].as<std::string>() != phase ||
        !EA::SchedulerCore::CanPreempt(
            candidate.schedulerPriority,
            reverified[0][2].as<std::string>()) ||
        !reverified[0][3].is_null() ||
        !reverified[0][4].is_null() ||
        !reverified[0][5].is_null() ||
        !reverified[0][6].is_null() ||
        reverified[0][7].as<std::string>() != "running" ||
        services.admission.capacityUsed(phase) < maximumCapacity)
    {
        transaction.commit();
        return false;
    }

    const auto inferAttemptHasAuthoritativeResult = [&] {
        if (phase != "infer")
            return false;
        const auto victimExperiment = LoadExperimentCheckpointIdentity(
            transaction, victimExperimentId);
        return victimExperiment.has_value() &&
               HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
                   transaction,
                   *victimExperiment,
                   workerAttemptId,
                   OperatorForcedFinalInferenceRerunRequested(
                       transaction, victimExperimentId));
    };
    if (inferAttemptHasAuthoritativeResult())
    {
        std::cout << "SCHEDULER_PREEMPTION_DEFERRED"
                  << ",candidate_experiment_id="
                  << candidate.experimentId
                  << ",victim_experiment_id=" << victimExperimentId
                  << ",worker_attempt_id=" << workerAttemptId
                  << ",phase=infer"
                  << ",reason=authoritative_inference_result_observed"
                  << std::endl;
        transaction.commit();
        return false;
    }

    auto processes =
        EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const auto worker = ManagedWorkerFromExactAttempt(*exact);
    bool newlyStopped = false;
    bool commitAttempted = false;
    try
    {
        const auto signal =
            EA::GlobalExperimentControl::PauseWorker(worker, *processes);
        newlyStopped =
            signal.success &&
            std::find(signal.signals.begin(), signal.signals.end(), SIGSTOP) !=
                signal.signals.end();
        if (!newlyStopped)
        {
            std::cout << "SCHEDULER_PREEMPTION_DEFERRED"
                      << ",candidate_experiment_id="
                      << candidate.experimentId
                      << ",victim_experiment_id="
                      << victimExperimentId
                      << ",worker_attempt_id=" << workerAttemptId
                      << ",phase=" << phase
                      << ",reason=" << signal.result
                      << ",detail=" << signal.detail
                      << std::endl;
            transaction.commit();
            return false;
        }
        // The worker is now quiescent. Recheck the exact-attempt result after
        // SIGSTOP so a completion persisted during the preemption decision
        // wins without rewriting the experiment as preempted.
        if (inferAttemptHasAuthoritativeResult())
        {
            const auto resumed =
                EA::GlobalExperimentControl::ResumeWorker(worker, *processes);
            std::cout << "SCHEDULER_PREEMPTION_DEFERRED"
                      << ",candidate_experiment_id="
                      << candidate.experimentId
                      << ",victim_experiment_id=" << victimExperimentId
                      << ",worker_attempt_id=" << workerAttemptId
                      << ",phase=infer"
                      << ",reason=authoritative_inference_result_observed"
                      << ",worker_resume_result=" << resumed.result
                      << std::endl;
            transaction.commit();
            return false;
        }
        if (SchedulerAuthorityTestFailpointEnabled(
                "preemption_after_sigstop_before_db"))
        {
            throw std::runtime_error(
                "injected_preemption_db_failure_after_sigstop");
        }

        const pqxx::result stopped = transaction.exec(
            "UPDATE experiment_scheduler_worker_attempt SET "
            "lifecycle_state='stopped',last_observed_at=clock_timestamp(),"
            "observed_by_scheduler_invocation_id=$1,signal_number=$2,"
            "reconciliation_result='scheduler_priority_preemption',"
            "diagnostic='verified_process_group_stopped_for_higher_priority' "
            "WHERE worker_attempt_id=$3 "
            "AND lifecycle_state IN ('spawned','running','observed') "
            "RETURNING worker_attempt_id;",
            pqxx::params{
                options.schedulerAuthority.schedulerInvocationId,
                SIGSTOP,
                workerAttemptId});
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            stopped, "preempt_exact_worker_attempt");
        const pqxx::result queued = transaction.exec(
            "UPDATE experiment SET status='pending',resume_requested=true,"
            "scheduler_resume_origin='preemption',"
            "worker_control_state='paused',updated_at=clock_timestamp() "
            "WHERE experiment_id=$1 AND status='running' AND phase=$2 "
            "AND scheduler_priority=$3 "
            "AND active_scheduler_worker_attempt_id=$4 "
            "AND cancellation_request_id IS NULL "
            "AND cancel_after_checkpoint_epoch IS NULL "
            "AND stop_after_checkpoint_epoch IS NULL "
            "AND worker_global_pause_request_id IS NULL "
            "RETURNING experiment_id;",
            pqxx::params{
                victimExperimentId,
                phase,
                victimPriority,
                workerAttemptId});
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            queued, "preempt_exact_experiment_lifecycle");
        commitAttempted = true;
        transaction.commit();
    }
    catch (...)
    {
        if (!commitAttempted)
        {
            transaction.abort();
            if (newlyStopped)
                CompensateSchedulerPreemptionRollback(
                    worker, *processes);
        }
        else
        {
            std::cerr << "SCHEDULER_PREEMPTION_COMMIT_OUTCOME_AMBIGUOUS"
                      << ",victim_experiment_id="
                      << victimExperimentId
                      << ",worker_attempt_id=" << workerAttemptId
                      << ",compensation=withheld"
                      << std::endl;
        }
        throw;
    }

    std::cout << "SCHEDULER_PRIORITY_PREEMPTED"
              << ",candidate_experiment_id=" << candidate.experimentId
              << ",candidate_priority=" << candidate.schedulerPriority
              << ",victim_experiment_id=" << victimExperimentId
              << ",victim_priority=" << victimPriority
              << ",worker_attempt_id=" << workerAttemptId
              << ",phase=" << phase
              << ",resume_origin=preemption"
              << ",victim_order=lowest_priority_then_newest_worker_started_at_then_experiment_id"
              << std::endl;
    return true;
}

StoppedWorkerAdmissionResult AdmitStoppedExperimentWorker(
    const SchedulerOptions& options,
    const ExperimentRow& experiment,
    const std::string& phase,
    int maximumCapacity)
{
    if (!experiment.activeWorkerAttemptId ||
        !experiment.resumeRequested)
        return StoppedWorkerAdmissionResult::NotApplicable;

    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    if (!SchedulerLaunchAllowed(transaction, phase))
    {
        transaction.commit();
        return StoppedWorkerAdmissionResult::DeferredUnsafe;
    }
    if (!services.admission.hasCapacity(phase, maximumCapacity))
    {
        transaction.commit();
        return StoppedWorkerAdmissionResult::DeferredNoCapacity;
    }

    const pqxx::result lifecycle = transaction.exec_params(
        "SELECT status,phase,resume_requested,scheduler_resume_origin,"
        "active_scheduler_worker_attempt_id "
        "FROM experiment WHERE experiment_id=$1;",
        experiment.experimentId);
    if (lifecycle.size() != 1 ||
        lifecycle[0][0].as<std::string>() != "pending" ||
        lifecycle[0][1].as<std::string>() != phase ||
        !lifecycle[0][2].as<bool>() ||
        lifecycle[0][3].as<std::string>() !=
            experiment.schedulerResumeOrigin ||
        lifecycle[0][4].is_null() ||
        lifecycle[0][4].as<long long>() !=
            *experiment.activeWorkerAttemptId)
    {
        transaction.commit();
        return StoppedWorkerAdmissionResult::DeferredUnsafe;
    }

    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = *experiment.activeWorkerAttemptId;
    expected.experimentId = experiment.experimentId;
    expected.workerKind = "experiment";
    expected.lifecyclePhase = phase;
    expected.capacityClass = phase;
    expected.requiredLifecycleState = "stopped";
    expected.requireSignalable = true;
    expected.requireCompleteProcessIdentity = true;
    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
    if (!exact)
    {
        std::cout << "SCHEDULER_STOPPED_WORKER_ADMISSION_DEFERRED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",worker_attempt_id="
                  << *experiment.activeWorkerAttemptId
                  << ",phase=" << phase
                  << ",reason=exact_attempt_verification_failed"
                  << std::endl;
        transaction.commit();
        return StoppedWorkerAdmissionResult::DeferredUnsafe;
    }

    EA::GlobalExperimentControl::ManagedWorker worker;
    worker.workerAttemptId = exact->workerAttemptId;
    worker.workerKind = exact->workerKind;
    worker.capacityClass = exact->capacityClass;
    worker.attemptLifecycleState = exact->lifecycleState;
    worker.launchAttemptIdentity = exact->launchAttemptIdentity;
    worker.experimentId = exact->experimentId;
    worker.phase = exact->lifecyclePhase;
    worker.lifecycleStatus = exact->lifecycleStatus;
    worker.pid = *exact->workerPid;
    worker.processGroupId = exact->processGroupId;
    worker.executable = exact->canonicalExecutablePath;
    worker.commandLine = exact->commandLine;
    worker.processStartIdentity = exact->processStartIdentity;

    auto processes =
        EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const EA::GlobalExperimentControl::SignalOutcome signal =
        EA::GlobalExperimentControl::
            ResumeStoppedWorkerForSchedulerAdmission(worker, *processes);
    if (signal.identity ==
        EA::GlobalExperimentControl::IdentityResult::ProcessMissing)
    {
        auto missingPlan =
            EA::SchedulerCore::PlanMissingStoppedWorker(
                phase,
                "pending",
                true,
                experiment.schedulerResumeOrigin,
                false);
        const bool preemptedTrainRestart =
            missingPlan.disposition ==
            EA::SchedulerCore::MissingStoppedWorkerDisposition::
                FailPreemptedTrainWithoutCheckpoint;
        std::optional<QueueResumeMeta> restartCheckpoint;
        if (preemptedTrainRestart)
        {
            const TrainingCheckpointSelection selection =
                SelectUsableTrainingCheckpoint(
                    transaction, experiment, std::nullopt, false);
            restartCheckpoint = selection.checkpoint;
            missingPlan =
                EA::SchedulerCore::PlanMissingStoppedWorker(
                    phase,
                    "pending",
                    true,
                    experiment.schedulerResumeOrigin,
                    restartCheckpoint.has_value());
        }
        pqxx::result retired = transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt SET "
            "lifecycle_state='abandoned',completed_at=clock_timestamp(),"
            "last_observed_at=clock_timestamp(),"
            "observed_by_scheduler_invocation_id=$1,"
            "reconciled_at=clock_timestamp(),"
            "reconciled_by_scheduler_invocation_id=$1,"
            "reconciliation_result='stopped_process_missing',"
            "diagnostic='resume_priority_preserved_for_restart' "
            "WHERE worker_attempt_id=$2 AND lifecycle_state='stopped' "
            "RETURNING worker_attempt_id;",
            options.schedulerAuthority.schedulerInvocationId,
            exact->workerAttemptId);
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            retired, "retire_missing_stopped_worker_attempt");
        pqxx::result detached;
        if (missingPlan.disposition ==
            EA::SchedulerCore::MissingStoppedWorkerDisposition::
                FailPreemptedTrainWithoutCheckpoint)
        {
            detached = transaction.exec(
                "UPDATE experiment SET status='failed',"
                "resume_requested=false,scheduler_resume_origin='none',"
                "worker_pid=NULL,worker_process_group_id=NULL,"
                "worker_process_start_identity=NULL,worker_executable=NULL,"
                "worker_command_line=NULL,worker_control_state='running',"
                "active_scheduler_worker_attempt_id=NULL,exit_code=-1,"
                "error_message='preempted_worker_missing_no_valid_checkpoint',"
                "completed_at=clock_timestamp(),updated_at=clock_timestamp() "
                "WHERE experiment_id=$1 AND status='pending' AND phase='train' "
                "AND resume_requested=true "
                "AND scheduler_resume_origin='preemption' "
                "AND active_scheduler_worker_attempt_id=$2 "
                "RETURNING experiment_id;",
                pqxx::params{
                    experiment.experimentId,
                    exact->workerAttemptId});
        }
        else if (missingPlan.disposition ==
                 EA::SchedulerCore::MissingStoppedWorkerDisposition::
                     RestartPreemptedTrainFromCheckpoint)
        {
            detached = transaction.exec(
                "UPDATE experiment SET worker_pid=NULL,"
                "worker_process_group_id=NULL,"
                "worker_process_start_identity=NULL,worker_executable=NULL,"
                "worker_command_line=NULL,worker_control_state='running',"
                "active_scheduler_worker_attempt_id=NULL,"
                "last_model_id=$1,"
                "resume_model_id=CASE "
                "WHEN continuation_source_model_id IS NULL THEN $1 "
                "ELSE resume_model_id END,"
                "current_operation='train',updated_at=clock_timestamp() "
                "WHERE experiment_id=$2 AND status='pending' "
                "AND phase='train' AND resume_requested=true "
                "AND scheduler_resume_origin='preemption' "
                "AND active_scheduler_worker_attempt_id=$3 "
                "RETURNING experiment_id;",
                pqxx::params{
                    restartCheckpoint->modelId,
                    experiment.experimentId,
                    exact->workerAttemptId});
        }
        else
        {
            detached = transaction.exec_params(
                "UPDATE experiment SET worker_pid=NULL,"
                "worker_process_group_id=NULL,"
                "worker_process_start_identity=NULL,worker_executable=NULL,"
                "worker_command_line=NULL,worker_control_state='running',"
                "active_scheduler_worker_attempt_id=NULL,"
                "updated_at=clock_timestamp() "
                "WHERE experiment_id=$1 AND status='pending' "
                "AND phase=$2 AND resume_requested=true "
                "AND active_scheduler_worker_attempt_id=$3 "
                "RETURNING experiment_id;",
                experiment.experimentId,
                phase,
                exact->workerAttemptId);
        }
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            detached, "detach_missing_stopped_worker_attempt");
        transaction.commit();
        std::cout << "SCHEDULER_STOPPED_WORKER_MISSING"
                  << ",experiment_id=" << experiment.experimentId
                  << ",worker_attempt_id=" << exact->workerAttemptId
                  << ",phase=" << phase
                  << ",resume_requested=true"
                  << ",resume_origin="
                  << experiment.schedulerResumeOrigin
                  << ",fallback="
                  << (preemptedTrainRestart
                          ? missingPlan.eventResult
                          : "checkpoint_restart")
                  << std::endl;
        // A preempted train restart may have promoted an earlier checkpoint.
        // Defer it to the next poll so command construction reloads that
        // authoritative identity instead of using this turn's stale snapshot.
        if (preemptedTrainRestart)
        {
            return StoppedWorkerAdmissionResult::DeferredUnsafe;
        }
        return StoppedWorkerAdmissionResult::MissingProcessFallbackReady;
    }
    if (!signal.success)
    {
        transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt SET "
            "lifecycle_state='identity_ambiguous',"
            "last_observed_at=clock_timestamp(),"
            "observed_by_scheduler_invocation_id=$1,"
            "reconciliation_result='stopped_resume_rejected',"
            "diagnostic=$2 WHERE worker_attempt_id=$3 "
            "AND lifecycle_state='stopped';",
            options.schedulerAuthority.schedulerInvocationId,
            signal.detail.empty() ? signal.result : signal.detail,
            exact->workerAttemptId);
        transaction.commit();
        std::cout << "SCHEDULER_STOPPED_WORKER_ADMISSION_DEFERRED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",worker_attempt_id=" << exact->workerAttemptId
                  << ",phase=" << phase
                  << ",reason=" << signal.result
                  << std::endl;
        return StoppedWorkerAdmissionResult::DeferredUnsafe;
    }

    pqxx::result activated = transaction.exec_params(
        "UPDATE experiment_scheduler_worker_attempt SET "
        "lifecycle_state='running',last_observed_at=clock_timestamp(),"
        "observed_by_scheduler_invocation_id=$1,"
        "reconciliation_result='stopped_worker_admitted',"
        "diagnostic='capacity_acquired_before_sigcont' "
        "WHERE worker_attempt_id=$2 AND lifecycle_state='stopped' "
        "RETURNING worker_attempt_id;",
        options.schedulerAuthority.schedulerInvocationId,
        exact->workerAttemptId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        activated, "activate_admitted_stopped_worker_attempt");
    pqxx::result running = transaction.exec_params(
        "UPDATE experiment SET status='running',resume_requested=false,"
        "scheduler_resume_origin='none',worker_control_state='running',"
        "worker_global_pause_request_id=NULL,"
        "updated_at=clock_timestamp() WHERE experiment_id=$1 "
        "AND status='pending' AND phase=$2 AND resume_requested=true "
        "AND active_scheduler_worker_attempt_id=$3 "
        "RETURNING experiment_id;",
        experiment.experimentId,
        phase,
        exact->workerAttemptId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        running, "activate_admitted_stopped_worker_lifecycle");
    transaction.commit();
    std::cout << "SCHEDULER_STOPPED_WORKER_ADMITTED"
              << ",experiment_id=" << experiment.experimentId
              << ",worker_attempt_id=" << exact->workerAttemptId
              << ",phase=" << phase
              << ",signal=" << signal.result
              << ",resume_requested=false"
              << std::endl;
    return StoppedWorkerAdmissionResult::Admitted;
}

std::optional<ReservedWorkerAttempt>
ReserveExperimentWorkerAttempt(
    const SchedulerOptions& options,
    const ExperimentRow& experiment,
    const std::string& phase,
    const std::string& logPath,
    int maximumCapacity,
    bool cancellationOnly = false)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    if (!SchedulerLaunchAllowed(
            transaction,
            phase,
            false,
            cancellationOnly))
    {
        transaction.commit();
        return std::nullopt;
    }

    std::string selectedWorkerExecutable =
        phase == "analyze"
            ? options.analyzeWorkerExecutablePath
            : options.currentWorkerExecutablePath;
    EA::Scheduler::SemanticAdmissionDecision semanticAdmission;
    if (phase == "infer")
    {
        const auto selection = LoadInferenceWorkerSelection(
            transaction, options, experiment.experimentId);
        semanticAdmission = {selection.selected, selection.diagnostic};
        if (selection.selected)
            selectedWorkerExecutable = selection.canonicalExecutablePath;
    }
    else
    {
        semanticAdmission = LoadSemanticWorkerAdmission(
            transaction, experiment.experimentId, phase);
    }
    if (!semanticAdmission.admissible)
    {
        std::cout << "SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",phase=" << phase
                  << ",diagnostic=" << semanticAdmission.diagnostic
                  << ",capacity_consumed=0,child_launched=0,"
                     "experiment_status_changed=false"
                  << std::endl;
        transaction.commit();
        return std::nullopt;
    }

    if (phase != "analyze")
    {
        const auto runtime = SemanticWorkerRegistryFor(options)
            .validateRuntimeForExecutable(selectedWorkerExecutable);
        if (!runtime.ready)
        {
            std::cout << "SCHEDULER_SEMANTIC_WORKER_RUNTIME_UNAVAILABLE"
                      << ",experiment_id=" << experiment.experimentId
                      << ",phase=" << phase
                      << ",worker_executable=" << selectedWorkerExecutable
                      << ",diagnostic=" << runtime.diagnostic
                      << ",capacity_consumed=0,child_launched=0,"
                         "experiment_status_changed=false"
                      << std::endl;
            transaction.commit();
            return std::nullopt;
        }
    }

    if (!services.admission.hasCapacity(phase, maximumCapacity))
    {
        transaction.commit();
        return std::nullopt;
    }

    const auto& registry = SemanticWorkerRegistryFor(options);
    const auto* artifact = phase == "analyze" ? nullptr :
        registry.findByCanonicalExecutable(selectedWorkerExecutable);
    if (phase != "analyze" && artifact == nullptr)
        throw std::runtime_error("selected_semantic_worker_not_in_registry");
    if ((phase == "train" && artifact->role != EA::Scheduler::SemanticWorkerRole::Train) ||
        (phase == "infer" && artifact->role != EA::Scheduler::SemanticWorkerRole::Infer))
        throw std::runtime_error("selected_semantic_worker_role_phase_mismatch");
    EA::SchedulerCore::WorkerAttemptLifecycleService lifecycle{
        services.repository,
        {options.schedulerAuthority.schedulerInvocationId,
         options.schedulerAuthority.fencingToken,
         selectedWorkerExecutable,
         artifact ? artifact->semanticLayoutVersion : 0,
         artifact ? artifact->modelInputWidth : 0,
         artifact ? (artifact->role == EA::Scheduler::SemanticWorkerRole::Train ? "train" : "infer") : "",
         artifact ? artifact->sourceCommit : "",
         artifact ? artifact->sha256 : "",
         artifact ? artifact->runtimeIdentity : "",
         artifact ? artifact->canonicalManifestPath : ""}};
    auto attempt = lifecycle.reserveExperiment({
        experiment.experimentId,
        phase,
        logPath,
        EA::SchedulerCore::GenerateSchedulerIdentityNonce(),
        cancellationOnly});
    transaction.commit();
    return attempt;
}

std::optional<ReservedWorkerAttempt>
ReserveCheckpointWorkerAttempt(
    const SchedulerOptions& options,
    const CheckpointEvalRow& eval,
    const std::string& logPath,
    int maximumCapacity)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    if (!SchedulerLaunchAllowed(
            transaction, "checkpoint_infer", true, false))
    {
        transaction.commit();
        return std::nullopt;
    }
    const auto selection = LoadInferenceWorkerSelection(
        transaction, options, eval.experiment.experimentId);
    if (!selection.selected)
    {
        std::cout << "SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE"
                  << ",experiment_id=" << eval.experiment.experimentId
                  << ",checkpoint_eval_id=" << eval.checkpointEvalId
                  << ",phase=infer,diagnostic="
                  << selection.diagnostic
                  << ",capacity_consumed=0,child_launched=0,"
                     "experiment_status_changed=false"
                  << std::endl;
        transaction.commit();
        return std::nullopt;
    }
    const auto runtime = SemanticWorkerRegistryFor(options)
        .validateRuntimeForExecutable(selection.canonicalExecutablePath);
    if (!runtime.ready)
    {
        std::cout << "SCHEDULER_SEMANTIC_WORKER_RUNTIME_UNAVAILABLE"
                  << ",experiment_id=" << eval.experiment.experimentId
                  << ",checkpoint_eval_id=" << eval.checkpointEvalId
                  << ",phase=infer,worker_executable="
                  << selection.canonicalExecutablePath
                  << ",diagnostic=" << runtime.diagnostic
                  << ",capacity_consumed=0,child_launched=0,"
                     "experiment_status_changed=false"
                  << std::endl;
        transaction.commit();
        return std::nullopt;
    }
    if (!services.admission.hasCapacity("infer", maximumCapacity))
    {
        transaction.commit();
        return std::nullopt;
    }

    const auto& registry = SemanticWorkerRegistryFor(options);
    const auto* artifact = registry.findByCanonicalExecutable(
        selection.canonicalExecutablePath);
    if (artifact == nullptr)
        throw std::runtime_error("selected_semantic_worker_not_in_registry");
    if (artifact->role != EA::Scheduler::SemanticWorkerRole::Infer)
        throw std::runtime_error("selected_semantic_worker_role_phase_mismatch");
    EA::SchedulerCore::WorkerAttemptLifecycleService lifecycle{
        services.repository,
        {options.schedulerAuthority.schedulerInvocationId,
         options.schedulerAuthority.fencingToken,
         selection.canonicalExecutablePath,
         artifact->semanticLayoutVersion, artifact->modelInputWidth,
         artifact->role == EA::Scheduler::SemanticWorkerRole::Train ? "train" : "infer",
         artifact->sourceCommit, artifact->sha256, artifact->runtimeIdentity,
         artifact->canonicalManifestPath}};
    auto attempt = lifecycle.reserveCheckpoint({
        eval.experiment.experimentId,
        eval.checkpointEvalId,
        logPath,
        EA::SchedulerCore::GenerateSchedulerIdentityNonce()});
    transaction.commit();
    return attempt;
}

[[noreturn]] void ThrowSchedulerChildLaunchError(
    long long experimentId,
    const std::string& phase,
    const std::string& commandLine,
    int errorNumber,
    const std::string& context)
{
    std::cout << SchedulerChildLaunchFailureDiagnostic(
                     experimentId,
                     phase,
                     errorNumber,
                     commandLine)
              << std::endl;
    throw SchedulerChildLaunchError(
        errorNumber,
        context + ": " + std::strerror(errorNumber));
}

void MarkReservedWorkerAttemptLaunchFailed(
    const SchedulerOptions& options,
    const ReservedWorkerAttempt& attempt,
    const std::string& diagnostic,
    int exitCode = 127)
{
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        RequireAndRefreshSchedulerAuthority(transaction, options);
        EA::SchedulerCore::PostgresSchedulerRepository repository{
            transaction};
        EA::SchedulerCore::WorkerAttemptLifecycleService lifecycle{
            repository,
            {options.schedulerAuthority.schedulerInvocationId,
             options.schedulerAuthority.fencingToken,
             attempt.canonicalExecutablePath}};
        lifecycle.recordLaunchFailure({attempt, exitCode, diagnostic});
        transaction.commit();
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_LAUNCH_FAILURE_PERSIST_FAILED"
                  << ",worker_attempt_id="
                  << attempt.workerAttemptId
                  << ",error=" << error.what()
                  << std::endl;
    }
}

void PersistSpawnedWorkerAttempt(
    const SchedulerOptions& options,
    const ReservedWorkerAttempt& attempt,
    pid_t pid,
    const std::string& processStartIdentity,
    const std::string& commandLine,
    bool requireSchedulerAuthority = true)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    if (requireSchedulerAuthority)
        RequireAndRefreshSchedulerAuthority(transaction, options);
    EA::SchedulerCore::PostgresSchedulerRepository repository{transaction};
    EA::SchedulerCore::WorkerAttemptLifecycleService lifecycle{
        repository,
        {options.schedulerAuthority.schedulerInvocationId,
         options.schedulerAuthority.fencingToken,
         attempt.canonicalExecutablePath}};
    lifecycle.recordSpawned({
        attempt,
        static_cast<int>(pid),
        processStartIdentity,
        commandLine,
        requireSchedulerAuthority});
    transaction.commit();
}

pid_t LaunchReservedChildProcess(
    const SchedulerOptions& options,
    const ReservedWorkerAttempt& attempt,
    std::vector<std::string> argv,
    const std::optional<long long>& expectedModelId = std::nullopt)
{
    bool executableIdentityValid = false;
    if (!argv.empty() && !attempt.canonicalExecutablePath.empty() &&
        attempt.canonicalExecutablePath.front() == '/' &&
        argv.front() == attempt.canonicalExecutablePath)
    {
        try
        {
            executableIdentityValid =
                EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
                    argv.front(), "reserved worker executable") ==
                attempt.canonicalExecutablePath;
        }
        catch (const std::invalid_argument&)
        {
            executableIdentityValid = false;
        }
    }
    if (!executableIdentityValid)
    {
        MarkReservedWorkerAttemptLaunchFailed(
            options,
            attempt,
            "canonical_executable_path_required",
            127);
        ThrowSchedulerChildLaunchError(
            attempt.experimentId,
            attempt.phase,
            CommandForDisplay(argv),
            EINVAL,
            "canonical executable path required");
    }
    AddCliOption(
        argv,
        "--scheduler-worker-attempt-id",
        std::to_string(attempt.workerAttemptId));
    const std::string commandLine =
        CommandForProcessObservation(argv);

    const int fd = ::open(
        attempt.logPath.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC,
        0644);
    if (fd < 0)
    {
        const int errorNumber = errno;
        MarkReservedWorkerAttemptLaunchFailed(
            options,
            attempt,
            SchedulerLaunchFailureError(errorNumber),
            127);
        ThrowSchedulerChildLaunchError(
            attempt.experimentId,
            attempt.phase,
            commandLine,
            errorNumber,
            "failed to open scheduler child log");
    }

    int gate[2] = {-1, -1};
    if (::pipe(gate) != 0)
    {
        const int errorNumber = errno;
        ::close(fd);
        MarkReservedWorkerAttemptLaunchFailed(
            options,
            attempt,
            SchedulerLaunchFailureError(errorNumber),
            127);
        ThrowSchedulerChildLaunchError(
            attempt.experimentId,
            attempt.phase,
            commandLine,
            errorNumber,
            "failed to create scheduler child launch gate");
    }
    (void)::fcntl(gate[0], F_SETFD, FD_CLOEXEC);
    (void)::fcntl(gate[1], F_SETFD, FD_CLOEXEC);

    std::vector<char*> childArgv;
    childArgv.reserve(argv.size() + 1);
    for (const auto& argument : argv)
        childArgv.push_back(const_cast<char*>(argument.c_str()));
    childArgv.push_back(nullptr);
    const char* phaseData = attempt.phase.data();
    const size_t phaseLength = attempt.phase.size();

    const pid_t pid = ::fork();
    if (pid < 0)
    {
        const int errorNumber = errno;
        ::close(gate[0]);
        ::close(gate[1]);
        ::close(fd);
        MarkReservedWorkerAttemptLaunchFailed(
            options,
            attempt,
            SchedulerLaunchFailureError(errorNumber),
            127);
        ThrowSchedulerChildLaunchError(
            attempt.experimentId,
            attempt.phase,
            commandLine,
            errorNumber,
            "fork failed");
    }

    if (pid == 0)
    {
        ::close(gate[1]);
        if (::setsid() < 0)
        {
            const int childError = errno;
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
            ::close(fd);
            WriteSchedulerChildExecFailureDiagnostic(
                STDERR_FILENO,
                attempt.experimentId,
                phaseData,
                phaseLength,
                childError);
            _exit(127);
        }
        ::signal(SIGHUP, SIG_IGN);
        ::dup2(fd, STDOUT_FILENO);
        ::dup2(fd, STDERR_FILENO);
        ::close(fd);

        try
        {
            const std::string childProcessStartIdentity =
                RequireProcessStartIdentity(::getpid());
            // The child may persist only the exact already-reserved attempt.
            // It cannot refresh the scheduler lease or claim/launch work.
            PersistSpawnedWorkerAttempt(
                options,
                attempt,
                ::getpid(),
                childProcessStartIdentity,
                commandLine,
                false);
        }
        catch (...)
        {
            WriteSchedulerChildExecFailureDiagnostic(
                STDERR_FILENO,
                attempt.experimentId,
                phaseData,
                phaseLength,
                EIO);
            _exit(126);
        }

        char permission = '\0';
        const ssize_t readResult =
            ::read(gate[0], &permission, 1);
        ::close(gate[0]);
        if (readResult != 1 || permission != 'G')
        {
            WriteSchedulerChildExecFailureDiagnostic(
                STDERR_FILENO,
                attempt.experimentId,
                phaseData,
                phaseLength,
                ECANCELED);
            _exit(126);
        }

        ::signal(SIGPIPE, SIG_DFL);
        ::execv(argv.front().c_str(), childArgv.data());
        const int childError = errno;
        WriteSchedulerChildExecFailureDiagnostic(
            STDERR_FILENO,
            attempt.experimentId,
            phaseData,
            phaseLength,
            childError);
        _exit(127);
    }

    ::close(gate[0]);
    ::close(fd);
    if (SchedulerAuthorityTestFailpointEnabled(
            "scheduler_parent_crash_after_child_registration"))
    {
        bool childRegistered = false;
        for (int observation = 0; observation < 500; ++observation)
        {
            pqxx::connection testConnection{
                LstmDbConnectionString()};
            pqxx::read_transaction testTransaction{
                testConnection};
            const pqxx::result registered =
                testTransaction.exec_params(
                    "SELECT 1 FROM "
                    "experiment_scheduler_worker_attempt "
                    "WHERE worker_attempt_id=$1 "
                    "AND lifecycle_state='spawned' "
                    "AND worker_pid=$2 "
                    "AND worker_process_group_id=$2 "
                    "AND worker_process_start_identity IS NOT NULL "
                    "AND canonical_executable_path=$3 "
                    "AND command_line=$4;",
                    attempt.workerAttemptId,
                    static_cast<int>(pid),
                    attempt.canonicalExecutablePath,
                    commandLine);
            if (registered.size() == 1)
            {
                childRegistered = true;
                break;
            }
            ::usleep(10000);
        }
        if (!childRegistered)
            throw std::runtime_error(
                "test_child_self_registration_not_observed");
        std::cout
            << "SCHEDULER_TEST_PARENT_CRASH_AFTER_CHILD_REGISTRATION"
            << ",worker_attempt_id=" << attempt.workerAttemptId
            << ",worker_pid=" << pid
            << std::endl;
        std::cout.flush();
        std::cerr.flush();
        ::_exit(87);
    }
    try
    {
        pid_t processGroupId = -1;
        for (int attemptNumber = 0;
             attemptNumber < 1000;
             ++attemptNumber)
        {
            processGroupId = ::getpgid(pid);
            if (processGroupId == pid)
                break;
            if (processGroupId < 0 && errno == ESRCH)
                break;
            ::usleep(1000);
        }
        if (processGroupId < 0 || processGroupId != pid)
            throw std::runtime_error(
                "scheduler_child_process_group_identity_mismatch");
        const std::string processStartIdentity =
            RequireProcessStartIdentity(pid);
        PersistSpawnedWorkerAttempt(
            options,
            attempt,
            pid,
            processStartIdentity,
            commandLine);
        const char permission = 'G';
        if (::write(gate[1], &permission, 1) != 1)
            throw std::runtime_error(
                "scheduler_child_launch_gate_write_failed");
        ::close(gate[1]);
    }
    catch (const std::exception& error)
    {
        ::close(gate[1]);
        (void)::waitpid(pid, nullptr, 0);
        MarkReservedWorkerAttemptLaunchFailed(
            options,
            attempt,
            std::string{"spawn_persistence_failed:"} + error.what(),
            126);
        throw;
    }

    SchedulerOwnedChild child;
    child.pid = pid;
    child.workerAttemptId = attempt.workerAttemptId;
    child.experimentId = attempt.experimentId;
    child.phase =
        attempt.workerKind == "checkpoint_infer"
            ? "checkpoint_infer"
            : attempt.phase;
    child.operation = attempt.phase;
    child.commandLine = commandLine;
    child.expectedModelId = expectedModelId;
    child.checkpointEvalId = attempt.checkpointEvalId;
    child.logPath = attempt.logPath;
    child.launchedAt = EA::RunMetadata::CurrentUtcTimestamp();
    child.launchedEpoch =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    SchedulerRuntime(options).ownedChildren.emplace(pid, std::move(child));
    std::cout << "SCHEDULER_CHILD_LAUNCHED"
              << ",worker_attempt_id=" << attempt.workerAttemptId
              << ",launch_attempt_identity="
              << attempt.launchAttemptIdentity
              << ",experiment_id=" << attempt.experimentId
              << ",worker_pid=" << pid
              << ",phase=" << attempt.phase
              << ",worker_kind=" << attempt.workerKind
              << ",command_line=" << commandLine
              << ",log_path=" << attempt.logPath
              << std::endl;
    return pid;
}

std::vector<std::string> BuildTrainCommand(const SchedulerOptions& options,
                                                  const ExperimentRow& experiment)
{
    // ResolvePersisted already rejected unknown identities. Re-canonicalize
    // here so only an exact supported objective can reach a training child.
    (void)EA::TrainingObjective::ParseSupportedCanonicalText(
        EA::TrainingObjective::CanonicalText(
            experiment.trainingObjective));
    std::vector<std::string> argv;
    argv.push_back(options.currentWorkerExecutablePath);
    AddCliFlag(argv, "--train");
    AddCliOption(argv, "--log-level", "summary");
    AddCliOption(argv, "--checkpoint-every", std::to_string(experiment.checkpointInterval));
    AddCliOption(argv, "--new-model-name", BaseModelName(experiment));
    AddCliOption(argv, "--donchian20-mode",
                 Donchian20ModeText(experiment.donchian20Mode));
    AddCliOption(argv, "--feature-warmup-scope",
                 EA::FeatureWarmupScopeText(experiment.featureWarmupScope));
    AddCliOption(argv, "--donchian-lookback",
                 std::to_string(experiment.donchianLookback));
    if (!experiment.resumeModelId && !experiment.lastModelId)
        AddCliOption(argv, "--fresh-initialization-seed",
                     std::to_string(experiment.freshInitializationSeed));
    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
    AddCliOption(
        argv,
        "--training-objective",
        experiment.trainingObjective.objectiveIdentifier);
    AddLstmProfileOptions(argv, options);

    const std::optional<long long> resumeFrom =
        experiment.lastModelId.has_value()
            ? experiment.lastModelId
            : experiment.resumeModelId;
    if (resumeFrom.has_value())
    {
        AddCliOption(argv, "--resume-model-id", std::to_string(*resumeFrom));
        if (experiment.resumeExpandInputWidth)
            AddCliFlag(argv, "--resume-expand-input-width");
        AddCliOption(argv, "--target-epochs", std::to_string(experiment.targetEpochs));
        return argv;
    }

    AddCliOption(argv, "--symbol", experiment.symbol);
    AddCliOption(argv, "--prediction-horizon", std::to_string(experiment.predictionHorizon));
    AddCliOption(argv, "--threshold", FormatDouble(experiment.cNextThreshold));
    AddCliOption(argv, "--epochs", std::to_string(experiment.targetEpochs));
    if (experiment.coreLrMult.has_value())
        AddCliOption(argv, "--core-lr-mult", FormatDouble(*experiment.coreLrMult));
    if (experiment.headLrMult.has_value())
        AddCliOption(argv, "--head-weight-lr-mult", FormatDouble(*experiment.headLrMult));
    AddCliPositional(argv, experiment.trainStart.substr(0, 10));
    AddCliPositional(argv, experiment.trainEnd.substr(0, 10));
    return argv;
}

std::vector<std::string> BuildInferCommand(
    const SchedulerOptions& options,
    const ExperimentRow& experiment,
    const std::string& selectedWorkerExecutable)
{
    if (!experiment.lastModelId.has_value())
        throw std::runtime_error("infer phase has no last_model_id");
    if (!experiment.inferStart.has_value() || !experiment.inferEnd.has_value())
        throw std::runtime_error("infer phase has no infer date range");

    std::vector<std::string> argv;
    argv.push_back(selectedWorkerExecutable);
    AddCliFlag(argv, "--infer");
    AddCliOption(argv, "--model", std::to_string(*experiment.lastModelId));
    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
    AddCliOption(argv, "--donchian20-mode",
                 Donchian20ModeText(experiment.donchian20Mode));
    AddCliOption(argv, "--feature-warmup-scope",
                 EA::FeatureWarmupScopeText(experiment.featureWarmupScope));
    AddCliOption(argv, "--donchian-lookback",
                 std::to_string(experiment.donchianLookback));
    AddCliOption(argv, "--log-level", "summary");
    AddLstmProfileOptions(argv, options);
    AddCliPositional(argv, experiment.inferStart->substr(0, 10));
    AddCliPositional(argv, experiment.inferEnd->substr(0, 10));
    return argv;
}

std::vector<std::string> BuildAnalyzeCommand(const SchedulerOptions& options,
                                                    const ExperimentRow& experiment)
{
    std::vector<std::string> argv;
    argv.push_back(options.analyzeWorkerExecutablePath);
    AddCliOption(argv, "--analyze-experiment", std::to_string(experiment.experimentId));
    if (options.autoGenerateReports)
        AddCliFlag(argv, "--auto-generate-reports");
    if (options.experimentReportDir != "experiment_reports")
        AddCliOption(argv, "--experiment-report-dir", options.experimentReportDir);
    return argv;
}

std::string CheckpointEvalLogPathFor(const SchedulerOptions& options,
                                     const CheckpointEvalRow& eval,
                                     const std::string& phase)
{
    std::filesystem::path dir{options.schedulerLogDir};
    std::ostringstream name;
    name << "checkpoint_eval_"
         << eval.experiment.experimentId
         << "_epoch" << eval.checkpointEpoch
         << "_model" << eval.checkpointModelId
         << "_" << phase << ".log";
    return (dir / name.str()).string();
}

std::vector<std::string> BuildCheckpointEvalInferCommand(
    const SchedulerOptions& options,
    const CheckpointEvalRow& eval,
    const std::string& selectedWorkerExecutable)
{
    if (!eval.experiment.inferStart.has_value() || !eval.experiment.inferEnd.has_value())
        throw std::runtime_error("checkpoint eval infer has no infer date range");

    std::vector<std::string> argv;
    argv.push_back(selectedWorkerExecutable);
    AddCliFlag(argv, "--infer");
    AddCliOption(argv, "--model", std::to_string(eval.checkpointModelId));
    AddCliOption(argv, "--scheduler-checkpoint-eval-id", std::to_string(eval.checkpointEvalId));
    AddCliOption(argv, "--donchian20-mode",
                 Donchian20ModeText(eval.experiment.donchian20Mode));
    AddCliOption(argv, "--feature-warmup-scope",
                 EA::FeatureWarmupScopeText(eval.experiment.featureWarmupScope));
    AddCliOption(argv, "--donchian-lookback",
                 std::to_string(eval.experiment.donchianLookback));
    AddCliOption(argv, "--log-level", "summary");
    AddLstmProfileOptions(argv, options);
    AddCliPositional(argv, eval.experiment.inferStart->substr(0, 10));
    AddCliPositional(argv, eval.experiment.inferEnd->substr(0, 10));
    return argv;
}

bool CheckpointEvalTableExists(pqxx::work& w)
{
    return TableExists(w, "experiment_checkpoint_eval");
}

void EnqueueCheckpointEvalRows(pqxx::work& w)
{
    const bool hasCheckpointInferEnabled = ColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = ColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    if (!CheckpointEvalTableExists(w) ||
        (!hasCheckpointInferEnabled && !hasOpportunisticCheckpointInfer) ||
        !ColumnExists(w, "model", "experiment_id"))
    {
        return;
    }
    const bool hasParentExperimentId = ColumnExists(w, "experiment_checkpoint_eval", "parent_experiment_id");
    const bool hasSymbol = ColumnExists(w, "experiment_checkpoint_eval", "symbol");
    const bool hasPredictionHorizon = ColumnExists(w, "experiment_checkpoint_eval", "prediction_horizon");

    std::ostringstream candidateSql;
    candidateSql
        << "WITH train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epoch "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ") "
        << "SELECT e.experiment_id, tm.completed_epoch, m.model_id, "
        << "       COALESCE(e.checkpoint_infer_interval, NULLIF(e.checkpoint_interval, 0)), "
        << "       e.symbol, e.prediction_horizon "
        << "FROM experiment e "
        << "JOIN model m ON m.experiment_id = e.experiment_id "
        << "JOIN train_meta tm ON tm.model_id = m.model_id "
        << "WHERE ";
    if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
        candidateSql << "(e.checkpoint_infer_enabled = true OR e.opportunistic_checkpoint_infer = true) ";
    else if (hasCheckpointInferEnabled)
        candidateSql << "e.checkpoint_infer_enabled = true ";
    else
        candidateSql << "e.opportunistic_checkpoint_infer = true ";
    candidateSql
        << "AND e.phase = 'train' "
        << "AND e.status IN ('pending', 'running', 'paused') "
        << "AND COALESCE(m.comment, '') ILIKE '%periodic training checkpoint%' "
        << "AND (e.checkpoint_infer_min_epoch IS NULL OR tm.completed_epoch >= e.checkpoint_infer_min_epoch) "
        << "ORDER BY e.experiment_id ASC, tm.completed_epoch ASC, m.model_id ASC;";

    pqxx::result candidates = w.exec(candidateSql.str());

    for (const auto& row : candidates)
    {
        const long long experimentId = row[0].as<long long>();
        const int epoch = row[1].as<int>();
        const long long modelId = row[2].as<long long>();
        const std::string symbol = row[4].as<std::string>();
        const int horizon = row[5].as<int>();
        const std::optional<int> interval =
            row[3].is_null() ? std::optional<int>{} : std::optional<int>{row[3].as<int>()};
        if (interval.has_value() && *interval > 0 && epoch % *interval != 0)
        {
            std::cout << "CHECKPOINT_EVAL_SKIPPED"
                      << " experiment_id=" << experimentId
                      << " epoch=" << epoch
                      << " reason=interval_mismatch"
                      << std::endl;
            continue;
        }

        std::ostringstream insertSql;
        insertSql << "INSERT INTO experiment_checkpoint_eval (experiment_id";
        if (hasParentExperimentId)
            insertSql << ", parent_experiment_id";
        insertSql << ", checkpoint_epoch, checkpoint_model_id";
        if (hasSymbol)
            insertSql << ", symbol";
        if (hasPredictionHorizon)
            insertSql << ", prediction_horizon";
        insertSql << ") VALUES ($1";
        int param = 2;
        if (hasParentExperimentId)
            insertSql << ", $" << param++;
        insertSql << ", $" << param++ << ", $" << param++;
        if (hasSymbol)
            insertSql << ", $" << param++;
        if (hasPredictionHorizon)
            insertSql << ", $" << param++;
        insertSql << ") ON CONFLICT ";
        if (hasParentExperimentId)
            insertSql << "(parent_experiment_id, checkpoint_model_id, checkpoint_epoch) ";
        else
            insertSql << "(experiment_id, checkpoint_epoch, checkpoint_model_id) ";
        insertSql << "DO NOTHING RETURNING checkpoint_eval_id;";

        pqxx::result inserted;
        if (hasParentExperimentId && hasSymbol && hasPredictionHorizon)
            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, symbol, horizon);
        else if (hasParentExperimentId && hasSymbol)
            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, symbol);
        else if (hasParentExperimentId && hasPredictionHorizon)
            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, horizon);
        else if (hasParentExperimentId)
            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId);
        else if (hasSymbol && hasPredictionHorizon)
            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, symbol, horizon);
        else if (hasSymbol)
            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, symbol);
        else if (hasPredictionHorizon)
            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, horizon);
        else
            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId);
        if (!inserted.empty())
        {
            std::cout << "CHECKPOINT_EVAL_ENQUEUED"
                      << " experiment_id=" << experimentId
                      << " epoch=" << epoch
                      << " model_id=" << modelId
                      << std::endl;
        }
    }
}

CheckpointEvalRow RowToCheckpointEval(const pqxx::row& row)
{
    CheckpointEvalRow eval;
    eval.checkpointEvalId = row[0].as<long long>();
    eval.checkpointEpoch = row[1].as<int>();
    eval.checkpointModelId = row[2].as<long long>();
    eval.status = row[3].as<std::string>();
    eval.phase = row[4].as<std::string>();
    eval.workerPid = row[5].is_null() ? std::optional<int>{} : std::optional<int>{row[5].as<int>()};
    eval.inferLogPath = OptionalStringCell(row, 6);
    eval.analysisLogPath = OptionalStringCell(row, 7);
    eval.experiment.experimentId = row[8].as<long long>();
    eval.experiment.symbol = EA::CanonicalSymbol::Normalize(row[9].as<std::string>());
    eval.experiment.predictionHorizon = row[10].as<int>();
    eval.experiment.cNextThreshold = row[11].as<double>();
    eval.experiment.coreLrMult = OptionalDoubleCell(row, 12);
    eval.experiment.headLrMult = OptionalDoubleCell(row, 13);
    eval.experiment.targetEpochs = row[14].as<int>();
    eval.experiment.checkpointInterval = row[15].as<int>();
    eval.experiment.trainStart = row[16].as<std::string>();
    eval.experiment.trainEnd = row[17].as<std::string>();
    eval.experiment.inferStart = OptionalStringCell(row, 18);
    eval.experiment.inferEnd = OptionalStringCell(row, 19);
    eval.experiment.lastModelId = eval.checkpointModelId;
    eval.experiment.resumeModelId = OptionalLongLongCell(row, 20);
    eval.experiment.trainLogPath = OptionalStringCell(row, 21);
    eval.inferStartedEpoch = row[22].is_null() ? 0.0 : row[22].as<double>();
    eval.experiment.inferLogPath = eval.inferLogPath;
    eval.experiment.analysisLogPath = eval.analysisLogPath;
    if (row.size() > 23 && !row[23].is_null())
        eval.cancellationRequestId = row[23].as<long long>();
    return eval;
}

std::vector<CheckpointEvalRow> LoadCheckpointEvalRows(pqxx::work& w,
                                                      const std::string& status,
                                                      const std::string& phase)
{
    if (!CheckpointEvalTableExists(w))
        return {};
    pqxx::result rows = w.exec_params(
        "SELECT ce.checkpoint_eval_id, ce.checkpoint_epoch, ce.checkpoint_model_id, "
        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
        "e.experiment_id, e.symbol, e.prediction_horizon, e.c_next_threshold, "
        "e.core_lr_mult, e.head_lr_mult, e.target_epochs, e.checkpoint_interval, "
        "e.train_start::text, e.train_end::text, e.infer_start::text, e.infer_end::text, "
        "e.resume_model_id, e.train_log_path, "
        "extract(epoch from COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at))::double precision, "
        "ce.cancellation_request_id "
        "FROM experiment_checkpoint_eval ce "
        "JOIN experiment e ON e.experiment_id = COALESCE(ce.parent_experiment_id, ce.experiment_id) "
        "WHERE ce.status = $1 AND ce.phase = $2 "
        "AND (ce.phase <> 'analyze' OR ce.cancellation_request_id IS NULL) "
        "ORDER BY ce.created_at ASC, ce.checkpoint_eval_id ASC;",
        status,
        phase);

    std::vector<CheckpointEvalRow> evals;
    evals.reserve(rows.size());
    for (const auto& row : rows)
        evals.push_back(RowToCheckpointEval(row));
    return evals;
}

std::string ReadFileIfExists(const std::optional<std::string>& path)
{
    if (!path.has_value())
        return {};
    std::ifstream in{*path};
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string ReadFileIfExists(const std::string& path)
{
    std::ifstream in{path};
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void WriteTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out{path};
    out << text;
}

bool IsValidInferenceLogText(const std::string& text)
{
    return text.find("Overall 3-class accuracy") != std::string::npos &&
           text.find("Overall 3-class confusion matrix") != std::string::npos &&
           text.find("MODEL_ACCEPTANCE") != std::string::npos;
}

bool FileWasModifiedForAttempt(const std::string& path,
                               double attemptStartedEpoch)
{
    std::error_code ec;
    const auto fileTime = std::filesystem::last_write_time(path, ec);
    if (ec)
        return false;
    const auto systemTime = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        fileTime - std::filesystem::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    const double modifiedEpoch = std::chrono::duration<double>(
        systemTime.time_since_epoch()).count();
    return IsCurrentWorkerAttemptEvidence(modifiedEpoch, attemptStartedEpoch);
}

bool HasValidInferenceLogPathForAttempt(
    const ExperimentRow& experiment,
    double attemptStartedEpoch)
{
    return experiment.inferLogPath.has_value() &&
           FileWasModifiedForAttempt(*experiment.inferLogPath, attemptStartedEpoch) &&
           IsValidInferenceLogText(ReadFileIfExists(experiment.inferLogPath));
}

std::optional<std::string> DiscoverValidInferenceLogForAttempt(
    const ExperimentRow& experiment,
    double attemptStartedEpoch)
{
    if (!experiment.lastModelId.has_value())
        return std::nullopt;

    const std::string experimentNeedle =
        "experiment_" + std::to_string(experiment.experimentId) + "_";
    const std::vector<std::filesystem::path> roots = {
        std::filesystem::current_path(),
        std::filesystem::current_path() / "experiment_logs"};

    for (const auto& root : roots)
    {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec))
            continue;
        for (const auto& entry : std::filesystem::directory_iterator(root, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file(ec))
                continue;
            const std::string filename = entry.path().filename().string();
            if (filename.find(experimentNeedle) == std::string::npos ||
                filename.find("infer") == std::string::npos)
                continue;
            const std::string path = entry.path().string();
            if (FileWasModifiedForAttempt(path, attemptStartedEpoch) &&
                IsValidInferenceLogText(ReadFileIfExists(path)))
                return path;
        }
    }
    return std::nullopt;
}

void ApplyPersistedSymbolToAnalysisExperiment(pqxx::work& w,
                                                     ExperimentRow& experiment,
                                                     const ParsedMetrics& metrics)
{
    const std::optional<long long> modelId =
        metrics.modelId.has_value() ? metrics.modelId : experiment.lastModelId;
    if (!modelId.has_value())
    {
        experiment.symbol = EA::CanonicalSymbol::Normalize(experiment.symbol);
        return;
    }

    const std::optional<std::string> persistedSymbol =
        TryLoadPersistedCanonicalSymbol(w, *modelId);
    if (persistedSymbol.has_value())
    {
        experiment.symbol = *persistedSymbol;
        std::cout << "MODEL_SYMBOL"
                  << ",source=database"
                  << ",model_id=" << *modelId
                  << ",symbol=" << experiment.symbol
                  << std::endl;
        return;
    }

    PrintModelSymbolMissing(*modelId);
    experiment.symbol = EA::CanonicalSymbol::Normalize(experiment.symbol);
    std::cout << "MODEL_SYMBOL"
              << ",source=legacy"
              << ",model_id=" << *modelId
              << ",symbol=" << experiment.symbol
              << ",warning=missing_metadata"
              << std::endl;
}

bool HasCompletedInferenceResult(pqxx::work& w,
                                        const ExperimentRow& experiment)
{
    if (!TableExists(w, "inference_eval_result") ||
        !experiment.lastModelId.has_value() ||
        !experiment.inferStart.has_value() ||
        !experiment.inferEnd.has_value())
    {
        return false;
    }

    pqxx::result rows = w.exec_params(
        "SELECT 1 "
        "FROM inference_eval_result "
        "WHERE model_id = $1 "
        "AND symbol = $2 "
        "AND prediction_horizon = $3 "
        "AND threshold_logret = $4 "
        "AND from_date = $5 "
        "AND to_date = $6 "
        "AND status = 'completed' "
        "AND inference_scope = 'final' "
        "AND checkpoint_eval_id IS NULL "
        "LIMIT 1;",
        *experiment.lastModelId,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.inferStart->substr(0, 10),
        experiment.inferEnd->substr(0, 10));
    return !rows.empty();
}

bool OperatorForcedFinalInferenceRerunRequested(
    pqxx::work& w,
    long long experimentId)
{
    const pqxx::result rows = w.exec(
        "SELECT operator_forced_final_inference_rerun_requested "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    return rows.size() == 1 && rows[0][0].as<bool>();
}

bool HasCompletedInferenceResultForAttempt(
    pqxx::work& w,
    const ExperimentRow& experiment,
    double attemptStartedEpoch,
    bool useThresholdTolerance = false)
{
    if (!TableExists(w, "inference_eval_result") ||
        !experiment.lastModelId.has_value() ||
        !experiment.inferStart.has_value() ||
        !experiment.inferEnd.has_value())
    {
        return false;
    }

    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM inference_eval_result "
        "WHERE model_id = $1 AND symbol = $2 AND prediction_horizon = $3 "
        "AND (($8::boolean AND abs(threshold_logret - $4) <= 1e-7) "
        "OR (NOT $8::boolean AND threshold_logret = $4)) "
        "AND from_date = $5 AND to_date = $6 "
        "AND status = 'completed' AND inference_scope = 'final' "
        "AND checkpoint_eval_id IS NULL "
        "AND completed_at >= to_timestamp($7) LIMIT 1;",
        *experiment.lastModelId,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.inferStart->substr(0, 10),
        experiment.inferEnd->substr(0, 10),
        attemptStartedEpoch,
        useThresholdTolerance);
    return !rows.empty();
}

bool HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
    pqxx::work& w,
    const ExperimentRow& experiment,
    long long workerAttemptId,
    bool useThresholdTolerance = false)
{
    EA::SchedulerCore::PostgresSchedulerRepository repository{w};
    const auto result =
        repository.findAuthoritativeFinalInferenceResultForWorkerAttempt(
            experiment.experimentId, workerAttemptId);
    if (!result) return false;
    if (result->forcedFinalInferenceRerun != useThresholdTolerance)
        throw std::runtime_error(
            "forced_final_inference_recovery_state_changed");
    return true;
}

std::optional<long long> FindCompletedCheckpointInferenceResultId(
    pqxx::work& w,
    const CheckpointEvalRow& eval)
{
    if (!TableExists(w, "inference_eval_result") ||
        !ColumnExists(w, "inference_eval_result", "checkpoint_eval_id"))
    {
        return std::nullopt;
    }

    pqxx::result rows = w.exec_params(
        "SELECT id "
        "FROM inference_eval_result "
        "WHERE checkpoint_eval_id = $1 "
        "AND model_id = $2 "
        "AND inference_scope = 'checkpoint' "
        "AND status = 'completed' "
        "LIMIT 1;",
        eval.checkpointEvalId,
        eval.checkpointModelId);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

std::optional<long long> FindCompletedCheckpointInferenceResultIdForAttempt(
    pqxx::work& w,
    const CheckpointEvalRow& eval,
    double attemptStartedEpoch)
{
    if (!TableExists(w, "inference_eval_result") ||
        !ColumnExists(w, "inference_eval_result", "checkpoint_eval_id"))
        return std::nullopt;

    pqxx::result rows = w.exec_params(
        "SELECT id FROM inference_eval_result "
        "WHERE checkpoint_eval_id = $1 AND model_id = $2 "
        "AND inference_scope = 'checkpoint' AND status = 'completed' "
        "AND completed_at >= to_timestamp($3) LIMIT 1;",
        eval.checkpointEvalId,
        eval.checkpointModelId,
        attemptStartedEpoch);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

std::optional<long long>
FindAuthoritativeCompletedCheckpointInferenceResultForWorkerAttempt(
    pqxx::work& w,
    const CheckpointEvalRow& eval,
    long long workerAttemptId)
{
    if (!TableExists(w, "inference_eval_result") ||
        !ColumnExists(w, "inference_eval_result", "checkpoint_eval_id"))
        return std::nullopt;
    const pqxx::result rows = w.exec_params(
        "SELECT r.id FROM inference_eval_result r "
        "JOIN model m ON m.model_id=r.model_id "
        "JOIN experiment_scheduler_worker_attempt a "
        " ON a.worker_attempt_id=$3 AND a.experiment_id=$4 "
        " AND a.checkpoint_eval_id=$1 "
        " AND a.worker_kind='checkpoint_infer' "
        " AND a.lifecycle_phase='infer' "
        "WHERE r.checkpoint_eval_id=$1 AND r.model_id=$2 "
        "AND m.experiment_id=$4 AND r.inference_scope='checkpoint' "
        "AND r.status='completed' AND r.completed_at>=a.reserved_at "
        "LIMIT 1;",
        eval.checkpointEvalId,
        eval.checkpointModelId,
        workerAttemptId,
        eval.experiment.experimentId);
    if (rows.empty()) return std::nullopt;
    return rows[0][0].as<long long>();
}

bool ApplyStructuredCheckpointInferenceMetrics(pqxx::work& w,
                                               const CheckpointEvalRow& eval,
                                               ParsedMetrics& metrics)
{
    pqxx::result rows = w.exec_params(
        "SELECT accuracy, accept_model, COALESCE(reject_reason, ''), completed_epochs "
        "FROM inference_eval_result "
        "WHERE checkpoint_eval_id = $1 "
        "AND model_id = $2 "
        "AND inference_scope = 'checkpoint' "
        "AND status = 'completed' "
        "LIMIT 1;",
        eval.checkpointEvalId,
        eval.checkpointModelId);
    if (rows.empty())
        return false;

    if (!rows[0][0].is_null())
        metrics.inferAccuracy = rows[0][0].as<double>();
    if (!rows[0][1].is_null())
        metrics.acceptModel = rows[0][1].as<bool>();
    const std::string rejectReason = rows[0][2].as<std::string>();
    if (!rejectReason.empty())
        metrics.rejectReason = rejectReason;
    if (!rows[0][3].is_null())
        metrics.completedEpochs = rows[0][3].as<int>();
    metrics.modelId = eval.checkpointModelId;
    metrics.acceptAccuracy = metrics.inferAccuracy;
    metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;
    return true;
}

bool HasCompletedAnalysisResult(pqxx::work& w,
                                       const ExperimentRow& experiment)
{
    if (!experiment.lastModelId.has_value())
        return false;

    pqxx::result rows = w.exec_params(
        "SELECT 1 "
        "FROM experiment_analysis_result "
        "WHERE experiment_id = $1 "
        "AND model_id = $2 "
        "AND COALESCE(analysis_scope, 'final') = 'final' "
        "AND analysis_status = 'completed' "
        "LIMIT 1;",
        experiment.experimentId,
        *experiment.lastModelId);
    return !rows.empty();
}

bool HasCompletedAnalysisResultForAttempt(
    pqxx::work& w,
    const ExperimentRow& experiment,
    double attemptStartedEpoch)
{
    if (!experiment.lastModelId.has_value())
        return false;

    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM experiment_analysis_result "
        "WHERE experiment_id = $1 AND model_id = $2 "
        "AND COALESCE(analysis_scope, 'final') = 'final' "
        "AND analysis_status = 'completed' "
        "AND updated_at >= to_timestamp($3) LIMIT 1;",
        experiment.experimentId,
        *experiment.lastModelId,
        attemptStartedEpoch);
    return !rows.empty();
}

std::optional<long long> FindCheckpointAnalysisResultId(pqxx::work& w,
                                                        long long checkpointEvalId)
{
    if (!ColumnExists(w, "experiment_analysis_result", "checkpoint_eval_id"))
        return std::nullopt;

    pqxx::result rows = w.exec_params(
        "SELECT analysis_id "
        "FROM experiment_analysis_result "
        "WHERE checkpoint_eval_id = $1 "
        "AND analysis_scope = 'checkpoint' "
        "ORDER BY updated_at DESC, analysis_id DESC "
        "LIMIT 1;",
        checkpointEvalId);
    if (rows.empty() || rows[0][0].is_null())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

double PredictionImbalancePenalty(const ParsedMetrics& metrics)
{
    if (!metrics.hasConfusion)
        return 1.0;

    long long pred[3] = {};
    long long total = 0;
    for (int actual = 0; actual < 3; ++actual)
    {
        for (int cls = 0; cls < 3; ++cls)
        {
            pred[cls] += metrics.confusion[actual][cls];
            total += metrics.confusion[actual][cls];
        }
    }
    if (total == 0)
        return 1.0;

    const double maxPredFrac = static_cast<double>(*std::max_element(pred, pred + 3)) /
                               static_cast<double>(total);
    if (maxPredFrac <= 0.60)
        return 1.0;
    return std::max(0.25, 1.0 - ((maxPredFrac - 0.60) / 0.40));
}

std::optional<double> ComputeLeaderScore(const ParsedMetrics& metrics)
{
    if (!metrics.inferAccuracy.has_value())
        return std::nullopt;
    const double acceptAccuracy = metrics.acceptAccuracy.value_or(*metrics.inferAccuracy);
    const double penalty = PredictionImbalancePenalty(metrics);
    return (*metrics.inferAccuracy) * (0.75 + 0.25 * acceptAccuracy) * penalty;
}

std::string MetricSql(pqxx::work&, const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "NULL";
}

std::string MetricSql(pqxx::work&, const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

std::string MetricSql(pqxx::work&, const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

std::string MetricSql(pqxx::work& w, const std::optional<std::string>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

std::string MarkdownCell(std::string value)
{
    if (value.empty())
        return "n/a";
    std::replace(value.begin(), value.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\r', ' ');
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value)
    {
        if (ch == '|')
            escaped += "\\|";
        else
            escaped.push_back(ch);
    }
    return escaped;
}

std::string CellText(const pqxx::field& field)
{
    return field.is_null() ? "n/a" : MarkdownCell(field.c_str());
}

std::string RowCell(const pqxx::row& row, pqxx::row::size_type index)
{
    return index < row.size() ? CellText(row[index]) : "n/a";
}

std::string RenderMarkdownReport(const std::string& title,
                                 const std::vector<std::string>& headers,
                                 const pqxx::result& rows)
{
    std::ostringstream out;
    out << "# " << title << "\n\n";
    out << "Rows: " << rows.size() << "\n\n";
    for (const std::string& header : headers)
        out << "| " << header << " ";
    out << "|\n";
    for (size_t i = 0; i < headers.size(); ++i)
        out << "|---";
    out << "|\n";
    for (const auto& row : rows)
    {
        for (pqxx::row::size_type i = 0; i < row.size(); ++i)
            out << "| " << CellText(row[i]) << " ";
        out << "|\n";
    }
    return out.str();
}

std::string RenderMarkdownTable(const std::vector<std::string>& headers,
                                const pqxx::result& rows)
{
    std::ostringstream out;
    for (const std::string& header : headers)
        out << "| " << header << " ";
    out << "|\n";
    for (size_t i = 0; i < headers.size(); ++i)
        out << "|---";
    out << "|\n";
    for (const auto& row : rows)
    {
        for (pqxx::row::size_type i = 0; i < row.size(); ++i)
            out << "| " << CellText(row[i]) << " ";
        out << "|\n";
    }
    return out.str();
}

std::vector<std::string> CompactLeaderHeaders()
{
    return {
        "experiment_id",
        "model_id",
        "symbol",
        "prediction_horizon",
        "target_epochs",
        "completed_epochs",
        "infer_accuracy",
        "accept_rate",
        "accept_accuracy",
        "leader_score"
    };
}

std::string RenderCompactLeaderTable(const pqxx::result& rows,
                                     size_t maxRows = std::numeric_limits<size_t>::max())
{
    std::ostringstream out;
    const std::vector<std::string> headers = CompactLeaderHeaders();
    for (const std::string& header : headers)
        out << "| " << header << " ";
    out << "|\n";
    for (size_t i = 0; i < headers.size(); ++i)
        out << "|---";
    out << "|\n";
    size_t rowCount = 0;
    for (const auto& row : rows)
    {
        if (rowCount++ >= maxRows)
            break;
        for (pqxx::row::size_type i = 0; i < 10 && i < row.size(); ++i)
            out << "| " << CellText(row[i]) << " ";
        out << "|\n";
    }
    return out.str();
}

pqxx::result ExecReportQuery(pqxx::work& w, const std::string& orderClause, const std::string& limitClause)
{
    return w.exec(
        std::string{
        "SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
        "a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
        "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
        "FROM experiment_analysis_result a "
        "JOIN experiment e ON e.experiment_id = a.experiment_id "
        "WHERE a.analysis_status = 'completed' "
        "AND COALESCE(a.analysis_scope, 'final') = 'final' "} +
        orderClause + " " + limitClause + ";");
}

std::vector<std::string> ExperimentReportHeaders()
{
    return {
        "experiment_id",
        "model_id",
        "symbol",
        "prediction_horizon",
        "target_epochs",
        "completed_epochs",
        "infer_accuracy",
        "accept_rate",
        "accept_accuracy",
        "leader_score",
        "analysis_status",
        "completed_at"
    };
}

std::vector<std::string> RecommendationReportHeaders()
{
    return {
        "experiment_id",
        "model_id",
        "symbol",
        "prediction_horizon",
        "target_epochs",
        "completed_epochs",
        "infer_accuracy",
        "accept_rate",
        "accept_accuracy",
        "leader_score",
        "analysis_status",
        "status",
        "phase",
        "completed_at",
        "priority",
        "recommendation",
        "suggested_action",
        "example_command"
    };
}

std::vector<std::string> PruningReportHeaders()
{
    return {
        "experiment_id",
        "model_id",
        "symbol",
        "prediction_horizon",
        "target_epochs",
        "completed_epochs",
        "status",
        "phase",
        "infer_accuracy",
        "accept_rate",
        "accept_accuracy",
        "leader_score",
        "completed_at",
        "reason",
        "suggested_action",
        "example_command"
    };
}

struct RecommendationPriorityCounts
{
    size_t critical = 0;
    size_t high = 0;
    size_t medium = 0;
    size_t low = 0;
    size_t informational = 0;

    size_t Total() const
    {
        return critical + high + medium + low + informational;
    }
};

std::string ActionCliPrefixSql()
{
    return "'./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release";
}

std::string QueueExperimentCommandSql(const std::string& targetEpochsExpression)
{
    return ActionCliPrefixSql() +
        " --queue-experiment --symbol=' || e.symbol || "
        "' --prediction-horizon=' || e.prediction_horizon::text || "
        "' --target-epochs=' || (" + targetEpochsExpression + ")::text || "
        "' --threshold=' || e.c_next_threshold::text || "
        "COALESCE(' --core-lr=' || e.core_lr_mult::text, '') || "
        "COALESCE(' --head-lr=' || e.head_lr_mult::text, '') || "
        "' --checkpoint-interval=' || e.checkpoint_interval::text";
}

std::string QueueResumeCommandSql(const std::string& targetEpochsExpression)
{
    return ActionCliPrefixSql() +
        " --queue-experiment --resume-model-id=' || COALESCE(a.model_id, e.last_model_id)::text || "
        "' --target-epochs=' || (" + targetEpochsExpression + ")::text";
}

std::string ExperimentControlCommandSql(const std::string& optionName)
{
    return ActionCliPrefixSql() + " " + optionName + "=' || e.experiment_id::text";
}

std::string ExperimentStatusCommandSql()
{
    return ActionCliPrefixSql() + " --status --experiment-id=' || e.experiment_id::text";
}

std::string ExperimentMetadataCommandSql()
{
    return ActionCliPrefixSql() + " --experiment-metadata=' || e.experiment_id::text";
}

std::string StaticCommandSql(const std::string& args)
{
    return "'./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release " + args + "'";
}

std::string RecommendationSelectColumns(const std::string& priorityExpression,
                                        const std::string& recommendationExpression,
                                        const std::string& suggestedActionExpression,
                                        const std::string& exampleCommandExpression)
{
    return
        "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
        "COALESCE(a.symbol, e.symbol), COALESCE(a.prediction_horizon, e.prediction_horizon), "
        "COALESCE(a.target_epochs, e.target_epochs), a.completed_epochs, "
        "a.infer_accuracy, a.accept_rate, a.accept_accuracy, a.leader_score, "
        "COALESCE(a.analysis_status, 'n/a'), e.status, e.phase, e.completed_at::text, " +
        priorityExpression + " AS priority, " +
        recommendationExpression + " AS recommendation, " +
        suggestedActionExpression + " AS suggested_action, " +
        exampleCommandExpression + " AS example_command ";
}

std::string PruningSelectColumns(const std::string& reasonExpression,
                                 const std::string& suggestedActionExpression,
                                 const std::string& exampleCommandExpression)
{
    return
        "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
        "COALESCE(a.symbol, e.symbol), COALESCE(a.prediction_horizon, e.prediction_horizon), "
        "COALESCE(a.target_epochs, e.target_epochs), a.completed_epochs, "
        "e.status, e.phase, a.infer_accuracy, a.accept_rate, a.accept_accuracy, "
        "a.leader_score, e.completed_at::text, " +
        reasonExpression + " AS reason, " +
        suggestedActionExpression + " AS suggested_action, " +
        exampleCommandExpression + " AS example_command ";
}

void AddRecommendationPriority(RecommendationPriorityCounts& counts, const pqxx::row& row)
{
    if (row.size() <= 14 || row[14].is_null())
        return;
    const std::string priority = row[14].as<std::string>();
    if (priority == "Critical")
        ++counts.critical;
    else if (priority == "High")
        ++counts.high;
    else if (priority == "Medium")
        ++counts.medium;
    else if (priority == "Low")
        ++counts.low;
    else if (priority == "Informational")
        ++counts.informational;
}

RecommendationPriorityCounts CountRecommendationPriorities(
    const std::vector<std::pair<std::string, pqxx::result>>& sections)
{
    RecommendationPriorityCounts counts;
    for (const auto& section : sections)
    {
        for (const auto& row : section.second)
            AddRecommendationPriority(counts, row);
    }
    return counts;
}

std::string RenderRecommendationReport(
    const std::vector<std::pair<std::string, pqxx::result>>& sections)
{
    std::ostringstream out;
    out << "# Experiment Recommendations\n\n";
    out << "This report is advisory only. It is generated from existing scheduler experiment "
        << "and analysis rows and does not queue, update, cancel, pause, retry, stop, or "
        << "launch experiments.\n\n";

    size_t total = 0;
    for (const auto& section : sections)
        total += section.second.size();
    out << "Total recommendations: " << total << "\n\n";

    const RecommendationPriorityCounts counts = CountRecommendationPriorities(sections);
    out << "## Priority Summary\n\n";
    out << "| priority | count |\n";
    out << "|---|---|\n";
    out << "| Critical | " << counts.critical << " |\n";
    out << "| High | " << counts.high << " |\n";
    out << "| Medium | " << counts.medium << " |\n";
    out << "| Low | " << counts.low << " |\n";
    out << "| Informational | " << counts.informational << " |\n\n";

    for (const auto& section : sections)
    {
        out << "## " << section.first << "\n\n";
        out << "Rows: " << section.second.size() << "\n\n";
        out << RenderMarkdownTable(RecommendationReportHeaders(), section.second);
        out << "\n";
    }
    return out.str();
}

std::string RenderPruningReport(
    const std::vector<std::pair<std::string, pqxx::result>>& sections)
{
    std::ostringstream out;
    out << "# Pruning / Archive Candidates\n\n";
    out << "This report is advisory only. It identifies scheduler experiment rows and model "
        << "references that may be obsolete, dominated, failed, cancelled, duplicate, "
        << "or missing analysis. It does not delete, archive, cancel, pause, retry, "
        << "resume, move, or mutate anything.\n\n";

    size_t total = 0;
    for (const auto& section : sections)
        total += section.second.size();
    out << "Total candidates: " << total << "\n\n";

    for (const auto& section : sections)
    {
        out << "## " << section.first << "\n\n";
        out << "Rows: " << section.second.size() << "\n\n";
        out << RenderMarkdownTable(PruningReportHeaders(), section.second);
        out << "\n";
    }
    return out.str();
}

RecommendationPriorityCounts WriteRecommendationReport(pqxx::work& w, const std::string& reportDir)
{
    std::vector<std::pair<std::string, pqxx::result>> sections;
    sections.reserve(10);

    sections.emplace_back(
        "Continue / Extend Promising Experiments",
        w.exec(
            RecommendationSelectColumns(
                "'High'",
                "'high leader_score; consider extending target_epochs or next checkpoint'",
                "'Queue resumed extension from this model'",
                QueueResumeCommandSql("GREATEST(e.target_epochs + e.checkpoint_interval, e.target_epochs + 20)")) +
            "FROM experiment_analysis_result a "
            "JOIN experiment e ON e.experiment_id = a.experiment_id "
            "WHERE e.status = 'completed' "
            "AND a.analysis_status = 'completed' "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "AND a.leader_score IS NOT NULL "
            "AND a.infer_accuracy IS NOT NULL "
            "ORDER BY a.leader_score DESC NULLS LAST, a.infer_accuracy DESC NULLS LAST, "
            "a.accept_accuracy DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 10;"));

    sections.emplace_back(
        "Replicate Current Leaders",
        w.exec(
            "WITH ranked AS ("
            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "         a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
            "         e.completed_at, e.c_next_threshold, e.core_lr_mult, e.head_lr_mult, "
            "         e.checkpoint_interval, "
            "         row_number() OVER (PARTITION BY a.symbol, a.prediction_horizon "
            "                            ORDER BY a.leader_score DESC NULLS LAST, "
            "                                     a.infer_accuracy DESC NULLS LAST, "
            "                                     e.experiment_id DESC) AS rn "
            "  FROM experiment_analysis_result a "
            "  JOIN experiment e ON e.experiment_id = a.experiment_id "
            "  WHERE e.status = 'completed' "
            "  AND a.analysis_status = 'completed' "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "  AND a.leader_score IS NOT NULL "
            "  AND a.infer_accuracy IS NOT NULL "
            ") "
            "SELECT e.experiment_id, e.model_id, e.symbol, e.prediction_horizon, e.target_epochs, "
            "e.completed_epochs, e.infer_accuracy, e.accept_rate, e.accept_accuracy, e.leader_score, "
            "e.analysis_status, e.status, e.phase, e.completed_at::text, "
            "'High' AS priority, "
            "'replicate current leader for symbol/horizon' AS recommendation, "
            "'Queue another run with the same configuration' AS suggested_action, "
            "(" + QueueExperimentCommandSql("e.target_epochs") + " || ' --allow-duplicate-experiment') AS example_command "
            "FROM ranked e "
            "WHERE rn = 1 "
            "ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, "
            "symbol ASC, prediction_horizon ASC "
            "LIMIT 20;"));

    sections.emplace_back(
        "Try Nearby Configuration Variants",
        w.exec(
            RecommendationSelectColumns(
                "'Medium'",
                "'try nearby core/head learning-rate variant around this leader'",
                "'Queue nearby LR variant'",
                ActionCliPrefixSql() +
                    " --queue-experiment --symbol=' || e.symbol || "
                    "' --prediction-horizon=' || e.prediction_horizon::text || "
                    "' --target-epochs=' || e.target_epochs::text || "
                    "' --threshold=' || e.c_next_threshold::text || "
                    "' --core-lr=' || COALESCE((e.core_lr_mult + 20)::text, '120') || "
                    "' --head-lr=' || COALESCE((e.head_lr_mult + 10)::text, '50') || "
                    "' --checkpoint-interval=' || e.checkpoint_interval::text") +
            "FROM experiment_analysis_result a "
            "JOIN experiment e ON e.experiment_id = a.experiment_id "
            "WHERE e.status = 'completed' "
            "AND a.analysis_status = 'completed' "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "AND a.leader_score IS NOT NULL "
            "AND a.leader_score > 0 "
            "AND a.infer_accuracy IS NOT NULL "
            "AND (e.core_lr_mult IS NOT NULL OR e.head_lr_mult IS NOT NULL) "
            "ORDER BY a.leader_score DESC NULLS LAST, a.infer_accuracy DESC NULLS LAST, "
            "e.symbol ASC, e.prediction_horizon ASC, e.experiment_id DESC "
            "LIMIT 10;"));

    sections.emplace_back(
        "Investigate Failures",
        w.exec(
            RecommendationSelectColumns(
                "'Critical'",
                "COALESCE('investigate failure: ' || NULLIF(e.error_message, ''), "
                "'investigate failure logs and configuration')",
                "'Review logs; retry failed experiment if still relevant'",
                ExperimentControlCommandSql("--retry-failed-experiment")) +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'failed' "
            "ORDER BY e.completed_at DESC NULLS LAST, e.updated_at DESC NULLS LAST, "
            "e.experiment_id DESC "
            "LIMIT 25;"));

    sections.emplace_back(
        "Avoid Or Pause Dominated Configurations",
        w.exec(
            "WITH eligible AS ("
            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "         a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
            "         e.completed_at "
            "  FROM experiment_analysis_result a "
            "  JOIN experiment e ON e.experiment_id = a.experiment_id "
            "  WHERE e.status = 'completed' "
            "  AND a.analysis_status = 'completed' "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "  AND a.leader_score IS NOT NULL "
            "  AND a.infer_accuracy IS NOT NULL "
            "), leaders AS ("
            "  SELECT eligible.*, row_number() OVER (PARTITION BY symbol, prediction_horizon "
            "    ORDER BY leader_score DESC, infer_accuracy DESC, experiment_id DESC) AS rn "
            "  FROM eligible "
            ") "
            "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
            "COALESCE(a.symbol, e.symbol), COALESCE(a.prediction_horizon, e.prediction_horizon), "
            "COALESCE(a.target_epochs, e.target_epochs), a.completed_epochs, "
            "a.infer_accuracy, a.accept_rate, a.accept_accuracy, a.leader_score, "
            "COALESCE(a.analysis_status, 'n/a'), e.status, e.phase, e.completed_at::text, "
            "CASE WHEN e.status = 'running' THEN 'High' "
            "     WHEN e.status = 'pending' THEN 'Medium' "
            "     ELSE 'Low' END AS priority, "
            "('dominated by experiment ' || l.experiment_id::text || "
            "' for same symbol/horizon') AS recommendation, "
            "CASE WHEN e.status = 'running' THEN 'Preview pause of running dominated experiment' "
            "     WHEN e.status = 'pending' THEN 'Preview cancel of pending dominated experiment' "
            "     ELSE 'No action required' END AS suggested_action, "
            "CASE WHEN e.status = 'running' THEN " + ExperimentControlCommandSql("--pause-experiment") + " "
            "     WHEN e.status = 'pending' THEN " + ExperimentControlCommandSql("--cancel-experiment") + " "
            "     ELSE 'No direct CLI action.' END AS example_command "
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "JOIN leaders l ON l.symbol = e.symbol "
            "AND l.prediction_horizon = e.prediction_horizon "
            "AND l.rn = 1 "
            "WHERE e.experiment_id <> l.experiment_id "
            "AND e.status IN ('pending', 'running', 'completed') "
            "AND (a.leader_score IS NULL OR l.leader_score > a.leader_score) "
            "ORDER BY priority ASC, e.status DESC, e.symbol ASC, e.prediction_horizon ASC, e.experiment_id ASC "
            "LIMIT 25;"));

    sections.emplace_back(
        "Fill Coverage Gaps By Symbol/Horizon",
        w.exec(
            RecommendationSelectColumns(
                "'Medium'",
                "'coverage gap: experiment lacks completed analysis evidence'",
                "CASE WHEN e.status IN ('pending', 'running') THEN "
                "'No action required; scheduler will process automatically' "
                "ELSE 'Queue a coverage experiment with matching symbol/horizon' END",
                "CASE WHEN e.status IN ('pending', 'running') THEN 'No direct CLI action.' "
                "ELSE " + QueueExperimentCommandSql("e.target_epochs") + " END") +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a "
            "  ON a.experiment_id = e.experiment_id "
            "  AND a.analysis_status = 'completed' "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE a.analysis_id IS NULL "
            "AND e.status <> 'cancelled' "
            "ORDER BY e.symbol ASC, e.prediction_horizon ASC, e.target_epochs DESC, "
            "e.created_at ASC, e.experiment_id ASC "
            "LIMIT 25;"));

    sections.emplace_back(
        "Run Missing Inference/Analysis Where Metrics Are Absent",
        w.exec(
            RecommendationSelectColumns(
                "'Medium'",
                "'missing ranking evidence: run or repair inference/analysis metrics'",
                "CASE WHEN e.status IN ('pending', 'running') THEN "
                "'No action required; scheduler will process automatically' "
                "WHEN e.last_model_id IS NOT NULL THEN 'Requeue analysis for this model' "
                "ELSE 'No direct CLI action' END",
                "CASE WHEN e.status IN ('pending', 'running') THEN 'No direct CLI action.' "
                "WHEN e.last_model_id IS NOT NULL THEN " + ExperimentControlCommandSql("--requeue-analysis") + " "
                "ELSE 'No direct CLI action.' END") +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'completed' "
            "AND (a.analysis_id IS NULL OR a.infer_accuracy IS NULL OR a.leader_score IS NULL) "
            "ORDER BY e.completed_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 25;"));

    sections.emplace_back(
        "Current Running Experiments",
        w.exec(
            RecommendationSelectColumns(
                "'Informational'",
                "'running experiment; continue unless operator intervention is required'",
                "'Monitor running experiment'",
                ExperimentStatusCommandSql()) +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'running' "
            "ORDER BY e.started_at ASC NULLS LAST, e.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Current Pending Queue",
        w.exec(
            RecommendationSelectColumns(
                "'Informational'",
                "'pending experiment; leave queued unless obsolete or superseded'",
                "'Leave queued'",
                "'No direct CLI action.'") +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'pending' "
            "ORDER BY e.created_at ASC, e.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Completed Experiment Actions",
        w.exec(
            "WITH ranked AS ("
            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "         a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
            "         e.completed_at, e.c_next_threshold, e.core_lr_mult, e.head_lr_mult, "
            "         e.checkpoint_interval, "
            "         row_number() OVER (ORDER BY a.leader_score DESC NULLS LAST, "
            "           a.infer_accuracy DESC NULLS LAST, e.experiment_id DESC) AS rn "
            "  FROM experiment_analysis_result a "
            "  JOIN experiment e ON e.experiment_id = a.experiment_id "
            "  WHERE e.status = 'completed' "
            "  AND a.analysis_status = 'completed' "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            ") "
            "SELECT e.experiment_id, e.model_id, e.symbol, e.prediction_horizon, "
            "e.target_epochs, e.completed_epochs, e.infer_accuracy, e.accept_rate, "
            "e.accept_accuracy, e.leader_score, e.analysis_status, e.status, e.phase, "
            "e.completed_at::text, "
            "CASE WHEN e.rn = 1 THEN 'High' ELSE 'Informational' END AS priority, "
            "CASE WHEN e.rn = 1 THEN 'Leader; replicate or extend' "
            "     WHEN e.leader_score IS NOT NULL THEN 'Completed comparison point; archive candidate' "
            "     ELSE 'Completed experiment without ranking evidence' END AS recommendation, "
            "CASE WHEN e.rn = 1 THEN 'Queue replication of current leader' "
            "     ELSE 'No action required' END AS suggested_action, "
            "CASE WHEN e.rn = 1 THEN (" + QueueExperimentCommandSql("e.target_epochs") + " || ' --allow-duplicate-experiment') "
            "     ELSE 'No direct CLI action.' END AS example_command "
            "FROM ranked e "
            "ORDER BY e.rn ASC, e.completed_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 25;"));

    const RecommendationPriorityCounts counts = CountRecommendationPriorities(sections);

    const std::filesystem::path path = std::filesystem::path{reportDir} / "recommendations.md";
    WriteTextFile(path.string(), RenderRecommendationReport(sections));
    std::cout << "EXPERIMENT_RECOMMENDATION_REPORT_GENERATED"
              << ",path=" << path.string()
              << ",recommendations=" << counts.Total()
              << ",critical=" << counts.critical
              << ",high=" << counts.high
              << ",medium=" << counts.medium
              << ",low=" << counts.low
              << ",informational=" << counts.informational
              << std::endl;
    return counts;
}

size_t WritePruningArchiveReport(pqxx::work& w, const std::string& reportDir)
{
    std::vector<std::pair<std::string, pqxx::result>> sections;
    sections.reserve(8);

    sections.emplace_back(
        "Obsolete Checkpoint Candidates",
        w.exec(
            PruningSelectColumns(
                "'superseded by later completed epoch for same symbol/horizon/config; verify references before manual cleanup'",
            "'Inspect metadata before considering manual archive'",
            ExperimentMetadataCommandSql()) +
            "FROM experiment_analysis_result a "
            "JOIN experiment e ON e.experiment_id = a.experiment_id "
            "WHERE e.status = 'completed' "
            "AND a.analysis_status = 'completed' "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "AND a.completed_epochs IS NOT NULL "
            "AND COALESCE(a.model_id, e.last_model_id) IS NOT NULL "
            "AND EXISTS ("
            "  SELECT 1 "
            "  FROM experiment_analysis_result a2 "
            "  JOIN experiment e2 ON e2.experiment_id = a2.experiment_id "
            "  WHERE e2.status = 'completed' "
            "  AND a2.analysis_status = 'completed' "
            "  AND COALESCE(a2.analysis_scope, 'final') = 'final' "
            "  AND a2.symbol = a.symbol "
            "  AND a2.prediction_horizon = a.prediction_horizon "
            "  AND COALESCE(a2.target_epochs, e2.target_epochs) = COALESCE(a.target_epochs, e.target_epochs) "
            "  AND abs(e2.c_next_threshold - e.c_next_threshold) <= 1e-12 "
            "  AND COALESCE(e2.core_lr_mult, '-infinity'::double precision) = COALESCE(e.core_lr_mult, '-infinity'::double precision) "
            "  AND COALESCE(e2.head_lr_mult, '-infinity'::double precision) = COALESCE(e.head_lr_mult, '-infinity'::double precision) "
            "  AND a2.completed_epochs > a.completed_epochs "
            "  AND (a2.leader_score IS NULL OR a.leader_score IS NULL OR a2.leader_score >= a.leader_score) "
            ") "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM experiment ref "
            "  WHERE ref.status IN ('pending', 'running') "
            "  AND (ref.last_model_id = COALESCE(a.model_id, e.last_model_id) "
            "       OR ref.resume_model_id = COALESCE(a.model_id, e.last_model_id))"
            ") "
            "ORDER BY a.symbol ASC, a.prediction_horizon ASC, a.completed_epochs ASC, "
            "a.leader_score ASC NULLS LAST, e.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Dominated Completed Experiments",
        w.exec(
            "WITH eligible AS ("
            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "         a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "         a.accept_accuracy, a.leader_score, e.status, e.phase, e.completed_at "
            "  FROM experiment_analysis_result a "
            "  JOIN experiment e ON e.experiment_id = a.experiment_id "
            "  WHERE e.status = 'completed' "
            "  AND a.analysis_status = 'completed' "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "  AND a.leader_score IS NOT NULL "
            "  AND a.infer_accuracy IS NOT NULL "
            ") "
            "SELECT d.experiment_id, d.model_id, d.symbol, d.prediction_horizon, "
            "d.target_epochs, d.completed_epochs, d.status, d.phase, d.infer_accuracy, "
            "d.accept_rate, d.accept_accuracy, d.leader_score, d.completed_at::text, "
            "('dominated by experiment ' || x.experiment_id::text || "
            "' with higher leader_score and no worse comparable accuracy') AS reason, "
            "'No automatic archive; retain if needed for comparison history' AS suggested_action, "
            "'No direct CLI action.' AS example_command "
            "FROM eligible d "
            "JOIN LATERAL ("
            "  SELECT e2.experiment_id "
            "  FROM eligible e2 "
            "  WHERE e2.symbol = d.symbol "
            "  AND e2.prediction_horizon = d.prediction_horizon "
            "  AND e2.experiment_id <> d.experiment_id "
            "  AND e2.leader_score > d.leader_score "
            "  AND e2.infer_accuracy >= d.infer_accuracy "
            "  AND (d.accept_accuracy IS NULL OR e2.accept_accuracy IS NULL OR "
            "       e2.accept_accuracy >= d.accept_accuracy) "
            "  ORDER BY e2.leader_score DESC, e2.infer_accuracy DESC, e2.experiment_id DESC "
            "  LIMIT 1 "
            ") x ON true "
            "ORDER BY d.leader_score ASC, d.infer_accuracy ASC, d.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Failed Experiment Archive Candidates",
        w.exec(
            PruningSelectColumns(
                "COALESCE('terminal failed run: ' || NULLIF(e.error_message, ''), 'terminal failed run')",
                "'Investigate logs and metadata before manual archive decision'",
                ExperimentMetadataCommandSql()) +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'failed' "
            "ORDER BY e.completed_at DESC NULLS LAST, e.updated_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Cancelled Experiment Archive Candidates",
        w.exec(
            PruningSelectColumns(
                "'terminal cancelled run; no automatic cleanup action'",
                "'No action required unless manually archiving old scheduler rows'",
                "'No direct CLI action.'") +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'cancelled' "
            "ORDER BY e.completed_at DESC NULLS LAST, e.updated_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Duplicate / Superseded Experiment Candidates",
        w.exec(
            "WITH ranked AS ("
            "  SELECT e.*, a.model_id, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "         a.accept_accuracy, a.leader_score, "
            "         row_number() OVER ("
            "           PARTITION BY e.symbol, e.prediction_horizon, e.target_epochs, "
            "                        e.checkpoint_interval, e.train_start, e.train_end, "
            "                        COALESCE(e.infer_start, '-infinity'::timestamptz), "
            "                        COALESCE(e.infer_end, '-infinity'::timestamptz), "
            "                        e.c_next_threshold, "
            "                        COALESCE(e.core_lr_mult, '-infinity'::double precision), "
            "                        COALESCE(e.head_lr_mult, '-infinity'::double precision) "
            "           ORDER BY COALESCE(a.leader_score, '-infinity'::double precision) DESC, "
            "                    COALESCE(a.infer_accuracy, '-infinity'::double precision) DESC, "
            "                    e.experiment_id DESC"
            "         ) AS rn "
            "  FROM experiment e "
            "  LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "  AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "  WHERE e.status IN ('completed', 'failed', 'cancelled') "
            ") "
            "SELECT e.experiment_id, COALESCE(e.model_id, e.last_model_id), e.symbol, "
            "e.prediction_horizon, e.target_epochs, e.completed_epochs, e.status, e.phase, "
            "e.infer_accuracy, e.accept_rate, e.accept_accuracy, e.leader_score, "
            "e.completed_at::text, "
            "'same symbol/horizon/config group has a higher-ranked or newer terminal experiment' AS reason, "
            "'Review duplicate/superseded lineage before manual archive' AS suggested_action, "
            "" + ExperimentMetadataCommandSql() + " AS example_command "
            "FROM ranked e "
            "WHERE e.rn > 1 "
            "ORDER BY e.symbol ASC, e.prediction_horizon ASC, e.target_epochs DESC, "
            "e.rn ASC, e.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Old Pending Experiments To Review",
        w.exec(
            PruningSelectColumns(
                "'pending for more than 7 days; verify still relevant before scheduler capacity is used'",
                "'Inspect current experiment status; cancel only if intentionally obsolete'",
            ExperimentStatusCommandSql()) +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'pending' "
            "AND e.created_at < now() - interval '7 days' "
            "ORDER BY e.created_at ASC, e.experiment_id ASC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Models / Checkpoints With Missing Analysis Metrics",
        w.exec(
            PruningSelectColumns(
                "'completed experiment lacks complete inference/analysis metrics'",
                "CASE WHEN e.phase IN ('infer', 'analyze') AND e.status IN ('pending', 'running') THEN "
                "'Wait for scheduler; missing metrics are expected while queued/running' "
                "WHEN e.last_model_id IS NOT NULL THEN 'Inspect status and consider requeue-analysis if appropriate' "
                "ELSE 'No direct CLI action' END",
                "CASE WHEN e.status IN ('pending', 'running') THEN " + ExperimentStatusCommandSql() + " "
                "WHEN e.last_model_id IS NOT NULL THEN " + ExperimentControlCommandSql("--requeue-analysis") + " "
            "ELSE 'No direct CLI action.' END") +
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'completed' "
            "AND (a.analysis_id IS NULL OR a.infer_accuracy IS NULL OR a.leader_score IS NULL) "
            "ORDER BY e.completed_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 50;"));

    sections.emplace_back(
        "Safe Cleanup Command Suggestions",
        w.exec(
            "SELECT NULL::bigint, NULL::bigint, 'all'::text, NULL::integer, NULL::integer, "
            "NULL::integer, 'n/a'::text, 'n/a'::text, NULL::double precision, "
            "NULL::double precision, NULL::double precision, NULL::double precision, "
            "NULL::text, "
            "'refresh reports after any manual archive review' AS reason, "
            "'Regenerate reports' AS suggested_action, "
            "" + StaticCommandSql("--generate-experiment-reports") + " AS example_command "
            "UNION ALL "
            "SELECT NULL::bigint, NULL::bigint, 'all'::text, NULL::integer, NULL::integer, "
            "NULL::integer, 'n/a'::text, 'n/a'::text, NULL::double precision, "
            "NULL::double precision, NULL::double precision, NULL::double precision, "
            "NULL::text, "
            "'inspect live scheduler state before pruning decisions' AS reason, "
            "'Review scheduler status' AS suggested_action, "
            "" + StaticCommandSql("--scheduler-status") + " AS example_command;"));

    size_t total = 0;
    for (const auto& section : sections)
        total += section.second.size();

    const std::filesystem::path path =
        std::filesystem::path{reportDir} / "pruning_archive_candidates.md";
    WriteTextFile(path.string(), RenderPruningReport(sections));
    std::cout << "EXPERIMENT_PRUNING_REPORT_GENERATED"
              << ",path=" << path.string()
              << ",candidates=" << total
              << std::endl;
    return total;
}

void WriteReportFile(const std::string& reportDir,
                     const std::string& fileName,
                     const std::string& title,
                     const pqxx::result& rows)
{
    const std::filesystem::path path = std::filesystem::path{reportDir} / fileName;
    WriteTextFile(path.string(), RenderMarkdownReport(title, ExperimentReportHeaders(), rows));
}

size_t WriteIndexReport(pqxx::work& w,
                        const std::string& reportDir,
                        const pqxx::result& latestLeaderboard,
                        const pqxx::result& bestBySymbol,
                        const pqxx::result& bestByHorizon)
{
    const pqxx::result generatedAt = w.exec("SELECT now()::text;");
    const pqxx::result counts = w.exec(
        "SELECT "
        "COUNT(*) AS total_experiments, "
        "COUNT(*) FILTER (WHERE status = 'completed') AS completed_experiments, "
        "COUNT(*) FILTER (WHERE status = 'failed') AS failed_experiments, "
        "COUNT(*) FILTER (WHERE status IN ('pending', 'running')) AS pending_running_experiments "
        "FROM experiment;");
    SchedulerServiceComposition services{w};
    const QueueSnapshot queue = LoadQueueSnapshot(services.admission);

    std::ostringstream out;
    out << "# Experiment Reports Dashboard\n\n";
    out << "Generated: " << (generatedAt.empty() ? "unknown" : RowCell(generatedAt[0], 0)) << "\n\n";

    out << "## Report Links\n\n";
    out << "- [Latest Leaderboard](latest_leaderboard.md)\n";
    out << "- [Best By Symbol](best_by_symbol.md)\n";
    out << "- [Best By Horizon](best_by_horizon.md)\n";
    out << "- [Recent Completed Experiments](recent_completed.md)\n";
    out << "- [Failed Experiments](failures.md)\n";
    out << "- [Recommendations](recommendations.md)\n";
    out << "- [Pruning / Archive Candidates](pruning_archive_candidates.md)\n\n";

    out << "## Summary\n\n";
    out << "| metric | value |\n";
    out << "|---|---|\n";
    if (counts.empty())
    {
        out << "| total_experiments | unknown |\n";
        out << "| completed_experiments | unknown |\n";
        out << "| failed_experiments | unknown |\n";
        out << "| pending_running_experiments | unknown |\n";
    }
    else
    {
        out << "| total_experiments | " << RowCell(counts[0], 0) << " |\n";
        out << "| completed_experiments | " << RowCell(counts[0], 1) << " |\n";
        out << "| failed_experiments | " << RowCell(counts[0], 2) << " |\n";
        out << "| pending_running_experiments | " << RowCell(counts[0], 3) << " |\n";
    }
    out << "\n";

    out << "## Current Overall Leader\n\n";
    if (latestLeaderboard.empty())
        out << "none\n\n";
    else
        out << RenderCompactLeaderTable(latestLeaderboard, 1) << "\n";

    out << "## Best By Symbol Summary\n\n";
    if (bestBySymbol.empty())
        out << "none\n\n";
    else
        out << RenderCompactLeaderTable(bestBySymbol) << "\n";

    out << "## Best By Horizon Summary\n\n";
    if (bestByHorizon.empty())
        out << "none\n\n";
    else
        out << RenderCompactLeaderTable(bestByHorizon) << "\n";

    out << "## Current Queue / Running Summary\n\n";
    out << "| phase | pending | running |\n";
    out << "|---|---|---|\n";
    out << "| train | " << queue.pendingTrain << " | " << queue.runningTrain << " |\n";
    out << "| infer | " << queue.pendingInfer << " | " << queue.runningInfer << " |\n";
    out << "| analyze | " << queue.pendingAnalyze << " | " << queue.runningAnalyze << " |\n\n";

    out << "## Process Note\n\n";
    out << "Unmanaged process detection is available from `--scheduler-status`; this index does not "
        << "scan OS processes during report generation.\n";

    const std::filesystem::path path = std::filesystem::path{reportDir} / "index.md";
    WriteTextFile(path.string(), out.str());
    std::cout << "EXPERIMENT_REPORT_INDEX_GENERATED"
              << ",path=" << path.string()
              << std::endl;
    return 1;
}

int GenerateExperimentReports(const std::string& reportDir, bool warnOnly)
{
    try
    {
        std::filesystem::create_directories(reportDir);

        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        if (!RequireSchedulerTables(w))
            return warnOnly ? 0 : 1;

        const pqxx::result latestLeaderboard = ExecReportQuery(
            w,
            "ORDER BY a.leader_score DESC NULLS LAST, a.infer_accuracy DESC NULLS LAST, e.experiment_id DESC",
            "LIMIT 100");
        const pqxx::result bestBySymbol = w.exec(
            "SELECT DISTINCT ON (a.symbol) e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
            "FROM experiment_analysis_result a "
            "JOIN experiment e ON e.experiment_id = a.experiment_id "
            "WHERE a.analysis_status = 'completed' "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "ORDER BY a.symbol ASC, a.leader_score DESC NULLS LAST, a.infer_accuracy DESC NULLS LAST, e.experiment_id DESC;");
        const pqxx::result bestByHorizon = w.exec(
            "SELECT DISTINCT ON (a.prediction_horizon) e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
            "a.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
            "FROM experiment_analysis_result a "
            "JOIN experiment e ON e.experiment_id = a.experiment_id "
            "WHERE a.analysis_status = 'completed' "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "ORDER BY a.prediction_horizon ASC, a.leader_score DESC NULLS LAST, a.infer_accuracy DESC NULLS LAST, e.experiment_id DESC;");
        const pqxx::result recentCompleted = ExecReportQuery(
            w,
            "ORDER BY e.completed_at DESC NULLS LAST, e.experiment_id DESC",
            "LIMIT 100");
        const pqxx::result failures = w.exec(
            "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), e.symbol, e.prediction_horizon, "
            "e.target_epochs, a.completed_epochs, a.infer_accuracy, a.accept_rate, "
            "a.accept_accuracy, a.leader_score, COALESCE(a.analysis_status, e.status), e.completed_at::text "
            "FROM experiment e "
            "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
            "AND COALESCE(a.analysis_scope, 'final') = 'final' "
            "WHERE e.status = 'failed' "
            "ORDER BY e.completed_at DESC NULLS LAST, e.updated_at DESC NULLS LAST, e.experiment_id DESC "
            "LIMIT 100;");
        const RecommendationPriorityCounts recommendationRows = WriteRecommendationReport(w, reportDir);
        const size_t pruningRows = WritePruningArchiveReport(w, reportDir);
        const size_t indexRows = WriteIndexReport(w,
                                                  reportDir,
                                                  latestLeaderboard,
                                                  bestBySymbol,
                                                  bestByHorizon);
        w.commit();

        WriteReportFile(reportDir, "latest_leaderboard.md", "Latest Leaderboard", latestLeaderboard);
        WriteReportFile(reportDir, "best_by_symbol.md", "Best By Symbol", bestBySymbol);
        WriteReportFile(reportDir, "best_by_horizon.md", "Best By Horizon", bestByHorizon);
        WriteReportFile(reportDir, "recent_completed.md", "Recent Completed Experiments", recentCompleted);
        WriteReportFile(reportDir, "failures.md", "Failed Experiments", failures);

        std::cout << "EXPERIMENT_REPORTS_GENERATED"
                  << ",dir=" << reportDir
                  << ",latest_leaderboard_rows=" << latestLeaderboard.size()
                  << ",best_by_symbol_rows=" << bestBySymbol.size()
                  << ",best_by_horizon_rows=" << bestByHorizon.size()
                  << ",recent_completed_rows=" << recentCompleted.size()
                  << ",failure_rows=" << failures.size()
                  << ",recommendation_rows=" << recommendationRows.Total()
                  << ",pruning_candidate_rows=" << pruningRows
                  << ",index_rows=" << indexRows
                  << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXPERIMENT_REPORT_GENERATION_WARNING"
                  << ",error=" << e.what()
                  << std::endl;
        return warnOnly ? 0 : 1;
    }
}

void TryGenerateExperimentReports(const SchedulerOptions& options)
{
    if (!options.autoGenerateReports)
        return;
    (void)GenerateExperimentReports(options.experimentReportDir, true);
}

void UpsertAnalysisResult(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 const ParsedMetrics& metrics,
                                 const std::optional<double>& leaderScore,
                                 const AnalysisScopeOptions& scopeOptions)
{
    const bool hasAnalysisScope = ColumnExists(w, "experiment_analysis_result", "analysis_scope");
    const bool hasCheckpointEvalId = ColumnExists(w, "experiment_analysis_result", "checkpoint_eval_id");
    const bool hasCheckpointEpoch = ColumnExists(w, "experiment_analysis_result", "checkpoint_epoch");
    const bool hasParentExperimentId = ColumnExists(w, "experiment_analysis_result", "parent_experiment_id");
    if (scopeOptions.scope == "checkpoint" &&
        (!hasAnalysisScope || !hasCheckpointEvalId || !hasCheckpointEpoch || !hasParentExperimentId))
    {
        throw std::runtime_error("checkpoint analysis scope migration required");
    }

    long long actual[3] = {};
    long long pred[3] = {};
    if (metrics.hasConfusion)
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int p = 0; p < 3; ++p)
            {
                actual[a] += metrics.confusion[a][p];
                pred[p] += metrics.confusion[a][p];
            }
        }
    }

    if (scopeOptions.scope == "final" && !metrics.modelId.has_value())
    {
        w.exec_params(
            "DELETE FROM experiment_analysis_result "
            "WHERE experiment_id = $1 AND model_id IS NULL "
            "AND COALESCE(analysis_scope, 'final') = 'final';",
            experiment.experimentId);
    }

    std::string analysisNotes = metrics.rejectReason.value_or("");
    if (scopeOptions.scope == "checkpoint")
    {
        std::ostringstream notes;
        notes << "checkpoint_analysis"
              << ";parent_experiment_id=" << scopeOptions.parentExperimentId.value_or(experiment.experimentId)
              << ";checkpoint_eval_id=" << scopeOptions.checkpointEvalId.value_or(-1)
              << ";checkpoint_epoch=" << scopeOptions.checkpointEpoch.value_or(-1);
        if (!analysisNotes.empty())
            notes << ";reject_reason=" << analysisNotes;
        analysisNotes = notes.str();
    }

    std::ostringstream sql;
    sql << "INSERT INTO experiment_analysis_result ("
        << "experiment_id, model_id, symbol, prediction_horizon, target_epochs, completed_epochs, "
        << "train_accuracy, validation_accuracy, infer_accuracy, "
        << "actual_down_count, actual_neutral_count, actual_up_count, "
        << "pred_down_count, pred_neutral_count, pred_up_count, "
        << "confusion_down_down, confusion_down_neutral, confusion_down_up, "
        << "confusion_neutral_down, confusion_neutral_neutral, confusion_neutral_up, "
        << "confusion_up_down, confusion_up_neutral, confusion_up_up, "
        << "accept_count, accept_rate, accept_accuracy, reject_count, loss_last, "
        << "best_metric_name, best_metric_value, leader_score, analysis_status, analysis_notes, "
        << "source_train_log_path, source_infer_log_path, updated_at"
        << (hasAnalysisScope ? ", analysis_scope" : "")
        << (hasCheckpointEvalId ? ", checkpoint_eval_id" : "")
        << (hasCheckpointEpoch ? ", checkpoint_epoch" : "")
        << (hasParentExperimentId ? ", parent_experiment_id" : "")
        << ") VALUES ("
        << experiment.experimentId << ","
        << MetricSql(w, metrics.modelId) << ","
        << w.quote(experiment.symbol) << ","
        << experiment.predictionHorizon << ","
        << experiment.targetEpochs << ","
        << MetricSql(w, metrics.completedEpochs) << ","
        << MetricSql(w, metrics.trainAccuracy) << ","
        << MetricSql(w, metrics.validationAccuracy) << ","
        << MetricSql(w, metrics.inferAccuracy) << ","
        << (metrics.hasConfusion ? std::to_string(actual[0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(actual[1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(actual[2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][2]) : "NULL") << ","
        << (metrics.acceptModel.has_value() && *metrics.acceptModel ? "1" : "0") << ","
        << MetricSql(w, metrics.acceptRate) << ","
        << MetricSql(w, metrics.acceptAccuracy) << ","
        << (metrics.acceptModel.has_value() && !*metrics.acceptModel ? "1" : "0") << ","
        << MetricSql(w, metrics.lossLast) << ","
        << w.quote("infer_accuracy") << ","
        << MetricSql(w, metrics.inferAccuracy) << ","
        << MetricSql(w, leaderScore) << ","
        << w.quote("completed") << ","
        << MetricSql(w, analysisNotes) << ","
        << MetricSql(w, experiment.trainLogPath) << ","
        << MetricSql(w, experiment.inferLogPath) << ","
        << "now()";
    if (hasAnalysisScope)
        sql << "," << w.quote(scopeOptions.scope);
    if (hasCheckpointEvalId)
        sql << "," << MetricSql(w, scopeOptions.checkpointEvalId);
    if (hasCheckpointEpoch)
        sql << "," << MetricSql(w, scopeOptions.checkpointEpoch);
    if (hasParentExperimentId)
        sql << "," << MetricSql(w, scopeOptions.parentExperimentId);
    sql << ") ";

    if (scopeOptions.scope == "checkpoint")
    {
        sql << "ON CONFLICT (checkpoint_eval_id) WHERE analysis_scope = 'checkpoint' AND checkpoint_eval_id IS NOT NULL DO UPDATE SET ";
    }
    else if (hasAnalysisScope)
    {
        sql << "ON CONFLICT (experiment_id, model_id) WHERE analysis_scope = 'final' DO UPDATE SET ";
    }
    else
    {
        sql << "ON CONFLICT (experiment_id, model_id) DO UPDATE SET ";
    }
    sql
        << "symbol = EXCLUDED.symbol, "
        << "prediction_horizon = EXCLUDED.prediction_horizon, "
        << "target_epochs = EXCLUDED.target_epochs, "
        << "completed_epochs = EXCLUDED.completed_epochs, "
        << "train_accuracy = EXCLUDED.train_accuracy, "
        << "validation_accuracy = EXCLUDED.validation_accuracy, "
        << "infer_accuracy = EXCLUDED.infer_accuracy, "
        << "actual_down_count = EXCLUDED.actual_down_count, "
        << "actual_neutral_count = EXCLUDED.actual_neutral_count, "
        << "actual_up_count = EXCLUDED.actual_up_count, "
        << "pred_down_count = EXCLUDED.pred_down_count, "
        << "pred_neutral_count = EXCLUDED.pred_neutral_count, "
        << "pred_up_count = EXCLUDED.pred_up_count, "
        << "confusion_down_down = EXCLUDED.confusion_down_down, "
        << "confusion_down_neutral = EXCLUDED.confusion_down_neutral, "
        << "confusion_down_up = EXCLUDED.confusion_down_up, "
        << "confusion_neutral_down = EXCLUDED.confusion_neutral_down, "
        << "confusion_neutral_neutral = EXCLUDED.confusion_neutral_neutral, "
        << "confusion_neutral_up = EXCLUDED.confusion_neutral_up, "
        << "confusion_up_down = EXCLUDED.confusion_up_down, "
        << "confusion_up_neutral = EXCLUDED.confusion_up_neutral, "
        << "confusion_up_up = EXCLUDED.confusion_up_up, "
        << "accept_count = EXCLUDED.accept_count, "
        << "accept_rate = EXCLUDED.accept_rate, "
        << "accept_accuracy = EXCLUDED.accept_accuracy, "
        << "reject_count = EXCLUDED.reject_count, "
        << "loss_last = EXCLUDED.loss_last, "
        << "best_metric_name = EXCLUDED.best_metric_name, "
        << "best_metric_value = EXCLUDED.best_metric_value, "
        << "leader_score = EXCLUDED.leader_score, "
        << "analysis_status = EXCLUDED.analysis_status, "
        << "analysis_notes = EXCLUDED.analysis_notes, "
        << "source_train_log_path = EXCLUDED.source_train_log_path, "
        << "source_infer_log_path = EXCLUDED.source_infer_log_path, "
        << "updated_at = now()";
    if (hasAnalysisScope)
        sql << ", analysis_scope = EXCLUDED.analysis_scope";
    if (hasCheckpointEvalId)
        sql << ", checkpoint_eval_id = EXCLUDED.checkpoint_eval_id";
    if (hasCheckpointEpoch)
        sql << ", checkpoint_epoch = EXCLUDED.checkpoint_epoch";
    if (hasParentExperimentId)
        sql << ", parent_experiment_id = EXCLUDED.parent_experiment_id";
    sql << ";";
    w.exec(sql.str());
}

std::string JoinCheckpointPolicyRules(const std::vector<std::string>& rules)
{
    if (rules.empty())
        return "none";
    std::ostringstream out;
    for (size_t i = 0; i < rules.size(); ++i)
    {
        if (i)
            out << "|";
        out << rules[i];
    }
    return out.str();
}

CheckpointPolicyEvaluationResult EvaluateCheckpointPolicyAfterAnalysis(
    pqxx::work& transaction,
    const CheckpointEvalRow& eval)
{
    using EA::SchedulerCore::CheckpointEvaluationOperations;
    using EA::SchedulerCore::CheckpointEvaluationRecord;
    using EA::SchedulerCore::CheckpointEvaluationService;

    const CheckpointEvaluationRecord evaluation{
        eval.checkpointEvalId,
        eval.experiment.experimentId,
        eval.checkpointModelId,
        eval.checkpointEpoch,
        eval.experiment.symbol,
        eval.experiment.predictionHorizon};
    EA::SchedulerCore::PostgresSchedulerRepository repository{transaction};
    CheckpointEvaluationOperations operations{
        [&] { return repository.checkpointPolicySchemaAvailable(); },
        [&](long long parentExperimentId) {
            return repository.loadCheckpointPolicyForEvaluation(
                parentExperimentId);
        },
        [&](const CheckpointEvaluationRecord& record,
            CheckpointPolicyConfig config) {
            return repository.reconcileCheckpointPolicyIdentity(
                record, std::move(config));
        },
        [&](const CheckpointEvaluationRecord& record) {
            return repository.loadCheckpointPolicyEvidence(record);
        },
        [&](long long parentExperimentId) {
            return repository.loadCompletedCheckpointPolicyPopulation(
                parentExperimentId);
        },
        [&](const CheckpointEvaluationRecord& record,
            const CheckpointPolicyConfig& config) {
            return repository.loadCheckpointPolicyRankPopulation(
                record, config);
        },
        [&](const CheckpointEvaluationRecord& record,
            const CheckpointPolicyConfig& config,
            const CheckpointPolicyDecision& decision,
            const ValidatedCheckpointPolicyEvidence& evidence,
            const CheckpointPolicyEvidenceIdentity& identity) {
            return repository.persistCheckpointPolicyDecision(
                record, config, decision, evidence, identity);
        },
        [&](const CheckpointEvaluationRecord& record,
            const CheckpointPolicyConfig& config,
            const CheckpointPolicyDecision& decision,
            const PersistedCheckpointPolicyDecision& persisted,
            const std::string& evidenceWatermark) {
            return repository.applyCheckpointPolicyStopRequest(
                record,
                config,
                decision,
                persisted,
                evidenceWatermark);
        }};
    CheckpointEvaluationService service{
        std::move(operations), std::cout};
    return service.evaluate(evaluation);
}

std::string StableFnv1aHash(const std::string& value)
{
    return StableContinuationPolicyHash(value);
}

ContinuationProfitabilityEvidence MapContinuationProfitabilityEvidence(
    const EA::InferenceProfitability::Observation& observation)
{
    ContinuationProfitabilityEvidence evidence;
    evidence.observationId = observation.observationId;
    evidence.observationIdentityHash =
        observation.observationIdentityHash;
    evidence.inferenceEvalResultId =
        observation.provenance.inferenceEvalResultId;
    evidence.inferenceScope = EA::InferenceProfitability::ScopeText(
        observation.provenance.scope);
    evidence.checkpointEvalId =
        observation.provenance.checkpointEvalId;
    evidence.metricDefinitionHash = observation.metricDefinitionHash;
    evidence.sourceContentHash = observation.sourceContentHash;
    evidence.predictionCount = observation.statistics.predictionCount;
    evidence.actionableCount = observation.statistics.actionableCount;
    evidence.winningActionableCount =
        observation.statistics.winningActionableCount;
    evidence.losingActionableCount =
        observation.statistics.losingActionableCount;
    evidence.grossPositiveTerminalHorizonLogReturnSum =
        observation.statistics
            .grossPositiveTerminalHorizonLogReturnSum;
    evidence.grossNegativeTerminalHorizonLogReturnSum =
        observation.statistics
            .grossNegativeTerminalHorizonLogReturnSum;
    evidence.aggregateTerminalHorizonLogReturnSum =
        observation.statistics.aggregateTerminalHorizonLogReturnSum;
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction =
        observation
            .averageTerminalHorizonLogReturnPerActionablePrediction;
    return evidence;
}

void AttachContinuationProfitabilityEvidence(
    pqxx::work& w,
    long long sourceExperimentId,
    bool profitabilitySchemaAvailable,
    ContinuationEvidence& evidence)
{
    evidence.profitability.reset();
    if (!evidence.inferenceEvalResultId.has_value())
    {
        if (evidence.profitabilityUnavailableReason.empty() ||
            evidence.profitabilityUnavailableReason ==
                "no_selected_continuation_source")
        {
            evidence.profitabilityUnavailableReason =
                "no_completed_inference_result";
        }
        return;
    }
    if (!profitabilitySchemaAvailable)
    {
        evidence.profitabilityUnavailableReason =
            "profitability_schema_unavailable";
        return;
    }

    try
    {
        EA::InferenceProfitability::AuthoritativeObservationSelector selector;
        selector.experimentId = sourceExperimentId;
        selector.modelId = evidence.modelId;
        selector.inferenceEvalResultId = *evidence.inferenceEvalResultId;
        selector.scope = evidence.analysisScope == "checkpoint"
            ? EA::InferenceProfitability::Scope::checkpointInference
            : EA::InferenceProfitability::Scope::finalInference;
        selector.checkpointEvalId = evidence.checkpointEvalId;
        selector.metricDefinitionCanonical =
            EA::InferenceProfitability::kMetricDefinitionCanonical;
        selector.metricDefinitionHash =
            EA::InferenceProfitability::MetricDefinitionHash();
        const auto selection =
            EA::InferenceProfitability::SelectAuthoritativeObservation(
                w,
                selector);
        if (!selection.observation.has_value())
        {
            evidence.profitabilityUnavailableReason =
                EA::InferenceProfitability::
                    AuthoritativeObservationStatusText(selection.status);
            return;
        }
        evidence.profitability = MapContinuationProfitabilityEvidence(
            *selection.observation);
        evidence.profitabilityUnavailableReason.clear();
    }
    catch (const std::exception&)
    {
        // Preserve the continuation evidence point. A configured Phase 2B
        // profitability gate consumes this unavailable state fail-closed.
        evidence.profitabilityUnavailableReason =
            "profitability_lookup_error";
    }
}

std::vector<ContinuationEvidence> LoadContinuationEvidence(
    pqxx::work& w,
    const ContinuationPolicyConfig& config)
{
    pqxx::result rows = w.exec_params(
        "WITH cfg AS ("
        "  SELECT model_id, round(max(value) FILTER (WHERE col_idx = 10))::integer AS completed_epoch "
        "  FROM matrix "
        "  WHERE param_name = 'train_config_meta' AND row_idx = 0 "
        "  GROUP BY model_id "
        "  HAVING count(DISTINCT col_idx) FILTER (WHERE col_idx BETWEEN 0 AND 13) >= 14"
        ") "
        "SELECT a.analysis_id, ce.checkpoint_eval_id, a.model_id, cfg.completed_epoch, "
        "       a.leader_score, a.infer_accuracy, ir.accept_model, a.analysis_scope, "
        "       COALESCE(ce.analyze_completed_at, ce.completed_at, a.updated_at)::text, "
        "       a.updated_at::text, ir.id "
        "FROM experiment e "
        "JOIN experiment_checkpoint_eval ce ON ce.parent_experiment_id = e.experiment_id "
        "JOIN experiment_analysis_result a ON a.analysis_id = ce.analysis_id "
        "JOIN model m ON m.model_id = ce.checkpoint_model_id AND m.experiment_id = e.experiment_id "
        "JOIN cfg ON cfg.model_id = m.model_id "
        "LEFT JOIN inference_eval_result ir "
        "  ON ir.checkpoint_eval_id = ce.checkpoint_eval_id "
        " AND ir.model_id = ce.checkpoint_model_id "
        " AND ir.inference_scope = 'checkpoint' AND ir.status = 'completed' "
        "WHERE e.experiment_id = $1 "
        "AND ce.status = 'completed' AND ce.phase = 'done' "
        "AND ce.symbol = e.symbol AND ce.prediction_horizon = e.prediction_horizon "
        "AND a.analysis_scope = 'checkpoint' AND a.analysis_status = 'completed' "
        "AND a.checkpoint_eval_id = ce.checkpoint_eval_id "
        "AND a.parent_experiment_id = e.experiment_id "
        "AND a.experiment_id = e.experiment_id "
        "AND a.model_id = ce.checkpoint_model_id "
        "AND a.checkpoint_epoch = ce.checkpoint_epoch "
        "AND a.completed_epochs = ce.checkpoint_epoch "
        "AND a.symbol = e.symbol AND a.prediction_horizon = e.prediction_horizon "
        "AND cfg.completed_epoch = ce.checkpoint_epoch "
        "UNION ALL "
        "SELECT a.analysis_id, NULL::bigint, a.model_id, cfg.completed_epoch, "
        "       a.leader_score, a.infer_accuracy, NULL::boolean, a.analysis_scope, "
        "       a.updated_at::text, a.updated_at::text, NULL::bigint "
        "FROM experiment e "
        "JOIN experiment_analysis_result a "
        "  ON a.experiment_id = e.experiment_id AND a.model_id = e.last_model_id "
        "JOIN model m ON m.model_id = a.model_id AND m.experiment_id = e.experiment_id "
        "JOIN cfg ON cfg.model_id = m.model_id "
        "WHERE e.experiment_id = $1 "
        "AND a.analysis_scope = 'final' AND a.analysis_status = 'completed' "
        "AND a.checkpoint_eval_id IS NULL AND a.parent_experiment_id IS NULL "
        "AND a.checkpoint_epoch IS NULL "
        "AND a.completed_epochs = cfg.completed_epoch "
        "AND a.symbol = e.symbol AND a.prediction_horizon = e.prediction_horizon "
        "ORDER BY completed_epoch ASC, 9 ASC, checkpoint_eval_id ASC NULLS LAST, analysis_id ASC;",
        config.sourceExperimentId);

    const bool profitabilitySchemaAvailable =
        EA::InferenceProfitability::SchemaExists(w);
    std::vector<ContinuationEvidence> evidence;
    evidence.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        ContinuationEvidence point;
        point.analysisId = row[0].as<long long>();
        point.checkpointEvalId = OptionalLongLongCell(row, 1);
        point.modelId = row[2].as<long long>();
        point.completedEpoch = row[3].as<int>();
        point.leaderScore = OptionalDoubleCell(row, 4);
        point.inferAccuracy = OptionalDoubleCell(row, 5);
        if (!row[6].is_null())
            point.acceptModel = row[6].as<bool>();
        point.analysisScope = row[7].as<std::string>();
        point.completedAt = row[8].as<std::string>();
        point.updatedAt = row[9].as<std::string>();
        point.inferenceEvalResultId = OptionalLongLongCell(row, 10);
        if (point.analysisScope == "final")
        {
            const auto finalInference =
                EA::InferenceProfitability::ResolveExactFinalInferenceResult(
                    w,
                    config.sourceExperimentId,
                    point.modelId);
            point.inferenceEvalResultId =
                finalInference.inferenceEvalResultId;
            point.acceptModel = finalInference.acceptModel;
            point.profitabilityUnavailableReason =
                EA::InferenceProfitability::
                    ExactFinalInferenceResultStatusText(
                        finalInference.status);
            if (finalInference.status ==
                EA::InferenceProfitability::
                    ExactFinalInferenceResultStatus::available)
            {
                point.profitabilityUnavailableReason.clear();
            }
        }
        AttachContinuationProfitabilityEvidence(
            w,
            config.sourceExperimentId,
            profitabilitySchemaAvailable,
            point);
        evidence.push_back(std::move(point));
    }
    return evidence;
}

void RefreshContinuationSelectedDiagnostics(
    ContinuationEvidence& selected,
    const std::vector<ContinuationEvidence>& evidence)
{
    selected.inferenceEvalResultId.reset();
    selected.profitability.reset();
    selected.profitabilityUnavailableReason =
        "selected_continuation_evidence_not_available";
    for (const ContinuationEvidence& point : evidence)
    {
        if (point.analysisId == selected.analysisId &&
            point.modelId == selected.modelId &&
            point.checkpointEvalId == selected.checkpointEvalId)
        {
            selected.analysisScope = point.analysisScope;
            selected.inferenceEvalResultId = point.inferenceEvalResultId;
            selected.profitability = point.profitability;
            selected.profitabilityUnavailableReason =
                point.profitabilityUnavailableReason;
            return;
        }
    }
}

bool ValidateContinuationResumeSource(pqxx::work& w,
                                      const ContinuationPolicyConfig& config,
                                      const ContinuationEvidence& selected,
                                      QueueResumeMeta* loadedMeta,
                                      std::string& reason)
{
    pqxx::result owner = w.exec_params(
        "SELECT experiment_id FROM model WHERE model_id = $1;",
        selected.modelId);
    if (owner.size() != 1 || owner[0][0].is_null() ||
        owner[0][0].as<long long>() != config.sourceExperimentId)
    {
        reason = "source_model_ownership_mismatch";
        return false;
    }

    QueueResumeMeta meta;
    try
    {
        meta = LoadQueueResumeMeta(w, selected.modelId);
    }
    catch (const std::exception& e)
    {
        reason = "source_model_not_resumable:" + std::string{e.what()};
        return false;
    }

    if (meta.completedEpochs != selected.completedEpoch)
    {
        reason = "source_model_completed_epoch_mismatch";
        return false;
    }
    if (!config.targetEpochs.has_value() || meta.completedEpochs >= *config.targetEpochs)
    {
        reason = "source_model_epoch_not_below_target";
        return false;
    }
    if (meta.symbol != config.source.symbol ||
        meta.predictionHorizon != config.source.predictionHorizon ||
        std::fabs(meta.threshold - config.source.cNextThreshold) > 1e-7 ||
        !SameDate(meta.trainStart, config.source.trainStart) ||
        !SameDate(meta.trainEnd, config.source.trainEnd))
    {
        reason = "source_model_resume_metadata_mismatch";
        return false;
    }
    const pqxx::result sourceMask = w.exec_params(
        "SELECT feature_ablation_mask FROM experiment WHERE experiment_id=$1;",
        config.sourceExperimentId);
    if (sourceMask.size() != 1)
    {
        reason = "source_experiment_feature_ablation_mask_missing";
        return false;
    }
    const std::string canonicalSourceMask = EA::FeatureAblationMask::Parse(
        sourceMask[0][0].as<std::string>()).CanonicalText();
    if (canonicalSourceMask != meta.featureAblationMask)
    {
        reason = "source_model_feature_ablation_mask_mismatch";
        return false;
    }
    if (loadedMeta)
        *loadedMeta = meta;
    return true;
}

std::optional<long long> FindEquivalentContinuationExperiment(
    pqxx::work& w,
    long long sourceExperimentId,
    long long sourceModelId,
    int targetEpochs)
{
    (void)DBIO::PgModelIO::validateModelInputSemanticsForLoad(
        w, sourceModelId);
    const std::size_t sourceInputWidth =
        DBIO::PgModelIO::loadRequiredModelMeta(
            w, sourceModelId).inputWidth;
    pqxx::result rows = w.exec_params(
        "SELECT experiment_id FROM experiment "
        "WHERE experiment_id <> $1 "
        "AND resume_model_id = $2 "
        "AND target_epochs = $3 "
        "AND model_input_width = $4 "
        "AND model_input_semantic_layout_version = $5 "
        "AND economic_calendar_snapshot_id IS NOT DISTINCT FROM "
        "(SELECT economic_calendar_snapshot_id FROM model WHERE model_id=$2) "
        "AND economic_calendar_snapshot_hash IS NOT DISTINCT FROM "
        "(SELECT economic_calendar_snapshot_hash FROM model WHERE model_id=$2) "
        "ORDER BY (continuation_source_model_id IS NOT NULL) DESC, experiment_id ASC "
        "LIMIT 1;",
        sourceExperimentId,
        sourceModelId,
        targetEpochs,
        sourceInputWidth,
        EA::kModelInputSemanticLayoutVersion);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

struct ContinuationRankingCandidate
{
    long long sourceExperimentId = -1;
    ContinuationEvidence selected;
};

struct ContinuationRankResult
{
    std::optional<int> rankValue;
    std::string populationWatermark;
};

bool BetterContinuationRankCandidate(const ContinuationRankingCandidate& lhs,
                                     const ContinuationRankingCandidate& rhs)
{
    if (lhs.selected.leaderScore.has_value() != rhs.selected.leaderScore.has_value())
        return lhs.selected.leaderScore.has_value();
    if (lhs.selected.leaderScore.has_value() &&
        *lhs.selected.leaderScore != *rhs.selected.leaderScore)
    {
        return *lhs.selected.leaderScore > *rhs.selected.leaderScore;
    }
    if (lhs.selected.inferAccuracy.has_value() != rhs.selected.inferAccuracy.has_value())
        return lhs.selected.inferAccuracy.has_value();
    if (lhs.selected.inferAccuracy.has_value() &&
        *lhs.selected.inferAccuracy != *rhs.selected.inferAccuracy)
    {
        return *lhs.selected.inferAccuracy > *rhs.selected.inferAccuracy;
    }
    if (lhs.selected.completedEpoch != rhs.selected.completedEpoch)
        return lhs.selected.completedEpoch > rhs.selected.completedEpoch;
    if (lhs.sourceExperimentId != rhs.sourceExperimentId)
        return lhs.sourceExperimentId < rhs.sourceExperimentId;
    return lhs.selected.modelId < rhs.selected.modelId;
}

ContinuationRankResult RankContinuationSource(
    pqxx::work& w,
    const ContinuationPolicyConfig& evaluatedConfig,
    const ContinuationEvidence& evaluatedSelection)
{
    ContinuationRankResult result;
    std::ostringstream sql;
    sql << "SELECT experiment_id FROM experiment "
        << "WHERE status = 'completed' AND phase = 'done' ";
    if (!evaluatedConfig.includeExcluded)
        sql << "AND continuation_candidate_excluded = false ";
    if (evaluatedConfig.scope == "symbol_horizon")
    {
        sql << "AND symbol = " << w.quote(evaluatedConfig.source.symbol)
            << " AND prediction_horizon = " << evaluatedConfig.source.predictionHorizon << " ";
    }
    else if (evaluatedConfig.scope == "horizon")
    {
        sql << "AND prediction_horizon = " << evaluatedConfig.source.predictionHorizon << " ";
    }
    else if (evaluatedConfig.scope != "global")
    {
        return result;
    }
    sql << "ORDER BY experiment_id ASC;";

    pqxx::result ids = w.exec(sql.str());
    std::vector<ContinuationRankingCandidate> candidates;
    for (const pqxx::row& idRow : ids)
    {
        const long long candidateId = idRow[0].as<long long>();
        std::optional<ContinuationPolicyConfig> loaded =
            FindContinuationPolicyConfig(w, candidateId);
        if (!loaded.has_value())
            continue;

        ContinuationPolicyConfig candidateConfig = *loaded;
        candidateConfig.targetEpochs = evaluatedConfig.targetEpochs;
        candidateConfig.sourceMode = evaluatedConfig.sourceMode;
        candidateConfig.includeExcluded = evaluatedConfig.includeExcluded;

        const std::vector<ContinuationEvidence> raw =
            LoadContinuationEvidence(w, candidateConfig);
        const std::vector<ContinuationEvidence> distinct =
            DeduplicateContinuationEvidence(raw);
        if (static_cast<int>(distinct.size()) < evaluatedConfig.minEvals)
            continue;
        if (evaluatedConfig.trendMode != "none" &&
            static_cast<int>(distinct.size()) < evaluatedConfig.patience)
        {
            continue;
        }

        const std::optional<ContinuationEvidence> selected =
            SelectContinuationSourceEvidence(candidateConfig, raw);
        if (!selected.has_value())
            continue;
        if (!selected->leaderScore.has_value() && !selected->inferAccuracy.has_value())
            continue;

        std::string resumeReason;
        if (!ValidateContinuationResumeSource(
                w,
                candidateConfig,
                *selected,
                nullptr,
                resumeReason))
        {
            continue;
        }

        if (candidateId != evaluatedConfig.sourceExperimentId &&
            FindEquivalentContinuationExperiment(
                w,
                candidateId,
                selected->modelId,
                *evaluatedConfig.targetEpochs).has_value())
        {
            continue;
        }

        candidates.push_back(ContinuationRankingCandidate{candidateId, *selected});
    }

    std::sort(candidates.begin(), candidates.end(), BetterContinuationRankCandidate);
    std::ostringstream population;
    population << "count=" << candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        population << "|source=" << candidates[i].sourceExperimentId
                   << ":model=" << candidates[i].selected.modelId
                   << ":epoch=" << candidates[i].selected.completedEpoch
                   << ":analysis=" << candidates[i].selected.analysisId
                   << ":leader="
                   << ContinuationOptionalDoubleText(candidates[i].selected.leaderScore)
                   << ":infer="
                   << ContinuationOptionalDoubleText(candidates[i].selected.inferAccuracy);
        if (candidates[i].sourceExperimentId == evaluatedConfig.sourceExperimentId &&
            candidates[i].selected.modelId == evaluatedSelection.modelId)
        {
            result.rankValue = static_cast<int>(i + 1);
        }
    }
    result.populationWatermark = StableFnv1aHash(population.str());
    return result;
}

void PrintContinuationPolicyLog(const std::string& marker,
                                const ContinuationPolicyConfig& config,
                                const ContinuationEvaluation& evaluation)
{
    std::cout << marker
              << ",source_experiment_id=" << config.sourceExperimentId
              << ",source_model_id=" << evaluation.selected.modelId
              << ",source_epoch=" << evaluation.selected.completedEpoch
              << ",checkpoint_eval_id="
              << (evaluation.selected.checkpointEvalId.has_value()
                      ? std::to_string(*evaluation.selected.checkpointEvalId)
                      : "NULL")
              << ",analysis_id=" << evaluation.selected.analysisId
              << ",analysis_scope="
              << (evaluation.selected.analysisScope.empty()
                      ? "NULL"
                      : evaluation.selected.analysisScope)
              << ",symbol=" << config.source.symbol
              << ",prediction_horizon=" << config.source.predictionHorizon
              << ",evidence_count=" << evaluation.evidenceCount
              << ",patience_window=" << config.patience
              << ",leader_score=" << ContinuationOptionalDoubleText(evaluation.selected.leaderScore)
              << ",infer_accuracy=" << ContinuationOptionalDoubleText(evaluation.selected.inferAccuracy)
              << ",trend_metric="
              << (evaluation.trendMetric.has_value() ? *evaluation.trendMetric : "NULL")
              << ",trend_value=" << ContinuationOptionalDoubleText(evaluation.trendValue)
              << ",rank=" << ContinuationOptionalIntText(evaluation.rankValue)
              << ",rank_scope=" << config.scope
              << ",target_epochs="
              << (config.targetEpochs.has_value() ? std::to_string(*config.targetEpochs) : "NULL")
              << ",continuation_decision_id="
              << (evaluation.decisionId > 0 ? std::to_string(evaluation.decisionId) : "NULL")
              << ",policy_revision=" << config.policyRevision
              << ",policy_hash="
              << (evaluation.policyHash.empty() ? "NULL" : evaluation.policyHash)
              << ",current_policy_hash="
              << (evaluation.currentPolicyHash.empty()
                      ? "NULL"
                      : evaluation.currentPolicyHash)
              << ",persisted_decision_policy_hash="
              << (evaluation.persistedDecisionPolicyHash.empty()
                      ? "NULL"
                      : evaluation.persistedDecisionPolicyHash)
              << ",evidence_watermark="
              << (evaluation.evidenceWatermark.empty() ? "NULL" : evaluation.evidenceWatermark)
              << ",queued_experiment_id="
              << (evaluation.queuedExperimentId.has_value()
                      ? std::to_string(*evaluation.queuedExperimentId)
                      : "NULL")
              << ContinuationProfitabilityEvidenceLogFields(
                     evaluation.selected)
              << ContinuationProfitabilityPolicyLogFields(
                     config,
                     evaluation.profitabilityGate)
              << ",decision=" << evaluation.decision
              << ",reason=" << evaluation.reason
              << std::endl;
}

std::string ContinuationDecisionMarker(const std::string& decision)
{
    if (decision == "eligible")
        return "CONTINUATION_POLICY_ELIGIBLE";
    if (decision == "insufficient_evidence")
        return "CONTINUATION_POLICY_INSUFFICIENT_EVIDENCE";
    if (decision == "rejected_threshold")
        return "CONTINUATION_POLICY_REJECTED_THRESHOLD";
    if (decision == "rejected_rank")
        return "CONTINUATION_POLICY_REJECTED_RANK";
    if (decision == "rejected_trend")
        return "CONTINUATION_POLICY_REJECTED_TREND";
    if (decision == "rejected_profitability")
        return "CONTINUATION_POLICY_REJECTED_PROFITABILITY";
    if (decision == "already_continued" || decision == "continuation_queued")
        return "CONTINUATION_POLICY_ALREADY_CONTINUED";
    if (decision == "error")
        return "CONTINUATION_POLICY_ERROR";
    return "CONTINUATION_POLICY_SKIPPED";
}

std::optional<pqxx::row> LoadContinuationDecision(
    pqxx::work& w,
    long long sourceExperimentId,
    int targetEpochs,
    bool lockRow,
    pqxx::result& storage)
{
    std::string sql =
        "SELECT continuation_decision_id, source_model_id, source_analysis_id, "
        "source_checkpoint_eval_id, source_epoch, decision, reason, leader_score, "
        "infer_accuracy, rank_value, rank_scope, observed_eval_count, patience_window, "
        "trend_metric, trend_value, policy_revision, policy_hash, evidence_watermark, "
        "queued_experiment_id "
        "FROM experiment_continuation_decision "
        "WHERE source_experiment_id = $1 AND target_epochs = $2";
    if (lockRow)
        sql += " FOR UPDATE";
    sql += ";";
    storage = w.exec_params(
        sql,
        sourceExperimentId,
        targetEpochs);
    if (storage.empty())
        return std::nullopt;
    return storage[0];
}

void FillContinuationEvaluationFromDecisionRow(
    ContinuationEvaluation& evaluation,
    const pqxx::row& row)
{
    evaluation.decisionId = row[0].as<long long>();
    evaluation.selected.modelId = row[1].as<long long>();
    evaluation.selected.analysisId = row[2].as<long long>();
    evaluation.selected.checkpointEvalId = OptionalLongLongCell(row, 3);
    evaluation.selected.completedEpoch = row[4].as<int>();
    evaluation.decision = row[5].as<std::string>();
    evaluation.reason = row[6].as<std::string>();
    evaluation.selected.leaderScore = OptionalDoubleCell(row, 7);
    evaluation.selected.inferAccuracy = OptionalDoubleCell(row, 8);
    if (!row[9].is_null())
        evaluation.rankValue = row[9].as<int>();
    evaluation.evidenceCount = row[11].as<int>();
    evaluation.trendMetric = OptionalStringCell(row, 13);
    evaluation.trendValue = OptionalDoubleCell(row, 14);
    evaluation.persistedDecisionPolicyHash = row[16].as<std::string>();
    evaluation.policyHash = evaluation.persistedDecisionPolicyHash;
    evaluation.evidenceWatermark = row[17].as<std::string>();
    evaluation.queuedExperimentId = OptionalLongLongCell(row, 18);
    evaluation.alreadyQueued = evaluation.queuedExperimentId.has_value();
}

long long PersistContinuationDecision(pqxx::work& w,
                                      const ContinuationPolicyConfig& config,
                                      const ContinuationEvaluation& evaluation,
                                      const std::optional<long long>& existingDecisionId)
{
    std::ostringstream sql;
    if (!existingDecisionId.has_value())
    {
        sql << "INSERT INTO experiment_continuation_decision ("
            << "source_experiment_id, source_model_id, source_analysis_id, "
            << "source_checkpoint_eval_id, source_epoch, target_epochs, decision, reason, "
            << "leader_score, infer_accuracy, rank_value, rank_scope, observed_eval_count, "
            << "patience_window, trend_metric, trend_value, policy_revision, policy_hash, "
            << "evidence_watermark, queued_experiment_id, updated_at) VALUES ("
            << config.sourceExperimentId << ","
            << evaluation.selected.modelId << ","
            << evaluation.selected.analysisId << ","
            << SqlNullable(w, evaluation.selected.checkpointEvalId) << ","
            << evaluation.selected.completedEpoch << ","
            << *config.targetEpochs << ","
            << w.quote(evaluation.decision) << ","
            << w.quote(evaluation.reason) << ","
            << SqlNullable(w, evaluation.selected.leaderScore) << ","
            << SqlNullable(w, evaluation.selected.inferAccuracy) << ","
            << SqlNullable(w, evaluation.rankValue) << ","
            << w.quote(config.scope) << ","
            << evaluation.evidenceCount << ","
            << config.patience << ","
            << SqlNullable(w, evaluation.trendMetric) << ","
            << SqlNullable(w, evaluation.trendValue) << ","
            << config.policyRevision << ","
            << w.quote(evaluation.policyHash) << ","
            << w.quote(evaluation.evidenceWatermark) << ","
            << SqlNullable(w, evaluation.queuedExperimentId) << ",now()) "
            << "RETURNING continuation_decision_id;";
    }
    else
    {
        sql << "UPDATE experiment_continuation_decision SET "
            << "source_model_id=" << evaluation.selected.modelId << ","
            << "source_analysis_id=" << evaluation.selected.analysisId << ","
            << "source_checkpoint_eval_id=" << SqlNullable(w, evaluation.selected.checkpointEvalId) << ","
            << "source_epoch=" << evaluation.selected.completedEpoch << ","
            << "decision=" << w.quote(evaluation.decision) << ","
            << "reason=" << w.quote(evaluation.reason) << ","
            << "leader_score=" << SqlNullable(w, evaluation.selected.leaderScore) << ","
            << "infer_accuracy=" << SqlNullable(w, evaluation.selected.inferAccuracy) << ","
            << "rank_value=" << SqlNullable(w, evaluation.rankValue) << ","
            << "rank_scope=" << w.quote(config.scope) << ","
            << "observed_eval_count=" << evaluation.evidenceCount << ","
            << "patience_window=" << config.patience << ","
            << "trend_metric=" << SqlNullable(w, evaluation.trendMetric) << ","
            << "trend_value=" << SqlNullable(w, evaluation.trendValue) << ","
            << "policy_revision=" << config.policyRevision << ","
            << "policy_hash=" << w.quote(evaluation.policyHash) << ","
            << "evidence_watermark=" << w.quote(evaluation.evidenceWatermark) << ","
            << "updated_at=now() "
            << "WHERE continuation_decision_id=" << *existingDecisionId
            << " AND queued_experiment_id IS NULL "
            << "RETURNING continuation_decision_id;";
    }

    pqxx::result persisted = w.exec(sql.str());
    if (persisted.size() != 1)
        throw std::runtime_error("continuation decision changed concurrently or is already queued");
    const long long decisionId = persisted[0][0].as<long long>();

    w.exec_params(
        "UPDATE experiment SET "
        "continuation_policy_last_decision = $1, "
        "continuation_policy_last_decision_at = now(), "
        "continuation_policy_last_reason = $2, "
        "continuation_policy_selected_model_id = $3, "
        "continuation_policy_queued_experiment_id = $4, "
        "updated_at = now() "
        "WHERE experiment_id = $5;",
        evaluation.decision,
        evaluation.reason,
        evaluation.selected.modelId,
        evaluation.queuedExperimentId,
        config.sourceExperimentId);
    return decisionId;
}

ContinuationEvaluation EvaluateContinuationPolicy(
    pqxx::work& w,
    long long sourceExperimentId,
    ContinuationPolicyConfig* loadedConfig,
    bool persistDecision)
{
    LogWorkerStarted(
        "CONTINUATION_EVALUATION_STARTED",
        persistDecision ? "continuation_evaluate" : "continuation_dry_run",
        sourceExperimentId);
    ContinuationEvaluation evaluation;
    std::optional<ContinuationPolicyConfig> configOption =
        persistDecision
            ? LockContinuationPolicyConfigForUpdate(w, sourceExperimentId)
            : FindContinuationPolicyConfig(w, sourceExperimentId);
    if (!configOption.has_value())
    {
        ContinuationPolicyConfig missing;
        missing.sourceExperimentId = sourceExperimentId;
        evaluation.reason = ContinuationPolicySchemaExists(w)
            ? "source_experiment_not_found"
            : "migration_required";
        if (loadedConfig)
            *loadedConfig = missing;
        PrintContinuationPolicyLog("CONTINUATION_POLICY_ERROR", missing, evaluation);
        return evaluation;
    }
    ContinuationPolicyConfig config = *configOption;
    evaluation.profitabilityGate = EvaluateContinuationProfitabilityGate(
        config,
        evaluation.selected);
    if (loadedConfig)
        *loadedConfig = config;

    if (!config.enabled)
    {
        evaluation.reason = "continuation_policy_disabled";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }
    const std::optional<std::string> configError =
        ContinuationPolicyConfigurationError(config, true);
    if (configError.has_value())
    {
        evaluation.reason = "invalid_configuration:" + *configError;
        PrintContinuationPolicyLog("CONTINUATION_POLICY_ERROR", config, evaluation);
        return evaluation;
    }
    if (!ContinuationPolicySourceCompletionReady(config))
    {
        evaluation.reason = "source_requires_completed_done";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }
    if (config.candidateExcluded && !config.includeExcluded)
    {
        evaluation.reason = "source_candidate_excluded";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }

    const std::vector<ContinuationEvidence> rawEvidence =
        LoadContinuationEvidence(w, config);
    const std::vector<ContinuationEvidence> evidence =
        DeduplicateContinuationEvidence(rawEvidence);
    evaluation.evidenceCount = static_cast<int>(evidence.size());
    evaluation.evidenceWatermark = ContinuationEvidenceWatermark(evidence);
    evaluation.policyHash = ContinuationPolicySemanticHash(config);
    evaluation.currentPolicyHash = evaluation.policyHash;

    const std::optional<ContinuationEvidence> selected =
        SelectContinuationSourceEvidence(config, rawEvidence);
    if (!selected.has_value())
    {
        evaluation.reason = "no_valid_source_analysis_for_source_mode";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }
    evaluation.selected = *selected;
    evaluation.profitabilityGate = EvaluateContinuationProfitabilityGate(
        config,
        evaluation.selected);
    if (ContinuationProfitabilityPolicyConfigured(config))
    {
        evaluation.evidenceWatermark = StableFnv1aHash(
            evaluation.evidenceWatermark +
            "|profitability=" +
            ContinuationProfitabilityEvidenceIdentity(
                evaluation.selected));
    }

    if (evidence.empty() || evidence.back().completedEpoch >= *config.targetEpochs)
    {
        evaluation.reason = "target_epochs_not_greater_than_source_completed_epoch";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }

    std::string resumeReason;
    if (!ValidateContinuationResumeSource(
            w,
            config,
            evaluation.selected,
            nullptr,
            resumeReason))
    {
        evaluation.reason = resumeReason;
        PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
        return evaluation;
    }

    if (config.topN.has_value())
    {
        const ContinuationRankResult rank = RankContinuationSource(
            w,
            config,
            evaluation.selected);
        evaluation.rankValue = rank.rankValue;
        evaluation.evidenceWatermark = StableFnv1aHash(
            evaluation.evidenceWatermark + "|ranking_population=" + rank.populationWatermark);
    }

    evaluation.decision = "evaluating";
    evaluation.reason = "policy_revision=" + std::to_string(config.policyRevision) +
                        ";policy_hash=" + evaluation.policyHash +
                        ";evidence_watermark=" + evaluation.evidenceWatermark +
                        ";rules=" + ContinuationPolicySemanticCanonicalText(config);
    PrintContinuationPolicyLog("CONTINUATION_POLICY_EVALUATING", config, evaluation);

    pqxx::result existingStorage;
    std::optional<pqxx::row> existing = LoadContinuationDecision(
        w,
        sourceExperimentId,
        *config.targetEpochs,
        persistDecision,
        existingStorage);
    if (existing.has_value() && !(*existing)[18].is_null())
    {
        FillContinuationEvaluationFromDecisionRow(evaluation, *existing);
        RefreshContinuationSelectedDiagnostics(
            evaluation.selected,
            rawEvidence);
        evaluation.profitabilityGate = EvaluateContinuationProfitabilityGate(
            config,
            evaluation.selected);
        evaluation.reused = true;
        evaluation.alreadyQueued = true;
        evaluation.reason = "continuation_already_queued";
        PrintContinuationPolicyLog("CONTINUATION_POLICY_ALREADY_CONTINUED", config, evaluation);
        return evaluation;
    }

    const std::optional<long long> equivalent = FindEquivalentContinuationExperiment(
        w,
        sourceExperimentId,
        evaluation.selected.modelId,
        *config.targetEpochs);
    if (equivalent.has_value())
    {
        evaluation.decision = "already_continued";
        evaluation.reason = "equivalent_resume_experiment_exists";
        evaluation.queuedExperimentId = equivalent;
    }
    else if (existing.has_value() &&
             (*existing)[15].as<long long>() == config.policyRevision &&
             (*existing)[16].as<std::string>() == evaluation.policyHash &&
             (*existing)[17].as<std::string>() == evaluation.evidenceWatermark)
    {
        FillContinuationEvaluationFromDecisionRow(evaluation, *existing);
        RefreshContinuationSelectedDiagnostics(
            evaluation.selected,
            rawEvidence);
        evaluation.profitabilityGate = EvaluateContinuationProfitabilityGate(
            config,
            evaluation.selected);
        evaluation.reused = true;
        evaluation.persisted = true;
        PrintContinuationPolicyLog(ContinuationDecisionMarker(evaluation.decision), config, evaluation);
        return evaluation;
    }
    else if (!evaluation.selected.leaderScore.has_value() &&
             !evaluation.selected.inferAccuracy.has_value())
    {
        evaluation.decision = "insufficient_evidence";
        evaluation.reason = "selected_analysis_has_no_rankable_metric";
    }
    else if (evaluation.evidenceCount < config.minEvals)
    {
        evaluation.decision = "insufficient_evidence";
        evaluation.reason = "observed_eval_count_below_min_evals";
    }
    else
    {
        std::string trendReason;
        const ContinuationTrendResult trendResult = EvaluateContinuationTrend(
            config,
            evidence,
            evaluation.trendMetric,
            evaluation.trendValue,
            trendReason);
        if (trendResult == ContinuationTrendResult::Insufficient)
        {
            evaluation.decision = "insufficient_evidence";
            evaluation.reason = trendReason;
        }
        else
        {
            std::vector<std::string> failedThresholds;
            if (config.minLeaderScore.has_value() &&
                (!evaluation.selected.leaderScore.has_value() ||
                 *evaluation.selected.leaderScore < *config.minLeaderScore))
            {
                failedThresholds.push_back("leader_score");
            }
            if (config.minInferAccuracy.has_value() &&
                (!evaluation.selected.inferAccuracy.has_value() ||
                 *evaluation.selected.inferAccuracy < *config.minInferAccuracy))
            {
                failedThresholds.push_back("infer_accuracy");
            }

            if (!failedThresholds.empty())
            {
                evaluation.decision = "rejected_threshold";
                evaluation.reason = "failed_thresholds=" + JoinCheckpointPolicyRules(failedThresholds);
            }
            else if (!evaluation.profitabilityGate.passed)
            {
                evaluation.decision = "rejected_profitability";
                evaluation.reason = evaluation.profitabilityGate.reason;
            }
            else
            {
                if (config.topN.has_value() &&
                    (!evaluation.rankValue.has_value() || *evaluation.rankValue > *config.topN))
                {
                    evaluation.decision = "rejected_rank";
                    evaluation.reason = "rank_outside_top_n";
                }
                else if (trendResult == ContinuationTrendResult::Reject)
                {
                    evaluation.decision = "rejected_trend";
                    evaluation.reason = trendReason;
                }
                else
                {
                    evaluation.decision = "eligible";
                    evaluation.reason = "all_configured_gates_passed";
                }
            }
        }
    }

    const std::optional<long long> existingDecisionId = existing.has_value()
        ? std::optional<long long>{(*existing)[0].as<long long>()}
        : std::nullopt;
    if (!persistDecision)
    {
        if (existingDecisionId.has_value())
            evaluation.decisionId = *existingDecisionId;
        PrintContinuationPolicyLog(ContinuationDecisionMarker(evaluation.decision), config, evaluation);
        return evaluation;
    }
    evaluation.decisionId = PersistContinuationDecision(
        w,
        config,
        evaluation,
        existingDecisionId);
    evaluation.persisted = true;

    PrintContinuationPolicyLog("CONTINUATION_POLICY_DECISION_PERSISTED", config, evaluation);
    PrintContinuationPolicyLog(ContinuationDecisionMarker(evaluation.decision), config, evaluation);
    return evaluation;
}

ContinuationChildPolicyPlan PrepareContinuationChildPolicy(
    pqxx::work& w,
    const ContinuationPolicyConfig& sourceConfig,
    SchedulerOptions& child,
    int continuationSourceEpoch)
{
    ContinuationChildPolicyPlan plan = PlanContinuationChildPolicy(sourceConfig);
    if (!sourceConfig.inheritToChild)
        return plan;

    if (!plan.terminal &&
        (sourceConfig.sourceMode == "best_checkpoint" ||
         sourceConfig.sourceMode == "latest_checkpoint"))
    {
        pqxx::result checkpointConfig = w.exec_params(
            "SELECT checkpoint_infer_enabled, checkpoint_infer_min_epoch, "
            "checkpoint_infer_interval FROM experiment WHERE experiment_id = $1;",
            sourceConfig.sourceExperimentId);
        if (checkpointConfig.size() != 1 || !checkpointConfig[0][0].as<bool>())
        {
            throw std::runtime_error(
                "inherited_checkpoint_source_mode_requires_checkpoint_inference");
        }
        child.queueCheckpointInfer = true;
        if (!checkpointConfig[0][1].is_null())
            child.queueCheckpointInferMinEpoch = checkpointConfig[0][1].as<int>();
        if (!checkpointConfig[0][2].is_null())
            child.queueCheckpointInferInterval = checkpointConfig[0][2].as<int>();
        if (child.queueCheckpointInferMinEpoch.has_value() &&
            *child.queueCheckpointInferMinEpoch > *sourceConfig.targetEpochs)
        {
            throw std::runtime_error(
                "inherited_checkpoint_min_epoch_exceeds_child_target");
        }
        const int checkpointInterval = sourceConfig.source.checkpointInterval;
        const int inferenceInterval = child.queueCheckpointInferInterval.value_or(checkpointInterval);
        if (checkpointInterval <= 0 || inferenceInterval <= 0)
            throw std::runtime_error("inherited_checkpoint_interval_must_be_positive");
        const auto gcd = [](long long lhs, long long rhs) {
            while (rhs != 0)
            {
                const long long remainder = lhs % rhs;
                lhs = rhs;
                rhs = remainder;
            }
            return lhs;
        };
        const long long commonInterval =
            static_cast<long long>(checkpointInterval) / gcd(checkpointInterval, inferenceInterval) *
            inferenceInterval;
        const long long minimumEpoch = std::max<long long>(
            static_cast<long long>(continuationSourceEpoch) + 1,
            child.queueCheckpointInferMinEpoch.value_or(1));
        const long long firstQualifyingEpoch =
            ((minimumEpoch + commonInterval - 1) / commonInterval) * commonInterval;
        if (firstQualifyingEpoch > *sourceConfig.targetEpochs)
        {
            throw std::runtime_error(
                "inherited_checkpoint_source_mode_has_no_qualifying_child_checkpoint");
        }
    }

    return plan;
}

void PersistContinuationChildPolicy(
    pqxx::work& w,
    long long childExperimentId,
    const ContinuationPolicyConfig& sourceConfig,
    const ContinuationChildPolicyPlan& plan)
{
    if (!plan.inherit)
        return;

    std::ostringstream sql;
    sql << "UPDATE experiment SET "
        << "continuation_policy_enabled = " << (plan.terminal ? "false" : "true") << ", "
        << "continuation_policy_target_epochs = "
        << (plan.terminal ? "NULL" : std::to_string(plan.targetEpochs)) << ", "
        << "continuation_policy_min_evals = " << sourceConfig.minEvals << ", "
        << "continuation_policy_patience = " << sourceConfig.patience << ", "
        << "continuation_policy_min_leader_score = " << SqlNullable(w, sourceConfig.minLeaderScore) << ", "
        << "continuation_policy_min_infer_accuracy = " << SqlNullable(w, sourceConfig.minInferAccuracy) << ", "
        << "continuation_policy_min_profit_actionable_count = "
        << SqlNullable(w, sourceConfig.minProfitabilityActionableCount) << ", "
        << "continuation_policy_min_profit_aggregate_log_return_sum = "
        << SqlNullable(
               w,
               sourceConfig
                   .minProfitabilityAggregateTerminalHorizonLogReturnSum)
        << ", "
        << "continuation_policy_min_profit_average_log_return = "
        << SqlNullable(
               w,
               sourceConfig
                   .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction)
        << ", "
        << "continuation_policy_min_improvement = " << SqlNullable(w, sourceConfig.minImprovement) << ", "
        << "continuation_policy_max_degradation = " << SqlNullable(w, sourceConfig.maxDegradation) << ", "
        << "continuation_policy_top_n = " << SqlNullable(w, sourceConfig.topN) << ", "
        << "continuation_policy_scope = " << w.quote(sourceConfig.scope) << ", "
        << "continuation_policy_trend_mode = " << w.quote(sourceConfig.trendMode) << ", "
        << "continuation_policy_source_mode = " << w.quote(sourceConfig.sourceMode) << ", "
        << "continuation_policy_include_excluded = " << (sourceConfig.includeExcluded ? "true" : "false") << ", "
        << "continuation_candidate_excluded = " << (sourceConfig.candidateExcluded ? "true" : "false") << ", "
        << "continuation_policy_inherit_to_child = " << (plan.terminal ? "false" : "true") << ", "
        << "continuation_policy_progression_mode = "
        << SqlNullable(w, plan.inheritedPolicy.progressionMode) << ", "
        << "continuation_policy_target_increment = "
        << SqlNullable(w, plan.inheritedPolicy.targetIncrement) << ", "
        << "continuation_policy_max_target_epochs = " << SqlNullable(w, sourceConfig.maxTargetEpochs) << ", "
        << "continuation_policy_target_sequence = "
        << SqlContinuationTargetSequence(plan.inheritedPolicy.targetSequence) << ", "
        << "continuation_policy_revision = 1, "
        << "continuation_policy_last_decision = NULL, "
        << "continuation_policy_last_decision_at = NULL, "
        << "continuation_policy_last_reason = NULL, "
        << "continuation_policy_selected_model_id = NULL, "
        << "continuation_policy_queued_experiment_id = NULL, "
        << "continuation_policy_inherited = true, "
        << "continuation_policy_inherited_from_experiment_id = " << sourceConfig.sourceExperimentId << ", "
        << "continuation_policy_inherited_from_revision = " << sourceConfig.policyRevision << ", "
        << "continuation_policy_inherited_from_hash = " << w.quote(plan.sourcePolicyHash) << ", "
        << "continuation_policy_inheritance_status = "
        << w.quote(plan.terminal ? "max_target_reached" : "valid") << ", "
        << "updated_at = now() "
        << "WHERE experiment_id = " << childExperimentId << ";";
    w.exec(sql.str());
}

int RunQueueContinuationCommand(const SchedulerOptions& options)
{
    const long long sourceExperimentId = *options.queueContinuationExperimentId;
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work w{connection};
        SetTransactionReadWrite(w);
        if (options.schedulerAuthority.held)
        {
            RequireAndRefreshSchedulerAuthority(w, options);
            InjectSchedulerAuthorityLossForTest(
                w,
                options,
                "continuation_before_evaluation_mutation");
            RequireAndRefreshSchedulerAuthority(w, options);
        }
        if (!SchedulerLaunchAllowed(w, "continuation_queue"))
        {
            w.commit();
            return 1;
        }

        ContinuationPolicyConfig config;
        ContinuationEvaluation evaluation = EvaluateContinuationPolicy(
            w,
            sourceExperimentId,
            &config);
        if (options.schedulerAuthority.held)
        {
            InjectSchedulerAuthorityLossForTest(
                w,
                options,
                "continuation_after_evaluation_before_queue");
            RequireAndRefreshSchedulerAuthority(w, options);
        }
        PrintContinuationPolicyLog("CONTINUATION_POLICY_QUEUE_REQUESTED", config, evaluation);

        if (evaluation.alreadyQueued || evaluation.queuedExperimentId.has_value())
        {
            PrintContinuationPolicyLog("CONTINUATION_POLICY_QUEUE_DUPLICATE", config, evaluation);
            w.commit();
            return 3;
        }
        if (!evaluation.persisted && !evaluation.reused)
        {
            evaluation.reason = "continuation_evaluation_not_persisted";
            PrintContinuationPolicyLog("CONTINUATION_POLICY_ERROR", config, evaluation);
            w.commit();
            return 1;
        }
        if (evaluation.decision != "eligible")
        {
            evaluation.reason = "current_decision_not_eligible:" + evaluation.decision;
            PrintContinuationPolicyLog("CONTINUATION_POLICY_SKIPPED", config, evaluation);
            w.commit();
            return 1;
        }

        pqxx::result decisionRows = w.exec_params(
            "SELECT decision, policy_revision, policy_hash, evidence_watermark, queued_experiment_id "
            "FROM experiment_continuation_decision "
            "WHERE continuation_decision_id = $1 FOR UPDATE;",
            evaluation.decisionId);
        if (decisionRows.size() != 1 ||
            decisionRows[0][0].as<std::string>() != "eligible" ||
            decisionRows[0][1].as<long long>() != config.policyRevision ||
            decisionRows[0][2].as<std::string>() != evaluation.policyHash ||
            decisionRows[0][3].as<std::string>() != evaluation.evidenceWatermark ||
            !decisionRows[0][4].is_null())
        {
            throw std::runtime_error("continuation decision changed before queueing");
        }

        const std::optional<long long> duplicate = FindEquivalentContinuationExperiment(
            w,
            sourceExperimentId,
            evaluation.selected.modelId,
            *config.targetEpochs);
        if (duplicate.has_value())
        {
            evaluation.queuedExperimentId = duplicate;
            evaluation.reason = "equivalent_resume_experiment_exists";
            PrintContinuationPolicyLog("CONTINUATION_POLICY_QUEUE_DUPLICATE", config, evaluation);
            w.commit();
            return 3;
        }

        QueueResumeMeta resumeMeta;
        std::string resumeReason;
        if (!ValidateContinuationResumeSource(
                w,
                config,
                evaluation.selected,
                &resumeMeta,
                resumeReason))
        {
            throw std::runtime_error(resumeReason);
        }

        SchedulerOptions child;
        child.schedulerExecutablePath = options.schedulerExecutablePath;
        child.schedulerAuthority = options.schedulerAuthority;
        child.queueExperiment = true;
        child.resumeModelId = evaluation.selected.modelId;
        child.targetEpochs = config.targetEpochs;
        child.checkpointInterval = config.source.checkpointInterval;
        child.inferStart = config.source.inferStart;
        child.inferEnd = config.source.inferEnd;
        child.queueContinuationCandidateExcluded = config.candidateExcluded;
        {
            const pqxx::result sourceMask = w.exec_params(
                "SELECT feature_ablation_mask FROM experiment WHERE experiment_id=$1;",
                sourceExperimentId);
            if (sourceMask.size() != 1)
                throw std::runtime_error("continuation_source_feature_ablation_mask_missing");
            child.featureAblationMask = EA::FeatureAblationMask::Parse(
                sourceMask[0][0].as<std::string>()).CanonicalText();
        }
        MergeResumeMetaIntoQueueOptions(child, resumeMeta);
        ResolveEconomicCalendarSnapshotForQueue(
            w, child, "continuation_inherit");
        const ContinuationChildPolicyPlan childPolicy =
            PrepareContinuationChildPolicy(
                w,
                config,
                child,
                evaluation.selected.completedEpoch);
        EnsureRequiredQueueOptions(child);

        if (options.schedulerAuthority.held)
        {
            InjectSchedulerAuthorityLossForTest(
                w,
                options,
                "continuation_before_child_creation");
            RequireAndRefreshSchedulerAuthority(w, options);
        }
        const long long childExperimentId = InsertExperimentRecord(
            w,
            child,
            config.source.symbol,
            0);
        const int childGeneration = config.continuationSourceExperimentId.has_value()
            ? config.continuationGeneration + 1
            : 1;
        pqxx::result childUpdated = w.exec_params(
            "UPDATE experiment SET "
            "parent_experiment_id = $1, "
            "continuation_source_experiment_id = $1, "
            "continuation_source_model_id = $2, "
            "continuation_source_epoch = $3, "
            "continuation_decision_id = $4, "
            "continuation_generation = $5, "
            "continuation_candidate_excluded = $6, "
            "continuation_policy_enabled = false, "
            "updated_at = now() "
            "WHERE experiment_id = $7 "
            "RETURNING experiment_id;",
            sourceExperimentId,
            evaluation.selected.modelId,
            evaluation.selected.completedEpoch,
            evaluation.decisionId,
            childGeneration,
            config.candidateExcluded,
            childExperimentId);
        if (childUpdated.size() != 1)
            throw std::runtime_error("failed to persist continuation experiment lineage");
        PersistContinuationChildPolicy(w, childExperimentId, config, childPolicy);

        pqxx::result decisionUpdated = w.exec_params(
            "UPDATE experiment_continuation_decision SET "
            "decision = 'continuation_queued', "
            "reason = 'continuation_experiment_created', "
            "queued_experiment_id = $1, updated_at = now() "
            "WHERE continuation_decision_id = $2 AND queued_experiment_id IS NULL "
            "RETURNING continuation_decision_id;",
            childExperimentId,
            evaluation.decisionId);
        if (decisionUpdated.size() != 1)
            throw std::runtime_error("continuation decision was queued concurrently");
        pqxx::result sourceUpdated = w.exec_params(
            "UPDATE experiment SET "
            "continuation_policy_last_decision = 'continuation_queued', "
            "continuation_policy_last_decision_at = now(), "
            "continuation_policy_last_reason = 'continuation_experiment_created', "
            "continuation_policy_selected_model_id = $1, "
            "continuation_policy_queued_experiment_id = $2, "
            "updated_at = now() "
            "WHERE experiment_id = $3 "
            "AND continuation_policy_queued_experiment_id IS NULL "
            "RETURNING experiment_id;",
            evaluation.selected.modelId,
            childExperimentId,
            sourceExperimentId);
        if (sourceUpdated.size() != 1)
            throw std::runtime_error(
                "continuation source lifecycle changed before queue commit");

        evaluation.decision = "continuation_queued";
        evaluation.reason = "continuation_experiment_created";
        evaluation.queuedExperimentId = childExperimentId;
        if (options.schedulerAuthority.held)
        {
            InjectSchedulerAuthorityLossForTest(
                w,
                options,
                "continuation_before_final_commit");
            RequireAndRefreshSchedulerAuthority(w, options);
        }
        w.commit();
        PrintContinuationPolicyLog("CONTINUATION_POLICY_QUEUED", config, evaluation);
        std::cout << "CONTINUATION_POLICY_CHILD_INITIALIZED"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",child_experiment_id=" << childExperimentId
                  << ",inherit_to_child="
                  << (childPolicy.inherit && !childPolicy.terminal ? "1" : "0")
                  << ",child_target_epochs=" << *config.targetEpochs
                  << ",inherited_policy_target_epochs="
                  << (childPolicy.inherit && !childPolicy.terminal
                          ? std::to_string(childPolicy.targetEpochs)
                          : "NULL")
                  << ",target_increment="
                  << (config.targetIncrement.has_value()
                          ? std::to_string(*config.targetIncrement)
                          : "NULL")
                  << ",max_target_epochs="
                  << (config.maxTargetEpochs.has_value()
                          ? std::to_string(*config.maxTargetEpochs)
                          : "NULL")
                  << ",policy_hash="
                  << (childPolicy.inherit ? childPolicy.policyHash : "NULL")
                  << ",inherited_from_revision="
                  << (childPolicy.inherit ? std::to_string(config.policyRevision) : "NULL")
                  << ",inherited_from_hash="
                  << (childPolicy.inherit ? childPolicy.sourcePolicyHash : "NULL")
                  << ",inheritance_status="
                  << (childPolicy.inherit
                          ? (childPolicy.terminal ? "max_target_reached" : "valid")
                          : "not_requested")
                  << ",progression_mode="
                  << (childPolicy.inherit
                          ? childPolicy.progressionMode
                          : EffectiveContinuationProgressionMode(config).value_or("NULL"))
                  << ",target_sequence="
                  << ContinuationTargetSequenceText(config.targetSequence)
                  << ",progression_diagnostic="
                  << (childPolicy.progressionDiagnostic.empty()
                          ? "not_requested"
                          : childPolicy.progressionDiagnostic)
                  << std::endl;
        return 0;
    }
    catch (const SchedulerAuthorityLost&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=" << e.what()
                  << std::endl;
        return 1;
    }
}

using ContinuationAutoScanCounts =
    EA::SchedulerCore::ContinuationAutomationCounts;

struct ContinuationAutoScanState
{
    std::chrono::steady_clock::time_point nextScan = std::chrono::steady_clock::now();
    bool hasRun = false;
    std::string lastScanAt;
    ContinuationAutoScanCounts lastCounts;
};

class SchedulerContinuationOrchestrationPort final
    : public EA::SchedulerCore::ContinuationOrchestrationPort
{
public:
    explicit SchedulerContinuationOrchestrationPort(
        const SchedulerOptions& options)
        : options_(options)
    {
    }

    bool automationAllowed() override
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        RequireAndRefreshSchedulerAuthority(transaction, options_);
        const bool allowed =
            SchedulerLaunchAllowed(transaction, "continuation_automation");
        transaction.commit();
        return allowed;
    }

    bool refreshAuthority() override
    {
        return RefreshSchedulerAuthority(options_);
    }

    std::vector<long long> loadCandidateIds() override
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadOnly(transaction);
        if (!ContinuationPolicySchemaExists(transaction))
            throw std::runtime_error("continuation_policy_schema_missing");
        const pqxx::result rows = transaction.exec(
            "SELECT experiment_id FROM experiment "
            "WHERE continuation_policy_enabled = true "
            "AND status = 'completed' AND phase = 'done' "
            "ORDER BY experiment_id ASC;");
        std::vector<long long> ids;
        ids.reserve(rows.size());
        for (const pqxx::row& row : rows)
            ids.push_back(row[0].as<long long>());
        transaction.commit();
        return ids;
    }

    EA::SchedulerCore::ContinuationAutomationPreflight preflight(
        long long sourceExperimentId,
        bool dryRun) override
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadOnly(transaction);
        const ContinuationAutoPreflightLookup preflight =
            LoadAutomaticContinuationPreflightReadOnly(
                transaction,
                sourceExperimentId);
        transaction.commit();
        return {
            preflight.satisfaction.alreadySatisfied,
            FormatAutomaticContinuationSatisfiedFields(preflight, dryRun)};
    }

    EA::SchedulerCore::ContinuationAutomationCandidate evaluate(
        long long sourceExperimentId,
        bool persist) override
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        if (persist)
        {
            SetTransactionReadWrite(transaction);
            RequireAndRefreshSchedulerAuthority(transaction, options_);
            InjectSchedulerAuthorityLossForTest(
                transaction,
                options_,
                "continuation_before_evaluation_mutation");
            RequireAndRefreshSchedulerAuthority(transaction, options_);
        }
        else
        {
            SetTransactionReadOnly(transaction);
        }

        EA::SchedulerCore::ContinuationAutomationCandidate candidate;
        candidate.sourceExperimentId = sourceExperimentId;
        candidate.evaluation = EvaluateContinuationPolicy(
            transaction,
            sourceExperimentId,
            &candidate.config,
            persist);
        if (persist)
            RequireAndRefreshSchedulerAuthority(transaction, options_);
        transaction.commit();
        return candidate;
    }

    EA::SchedulerCore::ContinuationQueueResult queue(
        long long sourceExperimentId) override
    {
        SchedulerOptions queueOptions;
        queueOptions.schedulerExecutablePath = options_.schedulerExecutablePath;
        queueOptions.schedulerAuthority = options_.schedulerAuthority;
        queueOptions.queueContinuationExperimentId = sourceExperimentId;
        const int result = RunQueueContinuationCommand(queueOptions);
        if (result == 0)
            return EA::SchedulerCore::ContinuationQueueResult::Queued;
        if (result == 3)
            return EA::SchedulerCore::ContinuationQueueResult::AlreadyQueued;
        return EA::SchedulerCore::ContinuationQueueResult::Failed;
    }

    void refreshQueuedIdentity(
        EA::SchedulerCore::ContinuationAutomationCandidate& candidate) override
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadOnly(transaction);
        const std::optional<ContinuationPolicyConfig> loaded =
            FindContinuationPolicyConfig(
                transaction,
                candidate.sourceExperimentId);
        if (loaded.has_value())
        {
            candidate.config = *loaded;
            candidate.evaluation.queuedExperimentId =
                loaded->queuedExperimentId;
            if (loaded->targetEpochs.has_value())
            {
                pqxx::result decisionStorage;
                const std::optional<pqxx::row> decision =
                    LoadContinuationDecision(
                        transaction,
                        candidate.sourceExperimentId,
                        *loaded->targetEpochs,
                        false,
                        decisionStorage);
                if (decision.has_value())
                {
                    FillContinuationEvaluationFromDecisionRow(
                        candidate.evaluation,
                        *decision);
                    const std::vector<ContinuationEvidence> rawEvidence =
                        LoadContinuationEvidence(transaction, *loaded);
                    RefreshContinuationSelectedDiagnostics(
                        candidate.evaluation.selected,
                        rawEvidence);
                    candidate.evaluation.profitabilityGate =
                        EvaluateContinuationProfitabilityGate(
                            *loaded,
                            candidate.evaluation.selected);
                }
            }
        }
        transaction.commit();
    }

private:
    const SchedulerOptions& options_;
};

ContinuationAutoScanCounts RunAutomaticContinuationScan(
    const SchedulerOptions& options)
{
    SchedulerContinuationOrchestrationPort port{options};
    EA::SchedulerCore::ContinuationOrchestrationService service{
        port,
        std::cout,
        std::cerr};
    return service.runAutomaticScan({
        options.autoQueueContinuations,
        options.continuationScanSeconds,
        options.continuationMaxQueuesPerScan,
        options.continuationDryRun || options.dryRun});
}

std::string RequireProcessStartIdentity(pid_t pid)
{
    const auto identity =
        EA::GlobalExperimentControl::ReadProcessStartIdentity(
            static_cast<int>(pid));
    if (!identity)
        throw std::runtime_error(
            "failed to capture worker process-start identity for pid " +
            std::to_string(pid));
    return *identity;
}

void LogPhaseTransition(long long experimentId,
                               const std::string& fromPhase,
                               const std::string& toPhase)
{
    std::cout << "SCHEDULER_PHASE_TRANSITION"
              << ",experiment_id=" << experimentId
              << ",from_phase=" << fromPhase
              << ",to_phase=" << toPhase
              << std::endl;
}

void MarkExperimentPendingPhase(pqxx::work& w,
                                       const ExperimentRow& experiment,
                                       const std::string& fromPhase,
                                       const std::string& toPhase,
                                       const std::optional<int>& exitCode = std::nullopt,
                                       const std::optional<long long>&
                                           workerAttemptId = std::nullopt)
{
    std::ostringstream sql;
    sql << "UPDATE experiment "
        << "SET status = 'pending', phase = " << w.quote(toPhase)
        << ", exit_code = " << (exitCode.has_value() ? std::to_string(*exitCode) : "NULL")
        << ", error_message = NULL, worker_pid = NULL, updated_at = now() "
        << "WHERE experiment_id = " << experiment.experimentId;
    if (workerAttemptId)
        sql << " AND active_scheduler_worker_attempt_id = "
            << *workerAttemptId;
    sql << " RETURNING experiment_id;";
    pqxx::result updated = w.exec(sql.str());
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "mark_experiment_pending_phase");
    LogPhaseTransition(experiment.experimentId, fromPhase, toPhase);
    std::cout << "EXPERIMENT_PHASE_CHANGED"
              << ",experiment_id=" << experiment.experimentId
              << ",phase=" << toPhase
              << ",status=pending"
              << std::endl;
}

void MarkExperimentDone(pqxx::work& w,
                               const ExperimentRow& experiment,
                               const std::string& fromPhase,
                               const std::optional<long long>&
                                   workerAttemptId = std::nullopt)
{
    pqxx::result updated = w.exec_params(
        "UPDATE experiment "
        "SET status = 'completed', phase = 'done', worker_pid = NULL, completed_at = COALESCE(completed_at, now()), updated_at = now() "
        "WHERE experiment_id = $1 "
        "AND ($2::bigint IS NULL OR "
        "active_scheduler_worker_attempt_id=$2) "
        "RETURNING experiment_id;",
        experiment.experimentId,
        workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "mark_experiment_done");
    LogPhaseTransition(experiment.experimentId, fromPhase, "done");
    std::cout << "SCHEDULER_PIPELINE_DONE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << std::endl;
}

void MarkExperimentFailed(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 const std::string& errorMessage,
                                 int exitCode = -1,
                                 const std::optional<long long>&
                                     workerAttemptId = std::nullopt)
{
    pqxx::result updated = w.exec_params(
        "UPDATE experiment "
        "SET status = 'failed', exit_code = $1, error_message = $2, worker_pid = NULL, completed_at = now(), updated_at = now() "
        "WHERE experiment_id = $3 "
        "AND ($4::bigint IS NULL OR "
        "active_scheduler_worker_attempt_id=$4) "
        "RETURNING experiment_id;",
        exitCode,
        errorMessage,
        experiment.experimentId,
        workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "mark_experiment_failed");
}

void UpdateInferLogPath(pqxx::work& w,
                               const ExperimentRow& experiment,
                               const std::string& inferLogPath,
                               const std::optional<long long>&
                                   workerAttemptId = std::nullopt)
{
    pqxx::result updated = w.exec_params(
        "UPDATE experiment "
        "SET infer_log_path = $1, updated_at = now() "
        "WHERE experiment_id = $2 "
        "AND ($3::bigint IS NULL OR "
        "active_scheduler_worker_attempt_id=$3) "
        "RETURNING experiment_id;",
        inferLogPath,
        experiment.experimentId,
        workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "update_exact_attempt_infer_log_path");
}

void TransitionRecoveredInferenceToAnalyze(pqxx::work& w,
                                                  const ExperimentRow& experiment,
                                                  const std::string& sourceMarker,
                                                  const std::string& reason,
                                                  const std::optional<std::string>& recoveredLogPath = std::nullopt,
                                                  const std::optional<long long>&
                                                      workerAttemptId = std::nullopt,
                                                  bool consumeForcedFinalInferenceRerun = false)
{
    if (recoveredLogPath.has_value())
        UpdateInferLogPath(
            w, experiment, *recoveredLogPath, workerAttemptId);

    std::cout << sourceMarker
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << ",phase=infer"
              << ",reason=" << reason;
    if (recoveredLogPath.has_value())
        std::cout << ",infer_log_path=" << *recoveredLogPath;
    else if (experiment.inferLogPath.has_value())
        std::cout << ",infer_log_path=" << *experiment.inferLogPath;
    std::cout << std::endl;

    if (consumeForcedFinalInferenceRerun && workerAttemptId)
    {
        const pqxx::result consumed = w.exec(
            "UPDATE experiment e SET "
            "operator_forced_final_inference_rerun_requested=false,"
            "updated_at=clock_timestamp() "
            "FROM experiment_scheduler_worker_attempt a "
            "WHERE e.experiment_id=$1 "
            "AND e.operator_forced_final_inference_rerun_requested "
            "AND e.phase='infer' "
            "AND (e.status='running' OR (e.status='pending' "
            " AND e.resume_requested=true "
            " AND e.scheduler_resume_origin='preemption')) "
            "AND e.active_scheduler_worker_attempt_id=$2 "
            "AND a.worker_attempt_id=$2 "
            "AND a.experiment_id=e.experiment_id "
            "AND a.worker_kind='experiment' "
            "AND a.lifecycle_phase='infer' "
            "AND EXISTS ("
            " SELECT 1 FROM inference_eval_result r "
            " WHERE r.model_id=e.last_model_id "
            " AND r.symbol=e.symbol "
            " AND r.prediction_horizon=e.prediction_horizon "
            " AND abs(r.threshold_logret-e.c_next_threshold)<=1e-7 "
            " AND r.from_date=$3 "
            " AND r.to_date=$4 "
            " AND r.status='completed' "
            " AND r.inference_scope='final' "
            " AND r.checkpoint_eval_id IS NULL "
            " AND r.completed_at>=a.reserved_at"
            ") RETURNING e.experiment_id;",
            pqxx::params{
                experiment.experimentId,
                *workerAttemptId,
                experiment.inferStart->substr(0, 10),
                experiment.inferEnd->substr(0, 10)});
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            consumed,
            "consume_operator_forced_final_inference_rerun");
        std::cout
            << "SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_CONSUMED"
            << ",experiment_id=" << experiment.experimentId
            << ",worker_attempt_id=" << *workerAttemptId
            << std::endl;
    }

    MarkExperimentPendingPhase(
        w,
        experiment,
        "infer",
        "analyze",
        0,
        workerAttemptId);
    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << std::endl;
}

bool TransitionAfterTrainModelAvailable(pqxx::work& w,
                                               const ExperimentRow& experiment,
                                               long long modelId,
                                               int exitCode,
                                               const std::string& fromPhase,
                                               const std::optional<long long>&
                                                   workerAttemptId = std::nullopt)
{
    ExperimentRow updatedExperiment = experiment;
    updatedExperiment.lastModelId = modelId;
    pqxx::result modelUpdated = w.exec_params(
        "UPDATE experiment "
        "SET last_model_id = $1, exit_code = $2, error_message = NULL, updated_at = now() "
        "WHERE experiment_id = $3 "
        "AND ($4::bigint IS NULL OR "
        "active_scheduler_worker_attempt_id=$4) "
        "RETURNING experiment_id;",
        modelId,
        exitCode,
        experiment.experimentId,
        workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            modelUpdated, "transition_train_model_exact_attempt");
    std::cout << "EXPERIMENT_LAST_MODEL_ID"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;

    if (HasCompletedAnalysisResult(w, updatedExperiment))
    {
        std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
        MarkExperimentDone(
            w, updatedExperiment, fromPhase, workerAttemptId);
    }
    else if (HasCompletedInferenceResult(w, updatedExperiment))
    {
        std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
        MarkExperimentPendingPhase(
            w,
            updatedExperiment,
            fromPhase,
            "analyze",
            exitCode,
            workerAttemptId);
        std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
    }
    else if (updatedExperiment.inferStart.has_value() && updatedExperiment.inferEnd.has_value())
    {
        MarkExperimentPendingPhase(
            w,
            updatedExperiment,
            fromPhase,
            "infer",
            exitCode,
            workerAttemptId);
        std::cout << "SCHEDULER_ENQUEUE_INFER"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
    }
    else
    {
        MarkExperimentFailed(
            w,
            updatedExperiment,
            "train_completed_missing_inference_range",
            exitCode,
            workerAttemptId);
        return false;
    }
    return true;
}

void RequeueTrainOrphanFromCheckpoint(pqxx::work& w,
                                      const ExperimentRow& experiment,
                                      const QueueResumeMeta& meta,
                                      const std::optional<long long>&
                                          workerAttemptId = std::nullopt)
{
    pqxx::result updated = w.exec_params(
        "UPDATE experiment "
        "SET status = 'pending', phase = 'train', last_model_id = $1, "
        "resume_model_id = CASE "
        "WHEN continuation_source_model_id IS NULL THEN $1 "
        "ELSE resume_model_id END, "
        "resume_requested=false,scheduler_resume_origin='none',"
        "exit_code = NULL, error_message = NULL, updated_at = updated_at "
        "WHERE experiment_id = $2 "
        "AND ($3::bigint IS NULL OR "
        "active_scheduler_worker_attempt_id=$3);",
        meta.modelId,
        experiment.experimentId,
        workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "requeue_train_orphan_exact_attempt");
    std::cout << "SCHEDULER_ORPHAN_RECOVERED"
              << ",experiment_id=" << experiment.experimentId
              << ",restart_model_id=" << meta.modelId
              << ",completed_epochs=" << meta.completedEpochs
              << ",target_epochs=" << experiment.targetEpochs
              << std::endl;
    LogPhaseTransition(experiment.experimentId, "train", "train");
}

bool RecoverTrainOrphanFromModel(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 long long modelId,
                                 const std::optional<long long>&
                                     workerAttemptId = std::nullopt)
{
    const std::optional<QueueResumeMeta> meta = TryLoadRecoverableModelMeta(w, modelId);
    if (!meta.has_value())
        return false;

    if (meta->symbol != experiment.symbol ||
        meta->predictionHorizon != experiment.predictionHorizon ||
        std::fabs(meta->threshold - experiment.cNextThreshold) > 1e-7 ||
        !SameDate(meta->trainStart, experiment.trainStart) ||
        !SameDate(meta->trainEnd, experiment.trainEnd))
    {
        std::cout << "SCHEDULER_ORPHAN_MODEL_UNUSABLE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << ",reason=config_mismatch"
                  << std::endl;
        return false;
    }

    if (meta->completedEpochs >= experiment.targetEpochs)
    {
        std::cout << "SCHEDULER_ORPHAN_ADVANCED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << ",completed_epochs=" << meta->completedEpochs
                  << ",target_epochs=" << experiment.targetEpochs
                  << ",next_phase=" << (experiment.inferStart.has_value() && experiment.inferEnd.has_value() ? "infer" : "none")
                  << std::endl;
        return TransitionAfterTrainModelAvailable(
            w, experiment, modelId, 0, "train", workerAttemptId);
    }

    RequeueTrainOrphanFromCheckpoint(
        w, experiment, *meta, workerAttemptId);
    return true;
}

bool CommandHasExactOptionValue(
    const std::string& command,
    const std::string& option,
    long long expectedValue)
{
    const std::string expected = std::to_string(expectedValue);
    size_t position = 0;
    while ((position = command.find(option, position)) !=
           std::string::npos)
    {
        const bool tokenStart =
            position == 0 ||
            std::isspace(
                static_cast<unsigned char>(command[position - 1]));
        size_t valueStart = position + option.size();
        if (tokenStart && valueStart < command.size() &&
            command[valueStart] == '=')
        {
            ++valueStart;
        }
        else if (
            tokenStart && valueStart < command.size() &&
            std::isspace(
                static_cast<unsigned char>(command[valueStart])))
        {
            while (
                valueStart < command.size() &&
                std::isspace(
                    static_cast<unsigned char>(
                        command[valueStart])))
                ++valueStart;
        }
        else
        {
            position += option.size();
            continue;
        }
        const size_t valueEnd = valueStart + expected.size();
        if (command.compare(
                valueStart, expected.size(), expected) == 0 &&
            (valueEnd == command.size() ||
             std::isspace(
                 static_cast<unsigned char>(
                     command[valueEnd]))))
            return true;
        position += option.size();
    }
    return false;
}

bool CommandHasExactToken(
    const std::string& command,
    const std::string& token)
{
    size_t position = 0;
    while ((position = command.find(token, position)) !=
           std::string::npos)
    {
        const bool starts =
            position == 0 ||
            std::isspace(
                static_cast<unsigned char>(command[position - 1]));
        const size_t end = position + token.size();
        if (starts &&
            (end == command.size() ||
             std::isspace(
                 static_cast<unsigned char>(command[end]))))
            return true;
        position += token.size();
    }
    return false;
}

struct LegacyNoPidProcessSearch
{
    bool inspectionSucceeded = false;
    bool matchingCommandObserved = false;
    bool inspectionDenied = false;
    std::optional<int> matchingPid;
};

LegacyNoPidProcessSearch SearchForLegacyNoPidProcess(
    long long experimentId,
    const std::optional<long long>& checkpointEvalId,
    const std::string& phase)
{
    LegacyNoPidProcessSearch result;
    FILE* pipe = ::popen("ps -axo pid=,command=", "r");
    if (pipe == nullptr)
        return result;
    char buffer[32768] = {};
    std::unique_ptr<
        EA::GlobalExperimentControl::ProcessOperations>
        processes =
            EA::GlobalExperimentControl::
                CreateNativeProcessOperations();
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        std::istringstream row{buffer};
        int pid = -1;
        if (!(row >> pid))
            continue;
        std::string command;
        std::getline(row, command);
        const bool commandMatches =
            checkpointEvalId
                ? CommandHasExactOptionValue(
                      command,
                      "--scheduler-checkpoint-eval-id",
                      *checkpointEvalId)
                : (CommandHasExactOptionValue(
                       command,
                       "--scheduler-experiment-id",
                       experimentId) &&
                   ((phase == "train" &&
                     CommandHasExactToken(command, "--train")) ||
                    (phase == "infer" &&
                     CommandHasExactToken(command, "--infer")) ||
                    (phase == "analyze" &&
                     CommandHasExactOptionValue(
                         command,
                         "--analyze-experiment",
                         experimentId))));
        if (!commandMatches)
            continue;
        result.matchingCommandObserved = true;
        result.matchingPid = pid;
        const auto observation = processes->Observe(pid);
        if (observation.permissionDenied ||
            (!observation.inspectionSucceeded &&
             observation.exists))
            result.inspectionDenied = true;
        break;
    }
    result.inspectionSucceeded = ::pclose(pipe) == 0;
    return result;
}

int RecoverOrphanedRunningExperiments(
    pqxx::work& transaction,
    const SchedulerOptions& options,
    SchedulerEventLogState* logState,
    bool verbose)
{
    (void)logState;
    (void)verbose;
    RequireAndRefreshSchedulerAuthority(transaction, options);
    pqxx::result attempts = transaction.exec(
        "SELECT worker_attempt_id,lifecycle_state,ownership_origin,"
        "worker_pid,worker_process_group_id,"
        "worker_process_start_identity,canonical_executable_path,"
        "command_line,experiment_id,checkpoint_eval_id,"
        "lifecycle_phase,worker_kind,"
        "extract(epoch from reserved_at)::double precision,"
        "(reserved_at <= clock_timestamp()-make_interval(secs=>"
        + std::to_string(kSchedulerLaunchRecoveryGraceSeconds) +
        ")) AS recovery_ready,"
        "scheduler_invocation_id,scheduler_fencing_token,"
        "capacity_class,command_identity "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE lifecycle_state IN "
        "('reserved','spawned','running','observed',"
        "'stopped','identity_ambiguous') "
        "ORDER BY worker_attempt_id FOR UPDATE;");
    const pqxx::row protocol = transaction.exec(
        "SELECT cutover_state,"
        "cutover_completed_at IS NOT NULL AND "
        "cutover_completed_at + "
        "make_interval(secs=>legacy_no_pid_grace_seconds) "
        "<= clock_timestamp() AS legacy_grace_elapsed "
        "FROM experiment_scheduler_protocol "
        "WHERE singleton=true;").one_row();
    const bool cutoverComplete =
        protocol[0].as<std::string>() == "complete";
    const bool legacyGraceElapsed =
        protocol[1].as<bool>();
    const SchedulerProcessAbsenceEvidence schedulerProcesses =
        InspectAllSchedulerDispatchProcesses();
    const bool foreignSchedulerObserved =
        !schedulerProcesses.inspectionSucceeded ||
        std::any_of(
            schedulerProcesses.schedulers.begin(),
            schedulerProcesses.schedulers.end(),
            [](const auto& scheduler) {
                return scheduler.first !=
                       static_cast<int>(::getpid());
            });

    std::unique_ptr<EA::GlobalExperimentControl::ProcessOperations>
        processes =
            EA::GlobalExperimentControl::CreateNativeProcessOperations();
    int reconciled = 0;
    for (const pqxx::row& row : attempts)
    {
        const long long attemptId = row[0].as<long long>();
        const std::string state = row[1].as<std::string>();
        const std::string origin = row[2].as<std::string>();
        const long long experimentId = row[8].as<long long>();
        const std::optional<long long> checkpointEvalId =
            row[9].is_null()
                ? std::nullopt
                : std::optional<long long>{
                      row[9].as<long long>()};
        const std::string phase = row[10].as<std::string>();
        const std::string workerKind =
            row[11].as<std::string>();
        const double attemptStartedEpoch = row[12].as<double>();
        const bool recoveryReady = row[13].as<bool>();
        const std::optional<std::string> attemptSchedulerInvocation =
            row[14].is_null()
                ? std::nullopt
                : std::optional<std::string>{
                      row[14].as<std::string>()};
        const std::optional<long long> attemptFence =
            row[15].is_null()
                ? std::nullopt
                : std::optional<long long>{
                      row[15].as<long long>()};

        if (workerKind == "checkpoint_analyze")
        {
            const bool currentInProcessAttempt =
                attemptSchedulerInvocation ==
                    std::optional<std::string>{
                        options.schedulerAuthority
                            .schedulerInvocationId} &&
                attemptFence ==
                    std::optional<long long>{
                        options.schedulerAuthority.fencingToken};
            if (currentInProcessAttempt)
                continue;
            pqxx::result abandoned = transaction.exec_params(
                "UPDATE experiment_scheduler_worker_attempt a SET "
                "lifecycle_state='abandoned',"
                "completed_at=clock_timestamp(),"
                "last_observed_at=clock_timestamp(),"
                "observed_by_scheduler_invocation_id=$1,"
                "reconciled_at=clock_timestamp(),"
                "reconciled_by_scheduler_invocation_id=$1,"
                "reconciliation_result="
                "'checkpoint_analysis_owner_lost',"
                "diagnostic='stale_in_process_analysis_requeued' "
                "WHERE a.worker_attempt_id=$2 "
                "AND a.worker_kind='checkpoint_analyze' "
                "AND a.lifecycle_state IN ('reserved','running') "
                "AND EXISTS (SELECT 1 "
                " FROM experiment_checkpoint_eval ce "
                " WHERE ce.checkpoint_eval_id=$3 "
                " AND ce.status='running' AND ce.phase='analyze' "
                " AND ce.active_scheduler_worker_attempt_id="
                "a.worker_attempt_id) "
                "RETURNING a.worker_attempt_id;",
                options.schedulerAuthority.schedulerInvocationId,
                attemptId,
                checkpointEvalId);
            if (abandoned.size() == 1)
            {
                pqxx::result requeued = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET "
                    "status='pending',phase='analyze',"
                    "active_scheduler_worker_attempt_id=NULL,"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "error_message="
                    "'checkpoint_analysis_owner_lost_retryable',"
                    "updated_at=clock_timestamp() "
                    "WHERE checkpoint_eval_id=$1 "
                    "AND status='running' AND phase='analyze' "
                    "AND active_scheduler_worker_attempt_id=$2 "
                    "RETURNING checkpoint_eval_id;",
                    *checkpointEvalId,
                    attemptId);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    requeued,
                    "requeue_stale_checkpoint_analysis_attempt");
                ++reconciled;
            }
            continue;
        }

        if (row[3].is_null())
        {
            if (origin == "legacy_unverified")
            {
                bool lifecycleMatches = false;
                if (checkpointEvalId)
                {
                    lifecycleMatches =
                        transaction.exec_params(
                            "SELECT 1 FROM "
                            "experiment_checkpoint_eval "
                            "WHERE checkpoint_eval_id=$1 "
                            "AND status='running' AND phase=$2 "
                            "AND active_scheduler_worker_attempt_id=$3 "
                            "FOR UPDATE;",
                            *checkpointEvalId,
                            phase,
                            attemptId).size() == 1;
                }
                else
                {
                    lifecycleMatches =
                        transaction.exec_params(
                            "SELECT 1 FROM experiment "
                            "WHERE experiment_id=$1 "
                            "AND status='running' AND phase=$2 "
                            "AND active_scheduler_worker_attempt_id=$3 "
                            "FOR UPDATE;",
                            experimentId,
                            phase,
                            attemptId).size() == 1;
                }
                const LegacyNoPidProcessSearch processSearch =
                    SearchForLegacyNoPidProcess(
                        experimentId,
                        checkpointEvalId,
                        phase);
                const bool safeEnvironment =
                    cutoverComplete &&
                    legacyGraceElapsed &&
                    !foreignSchedulerObserved &&
                    processSearch.inspectionSucceeded &&
                    !processSearch.matchingCommandObserved &&
                    !processSearch.inspectionDenied;
                if (safeEnvironment && !lifecycleMatches)
                {
                    pqxx::result detached =
                        transaction.exec_params(
                            "UPDATE "
                            "experiment_scheduler_worker_attempt a SET "
                            "lifecycle_state='abandoned',"
                            "completed_at=clock_timestamp(),"
                            "last_observed_at=clock_timestamp(),"
                            "observed_by_scheduler_invocation_id=$1,"
                            "reconciled_at=clock_timestamp(),"
                            "reconciled_by_scheduler_invocation_id=$1,"
                            "reconciliation_result="
                            "'legacy_no_pid_lifecycle_detached',"
                            "diagnostic="
                            "'exact_lifecycle_binding_absent_after_cutover' "
                            "WHERE a.worker_attempt_id=$2 "
                            "AND a.ownership_origin="
                            "'legacy_unverified' "
                            "AND a.worker_pid IS NULL "
                            "AND a.lifecycle_state="
                            "'identity_ambiguous' "
                            "AND NOT EXISTS (SELECT 1 "
                            " FROM experiment e "
                            " WHERE e.active_scheduler_worker_attempt_id="
                            "a.worker_attempt_id) "
                            "AND NOT EXISTS (SELECT 1 "
                            " FROM experiment_checkpoint_eval ce "
                            " WHERE ce.active_scheduler_worker_attempt_id="
                            "a.worker_attempt_id) "
                            "RETURNING a.worker_attempt_id;",
                            options.schedulerAuthority
                                .schedulerInvocationId,
                            attemptId);
                    if (detached.size() == 1)
                    {
                        ++reconciled;
                        continue;
                    }
                }
                const bool safeToReconcile =
                    lifecycleMatches && safeEnvironment;
                if (!safeToReconcile)
                {
                    std::string diagnostic =
                        !lifecycleMatches
                            ? "legacy_no_pid_lifecycle_no_longer_bound"
                        : !cutoverComplete
                            ? "legacy_no_pid_cutover_incomplete"
                        : !legacyGraceElapsed
                            ? "legacy_no_pid_grace_not_elapsed"
                        : foreignSchedulerObserved
                            ? "legacy_no_pid_scheduler_process_present"
                        : !processSearch.inspectionSucceeded
                            ? "legacy_no_pid_process_search_failed"
                        : processSearch.inspectionDenied
                            ? "legacy_no_pid_process_inspection_denied"
                            : "legacy_no_pid_matching_process_observed";
                    transaction.exec_params(
                        "UPDATE "
                        "experiment_scheduler_worker_attempt SET "
                        "lifecycle_state='identity_ambiguous',"
                        "last_observed_at=clock_timestamp(),"
                        "observed_by_scheduler_invocation_id=$1,"
                        "reconciliation_result="
                        "'legacy_no_pid_unresolved',"
                        "diagnostic=$2 "
                        "WHERE worker_attempt_id=$3 "
                        "AND ownership_origin='legacy_unverified' "
                        "AND worker_pid IS NULL "
                        "AND lifecycle_state='identity_ambiguous';",
                        options.schedulerAuthority
                            .schedulerInvocationId,
                        diagnostic,
                        attemptId);
                    continue;
                }

                pqxx::result terminal =
                    transaction.exec_params(
                        "UPDATE "
                        "experiment_scheduler_worker_attempt a SET "
                        "lifecycle_state='abandoned',"
                        "completed_at=clock_timestamp(),"
                        "last_observed_at=clock_timestamp(),"
                        "observed_by_scheduler_invocation_id=$1,"
                        "reconciled_at=clock_timestamp(),"
                        "reconciled_by_scheduler_invocation_id=$1,"
                        "reconciliation_result="
                        "'legacy_no_pid_proven_absent_after_cutover',"
                        "diagnostic="
                        "'bounded_legacy_no_pid_reconciliation' "
                        "WHERE a.worker_attempt_id=$2 "
                        "AND a.ownership_origin='legacy_unverified' "
                        "AND a.worker_pid IS NULL "
                        "AND a.lifecycle_state='identity_ambiguous' "
                        "AND EXISTS ("
                        " SELECT 1 FROM "
                        "experiment_scheduler_protocol p "
                        " WHERE p.singleton "
                        " AND p.required_generation=$3 "
                        " AND p.cutover_state='complete' "
                        " AND p.cutover_completed_at + "
                        "make_interval(secs=>"
                        "p.legacy_no_pid_grace_seconds) "
                        "<=clock_timestamp()) "
                        "AND EXISTS ("
                        " SELECT 1 FROM experiment e "
                        " WHERE $4::bigint IS NULL "
                        " AND e.experiment_id=$5 "
                        " AND e.status='running' AND e.phase=$6 "
                        " AND e.active_scheduler_worker_attempt_id="
                        "a.worker_attempt_id "
                        " UNION ALL "
                        " SELECT 1 FROM "
                        "experiment_checkpoint_eval ce "
                        " WHERE $4::bigint IS NOT NULL "
                        " AND ce.checkpoint_eval_id=$4 "
                        " AND ce.status='running' AND ce.phase=$6 "
                        " AND ce.active_scheduler_worker_attempt_id="
                        "a.worker_attempt_id"
                        ") RETURNING a.worker_attempt_id;",
                        options.schedulerAuthority
                            .schedulerInvocationId,
                        attemptId,
                        EA::SchedulerOwnership::
                            kProtocolGeneration,
                        checkpointEvalId,
                        experimentId,
                        phase);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    terminal,
                    "terminalize_legacy_no_pid_attempt");
                pqxx::result lifecycle;
                if (checkpointEvalId)
                {
                    lifecycle = transaction.exec_params(
                        "UPDATE experiment_checkpoint_eval SET "
                        "status='failed',completed_at=clock_timestamp(),"
                        "error_message="
                        "'legacy_no_pid_proven_absent_after_cutover',"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "updated_at=clock_timestamp() "
                        "WHERE checkpoint_eval_id=$1 "
                        "AND status='running' AND phase=$2 "
                        "AND active_scheduler_worker_attempt_id=$3 "
                        "RETURNING checkpoint_eval_id;",
                        *checkpointEvalId,
                        phase,
                        attemptId);
                }
                else
                {
                    lifecycle = transaction.exec_params(
                        "UPDATE experiment SET status='failed',"
                        "completed_at=clock_timestamp(),exit_code=-1,"
                        "error_message="
                        "'legacy_no_pid_proven_absent_after_cutover',"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$1 "
                        "AND status='running' AND phase=$2 "
                        "AND active_scheduler_worker_attempt_id=$3 "
                        "RETURNING experiment_id;",
                        experimentId,
                        phase,
                        attemptId);
                }
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    lifecycle,
                    "clear_legacy_no_pid_lifecycle_binding");
                std::cout
                    << "SCHEDULER_LEGACY_NO_PID_RECONCILED"
                    << ",worker_attempt_id=" << attemptId
                    << ",experiment_id=" << experimentId
                    << ",checkpoint_eval_id="
                    << (checkpointEvalId
                            ? std::to_string(
                                  *checkpointEvalId)
                            : "NULL")
                    << ",result=proven_absent_after_cutover"
                    << std::endl;
                ++reconciled;
                continue;
            }
            if (!recoveryReady)
            {
                transaction.exec_params(
                    "UPDATE experiment_scheduler_worker_attempt SET "
                    "lifecycle_state='identity_ambiguous',"
                    "last_observed_at=clock_timestamp(),"
                    "observed_by_scheduler_invocation_id=$1,"
                    "diagnostic="
                    "'launch_reservation_within_recovery_grace' "
                    "WHERE worker_attempt_id=$2 "
                    "AND lifecycle_state IN "
                    "('reserved','identity_ambiguous');",
                    options.schedulerAuthority
                        .schedulerInvocationId,
                    attemptId);
                continue;
            }

            pqxx::result terminal = transaction.exec_params(
                "UPDATE experiment_scheduler_worker_attempt a SET "
                "lifecycle_state='launch_failed',"
                "completed_at=clock_timestamp(),exit_code=126,"
                "reconciliation_result='never_spawned',"
                "diagnostic='interrupted_launch_never_spawned' "
                "WHERE a.worker_attempt_id=$1 "
                "AND a.lifecycle_state='reserved' "
                "AND EXISTS ("
                " SELECT 1 FROM experiment e "
                " WHERE $2::bigint IS NULL "
                " AND e.experiment_id=$3 "
                " AND e.status='running' AND e.phase=$4 "
                " AND e.active_scheduler_worker_attempt_id="
                "a.worker_attempt_id "
                " UNION ALL "
                " SELECT 1 FROM experiment_checkpoint_eval ce "
                " WHERE $2::bigint IS NOT NULL "
                " AND ce.checkpoint_eval_id=$2 "
                " AND ce.status='running' AND ce.phase='infer' "
                " AND ce.active_scheduler_worker_attempt_id="
                "a.worker_attempt_id"
                ") RETURNING a.worker_attempt_id;",
                attemptId,
                checkpointEvalId,
                experimentId,
                phase);
            if (terminal.empty())
                continue;
            if (checkpointEvalId)
            {
                pqxx::result lifecycle = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET "
                    "status='failed',completed_at=clock_timestamp(),"
                    "error_message='interrupted_launch_never_spawned',"
                    "active_scheduler_worker_attempt_id=NULL,"
                    "updated_at=clock_timestamp() "
                    "WHERE checkpoint_eval_id=$1 "
                    "AND active_scheduler_worker_attempt_id=$2 "
                    "AND status='running' AND phase='infer' "
                    "RETURNING checkpoint_eval_id;",
                    *checkpointEvalId,
                    attemptId);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    lifecycle,
                    "fail_never_spawned_checkpoint_attempt");
            }
            else
            {
                pqxx::result lifecycle = transaction.exec_params(
                    "UPDATE experiment SET status='failed',"
                    "completed_at=clock_timestamp(),exit_code=126,"
                    "error_message='interrupted_launch_never_spawned',"
                    "active_scheduler_worker_attempt_id=NULL,"
                    "updated_at=clock_timestamp() "
                    "WHERE experiment_id=$1 "
                    "AND active_scheduler_worker_attempt_id=$2 "
                    "AND status='running' AND phase=$3 "
                    "RETURNING experiment_id;",
                    experimentId,
                    attemptId,
                    phase);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    lifecycle,
                    "fail_never_spawned_experiment_attempt");
            }
            std::cout << "SCHEDULER_INTERRUPTED_LAUNCH_RECONCILED"
                      << ",worker_attempt_id=" << attemptId
                      << ",result=never_spawned"
                      << std::endl;
            ++reconciled;
            continue;
        }

        const int pid = row[3].as<int>();
        EA::GlobalExperimentControl::ProcessObservation observation;
        bool identityMatches = false;
        bool processMissing = false;
        EA::SchedulerCore::OrphanedRunningAttempt candidate;
        candidate.workerAttemptId = attemptId;
        candidate.experimentId = experimentId;
        candidate.checkpointEvalId = checkpointEvalId;
        candidate.lifecycleState = state;
        candidate.phase = phase;

        EA::SchedulerCore::OrphanedRunningExperimentReconciliationOperations
            reconciliationOperations;
        reconciliationOperations.loadCandidates =
            [&] { return std::vector{candidate}; };
        reconciliationOperations.observeProcess = [&](const auto&) {
            observation = processes->Observe(pid);
            identityMatches =
                observation.inspectionSucceeded &&
                observation.exists &&
                !row[4].is_null() &&
                !row[5].is_null() &&
                !row[6].is_null() &&
                observation.pid == pid &&
                observation.processGroupId == row[4].as<int>() &&
                observation.processStartIdentity ==
                    row[5].as<std::string>() &&
                CanonicalizeObservedExecutable(
                    observation.executable) ==
                    row[6].as<std::string>();
            if (origin != "legacy_unverified")
            {
                identityMatches =
                    identityMatches &&
                    CommandHasExactOptionValue(
                        observation.commandLine,
                        "--scheduler-worker-attempt-id",
                        attemptId);
            }
            if (workerKind == "checkpoint_infer")
            {
                identityMatches =
                    identityMatches && checkpointEvalId &&
                    CommandHasExactToken(
                        observation.commandLine, "--infer") &&
                    CommandHasExactOptionValue(
                        observation.commandLine,
                        "--scheduler-checkpoint-eval-id",
                        *checkpointEvalId);
            }
            else if (phase == "train")
            {
                identityMatches =
                    identityMatches &&
                    CommandHasExactToken(
                        observation.commandLine, "--train") &&
                    CommandHasExactOptionValue(
                        observation.commandLine,
                        "--scheduler-experiment-id",
                        experimentId);
            }
            else if (phase == "infer")
            {
                identityMatches =
                    identityMatches &&
                    CommandHasExactToken(
                        observation.commandLine, "--infer") &&
                    CommandHasExactOptionValue(
                        observation.commandLine,
                        "--scheduler-experiment-id",
                        experimentId);
            }
            else if (phase == "analyze")
            {
                identityMatches =
                    identityMatches &&
                    CommandHasExactOptionValue(
                        observation.commandLine,
                        "--analyze-experiment",
                        experimentId);
            }
            return EA::SchedulerCore::AttemptObservation{
                observation.inspectionSucceeded,
                observation.exists,
                identityMatches,
                state == "stopped",
                observation.stopped};
        };
        reconciliationOperations.persistProcessObservation =
            [&](const auto&, const auto& observationPlan) {
                if (observationPlan.action ==
                    EA::SchedulerCore::AttemptObservationAction::Defer)
                {
                    transaction.exec_params(
                        "UPDATE experiment_scheduler_worker_attempt SET "
                        "lifecycle_state=$1,"
                        "last_observed_at=clock_timestamp(),"
                        "observed_by_scheduler_invocation_id=$2,"
                        "diagnostic=$3 WHERE worker_attempt_id=$4;",
                        observationPlan.lifecycleState,
                        options.schedulerAuthority.schedulerInvocationId,
                        observationPlan.diagnostic,
                        attemptId);
                    return;
                }
                transaction.exec_params(
                    "UPDATE experiment_scheduler_worker_attempt SET "
                    "lifecycle_state=$1,last_observed_at=clock_timestamp(),"
                    "observed_by_scheduler_invocation_id=$2,"
                    "reconciliation_result=$3,diagnostic=$4 "
                    "WHERE worker_attempt_id=$5;",
                    observationPlan.lifecycleState,
                    options.schedulerAuthority.schedulerInvocationId,
                    observationPlan.reconciliationResult,
                    observationPlan.diagnostic,
                    attemptId);
                if (observationPlan.restoreRunningLifecycle &&
                    !checkpointEvalId)
                {
                    transaction.exec_params(
                        "UPDATE experiment SET status='running',"
                        "resume_requested=false,"
                        "scheduler_resume_origin='none',"
                        "worker_control_state='running',"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$1 AND phase=$2 "
                        "AND status IN ('paused','pending') "
                        "AND active_scheduler_worker_attempt_id=$3;",
                        experimentId,
                        phase,
                        attemptId);
                }
                if (observationPlan.action ==
                    EA::SchedulerCore::AttemptObservationAction::RetainLive)
                {
                    std::cout
                        << (identityMatches
                                ? "SCHEDULER_PRIOR_WORKER_OBSERVED"
                                : "SCHEDULER_WORKER_IDENTITY_AMBIGUOUS")
                        << ",worker_attempt_id=" << attemptId
                        << ",experiment_id=" << experimentId
                        << ",pid=" << pid
                        << ",capacity_consumed="
                        << (observationPlan.capacityConsumed ? 1 : 0)
                        << std::endl;
                }
            };
        reconciliationOperations.reconcileMissingProcess =
            [&](const auto&) {
                processMissing = true;
                return false;
            };
        EA::SchedulerCore::ReconciliationService reconciliationService{
            std::move(reconciliationOperations)};
        (void)reconciliationService.recoverOrphanedRunningExperiments();
        if (!processMissing)
            continue;

        bool stoppedInferHasAuthoritativeResult = false;
        if (state == "stopped" && !checkpointEvalId && phase == "infer")
        {
            const auto stoppedExperiment = LoadExperimentCheckpointIdentity(
                transaction, experimentId);
            stoppedInferHasAuthoritativeResult =
                stoppedExperiment.has_value() &&
                HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
                    transaction,
                    *stoppedExperiment,
                    attemptId,
                    OperatorForcedFinalInferenceRerunRequested(
                        transaction, experimentId));
        }

        if (state == "stopped" && !checkpointEvalId &&
            !stoppedInferHasAuthoritativeResult)
        {
            const pqxx::result stoppedLifecycle =
                transaction.exec_params(
                    "SELECT status,resume_requested,scheduler_resume_origin "
                    "FROM experiment "
                    "WHERE experiment_id=$1 AND phase=$2 "
                    "AND status IN ('paused','pending') "
                    "AND active_scheduler_worker_attempt_id=$3 "
                    "FOR UPDATE;",
                    experimentId,
                    phase,
                    attemptId);
            if (stoppedLifecycle.size() == 1)
            {
                const std::string stoppedStatus =
                    stoppedLifecycle[0][0].as<std::string>();
                const bool stoppedResumeRequested =
                    stoppedLifecycle[0][1].as<bool>();
                const std::string stoppedResumeOrigin =
                    stoppedLifecycle[0][2].as<std::string>();
                auto missingPlan =
                    EA::SchedulerCore::PlanMissingStoppedWorker(
                        phase,
                        stoppedStatus,
                        stoppedResumeRequested,
                        stoppedResumeOrigin,
                        false);
                const bool preemptedTrainRestart =
                    missingPlan.disposition ==
                    EA::SchedulerCore::MissingStoppedWorkerDisposition::
                        FailPreemptedTrainWithoutCheckpoint;
                std::optional<QueueResumeMeta> restartCheckpoint;
                if (preemptedTrainRestart)
                {
                    const auto checkpointExperiment =
                        LoadExperimentCheckpointIdentity(
                            transaction, experimentId);
                    if (!checkpointExperiment.has_value())
                        throw std::runtime_error(
                            "missing_preempted_checkpoint_experiment");
                    restartCheckpoint =
                        SelectUsableTrainingCheckpoint(
                            transaction,
                            *checkpointExperiment,
                            std::nullopt,
                            false)
                            .checkpoint;
                    missingPlan =
                        EA::SchedulerCore::PlanMissingStoppedWorker(
                            phase,
                            stoppedStatus,
                            stoppedResumeRequested,
                            stoppedResumeOrigin,
                            restartCheckpoint.has_value());
                }
                pqxx::result retired = transaction.exec_params(
                    "UPDATE experiment_scheduler_worker_attempt SET "
                    "lifecycle_state='abandoned',"
                    "completed_at=clock_timestamp(),"
                    "last_observed_at=clock_timestamp(),"
                    "observed_by_scheduler_invocation_id=$1,"
                    "reconciled_at=clock_timestamp(),"
                    "reconciled_by_scheduler_invocation_id=$1,"
                    "reconciliation_result='stopped_process_missing',"
                    "diagnostic='resume_priority_preserved_for_restart' "
                    "WHERE worker_attempt_id=$2 "
                    "AND lifecycle_state='stopped' "
                    "RETURNING worker_attempt_id;",
                    options.schedulerAuthority.schedulerInvocationId,
                    attemptId);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    retired,
                    "reconcile_missing_stopped_worker_attempt");
                pqxx::result detached;
                if (missingPlan.disposition ==
                    EA::SchedulerCore::MissingStoppedWorkerDisposition::
                        FailPreemptedTrainWithoutCheckpoint)
                {
                    detached = transaction.exec(
                        "UPDATE experiment SET status='failed',"
                        "resume_requested=false,"
                        "scheduler_resume_origin='none',worker_pid=NULL,"
                        "worker_process_group_id=NULL,"
                        "worker_process_start_identity=NULL,"
                        "worker_executable=NULL,worker_command_line=NULL,"
                        "worker_control_state='running',"
                        "active_scheduler_worker_attempt_id=NULL,exit_code=-1,"
                        "error_message="
                        "'preempted_worker_missing_no_valid_checkpoint',"
                        "completed_at=clock_timestamp(),"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$1 AND phase='train' "
                        "AND status='pending' AND resume_requested=true "
                        "AND scheduler_resume_origin='preemption' "
                        "AND active_scheduler_worker_attempt_id=$2 "
                        "RETURNING experiment_id;",
                        pqxx::params{experimentId, attemptId});
                }
                else if (missingPlan.disposition ==
                         EA::SchedulerCore::
                             MissingStoppedWorkerDisposition::
                                 RestartPreemptedTrainFromCheckpoint)
                {
                    detached = transaction.exec(
                        "UPDATE experiment SET worker_pid=NULL,"
                        "worker_process_group_id=NULL,"
                        "worker_process_start_identity=NULL,"
                        "worker_executable=NULL,worker_command_line=NULL,"
                        "worker_control_state='running',"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "last_model_id=$1,resume_model_id=$1,"
                        "current_operation='train',"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$2 AND phase='train' "
                        "AND status='pending' AND resume_requested=true "
                        "AND scheduler_resume_origin='preemption' "
                        "AND active_scheduler_worker_attempt_id=$3 "
                        "RETURNING experiment_id;",
                        pqxx::params{
                            restartCheckpoint->modelId,
                            experimentId,
                            attemptId});
                }
                else
                {
                    detached = transaction.exec_params(
                        "UPDATE experiment SET worker_pid=NULL,"
                        "worker_process_group_id=NULL,"
                        "worker_process_start_identity=NULL,"
                        "worker_executable=NULL,worker_command_line=NULL,"
                        "worker_control_state=CASE WHEN status='paused' "
                        "THEN 'paused' ELSE 'running' END,"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$1 AND phase=$2 "
                        "AND status IN ('paused','pending') "
                        "AND active_scheduler_worker_attempt_id=$3 "
                        "RETURNING experiment_id;",
                        experimentId,
                        phase,
                        attemptId);
                }
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    detached,
                    "detach_reconciled_missing_stopped_worker_attempt");
                std::cout
                    << "SCHEDULER_STOPPED_WORKER_RECONCILED"
                    << ",worker_attempt_id=" << attemptId
                    << ",experiment_id=" << experimentId
                    << ",phase=" << phase
                    << ",resume_requested="
                    << (stoppedLifecycle[0][1].as<bool>()
                            ? "true" : "false")
                    << ",resume_origin="
                    << stoppedLifecycle[0][2].as<std::string>()
                    << ",result="
                    << missingPlan.eventResult
                    << std::endl;
                ++reconciled;
                continue;
            }
        }

        // A missing process is destructive evidence only when the exact
        // durable attempt still owns the lifecycle row.
        bool lifecycleMatches = false;
        bool lifecycleAlreadyCompletedAnalyze = false;
        if (checkpointEvalId)
        {
            pqxx::result lifecycle = transaction.exec_params(
                "SELECT 1 FROM experiment_checkpoint_eval "
                "WHERE checkpoint_eval_id=$1 AND status='running' "
                "AND phase='infer' "
                "AND active_scheduler_worker_attempt_id=$2 "
                "FOR UPDATE;",
                *checkpointEvalId,
                attemptId);
            lifecycleMatches = lifecycle.size() == 1;
        }
        else
        {
            // Final analysis persists its durable result and advances the
            // experiment before the scheduler terminalizes the attempt.
            // Keep that exact bound crash window eligible for result recovery.
            pqxx::result lifecycle = transaction.exec_params(
                "SELECT status,phase FROM experiment "
                "WHERE experiment_id=$1 "
                "AND ((status='running' AND phase=$2) "
                " OR ($2='infer' AND status='pending' AND phase='infer' "
                "     AND resume_requested=true "
                "     AND scheduler_resume_origin='preemption') "
                " OR ($2='analyze' AND status='completed' "
                "     AND phase='done')) "
                "AND active_scheduler_worker_attempt_id=$3 "
                "FOR UPDATE;",
                experimentId,
                phase,
                attemptId);
            lifecycleMatches = lifecycle.size() == 1;
            lifecycleAlreadyCompletedAnalyze =
                lifecycleMatches &&
                lifecycle[0][0].as<std::string>() == "completed" &&
                lifecycle[0][1].as<std::string>() == "done";
        }
        if (!lifecycleMatches)
        {
            transaction.exec_params(
                "UPDATE experiment_scheduler_worker_attempt a SET "
                "lifecycle_state='failed',"
                "completed_at=clock_timestamp(),"
                "reconciliation_result='lifecycle_predicate_changed',"
                "diagnostic='process_missing_lifecycle_no_longer_owned' "
                "WHERE a.worker_attempt_id=$1 "
                "AND a.lifecycle_state IN "
                "('reserved','spawned','running','observed',"
                "'stopped','identity_ambiguous') "
                "AND NOT EXISTS (SELECT 1 FROM experiment e "
                " WHERE e.active_scheduler_worker_attempt_id="
                "a.worker_attempt_id) "
                "AND NOT EXISTS (SELECT 1 "
                " FROM experiment_checkpoint_eval ce "
                " WHERE ce.active_scheduler_worker_attempt_id="
                "a.worker_attempt_id);",
                attemptId);
            ++reconciled;
            continue;
        }

        bool completedEvidence = false;
        if (checkpointEvalId)
        {
            std::vector<CheckpointEvalRow> evaluations =
                LoadCheckpointEvalRows(
                    transaction, "running", "infer");
            const auto evaluation = std::find_if(
                evaluations.begin(),
                evaluations.end(),
                [&](const CheckpointEvalRow& value) {
                    return value.checkpointEvalId ==
                           *checkpointEvalId;
                });
            if (evaluation != evaluations.end())
            {
                const auto resultId =
                    FindAuthoritativeCompletedCheckpointInferenceResultForWorkerAttempt(
                        transaction, *evaluation, attemptId);
                if (resultId)
                {
                    AdvanceCheckpointEvalToAnalyze(
                        transaction,
                        *evaluation,
                        *resultId,
                        ColumnExists(
                            transaction,
                            "experiment_checkpoint_eval",
                            "infer_completed_at"),
                        attemptId);
                    completedEvidence = true;
                }
            }
            if (!completedEvidence)
            {
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET "
                    "status='failed',worker_pid=NULL,"
                    "worker_process_group_id=NULL,"
                    "completed_at=clock_timestamp(),"
                    "error_message='worker_process_missing_no_result',"
                    "updated_at=clock_timestamp() "
                    "WHERE checkpoint_eval_id=$1 "
                    "AND active_scheduler_worker_attempt_id=$2 "
                    "AND status='running' AND phase='infer';",
                    *checkpointEvalId,
                    attemptId);
            }
        }
        else
        {
            const bool forcedFinalInferenceRerun =
                phase == "infer" &&
                OperatorForcedFinalInferenceRerunRequested(
                    transaction, experimentId);
            pqxx::result experimentRows = transaction.exec_params(
                "SELECT experiment_id,symbol,prediction_horizon,"
                "c_next_threshold,core_lr_mult,head_lr_mult,"
                "target_epochs,checkpoint_interval,"
                "train_start::text,train_end::text,"
                "infer_start::text,infer_end::text,last_model_id,"
                "resume_model_id,train_log_path,infer_log_path,"
                "analysis_log_path,donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,fresh_initialization_seed,resume_expand_input_width,training_objective_canonical,training_objective_hash "
                "FROM experiment WHERE experiment_id=$1;",
                experimentId);
            if (experimentRows.size() == 1)
            {
                ExperimentRow experiment =
                    RowToExperiment(experimentRows[0]);
                if (phase == "infer" &&
                    HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
                        transaction, experiment, attemptId,
                        forcedFinalInferenceRerun))
                {
                    TransitionRecoveredInferenceToAnalyze(
                        transaction,
                        experiment,
                        "SCHEDULER_WORKER_RESULT_RECOVERED",
                        "completed_inference_result",
                        std::nullopt,
                        attemptId,
                        forcedFinalInferenceRerun);
                    if (state == "stopped")
                    {
                        const pqxx::result clearedPreemption =
                            transaction.exec(
                                "UPDATE experiment SET "
                                "resume_requested=false,"
                                "scheduler_resume_origin='none',"
                                "updated_at=clock_timestamp() "
                                "WHERE experiment_id=$1 "
                                "AND status='pending' AND phase='analyze' "
                                "AND active_scheduler_worker_attempt_id=$2 "
                                "RETURNING experiment_id;",
                                pqxx::params{experimentId, attemptId});
                        EA::SchedulerOwnership::RequireAffectedExactlyOne(
                            clearedPreemption,
                            "clear_recovered_infer_preemption_origin");
                    }
                    completedEvidence = true;
                }
                else if (
                    phase == "analyze" &&
                    HasCompletedAnalysisResultForAttempt(
                        transaction,
                        experiment,
                        attemptStartedEpoch))
                {
                    if (!lifecycleAlreadyCompletedAnalyze)
                    {
                        MarkExperimentDone(
                            transaction,
                            experiment,
                            "analyze",
                            attemptId);
                    }
                    completedEvidence = true;
                }
                else if (phase == "train")
                {
                    const TrainingCheckpointSelection selection =
                        SelectUsableTrainingCheckpoint(
                            transaction,
                            experiment,
                            attemptStartedEpoch,
                            true);
                    if (selection.checkpoint.has_value())
                    {
                        completedEvidence =
                            RecoverTrainOrphanFromModel(
                                transaction,
                                experiment,
                                selection.checkpoint->modelId,
                                attemptId);
                    }
                }
            }
            if (!completedEvidence)
            {
                transaction.exec_params(
                    "UPDATE experiment SET status='failed',"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "completed_at=clock_timestamp(),exit_code=-1,"
                    "error_message=$4,"
                    "updated_at=clock_timestamp() "
                    "WHERE experiment_id=$1 "
                    "AND active_scheduler_worker_attempt_id=$2 "
                    "AND status='running' AND phase=$3;",
                    experimentId,
                    attemptId,
                    phase,
                    forcedFinalInferenceRerun
                        ? "forced_final_inference_rerun_missing_attempt_result;worker_process_missing_no_result"
                        : "worker_process_missing_no_result");
            }
        }

        // This exact attempt no longer has a process. Mark the process-control
        // dimension non-running before detaching it; the next reservation, if
        // any, restores running for the newly admitted worker.
        if (checkpointEvalId)
        {
            transaction.exec_params(
                "UPDATE experiment_checkpoint_eval SET "
                "worker_pid=NULL,worker_process_group_id=NULL,"
                "worker_process_start_identity=NULL,worker_executable=NULL,"
                "worker_command_line=NULL,worker_control_state='paused' "
                "WHERE checkpoint_eval_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2;",
                *checkpointEvalId, attemptId);
        }
        else
        {
            transaction.exec_params(
                "UPDATE experiment SET worker_pid=NULL,"
                "worker_process_group_id=NULL,"
                "worker_process_start_identity=NULL,worker_executable=NULL,"
                "worker_command_line=NULL,current_operation=NULL,"
                "worker_control_state='paused' "
                "WHERE experiment_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2;",
                experimentId, attemptId);
        }

        const auto terminalPlan =
            EA::SchedulerCore::PlanMissingProcessTerminalization(
                completedEvidence);
        pqxx::result terminal = transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt a SET "
            "lifecycle_state=$1,completed_at=clock_timestamp(),"
            "last_observed_at=clock_timestamp(),"
            "observed_by_scheduler_invocation_id=$2,"
            "reconciliation_result=$3,diagnostic=$4 "
            "WHERE a.worker_attempt_id=$5 "
            "AND a.lifecycle_state IN "
            "('spawned','running','observed','stopped',"
            "'identity_ambiguous') "
            "AND EXISTS ("
            " SELECT 1 FROM experiment e "
            " WHERE $6::bigint IS NULL "
            " AND e.experiment_id=$7 "
            " AND e.active_scheduler_worker_attempt_id="
            "a.worker_attempt_id "
            " UNION ALL "
            " SELECT 1 FROM experiment_checkpoint_eval ce "
            " WHERE $6::bigint IS NOT NULL "
            " AND ce.checkpoint_eval_id=$6 "
            " AND ce.active_scheduler_worker_attempt_id="
            "a.worker_attempt_id"
            ") RETURNING a.worker_attempt_id;",
            terminalPlan.attemptLifecycleState,
            options.schedulerAuthority.schedulerInvocationId,
            terminalPlan.reconciliationResult,
            terminalPlan.diagnostic,
            attemptId,
            checkpointEvalId,
            experimentId);
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            terminal,
            "terminalize_reconciled_exact_attempt");
        pqxx::result cleared;
        if (checkpointEvalId)
        {
            cleared = transaction.exec_params(
                "UPDATE experiment_checkpoint_eval SET "
                "active_scheduler_worker_attempt_id=NULL "
                "WHERE checkpoint_eval_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2 "
                "RETURNING checkpoint_eval_id;",
                *checkpointEvalId,
                attemptId);
        }
        else
        {
            cleared = transaction.exec_params(
                "UPDATE experiment SET "
                "active_scheduler_worker_attempt_id=NULL "
                "WHERE experiment_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2 "
                "RETURNING experiment_id;",
                experimentId,
                attemptId);
        }
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            cleared,
            "clear_reconciled_exact_attempt_binding");
        std::cout << "SCHEDULER_WORKER_RECONCILED"
                  << ",worker_attempt_id=" << attemptId
                  << ",experiment_id=" << experimentId
                  << ",result="
                  << (completedEvidence
                          ? "completed_evidence"
                          : "failed_no_result")
                  << std::endl;
        ++reconciled;
    }
    return reconciled;
}

void PersistObservedExitFields(pqxx::work& w,
                               const SchedulerOwnedChild& child,
                               int exitCode,
                               const std::string& error,
                               bool preserveSuccessState)
{
    pqxx::result updated = w.exec_params(
        "UPDATE experiment SET exit_code = $1, error_message = $2, worker_pid = NULL, updated_at = now() "
        "WHERE experiment_id = $3 AND (worker_pid = $4 OR worker_pid IS NULL) "
        "AND status <> 'running' "
        "AND active_scheduler_worker_attempt_id=$5 "
        "RETURNING experiment_id;",
        exitCode,
        preserveSuccessState && exitCode == 0 ? std::optional<std::string>{} : std::optional<std::string>{error},
        child.experimentId,
        static_cast<int>(child.pid),
        child.workerAttemptId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        updated, "persist_observed_exit_fields");
}

void PersistObservedExperimentChild(pqxx::work& w,
                                    const SchedulerOwnedChild& child,
                                    const SchedulerChildCompletionEvidence&
                                        completion)
{
    const int exitCode = completion.exitCode;
    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, "
        "e.donchian20_mode,e.feature_warmup_scope,e.donchian_lookback,e.feature_ablation_mask,e.fresh_initialization_seed,e.resume_expand_input_width,e.training_objective_canonical,e.training_objective_hash,e.status,e.phase,e.worker_pid,e.cancellation_request_id,"
        "r.cancellation_mode,e.cancel_after_checkpoint_epoch "
        "FROM experiment e LEFT JOIN experiment_admin_request r "
        "ON r.request_id=e.cancellation_request_id "
        "WHERE e.experiment_id = $1 "
        "AND e.active_scheduler_worker_attempt_id=$2;",
        child.experimentId,
        child.workerAttemptId);
    if (rows.empty())
        return;

    ExperimentRow experiment = RowToExperiment(rows[0]);
    const std::string status = rows[0][25].as<std::string>();
    const std::string phase = rows[0][26].as<std::string>();
    const std::optional<int> workerPid = rows[0][27].is_null()
        ? std::nullopt
        : std::optional<int>{rows[0][27].as<int>()};
    const std::optional<long long> cancellationRequestId =
        rows[0][28].is_null()
            ? std::nullopt
        : std::optional<long long>{rows[0][28].as<long long>()};
    const std::optional<std::string> cancellationMode =
        rows[0][29].is_null()
            ? std::nullopt
        : std::optional<std::string>{rows[0][29].as<std::string>()};
    const std::optional<int> cancellationCheckpoint =
        rows[0][30].is_null()
            ? std::nullopt
        : std::optional<int>{rows[0][30].as<int>()};
    const std::string& error = completion.error;

    if (status != "running" || phase != child.phase ||
        (workerPid.has_value() && *workerPid != child.pid))
    {
        PersistObservedExitFields(w, child, exitCode, error, true);
        std::cout << "SCHEDULER_CHILD_RESULT_ALREADY_PERSISTED"
                  << ",experiment_id=" << child.experimentId
                  << ",phase=" << child.phase
                  << ",pid=" << child.pid
                  << ",status=" << status
                  << ",database_phase=" << phase
                  << ",exit_code=" << exitCode
                  << std::endl;
        return;
    }

    if (cancellationRequestId)
    {
        if (cancellationMode == "after_next_checkpoint" &&
            child.phase == "train" && cancellationCheckpoint)
        {
            const std::optional<long long> restartModel =
                FindLatestModelForExperiment(w, child.experimentId);
            if (restartModel)
                w.exec_params(
                    "UPDATE experiment SET status='pending',phase='train',"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "worker_control_state='running',last_model_id=$1,"
                    "current_operation='train',"
                    "error_message=$2,updated_at=now() "
                    "WHERE experiment_id=$3 AND status='running' "
                    "AND phase='train' "
                    "AND active_scheduler_worker_attempt_id=$4;",
                    *restartModel,
                    error,
                    child.experimentId,
                    child.workerAttemptId);
            else
            {
                w.exec_params(
                    "UPDATE experiment SET status='cancelled',"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "completed_at=now(),cancellation_completed_at=now(),"
                    "error_message='cancelled_missing_worker_no_checkpoint',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND phase='train' "
                    "AND active_scheduler_worker_attempt_id=$2;",
                    child.experimentId,
                    child.workerAttemptId);
                w.exec_params(
                    "UPDATE experiment_admin_worker_outcome SET "
                    "outcome_status='partial',"
                    "detail='worker_exited_before_checkpoint_no_restart_model',"
                    "updated_at=now() WHERE request_id=$1 "
                    "AND experiment_id=$2 "
                    "AND outcome_status='pending_checkpoint';",
                    *cancellationRequestId,
                    child.experimentId);
            }
        }
        else
        {
            w.exec_params(
                "UPDATE experiment SET status='cancelled',"
                "worker_pid=NULL,worker_process_group_id=NULL,"
                "completed_at=COALESCE(completed_at,now()),"
                "cancellation_completed_at=now(),exit_code=$1,"
                "error_message='cancelled_by_global_request',updated_at=now() "
                "WHERE experiment_id=$2 AND status='running' "
                "AND phase=$3 "
                "AND active_scheduler_worker_attempt_id=$4;",
                exitCode,
                child.experimentId,
                child.phase,
                child.workerAttemptId);
        }
        return;
    }

    if (child.phase == "analyze" &&
        HasCompletedAnalysisResultForAttempt(
            w, experiment, child.launchedEpoch))
    {
        MarkExperimentDone(
            w, experiment, "analyze", child.workerAttemptId);
        PersistObservedExitFields(w, child, exitCode, error, true);
        return;
    }

    if (child.phase == "infer")
    {
        const bool forcedFinalInferenceRerun =
            OperatorForcedFinalInferenceRerunRequested(
                w, experiment.experimentId);
        if (HasAuthoritativeCompletedInferenceResultForWorkerAttempt(
                w,
                experiment,
                *child.workerAttemptId,
                forcedFinalInferenceRerun))
        {
            TransitionRecoveredInferenceToAnalyze(
                w,
                experiment,
                "SCHEDULER_CHILD_RESULT_PERSISTED",
                "completed_inference_eval_result",
                std::nullopt,
                child.workerAttemptId,
                forcedFinalInferenceRerun);
            PersistObservedExitFields(w, child, exitCode, error, true);
            return;
        }
        if (!forcedFinalInferenceRerun &&
            HasValidInferenceLogPathForAttempt(
                experiment, child.launchedEpoch))
        {
            TransitionRecoveredInferenceToAnalyze(
                w,
                experiment,
                "SCHEDULER_CHILD_RESULT_PERSISTED",
                "valid_infer_log_path",
                std::nullopt,
                child.workerAttemptId);
            PersistObservedExitFields(w, child, exitCode, error, true);
            return;
        }
        if (!forcedFinalInferenceRerun)
        {
            if (const std::optional<std::string> discoveredLog =
                DiscoverValidInferenceLogForAttempt(
                    experiment, child.launchedEpoch);
                discoveredLog.has_value())
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    experiment,
                    "SCHEDULER_CHILD_RESULT_PERSISTED",
                    "discovered_valid_infer_log",
                    discoveredLog,
                    child.workerAttemptId);
                PersistObservedExitFields(w, child, exitCode, error, true);
                return;
            }
        }
        if (forcedFinalInferenceRerun)
        {
            MarkExperimentFailed(
                w,
                experiment,
                "forced_final_inference_rerun_missing_attempt_result;" +
                    error,
                exitCode,
                child.workerAttemptId);
            return;
        }
    }

    if (child.phase == "train")
    {
        if (const std::optional<long long> modelId =
                FindLatestModelForExperimentSince(
                    w,
                    experiment.experimentId,
                    child.launchedEpoch);
            modelId.has_value() &&
            (!child.expectedModelId.has_value() || *modelId != *child.expectedModelId))
        {
            (void)TransitionAfterTrainModelAvailable(
                w,
                experiment,
                *modelId,
                exitCode,
                "train",
                child.workerAttemptId);
            PersistObservedExitFields(w, child, exitCode, error, true);
            return;
        }
    }

    MarkExperimentFailed(
        w, experiment, error, exitCode, child.workerAttemptId);
}

void AdvanceCheckpointEvalToAnalyze(pqxx::work& w,
                                    const CheckpointEvalRow& eval,
                                    long long inferenceResultId,
                                    bool hasInferCompletedAt,
                                    const std::optional<long long>&
                                        workerAttemptId);

void PersistObservedCheckpointChild(pqxx::work& w,
                                    const SchedulerOwnedChild& child,
                                    const SchedulerChildCompletionEvidence&
                                        completion)
{
    if (!child.checkpointEvalId.has_value())
        return;
    const int exitCode = completion.exitCode;
    const std::string& error = completion.error;
    CheckpointEvalRow eval;
    eval.checkpointEvalId = *child.checkpointEvalId;
    eval.experiment.experimentId = child.experimentId;
    eval.checkpointModelId = child.expectedModelId.value_or(-1);
    const std::optional<long long> resultId =
        FindCompletedCheckpointInferenceResultIdForAttempt(
            w, eval, child.launchedEpoch);

    if (resultId.has_value())
    {
        const bool hasInferCompletedAt =
            ColumnExists(w, "experiment_checkpoint_eval", "infer_completed_at");
        pqxx::result cancellation = w.exec_params(
            "SELECT cancellation_request_id FROM experiment_checkpoint_eval "
            "WHERE checkpoint_eval_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2;",
            *child.checkpointEvalId,
            child.workerAttemptId);
        if (!cancellation.empty() && !cancellation[0][0].is_null())
        {
            std::ostringstream sql;
            sql << "UPDATE experiment_checkpoint_eval SET status='completed',"
                << "phase='done',worker_pid=NULL,completed_at=now(),"
                << "updated_at=now(),error_message=NULL";
            if (hasInferCompletedAt)
                sql << ",infer_completed_at=now()";
            sql << " WHERE checkpoint_eval_id=$1 "
                << "AND active_scheduler_worker_attempt_id=$2 "
                << "RETURNING checkpoint_eval_id;";
            pqxx::result completed = w.exec_params(
                sql.str(),
                *child.checkpointEvalId,
                child.workerAttemptId);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                completed,
                "complete_cancellation_checkpoint_exact_attempt");
        }
        else
        {
            AdvanceCheckpointEvalToAnalyze(
                w,
                eval,
                *resultId,
                hasInferCompletedAt,
                child.workerAttemptId);
        }
        if (exitCode != 0)
        {
            w.exec_params(
                "UPDATE experiment_checkpoint_eval SET error_message = $1, updated_at = now() "
                "WHERE checkpoint_eval_id = $2 "
                "AND active_scheduler_worker_attempt_id=$3;",
                error,
                *child.checkpointEvalId,
                child.workerAttemptId);
        }
        return;
    }
    w.exec_params(
        "UPDATE experiment_checkpoint_eval SET status = 'failed', worker_pid = NULL, completed_at = now(), "
        "updated_at = now(), error_message = $1 WHERE checkpoint_eval_id = $2 "
        "AND status = 'running' AND phase = 'infer' "
        "AND active_scheduler_worker_attempt_id=$3;",
        error,
        *child.checkpointEvalId,
        child.workerAttemptId);
}

void PersistUnexpectedChildStatus(pqxx::work& w,
                                  const SchedulerOwnedChild& child,
                                  int rawStatus)
{
    const std::string error =
        "child_abnormal_wait_status;phase=" + child.phase +
        ";raw_status=" + std::to_string(rawStatus);
    if (child.checkpointEvalId.has_value())
    {
        w.exec_params(
            "UPDATE experiment_checkpoint_eval SET status = 'failed', worker_pid = NULL, "
            "completed_at = now(), updated_at = now(), error_message = $1 "
            "WHERE checkpoint_eval_id = $2 AND status = 'running' "
            "AND phase='infer' "
            "AND active_scheduler_worker_attempt_id=$3;",
            error,
            *child.checkpointEvalId,
            child.workerAttemptId);
        return;
    }
    w.exec_params(
        "UPDATE experiment SET status = 'failed', exit_code = -1, error_message = $1, "
        "worker_pid = NULL, completed_at = now(), updated_at = now() "
        "WHERE experiment_id = $2 AND status = 'running' "
        "AND phase=$4 "
        "AND (worker_pid = $3 OR worker_pid IS NULL) "
        "AND active_scheduler_worker_attempt_id=$5;",
        error,
        child.experimentId,
        static_cast<int>(child.pid),
        child.phase,
        child.workerAttemptId);
}

void FinalizeObservedWorkerAttempt(
    pqxx::work& transaction,
    const SchedulerOwnedChild& child,
    const SchedulerOptions& options,
    int exitCode,
    const std::optional<int>& signalNumber,
    const std::string& diagnostic)
{
    if (!child.workerAttemptId)
        return;
    bool lifecycleFailed = true;
    if (child.checkpointEvalId)
    {
        pqxx::result lifecycle = transaction.exec_params(
            "SELECT status FROM experiment_checkpoint_eval "
            "WHERE checkpoint_eval_id=$1;",
            *child.checkpointEvalId);
        lifecycleFailed =
            lifecycle.size() != 1 ||
            lifecycle[0][0].as<std::string>() == "failed";
    }
    else
    {
        pqxx::result lifecycle = transaction.exec_params(
            "SELECT status FROM experiment WHERE experiment_id=$1;",
            child.experimentId);
        lifecycleFailed =
            lifecycle.size() != 1 ||
            lifecycle[0][0].as<std::string>() == "failed";
    }
    pqxx::result terminal = transaction.exec_params(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state=$1,completed_at=clock_timestamp(),"
        "last_observed_at=clock_timestamp(),exit_code=$2,"
        "signal_number=$3,reconciliation_result='parent_observed_exit',"
        "diagnostic=$4 "
        "WHERE a.worker_attempt_id=$5 "
        "AND a.scheduler_invocation_id=$6 "
        "AND a.scheduler_fencing_token=$7 "
        "AND a.worker_pid=$8 "
        "AND a.lifecycle_state IN "
        "('spawned','running','observed') "
        "AND EXISTS ("
        " SELECT 1 FROM experiment e "
        " WHERE $9::bigint IS NULL "
        " AND e.experiment_id=$10 "
        " AND e.active_scheduler_worker_attempt_id="
        "a.worker_attempt_id "
        " UNION ALL "
        " SELECT 1 FROM experiment_checkpoint_eval ce "
        " WHERE $9::bigint IS NOT NULL "
        " AND ce.checkpoint_eval_id=$9 "
        " AND ce.active_scheduler_worker_attempt_id="
        "a.worker_attempt_id"
        ") RETURNING a.worker_attempt_id;",
        lifecycleFailed ? "failed" : "completed",
        exitCode,
        signalNumber,
        diagnostic,
        *child.workerAttemptId,
        options.schedulerAuthority.schedulerInvocationId,
        options.schedulerAuthority.fencingToken,
        static_cast<int>(child.pid),
        child.checkpointEvalId,
        child.experimentId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        terminal, "finalize_reaped_exact_worker_attempt");
    pqxx::result lifecycleCleared;
    if (child.checkpointEvalId)
    {
        lifecycleCleared = transaction.exec_params(
            "UPDATE experiment_checkpoint_eval SET "
            "active_scheduler_worker_attempt_id=NULL "
            "WHERE checkpoint_eval_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2 "
            "RETURNING checkpoint_eval_id;",
            *child.checkpointEvalId,
            *child.workerAttemptId);
    }
    else
    {
        lifecycleCleared = transaction.exec_params(
            "UPDATE experiment SET "
            "active_scheduler_worker_attempt_id=NULL "
            "WHERE experiment_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2 "
            "RETURNING experiment_id;",
            child.experimentId,
            *child.workerAttemptId);
    }
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        lifecycleCleared,
        "clear_reaped_exact_worker_attempt_binding");
}

bool ObserveTerminalCheckpointStopAttempt(
    pqxx::work& transaction,
    const SchedulerOwnedChild& child,
    const SchedulerOptions& options,
    const ObservedChildStatus& observed)
{
    if (!child.workerAttemptId ||
        child.checkpointEvalId ||
        child.phase != "train" ||
        (observed.kind != ChildStatusKind::Exited &&
         observed.kind != ChildStatusKind::Signaled))
    {
        return false;
    }

    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = *child.workerAttemptId;
    expected.experimentId = child.experimentId;
    expected.workerKind = "experiment";
    expected.lifecyclePhase = "train";
    expected.capacityClass = "train";
    expected.schedulerInvocationId =
        options.schedulerAuthority.schedulerInvocationId;
    expected.schedulerFencingToken =
        options.schedulerAuthority.fencingToken;
    expected.requireCompleteProcessIdentity = true;
    const auto terminal =
        EA::SchedulerOwnership::LockAndVerifyExactTerminalAttempt(
            transaction,
            expected,
            "completed",
            "checkpoint_stop_completed",
            true);
    if (!terminal ||
        terminal->workerPid !=
            std::optional<int>{static_cast<int>(child.pid)})
    {
        return false;
    }

    const int exitCode =
        observed.kind == ChildStatusKind::Exited
            ? observed.exitCode
            : 128 + observed.signalNumber;
    const std::optional<int> signalNumber =
        observed.kind == ChildStatusKind::Signaled
            ? std::optional<int>{observed.signalNumber}
            : std::nullopt;
    const pqxx::result updated = transaction.exec_params(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "last_observed_at=clock_timestamp(),"
        "observed_by_scheduler_invocation_id=$1,"
        "exit_code=COALESCE(exit_code,$2),"
        "signal_number=COALESCE(signal_number,$3) "
        "WHERE a.worker_attempt_id=$4 "
        "AND a.scheduler_invocation_id=$1 "
        "AND a.scheduler_fencing_token=$5 "
        "AND a.experiment_id=$6 "
        "AND a.checkpoint_eval_id IS NULL "
        "AND a.worker_kind='experiment' "
        "AND a.lifecycle_phase='train' "
        "AND a.capacity_class='train' "
        "AND a.worker_pid=$7 "
        "AND a.lifecycle_state='completed' "
        "AND a.reconciliation_result='checkpoint_stop_completed' "
        "AND (a.exit_code IS NULL OR a.exit_code=$2) "
        "AND (a.signal_number IS NULL "
        "OR a.signal_number IS NOT DISTINCT FROM $3) "
        "RETURNING a.worker_attempt_id;",
        options.schedulerAuthority.schedulerInvocationId,
        exitCode,
        signalNumber,
        *child.workerAttemptId,
        options.schedulerAuthority.fencingToken,
        child.experimentId,
        static_cast<int>(child.pid));
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        updated,
        "observe_terminal_checkpoint_stop_attempt_exit");
    std::cout << "SCHEDULER_CHECKPOINT_STOP_EXIT_OBSERVED"
              << ",experiment_id=" << child.experimentId
              << ",worker_attempt_id=" << *child.workerAttemptId
              << ",pid=" << child.pid
              << ",exit_code=" << exitCode
              << ",lifecycle_rows_affected=0"
              << std::endl;
    return true;
}

void InjectStaleReaperReplacementForTest(
    pqxx::work& transaction,
    const SchedulerOwnedChild& child,
    const SchedulerOptions& options)
{
    constexpr const char* boundary =
        "stale_reaper_before_exact_verification";
    if (!SchedulerAuthorityTestFailpointEnabled(boundary) ||
        !child.workerAttemptId)
    {
        return;
    }

    const char* replacementPidText = std::getenv(
        "EA_SCHEDULER_OWNERSHIP_TEST_REPLACEMENT_PID");
    if (!replacementPidText || !*replacementPidText)
        throw std::runtime_error(
            "stale_reaper_test_replacement_pid_missing");
    const int replacementPid = std::stoi(replacementPidText);
    std::unique_ptr<
        EA::GlobalExperimentControl::ProcessOperations> processes =
        EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const auto replacementProcess =
        processes->Observe(replacementPid);
    const std::string replacementExecutable =
        CanonicalizeObservedExecutable(
            replacementProcess.executable);
    if (!replacementProcess.inspectionSucceeded ||
        !replacementProcess.exists ||
        replacementProcess.pid != replacementPid ||
        replacementProcess.processGroupId <= 0 ||
        replacementProcess.processStartIdentity.empty() ||
        replacementExecutable.empty() ||
        replacementProcess.commandLine.empty())
    {
        throw std::runtime_error(
            "stale_reaper_test_replacement_identity_incomplete");
    }

    RequireAndRefreshSchedulerAuthority(transaction, options);
    pqxx::result retired = transaction.exec_params(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state='abandoned',"
        "completed_at=clock_timestamp(),"
        "reconciliation_result='test_stale_reaper_replaced',"
        "diagnostic=$1 "
        "WHERE a.worker_attempt_id=$2 "
        "AND a.scheduler_invocation_id=$3 "
        "AND a.scheduler_fencing_token=$4 "
        "AND a.lifecycle_state IN ('spawned','running','observed') "
        "AND EXISTS ("
        " SELECT 1 FROM experiment e "
        " WHERE $5::bigint IS NULL "
        " AND e.experiment_id=$6 "
        " AND e.active_scheduler_worker_attempt_id="
        "a.worker_attempt_id "
        " UNION ALL "
        " SELECT 1 FROM experiment_checkpoint_eval ce "
        " WHERE $5::bigint IS NOT NULL "
        " AND ce.checkpoint_eval_id=$5 "
        " AND ce.active_scheduler_worker_attempt_id="
        "a.worker_attempt_id"
        ") RETURNING a.worker_attempt_id;",
        boundary,
        *child.workerAttemptId,
        options.schedulerAuthority.schedulerInvocationId,
        options.schedulerAuthority.fencingToken,
        child.checkpointEvalId,
        child.experimentId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        retired, "test_retire_stale_reaper_attempt");

    pqxx::result replacement = transaction.exec_params(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,scheduler_invocation_id,"
        "scheduler_fencing_token,experiment_id,checkpoint_eval_id,"
        "worker_kind,lifecycle_phase,capacity_class,"
        "ownership_origin,lifecycle_state,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "canonical_executable_path,command_line,command_identity,"
        "reserved_at,spawned_at,last_observed_at,diagnostic"
        ") SELECT "
        "'test-stale-reaper-replacement-' || "
        "worker_attempt_id::text,"
        "$1,$2,experiment_id,checkpoint_eval_id,worker_kind,"
        "lifecycle_phase,capacity_class,ownership_origin,'spawned',"
        "$4,$5,$6,$7,$8,command_identity,clock_timestamp(),"
        "clock_timestamp(),clock_timestamp(),$3 "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE worker_attempt_id=$9 "
        "RETURNING worker_attempt_id;",
        options.schedulerAuthority.schedulerInvocationId,
        options.schedulerAuthority.fencingToken,
        boundary,
        replacementPid,
        replacementProcess.processGroupId,
        replacementProcess.processStartIdentity,
        replacementExecutable,
        replacementProcess.commandLine,
        *child.workerAttemptId);
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        replacement, "test_create_stale_reaper_replacement");
    const long long replacementAttemptId =
        replacement[0][0].as<long long>();

    pqxx::result rebound;
    if (child.checkpointEvalId)
    {
        rebound = transaction.exec_params(
            "UPDATE experiment_checkpoint_eval SET "
            "active_scheduler_worker_attempt_id=$1 "
            "WHERE checkpoint_eval_id=$2 "
            "AND active_scheduler_worker_attempt_id=$3 "
            "RETURNING checkpoint_eval_id;",
            replacementAttemptId,
            *child.checkpointEvalId,
            *child.workerAttemptId);
    }
    else
    {
        rebound = transaction.exec_params(
            "UPDATE experiment SET "
            "active_scheduler_worker_attempt_id=$1 "
            "WHERE experiment_id=$2 "
            "AND active_scheduler_worker_attempt_id=$3 "
            "RETURNING experiment_id;",
            replacementAttemptId,
            child.experimentId,
            *child.workerAttemptId);
    }
    EA::SchedulerOwnership::RequireAffectedExactlyOne(
        rebound, "test_bind_stale_reaper_replacement");
}

bool VerifyExactActiveChildAttempt(
    pqxx::work& transaction,
    const SchedulerOwnedChild& child,
    const SchedulerOptions& options)
{
    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = *child.workerAttemptId;
    expected.experimentId = child.experimentId;
    expected.checkpointEvalId = child.checkpointEvalId;
    expected.workerKind = child.checkpointEvalId
        ? "checkpoint_infer"
        : "experiment";
    expected.lifecyclePhase = child.checkpointEvalId
        ? "infer"
        : child.phase;
    expected.capacityClass = child.checkpointEvalId
        ? "infer"
        : child.phase;
    expected.schedulerInvocationId =
        options.schedulerAuthority.schedulerInvocationId;
    expected.schedulerFencingToken =
        options.schedulerAuthority.fencingToken;
    expected.requireCompleteProcessIdentity = true;
    expected.allowTerminalLifecycle = true;
    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
    return exact &&
           exact->workerPid ==
               std::optional<int>{static_cast<int>(child.pid)};
}

void ReapSchedulerOwnedChildren(
    pqxx::work& w,
    const SchedulerOptions& options)
{
    EA::SchedulerCore::SchedulerChildCompletionOperations operations;
    operations.observeChild = [](pid_t pid) {
        return EA::SchedulerCore::NativeWorkerProcessController()
            .observeChild(pid);
    };
    operations.observeTerminalCheckpointStop =
        [&](const SchedulerOwnedChild& child,
            const ObservedChildStatus& observed) {
            return ObserveTerminalCheckpointStopAttempt(
                w, child, options, observed);
        };
    operations.injectStaleReaperReplacementForTest =
        [&](const SchedulerOwnedChild& child) {
            InjectStaleReaperReplacementForTest(w, child, options);
        };
    operations.verifyExactActiveAttempt =
        [&](const SchedulerOwnedChild& child) {
            return VerifyExactActiveChildAttempt(w, child, options);
        };
    operations.staleReaperFailpointEnabled = [] {
        return SchedulerAuthorityTestFailpointEnabled(
            "stale_reaper_before_exact_verification");
    };
    operations.requestSchedulerStop = [] {
        gSchedulerStopRequested = 1;
    };
    operations.persistUnexpectedStatus =
        [&](const SchedulerOwnedChild& child, int rawStatus) {
            PersistUnexpectedChildStatus(w, child, rawStatus);
        };
    operations.persistCheckpointCompletion =
        [&](const SchedulerOwnedChild& child,
            const SchedulerChildCompletionEvidence& completion) {
            PersistObservedCheckpointChild(w, child, completion);
        };
    operations.persistExperimentCompletion =
        [&](const SchedulerOwnedChild& child,
            const SchedulerChildCompletionEvidence& completion) {
            PersistObservedExperimentChild(w, child, completion);
        };
    operations.finalizeWorkerAttempt =
        [&](const SchedulerOwnedChild& child,
            int exitCode,
            const std::optional<int>& signalNumber,
            std::string_view diagnostic) {
            FinalizeObservedWorkerAttempt(
                w,
                child,
                options,
                exitCode,
                signalNumber,
                std::string{diagnostic});
        };
    EA::SchedulerCore::SchedulerChildCompletionService service{
        std::move(operations), std::cout};
    service.reap(SchedulerRuntime(options).ownedChildren);
}

void BeginSchedulerPollLogging(SchedulerEventLogState* logState)
{
    if (logState == nullptr)
        return;
    logState->currentSkipKeys.clear();
    logState->currentWorkerSelectionKeys.clear();
    logState->currentRunningPresentKeys.clear();
}

void FinishSchedulerPollLogging(SchedulerEventLogState* logState)
{
    if (logState == nullptr)
        return;
    logState->previousSkipKeys = logState->currentSkipKeys;
    logState->previousWorkerSelectionKeys =
        logState->currentWorkerSelectionKeys;
    logState->previousRunningPresentKeys = logState->currentRunningPresentKeys;
}

int CountRows(pqxx::work& w, const std::string& sql)
{
    return w.exec(sql).one_row()[0].as<int>();
}

void LogCheckpointEvalInferenceResultFound(const CheckpointEvalRow& eval,
                                           long long inferenceResultId)
{
    std::cout << "CHECKPOINT_EVAL_INFER_RESULT_FOUND"
              << ",checkpoint_eval_id=" << eval.checkpointEvalId
              << ",parent_experiment_id=" << eval.experiment.experimentId
              << ",checkpoint_model_id=" << eval.checkpointModelId
              << ",checkpoint_epoch=" << eval.checkpointEpoch
              << ",model_id=" << eval.checkpointModelId
              << ",inference_scope=checkpoint"
              << ",status=completed"
              << ",inference_result_id=" << inferenceResultId
              << std::endl;
}

void AdvanceCheckpointEvalToAnalyze(pqxx::work& w,
                                    const CheckpointEvalRow& eval,
                                    long long inferenceResultId,
                                    bool hasInferCompletedAt,
                                    const std::optional<long long>&
                                        workerAttemptId)
{
    LogCheckpointEvalInferenceResultFound(eval, inferenceResultId);
    std::ostringstream sql;
    sql << "UPDATE experiment_checkpoint_eval "
        << "SET status = 'pending', phase = 'analyze', worker_pid = NULL, "
        << "completed_at = NULL, error_message = NULL, updated_at = now()";
    if (hasInferCompletedAt)
        sql << ", infer_completed_at = COALESCE(infer_completed_at, now())";
    sql << " WHERE checkpoint_eval_id = $1 "
        << "AND ($2::bigint IS NULL OR "
        << "active_scheduler_worker_attempt_id=$2) "
        << "RETURNING checkpoint_eval_id;";
    pqxx::result updated = w.exec_params(
        sql.str(), eval.checkpointEvalId, workerAttemptId);
    if (workerAttemptId)
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated, "advance_checkpoint_exact_attempt_to_analyze");

    std::cout << "CHECKPOINT_EVAL_ADVANCED_TO_ANALYZE"
              << ",checkpoint_eval_id=" << eval.checkpointEvalId
              << ",parent_experiment_id=" << eval.experiment.experimentId
              << ",checkpoint_model_id=" << eval.checkpointModelId
              << ",checkpoint_epoch=" << eval.checkpointEpoch
              << ",model_id=" << eval.checkpointModelId
              << ",inference_scope=checkpoint"
              << ",status=pending"
              << ",inference_result_id=" << inferenceResultId
              << std::endl;
}

struct ClaimedCheckpointAnalysis
{
    CheckpointEvalRow evaluation;
    long long workerAttemptId = -1;
    std::string launchAttemptIdentity;
};

struct CheckpointAnalysisExecution
{
    bool success = false;
    std::string error;
    long long inferenceResultId = -1;
    ParsedMetrics metrics;
    std::optional<double> leaderScore;
};

std::optional<ClaimedCheckpointAnalysis>
ClaimCheckpointAnalysis(
    const SchedulerOptions& options)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    if (!CheckpointEvalTableExists(transaction) ||
        !SchedulerLaunchAllowed(
            transaction, "checkpoint_analyze"))
    {
        transaction.commit();
        return std::nullopt;
    }
    if (!services.admission.hasCapacity(
            "analyze", options.maxAnalyzeProcs))
    {
        transaction.commit();
        return std::nullopt;
    }
    const int pendingFinalAnalyze = CountRows(
        transaction,
        "SELECT count(*) FROM experiment "
        "WHERE status='pending' AND phase='analyze';");
    if (pendingFinalAnalyze > 0)
    {
        transaction.commit();
        return std::nullopt;
    }

    std::vector<CheckpointEvalRow> pending =
        LoadCheckpointEvalRows(
            transaction, "pending", "analyze");
    if (pending.empty())
    {
        transaction.commit();
        return std::nullopt;
    }
    ClaimedCheckpointAnalysis claim;
    claim.evaluation = pending.front();
    const std::string commandIdentity =
        "checkpoint_analyze:" +
        std::to_string(claim.evaluation.checkpointEvalId);
    claim.launchAttemptIdentity =
        options.schedulerAuthority.schedulerInvocationId + ":worker:" +
        EA::SchedulerCore::GenerateSchedulerIdentityNonce() + ":" +
        commandIdentity;
    const auto reserved = services.repository.reserveCheckpointAnalysisAttempt({
        claim.launchAttemptIdentity,
        options.schedulerAuthority.schedulerInvocationId,
        options.schedulerAuthority.fencingToken,
        claim.evaluation.experiment.experimentId,
        claim.evaluation.checkpointEvalId,
        options.schedulerExecutablePath,
        options.invocationCommandLine,
        commandIdentity,
        ColumnExists(transaction, "experiment_checkpoint_eval",
                     "analyze_started_at")});
    if (reserved.status ==
        EA::SchedulerCore::WorkerAttemptReservationStatus::ReservationInsertFailed)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "reserve_checkpoint_analysis_attempt:affected_rows=0");
    }
    if (reserved.status !=
            EA::SchedulerCore::WorkerAttemptReservationStatus::Reserved ||
        !reserved.attempt)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "claim_checkpoint_analysis_lifecycle:affected_rows=0");
    }
    claim.workerAttemptId = reserved.attempt->workerAttemptId;
    RequireAndRefreshSchedulerAuthority(
        transaction, options);
    transaction.commit();
    return claim;
}

CheckpointAnalysisExecution ExecuteCheckpointAnalysisWork(
    const ClaimedCheckpointAnalysis& claim)
{
    CheckpointAnalysisExecution result;
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadOnly(transaction);
        std::vector<CheckpointEvalRow> running =
            LoadCheckpointEvalRows(
                transaction, "running", "analyze");
        const auto current = std::find_if(
            running.begin(),
            running.end(),
            [&](const CheckpointEvalRow& candidate) {
                return candidate.checkpointEvalId ==
                       claim.evaluation.checkpointEvalId;
            });
        if (current == running.end())
            throw std::runtime_error(
                "checkpoint_analysis_claim_no_longer_active");
        const std::optional<long long> inferenceResultId =
            FindCompletedCheckpointInferenceResultId(
                transaction, *current);
        if (!inferenceResultId)
            throw std::runtime_error(
                "missing_completed_checkpoint_inference");
        result.inferenceResultId = *inferenceResultId;
        result.metrics.modelId =
            current->checkpointModelId;
        ExperimentRow analysisExperiment =
            current->experiment;
        ApplyPersistedSymbolToAnalysisExperiment(
            transaction,
            analysisExperiment,
            result.metrics);
        if (!ApplyStructuredCheckpointInferenceMetrics(
                transaction, *current, result.metrics))
        {
            throw std::runtime_error(
                "checkpoint_inference_result_disappeared");
        }
        result.leaderScore =
            ComputeLeaderScore(result.metrics);
        transaction.commit();
        result.success = true;
    }
    catch (const std::exception& error)
    {
        result.error = error.what();
    }
    return result;
}

bool FinalizeCheckpointAnalysis(
    const SchedulerOptions& options,
    const ClaimedCheckpointAnalysis& claim,
    const CheckpointAnalysisExecution& work)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    EA::SchedulerCore::PostgresSchedulerRepository repository{transaction};
    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = claim.workerAttemptId;
    expected.experimentId =
        claim.evaluation.experiment.experimentId;
    expected.checkpointEvalId =
        claim.evaluation.checkpointEvalId;
    expected.workerKind = "checkpoint_analyze";
    expected.lifecyclePhase = "analyze";
    expected.capacityClass = "analyze";
    expected.schedulerInvocationId =
        options.schedulerAuthority.schedulerInvocationId;
    expected.schedulerFencingToken =
        options.schedulerAuthority.fencingToken;
    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
    if (!exact)
        throw SchedulerAuthorityLost(
            "checkpoint_analysis_exact_attempt_replaced");

    const std::optional<long long> currentInferenceResult =
        FindCompletedCheckpointInferenceResultId(
            transaction, claim.evaluation);
    const auto plan =
        EA::SchedulerCore::PlanCheckpointAnalysisFinalization(
            {work.success, work.error, work.inferenceResultId},
            currentInferenceResult);
    const bool success = plan.complete;
    if (success)
    {
        AnalysisScopeOptions scope;
        scope.scope = "checkpoint";
        scope.checkpointEvalId =
            claim.evaluation.checkpointEvalId;
        scope.checkpointEpoch =
            claim.evaluation.checkpointEpoch;
        scope.parentExperimentId =
            claim.evaluation.experiment.experimentId;
        UpsertAnalysisResult(
            transaction,
            claim.evaluation.experiment,
            work.metrics,
            work.leaderScore,
            scope);
        const std::optional<long long> analysisId =
            FindCheckpointAnalysisResultId(
                transaction,
                claim.evaluation.checkpointEvalId);
        if (!repository.persistCheckpointAnalysisCompletion({
                claim.evaluation.checkpointEvalId,
                claim.workerAttemptId,
                analysisId,
                ColumnExists(transaction, "experiment_checkpoint_eval",
                             "analyze_completed_at")}))
        {
            throw std::runtime_error(
                "exact_attempt_predicate_rejected:"
                "complete_checkpoint_analysis_exact_attempt:affected_rows=0");
        }
        (void)EvaluateCheckpointPolicyAfterAnalysis(
            transaction, claim.evaluation);
    }

    const auto persisted = repository.persistCheckpointAnalysisTerminalState({
        claim.workerAttemptId,
        options.schedulerAuthority.schedulerInvocationId,
        options.schedulerAuthority.fencingToken,
        claim.evaluation.checkpointEvalId,
        plan.complete,
        plan.attemptLifecycleState,
        plan.reconciliationResult,
        plan.diagnostic});
    if (persisted == EA::SchedulerCore::CheckpointAnalysisPersistenceResult::
                         AttemptPreconditionRejected)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "terminalize_checkpoint_analysis_exact_attempt:affected_rows=0");
    }
    if (persisted == EA::SchedulerCore::CheckpointAnalysisPersistenceResult::
                         LifecyclePreconditionRejected)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "clear_checkpoint_analysis_exact_attempt_binding:affected_rows=0");
    }
    RequireAndRefreshSchedulerAuthority(
        transaction, options);
    transaction.commit();
    return success;
}

int RunCheckpointEvalAnalyzeJobs(
    const SchedulerOptions& options)
{
    using Claim = EA::SchedulerCore::CheckpointAnalysisClaim;
    using Work = EA::SchedulerCore::CheckpointAnalysisWorkResult;
    std::optional<ClaimedCheckpointAnalysis> claimed;
    std::optional<CheckpointAnalysisExecution> execution;
    EA::SchedulerCore::CheckpointAnalysisOperations operations{
        [&]() -> std::optional<Claim> {
            claimed = ClaimCheckpointAnalysis(options);
            if (!claimed)
                return std::nullopt;
            return Claim{
                claimed->evaluation.checkpointEvalId,
                claimed->evaluation.experiment.experimentId,
                claimed->evaluation.checkpointModelId,
                claimed->evaluation.checkpointEpoch,
                claimed->workerAttemptId};
        },
        [&](const Claim& claim) {
            if (SchedulerAuthorityTestFailpointEnabled(
                    "checkpoint_analysis_crash_after_claim"))
            {
                std::cout << "CHECKPOINT_ANALYSIS_TEST_CRASH_AFTER_CLAIM"
                          << ",checkpoint_eval_id=" << claim.checkpointEvalId
                          << ",worker_attempt_id=" << claim.workerAttemptId << std::endl;
                std::cout.flush();
                std::cerr.flush();
                ::_exit(86);
            }
            PersistSchedulerAuthorityLossForTest(
                options, "checkpoint_analysis_lease_loss_during_work");
        },
        [&](const Claim&) -> Work {
            execution = ExecuteCheckpointAnalysisWork(*claimed);
            return {execution->success, execution->error,
                    execution->inferenceResultId};
        },
        [&](const Claim&, const Work&) {
            PersistSchedulerAuthorityLossForTest(
                options, "checkpoint_analysis_lease_loss_before_finalize");
        },
        [&](const Claim&, const Work&) {
            return FinalizeCheckpointAnalysis(options, *claimed, *execution);
        },
        [&] { TryGenerateExperimentReports(options); }};
    EA::SchedulerCore::CheckpointAnalysisOrchestrationService service{
        std::move(operations), std::cout, std::cerr,
        static_cast<long long>(::getpid())};
    return service.runOne();
}

int GlobalCapacityUsed(
    const SchedulerOptions& options,
    const std::string& capacityClass)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    RequireAndRefreshSchedulerAuthority(transaction, options);
    SchedulerServiceComposition services{transaction};
    const int used = services.admission.capacityUsed(capacityClass);
    transaction.commit();
    return used;
}

std::string_view FinalExperimentPhaseName(
    EA::SchedulerCore::FinalExperimentPhase phase)
{
    using EA::SchedulerCore::FinalExperimentPhase;
    switch (phase)
    {
    case FinalExperimentPhase::Train:
        return "train";
    case FinalExperimentPhase::Infer:
        return "infer";
    case FinalExperimentPhase::Analyze:
        return "analyze";
    }
    throw std::logic_error("unknown final experiment phase");
}

EA::SchedulerCore::FinalExperimentStoppedAdmission
ToFinalExperimentStoppedAdmission(StoppedWorkerAdmissionResult result)
{
    using EA::SchedulerCore::FinalExperimentStoppedAdmission;
    switch (result)
    {
    case StoppedWorkerAdmissionResult::NotApplicable:
        return FinalExperimentStoppedAdmission::NotApplicable;
    case StoppedWorkerAdmissionResult::Admitted:
        return FinalExperimentStoppedAdmission::Admitted;
    case StoppedWorkerAdmissionResult::MissingProcessFallbackReady:
        return FinalExperimentStoppedAdmission::MissingProcessFallbackReady;
    case StoppedWorkerAdmissionResult::DeferredNoCapacity:
        return FinalExperimentStoppedAdmission::DeferredNoCapacity;
    case StoppedWorkerAdmissionResult::DeferredUnsafe:
        return FinalExperimentStoppedAdmission::DeferredUnsafe;
    }
    throw std::logic_error("unknown stopped worker admission result");
}

int RunFinalExperimentPhase(
    const SchedulerOptions& options,
    SchedulerEventLogState* logState,
    EA::SchedulerCore::FinalExperimentPhase requestedPhase,
    bool cancellationOnly = false)
{
    using EA::SchedulerCore::FinalExperimentDispatchBatch;
    using EA::SchedulerCore::FinalExperimentDispatchCandidate;
    using EA::SchedulerCore::FinalExperimentDispatchConfiguration;
    using EA::SchedulerCore::FinalExperimentDispatchOperations;
    using EA::SchedulerCore::FinalExperimentDispatchService;
    using EA::SchedulerCore::FinalExperimentDispatchStats;
    using EA::SchedulerCore::FinalExperimentEligibility;
    using EA::SchedulerCore::FinalExperimentPhase;

    std::vector<ExperimentRow> jobs;
    std::optional<ReservedWorkerAttempt> reservedAttempt;
    std::vector<std::string> preparedCommand;

    FinalExperimentDispatchOperations operations;
    operations.load =
        [&](FinalExperimentPhase phase, bool cancellation) {
            const std::string phaseName{FinalExperimentPhaseName(phase)};
            pqxx::connection connection{LstmDbConnectionString()};
            pqxx::work transaction{connection};
            SetTransactionReadWrite(transaction);
            RequireAndRefreshSchedulerAuthority(transaction, options);
            SchedulerServiceComposition services{transaction};

            FinalExperimentDispatchBatch batch;
            if (!SchedulerLaunchAllowed(
                    transaction,
                    phaseName,
                    false,
                    cancellation))
            {
                transaction.commit();
                return batch;
            }

            jobs = LoadPendingExperiments(
                services.admission, phaseName, cancellation);
            const int used = services.admission.capacityUsed(phaseName);
            transaction.commit();

            const int maximum =
                phase == FinalExperimentPhase::Train
                    ? options.maxTrainProcs
                    : (phase == FinalExperimentPhase::Infer
                           ? options.maxInferProcs
                           : options.maxAnalyzeProcs);
            batch.launchAllowed = true;
            batch.freeSlots = AvailableWorkerProcessSlots(maximum, used);
            batch.candidates.reserve(jobs.size());
            for (std::size_t index = 0; index < jobs.size(); ++index)
            {
                const ExperimentRow& job = jobs[index];
                batch.candidates.push_back({
                    index,
                    job.experimentId,
                    job.activeWorkerAttemptId.has_value(),
                    job.lastModelId.has_value()});
            }
            return batch;
        };
    operations.ensureLogDirectory =
        [&] { EnsureLogDir(options.schedulerLogDir); };
    operations.semanticPreflight =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase) {
            return SemanticWorkerPreflight(
                options,
                candidate.experimentId,
                jobs.at(candidate.sourceIndex).lastModelId,
                std::string{FinalExperimentPhaseName(phase)},
                logState,
                options.schedulerVerbose);
        };
    operations.preemptOneLowerPriorityWorker =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase,
            int maximumCapacity) {
            (void)PreemptOneLowerPriorityWorker(
                options,
                jobs.at(candidate.sourceIndex),
                std::string{FinalExperimentPhaseName(phase)},
                maximumCapacity);
        };
    operations.admitStoppedWorker =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase,
            int maximumCapacity) {
            return ToFinalExperimentStoppedAdmission(
                AdmitStoppedExperimentWorker(
                    options,
                    jobs.at(candidate.sourceIndex),
                    std::string{FinalExperimentPhaseName(phase)},
                    maximumCapacity));
        };
    operations.evaluateEligibility =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase) {
            const ExperimentRow& job = jobs.at(candidate.sourceIndex);
            if (phase == FinalExperimentPhase::Train)
            {
                const std::optional<long long> resumeFrom =
                    job.lastModelId ? job.lastModelId : job.resumeModelId;
                if (!resumeFrom)
                    return FinalExperimentEligibility::Eligible;

                pqxx::connection connection{LstmDbConnectionString()};
                pqxx::work transaction{connection};
                SetTransactionReadWrite(transaction);
                RequireAndRefreshSchedulerAuthority(transaction, options);
                if (!ModelExists(transaction, *resumeFrom))
                {
                    transaction.exec_params(
                        "UPDATE experiment SET status='failed',"
                        "completed_at=clock_timestamp(),exit_code=-1,"
                        "error_message='train_model_not_found',"
                        "updated_at=clock_timestamp() "
                        "WHERE experiment_id=$1 AND status='pending' "
                        "AND phase='train' "
                        "AND active_scheduler_worker_attempt_id IS NULL;",
                        job.experimentId);
                    transaction.commit();
                    return FinalExperimentEligibility::Failed;
                }
                transaction.commit();
                return FinalExperimentEligibility::Eligible;
            }

            if (phase == FinalExperimentPhase::Infer)
            {
                pqxx::connection connection{LstmDbConnectionString()};
                pqxx::work transaction{connection};
                SetTransactionReadWrite(transaction);
                RequireAndRefreshSchedulerAuthority(transaction, options);
                if (!job.lastModelId ||
                    !ModelExists(transaction, *job.lastModelId))
                {
                    if (!options.dryRun)
                    {
                        transaction.exec_params(
                            "UPDATE experiment SET status='failed',"
                            "completed_at=clock_timestamp(),exit_code=-1,"
                            "error_message=$1,updated_at=clock_timestamp() "
                            "WHERE experiment_id=$2 AND status='pending' "
                            "AND phase='infer' "
                            "AND active_scheduler_worker_attempt_id IS NULL;",
                            job.lastModelId
                                ? "infer_model_not_found"
                                : "infer_missing_last_model_id",
                            job.experimentId);
                    }
                    transaction.commit();
                    return FinalExperimentEligibility::Skipped;
                }

                const bool forcedFinalInferenceRerun =
                    OperatorForcedFinalInferenceRerunRequested(
                        transaction, job.experimentId);
                if (!forcedFinalInferenceRerun &&
                    HasCompletedInferenceResult(transaction, job))
                {
                    if (!options.dryRun)
                    {
                        std::cout
                            << "SCHEDULER_SKIP_EXISTING_INFERENCE"
                            << ",experiment_id=" << job.experimentId
                            << ",model_id=" << *job.lastModelId
                            << std::endl;
                        if (HasCompletedAnalysisResult(transaction, job))
                        {
                            MarkExperimentDone(transaction, job, "infer");
                        }
                        else
                        {
                            MarkExperimentPendingPhase(
                                transaction, job, "infer", "analyze");
                        }
                    }
                    transaction.commit();
                    return FinalExperimentEligibility::Skipped;
                }
                if (forcedFinalInferenceRerun)
                {
                    std::cout
                        << "SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_DISPATCH"
                        << ",experiment_id=" << job.experimentId
                        << ",model_id=" << *job.lastModelId
                        << std::endl;
                }
                transaction.commit();
            }
            return FinalExperimentEligibility::Eligible;
        };
    operations.emitDryRunCommand =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase) {
            const ExperimentRow& job = jobs.at(candidate.sourceIndex);
            std::vector<std::string> command;
            if (phase == FinalExperimentPhase::Train)
                command = BuildTrainCommand(options, job);
            else if (phase == FinalExperimentPhase::Infer)
            {
                pqxx::connection connection{LstmDbConnectionString()};
                pqxx::read_transaction transaction{connection};
                const auto selection = LoadInferenceWorkerSelection(
                    transaction, options, job.experimentId);
                if (!selection.selected)
                    throw std::runtime_error(selection.diagnostic);
                command = BuildInferCommand(
                    options, job, selection.canonicalExecutablePath);
            }
            else
                command = BuildAnalyzeCommand(options, job);
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=" << FinalExperimentPhaseName(phase)
                      << ",dry_run=1,argv="
                      << CommandForDisplay(command)
                      << std::endl;
        };
    operations.reserveWorkerAttempt =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase,
            int maximumCapacity,
            bool cancellation) -> std::optional<long long> {
            const ExperimentRow& job = jobs.at(candidate.sourceIndex);
            const std::string phaseName{FinalExperimentPhaseName(phase)};
            const std::string logPath = LogPathFor(
                options,
                job,
                phase == FinalExperimentPhase::Analyze
                    ? "analysis"
                    : phaseName);
            const auto attempt = ReserveExperimentWorkerAttempt(
                options,
                job,
                phaseName,
                logPath,
                maximumCapacity,
                cancellation);
            if (!attempt)
                return std::nullopt;
            const long long attemptId = attempt->workerAttemptId;
            reservedAttempt = std::move(*attempt);
            return attemptId;
        };
    operations.capacityUsed =
        [&](FinalExperimentPhase phase) {
            return GlobalCapacityUsed(
                options, std::string{FinalExperimentPhaseName(phase)});
        };
    operations.prepareReservedLaunch =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase,
            long long attemptId) {
            const ExperimentRow& job = jobs.at(candidate.sourceIndex);
            std::vector<std::string> command;
            if (phase == FinalExperimentPhase::Train)
            {
                command = BuildTrainCommand(options, job);
                std::cout << "EXPERIMENT_STARTED"
                          << ",experiment_id=" << job.experimentId
                          << ",phase=train,worker_attempt_id="
                          << attemptId
                          << std::endl;
            }
            else if (phase == FinalExperimentPhase::Infer)
            {
                command = BuildInferCommand(
                    options,
                    job,
                    reservedAttempt->canonicalExecutablePath);
            }
            else
            {
                command = BuildAnalyzeCommand(options, job);
            }
            preparedCommand = std::move(command);
        };
    operations.launchPreparedWorker =
        [&](const FinalExperimentDispatchCandidate& candidate,
            FinalExperimentPhase phase,
            long long) {
            const ExperimentRow& job = jobs.at(candidate.sourceIndex);
            PrintSchedulerExec(preparedCommand);
            std::optional<long long> expectedModelId;
            if (phase == FinalExperimentPhase::Train)
            {
                expectedModelId =
                    job.lastModelId ? job.lastModelId : job.resumeModelId;
            }
            else
            {
                expectedModelId = job.lastModelId;
            }
            (void)LaunchReservedChildProcess(
                options,
                *reservedAttempt,
                std::move(preparedCommand),
                expectedModelId);
        };
    operations.logSkip =
        [&](FinalExperimentPhase phase,
            long long experimentId,
            std::string_view reason) {
            LogSkip(
                std::string{FinalExperimentPhaseName(phase)},
                experimentId,
                std::string{reason},
                logState,
                options.schedulerVerbose);
        };
    operations.printStats =
        [&](const FinalExperimentDispatchStats& serviceStats) {
            PhaseSchedulingStats stats;
            stats.phase =
                std::string{FinalExperimentPhaseName(serviceStats.phase)};
            stats.examined = serviceStats.examined;
            stats.skipped = serviceStats.skipped;
            stats.launched = serviceStats.launched;
            stats.freeSlots = serviceStats.freeSlots;
            PrintPhaseSchedulingStats(
                stats, logState, options.schedulerVerbose);
        };

    FinalExperimentDispatchService service{
        FinalExperimentDispatchConfiguration{
            options.dryRun,
            options.maxTrainProcs,
            options.maxInferProcs,
            options.maxAnalyzeProcs},
        std::move(operations),
        std::cerr};
    switch (requestedPhase)
    {
    case FinalExperimentPhase::Train:
        return service.runTrain(cancellationOnly);
    case FinalExperimentPhase::Infer:
        return service.runInference();
    case FinalExperimentPhase::Analyze:
        return service.runAnalysis();
    }
    throw std::logic_error("unknown final experiment phase");
}

int RunTrainJobs(
    const SchedulerOptions& options,
    const QueueSnapshot&,
    SchedulerEventLogState* logState,
    bool cancellationOnly = false)
{
    return RunFinalExperimentPhase(
        options,
        logState,
        EA::SchedulerCore::FinalExperimentPhase::Train,
        cancellationOnly);
}

int RunInferJobs(
    const SchedulerOptions& options,
    const QueueSnapshot&,
    SchedulerEventLogState* logState)
{
    return RunFinalExperimentPhase(
        options,
        logState,
        EA::SchedulerCore::FinalExperimentPhase::Infer);
}

int RunAnalyzeJobs(
    const SchedulerOptions& options,
    const QueueSnapshot&,
    SchedulerEventLogState* logState)
{
    return RunFinalExperimentPhase(
        options,
        logState,
        EA::SchedulerCore::FinalExperimentPhase::Analyze);
}

int RunCheckpointEvalInferJobs(
    const SchedulerOptions& options,
    SchedulerEventLogState* logState)
{
    std::vector<CheckpointEvalRow> jobs;
    std::optional<long long> activeRequestId;
    bool cancellationInference = false;
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        RequireAndRefreshSchedulerAuthority(transaction, options);
        const auto control = LoadLockedGlobalControl(transaction);
        if (!control)
            return 1;
        const bool normal =
            EA::GlobalExperimentControl::NormalSchedulingAllowed(
                *control);
        cancellationInference =
            EA::GlobalExperimentControl::CancellationInferenceAllowed(
                *control);
        activeRequestId = control->activeRequestId;
        if (!normal && !cancellationInference)
        {
            transaction.commit();
            return 0;
        }
        if (CountRows(
                transaction,
                "SELECT count(*) FROM experiment "
                "WHERE status='pending' AND phase='infer';") == 0)
        {
            jobs = LoadCheckpointEvalRows(
                transaction, "pending", "infer");
        }
        transaction.commit();
    }

    EnsureLogDir(options.schedulerLogDir);
    int rc = 0;
    for (const CheckpointEvalRow& eval : jobs)
    {
        if (cancellationInference &&
            eval.cancellationRequestId != activeRequestId)
            continue;
        if (!eval.experiment.inferStart ||
            !eval.experiment.inferEnd)
            continue;
        if (!SemanticWorkerPreflight(
                options,
                eval.experiment.experimentId,
                eval.checkpointModelId,
                "infer", logState,
                options.schedulerVerbose))
            continue;
        if (options.dryRun)
            continue;

        const std::string logPath =
            CheckpointEvalLogPathFor(options, eval, "infer");
        const auto attempt = ReserveCheckpointWorkerAttempt(
            options,
            eval,
            logPath,
            options.maxInferProcs);
        if (!attempt)
        {
            if (!SchedulerWorkerCapacityHasSlot(
                    options.maxInferProcs,
                    GlobalCapacityUsed(options, "infer")))
                break;
            continue;
        }
        try
        {
            std::vector<std::string> command =
                BuildCheckpointEvalInferCommand(
                    options, eval, attempt->canonicalExecutablePath);
            PrintSchedulerExec(command);
            (void)LaunchReservedChildProcess(
                options,
                *attempt,
                std::move(command),
                eval.checkpointModelId);
        }
        catch (const std::exception& error)
        {
            std::cerr << "CHECKPOINT_EVAL_FAILED"
                      << ",checkpoint_eval_id="
                      << eval.checkpointEvalId
                      << ",worker_attempt_id="
                      << attempt->workerAttemptId
                      << ",error=" << error.what()
                      << std::endl;
            rc = 1;
        }
    }
    return rc;
}

std::string SchedulerCancellationReconciliationOwner(
    const SchedulerOptions& options)
{
    const int pid = static_cast<int>(::getpid());
    const std::optional<std::string> startIdentity =
        EA::GlobalExperimentControl::ReadProcessStartIdentity(pid);
    return "scheduler:pid:" + std::to_string(pid) +
           ";start:" + startIdentity.value_or("unavailable") +
           ";executable:" + options.schedulerExecutablePath;
}

int RunSchedulerOnce(const SchedulerOptions& options,
                     SchedulerEventLogState* logState)
{
    QueueSnapshot snapshot;
    EA::SchedulerCore::SchedulerCycleOperations operations;
    operations.beginPoll = [logState] { BeginSchedulerPollLogging(logState); };
    operations.prepare = [&] {
        EA::SchedulerCore::SchedulerCyclePreparation preparation;
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        if (!options.dryRun)
            SetTransactionReadWrite(w);
        RequireAndRefreshSchedulerAuthority(w, options);
        SchedulerServiceComposition services{w};
        if (!RequireSchedulerTables(w))
            return EA::SchedulerCore::SchedulerCyclePreparation{
                .result = 1};
        const auto control = LoadLockedGlobalControl(w);
        if (!control)
            return EA::SchedulerCore::SchedulerCyclePreparation{
                .result = 1};
        preparation.normalSchedulingAllowed =
            EA::GlobalExperimentControl::NormalSchedulingAllowed(*control);
        preparation.cancellationInferenceAllowed =
            EA::GlobalExperimentControl::CancellationInferenceAllowed(*control);
        preparation.cancellationCheckpointTrainAllowed =
            EA::GlobalExperimentControl::CancellationCheckpointTrainAllowed(
                *control);
        if (!options.dryRun)
        {
            (void)EA::GlobalExperimentControl::ReconcileActiveCancellation(
                w,
                SchedulerCancellationReconciliationOwner(options),
                true);
            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.schedulerExecutablePath, "scheduler_start");
            ReapSchedulerOwnedChildren(w, options);
            RecoverOrphanedRunningExperiments(
                w, options, logState, options.schedulerVerbose);
            if (preparation.normalSchedulingAllowed)
            {
                EnqueueCheckpointEvalRows(w);
                preparation.result |= FailInvalidSchedulerPhases(w);
            }
        }
        snapshot = LoadQueueSnapshot(services.admission);
        PrintQueueSnapshot(snapshot, logState, options.schedulerVerbose);
        w.commit();
        preparation.ready = true;
        return preparation;
    };
    operations.runTrain = [&](bool cancellationOnly) {
        return RunTrainJobs(options, snapshot, logState, cancellationOnly);
    };
    operations.runFinalInference = [&] { return RunInferJobs(options, snapshot, logState); };
    operations.runFinalAnalysis = [&] { return RunAnalyzeJobs(options, snapshot, logState); };
    operations.runCheckpointInference = [&] {
        return RunCheckpointEvalInferJobs(options, logState);
    };
    operations.runCheckpointAnalysis = [&] { return RunCheckpointEvalAnalyzeJobs(options); };
    operations.finishPoll = [logState] { FinishSchedulerPollLogging(logState); };
    EA::SchedulerCore::SchedulerCycleService service{
        std::move(operations)};
    return service.runOnce();
}

std::string ResolveAnalyzeWorkerExecutablePath(
    const SchedulerOptions& options,
    const EA::SchedulerCore::SchedulerDaemonConfiguration& configuration)
{
    if (configuration.analyzeWorkerExecutablePath.has_value())
        return *configuration.analyzeWorkerExecutablePath;

    // Both scheduler entry products are emitted beside the dedicated worker.
    // This is a role-specific product resolution, independent of the semantic
    // worker registry and its layout selection.
    const std::filesystem::path defaultPath =
        std::filesystem::path{options.schedulerExecutablePath}.parent_path() /
        "lstm-analyze-worker";
    return EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
        defaultPath.string(), "default analyze worker executable");
}

int RunScheduler(
    SchedulerOptions options,
    const EA::SchedulerCore::SchedulerDaemonConfiguration&
        daemonConfiguration)
{
    EA::SchedulerCore::SchedulerRuntimeContext runtimeContext;
    options.runtimeContext = &runtimeContext;
    gSchedulerStopRequested = 0;
    try
    {
        options.analyzeWorkerExecutablePath =
            ResolveAnalyzeWorkerExecutablePath(options, daemonConfiguration);
        options.semanticWorkerRegistry =
            EA::Scheduler::SemanticWorkerRegistry::Load({
                options.semanticWorkerRegistryPath,
                options.legacyLayout6InferWorkerPath});
        options.currentWorkerExecutablePath =
            options.semanticWorkerRegistry->currentWorker()
                .canonicalExecutablePath;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_START_REJECTED"
                  << ",diagnostic=" << error.what()
                  << ",authority_acquired=0,workers_launched=0"
                  << std::endl;
        return 1;
    }
    InstallSchedulerSignalHandlers();
    if (!AcquireSchedulerAuthority(options))
        return 3;
    SchedulerAuthorityReleaseGuard authorityGuard{options};

    int recoveryCount = 0;
    std::string initialGlobalState = "unknown";
    SchedulerEventLogState logState;
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        SetTransactionReadWrite(w);
        RequireAndRefreshSchedulerAuthority(w, options);
        if (!RequireSchedulerTables(w))
            return 1;
        const auto control = LoadLockedGlobalControl(w);
        if (!control)
            return 1;
        initialGlobalState = control->desiredState;
        if (!options.dryRun)
        {
            (void)EA::GlobalExperimentControl::ReconcileActiveCancellation(
                w,
                SchedulerCancellationReconciliationOwner(options),
                true);
            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.schedulerExecutablePath, "scheduler_start");
            BeginSchedulerPollLogging(&logState);
            recoveryCount = RecoverOrphanedRunningExperiments(
                w, options, &logState, options.schedulerVerbose);
            FinishSchedulerPollLogging(&logState);
            if (FailInvalidSchedulerPhases(w) != 0)
            {
                w.commit();
                return 1;
            }
        }
        w.commit();
    }

    std::cout << "SCHEDULER_START"
              << ",dry_run=" << (options.dryRun ? "1" : "0")
              << ",max_train_procs=" << options.maxTrainProcs
              << ",max_infer_procs=" << options.maxInferProcs
              << ",max_analyze_procs=" << options.maxAnalyzeProcs
              << ",scheduler_canonical_executable=" << options.schedulerExecutablePath
              << ",semantic_worker_registry="
              << options.semanticWorkerRegistry->canonicalRegistryPath()
              << ",current_worker_canonical_executable="
              << options.currentWorkerExecutablePath
              << ",analyze_worker_canonical_executable="
              << options.analyzeWorkerExecutablePath
              << ",legacy_layout6_identity_assertion="
              << options.legacyLayout6InferWorkerPath.value_or("NULL")
              << ",auto_evaluate_continuations="
              << (options.autoEvaluateContinuations ? "1" : "0")
              << ",auto_queue_continuations="
              << (options.autoQueueContinuations ? "1" : "0")
              << ",continuation_scan_seconds=" << options.continuationScanSeconds
              << ",continuation_max_queues_per_scan="
              << options.continuationMaxQueuesPerScan
              << ",continuation_dry_run="
              << ((options.continuationDryRun || options.dryRun) ? "1" : "0")
              << ",recover_orphans_only=" << (options.recoverOrphansOnly ? "1" : "0")
              << ",global_desired_state=" << initialGlobalState
              << std::endl;
    if (options.dryRun)
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
    if (options.recoverOrphansOnly)
    {
        if (options.dryRun)
        {
            std::cout << "SCHEDULER_ORPHAN_RECOVERY_SKIPPED"
                      << ",dry_run=1,reason=durable_reconciliation_disabled"
                      << std::endl;
        }
        else
        {
            std::cout << "SCHEDULER_ORPHAN_RECOVERY_DONE"
                      << ",recovered_or_failed=" << recoveryCount
                      << std::endl;
        }
        std::cout << "SCHEDULER_STOP"
                  << ",exit_code=0"
                  << std::endl;
        return 0;
    }

    ContinuationAutoScanState continuationScanState;
    EA::SchedulerCore::SchedulerDaemonOperations operations;
    operations.stopRequested = [] {
        return gSchedulerStopRequested != 0;
    };
    operations.refreshAuthority = [&] {
        return RefreshSchedulerAuthority(options);
    };
    operations.runCycle = [&] {
        return RunSchedulerOnce(options, &logState);
    };
    operations.runAutomaticContinuationScan = [&] {
        continuationScanState.lastCounts =
            RunAutomaticContinuationScan(options);
        continuationScanState.hasRun = true;
        continuationScanState.lastScanAt =
            EA::RunMetadata::CurrentUtcTimestamp();
    };
    operations.reportAuthorityLost = [&](std::string_view reason) {
        std::cerr << "SCHEDULER_OWNERSHIP_LOST"
                  << ",scheduler_invocation_id="
                  << options.schedulerAuthority.schedulerInvocationId
                  << ",fencing_token="
                  << options.schedulerAuthority.fencingToken
                  << ",reason=" << reason
                  << std::endl;
    };
    operations.sleepSeconds = [](unsigned int seconds) {
        ::sleep(seconds);
    };
    operations.reportStop = [](
        int result,
        bool ownershipLost,
        bool shutdownRequested) {
        std::cout << "SCHEDULER_STOP"
                  << ",exit_code=" << result
                  << ",ownership_lost=" << (ownershipLost ? 1 : 0)
                  << ",shutdown_requested="
                  << (shutdownRequested ? 1 : 0)
                  << std::endl;
    };
    return EA::SchedulerCore::SchedulerEngine().run(
        daemonConfiguration, operations);
}

} // namespace EA::SchedulerCore::ProductionRuntimeDetail

namespace EA::SchedulerCore
{

int RunProductionSchedulerDaemon(
    const SchedulerDaemonConfiguration& configuration)
{
    using namespace ProductionRuntimeDetail;
    SchedulerOptions options;
    options.scheduleExperiments = true;
    options.maxTrainProcs = configuration.maxTrainProcs;
    options.maxInferProcs = configuration.maxInferProcs;
    options.maxAnalyzeProcs = configuration.maxAnalyzeProcs;
    options.schedulerPollSeconds = configuration.schedulerPollSeconds;
    options.schedulerLogDir = configuration.schedulerLogDir;
    options.experimentReportDir = configuration.experimentReportDir;
    options.autoGenerateReports = configuration.autoGenerateReports;
    options.lstmProfileHotspots = configuration.lstmProfileHotspots;
    options.lstmProfileOutputPath = configuration.lstmProfileOutputPath;
    options.schedulerVerbose = configuration.schedulerVerbose;
    options.schedulerOnce = configuration.schedulerOnce;
    options.dryRun = configuration.dryRun;
    options.recoverOrphansOnly = configuration.recoverOrphansOnly;
    options.autoEvaluateContinuations =
        configuration.autoEvaluateContinuations;
    options.autoQueueContinuations = configuration.autoQueueContinuations;
    options.continuationDryRun = configuration.continuationDryRun;
    options.continuationScanSeconds = configuration.continuationScanSeconds;
    options.continuationMaxQueuesPerScan =
        configuration.continuationMaxQueuesPerScan;
    options.semanticWorkerRegistryPath =
        configuration.semanticWorkerRegistryPath;
    options.semanticWorkerRegistryPathSpecified = true;
    options.legacyLayout6InferWorkerPath =
        configuration.legacyLayout6InferWorkerPath;
    options.invocationCommandLine = configuration.invocationCommandLine;
    options.schedulerExecutablePath =
        EA::SchedulerCore::NativeWorkerProcessController()
            .resolveExecutablePath();
    return RunScheduler(std::move(options), configuration);
}

} // namespace EA::SchedulerCore
