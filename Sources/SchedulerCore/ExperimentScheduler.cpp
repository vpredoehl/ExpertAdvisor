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
#include "SchedulerStatusProcessRecognition.hpp"
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
#include "SchedulerCore/SchedulerDaemonCli.hpp"
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
#include "ExperimentPairComparisonService.hpp"
#include "ExperimentReplicationComparisonService.hpp"
#include "ExperimentReplicationPlanningService.hpp"
#include "FeatureAblationReplicationEvaluationService.hpp"
#include "CorrectedCausalSurpriseReplicationContinuationService.hpp"
#include "CausalSurpriseObservabilityService.hpp"
#include "EconomicEventRepository.hpp"
#include "SchedulerCore/ProductionSchedulerRuntimeInternal.hpp"

namespace EA::ExperimentScheduler
{
namespace
{

using namespace EA::SchedulerCore::ProductionRuntimeDetail;

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


struct QueueDefaults
{
    double threshold = default_c_next_threshold;
    double coreLrMult = default_core_lr_mult;
    double headLrMult = default_head_weight_lr_mult;
    int checkpointInterval = 20;
    std::string trainStart = "2010-01-01";
    std::string trainEnd = "2025-01-01";
    std::string inferStart = "2025-01-01";
    std::string inferEnd = "2026-01-01";
};


struct AutoResumeCandidate
{
    long long modelId = -1;
    std::string name;
    int completedEpochs = 0;
};



struct RunningExperimentChild
{
    ExperimentRow experiment;
    pid_t pid = -1;
    std::string logPath;
};


// The signal handler may only touch sig_atomic_t state. It is a bridge into
// the invocation-owned SchedulerRuntimeContext, whose child collection is
// consumed by normal scheduler control flow after the signal is observed.




struct RunningExperimentState
{
    ExperimentRow experiment;
    std::string phase;
    std::optional<int> workerPid;
    double attemptStartedEpoch = 0.0;
};


struct SchedulerStatusJob
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string phase;
    std::string status;
    int targetEpochs = 0;
    int checkpointInterval = 0;
    std::optional<long long> modelId;
    std::optional<int> completedEpochs;
    std::optional<int> currentEpoch;
    bool currentEpochFromTable = false;
    std::optional<int> lastCheckpointEpoch;
    std::optional<long long> lastCheckpointModelId;
    std::optional<int> nextCheckpointEpoch;
    std::optional<int> stopAfterCheckpointEpoch;
    std::optional<int> stoppedAtCheckpointEpoch;
    std::optional<long long> stoppedAtCheckpointModelId;
    std::optional<bool> opportunisticCheckpointInfer;
    std::optional<int> checkpointInferMinEpoch;
    std::optional<int> checkpointInferInterval;
    std::optional<bool> checkpointPolicyEnabled;
    std::optional<double> checkpointPolicyMinLeaderScore;
    std::optional<double> checkpointPolicyMinInferAccuracy;
    std::optional<int> checkpointPolicyTopN;
    std::optional<std::string> checkpointPolicyScope;
    std::optional<std::string> checkpointPolicyStopMode;
    std::optional<int> checkpointPolicyGraceEvals;
    std::optional<std::string> checkpointPolicyLastDecision;
    std::optional<long long> checkpointPolicyLastEvalId;
    std::optional<std::string> checkpointPolicyLastReason;
    int checkpointEvalPending = 0;
    int checkpointEvalRunning = 0;
    int checkpointEvalCompleted = 0;
    int checkpointEvalFailed = 0;
    std::optional<double> loss;
    std::optional<double> validationAccuracy;
    std::optional<double> elapsedSeconds;
    std::optional<double> etaSeconds;
    std::optional<int> pid;
    std::optional<double> cpuPercent;
    std::optional<double> memPercent;
    std::optional<double> rssMb;
    std::optional<std::string> recentProgress;
    std::optional<std::string> trainLogPath;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
    std::string currentOperation;
    std::string startedAt;
    std::string updatedAt;
    std::string completedAt;
    std::string errorMessage;
    bool operatorForcedFinalInferenceRerunRequested = false;
};

struct SchedulerCheckpointStatusJob
{
    long long checkpointEvalId = -1;
    long long experimentId = -1;
    int checkpointEpoch = 0;
    long long checkpointModelId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string status;
    std::string phase;
    std::optional<int> pid;
    std::optional<double> cpuPercent;
    std::optional<double> memPercent;
    std::optional<double> rssMb;
    std::string workerControlState;
    std::optional<std::string> inferLogPath;
};

struct SchedulerProcessResource
{
    int pid = -1;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerProcessInfo
{
    int pid = -1;
    std::string command;
    SchedulerProcessResource resource;
};

struct SchedulerResourceAggregate
{
    int workers = 0;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerStatusCounts
{
    int queued = 0;
    int paused = 0;
    int running = 0;
    int completed = 0;
    int failed = 0;
    int cancelled = 0;
};

struct SchedulerStopExperiment
{
    ExperimentRow experiment;
    std::string status;
    std::string phase;
    EA::GlobalExperimentControl::ManagedWorker worker;
    std::optional<long long> activeWorkerAttemptId;
};

struct SchedulerStopCandidate
{
    SchedulerStopExperiment experiment;
    std::optional<int> pid;
    std::string rejectionReason;
};

struct SchedulerDetectedWorker
{
    int pid = -1;
    std::string kind;
    std::string command;
    SchedulerProcessResource resource;
    bool stopped = false;
};

struct SchedulerUnmanagedWorker
{
    int pid = -1;
    std::string kind;
    std::string reason;
    std::string command;
};

struct SchedulerStatusProcessSnapshot
{
    bool processDetectionAvailable = false;
    std::vector<SchedulerProcessInfo> processes;
    std::vector<int> schedulerPids;
    std::map<int, SchedulerProcessResource> resourcesByPid;
    std::map<long long, int> trainPidByExperiment;
    std::map<long long, int> inferPidByExperiment;
    std::map<long long, int> analysisPidByExperiment;
    std::vector<SchedulerDetectedWorker> workerProcesses;
    SchedulerResourceAggregate schedulerResources;
    SchedulerResourceAggregate trainResources;
    SchedulerResourceAggregate inferResources;
    SchedulerResourceAggregate analysisResources;
    int trainWorkers = 0;
    int inferWorkers = 0;
    int analysisWorkers = 0;
    std::optional<int> maxTrainProcs;
    std::optional<int> maxInferProcs;
    std::optional<int> maxAnalyzeProcs;
    std::optional<int> schedulerPollSeconds;
    std::optional<bool> autoEvaluateContinuations;
    std::optional<bool> autoQueueContinuations;
    std::optional<bool> continuationDryRun;
    std::optional<int> continuationScanSeconds;
    std::optional<int> continuationMaxQueuesPerScan;
    std::optional<double> totalCpuPercent;
    std::optional<double> systemMemoryUsedMb;
    std::optional<double> systemMemoryTotalMb;
};

struct SchedulerWorkerAccounting
{
    int managedTrain = 0;
    int managedInfer = 0;
    int managedAnalyze = 0;
    int managedRunningTrain = 0;
    int managedRunningInfer = 0;
    int managedRunningAnalyze = 0;
    int managedPausedTrain = 0;
    int managedPausedInfer = 0;
    int managedPausedAnalyze = 0;
    int unmanagedTrain = 0;
    int unmanagedInfer = 0;
    int unmanagedAnalyze = 0;
    int identityMismatchTrain = 0;
    int identityMismatchInfer = 0;
    int identityMismatchAnalyze = 0;
    int expectedMissingTrain = 0;
    int expectedMissingInfer = 0;
    int expectedMissingAnalyze = 0;
    SchedulerResourceAggregate managedTrainResources;
    SchedulerResourceAggregate managedInferResources;
    SchedulerResourceAggregate managedAnalysisResources;
    SchedulerResourceAggregate unmanagedTrainResources;
    SchedulerResourceAggregate unmanagedInferResources;
    SchedulerResourceAggregate unmanagedAnalysisResources;
    std::vector<SchedulerUnmanagedWorker> unmanagedWorkers;
    std::vector<EA::GlobalExperimentControl::SchedulerWorkerClassification>
        workerClassifications;
};

struct SchedulerIntelligenceRecord
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string symbol;
    int predictionHorizon = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    int targetEpochs = 0;
    std::optional<int> completedEpochs;
};

struct SchedulerDominatedRecord
{
    SchedulerIntelligenceRecord dominated;
    long long dominatingExperimentId = -1;
    std::optional<double> dominatingLeaderScore;
    std::optional<double> leaderScoreDelta;
};

struct SchedulerIntelligenceSnapshot
{
    std::optional<SchedulerIntelligenceRecord> overallLeader;
    std::optional<SchedulerIntelligenceRecord> recentBest24h;
    std::vector<SchedulerIntelligenceRecord> leadersBySymbol;
    std::vector<SchedulerIntelligenceRecord> leadersByHorizon;
    std::vector<SchedulerIntelligenceRecord> top5;
    std::vector<SchedulerIntelligenceRecord> worst5;
    std::vector<SchedulerDominatedRecord> dominated;
    long long completedToday = 0;
    long long failedToday = 0;
    int waitingTrain = 0;
    int waitingInfer = 0;
    int waitingAnalyze = 0;
};

struct ModelInfoRecord
{
    long long modelId = -1;
    std::optional<long long> experimentId;
    std::string name = "unknown";
    std::string createdAt = "unknown";
    std::string comment = "unknown";
    std::string symbol = "unknown";
    std::optional<int> predictionHorizon;
    std::optional<double> threshold;
    std::optional<int> windowSize;
    std::optional<int> completedEpochs;
    std::optional<int> targetEpochs;
    std::string trainStart = "unknown";
    std::string trainEnd = "unknown";
    std::string inferStart = "unknown";
    std::string inferEnd = "unknown";
    std::optional<int> checkpointInterval;
    std::string optimizer = "unknown";
    std::optional<long long> optimizerUpdateCount;
    std::string targetType = "unknown";
    bool isCheckpoint = false;
    bool isResumable = false;
    std::optional<long long> parentResumeModelId;
};

struct ExperimentModelRow
{
    long long modelId = -1;
    std::optional<long long> experimentId;
    std::optional<long long> parentModelId;
    std::string name;
    std::string createdAt;
    std::string comment;
};





bool IsExperimentSchedulerCommandImpl(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        if (arg == "--verify-profitability-evidence" ||
            arg.rfind("--verify-profitability-evidence=", 0) == 0 ||
            arg == "--campaign-profitability-readiness" ||
            arg.rfind("--campaign-profitability-readiness=", 0) == 0 ||
            arg == "--shadow-rank-campaign-profitability" ||
            arg.rfind("--shadow-rank-campaign-profitability=", 0) == 0 ||
            arg == "--profitability-shadow-weights" ||
            arg.rfind("--profitability-shadow-weights=", 0) == 0 ||
            arg == "--calibrate-campaign-profitability" ||
            arg.rfind("--calibrate-campaign-profitability=", 0) == 0 ||
            arg == "--validate-campaign-profitability-temporal" ||
            arg == "--prepare-campaign-profitability-forward-validation" ||
            arg.rfind(
                "--prepare-campaign-profitability-forward-validation=", 0) == 0 ||
            arg == "--prepare-campaign-profitability-outcome-jobs" ||
            arg.rfind("--prepare-campaign-profitability-outcome-jobs=", 0) == 0 ||
            arg == "--compare-campaign-profitability-prospective" ||
            arg.rfind("--compare-campaign-profitability-prospective=", 0) == 0)
            return true;
        if (arg == "--compare-feature-ablation-pair" ||
            arg.rfind("--compare-feature-ablation-pair=", 0) == 0 ||
            arg == "--expected-ablation-mask" ||
            arg.rfind("--expected-ablation-mask=", 0) == 0 ||
            arg == "--compare-feature-ablation-replications" ||
            arg.rfind("--compare-feature-ablation-replications=", 0) == 0)
            return true;
        if (arg == "--compare-experiment-pair" ||
            arg.rfind("--compare-experiment-pair=", 0) == 0 ||
            arg == "--compare-experiment-replications" ||
            arg.rfind("--compare-experiment-replications=", 0) == 0 ||
            arg == "--plan-experiment-replications" ||
            arg.rfind("--plan-experiment-replications=", 0) == 0 ||
            arg == "--replication-seeds" ||
            arg.rfind("--replication-seeds=", 0) == 0)
            return true;
        if (arg == "--corrected-causal-surprise-replication-status" ||
            arg.rfind(
                "--corrected-causal-surprise-replication-status=", 0) == 0 ||
            arg == "--materialize-corrected-causal-surprise-replication" ||
            arg.rfind(
                "--materialize-corrected-causal-surprise-replication=", 0) == 0 ||
            arg == "--corrected-causal-surprise-anchor-pair" ||
            arg.rfind(
                "--corrected-causal-surprise-anchor-pair=", 0) == 0 ||
            arg == "--expected-corrected-replication-plan-hash" ||
            arg.rfind(
                "--expected-corrected-replication-plan-hash=", 0) == 0)
            return true;
        if (arg == "--causal-surprise-observability" ||
            arg.rfind("--causal-surprise-observability=", 0) == 0 ||
            arg == "--causal-surprise-observability-scope" ||
            arg.rfind("--causal-surprise-observability-scope=", 0) == 0)
            return true;
        if (arg == "--causal-surprise-coverage-gaps" ||
            arg.rfind("--causal-surprise-coverage-gaps=", 0) == 0 ||
            arg == "--causal-surprise-coverage-gaps-scope" ||
            arg.rfind("--causal-surprise-coverage-gaps-scope=", 0) == 0)
            return true;
        if (arg == "--compare-training-objective-pair" ||
            arg == "--pair-primary-profitability-metric" ||
            arg == "--pair-min-profitability-improvement" ||
            arg == "--pair-max-profitability-worsening" ||
            arg == "--pair-max-infer-accuracy-decrease" ||
            arg == "--pair-max-accept-accuracy-decrease" ||
            arg == "--pair-max-accept-rate-decrease" ||
            arg == "--pair-max-leader-score-decrease" ||
            arg == "--pair-max-neutral-proportion-increase" ||
            arg.rfind("--compare-training-objective-pair=", 0) == 0 ||
            arg.rfind("--pair-primary-profitability-metric=", 0) == 0 ||
            arg.rfind("--pair-min-profitability-improvement=", 0) == 0 ||
            arg.rfind("--pair-max-profitability-worsening=", 0) == 0 ||
            arg.rfind("--pair-max-infer-accuracy-decrease=", 0) == 0 ||
            arg.rfind("--pair-max-accept-accuracy-decrease=", 0) == 0 ||
            arg.rfind("--pair-max-accept-rate-decrease=", 0) == 0 ||
            arg.rfind("--pair-max-leader-score-decrease=", 0) == 0 ||
            arg.rfind("--pair-max-neutral-proportion-increase=", 0) == 0)
            return true;
        if (arg == "--schedule-experiments" ||
            arg == "--complete-scheduler-protocol-cutover" ||
            arg == "--create-economic-calendar-snapshot" ||
            arg == "--model-info" ||
            arg == "--status" ||
            arg == "--enqueue-experiment" ||
            arg == "--queue-experiment" ||
            arg == "--queue-sweep" ||
            arg == "--analyze-completed-experiments" ||
            arg == "--print-experiment-leaderboard" ||
            arg == "--generate-experiment-reports" ||
            arg == "--scheduler-status" ||
            arg == "--backfill-experiment-metadata" ||
            arg == "--backup-database" ||
            arg == "--backup-output" ||
            arg == "--experiment-metadata" ||
            arg == "--list-experiment-models" ||
            arg == "--list-experiment-lineage" ||
            arg == "--include-parent-models" ||
            arg == "--stop-experiment" ||
            arg == "--reconcile-worker-attempt" ||
            arg == "--recover-failed-inference" ||
            arg == "--stop-all-experiments" ||
            arg == "--pause-all-experiments" ||
            arg == "--resume-all-experiments" ||
            arg == "--pause-all" ||
            arg == "--resume-all" ||
            arg == "--cancel-all-experiments" ||
            arg == "--immediate" ||
            arg == "--after-next-checkpoint" ||
            arg == "--infer-before-cancel" ||
            arg == "--pause-experiment" ||
            arg == "--resume-experiment" ||
            arg == "--pause-campaign-materialization" ||
            arg == "--resume-campaign-materialization" ||
            arg == "--set-experiment-priority" ||
            arg == "--cancel-experiment" ||
            arg == "--retry-failed-experiment" ||
            arg == "--requeue-training" ||
            arg == "--retry-checkpoint-eval" ||
            arg == "--evaluate-checkpoint-policy" ||
            arg == "--checkpoint-policy-status" ||
            arg == "--enable-continuation-policy" ||
            arg == "--disable-continuation-policy" ||
            arg == "--set-continuation-policy" ||
            arg == "--evaluate-continuation" ||
            arg == "--queue-continuation" ||
            arg == "--continuation-status" ||
            arg == "--generate-experiment-recommendations" ||
            arg == "--list-experiment-recommendations" ||
            arg == "--recommendation-status" ||
            arg == "--list-experiment-recommendation-scans" ||
            arg == "--recommendation-scan-status" ||
            arg == "--recommendation-policy" ||
            arg == "--recommendation-symbol" ||
            arg == "--recommendation-horizon" ||
            arg == "--recommendation-source-experiment" ||
            arg == "--recommendation-max" ||
            arg == "--recommendation-status-filter" ||
            arg == "--recommendation-scan-id" ||
            arg == "--recommendation-limit" ||
            arg == "--score-experiment-recommendations" ||
            arg == "--list-experiment-recommendation-scores" ||
            arg == "--recommendation-score-status" ||
            arg == "--list-experiment-recommendation-score-runs" ||
            arg == "--recommendation-score-run-status" ||
            arg == "--explain-recommendation-score" ||
            arg == "--recommendation-scoring-policy" ||
            arg == "--recommendation-id" ||
            arg == "--recommendation-score-run-id" ||
            arg == "--recommendation-score-min" ||
            arg == "--recommendation-score-limit" ||
            arg == "--approve-experiment-recommendation" ||
            arg == "--reject-experiment-recommendation" ||
            arg == "--expire-experiment-recommendation" ||
            arg == "--list-experiment-recommendation-reviews" ||
            arg == "--recommendation-review-status" ||
            arg == "--recommendation-review-history" ||
            arg == "--recommendation-review-reason-code" ||
            arg == "--recommendation-review-reason" ||
            arg == "--recommendation-reviewer" ||
            arg == "--recommendation-review-note" ||
            arg == "--recommendation-review-score-id" ||
            arg == "--recommendation-review-action" ||
            arg == "--recommendation-review-limit" ||
            arg == "--evaluate-experiment-recommendations" ||
            arg == "--evaluate-experiment-recommendation" ||
            arg == "--list-experiment-recommendation-evaluations" ||
            arg == "--recommendation-evaluation-status" ||
            arg == "--explain-recommendation-evaluation" ||
            arg == "--list-experiment-recommendation-evaluation-runs" ||
            arg == "--recommendation-evaluation-run-status" ||
            arg == "--recommendation-evaluation-policy" ||
            arg == "--recommendation-evaluation-disposition" ||
            arg == "--recommendation-evaluation-limit" ||
            arg == "--recommendation-evaluation-dry-run" ||
            arg == "--rank-experiment-recommendation-evaluations" ||
            arg == "--recommendation-ranking-evaluation-run-id" ||
            arg == "--recommendation-ranking-scan-id" ||
            arg == "--recommendation-ranking-symbol" ||
            arg == "--recommendation-ranking-horizon" ||
            arg == "--recommendation-ranking-family" ||
            arg == "--recommendation-ranking-global" ||
            arg == "--recommendation-ranking-limit" ||
            arg == "--recommendation-ranking-dry-run" ||
            arg == "--list-experiment-recommendation-ranking-snapshots" ||
            arg == "--recommendation-ranking-status" ||
            arg == "--list-experiment-recommendation-ranking-members" ||
            arg == "--recommendation-ranking-member-status" ||
            arg == "--recommendation-ranking-bucket" ||
            arg == "--compare-experiment-recommendation-evaluations" ||
            arg == "--compare-experiment-recommendation-ranking-members" ||
            arg == "--approve-conversion-proposal" ||
            arg == "--reject-conversion-proposal" ||
            arg == "--conversion-proposal-review-request-id" ||
            arg == "--conversion-proposal-review-operator" ||
            arg == "--conversion-proposal-review-reason" ||
            arg == "--show-conversion-proposal" ||
            arg == "--list-conversion-proposal-reviews" ||
            arg == "--list-conversion-proposals-by-review-status" ||
            arg == "--conversion-proposal-review-limit" ||
            arg == "--execute-approved-conversion-proposal" ||
            arg == "--conversion-proposal-execution-status" ||
            arg == "--activate-recommendation-conversion-execution" ||
            arg == "--recommendation-conversion-activation-status" ||
            arg == "--recommendation-conversion-workflow" ||
            arg == "--list-recommendation-conversion-workflows" ||
            arg == "--conversion-workflow-state" ||
            arg == "--conversion-workflow-limit" ||
            arg == "--plan-recommendation-campaign" ||
            arg == "--review-recommendation-campaign" ||
            arg == "--approve-recommendation-campaign" ||
            arg == "--reject-recommendation-campaign" ||
            arg == "--campaign-review-identity-hash" ||
            arg == "--campaign-reviewer" ||
            arg == "--campaign-review-reason" ||
            arg == "--show-recommendation-campaign-approval" ||
            arg == "--list-recommendation-campaign-approvals" ||
            arg == "--campaign-approval-decision" ||
            arg == "--campaign-approval-limit" ||
            arg == "--materialize-recommendation-campaign" ||
            arg == "--campaign-approval-id" ||
            arg == "--campaign-materialized-by" ||
            arg == "--campaign-materialization-reason" ||
            arg == "--show-recommendation-campaign-materialization" ||
            arg == "--list-recommendation-campaign-materializations" ||
            arg == "--campaign-materialization-limit" ||
            arg == "--show-recommendation-campaign-handoff" ||
            arg == "--list-recommendation-campaign-handoffs" ||
            arg == "--campaign-handoff-limit" ||
            arg == "--review-recommendation-campaign-materialization" ||
            arg == "--campaign-proposal-review-decision" ||
            arg == "--campaign-proposal-review-operator" ||
            arg == "--campaign-proposal-review-reason" ||
            arg == "--execute-recommendation-campaign-materialization" ||
            arg == "--activate-recommendation-campaign-materialization" ||
            arg == "--launch-recommendation-campaign-materialization" ||
            arg == "--recommendation-campaign-status" ||
            arg == "--recommendation-campaign-outcome-assessment" ||
            arg == "--campaign-operations-budget-grant" ||
            arg == "--campaign-operations-budget-amend" ||
            arg == "--campaign-operations-budget-revoke" ||
            arg == "--campaign-operations-budget-supersede" ||
            arg == "--campaign-operations-admit" ||
            arg == "--campaign-operations-accept-request" ||
            arg == "--campaign-operations-budget-status" ||
            arg == "--campaign-operations-request-status" ||
            arg == "--campaign-operations-pause" ||
            arg == "--campaign-operations-resume" ||
            arg == "--campaign-operations-cancel" ||
            arg == "--campaign-operations-control-status" ||
            arg == "--campaign-operations-complete-if-settled" ||
            arg == "--campaign-operations-completion-status" ||
            arg == "--campaign-operations-production-readiness" ||
            arg == "--campaign-operations-production-status" ||
            arg == "--campaign-operations-production-enable" ||
            arg == "--campaign-operations-production-disable" ||
            arg == "--campaign-operations-dispatch-request" ||
            arg == "--campaign-operations-manager-run-once" ||
            arg == "--campaign-operations-reconcile-observe" ||
            arg == "--campaign-operations-reconcile-recover" ||
            arg == "--campaign-operations-expected-control-version" ||
            arg == "--campaign-operations-request-id" ||
            arg == "--campaign-operations-expected-request-version" ||
            arg == "--campaign-operations-expected-production-version" ||
            arg == "--campaign-operations-operation-key" ||
            arg == "--campaign-operations-independent-verification-reference" ||
            arg == "--campaign-operations-reconcile-after-request-id" ||
            arg == "--campaign-operations-reconcile-limit" ||
            arg == "--campaign-operations-expected-budget-version" ||
            arg == "--campaign-operations-budget-value" ||
            arg == "--campaign-operations-actor" ||
            arg == "--campaign-operations-reason" ||
            arg == "--campaign-operations-reservation-expires-at" ||
            arg == "--campaign-ranking-snapshot" ||
            arg == "--campaign-limit" ||
            arg == "--campaign-candidate-limit" ||
            arg == "--campaign-symbol" ||
            arg == "--campaign-horizon" ||
            arg == "--campaign-min-leader-score" ||
            arg == "--campaign-min-inference-accuracy" ||
            arg == "--campaign-max-neutral-proportion" ||
            arg == "--campaign-min-profitability" ||
            arg == "--campaign-max-per-symbol" ||
            arg == "--campaign-max-per-horizon" ||
            arg == "--campaign-max-per-source-experiment" ||
            arg == "--campaign-reconsider-rejected" ||
            arg == "--campaign-reconsider-failed" ||
            arg == "--campaign-reconsider-cancelled" ||
            arg == "--auto-evaluate-continuations" ||
            arg == "--auto-queue-continuations" ||
            arg == "--continuation-scan-seconds" ||
            arg == "--continuation-max-queues-per-scan" ||
            arg == "--continuation-dry-run" ||
            arg == "--requeue-analysis" ||
            arg == "--requeue-inference" ||
            arg == "--stop-after-checkpoint" ||
            arg == "--clear-stop-after-checkpoint" ||
            arg == "--stop-after-checkpoint-all" ||
            arg == "--clear-stop-after-checkpoint-all" ||
            arg == "--enable-checkpoint-infer" ||
            arg == "--disable-checkpoint-infer" ||
            arg == "--checkpoint-infer-min-epoch" ||
            arg == "--checkpoint-infer-interval" ||
            arg == "--enable-checkpoint-policy" ||
            arg == "--disable-checkpoint-policy" ||
            arg == "--set-checkpoint-policy" ||
            arg == "--help" ||
            arg == "--analyze-experiment" ||
            arg == "--experiment-id" ||
            arg.rfind("--analyze-experiment=", 0) == 0 ||
            arg.rfind("--experiment-id=", 0) == 0 ||
            arg.rfind("--stop-experiment=", 0) == 0 ||
            arg.rfind("--reconcile-worker-attempt=", 0) == 0 ||
            arg.rfind("--recover-failed-inference=", 0) == 0 ||
            arg.rfind("--pause-experiment=", 0) == 0 ||
            arg.rfind("--resume-experiment=", 0) == 0 ||
            arg.rfind("--pause-campaign-materialization=", 0) == 0 ||
            arg.rfind("--resume-campaign-materialization=", 0) == 0 ||
            arg.rfind("--set-experiment-priority=", 0) == 0 ||
            arg.rfind("--cancel-experiment=", 0) == 0 ||
            arg.rfind("--retry-failed-experiment=", 0) == 0 ||
            arg.rfind("--requeue-training=", 0) == 0 ||
            arg.rfind("--retry-checkpoint-eval=", 0) == 0 ||
            arg.rfind("--evaluate-checkpoint-policy=", 0) == 0 ||
            arg.rfind("--enable-continuation-policy=", 0) == 0 ||
            arg.rfind("--disable-continuation-policy=", 0) == 0 ||
            arg.rfind("--set-continuation-policy=", 0) == 0 ||
            arg.rfind("--evaluate-continuation=", 0) == 0 ||
            arg.rfind("--queue-continuation=", 0) == 0 ||
            arg.rfind("--continuation-status=", 0) == 0 ||
            arg.rfind("--recommendation-status=", 0) == 0 ||
            arg.rfind("--recommendation-scan-status=", 0) == 0 ||
            arg.rfind("--recommendation-policy=", 0) == 0 ||
            arg.rfind("--recommendation-symbol=", 0) == 0 ||
            arg.rfind("--recommendation-horizon=", 0) == 0 ||
            arg.rfind("--recommendation-source-experiment=", 0) == 0 ||
            arg.rfind("--recommendation-max=", 0) == 0 ||
            arg.rfind("--recommendation-status-filter=", 0) == 0 ||
            arg.rfind("--recommendation-scan-id=", 0) == 0 ||
            arg.rfind("--recommendation-limit=", 0) == 0 ||
            arg.rfind("--recommendation-score-status=", 0) == 0 ||
            arg.rfind("--recommendation-score-run-status=", 0) == 0 ||
            arg.rfind("--explain-recommendation-score=", 0) == 0 ||
            arg.rfind("--recommendation-scoring-policy=", 0) == 0 ||
            arg.rfind("--recommendation-id=", 0) == 0 ||
            arg.rfind("--recommendation-score-run-id=", 0) == 0 ||
            arg.rfind("--recommendation-score-min=", 0) == 0 ||
            arg.rfind("--recommendation-score-limit=", 0) == 0 ||
            arg.rfind("--approve-experiment-recommendation=", 0) == 0 ||
            arg.rfind("--reject-experiment-recommendation=", 0) == 0 ||
            arg.rfind("--expire-experiment-recommendation=", 0) == 0 ||
            arg.rfind("--recommendation-review-status=", 0) == 0 ||
            arg.rfind("--recommendation-review-history=", 0) == 0 ||
            arg.rfind("--recommendation-review-reason-code=", 0) == 0 ||
            arg.rfind("--recommendation-review-reason=", 0) == 0 ||
            arg.rfind("--recommendation-reviewer=", 0) == 0 ||
            arg.rfind("--recommendation-review-note=", 0) == 0 ||
            arg.rfind("--recommendation-review-score-id=", 0) == 0 ||
            arg.rfind("--recommendation-review-action=", 0) == 0 ||
            arg.rfind("--recommendation-review-limit=", 0) == 0 ||
            arg.rfind("--evaluate-experiment-recommendation=", 0) == 0 ||
            arg.rfind("--recommendation-evaluation-status=", 0) == 0 ||
            arg.rfind("--explain-recommendation-evaluation=", 0) == 0 ||
            arg.rfind("--recommendation-evaluation-run-status=", 0) == 0 ||
            arg.rfind("--recommendation-evaluation-policy=", 0) == 0 ||
            arg.rfind("--recommendation-evaluation-disposition=", 0) == 0 ||
            arg.rfind("--recommendation-evaluation-limit=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-evaluation-run-id=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-scan-id=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-symbol=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-horizon=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-family=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-limit=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-status=", 0) == 0 ||
            arg.rfind("--list-experiment-recommendation-ranking-members=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-member-status=", 0) == 0 ||
            arg.rfind("--recommendation-ranking-bucket=", 0) == 0 ||
            arg.rfind("--compare-experiment-recommendation-evaluations=", 0) == 0 ||
            arg.rfind("--compare-experiment-recommendation-ranking-members=", 0) == 0 ||
            arg.rfind("--approve-conversion-proposal=", 0) == 0 ||
            arg.rfind("--reject-conversion-proposal=", 0) == 0 ||
            arg.rfind("--conversion-proposal-review-request-id=", 0) == 0 ||
            arg.rfind("--conversion-proposal-review-operator=", 0) == 0 ||
            arg.rfind("--conversion-proposal-review-reason=", 0) == 0 ||
            arg.rfind("--show-conversion-proposal=", 0) == 0 ||
            arg.rfind("--list-conversion-proposal-reviews=", 0) == 0 ||
            arg.rfind("--list-conversion-proposals-by-review-status=", 0) == 0 ||
            arg.rfind("--conversion-proposal-review-limit=", 0) == 0 ||
            arg.rfind("--execute-approved-conversion-proposal=", 0) == 0 ||
            arg.rfind("--conversion-proposal-execution-status=", 0) == 0 ||
            arg.rfind(
                "--activate-recommendation-conversion-execution=", 0) == 0 ||
            arg.rfind(
                "--recommendation-conversion-activation-status=", 0) == 0 ||
            arg.rfind("--recommendation-conversion-workflow=", 0) == 0 ||
            arg.rfind("--conversion-workflow-state=", 0) == 0 ||
            arg.rfind("--conversion-workflow-limit=", 0) == 0 ||
            arg.rfind("--campaign-ranking-snapshot=", 0) == 0 ||
            arg.rfind("--campaign-limit=", 0) == 0 ||
            arg.rfind("--campaign-candidate-limit=", 0) == 0 ||
            arg.rfind("--campaign-symbol=", 0) == 0 ||
            arg.rfind("--campaign-horizon=", 0) == 0 ||
            arg.rfind("--campaign-min-leader-score=", 0) == 0 ||
            arg.rfind("--campaign-min-inference-accuracy=", 0) == 0 ||
            arg.rfind("--campaign-max-neutral-proportion=", 0) == 0 ||
            arg.rfind("--campaign-min-profitability=", 0) == 0 ||
            arg.rfind("--campaign-max-per-symbol=", 0) == 0 ||
            arg.rfind("--campaign-max-per-horizon=", 0) == 0 ||
            arg.rfind("--campaign-max-per-source-experiment=", 0) == 0 ||
            arg.rfind("--campaign-review-identity-hash=", 0) == 0 ||
            arg.rfind("--campaign-reviewer=", 0) == 0 ||
            arg.rfind("--campaign-review-reason=", 0) == 0 ||
            arg.rfind("--show-recommendation-campaign-approval=", 0) == 0 ||
            arg.rfind("--campaign-approval-decision=", 0) == 0 ||
            arg.rfind("--campaign-approval-limit=", 0) == 0 ||
            arg.rfind("--campaign-approval-id=", 0) == 0 ||
            arg.rfind("--campaign-materialized-by=", 0) == 0 ||
            arg.rfind("--campaign-materialization-reason=", 0) == 0 ||
            arg.rfind("--show-recommendation-campaign-materialization=", 0) == 0 ||
            arg.rfind("--campaign-materialization-limit=", 0) == 0 ||
            arg.rfind("--show-recommendation-campaign-handoff=", 0) == 0 ||
            arg.rfind("--campaign-handoff-limit=", 0) == 0 ||
            arg.rfind(
                "--review-recommendation-campaign-materialization=", 0) == 0 ||
            arg.rfind("--campaign-proposal-review-decision=", 0) == 0 ||
            arg.rfind("--campaign-proposal-review-operator=", 0) == 0 ||
            arg.rfind("--campaign-proposal-review-reason=", 0) == 0 ||
            arg.rfind(
                "--execute-recommendation-campaign-materialization=", 0) == 0 ||
            arg.rfind(
                "--activate-recommendation-campaign-materialization=", 0) == 0 ||
            arg.rfind(
                "--launch-recommendation-campaign-materialization=", 0) == 0 ||
            arg.rfind("--recommendation-campaign-status=", 0) == 0 ||
            arg.rfind(
                "--recommendation-campaign-outcome-assessment=", 0) == 0 ||
            arg.rfind("--continuation-scan-seconds=", 0) == 0 ||
            arg.rfind("--continuation-max-queues-per-scan=", 0) == 0 ||
            arg.rfind("--requeue-analysis=", 0) == 0 ||
            arg.rfind("--requeue-inference=", 0) == 0 ||
            arg.rfind("--stop-after-checkpoint=", 0) == 0 ||
            arg.rfind("--clear-stop-after-checkpoint=", 0) == 0 ||
            arg.rfind("--stop-after-checkpoint-all=", 0) == 0 ||
            arg.rfind("--enable-checkpoint-infer=", 0) == 0 ||
            arg.rfind("--disable-checkpoint-infer=", 0) == 0 ||
            arg.rfind("--checkpoint-infer-min-epoch=", 0) == 0 ||
            arg.rfind("--checkpoint-infer-interval=", 0) == 0 ||
            arg.rfind("--enable-checkpoint-policy=", 0) == 0 ||
            arg.rfind("--disable-checkpoint-policy=", 0) == 0 ||
            arg.rfind("--set-checkpoint-policy=", 0) == 0 ||
            arg.rfind("--backup-output=", 0) == 0 ||
            arg.rfind("--list-experiment-models=", 0) == 0 ||
            arg.rfind("--list-experiment-lineage=", 0) == 0 ||
            arg.rfind("--experiment-metadata=", 0) == 0)
            return true;
    }
    return false;
}



std::string ForexDbConnectionString()
{
    return "hostaddr=" + GetEnvOrDefault("FOREX_DB_HOST", "127.0.0.1") +
           " gssencmode=disable user=pqxx dbname=" +
           GetEnvOrDefault("FOREX_DB_NAME", "forex");
}

std::string LibpqConnectionValue(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('\'');
    for (const char character : value)
    {
        if (character == '\'' || character == '\\')
            escaped.push_back('\\');
        escaped.push_back(character);
    }
    escaped.push_back('\'');
    return escaped;
}

std::string CampaignOperationsProductionConnectionString(
    const char* principalEnvironmentVariable)
{
    const char* principal = std::getenv(principalEnvironmentVariable);
    if (principal == nullptr || *principal == '\0')
        throw std::runtime_error(
            std::string{"missing required Campaign Operations production "}
            + "principal environment variable " + principalEnvironmentVariable);

    // This is intentionally separate from LstmDbConnectionString(): only the
    // accepted Phase-H production command family may use a deployment LOGIN.
    return "hostaddr=" + LibpqConnectionValue(
               GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1")) +
           " gssencmode=disable user=" + LibpqConnectionValue(principal) +
           " dbname=" + LibpqConnectionValue(GetEnvOrDefault("LSTM_DB_NAME", "LSTM"));
}

std::string CampaignOperationsPrePhaseHConnectionString()
{
    const char* principal = std::getenv(
        "CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER");
    if (principal == nullptr || *principal == '\0')
        throw std::runtime_error(
            "missing required Campaign Operations pre-Phase-H principal "
            "environment variable CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER");

    // Pre-Phase-H capability assignment is a reviewed deployment concern.
    // It is intentionally separate from both the generic pqxx connection and
    // every Phase-H production-principal environment variable.
    return "hostaddr=" + LibpqConnectionValue(
               GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1")) +
           " gssencmode=disable user=" + LibpqConnectionValue(principal) +
           " dbname=" + LibpqConnectionValue(GetEnvOrDefault("LSTM_DB_NAME", "LSTM"));
}

void ValidateCampaignOperationsPrePhaseHPrincipal(
    const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    const bool prohibited = transaction.exec(
        "SELECT rolsuper OR EXISTS ("
        "SELECT 1 FROM pg_roles production_role WHERE "
        "production_role.rolname = ANY(ARRAY["
        "'campaign_operations_production_enabler',"
        "'campaign_operations_production_disabler',"
        "'campaign_operations_production_dispatcher',"
        "'campaign_operations_production_dispatch_service',"
        "'campaign_operations_production_phase5_transactional',"
        "'campaign_operations_production_reader',"
        "'campaign_operations_scheduler_protocol_evidence_reader']) "
        "AND pg_has_role(current_user, production_role.rolname, 'MEMBER')) "
        "FROM pg_roles WHERE rolname = current_user;").one_row()[0].as<bool>();
    if (prohibited)
        throw std::runtime_error(
            "campaign_operations_pre_phase_h_principal_invalid");
}

std::string CurrentLocalFilenameTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d_%H%M%S");
    return out.str();
}

std::string ShortGitCommit()
{
    const EA::RunMetadata::Snapshot metadata =
        EA::RunMetadata::Capture("LSTM_Release", "backup-database");
    if (metadata.gitCommit == "unknown" || metadata.gitCommit.empty())
        return "unknown";
    return metadata.gitCommit.substr(0, std::min<size_t>(7, metadata.gitCommit.size()));
}

int RunProcessAndWait(const std::vector<std::string>& args)
{
    if (args.empty())
        throw std::invalid_argument("RunProcessAndWait requires argv");

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args)
        argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error("fork failed for " + args.front() + ": errno=" + std::to_string(errno));

    if (pid == 0)
    {
        ::execvp(argv[0], argv.data());
        ::_exit(errno == ENOENT ? 127 : 126);
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0)
    {
        if (errno == EINTR)
            continue;
        throw std::runtime_error("waitpid failed for " + args.front() + ": errno=" + std::to_string(errno));
    }

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
}

std::string JsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (const char ch : value)
    {
        switch (ch)
        {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20)
                {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                }
                else
                {
                    out << ch;
                }
                break;
        }
    }
    return out.str();
}

std::string JsonString(const std::string& value)
{
    return "\"" + JsonEscape(value) + "\"";
}

std::string JsonOptionalLongLong(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "null";
}







bool SplitOptionWithValue(const std::string& arg,
                                 const std::string& optionName,
                                 std::string& value)
{
    const std::string prefix = optionName + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;
    value = arg.substr(prefix.size());
    return true;
}

std::string RequireNextArg(int argc, const char* argv[], int& i, const std::string& optionName)
{
    if (i + 1 >= argc)
        throw std::invalid_argument(optionName + " requires a value");
    return argv[++i];
}

int ParsePositiveInt(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return static_cast<int>(parsed);
}

long long ParsePositiveLongLong(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

std::tuple<long long, std::string, std::string>
ParseCampaignProfitabilityForwardValidationSpec(const std::string& value)
{
    const std::size_t first = value.find(',');
    const std::size_t second = first == std::string::npos
        ? std::string::npos : value.find(',', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        value.find(',', second + 1) != std::string::npos || first == 0 ||
        second == first + 1 || second + 1 == value.size())
        throw std::invalid_argument(
            "--prepare-campaign-profitability-forward-validation requires "
            "RANKING_SNAPSHOT_ID,OUTCOME_START,OUTCOME_END");
    return {
        ParsePositiveLongLong(
            "--prepare-campaign-profitability-forward-validation",
            value.substr(0, first)),
        value.substr(first + 1, second - first - 1),
        value.substr(second + 1)};
}

long long ParseSignedLongLong(
    const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size())
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    return parsed;
}

std::pair<long long, int> ParseExperimentEpochPair(const std::string& optionName,
                                                  const std::string& value)
{
    const size_t colon = value.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size())
        throw std::invalid_argument(optionName + " requires EXPERIMENT_ID:EPOCH");

    return {
        ParsePositiveLongLong(optionName, value.substr(0, colon)),
        ParsePositiveInt(optionName, value.substr(colon + 1))
    };
}

std::pair<long long, long long> ParsePositiveIdPair(
    const std::string& optionName,
    const std::string& value)
{
    const size_t colon = value.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size() ||
        value.find(':', colon + 1) != std::string::npos)
        throw std::invalid_argument(optionName + " requires LEFT_ID:RIGHT_ID");
    return {ParsePositiveLongLong(optionName, value.substr(0, colon)),
            ParsePositiveLongLong(optionName, value.substr(colon + 1))};
}

double ParsePositiveDouble(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

double ParsePositiveFiniteDouble(const std::string& optionName, const std::string& value)
{
    const double parsed = ParsePositiveDouble(optionName, value);
    if (!std::isfinite(parsed))
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

double ParseFiniteDouble(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

[[maybe_unused]] double ParseNonNegativeFiniteDouble(
    const std::string& optionName,
    const std::string& value)
{
    const double parsed = ParseFiniteDouble(optionName, value);
    if (parsed < 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

bool ParseBoolean(const std::string& optionName, const std::string& value)
{
    if (value == "true" || value == "1")
        return true;
    if (value == "false" || value == "0")
        return false;
    throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected true or false");
}

std::pair<long long, std::string> ParseExperimentConfigPair(const std::string& optionName,
                                                            const std::string& value)
{
    const size_t colon = value.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size())
        throw std::invalid_argument(optionName + " requires EXPERIMENT_ID:key=value[,key=value...]");

    return {
        ParsePositiveLongLong(optionName, value.substr(0, colon)),
        value.substr(colon + 1)
    };
}


SchedulerOptions ParseSchedulerArgs(int argc, const char* argv[])
{
    SchedulerOptions options;
    options.schedulerExecutablePath =
        EA::SchedulerCore::NativeWorkerProcessController()
            .resolveExecutablePath();
    options.semanticWorkerRegistryPath =
        (std::filesystem::current_path() / "Builds" /
         "SemanticWorkers" / "registry.json").string();
    for (int index = 0; index < argc; ++index)
    {
        if (index)
            options.invocationCommandLine.push_back(' ');
        options.invocationCommandLine += argv[index] != nullptr
            ? argv[index]
            : "";
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        std::string value;

        if (TryParseSchedulerWorkerLimitArgument(
                argc,
                argv,
                i,
                options.maxTrainProcs,
                options.maxInferProcs,
                options.maxAnalyzeProcs))
        {
            continue;
        }

        if (arg == "--model-info")
            options.modelInfo = true;
        else if (arg == "--status")
            options.compactStatus = true;
        else if (arg == "--create-economic-calendar-snapshot")
            options.createEconomicCalendarSnapshot = true;
        else if (arg == "--schedule-experiments")
            options.scheduleExperiments = true;
        else if (arg == "--enqueue-experiment")
            options.enqueueExperiment = true;
        else if (arg == "--queue-experiment")
            options.queueExperiment = true;
        else if (arg == "--queue-sweep")
            options.queueSweep = true;
        else if (arg == "--auto-resume")
            options.autoResume = true;
        else if (arg == "--analyze-completed-experiments")
            options.analyzeCompletedExperiments = true;
        else if (arg == "--print-experiment-leaderboard")
            options.printLeaderboard = true;
        else if (arg == "--generate-experiment-reports")
            options.generateExperimentReports = true;
        else if (arg == "--scheduler-status")
            options.schedulerStatus = true;
        else if (arg == "--verify-profitability-evidence")
        {
            if (options.verifyProfitabilityExperimentIds)
                throw std::invalid_argument(
                    "--verify-profitability-evidence specified more than once");
            options.verifyProfitabilityExperimentIds =
                EA::ProfitabilityVerification::ParseDeclaredExperimentIds(
                    RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-profitability-readiness")
        {
            if (options.campaignProfitabilityReadinessSnapshotId)
                throw std::invalid_argument(
                    "--campaign-profitability-readiness specified more than once");
            options.campaignProfitabilityReadinessSnapshotId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--shadow-rank-campaign-profitability")
        {
            if (options.campaignProfitabilityShadowSnapshotId)
                throw std::invalid_argument(
                    "--shadow-rank-campaign-profitability specified more than once");
            options.campaignProfitabilityShadowSnapshotId =
                ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--profitability-shadow-weights")
        {
            if (options.campaignProfitabilityShadowWeights)
                throw std::invalid_argument(
                    "--profitability-shadow-weights specified more than once");
            options.campaignProfitabilityShadowWeights =
                EA::ProfitabilityVerification::ParseProfitabilityShadowWeights(
                    RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--calibrate-campaign-profitability")
        {
            if (options.campaignProfitabilityCalibrationSnapshotId)
                throw std::invalid_argument(
                    "--calibrate-campaign-profitability specified more than once");
            options.campaignProfitabilityCalibrationSnapshotId =
                ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--validate-campaign-profitability-temporal")
        {
            if (options.campaignProfitabilityTemporalValidation)
                throw std::invalid_argument(
                    "--validate-campaign-profitability-temporal specified more than once");
            options.campaignProfitabilityTemporalValidation = true;
        }
        else if (arg ==
                 "--prepare-campaign-profitability-forward-validation")
        {
            if (options.campaignProfitabilityForwardValidationPrecommit)
                throw std::invalid_argument(
                    "--prepare-campaign-profitability-forward-validation specified more than once");
            options.campaignProfitabilityForwardValidationPrecommit =
                ParseCampaignProfitabilityForwardValidationSpec(
                    RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--prepare-campaign-profitability-outcome-jobs")
        {
            if (options.campaignProfitabilityOutcomePreparationCohort)
                throw std::invalid_argument(
                    "--prepare-campaign-profitability-outcome-jobs specified more than once");
            options.campaignProfitabilityOutcomePreparationCohort =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--compare-campaign-profitability-prospective")
        {
            if (options.campaignProfitabilityProspectiveComparisonCohort)
                throw std::invalid_argument(
                    "--compare-campaign-profitability-prospective specified more than once");
            options.campaignProfitabilityProspectiveComparisonCohort =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--complete-scheduler-protocol-cutover")
            options.completeSchedulerProtocolCutover = true;
        else if (arg == "--scheduler-worker-attempt-id")
        {
            if (options.schedulerWorkerAttemptId)
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id specified more than once");
            options.schedulerWorkerAttemptId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--legacy-layout6-infer-worker")
        {
            if (options.legacyLayout6InferWorkerPath)
                throw std::invalid_argument(
                    "--legacy-layout6-infer-worker specified more than once");
            options.legacyLayout6InferWorkerPath =
                EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
                    RequireNextArg(argc, argv, i, arg), arg);
        }
        else if (arg == "--semantic-worker-registry")
        {
            if (options.semanticWorkerRegistryPathSpecified)
                throw std::invalid_argument(
                    "--semantic-worker-registry specified more than once");
            options.semanticWorkerRegistryPath =
                RequireNextArg(argc, argv, i, arg);
            options.semanticWorkerRegistryPathSpecified = true;
        }
        else if (arg == "--backfill-experiment-metadata")
            options.backfillExperimentMetadata = true;
        else if (arg == "--backup-database")
            options.backupDatabase = true;
        else if (arg == "--backup-output")
        {
            options.backupOutputPath = RequireNextArg(argc, argv, i, arg);
            if (options.backupOutputPath->empty())
                throw std::invalid_argument("--backup-output requires a non-empty path");
        }
        else if (arg == "--experiment-metadata")
            options.experimentMetadataId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-models")
            options.listExperimentModelsId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-lineage")
            options.listExperimentLineageId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--include-parent-models")
            options.includeParentModels = true;
        else if (arg == "--model")
            options.modelInfoModelId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--experiment-id")
            options.statusExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--help")
            options.help = true;
        else if (arg == "--dry-run")
            options.dryRun = true;
        else if (arg == "--yes")
            options.yes = true;
        else if (arg == "--force")
            options.force = true;
        else if (arg == "--scheduler-verbose")
            options.schedulerVerbose = true;
        else if (arg == "--lstm-profile-hotspots")
            options.lstmProfileHotspots = true;
        else if (arg == "--auto-generate-reports")
            options.autoGenerateReports = true;
        else if (arg == "--scheduler-once")
            options.schedulerOnce = true;
        else if (arg == "--recover-orphans-only")
            options.recoverOrphansOnly = true;
        else if (arg == "--auto-evaluate-continuations")
            options.autoEvaluateContinuations = true;
        else if (arg == "--auto-queue-continuations")
            options.autoQueueContinuations = true;
        else if (arg == "--continuation-dry-run")
            options.continuationDryRun = true;
        else if (arg == "--continuation-scan-seconds")
        {
            options.continuationScanSeconds =
                ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.continuationScanSecondsSpecified = true;
        }
        else if (arg == "--continuation-max-queues-per-scan")
        {
            options.continuationMaxQueuesPerScan =
                ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.continuationMaxQueuesPerScanSpecified = true;
        }
        else if (arg == "--allow-duplicate-experiment")
            options.allowDuplicateExperiment = true;
        else if (arg == "--analyze-experiment")
            options.analyzeExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--compare-training-objective-pair")
            options.compareTrainingObjectivePair = ParsePositiveIdPair(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--compare-feature-ablation-pair")
            options.compareFeatureAblationPair =
                EA::FeatureAblationPairEvaluation::ParseExperimentIdPair(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--compare-experiment-pair")
            options.compareExperimentPair =
                EA::ExperimentPairComparison::ParseExperimentIdPair(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--compare-experiment-replications")
            options.compareExperimentReplications =
                EA::ExperimentReplicationComparison::ParseExperimentIdPairs(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--plan-experiment-replications")
        {
            if (options.planExperimentReplications)
                throw std::invalid_argument(
                    "--plan-experiment-replications specified more than once");
            options.planExperimentReplications =
                EA::ExperimentPairComparison::ParseExperimentIdPair(
                    RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--replication-seeds")
        {
            if (options.replicationSeeds)
                throw std::invalid_argument(
                    "--replication-seeds specified more than once");
            options.replicationSeeds =
                EA::ExperimentReplicationPlanning::ParseReplicationSeeds(
                    RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--summary")
            options.compareExperimentPairSummary = true;
        else if (arg == "--expected-ablation-mask")
            options.expectedFeatureAblationMask =
                EA::FeatureAblationMask::Parse(
                    RequireNextArg(argc, argv, i, arg)).CanonicalText();
        else if (arg == "--compare-feature-ablation-replications")
            options.compareFeatureAblationReplications =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--corrected-causal-surprise-replication-status")
            options.correctedCausalSurpriseReplicationStatus =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg ==
                 "--materialize-corrected-causal-surprise-replication")
            options.materializeCorrectedCausalSurpriseReplication =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--corrected-causal-surprise-anchor-pair")
            options.correctedCausalSurpriseAnchorPair =
                EA::FeatureAblationPairEvaluation::ParseExperimentIdPair(
                    RequireNextArg(argc, argv, i, arg));
        else if (arg == "--expected-corrected-replication-plan-hash")
            options.expectedCorrectedReplicationPlanHash =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--causal-surprise-observability")
            options.causalSurpriseObservabilityExperimentId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--causal-surprise-observability-scope")
        {
            options.causalSurpriseObservabilityScope =
                EA::CausalSurpriseObservability::ParseScope(
                    RequireNextArg(argc, argv, i, arg));
            options.causalSurpriseObservabilityScopeSpecified = true;
        }
        else if (arg == "--causal-surprise-coverage-gaps")
            options.causalSurpriseCoverageGapsExperimentId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--causal-surprise-coverage-gaps-scope")
        {
            options.causalSurpriseCoverageGapsScope =
                EA::CausalSurpriseObservability::ParseScope(
                    RequireNextArg(argc, argv, i, arg));
            options.causalSurpriseCoverageGapsScopeSpecified = true;
        }
        else if (arg == "--pair-primary-profitability-metric")
            options.pairPrimaryProfitabilityMetric =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--pair-min-profitability-improvement")
            options.pairMinimumProfitabilityImprovement =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-profitability-worsening")
            options.pairMaximumProfitabilityWorsening =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-infer-accuracy-decrease")
            options.pairMaximumInferenceAccuracyDecrease =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-accept-accuracy-decrease")
            options.pairMaximumAcceptAccuracyDecrease =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-accept-rate-decrease")
            options.pairMaximumAcceptRateDecrease =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-leader-score-decrease")
            options.pairMaximumLeaderScoreDecrease =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pair-max-neutral-proportion-increase")
            options.pairMaximumNeutralProportionIncrease =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--stop-experiment")
            options.stopExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--reconcile-worker-attempt")
            options.reconcileWorkerAttemptId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recover-failed-inference")
            options.recoverFailedInferenceExperimentId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--stop-all-experiments")
            options.stopAllExperiments = true;
        else if (arg == "--pause-all-experiments" || arg == "--pause-all")
            options.pauseAllExperiments = true;
        else if (arg == "--resume-all-experiments" || arg == "--resume-all")
            options.resumeAllExperiments = true;
        else if (arg == "--cancel-all-experiments")
            options.cancelAllExperiments = true;
        else if (arg == "--immediate")
            options.cancelImmediate = true;
        else if (arg == "--after-next-checkpoint")
            options.cancelAfterNextCheckpoint = true;
        else if (arg == "--infer-before-cancel")
            options.inferBeforeCancel = true;
        else if (arg == "--pause-experiment")
            options.pauseExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--resume-experiment")
            options.resumeExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--pause-campaign-materialization")
            options.pauseCampaignMaterializationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--resume-campaign-materialization")
            options.resumeCampaignMaterializationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--set-experiment-priority")
            options.setExperimentPriority = ParseExperimentConfigPair(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--cancel-experiment")
            options.cancelExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--retry-failed-experiment")
            options.retryFailedExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--requeue-training")
            options.requeueTrainingExperimentId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--retry-checkpoint-eval")
            options.retryCheckpointEvalId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--evaluate-checkpoint-policy")
            options.evaluateCheckpointPolicyEvalId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-policy-status")
            options.checkpointPolicyStatusEvalId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--enable-continuation-policy")
            options.enableContinuationPolicyExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--disable-continuation-policy")
            options.disableContinuationPolicyExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--set-continuation-policy")
            options.setContinuationPolicy = ParseExperimentConfigPair(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--evaluate-continuation")
            options.evaluateContinuationExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--queue-continuation")
            options.queueContinuationExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--continuation-status")
            options.continuationStatusExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--generate-experiment-recommendations")
            options.generateExperimentRecommendations = true;
        else if (arg == "--list-experiment-recommendations")
            options.listExperimentRecommendations = true;
        else if (arg == "--recommendation-status")
            options.recommendationStatusId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-scans")
            options.listExperimentRecommendationScans = true;
        else if (arg == "--recommendation-scan-status")
            options.recommendationScanStatusId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-policy")
            options.recommendationPolicy = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-symbol")
            options.recommendationSymbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-horizon")
            options.recommendationHorizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-source-experiment")
            options.recommendationSourceExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-max")
            options.recommendationMaximum = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-status-filter")
            options.recommendationStatusFilter = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-scan-id")
            options.recommendationScanId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-limit")
        {
            options.recommendationLimit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.recommendationLimitSpecified = true;
        }
        else if (arg == "--score-experiment-recommendations")
            options.scoreExperimentRecommendations = true;
        else if (arg == "--list-experiment-recommendation-scores")
            options.listExperimentRecommendationScores = true;
        else if (arg == "--recommendation-score-status")
            options.recommendationScoreStatusId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-score-runs")
            options.listExperimentRecommendationScoreRuns = true;
        else if (arg == "--recommendation-score-run-status")
            options.recommendationScoreRunStatusId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--explain-recommendation-score")
            options.explainRecommendationScoreId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-scoring-policy")
            options.recommendationScoringPolicy = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-id")
            options.recommendationIdFilter = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-score-run-id")
            options.recommendationScoreRunId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-score-min")
            options.recommendationScoreMinimum = ParseFiniteDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-score-limit")
        {
            options.recommendationScoreLimit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.recommendationScoreLimitSpecified = true;
        }
        else if (arg == "--approve-experiment-recommendation")
            options.approveRecommendationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--reject-experiment-recommendation")
            options.rejectRecommendationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--expire-experiment-recommendation")
            options.expireRecommendationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-reviews")
            options.listExperimentRecommendationReviews = true;
        else if (arg == "--recommendation-review-status")
            options.recommendationReviewStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-review-history")
            options.recommendationReviewHistoryId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-review-reason-code")
            options.recommendationReviewReasonCode =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-review-reason")
            options.recommendationReviewReason =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-reviewer")
            options.recommendationReviewer = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-review-note")
            options.recommendationReviewNote = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-review-score-id")
            options.recommendationReviewScoreId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-review-action")
            options.recommendationReviewActionFilter =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-review-limit")
        {
            options.recommendationReviewLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.recommendationReviewLimitSpecified = true;
        }
        else if (arg == "--evaluate-experiment-recommendations")
            options.evaluateExperimentRecommendations = true;
        else if (arg == "--evaluate-experiment-recommendation")
            options.evaluateExperimentRecommendationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-evaluations")
            options.listExperimentRecommendationEvaluations = true;
        else if (arg == "--recommendation-evaluation-status")
            options.recommendationEvaluationStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--explain-recommendation-evaluation")
            options.explainRecommendationEvaluationId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-evaluation-runs")
            options.listExperimentRecommendationEvaluationRuns = true;
        else if (arg == "--recommendation-evaluation-run-status")
            options.recommendationEvaluationRunStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-evaluation-policy")
            options.recommendationEvaluationPolicy =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-evaluation-disposition")
            options.recommendationEvaluationDisposition =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-evaluation-limit")
        {
            options.recommendationEvaluationLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.recommendationEvaluationLimitSpecified = true;
        }
        else if (arg == "--recommendation-evaluation-dry-run")
            options.recommendationEvaluationDryRun = true;
        else if (arg == "--rank-experiment-recommendation-evaluations")
            options.rankExperimentRecommendationEvaluations = true;
        else if (arg == "--recommendation-ranking-evaluation-run-id")
            options.recommendationRankingEvaluationRunId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-ranking-scan-id")
            options.recommendationRankingScanId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-ranking-symbol")
            options.recommendationRankingSymbol =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-ranking-horizon")
            options.recommendationRankingHorizon = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-ranking-family")
            options.recommendationRankingFamily =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--recommendation-ranking-global")
            options.recommendationRankingGlobal = true;
        else if (arg == "--recommendation-ranking-limit")
        {
            options.recommendationRankingLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.recommendationRankingLimitSpecified = true;
        }
        else if (arg == "--recommendation-ranking-dry-run")
            options.recommendationRankingDryRun = true;
        else if (arg == "--list-experiment-recommendation-ranking-snapshots")
            options.listExperimentRecommendationRankingSnapshots = true;
        else if (arg == "--recommendation-ranking-status")
            options.recommendationRankingStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-experiment-recommendation-ranking-members")
            options.listRecommendationRankingMembersId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-ranking-member-status")
            options.recommendationRankingMemberStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-ranking-bucket")
            options.recommendationRankingBucket =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--compare-experiment-recommendation-evaluations")
            options.compareRecommendationEvaluations = ParsePositiveIdPair(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--compare-experiment-recommendation-ranking-members")
            options.compareRecommendationRankingMembers = ParsePositiveIdPair(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--approve-conversion-proposal")
            options.approveConversionProposalId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--reject-conversion-proposal")
            options.rejectConversionProposalId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--conversion-proposal-review-request-id")
            options.conversionProposalReviewRequestId =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--conversion-proposal-review-operator")
            options.conversionProposalReviewOperator =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--conversion-proposal-review-reason")
            options.conversionProposalReviewReason =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--show-conversion-proposal")
            options.showConversionProposalId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-conversion-proposal-reviews")
            options.listConversionProposalReviewsId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-conversion-proposals-by-review-status")
            options.listConversionProposalsReviewStatus =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--conversion-proposal-review-limit")
        {
            options.conversionProposalReviewLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.conversionProposalReviewLimitSpecified = true;
        }
        else if (arg == "--execute-approved-conversion-proposal")
            options.executeApprovedConversionProposalId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--conversion-proposal-execution-status")
            options.conversionProposalExecutionStatusId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--activate-recommendation-conversion-execution")
            options.activateRecommendationConversionExecutionId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-conversion-activation-status")
            options.recommendationConversionActivationStatusId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-conversion-workflow")
            options.recommendationConversionWorkflowProposalId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-recommendation-conversion-workflows")
            options.listRecommendationConversionWorkflows = true;
        else if (arg == "--conversion-workflow-state")
        {
            const std::string value = RequireNextArg(argc, argv, i, arg);
            options.conversionWorkflowState = EA::ExperimentRecommendation::
                ParseRecommendationConversionWorkflowState(value);
            if (!options.conversionWorkflowState)
                throw std::invalid_argument(
                    "invalid --conversion-workflow-state value '" + value +
                    "'");
        }
        else if (arg == "--conversion-workflow-limit")
        {
            options.conversionWorkflowLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.conversionWorkflowLimitSpecified = true;
        }
        else if (arg == "--plan-recommendation-campaign")
        {
            options.planRecommendationCampaign = true;
            options.campaignPlanningPolicy.enabled = true;
        }
        else if (arg == "--review-recommendation-campaign")
        {
            options.reviewRecommendationCampaign = true;
            options.campaignPlanningPolicy.enabled = true;
        }
        else if (arg == "--approve-recommendation-campaign")
        {
            options.approveRecommendationCampaign = true;
            options.campaignPlanningPolicy.enabled = true;
        }
        else if (arg == "--reject-recommendation-campaign")
        {
            options.rejectRecommendationCampaign = true;
            options.campaignPlanningPolicy.enabled = true;
        }
        else if (arg == "--campaign-review-identity-hash")
            options.campaignReviewIdentityHash =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--campaign-reviewer")
            options.campaignReviewer = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--campaign-review-reason")
            options.campaignReviewReason = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--show-recommendation-campaign-approval")
            options.showRecommendationCampaignApprovalId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-recommendation-campaign-approvals")
            options.listRecommendationCampaignApprovals = true;
        else if (arg == "--campaign-approval-decision")
        {
            const std::string value = RequireNextArg(argc, argv, i, arg);
            options.campaignApprovalDecision = EA::ExperimentRecommendation::
                ParseRecommendationCampaignApprovalDecision(value);
            if (!options.campaignApprovalDecision)
                throw std::invalid_argument(
                    "invalid --campaign-approval-decision value '" + value +
                    "'");
        }
        else if (arg == "--campaign-approval-limit")
        {
            options.campaignApprovalLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignApprovalLimitSpecified = true;
        }
        else if (arg == "--materialize-recommendation-campaign")
            options.materializeRecommendationCampaign = true;
        else if (arg == "--campaign-approval-id")
            options.campaignMaterializationApprovalId = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--campaign-materialized-by")
            options.campaignMaterializedBy = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--campaign-materialization-reason")
            options.campaignMaterializationReason =
                RequireNextArg(argc, argv, i, arg);
        else if (arg == "--show-recommendation-campaign-materialization")
            options.showRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-recommendation-campaign-materializations")
            options.listRecommendationCampaignMaterializations = true;
        else if (arg == "--campaign-materialization-limit")
        {
            options.campaignMaterializationLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignMaterializationLimitSpecified = true;
        }
        else if (arg == "--show-recommendation-campaign-handoff")
            options.showRecommendationCampaignHandoffId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--list-recommendation-campaign-handoffs")
            options.listRecommendationCampaignHandoffs = true;
        else if (arg == "--campaign-handoff-limit")
        {
            options.campaignHandoffLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignHandoffLimitSpecified = true;
        }
        else if (arg == "--review-recommendation-campaign-materialization")
        {
            if (options.campaignProposalReviewCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --review-recommendation-campaign-materialization");
            options.campaignProposalReviewCommandSpecified = true;
            options.reviewRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-proposal-review-decision")
        {
            if (options.campaignProposalReviewDecisionSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-decision");
            options.campaignProposalReviewDecisionSpecified = true;
            const std::string decision = RequireNextArg(argc, argv, i, arg);
            options.campaignProposalReviewDecision =
                EA::ExperimentRecommendation::
                    ParseRecommendationConversionProposalReviewDecision(
                        decision);
            if (!options.campaignProposalReviewDecision)
                throw std::invalid_argument(
                    "invalid --campaign-proposal-review-decision value '" +
                    decision + "'");
        }
        else if (arg == "--campaign-proposal-review-operator")
        {
            if (options.campaignProposalReviewOperatorSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-operator");
            options.campaignProposalReviewOperatorSpecified = true;
            options.campaignProposalReviewOperator =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--campaign-proposal-review-reason")
        {
            if (options.campaignProposalReviewReasonSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-reason");
            options.campaignProposalReviewReasonSpecified = true;
            options.campaignProposalReviewReason =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--execute-recommendation-campaign-materialization")
        {
            if (options.campaignExecutionCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --execute-recommendation-campaign-materialization");
            options.campaignExecutionCommandSpecified = true;
            options.executeRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--activate-recommendation-campaign-materialization")
        {
            if (options.campaignActivationCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --activate-recommendation-campaign-materialization");
            options.campaignActivationCommandSpecified = true;
            options.activateRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--launch-recommendation-campaign-materialization")
        {
            if (options.campaignLaunchCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --launch-recommendation-campaign-materialization");
            options.campaignLaunchCommandSpecified = true;
            options.launchRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--recommendation-campaign-status")
        {
            if (options.campaignStatusCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --recommendation-campaign-status");
            options.campaignStatusCommandSpecified = true;
            options.recommendationCampaignStatusMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--recommendation-campaign-outcome-assessment")
        {
            if (options.campaignOutcomeAssessmentCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --recommendation-campaign-outcome-assessment");
            options.campaignOutcomeAssessmentCommandSpecified = true;
            options.recommendationCampaignOutcomeAssessmentMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-budget-grant")
        {
            if (options.campaignOperationsBudgetGrantCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-grant");
            options.campaignOperationsBudgetGrantCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-budget-amend")
        {
            if (options.campaignOperationsBudgetAmendCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-amend");
            options.campaignOperationsBudgetAmendCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-budget-revoke")
        {
            if (options.campaignOperationsBudgetRevokeCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-revoke");
            options.campaignOperationsBudgetRevokeCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-budget-supersede")
        {
            if (options.campaignOperationsBudgetSupersedeCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-supersede");
            options.campaignOperationsBudgetSupersedeCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-admit")
        {
            if (options.campaignOperationsAdmitMaterializationId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-admit");
            options.campaignOperationsAdmitMaterializationId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-accept-request")
        {
            if (options.campaignOperationsAcceptRequestCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-accept-request");
            options.campaignOperationsAcceptRequestCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-budget-status")
        {
            if (options.campaignOperationsBudgetStatusCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-status");
            options.campaignOperationsBudgetStatusCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-request-status")
        {
            if (options.campaignOperationsRequestStatusRequestId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-request-status");
            options.campaignOperationsRequestStatusRequestId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-pause")
        {
            if (options.campaignOperationsPauseCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-pause");
            options.campaignOperationsPauseCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-resume")
        {
            if (options.campaignOperationsResumeCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-resume");
            options.campaignOperationsResumeCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-cancel")
        {
            if (options.campaignOperationsCancelCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-cancel");
            options.campaignOperationsCancelCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-control-status")
        {
            if (options.campaignOperationsControlStatusCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-control-status");
            options.campaignOperationsControlStatusCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-complete-if-settled")
        {
            if (options.campaignOperationsCompleteCampaignId)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-complete-if-settled");
            options.campaignOperationsCompleteCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-completion-status")
        {
            if (options.campaignOperationsCompletionStatusCampaignId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-completion-status");
            options.campaignOperationsCompletionStatusCampaignId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-production-readiness")
        {
            if (options.campaignOperationsProductionReadiness)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-production-readiness");
            options.campaignOperationsProductionReadiness = true;
        }
        else if (arg == "--campaign-operations-production-status")
        {
            if (options.campaignOperationsProductionStatus)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-production-status");
            options.campaignOperationsProductionStatus = true;
        }
        else if (arg == "--campaign-operations-production-enable")
        {
            if (options.campaignOperationsProductionEnable)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-production-enable");
            options.campaignOperationsProductionEnable = true;
        }
        else if (arg == "--campaign-operations-production-disable")
        {
            if (options.campaignOperationsProductionDisable)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-production-disable");
            options.campaignOperationsProductionDisable = true;
        }
        else if (arg == "--campaign-operations-dispatch-request")
        {
            if (options.campaignOperationsProductionDispatchRequest)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-dispatch-request");
            options.campaignOperationsProductionDispatchRequest = true;
        }
        else if (arg == "--campaign-operations-manager-run-once")
        {
            if (options.campaignOperationsManagerRunOnceLimit)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-manager-run-once");
            options.campaignOperationsManagerRunOnceLimit = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-reconcile-observe" ||
                 arg == "--campaign-operations-reconcile-recover")
        {
            if (options.campaignOperationsReconcileRunKey)
                throw std::invalid_argument(
                    "duplicate Campaign Operations reconciliation command");
            options.campaignOperationsReconcileRunKey =
                RequireNextArg(argc, argv, i, arg);
            options.campaignOperationsReconcileRecover =
                arg == "--campaign-operations-reconcile-recover";
        }
        else if (
            arg == "--campaign-operations-expected-control-version")
        {
            if (options.campaignOperationsExpectedControlVersion)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-expected-control-version");
            const long long value = ParseSignedLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
            if (value < 0 || value > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "invalid "
                    "--campaign-operations-expected-control-version");
            options.campaignOperationsExpectedControlVersion =
                static_cast<int>(value);
        }
        else if (arg == "--campaign-operations-request-id")
        {
            if (options.campaignOperationsControlRequestId)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-request-id");
            options.campaignOperationsControlRequestId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (
            arg == "--campaign-operations-expected-request-version")
        {
            if (options.campaignOperationsExpectedRequestVersion)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-expected-request-version");
            const long long value = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
            if (value > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "invalid "
                    "--campaign-operations-expected-request-version");
            options.campaignOperationsExpectedRequestVersion =
                static_cast<int>(value);
        }
        else if (
            arg == "--campaign-operations-expected-production-version")
        {
            if (options.campaignOperationsExpectedProductionVersion)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-expected-production-version");
            const long long value = ParseSignedLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
            if (value < 0 || value > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "invalid "
                    "--campaign-operations-expected-production-version");
            options.campaignOperationsExpectedProductionVersion =
                static_cast<int>(value);
        }
        else if (arg == "--campaign-operations-operation-key")
        {
            if (options.campaignOperationsOperationKey)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-operation-key");
            options.campaignOperationsOperationKey =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg ==
                 "--campaign-operations-independent-verification-reference")
        {
            if (options.campaignOperationsIndependentVerificationReference)
                throw std::invalid_argument(
                    "duplicate independent verification reference");
            options.campaignOperationsIndependentVerificationReference =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (
            arg == "--campaign-operations-reconcile-after-request-id")
        {
            if (options.campaignOperationsReconcileAfterRequestId)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-reconcile-after-request-id");
            options.campaignOperationsReconcileAfterRequestId =
                ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-reconcile-limit")
        {
            if (options.campaignOperationsReconcileLimit)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-reconcile-limit");
            options.campaignOperationsReconcileLimit =
                ParsePositiveInt(
                    arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-expected-budget-version")
        {
            if (options.campaignOperationsExpectedBudgetVersion)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-expected-budget-version");
            const long long version = ParseSignedLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
            if (version < 0 ||
                version > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "invalid --campaign-operations-expected-budget-version");
            options.campaignOperationsExpectedBudgetVersion =
                static_cast<int>(version);
        }
        else if (arg == "--campaign-operations-budget-value")
        {
            if (options.campaignOperationsBudgetValue)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-budget-value");
            options.campaignOperationsBudgetValue = ParseSignedLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
        }
        else if (arg == "--campaign-operations-actor")
        {
            if (options.campaignOperationsActor)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-actor");
            options.campaignOperationsActor =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--campaign-operations-reason")
        {
            if (options.campaignOperationsReason)
                throw std::invalid_argument(
                    "duplicate --campaign-operations-reason");
            options.campaignOperationsReason =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (
            arg == "--campaign-operations-reservation-expires-at")
        {
            if (options.campaignOperationsReservationExpiresAt)
                throw std::invalid_argument(
                    "duplicate "
                    "--campaign-operations-reservation-expires-at");
            options.campaignOperationsReservationExpiresAt =
                RequireNextArg(argc, argv, i, arg);
        }
        else if (arg == "--campaign-ranking-snapshot")
        {
            options.campaignPlanningScope.rankingSnapshotId =
                ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-limit")
        {
            options.campaignPlanningPolicy.maximumSelectedRecommendations =
                ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-candidate-limit")
        {
            options.campaignPlanningPolicy.maximumCandidatesConsidered =
                ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-symbol")
        {
            options.campaignPlanningScope.symbol = EA::CanonicalSymbol::Normalize(
                RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-horizon")
        {
            options.campaignPlanningScope.horizon = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-donchian20-arms")
        {
            if (options.campaignDonchian20ArmsSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-donchian20-arms");
            options.campaignPlanningPolicy.donchian20Arms =
                EA::ExperimentRecommendation::
                    ParseRecommendationCampaignDonchian20Arms(
                        RequireNextArg(argc, argv, i, arg));
            options.campaignDonchian20ArmsSpecified = true;
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-min-leader-score")
        {
            options.campaignPlanningPolicy.minimumLeaderScore =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-min-inference-accuracy")
        {
            options.campaignPlanningPolicy.minimumInferenceAccuracy =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-max-neutral-proportion")
        {
            options.campaignPlanningPolicy.maximumPredictedNeutralProportion =
                ParseNonNegativeFiniteDouble(
                    arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-min-profitability")
        {
            options.campaignPlanningPolicy.minimumProfitability =
                ParseFiniteDouble(arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-max-per-symbol")
        {
            options.campaignPlanningPolicy.maximumPerSymbol = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-max-per-horizon")
        {
            options.campaignPlanningPolicy.maximumPerHorizon = ParsePositiveInt(
                arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-max-per-source-experiment")
        {
            options.campaignPlanningPolicy.maximumPerSourceExperiment =
                ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-reconsider-rejected")
        {
            options.campaignPlanningPolicy.reconsiderRejectedWorkflows = true;
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-reconsider-failed")
        {
            options.campaignPlanningPolicy.reconsiderFailedWorkflows = true;
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--campaign-reconsider-cancelled")
        {
            options.campaignPlanningPolicy.reconsiderCancelledWorkflows = true;
            options.campaignPolicyOptionSpecified = true;
        }
        else if (arg == "--requeue-analysis")
            options.requeueAnalysisExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--requeue-inference")
            options.requeueInferenceExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--stop-after-checkpoint")
            options.stopAfterCheckpoint = ParseExperimentEpochPair(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--clear-stop-after-checkpoint")
            options.clearStopAfterCheckpointExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--stop-after-checkpoint-all")
            options.stopAfterCheckpointAllEpoch = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--clear-stop-after-checkpoint-all")
            options.clearStopAfterCheckpointAll = true;
        else if (arg == "--enable-checkpoint-infer")
            options.enableCheckpointInferExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--disable-checkpoint-infer")
            options.disableCheckpointInferExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-infer")
            options.queueCheckpointInfer = true;
        else if (arg == "--checkpoint-infer-min-epoch")
        {
            const std::string next = RequireNextArg(argc, argv, i, arg);
            if (next.find(':') == std::string::npos)
                options.queueCheckpointInferMinEpoch = ParsePositiveInt(arg, next);
            else
                options.checkpointInferMinEpoch = ParseExperimentEpochPair(arg, next);
        }
        else if (arg == "--checkpoint-infer-interval")
        {
            const std::string next = RequireNextArg(argc, argv, i, arg);
            if (next.find(':') == std::string::npos)
                options.queueCheckpointInferInterval = ParsePositiveInt(arg, next);
            else
                options.checkpointInferInterval = ParseExperimentEpochPair(arg, next);
        }
        else if (arg == "--checkpoint-policy")
            options.queueCheckpointPolicy = true;
        else if (arg == "--continuation-candidate-excluded")
            options.queueContinuationCandidateExcluded = true;
        else if (arg == "--checkpoint-policy-min-leader-score")
            options.checkpointPolicyMinLeaderScore = ParsePositiveFiniteDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-policy-min-infer-accuracy")
            options.checkpointPolicyMinInferAccuracy = ParsePositiveFiniteDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-policy-top-n")
            options.checkpointPolicyTopN = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-policy-scope")
            options.checkpointPolicyScope = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--checkpoint-policy-stop-mode")
            options.checkpointPolicyStopMode = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--checkpoint-policy-grace-evals")
            options.checkpointPolicyGraceEvals = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--enable-checkpoint-policy")
            options.enableCheckpointPolicyExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--disable-checkpoint-policy")
            options.disableCheckpointPolicyExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--set-checkpoint-policy")
            options.setCheckpointPolicy = ParseExperimentConfigPair(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--symbol")
            options.symbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--prediction-horizon")
            options.predictionHorizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--c-next-threshold")
            options.cNextThreshold = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--threshold")
            options.cNextThreshold = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--core-lr-mult")
            options.coreLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--core-lr")
            options.coreLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--head-lr-mult")
            options.headLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--head-lr")
            options.headLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--training-objective")
        {
            const std::string objective = RequireNextArg(argc, argv, i, arg);
            try
            {
                options.trainingObjective =
                    EA::TrainingObjective::ParseCliSelection(objective);
            }
            catch (const std::invalid_argument&)
            {
                throw std::invalid_argument(
                    "unsupported --training-objective: " + objective);
            }
            options.trainingObjectiveSpecified = true;
        }
        else if (arg == "--target-epochs")
            options.targetEpochs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--epochs")
            options.epochs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-interval")
            options.checkpointInterval = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--donchian20-mode")
            options.donchian20Mode = ParseDonchian20Mode(
                RequireNextArg(argc, argv, i, arg));
        else if (arg == "--feature-warmup-scope")
        {
            options.featureWarmupScope = EA::ParseFeatureWarmupScope(
                RequireNextArg(argc, argv, i, arg));
            options.featureWarmupScopeSpecified = true;
        }
        else if (arg == "--donchian-lookback")
        {
            options.donchianLookback = ParseDonchianLookback(
                RequireNextArg(argc, argv, i, arg));
            options.donchianLookbackSpecified = true;
        }
        else if (arg == "--ablate-features")
        {
            options.featureAblationMask = EA::FeatureAblationMask::Parse(
                RequireNextArg(argc, argv, i, arg)).CanonicalText();
            options.featureAblationMaskSpecified = true;
        }
        else if (arg == "--fresh-initialization-seed")
        {
            const long long seed = ParsePositiveLongLong(
                arg, RequireNextArg(argc, argv, i, arg));
            if (seed > std::numeric_limits<unsigned int>::max())
                throw std::invalid_argument("--fresh-initialization-seed exceeds uint32 range");
            options.freshInitializationSeed = static_cast<unsigned int>(seed);
        }
        else if (arg == "--train-start")
            options.trainStart = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--train-end")
            options.trainEnd = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--infer-start")
            options.inferStart = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--infer-end")
            options.inferEnd = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--resume-model-id")
            options.resumeModelId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--resume-expand-input-width")
        {
            if (options.resumeExpandInputWidth)
                throw std::invalid_argument(
                    "--resume-expand-input-width specified more than once");
            options.resumeExpandInputWidth = true;
        }
        else if (arg == "--scheduler-poll-seconds")
            options.schedulerPollSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--scheduler-log-dir")
            options.schedulerLogDir = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--experiment-report-dir")
            options.experimentReportDir = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--leaderboard-symbol")
            options.leaderboardSymbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--leaderboard-horizon")
            options.leaderboardHorizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--leaderboard-limit")
            options.leaderboardLimit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--log-level")
            options.logLevel = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--lstm-profile-output")
            options.lstmProfileOutputPath = RequireNextArg(argc, argv, i, arg);
        else if (SplitOptionWithValue(arg, "--symbol", value))
            options.symbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--prediction-horizon", value))
            options.predictionHorizon = ParsePositiveInt("--prediction-horizon", value);
        else if (SplitOptionWithValue(arg, "--c-next-threshold", value))
            options.cNextThreshold = ParsePositiveDouble("--c-next-threshold", value);
        else if (SplitOptionWithValue(arg, "--threshold", value))
            options.cNextThreshold = ParsePositiveDouble("--threshold", value);
        else if (SplitOptionWithValue(arg, "--core-lr-mult", value))
            options.coreLrMult = ParsePositiveDouble("--core-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--core-lr", value))
            options.coreLrMult = ParsePositiveDouble("--core-lr", value);
        else if (SplitOptionWithValue(arg, "--head-lr-mult", value))
            options.headLrMult = ParsePositiveDouble("--head-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--head-lr", value))
            options.headLrMult = ParsePositiveDouble("--head-lr", value);
        else if (SplitOptionWithValue(arg, "--training-objective", value))
        {
            try
            {
                options.trainingObjective =
                    EA::TrainingObjective::ParseCliSelection(value);
            }
            catch (const std::invalid_argument&)
            {
                throw std::invalid_argument(
                    "unsupported --training-objective: " + value);
            }
            options.trainingObjectiveSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--target-epochs", value))
            options.targetEpochs = ParsePositiveInt("--target-epochs", value);
        else if (SplitOptionWithValue(arg, "--epochs", value))
            options.epochs = ParsePositiveInt("--epochs", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-interval", value))
            options.checkpointInterval = ParsePositiveInt("--checkpoint-interval", value);
        else if (SplitOptionWithValue(arg, "--donchian20-mode", value))
            options.donchian20Mode = ParseDonchian20Mode(value);
        else if (SplitOptionWithValue(arg, "--feature-warmup-scope", value))
        {
            options.featureWarmupScope = EA::ParseFeatureWarmupScope(value);
            options.featureWarmupScopeSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--donchian-lookback", value))
        {
            options.donchianLookback = ParseDonchianLookback(value);
            options.donchianLookbackSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--ablate-features", value))
        {
            options.featureAblationMask = EA::FeatureAblationMask::Parse(value).CanonicalText();
            options.featureAblationMaskSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--fresh-initialization-seed", value))
        {
            const long long seed = ParsePositiveLongLong("--fresh-initialization-seed", value);
            if (seed > std::numeric_limits<unsigned int>::max())
                throw std::invalid_argument("--fresh-initialization-seed exceeds uint32 range");
            options.freshInitializationSeed = static_cast<unsigned int>(seed);
        }
        else if (SplitOptionWithValue(arg, "--train-start", value))
            options.trainStart = value;
        else if (SplitOptionWithValue(arg, "--train-end", value))
            options.trainEnd = value;
        else if (SplitOptionWithValue(arg, "--infer-start", value))
            options.inferStart = value;
        else if (SplitOptionWithValue(arg, "--infer-end", value))
            options.inferEnd = value;
        else if (SplitOptionWithValue(arg, "--resume-model-id", value))
            options.resumeModelId = ParsePositiveLongLong("--resume-model-id", value);
        else if (SplitOptionWithValue(arg, "--scheduler-poll-seconds", value))
            options.schedulerPollSeconds = ParsePositiveInt("--scheduler-poll-seconds", value);
        else if (SplitOptionWithValue(arg, "--continuation-scan-seconds", value))
        {
            options.continuationScanSeconds = ParsePositiveInt("--continuation-scan-seconds", value);
            options.continuationScanSecondsSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--continuation-max-queues-per-scan", value))
        {
            options.continuationMaxQueuesPerScan =
                ParsePositiveInt("--continuation-max-queues-per-scan", value);
            options.continuationMaxQueuesPerScanSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--scheduler-log-dir", value))
            options.schedulerLogDir = value;
        else if (SplitOptionWithValue(
                     arg, "--scheduler-worker-attempt-id", value))
        {
            if (options.schedulerWorkerAttemptId)
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id specified more than once");
            options.schedulerWorkerAttemptId =
                ParsePositiveLongLong(
                    "--scheduler-worker-attempt-id", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--legacy-layout6-infer-worker", value))
        {
            if (options.legacyLayout6InferWorkerPath)
                throw std::invalid_argument(
                    "--legacy-layout6-infer-worker specified more than once");
            options.legacyLayout6InferWorkerPath =
                EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
                    value, "--legacy-layout6-infer-worker");
        }
        else if (SplitOptionWithValue(
                     arg, "--semantic-worker-registry", value))
        {
            if (options.semanticWorkerRegistryPathSpecified)
                throw std::invalid_argument(
                    "--semantic-worker-registry specified more than once");
            if (value.empty())
                throw std::invalid_argument(
                    "--semantic-worker-registry requires a path");
            options.semanticWorkerRegistryPath = value;
            options.semanticWorkerRegistryPathSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--experiment-report-dir", value))
            options.experimentReportDir = value;
        else if (SplitOptionWithValue(arg, "--analyze-experiment", value))
            options.analyzeExperimentId = ParsePositiveLongLong("--analyze-experiment", value);
        else if (SplitOptionWithValue(arg, "--stop-experiment", value))
            options.stopExperimentId = ParsePositiveLongLong("--stop-experiment", value);
        else if (SplitOptionWithValue(arg, "--reconcile-worker-attempt", value))
            options.reconcileWorkerAttemptId = ParsePositiveLongLong(
                "--reconcile-worker-attempt", value);
        else if (SplitOptionWithValue(
                     arg, "--recover-failed-inference", value))
            options.recoverFailedInferenceExperimentId =
                ParsePositiveLongLong("--recover-failed-inference", value);
        else if (SplitOptionWithValue(arg, "--pause-experiment", value))
            options.pauseExperimentId = ParsePositiveLongLong("--pause-experiment", value);
        else if (SplitOptionWithValue(arg, "--resume-experiment", value))
            options.resumeExperimentId = ParsePositiveLongLong("--resume-experiment", value);
        else if (SplitOptionWithValue(
                     arg, "--pause-campaign-materialization", value))
            options.pauseCampaignMaterializationId = ParsePositiveLongLong(
                "--pause-campaign-materialization", value);
        else if (SplitOptionWithValue(
                     arg, "--resume-campaign-materialization", value))
            options.resumeCampaignMaterializationId = ParsePositiveLongLong(
                "--resume-campaign-materialization", value);
        else if (SplitOptionWithValue(arg, "--set-experiment-priority", value))
            options.setExperimentPriority = ParseExperimentConfigPair(
                "--set-experiment-priority", value);
        else if (SplitOptionWithValue(arg, "--cancel-experiment", value))
            options.cancelExperimentId = ParsePositiveLongLong("--cancel-experiment", value);
        else if (SplitOptionWithValue(arg, "--retry-failed-experiment", value))
            options.retryFailedExperimentId = ParsePositiveLongLong("--retry-failed-experiment", value);
        else if (SplitOptionWithValue(arg, "--requeue-training", value))
            options.requeueTrainingExperimentId = ParsePositiveLongLong(
                "--requeue-training", value);
        else if (SplitOptionWithValue(arg, "--retry-checkpoint-eval", value))
            options.retryCheckpointEvalId = ParsePositiveLongLong("--retry-checkpoint-eval", value);
        else if (SplitOptionWithValue(arg, "--evaluate-checkpoint-policy", value))
            options.evaluateCheckpointPolicyEvalId = ParsePositiveLongLong("--evaluate-checkpoint-policy", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-status", value))
            options.checkpointPolicyStatusEvalId = ParsePositiveLongLong("--checkpoint-policy-status", value);
        else if (SplitOptionWithValue(arg, "--enable-continuation-policy", value))
            options.enableContinuationPolicyExperimentId = ParsePositiveLongLong("--enable-continuation-policy", value);
        else if (SplitOptionWithValue(arg, "--disable-continuation-policy", value))
            options.disableContinuationPolicyExperimentId = ParsePositiveLongLong("--disable-continuation-policy", value);
        else if (SplitOptionWithValue(arg, "--set-continuation-policy", value))
            options.setContinuationPolicy = ParseExperimentConfigPair("--set-continuation-policy", value);
        else if (SplitOptionWithValue(arg, "--evaluate-continuation", value))
            options.evaluateContinuationExperimentId = ParsePositiveLongLong("--evaluate-continuation", value);
        else if (SplitOptionWithValue(arg, "--queue-continuation", value))
            options.queueContinuationExperimentId = ParsePositiveLongLong("--queue-continuation", value);
        else if (SplitOptionWithValue(arg, "--continuation-status", value))
            options.continuationStatusExperimentId = ParsePositiveLongLong("--continuation-status", value);
        else if (SplitOptionWithValue(arg, "--recommendation-status", value))
            options.recommendationStatusId = ParsePositiveLongLong("--recommendation-status", value);
        else if (SplitOptionWithValue(arg, "--recommendation-scan-status", value))
            options.recommendationScanStatusId = ParsePositiveLongLong("--recommendation-scan-status", value);
        else if (SplitOptionWithValue(arg, "--recommendation-policy", value))
            options.recommendationPolicy = value;
        else if (SplitOptionWithValue(arg, "--recommendation-symbol", value))
            options.recommendationSymbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--recommendation-horizon", value))
            options.recommendationHorizon = ParsePositiveInt("--recommendation-horizon", value);
        else if (SplitOptionWithValue(arg, "--recommendation-source-experiment", value))
            options.recommendationSourceExperimentId = ParsePositiveLongLong("--recommendation-source-experiment", value);
        else if (SplitOptionWithValue(arg, "--recommendation-max", value))
            options.recommendationMaximum = ParsePositiveInt("--recommendation-max", value);
        else if (SplitOptionWithValue(arg, "--recommendation-status-filter", value))
            options.recommendationStatusFilter = value;
        else if (SplitOptionWithValue(arg, "--recommendation-scan-id", value))
            options.recommendationScanId = ParsePositiveLongLong("--recommendation-scan-id", value);
        else if (SplitOptionWithValue(arg, "--recommendation-limit", value))
        {
            options.recommendationLimit = ParsePositiveInt("--recommendation-limit", value);
            options.recommendationLimitSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--recommendation-score-status", value))
            options.recommendationScoreStatusId = ParsePositiveLongLong("--recommendation-score-status", value);
        else if (SplitOptionWithValue(arg, "--recommendation-score-run-status", value))
            options.recommendationScoreRunStatusId = ParsePositiveLongLong("--recommendation-score-run-status", value);
        else if (SplitOptionWithValue(arg, "--explain-recommendation-score", value))
            options.explainRecommendationScoreId = ParsePositiveLongLong("--explain-recommendation-score", value);
        else if (SplitOptionWithValue(arg, "--recommendation-scoring-policy", value))
            options.recommendationScoringPolicy = value;
        else if (SplitOptionWithValue(arg, "--recommendation-id", value))
            options.recommendationIdFilter = ParsePositiveLongLong("--recommendation-id", value);
        else if (SplitOptionWithValue(arg, "--recommendation-score-run-id", value))
            options.recommendationScoreRunId = ParsePositiveLongLong("--recommendation-score-run-id", value);
        else if (SplitOptionWithValue(arg, "--recommendation-score-min", value))
            options.recommendationScoreMinimum = ParseFiniteDouble("--recommendation-score-min", value);
        else if (SplitOptionWithValue(arg, "--recommendation-score-limit", value))
        {
            options.recommendationScoreLimit = ParsePositiveInt("--recommendation-score-limit", value);
            options.recommendationScoreLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--approve-experiment-recommendation", value))
            options.approveRecommendationId = ParsePositiveLongLong(
                "--approve-experiment-recommendation", value);
        else if (SplitOptionWithValue(
                     arg, "--reject-experiment-recommendation", value))
            options.rejectRecommendationId = ParsePositiveLongLong(
                "--reject-experiment-recommendation", value);
        else if (SplitOptionWithValue(
                     arg, "--expire-experiment-recommendation", value))
            options.expireRecommendationId = ParsePositiveLongLong(
                "--expire-experiment-recommendation", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-status", value))
            options.recommendationReviewStatusId = ParsePositiveLongLong(
                "--recommendation-review-status", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-history", value))
            options.recommendationReviewHistoryId = ParsePositiveLongLong(
                "--recommendation-review-history", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-reason-code", value))
            options.recommendationReviewReasonCode = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-reason", value))
            options.recommendationReviewReason = value;
        else if (SplitOptionWithValue(arg, "--recommendation-reviewer", value))
            options.recommendationReviewer = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-note", value))
            options.recommendationReviewNote = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-score-id", value))
            options.recommendationReviewScoreId = ParsePositiveLongLong(
                "--recommendation-review-score-id", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-action", value))
            options.recommendationReviewActionFilter = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-review-limit", value))
        {
            options.recommendationReviewLimit = ParsePositiveInt(
                "--recommendation-review-limit", value);
            options.recommendationReviewLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--evaluate-experiment-recommendation", value))
            options.evaluateExperimentRecommendationId = ParsePositiveLongLong(
                "--evaluate-experiment-recommendation", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-evaluation-status", value))
            options.recommendationEvaluationStatusId = ParsePositiveLongLong(
                "--recommendation-evaluation-status", value);
        else if (SplitOptionWithValue(
                     arg, "--explain-recommendation-evaluation", value))
            options.explainRecommendationEvaluationId = ParsePositiveLongLong(
                "--explain-recommendation-evaluation", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-evaluation-run-status", value))
            options.recommendationEvaluationRunStatusId = ParsePositiveLongLong(
                "--recommendation-evaluation-run-status", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-evaluation-policy", value))
            options.recommendationEvaluationPolicy = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-evaluation-disposition", value))
            options.recommendationEvaluationDisposition = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-evaluation-limit", value))
        {
            options.recommendationEvaluationLimit = ParsePositiveInt(
                "--recommendation-evaluation-limit", value);
            options.recommendationEvaluationLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-evaluation-run-id", value))
            options.recommendationRankingEvaluationRunId = ParsePositiveLongLong(
                "--recommendation-ranking-evaluation-run-id", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-scan-id", value))
            options.recommendationRankingScanId = ParsePositiveLongLong(
                "--recommendation-ranking-scan-id", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-symbol", value))
            options.recommendationRankingSymbol = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-horizon", value))
            options.recommendationRankingHorizon = ParsePositiveInt(
                "--recommendation-ranking-horizon", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-family", value))
            options.recommendationRankingFamily = value;
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-limit", value))
        {
            options.recommendationRankingLimit = ParsePositiveInt(
                "--recommendation-ranking-limit", value);
            options.recommendationRankingLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-status", value))
            options.recommendationRankingStatusId = ParsePositiveLongLong(
                "--recommendation-ranking-status", value);
        else if (SplitOptionWithValue(
                     arg, "--list-experiment-recommendation-ranking-members", value))
            options.listRecommendationRankingMembersId = ParsePositiveLongLong(
                "--list-experiment-recommendation-ranking-members", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-member-status", value))
            options.recommendationRankingMemberStatusId = ParsePositiveLongLong(
                "--recommendation-ranking-member-status", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-ranking-bucket", value))
            options.recommendationRankingBucket = value;
        else if (SplitOptionWithValue(
                     arg, "--compare-experiment-recommendation-evaluations", value))
            options.compareRecommendationEvaluations = ParsePositiveIdPair(
                "--compare-experiment-recommendation-evaluations", value);
        else if (SplitOptionWithValue(
                     arg, "--compare-experiment-recommendation-ranking-members", value))
            options.compareRecommendationRankingMembers = ParsePositiveIdPair(
                "--compare-experiment-recommendation-ranking-members", value);
        else if (SplitOptionWithValue(
                     arg, "--approve-conversion-proposal", value))
            options.approveConversionProposalId = ParsePositiveLongLong(
                "--approve-conversion-proposal", value);
        else if (SplitOptionWithValue(
                     arg, "--reject-conversion-proposal", value))
            options.rejectConversionProposalId = ParsePositiveLongLong(
                "--reject-conversion-proposal", value);
        else if (SplitOptionWithValue(
                     arg, "--conversion-proposal-review-request-id", value))
            options.conversionProposalReviewRequestId = value;
        else if (SplitOptionWithValue(
                     arg, "--conversion-proposal-review-operator", value))
            options.conversionProposalReviewOperator = value;
        else if (SplitOptionWithValue(
                     arg, "--conversion-proposal-review-reason", value))
            options.conversionProposalReviewReason = value;
        else if (SplitOptionWithValue(
                     arg, "--show-conversion-proposal", value))
            options.showConversionProposalId = ParsePositiveLongLong(
                "--show-conversion-proposal", value);
        else if (SplitOptionWithValue(
                     arg, "--list-conversion-proposal-reviews", value))
            options.listConversionProposalReviewsId = ParsePositiveLongLong(
                "--list-conversion-proposal-reviews", value);
        else if (SplitOptionWithValue(
                     arg, "--list-conversion-proposals-by-review-status", value))
            options.listConversionProposalsReviewStatus = value;
        else if (SplitOptionWithValue(
                     arg, "--conversion-proposal-review-limit", value))
        {
            options.conversionProposalReviewLimit = ParsePositiveInt(
                "--conversion-proposal-review-limit", value);
            options.conversionProposalReviewLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--execute-approved-conversion-proposal", value))
            options.executeApprovedConversionProposalId = ParsePositiveLongLong(
                "--execute-approved-conversion-proposal", value);
        else if (SplitOptionWithValue(
                     arg, "--conversion-proposal-execution-status", value))
            options.conversionProposalExecutionStatusId = ParsePositiveLongLong(
                "--conversion-proposal-execution-status", value);
        else if (SplitOptionWithValue(
                     arg,
                     "--activate-recommendation-conversion-execution",
                     value))
            options.activateRecommendationConversionExecutionId =
                ParsePositiveLongLong(
                    "--activate-recommendation-conversion-execution", value);
        else if (SplitOptionWithValue(
                     arg,
                     "--recommendation-conversion-activation-status",
                     value))
            options.recommendationConversionActivationStatusId =
                ParsePositiveLongLong(
                    "--recommendation-conversion-activation-status", value);
        else if (SplitOptionWithValue(
                     arg, "--recommendation-conversion-workflow", value))
            options.recommendationConversionWorkflowProposalId =
                ParsePositiveLongLong(
                    "--recommendation-conversion-workflow", value);
        else if (SplitOptionWithValue(
                     arg, "--conversion-workflow-state", value))
        {
            options.conversionWorkflowState = EA::ExperimentRecommendation::
                ParseRecommendationConversionWorkflowState(value);
            if (!options.conversionWorkflowState)
                throw std::invalid_argument(
                    "invalid --conversion-workflow-state value '" + value +
                    "'");
        }
        else if (SplitOptionWithValue(
                     arg, "--conversion-workflow-limit", value))
        {
            options.conversionWorkflowLimit = ParsePositiveInt(
                "--conversion-workflow-limit", value);
            options.conversionWorkflowLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-ranking-snapshot", value))
        {
            options.campaignPlanningScope.rankingSnapshotId =
                ParsePositiveLongLong("--campaign-ranking-snapshot", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--campaign-limit", value))
        {
            options.campaignPlanningPolicy.maximumSelectedRecommendations =
                ParsePositiveInt("--campaign-limit", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-candidate-limit", value))
        {
            options.campaignPlanningPolicy.maximumCandidatesConsidered =
                ParsePositiveInt("--campaign-candidate-limit", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--campaign-symbol", value))
        {
            options.campaignPlanningScope.symbol =
                EA::CanonicalSymbol::Normalize(value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--campaign-horizon", value))
        {
            options.campaignPlanningScope.horizon =
                ParsePositiveInt("--campaign-horizon", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-donchian20-arms", value))
        {
            if (options.campaignDonchian20ArmsSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-donchian20-arms");
            options.campaignPlanningPolicy.donchian20Arms =
                EA::ExperimentRecommendation::
                    ParseRecommendationCampaignDonchian20Arms(value);
            options.campaignDonchian20ArmsSpecified = true;
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-min-leader-score", value))
        {
            options.campaignPlanningPolicy.minimumLeaderScore =
                ParseNonNegativeFiniteDouble(
                    "--campaign-min-leader-score", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-min-inference-accuracy", value))
        {
            options.campaignPlanningPolicy.minimumInferenceAccuracy =
                ParseNonNegativeFiniteDouble(
                    "--campaign-min-inference-accuracy", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-max-neutral-proportion", value))
        {
            options.campaignPlanningPolicy.maximumPredictedNeutralProportion =
                ParseNonNegativeFiniteDouble(
                    "--campaign-max-neutral-proportion", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-min-profitability", value))
        {
            options.campaignPlanningPolicy.minimumProfitability =
                ParseFiniteDouble("--campaign-min-profitability", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-max-per-symbol", value))
        {
            options.campaignPlanningPolicy.maximumPerSymbol =
                ParsePositiveInt("--campaign-max-per-symbol", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-max-per-horizon", value))
        {
            options.campaignPlanningPolicy.maximumPerHorizon =
                ParsePositiveInt("--campaign-max-per-horizon", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-max-per-source-experiment", value))
        {
            options.campaignPlanningPolicy.maximumPerSourceExperiment =
                ParsePositiveInt(
                    "--campaign-max-per-source-experiment", value);
            options.campaignPolicyOptionSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-review-identity-hash", value))
            options.campaignReviewIdentityHash = value;
        else if (SplitOptionWithValue(arg, "--campaign-reviewer", value))
            options.campaignReviewer = value;
        else if (SplitOptionWithValue(arg, "--campaign-review-reason", value))
            options.campaignReviewReason = value;
        else if (SplitOptionWithValue(
                     arg, "--show-recommendation-campaign-approval", value))
            options.showRecommendationCampaignApprovalId =
                ParsePositiveLongLong(
                    "--show-recommendation-campaign-approval", value);
        else if (SplitOptionWithValue(
                     arg, "--campaign-approval-decision", value))
        {
            options.campaignApprovalDecision = EA::ExperimentRecommendation::
                ParseRecommendationCampaignApprovalDecision(value);
            if (!options.campaignApprovalDecision)
                throw std::invalid_argument(
                    "invalid --campaign-approval-decision value '" + value +
                    "'");
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-approval-limit", value))
        {
            options.campaignApprovalLimit = ParsePositiveInt(
                "--campaign-approval-limit", value);
            options.campaignApprovalLimitSpecified = true;
        }
        else if (SplitOptionWithValue(arg, "--campaign-approval-id", value))
            options.campaignMaterializationApprovalId = ParsePositiveLongLong(
                "--campaign-approval-id", value);
        else if (SplitOptionWithValue(arg, "--campaign-materialized-by", value))
            options.campaignMaterializedBy = value;
        else if (SplitOptionWithValue(
                     arg, "--campaign-materialization-reason", value))
            options.campaignMaterializationReason = value;
        else if (SplitOptionWithValue(
                     arg, "--show-recommendation-campaign-materialization", value))
            options.showRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    "--show-recommendation-campaign-materialization", value);
        else if (SplitOptionWithValue(
                     arg, "--campaign-materialization-limit", value))
        {
            options.campaignMaterializationLimit = ParsePositiveInt(
                "--campaign-materialization-limit", value);
            options.campaignMaterializationLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--show-recommendation-campaign-handoff", value))
            options.showRecommendationCampaignHandoffId =
                ParsePositiveLongLong(
                    "--show-recommendation-campaign-handoff", value);
        else if (SplitOptionWithValue(
                     arg, "--campaign-handoff-limit", value))
        {
            options.campaignHandoffLimit = ParsePositiveInt(
                "--campaign-handoff-limit", value);
            options.campaignHandoffLimitSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg,
                     "--review-recommendation-campaign-materialization",
                     value))
        {
            if (options.campaignProposalReviewCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --review-recommendation-campaign-materialization");
            options.campaignProposalReviewCommandSpecified = true;
            options.reviewRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    "--review-recommendation-campaign-materialization", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-proposal-review-decision", value))
        {
            if (options.campaignProposalReviewDecisionSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-decision");
            options.campaignProposalReviewDecisionSpecified = true;
            options.campaignProposalReviewDecision =
                EA::ExperimentRecommendation::
                    ParseRecommendationConversionProposalReviewDecision(value);
            if (!options.campaignProposalReviewDecision)
                throw std::invalid_argument(
                    "invalid --campaign-proposal-review-decision value '" +
                    value + "'");
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-proposal-review-operator", value))
        {
            if (options.campaignProposalReviewOperatorSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-operator");
            options.campaignProposalReviewOperatorSpecified = true;
            options.campaignProposalReviewOperator = value;
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-proposal-review-reason", value))
        {
            if (options.campaignProposalReviewReasonSpecified)
                throw std::invalid_argument(
                    "duplicate --campaign-proposal-review-reason");
            options.campaignProposalReviewReasonSpecified = true;
            options.campaignProposalReviewReason = value;
        }
        else if (SplitOptionWithValue(
                     arg,
                     "--execute-recommendation-campaign-materialization",
                     value))
        {
            if (options.campaignExecutionCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --execute-recommendation-campaign-materialization");
            options.campaignExecutionCommandSpecified = true;
            options.executeRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    "--execute-recommendation-campaign-materialization", value);
        }
        else if (SplitOptionWithValue(
                     arg,
                     "--activate-recommendation-campaign-materialization",
                     value))
        {
            if (options.campaignActivationCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --activate-recommendation-campaign-materialization");
            options.campaignActivationCommandSpecified = true;
            options.activateRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    "--activate-recommendation-campaign-materialization", value);
        }
        else if (SplitOptionWithValue(
                     arg,
                     "--launch-recommendation-campaign-materialization",
                     value))
        {
            if (options.campaignLaunchCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --launch-recommendation-campaign-materialization");
            options.campaignLaunchCommandSpecified = true;
            options.launchRecommendationCampaignMaterializationId =
                ParsePositiveLongLong(
                    "--launch-recommendation-campaign-materialization", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--recommendation-campaign-status", value))
        {
            if (options.campaignStatusCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --recommendation-campaign-status");
            options.campaignStatusCommandSpecified = true;
            options.recommendationCampaignStatusMaterializationId =
                ParsePositiveLongLong("--recommendation-campaign-status", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--recommendation-campaign-outcome-assessment", value))
        {
            if (options.campaignOutcomeAssessmentCommandSpecified)
                throw std::invalid_argument(
                    "duplicate --recommendation-campaign-outcome-assessment");
            options.campaignOutcomeAssessmentCommandSpecified = true;
            options.recommendationCampaignOutcomeAssessmentMaterializationId =
                ParsePositiveLongLong(
                    "--recommendation-campaign-outcome-assessment", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--compare-training-objective-pair", value))
            options.compareTrainingObjectivePair = ParsePositiveIdPair(
                "--compare-training-objective-pair", value);
        else if (SplitOptionWithValue(
                     arg, "--compare-feature-ablation-pair", value))
            options.compareFeatureAblationPair =
                EA::FeatureAblationPairEvaluation::ParseExperimentIdPair(value);
        else if (SplitOptionWithValue(
                     arg, "--compare-experiment-pair", value))
            options.compareExperimentPair =
                EA::ExperimentPairComparison::ParseExperimentIdPair(value);
        else if (SplitOptionWithValue(
                     arg, "--compare-experiment-replications", value))
            options.compareExperimentReplications =
                EA::ExperimentReplicationComparison::ParseExperimentIdPairs(
                    value);
        else if (SplitOptionWithValue(
                     arg, "--plan-experiment-replications", value))
        {
            if (options.planExperimentReplications)
                throw std::invalid_argument(
                    "--plan-experiment-replications specified more than once");
            options.planExperimentReplications =
                EA::ExperimentPairComparison::ParseExperimentIdPair(value);
        }
        else if (SplitOptionWithValue(arg, "--replication-seeds", value))
        {
            if (options.replicationSeeds)
                throw std::invalid_argument(
                    "--replication-seeds specified more than once");
            options.replicationSeeds =
                EA::ExperimentReplicationPlanning::ParseReplicationSeeds(value);
        }
        else if (SplitOptionWithValue(
                     arg, "--expected-ablation-mask", value))
            options.expectedFeatureAblationMask =
                EA::FeatureAblationMask::Parse(value).CanonicalText();
        else if (SplitOptionWithValue(
                     arg, "--compare-feature-ablation-replications", value))
            options.compareFeatureAblationReplications =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    value);
        else if (SplitOptionWithValue(
                     arg,
                     "--corrected-causal-surprise-replication-status",
                     value))
            options.correctedCausalSurpriseReplicationStatus =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    value);
        else if (SplitOptionWithValue(
                     arg,
                     "--materialize-corrected-causal-surprise-replication",
                     value))
            options.materializeCorrectedCausalSurpriseReplication =
                EA::FeatureAblationReplicationEvaluation::ParseExperimentIdPairs(
                    value);
        else if (SplitOptionWithValue(
                     arg, "--corrected-causal-surprise-anchor-pair", value))
            options.correctedCausalSurpriseAnchorPair =
                EA::FeatureAblationPairEvaluation::ParseExperimentIdPair(value);
        else if (SplitOptionWithValue(
                     arg,
                     "--expected-corrected-replication-plan-hash",
                     value))
            options.expectedCorrectedReplicationPlanHash = value;
        else if (SplitOptionWithValue(
                     arg, "--causal-surprise-observability", value))
            options.causalSurpriseObservabilityExperimentId =
                ParsePositiveLongLong(
                    "--causal-surprise-observability", value);
        else if (SplitOptionWithValue(
                     arg, "--causal-surprise-observability-scope", value))
        {
            options.causalSurpriseObservabilityScope =
                EA::CausalSurpriseObservability::ParseScope(value);
            options.causalSurpriseObservabilityScopeSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--causal-surprise-coverage-gaps", value))
            options.causalSurpriseCoverageGapsExperimentId =
                ParsePositiveLongLong(
                    "--causal-surprise-coverage-gaps", value);
        else if (SplitOptionWithValue(
                     arg, "--causal-surprise-coverage-gaps-scope", value))
        {
            options.causalSurpriseCoverageGapsScope =
                EA::CausalSurpriseObservability::ParseScope(value);
            options.causalSurpriseCoverageGapsScopeSpecified = true;
        }
        else if (SplitOptionWithValue(
                     arg, "--pair-primary-profitability-metric", value))
            options.pairPrimaryProfitabilityMetric = value;
        else if (SplitOptionWithValue(
                     arg, "--pair-min-profitability-improvement", value))
            options.pairMinimumProfitabilityImprovement =
                ParseNonNegativeFiniteDouble(
                    "--pair-min-profitability-improvement", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-profitability-worsening", value))
            options.pairMaximumProfitabilityWorsening =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-profitability-worsening", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-infer-accuracy-decrease", value))
            options.pairMaximumInferenceAccuracyDecrease =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-infer-accuracy-decrease", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-accept-accuracy-decrease", value))
            options.pairMaximumAcceptAccuracyDecrease =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-accept-accuracy-decrease", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-accept-rate-decrease", value))
            options.pairMaximumAcceptRateDecrease =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-accept-rate-decrease", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-leader-score-decrease", value))
            options.pairMaximumLeaderScoreDecrease =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-leader-score-decrease", value);
        else if (SplitOptionWithValue(
                     arg, "--pair-max-neutral-proportion-increase", value))
            options.pairMaximumNeutralProportionIncrease =
                ParseNonNegativeFiniteDouble(
                    "--pair-max-neutral-proportion-increase", value);
        else if (SplitOptionWithValue(arg, "--requeue-analysis", value))
            options.requeueAnalysisExperimentId = ParsePositiveLongLong("--requeue-analysis", value);
        else if (SplitOptionWithValue(arg, "--requeue-inference", value))
            options.requeueInferenceExperimentId = ParsePositiveLongLong("--requeue-inference", value);
        else if (SplitOptionWithValue(arg, "--stop-after-checkpoint", value))
            options.stopAfterCheckpoint = ParseExperimentEpochPair("--stop-after-checkpoint", value);
        else if (SplitOptionWithValue(arg, "--clear-stop-after-checkpoint", value))
            options.clearStopAfterCheckpointExperimentId = ParsePositiveLongLong("--clear-stop-after-checkpoint", value);
        else if (SplitOptionWithValue(arg, "--stop-after-checkpoint-all", value))
            options.stopAfterCheckpointAllEpoch = ParsePositiveInt("--stop-after-checkpoint-all", value);
        else if (SplitOptionWithValue(arg, "--enable-checkpoint-infer", value))
            options.enableCheckpointInferExperimentId = ParsePositiveLongLong("--enable-checkpoint-infer", value);
        else if (SplitOptionWithValue(arg, "--disable-checkpoint-infer", value))
            options.disableCheckpointInferExperimentId = ParsePositiveLongLong("--disable-checkpoint-infer", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-infer", value))
        {
            if (value != "1" && value != "true")
                throw std::invalid_argument("--checkpoint-infer does not accept a value; use --checkpoint-infer");
            options.queueCheckpointInfer = true;
        }
        else if (SplitOptionWithValue(arg, "--checkpoint-infer-min-epoch", value))
        {
            if (value.find(':') == std::string::npos)
                options.queueCheckpointInferMinEpoch = ParsePositiveInt("--checkpoint-infer-min-epoch", value);
            else
                options.checkpointInferMinEpoch = ParseExperimentEpochPair("--checkpoint-infer-min-epoch", value);
        }
        else if (SplitOptionWithValue(arg, "--checkpoint-infer-interval", value))
        {
            if (value.find(':') == std::string::npos)
                options.queueCheckpointInferInterval = ParsePositiveInt("--checkpoint-infer-interval", value);
            else
                options.checkpointInferInterval = ParseExperimentEpochPair("--checkpoint-infer-interval", value);
        }
        else if (SplitOptionWithValue(arg, "--checkpoint-policy", value))
        {
            if (value != "1" && value != "true")
                throw std::invalid_argument("--checkpoint-policy does not accept a value; use --checkpoint-policy");
            options.queueCheckpointPolicy = true;
        }
        else if (SplitOptionWithValue(arg, "--continuation-candidate-excluded", value))
            options.queueContinuationCandidateExcluded = ParseBoolean("--continuation-candidate-excluded", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-min-leader-score", value))
            options.checkpointPolicyMinLeaderScore = ParsePositiveFiniteDouble("--checkpoint-policy-min-leader-score", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-min-infer-accuracy", value))
            options.checkpointPolicyMinInferAccuracy = ParsePositiveFiniteDouble("--checkpoint-policy-min-infer-accuracy", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-top-n", value))
            options.checkpointPolicyTopN = ParsePositiveInt("--checkpoint-policy-top-n", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-scope", value))
            options.checkpointPolicyScope = value;
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-stop-mode", value))
            options.checkpointPolicyStopMode = value;
        else if (SplitOptionWithValue(arg, "--checkpoint-policy-grace-evals", value))
            options.checkpointPolicyGraceEvals = ParsePositiveInt("--checkpoint-policy-grace-evals", value);
        else if (SplitOptionWithValue(arg, "--enable-checkpoint-policy", value))
            options.enableCheckpointPolicyExperimentId = ParsePositiveLongLong("--enable-checkpoint-policy", value);
        else if (SplitOptionWithValue(arg, "--disable-checkpoint-policy", value))
            options.disableCheckpointPolicyExperimentId = ParsePositiveLongLong("--disable-checkpoint-policy", value);
        else if (SplitOptionWithValue(arg, "--set-checkpoint-policy", value))
            options.setCheckpointPolicy = ParseExperimentConfigPair("--set-checkpoint-policy", value);
        else if (SplitOptionWithValue(arg, "--leaderboard-symbol", value))
            options.leaderboardSymbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--leaderboard-horizon", value))
            options.leaderboardHorizon = ParsePositiveInt("--leaderboard-horizon", value);
        else if (SplitOptionWithValue(arg, "--leaderboard-limit", value))
            options.leaderboardLimit = ParsePositiveInt("--leaderboard-limit", value);
        else if (SplitOptionWithValue(arg, "--log-level", value))
            options.logLevel = value;
        else if (SplitOptionWithValue(arg, "--lstm-profile-output", value))
        {
            if (value.empty())
                throw std::invalid_argument("--lstm-profile-output requires a non-empty path");
            options.lstmProfileOutputPath = value;
        }
        else if (SplitOptionWithValue(arg, "--backup-output", value))
        {
            if (value.empty())
                throw std::invalid_argument("--backup-output requires a non-empty path");
            options.backupOutputPath = value;
        }
        else if (SplitOptionWithValue(arg, "--experiment-metadata", value))
            options.experimentMetadataId = ParsePositiveLongLong("--experiment-metadata", value);
        else if (SplitOptionWithValue(arg, "--list-experiment-models", value))
            options.listExperimentModelsId = ParsePositiveLongLong("--list-experiment-models", value);
        else if (SplitOptionWithValue(arg, "--list-experiment-lineage", value))
            options.listExperimentLineageId = ParsePositiveLongLong("--list-experiment-lineage", value);
        else if (SplitOptionWithValue(arg, "--model", value))
            options.modelInfoModelId = ParsePositiveLongLong("--model", value);
        else if (SplitOptionWithValue(arg, "--experiment-id", value))
            options.statusExperimentId = ParsePositiveLongLong("--experiment-id", value);
        else if (SplitOptionWithValue(
                     arg, "--verify-profitability-evidence", value))
        {
            if (options.verifyProfitabilityExperimentIds)
                throw std::invalid_argument(
                    "--verify-profitability-evidence specified more than once");
            options.verifyProfitabilityExperimentIds =
                EA::ProfitabilityVerification::ParseDeclaredExperimentIds(
                    value);
        }
        else if (SplitOptionWithValue(
                     arg, "--campaign-profitability-readiness", value))
        {
            if (options.campaignProfitabilityReadinessSnapshotId)
                throw std::invalid_argument(
                    "--campaign-profitability-readiness specified more than once");
            options.campaignProfitabilityReadinessSnapshotId =
                ParsePositiveLongLong(
                    "--campaign-profitability-readiness", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--shadow-rank-campaign-profitability", value))
        {
            if (options.campaignProfitabilityShadowSnapshotId)
                throw std::invalid_argument(
                    "--shadow-rank-campaign-profitability specified more than once");
            options.campaignProfitabilityShadowSnapshotId =
                ParsePositiveLongLong(
                    "--shadow-rank-campaign-profitability", value);
        }
        else if (SplitOptionWithValue(
                     arg, "--profitability-shadow-weights", value))
        {
            if (options.campaignProfitabilityShadowWeights)
                throw std::invalid_argument(
                    "--profitability-shadow-weights specified more than once");
            options.campaignProfitabilityShadowWeights =
                EA::ProfitabilityVerification::ParseProfitabilityShadowWeights(
                    value);
        }
        else if (SplitOptionWithValue(
                     arg, "--calibrate-campaign-profitability", value))
        {
            if (options.campaignProfitabilityCalibrationSnapshotId)
                throw std::invalid_argument(
                    "--calibrate-campaign-profitability specified more than once");
            options.campaignProfitabilityCalibrationSnapshotId =
                ParsePositiveLongLong(
                    "--calibrate-campaign-profitability", value);
        }
        else if (SplitOptionWithValue(
                     arg,
                     "--prepare-campaign-profitability-forward-validation",
                     value))
        {
            if (options.campaignProfitabilityForwardValidationPrecommit)
                throw std::invalid_argument(
                    "--prepare-campaign-profitability-forward-validation specified more than once");
            options.campaignProfitabilityForwardValidationPrecommit =
                ParseCampaignProfitabilityForwardValidationSpec(value);
        }
        else if (SplitOptionWithValue(
                     arg, "--prepare-campaign-profitability-outcome-jobs", value))
        {
            if (options.campaignProfitabilityOutcomePreparationCohort)
                throw std::invalid_argument(
                    "--prepare-campaign-profitability-outcome-jobs specified more than once");
            options.campaignProfitabilityOutcomePreparationCohort = value;
        }
        else if (SplitOptionWithValue(
                     arg, "--compare-campaign-profitability-prospective", value))
        {
            if (options.campaignProfitabilityProspectiveComparisonCohort)
                throw std::invalid_argument(
                    "--compare-campaign-profitability-prospective specified more than once");
            options.campaignProfitabilityProspectiveComparisonCohort = value;
        }
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown scheduler option '" + arg + "'");
        else
            throw std::invalid_argument("unexpected positional scheduler argument '" + arg + "'");
    }

    if (options.autoQueueContinuations)
        options.autoEvaluateContinuations = true;

    if (options.setExperimentPriority)
    {
        const std::string& priority = options.setExperimentPriority->second;
        if (priority != "high" && priority != "normal" && priority != "low")
            throw std::invalid_argument(
                "--set-experiment-priority requires high, normal, or low");
    }

    const int commandCount =
        (options.modelInfo ? 1 : 0) +
        (options.compactStatus ? 1 : 0) +
        (options.createEconomicCalendarSnapshot ? 1 : 0) +
        (options.scheduleExperiments ? 1 : 0) +
        (options.enqueueExperiment ? 1 : 0) +
        (options.queueExperiment ? 1 : 0) +
        (options.queueSweep ? 1 : 0) +
        (options.analyzeCompletedExperiments ? 1 : 0) +
        (options.analyzeExperimentId.has_value() ? 1 : 0) +
        (options.printLeaderboard ? 1 : 0) +
        (options.generateExperimentReports ? 1 : 0) +
        (options.schedulerStatus ? 1 : 0) +
        (options.completeSchedulerProtocolCutover ? 1 : 0) +
        (options.backfillExperimentMetadata ? 1 : 0) +
        (options.backupDatabase ? 1 : 0) +
        (options.experimentMetadataId.has_value() ? 1 : 0) +
        (options.listExperimentModelsId.has_value() ? 1 : 0) +
        (options.listExperimentLineageId.has_value() ? 1 : 0) +
        (options.stopExperimentId.has_value() ? 1 : 0) +
        (options.reconcileWorkerAttemptId.has_value() ? 1 : 0) +
        (options.recoverFailedInferenceExperimentId.has_value() ? 1 : 0) +
        (options.stopAllExperiments ? 1 : 0) +
        (options.pauseAllExperiments ? 1 : 0) +
        (options.resumeAllExperiments ? 1 : 0) +
        (options.cancelAllExperiments ? 1 : 0) +
        (options.pauseExperimentId.has_value() ? 1 : 0) +
        (options.resumeExperimentId.has_value() ? 1 : 0) +
        (options.pauseCampaignMaterializationId.has_value() ? 1 : 0) +
        (options.resumeCampaignMaterializationId.has_value() ? 1 : 0) +
        (options.setExperimentPriority.has_value() ? 1 : 0) +
        (options.cancelExperimentId.has_value() ? 1 : 0) +
        (options.retryFailedExperimentId.has_value() ? 1 : 0) +
        (options.requeueTrainingExperimentId.has_value() ? 1 : 0) +
        (options.retryCheckpointEvalId.has_value() ? 1 : 0) +
        (options.evaluateCheckpointPolicyEvalId.has_value() ? 1 : 0) +
        (options.checkpointPolicyStatusEvalId.has_value() ? 1 : 0) +
        (options.enableContinuationPolicyExperimentId.has_value() ? 1 : 0) +
        (options.disableContinuationPolicyExperimentId.has_value() ? 1 : 0) +
        (options.setContinuationPolicy.has_value() ? 1 : 0) +
        (options.evaluateContinuationExperimentId.has_value() ? 1 : 0) +
        (options.queueContinuationExperimentId.has_value() ? 1 : 0) +
        (options.continuationStatusExperimentId.has_value() ? 1 : 0) +
        (options.generateExperimentRecommendations ? 1 : 0) +
        (options.listExperimentRecommendations ? 1 : 0) +
        (options.recommendationStatusId.has_value() ? 1 : 0) +
        (options.listExperimentRecommendationScans ? 1 : 0) +
        (options.recommendationScanStatusId.has_value() ? 1 : 0) +
        (options.scoreExperimentRecommendations ? 1 : 0) +
        (options.listExperimentRecommendationScores ? 1 : 0) +
        (options.recommendationScoreStatusId.has_value() ? 1 : 0) +
        (options.listExperimentRecommendationScoreRuns ? 1 : 0) +
        (options.recommendationScoreRunStatusId.has_value() ? 1 : 0) +
        (options.explainRecommendationScoreId.has_value() ? 1 : 0) +
        (options.approveRecommendationId.has_value() ? 1 : 0) +
        (options.rejectRecommendationId.has_value() ? 1 : 0) +
        (options.expireRecommendationId.has_value() ? 1 : 0) +
        (options.listExperimentRecommendationReviews ? 1 : 0) +
        (options.recommendationReviewStatusId.has_value() ? 1 : 0) +
        (options.recommendationReviewHistoryId.has_value() ? 1 : 0) +
        (options.evaluateExperimentRecommendations ? 1 : 0) +
        (options.evaluateExperimentRecommendationId.has_value() ? 1 : 0) +
        (options.listExperimentRecommendationEvaluations ? 1 : 0) +
        (options.recommendationEvaluationStatusId.has_value() ? 1 : 0) +
        (options.explainRecommendationEvaluationId.has_value() ? 1 : 0) +
        (options.listExperimentRecommendationEvaluationRuns ? 1 : 0) +
        (options.recommendationEvaluationRunStatusId.has_value() ? 1 : 0) +
        (options.rankExperimentRecommendationEvaluations ? 1 : 0) +
        (options.listExperimentRecommendationRankingSnapshots ? 1 : 0) +
        (options.recommendationRankingStatusId.has_value() ? 1 : 0) +
        (options.listRecommendationRankingMembersId.has_value() ? 1 : 0) +
        (options.recommendationRankingMemberStatusId.has_value() ? 1 : 0) +
        (options.compareRecommendationEvaluations.has_value() ? 1 : 0) +
        (options.compareRecommendationRankingMembers.has_value() ? 1 : 0) +
        (options.compareTrainingObjectivePair.has_value() ? 1 : 0) +
        (options.compareFeatureAblationPair.has_value() ? 1 : 0) +
        (options.compareExperimentPair.has_value() ? 1 : 0) +
        (options.compareExperimentReplications.has_value() ? 1 : 0) +
        (options.planExperimentReplications.has_value() ? 1 : 0) +
        (options.compareFeatureAblationReplications.has_value() ? 1 : 0) +
        (options.correctedCausalSurpriseReplicationStatus.has_value()
             ? 1 : 0) +
        (options.materializeCorrectedCausalSurpriseReplication.has_value()
             ? 1 : 0) +
        (options.causalSurpriseObservabilityExperimentId.has_value()
             ? 1 : 0) +
        (options.causalSurpriseCoverageGapsExperimentId.has_value()
             ? 1 : 0) +
        (options.verifyProfitabilityExperimentIds.has_value() ? 1 : 0) +
        (options.campaignProfitabilityReadinessSnapshotId.has_value() ? 1 : 0) +
        (options.campaignProfitabilityShadowSnapshotId.has_value() ? 1 : 0) +
        (options.campaignProfitabilityCalibrationSnapshotId.has_value() ? 1 : 0) +
        (options.campaignProfitabilityTemporalValidation ? 1 : 0) +
        (options.campaignProfitabilityForwardValidationPrecommit.has_value()
             ? 1 : 0) +
        (options.campaignProfitabilityOutcomePreparationCohort.has_value()
             ? 1 : 0) +
        (options.campaignProfitabilityProspectiveComparisonCohort.has_value()
             ? 1 : 0) +
        (options.approveConversionProposalId.has_value() ? 1 : 0) +
        (options.rejectConversionProposalId.has_value() ? 1 : 0) +
        (options.showConversionProposalId.has_value() ? 1 : 0) +
        (options.listConversionProposalReviewsId.has_value() ? 1 : 0) +
        (options.listConversionProposalsReviewStatus.has_value() ? 1 : 0) +
        (options.executeApprovedConversionProposalId.has_value() ? 1 : 0) +
        (options.conversionProposalExecutionStatusId.has_value() ? 1 : 0) +
        (options.activateRecommendationConversionExecutionId.has_value() ? 1 : 0) +
        (options.recommendationConversionActivationStatusId.has_value() ? 1 : 0) +
        (options.recommendationConversionWorkflowProposalId.has_value() ? 1 : 0) +
        (options.listRecommendationConversionWorkflows ? 1 : 0) +
        (options.planRecommendationCampaign ? 1 : 0) +
        (options.reviewRecommendationCampaign ? 1 : 0) +
        (options.approveRecommendationCampaign ? 1 : 0) +
        (options.rejectRecommendationCampaign ? 1 : 0) +
        (options.showRecommendationCampaignApprovalId.has_value() ? 1 : 0) +
        (options.listRecommendationCampaignApprovals ? 1 : 0) +
        (options.materializeRecommendationCampaign ? 1 : 0) +
        (options.showRecommendationCampaignMaterializationId.has_value() ? 1 : 0) +
        (options.listRecommendationCampaignMaterializations ? 1 : 0) +
        (options.showRecommendationCampaignHandoffId.has_value() ? 1 : 0) +
        (options.listRecommendationCampaignHandoffs ? 1 : 0) +
        (options.reviewRecommendationCampaignMaterializationId.has_value()
             ? 1
             : 0) +
        (options.executeRecommendationCampaignMaterializationId.has_value()
             ? 1
             : 0) +
        (options.activateRecommendationCampaignMaterializationId.has_value()
             ? 1
             : 0) +
        (options.launchRecommendationCampaignMaterializationId.has_value()
             ? 1
             : 0) +
        (options.recommendationCampaignStatusMaterializationId.has_value()
             ? 1
             : 0) +
        (options.recommendationCampaignOutcomeAssessmentMaterializationId
                 .has_value()
             ? 1
             : 0) +
        (options.campaignOperationsBudgetGrantCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsBudgetAmendCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsBudgetRevokeCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsBudgetSupersedeCampaignId.has_value()
             ? 1
             : 0) +
        (options.campaignOperationsAdmitMaterializationId.has_value()
             ? 1
             : 0) +
        (options.campaignOperationsAcceptRequestCampaignId.has_value()
             ? 1
             : 0) +
        (options.campaignOperationsBudgetStatusCampaignId.has_value()
             ? 1
             : 0) +
        (options.campaignOperationsRequestStatusRequestId.has_value()
             ? 1
             : 0) +
        (options.campaignOperationsPauseCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsResumeCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsCancelCampaignId.has_value() ? 1 : 0) +
        (options.campaignOperationsControlStatusCampaignId.has_value()
             ? 1 : 0) +
        (options.campaignOperationsCompleteCampaignId.has_value()
             ? 1 : 0) +
        (options.campaignOperationsCompletionStatusCampaignId.has_value()
             ? 1 : 0) +
        (options.campaignOperationsProductionReadiness ? 1 : 0) +
        (options.campaignOperationsProductionStatus ? 1 : 0) +
        (options.campaignOperationsProductionEnable ? 1 : 0) +
        (options.campaignOperationsProductionDisable ? 1 : 0) +
        (options.campaignOperationsProductionDispatchRequest ? 1 : 0) +
        (options.campaignOperationsManagerRunOnceLimit.has_value() ? 1 : 0) +
        (options.campaignOperationsReconcileRunKey.has_value() ? 1 : 0) +
        (options.requeueAnalysisExperimentId.has_value() ? 1 : 0) +
        (options.requeueInferenceExperimentId.has_value() ? 1 : 0) +
        (options.stopAfterCheckpoint.has_value() ? 1 : 0) +
        (options.clearStopAfterCheckpointExperimentId.has_value() ? 1 : 0) +
        (options.stopAfterCheckpointAllEpoch.has_value() ? 1 : 0) +
        (options.clearStopAfterCheckpointAll ? 1 : 0) +
        (options.enableCheckpointInferExperimentId.has_value() ? 1 : 0) +
        (options.disableCheckpointInferExperimentId.has_value() ? 1 : 0) +
        (options.checkpointInferMinEpoch.has_value() ? 1 : 0) +
        (options.checkpointInferInterval.has_value() ? 1 : 0) +
        (options.enableCheckpointPolicyExperimentId.has_value() ? 1 : 0) +
        (options.disableCheckpointPolicyExperimentId.has_value() ? 1 : 0) +
        (options.setCheckpointPolicy.has_value() ? 1 : 0) +
        (options.help ? 1 : 0);
    const bool pairPolicyOption =
        options.pairPrimaryProfitabilityMetric.has_value() ||
        options.pairMinimumProfitabilityImprovement.has_value() ||
        options.pairMaximumProfitabilityWorsening.has_value() ||
        options.pairMaximumInferenceAccuracyDecrease.has_value() ||
        options.pairMaximumAcceptAccuracyDecrease.has_value() ||
        options.pairMaximumAcceptRateDecrease.has_value() ||
        options.pairMaximumLeaderScoreDecrease.has_value() ||
        options.pairMaximumNeutralProportionIncrease.has_value();
    if (pairPolicyOption && !options.compareTrainingObjectivePair)
        throw std::invalid_argument(
            "pair materiality options require "
            "--compare-training-objective-pair");
    if (options.compareExperimentPairSummary &&
        !options.compareExperimentPair)
        throw std::invalid_argument(
            "--summary requires --compare-experiment-pair");
    if (options.planExperimentReplications.has_value() !=
        options.replicationSeeds.has_value())
        throw std::invalid_argument(
            "--plan-experiment-replications and --replication-seeds "
            "must be specified together");
    if (options.expectedFeatureAblationMask &&
        !options.compareFeatureAblationPair &&
        !options.compareFeatureAblationReplications)
        throw std::invalid_argument(
            "--expected-ablation-mask requires "
            "--compare-feature-ablation-pair or "
            "--compare-feature-ablation-replications");
    if (options.expectedFeatureAblationMask &&
        options.expectedFeatureAblationMask->empty())
        throw std::invalid_argument(
            "--expected-ablation-mask must not be empty");
    const bool correctedReplicationCommand =
        options.correctedCausalSurpriseReplicationStatus.has_value() ||
        options.materializeCorrectedCausalSurpriseReplication.has_value();
    if (correctedReplicationCommand !=
        options.correctedCausalSurpriseAnchorPair.has_value())
        throw std::invalid_argument(
            "corrected causal-surprise replication commands require "
            "--corrected-causal-surprise-anchor-pair");
    if (options.expectedCorrectedReplicationPlanHash &&
        !options.materializeCorrectedCausalSurpriseReplication)
        throw std::invalid_argument(
            "--expected-corrected-replication-plan-hash requires "
            "--materialize-corrected-causal-surprise-replication");
    if (options.materializeCorrectedCausalSurpriseReplication &&
        !options.expectedCorrectedReplicationPlanHash)
        throw std::invalid_argument(
            "materialization requires "
            "--expected-corrected-replication-plan-hash");
    if (options.causalSurpriseObservabilityScopeSpecified &&
        !options.causalSurpriseObservabilityExperimentId)
        throw std::invalid_argument(
            "--causal-surprise-observability-scope requires "
            "--causal-surprise-observability");
    if (options.causalSurpriseCoverageGapsScopeSpecified &&
        !options.causalSurpriseCoverageGapsExperimentId)
        throw std::invalid_argument(
            "--causal-surprise-coverage-gaps-scope requires "
            "--causal-surprise-coverage-gaps");
    if (options.compareTrainingObjectivePair &&
        (!options.pairPrimaryProfitabilityMetric ||
         !options.pairMinimumProfitabilityImprovement ||
         !options.pairMaximumProfitabilityWorsening ||
         !options.pairMaximumInferenceAccuracyDecrease ||
         !options.pairMaximumAcceptAccuracyDecrease ||
         !options.pairMaximumAcceptRateDecrease ||
         !options.pairMaximumLeaderScoreDecrease ||
         !options.pairMaximumNeutralProportionIncrease))
        throw std::invalid_argument(
            "--compare-training-objective-pair requires the explicit primary "
            "metric and all seven pair materiality thresholds");
    if (options.pairPrimaryProfitabilityMetric &&
        *options.pairPrimaryProfitabilityMetric != "aggregate" &&
        *options.pairPrimaryProfitabilityMetric != "average")
        throw std::invalid_argument(
            "--pair-primary-profitability-metric must be aggregate or average");
    if (options.includeParentModels && !options.listExperimentLineageId.has_value())
        throw std::invalid_argument("--include-parent-models is only valid with --list-experiment-lineage=ID");
    if (options.backupOutputPath.has_value() && !options.backupDatabase)
        throw std::invalid_argument("--backup-output is only valid with --backup-database");
    if (options.queueContinuationCandidateExcluded && !options.queueExperiment)
        throw std::invalid_argument("--continuation-candidate-excluded is only valid with --queue-experiment");
    const bool recommendationGenerationOption =
        options.recommendationPolicy.has_value() ||
        options.recommendationSourceExperimentId.has_value() ||
        options.recommendationMaximum.has_value();
    if (recommendationGenerationOption &&
        !options.generateExperimentRecommendations)
        throw std::invalid_argument("recommendation policy, source, and maximum options require --generate-experiment-recommendations");
    const bool recommendationListOption =
        options.recommendationStatusFilter.has_value() ||
        options.recommendationScanId.has_value();
    if (recommendationListOption && !options.listExperimentRecommendations &&
        !options.scoreExperimentRecommendations &&
        !options.evaluateExperimentRecommendations &&
        !options.evaluateExperimentRecommendationId &&
        !options.listExperimentRecommendationEvaluations)
        throw std::invalid_argument("recommendation status and scan filters require recommendation listing, scoring, or evaluation");
    if (options.recommendationStatusFilter &&
        options.scoreExperimentRecommendations &&
        *options.recommendationStatusFilter != "proposed")
        throw std::invalid_argument("recommendation scoring supports only proposed status");
    if (options.recommendationStatusFilter &&
        (options.evaluateExperimentRecommendations ||
         options.evaluateExperimentRecommendationId))
        throw std::invalid_argument(
            "recommendation evaluation supports proposed status implicitly");
    const bool recommendationSourceFilter =
        options.recommendationSymbol.has_value() ||
        options.recommendationHorizon.has_value();
    if (recommendationSourceFilter &&
        !options.generateExperimentRecommendations &&
        !options.listExperimentRecommendations &&
        !options.scoreExperimentRecommendations &&
        !options.listExperimentRecommendationScores)
        throw std::invalid_argument("recommendation symbol and horizon filters require a recommendation generation, scoring, or list command");
    if (options.recommendationLimitSpecified &&
        !options.listExperimentRecommendations &&
        !options.listExperimentRecommendationScans)
        throw std::invalid_argument("--recommendation-limit requires a recommendation or scan list command");
    if (options.recommendationStatusFilter &&
        !EA::ExperimentRecommendation::ParseRecommendationStatus(
            *options.recommendationStatusFilter))
        throw std::invalid_argument("invalid recommendation status filter");
    if (options.recommendationScoringPolicy &&
        !options.scoreExperimentRecommendations)
        throw std::invalid_argument("--recommendation-scoring-policy requires --score-experiment-recommendations");
    if (options.recommendationIdFilter &&
        !options.scoreExperimentRecommendations &&
        !options.listExperimentRecommendationScores &&
        !options.listExperimentRecommendationReviews &&
        !options.listExperimentRecommendationEvaluations)
        throw std::invalid_argument(
            "--recommendation-id requires recommendation scoring, score listing, review listing, or evaluation listing");
    if (options.recommendationScoreRunId &&
        !options.listExperimentRecommendationScores)
        throw std::invalid_argument("--recommendation-score-run-id requires --list-experiment-recommendation-scores");
    if (options.recommendationScoreMinimum &&
        !options.listExperimentRecommendationScores)
        throw std::invalid_argument("--recommendation-score-min requires --list-experiment-recommendation-scores");
    if (options.recommendationScoreLimitSpecified &&
        !options.scoreExperimentRecommendations &&
        !options.listExperimentRecommendationScores &&
        !options.listExperimentRecommendationScoreRuns)
        throw std::invalid_argument("--recommendation-score-limit requires a recommendation scoring or score list command");
    const bool recommendationReviewActionCommand =
        options.approveRecommendationId.has_value() ||
        options.rejectRecommendationId.has_value() ||
        options.expireRecommendationId.has_value();
    const int recommendationReviewActionCount =
        (options.approveRecommendationId.has_value() ? 1 : 0) +
        (options.rejectRecommendationId.has_value() ? 1 : 0) +
        (options.expireRecommendationId.has_value() ? 1 : 0);
    if (recommendationReviewActionCount > 1)
        throw std::invalid_argument(
            "only one recommendation review action may be supplied");
    const bool recommendationReviewOnlyOption =
        options.recommendationReviewReasonCode.has_value() ||
        options.recommendationReviewReason.has_value() ||
        options.recommendationReviewer.has_value() ||
        options.recommendationReviewNote.has_value() ||
        options.recommendationReviewScoreId.has_value();
    if (recommendationReviewOnlyOption && !recommendationReviewActionCommand)
        throw std::invalid_argument(
            "recommendation review reason, reviewer, note, and score options require a review action");
    if (options.recommendationReviewActionFilter &&
        !options.listExperimentRecommendationReviews)
        throw std::invalid_argument(
            "--recommendation-review-action requires --list-experiment-recommendation-reviews");
    if (options.recommendationReviewActionFilter &&
        !EA::ExperimentRecommendation::ParseRecommendationReviewAction(
            *options.recommendationReviewActionFilter))
        throw std::invalid_argument("invalid recommendation review action filter");
    if (options.recommendationReviewLimitSpecified &&
        !options.listExperimentRecommendationReviews)
        throw std::invalid_argument(
            "--recommendation-review-limit requires --list-experiment-recommendation-reviews");
    const bool recommendationEvaluationCommand =
        options.evaluateExperimentRecommendations ||
        options.evaluateExperimentRecommendationId.has_value();
    if (options.evaluateExperimentRecommendations &&
        !options.recommendationScanId)
        throw std::invalid_argument(
            "--evaluate-experiment-recommendations requires --recommendation-scan-id");
    if (options.recommendationEvaluationPolicy && !recommendationEvaluationCommand)
        throw std::invalid_argument(
            "--recommendation-evaluation-policy requires an evaluation command");
    if (options.recommendationEvaluationDryRun && !recommendationEvaluationCommand)
        throw std::invalid_argument(
            "--recommendation-evaluation-dry-run requires an evaluation command");
    if (options.recommendationEvaluationDisposition &&
        !options.listExperimentRecommendationEvaluations)
        throw std::invalid_argument(
            "--recommendation-evaluation-disposition requires --list-experiment-recommendation-evaluations");
    if (options.recommendationEvaluationDisposition &&
        !EA::ExperimentRecommendation::ParseRecommendationEvaluationDisposition(
            *options.recommendationEvaluationDisposition))
        throw std::invalid_argument("invalid recommendation evaluation disposition");
    if (options.recommendationEvaluationLimitSpecified &&
        !recommendationEvaluationCommand &&
        !options.listExperimentRecommendationEvaluations &&
        !options.listExperimentRecommendationEvaluationRuns)
        throw std::invalid_argument(
            "--recommendation-evaluation-limit requires evaluation or evaluation listing");
    if (options.recommendationEvaluationLimit > 1000)
        throw std::invalid_argument(
            "--recommendation-evaluation-limit must not exceed 1000");
    const bool recommendationRankingScopeOption =
        options.recommendationRankingEvaluationRunId.has_value() ||
        options.recommendationRankingScanId.has_value() ||
        options.recommendationRankingSymbol.has_value() ||
        options.recommendationRankingHorizon.has_value() ||
        options.recommendationRankingFamily.has_value() ||
        options.recommendationRankingGlobal;
    if (recommendationRankingScopeOption &&
        !options.rankExperimentRecommendationEvaluations)
        throw std::invalid_argument(
            "recommendation ranking scope options require --rank-experiment-recommendation-evaluations");
    const int recommendationRankingScopeCount =
        (options.recommendationRankingEvaluationRunId ? 1 : 0) +
        (options.recommendationRankingScanId ? 1 : 0) +
        (options.recommendationRankingFamily ? 1 : 0) +
        (options.recommendationRankingGlobal ? 1 : 0) +
        ((options.recommendationRankingSymbol ||
          options.recommendationRankingHorizon) ? 1 : 0);
    if (options.rankExperimentRecommendationEvaluations &&
        recommendationRankingScopeCount != 1)
        throw std::invalid_argument(
            "ranking requires exactly one explicit evaluation-run, scan, symbol/horizon, family, or global scope");
    if ((options.recommendationRankingSymbol &&
         options.recommendationRankingSymbol->empty()) ||
        (options.recommendationRankingFamily &&
         options.recommendationRankingFamily->empty()))
        throw std::invalid_argument("recommendation ranking text scope is empty");
    if (options.recommendationRankingDryRun &&
        !options.rankExperimentRecommendationEvaluations)
        throw std::invalid_argument(
            "--recommendation-ranking-dry-run requires ranking creation");
    if (options.recommendationRankingLimitSpecified &&
        !options.rankExperimentRecommendationEvaluations &&
        !options.listExperimentRecommendationRankingSnapshots &&
        !options.listRecommendationRankingMembersId)
        throw std::invalid_argument(
            "--recommendation-ranking-limit requires ranking creation or listing");
    if (options.recommendationRankingLimit > 1000)
        throw std::invalid_argument(
            "--recommendation-ranking-limit must not exceed 1000");
    if (options.recommendationRankingBucket &&
        !options.listRecommendationRankingMembersId)
        throw std::invalid_argument(
            "--recommendation-ranking-bucket requires ranking member listing");
    if (options.recommendationRankingBucket &&
        !EA::ExperimentRecommendation::ParseRecommendationRankingBucket(
            *options.recommendationRankingBucket))
        throw std::invalid_argument("invalid recommendation ranking bucket");
    if (options.compareRecommendationEvaluations &&
        options.compareRecommendationEvaluations->first ==
            options.compareRecommendationEvaluations->second)
        throw std::invalid_argument(
            "recommendation evaluation comparison requires two different IDs");
    if (options.compareRecommendationRankingMembers &&
        options.compareRecommendationRankingMembers->first ==
            options.compareRecommendationRankingMembers->second)
        throw std::invalid_argument(
            "recommendation ranking member comparison requires two different IDs");
    if (recommendationReviewActionCommand)
    {
        EA::ExperimentRecommendation::RecommendationReviewRequest review;
        review.action = options.approveRecommendationId
            ? EA::ExperimentRecommendation::RecommendationReviewAction::approve
            : (options.rejectRecommendationId
                ? EA::ExperimentRecommendation::RecommendationReviewAction::reject
                : EA::ExperimentRecommendation::RecommendationReviewAction::expire);
        review.reasonCode = options.recommendationReviewReasonCode.value_or("");
        review.reasonText = options.recommendationReviewReason;
        review.reviewer = options.recommendationReviewer;
        review.note = options.recommendationReviewNote;
        review.recommendationScoreId = options.recommendationReviewScoreId;
        (void)EA::ExperimentRecommendation::NormalizeRecommendationReviewRequest(
            review);
    }
    const bool conversionProposalReviewAction =
        options.approveConversionProposalId.has_value() ||
        options.rejectConversionProposalId.has_value();
    if (options.approveConversionProposalId &&
        options.rejectConversionProposalId)
        throw std::invalid_argument(
            "only one conversion proposal review action may be supplied");
    const bool conversionProposalReviewOnlyOption =
        options.conversionProposalReviewRequestId.has_value() ||
        options.conversionProposalReviewOperator.has_value() ||
        options.conversionProposalReviewReason.has_value();
    if (conversionProposalReviewOnlyOption && !conversionProposalReviewAction)
        throw std::invalid_argument(
            "conversion proposal review request, operator, and reason options "
            "require an approve or reject action");
    if (conversionProposalReviewAction &&
        !options.conversionProposalReviewRequestId)
        throw std::invalid_argument(
            "conversion proposal review action requires "
            "--conversion-proposal-review-request-id");
    if (options.listConversionProposalsReviewStatus &&
        !EA::ExperimentRecommendation::
            ParseRecommendationConversionProposalReviewDisposition(
                *options.listConversionProposalsReviewStatus))
        throw std::invalid_argument(
            "invalid conversion proposal review status");
    if (options.conversionProposalReviewLimitSpecified &&
        !options.listConversionProposalReviewsId &&
        !options.listConversionProposalsReviewStatus)
        throw std::invalid_argument(
            "--conversion-proposal-review-limit requires a conversion proposal "
            "review list command");
    if (options.conversionProposalReviewLimit >
        EA::ExperimentRecommendation::
            kMaximumRecommendationConversionProposalReviewListLimit)
        throw std::invalid_argument(
            "--conversion-proposal-review-limit must not exceed 1000");
    if (conversionProposalReviewAction)
    {
        EA::ExperimentRecommendation::
            RecommendationConversionProposalReviewRequest review;
        review.proposalId = options.approveConversionProposalId
            ? *options.approveConversionProposalId
            : *options.rejectConversionProposalId;
        review.decision = options.approveConversionProposalId
            ? EA::ExperimentRecommendation::
                  RecommendationConversionProposalReviewDecision::approve
            : EA::ExperimentRecommendation::
                  RecommendationConversionProposalReviewDecision::reject;
        review.requestId = *options.conversionProposalReviewRequestId;
        review.operatorIdentity = options.conversionProposalReviewOperator;
        review.reasonText = options.conversionProposalReviewReason;
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationConversionProposalReviewRequest(review);
    }
    if (options.conversionWorkflowState &&
        !options.listRecommendationConversionWorkflows)
        throw std::invalid_argument(
            "--conversion-workflow-state requires "
            "--list-recommendation-conversion-workflows");
    if (options.conversionWorkflowLimitSpecified &&
        !options.listRecommendationConversionWorkflows)
        throw std::invalid_argument(
            "--conversion-workflow-limit requires "
            "--list-recommendation-conversion-workflows");
    if (options.conversionWorkflowLimit >
        EA::ExperimentRecommendation::
            kMaximumRecommendationConversionWorkflowListLimit)
        throw std::invalid_argument(
            "--conversion-workflow-limit must not exceed 1000");
    const int campaignApprovalActionCount =
        (options.approveRecommendationCampaign ? 1 : 0) +
        (options.rejectRecommendationCampaign ? 1 : 0);
    if (campaignApprovalActionCount > 1)
        throw std::invalid_argument(
            "campaign approval and rejection are mutually exclusive");
    const bool campaignApprovalAction = campaignApprovalActionCount == 1;
    if (options.campaignPolicyOptionSpecified &&
        !options.planRecommendationCampaign &&
        !options.reviewRecommendationCampaign &&
        !campaignApprovalAction)
        throw std::invalid_argument(
            "campaign options require campaign planning, review, approval, or "
            "rejection");
    if (options.planRecommendationCampaign ||
        options.reviewRecommendationCampaign || campaignApprovalAction)
    {
        if (const auto error = EA::ExperimentRecommendation::
                ValidateRecommendationCampaignPlanningPolicy(
                    options.campaignPlanningPolicy))
            throw std::invalid_argument(*error);
        if (const auto error = EA::ExperimentRecommendation::
                ValidateRecommendationCampaignPlanningScope(
                    options.campaignPlanningScope))
            throw std::invalid_argument(*error);
    }
    const bool campaignApprovalMetadata =
        options.campaignReviewIdentityHash.has_value() ||
        options.campaignReviewer.has_value() ||
        options.campaignReviewReason.has_value();
    if (campaignApprovalMetadata && !campaignApprovalAction)
        throw std::invalid_argument(
            "campaign review identity, reviewer, and reason require campaign "
            "approval or rejection");
    if (campaignApprovalAction)
    {
        if (!options.campaignReviewIdentityHash || !options.campaignReviewer ||
            !options.campaignReviewReason)
            throw std::invalid_argument(
                "campaign approval or rejection requires "
                "--campaign-review-identity-hash, --campaign-reviewer, and "
                "--campaign-review-reason");
        EA::ExperimentRecommendation::RecommendationCampaignApprovalRequest
            request;
        request.decision = options.approveRecommendationCampaign
            ? EA::ExperimentRecommendation::
                  RecommendationCampaignApprovalDecision::approved
            : EA::ExperimentRecommendation::
                  RecommendationCampaignApprovalDecision::rejected;
        request.expectedCampaignReviewIdentityHash =
            *options.campaignReviewIdentityHash;
        request.reviewerIdentity = *options.campaignReviewer;
        request.reasonText = *options.campaignReviewReason;
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignApprovalRequest(request);
    }
    if (options.campaignApprovalDecision &&
        !options.listRecommendationCampaignApprovals)
        throw std::invalid_argument(
            "--campaign-approval-decision requires "
            "--list-recommendation-campaign-approvals");
    if (options.campaignApprovalLimitSpecified &&
        !options.listRecommendationCampaignApprovals)
        throw std::invalid_argument(
            "--campaign-approval-limit requires "
            "--list-recommendation-campaign-approvals");
    if (options.campaignApprovalLimit > EA::ExperimentRecommendation::
            kMaximumRecommendationCampaignApprovalListLimit)
        throw std::invalid_argument(
            "--campaign-approval-limit must not exceed 1000");
    const bool campaignMaterializationMetadata =
        options.campaignMaterializationApprovalId.has_value() ||
        options.campaignMaterializedBy.has_value() ||
        options.campaignMaterializationReason.has_value();
    if (campaignMaterializationMetadata &&
        !options.materializeRecommendationCampaign &&
        !options.listRecommendationCampaignMaterializations)
        throw std::invalid_argument(
            "campaign materialization metadata requires materialize or list");
    if (options.materializeRecommendationCampaign)
    {
        if (!options.campaignMaterializationApprovalId ||
            !options.campaignMaterializedBy ||
            !options.campaignMaterializationReason)
            throw std::invalid_argument(
                "campaign materialization requires --campaign-approval-id, "
                "--campaign-materialized-by, and "
                "--campaign-materialization-reason");
        EA::ExperimentRecommendation::RecommendationCampaignMaterializationRequest
            request{*options.campaignMaterializationApprovalId,
                    *options.campaignMaterializedBy,
                    *options.campaignMaterializationReason};
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignMaterializationRequest(request);
    }
    if ((options.campaignMaterializedBy ||
         options.campaignMaterializationReason) &&
        !options.materializeRecommendationCampaign)
        throw std::invalid_argument(
            "materialized-by and materialization-reason require materialize");
    if (options.campaignMaterializationApprovalId &&
        !options.materializeRecommendationCampaign &&
        !options.listRecommendationCampaignMaterializations)
        throw std::invalid_argument(
            "--campaign-approval-id requires materialize or list");
    if (options.campaignMaterializationLimitSpecified &&
        !options.listRecommendationCampaignMaterializations)
        throw std::invalid_argument(
            "--campaign-materialization-limit requires list materializations");
    if (options.campaignMaterializationLimit >
        EA::ExperimentRecommendation::
            kMaximumRecommendationCampaignMaterializationListLimit)
        throw std::invalid_argument(
            "--campaign-materialization-limit must not exceed 1000");
    if (options.campaignHandoffLimitSpecified &&
        !options.listRecommendationCampaignHandoffs)
        throw std::invalid_argument(
            "--campaign-handoff-limit requires "
            "--list-recommendation-campaign-handoffs");
    if (options.campaignHandoffLimit > EA::ExperimentRecommendation::
            kMaximumRecommendationCampaignHandoffListLimit)
        throw std::invalid_argument(
            "--campaign-handoff-limit must not exceed 1000");
    const bool campaignProposalReviewMetadata =
        options.campaignProposalReviewDecisionSpecified ||
        options.campaignProposalReviewOperatorSpecified ||
        options.campaignProposalReviewReasonSpecified;
    if (campaignProposalReviewMetadata &&
        !options.reviewRecommendationCampaignMaterializationId)
        throw std::invalid_argument(
            "campaign proposal review metadata requires "
            "--review-recommendation-campaign-materialization");
    if (options.reviewRecommendationCampaignMaterializationId)
    {
        if (!options.campaignProposalReviewDecision ||
            !options.campaignProposalReviewOperator ||
            !options.campaignProposalReviewReason)
            throw std::invalid_argument(
                "campaign proposal review requires "
                "--campaign-proposal-review-decision, "
                "--campaign-proposal-review-operator, and "
                "--campaign-proposal-review-reason");
        if (!options.dryRun && !options.yes)
            throw std::invalid_argument(
                "campaign proposal review write requires --yes");
        EA::ExperimentRecommendation::RecommendationCampaignProposalReviewRequest
            request;
        request.materializationId =
            *options.reviewRecommendationCampaignMaterializationId;
        request.decision = *options.campaignProposalReviewDecision;
        request.operatorIdentity = *options.campaignProposalReviewOperator;
        request.reasonText = *options.campaignProposalReviewReason;
        request.dryRun = options.dryRun;
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignProposalReviewRequest(request);
    }
    if (options.executeRecommendationCampaignMaterializationId)
    {
        if (!options.dryRun && !options.yes)
            throw std::invalid_argument(
                "campaign execution write requires --yes");
        EA::ExperimentRecommendation::RecommendationCampaignExecutionRequest
            request{*options.executeRecommendationCampaignMaterializationId,
                    options.dryRun};
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignExecutionRequest(request);
    }
    if (options.activateRecommendationCampaignMaterializationId)
    {
        if (!options.dryRun && !options.yes)
            throw std::invalid_argument(
                "campaign activation write requires --yes");
        EA::ExperimentRecommendation::RecommendationCampaignActivationRequest
            request{*options.activateRecommendationCampaignMaterializationId,
                    options.dryRun};
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignActivationRequest(request);
    }
    if (options.launchRecommendationCampaignMaterializationId)
    {
        if (!options.dryRun && !options.yes)
            throw std::invalid_argument(
                "campaign launch write requires --yes");
        EA::ExperimentRecommendation::RecommendationCampaignLaunchRequest
            request{*options.launchRecommendationCampaignMaterializationId,
                    options.dryRun};
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignLaunchRequest(request);
    }
    if (options.recommendationCampaignStatusMaterializationId)
    {
        if (options.dryRun)
            throw std::invalid_argument(
                "--dry-run is not valid with --recommendation-campaign-status");
        if (options.yes)
            throw std::invalid_argument(
                "--yes is not valid with --recommendation-campaign-status");
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignStatusRequest(
                {*options.recommendationCampaignStatusMaterializationId});
    }
    if (options.recommendationCampaignOutcomeAssessmentMaterializationId)
    {
        if (options.dryRun)
            throw std::invalid_argument(
                "--dry-run is not valid with "
                "--recommendation-campaign-outcome-assessment");
        if (options.yes)
            throw std::invalid_argument(
                "--yes is not valid with "
                "--recommendation-campaign-outcome-assessment");
        (void)EA::ExperimentRecommendation::
            NormalizeRecommendationCampaignOutcomeAssessmentRequest(
                {*options.
                    recommendationCampaignOutcomeAssessmentMaterializationId});
    }
    const bool campaignOperationsBudgetMutation =
        options.campaignOperationsBudgetGrantCampaignId.has_value() ||
        options.campaignOperationsBudgetAmendCampaignId.has_value() ||
        options.campaignOperationsBudgetRevokeCampaignId.has_value() ||
        options.campaignOperationsBudgetSupersedeCampaignId.has_value();
    const bool campaignOperationsAdmissionMutation =
        options.campaignOperationsAdmitMaterializationId.has_value();
    const bool campaignOperationsRequestMutation =
        options.campaignOperationsAcceptRequestCampaignId.has_value();
    const bool campaignOperationsControlMutation =
        options.campaignOperationsPauseCampaignId.has_value() ||
        options.campaignOperationsResumeCampaignId.has_value() ||
        options.campaignOperationsCancelCampaignId.has_value() ||
        options.campaignOperationsReconcileRunKey.has_value();
    const bool campaignOperationsCompletionMutation =
        options.campaignOperationsCompleteCampaignId.has_value();
    const bool campaignOperationsProductionMutation =
        options.campaignOperationsProductionEnable ||
        options.campaignOperationsProductionDisable ||
        options.campaignOperationsProductionDispatchRequest;
    const bool campaignOperationsManagerMutation =
        options.campaignOperationsManagerRunOnceLimit.has_value();
    const bool campaignOperationsMutation =
        campaignOperationsBudgetMutation ||
        campaignOperationsAdmissionMutation ||
        campaignOperationsRequestMutation ||
        campaignOperationsControlMutation ||
        campaignOperationsCompletionMutation ||
        campaignOperationsProductionMutation ||
        campaignOperationsManagerMutation;
    const bool campaignOperationsStatus =
        options.campaignOperationsBudgetStatusCampaignId.has_value() ||
        options.campaignOperationsRequestStatusRequestId.has_value() ||
        options.campaignOperationsControlStatusCampaignId.has_value() ||
        options.campaignOperationsCompletionStatusCampaignId.has_value() ||
        options.campaignOperationsProductionReadiness ||
        options.campaignOperationsProductionStatus;
    const bool campaignOperationsMetadata =
        options.campaignOperationsExpectedBudgetVersion.has_value() ||
        options.campaignOperationsBudgetValue.has_value() ||
        options.campaignOperationsActor.has_value() ||
        options.campaignOperationsReason.has_value() ||
        options.campaignOperationsReservationExpiresAt.has_value() ||
        options.campaignOperationsExpectedControlVersion.has_value() ||
        options.campaignOperationsControlRequestId.has_value() ||
        options.campaignOperationsExpectedRequestVersion.has_value() ||
        options.campaignOperationsExpectedProductionVersion.has_value() ||
        options.campaignOperationsOperationKey.has_value() ||
        options.campaignOperationsIndependentVerificationReference.has_value() ||
        options.campaignOperationsReconcileAfterRequestId.has_value() ||
        options.campaignOperationsReconcileLimit.has_value();
    if (campaignOperationsMetadata && !campaignOperationsMutation)
        throw std::invalid_argument(
            "Campaign Operations metadata requires an authorized mutation "
            "or status command");
    if ((options.campaignOperationsProductionReadiness ||
            options.campaignOperationsProductionStatus) &&
        (options.dryRun || options.yes))
        throw std::invalid_argument(
            "Campaign Operations production readiness/status accepts "
            "neither --dry-run nor --yes");
    if (campaignOperationsMutation)
    {
        if (options.dryRun)
            throw std::invalid_argument(
                "--dry-run is not valid for durable Campaign Operations "
                "mutations");
        if (!options.yes)
            throw std::invalid_argument(
                "Campaign Operations mutation requires --yes");
        if (!options.campaignOperationsReconcileRunKey &&
            !options.campaignOperationsManagerRunOnceLimit &&
            (!options.campaignOperationsActor ||
             (!options.campaignOperationsReason &&
              !options.campaignOperationsProductionDispatchRequest)))
            throw std::invalid_argument(
                "Campaign Operations mutation requires "
                "--campaign-operations-actor and "
                "--campaign-operations-reason");
    }
    if (campaignOperationsManagerMutation &&
        (options.dryRun || !options.yes))
        throw std::invalid_argument(
            "Campaign Operations Manager run-once requires --yes and no --dry-run");
    if (campaignOperationsManagerMutation &&
        (options.campaignOperationsActor || options.campaignOperationsReason ||
         options.campaignOperationsOperationKey ||
         options.campaignOperationsControlRequestId ||
         options.campaignOperationsExpectedRequestVersion))
        throw std::invalid_argument(
            "Campaign Operations Manager run-once accepts no request metadata");
    if (campaignOperationsProductionMutation)
    {
        if (!options.campaignOperationsOperationKey)
            throw std::invalid_argument(
                "production mutation requires "
                "--campaign-operations-operation-key");
        if (!EA::CampaignOperations::IsValidProductionOperationKey(
                *options.campaignOperationsOperationKey))
            throw std::invalid_argument(
                "invalid --campaign-operations-operation-key for production");
        const int commandCount =
            (options.campaignOperationsProductionEnable ? 1 : 0) +
            (options.campaignOperationsProductionDisable ? 1 : 0) +
            (options.campaignOperationsProductionDispatchRequest ? 1 : 0);
        if (commandCount != 1)
            throw std::invalid_argument(
                "exactly one production H2 command is required");
        if (options.campaignOperationsProductionEnable ||
            options.campaignOperationsProductionDisable)
        {
            if (!options.campaignOperationsExpectedProductionVersion ||
                !options.campaignOperationsReason)
                throw std::invalid_argument(
                    "production enable/disable requires expected production "
                    "version and reason");
        }
        else if (options.campaignOperationsExpectedProductionVersion ||
                 options.campaignOperationsIndependentVerificationReference)
            throw std::invalid_argument(
                "production dispatch does not accept enable metadata");
        if (options.campaignOperationsProductionEnable &&
            !options.campaignOperationsIndependentVerificationReference)
            throw std::invalid_argument(
                "production enable requires independent verification reference");
        if (options.campaignOperationsProductionDispatchRequest &&
            (!options.campaignOperationsControlRequestId ||
             !options.campaignOperationsExpectedRequestVersion))
            throw std::invalid_argument(
                "production dispatch requires request ID and expected request version");
    }
    if (campaignOperationsStatus && (options.dryRun || options.yes))
        throw std::invalid_argument(
            "Campaign Operations status accepts neither --dry-run nor --yes");
    if (campaignOperationsBudgetMutation)
    {
        if (!options.campaignOperationsExpectedBudgetVersion)
            throw std::invalid_argument(
                "budget mutation requires "
                "--campaign-operations-expected-budget-version");
        EA::CampaignOperations::BudgetAdministrationRequest request;
        request.campaignId =
            options.campaignOperationsBudgetGrantCampaignId
                ? *options.campaignOperationsBudgetGrantCampaignId
                : (options.campaignOperationsBudgetAmendCampaignId
                          ? *options.campaignOperationsBudgetAmendCampaignId
                          : (options.campaignOperationsBudgetRevokeCampaignId
                                    ? *options.
                                          campaignOperationsBudgetRevokeCampaignId
                                    : *options.
                                          campaignOperationsBudgetSupersedeCampaignId));
        request.expectedLedgerVersion =
            *options.campaignOperationsExpectedBudgetVersion;
        request.kind = options.campaignOperationsBudgetGrantCampaignId
            ? EA::CampaignOperations::BudgetLedgerEntryKind::grant
            : (options.campaignOperationsBudgetAmendCampaignId
                      ? EA::CampaignOperations::BudgetLedgerEntryKind::amend
                      : (options.campaignOperationsBudgetRevokeCampaignId
                                ? EA::CampaignOperations::
                                      BudgetLedgerEntryKind::revoke
                                : EA::CampaignOperations::
                                      BudgetLedgerEntryKind::supersede));
        request.value = options.campaignOperationsBudgetValue;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        (void)EA::CampaignOperations::
            ValidateBudgetAdministrationRequest(request);
    }
    if (campaignOperationsAdmissionMutation)
    {
        EA::CampaignOperations::OperationalCampaignAdmissionRequest request;
        request.materializationId =
            *options.campaignOperationsAdmitMaterializationId;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        (void)EA::CampaignOperations::
            ValidateOperationalCampaignAdmissionRequest(request);
    }
    if (campaignOperationsRequestMutation)
    {
        if (options.campaignOperationsExpectedBudgetVersion ||
            options.campaignOperationsBudgetValue)
            throw std::invalid_argument(
                "request acceptance does not accept budget mutation metadata");
        EA::CampaignOperations::OperationalRequestAcceptanceRequest request;
        request.campaignId =
            *options.campaignOperationsAcceptRequestCampaignId;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        request.expiresAt =
            options.campaignOperationsReservationExpiresAt;
        (void)EA::CampaignOperations::
            ValidateOperationalRequestAcceptanceRequest(request);
    }
    if ((options.campaignOperationsExpectedBudgetVersion ||
            options.campaignOperationsBudgetValue) &&
        !campaignOperationsBudgetMutation &&
        !campaignOperationsRequestMutation)
        throw std::invalid_argument(
            "budget metadata requires a budget mutation");
    if (options.campaignOperationsReservationExpiresAt &&
        !campaignOperationsRequestMutation)
        throw std::invalid_argument(
            "reservation expiry requires request acceptance");
    const bool pauseOrResume =
        options.campaignOperationsPauseCampaignId.has_value() ||
        options.campaignOperationsResumeCampaignId.has_value();
    if (pauseOrResume)
    {
        if (!options.campaignOperationsExpectedControlVersion)
            throw std::invalid_argument(
                "pause/resume requires "
                "--campaign-operations-expected-control-version");
        EA::CampaignOperations::CampaignControlRequest control;
        control.campaignId = options.campaignOperationsPauseCampaignId
            ? *options.campaignOperationsPauseCampaignId
            : *options.campaignOperationsResumeCampaignId;
        control.expectedControlVersion =
            *options.campaignOperationsExpectedControlVersion;
        control.action = options.campaignOperationsPauseCampaignId
            ? EA::CampaignOperations::ControlEventKind::pause
            : EA::CampaignOperations::ControlEventKind::resume;
        control.actorIdentity = *options.campaignOperationsActor;
        control.reason = *options.campaignOperationsReason;
        (void)EA::CampaignOperations::ValidateCampaignControlRequest(
            control);
    }
    if (options.campaignOperationsExpectedControlVersion &&
        !pauseOrResume)
        throw std::invalid_argument(
            "control version requires pause or resume");
    if (options.campaignOperationsCancelCampaignId)
    {
        if (!options.campaignOperationsControlRequestId ||
            !options.campaignOperationsExpectedRequestVersion ||
            !options.campaignOperationsOperationKey)
            throw std::invalid_argument(
                "cancellation requires --campaign-operations-request-id, "
                "--campaign-operations-expected-request-version, and "
                "--campaign-operations-operation-key");
        EA::CampaignOperations::CampaignCancellationCommandRequest cancel;
        cancel.campaignId =
            *options.campaignOperationsCancelCampaignId;
        cancel.requestId = options.campaignOperationsControlRequestId;
        cancel.expectedRequestVersion =
            options.campaignOperationsExpectedRequestVersion;
        cancel.operationKey =
            *options.campaignOperationsOperationKey;
        cancel.actorIdentity = *options.campaignOperationsActor;
        cancel.reason = *options.campaignOperationsReason;
        (void)EA::CampaignOperations::
            ValidateCampaignCancellationRequest(cancel);
    }
    if ((options.campaignOperationsControlRequestId ||
            options.campaignOperationsExpectedRequestVersion) &&
        !options.campaignOperationsCancelCampaignId &&
        !options.campaignOperationsProductionDispatchRequest)
        throw std::invalid_argument(
            "cancellation metadata requires cancellation");
    if (options.campaignOperationsOperationKey &&
        !options.campaignOperationsCancelCampaignId &&
        !options.campaignOperationsCompleteCampaignId &&
        !campaignOperationsProductionMutation)
        throw std::invalid_argument(
            "cancellation metadata requires cancellation");
    if (options.campaignOperationsCompleteCampaignId)
    {
        if (!options.campaignOperationsOperationKey)
            throw std::invalid_argument(
                "complete-if-settled requires "
                "--campaign-operations-operation-key");
        EA::CampaignOperations::CompleteIfSettledRequest completion;
        completion.campaignId =
            *options.campaignOperationsCompleteCampaignId;
        completion.operationKey =
            *options.campaignOperationsOperationKey;
        completion.actorIdentity =
            *options.campaignOperationsActor;
        completion.reason = *options.campaignOperationsReason;
        (void)EA::CampaignOperations::ValidateCompleteIfSettledRequest(
            completion);
    }
    if (options.campaignOperationsReconcileRunKey)
    {
        if (options.campaignOperationsActor ||
            options.campaignOperationsReason)
            throw std::invalid_argument(
                "reconciliation uses its service identity and accepts no "
                "operator actor/reason");
        EA::CampaignOperations::ReconciliationObserveRequest reconcile;
        reconcile.runKey =
            *options.campaignOperationsReconcileRunKey;
        reconcile.afterRequestId =
            options.campaignOperationsReconcileAfterRequestId.value_or(0);
        reconcile.limit =
            options.campaignOperationsReconcileLimit.value_or(100);
        reconcile.resolveSafeTransitions =
            options.campaignOperationsReconcileRecover;
        (void)EA::CampaignOperations::
            ValidateReconciliationObserveRequest(reconcile);
    }
    if ((options.campaignOperationsReconcileAfterRequestId ||
            options.campaignOperationsReconcileLimit) &&
        !options.campaignOperationsReconcileRunKey)
        throw std::invalid_argument(
            "reconciliation metadata requires reconciliation");
    const bool hasAutomaticContinuationOption =
        options.autoEvaluateContinuations ||
        options.autoQueueContinuations ||
        options.continuationDryRun ||
        options.continuationScanSecondsSpecified ||
        options.continuationMaxQueuesPerScanSpecified;
    if (hasAutomaticContinuationOption && !options.scheduleExperiments)
        throw std::invalid_argument("automatic continuation options are only valid with --schedule-experiments");
    if (options.continuationDryRun && !options.autoEvaluateContinuations)
        throw std::invalid_argument("--continuation-dry-run requires --auto-evaluate-continuations or --auto-queue-continuations");
    if ((options.queueCheckpointInfer ||
         options.queueCheckpointInferMinEpoch.has_value() ||
         options.queueCheckpointInferInterval.has_value()) &&
        !options.queueExperiment &&
        !options.queueSweep)
    {
        throw std::invalid_argument("--checkpoint-infer options are only valid with --queue-experiment or --queue-sweep");
    }
    const bool hasQueuePolicyOptions =
        options.queueCheckpointPolicy ||
        options.checkpointPolicyMinLeaderScore.has_value() ||
        options.checkpointPolicyMinInferAccuracy.has_value() ||
        options.checkpointPolicyTopN.has_value() ||
        options.checkpointPolicyScope != "symbol_horizon" ||
        options.checkpointPolicyStopMode != "next_checkpoint" ||
        options.checkpointPolicyGraceEvals != 1;
    if (hasQueuePolicyOptions && !options.queueExperiment && !options.queueSweep)
        throw std::invalid_argument("--checkpoint-policy options are only valid with --queue-experiment or --queue-sweep");
    if (hasQueuePolicyOptions && !options.queueCheckpointPolicy)
        throw std::invalid_argument("--checkpoint-policy rule options require --checkpoint-policy");
    if (hasQueuePolicyOptions && !options.queueCheckpointInfer)
        throw std::invalid_argument("--checkpoint-policy options require --checkpoint-infer");
    ValidateCheckpointPolicyConfig(options);
    const int globalControlCommandCount =
        (options.pauseAllExperiments ? 1 : 0) +
        (options.resumeAllExperiments ? 1 : 0) +
        (options.cancelAllExperiments ? 1 : 0);
    if (globalControlCommandCount > 1)
        throw std::invalid_argument(
            "pause-all, resume-all, and cancel-all are mutually exclusive");
    if (options.cancelImmediate && options.cancelAfterNextCheckpoint)
        throw std::invalid_argument(
            "--immediate and --after-next-checkpoint are mutually exclusive");
    if ((options.cancelImmediate || options.cancelAfterNextCheckpoint) &&
        !options.cancelAllExperiments)
        throw std::invalid_argument(
            "cancellation mode requires --cancel-all-experiments");
    if (options.inferBeforeCancel && !options.cancelAllExperiments)
        throw std::invalid_argument(
            "--infer-before-cancel requires --cancel-all-experiments");
    if (options.cancelAllExperiments &&
        ((options.cancelImmediate ? 1 : 0) +
         (options.cancelAfterNextCheckpoint ? 1 : 0) != 1))
        throw std::invalid_argument(
            "--cancel-all-experiments requires exactly one of --immediate "
            "or --after-next-checkpoint");
    if (globalControlCommandCount > 0 && !options.dryRun && !options.yes)
        throw std::invalid_argument(
            "global experiment control writes require --yes");
    if (options.reconcileWorkerAttemptId.has_value() &&
        !options.dryRun && !options.yes)
    {
        throw std::invalid_argument(
            "--reconcile-worker-attempt requires --dry-run or --yes");
    }
    if (options.recoverFailedInferenceExperimentId.has_value() &&
        (options.dryRun == options.yes))
    {
        throw std::invalid_argument(
            "--recover-failed-inference requires exactly one of "
            "--dry-run or --yes");
    }
    if (options.completeSchedulerProtocolCutover &&
        (!options.yes || options.dryRun))
    {
        throw std::invalid_argument(
            "--complete-scheduler-protocol-cutover requires --yes "
            "and does not accept --dry-run");
    }
    if (options.campaignProfitabilityShadowSnapshotId.has_value() !=
        options.campaignProfitabilityShadowWeights.has_value())
        throw std::invalid_argument(
            "--shadow-rank-campaign-profitability requires "
            "--profitability-shadow-weights and vice versa");
    if (options.campaignProfitabilityShadowSnapshotId)
    {
        bool consumeValue = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (consumeValue)
            {
                consumeValue = false;
                continue;
            }
            if (argument == "--shadow-rank-campaign-profitability" ||
                argument == "--profitability-shadow-weights")
            {
                consumeValue = true;
                continue;
            }
            if (argument.rfind(
                    "--shadow-rank-campaign-profitability=", 0) == 0 ||
                argument.rfind("--profitability-shadow-weights=", 0) == 0)
                continue;
            throw std::invalid_argument(
                "--shadow-rank-campaign-profitability does not accept "
                "unrelated option '" + argument + "'");
        }
    }
    if (options.campaignProfitabilityCalibrationSnapshotId)
    {
        bool consumeValue = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (consumeValue)
            {
                consumeValue = false;
                continue;
            }
            if (argument == "--calibrate-campaign-profitability")
            {
                consumeValue = true;
                continue;
            }
            if (argument.rfind(
                    "--calibrate-campaign-profitability=", 0) == 0)
                continue;
            throw std::invalid_argument(
                "--calibrate-campaign-profitability does not accept "
                "unrelated option '" + argument + "'");
        }
    }
    if (options.campaignProfitabilityTemporalValidation && argc != 2)
        throw std::invalid_argument(
            "--validate-campaign-profitability-temporal does not accept "
            "unrelated options");
    if (options.campaignProfitabilityForwardValidationPrecommit)
    {
        bool consumeValue = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (consumeValue)
            {
                consumeValue = false;
                continue;
            }
            if (argument ==
                "--prepare-campaign-profitability-forward-validation")
            {
                consumeValue = true;
                continue;
            }
            if (argument.rfind(
                    "--prepare-campaign-profitability-forward-validation=",
                    0) == 0)
                continue;
            throw std::invalid_argument(
                "--prepare-campaign-profitability-forward-validation does "
                "not accept unrelated option '" + argument + "'");
        }
    }
    if (options.campaignProfitabilityOutcomePreparationCohort)
    {
        bool consumeValue = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (consumeValue) { consumeValue = false; continue; }
            if (argument == "--prepare-campaign-profitability-outcome-jobs")
            {
                consumeValue = true;
                continue;
            }
            if (argument.rfind(
                    "--prepare-campaign-profitability-outcome-jobs=", 0) == 0)
                continue;
            throw std::invalid_argument(
                "--prepare-campaign-profitability-outcome-jobs does not accept unrelated option '" +
                argument + "'");
        }
    }
    if (options.campaignProfitabilityProspectiveComparisonCohort)
    {
        bool consumeValue = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (consumeValue) { consumeValue = false; continue; }
            if (argument == "--compare-campaign-profitability-prospective")
            {
                consumeValue = true;
                continue;
            }
            if (argument.rfind(
                    "--compare-campaign-profitability-prospective=", 0) == 0)
                continue;
            throw std::invalid_argument(
                "--compare-campaign-profitability-prospective does not accept unrelated option '" +
                argument + "'");
        }
    }
    if (commandCount != 1)
        throw std::invalid_argument("expected exactly one experiment scheduler command");
    if (options.modelInfo && !options.modelInfoModelId.has_value())
        throw std::invalid_argument("--model-info requires --model=<model_id>");
    if (!options.modelInfo && options.modelInfoModelId.has_value())
        throw std::invalid_argument("--model is only valid with --model-info or existing inference commands");
    if (!options.compactStatus && options.statusExperimentId.has_value())
        throw std::invalid_argument("--experiment-id is only valid with --status");
    if (options.recoverOrphansOnly && !options.scheduleExperiments)
        throw std::invalid_argument("--recover-orphans-only requires --schedule-experiments");
    if (options.legacyLayout6InferWorkerPath &&
        !options.scheduleExperiments)
    {
        throw std::invalid_argument(
            "--legacy-layout6-infer-worker requires --schedule-experiments");
    }
    if (options.semanticWorkerRegistryPathSpecified &&
        !options.scheduleExperiments)
    {
        throw std::invalid_argument(
            "--semantic-worker-registry requires --schedule-experiments");
    }
    if (options.autoGenerateReports &&
        !options.scheduleExperiments &&
        !options.analyzeExperimentId.has_value())
    {
        throw std::invalid_argument("--auto-generate-reports requires --schedule-experiments or --analyze-experiment");
    }
    if (options.logLevel != "quiet" &&
        options.logLevel != "summary" &&
        options.logLevel != "diagnostic")
        throw std::invalid_argument("--log-level must be quiet, summary, or diagnostic");
    if (options.resumeExpandInputWidth &&
        (!options.resumeModelId.has_value() ||
         !options.queueExperiment))
    {
        throw std::invalid_argument(
            "--resume-expand-input-width requires --resume-model-id and "
            "--queue-experiment");
    }

    return options;
}

void EnsureRequiredEnqueueOptions(const SchedulerOptions& options)
{
    if ((!options.symbol.has_value() && !options.resumeModelId.has_value()) ||
        !options.predictionHorizon.has_value() ||
        !options.cNextThreshold.has_value() ||
        !options.targetEpochs.has_value() ||
        !options.trainStart.has_value() ||
        !options.trainEnd.has_value())
    {
        throw std::invalid_argument("--enqueue-experiment requires --symbol unless --resume-model-id is supplied, plus --prediction-horizon, --c-next-threshold, --target-epochs, --train-start, and --train-end");
    }
    if (options.inferStart.has_value() != options.inferEnd.has_value())
        throw std::invalid_argument("--infer-start and --infer-end must be supplied together");
}


std::optional<long long> CountRowsIfTableExists(pqxx::work& w, const std::string& tableName)
{
    if (!TableExists(w, tableName))
        return std::nullopt;

    if (tableName == "model")
        return w.exec("SELECT count(*) FROM model;").one_row()[0].as<long long>();
    if (tableName == "experiment")
        return w.exec("SELECT count(*) FROM experiment;").one_row()[0].as<long long>();
    if (tableName == "experiment_analysis_result")
        return w.exec("SELECT count(*) FROM experiment_analysis_result;").one_row()[0].as<long long>();

    throw std::invalid_argument("unsupported backup manifest row count table: " + tableName);
}














std::vector<AutoResumeCandidate> LoadAutoResumeCandidates(pqxx::work& w,
                                                                 const SchedulerOptions& options)
{
    const std::string canonicalSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
    const std::string trainRange = DateOnly(*options.trainStart) + "|" + DateOnly(*options.trainEnd);

    std::ostringstream sql;
    sql << "WITH cfg AS ("
        << "  SELECT model_id,"
        << "         max(value) FILTER (WHERE col_idx = 1) AS prediction_horizon,"
        << "         max(value) FILTER (WHERE col_idx = 2) AS threshold_logret,"
        << "         max(value) FILTER (WHERE col_idx = 10) AS completed_epochs"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << "), sym AS ("
        << "  SELECT model_id, string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS symbol"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_symbol_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << "), rng AS ("
        << "  SELECT model_id, string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS train_range"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_range_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << ") "
        << "SELECT m.model_id, COALESCE(m.name, ''), round(cfg.completed_epochs)::int "
        << "FROM model m "
        << "JOIN cfg ON cfg.model_id = m.model_id "
        << "JOIN sym ON sym.model_id = m.model_id "
        << "JOIN rng ON rng.model_id = m.model_id "
        << "WHERE sym.symbol = " << w.quote(canonicalSymbol)
        << " AND round(cfg.prediction_horizon)::int = " << *options.predictionHorizon
        << " AND abs(cfg.threshold_logret - " << FormatDouble(*options.cNextThreshold) << ") <= 1e-7"
        << " AND rng.train_range = " << w.quote(trainRange)
        << " AND round(cfg.completed_epochs)::int < " << *options.targetEpochs
        << " ORDER BY round(cfg.completed_epochs)::int DESC, m.model_id DESC;";

    pqxx::result rows = w.exec(sql.str());
    std::vector<AutoResumeCandidate> candidates;
    candidates.reserve(rows.size());
    for (const auto& row : rows)
    {
        candidates.push_back(AutoResumeCandidate{
            row[0].as<long long>(),
            row[1].as<std::string>(),
            row[2].as<int>()
        });
    }
    return candidates;
}

long long ResolveAutoResumeModelId(pqxx::work& w,
                                          const SchedulerOptions& options)
{
    const std::vector<AutoResumeCandidate> candidates = LoadAutoResumeCandidates(w, options);
    if (candidates.empty())
    {
        std::cout << "QUEUE_RESUME_AUTO_NO_MATCH"
                  << ",symbol=" << *options.symbol
                  << ",prediction_horizon=" << *options.predictionHorizon
                  << ",threshold=" << FormatDouble(*options.cNextThreshold)
                  << ",target_epochs=" << *options.targetEpochs
                  << ",train_start=" << *options.trainStart
                  << ",train_end=" << *options.trainEnd
                  << std::endl;
        throw std::invalid_argument("QUEUE_RESUME_AUTO_NO_MATCH");
    }

    const int bestCompletedEpochs = candidates.front().completedEpochs;
    int tiedBestCount = 0;
    for (const auto& candidate : candidates)
    {
        if (candidate.completedEpochs == bestCompletedEpochs)
            ++tiedBestCount;
    }

    for (const auto& candidate : candidates)
    {
        std::cout << "QUEUE_RESUME_AUTO_CANDIDATE"
                  << ",model_id=" << candidate.modelId
                  << ",name=" << candidate.name
                  << ",completed_epochs=" << candidate.completedEpochs
                  << std::endl;
    }

    if (tiedBestCount > 1)
    {
        std::cout << "QUEUE_RESUME_AUTO_AMBIGUOUS"
                  << ",best_completed_epochs=" << bestCompletedEpochs
                  << ",candidate_count=" << candidates.size()
                  << std::endl;
        throw std::invalid_argument("QUEUE_RESUME_AUTO_AMBIGUOUS");
    }

    std::cout << "QUEUE_RESUME_AUTO_SELECTED"
              << ",resume_model_id=" << candidates.front().modelId
              << ",completed_epochs=" << candidates.front().completedEpochs
              << std::endl;
    return candidates.front().modelId;
}




















int CompleteSchedulerProtocolCutover(
    const SchedulerOptions& options)
{
    const SchedulerProcessAbsenceEvidence evidence =
        InspectAllSchedulerDispatchProcesses();
    if (!evidence.inspectionSucceeded)
    {
        std::cerr << "SCHEDULER_PROTOCOL_CUTOVER_REJECTED"
                  << ",reason=scheduler_process_inspection_failed"
                  << ",mutations=0"
                  << std::endl;
        return 1;
    }
    if (!evidence.schedulers.empty())
    {
        std::cerr << "SCHEDULER_PROTOCOL_CUTOVER_REJECTED"
                  << ",reason=scheduler_dispatch_process_present"
                  << ",scheduler_count="
                  << evidence.schedulers.size()
                  << ",scheduler_pids=";
        for (size_t index = 0;
             index < evidence.schedulers.size();
             ++index)
        {
            if (index)
                std::cerr << "|";
            std::cerr << evidence.schedulers[index].first;
        }
        std::cerr << ",mutations=0" << std::endl;
        return 1;
    }

    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    if (!RequireSchedulerTables(transaction))
        return 2;
    SchedulerServiceComposition services{transaction};
    const auto result = services.authority.completeProtocolCutover(
        {options.schedulerExecutablePath,
         "ps_inspection_complete;active_scheduler_dispatch_processes=0"},
        [] {
            const std::optional<std::string> startIdentity =
                EA::GlobalExperimentControl::ReadProcessStartIdentity(
                    static_cast<int>(::getpid()));
            if (!startIdentity)
            {
                throw std::runtime_error(
                    "cutover_process_start_identity_unavailable");
            }
            return "pid:" + std::to_string(::getpid()) +
                   ";start:" + *startIdentity;
        });
    transaction.commit();
    if (result == EA::SchedulerCore::SchedulerProtocolCutoverResult::
                      GenerationMismatch)
    {
        std::cerr << "SCHEDULER_PROTOCOL_CUTOVER_REJECTED"
                  << ",reason=protocol_generation_mismatch"
                  << ",mutations=0"
                  << std::endl;
        return 1;
    }
    std::cout << "SCHEDULER_PROTOCOL_CUTOVER_COMPLETE"
              << ",generation="
              << EA::SchedulerCore::kSchedulerProtocolGeneration
              << ",result="
              << (result == EA::SchedulerCore::
                                SchedulerProtocolCutoverResult::AlreadyComplete
                      ? "already_complete"
                      : "completed")
              << (result == EA::SchedulerCore::
                               SchedulerProtocolCutoverResult::Completed
                      ? ",scheduler_processes=0,canonical_executable_path=" +
                            options.schedulerExecutablePath
                      : "")
              << std::endl;
    return 0;
}











void PrintModelSymbolMismatch(const std::string& runtimeSymbol,
                                     const std::string& modelSymbol)
{
    std::cerr << "MODEL_SYMBOL_MISMATCH"
              << ",runtime=" << runtimeSymbol
              << ",model=" << modelSymbol
              << std::endl;
}



std::string ResolveExperimentCanonicalSymbol(pqxx::work& w,
                                                    const SchedulerOptions& options)
{
    if (!options.resumeModelId.has_value())
        return EA::CanonicalSymbol::Normalize(*options.symbol);

    const std::optional<std::string> persistedSymbol =
        TryLoadPersistedCanonicalSymbol(w, *options.resumeModelId);
    if (persistedSymbol.has_value())
    {
        if (options.symbol.has_value())
        {
            const std::string runtimeSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
            if (runtimeSymbol != *persistedSymbol)
            {
                PrintModelSymbolMismatch(runtimeSymbol, *persistedSymbol);
                throw std::runtime_error("MODEL_SYMBOL_MISMATCH");
            }
        }
        std::cout << "MODEL_SYMBOL"
                  << ",source=database"
                  << ",model_id=" << *options.resumeModelId
                  << ",symbol=" << *persistedSymbol
                  << std::endl;
        return *persistedSymbol;
    }

    PrintModelSymbolMissing(*options.resumeModelId);
    if (options.symbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
        std::cout << "MODEL_SYMBOL"
                  << ",source=legacy"
                  << ",model_id=" << *options.resumeModelId
                  << ",symbol=" << legacySymbol
                  << ",warning=missing_metadata"
                  << std::endl;
        return legacySymbol;
    }

    throw std::runtime_error("resume-model-id is missing train_symbol_meta and --symbol was not supplied for legacy fallback");
}




std::string DuplicateWhereClause(pqxx::work& w,
                                        const SchedulerOptions& options,
                                        const std::string& canonicalSymbol)
{
    const QueuedModelInputIdentity inputIdentity =
        ResolveQueuedModelInputIdentity(w, options);
    std::ostringstream sql;
    sql << "symbol = " << w.quote(canonicalSymbol)
        << " AND prediction_horizon = " << *options.predictionHorizon
        << " AND c_next_threshold = " << FormatDouble(*options.cNextThreshold)
        << " AND core_lr_mult IS NOT DISTINCT FROM " << SqlNullable(w, options.coreLrMult)
        << " AND head_lr_mult IS NOT DISTINCT FROM " << SqlNullable(w, options.headLrMult)
        << " AND target_epochs = " << *options.targetEpochs
        << " AND checkpoint_interval = " << options.checkpointInterval
        << " AND train_start = " << w.quote(*options.trainStart) << "::timestamptz"
        << " AND train_end = " << w.quote(*options.trainEnd) << "::timestamptz"
        << " AND infer_start IS NOT DISTINCT FROM " << SqlNullable(w, options.inferStart)
        << "::timestamptz"
        << " AND infer_end IS NOT DISTINCT FROM " << SqlNullable(w, options.inferEnd)
        << "::timestamptz"
        << " AND donchian20_mode = " << w.quote(
            Donchian20ModeText(options.donchian20Mode.value_or(kDefaultDonchian20Mode)))
        << " AND feature_warmup_scope = " << w.quote(
            EA::FeatureWarmupScopeText(options.featureWarmupScope))
        << " AND donchian_lookback = " << DonchianLookbackDatabaseValue(
            options.donchianLookback)
        << " AND feature_ablation_mask = " << w.quote(options.featureAblationMask)
        << " AND fresh_initialization_seed = " << options.freshInitializationSeed
        << " AND resume_model_id IS NOT DISTINCT FROM " << SqlNullable(w, options.resumeModelId)
        << " AND resume_expand_input_width = "
        << (options.resumeExpandInputWidth ? "true" : "false")
        << " AND training_objective_hash = " << w.quote(
            EA::TrainingObjective::Identity(options.trainingObjective))
        << " AND training_objective_canonical = " << w.quote(
            EA::TrainingObjective::CanonicalText(options.trainingObjective))
        << " AND model_input_width = " << inputIdentity.width
        << " AND model_input_semantic_layout_version = "
            << inputIdentity.semanticLayoutVersion
        << " AND economic_calendar_snapshot_id IS NOT DISTINCT FROM "
        << (options.economicCalendarSnapshot
                ? std::to_string(options.economicCalendarSnapshot->snapshotId)
                : "NULL")
        << " AND economic_calendar_snapshot_hash IS NOT DISTINCT FROM "
        << (options.economicCalendarSnapshot
                ? w.quote(options.economicCalendarSnapshot->contentHash)
                : "NULL")
        << " AND status <> 'cancelled'";
    return sql.str();
}

std::string QueueDuplicateWhereClause(pqxx::work& w,
                                             const SchedulerOptions& options,
                                             const std::string& canonicalSymbol)
{
    const QueuedModelInputIdentity inputIdentity =
        ResolveQueuedModelInputIdentity(w, options);
    std::ostringstream sql;
    sql << "symbol = " << w.quote(canonicalSymbol)
        << " AND prediction_horizon = " << *options.predictionHorizon
        << " AND target_epochs = " << *options.targetEpochs
        << " AND c_next_threshold = " << FormatDouble(*options.cNextThreshold)
        << " AND donchian20_mode = " << w.quote(
            Donchian20ModeText(options.donchian20Mode.value_or(kDefaultDonchian20Mode)))
        << " AND feature_warmup_scope = " << w.quote(
            EA::FeatureWarmupScopeText(options.featureWarmupScope))
        << " AND feature_ablation_mask = " << w.quote(options.featureAblationMask)
        << " AND fresh_initialization_seed = " << options.freshInitializationSeed
        << " AND donchian_lookback = " << DonchianLookbackDatabaseValue(
            options.donchianLookback)
        << " AND resume_expand_input_width = "
        << (options.resumeExpandInputWidth ? "true" : "false")
        << " AND training_objective_hash = " << w.quote(
            EA::TrainingObjective::Identity(options.trainingObjective))
        << " AND training_objective_canonical = " << w.quote(
            EA::TrainingObjective::CanonicalText(options.trainingObjective))
        << " AND model_input_width = " << inputIdentity.width
        << " AND model_input_semantic_layout_version = "
            << inputIdentity.semanticLayoutVersion
        << " AND economic_calendar_snapshot_id IS NOT DISTINCT FROM "
        << (options.economicCalendarSnapshot
                ? std::to_string(options.economicCalendarSnapshot->snapshotId)
                : "NULL")
        << " AND economic_calendar_snapshot_hash IS NOT DISTINCT FROM "
        << (options.economicCalendarSnapshot
                ? w.quote(options.economicCalendarSnapshot->contentHash)
                : "NULL")
        << " AND train_start = " << w.quote(*options.trainStart) << "::timestamptz"
        << " AND train_end = " << w.quote(*options.trainEnd) << "::timestamptz"
        << " AND status NOT IN ('failed', 'cancelled')";
    return sql.str();
}

long long CurrentDuplicateNonce()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}


int EnqueueExperiment(const SchedulerOptions& rawOptions)
{
    SchedulerOptions options = rawOptions;
    EnsureRequiredEnqueueOptions(options);

    std::optional<std::string> dryRunSymbol;
    if (options.symbol.has_value())
        dryRunSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);

    if (options.dryRun)
    {
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        std::cout << "EXPERIMENT_ENQUEUE_DRY_RUN"
                  << ",symbol=" << (dryRunSymbol.has_value() ? *dryRunSymbol : "database_model")
                  << ",prediction_horizon=" << *options.predictionHorizon
                  << ",c_next_threshold=" << FormatDouble(*options.cNextThreshold)
                  << ",core_lr_mult=" << (options.coreLrMult.has_value() ? FormatDouble(*options.coreLrMult) : "NULL")
                  << ",head_lr_mult=" << (options.headLrMult.has_value() ? FormatDouble(*options.headLrMult) : "NULL")
                  << ",donchian20_mode=" << Donchian20ModeText(
                      options.donchian20Mode.value_or(kDefaultDonchian20Mode))
                  << ",feature_warmup_scope=" << EA::FeatureWarmupScopeText(options.featureWarmupScope)
                  << ",donchian_lookback=" << options.donchianLookback
                  << ",training_objective_id="
                  << options.trainingObjective.objectiveIdentifier
                  << ",training_objective_hash="
                  << EA::TrainingObjective::Identity(
                         options.trainingObjective)
                  << ",target_epochs=" << *options.targetEpochs
                  << ",checkpoint_interval=" << options.checkpointInterval
                  << ",train_start=" << *options.trainStart
                  << ",train_end=" << *options.trainEnd
                  << ",infer_start=" << (options.inferStart.has_value() ? *options.inferStart : "NULL")
                  << ",infer_end=" << (options.inferEnd.has_value() ? *options.inferEnd : "NULL")
                  << std::endl;
        return 0;
    }

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    w.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    SetTransactionReadWrite(w);
    w.exec("LOCK TABLE economic_calendar_snapshot IN SHARE ROW EXCLUSIVE MODE;");
    if (!RequireSchedulerTables(w))
        return 1;

    const std::string canonicalSymbol = ResolveExperimentCanonicalSymbol(w, options);
    ResolveEconomicCalendarSnapshotForQueue(
        w, options, "enqueue_experiment");

    if (!options.allowDuplicateExperiment)
    {
        pqxx::result duplicate = w.exec(
            "SELECT experiment_id FROM experiment WHERE " +
            DuplicateWhereClause(w, options, canonicalSymbol) +
            " LIMIT 1;");
        if (!duplicate.empty())
        {
            std::cout << "SCHEDULER_DUPLICATE_REJECTED"
                      << ",experiment_id=" << duplicate[0][0].as<long long>()
                      << ",symbol=" << canonicalSymbol
                      << ",prediction_horizon=" << *options.predictionHorizon
                      << std::endl;
            return 1;
        }
    }

    const long long duplicateNonce = options.allowDuplicateExperiment ? CurrentDuplicateNonce() : 0;
    const long long experimentId = InsertExperimentRecord(w, options, canonicalSymbol, duplicateNonce);
    w.commit();

    std::cout << "EXPERIMENT_ENQUEUED"
              << ",experiment_id=" << experimentId
              << ",symbol=" << canonicalSymbol
              << ",prediction_horizon=" << *options.predictionHorizon
              << ",target_epochs=" << *options.targetEpochs
              << std::endl;
    return 0;
}

SchedulerOptions ApplyQueueDefaults(SchedulerOptions options)
{
    const QueueDefaults defaults;
    if (!options.donchian20Mode.has_value())
        options.donchian20Mode = kDefaultDonchian20Mode;
    if (!options.cNextThreshold.has_value())
        options.cNextThreshold = defaults.threshold;
    if (!options.coreLrMult.has_value())
        options.coreLrMult = defaults.coreLrMult;
    if (!options.headLrMult.has_value())
        options.headLrMult = defaults.headLrMult;
    if (options.checkpointInterval <= 0)
        options.checkpointInterval = defaults.checkpointInterval;
    if (!options.trainStart.has_value())
        options.trainStart = defaults.trainStart;
    if (!options.trainEnd.has_value())
        options.trainEnd = defaults.trainEnd;
    if (!options.inferStart.has_value())
        options.inferStart = defaults.inferStart;
    if (!options.inferEnd.has_value())
        options.inferEnd = defaults.inferEnd;
    if (options.queueCheckpointInfer && !options.queueCheckpointInferInterval.has_value())
        options.queueCheckpointInferInterval = options.checkpointInterval;
    return options;
}


std::string CheckpointPolicyRuleText(const SchedulerOptions& options);

void PrintQueueConfig(const char* marker,
                             const SchedulerOptions& options,
                             const std::string& canonicalSymbol,
                             const std::optional<long long>& experimentId = std::nullopt)
{
    std::cout << marker;
    if (experimentId.has_value())
        std::cout << ",experiment_id=" << *experimentId;
    std::cout << ",symbol=" << canonicalSymbol
              << ",prediction_horizon=" << *options.predictionHorizon
              << ",target_epochs=" << *options.targetEpochs
              << ",resume_model_id=" << (options.resumeModelId.has_value() ? std::to_string(*options.resumeModelId) : "NULL")
              << ",resume_expand_input_width="
              << (options.resumeExpandInputWidth ? "1" : "0")
              << ",threshold=" << FormatDouble(*options.cNextThreshold)
              << ",core_lr=" << (options.coreLrMult.has_value() ? FormatDouble(*options.coreLrMult) : "NULL")
              << ",head_lr=" << (options.headLrMult.has_value() ? FormatDouble(*options.headLrMult) : "NULL")
              << ",donchian20_mode=" << Donchian20ModeText(
                  options.donchian20Mode.value_or(kDefaultDonchian20Mode))
              << ",feature_warmup_scope=" << EA::FeatureWarmupScopeText(options.featureWarmupScope)
              << ",donchian_lookback=" << options.donchianLookback
              << ",feature_ablation_mask=" << options.featureAblationMask
              << ",fresh_initialization_seed=" << options.freshInitializationSeed
              << ",training_objective_id="
              << options.trainingObjective.objectiveIdentifier
              << ",training_objective_hash="
              << EA::TrainingObjective::Identity(options.trainingObjective)
              << ",economic_calendar_snapshot_id="
              << (options.economicCalendarSnapshot
                      ? std::to_string(
                            options.economicCalendarSnapshot->snapshotId)
                      : "NULL")
              << ",economic_calendar_snapshot_hash="
              << (options.economicCalendarSnapshot
                      ? options.economicCalendarSnapshot->contentHash
                      : "NULL")
              << ",checkpoint_interval=" << options.checkpointInterval
              << ",checkpoint_infer=" << (options.queueCheckpointInfer ? "1" : "0")
              << ",checkpoint_infer_min_epoch="
              << (options.queueCheckpointInferMinEpoch.has_value() ? std::to_string(*options.queueCheckpointInferMinEpoch) : "NULL")
              << ",checkpoint_infer_interval="
              << (options.queueCheckpointInferInterval.has_value() ? std::to_string(*options.queueCheckpointInferInterval) : "NULL")
              << ",checkpoint_policy=" << (options.queueCheckpointPolicy ? "1" : "0")
              << ",checkpoint_policy_rules=" << CheckpointPolicyRuleText(options)
              << ",continuation_candidate_excluded="
              << (options.queueContinuationCandidateExcluded ? "1" : "0")
              << ",train_start=" << *options.trainStart
              << ",train_end=" << *options.trainEnd
              << ",infer_start=" << (options.inferStart.has_value() ? *options.inferStart : "NULL")
              << ",infer_end=" << (options.inferEnd.has_value() ? *options.inferEnd : "NULL")
              << std::endl;
}

std::optional<long long> FindQueueDuplicate(pqxx::work& w,
                                                   const SchedulerOptions& options,
                                                   const std::string& canonicalSymbol)
{
    pqxx::result duplicate = w.exec(
        "SELECT experiment_id FROM experiment WHERE " +
        QueueDuplicateWhereClause(w, options, canonicalSymbol) +
        " ORDER BY experiment_id ASC LIMIT 1;");
    if (duplicate.empty())
        return std::nullopt;
    return duplicate[0][0].as<long long>();
}

bool QueueOneExperiment(pqxx::work& w,
                               const SchedulerOptions& options,
                               const std::string& canonicalSymbol)
{
    const std::optional<long long> duplicateExperimentId =
        FindQueueDuplicate(w, options, canonicalSymbol);
    if (duplicateExperimentId.has_value())
    {
        PrintQueueConfig("QUEUE_ALREADY_EXISTS", options, canonicalSymbol, duplicateExperimentId);
        return false;
    }

    const long long experimentId = InsertExperimentRecord(w, options, canonicalSymbol, 0);
    PrintQueueConfig("QUEUE_EXPERIMENT_CREATED", options, canonicalSymbol, experimentId);
    return true;
}

int QueueExperiments(const SchedulerOptions& rawOptions)
{
    SchedulerOptions options = rawOptions;

    if (options.resumeModelId.has_value() && options.epochs.has_value())
    {
        ThrowQueueResumeInvalid("epochs_conflicts_with_absolute_target_epochs",
                                *options.resumeModelId);
    }

    if (options.dryRun && !options.resumeModelId.has_value() && !options.autoResume)
    {
        options = ApplyQueueDefaults(options);
        EnsureRequiredQueueOptions(options);
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        const auto& symbols = options.queueSweep
            ? EA::SupportedSymbols::TrainingSymbols()
            : std::vector<std::string>{EA::CanonicalSymbol::Normalize(*options.symbol)};
        for (const auto& symbol : symbols)
            PrintQueueConfig("QUEUE_EXPERIMENT_DRY_RUN", options, EA::CanonicalSymbol::Normalize(symbol));
        return 0;
    }

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    w.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    SetTransactionReadWrite(w);
    w.exec("LOCK TABLE economic_calendar_snapshot IN SHARE ROW EXCLUSIVE MODE;");
    if (!RequireSchedulerTables(w))
        return 2;

    if (options.autoResume)
    {
        options = ApplyQueueDefaults(options);
        EnsureRequiredQueueOptions(options);
        options.resumeModelId = ResolveAutoResumeModelId(w, options);
        options.autoResume = false;
    }

    if (options.resumeModelId.has_value())
    {
        if (rawOptions.coreLrMult.has_value())
            ThrowQueueResumeInvalid("core_lr_override_not_allowed_in_resume", *options.resumeModelId);
        if (rawOptions.headLrMult.has_value())
            ThrowQueueResumeInvalid("head_lr_override_not_allowed_in_resume", *options.resumeModelId);
        const QueueResumeMeta meta = LoadQueueResumeMeta(w, *options.resumeModelId);
        (void)DBIO::PgModelIO::validateModelInputSemanticsForLoad(
            w, *options.resumeModelId);
        if (options.resumeExpandInputWidth)
            DBIO::PgModelIO::validateModelInputSemanticsForExpansion(
                w, *options.resumeModelId);
        MergeResumeMetaIntoQueueOptions(options, meta);
    }

    options = ApplyQueueDefaults(options);
    EnsureRequiredQueueOptions(options);

    if (options.dryRun)
    {
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        const auto& symbols = options.queueSweep
            ? EA::SupportedSymbols::TrainingSymbols()
            : std::vector<std::string>{EA::CanonicalSymbol::Normalize(*options.symbol)};
        for (const auto& symbol : symbols)
            PrintQueueConfig("QUEUE_EXPERIMENT_DRY_RUN", options, EA::CanonicalSymbol::Normalize(symbol));
        w.commit();
        return 0;
    }

    ResolveEconomicCalendarSnapshotForQueue(
        w, options, options.queueSweep ? "queue_sweep" : "queue_experiment");

    int created = 0;
    int duplicates = 0;
    if (options.queueSweep)
    {
        for (const auto& symbol : EA::SupportedSymbols::TrainingSymbols())
        {
            SchedulerOptions perSymbol = options;
            perSymbol.symbol = EA::CanonicalSymbol::Normalize(symbol);
            if (QueueOneExperiment(w, perSymbol, *perSymbol.symbol))
                ++created;
            else
                ++duplicates;
        }
    }
    else
    {
        const std::string canonicalSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
        if (QueueOneExperiment(w, options, canonicalSymbol))
            ++created;
        else
            ++duplicates;
    }

    w.commit();
    std::cout << "QUEUE_DONE"
              << ",created=" << created
              << ",duplicates=" << duplicates
              << std::endl;
    return created > 0 ? 0 : (duplicates > 0 ? 3 : 0);
}

EA::CorrectedCausalSurpriseReplicationContinuation::Command
CorrectedReplicationCommand(const SchedulerOptions& options,
                            bool materialize)
{
    EA::CorrectedCausalSurpriseReplicationContinuation::Command command;
    command.evidencePairs = materialize
        ? *options.materializeCorrectedCausalSurpriseReplication
        : *options.correctedCausalSurpriseReplicationStatus;
    command.anchorPair = *options.correctedCausalSurpriseAnchorPair;
    return command;
}

SchedulerOptions CorrectedReplicationQueueOptions(
    const SchedulerOptions& commandOptions,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Plan& plan,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Pair& pair,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Arm& arm)
{
    const auto& configured = plan.configuration;
    SchedulerOptions options;
    options.schedulerExecutablePath = commandOptions.schedulerExecutablePath;
    options.queueExperiment = true;
    options.symbol = pair.symbol;
    options.predictionHorizon = pair.predictionHorizon;
    options.cNextThreshold = configured.threshold;
    options.coreLrMult = configured.coreLearningRateMultiplier;
    options.headLrMult = configured.headLearningRateMultiplier;
    options.targetEpochs = configured.targetEpochs;
    options.checkpointInterval = configured.checkpointInterval;
    options.trainStart = configured.trainStart;
    options.trainEnd = configured.trainEnd;
    options.inferStart = configured.inferenceStart;
    options.inferEnd = configured.inferenceEnd;
    options.donchian20Mode = ParseDonchian20Mode(configured.donchianMode);
    options.featureWarmupScope =
        EA::ParseFeatureWarmupScope(configured.featureWarmupScope);
    options.donchianLookback =
        static_cast<std::size_t>(configured.donchianLookback);
    options.featureAblationMask = arm.featureAblationMask;
    options.featureAblationMaskSpecified = true;
    options.trainingObjective =
        EA::TrainingObjective::ParseSupportedCanonicalText(
            configured.trainingObjectiveCanonical);
    options.trainingObjectiveSpecified = true;
    options.queueCheckpointInfer = configured.checkpointInferenceEnabled;
    options.queueCheckpointInferMinEpoch =
        configured.checkpointInferenceMinimumEpoch;
    options.queueCheckpointInferInterval =
        configured.checkpointInferenceInterval;
    options.queueCheckpointPolicy = configured.checkpointPolicyEnabled;
    options.checkpointPolicyMinLeaderScore =
        configured.checkpointPolicyMinimumLeaderScore;
    options.checkpointPolicyMinInferAccuracy =
        configured.checkpointPolicyMinimumInferenceAccuracy;
    options.checkpointPolicyTopN = configured.checkpointPolicyTopN;
    options.checkpointPolicyScope = configured.checkpointPolicyScope;
    options.checkpointPolicyStopMode = configured.checkpointPolicyStopMode;
    options.checkpointPolicyGraceEvals =
        configured.checkpointPolicyGraceEvaluations;
    options.queueContinuationCandidateExcluded = false;
    options.economicCalendarSnapshot =
        EA::EconomicCalendar::EconomicCalendarSnapshotIdentity{
            configured.economicCalendarSnapshotId,
            configured.economicCalendarSnapshotHash};
    options.queueInvocationMode =
        EA::CorrectedCausalSurpriseReplicationContinuation::
            MaterializationProvenance(plan, pair, arm);
    return options;
}

void ValidateCorrectedReplicationRuntimeContract(
    const EA::CorrectedCausalSurpriseReplicationContinuation::Plan& plan)
{
    const auto& configured = plan.configuration;
    if (configured.modelInputWidth !=
            static_cast<int>(EA::kCurrentModelInputWidth) ||
        configured.semanticLayoutVersion !=
            EA::kModelInputSemanticLayoutVersion ||
        configured.batchSize != static_cast<int>(batch_size) ||
        configured.baseLearningRate != 1.0e-3 / 3.0 ||
        configured.freshInitializationSeed !=
            std::optional<unsigned int>{42U})
        throw std::runtime_error(
            "corrected_replication_runtime_scientific_contract_mismatch");
}

void LockCorrectedReplicationEvidence(
    pqxx::work& transaction,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Command& command)
{
    std::set<long long> experimentIds;
    for (const auto& pair : command.evidencePairs)
    {
        experimentIds.insert(pair.first);
        experimentIds.insert(pair.second);
    }
    if (experimentIds.empty())
        throw std::invalid_argument(
            "corrected_replication_evidence_empty");
    std::ostringstream ids;
    bool first = true;
    for (const long long experimentId : experimentIds)
    {
        if (!first) ids << ',';
        first = false;
        ids << experimentId;
    }
    const pqxx::result locked = transaction.exec(
        "SELECT experiment_id FROM experiment WHERE experiment_id IN (" +
        ids.str() + ") ORDER BY experiment_id FOR SHARE;");
    if (locked.size() != experimentIds.size())
        throw std::invalid_argument(
            "corrected_replication_evidence_membership_missing");
}

struct ExistingCorrectedReplicationArm
{
    std::optional<long long> experimentId;
    bool exactPlanProvenance = false;
};

std::optional<long long> FindCorrectedReplicationArmByProvenance(
    pqxx::work& transaction,
    const SchedulerOptions& options)
{
    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id FROM experiment WHERE invocation_mode=$1 "
        "ORDER BY experiment_id;",
        pqxx::params{*options.queueInvocationMode});
    if (rows.size() > 1)
        throw std::runtime_error(
            "corrected_replication_materialization_provenance_ambiguous");
    if (rows.empty()) return std::nullopt;
    return rows.one_row()[0].as<long long>();
}

void ValidateCorrectedReplicationArm(
    pqxx::work& transaction,
    long long experimentId,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Plan& plan,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Pair& pair,
    const EA::CorrectedCausalSurpriseReplicationContinuation::Arm& arm)
{
    const auto evidence =
        EA::FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
            transaction, experimentId);
    EA::CorrectedCausalSurpriseReplicationContinuation::
        ValidatePlannedArmEvidence(plan, pair, arm, evidence);
    const pqxx::result provenance = transaction.exec(
        "SELECT invocation_mode FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (provenance.size() != 1 || provenance.one_row()[0].is_null() ||
        provenance.one_row()[0].as<std::string>() !=
            EA::CorrectedCausalSurpriseReplicationContinuation::
                MaterializationProvenance(plan, pair, arm))
        throw std::runtime_error(
            "corrected_replication_materialized_provenance_mismatch");
}

ExistingCorrectedReplicationArm FindExistingCorrectedReplicationArm(
    pqxx::work& transaction,
    const SchedulerOptions& options,
    const std::string& canonicalSymbol)
{
    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id,invocation_mode FROM experiment WHERE " +
        DuplicateWhereClause(transaction, options, canonicalSymbol) +
        " ORDER BY experiment_id;");
    if (rows.size() > 1)
        throw std::runtime_error(
            "corrected_replication_duplicate_identity_ambiguous");
    if (rows.empty()) return {};
    ExistingCorrectedReplicationArm result;
    result.experimentId = rows[0][0].as<long long>();
    result.exactPlanProvenance = !rows[0][1].is_null() &&
        rows[0][1].as<std::string>() == *options.queueInvocationMode;
    return result;
}

int RunCorrectedReplicationMaterializationAttempt(
    const SchedulerOptions& options)
{
    namespace Continuation =
        EA::CorrectedCausalSurpriseReplicationContinuation;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;");
    if (options.dryRun)
        SetTransactionReadOnly(transaction);
    else
        SetTransactionReadWrite(transaction);

    const Continuation::Command command =
        CorrectedReplicationCommand(options, true);
    if (!options.dryRun)
        LockCorrectedReplicationEvidence(transaction, command);
    const Continuation::Assessment assessment =
        Continuation::EvaluateCommand(transaction, command);
    ValidateCorrectedReplicationRuntimeContract(assessment.plan);
    std::cout << Continuation::RenderAssessment(assessment);
    if (*options.expectedCorrectedReplicationPlanHash != assessment.plan.hash)
    {
        std::cerr
            << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_REJECTED"
            << ",reason=stale_or_changed_plan_identity"
            << ",expected_plan_hash="
            << *options.expectedCorrectedReplicationPlanHash
            << ",actual_plan_hash=" << assessment.plan.hash << std::endl;
        return 3;
    }
    if (assessment.gate.nextAction !=
        Continuation::NextAction::PrepareAdditionalReplications)
    {
        std::cerr
            << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_REJECTED"
            << ",reason=continuation_gate_closed"
            << ",next_action="
            << Continuation::NextActionText(assessment.gate.nextAction)
            << std::endl;
        return assessment.gate.nextAction ==
                   Continuation::NextAction::AwaitPairCompletion
            ? 4 : 3;
    }
    const auto& pair = assessment.plan.pairs.at(
        *assessment.gate.nextPlanPairOrdinal - 1);
    const SchedulerOptions control = CorrectedReplicationQueueOptions(
        options, assessment.plan, pair, pair.control);
    const SchedulerOptions treatment = CorrectedReplicationQueueOptions(
        options, assessment.plan, pair, pair.treatment);

    if (options.dryRun)
    {
        PrintQueueConfig("CORRECTED_REPLICATION_CONTROL_DRY_RUN", control,
                         pair.symbol);
        PrintQueueConfig("CORRECTED_REPLICATION_TREATMENT_DRY_RUN", treatment,
                         pair.symbol);
        std::cout
            << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_PREVIEW"
            << ",plan_hash=" << assessment.plan.hash
            << ",pair_ordinal=" << pair.ordinal
            << ",scientific_policy_version="
            << assessment.plan.scientificPolicyVersion
            << ",replication_unit_hash=" << pair.replicationUnitHash
            << ",outcome_blind=true"
            << ",experiment_rows_created=0"
            << ",campaign_rows_modified=0"
            << ",scheduler_state_modified=false"
            << ",read_only=true" << std::endl;
        return 0;
    }
    if (!options.yes)
    {
        std::cerr
            << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_REJECTED"
            << ",reason=explicit_confirmation_required"
            << ",required_flag=--yes" << std::endl;
        return 1;
    }

    transaction.exec(
        "LOCK TABLE economic_calendar_snapshot IN SHARE ROW EXCLUSIVE MODE;");
    transaction.exec("LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE;");
    if (!RequireSchedulerTables(transaction)) return 2;
    const auto existingControl = FindExistingCorrectedReplicationArm(
        transaction, control, pair.symbol);
    const auto existingTreatment = FindExistingCorrectedReplicationArm(
        transaction, treatment, pair.symbol);
    const auto provenanceControl = FindCorrectedReplicationArmByProvenance(
        transaction, control);
    const auto provenanceTreatment = FindCorrectedReplicationArmByProvenance(
        transaction, treatment);
    if (provenanceControl != existingControl.experimentId ||
        provenanceTreatment != existingTreatment.experimentId)
        throw std::runtime_error(
            "corrected_replication_provenance_identity_disagreement");
    if (existingControl.experimentId || existingTreatment.experimentId)
    {
        if (existingControl.experimentId && existingTreatment.experimentId &&
            existingControl.exactPlanProvenance &&
            existingTreatment.exactPlanProvenance)
        {
            ValidateCorrectedReplicationArm(
                transaction, *existingControl.experimentId,
                assessment.plan, pair, pair.control);
            ValidateCorrectedReplicationArm(
                transaction, *existingTreatment.experimentId,
                assessment.plan, pair, pair.treatment);
            transaction.commit();
            std::cout
                << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_ALREADY_MATERIALIZED"
                << ",plan_hash=" << assessment.plan.hash
                << ",pair_ordinal=" << pair.ordinal
                << ",scientific_policy_version="
                << assessment.plan.scientificPolicyVersion
                << ",replication_unit_hash=" << pair.replicationUnitHash
                << ",outcome_blind=true"
                << ",control_experiment_id="
                << *existingControl.experimentId
                << ",treatment_experiment_id="
                << *existingTreatment.experimentId
                << ",experiment_rows_created=0" << std::endl;
            return 0;
        }
        std::cerr
            << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_REJECTED"
            << ",reason="
            << ((existingControl.experimentId &&
                 existingTreatment.experimentId)
                    ? "scientific_identity_owned_by_other_plan"
                    : "partial_pair_exists_requires_review")
            << ",control_experiment_id="
            << (existingControl.experimentId
                    ? std::to_string(*existingControl.experimentId) : "NULL")
            << ",treatment_experiment_id="
            << (existingTreatment.experimentId
                    ? std::to_string(*existingTreatment.experimentId) : "NULL")
            << std::endl;
        return 3;
    }

    const long long controlExperimentId = InsertExperimentRecord(
        transaction, control, pair.symbol, 0);
    const long long treatmentExperimentId = InsertExperimentRecord(
        transaction, treatment, pair.symbol, 0);
    const pqxx::result prioritized = transaction.exec_params(
        "UPDATE experiment SET scheduler_priority=$3,"
        "checkpoint_policy_revision=$4,checkpoint_policy_hash=$5,"
        "continuation_policy_enabled=false,updated_at=now() "
        "WHERE experiment_id IN ($1,$2) RETURNING experiment_id;",
        controlExperimentId,
        treatmentExperimentId,
        assessment.plan.configuration.schedulerPriority,
        assessment.plan.configuration.checkpointPolicyRevision,
        assessment.plan.configuration.checkpointPolicyHash);
    if (prioritized.size() != 2)
        throw std::runtime_error(
            "corrected_replication_pair_priority_assignment_failed");
    ValidateCorrectedReplicationArm(
        transaction, controlExperimentId, assessment.plan, pair,
        pair.control);
    ValidateCorrectedReplicationArm(
        transaction, treatmentExperimentId, assessment.plan, pair,
        pair.treatment);
    transaction.commit();
    std::cout
        << "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZED"
        << ",plan_hash=" << assessment.plan.hash
        << ",pair_ordinal=" << pair.ordinal
        << ",scientific_policy_version="
        << assessment.plan.scientificPolicyVersion
        << ",replication_unit_hash=" << pair.replicationUnitHash
        << ",outcome_blind=true"
        << ",control_experiment_id=" << controlExperimentId
        << ",treatment_experiment_id=" << treatmentExperimentId
        << ",scheduler_priority=high"
        << ",atomic_pair=true"
        << ",experiment_rows_created=2" << std::endl;
    return 0;
}

int RunCorrectedReplicationMaterializationCommand(
    const SchedulerOptions& options)
{
    constexpr int kMaximumTransactionAttempts = 3;
    if (options.dryRun)
        return RunCorrectedReplicationMaterializationAttempt(options);

    for (int attempt = 1; attempt <= kMaximumTransactionAttempts; ++attempt)
    {
        try
        {
            // Every attempt constructs a new connection and serializable
            // transaction, then repeats the complete authoritative
            // materialization workflow above.  A PostgreSQL transaction that
            // reports a serialization failure is aborted and is never reused.
            return RunCorrectedReplicationMaterializationAttempt(options);
        }
        catch (const pqxx::sql_error& error)
        {
            if (error.sqlstate() != "40001" ||
                attempt == kMaximumTransactionAttempts)
                throw;
        }
    }
    throw std::logic_error(
        "corrected_replication_materialization_retry_unreachable");
}

std::string FormatOptionalMetadataString(const pqxx::row& row, int index)
{
    return row[index].is_null() ? "unknown" : row[index].as<std::string>();
}

std::string FormatOptionalMetadataBool(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return "unknown";
    return row[index].as<bool>() ? "1" : "0";
}

void PrintEconomicCalendarSnapshotReport(
    const EA::EconomicCalendar::EconomicCalendarSnapshotReport& report,
    const char* marker)
{
    std::cout << marker
              << ",snapshot_id="
              << (report.snapshotId
                      ? std::to_string(*report.snapshotId) : "DRY_RUN")
              << ",content_hash=" << report.contentHash
              << ",canonical_event_count=" << report.canonicalEventCount
              << ",selected_consensus_count="
              << report.selectedConsensusCount
              << ",release_actual_count=" << report.releaseActualCount
              << ",proven_first_release_actual_count="
              << report.provenFirstReleaseActualCount
              << ",provenance_unavailable_count="
              << report.provenanceUnavailableCount
              << ",ambiguous_first_release_count="
              << report.ambiguousFirstReleaseCount
              << ",source_family_counts=" << report.sourceFamilyCountsJson
              << ",reused=" << (report.reused ? 1 : 0)
              << ",dry_run=" << (report.dryRun ? 1 : 0)
              << std::endl;
}

int CreateEconomicCalendarSnapshotCommand(const SchedulerOptions& options)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    if (options.dryRun)
        SetTransactionReadOnly(transaction);
    else
    {
        SetTransactionReadWrite(transaction);
        transaction.exec(
            "LOCK TABLE economic_calendar_snapshot "
            "IN SHARE ROW EXCLUSIVE MODE;");
    }
    const auto report =
        EA::EconomicCalendar::CreateOrReuseEconomicCalendarSnapshot(
            transaction, "manual_cli", std::nullopt, options.dryRun);
    transaction.commit();
    PrintEconomicCalendarSnapshotReport(
        report, options.dryRun
            ? "ECONOMIC_CALENDAR_SNAPSHOT_DRY_RUN"
            : "ECONOMIC_CALENDAR_SNAPSHOT_FINALIZED");
    return 0;
}

int PrintExperimentMetadata(long long experimentId)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    if (!EA::RunMetadata::ExperimentRunMetadataColumnsExist(w))
    {
        pqxx::result exists = w.exec_params(
            "SELECT 1 FROM experiment WHERE experiment_id = $1 LIMIT 1;",
            experimentId);
        if (exists.empty())
        {
            std::cerr << "EXPERIMENT_METADATA_FAILED"
                      << ",experiment_id=" << experimentId
                      << ",reason=not_found"
                      << std::endl;
            return 1;
        }
        std::cout << "EXPERIMENT_METADATA"
                  << ",experiment_id=" << experimentId
                  << ",metadata_available=0"
                  << ",reason=migration_required"
                  << std::endl;
        return 0;
    }

    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, status, phase, feature_ablation_mask, "
        "git_commit, git_branch, git_dirty, build_config, compiler_version, "
        "schema_version, scheduler_version, binary_name, invocation_mode, "
        "run_metadata_captured_at::text,economic_calendar_snapshot_id,"
        "economic_calendar_snapshot_hash "
        "FROM experiment WHERE experiment_id = $1 LIMIT 1;",
        experimentId);
    if (rows.empty())
    {
        std::cerr << "EXPERIMENT_METADATA_FAILED"
                  << ",experiment_id=" << experimentId
                  << ",reason=not_found"
                  << std::endl;
        return 1;
    }

    const auto& row = rows[0];
    std::cout << "EXPERIMENT_METADATA"
              << ",experiment_id=" << row[0].as<long long>()
              << ",symbol=" << row[1].as<std::string>()
              << ",prediction_horizon=" << row[2].as<int>()
              << ",status=" << row[3].as<std::string>()
              << ",phase=" << row[4].as<std::string>()
              << ",feature_ablation_mask=" << row[5].as<std::string>()
              << ",git_commit=" << FormatOptionalMetadataString(row, 6)
              << ",git_branch=" << FormatOptionalMetadataString(row, 7)
              << ",dirty=" << FormatOptionalMetadataBool(row, 8)
              << ",build_config=" << FormatOptionalMetadataString(row, 9)
              << ",compiler_version=" << FormatOptionalMetadataString(row, 10)
              << ",schema_version=" << FormatOptionalMetadataString(row, 11)
              << ",scheduler_version=" << FormatOptionalMetadataString(row, 12)
              << ",binary_name=" << FormatOptionalMetadataString(row, 13)
              << ",invocation_mode=" << FormatOptionalMetadataString(row, 14)
              << ",captured_at=" << FormatOptionalMetadataString(row, 15)
              << ",economic_calendar_snapshot_id="
              << (row[16].is_null() ? "NULL" :
                  std::to_string(row[16].as<long long>()))
              << ",economic_calendar_snapshot_hash="
              << (row[17].is_null() ? "NULL" :
                  row[17].as<std::string>())
              << ",economic_calendar_behavior="
              << (row[16].is_null() ? "legacy_live_corpus" :
                  "immutable_snapshot")
              << std::endl;
    return 0;
}

long long CountExperimentMetadataBackfillEligible(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT count(*) "
        "FROM experiment "
        "WHERE run_metadata_captured_at IS NULL "
        "AND status IN ('pending', 'running');");
    return rows[0][0].as<long long>();
}

int BackfillExperimentMetadata(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    if (!EA::RunMetadata::ExperimentRunMetadataColumnsExist(w))
    {
        std::cerr << "EXPERIMENT_METADATA_BACKFILL_FAILED"
                  << ",reason=migration_required"
                  << std::endl;
        return 1;
    }

    const long long eligible = CountExperimentMetadataBackfillEligible(w);
    const EA::RunMetadata::Snapshot metadata =
        EA::RunMetadata::Capture(options.schedulerExecutablePath, "metadata_backfill");
    const std::string schemaVersion = EA::RunMetadata::CurrentSchemaVersion(w);
    const long long updated = EA::RunMetadata::BackfillMissingExperimentRunMetadataWithCount(
        w, options.schedulerExecutablePath, "metadata_backfill");
    w.commit();

    std::cout << "EXPERIMENT_METADATA_BACKFILL"
              << ",updated=" << updated
              << ",eligible=" << eligible
              << ",schema_version=" << schemaVersion
              << ",dirty=" << EA::RunMetadata::SqlNullableBool(metadata.gitDirty)
              << std::endl;
    return 0;
}

struct BackupManifestStats
{
    std::string schemaVersion = "unknown";
    std::optional<long long> modelCount;
    std::optional<long long> experimentCount;
    std::optional<long long> analysisCount;
};

BackupManifestStats LoadBackupManifestStats()
{
    BackupManifestStats stats;
    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        stats.schemaVersion = EA::RunMetadata::CurrentSchemaVersion(w);
        stats.modelCount = CountRowsIfTableExists(w, "model");
        stats.experimentCount = CountRowsIfTableExists(w, "experiment");
        stats.analysisCount = CountRowsIfTableExists(w, "experiment_analysis_result");
        w.commit();
    }
    catch (const std::exception& e)
    {
        std::cerr << "DATABASE_BACKUP_MANIFEST_WARNING"
                  << ",reason=metadata_query_failed"
                  << ",error=" << JsonString(e.what())
                  << std::endl;
    }
    return stats;
}

bool WriteBackupManifest(const std::filesystem::path& manifestPath,
                         const std::string& database,
                         const std::filesystem::path& backupPath,
                         uintmax_t sizeBytes,
                         const std::string& gitCommit)
{
    const BackupManifestStats stats = LoadBackupManifestStats();
    std::ofstream out{manifestPath};
    if (!out)
        return false;

    out << "{\n"
        << "  \"database\": " << JsonString(database) << ",\n"
        << "  \"backup_path\": " << JsonString(backupPath.string()) << ",\n"
        << "  \"size_bytes\": " << sizeBytes << ",\n"
        << "  \"git_commit\": " << JsonString(gitCommit) << ",\n"
        << "  \"created_at\": " << JsonString(EA::RunMetadata::CurrentUtcTimestamp()) << ",\n"
        << "  \"format\": \"custom\",\n"
        << "  \"includes_schema\": true,\n"
        << "  \"includes_data\": true,\n"
        << "  \"schema_version\": " << JsonString(stats.schemaVersion) << ",\n"
        << "  \"model_count\": " << JsonOptionalLongLong(stats.modelCount) << ",\n"
        << "  \"experiment_count\": " << JsonOptionalLongLong(stats.experimentCount) << ",\n"
        << "  \"analysis_count\": " << JsonOptionalLongLong(stats.analysisCount) << "\n"
        << "}\n";
    return static_cast<bool>(out);
}

int BackupDatabase(const SchedulerOptions& options)
{
    const std::string database = GetEnvOrDefault("LSTM_DB_NAME", "LSTM");
    const std::string host = GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1");
    const std::string user = GetEnvOrDefault("LSTM_DB_USER", "pqxx");
    const std::string gitCommit = ShortGitCommit();
    const bool overwrite = options.backupOutputPath.has_value();

    std::filesystem::path backupPath;
    if (overwrite)
    {
        backupPath = std::filesystem::path{*options.backupOutputPath};
        if (backupPath.empty() || backupPath.filename().empty())
        {
            std::cerr << "DATABASE_BACKUP_FAILED"
                      << ",database=" << database
                      << ",path=" << backupPath.string()
                      << ",reason=invalid_output_path"
                      << std::endl;
            return 1;
        }
    }
    else
    {
        const std::filesystem::path backupDir{"Database/backups"};
        backupPath = backupDir / (database + "_" + CurrentLocalFilenameTimestamp() + "_" + gitCommit + ".dump");
    }

    const std::filesystem::path parent = backupPath.parent_path();
    if (!parent.empty())
    {
        std::error_code mkdirEc;
        std::filesystem::create_directories(parent, mkdirEc);
        if (mkdirEc)
        {
            std::cerr << "DATABASE_BACKUP_FAILED"
                      << ",database=" << database
                      << ",path=" << backupPath.string()
                      << ",reason=parent_directory_create_failed"
                      << ",error=" << JsonString(mkdirEc.message())
                      << std::endl;
            return 1;
        }
    }

    std::error_code dirEc;
    if (std::filesystem::is_directory(backupPath, dirEc))
    {
        std::cerr << "DATABASE_BACKUP_FAILED"
                  << ",database=" << database
                  << ",path=" << backupPath.string()
                  << ",reason=output_path_is_directory"
                  << std::endl;
        return 1;
    }

    const std::filesystem::path manifestPath{backupPath.string() + ".json"};

    std::cout << "DATABASE_BACKUP_STARTED"
              << ",database=" << database
              << ",path=" << backupPath.string()
              << ",git_commit=" << gitCommit
              << ",format=custom"
              << ",includes_schema=1"
              << ",includes_data=1"
              << ",overwrite=" << (overwrite ? 1 : 0)
              << std::endl;

    const std::vector<std::string> args = {
        "pg_dump",
        "-h", host,
        "-U", user,
        "-d", database,
        "-Fc",
        "-f", backupPath.string()
    };
    const int rc = RunProcessAndWait(args);
    if (rc != 0)
    {
        std::cerr << "DATABASE_BACKUP_FAILED"
                  << ",database=" << database
                  << ",path=" << backupPath.string()
                  << ",exit_code=" << rc;
        if (rc == 127)
            std::cerr << ",reason=pg_dump_not_found";
        else
            std::cerr << ",reason=pg_dump_failed";
        std::cerr << std::endl;
        return 1;
    }

    std::error_code ec;
    const uintmax_t sizeBytes = std::filesystem::file_size(backupPath, ec);
    if (ec)
        throw std::runtime_error("database backup created but file size is unavailable: " + ec.message());

    if (!WriteBackupManifest(manifestPath, database, backupPath, sizeBytes, gitCommit))
    {
        std::cerr << "DATABASE_BACKUP_FAILED"
                  << ",database=" << database
                  << ",path=" << backupPath.string()
                  << ",manifest_path=" << manifestPath.string()
                  << ",reason=manifest_write_failed"
                  << std::endl;
        return 1;
    }

    std::cout << "DATABASE_BACKUP_COMPLETE"
              << ",database=" << database
              << ",path=" << backupPath.string()
              << ",manifest_path=" << manifestPath.string()
              << ",size_bytes=" << sizeBytes
              << ",git_commit=" << gitCommit
              << ",format=custom"
              << ",includes_schema=1"
              << ",includes_data=1"
              << ",overwrite=" << (overwrite ? 1 : 0)
              << std::endl;
    return 0;
}











EA::SchedulerCore::ExperimentTransitionAction SchedulerControlAction(
    const SchedulerOptions& options)
{
    if (options.cancelExperimentId.has_value())
        return EA::SchedulerCore::ExperimentTransitionAction::Cancel;
    if (options.retryFailedExperimentId.has_value())
        return EA::SchedulerCore::ExperimentTransitionAction::RetryFailed;
    if (options.requeueTrainingExperimentId.has_value())
        return EA::SchedulerCore::ExperimentTransitionAction::RequeueTraining;
    if (options.requeueAnalysisExperimentId.has_value())
        return EA::SchedulerCore::ExperimentTransitionAction::RequeueAnalysis;
    if (options.requeueInferenceExperimentId.has_value())
        return EA::SchedulerCore::ExperimentTransitionAction::RequeueInference;
    throw std::invalid_argument("missing scheduler transition action");
}

long long SchedulerControlExperimentId(const SchedulerOptions& options)
{
    if (options.pauseExperimentId.has_value())
        return *options.pauseExperimentId;
    if (options.resumeExperimentId.has_value())
        return *options.resumeExperimentId;
    if (options.cancelExperimentId.has_value())
        return *options.cancelExperimentId;
    if (options.retryFailedExperimentId.has_value())
        return *options.retryFailedExperimentId;
    if (options.requeueTrainingExperimentId.has_value())
        return *options.requeueTrainingExperimentId;
    if (options.requeueAnalysisExperimentId.has_value())
        return *options.requeueAnalysisExperimentId;
    if (options.requeueInferenceExperimentId.has_value())
        return *options.requeueInferenceExperimentId;
    throw std::invalid_argument("missing scheduler control experiment id");
}

ExperimentRow SchedulerControlCheckpointExperiment(
    const EA::SchedulerCore::ExperimentTransitionRecord& row)
{
    const auto& record = row.experiment;
    ExperimentRow experiment;
    experiment.experimentId = record.experimentId;
    experiment.symbol = record.symbol;
    experiment.predictionHorizon = record.predictionHorizon;
    experiment.cNextThreshold = record.cNextThreshold;
    experiment.coreLrMult = record.coreLrMult;
    experiment.headLrMult = record.headLrMult;
    experiment.targetEpochs = record.targetEpochs;
    experiment.trainStart = record.trainStart;
    experiment.trainEnd = record.trainEnd;
    experiment.lastModelId = record.lastModelId;
    experiment.resumeModelId = record.resumeModelId;
    experiment.donchian20Mode = ParseDonchian20Mode(record.donchian20Mode);
    experiment.featureWarmupScope =
        EA::ParseFeatureWarmupScope(record.featureWarmupScope);
    experiment.donchianLookback =
        ParseDonchianLookback(record.donchianLookback);
    experiment.featureAblationMask = EA::FeatureAblationMask::Parse(
        record.featureAblationMask).CanonicalText();
    experiment.trainingObjective = EA::TrainingObjective::ResolvePersisted(
        record.trainingObjectiveCanonical,
        record.trainingObjectiveHash);
    experiment.schedulerPriority = row.schedulerPriority;
    experiment.schedulerResumeOrigin = row.schedulerResumeOrigin;
    experiment.activeWorkerAttemptId = row.activeWorkerAttemptId;
    return experiment;
}

QueueResumeCompatibilityRequirements RetryCheckpointRequirements(
    const EA::SchedulerCore::ExperimentTransitionRecord& row)
{
    const auto& experiment = row.experiment;
    return QueueResumeCompatibilityRequirements{
        experiment.targetEpochs,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.trainStart,
        experiment.trainEnd,
        experiment.coreLrMult.value_or(default_core_lr_mult),
        experiment.headLrMult.value_or(default_head_weight_lr_mult),
        ParseDonchian20Mode(experiment.donchian20Mode),
        EA::ParseFeatureWarmupScope(experiment.featureWarmupScope),
        ParseDonchianLookback(experiment.donchianLookback),
        EA::FeatureAblationMask::Parse(
            experiment.featureAblationMask).CanonicalText(),
        EA::TrainingObjective::ResolvePersisted(
            experiment.trainingObjectiveCanonical,
            experiment.trainingObjectiveHash)
    };
}

std::optional<QueueResumeMeta> TryLoadCompatibleRetryResumeMeta(
    pqxx::work& w,
    long long modelId,
    const QueueResumeCompatibilityRequirements& requirements)
{
    try
    {
        if (!DBIO::PgModelIO::hasTrainingResumeState(w, modelId))
            return std::nullopt;
        const QueueResumeMeta meta = LoadQueueResumeMeta(w, modelId);
        if (QueueResumeCompatibilityFailure(meta, requirements).has_value())
            return std::nullopt;
        return meta;
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

EA::SchedulerCore::RetryTrainingCheckpointSelection
SelectRetryTrainingCheckpoint(
    pqxx::work& w,
    const EA::SchedulerCore::ExperimentTransitionRecord& row)
{
    EA::SchedulerCore::RetryTrainingCheckpointSelection selection;
    selection.previousResumeModelId = row.experiment.resumeModelId;
    selection.selectedResumeModelId = row.experiment.resumeModelId;

    const QueueResumeCompatibilityRequirements requirements =
        RetryCheckpointRequirements(row);
    const std::optional<long long> effectiveExistingResume =
        row.experiment.resumeModelId.has_value()
            ? row.experiment.resumeModelId
            : row.experiment.lastModelId;
    std::optional<QueueResumeMeta> existingMeta;
    if (effectiveExistingResume.has_value())
    {
        existingMeta = TryLoadCompatibleRetryResumeMeta(
            w, *effectiveExistingResume, requirements);
    }
    if (row.experiment.resumeModelId.has_value() && existingMeta.has_value())
        selection.selectedCompletedEpoch = existingMeta->completedEpochs;

    const pqxx::result candidates = w.exec_params(
        "SELECT model_id FROM model "
        "WHERE experiment_id=$1 "
        "AND COALESCE(comment,'') ILIKE '%periodic training checkpoint%' "
        "ORDER BY model_id DESC;",
        row.experiment.experimentId);

    std::optional<QueueResumeMeta> bestCandidate;
    for (const auto& candidate : candidates)
    {
        const std::optional<QueueResumeMeta> meta =
            TryLoadCompatibleRetryResumeMeta(
                w, candidate[0].as<long long>(), requirements);
        if (!meta.has_value())
            continue;
        if (!bestCandidate.has_value() ||
            meta->completedEpochs > bestCandidate->completedEpochs ||
            (meta->completedEpochs == bestCandidate->completedEpochs &&
             meta->modelId > bestCandidate->modelId))
        {
            bestCandidate = meta;
        }
    }

    if (!bestCandidate.has_value())
    {
        selection.reason = "no_compatible_checkpoint";
        return selection;
    }

    if (existingMeta.has_value() &&
        bestCandidate->completedEpochs <= existingMeta->completedEpochs)
    {
        selection.reason = "no_newer_compatible_checkpoint";
        return selection;
    }

    selection.selectedResumeModelId = bestCandidate->modelId;
    selection.selectedCompletedEpoch = bestCandidate->completedEpochs;
    selection.promoted =
        row.experiment.resumeModelId != bestCandidate->modelId;
    selection.reason = selection.promoted ? "newer_compatible_checkpoint" :
                                            "existing_resume_already_selected";
    return selection;
}

class SchedulerTransitionCheckpointSelector final
    : public EA::SchedulerCore::ExperimentTransitionCheckpointSelector
{
public:
    explicit SchedulerTransitionCheckpointSelector(pqxx::work& transaction)
        : transaction_{transaction}
    {
    }

    EA::SchedulerCore::RetryTrainingCheckpointSelection
    selectRetryTrainingCheckpoint(
        const EA::SchedulerCore::ExperimentTransitionRecord& experiment)
        override
    {
        return SelectRetryTrainingCheckpoint(transaction_, experiment);
    }

    EA::SchedulerCore::RequeueTrainingCheckpointSelection
    selectRequeueTrainingCheckpoint(
        const EA::SchedulerCore::ExperimentTransitionRecord& experiment)
        override
    {
        const TrainingCheckpointSelection selection =
            SelectUsableTrainingCheckpoint(
                transaction_,
                SchedulerControlCheckpointExperiment(experiment),
                std::nullopt,
                false);
        if (!selection.checkpoint)
            return {std::nullopt, std::nullopt, selection.reason};
        return {
            selection.checkpoint->modelId,
            selection.checkpoint->completedEpochs,
            selection.reason};
    }

private:
    pqxx::work& transaction_;
};

int RunSchedulerControlCommand(const SchedulerOptions& options)
{
    const long long experimentId = SchedulerControlExperimentId(options);
    if (options.pauseExperimentId.has_value())
    {
        EA::GlobalExperimentControl::ExperimentPauseCommand command;
        command.experimentId = experimentId;
        command.dryRun = options.dryRun;
        command.confirmed = options.yes;
        return EA::GlobalExperimentControl::RunExperimentPauseCommand(
            LstmDbConnectionString(), command, std::cout, std::cerr);
    }
    if (options.resumeExperimentId.has_value())
    {
        EA::GlobalExperimentControl::ExperimentResumeCommand command;
        command.experimentId = experimentId;
        command.dryRun = options.dryRun;
        command.confirmed = options.yes;
        const auto invocationStarted =
            std::chrono::system_clock::now().time_since_epoch();
        command.invocationIdentity =
            std::string{"pid:"} + std::to_string(::getpid()) +
            ";started_ns:" +
            std::to_string(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    invocationStarted)
                    .count()) +
            ";executable:" + options.schedulerExecutablePath;
        return EA::GlobalExperimentControl::RunExperimentResumeCommand(
            LstmDbConnectionString(), command, std::cout, std::cerr);
    }
    const bool willApply = options.yes && !options.dryRun;

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (willApply)
        SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 2;

    EA::SchedulerCore::PostgresSchedulerRepository repository{w};
    SchedulerTransitionCheckpointSelector checkpointSelector{w};
    EA::SchedulerCore::ExperimentTransitionService service{
        repository, checkpointSelector, std::cout};
    const int result = service.run({
        SchedulerControlAction(options),
        experimentId,
        options.dryRun,
        options.yes});
    w.commit();
    return result;
}

int RunCampaignMaterializationSchedulerControlCommand(
    const SchedulerOptions& options)
{
    const bool pause = options.pauseCampaignMaterializationId.has_value();
    EA::GlobalExperimentControl::CampaignMaterializationControlCommand command;
    command.materializationId = pause
        ? *options.pauseCampaignMaterializationId
        : *options.resumeCampaignMaterializationId;
    command.dryRun = options.dryRun;
    command.confirmed = options.yes;
    const auto invocationStarted =
        std::chrono::system_clock::now().time_since_epoch();
    command.invocationIdentity =
        std::string{"pid:"} + std::to_string(::getpid()) +
        ";started_ns:" +
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                invocationStarted).count()) +
        ";action:" + (pause ? "pause" : "resume") +
        ";materialization:" + std::to_string(command.materializationId) +
        ";executable:" + options.schedulerExecutablePath;
    if (pause)
        return EA::GlobalExperimentControl::
            RunCampaignMaterializationPauseCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
    return EA::GlobalExperimentControl::
        RunCampaignMaterializationResumeCommand(
            LstmDbConnectionString(), command, std::cout, std::cerr);
}

int RunSetExperimentPriorityCommand(const SchedulerOptions& options)
{
    const auto& [experimentId, priority] = *options.setExperimentPriority;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    EA::GlobalExperimentControl::AcquireCoordinationLock(transaction);
    const pqxx::result updated = transaction.exec_params(
        "UPDATE experiment SET scheduler_priority=$1,"
        "updated_at=clock_timestamp() WHERE experiment_id=$2 "
        "RETURNING status,phase,resume_requested;",
        priority,
        experimentId);
    if (updated.empty())
    {
        std::cout << "SCHEDULER_PRIORITY_REJECTED,experiment_id="
                  << experimentId << ",reason=experiment_not_found\n";
        transaction.commit();
        return 1;
    }
    transaction.commit();
    std::cout << "SCHEDULER_PRIORITY_SET,experiment_id=" << experimentId
              << ",priority=" << priority
              << ",status=" << updated[0][0].as<std::string>()
              << ",phase=" << updated[0][1].as<std::string>()
              << ",resume_requested="
              << (updated[0][2].as<bool>() ? "true" : "false")
              << std::endl;
    return 0;
}

int RunRetryCheckpointEvalCommand(const SchedulerOptions& options)
{
    if (!options.retryCheckpointEvalId.has_value())
        throw std::invalid_argument("missing checkpoint_eval_id for retry");

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!options.dryRun)
        SetTransactionReadWrite(w);
    if (!TableExists(w, "experiment_checkpoint_eval"))
    {
        std::cerr << "DATABASE_MIGRATION_REQUIRED"
                  << ",missing=experiment_checkpoint_eval"
                  << ",command=./migrate_lstm_db.sh"
                  << std::endl;
        return 2;
    }

    std::ostringstream selectSql;
    selectSql
        << "SELECT checkpoint_eval_id, COALESCE(parent_experiment_id, experiment_id), "
        << "checkpoint_model_id, checkpoint_epoch, status, phase "
        << "FROM experiment_checkpoint_eval "
        << "WHERE checkpoint_eval_id = $1";
    if (!options.dryRun)
        selectSql << " FOR UPDATE";
    selectSql << ";";
    pqxx::result rows = w.exec_params(selectSql.str(), *options.retryCheckpointEvalId);
    if (rows.empty())
    {
        std::cerr << "CHECKPOINT_EVAL_RETRY_REJECTED"
                  << ",checkpoint_eval_id=" << *options.retryCheckpointEvalId
                  << ",reason=checkpoint_eval_not_found"
                  << std::endl;
        w.commit();
        return 1;
    }

    const long long checkpointEvalId = rows[0][0].as<long long>();
    const long long parentExperimentId = rows[0][1].as<long long>();
    const long long checkpointModelId = rows[0][2].as<long long>();
    const int checkpointEpoch = rows[0][3].as<int>();
    const std::string status = rows[0][4].as<std::string>();
    const std::string phase = rows[0][5].as<std::string>();
    if (status != "failed")
    {
        std::cerr << "CHECKPOINT_EVAL_RETRY_REJECTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << ",parent_experiment_id=" << parentExperimentId
                  << ",checkpoint_model_id=" << checkpointModelId
                  << ",checkpoint_epoch=" << checkpointEpoch
                  << ",status=" << status
                  << ",phase=" << phase
                  << ",reason=retry_requires_failed_status"
                  << std::endl;
        w.commit();
        return 1;
    }
    if (phase != "infer")
    {
        std::cerr << "CHECKPOINT_EVAL_RETRY_REJECTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << ",parent_experiment_id=" << parentExperimentId
                  << ",checkpoint_model_id=" << checkpointModelId
                  << ",checkpoint_epoch=" << checkpointEpoch
                  << ",status=" << status
                  << ",phase=" << phase
                  << ",reason=retry_requires_failed_infer_phase"
                  << std::endl;
        w.commit();
        return 1;
    }

    if (!options.dryRun)
    {
        w.exec_params(
            "UPDATE experiment_checkpoint_eval "
            "SET status = 'pending', phase = 'infer', worker_pid = NULL, "
            "started_at = NULL, completed_at = NULL, infer_started_at = NULL, "
            "infer_completed_at = NULL, analyze_started_at = NULL, analyze_completed_at = NULL, "
            "analysis_id = NULL, analysis_log_path = NULL, error_message = NULL, updated_at = now() "
            "WHERE checkpoint_eval_id = $1;",
            checkpointEvalId);
    }
    w.commit();

    std::cout << "CHECKPOINT_EVAL_RETRY_REQUESTED"
              << ",checkpoint_eval_id=" << checkpointEvalId
              << ",parent_experiment_id=" << parentExperimentId
              << ",checkpoint_model_id=" << checkpointModelId
              << ",checkpoint_epoch=" << checkpointEpoch
              << ",model_id=" << checkpointModelId
              << ",inference_scope=checkpoint"
              << ",status=" << (options.dryRun ? "dry_run" : "pending")
              << std::endl;
    return 0;
}

bool HasCheckpointControlCommand(const SchedulerOptions& options)
{
    return options.stopAfterCheckpoint.has_value() ||
           options.clearStopAfterCheckpointExperimentId.has_value() ||
           options.stopAfterCheckpointAllEpoch.has_value() ||
           options.clearStopAfterCheckpointAll ||
           options.enableCheckpointInferExperimentId.has_value() ||
           options.disableCheckpointInferExperimentId.has_value() ||
           options.checkpointInferMinEpoch.has_value() ||
           options.checkpointInferInterval.has_value() ||
           options.enableCheckpointPolicyExperimentId.has_value() ||
           options.disableCheckpointPolicyExperimentId.has_value() ||
           options.setCheckpointPolicy.has_value();
}

long long CheckpointControlExperimentId(const SchedulerOptions& options)
{
    if (options.stopAfterCheckpoint.has_value())
        return options.stopAfterCheckpoint->first;
    if (options.clearStopAfterCheckpointExperimentId.has_value())
        return *options.clearStopAfterCheckpointExperimentId;
    if (options.stopAfterCheckpointAllEpoch.has_value() || options.clearStopAfterCheckpointAll)
        return -1;
    if (options.enableCheckpointInferExperimentId.has_value())
        return *options.enableCheckpointInferExperimentId;
    if (options.disableCheckpointInferExperimentId.has_value())
        return *options.disableCheckpointInferExperimentId;
    if (options.checkpointInferMinEpoch.has_value())
        return options.checkpointInferMinEpoch->first;
    if (options.checkpointInferInterval.has_value())
        return options.checkpointInferInterval->first;
    if (options.enableCheckpointPolicyExperimentId.has_value())
        return *options.enableCheckpointPolicyExperimentId;
    if (options.disableCheckpointPolicyExperimentId.has_value())
        return *options.disableCheckpointPolicyExperimentId;
    if (options.setCheckpointPolicy.has_value())
        return options.setCheckpointPolicy->first;
    throw std::invalid_argument("missing checkpoint control experiment id");
}

int EffectiveCheckpointStopEpoch(int requestedEpoch, int checkpointInterval, int targetEpochs)
{
    if (checkpointInterval <= 0)
        return requestedEpoch;
    int effective = ((requestedEpoch + checkpointInterval - 1) / checkpointInterval) * checkpointInterval;
    if (targetEpochs > 0 && effective > targetEpochs)
        effective = targetEpochs;
    return effective;
}

std::vector<std::string> SplitCommaSeparated(const std::string& value)
{
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, ','))
    {
        if (!part.empty())
            parts.push_back(part);
    }
    return parts;
}

SchedulerOptions ParseCheckpointPolicyUpdateOptions(const std::string& config)
{
    SchedulerOptions update;
    for (const std::string& item : SplitCommaSeparated(config))
    {
        const size_t equals = item.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= item.size())
            throw std::invalid_argument("--set-checkpoint-policy requires key=value pairs");
        const std::string key = item.substr(0, equals);
        const std::string value = item.substr(equals + 1);
        if (key == "min_leader_score")
        {
            update.checkpointPolicyMinLeaderScore = ParsePositiveFiniteDouble(key, value);
            update.checkpointPolicySetKeys.insert(key);
        }
        else if (key == "min_infer_accuracy")
        {
            update.checkpointPolicyMinInferAccuracy = ParsePositiveFiniteDouble(key, value);
            update.checkpointPolicySetKeys.insert(key);
        }
        else if (key == "top_n")
        {
            update.checkpointPolicyTopN = ParsePositiveInt(key, value);
            update.checkpointPolicySetKeys.insert(key);
        }
        else if (key == "scope")
        {
            update.checkpointPolicyScope = value;
            update.checkpointPolicySetKeys.insert(key);
        }
        else if (key == "stop_mode")
        {
            update.checkpointPolicyStopMode = value;
            update.checkpointPolicySetKeys.insert(key);
        }
        else if (key == "grace_evals")
        {
            update.checkpointPolicyGraceEvals = ParsePositiveInt(key, value);
            update.checkpointPolicySetKeys.insert(key);
        }
        else
            throw std::invalid_argument("unsupported checkpoint policy key '" + key + "'");
    }
    ValidateCheckpointPolicyConfig(update);
    return update;
}

std::string CheckpointPolicyRuleText(const SchedulerOptions& options)
{
    std::ostringstream out;
    bool any = false;
    if (options.checkpointPolicyMinLeaderScore.has_value())
    {
        out << "leader>=" << FormatDouble(*options.checkpointPolicyMinLeaderScore);
        any = true;
    }
    if (options.checkpointPolicyMinInferAccuracy.has_value())
    {
        if (any)
            out << ";";
        out << "infer>=" << FormatDouble(*options.checkpointPolicyMinInferAccuracy);
        any = true;
    }
    if (options.checkpointPolicyTopN.has_value())
    {
        if (any)
            out << ";";
        out << "top_n=" << *options.checkpointPolicyTopN;
        any = true;
    }
    if (!any)
        out << "none";
    out << ";scope=" << options.checkpointPolicyScope
        << ";stop_mode=" << options.checkpointPolicyStopMode
        << ";grace_evals=" << options.checkpointPolicyGraceEvals;
    return out.str();
}

std::string CheckpointPolicyRuleText(const CheckpointPolicyConfig& config)
{
    std::ostringstream out;
    bool any = false;
    if (config.minLeaderScore.has_value())
    {
        out << "leader_score>=" << FormatDouble(*config.minLeaderScore);
        any = true;
    }
    if (config.minInferAccuracy.has_value())
    {
        if (any)
            out << "|";
        out << "infer_accuracy>=" << FormatDouble(*config.minInferAccuracy);
        any = true;
    }
    if (config.topN.has_value())
    {
        if (any)
            out << "|";
        out << "top_n<=" << *config.topN;
        any = true;
    }
    if (!any)
        out << "none";
    out << "|scope=" << config.scope
        << "|stop_mode=" << config.stopMode
        << "|grace_evals=" << config.graceEvals;
    return out.str();
}

CheckpointPolicyConfig LoadCheckpointPolicyControlConfig(
    pqxx::work& w,
    long long experimentId,
    bool hasCheckpointInferEnabled,
    bool hasOpportunisticCheckpointInfer)
{
    std::ostringstream sql;
    sql << "SELECT checkpoint_policy_enabled, "
        << "checkpoint_policy_min_leader_score, checkpoint_policy_min_infer_accuracy, "
        << "checkpoint_policy_top_n, checkpoint_policy_scope, checkpoint_policy_stop_mode, "
        << "checkpoint_policy_grace_evals, checkpoint_interval, target_epochs, current_epoch, "
        << "stop_after_checkpoint_epoch, status, phase, (";
    bool hasInferExpression = false;
    if (hasCheckpointInferEnabled)
    {
        sql << "checkpoint_infer_enabled";
        hasInferExpression = true;
    }
    if (hasOpportunisticCheckpointInfer)
    {
        if (hasInferExpression)
            sql << " OR ";
        sql << "opportunistic_checkpoint_infer";
        hasInferExpression = true;
    }
    if (!hasInferExpression)
        sql << "false";
    sql << ") AS checkpoint_infer_active, checkpoint_policy_revision, "
        << "checkpoint_policy_hash, active_scheduler_worker_attempt_id "
        << "FROM experiment WHERE experiment_id = $1 FOR UPDATE;";

    pqxx::result rows = w.exec_params(sql.str(), experimentId);
    if (rows.empty())
        throw std::runtime_error("checkpoint policy parent experiment disappeared");

    CheckpointPolicyConfig config;
    config.enabled = rows[0][0].as<bool>();
    config.minLeaderScore = OptionalDoubleCell(rows[0], 1);
    config.minInferAccuracy = OptionalDoubleCell(rows[0], 2);
    if (!rows[0][3].is_null())
        config.topN = rows[0][3].as<int>();
    config.scope = rows[0][4].as<std::string>();
    config.stopMode = rows[0][5].as<std::string>();
    config.graceEvals = rows[0][6].as<int>();
    config.checkpointInterval = rows[0][7].as<int>();
    config.targetEpochs = rows[0][8].as<int>();
    if (!rows[0][9].is_null())
        config.currentEpoch = rows[0][9].as<int>();
    if (!rows[0][10].is_null())
        config.stopAfterCheckpointEpoch = rows[0][10].as<int>();
    config.status = rows[0][11].as<std::string>();
    config.phase = rows[0][12].as<std::string>();
    config.checkpointInferEnabled = rows[0][13].as<bool>();
    config.policyRevision = rows[0][14].as<long long>();
    config.persistedPolicyHash = OptionalStringCell(rows[0], 15);
    config.activeTrainingAttemptId = OptionalLongLongCell(rows[0], 16);
    return config;
}

int RunCheckpointControlCommand(const SchedulerOptions& options)
{
    const long long experimentId = CheckpointControlExperimentId(options);
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 2;
    const bool hasCheckpointInferEnabled = ColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = ColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointPolicyEnabled = ColumnExists(w, "experiment", "checkpoint_policy_enabled");
    if (!ColumnExists(w, "experiment", "stop_after_checkpoint_epoch") ||
        (!hasCheckpointInferEnabled && !hasOpportunisticCheckpointInfer))
    {
        std::cerr << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,missing=checkpoint_stop_columns" << std::endl;
        w.commit();
        return 2;
    }
    if ((options.enableCheckpointPolicyExperimentId.has_value() ||
         options.disableCheckpointPolicyExperimentId.has_value() ||
         options.setCheckpointPolicy.has_value()) &&
        !hasCheckpointPolicyEnabled)
    {
        std::cerr << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,missing=checkpoint_policy_columns" << std::endl;
        w.commit();
        return 2;
    }

    if (options.stopAfterCheckpointAllEpoch.has_value())
    {
        const int epoch = *options.stopAfterCheckpointAllEpoch;
        pqxx::result updated = w.exec_params(
            "UPDATE experiment "
            "SET stop_after_checkpoint_epoch = $1, updated_at = now() "
            "WHERE status = 'running' AND phase = 'train' "
            "RETURNING experiment_id;",
            epoch);
        w.commit();
        std::cout << "CHECKPOINT_STOP_ALL_REQUESTED"
                  << ",count=" << updated.size()
                  << ",stop_after_checkpoint_epoch=" << epoch
                  << std::endl;
        return 0;
    }

    if (options.clearStopAfterCheckpointAll)
    {
        pqxx::result updated = w.exec(
            "UPDATE experiment "
            "SET stop_after_checkpoint_epoch = NULL, updated_at = now() "
            "WHERE status = 'running' AND phase = 'train' "
            "RETURNING experiment_id;");
        w.commit();
        std::cout << "CHECKPOINT_STOP_ALL_CLEARED"
                  << ",count=" << updated.size()
                  << std::endl;
        return 0;
    }

    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, status, phase, checkpoint_interval, target_epochs "
        "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
        experimentId);
    if (rows.empty())
    {
        std::cerr << "CHECKPOINT_CONTROL_REJECTED"
                  << ",experiment_id=" << experimentId
                  << ",reason=experiment_not_found"
                  << std::endl;
        w.commit();
        return 1;
    }

    const std::string status = rows[0][1].as<std::string>();
    const std::string phase = rows[0][2].as<std::string>();
    const int checkpointInterval = rows[0][3].as<int>();
    const int targetEpochs = rows[0][4].as<int>();
    if ((options.stopAfterCheckpoint.has_value() ||
         options.clearStopAfterCheckpointExperimentId.has_value()) &&
        (phase != "train" || status != "running"))
    {
        std::cerr << "CHECKPOINT_CONTROL_REJECTED"
                  << ",experiment_id=" << experimentId
                  << ",status=" << status
                  << ",phase=" << phase
                  << ",reason=requires_running_train"
                  << std::endl;
        w.commit();
        return 1;
    }

    if (phase != "train" || (status != "pending" && status != "running" && status != "paused"))
    {
        std::cerr << "CHECKPOINT_CONTROL_REJECTED"
                  << ",experiment_id=" << experimentId
                  << ",status=" << status
                  << ",phase=" << phase
                  << ",reason=requires_train_pending_running_or_paused"
                  << std::endl;
        w.commit();
        return 1;
    }

    if (options.stopAfterCheckpoint.has_value())
    {
        const int requestedEpoch = options.stopAfterCheckpoint->second;
        const int effectiveEpoch = EffectiveCheckpointStopEpoch(requestedEpoch, checkpointInterval, targetEpochs);
        w.exec_params(
            "UPDATE experiment "
            "SET stop_after_checkpoint_epoch = $1, updated_at = now() "
            "WHERE experiment_id = $2;",
            requestedEpoch,
            experimentId);
        w.commit();
        std::cout << "CHECKPOINT_STOP_REQUESTED"
                  << ",experiment_id=" << experimentId
                  << ",stop_after_checkpoint_epoch=" << requestedEpoch
                  << ",effective_epoch=" << effectiveEpoch
                  << std::endl;
        return 0;
    }

    if (options.clearStopAfterCheckpointExperimentId.has_value())
    {
        w.exec_params(
            "UPDATE experiment "
            "SET stop_after_checkpoint_epoch = NULL, updated_at = now() "
            "WHERE experiment_id = $1;",
            experimentId);
        w.commit();
        std::cout << "CHECKPOINT_STOP_CLEARED"
                  << ",experiment_id=" << experimentId
                  << std::endl;
        return 0;
    }

    if (options.enableCheckpointInferExperimentId.has_value() ||
        options.disableCheckpointInferExperimentId.has_value())
    {
        const bool enabled = options.enableCheckpointInferExperimentId.has_value();
        if (!enabled && hasCheckpointPolicyEnabled)
        {
            pqxx::result policyRows = w.exec_params(
                "SELECT checkpoint_policy_enabled FROM experiment WHERE experiment_id = $1;",
                experimentId);
            if (!policyRows.empty() && policyRows[0][0].as<bool>())
            {
                std::cerr << "CHECKPOINT_INFER_REJECTED"
                          << ",experiment_id=" << experimentId
                          << ",reason=disable_checkpoint_policy_first"
                          << std::endl;
                w.commit();
                return 1;
            }
        }
        std::ostringstream sql;
        sql << "UPDATE experiment SET updated_at = now()";
        if (hasCheckpointInferEnabled)
            sql << ", checkpoint_infer_enabled = " << (enabled ? "true" : "false");
        if (hasOpportunisticCheckpointInfer)
            sql << ", opportunistic_checkpoint_infer = " << (enabled ? "true" : "false");
        sql << " WHERE experiment_id = " << experimentId << ";";
        w.exec(sql.str());
        w.commit();
        std::cout << (enabled ? "CHECKPOINT_INFER_ENABLED" : "CHECKPOINT_INFER_DISABLED")
                  << ",experiment_id=" << experimentId
                  << std::endl;
        return 0;
    }

    if (options.checkpointInferMinEpoch.has_value())
    {
        const int epoch = options.checkpointInferMinEpoch->second;
        w.exec_params(
            "UPDATE experiment "
            "SET checkpoint_infer_min_epoch = $1, updated_at = now() "
            "WHERE experiment_id = $2;",
            epoch,
            experimentId);
        w.commit();
        std::cout << "CHECKPOINT_INFER_MIN_EPOCH_SET"
                  << ",experiment_id=" << experimentId
                  << ",epoch=" << epoch
                  << std::endl;
        return 0;
    }

    if (options.checkpointInferInterval.has_value())
    {
        const int interval = options.checkpointInferInterval->second;
        w.exec_params(
            "UPDATE experiment "
            "SET checkpoint_infer_interval = $1, updated_at = now() "
            "WHERE experiment_id = $2;",
            interval,
            experimentId);
        w.commit();
        std::cout << "CHECKPOINT_INFER_INTERVAL_SET"
                  << ",experiment_id=" << experimentId
                  << ",interval=" << interval
                  << std::endl;
        return 0;
    }

    if (options.enableCheckpointPolicyExperimentId.has_value() ||
        options.disableCheckpointPolicyExperimentId.has_value())
    {
        const bool enabled = options.enableCheckpointPolicyExperimentId.has_value();
        CheckpointPolicyConfig config = LoadCheckpointPolicyControlConfig(
            w,
            experimentId,
            hasCheckpointInferEnabled,
            hasOpportunisticCheckpointInfer);
        if (enabled && !config.checkpointInferEnabled)
        {
            std::cerr << "CHECKPOINT_POLICY_REJECTED"
                      << ",experiment_id=" << experimentId
                      << ",reason=checkpoint_infer_not_enabled"
                      << std::endl;
            w.commit();
            return 1;
        }
        const std::optional<std::string> configError =
            CheckpointPolicyConfigurationError(config, enabled);
        if (enabled && configError.has_value())
        {
            std::cerr << "CHECKPOINT_POLICY_REJECTED"
                      << ",experiment_id=" << experimentId
                      << ",reason=" << *configError
                      << std::endl;
            w.commit();
            return 1;
        }
        CheckpointPolicyConfig resultingConfig = config;
        resultingConfig.enabled = enabled;
        const bool changed =
            CheckpointPolicyCanonicalText(config) !=
            CheckpointPolicyCanonicalText(resultingConfig);
        const std::string resultingHash =
            CheckpointPolicySemanticHash(resultingConfig);
        pqxx::result revised = w.exec_params(
            "UPDATE experiment "
            "SET checkpoint_policy_enabled = $1, "
            "checkpoint_policy_revision = checkpoint_policy_revision + "
            "CASE WHEN $2 THEN 1 ELSE 0 END, "
            "checkpoint_policy_hash = $3, updated_at = now() "
            "WHERE experiment_id = $4 "
            "RETURNING checkpoint_policy_revision;",
            enabled,
            changed,
            resultingHash,
            experimentId);
        w.commit();
        std::cout << (enabled ? "CHECKPOINT_POLICY_ENABLED" : "CHECKPOINT_POLICY_DISABLED")
                  << ",experiment_id=" << experimentId
                  << ",policy_revision=" << revised[0][0].as<long long>()
                  << ",policy_hash=" << resultingHash
                  << std::endl;
        return 0;
    }

    if (options.setCheckpointPolicy.has_value())
    {
        SchedulerOptions update = ParseCheckpointPolicyUpdateOptions(options.setCheckpointPolicy->second);
        CheckpointPolicyConfig resultingConfig = LoadCheckpointPolicyControlConfig(
            w,
            experimentId,
            hasCheckpointInferEnabled,
            hasOpportunisticCheckpointInfer);
        if (update.checkpointPolicySetKeys.count("min_leader_score"))
            resultingConfig.minLeaderScore = update.checkpointPolicyMinLeaderScore;
        if (update.checkpointPolicySetKeys.count("min_infer_accuracy"))
            resultingConfig.minInferAccuracy = update.checkpointPolicyMinInferAccuracy;
        if (update.checkpointPolicySetKeys.count("top_n"))
            resultingConfig.topN = update.checkpointPolicyTopN;
        if (update.checkpointPolicySetKeys.count("scope"))
            resultingConfig.scope = update.checkpointPolicyScope;
        if (update.checkpointPolicySetKeys.count("stop_mode"))
            resultingConfig.stopMode = update.checkpointPolicyStopMode;
        if (update.checkpointPolicySetKeys.count("grace_evals"))
            resultingConfig.graceEvals = update.checkpointPolicyGraceEvals;

        const std::optional<std::string> configError =
            CheckpointPolicyConfigurationError(resultingConfig, resultingConfig.enabled);
        if (configError.has_value())
        {
            std::cerr << "CHECKPOINT_POLICY_REJECTED"
                      << ",experiment_id=" << experimentId
                      << ",reason=" << *configError
                      << std::endl;
            w.commit();
            return 1;
        }
        const bool changed =
            CheckpointPolicyCanonicalText(resultingConfig) !=
            CheckpointPolicyCanonicalText(
                LoadCheckpointPolicyControlConfig(
                    w,
                    experimentId,
                    hasCheckpointInferEnabled,
                    hasOpportunisticCheckpointInfer));
        const std::string resultingHash =
            CheckpointPolicySemanticHash(resultingConfig);
        std::ostringstream sql;
        sql << "UPDATE experiment SET updated_at = now(), "
            << "checkpoint_policy_hash = " << w.quote(resultingHash)
            << ", checkpoint_policy_revision = checkpoint_policy_revision + "
            << (changed ? "1" : "0");
        if (update.checkpointPolicySetKeys.count("min_leader_score"))
            sql << ", checkpoint_policy_min_leader_score = " << FormatDouble(*update.checkpointPolicyMinLeaderScore);
        if (update.checkpointPolicySetKeys.count("min_infer_accuracy"))
            sql << ", checkpoint_policy_min_infer_accuracy = " << FormatDouble(*update.checkpointPolicyMinInferAccuracy);
        if (update.checkpointPolicySetKeys.count("top_n"))
            sql << ", checkpoint_policy_top_n = " << *update.checkpointPolicyTopN;
        if (update.checkpointPolicySetKeys.count("scope"))
            sql << ", checkpoint_policy_scope = " << w.quote(update.checkpointPolicyScope);
        if (update.checkpointPolicySetKeys.count("stop_mode"))
            sql << ", checkpoint_policy_stop_mode = " << w.quote(update.checkpointPolicyStopMode);
        if (update.checkpointPolicySetKeys.count("grace_evals"))
            sql << ", checkpoint_policy_grace_evals = " << update.checkpointPolicyGraceEvals;
        sql << " WHERE experiment_id = " << experimentId
            << " RETURNING checkpoint_policy_revision;";
        pqxx::result revised = w.exec(sql.str());
        w.commit();
        std::cout << "CHECKPOINT_POLICY_SET"
                  << ",experiment_id=" << experimentId
                  << ",rules=" << CheckpointPolicyRuleText(resultingConfig)
                  << ",policy_revision=" << revised[0][0].as<long long>()
                  << ",policy_hash=" << resultingHash
                  << std::endl;
        return 0;
    }

    w.commit();
    return 1;
}














struct DiscoveredManagedProcess
{
    int pid = -1;
    int processGroupId = -1;
    std::string executable;
    std::string command;
    std::string processStartIdentity;
};


































std::optional<CheckpointEvalRow> LoadCheckpointEvalById(pqxx::work& w,
                                                        long long checkpointEvalId)
{
    if (!CheckpointEvalTableExists(w))
        return std::nullopt;
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
        "WHERE ce.checkpoint_eval_id = $1;",
        checkpointEvalId);
    if (rows.empty())
        return std::nullopt;
    return RowToCheckpointEval(rows[0]);
}




void AppendTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out{path, std::ios::app};
    out << text;
    out.flush();
}

std::optional<long long> ExtractLastModelId(const std::string& text)
{
    std::regex idRegex{
        "(Saved model with model_id=|Created new model_id=|RESUME_SAVED_NEW_MODEL_ID=|CHECKPOINT_SAVE_DONE[^\\n]*model_id=)([0-9]+)"};
    std::optional<long long> last;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), idRegex);
         it != std::sregex_iterator();
         ++it)
    {
        last = std::stoll((*it)[2].str());
    }
    return last;
}

std::optional<double> ExtractLastDouble(const std::string& text, const std::regex& regex)
{
    std::optional<double> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stod((*it)[1].str());
    }
    return value;
}

ParsedMetrics ParseMetricsFromLogs(const ExperimentRow& experiment)
{
    ParsedMetrics metrics;
    const std::string trainLog = ReadFileIfExists(experiment.trainLogPath);
    const std::string inferLog = ReadFileIfExists(experiment.inferLogPath);
    const std::string combined = trainLog + "\n" + inferLog;

    metrics.modelId = ExtractLastModelId(combined);
    if (!metrics.modelId.has_value())
        metrics.modelId = experiment.lastModelId;

    metrics.inferAccuracy = ExtractLastDouble(
        inferLog,
        std::regex{"Overall 3-class accuracy:\\s*([0-9]+(?:\\.[0-9]+)?)%"});
    if (metrics.inferAccuracy.has_value())
        metrics.inferAccuracy = *metrics.inferAccuracy / 100.0;

    metrics.validationAccuracy = ExtractLastDouble(
        combined,
        std::regex{"EPOCH_3CLASS_ACCURACY[^\\n]*(?:validation_accuracy|val_accuracy|accuracy)=([0-9]+(?:\\.[0-9]+)?)"});
    metrics.trainAccuracy = ExtractLastDouble(
        combined,
        std::regex{"EPOCH_3CLASS_ACCURACY[^\\n]*(?:train_accuracy)=([0-9]+(?:\\.[0-9]+)?)"});
    metrics.lossLast = ExtractLastDouble(
        combined,
        std::regex{"(?:loss|LOSS)[= :]([0-9]+(?:\\.[0-9]+)?)"});

    const std::optional<double> parsedCompletedEpochs = ExtractLastDouble(
        combined,
        std::regex{"(?:completed_epochs|epochs_trained|RESUME_TARGET_EPOCH)[= ]([0-9]+)"});
    if (parsedCompletedEpochs.has_value())
        metrics.completedEpochs = static_cast<int>(*parsedCompletedEpochs);

    std::regex confusionRegex{
        "Overall 3-class confusion matrix[^\\n]*\\[\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\],\\s*\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\],\\s*\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\]\\]"};
    std::smatch match;
    std::string::const_iterator searchStart = inferLog.cbegin();
    while (std::regex_search(searchStart, inferLog.cend(), match, confusionRegex))
    {
        for (int i = 0; i < 9; ++i)
            metrics.confusion[i / 3][i % 3] = std::stoll(match[i + 1].str());
        metrics.hasConfusion = true;
        searchStart = match.suffix().first;
    }

    std::regex acceptRegex{"ACCEPT_MODEL=(true|false)"};
    for (auto it = std::sregex_iterator(combined.begin(), combined.end(), acceptRegex);
         it != std::sregex_iterator();
         ++it)
    {
        metrics.acceptModel = ((*it)[1].str() == "true");
    }

    std::regex rejectRegex{"REJECT_REASON=([^\\n,]+)"};
    for (auto it = std::sregex_iterator(combined.begin(), combined.end(), rejectRegex);
         it != std::sregex_iterator();
         ++it)
    {
        metrics.rejectReason = (*it)[1].str();
    }

    if (metrics.hasConfusion)
    {
        long long total = 0;
        long long correct = 0;
        for (int actual = 0; actual < 3; ++actual)
        {
            for (int pred = 0; pred < 3; ++pred)
            {
                total += metrics.confusion[actual][pred];
                if (actual == pred)
                    correct += metrics.confusion[actual][pred];
            }
        }
        if (!metrics.inferAccuracy.has_value() && total > 0)
            metrics.inferAccuracy = static_cast<double>(correct) / static_cast<double>(total);
        metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;
        metrics.acceptAccuracy = metrics.inferAccuracy;
    }

    return metrics;
}






bool ApplyStructuredInferenceMetrics(pqxx::work& w,
                                            const ExperimentRow& experiment,
                                            ParsedMetrics& metrics)
{
    if (!TableExists(w, "inference_eval_result") ||
        !experiment.lastModelId.has_value() ||
        !experiment.inferStart.has_value() ||
        !experiment.inferEnd.has_value())
    {
        return false;
    }

    pqxx::result rows = w.exec_params(
        "SELECT accuracy, accept_model, COALESCE(reject_reason, ''), completed_epochs "
        "FROM inference_eval_result "
        "WHERE model_id = $1 "
        "AND symbol = $2 "
        "AND prediction_horizon = $3 "
        "AND abs(threshold_logret - $4) <= 1e-7 "
        "AND from_date = $5 "
        "AND to_date = $6 "
        "AND status = 'completed' "
        "AND inference_scope = 'final' "
        "AND checkpoint_eval_id IS NULL "
        "ORDER BY completed_at DESC LIMIT 1;",
        *experiment.lastModelId,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.inferStart->substr(0, 10),
        experiment.inferEnd->substr(0, 10));
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
    metrics.modelId = experiment.lastModelId;
    metrics.acceptAccuracy = metrics.inferAccuracy;
    metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;
    return true;
}




















































int RunEvaluateCheckpointPolicyCommand(const SchedulerOptions& options)
{
    if (!options.evaluateCheckpointPolicyEvalId.has_value())
        throw std::invalid_argument("missing checkpoint_eval_id for policy reevaluation");

    const long long checkpointEvalId = *options.evaluateCheckpointPolicyEvalId;
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);

    const std::optional<CheckpointEvalRow> eval = LoadCheckpointEvalById(w, checkpointEvalId);
    if (!eval.has_value())
    {
        std::cout << "CHECKPOINT_POLICY_REEVALUATION_REQUESTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << std::endl;
        std::cerr << "CHECKPOINT_POLICY_REEVALUATION_REJECTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << ",reason=checkpoint_eval_not_found"
                  << std::endl;
        w.commit();
        return 1;
    }

    std::cout << "CHECKPOINT_POLICY_REEVALUATION_REQUESTED"
              << ",parent_experiment_id=" << eval->experiment.experimentId
              << ",checkpoint_eval_id=" << eval->checkpointEvalId
              << ",checkpoint_epoch=" << eval->checkpointEpoch
              << ",checkpoint_model_id=" << eval->checkpointModelId
              << ",symbol=" << eval->experiment.symbol
              << ",prediction_horizon=" << eval->experiment.predictionHorizon
              << std::endl;

    const CheckpointPolicyEvaluationResult result =
        EvaluateCheckpointPolicyAfterAnalysis(w, *eval);
    if (!result.evaluated)
    {
        std::cerr << "CHECKPOINT_POLICY_REEVALUATION_REJECTED"
                  << ",parent_experiment_id=" << eval->experiment.experimentId
                  << ",checkpoint_eval_id=" << eval->checkpointEvalId
                  << ",checkpoint_epoch=" << eval->checkpointEpoch
                  << ",checkpoint_model_id=" << eval->checkpointModelId
                  << ",reason=" << result.reason
                  << std::endl;
        w.commit();
        return 1;
    }

    w.commit();
    std::cout << "CHECKPOINT_POLICY_REEVALUATION_COMPLETED"
              << ",parent_experiment_id=" << eval->experiment.experimentId
              << ",checkpoint_eval_id=" << eval->checkpointEvalId
              << ",checkpoint_epoch=" << eval->checkpointEpoch
              << ",checkpoint_model_id=" << eval->checkpointModelId
              << ",decision=" << result.decision
              << ",reason=" << result.reason
              << std::endl;
    return 0;
}

int RunCheckpointPolicyStatusCommand(const SchedulerOptions& options)
{
    const long long checkpointEvalId = *options.checkpointPolicyStatusEvalId;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work w{connection};
    SetTransactionReadOnly(w);
    EA::SchedulerCore::PostgresSchedulerRepository repository{w};
    if (!repository.checkpointPolicySchemaAvailable())
    {
        w.commit();
        std::cerr << "CHECKPOINT_POLICY_STATUS_REJECTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << ",reason=migration_required" << std::endl;
        return 1;
    }
    const std::optional<CheckpointEvalRow> eval =
        LoadCheckpointEvalById(w, checkpointEvalId);
    if (!eval.has_value())
    {
        w.commit();
        std::cerr << "CHECKPOINT_POLICY_STATUS_REJECTED"
                  << ",checkpoint_eval_id=" << checkpointEvalId
                  << ",reason=checkpoint_eval_not_found" << std::endl;
        return 1;
    }

    pqxx::result configRows = w.exec_params(
        "SELECT checkpoint_policy_enabled, "
        "checkpoint_policy_min_leader_score, checkpoint_policy_min_infer_accuracy, "
        "checkpoint_policy_top_n, checkpoint_policy_scope, checkpoint_policy_stop_mode, "
        "checkpoint_policy_grace_evals, checkpoint_interval, target_epochs, current_epoch, "
        "stop_after_checkpoint_epoch, status, phase, "
        "(checkpoint_infer_enabled OR opportunistic_checkpoint_infer), "
        "checkpoint_policy_revision, checkpoint_policy_hash, "
        "active_scheduler_worker_attempt_id, checkpoint_policy_last_decision_id, "
        "checkpoint_policy_stop_decision_id "
        "FROM experiment WHERE experiment_id=$1;",
        eval->experiment.experimentId);
    if (configRows.size() != 1)
        throw std::runtime_error("checkpoint policy status parent disappeared");
    const pqxx::row configRow = configRows[0];
    CheckpointPolicyConfig config;
    config.enabled = configRow[0].as<bool>();
    config.minLeaderScore = OptionalDoubleCell(configRow, 1);
    config.minInferAccuracy = OptionalDoubleCell(configRow, 2);
    if (!configRow[3].is_null())
        config.topN = configRow[3].as<int>();
    config.scope = configRow[4].as<std::string>();
    config.stopMode = configRow[5].as<std::string>();
    config.graceEvals = configRow[6].as<int>();
    config.checkpointInterval = configRow[7].as<int>();
    config.targetEpochs = configRow[8].as<int>();
    if (!configRow[9].is_null())
        config.currentEpoch = configRow[9].as<int>();
    if (!configRow[10].is_null())
        config.stopAfterCheckpointEpoch = configRow[10].as<int>();
    config.status = configRow[11].as<std::string>();
    config.phase = configRow[12].as<std::string>();
    config.checkpointInferEnabled = configRow[13].as<bool>();
    config.policyRevision = configRow[14].as<long long>();
    config.persistedPolicyHash = OptionalStringCell(configRow, 15);
    config.activeTrainingAttemptId = OptionalLongLongCell(configRow, 16);
    const std::optional<long long> lastDecisionId =
        OptionalLongLongCell(configRow, 17);
    const std::optional<long long> stopDecisionId =
        OptionalLongLongCell(configRow, 18);

    const EA::SchedulerCore::CheckpointEvaluationRecord evaluation{
        eval->checkpointEvalId,
        eval->experiment.experimentId,
        eval->checkpointModelId,
        eval->checkpointEpoch,
        eval->experiment.symbol,
        eval->experiment.predictionHorizon};
    const EA::SchedulerCore::CheckpointPolicyEvidenceLoadResult loadedEvidence =
        repository.loadCheckpointPolicyEvidence(evaluation);
    const bool exactEvidence = loadedEvidence.evidence.has_value();
    ValidatedCheckpointPolicyEvidence validated;
    std::string evidenceReason = loadedEvidence.rejectionReason;
    if (exactEvidence)
        validated = *loadedEvidence.evidence;
    std::optional<CheckpointPolicyDecisionContext> context;
    std::optional<CheckpointPolicyEvidenceIdentity> identity;
    std::string evidenceWatermark;
    if (exactEvidence)
    {
        CheckpointPolicyPopulation completedPopulation =
            repository.loadCompletedCheckpointPolicyPopulation(
                evaluation.parentExperimentId);
        CheckpointPolicyPopulation rankPopulation =
            repository.loadCheckpointPolicyRankPopulation(
                evaluation, config);
        context = PlanCheckpointPolicyDecision(
            evaluation,
            config,
            validated,
            std::move(completedPopulation),
            std::move(rankPopulation));
        identity = MakeCheckpointPolicyEvidenceIdentity(
            evaluation,
            validated,
            *context,
            config);
        evidenceWatermark = CheckpointPolicyEvidenceWatermark(*identity);
    }
    pqxx::result decisions = w.exec_params(
        "SELECT checkpoint_decision_id, decision, reason, policy_revision, "
        "policy_hash, evidence_watermark, analysis_id, inference_eval_result_id, "
        "rank_value, rank_scope, rank_population_watermark, identity_status, "
        "stop_request_applied, requested_stop_epoch, superseded_reason, "
        "superseded_by_decision_id, stop_action_worker_attempt_id, created_at::text "
        "FROM experiment_checkpoint_decision WHERE checkpoint_eval_id=$1 "
        "ORDER BY checkpoint_decision_id ASC;",
        checkpointEvalId);
    pqxx::result authoritativeRows = w.exec_params(
        "SELECT checkpoint_decision_id FROM experiment_checkpoint_decision "
        "WHERE parent_experiment_id=$1 "
        "AND identity_status IN ('active','action_applied') "
        "ORDER BY (identity_status='action_applied') DESC, checkpoint_epoch DESC, "
        "checkpoint_eval_id DESC, checkpoint_decision_id DESC LIMIT 1;",
        eval->experiment.experimentId);
    const std::optional<long long> authoritativeDecisionId =
        authoritativeRows.empty()
            ? std::nullopt
            : std::optional<long long>{authoritativeRows[0][0].as<long long>()};
    w.commit();

    const std::string currentPolicyHash =
        CheckpointPolicySemanticHash(config);
    std::cout << "CHECKPOINT POLICY STATUS\n"
              << "  Checkpoint Eval: " << checkpointEvalId << "\n"
              << "  Parent Experiment: " << eval->experiment.experimentId << "\n"
              << "  Checkpoint Model/Epoch: " << eval->checkpointModelId
              << "/" << eval->checkpointEpoch << "\n"
              << "  Checkpoint Lifecycle: " << eval->status << "/"
              << eval->phase << "\n"
              << "  Policy: " << CheckpointPolicyCanonicalText(config) << "\n"
              << "  Policy Revision/Hash: " << config.policyRevision << "/"
              << currentPolicyHash << " (persisted="
              << config.persistedPolicyHash.value_or("legacy_unmaterialized")
              << ")\n"
              << "  Exact Analysis: "
              << (exactEvidence ? std::to_string(validated.analysisId) : "unavailable")
              << "\n"
              << "  Exact Checkpoint Inference Result: "
              << (exactEvidence
                      ? std::to_string(validated.inferenceEvalResultId)
                      : "unavailable")
              << "\n"
              << "  Evidence Watermark: "
              << (exactEvidence ? evidenceWatermark : "unavailable")
              << "\n"
              << "  Evidence State: "
              << (exactEvidence ? "exact" : evidenceReason) << "\n"
              << "  Current Authoritative Decision: "
              << (authoritativeDecisionId.has_value()
                      ? std::to_string(*authoritativeDecisionId)
                      : "none")
              << "\n"
              << "  Last Observed Decision: "
              << (lastDecisionId.has_value()
                      ? std::to_string(*lastDecisionId)
                      : "none")
              << "\n"
              << "  Stop Decision: "
              << (stopDecisionId.has_value()
                      ? std::to_string(*stopDecisionId)
                      : "none")
              << " requested_epoch="
              << (config.stopAfterCheckpointEpoch.has_value()
                      ? std::to_string(*config.stopAfterCheckpointEpoch)
                      : "none")
              << "\n";
    for (const auto& row : decisions)
    {
        std::cout << "  Durable Decision: id=" << row[0].as<long long>()
                  << " decision=" << row[1].as<std::string>()
                  << " identity_status=" << row[11].as<std::string>()
                  << " policy_revision="
                  << (row[3].is_null() ? "legacy" : row[3].c_str())
                  << " policy_hash="
                  << (row[4].is_null() ? "legacy" : row[4].c_str())
                  << " evidence_watermark="
                  << (row[5].is_null() ? "legacy" : row[5].c_str())
                  << " analysis_id="
                  << (row[6].is_null() ? "legacy" : row[6].c_str())
                  << " inference_eval_result_id="
                  << (row[7].is_null() ? "legacy" : row[7].c_str())
                  << " stop_applied=" << (row[12].as<bool>() ? "yes" : "no")
                  << " requested_stop_epoch="
                  << (row[13].is_null() ? "none" : row[13].c_str())
                  << " superseded_reason="
                  << (row[14].is_null() ? "none" : row[14].c_str())
                  << " reason=" << row[2].as<std::string>() << "\n";
    }
    std::cout << "CHECKPOINT_POLICY_STATUS"
              << ",checkpoint_eval_id=" << checkpointEvalId
              << ",parent_experiment_id=" << eval->experiment.experimentId
              << ",checkpoint_model_id=" << eval->checkpointModelId
              << ",checkpoint_epoch=" << eval->checkpointEpoch
              << ",checkpoint_status=" << eval->status
              << ",checkpoint_phase=" << eval->phase
              << ",policy_revision=" << config.policyRevision
              << ",policy_hash=" << currentPolicyHash
              << ",persisted_policy_hash="
              << config.persistedPolicyHash.value_or("NULL")
              << ",analysis_id="
              << (exactEvidence ? std::to_string(validated.analysisId) : "NULL")
              << ",inference_eval_result_id="
              << (exactEvidence
                      ? std::to_string(validated.inferenceEvalResultId)
                      : "NULL")
              << ",evidence_watermark="
              << (exactEvidence ? evidenceWatermark : "NULL")
              << ",evidence_reason="
              << (exactEvidence ? "exact" : evidenceReason)
              << ",authoritative_decision_id="
              << (authoritativeDecisionId.has_value()
                      ? std::to_string(*authoritativeDecisionId)
                      : "NULL")
              << ",last_decision_id="
              << (lastDecisionId.has_value()
                      ? std::to_string(*lastDecisionId)
                      : "NULL")
              << ",stop_decision_id="
              << (stopDecisionId.has_value()
                      ? std::to_string(*stopDecisionId)
                      : "NULL")
              << ",requested_stop_epoch="
              << (config.stopAfterCheckpointEpoch.has_value()
                      ? std::to_string(*config.stopAfterCheckpointEpoch)
                      : "NULL")
              << ",decision_count=" << decisions.size()
              << std::endl;
    return 0;
}


















bool ValidateContinuationEvaluationSource(
    pqxx::work& w,
    const ContinuationPolicyConfig& config,
    std::string& reason,
    ContinuationEvidence* selectedEvidence = nullptr)
{
    const std::optional<std::string> configError =
        ContinuationPolicyConfigurationError(config, true);
    if (configError.has_value())
    {
        reason = *configError;
        return false;
    }
    if (!ContinuationPolicySourceCompletionReady(config))
    {
        reason = "source_requires_completed_done";
        return false;
    }

    const std::vector<ContinuationEvidence> raw = LoadContinuationEvidence(w, config);
    const std::vector<ContinuationEvidence> evidence = DeduplicateContinuationEvidence(raw);
    if (evidence.empty())
    {
        reason = "source_has_no_completed_analysis_evidence";
        return false;
    }
    if (evidence.back().completedEpoch >= *config.targetEpochs)
    {
        reason = "target_epochs_not_greater_than_source_completed_epoch";
        return false;
    }
    const std::optional<ContinuationEvidence> selected =
        SelectContinuationSourceEvidence(config, raw);
    if (!selected.has_value())
    {
        reason = "no_valid_source_analysis_for_source_mode";
        return false;
    }
    if (selectedEvidence)
        *selectedEvidence = *selected;
    return ValidateContinuationResumeSource(w, config, *selected, nullptr, reason);
}

long long ContinuationControlExperimentId(const SchedulerOptions& options)
{
    if (options.enableContinuationPolicyExperimentId.has_value())
        return *options.enableContinuationPolicyExperimentId;
    if (options.disableContinuationPolicyExperimentId.has_value())
        return *options.disableContinuationPolicyExperimentId;
    if (options.setContinuationPolicy.has_value())
        return options.setContinuationPolicy->first;
    throw std::invalid_argument("missing continuation policy experiment id");
}

bool HasContinuationControlCommand(const SchedulerOptions& options)
{
    return options.enableContinuationPolicyExperimentId.has_value() ||
           options.disableContinuationPolicyExperimentId.has_value() ||
           options.setContinuationPolicy.has_value();
}

int RunContinuationPolicyControlCommand(const SchedulerOptions& options)
{
    const long long sourceExperimentId = ContinuationControlExperimentId(options);
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work w{connection};
    SetTransactionReadWrite(w);
    if (!ContinuationPolicySchemaExists(w))
    {
        std::cerr << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,missing=continuation_policy_schema"
                  << std::endl;
        w.commit();
        return 2;
    }

    std::optional<ContinuationPolicyConfig> loaded =
        LockContinuationPolicyConfigForUpdate(w, sourceExperimentId);
    if (!loaded.has_value())
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=source_experiment_not_found"
                  << std::endl;
        w.commit();
        return 1;
    }
    ContinuationPolicyConfig config = *loaded;

    if (options.enableContinuationPolicyExperimentId.has_value() ||
        options.disableContinuationPolicyExperimentId.has_value())
    {
        const bool enabled = options.enableContinuationPolicyExperimentId.has_value();
        if (enabled)
        {
            config.enabled = true;
            const std::optional<std::string> reason =
                ContinuationPolicyEnablementError(config);
            if (reason.has_value())
            {
                std::cerr << "CONTINUATION_POLICY_ERROR"
                          << ",source_experiment_id=" << sourceExperimentId
                          << ",reason=" << *reason
                          << std::endl;
                w.commit();
                return 1;
            }
        }

        w.exec_params(
            "UPDATE experiment SET continuation_policy_enabled = $1, updated_at = now() "
            "WHERE experiment_id = $2;",
            enabled,
            sourceExperimentId);
        w.commit();
        std::cout << (enabled ? "CONTINUATION_POLICY_ENABLED" : "CONTINUATION_POLICY_DISABLED")
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",history_preserved=1"
                  << std::endl;
        return 0;
    }

    ContinuationPolicyUpdate update;
    try
    {
        update = ParseContinuationPolicyUpdate(options.setContinuationPolicy->second);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=" << e.what()
                  << std::endl;
        w.commit();
        return 1;
    }
    ContinuationPolicyConfig resulting = config;
    ApplyContinuationPolicyUpdate(resulting, update);

    const bool newlyEnablingInheritance =
        update.keys.count("inherit_to_child") &&
        resulting.inheritToChild &&
        !config.inheritToChild;
    const std::optional<std::string> resultingProgressionMode =
        EffectiveContinuationProgressionMode(resulting);
    if (newlyEnablingInheritance &&
        resultingProgressionMode != "target_sequence" &&
        !resulting.maxTargetEpochs.has_value())
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=inherit_to_child_requires_max_target_epochs"
                  << std::endl;
        w.commit();
        return 1;
    }
    if (update.keys.count("max_target_epochs") &&
        !resulting.maxTargetEpochs.has_value() &&
        resulting.inheritToChild &&
        resultingProgressionMode != "target_sequence")
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=cannot_clear_max_target_while_inheritance_enabled"
                  << std::endl;
        w.commit();
        return 1;
    }

    const std::optional<std::string> configError =
        ContinuationPolicyConfigurationError(resulting, resulting.enabled);
    if (configError.has_value())
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=" << *configError
                  << std::endl;
        w.commit();
        return 1;
    }
    if (resulting.enabled)
    {
        const std::optional<std::string> reason =
            ContinuationPolicyEnablementError(resulting);
        if (reason.has_value())
        {
            std::cerr << "CONTINUATION_POLICY_ERROR"
                      << ",source_experiment_id=" << sourceExperimentId
                      << ",reason=" << *reason
                      << std::endl;
            w.commit();
            return 1;
        }
    }

    const bool changed = ContinuationPolicySemanticCanonicalText(config) !=
                         ContinuationPolicySemanticCanonicalText(resulting);
    std::ostringstream sql;
    sql << "UPDATE experiment SET updated_at = now()";
    if (update.keys.count("target_epochs"))
        sql << ", continuation_policy_target_epochs = " << SqlNullable(w, update.targetEpochs);
    if (update.keys.count("min_evals"))
        sql << ", continuation_policy_min_evals = " << update.minEvals;
    if (update.keys.count("patience"))
        sql << ", continuation_policy_patience = " << update.patience;
    if (update.keys.count("min_leader_score"))
        sql << ", continuation_policy_min_leader_score = " << SqlNullable(w, update.minLeaderScore);
    if (update.keys.count("min_infer_accuracy"))
        sql << ", continuation_policy_min_infer_accuracy = " << SqlNullable(w, update.minInferAccuracy);
    if (update.keys.count("min_profitability_actionable_count"))
        sql << ", continuation_policy_min_profit_actionable_count = "
            << SqlNullable(w, update.minProfitabilityActionableCount);
    if (update.keys.count(
            "min_profitability_aggregate_terminal_horizon_log_return_sum"))
    {
        sql << ", continuation_policy_min_profit_aggregate_log_return_sum = "
            << SqlNullable(
                   w,
                   update
                       .minProfitabilityAggregateTerminalHorizonLogReturnSum);
    }
    if (update.keys.count(
            "min_profitability_average_terminal_horizon_log_return_per_actionable_prediction"))
    {
        sql << ", continuation_policy_min_profit_average_log_return = "
            << SqlNullable(
                   w,
                   update
                       .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction);
    }
    if (update.keys.count("min_improvement"))
        sql << ", continuation_policy_min_improvement = " << SqlNullable(w, update.minImprovement);
    if (update.keys.count("max_degradation"))
        sql << ", continuation_policy_max_degradation = " << SqlNullable(w, update.maxDegradation);
    if (update.keys.count("top_n"))
        sql << ", continuation_policy_top_n = " << SqlNullable(w, update.topN);
    if (update.keys.count("scope"))
        sql << ", continuation_policy_scope = " << w.quote(update.scope);
    if (update.keys.count("trend_mode"))
        sql << ", continuation_policy_trend_mode = " << w.quote(update.trendMode);
    if (update.keys.count("source_mode"))
        sql << ", continuation_policy_source_mode = " << w.quote(update.sourceMode);
    if (update.keys.count("include_excluded"))
        sql << ", continuation_policy_include_excluded = "
            << (update.includeExcluded ? "true" : "false");
    if (update.keys.count("candidate_excluded"))
        sql << ", continuation_candidate_excluded = "
            << (update.candidateExcluded ? "true" : "false");
    if (update.keys.count("inherit_to_child"))
        sql << ", continuation_policy_inherit_to_child = "
            << (update.inheritToChild ? "true" : "false");
    if (update.keys.count("progression_mode"))
        sql << ", continuation_policy_progression_mode = "
            << SqlNullable(w, update.progressionMode);
    if (update.keys.count("target_increment"))
        sql << ", continuation_policy_target_increment = "
            << SqlNullable(w, update.targetIncrement);
    if (update.keys.count("max_target_epochs"))
        sql << ", continuation_policy_max_target_epochs = "
            << SqlNullable(w, update.maxTargetEpochs);
    if (update.keys.count("target_sequence"))
        sql << ", continuation_policy_target_sequence = "
            << SqlContinuationTargetSequence(update.targetSequence);
    if (changed)
        sql << ", continuation_policy_revision = continuation_policy_revision + 1";
    sql << " WHERE experiment_id = " << sourceExperimentId
        << " RETURNING continuation_policy_revision;";
    pqxx::result updated = w.exec(sql.str());
    resulting.policyRevision = updated[0][0].as<long long>();
    w.commit();

    std::cout << "CONTINUATION_POLICY_SET"
              << ",source_experiment_id=" << sourceExperimentId
              << ",changed=" << (changed ? "1" : "0")
              << ",config=" << ContinuationPolicyDisplayText(resulting)
              << std::endl;
    return 0;
}

int RunEvaluateContinuationCommand(const SchedulerOptions& options)
{
    const long long sourceExperimentId = *options.evaluateContinuationExperimentId;
    try
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work w{connection};
        SetTransactionReadWrite(w);
        ContinuationPolicyConfig config;
        ContinuationEvaluation evaluation = EvaluateContinuationPolicy(
            w,
            sourceExperimentId,
            &config);
        if (!evaluation.persisted && !evaluation.reused && !evaluation.alreadyQueued)
        {
            w.commit();
            return 1;
        }
        w.commit();
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




int RunContinuationStatusCommand(const SchedulerOptions& options)
{
    const long long sourceExperimentId = *options.continuationStatusExperimentId;
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work w{connection};
    SetTransactionReadOnly(w);
    if (!ContinuationPolicySchemaExists(w))
    {
        std::cerr << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,missing=continuation_policy_schema"
                  << std::endl;
        w.commit();
        return 2;
    }
    const std::optional<ContinuationPolicyConfig> config =
        FindContinuationPolicyConfig(w, sourceExperimentId);
    if (!config.has_value())
    {
        std::cerr << "CONTINUATION_POLICY_ERROR"
                  << ",source_experiment_id=" << sourceExperimentId
                  << ",reason=source_experiment_not_found"
                  << std::endl;
        w.commit();
        return 1;
    }

    pqxx::result decisions = w.exec_params(
        "SELECT continuation_decision_id, source_model_id, source_epoch, target_epochs, "
        "decision, reason, queued_experiment_id, policy_revision, policy_hash, evidence_watermark "
        "FROM experiment_continuation_decision "
        "WHERE source_experiment_id = $1 "
        "ORDER BY updated_at DESC, continuation_decision_id DESC LIMIT 1;",
        sourceExperimentId);
    std::string outgoingInheritanceValidation = "disabled";
    std::optional<int> derivedNextTarget;
    std::string derivedPolicyHash;
    const bool sourceCompletionReady =
        ContinuationPolicySourceCompletionReady(*config);
    bool evaluationReady = false;
    std::string evaluationDeferredReason;
    ContinuationEvidence selectedEvidence;
    ContinuationEvaluation currentEvaluation;
    currentEvaluation.profitabilityGate =
        EvaluateContinuationProfitabilityGate(*config, selectedEvidence);
    bool currentEvaluationAvailable = false;
    if (!config->enabled)
    {
        evaluationDeferredReason = "policy_disabled";
    }
    else if (!sourceCompletionReady)
    {
        evaluationDeferredReason = "source_not_completed";
    }
    else
    {
        std::string reason;
        evaluationReady = ValidateContinuationEvaluationSource(
            w,
            *config,
            reason,
            &selectedEvidence);
        if (!evaluationReady)
            evaluationDeferredReason = reason;
        currentEvaluation = EvaluateContinuationPolicy(
            w,
            sourceExperimentId,
            nullptr,
            false);
        currentEvaluationAvailable = true;
        selectedEvidence = currentEvaluation.selected;
    }
    const std::string currentPolicyHash = ContinuationPolicySemanticHash(*config);
    const std::optional<std::string> effectiveProgressionMode =
        EffectiveContinuationProgressionMode(*config);
    std::optional<int> sequenceFinalTarget;
    if (config->targetSequence.has_value() && !config->targetSequence->empty())
        sequenceFinalTarget = config->targetSequence->back();
    const std::optional<int> effectiveFinalTarget =
        sequenceFinalTarget.has_value()
            ? sequenceFinalTarget
            : config->maxTargetEpochs;
    if (config->inheritToChild)
    {
        try
        {
            SchedulerOptions prospectiveChild;
            prospectiveChild.targetEpochs = config->targetEpochs;
            const ContinuationChildPolicyPlan plan =
                sourceCompletionReady
                    ? PrepareContinuationChildPolicy(
                          w,
                          *config,
                          prospectiveChild,
                          config->source.targetEpochs)
                    : PlanContinuationChildPolicy(*config);
            outgoingInheritanceValidation =
                plan.terminal ? "valid_terminal_child" : "valid";
            if (!plan.terminal)
                derivedNextTarget = plan.targetEpochs;
            derivedPolicyHash = plan.policyHash;
        }
        catch (const std::exception& e)
        {
            outgoingInheritanceValidation = e.what();
        }
    }
    else if (config->inheritanceStatus == "max_target_reached")
    {
        outgoingInheritanceValidation = "terminal_max_target_reached";
    }
    w.commit();

    std::cout << "CONTINUATION STATUS\n";
    std::cout << "  Experiment: " << sourceExperimentId << "\n";
    std::cout << "  Continuation Policy: " << (config->enabled ? "enabled" : "disabled") << "\n";
    std::cout << "  Continuation Evaluation: ";
    if (evaluationReady)
        std::cout << "ready";
    else if (config->enabled && !sourceCompletionReady)
        std::cout << "enabled and awaiting source completion";
    else
        std::cout << "deferred (" << evaluationDeferredReason << ")";
    std::cout << "\n";
    std::cout << "  Continuation Candidate: "
              << (config->candidateExcluded ? "excluded" : "production-eligible") << "\n";
    std::cout << "  Current Experiment Target: " << config->source.targetEpochs << "\n";
    std::cout << "  Continuation Progression Mode: "
              << effectiveProgressionMode.value_or("not configured") << "\n";
    std::cout << "  Continuation Target Sequence: "
              << ContinuationTargetSequenceText(config->targetSequence, "not configured")
              << "\n";
    std::cout << "  Continuation Target: "
              << (config->targetEpochs.has_value() ? std::to_string(*config->targetEpochs) : "not configured")
              << "\n";
    std::cout << "  Inherit Policy To Child: "
              << (config->inheritToChild ? "enabled" : "disabled") << "\n";
    std::cout << "  Continuation Target Increment: "
              << (config->targetIncrement.has_value()
                      ? std::to_string(*config->targetIncrement)
                      : "not configured")
              << "\n";
    std::cout << "  Continuation Maximum Target: "
              << (effectiveFinalTarget.has_value()
                      ? std::to_string(*effectiveFinalTarget)
                      : "not configured")
              << "\n";
    std::cout << "  Derived Next Policy Target: "
              << (derivedNextTarget.has_value()
                      ? std::to_string(*derivedNextTarget)
                      : "none")
              << "\n";
    std::cout << "  Sequence Next Target: "
              << (effectiveProgressionMode == "target_sequence" &&
                          derivedNextTarget.has_value()
                      ? std::to_string(*derivedNextTarget)
                      : "none")
              << "\n";
    std::cout << "  Sequence Final Target: "
              << (sequenceFinalTarget.has_value()
                      ? std::to_string(*sequenceFinalTarget)
                      : "none")
              << "\n";
    std::cout << "  Policy Inherited: " << (config->policyInherited ? "yes" : "no") << "\n";
    std::cout << "  Policy Inherited From: "
              << (config->inheritedFromExperimentId.has_value()
                      ? std::to_string(*config->inheritedFromExperimentId)
                      : "none")
              << "\n";
    std::cout << "  Policy Inherited From Revision: "
              << (config->inheritedFromRevision.has_value()
                      ? std::to_string(*config->inheritedFromRevision)
                      : "none")
              << "\n";
    std::cout << "  Policy Inherited From Hash: "
              << config->inheritedFromHash.value_or("none") << "\n";
    std::cout << "  Terminal Inheritance State: "
              << (config->inheritanceStatus == "max_target_reached" ? "yes" : "no")
              << "\n";
    std::cout << "  Inheritance Validation: " << outgoingInheritanceValidation << "\n";
    std::cout << "  Continuation Rules: " << ContinuationPolicyDisplayText(*config) << "\n";
    std::cout << "  Continuation Profitability Policy: "
              << (ContinuationProfitabilityPolicyConfigured(*config)
                      ? "enabled"
                      : "disabled")
              << " min_actionable_count="
              << ContinuationOptionalLongLongText(
                     config->minProfitabilityActionableCount)
              << " min_aggregate_terminal_horizon_log_return_sum="
              << ContinuationOptionalDoubleText(
                     config
                         ->minProfitabilityAggregateTerminalHorizonLogReturnSum)
              << " min_average_terminal_horizon_log_return_per_actionable_prediction="
              << ContinuationOptionalDoubleText(
                     config
                         ->minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction)
              << "\n";
    if (selectedEvidence.profitability.has_value())
    {
        const ContinuationProfitabilityEvidence& profitability =
            *selectedEvidence.profitability;
        std::cout << "  Continuation Profitability Evidence: available"
                  << " observation_id=" << profitability.observationId
                  << " scope=" << profitability.inferenceScope
                  << " inference_eval_result_id="
                  << profitability.inferenceEvalResultId
                  << " checkpoint_eval_id="
                  << (profitability.checkpointEvalId.has_value()
                          ? std::to_string(*profitability.checkpointEvalId)
                          : "none")
                  << " metric_definition_hash="
                  << profitability.metricDefinitionHash
                  << " actionable_count=" << profitability.actionableCount
                  << " prediction_count=" << profitability.predictionCount
                  << " aggregate_terminal_horizon_log_return_sum="
                  << ContinuationOptionalDoubleText(
                         profitability
                             .aggregateTerminalHorizonLogReturnSum)
                  << " average_terminal_horizon_log_return_per_actionable_prediction="
                  << ContinuationOptionalDoubleText(
                         profitability
                             .averageTerminalHorizonLogReturnPerActionablePrediction)
                  << "\n";
    }
    else
    {
        std::cout << "  Continuation Profitability Evidence: unavailable"
                  << " reason="
                  << selectedEvidence.profitabilityUnavailableReason
                  << "\n";
    }
    std::cout << "  Continuation Profitability Gate: "
              << currentEvaluation.profitabilityGate.reason
              << " passed="
              << (currentEvaluation.profitabilityGate.passed ? "yes" : "no")
              << "\n";
    std::cout << "  Continuation Current Read-Only Decision: "
              << (currentEvaluationAvailable
                      ? currentEvaluation.decision
                      : "not_evaluated")
              << " reason="
              << (currentEvaluationAvailable
                      ? currentEvaluation.reason
                      : evaluationDeferredReason)
              << "\n";
    std::cout << "CONTINUATION_POLICY_STATUS"
              << ",source_experiment_id=" << sourceExperimentId
              << ",enabled=" << (config->enabled ? "1" : "0")
              << ",policy_enabled=" << (config->enabled ? "1" : "0")
              << ",source_completion_ready=" << (sourceCompletionReady ? "1" : "0")
              << ",evaluation_ready=" << (evaluationReady ? "1" : "0")
              << ",evaluation_deferred_reason="
              << (evaluationDeferredReason.empty() ? "NULL" : evaluationDeferredReason)
              << ",progression_mode="
              << effectiveProgressionMode.value_or("NULL")
              << ",target_sequence="
              << ContinuationTargetSequenceText(config->targetSequence)
              << ",current_experiment_target=" << config->source.targetEpochs
              << ",inherit_to_child=" << (config->inheritToChild ? "1" : "0")
              << ",target_increment="
              << (config->targetIncrement.has_value()
                      ? std::to_string(*config->targetIncrement)
                      : "NULL")
              << ",max_target_epochs="
              << (config->maxTargetEpochs.has_value()
                      ? std::to_string(*config->maxTargetEpochs)
                      : "NULL")
              << ",policy_target_epochs="
              << (config->targetEpochs.has_value()
                      ? std::to_string(*config->targetEpochs)
                      : "NULL")
              << ",derived_next_target="
              << (derivedNextTarget.has_value()
                      ? std::to_string(*derivedNextTarget)
                      : "NULL")
              << ",sequence_next_target="
              << (effectiveProgressionMode == "target_sequence" &&
                          derivedNextTarget.has_value()
                      ? std::to_string(*derivedNextTarget)
                      : "NULL")
              << ",sequence_final_target="
              << (sequenceFinalTarget.has_value()
                      ? std::to_string(*sequenceFinalTarget)
                      : "NULL")
              << ",policy_inherited=" << (config->policyInherited ? "1" : "0")
              << ",inherited_from_experiment_id="
              << (config->inheritedFromExperimentId.has_value()
                      ? std::to_string(*config->inheritedFromExperimentId)
                      : "NULL")
              << ",inherited_from_revision="
              << (config->inheritedFromRevision.has_value()
                      ? std::to_string(*config->inheritedFromRevision)
                      : "NULL")
              << ",inherited_from_hash="
              << config->inheritedFromHash.value_or("NULL")
              << ",inheritance_status=" << config->inheritanceStatus
              << ",terminal="
              << (config->inheritanceStatus == "max_target_reached" ? "1" : "0")
              << ",inheritance_validation=" << outgoingInheritanceValidation
              << ",progression_validation=" << outgoingInheritanceValidation
              << ",current_policy_hash=" << currentPolicyHash
              << ",derived_policy_hash="
              << (derivedPolicyHash.empty() ? "NULL" : derivedPolicyHash)
              << ContinuationProfitabilityEvidenceLogFields(
                     selectedEvidence)
              << ContinuationProfitabilityPolicyLogFields(
                     *config,
                     currentEvaluation.profitabilityGate)
              << ",current_read_only_decision="
              << (currentEvaluationAvailable
                      ? currentEvaluation.decision
                      : "not_evaluated")
              << ",current_read_only_reason="
              << (currentEvaluationAvailable
                      ? currentEvaluation.reason
                      : (evaluationDeferredReason.empty()
                             ? "NULL"
                             : evaluationDeferredReason))
              << std::endl;
    std::cout << "  Continuation Last: decision="
              << (config->lastDecision.has_value() ? *config->lastDecision : "none")
              << " reason=" << (config->lastReason.has_value() ? *config->lastReason : "none")
              << "\n";
    std::cout << "  Continuation Policy Selection: model_id="
              << (config->selectedModelId.has_value() ? std::to_string(*config->selectedModelId) : "none");
    if (!decisions.empty())
        std::cout << " epoch=" << decisions[0][2].as<int>();
    else
        std::cout << " epoch=none";
    std::cout << "\n";
    std::cout << "  Continuation Queued Experiment: "
              << (config->queuedExperimentId.has_value()
                      ? std::to_string(*config->queuedExperimentId)
                      : "none")
              << "\n";
    const bool hasContinuationLineage =
        config->continuationSourceExperimentId.has_value() ||
        config->continuationSourceModelId.has_value() ||
        config->continuationSourceEpoch.has_value() ||
        config->continuationDecisionId.has_value();
    std::cout << "  Continuation Source Experiment: "
              << (config->continuationSourceExperimentId.has_value()
                      ? std::to_string(*config->continuationSourceExperimentId)
                      : "none")
              << "\n";
    std::cout << "  Continuation Source Model: "
              << (config->continuationSourceModelId.has_value()
                      ? std::to_string(*config->continuationSourceModelId)
                      : "none")
              << "\n";
    std::cout << "  Continuation Source Epoch: "
              << (config->continuationSourceEpoch.has_value()
                      ? std::to_string(*config->continuationSourceEpoch)
                      : "none")
              << "\n";
    std::cout << "  Continuation Decision: "
              << (config->continuationDecisionId.has_value()
                      ? std::to_string(*config->continuationDecisionId)
                      : "none")
              << "\n";
    std::cout << "  Continuation Generation: "
              << (hasContinuationLineage
                      ? std::to_string(config->continuationGeneration)
                      : "none")
              << "\n";
    if (!decisions.empty())
    {
        std::cout << "  Decision Identity: id=" << decisions[0][0].as<long long>()
                  << " policy_revision=" << decisions[0][7].as<long long>()
                  << " policy_hash=" << decisions[0][8].as<std::string>()
                  << " evidence_watermark=" << decisions[0][9].as<std::string>()
                  << "\n";
    }
    return 0;
}





int AnalyzeExperimentById(long long experimentId, const SchedulerOptions& options)
{
    LogWorkerStarted(
        "ANALYSIS_WORKER_STARTED",
        "analyze",
        experimentId);

    if (!options.schedulerWorkerAttemptId)
    {
        std::cerr << "EXPERIMENT_ANALYSIS_FAILED"
                  << ",experiment_id=" << experimentId
                  << ",reason=exact_worker_attempt_required"
                  << std::endl;
        return 1;
    }

    const long long workerAttemptId =
        *options.schedulerWorkerAttemptId;
    ExperimentRow experiment;
    ParsedMetrics metrics;
    bool usedStructuredInferenceMetrics = false;
    std::optional<double> leaderScore;
    std::string workError;
    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadOnly(transaction);
        if (!RequireSchedulerTables(transaction))
            return 1;

        pqxx::result rows = transaction.exec_params(
            "SELECT experiment_id,symbol,prediction_horizon,"
            "c_next_threshold,core_lr_mult,head_lr_mult,"
            "target_epochs,checkpoint_interval,train_start::text,"
            "train_end::text,infer_start::text,infer_end::text,"
            "last_model_id,resume_model_id,train_log_path,"
            "infer_log_path,analysis_log_path,donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,fresh_initialization_seed,resume_expand_input_width,training_objective_canonical,training_objective_hash "
            "FROM experiment WHERE experiment_id=$1 "
            "AND status='running' AND phase='analyze' "
            "AND active_scheduler_worker_attempt_id=$2;",
            experimentId,
            workerAttemptId);
        if (rows.size() != 1)
        {
            std::cerr << "EXPERIMENT_ANALYSIS_FAILED"
                      << ",experiment_id=" << experimentId
                      << ",worker_attempt_id=" << workerAttemptId
                      << ",reason=exact_active_attempt_not_bound"
                      << std::endl;
            return 1;
        }

        experiment = RowToExperiment(rows[0]);
        if (!experiment.lastModelId)
        {
            workError = "analyze_missing_last_model_id";
        }
        else
        {
            const long long modelId = *experiment.lastModelId;
            std::cout << "SCHEDULER_ANALYZE_STARTED"
                      << ",experiment_id="
                      << experiment.experimentId
                      << ",worker_attempt_id="
                      << workerAttemptId
                      << ",model_id=" << modelId
                      << std::endl;
            std::cout << "EXPERIMENT_ANALYSIS_STARTED"
                      << ",experiment_id="
                      << experiment.experimentId
                      << ",worker_attempt_id="
                      << workerAttemptId
                      << ",model_id=" << modelId
                      << std::endl;

            metrics = ParseMetricsFromLogs(experiment);
            metrics.modelId = modelId;
            ApplyPersistedSymbolToAnalysisExperiment(
                transaction, experiment, metrics);
            usedStructuredInferenceMetrics =
                ApplyStructuredInferenceMetrics(
                    transaction, experiment, metrics);
            if (usedStructuredInferenceMetrics)
            {
                std::cout
                    << "SCHEDULER_ANALYZE_USING_EXISTING_INFERENCE"
                    << ",model_id=" << modelId
                    << std::endl;
            }
            leaderScore = ComputeLeaderScore(metrics);
        }
        transaction.commit();
    }

    {
        pqxx::connection connection{LstmDbConnectionString()};
        pqxx::work transaction{connection};
        SetTransactionReadWrite(transaction);
        EA::SchedulerOwnership::ExactAttemptExpectation expected;
        expected.workerAttemptId = workerAttemptId;
        expected.experimentId = experimentId;
        expected.workerKind = "experiment";
        expected.lifecyclePhase = "analyze";
        expected.capacityClass = "analyze";
        expected.requireCompleteProcessIdentity = true;
        const auto exact =
            EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
                transaction, expected, true);
        if (!exact)
        {
            transaction.abort();
            std::cerr
                << "SCHEDULER_ANALYZE_STALE_FINALIZER_REJECTED"
                << ",experiment_id=" << experimentId
                << ",worker_attempt_id=" << workerAttemptId
                << ",reason=exact_active_attempt_changed"
                << std::endl;
            return 1;
        }

        if (workError.empty())
        {
            pqxx::result source = transaction.exec_params(
                "SELECT last_model_id FROM experiment "
                "WHERE experiment_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2 "
                "AND status='running' AND phase='analyze' "
                "AND last_model_id=$3;",
                experimentId,
                workerAttemptId,
                *experiment.lastModelId);
            if (source.size() != 1)
                workError =
                    "analyze_source_changed_before_finalize";
        }
        if (workError.empty())
        {
            UpsertAnalysisResult(
                transaction,
                experiment,
                metrics,
                leaderScore);
            pqxx::result completed = transaction.exec_params(
                "UPDATE experiment SET status='completed',"
                "phase='done',worker_pid=NULL,"
                "completed_at=COALESCE(completed_at,clock_timestamp()),"
                "updated_at=clock_timestamp() "
                "WHERE experiment_id=$1 "
                "AND status='running' AND phase='analyze' "
                "AND active_scheduler_worker_attempt_id=$2 "
                "AND last_model_id=$3 "
                "RETURNING experiment_id;",
                experimentId,
                workerAttemptId,
                *experiment.lastModelId);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                completed,
                "complete_final_analysis_exact_attempt");
        }
        else
        {
            pqxx::result failed = transaction.exec_params(
                "UPDATE experiment SET status='failed',"
                "exit_code=-1,error_message=$1,worker_pid=NULL,"
                "completed_at=clock_timestamp(),"
                "updated_at=clock_timestamp() "
                "WHERE experiment_id=$2 "
                "AND status='running' AND phase='analyze' "
                "AND active_scheduler_worker_attempt_id=$3 "
                "RETURNING experiment_id;",
                workError,
                experimentId,
                workerAttemptId);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                failed,
                "fail_final_analysis_exact_attempt");
        }
        transaction.commit();
    }

    if (!workError.empty())
    {
        std::cerr << "SCHEDULER_ANALYZE_FAILED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",worker_attempt_id=" << workerAttemptId
                  << ",model_id="
                  << (experiment.lastModelId
                          ? std::to_string(
                                *experiment.lastModelId)
                          : "none")
                  << ",error=" << workError
                  << std::endl;
        return 1;
    }

    const long long modelId = *experiment.lastModelId;
    std::cout << "SCHEDULER_PHASE_TRANSITION"
              << ",experiment_id=" << experiment.experimentId
              << ",from_phase=analyze"
              << ",to_phase=done"
              << std::endl;
    std::cout << "SCHEDULER_PIPELINE_DONE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_SOURCE"
              << ",experiment_id=" << experiment.experimentId
              << ",source=" << (usedStructuredInferenceMetrics ? "inference_eval_result" : "logs")
              << ",train_log=" << (experiment.trainLogPath.has_value() ? *experiment.trainLogPath : "none")
              << ",infer_log=" << (experiment.inferLogPath.has_value() ? *experiment.inferLogPath : "none")
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_METRIC"
              << ",experiment_id=" << experiment.experimentId
              << ",metric=infer_accuracy"
              << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
              << std::endl;
    std::cout << "EXPERIMENT_LEADER_SCORE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
              << std::endl;
    std::cout << "EXPERIMENT_LEADER_UPDATED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_COMPLETED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << std::endl;
    std::cout << "SCHEDULER_ANALYZE_COMPLETED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;
    std::cout.flush();
    std::cerr.flush();
    TryGenerateExperimentReports(options);
    std::cout.flush();
    std::cerr.flush();
    if (experiment.analysisLogPath.has_value())
    {
        std::ostringstream analysisLog;
        analysisLog << "EXPERIMENT_ANALYSIS_METRIC"
                    << ",experiment_id=" << experiment.experimentId
                    << ",metric=infer_accuracy"
                    << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
                    << "\n";
        analysisLog << "EXPERIMENT_LEADER_SCORE"
                    << ",experiment_id=" << experiment.experimentId
                    << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
                    << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
                    << "\n";
        AppendTextFile(*experiment.analysisLogPath, analysisLog.str());
    }
    return 0;
}



class SchedulerPidPersistenceError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

















































int PrintLeaderboard(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    std::ostringstream sql;
    sql << "SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
        << "a.target_epochs, a.infer_accuracy, a.accept_accuracy, a.accept_rate, a.leader_score "
        << "FROM experiment_analysis_result a "
        << "JOIN experiment e ON e.experiment_id = a.experiment_id "
        << "WHERE COALESCE(a.analysis_scope, 'final') = 'final' ";
    if (options.leaderboardSymbol.has_value())
        sql << "AND a.symbol = " << w.quote(*options.leaderboardSymbol) << " ";
    if (options.leaderboardHorizon.has_value())
        sql << "AND a.prediction_horizon = " << *options.leaderboardHorizon << " ";
    sql << "ORDER BY a.leader_score DESC NULLS LAST LIMIT " << options.leaderboardLimit << ";";

    pqxx::result rows = w.exec(sql.str());
    std::cout << "EXPERIMENT_LEADERBOARD_BEGIN"
              << ",rows=" << rows.size()
              << std::endl;
    std::cout << "experiment_id,model_id,symbol,prediction_horizon,target_epochs,infer_accuracy,accept_accuracy,accept_rate,leader_score"
              << std::endl;
    for (const auto& row : rows)
    {
        std::cout << row[0].c_str() << ","
                  << (row[1].is_null() ? "" : row[1].c_str()) << ","
                  << (row[2].is_null() ? "" : row[2].c_str()) << ","
                  << (row[3].is_null() ? "" : row[3].c_str()) << ","
                  << (row[4].is_null() ? "" : row[4].c_str()) << ","
                  << (row[5].is_null() ? "" : row[5].c_str()) << ","
                  << (row[6].is_null() ? "" : row[6].c_str()) << ","
                  << (row[7].is_null() ? "" : row[7].c_str()) << ","
                  << (row[8].is_null() ? "" : row[8].c_str()) << std::endl;
    }
    std::cout << "EXPERIMENT_LEADERBOARD_DONE"
              << ",rows=" << rows.size()
              << std::endl;
    return 0;
}

bool SchedulerStatusShouldEmitMachineRecords(const SchedulerOptions& options)
{
    return options.logLevel == "summary" || options.logLevel == "diagnostic";
}

bool UseAnsiColors()
{
    const char* term = std::getenv("TERM");
    return ::isatty(STDOUT_FILENO) && term && std::string{term} != "dumb";
}

std::string Colorize(const std::string& value,
                            const std::string& ansiCode,
                            bool useColor)
{
    if (!useColor)
        return value;
    return "\033[" + ansiCode + "m" + value + "\033[0m";
}

std::string ColorForStatus(const std::string& status, bool useColor)
{
    if (status == "running")
        return Colorize(status, "32", useColor);
    if (status == "pending")
        return Colorize(status, "33", useColor);
    if (status == "completed" || status == "done")
        return Colorize(status, "34", useColor);
    if (status == "failed")
        return Colorize(status, "31", useColor);
    return status;
}

std::string OptionalLongLongText(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

std::string OptionalIntText(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

std::string OptionalDoubleText(const std::optional<double>& value, int precision = 1)
{
    if (!value.has_value())
        return "unknown";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << *value;
    return oss.str();
}

std::string OptionalPercentText(const std::optional<double>& value, int precision = 1)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + "%" : "unknown";
}

std::string OptionalMbText(const std::optional<double>& value, int precision = 0)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + " MB" : "unknown";
}

std::string FormatPercentComplete(const SchedulerStatusJob& job)
{
    const std::optional<int> epoch = job.currentEpoch.has_value() ? job.currentEpoch : job.completedEpochs;
    if (!epoch.has_value() || job.targetEpochs <= 0)
        return "unknown";
    const double percent = std::min(100.0,
                                    100.0 * static_cast<double>(*epoch) /
                                        static_cast<double>(job.targetEpochs));
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << percent << "%";
    return oss.str();
}

std::string FormatDurationSeconds(double seconds)
{
    if (seconds < 0.0 || !std::isfinite(seconds))
        return "unknown";
    long long total = static_cast<long long>(std::llround(seconds));
    const long long days = total / 86400;
    total %= 86400;
    const long long hours = total / 3600;
    total %= 3600;
    const long long minutes = total / 60;
    const long long secs = total % 60;

    std::ostringstream oss;
    if (days > 0)
        oss << days << "d ";
    if (hours > 0 || days > 0)
        oss << hours << "h ";
    if (minutes > 0 || hours > 0 || days > 0)
        oss << minutes << "m ";
    oss << secs << "s";
    return oss.str();
}

std::string FormatOptionalDuration(const std::optional<double>& seconds)
{
    return seconds.has_value() ? FormatDurationSeconds(*seconds) : "unknown";
}


std::optional<double> EstimateEtaSeconds(const SchedulerStatusJob& job)
{
    if (!job.currentEpoch.has_value() ||
        !job.elapsedSeconds.has_value() ||
        *job.currentEpoch <= 0 ||
        job.targetEpochs <= 0 ||
        *job.currentEpoch >= job.targetEpochs)
    {
        return std::nullopt;
    }

    const double secondsPerEpoch = *job.elapsedSeconds / static_cast<double>(*job.currentEpoch);
    return secondsPerEpoch * static_cast<double>(job.targetEpochs - *job.currentEpoch);
}

std::string FormatProgressBar(const SchedulerStatusJob& job)
{
    constexpr int width = 20;
    if (!job.currentEpoch.has_value() || job.targetEpochs <= 0)
        return "[--------------------] unknown";

    const double clamped = std::clamp(static_cast<double>(*job.currentEpoch) /
                                          static_cast<double>(job.targetEpochs),
                                      0.0,
                                      1.0);
    const int filled = static_cast<int>(std::llround(clamped * width));
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < width; ++i)
        oss << (i < filled ? "#" : "-");
    oss << "] " << std::fixed << std::setprecision(1) << (100.0 * clamped) << "%";
    return oss.str();
}

std::string ReadFileTailIfExists(const std::optional<std::string>& path,
                                        std::streamoff maxBytes = 262144)
{
    if (!path.has_value() || path->empty())
        return {};

    std::ifstream in(*path, std::ios::binary);
    if (!in)
        return {};

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0)
        return {};

    const std::streamoff start = std::max<std::streamoff>(0, size - maxBytes);
    in.seekg(start, std::ios::beg);
    std::string text;
    text.resize(static_cast<size_t>(size - start));
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    return text;
}

std::optional<int> ExtractLastIntFromText(const std::string& text,
                                                 const std::regex& regex)
{
    std::optional<int> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stoi((*it)[1].str());
    }
    return value;
}

std::optional<long long> ExtractLastLongLongFromText(const std::string& text,
                                                            const std::regex& regex)
{
    std::optional<long long> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stoll((*it)[1].str());
    }
    return value;
}

std::optional<double> ExtractLastDoubleFromText(const std::string& text,
                                                       const std::regex& regex)
{
    std::optional<double> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stod((*it)[1].str());
    }
    return value;
}

std::optional<std::string> ExtractLastProgressLine(const std::string& text)
{
    std::optional<std::string> value;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.find("CHECKPOINT_SAVE") != std::string::npos ||
            line.find("EPOCH_3CLASS_ACCURACY") != std::string::npos ||
            line.find("Saved model with model_id=") != std::string::npos ||
            line.find("RESUME_") != std::string::npos ||
            line.find("Overall 3-class accuracy") != std::string::npos ||
            line.find("EXPERIMENT_ANALYSIS_") != std::string::npos)
        {
            if (line.size() > 160)
                value = line.substr(0, 157) + "...";
            else
                value = line;
        }
    }
    return value;
}

std::optional<long long> ExtractExperimentIdFromCommand(const std::string& command)
{
    std::smatch match;
    if (std::regex_search(command, match, std::regex{R"(experiment([0-9]+))"}))
        return std::stoll(match[1].str());
    if (std::regex_search(command, match, std::regex{R"(--analyze-experiment(?:=|\s+)([0-9]+))"}))
        return std::stoll(match[1].str());
    return std::nullopt;
}

std::string ReadCommandOutput(const std::string& command)
{
    std::string output;
    FILE* pipe = ::popen(command.c_str(), "r");
    if (!pipe)
        return output;

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe))
        output += buffer;
    ::pclose(pipe);
    return output;
}

std::optional<int> ExtractCommandIntOption(const std::string& command,
                                           const std::string& option)
{
    const std::regex regex(option + R"((?:=|\s+)([0-9]+))");
    std::smatch match;
    if (!std::regex_search(command, match, regex))
        return std::nullopt;
    return ParsePositiveInt(option, match[1].str());
}

void AddResourceToAggregate(SchedulerResourceAggregate& aggregate,
                                   const SchedulerProcessResource& resource)
{
    ++aggregate.workers;
    aggregate.cpuPercent += resource.cpuPercent;
    aggregate.memPercent += resource.memPercent;
    aggregate.rssMb += resource.rssMb;
}


std::optional<double> LoadSystemMemoryTotalMb()
{
    const std::string output = ReadCommandOutput("sysctl -n hw.memsize 2>/dev/null");
    std::smatch match;
    if (!std::regex_search(output, match, std::regex{R"(([0-9]+))"}))
        return std::nullopt;
    const double bytes = std::stod(match[1].str());
    return bytes / (1024.0 * 1024.0);
}

std::optional<double> ExtractVmStatPages(const std::string& text,
                                                const std::string& label)
{
    const std::regex regex(label + R"(:\s+([0-9]+)\.)");
    std::smatch match;
    if (!std::regex_search(text, match, regex))
        return std::nullopt;
    return std::stod(match[1].str());
}

std::optional<double> LoadSystemMemoryUsedMb()
{
    const std::string output = ReadCommandOutput("vm_stat 2>/dev/null");
    if (output.empty())
        return std::nullopt;

    double pageSize = 4096.0;
    std::smatch pageMatch;
    if (std::regex_search(output, pageMatch, std::regex{R"(page size of ([0-9]+) bytes)"}))
        pageSize = std::stod(pageMatch[1].str());

    double usedPages = 0.0;
    bool found = false;
    const std::vector<std::string> labels = {
        "Pages active",
        "Pages inactive",
        "Pages speculative",
        "Pages wired down",
        "Pages occupied by compressor"
    };
    for (const auto& label : labels)
    {
        if (const auto pages = ExtractVmStatPages(output, label))
        {
            usedPages += *pages;
            found = true;
        }
    }
    if (!found)
        return std::nullopt;
    return usedPages * pageSize / (1024.0 * 1024.0);
}

SchedulerStatusProcessSnapshot LoadSchedulerStatusProcessSnapshot()
{
    SchedulerStatusProcessSnapshot snapshot;
    const std::string psOutput = ReadCommandOutput(
        "ps -axo pid=,pcpu=,pmem=,rss=,state=,command= 2>/dev/null");
    snapshot.systemMemoryTotalMb = LoadSystemMemoryTotalMb();
    snapshot.systemMemoryUsedMb = LoadSystemMemoryUsedMb();
    if (psOutput.empty())
        return snapshot;

    snapshot.processDetectionAvailable = true;
    std::istringstream stream(psOutput);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
            continue;
        std::istringstream lineStream(line);
        int pid = -1;
        double cpuPercent = 0.0;
        double memPercent = 0.0;
        long long rssKb = 0;
        std::string processState;
        lineStream >> pid;
        lineStream >> cpuPercent;
        lineStream >> memPercent;
        lineStream >> rssKb;
        lineStream >> processState;
        std::string command;
        std::getline(lineStream, command);
        if (pid <= 0)
            continue;

        SchedulerProcessResource resource;
        resource.pid = pid;
        resource.cpuPercent = cpuPercent;
        resource.memPercent = memPercent;
        resource.rssMb = static_cast<double>(rssKb) / 1024.0;
        snapshot.resourcesByPid[pid] = resource;
        snapshot.processes.push_back(SchedulerProcessInfo{pid, command, resource});

        const bool isScheduler =
            IsSchedulerStatusSchedulerProcessCommand(command);
        const bool isLegacyLstm =
            SchedulerStatusCommandHasExecutableBasename(
                command, "LSTM_Release") ||
            SchedulerStatusCommandHasExecutableBasename(command, "LSTM");
        if (!isScheduler && !isLegacyLstm)
            continue;

        if (isScheduler)
        {
            snapshot.schedulerPids.push_back(pid);
            AddResourceToAggregate(snapshot.schedulerResources, resource);
            if (!snapshot.maxTrainProcs.has_value())
                snapshot.maxTrainProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-train-procs");
            if (!snapshot.maxInferProcs.has_value())
                snapshot.maxInferProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-infer-procs");
            if (!snapshot.maxAnalyzeProcs.has_value())
                snapshot.maxAnalyzeProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-analyze-procs");
            if (!snapshot.schedulerPollSeconds.has_value())
                snapshot.schedulerPollSeconds = ExtractCommandIntOption(command, "--scheduler-poll-seconds");
            if (!snapshot.autoQueueContinuations.has_value())
            {
                const bool autoQueue =
                    command.find("--auto-queue-continuations") != std::string::npos;
                const bool autoEvaluate = autoQueue ||
                    command.find("--auto-evaluate-continuations") != std::string::npos;
                snapshot.autoQueueContinuations = autoQueue;
                snapshot.autoEvaluateContinuations = autoEvaluate;
                snapshot.continuationDryRun =
                    command.find("--continuation-dry-run") != std::string::npos ||
                    command.find("--dry-run") != std::string::npos;
                snapshot.continuationScanSeconds =
                    ExtractCommandIntOption(command, "--continuation-scan-seconds").value_or(300);
                snapshot.continuationMaxQueuesPerScan =
                    ExtractCommandIntOption(command, "--continuation-max-queues-per-scan").value_or(1);
            }
        }
        else if (command.find("--train") != std::string::npos)
        {
            ++snapshot.trainWorkers;
            AddResourceToAggregate(snapshot.trainResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "train", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.trainPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--infer") != std::string::npos)
        {
            ++snapshot.inferWorkers;
            AddResourceToAggregate(snapshot.inferResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "infer", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.inferPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--analyze-experiment") != std::string::npos ||
                 command.find("--analyze-completed-experiments") != std::string::npos)
        {
            ++snapshot.analysisWorkers;
            AddResourceToAggregate(snapshot.analysisResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "analyze", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.analysisPidByExperiment[*experimentId] = pid;
        }
    }
    snapshot.totalCpuPercent = snapshot.schedulerResources.cpuPercent +
                               snapshot.trainResources.cpuPercent +
                               snapshot.inferResources.cpuPercent +
                               snapshot.analysisResources.cpuPercent;
    return snapshot;
}


bool CommandContainsOptionValue(const std::string& command,
                                       const std::string& option,
                                       long long value)
{
    const std::string valueText = std::to_string(value);
    return command.find(option + "=" + valueText) != std::string::npos ||
           command.find(option + " " + valueText) != std::string::npos;
}

std::optional<SchedulerStopExperiment> LoadStopExperiment(pqxx::work& w,
                                                                 long long experimentId,
                                                                 bool forUpdate)
{
    std::ostringstream sql;
    sql << "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        << "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        << "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        << "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, "
        << "donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,fresh_initialization_seed,resume_expand_input_width,training_objective_canonical,training_objective_hash,status, phase,worker_pid,worker_process_group_id,"
        << "worker_executable,worker_command_line,"
        << "worker_process_start_identity,"
        << "active_scheduler_worker_attempt_id "
        << "FROM experiment WHERE experiment_id = " << experimentId;
    if (forUpdate)
        sql << " FOR UPDATE";
    sql << ";";

    pqxx::result rows = w.exec(sql.str());
    if (rows.empty())
        return std::nullopt;

    SchedulerStopExperiment result;
    result.experiment = RowToExperiment(rows[0]);
    result.status = rows[0][25].as<std::string>();
    result.phase = rows[0][26].as<std::string>();
    result.worker.experimentId = result.experiment.experimentId;
    result.worker.phase = result.phase;
    result.worker.lifecycleStatus = result.status;
    if (!rows[0][27].is_null()) result.worker.pid = rows[0][27].as<int>();
    if (!rows[0][28].is_null()) result.worker.processGroupId = rows[0][28].as<int>();
    result.worker.executable = OptionalStringCell(rows[0], 29);
    result.worker.commandLine = OptionalStringCell(rows[0], 30);
    result.worker.processStartIdentity =
        OptionalStringCell(rows[0], 31);
    result.activeWorkerAttemptId =
        OptionalLongLongCell(rows[0], 32);
    result.worker.workerAttemptId =
        result.activeWorkerAttemptId;
    result.worker.workerKind = "experiment";
    result.worker.capacityClass = result.phase;
    return result;
}

std::vector<SchedulerStopExperiment> LoadRunningStopExperiments(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, "
        "donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,fresh_initialization_seed,resume_expand_input_width,training_objective_canonical,training_objective_hash,status, phase,worker_pid,worker_process_group_id,"
        "worker_executable,worker_command_line,"
        "worker_process_start_identity,"
        "active_scheduler_worker_attempt_id "
        "FROM experiment "
        "WHERE status = 'running' AND phase IN ('train', 'infer', 'analyze') "
        "ORDER BY updated_at ASC, experiment_id ASC;");

    std::vector<SchedulerStopExperiment> experiments;
    experiments.reserve(rows.size());
    for (const auto& row : rows)
    {
        SchedulerStopExperiment item;
        item.experiment = RowToExperiment(row);
        item.status = row[25].as<std::string>();
        item.phase = row[26].as<std::string>();
        item.worker.experimentId =
            item.experiment.experimentId;
        item.worker.phase = item.phase;
        item.worker.lifecycleStatus = item.status;
        if (!row[27].is_null()) item.worker.pid = row[27].as<int>();
        if (!row[28].is_null()) item.worker.processGroupId = row[28].as<int>();
        item.worker.executable = OptionalStringCell(row, 29);
        item.worker.commandLine = OptionalStringCell(row, 30);
        item.worker.processStartIdentity =
            OptionalStringCell(row, 31);
        item.activeWorkerAttemptId =
            OptionalLongLongCell(row, 32);
        item.worker.workerAttemptId =
            item.activeWorkerAttemptId;
        item.worker.workerKind = "experiment";
        item.worker.capacityClass = item.phase;
        experiments.push_back(item);
    }
    return experiments;
}

SchedulerStopCandidate BuildStopCandidate(const SchedulerStopExperiment& experiment,
                                                 const SchedulerStatusProcessSnapshot& processes)
{
    (void)processes;
    SchedulerStopCandidate candidate;
    candidate.experiment = experiment;

    if (experiment.status != "running")
    {
        candidate.rejectionReason = "experiment_not_running";
        return candidate;
    }
    if (experiment.phase != "train" &&
        experiment.phase != "infer" &&
        experiment.phase != "analyze")
    {
        candidate.rejectionReason = "invalid_running_phase";
        return candidate;
    }
    if (!experiment.activeWorkerAttemptId.has_value())
    {
        candidate.rejectionReason =
            "active_worker_attempt_identity_missing";
        return candidate;
    }

    std::unique_ptr<EA::GlobalExperimentControl::ProcessOperations>
        processOperations =
            EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const EA::GlobalExperimentControl::ValidatedWorker validated =
        EA::GlobalExperimentControl::ValidateManagedWorker(
            experiment.worker, *processOperations);
    if (validated.identity !=
        EA::GlobalExperimentControl::IdentityResult::Validated)
    {
        candidate.rejectionReason =
            "worker_identity_not_validated:" + validated.detail;
        return candidate;
    }
    candidate.pid = validated.observation.pid;
    return candidate;
}

void PrintSchedulerStopAttempt(const SchedulerStopCandidate& candidate, bool force)
{
    std::cout << "SCHEDULER_STOP_ATTEMPT"
              << ",experiment_id=" << candidate.experiment.experiment.experimentId
              << ",phase=" << candidate.experiment.phase
              << ",pid=" << (candidate.pid.has_value() ? std::to_string(*candidate.pid) : "unknown")
              << ",force=" << (force ? "1" : "0")
              << std::endl;
}

void PrintSchedulerStopRejected(long long experimentId, const std::string& reason)
{
    std::cout << "SCHEDULER_STOP_REJECTED"
              << ",experiment_id=" << experimentId
              << ",reason=" << reason
              << std::endl;
}

void PrintSchedulerStopPreview(const SchedulerStopCandidate& candidate, bool force)
{
    const auto& row = candidate.experiment;
    std::cout << "Experiment " << row.experiment.experimentId << "\n"
              << "Current: status=" << row.status << " phase=" << row.phase << "\n"
              << "Matched PID: " << (candidate.pid.has_value() ? std::to_string(*candidate.pid) : "unknown") << "\n"
              << "Requested: stop worker with SIGTERM";
    if (force)
        std::cout << " then SIGKILL if still running";
    std::cout << "\n";
}

bool SignalProcessForStop(const SchedulerStopCandidate& candidate,
                                 bool force,
                                 std::string& errorMessage,
                                 bool& usedSigkill)
{
    usedSigkill = false;
    if (!candidate.pid.has_value())
    {
        errorMessage = "pid_not_found";
        return false;
    }

    const pid_t pid = static_cast<pid_t>(*candidate.pid);
    const long long experimentId =
        candidate.experiment.experiment.experimentId;
    std::unique_ptr<EA::GlobalExperimentControl::ProcessOperations>
        processes =
            EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const EA::GlobalExperimentControl::ValidatedWorker validated =
        EA::GlobalExperimentControl::ValidateManagedWorker(
            candidate.experiment.worker, *processes);
    if (validated.identity !=
        EA::GlobalExperimentControl::IdentityResult::Validated)
    {
        errorMessage =
            "worker_identity_not_validated:" + validated.detail;
        return false;
    }

    std::cout << "SCHEDULER_STOP_SIGNAL"
              << ",experiment_id=" << experimentId
              << ",pid=" << pid
              << ",signal=SIGTERM"
              << std::endl;
    if (force)
    {
        const EA::GlobalExperimentControl::SignalOutcome outcome =
            EA::GlobalExperimentControl::CancelWorker(
                candidate.experiment.worker,
                false,
                std::chrono::seconds(3),
                *processes);
        usedSigkill =
            std::find(
                outcome.signals.begin(),
                outcome.signals.end(),
                SIGKILL) != outcome.signals.end();
        if (!outcome.success)
        {
            errorMessage =
                outcome.result + ":" + outcome.detail;
            return false;
        }
        return true;
    }

    int errorNumber = 0;
    if (!processes->SignalProcessGroup(
            validated.observation.processGroupId,
            SIGTERM,
            errorNumber))
    {
        errorMessage =
            "sigterm_failed:" +
            std::string{std::strerror(errorNumber)};
        return false;
    }
    if (processes->WaitForProcessGroupExit(
            validated.observation.processGroupId,
            std::chrono::seconds(3)))
        return true;

    errorMessage = "process_still_running";
    return false;
}

void MarkExperimentStopped(pqxx::work& w,
                                  const SchedulerStopExperiment& stopped,
                                  const std::string& errorMessage,
                                  int exitCode)
{
    const long long experimentId =
        stopped.experiment.experimentId;
    const long long workerAttemptId =
        *stopped.activeWorkerAttemptId;
    pqxx::result attempt = w.exec_params(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state='failed',completed_at=clock_timestamp(),"
        "exit_code=$1,reconciliation_result='operator_stop',"
        "diagnostic=$2,last_observed_at=clock_timestamp() "
        "WHERE a.worker_attempt_id=$3 "
        "AND a.worker_kind='experiment' "
        "AND a.experiment_id=$4 "
        "AND a.lifecycle_phase=$5 "
        "AND a.lifecycle_state IN "
        "('spawned','running','observed') "
        "AND EXISTS (SELECT 1 FROM experiment e "
        " WHERE e.experiment_id=$4 "
        " AND e.status='running' AND e.phase=$5 "
        " AND e.active_scheduler_worker_attempt_id="
        "a.worker_attempt_id) "
        "RETURNING a.worker_attempt_id;",
        exitCode,
        errorMessage,
        workerAttemptId,
        experimentId,
        stopped.phase);
    if (attempt.size() != 1)
        throw std::runtime_error(
            "operator_stop_worker_attempt_predicate_rejected");
    pqxx::result lifecycle = w.exec_params(
        "UPDATE experiment "
        "SET status = 'cancelled', completed_at = now(), updated_at = now(), "
        "exit_code = $1, error_message = $2,"
        "active_scheduler_worker_attempt_id=NULL,"
        "worker_pid=NULL,worker_process_group_id=NULL "
        "WHERE experiment_id = $3 "
        "AND active_scheduler_worker_attempt_id=$4 "
        "AND status='running' AND phase=$5 "
        "AND worker_pid=$6 "
        "AND worker_process_group_id IS NOT DISTINCT FROM $7 "
        "AND worker_process_start_identity IS NOT DISTINCT FROM $8 "
        "AND worker_executable IS NOT DISTINCT FROM $9 "
        "AND worker_command_line IS NOT DISTINCT FROM $10 "
        "RETURNING experiment_id;",
        exitCode,
        errorMessage,
        experimentId,
        workerAttemptId,
        stopped.phase,
        stopped.worker.pid,
        stopped.worker.processGroupId,
        stopped.worker.processStartIdentity,
        stopped.worker.executable,
        stopped.worker.commandLine);
    if (lifecycle.size() != 1)
        throw std::runtime_error(
            "operator_stop_lifecycle_predicate_rejected");
}

bool ApplyStopCandidate(const SchedulerStopCandidate& originalCandidate,
                               const SchedulerOptions& options,
                               std::string& failureReason)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
    {
        failureReason = "scheduler_tables_missing";
        return false;
    }

    const long long experimentId =
        originalCandidate.experiment.experiment.experimentId;
    if (!originalCandidate.experiment.activeWorkerAttemptId)
    {
        failureReason = "active_worker_attempt_identity_missing";
        w.commit();
        return false;
    }
    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId =
        *originalCandidate.experiment.activeWorkerAttemptId;
    expected.experimentId = experimentId;
    expected.workerKind = "experiment";
    expected.lifecyclePhase =
        originalCandidate.experiment.phase;
    expected.capacityClass =
        originalCandidate.experiment.phase;
    expected.requireSignalable = true;
    expected.requireCompleteProcessIdentity = true;
    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            w, expected, true);
    if (!exact)
    {
        failureReason =
            "exact_active_worker_attempt_verification_failed";
        w.commit();
        return false;
    }
    const std::optional<SchedulerStopExperiment> locked =
        LoadStopExperiment(w, experimentId, false);
    if (!locked.has_value())
    {
        failureReason = "experiment_not_found";
        return false;
    }

    const SchedulerStatusProcessSnapshot freshProcesses = LoadSchedulerStatusProcessSnapshot();
    SchedulerStopCandidate candidate = BuildStopCandidate(*locked, freshProcesses);
    if (!candidate.pid.has_value())
    {
        failureReason = candidate.rejectionReason;
        w.commit();
        return false;
    }
    if (originalCandidate.pid.has_value() && candidate.pid != originalCandidate.pid)
    {
        failureReason = "pid_changed";
        w.commit();
        return false;
    }

    std::string signalError;
    bool usedSigkill = false;
    if (!SignalProcessForStop(candidate, options.force, signalError, usedSigkill))
    {
        failureReason = signalError;
        w.commit();
        return false;
    }

    const std::string message = options.force ? "force_stopped_by_user" : "stopped_by_user";
    MarkExperimentStopped(
        w,
        *locked,
        message,
        usedSigkill ? 137 : 143);
    std::cout << "SCHEDULER_STOP_APPLIED"
              << ",experiment_id=" << experimentId
              << ",new_status=cancelled"
              << ",new_phase=" << locked->phase
              << ",error_message=" << message
              << std::endl;
    w.commit();
    return true;
}

int RunStopExperimentCommand(const SchedulerOptions& options)
{
    const bool isStopAll = options.stopAllExperiments;
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 2;

    std::vector<SchedulerStopExperiment> experiments;
    if (isStopAll)
    {
        experiments = LoadRunningStopExperiments(w);
    }
    else
    {
        const std::optional<SchedulerStopExperiment> experiment =
            LoadStopExperiment(w, *options.stopExperimentId, false);
        if (!experiment.has_value())
        {
            PrintSchedulerStopRejected(*options.stopExperimentId, "experiment_not_found");
            w.commit();
            return 1;
        }
        experiments.push_back(*experiment);
    }
    w.commit();

    int matched = 0;
    int stopped = 0;
    int rejected = 0;
    int failed = 0;

    for (const auto& experiment : experiments)
    {
        SchedulerStopCandidate candidate = BuildStopCandidate(experiment, processes);
        if (!candidate.pid.has_value())
        {
            ++rejected;
            PrintSchedulerStopRejected(experiment.experiment.experimentId, candidate.rejectionReason);
            continue;
        }

        ++matched;
        PrintSchedulerStopAttempt(candidate, options.force);
        PrintSchedulerStopPreview(candidate, options.force);

        if (options.dryRun)
        {
            std::cout << "SCHEDULER_STOP_DRY_RUN"
                      << ",experiment_id=" << experiment.experiment.experimentId
                      << ",phase=" << experiment.phase
                      << ",pid=" << *candidate.pid
                      << std::endl;
            continue;
        }

        if (!options.yes)
        {
            std::cout << "Use --yes to apply." << std::endl;
            continue;
        }

        std::string failureReason;
        if (ApplyStopCandidate(candidate, options, failureReason))
        {
            ++stopped;
        }
        else
        {
            ++failed;
            std::cout << "SCHEDULER_STOP_FAILED"
                      << ",experiment_id=" << experiment.experiment.experimentId
                      << ",reason=" << failureReason
                      << std::endl;
        }
    }

    if (isStopAll)
    {
        std::cout << "SCHEDULER_STOP_ALL_SUMMARY"
                  << ",matched=" << matched
                  << ",stopped=" << stopped
                  << ",rejected=" << rejected
                  << ",failed=" << failed
                  << std::endl;
    }

    return failed > 0 || (!isStopAll && rejected > 0) ? 1 : 0;
}

SchedulerStatusCounts LoadSchedulerStatusCounts(pqxx::work& w)
{
    SchedulerStatusCounts counts;
    pqxx::result rows = w.exec(
        "SELECT status, count(*) "
        "FROM experiment "
        "GROUP BY status;");
    for (const auto& row : rows)
    {
        const std::string status = row[0].as<std::string>();
        const int count = row[1].as<int>();
        if (status == "pending")
            counts.queued = count;
        else if (status == "paused")
            counts.paused = count;
        else if (status == "running")
            counts.running = count;
        else if (status == "completed")
            counts.completed = count;
        else if (status == "failed")
            counts.failed = count;
        else if (status == "cancelled")
            counts.cancelled = count;
    }
    return counts;
}

std::vector<EA::GlobalExperimentControl::ManagedWorker>
LoadAuthoritativeSchedulerWorkers(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT e.experiment_id,NULL::bigint AS checkpoint_eval_id,"
        "e.status,e.phase,a.worker_pid,a.worker_process_group_id,"
        "a.canonical_executable_path,a.command_line,"
        "a.worker_process_start_identity,a.worker_attempt_id,"
        "a.worker_kind,a.capacity_class,a.ownership_origin,"
        "a.lifecycle_state,"
        "a.launch_attempt_identity,"
        "(e.worker_pid IS NOT DISTINCT FROM a.worker_pid AND "
        " e.worker_process_group_id IS NOT DISTINCT FROM "
        "     a.worker_process_group_id AND "
        " e.worker_process_start_identity IS NOT DISTINCT FROM "
        "     a.worker_process_start_identity AND "
        " e.worker_executable IS NOT DISTINCT FROM "
        "     a.canonical_executable_path AND "
        " e.worker_command_line IS NOT DISTINCT FROM a.command_line) "
        "FROM experiment e "
        "JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "WHERE e.status IN ('running','paused') "
        "AND e.phase IN ('train','infer','analyze') "
        "UNION ALL "
        "SELECT COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.checkpoint_eval_id,ce.status,'checkpoint_infer',a.worker_pid,"
        "a.worker_process_group_id,a.canonical_executable_path,"
        "a.command_line,a.worker_process_start_identity,"
        "a.worker_attempt_id,a.worker_kind,a.capacity_class,"
        "a.ownership_origin,a.lifecycle_state,a.launch_attempt_identity,"
        "(ce.worker_pid IS NOT DISTINCT FROM a.worker_pid AND "
        " ce.worker_process_group_id IS NOT DISTINCT FROM "
        "     a.worker_process_group_id AND "
        " ce.worker_process_start_identity IS NOT DISTINCT FROM "
        "     a.worker_process_start_identity AND "
        " ce.worker_executable IS NOT DISTINCT FROM "
        "     a.canonical_executable_path AND "
        " ce.worker_command_line IS NOT DISTINCT FROM a.command_line) "
        "FROM experiment_checkpoint_eval ce "
        "JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=ce.active_scheduler_worker_attempt_id "
        "WHERE ce.status='running' AND ce.phase='infer' "
        "ORDER BY 1,2 NULLS FIRST;");

    std::vector<EA::GlobalExperimentControl::ManagedWorker> workers;
    workers.reserve(rows.size());
    for (const auto& row : rows)
    {
        EA::GlobalExperimentControl::ManagedWorker worker;
        worker.experimentId = row[0].as<long long>();
        worker.checkpointEvalId = OptionalLongLongCell(row, 1);
        worker.lifecycleStatus = row[2].as<std::string>();
        worker.phase = row[3].as<std::string>();
        if (!row[4].is_null())
            worker.pid = row[4].as<int>();
        if (!row[5].is_null())
            worker.processGroupId = row[5].as<int>();
        worker.executable = OptionalStringCell(row, 6);
        worker.commandLine = OptionalStringCell(row, 7);
        worker.processStartIdentity = OptionalStringCell(row, 8);
        worker.workerAttemptId = OptionalLongLongCell(row, 9);
        worker.workerKind = row[10].as<std::string>();
        worker.capacityClass = row[11].as<std::string>();
        worker.ownershipOrigin = row[12].as<std::string>();
        worker.attemptLifecycleState = row[13].as<std::string>();
        worker.launchAttemptIdentity = row[14].as<std::string>();
        worker.authoritativeBindingMatches = row[15].as<bool>();
        workers.push_back(std::move(worker));
    }
    return workers;
}

std::vector<SchedulerCheckpointStatusJob>
LoadActiveCheckpointStatusJobs(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT ce.checkpoint_eval_id,"
        "COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.checkpoint_epoch,ce.checkpoint_model_id,"
        "COALESCE(ce.symbol,e.symbol),"
        "COALESCE(ce.prediction_horizon,e.prediction_horizon),"
        "ce.status,ce.phase,ce.worker_pid,ce.worker_control_state,"
        "ce.infer_log_path "
        "FROM experiment_checkpoint_eval ce "
        "LEFT JOIN experiment e "
        "ON e.experiment_id="
        "COALESCE(ce.parent_experiment_id,ce.experiment_id) "
        "WHERE ce.status='running' AND ce.phase='infer' "
        "ORDER BY ce.created_at,ce.checkpoint_eval_id;");

    std::vector<SchedulerCheckpointStatusJob> jobs;
    jobs.reserve(rows.size());
    for (const auto& row : rows)
    {
        SchedulerCheckpointStatusJob job;
        job.checkpointEvalId = row[0].as<long long>();
        job.experimentId = row[1].as<long long>();
        job.checkpointEpoch = row[2].as<int>();
        job.checkpointModelId = row[3].as<long long>();
        job.symbol =
            row[4].is_null() ? "unknown" : row[4].as<std::string>();
        job.predictionHorizon =
            row[5].is_null() ? 0 : row[5].as<int>();
        job.status = row[6].as<std::string>();
        job.phase = row[7].as<std::string>();
        if (!row[8].is_null())
            job.pid = row[8].as<int>();
        job.workerControlState =
            row[9].is_null() ? "unknown" : row[9].as<std::string>();
        job.inferLogPath = OptionalStringCell(row, 10);
        jobs.push_back(std::move(job));
    }
    return jobs;
}

std::string OptionalLongLongText(const std::optional<long long>& value);
std::string OptionalIntText(const std::optional<int>& value);
std::string OptionalDoubleText(const std::optional<double>& value, int precision);
std::string CurrentOperationForStatusJob(const SchedulerStatusJob& job);

std::vector<double> LoadMatrixRowValuesOrEmpty(pqxx::work& w,
                                               long long modelId,
                                               const std::string& paramName)
{
    pqxx::result rows = w.exec_params(
        "SELECT value FROM matrix "
        "WHERE model_id = $1 AND param_name = $2 AND row_idx = 0 "
        "ORDER BY col_idx;",
        modelId,
        paramName);
    std::vector<double> values;
    values.reserve(rows.size());
    for (const auto& row : rows)
        values.push_back(row[0].as<double>());
    return values;
}

std::string DecodeAsciiMatrixOrUnknown(pqxx::work& w,
                                       long long modelId,
                                       const std::string& paramName)
{
    try
    {
        const std::vector<double> values = LoadMatrixRowValuesOrEmpty(w, modelId, paramName);
        if (values.empty())
            return "unknown";
        std::string decoded;
        decoded.reserve(values.size());
        for (double value : values)
        {
            const long long code = static_cast<long long>(std::llround(value));
            if (code <= 0 || code > 255)
                return "unknown";
            decoded.push_back(static_cast<char>(code));
        }
        return decoded.empty() ? "unknown" : decoded;
    }
    catch (const std::exception&)
    {
        return "unknown";
    }
}

std::pair<std::string, std::string> DecodeRangeMetaOrUnknown(pqxx::work& w,
                                                             long long modelId,
                                                             const std::string& paramName)
{
    const std::string encoded = DecodeAsciiMatrixOrUnknown(w, modelId, paramName);
    const size_t sep = encoded.find('|');
    if (encoded == "unknown" || sep == std::string::npos)
        return {"unknown", "unknown"};
    return {encoded.substr(0, sep), encoded.substr(sep + 1)};
}

std::string TargetTypeNameFromMetaValue(int value)
{
    switch (value)
    {
        case 0: return "Return";
        case 1: return "Direction";
        case 2: return "UpNeutralDownReturn";
        default: return "unknown";
    }
}

std::string OptimizerNameFromMetaValue(int value)
{
    if (value == DBIO::PgModelIO::kOptimizerTypeSgd)
        return "SGD";
    return "unknown";
}

std::string OptionalDateRangeText(const std::string& start, const std::string& end)
{
    if (start == "unknown" && end == "unknown")
        return "unknown";
    return start + " -> " + end;
}

std::string YesNoText(bool value)
{
    return value ? "yes" : "no";
}

void PrintModelInfoField(const std::string& label, const std::string& value)
{
    std::cout << std::left << std::setw(24) << (label + ":") << value << "\n";
}

ModelInfoRecord LoadModelInfo(pqxx::work& w, long long modelId)
{
    const bool hasModelExperimentId = ColumnExists(w, "model", "experiment_id");
    pqxx::result modelRows = w.exec_params(
        "SELECT model_id, " +
        std::string{hasModelExperimentId ? "experiment_id" : "NULL::bigint"} +
        ", name, created_at::text, COALESCE(comment, '') "
        "FROM model WHERE model_id = $1;",
        modelId);
    if (modelRows.empty())
        throw std::runtime_error("model not found");

    ModelInfoRecord info;
    info.modelId = modelRows[0][0].as<long long>();
    if (!modelRows[0][1].is_null())
        info.experimentId = modelRows[0][1].as<long long>();
    info.name = modelRows[0][2].as<std::string>();
    info.createdAt = modelRows[0][3].as<std::string>();
    info.comment = modelRows[0][4].as<std::string>();
    if (info.comment.empty())
        info.comment = "unknown";

    info.symbol = DecodeAsciiMatrixOrUnknown(w, modelId, "train_symbol_meta");
    const auto trainRange = DecodeRangeMetaOrUnknown(w, modelId, "train_range_meta");
    info.trainStart = trainRange.first;
    info.trainEnd = trainRange.second;

    const std::vector<double> trainConfig = LoadMatrixRowValuesOrEmpty(w, modelId, "train_config_meta");
    if (trainConfig.size() > 1)
        info.predictionHorizon = static_cast<int>(std::llround(trainConfig[1]));
    if (trainConfig.size() > 2)
        info.threshold = trainConfig[2];
    if (trainConfig.size() > 3)
        info.windowSize = static_cast<int>(std::llround(trainConfig[3]));
    if (trainConfig.size() > 10)
        info.completedEpochs = static_cast<int>(std::llround(trainConfig[10]));

    const std::vector<double> targetMeta = LoadMatrixRowValuesOrEmpty(w, modelId, "target_meta");
    if (!targetMeta.empty())
        info.targetType = TargetTypeNameFromMetaValue(static_cast<int>(std::llround(targetMeta[0])));

    const std::vector<double> optimizerMeta = LoadMatrixRowValuesOrEmpty(w, modelId, "optimizer_meta");
    if (optimizerMeta.size() > 1)
        info.optimizer = OptimizerNameFromMetaValue(static_cast<int>(std::llround(optimizerMeta[1])));
    if (optimizerMeta.size() > 2)
        info.optimizerUpdateCount = static_cast<long long>(std::llround(optimizerMeta[2]));

    pqxx::result experimentRows;
    if (info.experimentId.has_value())
    {
        experimentRows = w.exec_params(
            "SELECT target_epochs, checkpoint_interval, "
            "train_start::date::text, train_end::date::text, "
            "infer_start::date::text, infer_end::date::text, "
            "resume_model_id, last_model_id "
            "FROM experiment "
            "WHERE experiment_id = $1 OR last_model_id = $2 OR resume_model_id = $2 "
            "ORDER BY CASE WHEN experiment_id = $1 THEN 0 WHEN last_model_id = $2 THEN 1 ELSE 2 END, "
            "updated_at DESC "
            "LIMIT 1;",
            *info.experimentId,
            modelId);
    }
    else
    {
        experimentRows = w.exec_params(
            "SELECT target_epochs, checkpoint_interval, "
            "train_start::date::text, train_end::date::text, "
            "infer_start::date::text, infer_end::date::text, "
            "resume_model_id, last_model_id "
            "FROM experiment "
            "WHERE last_model_id = $1 OR resume_model_id = $1 "
            "ORDER BY CASE WHEN last_model_id = $1 THEN 0 ELSE 1 END, updated_at DESC "
            "LIMIT 1;",
            modelId);
    }
    if (!experimentRows.empty())
    {
        info.targetEpochs = experimentRows[0][0].as<int>();
        info.checkpointInterval = experimentRows[0][1].as<int>();
        if (!experimentRows[0][2].is_null())
            info.trainStart = experimentRows[0][2].as<std::string>();
        if (!experimentRows[0][3].is_null())
            info.trainEnd = experimentRows[0][3].as<std::string>();
        if (!experimentRows[0][4].is_null())
            info.inferStart = experimentRows[0][4].as<std::string>();
        if (!experimentRows[0][5].is_null())
            info.inferEnd = experimentRows[0][5].as<std::string>();
        if (!experimentRows[0][6].is_null())
            info.parentResumeModelId = experimentRows[0][6].as<long long>();
    }

    pqxx::result analysisRows = w.exec_params(
        "SELECT target_epochs, completed_epochs "
        "FROM experiment_analysis_result "
        "WHERE model_id = $1 "
        "ORDER BY updated_at DESC "
        "LIMIT 1;",
        modelId);
    if (!analysisRows.empty())
    {
        if (!info.targetEpochs.has_value() && !analysisRows[0][0].is_null())
            info.targetEpochs = analysisRows[0][0].as<int>();
        if (!info.completedEpochs.has_value() && !analysisRows[0][1].is_null())
            info.completedEpochs = analysisRows[0][1].as<int>();
    }

    pqxx::result inferenceRows = w.exec_params(
        "SELECT from_date::text, to_date::text, completed_epochs "
        "FROM inference_eval_result "
        "WHERE model_id = $1 AND status = 'completed' "
        "AND inference_scope = 'final' AND checkpoint_eval_id IS NULL "
        "ORDER BY completed_at DESC, id DESC "
        "LIMIT 1;",
        modelId);
    if (!inferenceRows.empty())
    {
        info.inferStart = inferenceRows[0][0].as<std::string>();
        info.inferEnd = inferenceRows[0][1].as<std::string>();
        if (!info.completedEpochs.has_value() && !inferenceRows[0][2].is_null())
            info.completedEpochs = static_cast<int>(inferenceRows[0][2].as<long long>());
    }

    info.isCheckpoint =
        info.comment.find("periodic training checkpoint") != std::string::npos ||
        std::regex_search(info.name, std::regex{R"(_epoch[0-9]+)"});
    try
    {
        (void)LoadQueueResumeMeta(w, modelId);
        info.isResumable = DBIO::PgModelIO::hasTrainingResumeState(w, modelId);
    }
    catch (const std::exception&)
    {
        info.isResumable = false;
    }

    return info;
}

int PrintModelInfo(long long modelId)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    ModelInfoRecord info;
    try
    {
        info = LoadModelInfo(w, modelId);
    }
    catch (const std::exception& e)
    {
        w.commit();
        if (std::string{e.what()} == "model not found")
        {
            std::cerr << "ERROR: model not found." << std::endl;
            return 1;
        }
        throw;
    }
    w.commit();

    std::cout << "MODEL INFORMATION\n"
              << "-----------------\n";
    PrintModelInfoField("Model ID", std::to_string(info.modelId));
    PrintModelInfoField("Experiment ID", OptionalLongLongText(info.experimentId));
    PrintModelInfoField("Name", info.name);
    PrintModelInfoField("Symbol", info.symbol);
    PrintModelInfoField("Prediction Horizon", OptionalIntText(info.predictionHorizon));
    PrintModelInfoField("Completed Epochs", OptionalIntText(info.completedEpochs));
    PrintModelInfoField("Target Epochs", OptionalIntText(info.targetEpochs));
    PrintModelInfoField("Training Range", OptionalDateRangeText(info.trainStart, info.trainEnd));
    PrintModelInfoField("Inference Range", OptionalDateRangeText(info.inferStart, info.inferEnd));
    PrintModelInfoField("Threshold", OptionalDoubleText(info.threshold, 8));
    PrintModelInfoField("Window Size", OptionalIntText(info.windowSize));
    PrintModelInfoField("Checkpoint Interval", OptionalIntText(info.checkpointInterval));
    PrintModelInfoField("Optimizer", info.optimizer);
    PrintModelInfoField("Optimizer Updates", OptionalLongLongText(info.optimizerUpdateCount));
    PrintModelInfoField("Target Type", info.targetType);
    PrintModelInfoField("Created", info.createdAt);
    PrintModelInfoField("Comment", info.comment);
    PrintModelInfoField("Checkpoint", YesNoText(info.isCheckpoint));
    PrintModelInfoField("Resumable", YesNoText(info.isResumable));
    PrintModelInfoField("Parent Resume Model", OptionalLongLongText(info.parentResumeModelId));
    return 0;
}

std::vector<ExperimentModelRow> LoadExperimentModelRows(pqxx::work& w, long long experimentId)
{
    const bool hasExperimentId = ColumnExists(w, "model", "experiment_id");
    const bool hasParentModelId = ColumnExists(w, "model", "parent_model_id");
    const std::string legacyPattern = "%experiment" + std::to_string(experimentId) + "%";
    pqxx::result rows;
    if (hasExperimentId)
    {
        rows = w.exec_params(
            "SELECT model_id, experiment_id, " +
            std::string{hasParentModelId ? "parent_model_id" : "NULL::bigint"} +
            ", COALESCE(name, ''), created_at::text, COALESCE(comment, '') "
            "FROM model "
            "WHERE experiment_id = $1 "
            "   OR (experiment_id IS NULL AND COALESCE(name, '') LIKE $2) "
            "ORDER BY model_id;",
            experimentId,
            legacyPattern);
    }
    else
    {
        rows = w.exec_params(
            "SELECT model_id, NULL::bigint AS experiment_id, " +
            std::string{hasParentModelId ? "parent_model_id" : "NULL::bigint"} +
            ", COALESCE(name, ''), created_at::text, COALESCE(comment, '') "
            "FROM model "
            "WHERE COALESCE(name, '') LIKE $1 "
            "ORDER BY model_id;",
            legacyPattern);
    }

    std::vector<ExperimentModelRow> models;
    models.reserve(rows.size());
    for (const auto& row : rows)
    {
        ExperimentModelRow model;
        model.modelId = row[0].as<long long>();
        if (!row[1].is_null())
            model.experimentId = row[1].as<long long>();
        if (!row[2].is_null())
            model.parentModelId = row[2].as<long long>();
        model.name = row[3].as<std::string>();
        model.createdAt = row[4].as<std::string>();
        model.comment = row[5].as<std::string>();
        models.push_back(std::move(model));
    }
    return models;
}

std::optional<ExperimentModelRow> LoadModelRowById(pqxx::work& w, long long modelId)
{
    const bool hasExperimentId = ColumnExists(w, "model", "experiment_id");
    const bool hasParentModelId = ColumnExists(w, "model", "parent_model_id");
    pqxx::result rows = w.exec_params(
        "SELECT model_id, " +
        std::string{hasExperimentId ? "experiment_id" : "NULL::bigint"} +
        ", " +
        std::string{hasParentModelId ? "parent_model_id" : "NULL::bigint"} +
        ", COALESCE(name, ''), created_at::text, COALESCE(comment, '') "
        "FROM model WHERE model_id = $1;",
        modelId);
    if (rows.empty())
        return std::nullopt;

    ExperimentModelRow model;
    model.modelId = rows[0][0].as<long long>();
    if (!rows[0][1].is_null())
        model.experimentId = rows[0][1].as<long long>();
    if (!rows[0][2].is_null())
        model.parentModelId = rows[0][2].as<long long>();
    model.name = rows[0][3].as<std::string>();
    model.createdAt = rows[0][4].as<std::string>();
    model.comment = rows[0][5].as<std::string>();
    return model;
}

std::string ExperimentIdText(const std::optional<long long>& experimentId)
{
    return experimentId.has_value() ? std::to_string(*experimentId) : "legacy_null";
}

std::string ParentModelIdText(const std::optional<long long>& parentModelId)
{
    return parentModelId.has_value() ? std::to_string(*parentModelId) : "none";
}

std::string ModelLineageRole(const ExperimentModelRow& model,
                             const std::optional<long long>& lastModelId)
{
    if (lastModelId.has_value() && model.modelId == *lastModelId)
        return "final";
    if (model.comment.find("periodic training checkpoint") != std::string::npos ||
        std::regex_search(model.name, std::regex{R"(_epoch[0-9]+)"}))
    {
        return "checkpoint";
    }
    return "created";
}

void PrintExperimentLineageModel(const std::string& marker,
                                 const ExperimentModelRow& model,
                                 const std::string& role,
                                 bool includeParentModelId = false)
{
    std::cout << marker
              << ",model_id=" << model.modelId
              << ",experiment_id=" << ExperimentIdText(model.experimentId);
    if (includeParentModelId)
        std::cout << ",parent_model_id=" << ParentModelIdText(model.parentModelId);
    std::cout << ",role=" << role
              << ",name=" << model.name
              << ",created_at=" << model.createdAt
              << ",comment=" << model.comment
              << std::endl;
}

std::vector<ExperimentModelRow> LoadAncestorModels(pqxx::work& w,
                                                   long long startModelId,
                                                   long long experimentId)
{
    std::vector<ExperimentModelRow> ancestors;
    std::set<long long> seen;
    long long currentModelId = startModelId;
    constexpr int kMaxLineageDepth = 100;

    for (int depth = 0; depth < kMaxLineageDepth; ++depth)
    {
        const std::optional<ExperimentModelRow> current = LoadModelRowById(w, currentModelId);
        if (!current.has_value() || !current->parentModelId.has_value())
            return ancestors;

        const long long parentModelId = *current->parentModelId;
        if (!seen.insert(parentModelId).second)
        {
            std::cout << "EXPERIMENT_LINEAGE_ANCESTRY_CYCLE"
                      << ",experiment_id=" << experimentId
                      << ",model_id=" << parentModelId
                      << std::endl;
            return ancestors;
        }

        std::optional<ExperimentModelRow> parent = LoadModelRowById(w, parentModelId);
        if (!parent.has_value())
        {
            std::cout << "EXPERIMENT_LINEAGE_ANCESTRY_MISSING_PARENT"
                      << ",experiment_id=" << experimentId
                      << ",model_id=" << currentModelId
                      << ",parent_model_id=" << parentModelId
                      << std::endl;
            return ancestors;
        }

        ancestors.push_back(*parent);
        currentModelId = parentModelId;
    }

    std::cout << "EXPERIMENT_LINEAGE_ANCESTRY_DEPTH_LIMIT"
              << ",experiment_id=" << experimentId
              << ",max_depth=" << kMaxLineageDepth
              << std::endl;
    return ancestors;
}

int ListExperimentModels(long long experimentId)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    std::vector<ExperimentModelRow> models = LoadExperimentModelRows(w, experimentId);
    w.commit();

    std::cout << "EXPERIMENT_MODELS"
              << ",experiment_id=" << experimentId
              << ",count=" << models.size()
              << std::endl;
    for (const auto& model : models)
    {
        std::cout << "EXPERIMENT_MODEL"
                  << ",model_id=" << model.modelId
                  << ",experiment_id=" << ExperimentIdText(model.experimentId)
                  << ",name=" << model.name
                  << ",created_at=" << model.createdAt
                  << ",comment=" << model.comment
                  << std::endl;
    }
    return 0;
}

int ListExperimentLineage(long long experimentId, bool includeParentModels)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    pqxx::result experiments = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, target_epochs, "
        "resume_model_id, last_model_id, status, phase "
        "FROM experiment WHERE experiment_id = $1;",
        experimentId);
    if (experiments.empty())
    {
        w.commit();
        std::cerr << "ERROR: experiment not found." << std::endl;
        return 1;
    }

    const auto& experiment = experiments[0];
    std::optional<long long> resumeModelId;
    std::optional<long long> lastModelId;
    if (!experiment[4].is_null())
        resumeModelId = experiment[4].as<long long>();
    if (!experiment[5].is_null())
        lastModelId = experiment[5].as<long long>();

    std::optional<ExperimentModelRow> resumeSource;
    if (resumeModelId.has_value())
        resumeSource = LoadModelRowById(w, *resumeModelId);
    std::vector<ExperimentModelRow> ancestors;
    if (includeParentModels && resumeModelId.has_value())
        ancestors = LoadAncestorModels(w, *resumeModelId, experimentId);
    std::vector<ExperimentModelRow> models = LoadExperimentModelRows(w, experimentId);
    w.commit();

    std::cout << "EXPERIMENT_LINEAGE"
              << ",experiment_id=" << experiment[0].as<long long>()
              << ",symbol=" << experiment[1].as<std::string>()
              << ",prediction_horizon=" << experiment[2].as<int>()
              << ",target_epochs=" << experiment[3].as<int>()
              << ",resume_model_id=" << (resumeModelId.has_value() ? std::to_string(*resumeModelId) : "none")
              << ",last_model_id=" << (lastModelId.has_value() ? std::to_string(*lastModelId) : "none")
              << ",status=" << experiment[6].as<std::string>()
              << ",phase=" << experiment[7].as<std::string>();
    if (includeParentModels)
        std::cout << ",include_parent_models=1"
                  << ",ancestor_count=" << ancestors.size();
    std::cout << ",created_model_count=" << models.size()
              << std::endl;

    for (auto it = ancestors.rbegin(); it != ancestors.rend(); ++it)
    {
        PrintExperimentLineageModel("EXPERIMENT_LINEAGE_ANCESTOR",
                                    *it,
                                    "ancestor",
                                    true);
    }

    if (resumeModelId.has_value())
    {
        if (resumeSource.has_value())
        {
            PrintExperimentLineageModel("EXPERIMENT_LINEAGE_RESUME_SOURCE",
                                        *resumeSource,
                                        "resume_source");
        }
        else
        {
            std::cout << "EXPERIMENT_LINEAGE_RESUME_SOURCE"
                      << ",model_id=" << *resumeModelId
                      << ",experiment_id=unknown"
                      << ",parent_model_id=unknown"
                      << ",role=resume_source"
                      << ",name=unknown"
                      << ",created_at=unknown"
                      << ",comment=model_not_found"
                      << std::endl;
        }
    }

    for (const auto& model : models)
    {
        PrintExperimentLineageModel("EXPERIMENT_LINEAGE_MODEL",
                                    model,
                                    ModelLineageRole(model, lastModelId));
    }
    return 0;
}

SchedulerStatusJob RowToSchedulerStatusJob(const pqxx::row& row)
{
    SchedulerStatusJob job;
    job.experimentId = row[0].as<long long>();
    job.symbol = row[1].as<std::string>();
    job.predictionHorizon = row[2].as<int>();
    job.phase = row[3].as<std::string>();
    job.status = row[4].as<std::string>();
    job.targetEpochs = row[5].as<int>();
    job.checkpointInterval = row[6].as<int>();
    job.modelId = OptionalLongLongCell(row, 7);
    if (!row[8].is_null())
        job.completedEpochs = row[8].as<int>();
    if (!row[9].is_null())
        job.elapsedSeconds = row[9].as<double>();
    job.startedAt = row[10].is_null() ? "" : row[10].as<std::string>();
    job.updatedAt = row[11].is_null() ? "" : row[11].as<std::string>();
    job.completedAt = row[12].is_null() ? "" : row[12].as<std::string>();
    job.errorMessage = row[13].is_null() ? "" : row[13].as<std::string>();
    job.trainLogPath = OptionalStringCell(row, 14);
    job.inferLogPath = OptionalStringCell(row, 15);
    job.analysisLogPath = OptionalStringCell(row, 16);
    if (!row[17].is_null())
    {
        job.currentEpoch = row[17].as<int>();
        job.currentEpochFromTable = true;
    }
    if (!row[18].is_null())
        job.pid = row[18].as<int>();
    if (!row[19].is_null())
    {
        const auto currentOperation =
            EA::ExperimentLifecycle::NormalizePersistedCurrentOperation(
                row[19].as<std::string>());
        job.currentOperation = currentOperation;
    }
    if (!row[20].is_null())
        job.stopAfterCheckpointEpoch = row[20].as<int>();
    if (!row[21].is_null())
        job.stoppedAtCheckpointEpoch = row[21].as<int>();
    if (!row[22].is_null())
        job.stoppedAtCheckpointModelId = row[22].as<long long>();
    if (!row[23].is_null())
        job.opportunisticCheckpointInfer = row[23].as<bool>();
    if (!row[24].is_null())
        job.checkpointInferMinEpoch = row[24].as<int>();
    if (!row[25].is_null())
        job.checkpointInferInterval = row[25].as<int>();
    if (!row[26].is_null())
        job.checkpointPolicyEnabled = row[26].as<bool>();
    job.checkpointPolicyMinLeaderScore = OptionalDoubleCell(row, 27);
    job.checkpointPolicyMinInferAccuracy = OptionalDoubleCell(row, 28);
    if (!row[29].is_null())
        job.checkpointPolicyTopN = row[29].as<int>();
    if (!row[30].is_null())
        job.checkpointPolicyScope = row[30].as<std::string>();
    if (!row[31].is_null())
        job.checkpointPolicyStopMode = row[31].as<std::string>();
    if (!row[32].is_null())
        job.checkpointPolicyGraceEvals = row[32].as<int>();
    if (!row[33].is_null())
        job.checkpointPolicyLastDecision = row[33].as<std::string>();
    if (!row[34].is_null())
        job.checkpointPolicyLastEvalId = row[34].as<long long>();
    if (!row[35].is_null())
        job.checkpointPolicyLastReason = row[35].as<std::string>();
    job.checkpointEvalPending = row[36].as<int>();
    job.checkpointEvalRunning = row[37].as<int>();
    job.checkpointEvalCompleted = row[38].as<int>();
    job.checkpointEvalFailed = row[39].as<int>();
    job.operatorForcedFinalInferenceRerunRequested = row[40].as<bool>();
    return job;
}

std::vector<SchedulerStatusJob> LoadSchedulerStatusJobs(pqxx::work& w,
                                                               const std::string& status,
                                                               const std::optional<std::string>& phase,
                                                               int limit,
                                                               bool newestFirst)
{
    const bool hasCheckpointColumns = ColumnExists(w, "experiment", "stop_after_checkpoint_epoch");
    const bool hasCheckpointInferEnabled = ColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = ColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointEvalTable = TableExists(w, "experiment_checkpoint_eval");
    const bool hasCheckpointPolicy = ColumnExists(w, "experiment", "checkpoint_policy_enabled");
    std::ostringstream sql;
    sql << "WITH latest_analysis AS ("
        << "  SELECT experiment_id, model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM experiment_analysis_result "
        << "  WHERE completed_epochs IS NOT NULL "
        << "  AND COALESCE(analysis_scope, 'final') = 'final' "
        << "  GROUP BY experiment_id, model_id"
        << "), latest_infer AS ("
        << "  SELECT model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM inference_eval_result "
        << "  WHERE completed_epochs IS NOT NULL AND status = 'completed' "
        << "  AND inference_scope = 'final' AND checkpoint_eval_id IS NULL "
        << "  GROUP BY model_id"
        << "), train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epochs "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ")";
    if (hasCheckpointEvalTable)
    {
        sql << ", checkpoint_eval_counts AS ("
            << "  SELECT COALESCE(parent_experiment_id, experiment_id) AS parent_experiment_id, "
            << "         count(*) FILTER (WHERE status = 'pending') AS pending_count, "
            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
            << "         count(*) FILTER (WHERE status = 'completed') AS completed_count, "
            << "         count(*) FILTER (WHERE status = 'failed') AS failed_count "
            << "  FROM experiment_checkpoint_eval "
            << "  GROUP BY COALESCE(parent_experiment_id, experiment_id)"
            << ")";
    }
    sql
        << " SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.phase, e.status, "
        << "e.target_epochs, e.checkpoint_interval, COALESCE(e.last_model_id, e.resume_model_id) AS model_id, "
        << "COALESCE(la.completed_epochs, li.completed_epochs, tm.completed_epochs) AS completed_epochs, "
        << "CASE WHEN e.started_at IS NULL THEN NULL "
        << "     WHEN e.completed_at IS NULL THEN EXTRACT(EPOCH FROM (now() - e.started_at)) "
        << "     ELSE EXTRACT(EPOCH FROM (e.completed_at - e.started_at)) END AS elapsed_seconds, "
        << "e.started_at::text, e.updated_at::text, e.completed_at::text, e.error_message, "
        << "e.train_log_path, e.infer_log_path, e.analysis_log_path, "
        << "e.current_epoch, e.worker_pid, e.current_operation, ";
    if (hasCheckpointColumns)
    {
        sql << "e.stop_after_checkpoint_epoch, e.stopped_at_checkpoint_epoch, "
            << "e.stopped_at_checkpoint_model_id, ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(e.checkpoint_infer_enabled OR e.opportunistic_checkpoint_infer), ";
        else if (hasCheckpointInferEnabled)
            sql << "e.checkpoint_infer_enabled, ";
        else if (hasOpportunisticCheckpointInfer)
            sql << "e.opportunistic_checkpoint_infer, ";
        else
            sql << "NULL::boolean, ";
        sql << "e.checkpoint_infer_min_epoch, e.checkpoint_infer_interval, ";
    }
    else
    {
        sql << "NULL::integer, NULL::integer, NULL::bigint, NULL::boolean, NULL::integer, NULL::integer, ";
    }
    if (hasCheckpointPolicy)
    {
        sql << "e.checkpoint_policy_enabled, e.checkpoint_policy_min_leader_score, "
            << "e.checkpoint_policy_min_infer_accuracy, e.checkpoint_policy_top_n, "
            << "e.checkpoint_policy_scope, e.checkpoint_policy_stop_mode, "
            << "e.checkpoint_policy_grace_evals, e.checkpoint_policy_last_decision, "
            << "e.checkpoint_policy_last_checkpoint_eval_id, e.checkpoint_policy_last_reason, ";
    }
    else
    {
        sql << "NULL::boolean, NULL::double precision, NULL::double precision, NULL::integer, "
            << "NULL::text, NULL::text, NULL::integer, NULL::text, NULL::bigint, NULL::text, ";
    }
    if (hasCheckpointEvalTable)
    {
        sql << "COALESCE(cec.pending_count, 0)::int, "
            << "COALESCE(cec.running_count, 0)::int, "
            << "COALESCE(cec.completed_count, 0)::int, "
            << "COALESCE(cec.failed_count, 0)::int, ";
    }
    else
    {
        sql << "0::int, 0::int, 0::int, 0::int, ";
    }
    sql << "e.operator_forced_final_inference_rerun_requested ";
    sql
        << "FROM experiment e "
        << "LEFT JOIN latest_analysis la ON la.experiment_id = e.experiment_id "
        << "LEFT JOIN latest_infer li ON li.model_id = e.last_model_id "
        << "LEFT JOIN train_meta tm ON tm.model_id = COALESCE(e.last_model_id, e.resume_model_id) ";
    if (hasCheckpointEvalTable)
        sql << "LEFT JOIN checkpoint_eval_counts cec ON cec.parent_experiment_id = e.experiment_id ";
    sql
        << "WHERE e.status = " << w.quote(status) << " ";
    if (phase.has_value())
        sql << "AND e.phase = " << w.quote(*phase) << " ";
    sql << "ORDER BY ";
    if (status == "pending")
        sql << "e.updated_at ASC, e.experiment_id ASC ";
    else if (newestFirst)
        sql << "COALESCE(e.completed_at, e.updated_at) DESC, e.experiment_id DESC ";
    else
        sql << "COALESCE(e.started_at, e.updated_at) ASC, e.experiment_id ASC ";
    sql << "LIMIT " << limit << ";";

    pqxx::result rows = w.exec(sql.str());
    std::vector<SchedulerStatusJob> jobs;
    jobs.reserve(rows.size());
    for (const auto& row : rows)
        jobs.push_back(RowToSchedulerStatusJob(row));
    return jobs;
}

std::optional<SchedulerStatusJob> LoadSchedulerStatusJobById(pqxx::work& w, long long experimentId)
{
    const bool hasCheckpointColumns = ColumnExists(w, "experiment", "stop_after_checkpoint_epoch");
    const bool hasCheckpointInferEnabled = ColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = ColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointEvalTable = TableExists(w, "experiment_checkpoint_eval");
    const bool hasCheckpointPolicy = ColumnExists(w, "experiment", "checkpoint_policy_enabled");
    std::ostringstream sql;
    sql << "WITH latest_analysis AS ("
        << "  SELECT experiment_id, model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM experiment_analysis_result "
        << "  WHERE completed_epochs IS NOT NULL "
        << "  AND COALESCE(analysis_scope, 'final') = 'final' "
        << "  GROUP BY experiment_id, model_id"
        << "), latest_infer AS ("
        << "  SELECT model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM inference_eval_result "
        << "  WHERE completed_epochs IS NOT NULL AND status = 'completed' "
        << "  AND inference_scope = 'final' AND checkpoint_eval_id IS NULL "
        << "  GROUP BY model_id"
        << "), train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epochs "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ")";
    if (hasCheckpointEvalTable)
    {
        sql << ", checkpoint_eval_counts AS ("
            << "  SELECT COALESCE(parent_experiment_id, experiment_id) AS parent_experiment_id, "
            << "         count(*) FILTER (WHERE status = 'pending') AS pending_count, "
            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
            << "         count(*) FILTER (WHERE status = 'completed') AS completed_count, "
            << "         count(*) FILTER (WHERE status = 'failed') AS failed_count "
            << "  FROM experiment_checkpoint_eval "
            << "  GROUP BY COALESCE(parent_experiment_id, experiment_id)"
            << ")";
    }
    sql
        << " SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.phase, e.status, "
        << "e.target_epochs, e.checkpoint_interval, COALESCE(e.last_model_id, e.resume_model_id) AS model_id, "
        << "COALESCE(la.completed_epochs, li.completed_epochs, tm.completed_epochs) AS completed_epochs, "
        << "CASE WHEN e.started_at IS NULL THEN NULL "
        << "     WHEN e.completed_at IS NULL THEN EXTRACT(EPOCH FROM (now() - e.started_at)) "
        << "     ELSE EXTRACT(EPOCH FROM (e.completed_at - e.started_at)) END AS elapsed_seconds, "
        << "e.started_at::text, e.updated_at::text, e.completed_at::text, e.error_message, "
        << "e.train_log_path, e.infer_log_path, e.analysis_log_path, "
        << "e.current_epoch, e.worker_pid, e.current_operation, ";
    if (hasCheckpointColumns)
    {
        sql << "e.stop_after_checkpoint_epoch, e.stopped_at_checkpoint_epoch, "
            << "e.stopped_at_checkpoint_model_id, ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(e.checkpoint_infer_enabled OR e.opportunistic_checkpoint_infer), ";
        else if (hasCheckpointInferEnabled)
            sql << "e.checkpoint_infer_enabled, ";
        else if (hasOpportunisticCheckpointInfer)
            sql << "e.opportunistic_checkpoint_infer, ";
        else
            sql << "NULL::boolean, ";
        sql << "e.checkpoint_infer_min_epoch, e.checkpoint_infer_interval, ";
    }
    else
    {
        sql << "NULL::integer, NULL::integer, NULL::bigint, NULL::boolean, NULL::integer, NULL::integer, ";
    }
    if (hasCheckpointPolicy)
    {
        sql << "e.checkpoint_policy_enabled, e.checkpoint_policy_min_leader_score, "
            << "e.checkpoint_policy_min_infer_accuracy, e.checkpoint_policy_top_n, "
            << "e.checkpoint_policy_scope, e.checkpoint_policy_stop_mode, "
            << "e.checkpoint_policy_grace_evals, e.checkpoint_policy_last_decision, "
            << "e.checkpoint_policy_last_checkpoint_eval_id, e.checkpoint_policy_last_reason, ";
    }
    else
    {
        sql << "NULL::boolean, NULL::double precision, NULL::double precision, NULL::integer, "
            << "NULL::text, NULL::text, NULL::integer, NULL::text, NULL::bigint, NULL::text, ";
    }
    if (hasCheckpointEvalTable)
    {
        sql << "COALESCE(cec.pending_count, 0)::int, "
            << "COALESCE(cec.running_count, 0)::int, "
            << "COALESCE(cec.completed_count, 0)::int, "
            << "COALESCE(cec.failed_count, 0)::int, ";
    }
    else
    {
        sql << "0::int, 0::int, 0::int, 0::int, ";
    }
    sql << "e.operator_forced_final_inference_rerun_requested ";
    sql
        << "FROM experiment e "
        << "LEFT JOIN latest_analysis la ON la.experiment_id = e.experiment_id "
        << "LEFT JOIN latest_infer li ON li.model_id = e.last_model_id "
        << "LEFT JOIN train_meta tm ON tm.model_id = COALESCE(e.last_model_id, e.resume_model_id) ";
    if (hasCheckpointEvalTable)
        sql << "LEFT JOIN checkpoint_eval_counts cec ON cec.parent_experiment_id = e.experiment_id ";
    sql
        << "WHERE e.experiment_id = " << experimentId << " "
        << "LIMIT 1;";

    pqxx::result rows = w.exec(sql.str());
    if (rows.empty())
        return std::nullopt;
    return RowToSchedulerStatusJob(rows[0]);
}

void EnrichSchedulerStatusJobFromLogs(SchedulerStatusJob& job,
                                             const SchedulerStatusProcessSnapshot& processes)
{
    const auto assignPid = [&](const std::map<long long, int>& pids) {
        const auto it = pids.find(job.experimentId);
        if (it != pids.end())
            job.pid = it->second;
    };

    if (job.phase == "train")
        assignPid(processes.trainPidByExperiment);
    else if (job.phase == "infer")
    {
        assignPid(processes.inferPidByExperiment);
        if (!job.pid.has_value() && job.modelId.has_value())
        {
            for (const auto& process : processes.workerProcesses)
            {
                if (process.kind == "infer" &&
                    process.command.find(
                        "--scheduler-checkpoint-eval-id") ==
                        std::string::npos &&
                    CommandContainsOptionValue(process.command, "--model", *job.modelId))
                {
                    job.pid = process.pid;
                    break;
                }
            }
        }
    }
    else if (job.phase == "analyze")
        assignPid(processes.analysisPidByExperiment);

    if (job.pid.has_value())
    {
        const auto resourceIt = processes.resourcesByPid.find(*job.pid);
        if (resourceIt != processes.resourcesByPid.end())
        {
            job.cpuPercent = resourceIt->second.cpuPercent;
            job.memPercent = resourceIt->second.memPercent;
            job.rssMb = resourceIt->second.rssMb;
        }
    }

    std::optional<std::string> logPath;
    if (job.phase == "train")
        logPath = job.trainLogPath;
    else if (job.phase == "infer")
        logPath = job.inferLogPath;
    else if (job.phase == "analyze")
        logPath = job.analysisLogPath;

    const std::string tail = ReadFileTailIfExists(logPath);
    if (!tail.empty())
    {
        job.recentProgress = ExtractLastProgressLine(tail);

        const auto resumeCompleted = ExtractLastIntFromText(
            tail,
            std::regex{R"(RESUME_COMPLETED_EPOCH=([0-9]+))"});
        const auto resumeTarget = ExtractLastIntFromText(
            tail,
            std::regex{R"(RESUME_TARGET_EPOCH=([0-9]+))"});
        const auto metaEpoch = ExtractLastIntFromText(
            tail,
            std::regex{R"(epochs_trained=([0-9]+))"});
        const auto epochKv = ExtractLastIntFromText(
            tail,
            std::regex{R"((?:^|[,[:space:]])(?:epoch|current_epoch|completed_epoch|completed_epochs)=([0-9]+))"});
        const auto checkpointEpoch = ExtractLastIntFromText(
            tail,
            std::regex{R"(CHECKPOINT_SAVE_DONE[^[:cntrl:]]*epoch=([0-9]+))"});
        const auto checkpointModel = ExtractLastLongLongFromText(
            tail,
            std::regex{R"(CHECKPOINT_SAVE_DONE[^[:cntrl:]]*model_id=([0-9]+))"});
        const auto finalModel = ExtractLastLongLongFromText(
            tail,
            std::regex{R"((?:Saved model with model_id=|RESUME_SAVED_NEW_MODEL_ID=)([0-9]+))"});
        const auto loss = ExtractLastDoubleFromText(
            tail,
            std::regex{R"((?:^|[,[:space:]])(?:loss|loss_last|weighted_loss)=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?))"});
        const auto validationAccuracy = ExtractLastDoubleFromText(
            tail,
            std::regex{R"((?:validation_accuracy|val_accuracy|acc)=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?))"});

        if (checkpointEpoch.has_value())
            job.lastCheckpointEpoch = checkpointEpoch;
        if (checkpointModel.has_value())
            job.lastCheckpointModelId = checkpointModel;
        else if (finalModel.has_value())
            job.lastCheckpointModelId = finalModel;
        if (loss.has_value())
            job.loss = loss;
        if (validationAccuracy.has_value())
            job.validationAccuracy = validationAccuracy;

        if (!job.currentEpochFromTable)
        {
            int currentEpoch = 0;
            if (job.completedEpochs.has_value())
                currentEpoch = std::max(currentEpoch, *job.completedEpochs);
            if (resumeCompleted.has_value())
                currentEpoch = std::max(currentEpoch, *resumeCompleted);
            if (metaEpoch.has_value())
                currentEpoch = std::max(currentEpoch, *metaEpoch);
            if (epochKv.has_value())
                currentEpoch = std::max(currentEpoch, *epochKv);
            if (checkpointEpoch.has_value())
                currentEpoch = std::max(currentEpoch, *checkpointEpoch);
            if (currentEpoch > 0)
                job.currentEpoch = currentEpoch;
        }

        if (resumeTarget.has_value() && job.targetEpochs <= 0)
            job.targetEpochs = *resumeTarget;
    }

    if (!job.currentEpochFromTable && !job.currentEpoch.has_value() && job.completedEpochs.has_value())
        job.currentEpoch = job.completedEpochs;

    if (!job.lastCheckpointEpoch.has_value() && job.completedEpochs.has_value())
        job.lastCheckpointEpoch = job.completedEpochs;
    if (!job.lastCheckpointModelId.has_value() && job.modelId.has_value())
        job.lastCheckpointModelId = job.modelId;

    if (job.checkpointInterval > 0 && job.currentEpoch.has_value() && job.targetEpochs > 0)
    {
        const int next = std::min(job.targetEpochs,
                                  ((*job.currentEpoch / job.checkpointInterval) + 1) * job.checkpointInterval);
        if (next > *job.currentEpoch)
            job.nextCheckpointEpoch = next;
    }

    job.etaSeconds = EstimateEtaSeconds(job);
}

void EnrichSchedulerStatusJobs(std::vector<SchedulerStatusJob>& jobs,
                                      const SchedulerStatusProcessSnapshot& processes)
{
    for (auto& job : jobs)
        EnrichSchedulerStatusJobFromLogs(job, processes);
}

void EnrichCheckpointStatusJobs(
    std::vector<SchedulerCheckpointStatusJob>& jobs,
    const SchedulerStatusProcessSnapshot& processes)
{
    for (auto& job : jobs)
    {
        if (!job.pid)
            continue;
        const auto resource = processes.resourcesByPid.find(*job.pid);
        if (resource == processes.resourcesByPid.end())
            continue;
        job.cpuPercent = resource->second.cpuPercent;
        job.memPercent = resource->second.memPercent;
        job.rssMb = resource->second.rssMb;
    }
}

std::string TruncateCommandForStatus(const std::string& command)
{
    constexpr size_t kMaxCommandLength = 220;
    if (command.size() <= kMaxCommandLength)
        return command;
    return command.substr(0, kMaxCommandLength - 3) + "...";
}

SchedulerWorkerAccounting ComputeSchedulerWorkerAccounting(
    const SchedulerStatusProcessSnapshot& processes,
    const std::vector<EA::GlobalExperimentControl::ManagedWorker>&
        authoritativeWorkers,
    EA::GlobalExperimentControl::ProcessOperations& processOperations)
{
    std::vector<EA::GlobalExperimentControl::SchedulerWorkerCandidate>
        candidates;
    candidates.reserve(processes.workerProcesses.size());
    for (const auto& process : processes.workerProcesses)
    {
        candidates.push_back(
            EA::GlobalExperimentControl::SchedulerWorkerCandidate{
                process.pid,
                process.kind,
                process.command,
                process.resource.cpuPercent,
                process.resource.memPercent,
                process.resource.rssMb,
                process.stopped});
    }
    const auto classifications =
        EA::GlobalExperimentControl::ClassifySchedulerWorkers(
            candidates, authoritativeWorkers, processOperations);
    const auto summary =
        EA::GlobalExperimentControl::SummarizeSchedulerWorkers(
            classifications);

    SchedulerWorkerAccounting accounting;
    const auto copyAggregate = [](
        const EA::GlobalExperimentControl::SchedulerWorkerAggregate& from,
        SchedulerResourceAggregate& to) {
        to.workers = from.workers;
        to.cpuPercent = from.cpuPercent;
        to.memPercent = from.memPercent;
        to.rssMb = from.rssMb;
    };
    accounting.managedTrain = summary.managedTrain.workers;
    accounting.managedInfer = summary.managedInfer.workers;
    accounting.managedAnalyze = summary.managedAnalyze.workers;
    accounting.managedRunningTrain = summary.managedRunningTrain.workers;
    accounting.managedRunningInfer = summary.managedRunningInfer.workers;
    accounting.managedRunningAnalyze = summary.managedRunningAnalyze.workers;
    accounting.managedPausedTrain = summary.managedPausedTrain.workers;
    accounting.managedPausedInfer = summary.managedPausedInfer.workers;
    accounting.managedPausedAnalyze = summary.managedPausedAnalyze.workers;
    accounting.unmanagedTrain = summary.unmanagedTrain.workers;
    accounting.unmanagedInfer = summary.unmanagedInfer.workers;
    accounting.unmanagedAnalyze = summary.unmanagedAnalyze.workers;
    accounting.identityMismatchTrain = summary.identityMismatchTrain.workers;
    accounting.identityMismatchInfer = summary.identityMismatchInfer.workers;
    accounting.identityMismatchAnalyze = summary.identityMismatchAnalyze.workers;
    accounting.expectedMissingTrain = summary.expectedMissingTrain.workers;
    accounting.expectedMissingInfer = summary.expectedMissingInfer.workers;
    accounting.expectedMissingAnalyze = summary.expectedMissingAnalyze.workers;
    copyAggregate(
        summary.managedTrain, accounting.managedTrainResources);
    copyAggregate(
        summary.managedInfer, accounting.managedInferResources);
    copyAggregate(
        summary.managedAnalyze, accounting.managedAnalysisResources);
    copyAggregate(
        summary.unmanagedTrain, accounting.unmanagedTrainResources);
    copyAggregate(
        summary.unmanagedInfer, accounting.unmanagedInferResources);
    copyAggregate(
        summary.unmanagedAnalyze, accounting.unmanagedAnalysisResources);

    accounting.workerClassifications = classifications;
    for (size_t i = 0; i < processes.workerProcesses.size(); ++i)
    {
        const auto& process = processes.workerProcesses[i];
        const auto& classification = classifications[i];
        if (classification.managed || classification.authoritative)
            continue;
        accounting.unmanagedWorkers.push_back(SchedulerUnmanagedWorker{
            process.pid,
            process.kind,
            classification.reason,
            TruncateCommandForStatus(process.command)
        });
    }
    return accounting;
}

void PrintCheckpointStatusJobs(
    const std::vector<SchedulerCheckpointStatusJob>& jobs)
{
    std::cout << "\nActive Checkpoint Inference Jobs\n";
    if (jobs.empty())
    {
        std::cout << "  none\n";
        return;
    }
    for (const auto& job : jobs)
    {
        std::cout << "  checkpoint_eval_id=" << job.checkpointEvalId
                  << " experiment_id=" << job.experimentId
                  << " epoch=" << job.checkpointEpoch
                  << " model_id=" << job.checkpointModelId
                  << " symbol=" << job.symbol
                  << " horizon=" << job.predictionHorizon
                  << " pid=" << OptionalIntText(job.pid)
                  << " control=" << job.workerControlState
                  << " cpu=" << OptionalDoubleText(job.cpuPercent, 1)
                  << "% rss_mb=" << OptionalDoubleText(job.rssMb, 0)
                  << " log=" << job.inferLogPath.value_or("none")
                  << "\n";
    }
}

void PrintCheckpointStatusJobMachine(
    const SchedulerCheckpointStatusJob& job)
{
    std::cout << "SCHEDULER_STATUS_CHECKPOINT_JOB"
              << ",checkpoint_eval_id=" << job.checkpointEvalId
              << ",experiment_id=" << job.experimentId
              << ",checkpoint_epoch=" << job.checkpointEpoch
              << ",checkpoint_model_id=" << job.checkpointModelId
              << ",status=" << job.status
              << ",phase=" << job.phase
              << ",symbol=" << job.symbol
              << ",prediction_horizon=" << job.predictionHorizon
              << ",pid=" << OptionalIntText(job.pid)
              << ",worker_control_state=" << job.workerControlState
              << ",cpu_percent="
              << OptionalDoubleText(job.cpuPercent, 1)
              << ",rss_mb=" << OptionalDoubleText(job.rssMb, 0)
              << ",mem_percent="
              << OptionalDoubleText(job.memPercent, 1)
              << ",infer_log_path="
              << job.inferLogPath.value_or("NULL")
              << std::endl;
}

void PrintSchedulerStatusJobMachine(const SchedulerStatusJob& job)
{
    std::cout << "SCHEDULER_STATUS_JOB"
              << ",experiment_id=" << job.experimentId
              << ",phase=" << job.phase
              << ",status=" << job.status
              << ",current_operation=" << CurrentOperationForStatusJob(job)
              << ",symbol=" << job.symbol
              << ",prediction_horizon=" << job.predictionHorizon
              << ",pid=" << OptionalIntText(job.pid)
              << ",cpu_percent=" << OptionalDoubleText(job.cpuPercent, 1)
              << ",rss_mb=" << OptionalDoubleText(job.rssMb, 0)
              << ",mem_percent=" << OptionalDoubleText(job.memPercent, 1)
              << ",current_epoch=" << OptionalIntText(job.currentEpoch)
              << ",target_epochs=" << job.targetEpochs
              << ",percent_complete=" << (job.currentEpoch.has_value() && job.targetEpochs > 0
                                               ? OptionalDoubleText(100.0 * static_cast<double>(*job.currentEpoch) /
                                                                        static_cast<double>(job.targetEpochs),
                                                                    1)
                                               : "unknown")
              << ",completed_epochs=" << OptionalIntText(job.completedEpochs)
              << ",model_id=" << OptionalLongLongText(job.modelId)
              << ",last_checkpoint_epoch=" << OptionalIntText(job.lastCheckpointEpoch)
              << ",last_checkpoint_model_id=" << OptionalLongLongText(job.lastCheckpointModelId)
              << ",next_checkpoint_epoch=" << OptionalIntText(job.nextCheckpointEpoch)
              << ",stop_after_checkpoint_epoch=" << OptionalIntText(job.stopAfterCheckpointEpoch)
              << ",stopped_at_checkpoint_epoch=" << OptionalIntText(job.stoppedAtCheckpointEpoch)
              << ",stopped_at_checkpoint_model_id=" << OptionalLongLongText(job.stoppedAtCheckpointModelId)
              << ",opportunistic_checkpoint_infer="
              << (job.opportunisticCheckpointInfer.has_value() ? (*job.opportunisticCheckpointInfer ? "1" : "0") : "unknown")
              << ",checkpoint_infer_min_epoch=" << OptionalIntText(job.checkpointInferMinEpoch)
              << ",checkpoint_infer_interval=" << OptionalIntText(job.checkpointInferInterval)
              << ",checkpoint_policy="
              << (job.checkpointPolicyEnabled.has_value() ? (*job.checkpointPolicyEnabled ? "1" : "0") : "unknown")
              << ",checkpoint_policy_min_leader_score=" << OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6)
              << ",checkpoint_policy_min_infer_accuracy=" << OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6)
              << ",checkpoint_policy_top_n=" << OptionalIntText(job.checkpointPolicyTopN)
              << ",checkpoint_policy_scope=" << (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown")
              << ",checkpoint_policy_stop_mode=" << (job.checkpointPolicyStopMode.has_value() ? *job.checkpointPolicyStopMode : "unknown")
              << ",checkpoint_policy_grace_evals=" << OptionalIntText(job.checkpointPolicyGraceEvals)
              << ",checkpoint_policy_last_decision=" << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown")
              << ",checkpoint_policy_last_eval_id=" << OptionalLongLongText(job.checkpointPolicyLastEvalId)
              << ",checkpoint_policy_last_reason=" << (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown")
              << ",checkpoint_eval_pending=" << job.checkpointEvalPending
              << ",checkpoint_eval_running=" << job.checkpointEvalRunning
              << ",checkpoint_eval_completed=" << job.checkpointEvalCompleted
              << ",checkpoint_eval_failed=" << job.checkpointEvalFailed
              << ",operator_forced_final_inference_rerun_requested="
              << (job.operatorForcedFinalInferenceRerunRequested ? "1" : "0")
              << ",eta_seconds=" << OptionalDoubleText(job.etaSeconds, 0)
              << ",loss=" << OptionalDoubleText(job.loss, 6)
              << ",validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4)
              << std::endl;
}

void PrintStatusJobTable(const std::string& title,
                                const std::vector<SchedulerStatusJob>& jobs,
                                bool useColor,
                                bool showEta,
                                bool showError)
{
    (void)showEta;
    std::cout << "\n" << title << "\n";
    if (jobs.empty())
    {
        std::cout << "  none\n";
        return;
    }

    for (const auto& job : jobs)
    {
        const bool active = job.status == "running" &&
                            (job.phase == "train" || job.phase == "infer" || job.phase == "analyze");
        if (!active)
        {
            std::cout << "  experiment_id=" << job.experimentId
                      << " symbol=" << job.symbol
                      << " H=" << job.predictionHorizon
                      << " phase=" << job.phase
                      << " status=" << ColorForStatus(job.status, useColor)
                      << " model_id=" << OptionalLongLongText(job.modelId)
                      << " completed_epochs=" << OptionalIntText(job.completedEpochs)
                      << " target_epochs=" << job.targetEpochs
                      << " percent=" << FormatPercentComplete(job)
                      << " elapsed=" << FormatOptionalDuration(job.elapsedSeconds)
                      << " forced_final_infer_rerun="
                      << (job.operatorForcedFinalInferenceRerunRequested
                              ? "requested"
                              : "none");
            if (job.stopAfterCheckpointEpoch.has_value())
                std::cout << " stop_after_checkpoint=" << *job.stopAfterCheckpointEpoch;
            if (job.stoppedAtCheckpointEpoch.has_value())
                std::cout << " stopped_checkpoint=" << *job.stoppedAtCheckpointEpoch
                          << "/" << OptionalLongLongText(job.stoppedAtCheckpointModelId);
            if (job.opportunisticCheckpointInfer.has_value() && *job.opportunisticCheckpointInfer)
                std::cout << " checkpoint_eval="
                          << job.checkpointEvalPending << "/"
                          << job.checkpointEvalRunning << "/"
                          << job.checkpointEvalCompleted << "/"
                          << job.checkpointEvalFailed;
            if (job.checkpointPolicyEnabled.has_value() && *job.checkpointPolicyEnabled)
                std::cout << " checkpoint_policy=enabled"
                          << " last_decision="
                          << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown");
            if (showError && !job.errorMessage.empty())
                std::cout << " error=" << job.errorMessage;
            std::cout << "\n";
            continue;
        }

        std::cout << "  experiment_id=" << job.experimentId
                  << " symbol=" << job.symbol
                  << " H=" << job.predictionHorizon
                  << " phase=" << job.phase
                  << " status=" << ColorForStatus(job.status, useColor)
                  << " pid=" << OptionalIntText(job.pid)
                  << " forced_final_infer_rerun="
                  << (job.operatorForcedFinalInferenceRerunRequested
                          ? "requested"
                          : "none")
                  << "\n";
        std::cout << "    cpu=" << OptionalPercentText(job.cpuPercent)
                  << " rss=" << OptionalMbText(job.rssMb)
                  << " mem=" << OptionalPercentText(job.memPercent)
                  << "\n";
        std::cout << "    model_id=" << OptionalLongLongText(job.modelId)
                  << " current_epoch=" << OptionalIntText(job.currentEpoch)
                  << " target_epochs=" << job.targetEpochs
                  << " progress=" << FormatProgressBar(job)
                  << "\n";
        std::cout << "    elapsed=" << FormatOptionalDuration(job.elapsedSeconds)
                  << " eta=" << (job.etaSeconds.has_value() ? FormatDurationSeconds(*job.etaSeconds) : "unknown")
                  << " last_checkpoint_epoch=" << OptionalIntText(job.lastCheckpointEpoch)
                  << " last_checkpoint_model_id=" << OptionalLongLongText(job.lastCheckpointModelId)
                  << " next_checkpoint_epoch=" << OptionalIntText(job.nextCheckpointEpoch)
                  << "\n";
        std::cout << "    loss=" << OptionalDoubleText(job.loss, 6)
                  << " validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4);
        if (job.recentProgress.has_value())
            std::cout << " recent=\"" << *job.recentProgress << "\"";
        std::cout << "\n";
        if (job.stopAfterCheckpointEpoch.has_value() ||
            job.stoppedAtCheckpointEpoch.has_value() ||
            (job.opportunisticCheckpointInfer.has_value() && *job.opportunisticCheckpointInfer) ||
            (job.checkpointPolicyEnabled.has_value() && *job.checkpointPolicyEnabled))
        {
            std::cout << "    checkpoint_stop_after=" << OptionalIntText(job.stopAfterCheckpointEpoch)
                      << " stopped_epoch=" << OptionalIntText(job.stoppedAtCheckpointEpoch)
                      << " stopped_model_id=" << OptionalLongLongText(job.stoppedAtCheckpointModelId)
                      << " checkpoint_infer="
                      << (job.opportunisticCheckpointInfer.has_value() ? (*job.opportunisticCheckpointInfer ? "enabled" : "disabled") : "unknown")
                      << " min_epoch=" << OptionalIntText(job.checkpointInferMinEpoch)
                      << " interval=" << OptionalIntText(job.checkpointInferInterval)
                      << " evals=pending:" << job.checkpointEvalPending
                      << ",running:" << job.checkpointEvalRunning
                      << ",completed:" << job.checkpointEvalCompleted
                      << ",failed:" << job.checkpointEvalFailed
                      << "\n";
            std::cout << "    checkpoint_policy="
                      << (job.checkpointPolicyEnabled.has_value() ? (*job.checkpointPolicyEnabled ? "enabled" : "disabled") : "unknown")
                      << " rules=leader>=" << OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6)
                      << ",infer>=" << OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6)
                      << ",top_n=" << OptionalIntText(job.checkpointPolicyTopN)
                      << ",scope=" << (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown")
                      << ",stop_mode=" << (job.checkpointPolicyStopMode.has_value() ? *job.checkpointPolicyStopMode : "unknown")
                      << ",grace_evals=" << OptionalIntText(job.checkpointPolicyGraceEvals)
                      << " last=decision=" << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown")
                      << ",checkpoint_eval_id=" << OptionalLongLongText(job.checkpointPolicyLastEvalId)
                      << ",reason=" << (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown")
                      << "\n";
        }
    }
}

void PrintCompactStatusField(const std::string& label, const std::string& value)
{
    std::cout << std::left << std::setw(24) << (label + ":") << value << "\n";
}

std::string CurrentOperationForStatusJob(const SchedulerStatusJob& job)
{
    const std::optional<std::string> currentOperation =
        job.currentOperation.empty()
            ? std::nullopt
            : std::optional<std::string>{job.currentOperation};
    return EA::ExperimentLifecycle::CurrentOperationForStatus(
        currentOperation,
        job.phase);
}

void PrintCompactStatusJob(const SchedulerStatusJob& job)
{
    std::cout << "\nExperiment " << job.experimentId << "\n"
              << "---------------\n";
    PrintCompactStatusField("Status", job.status);
    PrintCompactStatusField("Phase", job.phase);
    PrintCompactStatusField("Symbol", job.symbol);
    PrintCompactStatusField("Horizon", std::to_string(job.predictionHorizon));
    PrintCompactStatusField("Model ID", OptionalLongLongText(job.modelId));
    PrintCompactStatusField("Epoch", OptionalIntText(job.currentEpoch) + " / " + std::to_string(job.targetEpochs));
    PrintCompactStatusField("Progress", FormatPercentComplete(job));
    PrintCompactStatusField("Runtime", FormatOptionalDuration(job.elapsedSeconds));
    PrintCompactStatusField("PID", OptionalIntText(job.pid));
    PrintCompactStatusField("Operation", CurrentOperationForStatusJob(job));
    PrintCompactStatusField(
        "Forced FINAL Infer Rerun",
        job.operatorForcedFinalInferenceRerunRequested
            ? "requested"
            : "none");
    if (job.lastCheckpointModelId.has_value())
        PrintCompactStatusField("Checkpoint Model", OptionalLongLongText(job.lastCheckpointModelId));
    if (job.lastCheckpointEpoch.has_value())
        PrintCompactStatusField("Checkpoint Epoch", OptionalIntText(job.lastCheckpointEpoch));
    if (job.stopAfterCheckpointEpoch.has_value())
        PrintCompactStatusField("Stop After Checkpoint", OptionalIntText(job.stopAfterCheckpointEpoch));
    if (job.stoppedAtCheckpointEpoch.has_value())
    {
        PrintCompactStatusField("Stopped Checkpoint", OptionalIntText(job.stoppedAtCheckpointEpoch));
        PrintCompactStatusField("Stopped Model ID", OptionalLongLongText(job.stoppedAtCheckpointModelId));
    }
    if (job.opportunisticCheckpointInfer.has_value())
    {
        PrintCompactStatusField("Checkpoint Infer",
                                *job.opportunisticCheckpointInfer ? "enabled" : "disabled");
        PrintCompactStatusField("Checkpoint Infer Min", OptionalIntText(job.checkpointInferMinEpoch));
        PrintCompactStatusField("Checkpoint Infer Interval", OptionalIntText(job.checkpointInferInterval));
        PrintCompactStatusField("Checkpoint Evals",
                                "pending=" + std::to_string(job.checkpointEvalPending) +
                                " running=" + std::to_string(job.checkpointEvalRunning) +
                                " completed=" + std::to_string(job.checkpointEvalCompleted) +
                                " failed=" + std::to_string(job.checkpointEvalFailed));
    }
    if (job.checkpointPolicyEnabled.has_value())
    {
        PrintCompactStatusField("Checkpoint Policy",
                                *job.checkpointPolicyEnabled ? "enabled" : "disabled");
        PrintCompactStatusField("Checkpoint Policy Rules",
                                "leader>=" + OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6) +
                                " infer>=" + OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6) +
                                " top_n=" + OptionalIntText(job.checkpointPolicyTopN) +
                                " scope=" + (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown"));
        PrintCompactStatusField("Checkpoint Policy Last",
                                "decision=" + (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown") +
                                " checkpoint_eval_id=" + OptionalLongLongText(job.checkpointPolicyLastEvalId) +
                                " reason=" + (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown"));
    }
    if (!job.errorMessage.empty())
        PrintCompactStatusField("Error", job.errorMessage);
}

int PrintCompactExperimentStatus(const SchedulerOptions& options)
{
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadOnly(w);
    if (!RequireSchedulerTables(w))
        return 1;

    std::vector<SchedulerStatusJob> jobs;
    if (options.statusExperimentId.has_value())
    {
        auto job = LoadSchedulerStatusJobById(w, *options.statusExperimentId);
        if (!job.has_value())
        {
            w.commit();
            std::cerr << "ERROR: experiment not found." << std::endl;
            return 1;
        }
        jobs.push_back(*job);
    }
    else
    {
        std::vector<SchedulerStatusJob> train = LoadSchedulerStatusJobs(w, "running", "train", 500, false);
        std::vector<SchedulerStatusJob> infer = LoadSchedulerStatusJobs(w, "running", "infer", 500, false);
        std::vector<SchedulerStatusJob> analyze = LoadSchedulerStatusJobs(w, "running", "analyze", 500, false);
        jobs.reserve(train.size() + infer.size() + analyze.size());
        jobs.insert(jobs.end(), train.begin(), train.end());
        jobs.insert(jobs.end(), infer.begin(), infer.end());
        jobs.insert(jobs.end(), analyze.begin(), analyze.end());
    }
    w.commit();

    EnrichSchedulerStatusJobs(jobs, processes);

    if (!options.statusExperimentId.has_value() && jobs.empty())
    {
        std::cout << "No running experiments." << std::endl;
        return 0;
    }

    std::cout << (options.statusExperimentId.has_value() ? "EXPERIMENT STATUS" : "RUNNING EXPERIMENTS") << "\n";
    for (const auto& job : jobs)
        PrintCompactStatusJob(job);
    return 0;
}

std::string FormatAggregateResource(const SchedulerResourceAggregate& aggregate)
{
    std::ostringstream oss;
    oss << "workers=" << aggregate.workers
        << " cpu=" << std::fixed << std::setprecision(1) << aggregate.cpuPercent << "%"
        << " rss=" << std::fixed << std::setprecision(0) << aggregate.rssMb << " MB"
        << " mem=" << std::fixed << std::setprecision(1) << aggregate.memPercent << "%";
    return oss.str();
}

std::vector<std::string> BuildSchedulerStatusWarnings(const SchedulerStatusProcessSnapshot& processes,
                                                      const SchedulerWorkerAccounting& accounting)
{
    std::vector<std::string> warnings;
    if (!processes.processDetectionAvailable)
        warnings.push_back("process detection unavailable");
    if (processes.schedulerPids.size() > 1)
        warnings.push_back("multiple scheduler processes detected: " + std::to_string(processes.schedulerPids.size()));
    if (processes.maxTrainProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxTrainProcs, accounting.managedTrain))
        warnings.push_back("train worker count exceeds max-train-procs");
    if (processes.maxInferProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxInferProcs, accounting.managedInfer))
        warnings.push_back("infer worker count exceeds max-infer-procs");
    if (processes.maxAnalyzeProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxAnalyzeProcs, accounting.managedAnalyze))
        warnings.push_back("analysis worker count exceeds max-analyze-procs");
    if (!accounting.unmanagedWorkers.empty())
        warnings.push_back("unmanaged LSTM_Release worker processes detected: " + std::to_string(accounting.unmanagedWorkers.size()));
    const int identityMismatches = accounting.identityMismatchTrain +
        accounting.identityMismatchInfer + accounting.identityMismatchAnalyze;
    if (identityMismatches > 0)
        warnings.push_back(
            "authoritative worker identity mismatches detected: " +
            std::to_string(identityMismatches));
    const int expectedMissing = accounting.expectedMissingTrain +
        accounting.expectedMissingInfer + accounting.expectedMissingAnalyze;
    if (expectedMissing > 0)
        warnings.push_back(
            "expected worker processes missing: " +
            std::to_string(expectedMissing));

    if (processes.systemMemoryUsedMb.has_value() &&
        processes.systemMemoryTotalMb.has_value() &&
        *processes.systemMemoryTotalMb > 0.0)
    {
        const double fraction = *processes.systemMemoryUsedMb / *processes.systemMemoryTotalMb;
        if (fraction >= 0.90)
            warnings.push_back("system memory usage is high");
        const double workerRss = processes.trainResources.rssMb +
                                 processes.inferResources.rssMb +
                                 processes.analysisResources.rssMb;
        if (workerRss / *processes.systemMemoryTotalMb >= 0.80)
            warnings.push_back("worker RSS is high relative to system memory");
    }

    return warnings;
}

void PrintSchedulerResourceUsage(const SchedulerStatusProcessSnapshot& processes,
                                 const SchedulerWorkerAccounting& accounting)
{
    std::cout << "\nRESOURCE USAGE\n";
    std::cout << "  total_cpu=" << OptionalPercentText(processes.totalCpuPercent) << "\n";
    std::cout << "  system_memory_used=" << OptionalMbText(processes.systemMemoryUsedMb)
              << " total=" << OptionalMbText(processes.systemMemoryTotalMb);
    if (processes.systemMemoryUsedMb.has_value() &&
        processes.systemMemoryTotalMb.has_value() &&
        *processes.systemMemoryTotalMb > 0.0)
    {
        const double pct = 100.0 * *processes.systemMemoryUsedMb / *processes.systemMemoryTotalMb;
        std::cout << " used_percent=" << OptionalPercentText(pct);
    }
    std::cout << "\n";
    std::cout << "  scheduler " << FormatAggregateResource(processes.schedulerResources) << "\n";
    std::cout << "  managed train     " << FormatAggregateResource(accounting.managedTrainResources) << "\n";
    std::cout << "  managed infer     " << FormatAggregateResource(accounting.managedInferResources) << "\n";
    std::cout << "  managed analysis  " << FormatAggregateResource(accounting.managedAnalysisResources) << "\n";
    std::cout << "  unmanaged train   " << FormatAggregateResource(accounting.unmanagedTrainResources) << "\n";
    std::cout << "  unmanaged infer   " << FormatAggregateResource(accounting.unmanagedInferResources) << "\n";
    std::cout << "  unmanaged analysis " << FormatAggregateResource(accounting.unmanagedAnalysisResources) << "\n";
}

void PrintUnmanagedWorkers(const SchedulerWorkerAccounting& accounting)
{
    std::cout << "\nUNMANAGED LSTM PROCESSES\n";
    if (accounting.unmanagedWorkers.empty())
    {
        std::cout << "  none\n";
        return;
    }
    for (const auto& worker : accounting.unmanagedWorkers)
    {
        std::cout << "  pid=" << worker.pid
                  << " kind=" << worker.kind
                  << " reason=" << worker.reason
                  << " command=" << worker.command
                  << "\n";
    }
}

void PrintSchedulerWarnings(const std::vector<std::string>& warnings)
{
    std::cout << "\nWARNINGS\n";
    if (warnings.empty())
    {
        std::cout << "  none\n";
        return;
    }
    for (const auto& warning : warnings)
        std::cout << "  " << warning << "\n";
}

SchedulerIntelligenceRecord RowToIntelligenceRecord(const pqxx::row& row)
{
    SchedulerIntelligenceRecord record;
    record.experimentId = row[0].as<long long>();
    record.modelId = OptionalLongLongCell(row, 1);
    record.symbol = row[2].is_null() ? "unknown" : row[2].as<std::string>();
    record.predictionHorizon = row[3].is_null() ? 0 : row[3].as<int>();
    record.leaderScore = OptionalDoubleCell(row, 4);
    record.inferAccuracy = OptionalDoubleCell(row, 5);
    record.acceptAccuracy = OptionalDoubleCell(row, 6);
    record.targetEpochs = row[7].is_null() ? 0 : row[7].as<int>();
    if (!row[8].is_null())
        record.completedEpochs = row[8].as<int>();
    return record;
}

std::vector<SchedulerIntelligenceRecord> RowsToIntelligenceRecords(const pqxx::result& rows)
{
    std::vector<SchedulerIntelligenceRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
        records.push_back(RowToIntelligenceRecord(row));
    return records;
}

std::string IntelligenceBaseSql()
{
    return
        "WITH eligible AS ("
        "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
        "         a.leader_score, a.infer_accuracy, a.accept_accuracy, "
        "         a.target_epochs, a.completed_epochs, "
        "         e.completed_at, a.updated_at "
        "  FROM experiment_analysis_result a "
        "  JOIN experiment e ON e.experiment_id = a.experiment_id "
        "  WHERE e.status = 'completed' "
        "    AND a.analysis_status = 'completed' "
        "    AND COALESCE(a.analysis_scope, 'final') = 'final' "
        "    AND a.leader_score IS NOT NULL "
        "    AND a.infer_accuracy IS NOT NULL "
        ")";
}

SchedulerIntelligenceSnapshot LoadSchedulerIntelligenceSnapshot(pqxx::work& w,
                                                                       const QueueSnapshot& queueSnapshot)
{
    SchedulerIntelligenceSnapshot snapshot;
    snapshot.waitingTrain = queueSnapshot.pendingTrain;
    snapshot.waitingInfer = queueSnapshot.pendingInfer;
    snapshot.waitingAnalyze = queueSnapshot.pendingAnalyze;

    pqxx::result counts = w.exec(
        "SELECT "
        "COUNT(*) FILTER (WHERE status = 'completed' AND completed_at >= date_trunc('day', now())), "
        "COUNT(*) FILTER (WHERE status = 'failed' AND completed_at >= date_trunc('day', now())) "
        "FROM experiment;");
    if (!counts.empty())
    {
        snapshot.completedToday = counts[0][0].as<long long>();
        snapshot.failedToday = counts[0][1].as<long long>();
    }

    const std::string base = IntelligenceBaseSql();

    pqxx::result overall = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 1;");
    if (!overall.empty())
        snapshot.overallLeader = RowToIntelligenceRecord(overall[0]);

    pqxx::result recent = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " WHERE completed_at >= now() - interval '24 hours' "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 1;");
    if (!recent.empty())
        snapshot.recentBest24h = RowToIntelligenceRecord(recent[0]);

    pqxx::result bySymbol = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM ("
        "   SELECT eligible.*, row_number() OVER (PARTITION BY symbol ORDER BY leader_score DESC, infer_accuracy DESC, experiment_id DESC) AS rn "
        "   FROM eligible"
        " ) ranked "
        " WHERE rn = 1 "
        " ORDER BY symbol ASC "
        " LIMIT 20;");
    snapshot.leadersBySymbol = RowsToIntelligenceRecords(bySymbol);

    pqxx::result byHorizon = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM ("
        "   SELECT eligible.*, row_number() OVER (PARTITION BY prediction_horizon ORDER BY leader_score DESC, infer_accuracy DESC, experiment_id DESC) AS rn "
        "   FROM eligible"
        " ) ranked "
        " WHERE rn = 1 "
        " ORDER BY prediction_horizon ASC "
        " LIMIT 20;");
    snapshot.leadersByHorizon = RowsToIntelligenceRecords(byHorizon);

    pqxx::result top = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 5;");
    snapshot.top5 = RowsToIntelligenceRecords(top);

    pqxx::result worst = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score ASC NULLS LAST, infer_accuracy ASC NULLS LAST, experiment_id ASC "
        " LIMIT 5;");
    snapshot.worst5 = RowsToIntelligenceRecords(worst);

    pqxx::result dominatedRows = w.exec(base +
        " SELECT d.experiment_id, d.model_id, d.symbol, d.prediction_horizon, d.leader_score, d.infer_accuracy, d.accept_accuracy, "
        "        d.target_epochs, d.completed_epochs, x.experiment_id AS dominating_experiment_id, "
        "        x.leader_score AS dominating_leader_score, (x.leader_score - d.leader_score) AS leader_score_delta "
        " FROM eligible d "
        " JOIN LATERAL ("
        "   SELECT e2.experiment_id, e2.leader_score "
        "   FROM eligible e2 "
        "   WHERE e2.symbol = d.symbol "
        "     AND e2.prediction_horizon = d.prediction_horizon "
        "     AND e2.experiment_id <> d.experiment_id "
        "     AND e2.leader_score > d.leader_score "
        "     AND e2.infer_accuracy >= d.infer_accuracy "
        "     AND (d.accept_accuracy IS NULL OR e2.accept_accuracy IS NULL OR e2.accept_accuracy >= d.accept_accuracy) "
        "   ORDER BY e2.leader_score DESC, e2.infer_accuracy DESC, e2.experiment_id DESC "
        "   LIMIT 1 "
        " ) x ON true "
        " ORDER BY leader_score_delta DESC NULLS LAST, d.leader_score ASC "
        " LIMIT 10;");
    snapshot.dominated.reserve(dominatedRows.size());
    for (const auto& row : dominatedRows)
    {
        SchedulerDominatedRecord record;
        record.dominated = RowToIntelligenceRecord(row);
        record.dominatingExperimentId = row[9].as<long long>();
        record.dominatingLeaderScore = OptionalDoubleCell(row, 10);
        record.leaderScoreDelta = OptionalDoubleCell(row, 11);
        snapshot.dominated.push_back(record);
    }

    return snapshot;
}

void PrintIntelligenceRecord(const SchedulerIntelligenceRecord& record,
                                    const std::string& prefix = "  ")
{
    std::cout << prefix
              << "experiment_id=" << record.experimentId
              << " model_id=" << OptionalLongLongText(record.modelId)
              << " symbol=" << record.symbol
              << " H=" << record.predictionHorizon
              << " leader_score=" << OptionalDoubleText(record.leaderScore, 4)
              << " infer_accuracy=" << OptionalDoubleText(record.inferAccuracy, 4)
              << " accept_accuracy=" << OptionalDoubleText(record.acceptAccuracy, 4)
              << "\n";
}

void PrintIntelligenceList(const std::string& title,
                                  const std::vector<SchedulerIntelligenceRecord>& records)
{
    std::cout << "\n" << title << "\n";
    if (records.empty())
    {
        std::cout << "  none\n";
        return;
    }
    for (const auto& record : records)
        PrintIntelligenceRecord(record);
}

void PrintExperimentIntelligence(const SchedulerIntelligenceSnapshot& intelligence)
{
    std::cout << "\nEXPERIMENT INTELLIGENCE\n";
    std::cout << "Completed today: " << intelligence.completedToday
              << " failed today: " << intelligence.failedToday
              << " waiting train=" << intelligence.waitingTrain
              << " infer=" << intelligence.waitingInfer
              << " analyze=" << intelligence.waitingAnalyze
              << "\n";

    std::cout << "\nOverall leader:\n";
    if (intelligence.overallLeader.has_value())
        PrintIntelligenceRecord(*intelligence.overallLeader);
    else
        std::cout << "  none\n";

    std::cout << "\nBest completed in last 24h:\n";
    if (intelligence.recentBest24h.has_value())
        PrintIntelligenceRecord(*intelligence.recentBest24h);
    else
        std::cout << "  none\n";

    PrintIntelligenceList("Leaders by symbol:", intelligence.leadersBySymbol);
    PrintIntelligenceList("Leaders by horizon:", intelligence.leadersByHorizon);
    PrintIntelligenceList("Top 5:", intelligence.top5);
    PrintIntelligenceList("Worst 5:", intelligence.worst5);

    std::cout << "\nDominated candidates:\n";
    if (intelligence.dominated.empty())
    {
        std::cout << "  none\n";
    }
    else
    {
        for (const auto& record : intelligence.dominated)
        {
            std::cout << "  experiment_id=" << record.dominated.experimentId
                      << " symbol=" << record.dominated.symbol
                      << " H=" << record.dominated.predictionHorizon
                      << " leader_score=" << OptionalDoubleText(record.dominated.leaderScore, 4)
                      << " dominated_by=" << record.dominatingExperimentId
                      << " leader_score_delta=" << OptionalDoubleText(record.leaderScoreDelta, 4)
                      << "\n";
        }
    }
}

void PrintSchedulerStatusLeaderMachine(const std::string& scope,
                                              const SchedulerIntelligenceRecord& record,
                                              const std::optional<std::string>& scopeValue = std::nullopt)
{
    std::cout << "SCHEDULER_STATUS_LEADER"
              << ",scope=" << scope;
    if (scopeValue.has_value())
        std::cout << "," << *scopeValue;
    std::cout << ",experiment_id=" << record.experimentId
              << ",model_id=" << OptionalLongLongText(record.modelId)
              << ",symbol=" << record.symbol
              << ",prediction_horizon=" << record.predictionHorizon
              << ",leader_score=" << OptionalDoubleText(record.leaderScore, 4)
              << ",infer_accuracy=" << OptionalDoubleText(record.inferAccuracy, 4)
              << ",accept_accuracy=" << OptionalDoubleText(record.acceptAccuracy, 4)
              << std::endl;
}

void PrintExperimentIntelligenceMachine(const SchedulerIntelligenceSnapshot& intelligence)
{
    std::cout << "SCHEDULER_STATUS_INTELLIGENCE"
              << ",completed_today=" << intelligence.completedToday
              << ",failed_today=" << intelligence.failedToday
              << ",waiting_train=" << intelligence.waitingTrain
              << ",waiting_infer=" << intelligence.waitingInfer
              << ",waiting_analyze=" << intelligence.waitingAnalyze
              << ",dominated_count=" << intelligence.dominated.size()
              << ",overall_leader_experiment_id="
              << (intelligence.overallLeader.has_value()
                      ? std::to_string(intelligence.overallLeader->experimentId)
                      : "unknown")
              << ",overall_leader_score="
              << (intelligence.overallLeader.has_value()
                      ? OptionalDoubleText(intelligence.overallLeader->leaderScore, 4)
                      : "unknown")
              << std::endl;

    if (intelligence.overallLeader.has_value())
        PrintSchedulerStatusLeaderMachine("overall", *intelligence.overallLeader);
    if (intelligence.recentBest24h.has_value())
        PrintSchedulerStatusLeaderMachine("recent_24h", *intelligence.recentBest24h);
    for (const auto& record : intelligence.leadersBySymbol)
        PrintSchedulerStatusLeaderMachine("symbol", record, "symbol=" + record.symbol);
    for (const auto& record : intelligence.leadersByHorizon)
        PrintSchedulerStatusLeaderMachine("horizon", record, "prediction_horizon=" + std::to_string(record.predictionHorizon));
    for (const auto& record : intelligence.dominated)
    {
        std::cout << "SCHEDULER_STATUS_DOMINATED"
                  << ",experiment_id=" << record.dominated.experimentId
                  << ",dominated_by_experiment_id=" << record.dominatingExperimentId
                  << ",symbol=" << record.dominated.symbol
                  << ",prediction_horizon=" << record.dominated.predictionHorizon
                  << ",leader_score=" << OptionalDoubleText(record.dominated.leaderScore, 4)
                  << ",dominating_leader_score=" << OptionalDoubleText(record.dominatingLeaderScore, 4)
                  << std::endl;
    }
}

int PrintSchedulerStatus(const SchedulerOptions& options)
{
    const bool useColor = UseAnsiColors();
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;
    const auto globalControl = LoadLockedGlobalControl(w);
    if (!globalControl)
        return 1;
    const int globallyPausedWorkers = CountRows(
        w,
        "SELECT ("
        " (SELECT count(*) FROM experiment WHERE status='running' "
        "  AND worker_control_state='paused') +"
        " (SELECT count(*) FROM experiment_checkpoint_eval "
        "  WHERE status='running' AND phase='infer' "
        "  AND worker_control_state='paused')"
        ")::bigint;");
    const int selectivelyReleasedWorkers = CountRows(
        w,
        "SELECT ("
        " (SELECT count(*) FROM experiment e "
        "  JOIN experiment_global_control c ON c.singleton "
        "  WHERE e.status='running' "
        "  AND e.worker_control_state='running' "
        "  AND e.worker_global_pause_request_id=c.current_pause_request_id) +"
        " (SELECT count(*) FROM experiment_checkpoint_eval ce "
        "  JOIN experiment_global_control c ON c.singleton "
        "  WHERE ce.status='running' AND ce.phase='infer' "
        "  AND ce.worker_control_state='running' "
        "  AND ce.worker_global_pause_request_id=c.current_pause_request_id)"
        ")::bigint;");
    const int pendingCheckpointCancellations = CountRows(
        w,
        "SELECT count(*) FROM experiment "
        "WHERE status IN ('running','pending') "
        "AND cancel_after_checkpoint_epoch IS NOT NULL;");
    pqxx::result latestAdmin = w.exec(
        "SELECT request_id,action,COALESCE(cancellation_mode,'NULL'),"
        "infer_before_cancel,status,requested_at::text,"
        "COALESCE(completed_at::text,'NULL'),successful_count,"
        "missing_count,rejected_count,failed_count "
        "FROM experiment_admin_request "
        "ORDER BY request_id DESC LIMIT 1;");
    SchedulerServiceComposition services{w};
    const QueueSnapshot queueSnapshot =
        LoadQueueSnapshot(services.admission);
    const SchedulerStatusCounts counts = LoadSchedulerStatusCounts(w);
    const SchedulerIntelligenceSnapshot intelligence = LoadSchedulerIntelligenceSnapshot(w, queueSnapshot);
    std::vector<SchedulerStatusJob> runningTrain = LoadSchedulerStatusJobs(w, "running", "train", 50, false);
    std::vector<SchedulerStatusJob> runningInfer = LoadSchedulerStatusJobs(w, "running", "infer", 50, false);
    std::vector<SchedulerStatusJob> runningAnalyze = LoadSchedulerStatusJobs(w, "running", "analyze", 50, false);
    std::vector<SchedulerStatusJob> queued = LoadSchedulerStatusJobs(w, "pending", std::nullopt, 50, false);
    std::vector<SchedulerStatusJob> paused = LoadSchedulerStatusJobs(w, "paused", std::nullopt, 50, false);
    std::vector<SchedulerStatusJob> completed = LoadSchedulerStatusJobs(w, "completed", std::nullopt, 10, true);
    std::vector<SchedulerStatusJob> failed = LoadSchedulerStatusJobs(w, "failed", std::nullopt, 20, true);
    const auto authoritativeWorkers = LoadAuthoritativeSchedulerWorkers(w);
    std::vector<SchedulerCheckpointStatusJob> activeCheckpointInfer =
        LoadActiveCheckpointStatusJobs(w);
    pqxx::result schedulerLease = w.exec(
        "SELECT l.authority_state,"
        "COALESCE(l.owner_scheduler_invocation_id,'NULL'),"
        "l.fencing_token,COALESCE(l.acquired_at::text,'NULL'),"
        "COALESCE(l.heartbeat_at::text,'NULL'),"
        "COALESCE(l.expires_at::text,'NULL'),l.transition_reason,"
        "(l.expires_at IS NOT NULL "
        " AND l.expires_at<=clock_timestamp()) AS expired,"
        "COALESCE(i.canonical_executable_path,'NULL'),"
        "i.process_pid,i.process_group_id,i.process_start_identity "
        "FROM experiment_scheduler_lease l "
        "LEFT JOIN experiment_scheduler_invocation i "
        "ON i.scheduler_invocation_id="
        "l.owner_scheduler_invocation_id "
        "WHERE l.singleton=true;");
    pqxx::result schedulerProtocol = w.exec(
        "SELECT required_generation,cutover_state,"
        "COALESCE(cutover_completed_at::text,'NULL'),"
        "COALESCE(cutover_completed_by,'NULL'),"
        "COALESCE(cutover_process_evidence,'NULL'),"
        "legacy_no_pid_grace_seconds,"
        "COALESCE(failure_diagnostic,'NULL') "
        "FROM experiment_scheduler_protocol "
        "WHERE singleton=true;");
    pqxx::result durableWorkerCounts = w.exec(
        "SELECT capacity_class,"
        "count(*) FILTER (WHERE lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'identity_ambiguous')) AS consuming,"
        "count(*) FILTER (WHERE lifecycle_state='reserved') "
        " AS reservations,"
        "count(*) FILTER (WHERE lifecycle_state='identity_ambiguous') "
        " AS identity_mismatches,"
        "count(*) FILTER (WHERE lifecycle_state='observed') "
        " AS observed,"
        "count(*) FILTER (WHERE worker_kind='checkpoint_infer' "
        " AND lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'identity_ambiguous')) AS checkpoint_workers "
        "FROM experiment_scheduler_worker_attempt "
        "GROUP BY capacity_class ORDER BY capacity_class;");
    pqxx::result attemptOwnershipCounts = w.exec(
        "SELECT "
        "count(*) FILTER (WHERE a.lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'stopped','identity_ambiguous') "
        " AND a.scheduler_invocation_id="
        "     l.owner_scheduler_invocation_id) AS current_owner,"
        "count(*) FILTER (WHERE a.lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'stopped','identity_ambiguous') "
        " AND a.scheduler_invocation_id IS DISTINCT FROM "
        "     l.owner_scheduler_invocation_id) AS prior_or_legacy,"
        "count(*) FILTER (WHERE a.lifecycle_state='reserved') "
        " AS unresolved_launches,"
        "count(*) FILTER (WHERE a.lifecycle_state="
        " 'identity_ambiguous') AS orphan_candidates "
        "FROM experiment_scheduler_worker_attempt a "
        "CROSS JOIN experiment_scheduler_lease l "
        "WHERE l.singleton=true;");
    pqxx::result activeAttemptRows = w.exec(
        "SELECT a.worker_attempt_id,a.worker_kind,"
        "a.lifecycle_phase,a.capacity_class,a.lifecycle_state,"
        "COALESCE(a.scheduler_invocation_id,'legacy'),"
        "a.experiment_id,a.checkpoint_eval_id,a.worker_pid,"
        "COALESCE(a.observed_by_scheduler_invocation_id,'NULL'),"
        "COALESCE(a.reconciliation_result,'NULL'),"
        "COALESCE(a.diagnostic,'NULL') "
        "FROM experiment_scheduler_worker_attempt a "
        "WHERE a.lifecycle_state IN "
        "('reserved','spawned','running','observed',"
        "'stopped','identity_ambiguous') "
        "ORDER BY a.worker_attempt_id;");
    pqxx::result unresolvedLegacyNoPid = w.exec(
        "SELECT capacity_class,count(*) "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE ownership_origin='legacy_unverified' "
        "AND lifecycle_state='identity_ambiguous' "
        "AND worker_pid IS NULL "
        "GROUP BY capacity_class ORDER BY capacity_class;");
    w.commit();

    EnrichSchedulerStatusJobs(runningTrain, processes);
    EnrichSchedulerStatusJobs(runningInfer, processes);
    EnrichSchedulerStatusJobs(runningAnalyze, processes);
    EnrichSchedulerStatusJobs(queued, processes);
    EnrichSchedulerStatusJobs(paused, processes);
    EnrichSchedulerStatusJobs(completed, processes);
    EnrichSchedulerStatusJobs(failed, processes);
    EnrichCheckpointStatusJobs(activeCheckpointInfer, processes);
    auto processOperations =
        EA::GlobalExperimentControl::CreateNativeProcessOperations();
    const SchedulerWorkerAccounting workerAccounting =
        ComputeSchedulerWorkerAccounting(
            processes, authoritativeWorkers, *processOperations);

    const bool schedulerRunning = !processes.schedulerPids.empty();
    const std::string schedulerPid =
        schedulerRunning ? std::to_string(processes.schedulerPids.front()) : "unknown";
    const auto automationStateText = [](const std::optional<bool>& enabled) {
        if (!enabled.has_value())
            return std::string{"unknown"};
        return std::string{*enabled ? "enabled" : "disabled"};
    };
    SchedulerOwnerProcessEvidence schedulerExecutableEvidence =
        SchedulerOwnerProcessEvidence::Ambiguous;
    EA::GlobalExperimentControl::ProcessObservation
        schedulerOwnerObservation;
    const std::string schedulerCanonicalExecutable =
        schedulerLease.size() == 1
            ? schedulerLease[0][8].as<std::string>()
            : "NULL";
    if (schedulerLease.size() == 1 &&
        schedulerLease[0][0].as<std::string>() == "active" &&
        schedulerCanonicalExecutable != "NULL" &&
        !schedulerLease[0][9].is_null() &&
        !schedulerLease[0][10].is_null() &&
        !schedulerLease[0][11].is_null())
    {
        schedulerExecutableEvidence = InspectSchedulerOwnerProcess(
            schedulerLease[0][9].as<int>(),
            schedulerLease[0][10].as<int>(),
            schedulerLease[0][11].as<std::string>(),
            schedulerCanonicalExecutable,
            &schedulerOwnerObservation);
    }
    const std::string schedulerObservedExecutable =
        schedulerOwnerObservation.executable.empty()
            ? "NULL" : schedulerOwnerObservation.executable;
    const std::string schedulerExecutableIdentityMatch =
        schedulerExecutableEvidence == SchedulerOwnerProcessEvidence::Valid
            ? "1"
            : (schedulerExecutableEvidence ==
                       SchedulerOwnerProcessEvidence::Ambiguous
                   ? "unknown" : "0");

    std::cout << "Scheduler Status\n";
    if (schedulerLease.size() == 1)
    {
        std::cout << "Scheduler authority: "
                  << schedulerLease[0][0].as<std::string>()
                  << " owner=" << schedulerLease[0][1].as<std::string>()
                  << " fence=" << schedulerLease[0][2].as<long long>()
                  << " acquired=" << schedulerLease[0][3].as<std::string>()
                  << " heartbeat=" << schedulerLease[0][4].as<std::string>()
                  << " expires=" << schedulerLease[0][5].as<std::string>()
                  << " expired="
                  << (schedulerLease[0][7].as<bool>() ? 1 : 0)
                  << "\n";
    }
    if (schedulerProtocol.size() == 1)
    {
        std::cout << "Scheduler protocol: generation="
                  << schedulerProtocol[0][0].as<int>()
                  << " cutover_state="
                  << schedulerProtocol[0][1].as<std::string>()
                  << " completed_at="
                  << schedulerProtocol[0][2].as<std::string>()
                  << " legacy_no_pid_grace_seconds="
                  << schedulerProtocol[0][5].as<int>()
                  << "\n";
    }
    std::cout << "Status reporter executable: " << options.schedulerExecutablePath << "\n"
              << "Scheduler canonical executable: "
              << schedulerCanonicalExecutable << "\n"
              << "Scheduler executable identity: "
              << EA::SchedulerCore::SchedulerOwnerProcessEvidenceText(
                     schedulerExecutableEvidence)
              << " observed=" << schedulerObservedExecutable << "\n";
    if (attemptOwnershipCounts.size() == 1)
    {
        std::cout << "Durable workers: current_owner="
                  << attemptOwnershipCounts[0][0].as<long long>()
                  << " prior_or_legacy="
                  << attemptOwnershipCounts[0][1].as<long long>()
                  << " unresolved_launches="
                  << attemptOwnershipCounts[0][2].as<long long>()
                  << " orphan_candidates="
                  << attemptOwnershipCounts[0][3].as<long long>()
                  << "\n";
    }
    std::cout << "Global experiment execution: "
              << globalControl->desiredState
              << " active_request_id="
              << (globalControl->activeRequestId
                      ? std::to_string(*globalControl->activeRequestId)
                      : "none")
              << " current_pause_request_id="
              << (globalControl->currentPauseRequestId
                      ? std::to_string(*globalControl->currentPauseRequestId)
                      : "none")
              << " paused_workers=" << globallyPausedWorkers
              << " selectively_released_workers="
              << selectivelyReleasedWorkers
              << " cancellation_mode="
              << globalControl->cancellationMode.value_or("none")
              << " infer_before_cancel="
              << (globalControl->inferBeforeCancel ? "yes" : "no")
              << " pending_checkpoint_cancellations="
              << pendingCheckpointCancellations
              << "\n";
    if (!latestAdmin.empty())
    {
        std::cout << "Latest administrative request: id="
                  << latestAdmin[0][0].as<long long>()
                  << " action=" << latestAdmin[0][1].as<std::string>()
                  << " mode=" << latestAdmin[0][2].as<std::string>()
                  << " infer_before_cancel="
                  << (latestAdmin[0][3].as<bool>() ? "yes" : "no")
                  << " status=" << latestAdmin[0][4].as<std::string>()
                  << " successful="
                  << latestAdmin[0][7].as<int>()
                  << " missing=" << latestAdmin[0][8].as<int>()
                  << " rejected=" << latestAdmin[0][9].as<int>()
                  << " failed=" << latestAdmin[0][10].as<int>()
                  << "\n";
    }
    if (schedulerRunning)
    {
        std::cout << "Scheduler process: "
                  << Colorize("running", "32", useColor)
                  << " pid=" << schedulerPid;
        if (processes.schedulerPids.size() > 1)
            std::cout << " additional_pids=" << (processes.schedulerPids.size() - 1);
        std::cout << "\n";
    }
    else if (!processes.processDetectionAvailable)
    {
        std::cout << "Scheduler process: unknown (process detection unavailable)\n";
    }
    else
    {
        std::cout << Colorize("Scheduler process not detected.", "31", useColor) << "\n";
    }

    std::cout << "Poll interval: "
              << (processes.schedulerPollSeconds.has_value()
                      ? std::to_string(*processes.schedulerPollSeconds) + "s"
                      : "unknown")
              << "\n";
    std::cout << "Configured worker limits: train="
              << OptionalIntText(processes.maxTrainProcs)
              << " infer=" << OptionalIntText(processes.maxInferProcs)
              << " analyze=" << OptionalIntText(processes.maxAnalyzeProcs)
              << "\n";
    std::cout << "Continuation Automation:\n"
              << "  automatic evaluation="
              << automationStateText(processes.autoEvaluateContinuations)
              << " automatic queueing="
              << automationStateText(processes.autoQueueContinuations)
              << " dry-run=" << automationStateText(processes.continuationDryRun)
              << "\n"
              << "  scan interval="
              << (processes.continuationScanSeconds.has_value()
                      ? std::to_string(*processes.continuationScanSeconds) + "s"
                      : "unknown")
              << " maximum queues per scan="
              << OptionalIntText(processes.continuationMaxQueuesPerScan)
              << "\n"
              << "  last scan=unavailable next scan=unavailable"
              << " counts=unavailable (scheduler process memory only)\n";
    std::cout << "Detected worker processes:\n"
              << "  managed train=" << workerAccounting.managedTrain
              << " infer=" << workerAccounting.managedInfer
              << " analyze=" << workerAccounting.managedAnalyze
              << "\n"
              << "  managed running train="
              << workerAccounting.managedRunningTrain
              << " infer=" << workerAccounting.managedRunningInfer
              << " analyze=" << workerAccounting.managedRunningAnalyze
              << "\n"
              << "  managed paused train="
              << workerAccounting.managedPausedTrain
              << " infer=" << workerAccounting.managedPausedInfer
              << " analyze=" << workerAccounting.managedPausedAnalyze
              << "\n"
              << "  unmanaged train=" << workerAccounting.unmanagedTrain
              << " infer=" << workerAccounting.unmanagedInfer
              << " analyze=" << workerAccounting.unmanagedAnalyze
              << "\n"
              << "  identity mismatch train="
              << workerAccounting.identityMismatchTrain
              << " infer=" << workerAccounting.identityMismatchInfer
              << " analyze=" << workerAccounting.identityMismatchAnalyze
              << "\n"
              << "  expected missing train="
              << workerAccounting.expectedMissingTrain
              << " infer=" << workerAccounting.expectedMissingInfer
              << " analyze=" << workerAccounting.expectedMissingAnalyze
              << "\n";

    PrintSchedulerResourceUsage(processes, workerAccounting);
    PrintUnmanagedWorkers(workerAccounting);
    const std::vector<std::string> warnings = BuildSchedulerStatusWarnings(processes, workerAccounting);
    PrintSchedulerWarnings(warnings);

    std::cout << "\nOverall Counts\n"
              << "  queued=" << counts.queued
              << " paused=" << counts.paused
              << " running=" << counts.running
              << " completed=" << counts.completed
              << " failed=" << counts.failed
              << " cancelled=" << counts.cancelled
              << "\n";
    std::cout << "  pending_train=" << queueSnapshot.pendingTrain
              << " pending_infer=" << queueSnapshot.pendingInfer
              << " pending_analyze=" << queueSnapshot.pendingAnalyze
              << " running_train=" << queueSnapshot.runningTrain
              << " running_infer=" << queueSnapshot.runningInfer
              << " running_analyze=" << queueSnapshot.runningAnalyze
              << "\n";

    PrintExperimentIntelligence(intelligence);

    PrintStatusJobTable("Active Training Jobs", runningTrain, useColor, true, false);
    PrintStatusJobTable("Active Inference Jobs", runningInfer, useColor, false, false);
    PrintCheckpointStatusJobs(activeCheckpointInfer);
    PrintStatusJobTable("Active Analysis Jobs", runningAnalyze, useColor, false, false);
    PrintStatusJobTable("Queued Jobs", queued, useColor, false, false);
    PrintStatusJobTable("Paused Jobs", paused, useColor, false, false);
    PrintStatusJobTable("Recent Completed Experiments", completed, useColor, false, false);
    PrintStatusJobTable("Failed Experiments Summary", failed, useColor, false, true);

    if (SchedulerStatusShouldEmitMachineRecords(options))
    {
        if (schedulerLease.size() == 1)
        {
            std::cout << "\nSCHEDULER_STATUS_OWNERSHIP"
                      << ",authority_state="
                      << schedulerLease[0][0].as<std::string>()
                      << ",owner_scheduler_invocation_id="
                      << schedulerLease[0][1].as<std::string>()
                      << ",fencing_token="
                      << schedulerLease[0][2].as<long long>()
                      << ",acquired_at="
                      << schedulerLease[0][3].as<std::string>()
                      << ",heartbeat_at="
                      << schedulerLease[0][4].as<std::string>()
                      << ",expires_at="
                      << schedulerLease[0][5].as<std::string>()
                      << ",transition_reason="
                      << schedulerLease[0][6].as<std::string>()
                      << ",expired="
                      << (schedulerLease[0][7].as<bool>() ? 1 : 0)
                      << ",canonical_executable_path="
                      << schedulerLease[0][8].as<std::string>()
                      << ",scheduler_canonical_executable_path="
                      << schedulerCanonicalExecutable
                      << std::endl;
        }
        std::cout << "SCHEDULER_STATUS_EXECUTABLE_IDENTITY"
                  << ",status_reporter_executable_path="
                  << options.schedulerExecutablePath
                  << ",scheduler_canonical_executable_path="
                  << schedulerCanonicalExecutable
                  << ",scheduler_observed_executable_path="
                  << schedulerObservedExecutable
                  << ",scheduler_identity_result="
                  << EA::SchedulerCore::SchedulerOwnerProcessEvidenceText(
                         schedulerExecutableEvidence)
                  << ",scheduler_identity_match="
                  << schedulerExecutableIdentityMatch
                  << std::endl;
        if (schedulerProtocol.size() == 1)
        {
            std::cout
                << "SCHEDULER_STATUS_PROTOCOL"
                << ",required_generation="
                << schedulerProtocol[0][0].as<int>()
                << ",cutover_state="
                << schedulerProtocol[0][1].as<std::string>()
                << ",cutover_completed_at="
                << schedulerProtocol[0][2].as<std::string>()
                << ",cutover_completed_by="
                << schedulerProtocol[0][3].as<std::string>()
                << ",cutover_process_evidence="
                << schedulerProtocol[0][4].as<std::string>()
                << ",legacy_no_pid_grace_seconds="
                << schedulerProtocol[0][5].as<int>()
                << ",failure_diagnostic="
                << schedulerProtocol[0][6].as<std::string>()
                << std::endl;
        }
        for (const pqxx::row& legacy :
             unresolvedLegacyNoPid)
        {
            std::cout
                << "SCHEDULER_STATUS_LEGACY_NO_PID"
                << ",capacity_class="
                << legacy[0].as<std::string>()
                << ",unresolved="
                << legacy[1].as<long long>()
                << ",capacity_consumed="
                << legacy[1].as<long long>()
                << std::endl;
        }
        for (const pqxx::row& capacity : durableWorkerCounts)
        {
            std::cout << "SCHEDULER_STATUS_GLOBAL_CAPACITY"
                      << ",capacity_class="
                      << capacity[0].as<std::string>()
                      << ",consuming=" << capacity[1].as<long long>()
                      << ",reservations=" << capacity[2].as<long long>()
                      << ",identity_mismatches="
                      << capacity[3].as<long long>()
                      << ",observed_prior_workers="
                      << capacity[4].as<long long>()
                      << ",checkpoint_workers="
                      << capacity[5].as<long long>()
                      << std::endl;
        }
        if (attemptOwnershipCounts.size() == 1)
        {
            std::cout << "SCHEDULER_STATUS_WORKER_OWNERSHIP"
                      << ",current_owner="
                      << attemptOwnershipCounts[0][0].as<long long>()
                      << ",prior_or_legacy="
                      << attemptOwnershipCounts[0][1].as<long long>()
                      << ",unresolved_launches="
                      << attemptOwnershipCounts[0][2].as<long long>()
                      << ",orphan_candidates="
                      << attemptOwnershipCounts[0][3].as<long long>()
                      << std::endl;
        }
        for (const pqxx::row& attempt : activeAttemptRows)
        {
            std::cout << "SCHEDULER_STATUS_WORKER_ATTEMPT"
                      << ",worker_attempt_id="
                      << attempt[0].as<long long>()
                      << ",worker_kind="
                      << attempt[1].as<std::string>()
                      << ",phase=" << attempt[2].as<std::string>()
                      << ",capacity_class="
                      << attempt[3].as<std::string>()
                      << ",state=" << attempt[4].as<std::string>()
                      << ",launch_scheduler_invocation_id="
                      << attempt[5].as<std::string>()
                      << ",experiment_id="
                      << attempt[6].as<long long>()
                      << ",checkpoint_eval_id="
                      << (attempt[7].is_null()
                              ? "NULL"
                              : attempt[7].c_str())
                      << ",pid="
                      << (attempt[8].is_null()
                              ? "NULL"
                              : attempt[8].c_str())
                      << ",observed_by="
                      << attempt[9].as<std::string>()
                      << ",reconciliation_result="
                      << attempt[10].as<std::string>()
                      << ",diagnostic="
                      << attempt[11].as<std::string>()
                      << std::endl;
        }
        std::cout << "\nSCHEDULER_STATUS"
                  << ",running=" << (schedulerRunning ? "1" : "0")
                  << ",global_desired_state=" << globalControl->desiredState
                  << ",active_admin_request_id="
                  << (globalControl->activeRequestId
                          ? std::to_string(*globalControl->activeRequestId)
                          : "NULL")
                  << ",current_pause_request_id="
                  << (globalControl->currentPauseRequestId
                          ? std::to_string(
                                *globalControl->currentPauseRequestId)
                          : "NULL")
                  << ",globally_paused_workers=" << globallyPausedWorkers
                  << ",selectively_released_workers="
                  << selectivelyReleasedWorkers
                  << ",active_cancellation_mode="
                  << globalControl->cancellationMode.value_or("NULL")
                  << ",active_infer_before_cancel="
                  << (globalControl->inferBeforeCancel ? "1" : "0")
                  << ",pending_checkpoint_cancellations="
                  << pendingCheckpointCancellations
                  << ",latest_admin_request_id="
                  << (latestAdmin.empty()
                          ? "NULL"
                          : std::to_string(
                                latestAdmin[0][0].as<long long>()))
                  << ",latest_admin_status="
                  << (latestAdmin.empty()
                          ? "NULL"
                          : latestAdmin[0][4].as<std::string>())
                  << ",pid=" << (schedulerRunning ? schedulerPid : "unknown")
                  << ",train_workers=" << processes.trainWorkers
                  << ",infer_workers=" << processes.inferWorkers
                  << ",analysis_workers=" << processes.analysisWorkers
                  << ",managed_train_workers=" << workerAccounting.managedTrain
                  << ",managed_infer_workers=" << workerAccounting.managedInfer
                  << ",managed_analysis_workers=" << workerAccounting.managedAnalyze
                  << ",managed_running_train_workers="
                  << workerAccounting.managedRunningTrain
                  << ",managed_running_infer_workers="
                  << workerAccounting.managedRunningInfer
                  << ",managed_running_analysis_workers="
                  << workerAccounting.managedRunningAnalyze
                  << ",managed_paused_train_workers="
                  << workerAccounting.managedPausedTrain
                  << ",managed_paused_infer_workers="
                  << workerAccounting.managedPausedInfer
                  << ",managed_paused_analysis_workers="
                  << workerAccounting.managedPausedAnalyze
                  << ",unmanaged_train_workers=" << workerAccounting.unmanagedTrain
                  << ",unmanaged_infer_workers=" << workerAccounting.unmanagedInfer
                  << ",unmanaged_analysis_workers=" << workerAccounting.unmanagedAnalyze
                  << ",identity_mismatch_train_workers="
                  << workerAccounting.identityMismatchTrain
                  << ",identity_mismatch_infer_workers="
                  << workerAccounting.identityMismatchInfer
                  << ",identity_mismatch_analysis_workers="
                  << workerAccounting.identityMismatchAnalyze
                  << ",expected_missing_train_workers="
                  << workerAccounting.expectedMissingTrain
                  << ",expected_missing_infer_workers="
                  << workerAccounting.expectedMissingInfer
                  << ",expected_missing_analysis_workers="
                  << workerAccounting.expectedMissingAnalyze
                  << ",poll_seconds=" << OptionalIntText(processes.schedulerPollSeconds)
                  << ",max_train_procs=" << OptionalIntText(processes.maxTrainProcs)
                  << ",max_infer_procs=" << OptionalIntText(processes.maxInferProcs)
                  << ",max_analyze_procs=" << OptionalIntText(processes.maxAnalyzeProcs)
                  << std::endl;
        std::cout << "SCHEDULER_STATUS_RESOURCE"
                  << ",total_cpu_percent=" << OptionalDoubleText(processes.totalCpuPercent, 1)
                  << ",system_memory_used_mb=" << OptionalDoubleText(processes.systemMemoryUsedMb, 0)
                  << ",system_memory_total_mb=" << OptionalDoubleText(processes.systemMemoryTotalMb, 0)
                  << ",scheduler_cpu_percent=" << OptionalDoubleText(processes.schedulerResources.cpuPercent, 1)
                  << ",scheduler_rss_mb=" << OptionalDoubleText(processes.schedulerResources.rssMb, 0)
                  << ",scheduler_mem_percent=" << OptionalDoubleText(processes.schedulerResources.memPercent, 1)
                  << ",train_cpu_percent=" << OptionalDoubleText(processes.trainResources.cpuPercent, 1)
                  << ",train_rss_mb=" << OptionalDoubleText(processes.trainResources.rssMb, 0)
                  << ",train_mem_percent=" << OptionalDoubleText(processes.trainResources.memPercent, 1)
                  << ",train_workers=" << processes.trainWorkers
                  << ",managed_train_cpu_percent=" << OptionalDoubleText(workerAccounting.managedTrainResources.cpuPercent, 1)
                  << ",managed_train_rss_mb=" << OptionalDoubleText(workerAccounting.managedTrainResources.rssMb, 0)
                  << ",managed_train_workers=" << workerAccounting.managedTrain
                  << ",unmanaged_train_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedTrainResources.cpuPercent, 1)
                  << ",unmanaged_train_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedTrainResources.rssMb, 0)
                  << ",unmanaged_train_workers=" << workerAccounting.unmanagedTrain
                  << ",infer_cpu_percent=" << OptionalDoubleText(processes.inferResources.cpuPercent, 1)
                  << ",infer_rss_mb=" << OptionalDoubleText(processes.inferResources.rssMb, 0)
                  << ",infer_mem_percent=" << OptionalDoubleText(processes.inferResources.memPercent, 1)
                  << ",infer_workers=" << processes.inferWorkers
                  << ",managed_infer_cpu_percent=" << OptionalDoubleText(workerAccounting.managedInferResources.cpuPercent, 1)
                  << ",managed_infer_rss_mb=" << OptionalDoubleText(workerAccounting.managedInferResources.rssMb, 0)
                  << ",managed_infer_workers=" << workerAccounting.managedInfer
                  << ",unmanaged_infer_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedInferResources.cpuPercent, 1)
                  << ",unmanaged_infer_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedInferResources.rssMb, 0)
                  << ",unmanaged_infer_workers=" << workerAccounting.unmanagedInfer
                  << ",analysis_cpu_percent=" << OptionalDoubleText(processes.analysisResources.cpuPercent, 1)
                  << ",analysis_rss_mb=" << OptionalDoubleText(processes.analysisResources.rssMb, 0)
                  << ",analysis_mem_percent=" << OptionalDoubleText(processes.analysisResources.memPercent, 1)
                  << ",analysis_workers=" << processes.analysisWorkers
                  << ",managed_analysis_cpu_percent=" << OptionalDoubleText(workerAccounting.managedAnalysisResources.cpuPercent, 1)
                  << ",managed_analysis_rss_mb=" << OptionalDoubleText(workerAccounting.managedAnalysisResources.rssMb, 0)
                  << ",managed_analysis_workers=" << workerAccounting.managedAnalyze
                  << ",unmanaged_analysis_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedAnalysisResources.cpuPercent, 1)
                  << ",unmanaged_analysis_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedAnalysisResources.rssMb, 0)
                  << ",unmanaged_analysis_workers=" << workerAccounting.unmanagedAnalyze
                  << std::endl;
        std::cout << "SCHEDULER_STATUS_CONTINUATION"
                  << ",auto_evaluate="
                  << (processes.autoEvaluateContinuations.has_value()
                          ? (*processes.autoEvaluateContinuations ? "1" : "0")
                          : "unknown")
                  << ",auto_queue="
                  << (processes.autoQueueContinuations.has_value()
                          ? (*processes.autoQueueContinuations ? "1" : "0")
                          : "unknown")
                  << ",dry_run="
                  << (processes.continuationDryRun.has_value()
                          ? (*processes.continuationDryRun ? "1" : "0")
                          : "unknown")
                  << ",scan_seconds=" << OptionalIntText(processes.continuationScanSeconds)
                  << ",max_queues_per_scan="
                  << OptionalIntText(processes.continuationMaxQueuesPerScan)
                  << ",last_scan=unavailable"
                  << ",next_scan=unavailable"
                  << ",scan_candidates=unavailable"
                  << ",scan_evaluated=unavailable"
                  << ",scan_already_satisfied=unavailable"
                  << ",scan_eligible=unavailable"
                  << ",scan_queued=unavailable"
                  << ",scan_errors=unavailable"
                  << std::endl;
        for (const auto& worker : workerAccounting.unmanagedWorkers)
        {
            std::cout << "SCHEDULER_STATUS_UNMANAGED_WORKER"
                      << ",pid=" << worker.pid
                      << ",kind=" << worker.kind
                      << ",reason=" << worker.reason
                      << ",command=" << worker.command
                      << std::endl;
        }
        for (const auto& worker :
             workerAccounting.workerClassifications)
        {
            std::cout << "SCHEDULER_STATUS_WORKER"
                      << ",pid=" << worker.pid
                      << ",kind=" << worker.kind
                      << ",managed=" << (worker.managed ? 1 : 0)
                      << ",authoritative="
                      << (worker.authoritative ? 1 : 0)
                      << ",detected=" << (worker.detected ? 1 : 0)
                      << ",execution_state="
                      << EA::GlobalExperimentControl::ToString(
                             worker.executionState)
                      << ",lifecycle_status="
                      << (worker.lifecycleStatus.empty()
                              ? "NULL" : worker.lifecycleStatus)
                      << ",attempt_state="
                      << (worker.attemptLifecycleState.empty()
                              ? "NULL" : worker.attemptLifecycleState)
                      << ",identity_result="
                      << EA::GlobalExperimentControl::ToString(
                             worker.identity)
                      << ",executable_identity_match="
                      << (worker.executableIdentityMatch
                              ? (*worker.executableIdentityMatch ? "1" : "0")
                              : "unknown")
                      << ",canonical_executable_path="
                      << worker.expectedExecutable.value_or("NULL")
                      << ",observed_executable_path="
                      << worker.observedExecutable.value_or("NULL")
                      << ",experiment_id="
                      << (worker.experimentId
                              ? std::to_string(*worker.experimentId)
                              : "NULL")
                      << ",checkpoint_eval_id="
                      << (worker.checkpointEvalId
                              ? std::to_string(*worker.checkpointEvalId)
                              : "NULL")
                      << ",reason=" << worker.reason
                      << std::endl;
        }
        std::cout << "SCHEDULER_STATUS_COUNT"
                  << ",queued=" << counts.queued
                  << ",paused=" << counts.paused
                  << ",running=" << counts.running
                  << ",completed=" << counts.completed
                  << ",failed=" << counts.failed
                  << ",cancelled=" << counts.cancelled
                  << ",pending_train=" << queueSnapshot.pendingTrain
                  << ",pending_infer=" << queueSnapshot.pendingInfer
                  << ",pending_analyze=" << queueSnapshot.pendingAnalyze
                  << ",running_train=" << queueSnapshot.runningTrain
                  << ",running_infer=" << queueSnapshot.runningInfer
                  << ",running_analyze=" << queueSnapshot.runningAnalyze
                  << std::endl;
        PrintExperimentIntelligenceMachine(intelligence);
        for (const auto& job : runningTrain)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : runningInfer)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : activeCheckpointInfer)
            PrintCheckpointStatusJobMachine(job);
        for (const auto& job : runningAnalyze)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : queued)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : paused)
            PrintSchedulerStatusJobMachine(job);
    }

    return 0;
}

void PrintExperimentSchedulerHelp(const char* executable)
{
    const std::string exe = executable ? executable : "LSTM_Release";
    std::cout
        << "Usage: " << exe << " --queue-experiment --symbol=SYMBOL --prediction-horizon=N --target-epochs=N "
        << "[--threshold=VALUE] [--core-lr=VALUE] [--head-lr=VALUE] [--checkpoint-interval=N] "
        << "[--training-objective=legacy|profitability_auxiliary_v1] "
        << "[--ablate-features=NAME[,NAME...]] "
        << "[--train-start=YYYY-MM-DD] [--train-end=YYYY-MM-DD] [--infer-start=YYYY-MM-DD] [--infer-end=YYYY-MM-DD] "
        << "[--checkpoint-infer] [--checkpoint-infer-min-epoch=N] [--checkpoint-infer-interval=N] "
        << "[--checkpoint-policy --checkpoint-policy-min-leader-score=VALUE|--checkpoint-policy-min-infer-accuracy=VALUE|--checkpoint-policy-top-n=N]\n"
        << "Validation sources may add --continuation-candidate-excluded; they remain ineligible unless include_excluded=true is configured explicitly.\n"
        << "Example: " << exe << " --queue-experiment --symbol=eurusdrmp --prediction-horizon=12 --target-epochs=240\n"
        << "Resume: " << exe << " --queue-experiment --resume-model-id=MODEL_ID --target-epochs=240 "
        << "[--resume-expand-input-width] [--checkpoint-interval=N] [--infer-start=YYYY-MM-DD] [--infer-end=YYYY-MM-DD]\n"
        << "Auto-resume: " << exe << " --queue-experiment --auto-resume --symbol=SYMBOL --prediction-horizon=N "
        << "--target-epochs=240 [--threshold=VALUE] [--train-start=YYYY-MM-DD] [--train-end=YYYY-MM-DD]\n"
        << "Usage: " << exe << " --queue-sweep --prediction-horizon=N --target-epochs=N "
        << "[--threshold=VALUE] [--core-lr=VALUE] [--head-lr=VALUE] [--checkpoint-interval=N] "
        << "[--checkpoint-infer] [--checkpoint-infer-min-epoch=N] [--checkpoint-infer-interval=N] "
        << "[--checkpoint-policy --checkpoint-policy-min-leader-score=VALUE|--checkpoint-policy-min-infer-accuracy=VALUE|--checkpoint-policy-top-n=N]\n"
        << "Example: " << exe << " --queue-sweep --prediction-horizon=12 --target-epochs=240\n"
        << "Supported sweep symbols:";
    for (const auto& symbol : EA::SupportedSymbols::TrainingSymbols())
        std::cout << " " << symbol;
    std::cout << "\n"
        << "Usage: " << exe
        << " --enqueue-experiment --symbol=SYMBOL --prediction-horizon=N --c-next-threshold=VALUE "
        << "--core-lr-mult=VALUE --head-lr-mult=VALUE --target-epochs=N --checkpoint-interval=N "
        << "--train-start=YYYY-MM-DD --train-end=YYYY-MM-DD [--infer-start=YYYY-MM-DD --infer-end=YYYY-MM-DD] "
        << "[--resume-model-id=MODEL_ID] [--allow-duplicate-experiment]\n"
        << "Usage: " << exe
        << " --schedule-experiments [--max-train-procs=N] [--max-infer-procs=N] "
        << "[--max-analyze-procs=N] [--scheduler-poll-seconds=N] [--scheduler-once] "
        << "[--semantic-worker-registry=/absolute/path/to/registry.json] "
        << "[--legacy-layout6-infer-worker=/absolute/path/to/LSTM_Release] "
        << "[--scheduler-log-dir=PATH] [--auto-generate-reports] [--experiment-report-dir=PATH] "
        << "[--lstm-profile-hotspots] [--lstm-profile-output=PATH] "
        << "[--scheduler-verbose] [--dry-run] [--recover-orphans-only] "
        << "[--auto-evaluate-continuations] [--auto-queue-continuations] "
        << "[--continuation-scan-seconds=N] [--continuation-max-queues-per-scan=N] "
        << "[--continuation-dry-run]\n"
        << "Scheduler worker limits accept non-negative integers. Zero prevents new workers "
        << "in that capacity class without stopping the scheduler or existing workers.\n"
        << "The semantic-worker registry is validated before scheduler authority "
        << "is acquired. Train and final analysis use its current published worker; "
        << "final and checkpoint inference select an exact registered semantic layout. "
        << "The transitional layout-6 option is only an exact-identity assertion "
        << "against the registry and is never forwarded to child workers.\n"
        << "Usage: " << exe
        << " --create-economic-calendar-snapshot [--dry-run]\n"
        << "Computes the deterministic current economic-calendar corpus identity. "
        << "Dry-run reports without persistence; apply creates or reuses one "
        << "finalized immutable snapshot.\n"
        << "Continuation automation is disabled by default. Evaluation-only persists/reuses Phase 3A "
        << "decisions without creating children. --auto-queue-continuations implies evaluation; children "
        << "enter the normal pending/train queue and obey ordinary scheduler capacity. Defaults: scan "
        << "interval 300 seconds, maximum 1 queue per scan. --continuation-dry-run computes and logs "
        << "evaluation/queue proposals without database writes.\n"
        << "Usage: " << exe
        << " --scheduler-status [--log-level=quiet|summary|diagnostic]\n"
        << "Usage: " << exe
        << " --verify-profitability-evidence=EXPERIMENT_ID[,EXPERIMENT_ID...]\n"
        << "Exact-final profitability verification preserves declared order, "
        << "rejects duplicate IDs, forbids checkpoint substitution, and uses "
        << "one repeatable-read, read-only transaction. Exit codes: 0 all "
        << "valid, 4 unavailable/incomplete evidence, 3 invalid/ambiguous "
        << "evidence, 2 database/tool error, 1 argument error.\n"
        << "Usage: " << exe
        << " --campaign-profitability-readiness=RANKING_SNAPSHOT_ID\n"
        << "Campaign profitability readiness validates frozen exact-FINAL "
        << "provenance and renders a deterministic shadow ordering only. "
        << "Current rank remains authoritative; live profitability weight and "
        << "score contribution remain zero; activation is never performed.\n"
        << "Usage: " << exe
        << " --shadow-rank-campaign-profitability=RANKING_SNAPSHOT_ID "
        << "--profitability-shadow-weights=WEIGHT[,WEIGHT...]\n"
        << "Campaign profitability weighted ranking is shadow-only and uses "
        << "the exact frozen control snapshot/evaluation evidence. Current/live "
        << "rank remains authoritative; live profitability weight and score "
        << "contribution remain zero. It performs no activation, database write, "
        << "experiment creation/queueing, or scheduler modification. Phase 9 "
        << "accepts finite unique weights from 0 through 0.05.\n"
        << "Usage: " << exe
        << " --calibrate-campaign-profitability=RANKING_SNAPSHOT_ID\n"
        << "Phase 10 performs the fixed 0 through 0.05 empirical sweep, exact "
        << "coverage classification, anchor comparisons, stability analysis, "
        << "and advisory production-readiness assessment in one repeatable-read, "
        << "read-only transaction. It cannot backfill frozen evidence or modify "
        << "ranking, recommendation, experiment, scheduler, or worker state.\n"
        << "Usage: " << exe
        << " --validate-campaign-profitability-temporal\n"
        << "Phase 11 audits every immutable historical ranking snapshot for "
        << "exact point-in-time reconstruction and strictly subsequent exact-"
        << "identity FINAL profitability outcomes. Contaminated or incomplete "
        << "cohorts fail closed; the precommitted candidate weight is 0.025 and "
        << "is never selected from holdout outcomes. The command is read-only.\n"
        << "Usage: " << exe
        << " --prepare-campaign-profitability-forward-validation="
           "RANKING_SNAPSHOT_ID,OUTCOME_START,OUTCOME_END\n"
        << "Phase 11 emits a deterministic no-write forward-validation "
        << "precommit for the exact zero-weight control and precommitted 0.025 "
        << "shadow ranking. OUTCOME_START and OUTCOME_END are ISO dates and must "
        << "be strictly after the immutable snapshot decision date. It neither "
        << "launches outcome inference nor creates experiments or authority.\n"
        << "Usage: " << exe
        << " --prepare-campaign-profitability-outcome-jobs="
           "VALIDATION_COHORT_IDENTITY_HASH\n"
        << "Phase 12 verifies the committed prospective artifact and emits "
        << "deduplicated frozen-model outcome jobs and deferred execution "
        << "commands. It is repeatable-read and read-only; it never runs "
        << "inference, training, scheduling, ranking, or activation.\n"
        << "Usage: " << exe
        << " --compare-campaign-profitability-prospective="
           "VALIDATION_COHORT_IDENTITY_HASH\n"
        << "Phase 13 verifies the immutable Phase 11/12 artifacts and reads "
        << "persisted frozen-model outcomes to compare the precommitted "
        << "Top-5/10/20 candidate and control selections. It fails closed on "
        << "identity, metric, window, compatibility, or changed-source "
        << "coverage gaps and remains pending until the window is complete. "
        << "The command is repeatable-read and read-only.\n"
        << "Usage: " << exe
        << " --compare-experiment-pair=EXPERIMENT_A_ID:EXPERIMENT_B_ID "
           "[--summary]\n"
        << "Generic experiment-pair comparison preserves argument order as "
        << "arm A and arm B, loads exact persisted FINAL evidence in one "
        << "repeatable-read transaction, and renders B-minus-A deltas without "
        << "a winner or database writes. A control-versus-feature-ablation "
        << "mask difference is recognized only when exactly one mask is empty. "
        << "Exit 0 includes complete, incomplete, and scientifically "
        << "incompatible reports; exit 3 is missing or invalid evidence; exit "
        << "2 is a database/tool error.\n"
        << "Usage: " << exe
        << " --compare-experiment-replications=A_ID:B_ID,C_ID:D_ID\n"
        << "Generic multi-pair replication comparison preserves pair and arm "
        << "argument order, requires globally unique experiment IDs, and "
        << "loads every pair in one repeatable-read transaction. It reports "
        << "exact B-minus-A pair deltas and unweighted descriptive aggregates "
        << "only after replication compatibility is established. Seed may "
        << "differ across pairs but must match within each pair. Missing "
        << "evidence remains NULL; no winner, ranking, recommendation, or "
        << "database write is produced.\n"
        << "Usage: " << exe
        << " --plan-experiment-replications=SOURCE_A_ID:SOURCE_B_ID "
           "--replication-seeds=SEED[,SEED...]\n"
        << "Plans an ordered, read-only replication wave by copying the exact "
        << "authoritative source-arm scientific identities and changing only "
        << "fresh_initialization_seed. Seeds are strict positive uint32 values, "
        << "must be unique, and retain argument order. Reusing the source seed "
        << "is allowed and explicitly classified as same_seed_repeated_pairs. "
        << "Equivalent experiments are reported without selecting or mutating "
        << "them. Statistical independence is never inferred.\n"
        << "Usage: " << exe
        << " --compare-feature-ablation-pair=CONTROL_ID:ABLATION_ID "
        << "--expected-ablation-mask=FEATURE[,FEATURE...]\n"
        << "Feature-ablation pair comparison canonicalizes the requested mask, "
        << "requires an empty-mask control and exact-mask ablation, resolves "
        << "exact FINAL inference and "
        << "profitability evidence, and performs no database writes. Omitting "
        << "--expected-ablation-mask retains the legacy consensus-family "
        << "ABLATION_ID:ENABLED_ID argument order; new comparisons must provide "
        << "the expected mask and use CONTROL_ID:ABLATION_ID. Exit codes: "
        << "0 complete comparison, 4 incomplete evidence, 3 invalid pair, "
        << "2 database/tool error.\n"
        << "Usage: " << exe
        << " --causal-surprise-observability=EXPERIMENT_ID "
        << "[--causal-surprise-observability-scope="
           "train|infer|combined]\n"
        << "Reports read-only causal first-release surprise coverage over "
        << "the experiment's exact post-warmup feature-row populations. "
        << "Combined (the default) sums independently generated train and "
        << "inference populations. The feature-ablation mask is reported but "
        << "is downstream of this diagnostic.\n"
        << "Usage: " << exe
        << " --causal-surprise-coverage-gaps=EXPERIMENT_ID "
        << "[--causal-surprise-coverage-gaps-scope="
           "train|infer|combined]\n"
        << "Attributes the same causal-surprise feature-row population by "
        << "raw provenance reason, event family, authoritative agency, UTC "
        << "calendar year, compatibility reason, and deterministic remediation "
        << "priority. The command is repeatable-read and read-only.\n"
        << "Usage: " << exe
        << " --compare-feature-ablation-replications="
        << "CONTROL_ID:ABLATION_ID[,CONTROL_ID:ABLATION_ID...] "
        << "--expected-ablation-mask=FEATURE[,FEATURE...]\n"
        << "Replication aggregation preserves declared order, uses the "
        << "exact requested treatment mask, preserves each pair's economic-"
        << "calendar snapshot provenance without treating it as treatment, "
        << "versioned profitability-primary policy, performs no writes or "
        << "activation, and separates software readiness from scientific "
        << "decision. Exit codes: 0 all evidence complete, 4 incomplete or "
        << "profitability unavailable, 3 invalid or missing evidence, "
        << "2 database/tool error. Omitting --expected-ablation-mask retains "
        << "the legacy consensus ABLATED_ID:ENABLED_ID mode.\n"
        << "Usage: " << exe
        << " --corrected-causal-surprise-replication-status="
           "CONTROL_ID:TREATMENT_ID[,CONTROL_ID:TREATMENT_ID...] "
        << "--corrected-causal-surprise-anchor-pair="
           "CONTROL_ID:TREATMENT_ID\n"
        << "Evaluates corrected evidence and renders the immutable, outcome-"
           "blind follow-on plan in one repeatable-read transaction. It never "
           "creates experiments or changes scheduler state.\n"
        << "Usage: " << exe
        << " --materialize-corrected-causal-surprise-replication="
           "CONTROL_ID:TREATMENT_ID[,CONTROL_ID:TREATMENT_ID...] "
        << "--corrected-causal-surprise-anchor-pair="
           "CONTROL_ID:TREATMENT_ID "
        << "--expected-corrected-replication-plan-hash=HASH "
           "[--dry-run | --yes]\n"
        << "Materialization re-evaluates the validity-only continuation gate, "
           "requires the exact previewed plan hash and explicit --yes, and "
           "creates one control/treatment pair atomically. --dry-run performs "
           "no writes.\n"
        << "Usage: " << exe
        << " --compare-training-objective-pair=CONTROL_ID:TREATMENT_ID "
        << "--pair-primary-profitability-metric=aggregate|average "
        << "--pair-min-profitability-improvement=VALUE "
        << "--pair-max-profitability-worsening=VALUE "
        << "--pair-max-infer-accuracy-decrease=VALUE "
        << "--pair-max-accept-accuracy-decrease=VALUE "
        << "--pair-max-accept-rate-decrease=VALUE "
        << "--pair-max-leader-score-decrease=VALUE "
        << "--pair-max-neutral-proportion-increase=VALUE\n"
        << "Training-objective pair comparison uses one repeatable-read, "
        << "read-only transaction and never persists comparison results. "
        << "Exit 0 includes every scientific disposition; exit 3 is a "
        << "missing, ambiguous, or invalid evidence contract.\n"
        << "Usage: " << exe
        << " --pause-all-experiments|--pause-all [--dry-run | --yes]\n"
        << "Usage: " << exe
        << " --resume-all-experiments|--resume-all [--dry-run | --yes]\n"
        << "Usage: " << exe
        << " --cancel-all-experiments (--immediate | --after-next-checkpoint) "
        << "[--infer-before-cancel] [--dry-run | --yes]\n"
        << "Global controls are database-authoritative and work without a running "
        << "scheduler. Pause stops identity-validated train/infer workers; resume "
        << "queues generation members for capacity-limited scheduler admission. "
        << "Cancellation uses durable checkpoints for optional inference.\n"
        << "Usage: " << exe
        << " --generate-experiment-reports [--experiment-report-dir=PATH]\n"
        << "Usage: " << exe
        << " --backup-database [--backup-output=PATH]\n"
        << "Usage: " << exe
        << " --status [--experiment-id=ID]\n"
        << "Usage: " << exe
        << " --model-info --model=MODEL_ID\n"
        << "Usage: " << exe
        << " --experiment-metadata=ID\n"
        << "Usage: " << exe
        << " --list-experiment-models=ID\n"
        << "Usage: " << exe
        << " --list-experiment-lineage=ID [--include-parent-models]\n"
        << "Usage: " << exe
        << " --backfill-experiment-metadata\n"
        << "Usage: " << exe
        << " --pause-experiment=ID | --resume-experiment=ID | --cancel-experiment=ID | "
        << "--retry-failed-experiment=ID | --requeue-training=ID | "
        << "--requeue-inference=ID | --requeue-analysis=ID "
        << "[--dry-run] [--yes]\n"
        << "Training requeue retains the same experiment ID and persistent priority, "
        << "requires an authoritative intermediate checkpoint, and rejects any "
        << "attached or residual worker identity. It never signals a worker.\n"
        << "Usage: " << exe
        << " --set-experiment-priority=ID:high|normal|low\n"
        << "Individual resume queues pending work with temporary resume priority; "
        << "SIGCONT occurs only after compatible scheduler capacity is admitted.\n"
        << "Usage: " << exe
        << " --pause-campaign-materialization=ID | "
        << "--resume-campaign-materialization=ID [--dry-run | --yes]\n"
        << "Campaign-materialization pause freezes the immutable manifest and "
        << "controls only exactly resolved experiments. Resume releases only "
        << "group-owned pauses into ordinary capacity-limited priority admission.\n"
        << "Usage: " << exe
        << " --retry-checkpoint-eval=CHECKPOINT_EVAL_ID [--dry-run]\n"
        << "Usage: " << exe
        << " --evaluate-checkpoint-policy=CHECKPOINT_EVAL_ID\n"
        << "Usage: " << exe
        << " --checkpoint-policy-status=CHECKPOINT_EVAL_ID\n"
        << "Usage: " << exe
        << " --enable-continuation-policy=EXPERIMENT_ID | --disable-continuation-policy=EXPERIMENT_ID | "
        << "--set-continuation-policy=EXPERIMENT_ID:key=value,key=value\n"
        << "Continuation keys: target_epochs, min_evals, patience, min_leader_score, "
        << "min_infer_accuracy, min_profitability_actionable_count, "
        << "min_profitability_aggregate_terminal_horizon_log_return_sum, "
        << "min_profitability_average_terminal_horizon_log_return_per_actionable_prediction, "
        << "min_improvement, max_degradation, top_n, scope, trend_mode, "
        << "source_mode, include_excluded, candidate_excluded, inherit_to_child, progression_mode, "
        << "target_increment, max_target_epochs, target_sequence\n"
        << "Continuation policy inheritance is disabled by default. When inherit_to_child=true, "
        << "fixed_increment policies require positive target_increment and max_target_epochs; "
        << "target_sequence policies use colon-separated strictly increasing targets such as "
        << "140:160:180:200:220:240. The final configured target creates a terminal child with "
        << "outgoing inheritance disabled. "
        << "Legacy inherited policies without a maximum retain compatibility until reconfigured.\n"
        << "Continuation gates use AND semantics. trend_delta=latest_metric-first_metric; "
        << "non_degrading requires trend_delta>=-max_degradation and improving requires "
        << "trend_delta>=min_improvement. trend_mode defaults to none. Profitability gates "
        << "are disabled when all three profitability keys are null. When any is configured, "
        << "the exact authoritative observation for the already-selected final/best-checkpoint/"
        << "latest-checkpoint source is required; no cross-scope fallback or source reselection "
        << "occurs. Configured profitability requirements use AND semantics. Actionable count is "
        << "independent from prediction count. Aggregate and average fields are terminal-horizon "
        << "directional log-return primitives, not portfolio P&L; a zero-actionable observation "
        << "has a valid zero aggregate and undefined average. Profitability does not affect "
        << "ranking or trend. Use value=null to clear an optional profitability key.\n"
        << "Usage: " << exe
        << " --evaluate-continuation=EXPERIMENT_ID | --queue-continuation=EXPERIMENT_ID | "
        << "--continuation-status=EXPERIMENT_ID\n"
        << "Usage: " << exe
        << " --generate-experiment-recommendations [--recommendation-policy=key=value,...] "
        << "[--recommendation-symbol=SYMBOL] [--recommendation-horizon=N] "
        << "[--recommendation-source-experiment=ID] [--recommendation-max=N]\n"
        << "Usage: " << exe
        << " --list-experiment-recommendations [--recommendation-status-filter=proposed|rejected|expired|approved] "
        << "[--recommendation-symbol=SYMBOL] [--recommendation-horizon=N] "
        << "[--recommendation-scan-id=ID] [--recommendation-limit=N]\n"
        << "Usage: " << exe
        << " --recommendation-status=ID | --list-experiment-recommendation-scans "
        << "[--recommendation-limit=N] | --recommendation-scan-status=ID\n"
        << "Usage: " << exe
        << " --score-experiment-recommendations [--recommendation-scoring-policy=key=value,...] "
        << "[--recommendation-status-filter=proposed] "
        << "[--recommendation-symbol=SYMBOL] [--recommendation-horizon=N] "
        << "[--recommendation-scan-id=ID] [--recommendation-id=ID] [--recommendation-score-limit=N]\n"
        << "Usage: " << exe
        << " --list-experiment-recommendation-scores [--recommendation-score-run-id=ID] "
        << "[--recommendation-id=ID] [--recommendation-score-min=VALUE] "
        << "[--recommendation-symbol=SYMBOL] [--recommendation-horizon=N] "
        << "[--recommendation-score-limit=N]\n"
        << "Usage: " << exe
        << " --recommendation-score-status=ID | --explain-recommendation-score=ID | "
        << "--list-experiment-recommendation-score-runs [--recommendation-score-limit=N] | "
        << "--recommendation-score-run-status=ID\n"
        << "Usage: " << exe
        << " --approve-experiment-recommendation=ID | --reject-experiment-recommendation=ID | "
        << "--expire-experiment-recommendation=ID "
        << "[--recommendation-review-reason-code=CODE] [--recommendation-review-reason=TEXT] "
        << "[--recommendation-reviewer=TEXT] [--recommendation-review-note=TEXT] "
        << "[--recommendation-review-score-id=ID]\n"
        << "Usage: " << exe
        << " --list-experiment-recommendation-reviews [--recommendation-id=ID] "
        << "[--recommendation-review-action=approve|reject|expire] "
        << "[--recommendation-review-limit=N] | --recommendation-review-status=ID | "
        << "--recommendation-review-history=RECOMMENDATION_ID\n"
        << "Phase 4A recommendations, scores, and reviews are advisory only: review never creates or queues experiments.\n"
        << "Usage: " << exe
        << " --evaluate-experiment-recommendations --recommendation-scan-id=ID | "
        << "--evaluate-experiment-recommendation=ID "
        << "[--recommendation-evaluation-policy=key=value,...] "
        << "[--recommendation-evaluation-limit=N] "
        << "[--recommendation-evaluation-dry-run]\n"
        << "Usage: " << exe
        << " --list-experiment-recommendation-evaluations "
        << "[--recommendation-scan-id=ID] [--recommendation-id=ID] "
        << "[--recommendation-evaluation-disposition=STATE] "
        << "[--recommendation-evaluation-limit=N] | "
        << "--recommendation-evaluation-status=ID | "
        << "--explain-recommendation-evaluation=ID\n"
        << "Usage: " << exe
        << " --list-experiment-recommendation-evaluation-runs "
        << "[--recommendation-evaluation-limit=N] | "
        << "--recommendation-evaluation-run-status=ID\n"
        << "Phase 4B evaluation is advisory only: it never creates or queues experiments or changes scheduler state.\n"
        << "Usage: " << exe
        << " --rank-experiment-recommendation-evaluations "
        << "(--recommendation-ranking-evaluation-run-id=ID | "
        << "--recommendation-ranking-scan-id=ID | "
        << "--recommendation-ranking-symbol=SYMBOL [--recommendation-ranking-horizon=N] | "
        << "--recommendation-ranking-horizon=N | --recommendation-ranking-family=NAME | "
        << "--recommendation-ranking-global) [--recommendation-ranking-limit=N] "
        << "[--recommendation-ranking-dry-run]\n"
        << "Usage: " << exe
        << " --list-experiment-recommendation-ranking-snapshots "
        << "[--recommendation-ranking-limit=N] | --recommendation-ranking-status=ID | "
        << "--list-experiment-recommendation-ranking-members=ID "
        << "[--recommendation-ranking-bucket=advisory_ready|blocked|non_actionable] "
        << "[--recommendation-ranking-limit=N] | --recommendation-ranking-member-status=ID\n"
        << "Usage: " << exe
        << " --compare-experiment-recommendation-evaluations=LEFT:RIGHT | "
        << "--compare-experiment-recommendation-ranking-members=LEFT:RIGHT\n"
        << "Phase 4B ranking and comparison are advisory only: they never create or queue experiments or change scheduler state.\n"
        << "Usage: " << exe
        << " --approve-conversion-proposal=ID | --reject-conversion-proposal=ID "
        << "--conversion-proposal-review-request-id=TOKEN "
        << "[--conversion-proposal-review-operator=TEXT] "
        << "[--conversion-proposal-review-reason=TEXT]\n"
        << "Usage: " << exe
        << " --show-conversion-proposal=ID | "
        << "--list-conversion-proposal-reviews=ID "
        << "[--conversion-proposal-review-limit=N] | "
        << "--list-conversion-proposals-by-review-status="
        << "pending_review|approved|rejected "
        << "[--conversion-proposal-review-limit=N]\n"
        << "Phase 4C proposal review is administrative only: it never creates "
        << "or queues an experiment or changes scheduler state.\n"
        << "Usage: " << exe
        << " --execute-approved-conversion-proposal=PROPOSAL_ID | "
        << "--conversion-proposal-execution-status=PROPOSAL_ID\n"
        << "Phase 4C conversion creates one paused experiment only: it does not "
        << "queue, start, resume, or schedule the experiment.\n"
        << "Usage: " << exe
        << " --activate-recommendation-conversion-execution=EXECUTION_ID | "
        << "--recommendation-conversion-activation-status=ACTIVATION_ID\n"
        << "Phase 4C activation moves that existing paused experiment to "
        << "pending/train. It starts no worker and does not bypass the scheduler.\n"
        << "Usage: " << exe
        << " --recommendation-conversion-workflow=PROPOSAL_ID | "
        << "--list-recommendation-conversion-workflows "
        << "[--conversion-workflow-state=STATE] "
        << "[--conversion-workflow-limit=N]\n"
        << "Phase 4C workflow observation is read-only and never changes an "
        << "experiment, audit record, worker, or scheduler state.\n"
        << "Usage: " << exe
        << " --plan-recommendation-campaign "
        << "--campaign-ranking-snapshot=ID "
        << "[--campaign-limit=N] [--campaign-candidate-limit=N] "
        << "[--campaign-symbol=SYMBOL] [--campaign-horizon=N] "
        << "[--campaign-donchian20-arms=enabled|zero_ablation|enabled:zero_ablation] "
        << "[--campaign-min-leader-score=VALUE] "
        << "[--campaign-min-inference-accuracy=VALUE] "
        << "[--campaign-max-neutral-proportion=VALUE] "
        << "[--campaign-min-profitability=VALUE] "
        << "[--campaign-max-per-symbol=N] "
        << "[--campaign-max-per-horizon=N] "
        << "[--campaign-max-per-source-experiment=N] "
        << "[--campaign-reconsider-rejected] "
        << "[--campaign-reconsider-failed] "
        << "[--campaign-reconsider-cancelled]\n"
        << "Phase 4D campaign planning reads one explicit durable ranking "
        << "snapshot and Phase 4C workflow history. It creates no proposal or "
        << "experiment and never starts the scheduler or a worker.\n"
        << "Usage: " << exe
        << " --review-recommendation-campaign "
        << "--campaign-ranking-snapshot=ID [campaign policy options]\n"
        << "Phase 4D campaign review deterministically explains the selected, "
        << "excluded, duplicate, family, symbol, and horizon structure of the "
        << "read-only campaign plan. It writes no database row.\n"
        << "Usage: " << exe
        << " --approve-recommendation-campaign | "
        << "--reject-recommendation-campaign "
        << "--campaign-ranking-snapshot=ID "
        << "--campaign-review-identity-hash=HASH "
        << "--campaign-reviewer=TEXT --campaign-review-reason=TEXT "
        << "[campaign policy options]\n"
        << "Usage: " << exe
        << " --show-recommendation-campaign-approval=ID | "
        << "--list-recommendation-campaign-approvals "
        << "[--campaign-approval-decision=approved|rejected] "
        << "[--campaign-approval-limit=N]\n"
        << "Phase 4D campaign approval records immutable human authorization "
        << "for one exact reconstructed review. It never executes a campaign "
        << "or creates, queues, or modifies an experiment.\n"
        << "Usage: " << exe
        << " --materialize-recommendation-campaign "
        << "--campaign-approval-id=ID --campaign-materialized-by=IDENTITY "
        << "--campaign-materialization-reason=TEXT\n"
        << "Usage: " << exe
        << " --show-recommendation-campaign-materialization=ID | "
        << "--list-recommendation-campaign-materializations "
        << "[--campaign-approval-id=ID] "
        << "[--campaign-materialization-limit=N]\n"
        << "Campaign materialization creates only the exact Phase 4C proposal "
        << "set. Conversion review, execution, activation, and experiments "
        << "remain separate explicit actions.\n"
        << "Usage: " << exe
        << " --show-recommendation-campaign-handoff=ID | "
        << "--list-recommendation-campaign-handoffs "
        << "[--campaign-handoff-limit=N]\n"
        << "Campaign handoff is a read-only projection of each materialized "
        << "proposal's current Phase 4C review, execution, and activation "
        << "evidence. It never advances or repairs workflow state.\n"
        << "Usage: " << exe
        << " --review-recommendation-campaign-materialization=ID "
        << "--campaign-proposal-review-decision=approve|reject "
        << "--campaign-proposal-review-operator=TEXT "
        << "--campaign-proposal-review-reason=TEXT [--dry-run] [--yes]\n"
        << "Campaign proposal review atomically records ordinary Phase 4C "
        << "review rows for every exact persisted materialization member. "
        << "It never executes or activates proposals, creates experiments, "
        << "or starts the scheduler or workers. --yes is required unless "
        << "--dry-run is supplied.\n"
        << "Usage: " << exe
        << " --execute-recommendation-campaign-materialization=ID "
        << "[--dry-run] [--yes]\n"
        << "Phase 5 campaign execution atomically invokes the existing Phase "
        << "4C paused conversion for every exact persisted materialization "
        << "member. It never activates or queues experiments and never starts "
        << "the scheduler or workers. --yes is required unless --dry-run is "
        << "supplied.\n"
        << "Usage: " << exe
        << " --activate-recommendation-campaign-materialization=ID "
        << "[--dry-run] [--yes]\n"
        << "Phase 5 campaign activation atomically invokes the existing Phase "
        << "4C paused/train to pending/train activation for every exact "
        << "persisted materialization member. It starts neither scheduler nor "
        << "workers and performs no automatic follow-up. --yes is required "
        << "unless --dry-run is supplied.\n"
        << "Usage: " << exe
        << " --launch-recommendation-campaign-materialization=ID "
        << "[--dry-run] [--yes]\n"
        << "Phase 5 atomic campaign launch invokes the existing Phase 4C "
        << "execution and activation authorities for every exact materialized "
        << "member in one transaction, leaving successful members pending/train. "
        << "It starts neither scheduler nor workers and performs no automatic "
        << "follow-up. --yes is required unless --dry-run is supplied.\n"
        << "Usage: " << exe
        << " --recommendation-campaign-status=MATERIALIZATION_ID\n"
        << "Phase 5 campaign status reads one repeatable-read snapshot of every "
        << "exact materialized member and its Phase 4C and experiment lifecycle "
        << "evidence. It writes nothing, takes no locks, and neither polls nor "
        << "controls the scheduler or workers.\n"
        << "Usage: " << exe
        << " --recommendation-campaign-outcome-assessment=MATERIALIZATION_ID\n"
        << "Phase 5 campaign outcome assessment reads one repeatable-read "
        << "snapshot, reuses campaign-status lifecycle evidence, and compares "
        << "the exact persisted source and result scientific evidence. It "
        << "persists nothing and authorizes no follow-up.\n"
        << "Usage: " << exe
        << " --campaign-operations-admit MATERIALIZATION_ID "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Campaign Operations admission explicitly binds one exact, "
        << "validated Phase 4D recommendation campaign materialization to "
        << "one immutable operational campaign. It grants no budget, "
        << "accepts no request, dispatches nothing, and changes no experiment "
        << "or scheduler state.\n"
        << "Usage: " << exe
        << " --campaign-operations-budget-grant CAMPAIGN_ID | "
        << "--campaign-operations-budget-amend CAMPAIGN_ID | "
        << "--campaign-operations-budget-supersede CAMPAIGN_ID "
        << "--campaign-operations-expected-budget-version N "
        << "--campaign-operations-budget-value UNITS "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-budget-revoke CAMPAIGN_ID "
        << "--campaign-operations-expected-budget-version N "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-accept-request CAMPAIGN_ID "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON "
        << "[--campaign-operations-reservation-expires-at "
           "YYYY-MM-DDTHH:MM:SS.ffffffZ] --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-budget-status CAMPAIGN_ID | "
        << "--campaign-operations-request-status REQUEST_ID\n"
        << "Campaign Operations Phase 2 administers the member-unit budget "
        << "ledger and atomically persists one held reservation plus one "
        << "ready durable request. It never dispatches, invokes Phase 5, "
        << "changes an experiment, starts the scheduler, or launches a "
        << "worker. Capability role assignment remains a separate "
        << "administrator action.\n"
        << "Usage: " << exe
        << " --campaign-operations-pause CAMPAIGN_ID | "
        << "--campaign-operations-resume CAMPAIGN_ID "
        << "--campaign-operations-expected-control-version N "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-cancel CAMPAIGN_ID "
        << "--campaign-operations-request-id REQUEST_ID "
        << "--campaign-operations-expected-request-version N "
        << "--campaign-operations-operation-key KEY "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-control-status CAMPAIGN_ID | "
        << "(--campaign-operations-reconcile-observe RUN_KEY | "
        << "--campaign-operations-reconcile-recover RUN_KEY) "
        << "[--campaign-operations-reconcile-after-request-id ID] "
        << "[--campaign-operations-reconcile-limit N] --yes\n"
        << "Campaign Operations Phase 4 controls only future Campaign "
        << "Operations actions. It records cancellation intent separately "
        << "from settlement, never refunds committed units, and delegates "
        << "bound pending cancellation to lifecycle authority after releasing "
        << "Campaign Operations locks. Reconciliation observes first; only "
        << "the recover form invokes a named safe owning transition. Neither "
        << "form signals workers or controls scheduler processes.\n"
        << "Usage: " << exe
        << " --campaign-operations-complete-if-settled CAMPAIGN_ID "
        << "--campaign-operations-operation-key KEY "
        << "--campaign-operations-actor ACTOR "
        << "--campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-completion-status CAMPAIGN_ID\n"
        << "Campaign Operations Phase 5 records one immutable operational "
        << "completion only when every accepted obligation is proven settled "
        << "in one serializable evidence transaction. Completion never changes "
        << "experiment lifecycle and never means scientific success. There "
        << "is no force-complete, reopen, override, or delete command.\n"
        << "Usage: " << exe
        << " --campaign-operations-production-readiness | "
        << "--campaign-operations-production-status\n"
        << "Campaign Operations Phase H1 commands are read-only. They report "
        << "migration 055, exact generation-52 evidence, immutable admission "
        << "and Attempt V2 evidence, role readiness, blocked leases, and "
        << "Completion V1 nested-V2 proof.\n"
        << "Usage: " << exe
        << " --campaign-operations-production-enable "
        << "--campaign-operations-operation-key KEY "
        << "--campaign-operations-expected-production-version N "
        << "--campaign-operations-independent-verification-reference REF "
        << "--campaign-operations-actor ACTOR --campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-production-disable "
        << "--campaign-operations-operation-key KEY "
        << "--campaign-operations-expected-production-version N "
        << "--campaign-operations-actor ACTOR --campaign-operations-reason REASON --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-dispatch-request "
        << "--campaign-operations-request-id ID "
        << "--campaign-operations-expected-request-version N "
        << "--campaign-operations-operation-key KEY "
        << "--campaign-operations-actor ACTOR --yes\n"
        << "Usage: " << exe
        << " --campaign-operations-manager-run-once LIMIT --yes\n"
        << "Phase H3 run-once takes one optimistic read-only candidate snapshot,"
        << " processes at most LIMIT requests sequentially, and has no daemon,"
        << " polling, sleep, or continuous CLI/daemon mode. ADR-0020 accepts"
        << " external deployment-owned H4 supervision that repeatedly invokes"
        << " this bounded command.\n"
        << "Phase H2 mutations are default-off, caller-keyed, and single-request "
        << "only. They require the dedicated deployed roles and a clean Release "
        << "build; H4 adds no scheduler polling or database singleton, heartbeat,"
        << " lease, or leader-election authority.\n"
        << "Usage: " << exe
        << " --stop-after-checkpoint=ID:EPOCH | --clear-stop-after-checkpoint=ID | "
        << "--stop-after-checkpoint-all=EPOCH | --clear-stop-after-checkpoint-all | "
        << "--enable-checkpoint-infer=ID | --disable-checkpoint-infer=ID | "
        << "--checkpoint-infer-min-epoch=ID:EPOCH | --checkpoint-infer-interval=ID:EPOCH_INTERVAL | "
        << "--enable-checkpoint-policy=ID | --disable-checkpoint-policy=ID | "
        << "--set-checkpoint-policy=ID:key=value,key=value\n"
        << "Usage: " << exe
        << " --stop-experiment=ID | --stop-all-experiments [--dry-run] [--yes] [--force]\n"
        << "Usage: " << exe
        << " --reconcile-worker-attempt=WORKER_ATTEMPT_ID [--dry-run | --yes]\n"
        << "Reconciles only an exact identity_ambiguous experiment-worker attempt after "
        << "locked lifecycle and live process identity verification; it never signals or dispatches work.\n"
        << "Usage: " << exe
        << " --recover-failed-inference=EXPERIMENT_ID [--dry-run | --yes]\n"
        << "Repairs only uniquely proven historical failed final-inference completion; "
        << "it never reruns inference or creates an attempt or result.\n"
        << "Usage: " << exe
        << " --analyze-experiment=EXPERIMENT_ID | --analyze-completed-experiments | "
        << "--print-experiment-leaderboard [--leaderboard-symbol=SYMBOL] "
        << "[--leaderboard-horizon=N] [--leaderboard-limit=N]\n"
        << "Backup note: --backup-database writes a PostgreSQL custom-format dump with schema and data; "
        << "migrations remain schema history, and --backup-output=Database/backups/LSTM_latest.dump overwrites a stable file.\n"
        << "Queue exit codes: 0=created, 1=invalid_arguments, 2=database_error, 3=duplicates_only\n";
}

int RunCampaignOperationsCommand(const SchedulerOptions& options)
{
    if (options.campaignOperationsProductionReadiness)
        return EA::CampaignOperations::RunProductionReadinessCommand(
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER"),
            std::cout, std::cerr,
            EA::CampaignOperations::CaptureActualManagerBuildContract(
                options.schedulerExecutablePath));
    if (options.campaignOperationsProductionStatus)
        return EA::CampaignOperations::RunProductionStatusCommand(
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER"),
            std::cout, std::cerr);
    if (options.campaignOperationsManagerRunOnceLimit)
        return EA::CampaignOperations::RunCampaignOperationsManagerOnceCommand(
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER"),
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER"),
            *options.campaignOperationsManagerRunOnceLimit,
            options.schedulerExecutablePath, std::cout, std::cerr);
    if (options.campaignOperationsProductionEnable)
    {
        const auto build =
            EA::CampaignOperations::CaptureActualManagerBuildContract(
                options.schedulerExecutablePath);
        if (!build)
            throw std::runtime_error(
                "production enable requires a clean Release build identity");
        EA::CampaignOperations::ProductionEnableRequest request{
            *options.campaignOperationsOperationKey,
            *options.campaignOperationsExpectedProductionVersion,
            *options.campaignOperationsIndependentVerificationReference,
            EA::CampaignOperations::ActorIdentity(
                *options.campaignOperationsActor),
            EA::CampaignOperations::Reason(*options.campaignOperationsReason),
            *build,
            true};
        return EA::CampaignOperations::RunProductionEnableCommand(
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_ENABLER_DB_USER"),
            request, std::cout, std::cerr);
    }
    if (options.campaignOperationsProductionDisable)
    {
        EA::CampaignOperations::ProductionDisableRequest request{
            *options.campaignOperationsOperationKey,
            *options.campaignOperationsExpectedProductionVersion,
            EA::CampaignOperations::ActorIdentity(
                *options.campaignOperationsActor),
            EA::CampaignOperations::Reason(*options.campaignOperationsReason),
            true};
        return EA::CampaignOperations::RunProductionDisableCommand(
            CampaignOperationsProductionConnectionString(
                "CAMPAIGN_OPERATIONS_PRODUCTION_DISABLER_DB_USER"),
            request, std::cout, std::cerr);
    }
    if (options.campaignOperationsProductionDispatchRequest)
    {
        const auto build =
            EA::CampaignOperations::CaptureActualManagerBuildContract(
                options.schedulerExecutablePath);
        if (!build)
            throw std::runtime_error(
                "production dispatch requires a clean Release build identity");
        EA::CampaignOperations::ProductionDispatchRequest request{
            EA::CampaignOperations::OperationalRequestId(
                *options.campaignOperationsControlRequestId),
            *options.campaignOperationsExpectedRequestVersion,
            *options.campaignOperationsOperationKey,
            EA::CampaignOperations::ActorIdentity(
                *options.campaignOperationsActor),
            *build,
            true};
        const auto result =
            EA::CampaignOperations::DispatchOneRequestForProduction(
                CampaignOperationsProductionConnectionString(
                    "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER"),
                CampaignOperationsProductionConnectionString(
                    "CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER"),
                request, options.schedulerExecutablePath);
        std::cout << "CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH"
                  << ",request_id=" << result.requestId.value()
                  << ",classification="
                  << EA::CampaignOperations::ToText(result.classification)
                  << ",downstream_evidence="
                  << EA::CampaignOperations::ToText(result.downstreamEvidence)
                  << ",replay_disposition="
                  << EA::CampaignOperations::ToText(result.replayDisposition)
                  << ",recovery="
                  << EA::CampaignOperations::ToText(result.recovery)
                  << ",transaction_attempts=" << result.transactionAttempts
                  << ",binding_set_identity_hash="
                  << (result.bindingSetIdentityHash.empty()
                          ? "none" : result.bindingSetIdentityHash)
                  << ",diagnostic_code=" << result.diagnosticCode << '\n';
        return result.classification ==
                    EA::CampaignOperations::DispatchResultClassification::
                        createdAndBound ||
                result.classification ==
                    EA::CampaignOperations::DispatchResultClassification::
                        adoptedExistingPendingAndBound ||
                result.classification ==
                    EA::CampaignOperations::DispatchResultClassification::
                        existingIdentical ? 0 : 2;
    }

    // Pre-Phase-H commands require a separately reviewed deployment LOGIN for
    // their Phase 2 capability roles. Production-capable or superuser
    // principals are rejected before any Campaign Operations workflow opens.
    const std::string connectionString =
        CampaignOperationsPrePhaseHConnectionString();
    ValidateCampaignOperationsPrePhaseHPrincipal(connectionString);
    if (options.campaignOperationsAdmitMaterializationId)
    {
        EA::CampaignOperations::OperationalCampaignAdmissionRequest request;
        request.materializationId =
            *options.campaignOperationsAdmitMaterializationId;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        return EA::CampaignOperations::RunOperationalCampaignAdmissionCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.campaignOperationsCompletionStatusCampaignId)
        return EA::CampaignOperations::RunCampaignCompletionStatusCommand(
            connectionString,
            EA::CampaignOperations::OperationalCampaignId(
                *options.campaignOperationsCompletionStatusCampaignId),
            std::cout, std::cerr);
    if (options.campaignOperationsCompleteCampaignId)
    {
        EA::CampaignOperations::CompleteIfSettledRequest request;
        request.campaignId =
            *options.campaignOperationsCompleteCampaignId;
        request.operationKey =
            *options.campaignOperationsOperationKey;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        return EA::CampaignOperations::
            RunCompleteCampaignIfSettledCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.campaignOperationsControlStatusCampaignId)
        return EA::CampaignOperations::RunCampaignControlStatusCommand(
            connectionString,
            EA::CampaignOperations::OperationalCampaignId(
                *options.campaignOperationsControlStatusCampaignId),
            std::cout, std::cerr);
    if (options.campaignOperationsReconcileRunKey)
    {
        EA::CampaignOperations::ReconciliationObserveRequest request;
        request.runKey = *options.campaignOperationsReconcileRunKey;
        request.afterRequestId =
            options.campaignOperationsReconcileAfterRequestId.value_or(0);
        request.limit =
            options.campaignOperationsReconcileLimit.value_or(100);
        request.resolveSafeTransitions =
            options.campaignOperationsReconcileRecover;
        return EA::CampaignOperations::RunCampaignReconciliationCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.campaignOperationsPauseCampaignId ||
        options.campaignOperationsResumeCampaignId)
    {
        EA::CampaignOperations::CampaignControlRequest request;
        request.campaignId = options.campaignOperationsPauseCampaignId
            ? *options.campaignOperationsPauseCampaignId
            : *options.campaignOperationsResumeCampaignId;
        request.expectedControlVersion =
            *options.campaignOperationsExpectedControlVersion;
        request.action = options.campaignOperationsPauseCampaignId
            ? EA::CampaignOperations::ControlEventKind::pause
            : EA::CampaignOperations::ControlEventKind::resume;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        return EA::CampaignOperations::RunCampaignControlCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.campaignOperationsCancelCampaignId)
    {
        EA::CampaignOperations::CampaignCancellationCommandRequest request;
        request.campaignId =
            *options.campaignOperationsCancelCampaignId;
        request.requestId =
            options.campaignOperationsControlRequestId;
        request.expectedRequestVersion =
            options.campaignOperationsExpectedRequestVersion;
        request.operationKey =
            *options.campaignOperationsOperationKey;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        return EA::CampaignOperations::RunCampaignCancellationCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.campaignOperationsBudgetStatusCampaignId)
        return EA::CampaignOperations::RunCampaignBudgetStatusCommand(
            connectionString,
            EA::CampaignOperations::OperationalCampaignId(
                *options.campaignOperationsBudgetStatusCampaignId),
            std::cout, std::cerr);
    if (options.campaignOperationsRequestStatusRequestId)
        return EA::CampaignOperations::
            RunCampaignOperationalRequestStatusCommand(
                connectionString,
                EA::CampaignOperations::OperationalRequestId(
                    *options.campaignOperationsRequestStatusRequestId),
                std::cout, std::cerr);
    if (options.campaignOperationsAcceptRequestCampaignId)
    {
        EA::CampaignOperations::OperationalRequestAcceptanceRequest request;
        request.campaignId =
            *options.campaignOperationsAcceptRequestCampaignId;
        request.actorIdentity = *options.campaignOperationsActor;
        request.reason = *options.campaignOperationsReason;
        request.expiresAt =
            options.campaignOperationsReservationExpiresAt;
        return EA::CampaignOperations::
            RunCampaignOperationalRequestAcceptanceCommand(
                connectionString, request, std::cout, std::cerr);
    }
    EA::CampaignOperations::BudgetAdministrationRequest request;
    request.campaignId = options.campaignOperationsBudgetGrantCampaignId
        ? *options.campaignOperationsBudgetGrantCampaignId
        : (options.campaignOperationsBudgetAmendCampaignId
                  ? *options.campaignOperationsBudgetAmendCampaignId
                  : (options.campaignOperationsBudgetRevokeCampaignId
                            ? *options.
                                  campaignOperationsBudgetRevokeCampaignId
                            : *options.
                                  campaignOperationsBudgetSupersedeCampaignId));
    request.expectedLedgerVersion =
        *options.campaignOperationsExpectedBudgetVersion;
    request.kind = options.campaignOperationsBudgetGrantCampaignId
        ? EA::CampaignOperations::BudgetLedgerEntryKind::grant
        : (options.campaignOperationsBudgetAmendCampaignId
                  ? EA::CampaignOperations::BudgetLedgerEntryKind::amend
                  : (options.campaignOperationsBudgetRevokeCampaignId
                            ? EA::CampaignOperations::
                                  BudgetLedgerEntryKind::revoke
                            : EA::CampaignOperations::
                                  BudgetLedgerEntryKind::supersede));
    request.value = options.campaignOperationsBudgetValue;
    request.actorIdentity = *options.campaignOperationsActor;
    request.reason = *options.campaignOperationsReason;
    return EA::CampaignOperations::RunCampaignBudgetAdministrationCommand(
        connectionString, request, std::cout, std::cerr);
}

int RunExperimentRecommendationCommand(const SchedulerOptions& options)
{
    const std::string connectionString = LstmDbConnectionString();
    if (options.generateExperimentRecommendations)
    {
        EA::ExperimentRecommendation::RecommendationGenerationCommandRequest request;
        request.policy = options.recommendationPolicy
            ? EA::ExperimentRecommendation::ParseRecommendationPolicy(
                  *options.recommendationPolicy)
            : EA::ExperimentRecommendation::RecommendationPolicy{};
        request.symbol = options.recommendationSymbol;
        request.predictionHorizon = options.recommendationHorizon;
        request.sourceExperimentId = options.recommendationSourceExperimentId;
        request.requestedMaximum = options.recommendationMaximum;
        return EA::ExperimentRecommendation::RunGenerateExperimentRecommendationsCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.listExperimentRecommendations)
    {
        EA::ExperimentRecommendation::RecommendationListCommandRequest request;
        request.status = options.recommendationStatusFilter;
        request.symbol = options.recommendationSymbol;
        request.predictionHorizon = options.recommendationHorizon;
        request.recommendationScanId = options.recommendationScanId;
        request.limit = options.recommendationLimit;
        return EA::ExperimentRecommendation::RunListExperimentRecommendationsCommand(
            connectionString, request, std::cout);
    }
    if (options.recommendationStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationStatusCommand(
            connectionString, *options.recommendationStatusId, std::cout);
    if (options.listExperimentRecommendationScans)
        return EA::ExperimentRecommendation::RunListExperimentRecommendationScansCommand(
            connectionString, options.recommendationLimit, std::cout);
    if (options.recommendationScanStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationScanStatusCommand(
            connectionString, *options.recommendationScanStatusId, std::cout);
    if (options.scoreExperimentRecommendations)
    {
        EA::ExperimentRecommendation::RecommendationScoringCommandRequest request;
        request.policy = options.recommendationScoringPolicy
            ? EA::ExperimentRecommendation::ParseRecommendationScoringPolicy(
                  *options.recommendationScoringPolicy)
            : EA::ExperimentRecommendation::RecommendationScoringPolicy{};
        request.symbol = options.recommendationSymbol;
        request.predictionHorizon = options.recommendationHorizon;
        request.recommendationScanId = options.recommendationScanId;
        request.recommendationId = options.recommendationIdFilter;
        if (options.recommendationScoreLimitSpecified)
            request.requestedLimit = options.recommendationScoreLimit;
        return EA::ExperimentRecommendation::RunScoreExperimentRecommendationsCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.listExperimentRecommendationScores)
    {
        EA::ExperimentRecommendation::RecommendationScoreListCommandRequest request;
        request.scoreRunId = options.recommendationScoreRunId;
        request.recommendationId = options.recommendationIdFilter;
        request.symbol = options.recommendationSymbol;
        request.predictionHorizon = options.recommendationHorizon;
        request.minimumScore = options.recommendationScoreMinimum;
        request.limit = options.recommendationScoreLimit;
        return EA::ExperimentRecommendation::RunListExperimentRecommendationScoresCommand(
            connectionString, request, std::cout);
    }
    if (options.recommendationScoreStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationScoreStatusCommand(
            connectionString, *options.recommendationScoreStatusId, std::cout);
    if (options.listExperimentRecommendationScoreRuns)
        return EA::ExperimentRecommendation::RunListExperimentRecommendationScoreRunsCommand(
            connectionString, options.recommendationScoreLimit, std::cout);
    if (options.recommendationScoreRunStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationScoreRunStatusCommand(
            connectionString, *options.recommendationScoreRunStatusId, std::cout);
    if (options.approveRecommendationId || options.rejectRecommendationId ||
        options.expireRecommendationId)
    {
        EA::ExperimentRecommendation::RecommendationReviewCommandRequest request;
        if (options.approveRecommendationId)
        {
            request.recommendationId = *options.approveRecommendationId;
            request.review.action =
                EA::ExperimentRecommendation::RecommendationReviewAction::approve;
        }
        else if (options.rejectRecommendationId)
        {
            request.recommendationId = *options.rejectRecommendationId;
            request.review.action =
                EA::ExperimentRecommendation::RecommendationReviewAction::reject;
        }
        else
        {
            request.recommendationId = *options.expireRecommendationId;
            request.review.action =
                EA::ExperimentRecommendation::RecommendationReviewAction::expire;
        }
        request.review.reasonCode =
            options.recommendationReviewReasonCode.value_or("");
        request.review.reasonText = options.recommendationReviewReason;
        request.review.reviewer = options.recommendationReviewer;
        request.review.note = options.recommendationReviewNote;
        request.review.recommendationScoreId =
            options.recommendationReviewScoreId;
        return EA::ExperimentRecommendation::RunExperimentRecommendationReviewCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.listExperimentRecommendationReviews)
    {
        EA::ExperimentRecommendation::RecommendationReviewListCommandRequest request;
        request.recommendationId = options.recommendationIdFilter;
        if (options.recommendationReviewActionFilter)
            request.action =
                *EA::ExperimentRecommendation::ParseRecommendationReviewAction(
                    *options.recommendationReviewActionFilter);
        request.limit = options.recommendationReviewLimit;
        return EA::ExperimentRecommendation::RunListExperimentRecommendationReviewsCommand(
            connectionString, request, std::cout);
    }
    if (options.recommendationReviewStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationReviewStatusCommand(
            connectionString, *options.recommendationReviewStatusId, std::cout);
    if (options.recommendationReviewHistoryId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationReviewHistoryCommand(
            connectionString, *options.recommendationReviewHistoryId, std::cout);
    if (options.approveConversionProposalId ||
        options.rejectConversionProposalId)
    {
        EA::ExperimentRecommendation::
            RecommendationConversionProposalReviewRequest request;
        request.proposalId = options.approveConversionProposalId
            ? *options.approveConversionProposalId
            : *options.rejectConversionProposalId;
        request.decision = options.approveConversionProposalId
            ? EA::ExperimentRecommendation::
                  RecommendationConversionProposalReviewDecision::approve
            : EA::ExperimentRecommendation::
                  RecommendationConversionProposalReviewDecision::reject;
        request.requestId = *options.conversionProposalReviewRequestId;
        request.operatorIdentity = options.conversionProposalReviewOperator;
        request.reasonText = options.conversionProposalReviewReason;
        return EA::ExperimentRecommendation::
            RunRecommendationConversionProposalReviewCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.showConversionProposalId)
        return EA::ExperimentRecommendation::
            RunShowRecommendationConversionProposalCommand(
                connectionString, *options.showConversionProposalId, std::cout);
    if (options.listConversionProposalReviewsId)
        return EA::ExperimentRecommendation::
            RunListRecommendationConversionProposalReviewsCommand(
                connectionString, *options.listConversionProposalReviewsId,
                options.conversionProposalReviewLimit, std::cout);
    if (options.listConversionProposalsReviewStatus)
    {
        EA::ExperimentRecommendation::
            RecommendationConversionProposalReviewListRequest request;
        request.disposition = *EA::ExperimentRecommendation::
            ParseRecommendationConversionProposalReviewDisposition(
                *options.listConversionProposalsReviewStatus);
        request.limit = options.conversionProposalReviewLimit;
        return EA::ExperimentRecommendation::
            RunListRecommendationConversionProposalsByReviewDispositionCommand(
                connectionString, request, std::cout);
    }
    if (options.executeApprovedConversionProposalId)
        return EA::ExperimentRecommendation::
            RunExecuteApprovedRecommendationConversionProposalCommand(
                connectionString, *options.executeApprovedConversionProposalId,
                std::cout, std::cerr);
    if (options.conversionProposalExecutionStatusId)
        return EA::ExperimentRecommendation::
            RunRecommendationConversionExecutionStatusCommand(
                connectionString,
                *options.conversionProposalExecutionStatusId,
                std::cout);
    if (options.activateRecommendationConversionExecutionId)
        return EA::ExperimentRecommendation::
            RunActivateRecommendationConversionExecutionCommand(
                connectionString,
                *options.activateRecommendationConversionExecutionId,
                std::cout,
                std::cerr);
    if (options.recommendationConversionActivationStatusId)
        return EA::ExperimentRecommendation::
            RunRecommendationConversionActivationStatusCommand(
                connectionString,
                *options.recommendationConversionActivationStatusId,
                std::cout);
    if (options.recommendationConversionWorkflowProposalId)
        return EA::ExperimentRecommendation::
            RunRecommendationConversionWorkflowCommand(
                connectionString,
                *options.recommendationConversionWorkflowProposalId,
                std::cout,
                std::cerr);
    if (options.listRecommendationConversionWorkflows)
        return EA::ExperimentRecommendation::
            RunListRecommendationConversionWorkflowsCommand(
                connectionString,
                options.conversionWorkflowState,
                options.conversionWorkflowLimit,
                std::cout,
                std::cerr);
    if (options.planRecommendationCampaign)
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignPlanningCommand(
                connectionString,
                options.campaignPlanningPolicy,
                options.campaignPlanningScope,
                std::cout,
                std::cerr);
    if (options.reviewRecommendationCampaign)
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignReviewCommand(
                connectionString,
                options.campaignPlanningPolicy,
                options.campaignPlanningScope,
                std::cout,
                std::cerr);
    if (options.approveRecommendationCampaign ||
        options.rejectRecommendationCampaign)
    {
        EA::ExperimentRecommendation::RecommendationCampaignApprovalRequest
            request;
        request.decision = options.approveRecommendationCampaign
            ? EA::ExperimentRecommendation::
                  RecommendationCampaignApprovalDecision::approved
            : EA::ExperimentRecommendation::
                  RecommendationCampaignApprovalDecision::rejected;
        request.expectedCampaignReviewIdentityHash =
            *options.campaignReviewIdentityHash;
        request.reviewerIdentity = *options.campaignReviewer;
        request.reasonText = *options.campaignReviewReason;
        return EA::ExperimentRecommendation::
            RunRecordRecommendationCampaignApprovalCommand(
                connectionString,
                options.campaignPlanningPolicy,
                options.campaignPlanningScope,
                request,
                std::cout,
                std::cerr);
    }
    if (options.showRecommendationCampaignApprovalId)
        return EA::ExperimentRecommendation::
            RunShowRecommendationCampaignApprovalCommand(
                connectionString,
                *options.showRecommendationCampaignApprovalId,
                std::cout,
                std::cerr);
    if (options.listRecommendationCampaignApprovals)
        return EA::ExperimentRecommendation::
            RunListRecommendationCampaignApprovalsCommand(
                connectionString,
                options.campaignApprovalDecision,
                options.campaignApprovalLimit,
                std::cout,
                std::cerr);
    if (options.materializeRecommendationCampaign)
    {
        EA::ExperimentRecommendation::RecommendationCampaignMaterializationRequest
            request{*options.campaignMaterializationApprovalId,
                    *options.campaignMaterializedBy,
                    *options.campaignMaterializationReason};
        return EA::ExperimentRecommendation::
            RunMaterializeRecommendationCampaignCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.showRecommendationCampaignMaterializationId)
        return EA::ExperimentRecommendation::
            RunShowRecommendationCampaignMaterializationCommand(
                connectionString,
                *options.showRecommendationCampaignMaterializationId,
                std::cout, std::cerr);
    if (options.listRecommendationCampaignMaterializations)
        return EA::ExperimentRecommendation::
            RunListRecommendationCampaignMaterializationsCommand(
                connectionString, options.campaignMaterializationApprovalId,
                options.campaignMaterializationLimit,
                std::cout, std::cerr);
    if (options.showRecommendationCampaignHandoffId)
        return EA::ExperimentRecommendation::
            RunShowRecommendationCampaignHandoffCommand(
                connectionString,
                *options.showRecommendationCampaignHandoffId,
                std::cout, std::cerr);
    if (options.listRecommendationCampaignHandoffs)
        return EA::ExperimentRecommendation::
            RunListRecommendationCampaignHandoffsCommand(
                connectionString, options.campaignHandoffLimit,
                std::cout, std::cerr);
    if (options.reviewRecommendationCampaignMaterializationId)
    {
        EA::ExperimentRecommendation::
            RecommendationCampaignProposalReviewRequest request;
        request.materializationId =
            *options.reviewRecommendationCampaignMaterializationId;
        request.decision = *options.campaignProposalReviewDecision;
        request.operatorIdentity = *options.campaignProposalReviewOperator;
        request.reasonText = *options.campaignProposalReviewReason;
        request.dryRun = options.dryRun;
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignProposalReviewCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.executeRecommendationCampaignMaterializationId)
    {
        EA::ExperimentRecommendation::RecommendationCampaignExecutionRequest
            request{*options.executeRecommendationCampaignMaterializationId,
                    options.dryRun};
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignExecutionCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.activateRecommendationCampaignMaterializationId)
    {
        EA::ExperimentRecommendation::RecommendationCampaignActivationRequest
            request{*options.activateRecommendationCampaignMaterializationId,
                    options.dryRun};
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignActivationCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.launchRecommendationCampaignMaterializationId)
    {
        EA::ExperimentRecommendation::RecommendationCampaignLaunchRequest
            request{*options.launchRecommendationCampaignMaterializationId,
                    options.dryRun};
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignLaunchCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.recommendationCampaignStatusMaterializationId)
    {
        EA::ExperimentRecommendation::RecommendationCampaignStatusRequest request{
            *options.recommendationCampaignStatusMaterializationId};
        return EA::ExperimentRecommendation::RunRecommendationCampaignStatusCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.recommendationCampaignOutcomeAssessmentMaterializationId)
    {
        EA::ExperimentRecommendation::
            RecommendationCampaignOutcomeAssessmentRequest request{
                *options.
                    recommendationCampaignOutcomeAssessmentMaterializationId};
        return EA::ExperimentRecommendation::
            RunRecommendationCampaignOutcomeAssessmentCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.evaluateExperimentRecommendations ||
        options.evaluateExperimentRecommendationId)
    {
        EA::ExperimentRecommendation::RecommendationEvaluationCommandRequest request;
        if (options.recommendationEvaluationPolicy)
            request.policy.scoringPolicy =
                EA::ExperimentRecommendation::ParseRecommendationScoringPolicy(
                    *options.recommendationEvaluationPolicy);
        request.recommendationScanId = options.recommendationScanId;
        request.recommendationId = options.evaluateExperimentRecommendationId;
        request.limit = options.recommendationEvaluationLimit;
        request.dryRun = options.recommendationEvaluationDryRun;
        return EA::ExperimentRecommendation::RunEvaluateExperimentRecommendationsCommand(
            connectionString, request, std::cout, std::cerr);
    }
    if (options.listExperimentRecommendationEvaluations)
    {
        EA::ExperimentRecommendation::RecommendationEvaluationFilters filters;
        filters.recommendationScanId = options.recommendationScanId;
        filters.recommendationId = options.recommendationIdFilter;
        if (options.recommendationEvaluationDisposition)
            filters.disposition =
                *EA::ExperimentRecommendation::ParseRecommendationEvaluationDisposition(
                    *options.recommendationEvaluationDisposition);
        filters.limit = options.recommendationEvaluationLimit;
        return EA::ExperimentRecommendation::RunListExperimentRecommendationEvaluationsCommand(
            connectionString, filters, std::cout);
    }
    if (options.recommendationEvaluationStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationEvaluationStatusCommand(
            connectionString, *options.recommendationEvaluationStatusId,
            std::cout);
    if (options.explainRecommendationEvaluationId)
        return EA::ExperimentRecommendation::RunExplainExperimentRecommendationEvaluationCommand(
            connectionString, *options.explainRecommendationEvaluationId,
            std::cout);
    if (options.listExperimentRecommendationEvaluationRuns)
        return EA::ExperimentRecommendation::RunListExperimentRecommendationEvaluationRunsCommand(
            connectionString, options.recommendationEvaluationLimit,
            std::cout);
    if (options.recommendationEvaluationRunStatusId)
        return EA::ExperimentRecommendation::RunExperimentRecommendationEvaluationRunStatusCommand(
            connectionString, *options.recommendationEvaluationRunStatusId,
            std::cout);
    if (options.rankExperimentRecommendationEvaluations)
    {
        EA::ExperimentRecommendation::RecommendationRankingCommandRequest request;
        request.limit = options.recommendationRankingLimit;
        request.dryRun = options.recommendationRankingDryRun;
        if (options.recommendationRankingEvaluationRunId)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::evaluationRun;
            request.scope.evaluationRunId =
                options.recommendationRankingEvaluationRunId;
        }
        else if (options.recommendationRankingScanId)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::recommendationScan;
            request.scope.recommendationScanId =
                options.recommendationRankingScanId;
        }
        else if (options.recommendationRankingSymbol &&
                 options.recommendationRankingHorizon)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::symbolHorizon;
            request.scope.symbol = options.recommendationRankingSymbol;
            request.scope.horizon = options.recommendationRankingHorizon;
        }
        else if (options.recommendationRankingSymbol)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::symbol;
            request.scope.symbol = options.recommendationRankingSymbol;
        }
        else if (options.recommendationRankingHorizon)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::horizon;
            request.scope.horizon = options.recommendationRankingHorizon;
        }
        else if (options.recommendationRankingFamily)
        {
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::family;
            request.scope.family = options.recommendationRankingFamily;
        }
        else
            request.scope.type = EA::ExperimentRecommendation::
                RecommendationRankingScopeType::global;
        return EA::ExperimentRecommendation::
            RunRankExperimentRecommendationEvaluationsCommand(
                connectionString, request, std::cout, std::cerr);
    }
    if (options.listExperimentRecommendationRankingSnapshots)
        return EA::ExperimentRecommendation::
            RunListExperimentRecommendationRankingSnapshotsCommand(
                connectionString, options.recommendationRankingLimit, std::cout);
    if (options.recommendationRankingStatusId)
        return EA::ExperimentRecommendation::
            RunExperimentRecommendationRankingStatusCommand(
                connectionString, *options.recommendationRankingStatusId,
                std::cout);
    if (options.listRecommendationRankingMembersId)
    {
        std::optional<EA::ExperimentRecommendation::RecommendationRankingBucket>
            bucket;
        if (options.recommendationRankingBucket)
            bucket = *EA::ExperimentRecommendation::
                ParseRecommendationRankingBucket(
                    *options.recommendationRankingBucket);
        return EA::ExperimentRecommendation::
            RunListExperimentRecommendationRankingMembersCommand(
                connectionString, *options.listRecommendationRankingMembersId,
                bucket, options.recommendationRankingLimit, std::cout);
    }
    if (options.recommendationRankingMemberStatusId)
        return EA::ExperimentRecommendation::
            RunExperimentRecommendationRankingMemberStatusCommand(
                connectionString,
                *options.recommendationRankingMemberStatusId, std::cout);
    if (options.compareRecommendationEvaluations)
        return EA::ExperimentRecommendation::
            RunCompareExperimentRecommendationEvaluationsCommand(
                connectionString, *options.compareRecommendationEvaluations,
                std::cout);
    if (options.compareRecommendationRankingMembers)
        return EA::ExperimentRecommendation::
            RunCompareExperimentRecommendationRankingMembersCommand(
                connectionString,
                *options.compareRecommendationRankingMembers, std::cout);
    return EA::ExperimentRecommendation::RunExplainExperimentRecommendationScoreCommand(
        connectionString, *options.explainRecommendationScoreId, std::cout);
}

} // namespace

bool IsExperimentSchedulerCommand(int argc, const char* argv[])
{
    return IsExperimentSchedulerCommandImpl(argc, argv);
}


int RunExperimentSchedulerCli(int argc, const char* argv[])
{
    if (EA::SchedulerCore::IsSchedulerDaemonCommand(argc, argv))
        return EA::SchedulerCore::RunSchedulerDaemonCli(argc, argv);

    SchedulerOptions options;
    try
    {
        options = ParseSchedulerArgs(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        PrintExperimentSchedulerHelp(argc > 0 ? argv[0] : "LSTM_Release");
        return 1;
    }

    try
    {
        if (options.schedulerWorkerAttemptId &&
            !EA::SchedulerCore::RegisterSchedulerWorker({
                *options.schedulerWorkerAttemptId,
                options.analyzeExperimentId,
                std::nullopt,
                "experiment",
                "analyze"}))
        {
            return 125;
        }
        if (options.help)
        {
            PrintExperimentSchedulerHelp(argc > 0 ? argv[0] : "LSTM_Release");
            return 0;
        }
        if (options.verifyProfitabilityExperimentIds)
        {
            try
            {
                return EA::ProfitabilityVerification::RunVerificationCommand(
                    LstmDbConnectionString(),
                    *options.verifyProfitabilityExperimentIds,
                    std::cout,
                    std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr << "PROFITABILITY_VERIFICATION_TOOL_ERROR"
                          << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityReadinessSnapshotId)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignReadinessCommand(
                        LstmDbConnectionString(),
                        *options.campaignProfitabilityReadinessSnapshotId,
                        std::cout,
                        std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr << "CAMPAIGN_PROFITABILITY_READINESS_TOOL_ERROR"
                          << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityShadowSnapshotId)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignShadowRankingCommand(
                        LstmDbConnectionString(),
                        *options.campaignProfitabilityShadowSnapshotId,
                        *options.campaignProfitabilityShadowWeights,
                        std::cout,
                        std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr << "CAMPAIGN_PROFITABILITY_SHADOW_TOOL_ERROR"
                          << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityCalibrationSnapshotId)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignProfitabilityCalibrationCommand(
                        LstmDbConnectionString(),
                        *options.campaignProfitabilityCalibrationSnapshotId,
                        std::cout,
                        std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "CAMPAIGN_PROFITABILITY_CALIBRATION_TOOL_ERROR"
                    << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityTemporalValidation)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignProfitabilityTemporalValidationCommand(
                        LstmDbConnectionString(), std::cout, std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "CAMPAIGN_PROFITABILITY_TEMPORAL_VALIDATION_TOOL_ERROR"
                    << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityForwardValidationPrecommit)
        {
            try
            {
                const auto& [snapshotId, outcomeStart, outcomeEnd] =
                    *options.campaignProfitabilityForwardValidationPrecommit;
                return EA::ProfitabilityVerification::
                    RunCampaignProfitabilityForwardValidationPrecommitCommand(
                        LstmDbConnectionString(), snapshotId, outcomeStart,
                        outcomeEnd, std::cout, std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "CAMPAIGN_PROFITABILITY_FORWARD_VALIDATION_TOOL_ERROR"
                    << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityOutcomePreparationCohort)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignProfitabilityOutcomePreparationCommand(
                        LstmDbConnectionString(),
                        *options.campaignProfitabilityOutcomePreparationCohort,
                        std::cout, std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "CAMPAIGN_PROFITABILITY_OUTCOME_PREPARATION_TOOL_ERROR"
                    << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.campaignProfitabilityProspectiveComparisonCohort)
        {
            try
            {
                return EA::ProfitabilityVerification::
                    RunCampaignProfitabilityProspectiveComparisonCommand(
                        LstmDbConnectionString(),
                        *options.campaignProfitabilityProspectiveComparisonCohort,
                        std::cout, std::cerr);
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "CAMPAIGN_PROFITABILITY_PROSPECTIVE_COMPARISON_TOOL_ERROR"
                    << ",error=" << error.what() << std::endl;
                return 2;
            }
        }
        if (options.compareExperimentPair)
        {
            EA::ExperimentPairComparison::ComparisonCommand command;
            command.experimentIds = *options.compareExperimentPair;
            command.summary = options.compareExperimentPairSummary;
            return EA::ExperimentPairComparison::RunComparisonCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.compareExperimentReplications)
        {
            EA::ExperimentReplicationComparison::ComparisonCommand command;
            command.experimentPairs =
                *options.compareExperimentReplications;
            return EA::ExperimentReplicationComparison::RunComparisonCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.planExperimentReplications)
        {
            EA::ExperimentReplicationPlanning::PlanningCommand command;
            command.sourceExperimentIds =
                *options.planExperimentReplications;
            command.requestedSeeds = *options.replicationSeeds;
            return EA::ExperimentReplicationPlanning::RunPlanningCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.compareFeatureAblationPair)
        {
            EA::FeatureAblationPairEvaluation::ComparisonCommand command;
            command.experimentIds = *options.compareFeatureAblationPair;
            command.expectedAblationMask =
                options.expectedFeatureAblationMask;
            return EA::FeatureAblationPairEvaluation::RunComparisonCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.causalSurpriseObservabilityExperimentId)
        {
            return EA::CausalSurpriseObservability::RunCommand(
                LstmDbConnectionString(), ForexDbConnectionString(),
                *options.causalSurpriseObservabilityExperimentId,
                options.causalSurpriseObservabilityScope,
                std::cout, std::cerr);
        }
        if (options.causalSurpriseCoverageGapsExperimentId)
        {
            return EA::CausalSurpriseObservability::
                RunGapAttributionCommand(
                    LstmDbConnectionString(), ForexDbConnectionString(),
                    *options.causalSurpriseCoverageGapsExperimentId,
                    options.causalSurpriseCoverageGapsScope,
                    std::cout, std::cerr);
        }
        if (options.compareFeatureAblationReplications)
        {
            EA::FeatureAblationReplicationEvaluation::ComparisonCommand command;
            command.experimentIdPairs =
                *options.compareFeatureAblationReplications;
            command.expectedAblationMask =
                options.expectedFeatureAblationMask;
            return EA::FeatureAblationReplicationEvaluation::
                RunComparisonCommand(
                    LstmDbConnectionString(), command,
                    std::cout, std::cerr);
        }
        if (options.correctedCausalSurpriseReplicationStatus)
            return EA::CorrectedCausalSurpriseReplicationContinuation::
                RunStatusCommand(
                    LstmDbConnectionString(),
                    CorrectedReplicationCommand(options, false),
                    std::cout, std::cerr);
        if (options.materializeCorrectedCausalSurpriseReplication)
            return RunCorrectedReplicationMaterializationCommand(options);
        if (options.compareTrainingObjectivePair)
        {
            EA::PairedTrainingObjectiveEvaluation::ComparisonCommand command;
            command.experimentIds = *options.compareTrainingObjectivePair;
            command.policy.primaryProfitabilityMetric =
                *options.pairPrimaryProfitabilityMetric == "aggregate"
                    ? EA::PairedTrainingObjectiveEvaluation::
                          ProfitabilityPrimaryMetric::
                              AggregateTerminalHorizonLogReturnSum
                    : EA::PairedTrainingObjectiveEvaluation::
                          ProfitabilityPrimaryMetric::
                              AverageTerminalHorizonLogReturnPerActionablePrediction;
            command.policy.minimumProfitabilityImprovement =
                *options.pairMinimumProfitabilityImprovement;
            command.policy.maximumProfitabilityWorsening =
                *options.pairMaximumProfitabilityWorsening;
            command.policy.classification =
                EA::PairedTrainingObjectiveEvaluation::
                    ClassificationDegradationPolicy{
                        *options.pairMaximumInferenceAccuracyDecrease,
                        *options.pairMaximumAcceptAccuracyDecrease,
                        *options.pairMaximumAcceptRateDecrease,
                        *options.pairMaximumLeaderScoreDecrease,
                        *options.pairMaximumNeutralProportionIncrease};
            return EA::PairedTrainingObjectiveEvaluation::
                RunComparisonCommand(
                    LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.modelInfo)
            return PrintModelInfo(*options.modelInfoModelId);
        if (options.createEconomicCalendarSnapshot)
            return CreateEconomicCalendarSnapshotCommand(options);
        if (options.compactStatus)
            return PrintCompactExperimentStatus(options);
        if (options.queueExperiment || options.queueSweep)
            return QueueExperiments(options);
        if (options.enqueueExperiment)
            return EnqueueExperiment(options);
        if (options.pauseAllExperiments ||
            options.resumeAllExperiments ||
            options.cancelAllExperiments)
        {
            EA::GlobalExperimentControl::Command command;
            command.action = options.pauseAllExperiments
                ? EA::GlobalExperimentControl::Action::PauseAll
                : (options.resumeAllExperiments
                       ? EA::GlobalExperimentControl::Action::ResumeAll
                       : EA::GlobalExperimentControl::Action::CancelAll);
            if (options.cancelImmediate)
                command.cancellationMode =
                    EA::GlobalExperimentControl::CancellationMode::Immediate;
            else if (options.cancelAfterNextCheckpoint)
                command.cancellationMode =
                    EA::GlobalExperimentControl::CancellationMode::
                        AfterNextCheckpoint;
            command.inferBeforeCancel = options.inferBeforeCancel;
            command.dryRun = options.dryRun;
            command.confirmed = options.yes;
            const auto invocationStarted =
                std::chrono::system_clock::now().time_since_epoch();
            command.invocationIdentity =
                std::string{"pid:"} + std::to_string(::getpid()) +
                ";started_ns:" +
                std::to_string(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        invocationStarted)
                        .count()) +
                ";executable:" + options.schedulerExecutablePath;
            return EA::GlobalExperimentControl::RunCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.scheduleExperiments)
        {
            throw std::logic_error(
                "scheduler daemon must use the typed compatibility adapter");
        }
        if (options.completeSchedulerProtocolCutover)
            return CompleteSchedulerProtocolCutover(options);
        if (options.stopExperimentId.has_value() || options.stopAllExperiments)
            return RunStopExperimentCommand(options);
        if (options.reconcileWorkerAttemptId.has_value())
        {
            EA::GlobalExperimentControl::WorkerAttemptReconciliationCommand command;
            command.workerAttemptId = *options.reconcileWorkerAttemptId;
            command.dryRun = options.dryRun;
            command.confirmed = options.yes;
            return EA::GlobalExperimentControl::RunWorkerAttemptReconciliationCommand(
                LstmDbConnectionString(), command, std::cout, std::cerr);
        }
        if (options.recoverFailedInferenceExperimentId.has_value())
        {
            EA::GlobalExperimentControl::
                HistoricalFailedInferenceRecoveryCommand command;
            command.experimentId =
                *options.recoverFailedInferenceExperimentId;
            command.dryRun = options.dryRun;
            command.confirmed = options.yes;
            return EA::GlobalExperimentControl::
                RunHistoricalFailedInferenceRecoveryCommand(
                    LstmDbConnectionString(), command,
                    std::cout, std::cerr);
        }
        if (options.retryCheckpointEvalId.has_value())
            return RunRetryCheckpointEvalCommand(options);
        if (options.evaluateCheckpointPolicyEvalId.has_value())
            return RunEvaluateCheckpointPolicyCommand(options);
        if (options.checkpointPolicyStatusEvalId.has_value())
            return RunCheckpointPolicyStatusCommand(options);
        if (HasContinuationControlCommand(options))
            return RunContinuationPolicyControlCommand(options);
        if (options.evaluateContinuationExperimentId.has_value())
            return RunEvaluateContinuationCommand(options);
        if (options.queueContinuationExperimentId.has_value())
            return RunQueueContinuationCommand(options);
        if (options.continuationStatusExperimentId.has_value())
            return RunContinuationStatusCommand(options);
        if (options.campaignOperationsBudgetGrantCampaignId ||
            options.campaignOperationsBudgetAmendCampaignId ||
            options.campaignOperationsBudgetRevokeCampaignId ||
            options.campaignOperationsBudgetSupersedeCampaignId ||
            options.campaignOperationsAdmitMaterializationId ||
            options.campaignOperationsAcceptRequestCampaignId ||
            options.campaignOperationsBudgetStatusCampaignId ||
            options.campaignOperationsRequestStatusRequestId ||
            options.campaignOperationsPauseCampaignId ||
            options.campaignOperationsResumeCampaignId ||
            options.campaignOperationsCancelCampaignId ||
            options.campaignOperationsControlStatusCampaignId ||
            options.campaignOperationsCompleteCampaignId ||
            options.campaignOperationsCompletionStatusCampaignId ||
            options.campaignOperationsProductionReadiness ||
            options.campaignOperationsProductionStatus ||
            options.campaignOperationsProductionEnable ||
            options.campaignOperationsProductionDisable ||
            options.campaignOperationsProductionDispatchRequest ||
            options.campaignOperationsManagerRunOnceLimit ||
            options.campaignOperationsReconcileRunKey)
            return RunCampaignOperationsCommand(options);
        if (options.generateExperimentRecommendations ||
            options.listExperimentRecommendations ||
            options.recommendationStatusId.has_value() ||
            options.listExperimentRecommendationScans ||
            options.recommendationScanStatusId.has_value() ||
            options.scoreExperimentRecommendations ||
            options.listExperimentRecommendationScores ||
            options.recommendationScoreStatusId.has_value() ||
            options.listExperimentRecommendationScoreRuns ||
            options.recommendationScoreRunStatusId.has_value() ||
            options.explainRecommendationScoreId.has_value() ||
            options.approveRecommendationId.has_value() ||
            options.rejectRecommendationId.has_value() ||
            options.expireRecommendationId.has_value() ||
            options.listExperimentRecommendationReviews ||
            options.recommendationReviewStatusId.has_value() ||
            options.recommendationReviewHistoryId.has_value() ||
            options.evaluateExperimentRecommendations ||
            options.evaluateExperimentRecommendationId.has_value() ||
            options.listExperimentRecommendationEvaluations ||
            options.recommendationEvaluationStatusId.has_value() ||
            options.explainRecommendationEvaluationId.has_value() ||
            options.listExperimentRecommendationEvaluationRuns ||
            options.recommendationEvaluationRunStatusId.has_value() ||
            options.rankExperimentRecommendationEvaluations ||
            options.listExperimentRecommendationRankingSnapshots ||
            options.recommendationRankingStatusId.has_value() ||
            options.listRecommendationRankingMembersId.has_value() ||
            options.recommendationRankingMemberStatusId.has_value() ||
            options.compareRecommendationEvaluations.has_value() ||
            options.compareRecommendationRankingMembers.has_value() ||
            options.approveConversionProposalId.has_value() ||
            options.rejectConversionProposalId.has_value() ||
            options.showConversionProposalId.has_value() ||
            options.listConversionProposalReviewsId.has_value() ||
            options.listConversionProposalsReviewStatus.has_value() ||
            options.executeApprovedConversionProposalId.has_value() ||
            options.conversionProposalExecutionStatusId.has_value() ||
            options.activateRecommendationConversionExecutionId.has_value() ||
            options.recommendationConversionActivationStatusId.has_value() ||
            options.recommendationConversionWorkflowProposalId.has_value() ||
            options.listRecommendationConversionWorkflows ||
            options.planRecommendationCampaign ||
            options.reviewRecommendationCampaign ||
            options.approveRecommendationCampaign ||
            options.rejectRecommendationCampaign ||
            options.showRecommendationCampaignApprovalId.has_value() ||
            options.listRecommendationCampaignApprovals ||
            options.materializeRecommendationCampaign ||
            options.showRecommendationCampaignMaterializationId.has_value() ||
            options.listRecommendationCampaignMaterializations ||
            options.showRecommendationCampaignHandoffId.has_value() ||
            options.listRecommendationCampaignHandoffs ||
            options.reviewRecommendationCampaignMaterializationId.has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (options.executeRecommendationCampaignMaterializationId.has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (options.activateRecommendationCampaignMaterializationId.has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (options.launchRecommendationCampaignMaterializationId.has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (options.recommendationCampaignStatusMaterializationId.has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (options.recommendationCampaignOutcomeAssessmentMaterializationId
                .has_value())
            // Handled by the standalone recommendation/campaign dispatcher.
            return RunExperimentRecommendationCommand(options);
        if (HasCheckpointControlCommand(options))
            return RunCheckpointControlCommand(options);
        if (options.schedulerStatus)
            return PrintSchedulerStatus(options);
        if (options.backfillExperimentMetadata)
            return BackfillExperimentMetadata(options);
        if (options.generateExperimentReports)
            return GenerateExperimentReports(options.experimentReportDir, false);
        if (options.backupDatabase)
            return BackupDatabase(options);
        if (options.experimentMetadataId.has_value())
            return PrintExperimentMetadata(*options.experimentMetadataId);
        if (options.listExperimentModelsId.has_value())
            return ListExperimentModels(*options.listExperimentModelsId);
        if (options.listExperimentLineageId.has_value())
            return ListExperimentLineage(*options.listExperimentLineageId, options.includeParentModels);
        if (options.setExperimentPriority.has_value())
            return RunSetExperimentPriorityCommand(options);
        if (options.pauseCampaignMaterializationId.has_value() ||
            options.resumeCampaignMaterializationId.has_value())
            return RunCampaignMaterializationSchedulerControlCommand(options);
        if (options.pauseExperimentId.has_value() ||
            options.resumeExperimentId.has_value() ||
            options.cancelExperimentId.has_value() ||
            options.retryFailedExperimentId.has_value() ||
            options.requeueTrainingExperimentId.has_value() ||
            options.requeueAnalysisExperimentId.has_value() ||
            options.requeueInferenceExperimentId.has_value())
            return RunSchedulerControlCommand(options);
        if (options.analyzeExperimentId.has_value())
        {
            if (!options.schedulerWorkerAttemptId)
            {
                std::cerr
                    << "DIRECT_CLI_MANAGED_WORK_REJECTED"
                    << ",operation=analyze"
                    << ",experiment_id="
                    << *options.analyzeExperimentId
                    << ",reason=exact_worker_attempt_required"
                    << std::endl;
                return 1;
            }
            return AnalyzeExperimentById(*options.analyzeExperimentId, options);
        }
        if (options.analyzeCompletedExperiments)
        {
            std::cerr
                << "DIRECT_CLI_MANAGED_WORK_REJECTED"
                << ",operation=bulk_analyze"
                << ",reason=scheduler_owned_analysis_only"
                << std::endl;
            return 1;
        }
        if (options.printLeaderboard)
            return PrintLeaderboard(options);
    }
    catch (const pqxx::failure& e)
    {
        std::cerr << "EXPERIMENT_DATABASE_ERROR"
                  << ",error=" << e.what()
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXPERIMENT_FAILED"
                  << ",error=" << e.what()
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
    catch (...)
    {
        std::cerr << "EXPERIMENT_FAILED,error=unknown_exception" << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }

    return 1;
}

int RunStandaloneAnalyzeWorkerCli(int argc, const char* argv[])
{
    SchedulerOptions options;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg{argv[i]};
            std::string value;
            if (arg == "--analyze-experiment")
            {
                if (options.analyzeExperimentId)
                    throw std::invalid_argument(
                        "--analyze-experiment specified more than once");
                options.analyzeExperimentId = ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
            }
            else if (arg == "--scheduler-worker-attempt-id")
            {
                if (options.schedulerWorkerAttemptId)
                    throw std::invalid_argument(
                        "--scheduler-worker-attempt-id specified more than once");
                options.schedulerWorkerAttemptId = ParsePositiveLongLong(
                    arg, RequireNextArg(argc, argv, i, arg));
            }
            else if (arg == "--auto-generate-reports")
            {
                options.autoGenerateReports = true;
            }
            else if (arg == "--experiment-report-dir")
            {
                options.experimentReportDir =
                    RequireNextArg(argc, argv, i, arg);
            }
            else if (SplitOptionWithValue(arg, "--analyze-experiment", value))
            {
                if (options.analyzeExperimentId)
                    throw std::invalid_argument(
                        "--analyze-experiment specified more than once");
                options.analyzeExperimentId = ParsePositiveLongLong(
                    "--analyze-experiment", value);
            }
            else if (SplitOptionWithValue(
                         arg, "--scheduler-worker-attempt-id", value))
            {
                if (options.schedulerWorkerAttemptId)
                    throw std::invalid_argument(
                        "--scheduler-worker-attempt-id specified more than once");
                options.schedulerWorkerAttemptId = ParsePositiveLongLong(
                    "--scheduler-worker-attempt-id", value);
            }
            else if (SplitOptionWithValue(
                         arg, "--experiment-report-dir", value))
            {
                options.experimentReportDir = value;
            }
            else
            {
                throw std::invalid_argument(
                    "unsupported analyze worker option '" + arg + "'");
            }
        }

        if (!options.analyzeExperimentId)
            throw std::invalid_argument("--analyze-experiment is required");
        if (!options.schedulerWorkerAttemptId)
        {
            std::cerr << "DIRECT_CLI_MANAGED_WORK_REJECTED"
                      << ",operation=analyze"
                      << ",experiment_id=" << *options.analyzeExperimentId
                      << ",reason=exact_worker_attempt_required"
                      << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "lstm-analyze-worker")
                  << " --analyze-experiment=EXPERIMENT_ID"
                  << " --scheduler-worker-attempt-id=ATTEMPT_ID"
                  << " [--auto-generate-reports]"
                  << " [--experiment-report-dir=PATH]\n";
        return 1;
    }

    if (!EA::SchedulerCore::RegisterSchedulerWorker({
            *options.schedulerWorkerAttemptId,
            options.analyzeExperimentId,
            std::nullopt,
            "experiment",
            "analyze"}))
    {
        return 125;
    }
    return AnalyzeExperimentById(*options.analyzeExperimentId, options);
}

} // namespace EA::ExperimentScheduler
