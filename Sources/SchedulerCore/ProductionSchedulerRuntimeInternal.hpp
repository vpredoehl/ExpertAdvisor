#pragma once

namespace EA::SchedulerCore::ProductionRuntimeDetail
{

using namespace EA::ExperimentScheduler;

EA::SchedulerCore::SchedulerOwnerProcessEvidence InspectSchedulerOwnerProcess(int, int, const std::string&, const std::string&, EA::GlobalExperimentControl::ProcessObservation* = nullptr);

struct SchedulerOptions
{
    bool modelInfo = false;
    bool compactStatus = false;
    bool createEconomicCalendarSnapshot = false;
    bool scheduleExperiments = false;
    bool enqueueExperiment = false;
    bool queueExperiment = false;
    bool queueSweep = false;
    bool autoResume = false;
    bool analyzeCompletedExperiments = false;
    std::optional<long long> analyzeExperimentId;
    bool printLeaderboard = false;
    bool generateExperimentReports = false;
    bool autoGenerateReports = false;
    bool schedulerStatus = false;
    bool completeSchedulerProtocolCutover = false;
    bool backfillExperimentMetadata = false;
    bool backupDatabase = false;
    std::optional<std::string> backupOutputPath;
    std::optional<long long> experimentMetadataId;
    std::optional<long long> listExperimentModelsId;
    std::optional<long long> listExperimentLineageId;
    bool includeParentModels = false;
    std::optional<long long> modelInfoModelId;
    std::optional<long long> statusExperimentId;
    std::optional<long long> stopExperimentId;
    std::optional<long long> reconcileWorkerAttemptId;
    std::optional<long long> recoverFailedInferenceExperimentId;
    bool stopAllExperiments = false;
    bool pauseAllExperiments = false;
    bool resumeAllExperiments = false;
    bool cancelAllExperiments = false;
    bool cancelImmediate = false;
    bool cancelAfterNextCheckpoint = false;
    bool inferBeforeCancel = false;
    std::optional<long long> pauseExperimentId;
    std::optional<long long> resumeExperimentId;
    std::optional<long long> pauseCampaignMaterializationId;
    std::optional<long long> resumeCampaignMaterializationId;
    std::optional<std::pair<long long, std::string>> setExperimentPriority;
    std::optional<long long> cancelExperimentId;
    std::optional<long long> retryFailedExperimentId;
    std::optional<long long> requeueTrainingExperimentId;
    std::optional<long long> retryCheckpointEvalId;
    std::optional<long long> evaluateCheckpointPolicyEvalId;
    std::optional<long long> checkpointPolicyStatusEvalId;
    std::optional<long long> enableContinuationPolicyExperimentId;
    std::optional<long long> disableContinuationPolicyExperimentId;
    std::optional<std::pair<long long, std::string>> setContinuationPolicy;
    std::optional<long long> evaluateContinuationExperimentId;
    std::optional<long long> queueContinuationExperimentId;
    std::optional<long long> continuationStatusExperimentId;
    bool generateExperimentRecommendations = false;
    bool listExperimentRecommendations = false;
    std::optional<long long> recommendationStatusId;
    bool listExperimentRecommendationScans = false;
    std::optional<long long> recommendationScanStatusId;
    std::optional<std::string> recommendationPolicy;
    std::optional<std::string> recommendationSymbol;
    std::optional<int> recommendationHorizon;
    std::optional<long long> recommendationSourceExperimentId;
    std::optional<int> recommendationMaximum;
    std::optional<std::string> recommendationStatusFilter;
    std::optional<long long> recommendationScanId;
    int recommendationLimit = 100;
    bool recommendationLimitSpecified = false;
    bool scoreExperimentRecommendations = false;
    bool listExperimentRecommendationScores = false;
    std::optional<long long> recommendationScoreStatusId;
    bool listExperimentRecommendationScoreRuns = false;
    std::optional<long long> recommendationScoreRunStatusId;
    std::optional<long long> explainRecommendationScoreId;
    std::optional<std::string> recommendationScoringPolicy;
    std::optional<long long> recommendationIdFilter;
    std::optional<long long> recommendationScoreRunId;
    std::optional<double> recommendationScoreMinimum;
    int recommendationScoreLimit = 100;
    bool recommendationScoreLimitSpecified = false;
    std::optional<long long> approveRecommendationId;
    std::optional<long long> rejectRecommendationId;
    std::optional<long long> expireRecommendationId;
    bool listExperimentRecommendationReviews = false;
    std::optional<long long> recommendationReviewStatusId;
    std::optional<long long> recommendationReviewHistoryId;
    std::optional<std::string> recommendationReviewReasonCode;
    std::optional<std::string> recommendationReviewReason;
    std::optional<std::string> recommendationReviewer;
    std::optional<std::string> recommendationReviewNote;
    std::optional<long long> recommendationReviewScoreId;
    std::optional<std::string> recommendationReviewActionFilter;
    int recommendationReviewLimit = 100;
    bool recommendationReviewLimitSpecified = false;
    bool evaluateExperimentRecommendations = false;
    std::optional<long long> evaluateExperimentRecommendationId;
    bool listExperimentRecommendationEvaluations = false;
    std::optional<long long> recommendationEvaluationStatusId;
    std::optional<long long> explainRecommendationEvaluationId;
    bool listExperimentRecommendationEvaluationRuns = false;
    std::optional<long long> recommendationEvaluationRunStatusId;
    std::optional<std::string> recommendationEvaluationPolicy;
    std::optional<std::string> recommendationEvaluationDisposition;
    int recommendationEvaluationLimit = 100;
    bool recommendationEvaluationLimitSpecified = false;
    bool recommendationEvaluationDryRun = false;
    bool rankExperimentRecommendationEvaluations = false;
    std::optional<long long> recommendationRankingEvaluationRunId;
    std::optional<long long> recommendationRankingScanId;
    std::optional<std::string> recommendationRankingSymbol;
    std::optional<int> recommendationRankingHorizon;
    std::optional<std::string> recommendationRankingFamily;
    bool recommendationRankingGlobal = false;
    int recommendationRankingLimit = 100;
    bool recommendationRankingLimitSpecified = false;
    bool recommendationRankingDryRun = false;
    bool listExperimentRecommendationRankingSnapshots = false;
    std::optional<long long> recommendationRankingStatusId;
    std::optional<long long> listRecommendationRankingMembersId;
    std::optional<long long> recommendationRankingMemberStatusId;
    std::optional<std::string> recommendationRankingBucket;
    std::optional<std::pair<long long, long long>> compareRecommendationEvaluations;
    std::optional<std::pair<long long, long long>> compareRecommendationRankingMembers;
    std::optional<std::pair<long long, long long>> compareTrainingObjectivePair;
    std::optional<std::pair<long long, long long>> compareFeatureAblationPair;
    std::optional<std::pair<long long, long long>> compareExperimentPair;
    bool compareExperimentPairSummary = false;
    std::optional<std::vector<std::pair<long long, long long>>>
        compareExperimentReplications;
    std::optional<std::pair<long long, long long>>
        planExperimentReplications;
    std::optional<std::pair<long long, long long>>
        materializeExperimentReplications;
    std::optional<std::vector<unsigned int>> replicationSeeds;
    std::optional<std::string> expectedFeatureAblationMask;
    std::optional<std::vector<std::pair<long long, long long>>>
        compareFeatureAblationReplications;
    std::optional<std::vector<std::pair<long long, long long>>>
        correctedCausalSurpriseReplicationStatus;
    std::optional<std::vector<std::pair<long long, long long>>>
        materializeCorrectedCausalSurpriseReplication;
    std::optional<std::pair<long long, long long>>
        correctedCausalSurpriseAnchorPair;
    std::optional<std::string> expectedCorrectedReplicationPlanHash;
    std::optional<long long> causalSurpriseObservabilityExperimentId;
    EA::CausalSurpriseObservability::Scope causalSurpriseObservabilityScope =
        EA::CausalSurpriseObservability::Scope::combined;
    bool causalSurpriseObservabilityScopeSpecified = false;
    std::optional<long long> causalSurpriseCoverageGapsExperimentId;
    EA::CausalSurpriseObservability::Scope causalSurpriseCoverageGapsScope =
        EA::CausalSurpriseObservability::Scope::combined;
    bool causalSurpriseCoverageGapsScopeSpecified = false;
    std::optional<std::vector<long long>> verifyProfitabilityExperimentIds;
    std::optional<long long> campaignProfitabilityReadinessSnapshotId;
    std::optional<long long> campaignProfitabilityShadowSnapshotId;
    std::optional<std::vector<double>> campaignProfitabilityShadowWeights;
    std::optional<long long> campaignProfitabilityCalibrationSnapshotId;
    bool campaignProfitabilityTemporalValidation = false;
    std::optional<std::tuple<long long, std::string, std::string>>
        campaignProfitabilityForwardValidationPrecommit;
    std::optional<std::string> campaignProfitabilityOutcomePreparationCohort;
    std::optional<std::string> campaignProfitabilityProspectiveComparisonCohort;
    std::optional<std::string> pairPrimaryProfitabilityMetric;
    std::optional<double> pairMinimumProfitabilityImprovement;
    std::optional<double> pairMaximumProfitabilityWorsening;
    std::optional<double> pairMaximumInferenceAccuracyDecrease;
    std::optional<double> pairMaximumAcceptAccuracyDecrease;
    std::optional<double> pairMaximumAcceptRateDecrease;
    std::optional<double> pairMaximumLeaderScoreDecrease;
    std::optional<double> pairMaximumNeutralProportionIncrease;
    std::optional<long long> approveConversionProposalId;
    std::optional<long long> rejectConversionProposalId;
    std::optional<long long> showConversionProposalId;
    std::optional<long long> listConversionProposalReviewsId;
    std::optional<std::string> listConversionProposalsReviewStatus;
    std::optional<std::string> conversionProposalReviewRequestId;
    std::optional<std::string> conversionProposalReviewOperator;
    std::optional<std::string> conversionProposalReviewReason;
    int conversionProposalReviewLimit = 100;
    bool conversionProposalReviewLimitSpecified = false;
    std::optional<long long> executeApprovedConversionProposalId;
    std::optional<long long> conversionProposalExecutionStatusId;
    std::optional<long long> activateRecommendationConversionExecutionId;
    std::optional<long long> recommendationConversionActivationStatusId;
    std::optional<long long> recommendationConversionWorkflowProposalId;
    bool listRecommendationConversionWorkflows = false;
    std::optional<EA::ExperimentRecommendation::
        RecommendationConversionWorkflowState> conversionWorkflowState;
    int conversionWorkflowLimit = EA::ExperimentRecommendation::
        kDefaultRecommendationConversionWorkflowListLimit;
    bool conversionWorkflowLimitSpecified = false;
    bool planRecommendationCampaign = false;
    bool reviewRecommendationCampaign = false;
    bool approveRecommendationCampaign = false;
    bool rejectRecommendationCampaign = false;
    std::optional<std::string> campaignReviewIdentityHash;
    std::optional<std::string> campaignReviewer;
    std::optional<std::string> campaignReviewReason;
    std::optional<long long> showRecommendationCampaignApprovalId;
    bool listRecommendationCampaignApprovals = false;
    std::optional<EA::ExperimentRecommendation::
        RecommendationCampaignApprovalDecision> campaignApprovalDecision;
    int campaignApprovalLimit = 100;
    bool campaignApprovalLimitSpecified = false;
    bool materializeRecommendationCampaign = false;
    std::optional<long long> campaignMaterializationApprovalId;
    std::optional<std::string> campaignMaterializedBy;
    std::optional<std::string> campaignMaterializationReason;
    std::optional<long long> showRecommendationCampaignMaterializationId;
    bool listRecommendationCampaignMaterializations = false;
    int campaignMaterializationLimit = 100;
    bool campaignMaterializationLimitSpecified = false;
    std::optional<long long> showRecommendationCampaignHandoffId;
    bool listRecommendationCampaignHandoffs = false;
    int campaignHandoffLimit = EA::ExperimentRecommendation::
        kDefaultRecommendationCampaignHandoffListLimit;
    bool campaignHandoffLimitSpecified = false;
    std::optional<long long> reviewRecommendationCampaignMaterializationId;
    std::optional<EA::ExperimentRecommendation::
        RecommendationConversionProposalReviewDecision>
        campaignProposalReviewDecision;
    std::optional<std::string> campaignProposalReviewOperator;
    std::optional<std::string> campaignProposalReviewReason;
    bool campaignProposalReviewCommandSpecified = false;
    bool campaignProposalReviewDecisionSpecified = false;
    bool campaignProposalReviewOperatorSpecified = false;
    bool campaignProposalReviewReasonSpecified = false;
    std::optional<long long> executeRecommendationCampaignMaterializationId;
    bool campaignExecutionCommandSpecified = false;
    std::optional<long long> activateRecommendationCampaignMaterializationId;
    bool campaignActivationCommandSpecified = false;
    std::optional<long long> launchRecommendationCampaignMaterializationId;
    bool campaignLaunchCommandSpecified = false;
    std::optional<long long> recommendationCampaignStatusMaterializationId;
    bool campaignStatusCommandSpecified = false;
    std::optional<long long>
        recommendationCampaignOutcomeAssessmentMaterializationId;
    bool campaignOutcomeAssessmentCommandSpecified = false;
    std::optional<long long> campaignOperationsBudgetGrantCampaignId;
    std::optional<long long> campaignOperationsBudgetAmendCampaignId;
    std::optional<long long> campaignOperationsBudgetRevokeCampaignId;
    std::optional<long long> campaignOperationsBudgetSupersedeCampaignId;
    std::optional<long long> campaignOperationsAdmitMaterializationId;
    std::optional<long long> campaignOperationsAcceptRequestCampaignId;
    std::optional<long long> campaignOperationsBudgetStatusCampaignId;
    std::optional<long long> campaignOperationsRequestStatusRequestId;
    std::optional<long long> campaignOperationsPauseCampaignId;
    std::optional<long long> campaignOperationsResumeCampaignId;
    std::optional<long long> campaignOperationsCancelCampaignId;
    std::optional<long long> campaignOperationsControlStatusCampaignId;
    std::optional<long long> campaignOperationsCompleteCampaignId;
    std::optional<long long> campaignOperationsCompletionStatusCampaignId;
    bool campaignOperationsProductionReadiness = false;
    bool campaignOperationsProductionStatus = false;
    bool campaignOperationsProductionEnable = false;
    bool campaignOperationsProductionDisable = false;
    bool campaignOperationsProductionDispatchRequest = false;
    std::optional<int> campaignOperationsManagerRunOnceLimit;
    std::optional<std::string> campaignOperationsReconcileRunKey;
    bool campaignOperationsReconcileRecover = false;
    std::optional<int> campaignOperationsExpectedControlVersion;
    std::optional<long long> campaignOperationsControlRequestId;
    std::optional<int> campaignOperationsExpectedRequestVersion;
    std::optional<int> campaignOperationsExpectedProductionVersion;
    std::optional<std::string> campaignOperationsOperationKey;
    std::optional<std::string>
        campaignOperationsIndependentVerificationReference;
    std::optional<long long> campaignOperationsReconcileAfterRequestId;
    std::optional<int> campaignOperationsReconcileLimit;
    std::optional<int> campaignOperationsExpectedBudgetVersion;
    std::optional<long long> campaignOperationsBudgetValue;
    std::optional<std::string> campaignOperationsActor;
    std::optional<std::string> campaignOperationsReason;
    std::optional<std::string> campaignOperationsReservationExpiresAt;
    EA::ExperimentRecommendation::RecommendationCampaignPlanningPolicy
        campaignPlanningPolicy;
    EA::ExperimentRecommendation::RecommendationCampaignPlanningScope
        campaignPlanningScope;
    bool campaignPolicyOptionSpecified = false;
    bool campaignDonchian20ArmsSpecified = false;
    std::optional<long long> requeueAnalysisExperimentId;
    std::optional<long long> requeueInferenceExperimentId;
    std::optional<std::pair<long long, int>> stopAfterCheckpoint;
    std::optional<long long> clearStopAfterCheckpointExperimentId;
    std::optional<int> stopAfterCheckpointAllEpoch;
    bool clearStopAfterCheckpointAll = false;
    std::optional<long long> enableCheckpointInferExperimentId;
    std::optional<long long> disableCheckpointInferExperimentId;
    std::optional<std::pair<long long, int>> checkpointInferMinEpoch;
    std::optional<std::pair<long long, int>> checkpointInferInterval;
    std::optional<long long> enableCheckpointPolicyExperimentId;
    std::optional<long long> disableCheckpointPolicyExperimentId;
    std::optional<std::pair<long long, std::string>> setCheckpointPolicy;
    bool queueCheckpointInfer = false;
    std::optional<int> queueCheckpointInferMinEpoch;
    std::optional<int> queueCheckpointInferInterval;
    bool queueCheckpointPolicy = false;
    std::optional<double> checkpointPolicyMinLeaderScore;
    std::optional<double> checkpointPolicyMinInferAccuracy;
    std::optional<int> checkpointPolicyTopN;
    std::string checkpointPolicyScope = "symbol_horizon";
    std::string checkpointPolicyStopMode = "next_checkpoint";
    int checkpointPolicyGraceEvals = 1;
    std::set<std::string> checkpointPolicySetKeys;
    bool queueContinuationCandidateExcluded = false;
    std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity>
        economicCalendarSnapshot;
    bool help = false;
    bool dryRun = false;
    bool yes = false;
    bool force = false;
    bool schedulerVerbose = false;
    bool schedulerOnce = false;
    bool recoverOrphansOnly = false;
    bool autoEvaluateContinuations = false;
    bool autoQueueContinuations = false;
    bool continuationDryRun = false;
    int continuationScanSeconds = 300;
    int continuationMaxQueuesPerScan = 1;
    bool continuationScanSecondsSpecified = false;
    bool continuationMaxQueuesPerScanSpecified = false;
    int maxTrainProcs = kDefaultMaxTrainProcs;
    int maxInferProcs = kDefaultMaxInferProcs;
    int maxAnalyzeProcs = kDefaultMaxAnalyzeProcs;
    int schedulerPollSeconds = 30;
    std::string schedulerLogDir = "experiment_logs";
    std::string experimentReportDir = "experiment_reports";
    bool lstmProfileHotspots = false;
    std::optional<std::string> lstmProfileOutputPath;
    std::string schedulerExecutablePath;
    std::string semanticWorkerRegistryPath;
    bool semanticWorkerRegistryPathSpecified = false;
    std::string currentWorkerExecutablePath;
    std::string analyzeWorkerExecutablePath;
    std::optional<EA::Scheduler::SemanticWorkerRegistry> semanticWorkerRegistry;
    std::optional<std::string> legacyLayout6InferWorkerPath;
    std::string invocationCommandLine;
    EA::SchedulerCore::SchedulerAuthorityContext schedulerAuthority;
    std::optional<long long> schedulerWorkerAttemptId;
    EA::SchedulerCore::SchedulerRuntimeContext* runtimeContext = nullptr;

    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<double> cNextThreshold;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    std::optional<Donchian20Mode> donchian20Mode;
    EA::FeatureWarmupScope featureWarmupScope = EA::kDefaultFeatureWarmupScope;
    bool featureWarmupScopeSpecified = false;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    bool donchianLookbackSpecified = false;
    std::string featureAblationMask;
    bool featureAblationMaskSpecified = false;
    unsigned int freshInitializationSeed = 42U;
    bool freshInitializationSeedSpecified = false;
    std::optional<int> targetEpochs;
    std::optional<int> epochs;
    int checkpointInterval = 20;
    std::optional<std::string> trainStart;
    std::optional<std::string> trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> resumeModelId;
    bool resumeExpandInputWidth = false;
    bool allowDuplicateExperiment = false;
    EA::TrainingObjective::Configuration trainingObjective =
        EA::TrainingObjective::Legacy();
    bool trainingObjectiveSpecified = false;
    std::optional<std::string> queueInvocationMode;

    std::optional<std::string> leaderboardSymbol;
    std::optional<int> leaderboardHorizon;
    int leaderboardLimit = 20;
    std::string logLevel = "summary";
};

struct QueueResumeMeta
{
    long long modelId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    std::string trainStart;
    std::string trainEnd;
    int completedEpochs = 0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    EA::FeatureWarmupScope featureWarmupScope =
        EA::FeatureWarmupScope::LegacyColdBoundary;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    std::string featureAblationMask;
    unsigned int freshInitializationSeed = 42U;
    std::size_t modelInputWidth = 0;
    EA::TrainingObjective::Configuration trainingObjective =
        EA::TrainingObjective::Legacy();
};

struct ExperimentRow
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double cNextThreshold = 0.0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    int targetEpochs = 0;
    int checkpointInterval = 20;
    std::string trainStart;
    std::string trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> lastModelId;
    std::optional<long long> resumeModelId;
    std::optional<std::string> trainLogPath;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    EA::FeatureWarmupScope featureWarmupScope =
        EA::FeatureWarmupScope::LegacyColdBoundary;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    std::string featureAblationMask;
    unsigned int freshInitializationSeed = 42U;
    bool resumeExpandInputWidth = false;
    EA::TrainingObjective::Configuration trainingObjective =
        EA::TrainingObjective::Legacy();
    std::string schedulerPriority = "normal";
    bool resumeRequested = false;
    std::string schedulerResumeOrigin = "none";
    std::optional<long long> activeWorkerAttemptId;
};

struct CheckpointEvalRow
{
    long long checkpointEvalId = -1;
    ExperimentRow experiment;
    int checkpointEpoch = 0;
    long long checkpointModelId = -1;
    std::string status;
    std::string phase;
    std::optional<int> workerPid;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
    double inferStartedEpoch = 0.0;
    std::optional<long long> cancellationRequestId;
};

struct QueueSnapshot
{
    int pendingTrain = 0;
    int pendingInfer = 0;
    int pendingAnalyze = 0;
    int runningTrain = 0;
    int runningInfer = 0;
    int runningAnalyze = 0;
};

struct ParsedMetrics
{
    std::optional<long long> modelId;
    std::optional<int> completedEpochs;
    std::optional<double> trainAccuracy;
    std::optional<double> validationAccuracy;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> lossLast;
    std::optional<bool> acceptModel;
    std::optional<std::string> rejectReason;
    long long confusion[3][3] = {};
    bool hasConfusion = false;
};

struct AnalysisScopeOptions
{
    std::string scope = "final";
    std::optional<long long> checkpointEvalId;
    std::optional<int> checkpointEpoch;
    std::optional<long long> parentExperimentId;
};

struct QueueResumeCompatibilityRequirements
{
    int targetEpochs = 0;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<double> threshold;
    std::optional<std::string> trainStart;
    std::optional<std::string> trainEnd;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    std::optional<Donchian20Mode> donchian20Mode;
    std::optional<EA::FeatureWarmupScope> featureWarmupScope;
    std::optional<std::size_t> donchianLookback;
    std::optional<std::string> featureAblationMask;
    EA::TrainingObjective::Configuration trainingObjective =
        EA::TrainingObjective::Legacy();
};

struct TrainingCheckpointSelection
{
    std::optional<QueueResumeMeta> checkpoint;
    std::string reason;
};

struct TrainingCheckpointCandidate
{
    long long modelId = -1;
    std::optional<int> completedEpoch;
    bool periodicCheckpoint = false;
    std::optional<std::size_t> experimentInputWidth;
};

class NativeSchedulerOwnerProcessInspector final
    : public EA::SchedulerCore::SchedulerOwnerProcessInspector
{
public:
    SchedulerOwnerProcessEvidence inspectOwner(
        int processPid,
        int processGroupId,
        const std::string& processStartIdentity,
        const std::string& canonicalExecutablePath) override
    {
        return InspectSchedulerOwnerProcess(
            processPid,
            processGroupId,
            processStartIdentity,
            canonicalExecutablePath);
    }
};

class SchedulerServiceComposition final
{
public:
    explicit SchedulerServiceComposition(pqxx::transaction_base& transaction)
        : repository{transaction},
          admission{repository},
          workerControl{
              EA::SchedulerCore::NativeWorkerProcessController()},
          authority{repository, authorityProcessInspector}
    {
    }

    EA::SchedulerCore::PostgresSchedulerRepository repository;
    EA::SchedulerCore::SchedulerAdmissionService admission;
    EA::SchedulerCore::WorkerControlService workerControl;
    NativeSchedulerOwnerProcessInspector authorityProcessInspector;
    EA::SchedulerCore::SchedulerAuthorityService authority;
};

struct SchedulerProcessAbsenceEvidence
{
    bool inspectionSucceeded = false;
    std::vector<std::pair<int, std::string>> schedulers;
};

struct QueuedModelInputIdentity
{
    std::size_t width = EA::kCurrentModelInputWidth;
    int semanticLayoutVersion = EA::kModelInputSemanticLayoutVersion;
};

enum class StoppedWorkerAdmissionResult
{
    NotApplicable,
    Admitted,
    MissingProcessFallbackReady,
    DeferredNoCapacity,
    DeferredUnsafe
};

std::string GetEnvOrDefault(const char* name, const char* fallback);

std::string LstmDbConnectionString();

std::string SqlNullable(pqxx::work& w, const std::optional<std::string>& value);

std::string SqlNullable(pqxx::work& w, const std::optional<double>& value);

std::string SqlNullable(pqxx::work& w, const std::optional<int>& value);

std::string SqlNullable(pqxx::work& w, const std::optional<long long>& value);

std::string SqlContinuationTargetSequence(
    const std::optional<std::vector<int>>& sequence);

std::string FormatDouble(double value);

void ValidateCheckpointPolicyConfig(const SchedulerOptions& options);

bool TableExists(pqxx::work& w, const std::string& tableName);

bool ColumnExists(pqxx::work& w, const std::string& tableName, const std::string& columnName);

std::string DateOnly(const std::string& value);

QueueResumeMeta LoadQueueResumeMeta(pqxx::work& w, long long modelId);

void ThrowQueueResumeInvalid(const std::string& reason,
                                    long long modelId,
                                    const std::string& detail = {});

std::optional<std::string> QueueResumeCompatibilityFailure(
    const QueueResumeMeta& meta,
    const QueueResumeCompatibilityRequirements& requirements);

void MergeResumeMetaIntoQueueOptions(SchedulerOptions& options,
                                            const QueueResumeMeta& meta);

TrainingCheckpointSelection SelectUsableTrainingCheckpoint(
    pqxx::work& w,
    const ExperimentRow& experiment,
    const std::optional<double>& createdAtOrAfter = std::nullopt,
    bool allowFinalModel = true);

bool RequireSchedulerTables(pqxx::work& w);

std::optional<EA::GlobalExperimentControl::ControlSnapshot>
LoadLockedGlobalControl(pqxx::work& w);

void SetTransactionReadWrite(pqxx::work& w);

void SetTransactionReadOnly(pqxx::work& w);

SchedulerProcessAbsenceEvidence
InspectAllSchedulerDispatchProcesses();

void PrintModelSymbolMissing(long long modelId);

std::optional<std::string> TryLoadPersistedCanonicalSymbol(pqxx::work& w,
                                                                  long long modelId);

void ResolveEconomicCalendarSnapshotForQueue(
    pqxx::work& transaction,
    SchedulerOptions& options,
    const std::string& creationMode);

QueuedModelInputIdentity ResolveQueuedModelInputIdentity(
    pqxx::work& w, const SchedulerOptions& options);

long long InsertExperimentRecord(pqxx::work& w,
                                        const SchedulerOptions& options,
                                        const std::string& canonicalSymbol,
                                        long long duplicateNonce);

void EnsureRequiredQueueOptions(const SchedulerOptions& options);

std::optional<double> OptionalDoubleCell(const pqxx::row& row, int index);

std::optional<long long> OptionalLongLongCell(const pqxx::row& row, int index);

std::optional<std::string> OptionalStringCell(const pqxx::row& row, int index);

ExperimentRow RowToExperiment(const pqxx::row& row);

QueueSnapshot LoadQueueSnapshot(
    EA::SchedulerCore::SchedulerAdmissionService& admission);

bool CheckpointEvalTableExists(pqxx::work& w);

CheckpointEvalRow RowToCheckpointEval(const pqxx::row& row);

std::string ReadFileIfExists(const std::optional<std::string>& path);

std::string ReadFileIfExists(const std::string& path);

void ApplyPersistedSymbolToAnalysisExperiment(pqxx::work& w,
                                                     ExperimentRow& experiment,
                                                     const ParsedMetrics& metrics);

std::optional<double> ComputeLeaderScore(const ParsedMetrics& metrics);

int GenerateExperimentReports(const std::string& reportDir, bool warnOnly);

void TryGenerateExperimentReports(const SchedulerOptions& options);

void UpsertAnalysisResult(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 const ParsedMetrics& metrics,
                                 const std::optional<double>& leaderScore,
                                 const AnalysisScopeOptions& scopeOptions = AnalysisScopeOptions{});

CheckpointPolicyEvaluationResult EvaluateCheckpointPolicyAfterAnalysis(
    pqxx::work& transaction,
    const CheckpointEvalRow& eval);

std::vector<ContinuationEvidence> LoadContinuationEvidence(
    pqxx::work& w,
    const ContinuationPolicyConfig& config);

bool ValidateContinuationResumeSource(pqxx::work& w,
                                      const ContinuationPolicyConfig& config,
                                      const ContinuationEvidence& selected,
                                      QueueResumeMeta* loadedMeta,
                                      std::string& reason);

ContinuationEvaluation EvaluateContinuationPolicy(
    pqxx::work& w,
    long long sourceExperimentId,
    ContinuationPolicyConfig* loadedConfig,
    bool persistDecision = true);

ContinuationChildPolicyPlan PrepareContinuationChildPolicy(
    pqxx::work& w,
    const ContinuationPolicyConfig& sourceConfig,
    SchedulerOptions& child,
    int continuationSourceEpoch);

int RunQueueContinuationCommand(const SchedulerOptions& options);

int CountRows(pqxx::work& w, const std::string& sql);

} // namespace EA::SchedulerCore::ProductionRuntimeDetail
