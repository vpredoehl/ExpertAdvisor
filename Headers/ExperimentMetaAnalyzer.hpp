#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentMetaAnalyzer
{

constexpr int kDefaultMetaAnalysisIntervalSeconds = 300;

enum class RecommendationEpochPolicy
{
    Leader,
    Highest
};

struct MetaAnalysisOptions
{
    bool metaAnalyze = false;
    bool metaAnalyzeOnce = false;
    bool metaAnalysisReport = false;
    bool queueMetaRecommendations = false;
    bool outputJson = false;
    bool outputMarkdown = false;
    bool dryRun = false;
    int limit = 20;
    int intervalSeconds = kDefaultMetaAnalysisIntervalSeconds;
    RecommendationEpochPolicy recommendationEpochPolicy = RecommendationEpochPolicy::Highest;
    std::optional<std::string> symbol;
    std::optional<int> horizon;
    std::optional<std::string> outputFile;
};

struct ExperimentRecord
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string modelName;
    std::string symbol;
    int horizon = 0;
    double threshold = 0.0;
    std::optional<double> coreLr;
    std::optional<double> headLr;
    int targetEpochs = 0;
    int checkpointInterval = 0;
    std::string status;
    std::string phase;
    bool resumed = false;
    std::optional<std::string> createdAt;
    std::optional<std::string> completedAt;
    std::optional<int> completedEpochs;
    std::optional<double> trainAccuracy;
    std::optional<double> validationAccuracy;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> acceptAccuracy;
    std::optional<double> leaderScore;
    std::string dataSource;
};

struct DataSourceSummary
{
    long long schedulerExperiments = 0;
    long long legacyModels = 0;
    long long mergedRecords = 0;
    long long duplicateModelsSkipped = 0;
    long long legacyModelsWithInferenceMetrics = 0;
    long long legacyModelsMissingInferenceMetrics = 0;
    long long schedulerRecordsWithMetrics = 0;
    long long mergedRecordsWithMetrics = 0;
    double metricCoveragePercentage = 0.0;
    double coveragePercentage = 0.0;
};

struct ScalarStats
{
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double best = 0.0;
    double worst = 0.0;
    double stddev = 0.0;
};

struct GroupStats
{
    std::string dimension;
    std::string key;
    std::size_t sampleCount = 0;
    ScalarStats leaderScore;
    ScalarStats inferAccuracy;
    ScalarStats acceptRate;
    ScalarStats acceptAccuracy;
};

struct LeaderRow
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string symbol;
    int horizon = 0;
    int targetEpochs = 0;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> leaderScore;
    std::string modelName;
};

struct Recommendation
{
    int priority = 0;
    std::string action;
    std::string reason;
    std::string evidence;
    std::string expectedInformationGain;
    std::string confidence;
};

struct NextExperimentRecommendation
{
    int rank = 0;
    std::string symbol;
    int horizon = 0;
    int targetEpochs = 0;
    double threshold = 0.0;
    double coreLr = 0.0;
    double headLr = 0.0;
    std::string reason;
    long long sourceLeaderExperimentId = -1;
    std::optional<long long> sourceModelId;
};

struct PlateauSignal
{
    std::string key;
    int earlierEpoch = 0;
    int laterEpoch = 0;
    double earlierScore = 0.0;
    double laterScore = 0.0;
    double delta = 0.0;
    std::string classification;
    std::string confidence;
};

struct MetaAnalysisResult
{
    std::string scope;
    std::string generatedAt;
    std::string generatedGitCommit = "unknown";
    std::string generatedGitBranch = "unknown";
    std::string generatedGitDirty = "unknown";
    std::string generatedBuildConfig = "unknown";
    std::string generatedCompilerVersion = "unknown";
    std::string generatedSchemaVersion = "unknown";
    std::string generatedSchedulerVersion = "unknown";
    RecommendationEpochPolicy recommendationEpochPolicy = RecommendationEpochPolicy::Highest;
    long long totalExperiments = 0;
    long long completedExperiments = 0;
    long long failedExperiments = 0;
    long long runningExperiments = 0;
    long long pendingExperiments = 0;
    long long completedModels = 0;
    double successRate = 0.0;
    std::vector<ExperimentRecord> records;
    std::vector<GroupStats> groupStats;
    std::vector<LeaderRow> leaders;
    std::vector<PlateauSignal> plateauSignals;
    std::vector<Recommendation> recommendations;
    std::vector<NextExperimentRecommendation> nextExperimentRecommendations;
    std::vector<std::string> nextExperimentNotes;
    DataSourceSummary dataSources;
    std::string statisticsJson;
    std::string recommendationsJson;
    std::string leaderboardJson;
    std::string markdown;
    long long metaAnalysisId = -1;
};

bool IsMetaAnalysisCommand(int argc, const char* argv[]);
MetaAnalysisOptions ParseMetaAnalysisArgs(int argc, const char* argv[]);

std::string ConfidenceForSampleSize(std::size_t n);
std::string ScopeForOptions(const MetaAnalysisOptions& options);

int RunMetaAnalysisOnce(const MetaAnalysisOptions& options);
int RunContinuousMetaAnalysis(MetaAnalysisOptions options);
int PrintLatestReport(const MetaAnalysisOptions& options);
int RunMetaAnalysisCli(int argc, const char* argv[]);

} // namespace EA::ExperimentMetaAnalyzer
