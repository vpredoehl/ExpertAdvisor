#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentMetaAnalyzer
{

constexpr int kDefaultMetaAnalysisIntervalSeconds = 300;

struct MetaAnalysisOptions
{
    bool metaAnalyze = false;
    bool metaAnalyzeOnce = false;
    bool metaAnalysisReport = false;
    bool outputJson = false;
    bool outputMarkdown = false;
    int limit = 20;
    int intervalSeconds = kDefaultMetaAnalysisIntervalSeconds;
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
    std::optional<int> completedEpochs;
    std::optional<double> trainAccuracy;
    std::optional<double> validationAccuracy;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> acceptAccuracy;
    std::optional<double> leaderScore;
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
