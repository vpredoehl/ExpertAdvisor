#ifndef TG4HistoricalEmpiricalEvaluation_hpp
#define TG4HistoricalEmpiricalEvaluation_hpp

#include "CausalFibonacciConfluenceIntegration.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::TG4
{

inline constexpr std::string_view kObservationSchemaVersion =
    "tg4-inner-break-observation-v1";
inline constexpr std::string_view kAggregateSchemaVersion =
    "tg4-aggregate-v1";
inline constexpr std::string_view kStudyContractVersion =
    "tg4-historical-empirical-evaluation-v1";

enum class TemporalPartition
{
    Exploratory2010To2019,
    Calibration2020To2022,
    Validation2023To2024,
    Confirmation2025,
    OutsideNamedStudy
};

struct TemporalRange
{
    std::int64_t warmupStart = 0;
    std::int64_t scoreStart = 0;
    std::int64_t scoreEnd = 0;
    std::int64_t outcomeEnd = 0;
};

struct EvaluationConfiguration
{
    std::string configurationSchema;
    std::string name;
    std::string provenance;
    std::string timeframe;
    int candlePeriod = 0;
    std::string candleUnit;
    std::int64_t expectedIntervalSeconds = 0;
    double materialGapMultiple = 0.0;
    std::size_t minimumHumanReportResolvedN = 0;
    std::size_t maxPendingTG4Records = 0;

    TG1A::Configuration geometry;
    double referenceBarScale = 0.0;
    TG2::Configuration behavior;
    TG3::Configuration fibonacci;
};

struct WilsonInterval
{
    std::size_t numerator = 0;
    std::size_t denominator = 0;
    std::optional<double> rate;
    std::optional<double> lower95;
    std::optional<double> upper95;
};

struct GapObservation
{
    std::int64_t previousTimestamp = 0;
    std::int64_t nextTimestamp = 0;
    std::int64_t gapSeconds = 0;
    bool intersectsScoredRange = false;
};

struct DataQualityAudit
{
    std::string symbol;
    bool excluded = false;
    std::string exclusionReason;
    std::size_t sourceRows = 0;
    std::size_t usableRows = 0;
    std::size_t warmupRows = 0;
    std::size_t scoredRows = 0;
    std::size_t outcomeOnlyRows = 0;
    std::size_t duplicateTimestamps = 0;
    std::size_t outOfOrderTimestamps = 0;
    std::optional<std::int64_t> firstUsableTimestamp;
    std::optional<std::int64_t> lastUsableTimestamp;
    std::map<TemporalPartition, std::size_t> partitionRows;
    std::vector<GapObservation> materialGaps;
};

struct ObservationRecord
{
    std::string eventIdentity;
    std::string symbol;
    std::string timeframe;
    TemporalPartition partition = TemporalPartition::OutsideNamedStudy;
    TG2::BreakObservation behavior;
    TG3::ConfluenceObservation confluence;
    std::optional<TG1B::ClassifiedTrendLineCandidate> frozenInner;
    std::optional<TG1B::ClassifiedTrendLineCandidate> frozenOuter;
    std::string outerPairingEligibilityReason;
};

struct OutcomeTally
{
    std::size_t structurallyEligible = 0;
    std::size_t structurallyIneligible = 0;
    std::size_t pending = 0;
    std::size_t censored = 0;
    std::size_t resolved = 0;
    std::size_t successes = 0;
    std::size_t failures = 0;
};

struct CohortTally
{
    std::size_t totalObservations = 0;
    std::size_t paired = 0;
    std::size_t unpaired = 0;
    std::size_t confluent = 0;
    std::size_t nonConfluent = 0;
    std::size_t confluenceIneligible = 0;
    OutcomeTally retest;
    OutcomeTally outerTarget;
    OutcomeTally outerTargetAfterRetest;
};

struct CohortKey
{
    std::string symbol;
    std::string partition;
    std::string direction;
    std::string cohort;

    bool operator<(const CohortKey&) const;
};

struct AngleMoments
{
    std::size_t count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double m2 = 0.0;

    void Add(double value);
    std::optional<double> SampleStandardDeviation() const;
};

struct AngleTally
{
    AngleMoments inner;
    AngleMoments outer;
};

struct ComparisonCounts
{
    std::size_t groupASuccesses = 0;
    std::size_t groupAFailures = 0;
    std::size_t groupBSuccesses = 0;
    std::size_t groupBFailures = 0;
};

class AggregateAccumulator
{
public:
    void Add(const ObservationRecord& record);

    const std::map<CohortKey, CohortTally>& Cohorts() const;
    const std::map<CohortKey, AngleTally>& Angles() const;
    const std::map<CohortKey, ComparisonCounts>& Comparisons() const;

private:
    std::map<CohortKey, CohortTally> cohorts_;
    std::map<CohortKey, AngleTally> angles_;
    std::map<CohortKey, ComparisonCounts> comparisons_;
};

class HistoricalEvaluator
{
public:
    using RecordSink = std::function<void(const ObservationRecord&)>;

    HistoricalEvaluator(std::string symbol,
                        EvaluationConfiguration configuration,
                        TemporalRange range,
                        RecordSink sink);

    void AddCompletedBar(const TG1A::Candle& candle);
    void Finalize();

    const DataQualityAudit& DataQuality() const;
    const std::deque<ObservationRecord>& PendingRecords() const;
    std::size_t EmittedRecordCount() const;
    std::size_t PeakPendingRecordCount() const;
    bool IsFinalized() const;

private:
    std::string symbol_;
    EvaluationConfiguration configuration_;
    TemporalRange range_;
    RecordSink sink_;
    TG3::CausalFibonacciConfluenceIntegration integration_;
    DataQualityAudit dataQuality_;
    std::deque<ObservationRecord> pending_;
    std::size_t emittedRecordCount_ = 0;
    std::size_t peakPendingRecordCount_ = 0;
    std::optional<std::int64_t> lastTimestamp_;
    bool finalized_ = false;

    void UpdateDataQuality(const TG1A::Candle& candle);
    void SynchronizePending(std::size_t currentBar,
                            std::int64_t currentTimestamp);
    void CaptureNewRecords(const TG3::Update& update);
    void FlushTerminalPrefix();
};

class StudyArtifactWriter
{
public:
    StudyArtifactWriter(std::filesystem::path outputDirectory,
                        EvaluationConfiguration configuration,
                        TemporalRange range,
                        std::string baselineCommit);
    ~StudyArtifactWriter();

    StudyArtifactWriter(const StudyArtifactWriter&) = delete;
    StudyArtifactWriter& operator=(const StudyArtifactWriter&) = delete;

    void Write(const ObservationRecord& record);
    void AddDataQuality(DataQualityAudit audit);
    void Complete();

    const AggregateAccumulator& Aggregates() const;
    std::size_t ObservationCount() const;

private:
    class Implementation;
    Implementation* implementation_;
};

EvaluationConfiguration LoadConfigurationFile(
    const std::filesystem::path& path);
void ValidateConfiguration(const EvaluationConfiguration& configuration);
void ValidateTemporalRange(const TemporalRange& range);

TemporalPartition PartitionForTimestamp(std::int64_t timestamp);
std::string TemporalPartitionName(TemporalPartition partition);
std::int64_t ParseUtcDateOrTimestamp(std::string_view text);
std::string FormatUtcTimestamp(std::int64_t timestamp);
WilsonInterval Wilson95(std::size_t successes, std::size_t failures);

std::string ObservationCsvHeader();
std::string ObservationCsvRow(const ObservationRecord& record,
                              const EvaluationConfiguration& configuration);
std::string FrozenCausalSnapshot(const ObservationRecord& record);
std::string EffectiveConfigurationJson(
    const EvaluationConfiguration& configuration,
    const TemporalRange& range,
    std::string_view baselineCommit);

} // namespace EA::TG4

#endif /* TG4HistoricalEmpiricalEvaluation_hpp */
