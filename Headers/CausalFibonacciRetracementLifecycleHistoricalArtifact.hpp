#pragma once

#include "CausalFibonacciRetracementLifecycleHistoricalEvaluation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::FibonacciResearch::RetracementLifecycle
{

struct HistoricalArtifactProvenance
{
    std::string symbol;
    std::string timeframe;
    std::filesystem::path configurationPath;
    std::string configurationFingerprint;
    std::string baselineCommit;
    std::string requestedRange;
    std::string reproductionCommand;
};

struct HistoricalArtifactDataQuality
{
    std::size_t preflightRows = 0;
    std::size_t streamedRows = 0;
    std::size_t duplicateTimestampCount = 0;
    std::optional<std::int64_t> firstTimestamp;
    std::optional<std::int64_t> lastTimestamp;
};

class HistoricalArtifactWriter
{
public:
    HistoricalArtifactWriter(std::filesystem::path outputDirectory,
                             HistoricalArtifactProvenance provenance);
    ~HistoricalArtifactWriter();
    HistoricalArtifactWriter(const HistoricalArtifactWriter&) = delete;
    HistoricalArtifactWriter& operator=(const HistoricalArtifactWriter&) = delete;

    void AddRecord(HistoricalRecord record);
    void SetDataQuality(HistoricalArtifactDataQuality quality);
    void Complete();
    std::size_t ObservationCount() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

std::string LifecycleObservationCsvHeader();
std::string LifecycleObservationCsvRow(const HistoricalRecord& record,
                                       std::string_view symbol,
                                       std::string_view timeframe);

} // namespace EA::FibonacciResearch::RetracementLifecycle
