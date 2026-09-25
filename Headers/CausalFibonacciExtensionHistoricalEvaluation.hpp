#pragma once

#include "CausalFibonacciExtensionResearch.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace EA::FibonacciResearch
{

inline constexpr std::int64_t kFirstStudyStart = 1'262'304'000;
inline constexpr std::int64_t kFirstStudyEndExclusive = 1'735'689'600;
inline constexpr std::string_view kFirstStudyIdentity =
    "causal-fibonacci-extension-pre-2025-first-study-v1";

struct HistoricalDataQuality
{
    std::string symbol;
    bool excluded = false;
    std::string exclusionReason;
    std::size_t sourceRows = 0;
    std::size_t usableRows = 0;
    std::size_t materialGaps = 0;
    std::optional<std::int64_t> firstTimestamp;
    std::optional<std::int64_t> lastTimestamp;
};

class HistoricalEvaluator
{
public:
    using RecordSink = std::function<void(ObservationRecord)>;

    HistoricalEvaluator(std::string symbol,
                        TG4::EvaluationConfiguration configuration,
                        RecordSink sink);
    ~HistoricalEvaluator();
    HistoricalEvaluator(HistoricalEvaluator&&) noexcept;
    HistoricalEvaluator& operator=(HistoricalEvaluator&&) noexcept;
    HistoricalEvaluator(const HistoricalEvaluator&) = delete;
    HistoricalEvaluator& operator=(const HistoricalEvaluator&) = delete;

    void AddCompletedBar(const TG1A::Candle& candle);
    void Finalize();
    const HistoricalDataQuality& DataQuality() const;
    std::size_t RecordCount() const;
    bool IsFinalized() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

class HistoricalArtifactWriter
{
public:
    HistoricalArtifactWriter(std::filesystem::path outputDirectory,
                             TG4::EvaluationConfiguration configuration,
                             std::string baselineCommit,
                             std::vector<std::string> symbols,
                             std::string reproductionCommand);
    ~HistoricalArtifactWriter();
    HistoricalArtifactWriter(const HistoricalArtifactWriter&) = delete;
    HistoricalArtifactWriter& operator=(const HistoricalArtifactWriter&) = delete;

    void AddRecord(ObservationRecord record);
    void AddDataQuality(HistoricalDataQuality audit);
    void Complete();
    std::size_t ObservationCount() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace EA::FibonacciResearch
