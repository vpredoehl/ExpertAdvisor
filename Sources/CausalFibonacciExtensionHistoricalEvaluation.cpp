#include "CausalFibonacciExtensionHistoricalEvaluation.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace EA::FibonacciResearch
{
namespace
{

using Index = std::size_t;

bool SameIdentity(const TG3::ABIdentity& left, const TG3::ABIdentity& right)
{
    return left == right;
}

std::string PeriodAt(std::int64_t timestamp)
{
    return TG4::TemporalPartitionName(TG4::PartitionForTimestamp(timestamp));
}

bool Terminal(OutcomeState state)
{
    return state == OutcomeState::Success || state == OutcomeState::Failure ||
        state == OutcomeState::RightCensored ||
        state == OutcomeState::AmbiguousIneligible;
}

void Start(Outcome& outcome, const EventOccurrence& eligibility,
           std::string detail)
{
    outcome.state = OutcomeState::Pending;
    outcome.eligibility = eligibility;
    outcome.detail = std::move(detail);
}

void Succeed(Outcome& outcome, const EventOccurrence& occurrence)
{
    outcome.state = OutcomeState::Success;
    outcome.resolution = occurrence;
    outcome.barsToTarget = occurrence.bar - outcome.eligibility->bar;
}

void Fail(Outcome& outcome, std::size_t bar, const TG1A::Candle& candle)
{
    outcome.state = OutcomeState::Failure;
    outcome.resolution = EventOccurrence{
        outcome.eligibility->type, bar, candle.timestamp, candle.close};
    outcome.detail += ":frozen_horizon_expired";
}

std::string Optional(const std::optional<double>& value)
{
    return value.has_value() ? Detail::Number(*value) : std::string{};
}

std::string CsvLine(const std::vector<std::string>& fields)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0) output << ',';
        output << Detail::Csv(fields[index]);
    }
    output << '\n';
    return output.str();
}

} // namespace

void ValidateHistoricalStudySpecification(
    const HistoricalStudySpecification& specification)
{
    const TG4::TemporalRange& range = specification.range;
    if (specification.identity.empty() ||
        range.warmupStart > range.scoreStart ||
        range.scoreStart >= range.scoreEnd ||
        range.scoreEnd > range.outcomeEnd)
        throw std::invalid_argument("invalid Fibonacci historical study specification");
}

bool IsHistoricalStudyEligibilityTimestamp(
    const HistoricalStudySpecification& specification, std::int64_t timestamp)
{
    return timestamp >= specification.range.scoreStart &&
        timestamp < specification.range.scoreEnd;
}

class HistoricalEvaluator::Implementation
{
public:
    Implementation(std::string symbol,
                   TG4::EvaluationConfiguration configuration,
                   HistoricalStudySpecification specification,
                   RecordSink sink)
        : symbol_(std::move(symbol)), configuration_(std::move(configuration)),
          specification_(specification), sink_(std::move(sink)), integration_(
              TG1B::CalibrationConfiguration(configuration_.referenceBarScale),
              FibonacciConfiguration(), configuration_.geometry,
              configuration_.behavior, {symbol_, configuration_.timeframe})
    {
        TG4::ValidateConfiguration(configuration_);
        ValidateHistoricalStudySpecification(specification_);
        if (!sink_) throw std::invalid_argument("Fibonacci record sink is empty");
        if (configuration_.timeframe != "15m" ||
            configuration_.candlePeriod != 15 ||
            configuration_.candleUnit != "minute")
            throw std::invalid_argument(
                "Fibonacci historical study requires canonical 15-minute bars");
        (void)FrozenProspectiveEvaluationConfiguration(symbol_);
        audit_.symbol = symbol_;
    }

    void Add(const TG1A::Candle& candle)
    {
        if (finalized_)
            throw std::logic_error("Fibonacci evaluator is finalized");
        if (candle.timestamp < specification_.range.warmupStart ||
            candle.timestamp >= specification_.range.outcomeEnd)
            throw std::invalid_argument(
                "Fibonacci study rejects bars outside its temporal range");
        Audit(candle);
        const TG3::Update update = integration_.AddCompletedBar(candle);
        AddStructures(update.newlyAvailableABStructures);
        ResolvePending(update.bar, candle);
        ObserveRejections(update.bar, candle);
        ObserveTouches(update.bar, candle);
        ObserveBeyond(update.bar, candle);
    }

    void Finalize()
    {
        if (finalized_) return;
        integration_.Finalize();
        for (State& state : states_)
        {
            Censor(state.record.h1);
            Censor(state.record.h2);
            Censor(state.record.h2Pullback0500Descriptive);
            Censor(state.record.h2Pullback0618Descriptive);
        }
        std::sort(states_.begin(), states_.end(),
            [](const State& left, const State& right)
            {
                return left.record.eventIdentity < right.record.eventIdentity;
            });
        emitted_ = std::count_if(states_.begin(), states_.end(),
            [this](const State& state) { return ShouldEmit(state); });
        for (State& state : states_)
            if (ShouldEmit(state)) sink_(std::move(state.record));
        finalized_ = true;
    }

    const HistoricalDataQuality& DataQuality() const { return audit_; }
    std::size_t RecordCount() const { return emitted_; }
    bool IsFinalized() const { return finalized_; }

private:
    struct State
    {
        ObservationRecord record;
        bool h1PendingListed = false;
        bool h2PendingListed = false;
    };

    std::string symbol_;
    TG4::EvaluationConfiguration configuration_;
    HistoricalStudySpecification specification_;
    RecordSink sink_;
    TG3::CausalFibonacciConfluenceIntegration integration_;
    HistoricalDataQuality audit_;
    std::optional<std::int64_t> lastTimestamp_;
    std::vector<State> states_;
    std::set<std::string> identities_;
    std::multimap<double, Index> upTouch_;
    std::multimap<double, Index> downTouch_;
    std::multimap<double, Index> upBeyond_;
    std::multimap<double, Index> downBeyond_;
    std::multimap<double, Index> upRejection_;
    std::multimap<double, Index> downRejection_;
    std::vector<Index> h1Pending_;
    std::vector<Index> h2Pending_;
    std::size_t emitted_ = 0;
    bool finalized_ = false;

    bool IsScoreTimestamp(std::int64_t timestamp) const
    {
        return IsHistoricalStudyEligibilityTimestamp(specification_, timestamp);
    }

    static bool IsScored(const Outcome& outcome)
    {
        return outcome.eligibility.has_value();
    }

    static bool RelevantToScoringWindow(const State& state)
    {
        return IsScored(state.record.h1) || IsScored(state.record.h2) ||
            IsScored(state.record.h2Pullback0500Descriptive) ||
            IsScored(state.record.h2Pullback0618Descriptive);
    }

    bool ShouldEmit(const State& state) const
    {
        // The completed discovery study had no distinct warmup interval, so
        // retaining every structural observation preserves its frozen artifact.
        return specification_.range.warmupStart == specification_.range.scoreStart ||
            RelevantToScoringWindow(state);
    }

    TG3::Configuration FibonacciConfiguration() const
    {
        TG3::Configuration result = configuration_.fibonacci;
        result.absolutePriceTolerance =
            TG4::EffectiveFibonacciAbsolutePriceTolerance(
                configuration_, symbol_);
        return result;
    }

    void Audit(const TG1A::Candle& candle)
    {
        ++audit_.sourceRows;
        if (lastTimestamp_.has_value())
        {
            if (candle.timestamp <= *lastTimestamp_)
                throw std::invalid_argument(
                    "Fibonacci bars require unique increasing timestamps");
            if (static_cast<double>(candle.timestamp - *lastTimestamp_) >
                static_cast<double>(configuration_.expectedIntervalSeconds) *
                    configuration_.materialGapMultiple)
                ++audit_.materialGaps;
        }
        lastTimestamp_ = candle.timestamp;
        if (!audit_.firstTimestamp.has_value())
            audit_.firstTimestamp = candle.timestamp;
        audit_.lastTimestamp = candle.timestamp;
        ++audit_.usableRows;
    }

    void AddStructures(const std::vector<TG3::ABIdentity>& identities)
    {
        for (const TG3::ABIdentity& identity : identities)
        {
            const auto found = std::find_if(
                integration_.ABStructures().begin(),
                integration_.ABStructures().end(),
                [&identity](const TG3::ABStructure& structure)
                {
                    return SameIdentity(structure.identity, identity);
                });
            if (found == integration_.ABStructures().end())
                throw std::logic_error("new TG3 A/B structure was not retained");
            GroupingMetadata grouping{
                symbol_, configuration_.timeframe,
                PeriodAt(identity.availabilityTimestamp), std::nullopt,
                std::nullopt};
            CausalExtensionTracker initial(
                *found, grouping,
                FrozenProspectiveEvaluationConfiguration(symbol_));
            if (!identities_.insert(initial.Record().eventIdentity).second)
                throw std::runtime_error(
                    "duplicate Fibonacci structural event identity");
            const Index index = states_.size();
            states_.push_back({initial.Record()});
            const ObservationRecord& record = states_.back().record;
            if (identity.direction == TG3::ABDirection::UpAB)
            {
                upTouch_.emplace(record.extension1272.zoneLowerPrice, index);
                upBeyond_.emplace(record.extension1272.zoneUpperPrice, index);
            }
            else
            {
                downTouch_.emplace(record.extension1272.zoneUpperPrice, index);
                downBeyond_.emplace(record.extension1272.zoneLowerPrice, index);
            }
        }
    }

    template<typename Map, typename Function>
    static void ExtractRange(Map& values, typename Map::iterator begin,
                             typename Map::iterator end, Function function)
    {
        for (auto iterator = begin; iterator != end;)
        {
            const Index index = iterator->second;
            iterator = values.erase(iterator);
            function(index);
        }
    }

    void ObserveTouches(std::size_t bar, const TG1A::Candle& candle)
    {
        const auto touch = [this, bar, &candle](Index index)
        {
            State& state = states_[index];
            const bool up = state.record.sourceAB.identity.direction ==
                TG3::ABDirection::UpAB;
            state.record.extension1272Touched = EventOccurrence{
                EventType::Extension1272Touched, bar, candle.timestamp,
                up ? candle.high : candle.low};
            if (up)
                upRejection_.emplace(
                    state.record.extension1272.zoneLowerPrice, index);
            else
                downRejection_.emplace(
                    state.record.extension1272.zoneUpperPrice, index);
        };
        ExtractRange(upTouch_, upTouch_.begin(), upTouch_.upper_bound(candle.high),
                     touch);
        ExtractRange(downTouch_, downTouch_.lower_bound(candle.low),
                     downTouch_.end(), touch);
    }

    void ObserveBeyond(std::size_t bar, const TG1A::Candle& candle)
    {
        const auto beyond = [this, bar, &candle](Index index)
        {
            State& state = states_[index];
            EventOccurrence occurrence{EventType::Extension1272Beyond, bar,
                                       candle.timestamp, candle.close};
            state.record.extension1272Beyond = occurrence;
            if (IsScoreTimestamp(candle.timestamp))
            {
                Start(state.record.h1, occurrence,
                      "h1_1.618_after_1.272_beyond");
                if (!state.h1PendingListed)
                {
                    h1Pending_.push_back(index);
                    state.h1PendingListed = true;
                }
            }
        };
        ExtractRange(upBeyond_, upBeyond_.begin(),
                     upBeyond_.lower_bound(candle.close), beyond);
        ExtractRange(downBeyond_, downBeyond_.upper_bound(candle.close),
                     downBeyond_.end(), beyond);
    }

    void ObserveRejections(std::size_t bar, const TG1A::Candle& candle)
    {
        const auto reject = [this, bar, &candle](Index index)
        {
            State& state = states_[index];
            if (bar <= state.record.extension1272Touched->bar)
                throw std::logic_error("same-bar Fibonacci rejection leaked");
            EventOccurrence occurrence{
                EventType::Extension1272RejectionConfirmed, bar,
                candle.timestamp, candle.close};
            state.record.rejectionConfirmed = occurrence;
            if (IsScoreTimestamp(candle.timestamp))
            {
                Start(state.record.h2, occurrence,
                      "h2_0.382_after_rejection");
                Start(state.record.h2Pullback0500Descriptive, occurrence,
                      "h2_descriptive_0.500_after_rejection");
                Start(state.record.h2Pullback0618Descriptive, occurrence,
                      "h2_secondary_0.618_after_rejection");
                if (!state.h2PendingListed)
                {
                    h2Pending_.push_back(index);
                    state.h2PendingListed = true;
                }
            }
        };
        ExtractRange(upRejection_, upRejection_.upper_bound(candle.close),
                     upRejection_.end(), reject);
        ExtractRange(downRejection_, downRejection_.begin(),
                     downRejection_.lower_bound(candle.close), reject);
    }

    void ResolvePending(std::size_t bar, const TG1A::Candle& candle)
    {
        std::vector<Index> nextH1;
        for (Index index : h1Pending_)
        {
            State& state = states_[index];
            Outcome& outcome = state.record.h1;
            const std::size_t elapsed = bar - outcome.eligibility->bar;
            const bool up = state.record.sourceAB.identity.direction ==
                TG3::ABDirection::UpAB;
            const bool reached = up
                ? candle.high >= state.record.extension1618.zoneLowerPrice
                : candle.low <= state.record.extension1618.zoneUpperPrice;
            if (reached)
            {
                EventOccurrence occurrence{
                    EventType::Extension1618ReachedAfterBeyond, bar,
                    candle.timestamp, up ? candle.high : candle.low};
                state.record.extension1618ReachedAfterBeyond = occurrence;
                Succeed(outcome, occurrence);
            }
            else if (elapsed >= kPrimaryEvaluationHorizonBars)
                Fail(outcome, bar, candle);
            else
                nextH1.push_back(index);
        }
        h1Pending_.swap(nextH1);

        std::vector<Index> nextH2;
        for (Index index : h2Pending_)
        {
            State& state = states_[index];
            const bool up = state.record.sourceAB.identity.direction ==
                TG3::ABDirection::UpAB;
            ResolvePullback(state.record.h2,
                state.record.pullback0382ReachedAfterRejection,
                state.record.pullback0382,
                EventType::Pullback0382ReachedAfterRejection,
                up, bar, candle);
            ResolvePullback(state.record.h2Pullback0500Descriptive,
                state.record.pullback0500ReachedAfterRejection,
                state.record.pullback0500,
                EventType::Pullback0500ReachedAfterRejection,
                up, bar, candle);
            ResolvePullback(state.record.h2Pullback0618Descriptive,
                state.record.pullback0618ReachedAfterRejection,
                state.record.pullback0618,
                EventType::Pullback0618ReachedAfterRejection,
                up, bar, candle);
            if (!Terminal(state.record.h2.state) ||
                !Terminal(state.record.h2Pullback0500Descriptive.state) ||
                !Terminal(state.record.h2Pullback0618Descriptive.state))
                nextH2.push_back(index);
        }
        h2Pending_.swap(nextH2);
    }

    static void ResolvePullback(
        Outcome& outcome, std::optional<EventOccurrence>& destination,
        const FibonacciLevel& level, EventType type, bool up,
        std::size_t bar, const TG1A::Candle& candle)
    {
        if (outcome.state != OutcomeState::Pending) return;
        const bool reached = up ? candle.low <= level.zoneUpperPrice
                                : candle.high >= level.zoneLowerPrice;
        if (reached)
        {
            EventOccurrence occurrence{type, bar, candle.timestamp,
                                       up ? candle.low : candle.high};
            destination = occurrence;
            Succeed(outcome, occurrence);
        }
        else if (bar - outcome.eligibility->bar >=
                 kPrimaryEvaluationHorizonBars)
            Fail(outcome, bar, candle);
    }

    static void Censor(Outcome& outcome)
    {
        if (outcome.state != OutcomeState::Pending) return;
        outcome.state = OutcomeState::RightCensored;
        outcome.censorReason = CensorReason::StudyWindowBoundary;
    }
};

HistoricalEvaluator::HistoricalEvaluator(
    std::string symbol, TG4::EvaluationConfiguration configuration,
    HistoricalStudySpecification specification, RecordSink sink)
    : implementation_(std::make_unique<Implementation>(
          std::move(symbol), std::move(configuration), specification,
          std::move(sink))) {}

HistoricalEvaluator::HistoricalEvaluator(
    std::string symbol, TG4::EvaluationConfiguration configuration,
    RecordSink sink)
    : implementation_(std::make_unique<Implementation>(
          std::move(symbol), std::move(configuration),
          Pre2025FirstStudySpecification(), std::move(sink))) {}

HistoricalEvaluator::~HistoricalEvaluator() = default;
HistoricalEvaluator::HistoricalEvaluator(HistoricalEvaluator&&) noexcept = default;
HistoricalEvaluator& HistoricalEvaluator::operator=(HistoricalEvaluator&&) noexcept = default;
void HistoricalEvaluator::AddCompletedBar(const TG1A::Candle& candle)
{ implementation_->Add(candle); }
void HistoricalEvaluator::Finalize() { implementation_->Finalize(); }
const HistoricalDataQuality& HistoricalEvaluator::DataQuality() const
{ return implementation_->DataQuality(); }
std::size_t HistoricalEvaluator::RecordCount() const
{ return implementation_->RecordCount(); }
bool HistoricalEvaluator::IsFinalized() const
{ return implementation_->IsFinalized(); }

class HistoricalArtifactWriter::Implementation
{
public:
    Implementation(std::filesystem::path outputDirectory,
                   TG4::EvaluationConfiguration configuration,
                   HistoricalStudySpecification specification,
                   std::string baselineCommit,
                   std::vector<std::string> symbols,
                   std::string reproductionCommand)
        : outputDirectory_(std::move(outputDirectory)),
          configuration_(std::move(configuration)),
          specification_(specification),
          baselineCommit_(std::move(baselineCommit)),
          symbols_(std::move(symbols)),
          reproductionCommand_(std::move(reproductionCommand))
    {
        ValidateHistoricalStudySpecification(specification_);
    }

    void Complete()
    {
        if (complete_) return;
        std::sort(records_.begin(), records_.end(),
            [](const ObservationRecord& left, const ObservationRecord& right)
            {
                return std::tie(left.grouping.symbol, left.eventIdentity) <
                    std::tie(right.grouping.symbol, right.eventIdentity);
            });
        std::sort(audits_.begin(), audits_.end(),
            [](const HistoricalDataQuality& left,
               const HistoricalDataQuality& right)
            { return left.symbol < right.symbol; });
        std::filesystem::create_directories(outputDirectory_);
        WriteObservations();
        WriteAggregates();
        WriteCohorts();
        WriteDataQuality();
        WriteManifest();
        WriteSummary();
        complete_ = true;
    }

    std::filesystem::path outputDirectory_;
    TG4::EvaluationConfiguration configuration_;
    HistoricalStudySpecification specification_;
    std::string baselineCommit_;
    std::vector<std::string> symbols_;
    std::string reproductionCommand_;
    std::vector<ObservationRecord> records_;
    std::vector<HistoricalDataQuality> audits_;
    bool complete_ = false;

    static constexpr Endpoint kEndpoints[] = {
        Endpoint::H1Extension1618, Endpoint::H2Pullback0382,
        Endpoint::H2Pullback0500Descriptive,
        Endpoint::H2Pullback0618Descriptive};

    std::ofstream Open(std::string_view name)
    {
        std::ofstream output(outputDirectory_ / name,
                             std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error(
            "cannot create Fibonacci artifact " + std::string(name));
        return output;
    }

    AggregateAccumulator Accumulator() const
    {
        AggregateAccumulator result;
        for (const ObservationRecord& record : records_) result.Add(record);
        return result;
    }

    void WriteObservations()
    {
        auto output = Open("observations.csv");
        output << ObservationCsvHeader() << '\n';
        for (const ObservationRecord& record : records_)
            output << ObservationCsvRow(record) << '\n';
    }

    void WriteAggregates()
    {
        const AggregateAccumulator accumulator = Accumulator();
        auto output = Open("aggregate.csv");
        output << EndpointSummaryCsvHeader() << '\n';
        for (Endpoint endpoint : kEndpoints)
            output << EndpointSummaryCsvRow(
                endpoint, CountingUnit::UniqueStructuralEvent,
                accumulator.Summarize(
                    endpoint, CountingUnit::UniqueStructuralEvent)) << '\n';
    }

    void WriteCohorts()
    {
        const AggregateAccumulator accumulator = Accumulator();
        auto output = Open("cohorts.csv");
        output << "endpoint,symbol,direction,temporal_partition,"
                  "volatility_regime,tg3_confluence_0_618,";
        output << EndpointSummaryCsvHeader() << '\n';
        for (Endpoint endpoint : kEndpoints)
            for (const auto& [key, summary] : accumulator.SummarizeCohorts(
                     endpoint, CountingUnit::UniqueStructuralEvent))
                output << Detail::Csv(EndpointName(endpoint)) << ','
                       << Detail::Csv(key.symbol) << ','
                       << Detail::Csv(key.direction) << ','
                       << Detail::Csv(key.calendarPeriod) << ','
                       << Detail::Csv(key.volatilityRegime) << ','
                       << Detail::Csv(key.tg3Confluence0618) << ','
                       << EndpointSummaryCsvRow(endpoint,
                           CountingUnit::UniqueStructuralEvent, summary)
                       << '\n';
    }

    void WriteDataQuality()
    {
        auto output = Open("data_quality.csv");
        output << "symbol,excluded,exclusion_reason,source_rows,usable_rows,"
                  "material_gaps,first_timestamp,last_timestamp\n";
        for (const HistoricalDataQuality& audit : audits_)
            output << CsvLine({audit.symbol, audit.excluded ? "true" : "false",
                audit.exclusionReason, std::to_string(audit.sourceRows),
                std::to_string(audit.usableRows),
                std::to_string(audit.materialGaps),
                audit.firstTimestamp.has_value()
                    ? TG4::FormatUtcTimestamp(*audit.firstTimestamp) : "",
                audit.lastTimestamp.has_value()
                    ? TG4::FormatUtcTimestamp(*audit.lastTimestamp) : ""});
    }

    void WriteManifest()
    {
        auto output = Open("manifest.json");
        output << "{\n"
            "  \"schema_version\": \"causal-fibonacci-extension-manifest-v1\",\n"
            "  \"study_identity\": \"" << specification_.identity << "\",\n"
            "  \"baseline_commit\": \"" << baselineCommit_ << "\",\n"
            "  \"policy_id\": \"" << kStudyContractVersion << "\",\n"
            "  \"observation_schema\": \"" << kObservationSchemaVersion << "\",\n"
            "  \"aggregate_schema\": \"" << kAggregateSchemaVersion << "\",\n"
            "  \"warmup_start\": \"" << TG4::FormatUtcTimestamp(
                specification_.range.warmupStart) << "\",\n"
            "  \"score_start\": \"" << TG4::FormatUtcTimestamp(
                specification_.range.scoreStart) << "\",\n"
            "  \"score_end_exclusive\": \"" << TG4::FormatUtcTimestamp(
                specification_.range.scoreEnd) << "\",\n"
            "  \"outcome_end_exclusive\": \"" << TG4::FormatUtcTimestamp(
                specification_.range.outcomeEnd) << "\",\n"
            "  \"timeframe\": \"15m\",\n"
            "  \"configuration_fingerprint\": \""
               << TG4::ConfigurationFingerprint(configuration_) << "\",\n"
            "  \"symbols\": [";
        for (std::size_t index = 0; index < symbols_.size(); ++index)
        {
            if (index != 0) output << ", ";
            output << '"' << symbols_[index] << '"';
        }
        output << "],\n  \"reproduction_command\": \"";
        for (char character : reproductionCommand_)
        {
            if (character == '\\' || character == '"') output << '\\';
            output << character;
        }
        output << "\",\n  \"h3_status\": "
                  "\"missing_authoritative_d_extension_contract\",\n"
                  "  \"read_only_database\": true,\n"
                  "  \"uses_2025_bars\": "
               << (specification_.uses2025Bars ? "true" : "false") << "\n}\n";
    }

    void WriteSummary()
    {
        const AggregateAccumulator accumulator = Accumulator();
        auto output = Open("summary.md");
        output << "# " << specification_.summaryTitle << "\n\n"
                  "Policy: `" << kStudyContractVersion << "`\n\n"
                  "Warmup: `[" << TG4::FormatUtcTimestamp(
                      specification_.range.warmupStart) << ", "
               << TG4::FormatUtcTimestamp(specification_.range.scoreStart)
               << ")`; scoring: `[" << TG4::FormatUtcTimestamp(
                      specification_.range.scoreStart) << ", "
               << TG4::FormatUtcTimestamp(specification_.range.scoreEnd)
               << ")`; outcomes end before `" << TG4::FormatUtcTimestamp(
                      specification_.range.outcomeEnd) << "`."
               << (specification_.uses2025Bars
                       ? "\n\n"
                       : " 2025 bars were not loaded.\n\n")
               <<
                  "| Endpoint | Eligible | Success | Failure | Censored | "
                  "Ineligible | Resolved | Probability | Wilson 95% | "
                  "Bars to target (min/median/max) |\n"
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
        for (Endpoint endpoint : kEndpoints)
        {
            const EndpointSummary summary = accumulator.Summarize(
                endpoint, CountingUnit::UniqueStructuralEvent);
            output << "| " << EndpointName(endpoint) << " | "
                   << summary.eligible << " | " << summary.successes << " | "
                   << summary.failures << " | " << summary.censored << " | "
                   << summary.ambiguousOrIneligible << " | "
                   << summary.measured.denominator << " | "
                   << Optional(summary.measured.rate) << " | ["
                   << Optional(summary.measured.lower95) << ", "
                   << Optional(summary.measured.upper95) << "] | "
                   << Optional(summary.timeToTarget.minimumBars) << "/"
                   << Optional(summary.timeToTarget.medianBars) << "/"
                   << Optional(summary.timeToTarget.maximumBars) << " |\n";
        }
        const EndpointSummary h2 = accumulator.Summarize(
            Endpoint::H2Pullback0382,
            CountingUnit::UniqueStructuralEvent);
        std::size_t totalRows = 0;
        std::size_t totalGaps = 0;
        std::size_t excluded = 0;
        for (const HistoricalDataQuality& audit : audits_)
        {
            totalRows += audit.usableRows;
            totalGaps += audit.materialGaps;
            excluded += audit.excluded ? 1U : 0U;
        }
        output << "\nH2 primary expert prior: 0.80 (metadata only); "
                  "measured minus prior: "
               << Optional(h2.measuredMinusBenchmark) << ".\n\n"
                  "Data quality: " << totalRows << " usable completed bars, "
               << totalGaps << " material timestamp gaps, " << excluded
               << " excluded symbols. Gaps are retained observed-market gaps; "
                  "no synthetic bars were inserted.\n\n"
                  "H3 remains closed: "
                  "`missing_authoritative_d_extension_contract`. "
                  "Rows share market paths and overlapping structures; "
                  "intervals do not model that dependence.\n";
    }
};

HistoricalArtifactWriter::HistoricalArtifactWriter(
    std::filesystem::path outputDirectory,
    TG4::EvaluationConfiguration configuration,
    HistoricalStudySpecification specification, std::string baselineCommit,
    std::vector<std::string> symbols, std::string reproductionCommand)
    : implementation_(std::make_unique<Implementation>(
          std::move(outputDirectory), std::move(configuration),
          specification, std::move(baselineCommit), std::move(symbols),
          std::move(reproductionCommand))) {}
HistoricalArtifactWriter::~HistoricalArtifactWriter() = default;
void HistoricalArtifactWriter::AddRecord(ObservationRecord record)
{ implementation_->records_.push_back(std::move(record)); }
void HistoricalArtifactWriter::AddDataQuality(HistoricalDataQuality audit)
{ implementation_->audits_.push_back(std::move(audit)); }
void HistoricalArtifactWriter::Complete() { implementation_->Complete(); }
std::size_t HistoricalArtifactWriter::ObservationCount() const
{ return implementation_->records_.size(); }

} // namespace EA::FibonacciResearch
