#ifndef CausalTrendLineBreakRetestBehavior_hpp
#define CausalTrendLineBreakRetestBehavior_hpp

#include "CausalFractalTrendLineAngleClassification.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace EA::TG2
{

enum class BreakPolicy
{
    CompletedCloseBeyondLine,
    CompletedWickBeyondLine
};

enum class RearmPolicy
{
    CompletedBarReturnsToValidSide
};

enum class RetestContactPolicy
{
    WickReachesProjectedLineFromBrokenSide
};

enum class OuterPairingPolicy
{
    NearestCoexistingOuterBeyondBreakCandle
};

enum class ResolutionState
{
    Pending,
    Succeeded,
    Failed,
    Censored,
    StructurallyIneligible
};

enum class CensorReason
{
    None,
    EndOfInput,
    CapacityEviction
};

struct CandidateIdentity
{
    TG1A::TrendLineDirection direction = TG1A::TrendLineDirection::UTL;
    std::size_t anchor1Bar = 0;
    std::int64_t anchor1Timestamp = 0;
    std::size_t anchor2Bar = 0;
    std::int64_t anchor2Timestamp = 0;
    std::size_t creationBar = 0;
    std::int64_t creationTimestamp = 0;

    bool operator==(const CandidateIdentity&) const = default;
};

struct Configuration
{
    BreakPolicy breakPolicy = BreakPolicy::CompletedCloseBeyondLine;
    RearmPolicy rearmPolicy = RearmPolicy::CompletedBarReturnsToValidSide;
    RetestContactPolicy retestContactPolicy =
        RetestContactPolicy::WickReachesProjectedLineFromBrokenSide;
    OuterPairingPolicy outerPairingPolicy =
        OuterPairingPolicy::NearestCoexistingOuterBeyondBreakCandle;

    // Absolute price-unit tolerances. Zero means an exact comparison.
    double breakPriceTolerance = 0.0;
    double retestPriceTolerance = 0.0;
    double outerTargetPriceTolerance = 0.0;

    // Both outcome windows contain bars break+1 through break+horizon.
    std::size_t retestHorizonBars = 20;
    std::size_t outerTargetHorizonBars = 20;

    // Pending observations and retained auditable observations are separately
    // bounded. Capacity pressure is explicit censoring, never a silent drop.
    std::size_t maxActiveBreakObservations = 4096;
    std::size_t maxRetainedBreakObservations = 4096;
};

struct OutcomeResolution
{
    ResolutionState state = ResolutionState::Pending;
    std::optional<std::size_t> resolutionBar;
    std::optional<std::int64_t> resolutionTimestamp;
    std::optional<std::size_t> latencyBars;
    std::optional<double> projectedLinePrice;
    CensorReason censorReason = CensorReason::None;
};

struct BreakEvent
{
    std::uint64_t eventSequence = 0;
    CandidateIdentity candidate;
    TG1B::SteepnessClassification frozenClassification =
        TG1B::SteepnessClassification::Unclassified;
    BreakPolicy policy = BreakPolicy::CompletedCloseBeyondLine;
    std::size_t bar = 0;
    std::int64_t timestamp = 0;
    double projectedLinePrice = 0.0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double observedBreakComponent = 0.0;
    // Positive penetration is the distance beyond the line.
    double penetration = 0.0;
    // TG1A sign convention: positive valid-side, negative broken-side.
    double directionalDistance = 0.0;
};

struct PairedOuterLine
{
    CandidateIdentity candidate;
    double rawPriceSlopePerBar = 0.0;
    double anchor1Price = 0.0;
    double projectedPriceAtBreak = 0.0;
};

struct BreakObservation
{
    BreakEvent breakEvent;
    double innerRawPriceSlopePerBar = 0.0;
    double innerAnchor1Price = 0.0;
    std::optional<PairedOuterLine> pairedOuter;
    OutcomeResolution retest;
    OutcomeResolution outerTarget;
    // This outcome starts only after a retest contact and accepts target
    // contact only on a later completed bar, avoiding unknown intrabar order.
    OutcomeResolution outerTargetAfterRetest;
};

struct ContactEvent
{
    std::uint64_t breakEventSequence = 0;
    std::size_t bar = 0;
    std::int64_t timestamp = 0;
    std::size_t latencyBars = 0;
    double projectedLinePrice = 0.0;
};

struct Update
{
    std::size_t bar = 0;
    TG1B::Update classificationUpdate;
    std::vector<BreakEvent> newBreakEvents;
    std::vector<ContactEvent> newRetestContacts;
    std::vector<ContactEvent> newOuterTargetContacts;
    std::vector<ContactEvent> newOuterTargetAfterRetestContacts;
};

struct OutcomeCounts
{
    std::size_t eligible = 0;
    std::size_t resolved = 0;
    std::size_t successes = 0;
    std::size_t failures = 0;
    std::size_t censored = 0;
    std::size_t pending = 0;
    std::size_t structurallyIneligible = 0;
    std::optional<double> empiricalRate;
};

struct Summary
{
    std::size_t breakEvents = 0;
    OutcomeCounts allBreakRetests;
    std::size_t innerBreaks = 0;
    std::size_t pairedInnerBreaks = 0;
    std::size_t unpairedInnerBreaks = 0;
    OutcomeCounts allEligibleInnerToOuter;
    std::size_t retestedInnerBreaks = 0;
    std::size_t retestedPairedInnerBreaks = 0;
    OutcomeCounts retestThenOuter;
};

class TrendLineBehaviorTracker
{
public:
    explicit TrendLineBehaviorTracker(
        Configuration configuration = {}, TG1A::SeriesIdentity identity = {})
        : configuration_(configuration), identity_(std::move(identity))
    {
        ValidateConfiguration();
    }

    Update ObserveCompletedBar(
        std::size_t bar,
        const TG1A::Candle& candle,
        const std::vector<TG1B::ClassifiedTrendLineCandidate>& liveCandidates,
        TG1B::Update classificationUpdate = {})
    {
        if (finalized_)
            throw std::logic_error("TG2 cannot accept bars after finalization");
        ValidateCandle(candle);
        if (lastBar_.has_value() && bar != *lastBar_ + 1)
            throw std::invalid_argument(
                "TG2 completed bars must use consecutive bar indexes");
        if (lastTimestamp_.has_value() && candle.timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "TG2 completed bars must have unique increasing timestamps");
        lastBar_ = bar;
        lastTimestamp_ = candle.timestamp;

        Update update;
        update.bar = bar;
        update.classificationUpdate = std::move(classificationUpdate);

        ResolvePendingObservations(bar, candle, update);
        ObserveCandidateTransitions(bar, candle, liveCandidates, update);
        SynchronizeCandidateStates(liveCandidates);
        return update;
    }

    void Finalize()
    {
        if (finalized_) return;
        finalized_ = true;
        const std::size_t bar = lastBar_.value_or(0);
        const std::int64_t timestamp = lastTimestamp_.value_or(0);
        for (BreakObservation& observation : observations_)
            CensorPending(observation, CensorReason::EndOfInput,
                          bar, timestamp);
    }

    const std::vector<BreakObservation>& Observations() const
    {
        return observations_;
    }

    const Configuration& GetConfiguration() const { return configuration_; }
    const TG1A::SeriesIdentity& GetSeriesIdentity() const { return identity_; }
    bool IsFinalized() const { return finalized_; }

    std::size_t ActiveObservationCount() const
    {
        return static_cast<std::size_t>(std::count_if(
            observations_.begin(), observations_.end(), IsActive));
    }

    Summary AggregateSummary() const
    {
        Summary result = archivedSummary_;
        for (const BreakObservation& observation : observations_)
            Accumulate(result, observation);
        FinalizeRates(result);
        return result;
    }

    std::string FormatDiagnostic(const BreakObservation& observation) const
    {
        const BreakEvent& event = observation.breakEvent;
        std::ostringstream output;
        output << std::setprecision(12)
               << "symbol=" << identity_.symbol
               << ",timeframe=" << identity_.timeframe
               << ",event_sequence=" << event.eventSequence
               << ",direction=" << DirectionName(event.candidate.direction)
               << ",anchor1_bar=" << event.candidate.anchor1Bar
               << ",anchor1_timestamp="
               << FormatTimestamp(event.candidate.anchor1Timestamp)
               << ",anchor2_bar=" << event.candidate.anchor2Bar
               << ",anchor2_timestamp="
               << FormatTimestamp(event.candidate.anchor2Timestamp)
               << ",creation_bar=" << event.candidate.creationBar
               << ",creation_timestamp="
               << FormatTimestamp(event.candidate.creationTimestamp)
               << ",classification="
               << ClassificationName(event.frozenClassification)
               << ",break_bar=" << event.bar
               << ",break_timestamp=" << FormatTimestamp(event.timestamp)
               << ",break_policy=" << BreakPolicyName(event.policy)
               << ",break_tolerance="
               << configuration_.breakPriceTolerance
               << ",break_projection=" << event.projectedLinePrice
               << ",open=" << event.open
               << ",high=" << event.high
               << ",low=" << event.low
               << ",close=" << event.close
               << ",break_component=" << event.observedBreakComponent
               << ",penetration=" << event.penetration
               << ",directional_distance=" << event.directionalDistance
               << ",retest_policy="
               << RetestPolicyName(configuration_.retestContactPolicy)
               << ",retest_contact_component=high_low_range"
               << ",retest_tolerance="
               << configuration_.retestPriceTolerance
               << ",retest_horizon_bars="
               << configuration_.retestHorizonBars
               << ",retest=" << ResolutionName(observation.retest.state)
               << ",retest_latency_bars="
               << OptionalSize(observation.retest.latencyBars)
               << ",retest_resolution_bar="
               << OptionalSize(observation.retest.resolutionBar)
               << ",retest_resolution_timestamp="
               << OptionalTimestamp(
                      observation.retest.resolutionTimestamp)
               << ",retest_contact_projection="
               << OptionalNumber(observation.retest.projectedLinePrice)
               << ",outer_pairing_policy="
               << OuterPairingPolicyName(configuration_.outerPairingPolicy)
               << ",paired_outer="
               << (observation.pairedOuter.has_value() ? "yes" : "no");
        if (observation.pairedOuter.has_value())
        {
            const PairedOuterLine& outer = *observation.pairedOuter;
            output << ",outer_anchor1_bar=" << outer.candidate.anchor1Bar
                   << ",outer_anchor1_timestamp="
                   << FormatTimestamp(outer.candidate.anchor1Timestamp)
                   << ",outer_anchor2_bar=" << outer.candidate.anchor2Bar
                   << ",outer_anchor2_timestamp="
                   << FormatTimestamp(outer.candidate.anchor2Timestamp)
                   << ",outer_creation_bar=" << outer.candidate.creationBar
                   << ",outer_creation_timestamp="
                   << FormatTimestamp(outer.candidate.creationTimestamp)
                   << ",outer_projection_at_break="
                   << outer.projectedPriceAtBreak;
        }
        output << ",outer_tolerance="
               << configuration_.outerTargetPriceTolerance
               << ",outer_contact_component="
               << (event.candidate.direction ==
                           TG1A::TrendLineDirection::UTL
                       ? "low" : "high")
               << ",outer_horizon_bars="
               << configuration_.outerTargetHorizonBars
               << ",outer_outcome="
               << ResolutionName(observation.outerTarget.state)
               << ",outer_latency_bars="
               << OptionalSize(observation.outerTarget.latencyBars)
               << ",outer_resolution_bar="
               << OptionalSize(observation.outerTarget.resolutionBar)
               << ",outer_resolution_timestamp="
               << OptionalTimestamp(
                      observation.outerTarget.resolutionTimestamp)
               << ",outer_contact_projection="
               << OptionalNumber(observation.outerTarget.projectedLinePrice)
               << ",retest_then_outer_outcome="
               << ResolutionName(observation.outerTargetAfterRetest.state)
               << ",retest_then_outer_latency_bars="
               << OptionalSize(
                      observation.outerTargetAfterRetest.latencyBars)
               << ",retest_then_outer_resolution_bar="
               << OptionalSize(
                      observation.outerTargetAfterRetest.resolutionBar)
               << ",retest_then_outer_resolution_timestamp="
               << OptionalTimestamp(
                      observation.outerTargetAfterRetest.resolutionTimestamp)
               << ",retest_censor_reason="
               << CensorReasonName(observation.retest.censorReason)
               << ",outer_censor_reason="
               << CensorReasonName(observation.outerTarget.censorReason);
        return output.str();
    }

private:
    using IdentityKey =
        std::tuple<int, std::size_t, std::size_t, std::size_t>;

    struct CandidateState
    {
        CandidateIdentity identity;
        bool armed = false;
    };

    Configuration configuration_;
    TG1A::SeriesIdentity identity_;
    std::vector<CandidateState> candidateStates_;
    std::vector<BreakObservation> observations_;
    Summary archivedSummary_;
    std::uint64_t nextEventSequence_ = 1;
    std::optional<std::size_t> lastBar_;
    std::optional<std::int64_t> lastTimestamp_;
    bool finalized_ = false;

    static bool FiniteNonnegative(double value)
    {
        return std::isfinite(value) && value >= 0.0;
    }

    void ValidateConfiguration() const
    {
        if (!FiniteNonnegative(configuration_.breakPriceTolerance) ||
            !FiniteNonnegative(configuration_.retestPriceTolerance) ||
            !FiniteNonnegative(configuration_.outerTargetPriceTolerance))
            throw std::invalid_argument(
                "TG2 tolerances must be finite and nonnegative");
        if (configuration_.retestHorizonBars == 0 ||
            configuration_.outerTargetHorizonBars == 0 ||
            configuration_.retestHorizonBars >
                configuration_.outerTargetHorizonBars)
            throw std::invalid_argument(
                "TG2 horizons must be positive and retest horizon must not "
                "exceed outer-target horizon");
        if (configuration_.maxActiveBreakObservations == 0 ||
            configuration_.maxRetainedBreakObservations == 0)
            throw std::invalid_argument("TG2 state bounds must be positive");
    }

    static void ValidateCandle(const TG1A::Candle& candle)
    {
        if (!std::isfinite(candle.open) || !std::isfinite(candle.high) ||
            !std::isfinite(candle.low) || !std::isfinite(candle.close) ||
            candle.high < candle.low || candle.open < candle.low ||
            candle.open > candle.high || candle.close < candle.low ||
            candle.close > candle.high)
            throw std::invalid_argument("TG2 requires finite valid OHLC bars");
    }

    static CandidateIdentity Identity(
        const TG1B::ClassifiedTrendLineCandidate& candidate)
    {
        return {candidate.direction,
                candidate.anchor1Bar, candidate.anchor1Timestamp,
                candidate.anchor2Bar, candidate.anchor2Timestamp,
                candidate.creationBar, candidate.creationTimestamp};
    }

    static IdentityKey Key(const CandidateIdentity& identity)
    {
        return {static_cast<int>(identity.direction), identity.anchor1Bar,
                identity.anchor2Bar, identity.creationBar};
    }

    static IdentityKey Key(
        const TG1B::ClassifiedTrendLineCandidate& candidate)
    {
        return {static_cast<int>(candidate.direction), candidate.anchor1Bar,
                candidate.anchor2Bar, candidate.creationBar};
    }

    static double ProjectedPrice(
        const TG1B::ClassifiedTrendLineCandidate& candidate,
        std::size_t bar)
    {
        return candidate.anchor1Price + candidate.rawPriceSlopePerBar *
            (static_cast<double>(bar) -
             static_cast<double>(candidate.anchor1Bar));
    }

    static double ProjectedPrice(const BreakObservation& observation,
                                 std::size_t bar)
    {
        return observation.innerAnchor1Price +
            observation.innerRawPriceSlopePerBar *
            (static_cast<double>(bar) - static_cast<double>(
                observation.breakEvent.candidate.anchor1Bar));
    }

    static double ProjectedPrice(const PairedOuterLine& outer,
                                 std::size_t bar)
    {
        return outer.anchor1Price + outer.rawPriceSlopePerBar *
            (static_cast<double>(bar) -
             static_cast<double>(outer.candidate.anchor1Bar));
    }

    double BreakComponent(
        const TG1B::ClassifiedTrendLineCandidate& candidate,
        const TG1A::Candle& candle) const
    {
        if (configuration_.breakPolicy ==
            BreakPolicy::CompletedCloseBeyondLine)
            return candle.close;
        return candidate.direction == TG1A::TrendLineDirection::UTL
            ? candle.low : candle.high;
    }

    bool IsBroken(const TG1B::ClassifiedTrendLineCandidate& candidate,
                  std::size_t bar, const TG1A::Candle& candle) const
    {
        const double projected = ProjectedPrice(candidate, bar);
        const double observed = BreakComponent(candidate, candle);
        if (candidate.direction == TG1A::TrendLineDirection::UTL)
            return observed < projected - configuration_.breakPriceTolerance;
        return observed > projected + configuration_.breakPriceTolerance;
    }

    static bool SameIdentity(
        const CandidateState& state,
        const TG1B::ClassifiedTrendLineCandidate& candidate)
    {
        return Key(state.identity) == Key(candidate);
    }

    CandidateState* FindState(
        const TG1B::ClassifiedTrendLineCandidate& candidate)
    {
        const auto found = std::find_if(
            candidateStates_.begin(), candidateStates_.end(),
            [&candidate](const CandidateState& state)
            {
                return SameIdentity(state, candidate);
            });
        return found == candidateStates_.end() ? nullptr : &*found;
    }

    void ObserveCandidateTransitions(
        std::size_t bar,
        const TG1A::Candle& candle,
        const std::vector<TG1B::ClassifiedTrendLineCandidate>& liveCandidates,
        Update& update)
    {
        for (const TG1B::ClassifiedTrendLineCandidate& candidate :
             liveCandidates)
        {
            CandidateState* state = FindState(candidate);
            const bool broken = IsBroken(candidate, bar, candle);
            if (state == nullptr)
            {
                candidateStates_.push_back({Identity(candidate), !broken});
                continue; // Creation/current discovery is not a transition.
            }
            if (state->armed && broken)
            {
                state->armed = false;
                EmitBreak(bar, candle, candidate, liveCandidates, update);
            }
            else if (!state->armed && !broken)
            {
                // The named re-arm policy requires one completed valid-side
                // observation; no candidate expiry or time passage re-arms it.
                state->armed = true;
            }
        }
    }

    void SynchronizeCandidateStates(
        const std::vector<TG1B::ClassifiedTrendLineCandidate>& liveCandidates)
    {
        candidateStates_.erase(
            std::remove_if(candidateStates_.begin(), candidateStates_.end(),
                [&liveCandidates](const CandidateState& state)
                {
                    return std::none_of(
                        liveCandidates.begin(), liveCandidates.end(),
                        [&state](
                            const TG1B::ClassifiedTrendLineCandidate& candidate)
                        {
                            return Key(state.identity) == Key(candidate);
                        });
                }),
            candidateStates_.end());
        std::sort(candidateStates_.begin(), candidateStates_.end(),
            [](const CandidateState& left, const CandidateState& right)
            {
                return Key(left.identity) < Key(right.identity);
            });
    }

    void EmitBreak(
        std::size_t bar,
        const TG1A::Candle& candle,
        const TG1B::ClassifiedTrendLineCandidate& candidate,
        const std::vector<TG1B::ClassifiedTrendLineCandidate>& liveCandidates,
        Update& update)
    {
        EnsureCapacity(bar, candle.timestamp);
        const double projected = ProjectedPrice(candidate, bar);
        const double observed = BreakComponent(candidate, candle);
        const double directionalDistance =
            candidate.direction == TG1A::TrendLineDirection::UTL
                ? observed - projected : projected - observed;
        BreakEvent event;
        event.eventSequence = nextEventSequence_++;
        event.candidate = Identity(candidate);
        event.frozenClassification = candidate.classification;
        event.policy = configuration_.breakPolicy;
        event.bar = bar;
        event.timestamp = candle.timestamp;
        event.projectedLinePrice = projected;
        event.open = candle.open;
        event.high = candle.high;
        event.low = candle.low;
        event.close = candle.close;
        event.observedBreakComponent = observed;
        event.penetration = -directionalDistance;
        event.directionalDistance = directionalDistance;

        BreakObservation observation;
        observation.breakEvent = event;
        observation.innerRawPriceSlopePerBar = candidate.rawPriceSlopePerBar;
        observation.innerAnchor1Price = candidate.anchor1Price;
        observation.retest.state = ResolutionState::Pending;
        observation.outerTargetAfterRetest.state =
            ResolutionState::StructurallyIneligible;
        if (candidate.classification == TG1B::SteepnessClassification::Inner)
        {
            observation.pairedOuter = PairOuter(
                candidate, bar, candle, liveCandidates);
            observation.outerTarget.state = observation.pairedOuter.has_value()
                ? ResolutionState::Pending
                : ResolutionState::StructurallyIneligible;
        }
        else
        {
            observation.outerTarget.state =
                ResolutionState::StructurallyIneligible;
        }
        observations_.push_back(observation);
        update.newBreakEvents.push_back(std::move(event));
    }

    std::optional<PairedOuterLine> PairOuter(
        const TG1B::ClassifiedTrendLineCandidate& inner,
        std::size_t bar,
        const TG1A::Candle& candle,
        const std::vector<TG1B::ClassifiedTrendLineCandidate>& liveCandidates)
        const
    {
        const double innerProjection = ProjectedPrice(inner, bar);
        const TG1B::ClassifiedTrendLineCandidate* best = nullptr;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (const TG1B::ClassifiedTrendLineCandidate& outer : liveCandidates)
        {
            if (outer.direction != inner.direction ||
                outer.classification != TG1B::SteepnessClassification::Outer ||
                outer.creationBar > bar)
                continue;
            const double outerProjection = ProjectedPrice(outer, bar);
            bool eligible = false;
            if (inner.direction == TG1A::TrendLineDirection::UTL)
            {
                eligible = outerProjection < innerProjection &&
                    candle.low > outerProjection +
                        configuration_.outerTargetPriceTolerance;
            }
            else
            {
                eligible = outerProjection > innerProjection &&
                    candle.high < outerProjection -
                        configuration_.outerTargetPriceTolerance;
            }
            if (!eligible) continue;
            const double distance = std::fabs(
                innerProjection - outerProjection);
            if (best == nullptr || distance < bestDistance ||
                (distance == bestDistance && Key(outer) < Key(*best)))
            {
                best = &outer;
                bestDistance = distance;
            }
        }
        if (best == nullptr) return std::nullopt;
        return PairedOuterLine{Identity(*best), best->rawPriceSlopePerBar,
                               best->anchor1Price,
                               ProjectedPrice(*best, bar)};
    }

    bool RetestContact(const BreakObservation& observation,
                       std::size_t bar, const TG1A::Candle& candle,
                       double projected) const
    {
        (void)observation;
        (void)bar;
        // A retest is actual candle-range contact with the tolerance band.
        // A gap wholly across the line is not silently called wick contact.
        return candle.high >=
                   projected - configuration_.retestPriceTolerance &&
               candle.low <=
                   projected + configuration_.retestPriceTolerance;
    }

    bool OuterContact(const BreakObservation& observation,
                      const TG1A::Candle& candle, double projected) const
    {
        if (observation.breakEvent.candidate.direction ==
            TG1A::TrendLineDirection::UTL)
            return candle.low <=
                projected + configuration_.outerTargetPriceTolerance;
        return candle.high >=
            projected - configuration_.outerTargetPriceTolerance;
    }

    static void ResolveSuccess(OutcomeResolution& outcome,
                               std::size_t bar, std::int64_t timestamp,
                               std::size_t breakBar, double projected)
    {
        outcome.state = ResolutionState::Succeeded;
        outcome.resolutionBar = bar;
        outcome.resolutionTimestamp = timestamp;
        outcome.latencyBars = bar - breakBar;
        outcome.projectedLinePrice = projected;
    }

    static void ResolveFailure(OutcomeResolution& outcome,
                               std::size_t bar, std::int64_t timestamp,
                               std::size_t breakBar)
    {
        outcome.state = ResolutionState::Failed;
        outcome.resolutionBar = bar;
        outcome.resolutionTimestamp = timestamp;
        outcome.latencyBars = bar - breakBar;
    }

    void ResolvePendingObservations(std::size_t bar,
                                    const TG1A::Candle& candle,
                                    Update& update)
    {
        for (BreakObservation& observation : observations_)
        {
            const std::size_t breakBar = observation.breakEvent.bar;
            if (bar <= breakBar) continue;
            const std::size_t retestDeadline =
                breakBar + configuration_.retestHorizonBars;
            const std::size_t outerDeadline =
                breakBar + configuration_.outerTargetHorizonBars;

            bool retestedNow = false;
            if (observation.retest.state == ResolutionState::Pending &&
                bar <= retestDeadline)
            {
                const double projected = ProjectedPrice(observation, bar);
                if (RetestContact(observation, bar, candle, projected))
                {
                    ResolveSuccess(observation.retest, bar, candle.timestamp,
                                   breakBar, projected);
                    retestedNow = true;
                    update.newRetestContacts.push_back(
                        {observation.breakEvent.eventSequence, bar,
                         candle.timestamp, bar - breakBar, projected});
                    if (observation.pairedOuter.has_value())
                    {
                        observation.outerTargetAfterRetest.state =
                            bar < outerDeadline ? ResolutionState::Pending
                                                : ResolutionState::Failed;
                        if (bar >= outerDeadline)
                            ResolveFailure(
                                observation.outerTargetAfterRetest, bar,
                                candle.timestamp, breakBar);
                    }
                }
                else if (bar == retestDeadline)
                {
                    ResolveFailure(observation.retest, bar,
                                   candle.timestamp, breakBar);
                }
            }

            if (observation.outerTarget.state == ResolutionState::Pending &&
                bar <= outerDeadline && observation.pairedOuter.has_value())
            {
                const double projected =
                    ProjectedPrice(*observation.pairedOuter, bar);
                if (OuterContact(observation, candle, projected))
                {
                    ResolveSuccess(observation.outerTarget, bar,
                                   candle.timestamp, breakBar, projected);
                    update.newOuterTargetContacts.push_back(
                        {observation.breakEvent.eventSequence, bar,
                         candle.timestamp, bar - breakBar, projected});
                }
                else if (bar == outerDeadline)
                {
                    ResolveFailure(observation.outerTarget, bar,
                                   candle.timestamp, breakBar);
                }
            }

            // Same-bar retest/target contact is not ordered causally. The
            // conditioned outcome starts on the next completed bar.
            if (!retestedNow &&
                observation.outerTargetAfterRetest.state ==
                    ResolutionState::Pending &&
                bar <= outerDeadline && observation.pairedOuter.has_value())
            {
                const double projected =
                    ProjectedPrice(*observation.pairedOuter, bar);
                if (OuterContact(observation, candle, projected))
                {
                    ResolveSuccess(observation.outerTargetAfterRetest, bar,
                                   candle.timestamp, breakBar, projected);
                    update.newOuterTargetAfterRetestContacts.push_back(
                        {observation.breakEvent.eventSequence, bar,
                         candle.timestamp, bar - breakBar, projected});
                }
                else if (bar == outerDeadline)
                {
                    ResolveFailure(observation.outerTargetAfterRetest, bar,
                                   candle.timestamp, breakBar);
                }
            }
        }
    }

    static bool IsActive(const BreakObservation& observation)
    {
        return observation.retest.state == ResolutionState::Pending ||
            observation.outerTarget.state == ResolutionState::Pending ||
            observation.outerTargetAfterRetest.state == ResolutionState::Pending;
    }

    static void Censor(OutcomeResolution& outcome, CensorReason reason,
                       std::size_t bar, std::int64_t timestamp,
                       std::size_t breakBar)
    {
        if (outcome.state != ResolutionState::Pending) return;
        outcome.state = ResolutionState::Censored;
        outcome.censorReason = reason;
        outcome.resolutionBar = bar;
        outcome.resolutionTimestamp = timestamp;
        outcome.latencyBars = bar >= breakBar
            ? std::optional<std::size_t>(bar - breakBar) : std::nullopt;
    }

    static void CensorPending(BreakObservation& observation,
                              CensorReason reason,
                              std::size_t bar, std::int64_t timestamp)
    {
        const std::size_t breakBar = observation.breakEvent.bar;
        Censor(observation.retest, reason, bar, timestamp, breakBar);
        Censor(observation.outerTarget, reason, bar, timestamp, breakBar);
        Censor(observation.outerTargetAfterRetest,
               reason, bar, timestamp, breakBar);
    }

    void ArchiveAndErase(std::size_t index,
                         CensorReason reason,
                         std::size_t bar,
                         std::int64_t timestamp)
    {
        CensorPending(observations_[index], reason, bar, timestamp);
        Accumulate(archivedSummary_, observations_[index]);
        observations_.erase(observations_.begin() +
                            static_cast<std::ptrdiff_t>(index));
    }

    void EnsureCapacity(std::size_t bar, std::int64_t timestamp)
    {
        while (ActiveObservationCount() >=
               configuration_.maxActiveBreakObservations)
        {
            const auto active = std::find_if(
                observations_.begin(), observations_.end(), IsActive);
            if (active == observations_.end()) break;
            ArchiveAndErase(static_cast<std::size_t>(
                std::distance(observations_.begin(), active)),
                CensorReason::CapacityEviction, bar, timestamp);
        }
        while (observations_.size() >=
               configuration_.maxRetainedBreakObservations)
            ArchiveAndErase(0, CensorReason::CapacityEviction,
                            bar, timestamp);
    }

    static void AddOutcome(OutcomeCounts& counts,
                           const OutcomeResolution& outcome)
    {
        if (outcome.state == ResolutionState::StructurallyIneligible)
        {
            ++counts.structurallyIneligible;
            return;
        }
        ++counts.eligible;
        switch (outcome.state)
        {
            case ResolutionState::Succeeded:
                ++counts.successes;
                ++counts.resolved;
                break;
            case ResolutionState::Failed:
                ++counts.failures;
                ++counts.resolved;
                break;
            case ResolutionState::Censored: ++counts.censored; break;
            case ResolutionState::Pending: ++counts.pending; break;
            case ResolutionState::StructurallyIneligible: break;
        }
    }

    static void Accumulate(Summary& summary,
                           const BreakObservation& observation)
    {
        ++summary.breakEvents;
        AddOutcome(summary.allBreakRetests, observation.retest);
        if (observation.breakEvent.frozenClassification !=
            TG1B::SteepnessClassification::Inner)
            return;
        ++summary.innerBreaks;
        if (observation.pairedOuter.has_value())
            ++summary.pairedInnerBreaks;
        else
            ++summary.unpairedInnerBreaks;
        AddOutcome(summary.allEligibleInnerToOuter,
                   observation.outerTarget);
        if (observation.retest.state == ResolutionState::Succeeded)
        {
            ++summary.retestedInnerBreaks;
            if (observation.pairedOuter.has_value())
            {
                ++summary.retestedPairedInnerBreaks;
                AddOutcome(summary.retestThenOuter,
                           observation.outerTargetAfterRetest);
            }
        }
    }

    static void SetRate(OutcomeCounts& counts)
    {
        const std::size_t denominator = counts.successes + counts.failures;
        counts.empiricalRate = denominator == 0
            ? std::nullopt
            : std::optional<double>(
                static_cast<double>(counts.successes) /
                static_cast<double>(denominator));
    }

    static void FinalizeRates(Summary& summary)
    {
        SetRate(summary.allBreakRetests);
        SetRate(summary.allEligibleInnerToOuter);
        SetRate(summary.retestThenOuter);
    }

    static const char* DirectionName(TG1A::TrendLineDirection direction)
    {
        return direction == TG1A::TrendLineDirection::UTL ? "UTL" : "DTL";
    }

    static const char* ClassificationName(
        TG1B::SteepnessClassification classification)
    {
        switch (classification)
        {
            case TG1B::SteepnessClassification::LongTerm: return "LongTerm";
            case TG1B::SteepnessClassification::Outer: return "Outer";
            case TG1B::SteepnessClassification::Inner: return "Inner";
            case TG1B::SteepnessClassification::Unclassified:
                return "Unclassified";
        }
        return "Unclassified";
    }

    static const char* BreakPolicyName(BreakPolicy policy)
    {
        return policy == BreakPolicy::CompletedCloseBeyondLine
            ? "completed_close_beyond_line"
            : "completed_wick_beyond_line";
    }

    static const char* RetestPolicyName(RetestContactPolicy)
    {
        return "wick_reaches_projected_line_from_broken_side";
    }

    static const char* OuterPairingPolicyName(OuterPairingPolicy)
    {
        return "nearest_coexisting_outer_beyond_break_candle";
    }

    static const char* ResolutionName(ResolutionState state)
    {
        switch (state)
        {
            case ResolutionState::Pending: return "pending";
            case ResolutionState::Succeeded: return "succeeded";
            case ResolutionState::Failed: return "failed";
            case ResolutionState::Censored: return "censored";
            case ResolutionState::StructurallyIneligible:
                return "structurally_ineligible";
        }
        return "pending";
    }

    static const char* CensorReasonName(CensorReason reason)
    {
        switch (reason)
        {
            case CensorReason::None: return "none";
            case CensorReason::EndOfInput: return "end_of_input";
            case CensorReason::CapacityEviction: return "capacity_eviction";
        }
        return "none";
    }

    static std::string OptionalSize(const std::optional<std::size_t>& value)
    {
        return value.has_value() ? std::to_string(*value) : "unavailable";
    }

    static std::string OptionalNumber(const std::optional<double>& value)
    {
        if (!value.has_value()) return "unavailable";
        std::ostringstream output;
        output << std::setprecision(12) << *value;
        return output.str();
    }

    static std::string OptionalTimestamp(
        const std::optional<std::int64_t>& value)
    {
        return value.has_value() ? FormatTimestamp(*value) : "unavailable";
    }

    static std::string FormatTimestamp(std::int64_t epochSeconds)
    {
        const std::time_t raw = static_cast<std::time_t>(epochSeconds);
        std::tm utc{};
        if (gmtime_r(&raw, &utc) == nullptr)
            return std::to_string(epochSeconds);
        std::ostringstream output;
        output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
        return output.str();
    }
};

class CausalTrendLineBreakRetestBehavior
{
public:
    explicit CausalTrendLineBreakRetestBehavior(
        TG1B::CalibrationConfiguration calibration,
        TG1A::Configuration geometryConfiguration = {},
        Configuration behaviorConfiguration = {},
        TG1A::SeriesIdentity identity = {})
        : classification_(calibration, geometryConfiguration, identity),
          tracker_(behaviorConfiguration, std::move(identity))
    {
    }

    static CausalTrendLineBreakRetestBehavior FromHistorical(
        std::vector<TG1A::Candle> candles,
        TG1B::CalibrationConfiguration calibration,
        TG1A::Configuration geometryConfiguration = {},
        Configuration behaviorConfiguration = {},
        TG1A::SeriesIdentity identity = {})
    {
        std::stable_sort(candles.begin(), candles.end(),
            [](const TG1A::Candle& left, const TG1A::Candle& right)
            {
                return left.timestamp < right.timestamp;
            });
        CausalTrendLineBreakRetestBehavior result(
            calibration, geometryConfiguration, behaviorConfiguration,
            std::move(identity));
        for (const TG1A::Candle& candle : candles)
            result.AddCompletedBar(candle);
        result.Finalize();
        return result;
    }

    Update AddCompletedBar(const TG1A::Candle& candle)
    {
        if (tracker_.IsFinalized())
            throw std::logic_error("TG2 cannot accept bars after finalization");
        TG1B::Update classificationUpdate =
            classification_.AddCompletedBar(candle);
        return tracker_.ObserveCompletedBar(
            classificationUpdate.bar, candle,
            classification_.ClassifiedCandidates(),
            std::move(classificationUpdate));
    }

    void Finalize() { tracker_.Finalize(); }

    const std::vector<BreakObservation>& Observations() const
    {
        return tracker_.Observations();
    }

    Summary AggregateSummary() const { return tracker_.AggregateSummary(); }

    std::string FormatDiagnostic(const BreakObservation& observation) const
    {
        return tracker_.FormatDiagnostic(observation);
    }

    const TG1B::CausalFractalTrendLineAngleClassification& Classification()
        const
    {
        return classification_;
    }

    const TrendLineBehaviorTracker& Tracker() const { return tracker_; }

private:
    TG1B::CausalFractalTrendLineAngleClassification classification_;
    TrendLineBehaviorTracker tracker_;
};

} // namespace EA::TG2

#endif /* CausalTrendLineBreakRetestBehavior_hpp */
