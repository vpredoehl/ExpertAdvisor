#ifndef CausalFibonacciConfluenceIntegration_hpp
#define CausalFibonacciConfluenceIntegration_hpp

#include "CausalTrendLineBreakRetestBehavior.hpp"

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

namespace EA::TG3
{

enum class ABDirection
{
    UpAB,
    DownAB
};

// The source material does not define A/B selection. This deliberately named
// convention pairs each newly confirmed B pivot with the most recent earlier
// confirmed opposite-kind pivot that gives the requested price direction.
enum class AnchorSelectionPolicy
{
    MostRecentPriorOppositeConfirmedFractal
};

enum class FibonacciLevelType
{
    Retracement
};

// The source mentions UTL/up-AB only. Symmetric DTL/down-AB measurement is an
// opt-in diagnostic hypothesis and is never enabled implicitly.
enum class DirectionalStudyPolicy
{
    SourceUTLUpABOnly,
    SymmetricDirectionalDiagnostic
};

enum class ConfluencePolicy
{
    AbsolutePriceToleranceAroundExactRetracementLevel
};

enum class ConfluenceState
{
    Confluence,
    NoConfluence,
    StructurallyIneligible
};

enum class IneligibleReason
{
    None,
    NoTG2PairedOuter,
    DirectionUnsupportedByStudy,
    NoCausallyEligibleAB
};

struct ABIdentity
{
    ABDirection direction = ABDirection::UpAB;
    std::size_t aBar = 0;
    std::int64_t aTimestamp = 0;
    std::size_t bBar = 0;
    std::int64_t bTimestamp = 0;
    std::size_t availabilityBar = 0;
    std::int64_t availabilityTimestamp = 0;

    bool operator==(const ABIdentity&) const = default;
};

struct ABStructure
{
    ABIdentity identity;
    double aPrice = 0.0;
    std::size_t aConfirmationBar = 0;
    std::int64_t aConfirmationTimestamp = 0;
    double bPrice = 0.0;
    std::size_t bConfirmationBar = 0;
    std::int64_t bConfirmationTimestamp = 0;
    double priceRange = 0.0;
    AnchorSelectionPolicy anchorSelectionPolicy =
        AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal;
};

struct FibonacciLevel
{
    FibonacciLevelType type = FibonacciLevelType::Retracement;
    double ratio = 0.0;
    double price = 0.0;
};

struct LevelMeasurement
{
    double ratio = 0.0;
    double levelPrice = 0.0;
    double zoneLowerPrice = 0.0;
    double zoneUpperPrice = 0.0;
    double rawPriceDistance = 0.0;
    bool matched = false;
};

struct Configuration
{
    // Ratios are explicit caller configuration because the audited source
    // supplies no ratio set. Only retracements in [0, 1] are implemented.
    std::vector<double> retracementRatios;
    double absolutePriceTolerance = 0.0;
    AnchorSelectionPolicy anchorSelectionPolicy =
        AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal;
    DirectionalStudyPolicy directionalStudyPolicy =
        DirectionalStudyPolicy::SourceUTLUpABOnly;
    ConfluencePolicy confluencePolicy =
        ConfluencePolicy::AbsolutePriceToleranceAroundExactRetracementLevel;

    std::size_t maxConfirmedFractalsPerKind = 128;
    std::size_t maxABAgeBars = 2048;
    std::size_t maxActiveABStructures = 512;
    std::size_t maxActiveConfluenceObservations = 4096;
    std::size_t maxRetainedConfluenceObservations = 4096;
};

struct ConfluenceObservation
{
    std::uint64_t breakEventSequence = 0;
    TG2::CandidateIdentity innerCandidate;
    TG1B::SteepnessClassification frozenClassification =
        TG1B::SteepnessClassification::Unclassified;
    std::size_t innerBreakBar = 0;
    std::int64_t innerBreakTimestamp = 0;
    std::optional<TG2::PairedOuterLine> pairedOuter;
    std::optional<ABStructure> selectedAB;
    std::size_t observationBar = 0;
    std::int64_t observationTimestamp = 0;
    std::optional<double> outerProjectedPriceAtObservation;
    std::optional<double> observationAtr;
    std::vector<LevelMeasurement> levels;
    std::vector<double> matchedRatios;
    ConfluenceState confluenceState = ConfluenceState::StructurallyIneligible;
    IneligibleReason ineligibleReason = IneligibleReason::None;
    std::optional<double> minimumRawPriceDistance;
    std::optional<double> minimumAtrNormalizedDistance;
    AnchorSelectionPolicy anchorSelectionPolicy =
        AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal;
    DirectionalStudyPolicy directionalStudyPolicy =
        DirectionalStudyPolicy::SourceUTLUpABOnly;
    ConfluencePolicy confluencePolicy =
        ConfluencePolicy::AbsolutePriceToleranceAroundExactRetracementLevel;
    double absolutePriceTolerance = 0.0;

    // These are synchronized snapshots of TG2 outcomes. Synchronization may
    // update outcome state, but never the immutable confluence classification.
    TG2::OutcomeResolution retest;
    TG2::OutcomeResolution outerTarget;
    TG2::OutcomeResolution outerTargetAfterRetest;
};

struct ConfluenceGroupCounts
{
    std::size_t observations = 0;
    std::size_t retestedObservations = 0;
    TG2::OutcomeCounts outerTarget;
    TG2::OutcomeCounts retestThenOuter;
};

struct Summary
{
    std::size_t innerBreakObservations = 0;
    std::size_t eligibleWithConfluence = 0;
    std::size_t eligibleWithoutConfluence = 0;
    std::size_t structurallyIneligible = 0;
    std::size_t noPairedOuter = 0;
    std::size_t noEligibleAB = 0;
    std::size_t unsupportedDirection = 0;
    std::size_t capacityEvictedObservations = 0;
    ConfluenceGroupCounts confluence;
    ConfluenceGroupCounts noConfluence;
    ConfluenceGroupCounts ineligible;
};

struct Update
{
    std::size_t bar = 0;
    TG2::Update behaviorUpdate;
    std::vector<ABIdentity> newlyAvailableABStructures;
    std::vector<std::uint64_t> newConfluenceObservations;
    // Captured at creation time so a streaming consumer can classify every
    // same-bar event before later outcome synchronization or retention limits
    // can affect the tracker-visible observation set.
    std::vector<ConfluenceObservation> newlyCreatedConfluenceObservations;
};

class FibonacciConfluenceTracker
{
public:
    explicit FibonacciConfluenceTracker(
        Configuration configuration,
        TG1A::SeriesIdentity identity = {})
        : configuration_(std::move(configuration)),
          identity_(std::move(identity))
    {
        ValidateAndNormalizeConfiguration();
    }

    void Advance(std::size_t bar, std::int64_t timestamp)
    {
        if (lastBar_.has_value() && bar != *lastBar_ + 1)
            throw std::invalid_argument(
                "TG3 completed bars must use consecutive bar indexes");
        if (lastTimestamp_.has_value() && timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "TG3 completed bars must have unique increasing timestamps");
        lastBar_ = bar;
        lastTimestamp_ = timestamp;
        ExpireABStructures(bar);
    }

    std::vector<ABIdentity> ObserveConfirmedFractals(
        std::vector<TG1A::ConfirmedFractal> fractals)
    {
        std::sort(fractals.begin(), fractals.end(), FractalLess);
        std::vector<ABIdentity> created;
        for (const TG1A::ConfirmedFractal& fractal : fractals)
        {
            if (!std::isfinite(fractal.price) ||
                fractal.confirmationBar < fractal.anchorBar ||
                fractal.confirmationTimestamp < fractal.anchorTimestamp)
                throw std::invalid_argument(
                    "TG3 requires finite causally ordered confirmed fractals");
            if (!lastBar_.has_value() || !lastTimestamp_.has_value() ||
                fractal.confirmationBar != *lastBar_ ||
                fractal.confirmationTimestamp != *lastTimestamp_)
                throw std::invalid_argument(
                    "TG3 newly confirmed fractals must be observed exactly at causal confirmation");
            if (AlreadyStored(fractal)) continue;

            const std::optional<TG1A::ConfirmedFractal> a =
                SelectAnchorFor(fractal);
            if (a.has_value())
            {
                const ABDirection direction =
                    fractal.kind == TG1A::FractalKind::High
                    ? ABDirection::UpAB : ABDirection::DownAB;
                ABStructure structure;
                structure.identity = {
                    direction,
                    a->anchorBar, a->anchorTimestamp,
                    fractal.anchorBar, fractal.anchorTimestamp,
                    fractal.confirmationBar, fractal.confirmationTimestamp};
                structure.aPrice = a->price;
                structure.aConfirmationBar = a->confirmationBar;
                structure.aConfirmationTimestamp = a->confirmationTimestamp;
                structure.bPrice = fractal.price;
                structure.bConfirmationBar = fractal.confirmationBar;
                structure.bConfirmationTimestamp =
                    fractal.confirmationTimestamp;
                structure.priceRange = std::fabs(
                    fractal.price - a->price);
                structure.anchorSelectionPolicy =
                    configuration_.anchorSelectionPolicy;
                if (structure.priceRange > 0.0 &&
                    std::isfinite(structure.priceRange))
                {
                    abStructures_.push_back(structure);
                    ++totalABStructuresCreated_;
                    created.push_back(structure.identity);
                    EnforceABCapacity();
                }
            }
            StoreFractal(fractal);
        }
        SortABStructures();
        return created;
    }

    std::optional<std::uint64_t> ObserveInnerBreak(
        const TG2::BreakObservation& behaviorObservation,
        std::optional<double> observationAtr = std::nullopt)
    {
        if (behaviorObservation.breakEvent.frozenClassification !=
            TG1B::SteepnessClassification::Inner)
            return std::nullopt;
        if (!lastBar_.has_value() ||
            behaviorObservation.breakEvent.bar != *lastBar_ ||
            !lastTimestamp_.has_value() ||
            behaviorObservation.breakEvent.timestamp != *lastTimestamp_)
            throw std::invalid_argument(
                "TG3 confluence must be observed on the completed Inner-break bar");
        if (FindObservation(behaviorObservation.breakEvent.eventSequence) !=
            nullptr)
            throw std::invalid_argument(
                "TG3 break event sequence must be observed once");
        if (behaviorObservation.breakEvent.eventSequence == 0 ||
            (lastBreakEventSequence_.has_value() &&
             behaviorObservation.breakEvent.eventSequence <=
                 *lastBreakEventSequence_))
            throw std::invalid_argument(
                "TG3 break event sequences must be positive and increasing");
        if (observationAtr.has_value() &&
            (!std::isfinite(*observationAtr) || *observationAtr <= 0.0))
            observationAtr = std::nullopt;

        EnsureObservationCapacity(*lastBar_, *lastTimestamp_);

        ConfluenceObservation observation;
        const TG2::BreakEvent& event = behaviorObservation.breakEvent;
        observation.breakEventSequence = event.eventSequence;
        observation.innerCandidate = event.candidate;
        observation.frozenClassification = event.frozenClassification;
        observation.innerBreakBar = event.bar;
        observation.innerBreakTimestamp = event.timestamp;
        observation.pairedOuter = behaviorObservation.pairedOuter;
        observation.observationBar = event.bar;
        observation.observationTimestamp = event.timestamp;
        observation.observationAtr = observationAtr;
        observation.anchorSelectionPolicy = configuration_.anchorSelectionPolicy;
        observation.directionalStudyPolicy =
            configuration_.directionalStudyPolicy;
        observation.confluencePolicy = configuration_.confluencePolicy;
        observation.absolutePriceTolerance =
            configuration_.absolutePriceTolerance;
        observation.retest = behaviorObservation.retest;
        observation.outerTarget = behaviorObservation.outerTarget;
        observation.outerTargetAfterRetest =
            behaviorObservation.outerTargetAfterRetest;

        ClassifyConfluence(observation);
        observations_.push_back(std::move(observation));
        lastBreakEventSequence_ = event.eventSequence;
        std::sort(observations_.begin(), observations_.end(),
            [](const ConfluenceObservation& left,
               const ConfluenceObservation& right)
            {
                return left.breakEventSequence < right.breakEventSequence;
            });
        return event.eventSequence;
    }

    void SynchronizeOutcomes(
        const std::vector<TG2::BreakObservation>& behaviorObservations,
        std::size_t currentBar,
        std::int64_t currentTimestamp)
    {
        for (ConfluenceObservation& observation : observations_)
        {
            const auto found = std::find_if(
                behaviorObservations.begin(), behaviorObservations.end(),
                [&observation](const TG2::BreakObservation& candidate)
                {
                    return candidate.breakEvent.eventSequence ==
                        observation.breakEventSequence;
                });
            if (found != behaviorObservations.end())
            {
                observation.retest = found->retest;
                observation.outerTarget = found->outerTarget;
                observation.outerTargetAfterRetest =
                    found->outerTargetAfterRetest;
            }
            else
            {
                // TG2 only removes detailed records under deterministic
                // capacity pressure. Preserve that as censoring, not failure.
                CensorPending(observation.retest, currentBar,
                              currentTimestamp, observation.innerBreakBar);
                CensorPending(observation.outerTarget, currentBar,
                              currentTimestamp, observation.innerBreakBar);
                CensorPending(observation.outerTargetAfterRetest, currentBar,
                              currentTimestamp, observation.innerBreakBar);
            }
        }
    }

    const std::vector<ABStructure>& ABStructures() const
    {
        return abStructures_;
    }

    const std::vector<ConfluenceObservation>& Observations() const
    {
        return observations_;
    }

    const Configuration& GetConfiguration() const { return configuration_; }
    const TG1A::SeriesIdentity& GetSeriesIdentity() const { return identity_; }
    std::size_t TotalABStructuresCreated() const
    {
        return totalABStructuresCreated_;
    }
    std::size_t ABCapacityEvictions() const { return abCapacityEvictions_; }
    std::size_t ABAgeExpirations() const { return abAgeExpirations_; }

    std::size_t ActiveObservationCount() const
    {
        return static_cast<std::size_t>(std::count_if(
            observations_.begin(), observations_.end(),
            [](const ConfluenceObservation& observation)
            {
                return IsPending(observation.retest) ||
                    IsPending(observation.outerTarget) ||
                    IsPending(observation.outerTargetAfterRetest);
            }));
    }

    Summary AggregateSummary() const
    {
        Summary result = archivedSummary_;
        for (const ConfluenceObservation& observation : observations_)
            Accumulate(result, observation);
        FinalizeRates(result);
        return result;
    }

    std::string FormatDiagnostic(
        const ConfluenceObservation& observation) const
    {
        std::ostringstream output;
        output << std::setprecision(12)
               << "symbol=" << identity_.symbol
               << ",timeframe=" << identity_.timeframe
               << ",tg2_break_event_sequence="
               << observation.breakEventSequence
               << ",direction="
               << DirectionName(observation.innerCandidate.direction)
               << ",classification="
               << ClassificationName(observation.frozenClassification)
               << ",inner_anchor1_bar="
               << observation.innerCandidate.anchor1Bar
               << ",inner_anchor1_timestamp="
               << FormatTimestamp(
                      observation.innerCandidate.anchor1Timestamp)
               << ",inner_anchor2_bar="
               << observation.innerCandidate.anchor2Bar
               << ",inner_anchor2_timestamp="
               << FormatTimestamp(
                      observation.innerCandidate.anchor2Timestamp)
               << ",inner_creation_bar="
               << observation.innerCandidate.creationBar
               << ",inner_creation_timestamp="
               << FormatTimestamp(
                      observation.innerCandidate.creationTimestamp)
               << ",inner_break_bar=" << observation.innerBreakBar
               << ",inner_break_timestamp="
               << FormatTimestamp(observation.innerBreakTimestamp)
               << ",observation_timing=completed_inner_break_bar"
               << ",observation_bar=" << observation.observationBar
               << ",observation_timestamp="
               << FormatTimestamp(observation.observationTimestamp)
               << ",paired_outer="
               << (observation.pairedOuter.has_value() ? "yes" : "no");
        if (observation.pairedOuter.has_value())
        {
            const TG2::PairedOuterLine& outer = *observation.pairedOuter;
            output << ",outer_anchor1_bar=" << outer.candidate.anchor1Bar
                   << ",outer_anchor1_timestamp="
                   << FormatTimestamp(outer.candidate.anchor1Timestamp)
                   << ",outer_anchor2_bar=" << outer.candidate.anchor2Bar
                   << ",outer_anchor2_timestamp="
                   << FormatTimestamp(outer.candidate.anchor2Timestamp)
                   << ",outer_creation_bar=" << outer.candidate.creationBar
                   << ",outer_creation_timestamp="
                   << FormatTimestamp(outer.candidate.creationTimestamp)
                   << ",outer_projected_price_at_observation="
                   << OptionalNumber(
                          observation.outerProjectedPriceAtObservation);
        }
        output << ",anchor_selection_policy="
               << AnchorPolicyName(observation.anchorSelectionPolicy)
               << ",anchor_policy_provenance=implementation_convention"
               << ",ratio_provenance=explicit_caller_configuration"
               << ",fibonacci_level_type=retracement"
               << ",directional_study_policy="
               << DirectionalPolicyName(observation.directionalStudyPolicy)
               << ",confluence_policy="
               << ConfluencePolicyName(observation.confluencePolicy)
               << ",confluence_policy_provenance=implementation_convention"
               << ",absolute_price_tolerance="
               << observation.absolutePriceTolerance;
        if (observation.selectedAB.has_value())
        {
            const ABStructure& ab = *observation.selectedAB;
            output << ",ab_direction=" << ABDirectionName(ab.identity.direction)
                   << ",a_bar=" << ab.identity.aBar
                   << ",a_timestamp="
                   << FormatTimestamp(ab.identity.aTimestamp)
                   << ",a_price=" << ab.aPrice
                   << ",a_confirmation_bar=" << ab.aConfirmationBar
                   << ",a_confirmation_timestamp="
                   << FormatTimestamp(ab.aConfirmationTimestamp)
                   << ",b_bar=" << ab.identity.bBar
                   << ",b_timestamp="
                   << FormatTimestamp(ab.identity.bTimestamp)
                   << ",b_price=" << ab.bPrice
                   << ",b_confirmation_bar=" << ab.bConfirmationBar
                   << ",b_confirmation_timestamp="
                   << FormatTimestamp(ab.bConfirmationTimestamp)
                   << ",ab_availability_bar="
                   << ab.identity.availabilityBar
                   << ",ab_availability_timestamp="
                   << FormatTimestamp(ab.identity.availabilityTimestamp)
                   << ",ab_price_range=" << ab.priceRange;
        }
        output << ",levels=" << FormatLevels(observation.levels)
               << ",matched_ratios="
               << FormatRatios(observation.matchedRatios)
               << ",minimum_raw_price_distance="
               << OptionalNumber(observation.minimumRawPriceDistance)
               << ",observation_atr="
               << OptionalNumber(observation.observationAtr)
               << ",minimum_atr_normalized_distance="
               << OptionalNumber(
                      observation.minimumAtrNormalizedDistance)
               << ",confluence="
               << ConfluenceStateName(observation.confluenceState)
               << ",ineligible_reason="
               << IneligibleReasonName(observation.ineligibleReason)
               << ",outer_outcome="
               << ResolutionName(observation.outerTarget.state)
               << ",retest_outcome="
               << ResolutionName(observation.retest.state)
               << ",retest_censor_reason="
               << CensorReasonName(observation.retest.censorReason)
               << ",retest_then_outer_outcome="
               << ResolutionName(
                      observation.outerTargetAfterRetest.state)
               << ",retest_then_outer_censor_reason="
               << CensorReasonName(
                      observation.outerTargetAfterRetest.censorReason)
               << ",outer_censor_reason="
               << CensorReasonName(observation.outerTarget.censorReason)
               << ",empirical_interpretation=observed_frequency_not_probability";
        return output.str();
    }

    std::string FormatSummaryDiagnostic() const
    {
        const Summary summary = AggregateSummary();
        std::ostringstream output;
        output << std::setprecision(12)
               << "symbol=" << identity_.symbol
               << ",timeframe=" << identity_.timeframe
               << ",inner_break_observations="
               << summary.innerBreakObservations
               << ",eligible_with_confluence="
               << summary.eligibleWithConfluence
               << ",eligible_without_confluence="
               << summary.eligibleWithoutConfluence
               << ",structurally_ineligible="
               << summary.structurallyIneligible
               << ",no_paired_outer=" << summary.noPairedOuter
               << ",no_eligible_ab=" << summary.noEligibleAB
               << ",unsupported_direction="
               << summary.unsupportedDirection
               << ",capacity_evicted_observations="
               << summary.capacityEvictedObservations;
        AppendGroupDiagnostic(output, "confluence", summary.confluence);
        AppendGroupDiagnostic(output, "no_confluence", summary.noConfluence);
        AppendGroupDiagnostic(output, "structurally_ineligible",
                              summary.ineligible);
        output << ",empirical_rate_formula=successes/(successes+failures)"
               << ",empirical_interpretation=observed_frequency_not_probability";
        return output.str();
    }

    static std::vector<FibonacciLevel> CalculateRetracementLevels(
        const ABStructure& structure,
        const std::vector<double>& ratios)
    {
        if (!(structure.priceRange > 0.0) ||
            !std::isfinite(structure.priceRange) ||
            !std::isfinite(structure.aPrice) ||
            !std::isfinite(structure.bPrice))
            throw std::invalid_argument(
                "TG3 Fibonacci levels require a finite nondegenerate AB range");
        std::vector<FibonacciLevel> levels;
        levels.reserve(ratios.size());
        for (double ratio : ratios)
        {
            if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0)
                throw std::invalid_argument(
                    "TG3 retracement ratios must be finite and in [0, 1]");
            const double price = structure.identity.direction ==
                    ABDirection::UpAB
                ? structure.bPrice - ratio * structure.priceRange
                : structure.bPrice + ratio * structure.priceRange;
            levels.push_back(
                {FibonacciLevelType::Retracement, ratio, price});
        }
        std::sort(levels.begin(), levels.end(),
            [](const FibonacciLevel& left, const FibonacciLevel& right)
            {
                if (left.price != right.price) return left.price < right.price;
                return left.ratio < right.ratio;
            });
        return levels;
    }

private:
    Configuration configuration_;
    TG1A::SeriesIdentity identity_;
    std::vector<TG1A::ConfirmedFractal> highFractals_;
    std::vector<TG1A::ConfirmedFractal> lowFractals_;
    std::vector<ABStructure> abStructures_;
    std::vector<ConfluenceObservation> observations_;
    Summary archivedSummary_;
    std::optional<std::size_t> lastBar_;
    std::optional<std::int64_t> lastTimestamp_;
    std::optional<std::uint64_t> lastBreakEventSequence_;
    std::size_t totalABStructuresCreated_ = 0;
    std::size_t abCapacityEvictions_ = 0;
    std::size_t abAgeExpirations_ = 0;

    using ABKey = std::tuple<std::size_t, std::size_t, int,
                             std::int64_t, std::int64_t>;

    void ValidateAndNormalizeConfiguration()
    {
        if (configuration_.retracementRatios.empty())
            throw std::invalid_argument(
                "TG3 requires explicit caller-configured retracement ratios");
        for (double& ratio : configuration_.retracementRatios)
        {
            if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0)
                throw std::invalid_argument(
                    "TG3 retracement ratios must be finite and in [0, 1]");
            if (ratio == 0.0) ratio = 0.0; // Canonicalize negative zero.
        }
        std::sort(configuration_.retracementRatios.begin(),
                  configuration_.retracementRatios.end());
        configuration_.retracementRatios.erase(
            std::unique(configuration_.retracementRatios.begin(),
                        configuration_.retracementRatios.end()),
            configuration_.retracementRatios.end());
        if (!std::isfinite(configuration_.absolutePriceTolerance) ||
            configuration_.absolutePriceTolerance < 0.0)
            throw std::invalid_argument(
                "TG3 confluence tolerance must be finite and nonnegative");
        if (configuration_.maxConfirmedFractalsPerKind == 0 ||
            configuration_.maxABAgeBars == 0 ||
            configuration_.maxActiveABStructures == 0 ||
            configuration_.maxActiveConfluenceObservations == 0 ||
            configuration_.maxRetainedConfluenceObservations == 0)
            throw std::invalid_argument("TG3 state bounds must be positive");
    }

    static bool FractalLess(const TG1A::ConfirmedFractal& left,
                            const TG1A::ConfirmedFractal& right)
    {
        return std::tuple(left.confirmationBar, left.anchorBar,
                          static_cast<int>(left.kind), left.anchorTimestamp) <
               std::tuple(right.confirmationBar, right.anchorBar,
                          static_cast<int>(right.kind), right.anchorTimestamp);
    }

    static bool SameFractal(const TG1A::ConfirmedFractal& left,
                            const TG1A::ConfirmedFractal& right)
    {
        return left.kind == right.kind &&
            left.anchorBar == right.anchorBar &&
            left.anchorTimestamp == right.anchorTimestamp &&
            left.confirmationBar == right.confirmationBar &&
            left.confirmationTimestamp == right.confirmationTimestamp;
    }

    bool AlreadyStored(const TG1A::ConfirmedFractal& fractal) const
    {
        const std::vector<TG1A::ConfirmedFractal>& store =
            fractal.kind == TG1A::FractalKind::High
            ? highFractals_ : lowFractals_;
        return std::any_of(store.begin(), store.end(),
            [&fractal](const TG1A::ConfirmedFractal& candidate)
            {
                return SameFractal(candidate, fractal);
            });
    }

    std::optional<TG1A::ConfirmedFractal> SelectAnchorFor(
        const TG1A::ConfirmedFractal& b) const
    {
        const std::vector<TG1A::ConfirmedFractal>& candidates =
            b.kind == TG1A::FractalKind::High ? lowFractals_ : highFractals_;
        const TG1A::ConfirmedFractal* best = nullptr;
        for (const TG1A::ConfirmedFractal& candidate : candidates)
        {
            if (candidate.anchorBar >= b.anchorBar ||
                candidate.confirmationBar > b.confirmationBar)
                continue;
            const bool directional = b.kind == TG1A::FractalKind::High
                ? b.price > candidate.price
                : b.price < candidate.price;
            if (!directional) continue;
            if (best == nullptr ||
                std::tuple(candidate.anchorBar, candidate.confirmationBar,
                           candidate.anchorTimestamp) >
                std::tuple(best->anchorBar, best->confirmationBar,
                           best->anchorTimestamp))
                best = &candidate;
        }
        return best == nullptr
            ? std::nullopt
            : std::optional<TG1A::ConfirmedFractal>(*best);
    }

    void StoreFractal(const TG1A::ConfirmedFractal& fractal)
    {
        std::vector<TG1A::ConfirmedFractal>& store =
            fractal.kind == TG1A::FractalKind::High
            ? highFractals_ : lowFractals_;
        store.push_back(fractal);
        std::sort(store.begin(), store.end(), FractalLess);
        while (store.size() > configuration_.maxConfirmedFractalsPerKind)
            store.erase(store.begin());
    }

    static ABKey Key(const ABStructure& structure)
    {
        return {structure.identity.availabilityBar,
                structure.identity.bBar,
                static_cast<int>(structure.identity.direction),
                structure.identity.aTimestamp,
                structure.identity.bTimestamp};
    }

    void SortABStructures()
    {
        std::sort(abStructures_.begin(), abStructures_.end(),
            [](const ABStructure& left, const ABStructure& right)
            {
                return Key(left) < Key(right);
            });
    }

    void EnforceABCapacity()
    {
        SortABStructures();
        while (abStructures_.size() > configuration_.maxActiveABStructures)
        {
            abStructures_.erase(abStructures_.begin());
            ++abCapacityEvictions_;
        }
    }

    void ExpireABStructures(std::size_t currentBar)
    {
        const auto oldSize = abStructures_.size();
        abStructures_.erase(
            std::remove_if(abStructures_.begin(), abStructures_.end(),
                [this, currentBar](const ABStructure& structure)
                {
                    return currentBar > structure.identity.availabilityBar &&
                        currentBar - structure.identity.availabilityBar >
                            configuration_.maxABAgeBars;
                }),
            abStructures_.end());
        abAgeExpirations_ += oldSize - abStructures_.size();
    }

    std::optional<ABDirection> RequiredABDirection(
        TG1A::TrendLineDirection direction) const
    {
        if (direction == TG1A::TrendLineDirection::UTL)
            return ABDirection::UpAB;
        if (configuration_.directionalStudyPolicy ==
            DirectionalStudyPolicy::SymmetricDirectionalDiagnostic)
            return ABDirection::DownAB;
        return std::nullopt;
    }

    std::optional<ABStructure> SelectAB(
        ABDirection direction, std::size_t observationBar) const
    {
        const ABStructure* best = nullptr;
        for (const ABStructure& structure : abStructures_)
        {
            if (structure.identity.direction != direction ||
                structure.identity.availabilityBar > observationBar)
                continue;
            if (best == nullptr || Key(structure) > Key(*best))
                best = &structure;
        }
        return best == nullptr
            ? std::nullopt : std::optional<ABStructure>(*best);
    }

    void ClassifyConfluence(ConfluenceObservation& observation) const
    {
        if (!observation.pairedOuter.has_value())
        {
            observation.ineligibleReason = IneligibleReason::NoTG2PairedOuter;
            return;
        }
        const std::optional<ABDirection> required = RequiredABDirection(
            observation.innerCandidate.direction);
        if (!required.has_value())
        {
            observation.ineligibleReason =
                IneligibleReason::DirectionUnsupportedByStudy;
            return;
        }
        observation.selectedAB = SelectAB(*required, observation.observationBar);
        if (!observation.selectedAB.has_value())
        {
            observation.ineligibleReason =
                IneligibleReason::NoCausallyEligibleAB;
            return;
        }

        observation.outerProjectedPriceAtObservation =
            observation.pairedOuter->projectedPriceAtBreak;
        const std::vector<FibonacciLevel> levels =
            CalculateRetracementLevels(*observation.selectedAB,
                                       configuration_.retracementRatios);
        double minimumDistance = std::numeric_limits<double>::infinity();
        for (const FibonacciLevel& level : levels)
        {
            const double distance = std::fabs(
                *observation.outerProjectedPriceAtObservation - level.price);
            const bool matched =
                distance <= configuration_.absolutePriceTolerance;
            observation.levels.push_back(
                {level.ratio, level.price,
                 level.price - configuration_.absolutePriceTolerance,
                 level.price + configuration_.absolutePriceTolerance,
                 distance, matched});
            if (matched) observation.matchedRatios.push_back(level.ratio);
            minimumDistance = std::min(minimumDistance, distance);
        }
        observation.minimumRawPriceDistance = minimumDistance;
        if (observation.observationAtr.has_value())
            observation.minimumAtrNormalizedDistance =
                minimumDistance / *observation.observationAtr;
        observation.confluenceState = observation.matchedRatios.empty()
            ? ConfluenceState::NoConfluence
            : ConfluenceState::Confluence;
        observation.ineligibleReason = IneligibleReason::None;
    }

    ConfluenceObservation* FindObservation(std::uint64_t eventSequence)
    {
        const auto found = std::find_if(
            observations_.begin(), observations_.end(),
            [eventSequence](const ConfluenceObservation& observation)
            {
                return observation.breakEventSequence == eventSequence;
            });
        return found == observations_.end() ? nullptr : &*found;
    }

    static bool IsPending(const TG2::OutcomeResolution& outcome)
    {
        return outcome.state == TG2::ResolutionState::Pending;
    }

    static void CensorPending(TG2::OutcomeResolution& outcome,
                              std::size_t bar,
                              std::int64_t timestamp,
                              std::size_t breakBar)
    {
        if (!IsPending(outcome)) return;
        outcome.state = TG2::ResolutionState::Censored;
        outcome.censorReason = TG2::CensorReason::CapacityEviction;
        outcome.resolutionBar = bar;
        outcome.resolutionTimestamp = timestamp;
        if (bar >= breakBar) outcome.latencyBars = bar - breakBar;
    }

    static bool ObservationActive(const ConfluenceObservation& observation)
    {
        return IsPending(observation.retest) ||
            IsPending(observation.outerTarget) ||
            IsPending(observation.outerTargetAfterRetest);
    }

    void ArchiveAndErase(std::size_t index,
                         std::size_t bar,
                         std::int64_t timestamp)
    {
        ConfluenceObservation& observation = observations_[index];
        CensorPending(observation.retest, bar, timestamp,
                      observation.innerBreakBar);
        CensorPending(observation.outerTarget, bar, timestamp,
                      observation.innerBreakBar);
        CensorPending(observation.outerTargetAfterRetest, bar, timestamp,
                      observation.innerBreakBar);
        Accumulate(archivedSummary_, observation);
        ++archivedSummary_.capacityEvictedObservations;
        observations_.erase(observations_.begin() +
                            static_cast<std::ptrdiff_t>(index));
    }

    void EnsureObservationCapacity(std::size_t bar, std::int64_t timestamp)
    {
        while (ActiveObservationCount() >=
               configuration_.maxActiveConfluenceObservations)
        {
            const auto active = std::find_if(
                observations_.begin(), observations_.end(),
                ObservationActive);
            if (active == observations_.end()) break;
            ArchiveAndErase(static_cast<std::size_t>(
                std::distance(observations_.begin(), active)),
                bar, timestamp);
        }
        while (observations_.size() >=
               configuration_.maxRetainedConfluenceObservations)
            ArchiveAndErase(0, bar, timestamp);
    }

    static void AddOutcome(TG2::OutcomeCounts& counts,
                           const TG2::OutcomeResolution& outcome)
    {
        if (outcome.state == TG2::ResolutionState::StructurallyIneligible)
        {
            ++counts.structurallyIneligible;
            return;
        }
        ++counts.eligible;
        switch (outcome.state)
        {
            case TG2::ResolutionState::Succeeded:
                ++counts.successes;
                ++counts.resolved;
                break;
            case TG2::ResolutionState::Failed:
                ++counts.failures;
                ++counts.resolved;
                break;
            case TG2::ResolutionState::Censored: ++counts.censored; break;
            case TG2::ResolutionState::Pending: ++counts.pending; break;
            case TG2::ResolutionState::StructurallyIneligible: break;
        }
    }

    static void AccumulateGroup(ConfluenceGroupCounts& group,
                                const ConfluenceObservation& observation)
    {
        ++group.observations;
        AddOutcome(group.outerTarget, observation.outerTarget);
        if (observation.retest.state == TG2::ResolutionState::Succeeded &&
            observation.pairedOuter.has_value())
        {
            ++group.retestedObservations;
            AddOutcome(group.retestThenOuter,
                       observation.outerTargetAfterRetest);
        }
    }

    static void Accumulate(Summary& summary,
                           const ConfluenceObservation& observation)
    {
        ++summary.innerBreakObservations;
        switch (observation.confluenceState)
        {
            case ConfluenceState::Confluence:
                ++summary.eligibleWithConfluence;
                AccumulateGroup(summary.confluence, observation);
                break;
            case ConfluenceState::NoConfluence:
                ++summary.eligibleWithoutConfluence;
                AccumulateGroup(summary.noConfluence, observation);
                break;
            case ConfluenceState::StructurallyIneligible:
                ++summary.structurallyIneligible;
                if (observation.ineligibleReason ==
                    IneligibleReason::NoTG2PairedOuter)
                    ++summary.noPairedOuter;
                else if (observation.ineligibleReason ==
                    IneligibleReason::NoCausallyEligibleAB)
                    ++summary.noEligibleAB;
                else if (observation.ineligibleReason ==
                    IneligibleReason::DirectionUnsupportedByStudy)
                    ++summary.unsupportedDirection;
                AccumulateGroup(summary.ineligible, observation);
                break;
        }
    }

    static void SetRate(TG2::OutcomeCounts& counts)
    {
        const std::size_t denominator = counts.successes + counts.failures;
        counts.empiricalRate = denominator == 0
            ? std::nullopt
            : std::optional<double>(
                static_cast<double>(counts.successes) /
                static_cast<double>(denominator));
    }

    static void FinalizeGroupRates(ConfluenceGroupCounts& group)
    {
        SetRate(group.outerTarget);
        SetRate(group.retestThenOuter);
    }

    static void FinalizeRates(Summary& summary)
    {
        FinalizeGroupRates(summary.confluence);
        FinalizeGroupRates(summary.noConfluence);
        FinalizeGroupRates(summary.ineligible);
    }

    static void AppendGroupDiagnostic(
        std::ostringstream& output,
        const char* prefix,
        const ConfluenceGroupCounts& group)
    {
        const std::size_t outerDenominator =
            group.outerTarget.successes + group.outerTarget.failures;
        const std::size_t conditionedDenominator =
            group.retestThenOuter.successes +
            group.retestThenOuter.failures;
        output << ',' << prefix << "_observations=" << group.observations
               << ',' << prefix << "_outer_successes="
               << group.outerTarget.successes
               << ',' << prefix << "_outer_failures="
               << group.outerTarget.failures
               << ',' << prefix << "_outer_censored="
               << group.outerTarget.censored
               << ',' << prefix << "_outer_pending="
               << group.outerTarget.pending
               << ',' << prefix << "_outer_denominator="
               << outerDenominator
               << ',' << prefix << "_outer_empirical_rate="
               << OptionalNumber(group.outerTarget.empiricalRate)
               << ',' << prefix << "_retested_observations="
               << group.retestedObservations
               << ',' << prefix << "_retest_then_outer_successes="
               << group.retestThenOuter.successes
               << ',' << prefix << "_retest_then_outer_failures="
               << group.retestThenOuter.failures
               << ',' << prefix << "_retest_then_outer_censored="
               << group.retestThenOuter.censored
               << ',' << prefix << "_retest_then_outer_pending="
               << group.retestThenOuter.pending
               << ',' << prefix << "_retest_then_outer_denominator="
               << conditionedDenominator
               << ',' << prefix << "_retest_then_outer_empirical_rate="
               << OptionalNumber(group.retestThenOuter.empiricalRate);
    }

    static const char* DirectionName(TG1A::TrendLineDirection direction)
    {
        return direction == TG1A::TrendLineDirection::UTL ? "UTL" : "DTL";
    }

    static const char* ABDirectionName(ABDirection direction)
    {
        return direction == ABDirection::UpAB ? "UpAB" : "DownAB";
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

    static const char* AnchorPolicyName(AnchorSelectionPolicy)
    {
        return "most_recent_prior_opposite_confirmed_fractal";
    }

    static const char* DirectionalPolicyName(DirectionalStudyPolicy policy)
    {
        return policy == DirectionalStudyPolicy::SourceUTLUpABOnly
            ? "source_utl_up_ab_only"
            : "symmetric_directional_diagnostic_hypothesis";
    }

    static const char* ConfluencePolicyName(ConfluencePolicy)
    {
        return "absolute_price_tolerance_around_exact_retracement_level";
    }

    static const char* ConfluenceStateName(ConfluenceState state)
    {
        switch (state)
        {
            case ConfluenceState::Confluence: return "yes";
            case ConfluenceState::NoConfluence: return "no";
            case ConfluenceState::StructurallyIneligible:
                return "structurally_ineligible";
        }
        return "structurally_ineligible";
    }

    static const char* IneligibleReasonName(IneligibleReason reason)
    {
        switch (reason)
        {
            case IneligibleReason::None: return "none";
            case IneligibleReason::NoTG2PairedOuter:
                return "no_tg2_paired_outer";
            case IneligibleReason::DirectionUnsupportedByStudy:
                return "direction_unsupported_by_source_study";
            case IneligibleReason::NoCausallyEligibleAB:
                return "no_causally_eligible_ab";
        }
        return "none";
    }

    static const char* ResolutionName(TG2::ResolutionState state)
    {
        switch (state)
        {
            case TG2::ResolutionState::Pending: return "pending";
            case TG2::ResolutionState::Succeeded: return "succeeded";
            case TG2::ResolutionState::Failed: return "failed";
            case TG2::ResolutionState::Censored: return "censored";
            case TG2::ResolutionState::StructurallyIneligible:
                return "structurally_ineligible";
        }
        return "pending";
    }

    static const char* CensorReasonName(TG2::CensorReason reason)
    {
        switch (reason)
        {
            case TG2::CensorReason::None: return "none";
            case TG2::CensorReason::EndOfInput: return "end_of_input";
            case TG2::CensorReason::CapacityEviction:
                return "capacity_eviction";
        }
        return "none";
    }

    static std::string OptionalNumber(const std::optional<double>& value)
    {
        if (!value.has_value()) return "unavailable";
        std::ostringstream output;
        output << std::setprecision(12) << *value;
        return output.str();
    }

    static std::string FormatRatios(const std::vector<double>& ratios)
    {
        std::ostringstream output;
        output << '[';
        for (std::size_t index = 0; index < ratios.size(); ++index)
        {
            if (index != 0) output << ';';
            output << std::setprecision(12) << ratios[index];
        }
        output << ']';
        return output.str();
    }

    static std::string FormatLevels(
        const std::vector<LevelMeasurement>& levels)
    {
        std::ostringstream output;
        output << '[';
        for (std::size_t index = 0; index < levels.size(); ++index)
        {
            if (index != 0) output << ';';
            const LevelMeasurement& level = levels[index];
            output << std::setprecision(12)
                   << "ratio:" << level.ratio
                   << "|price:" << level.levelPrice
                   << "|lower:" << level.zoneLowerPrice
                   << "|upper:" << level.zoneUpperPrice
                   << "|distance:" << level.rawPriceDistance
                   << "|matched:" << (level.matched ? "yes" : "no");
        }
        output << ']';
        return output.str();
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

class CausalFibonacciConfluenceIntegration
{
public:
    CausalFibonacciConfluenceIntegration(
        TG1B::CalibrationConfiguration calibration,
        Configuration fibonacciConfiguration,
        TG1A::Configuration geometryConfiguration = {},
        TG2::Configuration behaviorConfiguration = {},
        TG1A::SeriesIdentity identity = {})
        : behavior_(calibration, geometryConfiguration,
                    behaviorConfiguration, identity),
          tracker_(std::move(fibonacciConfiguration), std::move(identity))
    {
    }

    static CausalFibonacciConfluenceIntegration FromHistorical(
        std::vector<TG1A::Candle> candles,
        TG1B::CalibrationConfiguration calibration,
        Configuration fibonacciConfiguration,
        TG1A::Configuration geometryConfiguration = {},
        TG2::Configuration behaviorConfiguration = {},
        TG1A::SeriesIdentity identity = {})
    {
        std::stable_sort(candles.begin(), candles.end(),
            [](const TG1A::Candle& left, const TG1A::Candle& right)
            {
                return left.timestamp < right.timestamp;
            });
        CausalFibonacciConfluenceIntegration result(
            calibration, std::move(fibonacciConfiguration),
            geometryConfiguration, behaviorConfiguration,
            std::move(identity));
        for (const TG1A::Candle& candle : candles)
            result.AddCompletedBar(candle);
        result.Finalize();
        return result;
    }

    Update AddCompletedBar(const TG1A::Candle& candle)
    {
        Update result;
        result.behaviorUpdate = behavior_.AddCompletedBar(candle);
        result.bar = result.behaviorUpdate.bar;
        lastBar_ = result.bar;
        lastTimestamp_ = candle.timestamp;
        tracker_.Advance(result.bar, candle.timestamp);
        result.newlyAvailableABStructures = tracker_.ObserveConfirmedFractals(
            result.behaviorUpdate.classificationUpdate.geometryUpdate
                .newlyConfirmedFractals);
        tracker_.SynchronizeOutcomes(
            behavior_.Observations(), result.bar, candle.timestamp);
        for (const TG2::BreakEvent& event :
             result.behaviorUpdate.newBreakEvents)
        {
            const TG2::BreakObservation* observation =
                FindBehaviorObservation(event.eventSequence);
            if (observation == nullptr)
                throw std::logic_error(
                    "TG3 could not find newly emitted TG2 observation");
            const std::optional<std::uint64_t> created =
                tracker_.ObserveInnerBreak(
                    *observation, ObservationAtr(*observation));
            if (created.has_value())
            {
                result.newConfluenceObservations.push_back(*created);
                const ConfluenceObservation* createdObservation =
                    FindConfluenceObservation(*created);
                if (createdObservation == nullptr)
                    throw std::logic_error(
                        "TG3 could not find newly created confluence observation");
                result.newlyCreatedConfluenceObservations.push_back(
                    *createdObservation);
            }
        }
        return result;
    }

    void Finalize()
    {
        behavior_.Finalize();
        if (lastBar_.has_value() && lastTimestamp_.has_value())
            tracker_.SynchronizeOutcomes(
                behavior_.Observations(), *lastBar_, *lastTimestamp_);
    }

    const std::vector<ABStructure>& ABStructures() const
    {
        return tracker_.ABStructures();
    }

    const std::vector<ConfluenceObservation>& Observations() const
    {
        return tracker_.Observations();
    }

    Summary AggregateSummary() const { return tracker_.AggregateSummary(); }

    std::string FormatDiagnostic(
        const ConfluenceObservation& observation) const
    {
        return tracker_.FormatDiagnostic(observation);
    }

    std::string FormatSummaryDiagnostic() const
    {
        return tracker_.FormatSummaryDiagnostic();
    }

    const TG2::CausalTrendLineBreakRetestBehavior& Behavior() const
    {
        return behavior_;
    }

    const FibonacciConfluenceTracker& Tracker() const { return tracker_; }

private:
    TG2::CausalTrendLineBreakRetestBehavior behavior_;
    FibonacciConfluenceTracker tracker_;
    std::optional<std::size_t> lastBar_;
    std::optional<std::int64_t> lastTimestamp_;

    const TG2::BreakObservation* FindBehaviorObservation(
        std::uint64_t eventSequence) const
    {
        const auto& observations = behavior_.Observations();
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [eventSequence](const TG2::BreakObservation& observation)
            {
                return observation.breakEvent.eventSequence == eventSequence;
            });
        return found == observations.end() ? nullptr : &*found;
    }

    const ConfluenceObservation* FindConfluenceObservation(
        std::uint64_t eventSequence) const
    {
        const auto& observations = tracker_.Observations();
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [eventSequence](const ConfluenceObservation& observation)
            {
                return observation.breakEventSequence == eventSequence;
            });
        return found == observations.end() ? nullptr : &*found;
    }

    std::optional<double> ObservationAtr(
        const TG2::BreakObservation& observation) const
    {
        if (!observation.pairedOuter.has_value()) return std::nullopt;
        const auto& candidates =
            behavior_.Classification().Geometry().Candidates();
        const TG2::CandidateIdentity& identity =
            observation.pairedOuter->candidate;
        const auto found = std::find_if(
            candidates.begin(), candidates.end(),
            [&identity](const TG1A::TrendLineCandidate& candidate)
            {
                return candidate.direction == identity.direction &&
                    candidate.anchor1Bar == identity.anchor1Bar &&
                    candidate.anchor2Bar == identity.anchor2Bar &&
                    candidate.creationBar == identity.creationBar;
            });
        return found == candidates.end() ? std::nullopt : found->currentAtr;
    }

};

} // namespace EA::TG3

#endif /* CausalFibonacciConfluenceIntegration_hpp */
