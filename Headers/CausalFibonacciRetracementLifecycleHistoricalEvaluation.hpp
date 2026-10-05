#pragma once

#include "CausalFibonacciConfluenceIntegration.hpp"
#include "CausalFibonacciRetracementLifecycle.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace EA::FibonacciResearch::RetracementLifecycle
{

enum class DTargetHypothesis { Extension1272, Extension1618 };

struct HistoricalRecord
{
    DTargetHypothesis dTarget = DTargetHypothesis::Extension1272;
    Record lifecycle;
};

// This evaluator is deliberately independent of the frozen H1/H2 study. It
// snapshots each newly causal A/B while TG3 retains it, then advances the copy
// even when TG3 later ages or capacity-prunes its active A/B collection.
class HistoricalEvaluator
{
public:
    using RecordSink = std::function<void(HistoricalRecord)>;

    HistoricalEvaluator(TG1B::CalibrationConfiguration calibration,
                        TG3::Configuration fibonacci,
                        TG1A::Configuration geometry,
                        TG2::Configuration behavior,
                        TG1A::SeriesIdentity identity,
                        RecordSink sink)
        : tolerance_(fibonacci.absolutePriceTolerance),
          integration_(std::move(calibration), std::move(fibonacci),
                       std::move(geometry), std::move(behavior),
                       std::move(identity)), sink_(std::move(sink))
    {
        if (!sink_) throw std::invalid_argument("retracement lifecycle sink is empty");
    }

    void AddCompletedBar(const TG1A::Candle& candle)
    {
        if (finalized_)
            throw std::logic_error("retracement lifecycle evaluator is finalized");
        const TG3::Update update = integration_.AddCompletedBar(candle);
        AddNewABs(update.newlyAvailableABStructures);
        for (State& state : states_)
            state.tracker.AddCompletedBar(update.bar, candle);
    }

    void Finalize()
    {
        if (finalized_) return;
        integration_.Finalize();
        for (State& state : states_) state.tracker.Finalize();
        std::sort(states_.begin(), states_.end(), [](const State& left,
                                                      const State& right)
        {
            return std::tie(left.identity.availabilityBar, left.identity.bBar,
                            left.identity.direction, left.hypothesis) <
                   std::tie(right.identity.availabilityBar, right.identity.bBar,
                            right.identity.direction, right.hypothesis);
        });
        for (State& state : states_)
            sink_({state.hypothesis, state.tracker.GetRecord()});
        finalized_ = true;
    }

    std::size_t RecordCount() const { return states_.size(); }
    bool IsFinalized() const { return finalized_; }

private:
    struct State
    {
        TG3::ABIdentity identity;
        DTargetHypothesis hypothesis;
        Tracker tracker;
    };

    double tolerance_;
    TG3::CausalFibonacciConfluenceIntegration integration_;
    RecordSink sink_;
    std::vector<State> states_;
    std::set<std::tuple<std::size_t, std::size_t, TG3::ABDirection,
                        std::int64_t, DTargetHypothesis>> identities_;
    bool finalized_ = false;

    void AddNewABs(const std::vector<TG3::ABIdentity>& identities)
    {
        for (const TG3::ABIdentity& identity : identities)
        {
            const auto found = std::find_if(integration_.ABStructures().begin(),
                integration_.ABStructures().end(), [&identity](const auto& ab)
                { return ab.identity == identity; });
            if (found == integration_.ABStructures().end())
                throw std::logic_error("new causal A/B was not retained");
            for (const auto hypothesis : {DTargetHypothesis::Extension1272,
                                          DTargetHypothesis::Extension1618})
            {
                const auto key = std::make_tuple(identity.availabilityBar,
                    identity.bBar, identity.direction, identity.bTimestamp,
                    hypothesis);
                if (!identities_.insert(key).second)
                    throw std::logic_error("duplicate retracement lifecycle A/B");
                const double ratio = hypothesis == DTargetHypothesis::Extension1272
                    ? kExtension1272 : kExtension1618;
                const double d = CalculateExtensionLevel(*found, ratio,
                                                          tolerance_).price;
                states_.push_back({identity, hypothesis, Tracker(*found, d)});
            }
        }
    }
};

} // namespace EA::FibonacciResearch::RetracementLifecycle
