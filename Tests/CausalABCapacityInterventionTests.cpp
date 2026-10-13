#include "CausalABCapacityCounterfactual.hpp"
#include "CausalFibonacciTrendlineHistoricalEvaluation.hpp"
#include <iostream>
namespace A = EA::FibonacciResearch::ABCapacity;
namespace H = EA::FibonacciResearch::HistoryShadow;
namespace S = EA::FibonacciResearch::TrendlineHistorical;
namespace F = EA::TG3;
namespace G = EA::TG1A;
void Check(bool x)
{
    if (!x)
        throw std::runtime_error("intervention focused assertion");
}
template <class Fn> void Reject(Fn fn)
{
    bool failed = false;
    try
    {
        fn();
    }
    catch (...)
    {
        failed = true;
    }
    Check(failed);
}
template <class Fn> void RejectReason(Fn fn, std::string_view reason)
{
    bool failed = false;
    try
    {
        fn();
    }
    catch (const std::runtime_error &e)
    {
        failed = std::string_view(e.what()) == reason;
    }
    Check(failed);
}
std::int64_t Time(std::size_t bar) { return S::kOrigin + static_cast<std::int64_t>(bar) * 900; }
F::ABStructure AB(double a, double b, std::size_t birth)
{
    F::ABStructure v;
    const auto anchorA = birth > 6 ? birth - 6 : 0, anchorB = birth > 2 ? birth - 2 : 1;
    v.identity = {F::ABDirection::UpAB, anchorA, Time(anchorA), anchorB, Time(anchorB), birth, Time(birth)};
    v.aPrice = a;
    v.bPrice = b;
    v.aConfirmationBar = birth > 4 ? birth - 4 : 1;
    v.aConfirmationTimestamp = Time(v.aConfirmationBar);
    v.bConfirmationBar = birth;
    v.bConfirmationTimestamp = Time(birth);
    v.priceRange = b - a;
    return v;
}
G::TrendLineCandidate Line(double projected)
{
    G::TrendLineCandidate l;
    l.direction = G::TrendLineDirection::UTL;
    l.anchor1Bar = 2030;
    l.anchor2Bar = 2033;
    l.anchor1Timestamp = Time(2030);
    l.anchor2Timestamp = Time(2033);
    l.anchor1Price = projected - .21;
    l.anchor2Price = l.anchor1Price + .03;
    l.anchor1ConfirmationBar = 2032;
    l.anchor2ConfirmationBar = 2035;
    l.anchor1ConfirmationTimestamp = Time(2032);
    l.anchor2ConfirmationTimestamp = Time(2035);
    l.creationBar = 2035;
    l.creationTimestamp = Time(2035);
    l.anchorSeparationBars = 3;
    l.rawPriceSlopePerBar = .01;
    l.currentBar = 2051;
    l.currentTimestamp = Time(2051);
    return l;
}
int main()
{
    try
    {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        Check(A::RemainingInterventionPairs(6, 6) == 0);
        Check(A::AccumulateInterventionPairs(6, 4, 2, 1) == 6);
        Check(A::AccumulateInterventionPairs(6, 6, 0, maximum) == 6);
        RejectReason([&] { A::RemainingInterventionPairs(6, 7); },
                     "intervention_accumulated_pairs_exceed_budget");
        RejectReason([&] { A::AccumulateInterventionPairs(6, 6, 1, 1); },
                     "intervention_evaluation_exceeds_remaining_pairs");
        RejectReason([&] { A::AccumulateInterventionPairs(maximum, maximum, 1, 1); },
                     "intervention_pair_accumulation_overflow");
        RejectReason([&] { A::AccumulateInterventionPairs(maximum, 0, maximum, 2); },
                     "intervention_pair_multiplication_overflow");
        auto base = AB(104, 106, 2044), near = AB(99, 101, 2046), near2 = AB(99, 101, 2047),
             offset = AB(100.2, 100.24, 2048);
        G::Candle c{Time(2051), 100, 101, 99, 100};
        auto run = [&](std::vector<A::CapacityVictim> victims, double price, std::vector<std::string> q)
        {
            auto l = Line(price);
            return A::EvaluateInterventions(2051, c, 1, std::span{&base, 1}, std::span{&l, 1}, victims, q);
        };
        auto key = [](const auto &v) { return H::CandidateIdentityReconciliation::Native(v.identity); };
        auto one = run({{near, 2050, Time(2050)}}, 100.1, {key(near)});
        Check(one.actual.cohort == "B" && one.complete.cohort == "C" && one.subsets.at(1).minimal);
        auto budgetLine = Line(100.1);
        std::array victims{A::CapacityVictim{near, 2050, Time(2050)}};
        std::array qualifiers{key(near)};
        auto exact = A::EvaluateInterventions(2051, c, 1, std::span{&base, 1}, std::span{&budgetLine, 1},
                                              victims, qualifiers, one.pairs);
        Check(exact.pairs == one.pairs && exact.complete.cohort == one.complete.cohort);
        Reject(
            [&]
            {
                A::EvaluateInterventions(2051, c, 1, std::span{&base, 1}, std::span{&budgetLine, 1}, victims,
                                         qualifiers, one.pairs - 1);
            });
        auto many =
            run({{near, 2050, Time(2050)}, {near2, 2050, Time(2050)}}, 100.1, {key(near2), key(near)});
        Check(many.subsets.at(1).minimal && many.subsets.at(2).minimal && !many.subsets.at(3).minimal);
        Check(many.subsets.at(0).geometry.abs.size() == 1 && many.complete.abs.size() == 3);
        auto fib = run({{near, 2050, Time(2050)}}, 102, {key(near)});
        Check(fib.actual.cohort == "D" && fib.complete.cohort == "A");
        auto fs = run({{offset, 2050, Time(2050)}}, 99.8, {key(offset)});
        Check(fs.actual.cohort == "B" && fs.complete.cohort == "F_stratum");
        const auto original = base;
        base = offset;
        auto fToC = run({{near, 2050, Time(2050)}}, 99.8, {key(near)});
        Check(fToC.actual.cohort == "F_stratum" && fToC.complete.cohort == "C" && fToC.subsets[1].sufficient);
        base = original;
        auto unchanged = run({{AB(110, 112, 2049), 2050, Time(2050)}}, 100.1, {});
        Check(unchanged.qualifyingIdentities.empty() && unchanged.subsets.size() == 1 &&
              unchanged.subsets[0].minimal && unchanged.actual.cohort == unchanged.complete.cohort);
        auto repeat =
            run({{near2, 2050, Time(2050)}, {near, 2050, Time(2050)}}, 100.1, {key(near), key(near2)});
        for (std::size_t i = 0; i < many.subsets.size(); ++i)
        {
            Check(many.subsets[i].geometry.cohort == repeat.subsets[i].geometry.cohort);
            Check(many.subsets[i].minimal == repeat.subsets[i].minimal);
        }
        auto far = AB(110, 112, 2049);
        auto mixed = run({{far, 2050, Time(2050)}, {near, 2050, Time(2050)}}, 100.1, {key(near)});
        Check(mixed.complete.abs.size() == 3 && mixed.subsets.back().geometry.abs.size() == 2 &&
              mixed.subsets.back().sufficient);
        Reject(
            [&]
            {
                run({{far, 2050, Time(2050)}, {near, 2050, Time(2050)}}, 100.1, {});
            }); // no singleton in incomplete shortlist: reject
        Reject([&] { run({{near, 2052, Time(2052)}}, 100.1, {key(near)}); }); // future eviction provenance
        auto boundary = AB(99, 101, 3);                                       // age 2048
        auto age = run({{boundary, 2049, Time(2049)}}, 100.1, {key(boundary)});
        Check(age.subsets[1].sufficient);
        auto expired = AB(99, 101, 2);
        Reject([&] { run({{expired, 2049, Time(2049)}}, 100.1, {key(expired)}); });
        Reject(
            [&]
            {
                auto l = Line(100);
                A::EvaluateInterventions(2051, c, 1, std::span{&base, 1}, std::span{&l, 1}, {}, {}, 1);
            });
        // With fixed nonempty lines and frozen existential F/I aggregation, a changed
        // cohort always has a singleton witness. Exhaustively test all full predicate
        // triples and two candidate contribution triples: joint-only effects cannot
        // occur. Do not manufacture a non-frozen cross-AB conjunction to test them.
        std::size_t cases = 0;
        for (bool f : {false, true})
            for (bool l : {false, true})
                for (bool i : {false, true})
                {
                    if (i && (!f || !l))
                        continue;
                    for (bool af : {false, true})
                        for (bool ai : {false, true})
                            for (bool bf : {false, true})
                                for (bool bi : {false, true})
                                {
                                    if ((ai && (!af || !l)) || (bi && (!bf || !l)))
                                        continue;
                                    auto baseline = S::Classify(f, l, i),
                                         joint = S::Classify(f || af || bf, l, i || ai || bi);
                                    if (joint != baseline)
                                        Check(S::Classify(f || af, l, i || ai) == joint ||
                                              S::Classify(f || bf, l, i || bi) == joint);
                                    ++cases;
                                }
                }
        // Real native reconstruction exercises both directions and capacity provenance.
        F::Configuration fc;
        fc.retracementRatios = {.382, .5, .618};
        G::Configuration gc;
        A::InterventionReplay replay(fc, gc);
        for (std::size_t b = 0; b < 2220; ++b)
        {
            bool center = b % 3 == 0;
            double low = 100 - static_cast<double>(b) * .0001;
            replay.AddCompletedBar({Time(b), 101.4, center ? 107. : 106., low + (center ? 0 : .5), 101.4});
        }
        Check(replay.ParityChecks() == 2220 && replay.Actual().ABCapacityEvictions() > 0 &&
              !replay.Victims().empty());
        for (const auto &v : replay.Victims())
            Check(v.evictionBar >= v.ab.identity.availabilityBar &&
                  2219 - v.ab.identity.availabilityBar <= 2048);
        std::cout << "INTERVENTION_TESTS exact_budget=true one_pair_beyond=true accumulation_overflow=true "
                     "multiplication_overflow=true underflow_rejected=true "
                     "single_multiple_sufficiency=true complete_universe=true "
                     "fib_conjunction_Fstratum=true age_provenance=true deterministic=true "
                     "native_parity=true joint_only_impossibility_cases="
                  << cases << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
