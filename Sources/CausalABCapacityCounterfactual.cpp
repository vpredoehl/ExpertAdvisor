#include "CausalABCapacityCounterfactual.hpp"
#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include <fstream>
#include <set>

namespace EA::FibonacciResearch::ABCapacity
{
namespace
{
namespace H = HistoryShadow;
namespace S = TrendlineHistorical;
using AB = TG3::ABStructure;
using Line = TG1A::TrendLineCandidate;
constexpr std::size_t kMetadataReserve = 16384;
void Require(bool ok, const char *reason)
{
    if (!ok)
        throw std::runtime_error(reason);
}
std::string Read(const std::filesystem::path &p)
{
    std::ifstream f(p, std::ios::binary);
    Require(bool(f), "missing_publication_file");
    // Metadata is bounded; large CSVs are hashed/streamed instead.
    Require(std::filesystem::file_size(p) <= 1024 * 1024, "metadata_read_budget");
    return {std::istreambuf_iterator<char>(f), {}};
}
std::string Json(std::string_view s)
{
    std::string r = "\"";
    for (unsigned char c : s)
    {
        if (c == '"' || c == '\\')
            r += '\\';
        Require(c >= 32, "metadata_control_character");
        r += static_cast<char>(c);
    }
    return r + '"';
}
std::string Csv(std::string_view value)
{
    if (value.find_first_of(",\"\r\n") == value.npos)
        return std::string(value);
    std::string out = "\"";
    for (char c : value)
    {
        if (c == '\"')
            out += '\"';
        out += c;
    }
    return out + '\"';
}
std::string N(std::size_t v) { return std::to_string(v); }
std::string B(bool v) { return v ? "1" : "0"; }
std::string Value(std::optional<bool> v) { return v ? B(*v) : "UNKNOWN"; }
auto Identity(const TG3::ABIdentity &i)
{
    return std::tuple{
        i.direction, i.aBar, i.aTimestamp, i.bBar, i.bTimestamp, i.availabilityBar, i.availabilityTimestamp};
}
auto Snapshot(const AB &a)
{
    return std::tuple{Identity(a.identity),     a.aPrice,     a.aConfirmationBar,
                      a.aConfirmationTimestamp, a.bPrice,     a.bConfirmationBar,
                      a.bConfirmationTimestamp, a.priceRange, a.anchorSelectionPolicy};
}
auto Snapshot(const Line &l)
{
    return std::tuple{l.direction,
                      l.anchor1Bar,
                      l.anchor1Timestamp,
                      l.anchor1Price,
                      l.anchor1ConfirmationBar,
                      l.anchor1ConfirmationTimestamp,
                      l.anchor2Bar,
                      l.anchor2Timestamp,
                      l.anchor2Price,
                      l.anchor2ConfirmationBar,
                      l.anchor2ConfirmationTimestamp,
                      l.anchorSeparationBars,
                      l.rawPriceSlopePerBar,
                      l.creationBar,
                      l.creationTimestamp,
                      l.candidateAgeBars,
                      l.currentBar,
                      l.currentTimestamp,
                      l.projectedLinePriceAtCurrentBar,
                      l.rawPriceToLineDistance,
                      l.currentAtr,
                      l.atrNormalizedSlope,
                      l.atrNormalizedPriceToLineDistance,
                      l.candleTouchCount,
                      l.fractalTouchCount};
}
auto Snapshot(const TG1A::ConfirmedFractal &f)
{
    return std::tuple{f.kind,  f.anchorBar,       f.anchorTimestamp,
                      f.price, f.confirmationBar, f.confirmationTimestamp};
}
template <class A, class C, class Key> bool Equal(const A &a, const C &c, Key key)
{
    return a.size() == c.size() && std::equal(a.begin(), a.end(), c.begin(),
                                              [&](const auto &x, const auto &y) { return key(x) == key(y); });
}
auto Order(const AB &a)
{
    const auto &i = a.identity;
    return std::tuple{i.availabilityBar, i.bBar, static_cast<int>(i.direction), i.aTimestamp, i.bTimestamp};
}
std::set<std::string> VerifyChecksums(const std::filesystem::path &dir, std::string_view manifest)
{
    std::set<std::string> names;
    std::istringstream rows{Read(dir / manifest)};
    std::string row;
    while (std::getline(rows, row))
    {
        Require(row.size() > 66 && row.substr(64, 2) == "  " && S::IsHash(row.substr(0, 64)),
                "invalid_checksum_row");
        auto name = row.substr(66);
        Require(name.find('/') == name.npos && name.find('\\') == name.npos && name != "." && name != ".." &&
                    !std::filesystem::is_symlink(dir / name) && names.insert(name).second,
                "unsafe_checksum_path");
        Require(S::FileHash(dir / name) == row.substr(0, 64), "checksum_mismatch");
    }
    return names;
}
std::string CheckParent(const H::Provenance &p)
{
    const auto files = VerifyChecksums(p.parentPrimary, "checksums.sha256");
    for (const auto *name : {"protocol.json", "provenance_manifest.json", "coverage.csv", "decisions.csv",
                             "detector_configuration.json", "geometry_measurements.csv.gz"})
        Require(files.contains(name), "incomplete_parent_integrity_manifest");
    const bool invalid = std::filesystem::exists(p.parentPrimary / "INVALID");
    Require(invalid != std::filesystem::exists(p.parentPrimary / "COMPLETE"), "ambiguous_parent_publication");
    if (invalid)
        Require(Read(p.parentPrimary / "INVALID") ==
                    "processing finished; primary geometry/calendar invalid; no "
                    "inferential completion\n",
                "unrecognized_parent_INVALID");
    else
        S::ValidateArtifactPublication(p.parentPrimary);
    for (const auto &name : files)
        if (name == "geometry_measurements.csv.gz" || name == "interaction_episodes.csv.gz")
        {
            // Validate complete CSV gzip members, including CRC and trailer.
            S::InspectGzip(p.parentPrimary / name);
        }
    Require(Read(p.parentPrimary / "protocol.json")
                    .find("\"protocol_fingerprint\":" + Json(p.semanticFingerprint)) != std::string::npos,
            "parent_protocol_mismatch");
    const auto provenance = Read(p.parentPrimary / "provenance_manifest.json");
    for (const auto &[key, value] : std::array<std::pair<std::string, std::string>, 3>{
             {{"git_commit", p.gitCommit},
              {"source_manifest_sha256", p.sourceSnapshotDigest},
              {"canonical_candle_content_sha256", p.candleDigest}}})
        Require(provenance.find(Json(key) + ":" + Json(value)) != std::string::npos,
                "parent_provenance_mismatch");
    return invalid ? "INVALID" : "COMPLETE";
}
} // namespace
struct Audit::State
{
    TG3::Configuration fc;
    TG1A::Configuration gc;
    H::DecisionAuditConfiguration config;
    std::filesystem::path dir;
    Budgets budgets;
    Hook hook;
    Summary summary;
    TG3::FibonacciConfluenceTracker fibonacci;
    TG1A::CausalFractalTrendLineGeometry geometry;
    std::vector<AB> victims;
    std::optional<H::PrimaryDecisionObservation> pending;
    std::int64_t nextSlot;
    S::Sha256 candleHash;
    std::ofstream decisions, candidates, evictions;
    bool stopped = false, finished = false, outputStopped = false;
    State(TG3::Configuration f, TG1A::Configuration g, H::DecisionAuditConfiguration c,
          std::filesystem::path d, Budgets b, Hook h)
        : fc(std::move(f)), gc(g), config(std::move(c)), dir(std::move(d)), budgets(b), hook(std::move(h)),
          fibonacci(fc, {"AUDCAD", "15m"}), geometry(gc, {"AUDCAD", "15m"}),
          nextSlot(config.scoreStart / 3600 * 3600 + 2700)
    {
        if (nextSlot < config.scoreStart)
            nextSlot += 3600;
        Require(fc.maxActiveABStructures == 512 && fc.maxConfirmedFractalsPerKind == 128 &&
                    fc.maxABAgeBars == 2048 &&
                    fibonacci.GetConfiguration().retracementRatios == std::vector<double>({.382, .5, .618}) &&
                    gc.maxCandidates == 4096 && gc.maxConfirmedFractalsPerKind == 64 &&
                    gc.maxFractalAnchorLookbackBars == 512 && gc.maxCandidateAgeBars == 512 &&
                    gc.atrPeriod == 14,
                "non_frozen_detector_configuration");
        Require(S::IsHash(config.semanticFingerprint) && config.scoreStart >= S::kOrigin &&
                    config.scoreStart < config.scoreEnd && config.scoreEnd <= S::kFinalStart &&
                    budgets.liveABRecords && budgets.memoryBytes &&
                    budgets.decodedOutputBytes > kMetadataReserve + 4096 && budgets.pairsPerDecision &&
                    budgets.totalPairs,
                "invalid_diagnostic_configuration");
        Require(!std::filesystem::exists(dir) && std::filesystem::create_directories(dir),
                "diagnostic_directory_must_be_new");
        for (auto *fptr : {&decisions, &candidates, &evictions})
            fptr->exceptions(std::ios::badbit | std::ios::failbit);
        decisions.open(dir / "decisions.csv");
        candidates.open(dir / "restored_candidates.csv");
        evictions.open(dir / "source_evictions.csv");
        Append(decisions, "decision_id,bar_start_utc_epoch,primary_status,decision_window_"
                          "eligible,status,actual_status,counterfactual_status,actual_Ffib,"
                          "actual_Lline,actual_I,actual_cohort,counterfactual_Ffib,"
                          "counterfactual_Lline,counterfactual_I,counterfactual_cohort,actual_"
                          "ABs,counterfactual_ABs,lines,restored_present,restored_age_"
                          "eligible,restored_upward,restored_eligible_pairs,restored_near_ABs,"
                          "restored_conjunction_ABs,changed_Ffib,changed_Lline,changed_I,"
                          "changed_cohort,reason\n");
        Append(candidates, "decision_id,AB_identity,availability_bar,age,present,age_eligible,"
                           "upward,eligible_pairs,fib_near,conjunction\n");
        Append(evictions, "bar,bar_start_utc_epoch,phase,AB_identity,direction,a_bar,a_utc_"
                          "epoch,a_price,a_confirmation_bar,a_confirmation_utc_epoch,b_bar,b_"
                          "utc_epoch,b_price,b_confirmation_bar,b_confirmation_utc_epoch,"
                          "availability_bar,availability_utc_epoch,reason\n");
    }
    void Call(Step s)
    {
        if (hook)
            hook(s);
    }
    void Fail(std::string reason, bool actual = true)
    {
        if (summary.failure.empty())
            summary.failure = std::move(reason);
        summary.actualComplete = summary.actualComplete && !actual;
        summary.counterfactualComplete = false;
        stopped = true;
    }
    void Resources(std::size_t live, std::size_t lines)
    {
        // Conservative reservation covers native mirrors including spare vector
        // capacity, streams, transient copies, two geometry maps and strings.
        // Forecast BEFORE allocation; counters report this bound, not RSS.
        const std::size_t memory = 16 * 1024 * 1024 + live * 2048 + lines * 4096;
        summary.peakLiveAB = std::max(summary.peakLiveAB, live);
        summary.peakMemoryBytes = std::max(summary.peakMemoryBytes, memory);
        Require(live <= budgets.liveABRecords, "live_AB_record_budget");
        Require(memory <= budgets.memoryBytes, "observer_memory_budget");
    }
    void Append(std::ofstream &stream, std::string_view row)
    {
        if (outputStopped)
            throw std::runtime_error("diagnostic_output_budget");
        if (row.size() > budgets.decodedOutputBytes - kMetadataReserve - summary.outputBytes)
        {
            outputStopped = true;
            throw std::runtime_error("diagnostic_output_budget");
        }
        Call(Step::Write);
        stream << row;
        summary.outputBytes += row.size();
    }
    void Unknown(const H::PrimaryDecisionObservation &p, std::string_view status, bool missing = false)
    {
        if (missing)
            ++summary.missing;
        else
            ++summary.unknown;
        try
        {
            Append(decisions, p.identity + "," + std::to_string(p.timestamp) + "," + p.status + ",UNKNOWN," +
                                  std::string(status) +
                                  ",UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,"
                                  "UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,"
                                  "UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,"
                                  "UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN," +
                                  Csv(summary.failure.empty() ? std::string(status) : summary.failure) +
                                  "\n");
            ++summary.decisionRows;
        }
        catch (...)
        {
            ++summary.unpublishedDecisions;
            Fail("diagnostic_output_failure");
        }
    }
    void Primary(const H::PrimaryDecisionObservation &p)
    {
        Require(!finished && p.timestamp == nextSlot && p.timestamp < config.scoreEnd &&
                    p.identity == config.semanticFingerprint + ":" + std::to_string(p.timestamp),
                "primary_decision_clock_parity");
        Require(!pending, "unconsumed_primary_decision");
        Require(p.completedCandle || p.status == "missing_slot_unverified" ||
                    p.status == "missing_open_session" || p.status == "closed_market",
                "missing_slot_status_parity");
        Require(p.cohort.empty() || p.status == "classified", "primary_cohort_status_parity");
        nextSlot += 3600;
        ++summary.scheduled;
        if (!p.completedCandle)
        {
            Unknown(p, "UNOBSERVABLE_MISSING_SLOT", true);
            return;
        }
        if (stopped)
        {
            summary.actualComplete = false;
            Unknown(p, "UNKNOWN_AFTER_FAILURE");
            return;
        }
        pending = p;
    }
    void Capture(const H::CompletedBarObservation &o)
    {
        Call(Step::ActualParity);
        auto gu = geometry.AddCompletedBar(o.candle);
        Require(Equal(gu.newlyConfirmedFractals, o.confirmed, [](const auto &f) { return Snapshot(f); }),
                "TG1A_confirmation_parity");
        Require(Equal(gu.newlyCreatedCandidates, o.lineBirths, [](const auto &l) { return Snapshot(l); }),
                "TG1A_birth_order_parity");
        Require(Equal(geometry.Candidates(), o.retainedLines, [](const auto &l) { return Snapshot(l); }) &&
                    geometry.CurrentAtr() == o.atr,
                "TG1A_retained_order_or_ATR_parity");
        // Expiration is native and precedes capacity enforcement. Never label
        // an Advance removal as a capacity victim.
        fibonacci.Advance(o.bar, o.candle.timestamp);
        std::vector<TG1A::ConfirmedFractal> confirmed(o.confirmed.begin(), o.confirmed.end());
        std::sort(confirmed.begin(), confirmed.end(),
                  [](const auto &a, const auto &b)
                  {
                      return std::tuple(a.confirmationBar, a.anchorBar, static_cast<int>(a.kind),
                                        a.anchorTimestamp) < std::tuple(b.confirmationBar, b.anchorBar,
                                                                        static_cast<int>(b.kind),
                                                                        b.anchorTimestamp);
                  });
        std::vector<TG3::ABIdentity> births;
        for (const auto &f : confirmed)
        {
            const auto before = fibonacci.ABStructures();
            const auto evicted = fibonacci.ABCapacityEvictions();
            auto born = fibonacci.ObserveConfirmedFractals({f});
            births.insert(births.end(), born.begin(), born.end());
            const auto &after = fibonacci.ABStructures();
            std::size_t removed = 0;
            for (const auto &a : before)
                if (std::none_of(after.begin(), after.end(),
                                 [&](const auto &x) { return Identity(a.identity) == Identity(x.identity); }))
                {
                    ++removed;
                    victims.push_back(a);
                    ++summary.evictions[0];
                    ++summary.evictions[o.scoring ? 2 : 1];
                    if (!summary.firstEvictionBar)
                    {
                        summary.firstEvictionBar = o.bar;
                        summary.firstEvictionTimestamp = o.candle.timestamp;
                    }
                    const auto &i = a.identity;
                    Append(evictions,
                           N(o.bar) + "," + std::to_string(o.candle.timestamp) + "," +
                               (o.scoring ? "scoring" : "prefix") + "," +
                               H::CandidateIdentityReconciliation::Native(i) + "," +
                               N(static_cast<std::size_t>(i.direction)) + "," + N(i.aBar) + "," +
                               std::to_string(i.aTimestamp) + "," + S::Number(a.aPrice) + "," +
                               N(a.aConfirmationBar) + "," + std::to_string(a.aConfirmationTimestamp) + "," +
                               N(i.bBar) + "," + std::to_string(i.bTimestamp) + "," + S::Number(a.bPrice) +
                               "," + N(a.bConfirmationBar) + "," + std::to_string(a.bConfirmationTimestamp) +
                               "," + N(i.availabilityBar) + "," + std::to_string(i.availabilityTimestamp) +
                               ",SOURCE_AB_CAPACITY\n");
                }
            Require(removed == fibonacci.ABCapacityEvictions() - evicted,
                    "uncaptured_same_bar_capacity_victim");
        }
        Require(Equal(births, o.abBirths, [](const auto &i) { return Identity(i); }),
                "TG3_birth_order_parity");
        Require(Equal(fibonacci.ABStructures(), o.retainedAB, [](const auto &a) { return Snapshot(a); }),
                "TG3_retained_order_parity");
        Require(fibonacci.TotalABStructuresCreated() == o.totalABCreated &&
                    fibonacci.ABCapacityEvictions() == o.abCapacityEvictions &&
                    fibonacci.ABAgeExpirations() == o.abAgeExpirations &&
                    summary.evictions[0] == o.abCapacityEvictions,
                "TG3_lifecycle_parity");
        ++summary.parityChecks;
    }
    H::DecisionGeometry Evaluate(const H::CompletedBarObservation &o, std::span<const AB> abs,
                                 std::size_t &work)
    {
        auto remaining = std::min(budgets.pairsPerDecision - work, budgets.totalPairs - summary.totalPairs);
        auto r = H::EvaluateDecisionGeometry(o.bar, o.candle, o.atr, abs, o.retainedLines,
                                             {budgets.liveABRecords, 4096, remaining});
        if (!r.reconstructed && r.reason == "pair_work_budget")
            r.reason = r.eligibleAB * r.eligibleLines > budgets.totalPairs - summary.totalPairs
                           ? "total_pair_work_budget"
                           : "decision_pair_work_budget";
        if (r.reconstructed)
        {
            auto pairs = r.eligibleAB * r.eligibleLines;
            work += pairs;
            summary.totalPairs += pairs;
        }
        summary.peakPairs = std::max(summary.peakPairs, work);
        return r;
    }
    void Decision(const H::CompletedBarObservation &o, const H::PrimaryDecisionObservation &p)
    {
        std::size_t work = 0;
        auto actual = Evaluate(o, o.retainedAB, work);
        if (!actual.reconstructed)
        {
            Fail("actual_" + actual.reason);
            Unknown(p, "UNKNOWN_ACTUAL");
            return;
        }
        if (p.geometryComplete && p.geometryEvaluated)
        {
            Require(p.eligibleAB == actual.eligibleAB && p.eligibleLines == actual.eligibleLines &&
                        p.fibNear == actual.fibNear.value_or(false) &&
                        p.lineNear == actual.lineNear.value_or(false) &&
                        p.intersection == actual.intersection.value_or(false) &&
                        (p.cohort.empty() || p.cohort == actual.cohort),
                    "primary_geometry_parity");
            ++summary.primaryParityChecks;
        }
        // Capacity-invalid primary predicates are deliberately never consulted.
        H::DecisionGeometry cf;
        try
        {
            Call(Step::Counterfactual);
            std::vector<AB> restored(o.retainedAB.begin(), o.retainedAB.end());
            restored.insert(restored.end(), victims.begin(), victims.end());
            std::sort(restored.begin(), restored.end(),
                      [](const auto &a, const auto &b) { return Order(a) < Order(b); });
            cf = Evaluate(o, restored, work);
        }
        catch (const std::exception &e)
        {
            cf.reason = e.what();
        }
        catch (...)
        {
            cf.reason = "allocation_or_unknown_failure";
        }
        if (!cf.reconstructed)
        {
            Fail(cf.reason.starts_with("counterfactual_") ? cf.reason : "counterfactual_" + cf.reason, false);
            ++summary.unknown;
            // Retain independently reconstructed ACTUAL values when CF work is
            // unavailable; do not replace them with primary defaults.
            Append(decisions, p.identity + "," + std::to_string(p.timestamp) + "," + p.status + "," +
                                  B(o.decisionWindowEligible) +
                                  ",UNKNOWN_COUNTERFACTUAL,RECONSTRUCTED,UNKNOWN," + Value(actual.fibNear) +
                                  "," + Value(actual.lineNear) + "," + Value(actual.intersection) + "," +
                                  actual.cohort + ",UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN," + N(actual.eligibleAB) +
                                  ",UNKNOWN," + N(actual.eligibleLines) +
                                  ",UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,"
                                  "UNKNOWN,UNKNOWN,UNKNOWN," +
                                  Csv(summary.failure) + "\n");
            ++summary.decisionRows;
            return;
        }
        std::size_t up = 0, near = 0, conjunction = 0;
        for (const auto &a : victims)
        {
            const auto key = S::ABKey(a);
            const bool eligible = a.identity.direction == TG3::ABDirection::UpAB && cf.abs.contains(key);
            const bool n = eligible && cf.abGeometry.at(key).near,
                       c = eligible && cf.abGeometry.at(key).conjunction;
            up += eligible;
            near += n;
            conjunction += c;
            Append(candidates,
                   p.identity + "," + H::CandidateIdentityReconciliation::Native(a.identity) + "," +
                       N(a.identity.availabilityBar) + "," + N(o.bar - a.identity.availabilityBar) + ",1,1," +
                       B(a.identity.direction == TG3::ABDirection::UpAB) + "," +
                       N(eligible ? cf.eligibleLines : 0) + "," + (eligible ? B(n) : "INELIGIBLE") + "," +
                       (eligible ? B(c) : "INELIGIBLE") + "\n");
            ++summary.candidateRows;
        }
        auto changed = [](std::optional<bool> a, std::optional<bool> b) -> std::optional<bool>
        { return a && b ? std::optional{*a != *b} : std::nullopt; };
        auto f = changed(actual.fibNear, cf.fibNear), l = changed(actual.lineNear, cf.lineNear),
             i = changed(actual.intersection, cf.intersection);
        const bool cohort = actual.cohort != cf.cohort;
        summary.changedF += f.value_or(false);
        summary.changedL += l.value_or(false);
        summary.changedI += i.value_or(false);
        summary.changedCohort += cohort;
        Append(decisions, p.identity + "," + std::to_string(p.timestamp) + "," + p.status + "," +
                              B(o.decisionWindowEligible) + ",RECONSTRUCTED,RECONSTRUCTED,RECONSTRUCTED," +
                              Value(actual.fibNear) + "," + Value(actual.lineNear) + "," +
                              Value(actual.intersection) + "," + actual.cohort + "," + Value(cf.fibNear) +
                              "," + Value(cf.lineNear) + "," + Value(cf.intersection) + "," + cf.cohort +
                              "," + N(actual.eligibleAB) + "," + N(cf.eligibleAB) + "," +
                              N(cf.eligibleLines) + "," + N(victims.size()) + "," + N(victims.size()) + "," +
                              N(up) + "," + N(up * cf.eligibleLines) + "," + N(near) + "," + N(conjunction) +
                              "," + Value(f) + "," + Value(l) + "," + Value(i) + "," + B(cohort) + ",\n");
        ++summary.decisionRows;
        ++summary.reconstructed;
    }
    void Observe(const H::CompletedBarObservation &o)
    {
        Require(!finished && o.bar == summary.bars && o.candle.timestamp >= S::kOrigin &&
                    o.candle.timestamp < config.scoreEnd && o.candle.timestamp % 900 == 0 &&
                    o.scoring == (o.candle.timestamp >= config.scoreStart),
                "completed_bar_clock_parity");
        candleHash.Add(std::to_string(o.candle.timestamp) + "," + S::Number(o.candle.open) + "," +
                       S::Number(o.candle.high) + "," + S::Number(o.candle.low) + "," +
                       S::Number(o.candle.close) + "\n");
        ++summary.bars;
        if (stopped)
            return;
        Require(o.scheduled == (o.scoring && o.candle.timestamp % 3600 == 2700), "scheduled_clock_parity");
        const auto beforeExpiration = victims.size();
        std::erase_if(victims,
                      [&](const auto &a)
                      {
                          return o.bar > a.identity.availabilityBar &&
                                 o.bar - a.identity.availabilityBar > fc.maxABAgeBars;
                      });
        summary.victimAgeExpirations += beforeExpiration - victims.size();
        Require(o.retainedAB.size() <= 512 && o.retainedLines.size() <= 4096 && o.confirmed.size() <= 2,
                "frozen_snapshot_bound");
        Resources(victims.size() + o.retainedAB.size() + o.confirmed.size(),
                  std::max(o.retainedLines.size(), geometry.Candidates().size()) + 128);
        Capture(o);
        Resources(victims.size() + o.retainedAB.size(), o.retainedLines.size());
        if (o.scheduled)
        {
            Require(pending && pending->timestamp == o.candle.timestamp, "missing_primary_decision");
            auto p = *pending;
            Decision(o, p);
            pending.reset();
        }
    }
    void Metadata(std::string_view name, std::string_view data)
    {
        Require(data.size() <= budgets.decodedOutputBytes - summary.outputBytes, "metadata_output_budget");
        Call(Step::Write);
        std::ofstream f(dir / name, std::ios::binary);
        f.exceptions(std::ios::badbit | std::ios::failbit);
        f << data;
        f.close();
        summary.outputBytes += data.size();
    }
    bool Complete(const H::Provenance &p)
    {
        Require(!finished, "already_finalized");
        finished = true;
        if (pending)
        {
            Unknown(*pending, "UNKNOWN_UNCONSUMED_DECISION");
            pending.reset();
            Fail("unconsumed_primary_decision");
        }
        if (nextSlot < config.scoreEnd)
            Fail("incomplete_decision_clock");
        Require(p.semanticFingerprint == config.semanticFingerprint && p.candleDigest == candleHash.Digest(),
                "diagnostic_provenance_parity");
        auto parent = CheckParent(p);
        Call(Step::Finalize);
        for (auto *fptr : {&decisions, &candidates, &evictions})
            fptr->close();
        std::string protocol = "{\"schema\":" + Json(kSchema) +
                               ",\"semantic_protocol\":\"fib-trendline-decision-study-v2\","
                               "\"intervention\":\"source_AB_capacity_only\",\"restore_fractal_"
                               "history\":false,\"source_AB_capacity\":512,\"AB_age_inclusive\":2048,"
                               "\"TG3_history_per_kind\":128,\"TG1A_history_per_kind\":64,\"line_"
                               "capacity\":4096,\"line_age_inclusive\":512,\"ATR_period\":14,"
                               "\"ratios\":[0.382,0.500,0.618],\"market_threshold\":0.25,\"structural_"
                               "threshold\":0.25,\"score_start\":" +
                               std::to_string(config.scoreStart) +
                               ",\"score_end\":" + std::to_string(config.scoreEnd) +
                               ",\"causal_origin\":" + std::to_string(S::kOrigin) +
                               ",\"live_AB_budget\":" + N(budgets.liveABRecords) +
                               ",\"memory_byte_budget\":" + N(budgets.memoryBytes) +
                               ",\"decoded_output_byte_budget\":" + N(budgets.decodedOutputBytes) +
                               ",\"pair_work_per_decision\":" + N(budgets.pairsPerDecision) +
                               ",\"pair_work_total\":" + N(budgets.totalPairs) +
                               ",\"metadata_reserve_bytes\":" + N(kMetadataReserve) +
                               ",\"memory_accounting\":\"conservative reservation including transient "
                               "geometry; not RSS\",\"missing_geometry\":\"INELIGIBLE_NO_GEOMETRY "
                               "with UNKNOWN "
                               "predicates\",\"primary_validity\":\"UNCHANGED\",\"statistical_"
                               "estimation\":\"NOT_EXECUTED\"}\n";
        Metadata("audit_protocol.json", protocol);
        Metadata("audit_provenance.json",
                 "{\"schema\":" + Json(kSchema) + ",\"semantic_fingerprint\":" + Json(p.semanticFingerprint) +
                     ",\"git_commit\":" + Json(p.gitCommit) + ",\"source_manifest_sha256\":" +
                     Json(p.sourceSnapshotDigest) + ",\"canonical_candle_sha256\":" + Json(p.candleDigest) +
                     ",\"parent_status\":" + Json(parent) + ",\"parent_checksums_sha256\":" +
                     Json(S::FileHash(p.parentPrimary / "checksums.sha256")) +
                     ",\"parent_marker_sha256\":" + Json(S::FileHash(p.parentPrimary / parent)) +
                     ",\"parent_detector_configuration_sha256\":" +
                     Json(S::FileHash(p.parentPrimary / "detector_configuration.json")) +
                     ",\"TG3_source_sha256\":" + Json(H::kTG3SourceSHA256) +
                     ",\"TG1A_source_sha256\":" + Json(H::kTG1ASourceSHA256) + "}\n");
        std::string stats = "metric,value,status\n";
        auto count = [&](std::string_view key, std::size_t n)
        { stats += std::string(key) + "," + N(n) + ",OBSERVED\n"; };
        for (auto [k, n] : std::initializer_list<std::pair<const char *, std::size_t>>{
                 {"bars", summary.bars},
                 {"parity_checks", summary.parityChecks},
                 {"scheduled", summary.scheduled},
                 {"reconstructed", summary.reconstructed},
                 {"unknown", summary.unknown},
                 {"missing", summary.missing},
                 {"unpublished_decisions", summary.unpublishedDecisions},
                 {"primary_parity_checks", summary.primaryParityChecks},
                 {"source_evictions_full", summary.evictions[0]},
                 {"source_evictions_prefix", summary.evictions[1]},
                 {"source_evictions_scoring", summary.evictions[2]},
                 {"victim_age_expirations", summary.victimAgeExpirations},
                 {"peak_live_AB_records", summary.peakLiveAB},
                 {"peak_accounted_memory_bytes", summary.peakMemoryBytes},
                 {"peak_decision_pairs", summary.peakPairs},
                 {"total_pairs", summary.totalPairs},
                 {"decision_rows", summary.decisionRows},
                 {"candidate_rows", summary.candidateRows}})
            count(k, n);
        const bool complete = summary.actualComplete && summary.counterfactualComplete && summary.parity &&
                              !summary.unknown && !summary.unpublishedDecisions;
        for (auto [k, n] : std::initializer_list<std::pair<const char *, std::size_t>>{
                 {"changed_Ffib", summary.changedF},
                 {"changed_Lline", summary.changedL},
                 {"changed_I", summary.changedI},
                 {"changed_cohort", summary.changedCohort}})
        {
            count(std::string("verified_") + k, n);
            stats += std::string(k) + "," + (complete ? N(n) : "UNKNOWN") + "," +
                     (complete ? "COMPLETE_OBSERVED_CLOCK" : "INCOMPLETE") + "\n";
        }
        stats += "first_eviction_bar," +
                 (summary.firstEvictionBar ? N(*summary.firstEvictionBar) : "UNKNOWN") +
                 ",OBSERVED_OR_UNAVAILABLE\n";
        stats +=
            "first_eviction_utc_epoch," +
            (summary.firstEvictionTimestamp ? std::to_string(*summary.firstEvictionTimestamp) : "UNKNOWN") +
            ",OBSERVED_OR_UNAVAILABLE\n";
        stats += "actual_complete," + B(summary.actualComplete) + ",DIAGNOSTIC\ncounterfactual_complete," +
                 B(summary.counterfactualComplete) + ",DIAGNOSTIC\nparity," + B(summary.parity) +
                 ",DIAGNOSTIC\nfailure," + Csv(summary.failure) + ",DIAGNOSTIC\n";
        // A final deterministic byte total includes summary, ledger and marker.
        // Fixed-point decimal length avoids excluding publication metadata.
        std::size_t finalBytes = summary.outputBytes + stats.size() + 1024;
        for (int i = 0; i < 4; ++i)
            finalBytes = summary.outputBytes + stats.size() +
                         std::string("decoded_output_bytes," + N(finalBytes) + ",OBSERVED\n").size() +
                         6 * (64 + 2 + 1) +
                         std::string("audit_protocol.jsonaudit_provenance.jsonsummary.csvdecisions."
                                     "csvrestored_candidates.csvsource_evictions.csv")
                             .size() +
                         (complete ? std::string("schema=" + std::string(kSchema) +
                                                 "\nprocessing=AUDIT_COMPLETE\nprimary_"
                                                 "validity=UNCHANGED\nparent_status=" +
                                                 parent + "\nchecksums_sha256=" + std::string(64, '0') + "\n")
                                         .size()
                                   : std::string("schema=" + std::string(kSchema) +
                                                 "\nprocessing=INCOMPLETE\nreason=" + summary.failure + "\n")
                                         .size());
        stats += "decoded_output_bytes," + N(finalBytes) + ",OBSERVED\n";
        Metadata("summary.csv", stats);
        std::string ledger;
        Call(Step::Checksum);
        for (const auto *name : {"audit_protocol.json", "audit_provenance.json", "decisions.csv",
                                 "restored_candidates.csv", "source_evictions.csv", "summary.csv"})
            ledger += S::FileHash(dir / name) + "  " + name + "\n";
        Metadata("audit_checksums.sha256", ledger);
        VerifyChecksums(dir, "audit_checksums.sha256");
        if (!complete)
        {
            Metadata("INCOMPLETE", "schema=" + std::string(kSchema) +
                                       "\nprocessing=INCOMPLETE\nreason=" + summary.failure + "\n");
            return false;
        }
        Call(Step::Marker);
        Metadata("AUDIT_COMPLETE.pending",
                 "schema=" + std::string(kSchema) +
                     "\nprocessing=AUDIT_COMPLETE\nprimary_validity="
                     "UNCHANGED\nparent_status=" +
                     parent + "\nchecksums_sha256=" + S::FileHash(dir / "audit_checksums.sha256") + "\n");
        std::filesystem::rename(dir / "AUDIT_COMPLETE.pending", dir / "AUDIT_COMPLETE");
        return true;
    }
};
Audit::Audit(TG3::Configuration f, TG1A::Configuration g, H::DecisionAuditConfiguration c,
             std::filesystem::path d, Budgets b, Hook h)
    : state_(std::make_unique<State>(std::move(f), g, std::move(c), std::move(d), b, std::move(h)))
{
}
Audit::~Audit() = default;
void Audit::ObservePrimaryDecision(const H::PrimaryDecisionObservation &p) noexcept
{
    try
    {
        state_->Primary(p);
    }
    catch (const std::exception &e)
    {
        state_->summary.parity = false;
        state_->Fail(e.what());
    }
    catch (...)
    {
        state_->Fail("primary_observer_failure");
    }
}
void Audit::Observe(const H::CompletedBarObservation &o) noexcept
{
    try
    {
        state_->Observe(o);
    }
    catch (const std::exception &e)
    {
        const bool cf = std::string_view(e.what()).starts_with("counterfactual_");
        if (!cf && std::string_view(e.what()).find("parity") != std::string_view::npos)
            state_->summary.parity = false;
        state_->Fail(e.what(), !cf);
        if (state_->pending)
        {
            state_->Unknown(*state_->pending, "UNKNOWN_RECONSTRUCTION");
            state_->pending.reset();
        }
    }
    catch (...)
    {
        state_->Fail("observer_allocation_or_unknown_failure");
    }
}
const Summary &Audit::GetSummary() const { return state_->summary; }
bool Audit::Complete(const H::Provenance &p) noexcept
{
    try
    {
        return state_->Complete(p);
    }
    catch (const std::exception &e)
    {
        state_->Fail(e.what());
    }
    catch (...)
    {
        state_->Fail("publication_failure");
    }
    // Best-effort bounded failure marker. Never replace a primary artifact or
    // create a completion marker when stream finalization/publication failed.
    try
    {
        const auto text = "schema=" + std::string(kSchema) +
                          "\nprocessing=INCOMPLETE\nreason=" + state_->summary.failure + "\n";
        if (text.size() <= state_->budgets.decodedOutputBytes - state_->summary.outputBytes)
        {
            std::ofstream status(state_->dir / "INCOMPLETE", std::ios::binary);
            status << text;
            status.close();
        }
    }
    catch (...)
    {
    }
    return false;
}
void ValidateSidecar(const std::filesystem::path &dir, const std::filesystem::path &parent)
{
    const auto files = VerifyChecksums(dir, "audit_checksums.sha256");
    Require(files == std::set<std::string>{"audit_protocol.json", "audit_provenance.json", "decisions.csv",
                                           "restored_candidates.csv", "source_evictions.csv", "summary.csv"},
            "incomplete_diagnostic_manifest");
    auto marker = Read(dir / "AUDIT_COMPLETE"), protocol = Read(dir / "audit_protocol.json"),
         provenance = Read(dir / "audit_provenance.json"), stats = Read(dir / "summary.csv");
    Require(marker.find("schema=" + std::string(kSchema) +
                        "\nprocessing=AUDIT_COMPLETE\nprimary_validity=UNCHANGED\n") == 0 &&
                marker.find("checksums_sha256=" + S::FileHash(dir / "audit_checksums.sha256") + "\n") !=
                    std::string::npos &&
                protocol.find("\"schema\":" + Json(kSchema)) != std::string::npos &&
                stats.find("actual_complete,1,DIAGNOSTIC\n") != std::string::npos &&
                stats.find("counterfactual_complete,1,DIAGNOSTIC\n") != std::string::npos &&
                stats.find("parity,1,DIAGNOSTIC\n") != std::string::npos &&
                stats.find("unknown,0,OBSERVED\n") != std::string::npos &&
                stats.find("unpublished_decisions,0,OBSERVED\n") != std::string::npos,
            "incomplete_diagnostic_marker");
    const auto status = std::filesystem::exists(parent / "INVALID") ? "INVALID" : "COMPLETE";
    Require(!std::filesystem::exists(dir / "INCOMPLETE") &&
                std::filesystem::exists(parent / "INVALID") != std::filesystem::exists(parent / "COMPLETE") &&
                provenance.find("\"parent_checksums_sha256\":" +
                                Json(S::FileHash(parent / "checksums.sha256"))) != std::string::npos &&
                provenance.find("\"parent_marker_sha256\":" + Json(S::FileHash(parent / status))) !=
                    std::string::npos &&
                marker.find(std::string("parent_status=") + status + "\n") != std::string::npos,
            "parent_linkage_mismatch");
    VerifyChecksums(parent, "checksums.sha256");
}

InterventionReplay::InterventionReplay(TG3::Configuration f, TG1A::Configuration g)
    : actual_(f, {"AUDCAD", "15m"}), mirror_(f, {"AUDCAD", "15m"}), geometry_(g, {"AUDCAD", "15m"})
{
    Require(f.maxActiveABStructures == 512 && f.maxConfirmedFractalsPerKind == 128 &&
                f.maxABAgeBars == 2048 && g.maxCandidates == 4096 && g.maxConfirmedFractalsPerKind == 64 &&
                g.maxCandidateAgeBars == 512 && g.maxFractalAnchorLookbackBars == 512 && g.atrPeriod == 14,
            "non_frozen_intervention_configuration");
}
void InterventionReplay::AddCompletedBar(const TG1A::Candle &c)
{
    Require(c.timestamp >= S::kOrigin && c.timestamp < S::kFinalStart && c.timestamp % 900 == 0,
            "intervention_causal_range");
    const auto bar = geometry_.CompletedBarCount();
    std::erase_if(
        victims_, [&](const auto &v)
        { return bar > v.ab.identity.availabilityBar && bar - v.ab.identity.availabilityBar > 2048; });
    // Reserve before transient births. Diagnostic bounds do not alter detectors.
    Require(victims_.size() + actual_.ABStructures().size() + 2 <= 8192, "intervention_live_AB_budget");
    auto update = geometry_.AddCompletedBar(c);
    actual_.Advance(bar, c.timestamp);
    mirror_.Advance(bar, c.timestamp);
    auto births = actual_.ObserveConfirmedFractals(update.newlyConfirmedFractals);
    auto confirmed = update.newlyConfirmedFractals;
    std::sort(
        confirmed.begin(), confirmed.end(),
        [](const auto &a, const auto &b)
        {
            return std::tuple(a.confirmationBar, a.anchorBar, static_cast<int>(a.kind), a.anchorTimestamp) <
                   std::tuple(b.confirmationBar, b.anchorBar, static_cast<int>(b.kind), b.anchorTimestamp);
        });
    std::vector<TG3::ABIdentity> mirrorBirths;
    for (const auto &f : confirmed)
    {
        const auto before = mirror_.ABStructures();
        const auto count = mirror_.ABCapacityEvictions();
        auto created = mirror_.ObserveConfirmedFractals({f});
        mirrorBirths.insert(mirrorBirths.end(), created.begin(), created.end());
        std::size_t removed = 0;
        for (const auto &a : before)
            if (std::none_of(mirror_.ABStructures().begin(), mirror_.ABStructures().end(),
                             [&](const auto &b) { return Identity(a.identity) == Identity(b.identity); }))
            {
                victims_.push_back({a, bar, c.timestamp});
                ++removed;
            }
        Require(removed == mirror_.ABCapacityEvictions() - count, "intervention_eviction_capture_parity");
    }
    Require(Equal(births, mirrorBirths, [](const auto &x) { return Identity(x); }) &&
                Equal(actual_.ABStructures(), mirror_.ABStructures(),
                      [](const auto &x) { return Snapshot(x); }) &&
                actual_.ABCapacityEvictions() == mirror_.ABCapacityEvictions() &&
                actual_.ABAgeExpirations() == mirror_.ABAgeExpirations() &&
                actual_.TotalABStructuresCreated() == mirror_.TotalABStructuresCreated(),
            "intervention_native_identity_order_parity");
    ++parityChecks_;
}
std::size_t RemainingInterventionPairs(std::size_t budget, std::size_t accumulated)
{
    Require(accumulated <= budget, "intervention_accumulated_pairs_exceed_budget");
    return budget - accumulated;
}
std::size_t AccumulateInterventionPairs(std::size_t budget, std::size_t accumulated, std::size_t eligibleAB,
                                        std::size_t eligibleLines)
{
    const auto remaining = RemainingInterventionPairs(budget, accumulated);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    Require(!eligibleLines || eligibleAB <= maximum / eligibleLines,
            "intervention_pair_multiplication_overflow");
    const auto pairs = eligibleAB * eligibleLines;
    Require(pairs <= maximum - accumulated, "intervention_pair_accumulation_overflow");
    Require(pairs <= remaining, "intervention_evaluation_exceeds_remaining_pairs");
    return accumulated + pairs;
}
InterventionResult EvaluateInterventions(std::size_t bar, const TG1A::Candle &candle,
                                         std::optional<double> atr, std::span<const AB> actual,
                                         std::span<const Line> lines, std::span<const CapacityVictim> victims,
                                         std::span<const std::string> qualifying, std::size_t budget)
{
    Require(candle.timestamp % 3600 == 2700 && candle.timestamp >= S::kOrigin &&
                candle.timestamp < S::kFinalStart,
            "intervention_decision_timestamp");
    Require(actual.size() <= 512 && lines.size() <= 4096 && actual.size() + victims.size() <= 8192 &&
                qualifying.size() <= 5,
            "intervention_input_budget");
    InterventionResult out;
    auto evaluate = [&](std::span<const AB> abs)
    {
        const auto remaining = RemainingInterventionPairs(budget, out.pairs);
        auto g = H::EvaluateDecisionGeometry(bar, candle, atr, abs, lines, {8192, 4096, remaining});
        Require(g.reconstructed, "intervention_geometry_incomplete");
        out.pairs = AccumulateInterventionPairs(budget, out.pairs, g.eligibleAB, g.eligibleLines);
        return g;
    };
    out.actual = evaluate(actual);
    std::vector<AB> complete(actual.begin(), actual.end());
    std::map<std::string, AB> live;
    for (const auto &v : victims)
    {
        Require(v.evictionBar <= bar && v.evictionTimestamp <= candle.timestamp &&
                    v.evictionBar >= v.ab.identity.availabilityBar &&
                    v.evictionTimestamp >= v.ab.identity.availabilityTimestamp,
                "intervention_eviction_provenance");
        Require(bar >= v.ab.identity.availabilityBar && bar - v.ab.identity.availabilityBar <= 2048,
                "intervention_expired_victim");
        const auto native = H::CandidateIdentityReconciliation::Native(v.ab.identity);
        Require(live.emplace(native, v.ab).second, "intervention_duplicate_victim");
        Require(std::none_of(actual.begin(), actual.end(),
                             [&](const auto &a) { return Identity(a.identity) == Identity(v.ab.identity); }),
                "intervention_victim_retained_overlap");
        complete.push_back(v.ab);
    }
    std::sort(complete.begin(), complete.end(),
              [](const auto &a, const auto &b) { return Order(a) < Order(b); });
    out.complete = evaluate(complete);
    // Verify the entire restoration universe, including nonqualifying/direction
    // victims. A forensic short list is never substituted for complete state.
    std::set<std::string> discovered;
    for (const auto &[native, a] : live)
        if (a.identity.direction == TG3::ABDirection::UpAB && out.complete.abs.contains(S::ABKey(a)))
        {
            const auto &g = out.complete.abGeometry.at(S::ABKey(a));
            if (g.near || g.conjunction)
                discovered.insert(native);
        }
    std::set<std::string> expected(qualifying.begin(), qualifying.end());
    Require(expected.size() == qualifying.size() && expected == discovered,
            "intervention_complete_qualifier_parity");
    out.qualifyingIdentities.assign(expected.begin(), expected.end());
    auto same = [](const auto &a, const auto &b)
    {
        return a.fibNear == b.fibNear && a.lineNear == b.lineNear && a.intersection == b.intersection &&
               a.cohort == b.cohort;
    };
    for (std::size_t mask = 0; mask < (std::size_t{1} << expected.size()); ++mask)
    {
        std::vector<AB> selected(actual.begin(), actual.end());
        for (std::size_t i = 0; i < out.qualifyingIdentities.size(); ++i)
            if (mask & (std::size_t{1} << i))
                selected.push_back(live.at(out.qualifyingIdentities[i]));
        std::sort(selected.begin(), selected.end(),
                  [](const auto &a, const auto &b) { return Order(a) < Order(b); });
        auto g = evaluate(selected);
        const bool sufficient = same(g, out.complete);
        out.subsets.push_back({mask, std::move(g), sufficient, false});
    }
    Require(same(out.subsets.front().geometry, out.actual), "intervention_baseline_parity");
    Require(same(out.subsets.back().geometry, out.complete), "intervention_joint_complete_parity");
    for (auto &subset : out.subsets)
    {
        subset.minimal = subset.sufficient;
        for (const auto &proper : out.subsets)
            if (proper.mask != subset.mask && (proper.mask & subset.mask) == proper.mask && proper.sufficient)
                subset.minimal = false;
    }
    return out;
}
} // namespace EA::FibonacciResearch::ABCapacity
