#pragma once

#include "CausalFibonacciTrendlineConfluenceDiagnostic.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <functional>
#include <locale>
#include <map>
#include <set>

// Standalone research only. Geometry and Wilder ATR belong to authoritative
// TG3. Primary opportunity IDs contain the clock and protocol, never a pair ID.
namespace EA::FibonacciResearch::TrendlineHistorical
{
namespace Diagnostic = TG3::TrendlineDiagnostic;
inline constexpr std::int64_t kOrigin = 1262304000;
inline constexpr std::int64_t kCalibrationStart = 1577836800;
inline constexpr std::int64_t kFinalStart = 1735689600;
inline constexpr std::int64_t kFinalEnd = 1767225600;
inline constexpr std::size_t kHorizon = 4;
inline constexpr std::string_view kSchema = "fib-trendline-decision-study-v2";
inline constexpr std::array<double, 3> kRatios{0.382, 0.500, 0.618};

class Sha256
{
public:
    Sha256() { if (!CC_SHA256_Init(&context_)) throw std::runtime_error("SHA256 init failed"); }
    void Add(std::string_view value)
    {
        while (!value.empty())
        {
            const auto count = std::min<std::size_t>(value.size(), std::numeric_limits<CC_LONG>::max());
            if (!CC_SHA256_Update(&context_, value.data(), static_cast<CC_LONG>(count)))
                throw std::runtime_error("SHA256 update failed");
            value.remove_prefix(count);
        }
    }
    std::string Digest() const
    {
        auto copy = context_;
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
        if (!CC_SHA256_Final(bytes.data(), &copy)) throw std::runtime_error("SHA256 final failed");
        std::ostringstream out; out.imbue(std::locale::classic());
        out << std::hex << std::setfill('0');
        for (unsigned char byte : bytes) out << std::setw(2) << static_cast<unsigned>(byte);
        return out.str();
    }
private:
    CC_SHA256_CTX context_{};
};
inline std::string Hash(std::string_view value) { Sha256 hash; hash.Add(value); return hash.Digest(); }
inline std::string Number(double value)
{
    std::ostringstream out; out.imbue(std::locale::classic());
    out << std::setprecision(17) << (value == 0 ? 0.0 : value);
    return out.str();
}
inline bool IsHash(std::string_view value)
{ return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == value.npos; }

enum class Partition { Development, Calibration, Final2025 };
enum class Cohort { A, B, C, D, FStratum };
inline const char* Name(Cohort c)
{ constexpr std::array names{"A", "B", "C", "D", "F_stratum"}; return names.at(static_cast<std::size_t>(c)); }
inline const char* Name(Partition p)
{ constexpr std::array names{"development_2010_2019", "calibration_2020_2024", "final_2025"}; return names.at(static_cast<std::size_t>(p)); }
inline std::pair<std::int64_t, std::int64_t> Bounds(Partition p)
{
    if (p == Partition::Development) return {kOrigin, kCalibrationStart};
    if (p == Partition::Calibration) return {kCalibrationStart, kFinalStart};
    if (p == Partition::Final2025) return {kFinalStart, kFinalEnd};
    throw std::invalid_argument("unknown partition");
}
struct ScoreRange { std::int64_t start, end; };
struct Protocol
{
    Partition partition = Partition::Development;
    std::size_t warmupBars = 2048;
    std::size_t lineBatchSize = 128;
    std::size_t maxPairsPerCandle = 512 * 4096;
    std::size_t maxDetectorIdentities = 65536;
    std::size_t maxEpisodeStates = 65536;
    std::size_t maxABAgeBars = 2048, maxLineAgeBars = 512;
    std::size_t maxABInputs = 512, maxLineInputs = 4096;
};
enum class Session { Open, Closed, Unverified };
struct SessionCalendar
{
    std::string identity = "UNVERIFIED", sha256;
    std::function<Session(std::int64_t)> slot;
    Session At(std::int64_t timestamp) const { return slot ? slot(timestamp) : Session::Unverified; }
};
inline std::string CanonicalProtocol(const Protocol& p, ScoreRange r, std::string_view detector, const SessionCalendar& calendar)
{
    return std::string(kSchema) + "|AUDCAD|15m|UTC_bar_start_minute_45|horizon=4|UpAB_UTL|"
        "ratios=0.382,0.500,0.618|market=0.25|structure=0.25|origin=1262304000|"
        "complete_commutative_universe|missing_not_D|strict_900_second_gap_censor|"
        "continuous_prefix|calibration_embargo=4_completed_candles|" + Name(p.partition) + "|"
        + std::to_string(r.start) + "|" + std::to_string(r.end) + "|warmup=" + std::to_string(p.warmupBars)
        + "|max_pairs=" + std::to_string(p.maxPairsPerCandle) + "|max_identities=" + std::to_string(p.maxDetectorIdentities)
        + "|max_episodes=" + std::to_string(p.maxEpisodeStates) + "|ab_age=" + std::to_string(p.maxABAgeBars)
        + "|line_age=" + std::to_string(p.maxLineAgeBars) + "|max_ab_inputs=" + std::to_string(p.maxABInputs)
        + "|max_line_inputs=" + std::to_string(p.maxLineInputs) + "|calendar=" + calendar.identity + ":" + calendar.sha256
        + "|detector=" + std::string(detector);
    // Batch size is an execution parameter; it cannot change semantic identity.
}
inline Cohort Classify(bool fib, bool line, bool intersection)
{
    if (intersection && (!fib || !line)) throw std::logic_error("inconsistent aggregate intersection");
    if (intersection) return Cohort::C;
    if (fib && line) return Cohort::FStratum;
    if (fib) return Cohort::A;
    if (line) return Cohort::B;
    return Cohort::D;
}
inline std::string ABKey(const TG3::ABStructure& ab)
{
    const auto& id = ab.identity;
    return "UpAB:" + std::to_string(id.aTimestamp) + ":" + std::to_string(id.bTimestamp) + ":"
        + std::to_string(id.availabilityTimestamp);
}
inline std::string LineKey(const TG1A::TrendLineCandidate& line)
{
    return "UTL:" + std::to_string(line.anchor1Timestamp) + ":" + std::to_string(line.anchor2Timestamp) + ":"
        + std::to_string(line.creationTimestamp);
}
inline std::string Snapshot(const TG3::ABStructure& ab)
{ return Number(ab.aPrice) + ":" + Number(ab.bPrice) + ":" + std::to_string(ab.aConfirmationTimestamp) + ":" + std::to_string(ab.bConfirmationTimestamp); }
inline std::string Snapshot(const TG1A::TrendLineCandidate& line)
{ return Number(line.anchor1Price) + ":" + Number(line.anchor2Price) + ":" + Number(line.rawPriceSlopePerBar) + ":" + std::to_string(line.anchorSeparationBars) + ":" + std::to_string(line.anchor1ConfirmationTimestamp) + ":" + std::to_string(line.anchor2ConfirmationTimestamp); }

struct GeometryMeasurement
{
    std::int64_t timestamp = 0;
    std::string ab, line;
    double ratio = 0, fibPrice = 0, linePrice = 0, dF = 0, dL = 0, dS = 0;
    Cohort cohort = Cohort::D;
};
struct Episode
{
    GeometryMeasurement measurement;
    std::size_t sequence = 0;
};
struct Aggregate
{
    bool fibNear = false, lineNear = false, intersection = false;
    bool complete = true, evaluated = false;
    std::size_t abs = 0, lines = 0, pairs = 0, measurements = 0, intersections = 0;
    std::size_t nearFibLevels = 0, nearLines = 0;
    std::optional<double> minDF, minDL, minDS;
    std::array<bool, 3> supportingRatios{};
};
struct Covariates
{
    std::optional<double> logAtrClose, return4, return16, return64, logAtrTR64;
    double meanABAge = 0, meanLineAge = 0, meanABRangeAtr = 0, meanLineSlopeAtr = 0;
    double newABFraction = 0, newLineFraction = 0;
    std::size_t barsSinceGap = 0;
    int utcHour = 0, utcWeekday = 0;
};
struct Decision
{
    std::string identity, status, reason;
    std::int64_t timestamp = 0;
    std::optional<Cohort> cohort;
    std::optional<double> close, atr;
    Aggregate aggregate;
    Covariates covariates;
};
struct Outcome
{
    double signedAtrChange = 0, signedLogReturn = 0, favorableAtr = 0, adverseAtr = 0;
    bool directionalHit = false, flat = false;
    bool operator==(const Outcome&) const = default;
};
struct OutcomeRecord
{
    std::string decision, status, reason;
    std::optional<std::int64_t> resolutionTimestamp;
    std::optional<Outcome> outcome;
};
struct Gap { std::int64_t previous, next, seconds; };
struct Audit
{
    std::size_t candles = 0, prefixCandles = 0, scoredRangeCandles = 0, scheduled = 0;
    std::size_t classified = 0, resolved = 0, censored = 0, peakIdentities = 0, peakEpisodes = 0;
    std::optional<std::int64_t> firstTimestamp, lastTimestamp;
    std::map<std::string, std::size_t> decisions, censoring, diagnostics, sourceYears;
    bool geometryComplete = true, diagnosticsComplete = true, calendarConsistent = true;
};
struct Sinks
{
    std::function<void(const Decision&)> decision;
    std::function<void(const OutcomeRecord&)> outcome;
    std::function<void(const GeometryMeasurement&)> geometry;
    std::function<void(const Episode&)> episode;
    std::function<void(const Gap&)> gap;
};

class EventEngine
{
public:
    EventEngine(Protocol protocol, std::string detectorFingerprint, Sinks sinks,
                std::optional<ScoreRange> range = {}, SessionCalendar calendar = {})
        : protocol_(protocol), calendar_(std::move(calendar)), sinks_(std::move(sinks))
    {
        if (protocol.partition == Partition::Final2025) throw std::invalid_argument("2025 evaluation disabled");
        const auto [start, end] = Bounds(protocol.partition);
        range_ = range.value_or(ScoreRange{start, end});
        if (range_.start < start || range_.end > end || range_.start >= range_.end ||
            protocol.warmupBars < 2048 || !protocol.lineBatchSize || !protocol.maxPairsPerCandle ||
            !protocol.maxDetectorIdentities || !protocol.maxEpisodeStates || !protocol.maxABAgeBars ||
            !protocol.maxLineAgeBars || !protocol.maxABInputs || !protocol.maxLineInputs ||
            !sinks_.decision || !sinks_.outcome ||
            (calendar_.slot && (calendar_.identity == "UNVERIFIED" || !IsHash(calendar_.sha256))))
            throw std::invalid_argument("invalid v2 protocol/range/calendar/sinks");
        canonical_ = CanonicalProtocol(protocol_, range_, detectorFingerprint, calendar_);
        fingerprint_ = Hash(canonical_);
        nextSlot_ = FirstSlot(range_.start);
    }
    void ReportSourceCapacity(std::size_t abEvictions, std::size_t lineEvictions)
    {
        if (!abEvictions && !lineEvictions) return;
        sourceCapacityStopped_ = true;
        audit_.geometryComplete = false; audit_.diagnosticsComplete = false;
        audit_.diagnostics["source_ab_capacity_evictions"] += abEvictions;
        audit_.diagnostics["source_line_capacity_evictions"] += lineEvictions;
    }
    void ValidateNext(const TG1A::Candle& c) const
    {
        if (finalized_) throw std::logic_error("finalized v2 replay");
        if (c.timestamp < kOrigin || c.timestamp >= kFinalStart || c.timestamp >= range_.end)
            throw std::invalid_argument("candle outside authorized pre2025 replay, rejected before detector update");
        if (c.timestamp % 900 != 0 || (previous_ && c.timestamp <= previous_->timestamp) ||
            !std::isfinite(c.open) || !std::isfinite(c.high) || !std::isfinite(c.low) || !std::isfinite(c.close) ||
            c.low <= 0 || c.high < c.low || c.open < c.low || c.open > c.high || c.close < c.low || c.close > c.high)
            throw std::invalid_argument("invalid chronological canonical candle");
    }
    void Add(std::size_t bar, const TG1A::Candle& c, std::optional<double> atr,
             std::span<const TG3::ABStructure> abs, std::span<const TG1A::TrendLineCandidate> lines)
    {
        ValidateNext(c);
        if (bar != audit_.candles) throw std::invalid_argument("replay must update exactly once");
        const bool gap = previous_ && c.timestamp - previous_->timestamp != 900;
        if (gap)
        {
            ++audit_.diagnostics["timestamp_gaps"];
            if (sinks_.gap) sinks_.gap({previous_->timestamp, c.timestamp, c.timestamp - previous_->timestamp});
            barsSinceGap_ = 0;
        }
        else ++barsSinceGap_;
        Advance(c, gap); // Outcomes cannot influence any decision calculations.
        FillAbsent(c.timestamp, false);
        if (c.timestamp >= kCalibrationStart) ++calibrationCandles_;
        UpdateHistory(c);
        ++audit_.candles;
        if (c.timestamp < range_.start) ++audit_.prefixCandles; else ++audit_.scoredRangeCandles;
        if (!audit_.firstTimestamp) audit_.firstTimestamp = c.timestamp;
        audit_.lastTimestamp = c.timestamp;
        const std::time_t raw = c.timestamp; std::tm utc{};
        if (!gmtime_r(&raw, &utc)) throw std::invalid_argument("invalid UTC timestamp");
        ++audit_.sourceYears[std::to_string(utc.tm_year + 1900)];
        candleHash_.Add(std::to_string(c.timestamp) + "," + Number(c.open) + "," + Number(c.high) + ","
                        + Number(c.low) + "," + Number(c.close) + "\n");
        Prune(bar);
        bool complete = abs.size() <= protocol_.maxABInputs && lines.size() <= protocol_.maxLineInputs;
        if (!complete) ++audit_.diagnostics["input_capacity"];
        auto causalAB = complete ? Canonical(abs, bar, c.timestamp) : std::vector<TG3::ABStructure>{};
        auto causalLines = complete ? Canonical(lines, bar, c.timestamp) : std::vector<TG1A::TrendLineCandidate>{};
        complete = complete && !identityStopped_ && !sourceCapacityStopped_;
        const bool validAtr = atr && std::isfinite(*atr) && *atr > 0;
        const bool scheduled = c.timestamp == nextSlot_ && c.timestamp >= range_.start;
        Aggregate a;
        if (complete && validAtr) a = Measure(bar, c, *atr, causalAB, causalLines, scheduled);
        else { a.abs = causalAB.size(); a.lines = causalLines.size(); a.complete = complete; }
        if (!a.complete) audit_.geometryComplete = false;
        if (scheduled)
        {
            Decision d;
            d.timestamp = c.timestamp; d.identity = fingerprint_ + ":" + std::to_string(c.timestamp);
            d.close = c.close; if (validAtr) d.atr = *atr;
            d.aggregate = a;
            d.covariates = Features(bar, c, atr, causalAB, causalLines);
            const Session session = calendar_.At(c.timestamp);
            if (session == Session::Closed)
            { d.status = "calendar_conflict"; audit_.calendarConsistent = false; }
            else if (!a.complete) d.status = "capacity_invalid";
            else if (bar < protocol_.warmupBars) d.status = "warmup";
            else if (protocol_.partition == Partition::Calibration && calibrationCandles_ <= 4) d.status = "embargo";
            else if (!validAtr) d.status = "invalid_atr";
            else if (causalAB.empty() || causalLines.empty()) d.status = "no_geometry";
            else if (!d.covariates.logAtrClose || !d.covariates.logAtrTR64 || !d.covariates.return4 ||
                !d.covariates.return16 || !d.covariates.return64 || !std::isfinite(d.covariates.meanABRangeAtr) ||
                !std::isfinite(d.covariates.meanLineSlopeAtr)) d.status = "invalid_covariates";
            else if (c.timestamp + 4 * 900 >= range_.end) d.status = "partition_purged";
            else if (ClosedHorizon(c.timestamp)) d.status = "closed_horizon";
            else
            {
                d.status = "classified"; d.cohort = Classify(a.fibNear, a.lineNear, a.intersection);
                if (pending_) throw std::logic_error("overlapping fixed-clock horizons");
                pending_ = Pending{d.identity, c.timestamp, c.close, *atr, 0, c.close, c.close};
                ++audit_.classified;
            }
            d.reason = d.status == "classified" ? "" : d.status;
            Emit(d); nextSlot_ += 3600;
        }
        previous_ = c;
    }
    void Finalize()
    {
        if (finalized_) return;
        if (pending_)
            Censor("source_coverage_shortfall", std::nullopt);
        FillAbsent(range_.end, true);
        std::size_t sum = 0; for (const auto& [key, count] : audit_.decisions) { (void)key; sum += count; }
        if (sum != audit_.scheduled || audit_.classified != audit_.resolved + audit_.censored)
            throw std::logic_error("v2 count reconciliation failed");
        finalized_ = true;
    }
    const Audit& GetAudit() const { return audit_; }
    const Protocol& GetProtocol() const { return protocol_; }
    ScoreRange Range() const { return range_; }
    const SessionCalendar& Calendar() const { return calendar_; }
    const std::string& Fingerprint() const { return fingerprint_; }
    const std::string& CanonicalConfiguration() const { return canonical_; }
    std::string CandleDigest() const { return candleHash_.Digest(); }
    bool IsFinalized() const { return finalized_; }
private:
    struct Stored { std::size_t birth; std::string snapshot; };
    struct EpisodeState { bool near; std::size_t sequence; std::size_t abBirth, lineBirth; };
    struct History { double close, trueRange; };
    struct Pending { std::string id; std::int64_t timestamp; double close, atr; std::size_t bars; double high, low; };
    Protocol protocol_;
    ScoreRange range_{};
    SessionCalendar calendar_;
    Sinks sinks_;
    std::string fingerprint_, canonical_;
    Audit audit_;
    Sha256 candleHash_;
    std::int64_t nextSlot_ = 0;
    std::size_t calibrationCandles_ = 0, barsSinceGap_ = 0;
    std::optional<TG1A::Candle> previous_;
    std::deque<History> history_;
    std::optional<Pending> pending_;
    std::map<std::string, Stored> abSnapshots_, lineSnapshots_;
    std::map<std::string, EpisodeState> episodes_;
    bool finalized_ = false, identityStopped_ = false, episodesStopped_ = false, sourceCapacityStopped_ = false;
    bool ClosedHorizon(std::int64_t timestamp) const
    {
        for (std::size_t i = 1; i <= kHorizon; ++i)
            if (calendar_.At(timestamp + static_cast<std::int64_t>(i) * 900) == Session::Closed) return true;
        return false;
    }
    static std::int64_t FirstSlot(std::int64_t start)
    { const auto slot = start / 3600 * 3600 + 2700; return slot >= start ? slot : slot + 3600; }
    void Emit(const Decision& d)
    { ++audit_.scheduled; ++audit_.decisions[d.status]; sinks_.decision(d); }
    void FillAbsent(std::int64_t before, bool trailing)
    {
        while (nextSlot_ < before && nextSlot_ < range_.end)
        {
            Decision d; d.timestamp = nextSlot_; d.identity = fingerprint_ + ":" + std::to_string(nextSlot_);
            const Session session = calendar_.At(nextSlot_);
            d.status = session == Session::Closed ? "closed_market" :
                session == Session::Open ? "missing_open_session" : "missing_slot_unverified";
            d.reason = trailing && session != Session::Closed ? "source_coverage_shortfall" : d.status;
            Emit(d); nextSlot_ += 3600;
        }
    }
    void UpdateHistory(const TG1A::Candle& c)
    {
        double tr = c.high - c.low;
        if (previous_) tr = std::max({tr, std::abs(c.high - previous_->close), std::abs(c.low - previous_->close)});
        history_.push_back({c.close, tr}); if (history_.size() > 65) history_.pop_front();
    }
    void Prune(std::size_t bar)
    {
        const auto oldAB = [&](std::size_t birth) { return bar > birth && bar - birth > protocol_.maxABAgeBars; };
        const auto oldLine = [&](std::size_t birth) { return bar > birth && bar - birth > protocol_.maxLineAgeBars; };
        std::erase_if(abSnapshots_, [&](const auto& item) { return oldAB(item.second.birth); });
        std::erase_if(lineSnapshots_, [&](const auto& item) { return oldLine(item.second.birth); });
        std::erase_if(episodes_, [&](const auto& item) { return oldAB(item.second.abBirth) || oldLine(item.second.lineBirth); });
    }
    void Remember(std::map<std::string, Stored>& map, const std::string& key, std::size_t birth, const std::string& snapshot)
    {
        const auto found = map.find(key);
        if (found != map.end())
        {
            if (found->second.snapshot != snapshot || found->second.birth != birth)
                throw std::invalid_argument("conflicting natural snapshots across replay");
            return;
        }
        if (abSnapshots_.size() + lineSnapshots_.size() >= protocol_.maxDetectorIdentities)
        {
            if (!identityStopped_) ++audit_.diagnostics["identity_capacity_stop"];
            identityStopped_ = true; ++audit_.diagnostics["identity_capacity_skips"]; return;
        }
        map.emplace(key, Stored{birth, snapshot});
        audit_.peakIdentities = std::max(audit_.peakIdentities, abSnapshots_.size() + lineSnapshots_.size());
    }
    std::vector<TG3::ABStructure> Canonical(std::span<const TG3::ABStructure> input, std::size_t bar, std::int64_t time)
    {
        std::map<std::string, TG3::ABStructure> unique;
        for (const auto& ab : input)
        {
            const auto& id = ab.identity;
            if (!Diagnostic::Detail::Eligible(ab, bar) || id.direction != TG3::ABDirection::UpAB ||
                bar - id.availabilityBar > protocol_.maxABAgeBars || id.aTimestamp >= id.bTimestamp ||
                id.aTimestamp > ab.aConfirmationTimestamp || id.bTimestamp > ab.bConfirmationTimestamp ||
                ab.aConfirmationTimestamp > id.availabilityTimestamp || ab.bConfirmationTimestamp > id.availabilityTimestamp ||
                id.availabilityTimestamp > time)
            { ++audit_.diagnostics["rejected_ab_snapshots"]; continue; }
            const auto [it, inserted] = unique.emplace(ABKey(ab), ab);
            if (!inserted)
            {
                if (Snapshot(it->second) != Snapshot(ab) || it->second.identity.availabilityBar != id.availabilityBar)
                    throw std::invalid_argument("conflicting natural AB snapshots");
                ++audit_.diagnostics["duplicate_ab_snapshots"];
            }
        }
        std::vector<TG3::ABStructure> out;
        for (const auto& [key, ab] : unique)
        { Remember(abSnapshots_, key, ab.identity.availabilityBar, Snapshot(ab)); out.push_back(ab); }
        return out;
    }
    std::vector<TG1A::TrendLineCandidate> Canonical(std::span<const TG1A::TrendLineCandidate> input, std::size_t bar, std::int64_t time)
    {
        std::map<std::string, TG1A::TrendLineCandidate> unique;
        for (const auto& line : input)
        {
            if (!Diagnostic::Detail::Eligible(line, bar) || line.direction != TG1A::TrendLineDirection::UTL ||
                bar - line.creationBar > protocol_.maxLineAgeBars || line.anchor1Timestamp >= line.anchor2Timestamp ||
                line.anchor1Timestamp > line.anchor1ConfirmationTimestamp || line.anchor2Timestamp > line.anchor2ConfirmationTimestamp ||
                line.anchor1ConfirmationTimestamp > line.creationTimestamp || line.anchor2ConfirmationTimestamp > line.creationTimestamp ||
                line.creationTimestamp > time || line.currentTimestamp != time)
            { ++audit_.diagnostics["rejected_line_snapshots"]; continue; }
            const auto [it, inserted] = unique.emplace(LineKey(line), line);
            if (!inserted)
            {
                if (Snapshot(it->second) != Snapshot(line) || it->second.creationBar != line.creationBar ||
                    it->second.ProjectedPrice(bar) != line.ProjectedPrice(bar))
                    throw std::invalid_argument("conflicting natural line snapshots");
                ++audit_.diagnostics["duplicate_line_snapshots"];
            }
        }
        std::vector<TG1A::TrendLineCandidate> out;
        for (const auto& [key, line] : unique)
        { Remember(lineSnapshots_, key, line.creationBar, Snapshot(line)); out.push_back(line); }
        return out;
    }
    Aggregate Measure(std::size_t bar, const TG1A::Candle& c, double atr,
                      const std::vector<TG3::ABStructure>& abs, const std::vector<TG1A::TrendLineCandidate>& lines, bool scheduled)
    {
        Aggregate a; a.evaluated = true; a.abs = abs.size(); a.lines = lines.size();
        if (!lines.empty() && abs.size() > protocol_.maxPairsPerCandle / lines.size())
        { a.complete = false; ++audit_.diagnostics["pair_capacity_candles"]; return a; }
        std::set<std::string> fibNear, lineNear;
        const auto minimum = [](std::optional<double>& target, double value) { if (!target || value < *target) target = value; };
        for (const auto& ab : abs)
            for (std::size_t first = 0; first < lines.size(); first += protocol_.lineBatchSize)
            {
                const auto count = std::min(protocol_.lineBatchSize, lines.size() - first);
                using CandidateKey = decltype(Diagnostic::Detail::Key(TG2::CandidateIdentity{}));
                std::map<CandidateKey, const TG1A::TrendLineCandidate*> candidates;
                std::map<std::string, std::size_t> lineBirths;
                for (std::size_t i = first; i < first + count; ++i)
                {
                    candidates.emplace(Diagnostic::Detail::Key(Diagnostic::Detail::LineIdentity(lines[i])), &lines[i]);
                    lineBirths.emplace(LineKey(lines[i]), lines[i].creationBar);
                }
                Diagnostic::Configuration config(.25);
                config.retracementRatios.assign(kRatios.begin(), kRatios.end());
                config.maxABStructures = 1; config.maxTrendlines = count;
                config.maxPairEvaluations = count; config.maxMeasurements = count * 3;
                const auto batch = Diagnostic::Evaluate(bar, c.close, atr, std::span{&ab, 1},
                    std::span{lines.data() + first, count}, config);
                if (batch.pairCapacityReached || batch.measurementCapacityReached || batch.measurements.size() != count * 3)
                    throw std::logic_error("incomplete bounded diagnostic batch");
                std::vector<GeometryMeasurement> measurements;
                for (const auto& m : batch.measurements)
                {
                    const auto* candidate = candidates.at(Diagnostic::Detail::Key(m.identity.line));
                    const auto f = Diagnostic::Detail::NormalizedDistance(c.close, m.fibPrice, atr);
                    const auto l = Diagnostic::Detail::NormalizedDistance(c.close, m.linePrice, atr);
                    if (!f || !l || !m.normalizedStructureDistance)
                        throw std::invalid_argument("nonrepresentable geometry distance; cannot classify complete universe");
                    measurements.push_back({c.timestamp, ABKey(ab), LineKey(*candidate), m.identity.ratio,
                        m.fibPrice, m.linePrice, *f, *l, *m.normalizedStructureDistance,
                        Classify(*f <= .25, *l <= .25, *f <= .25 && *l <= .25 && *m.normalizedStructureDistance <= .25)});
                }
                std::sort(measurements.begin(), measurements.end(), [](const auto& x, const auto& y)
                    { return std::tie(x.line, x.ratio) < std::tie(y.line, y.ratio); });
                a.pairs += count;
                for (const auto& m : measurements)
                {
                    ++a.measurements;
                    minimum(a.minDF, m.dF); minimum(a.minDL, m.dL); minimum(a.minDS, m.dS);
                    if (m.dF <= .25) { a.fibNear = true; fibNear.insert(m.ab + ":" + Number(m.ratio)); }
                    if (m.dL <= .25) { a.lineNear = true; lineNear.insert(m.line); }
                    if (m.cohort == Cohort::C)
                    {
                        a.intersection = true; ++a.intersections;
                        for (std::size_t i = 0; i < kRatios.size(); ++i) if (m.ratio == kRatios[i]) a.supportingRatios[i] = true;
                    }
                    if (scheduled && sinks_.geometry) sinks_.geometry(m);
                    ObserveEpisode(m, ab.identity.availabilityBar, lineBirths.at(m.line));
                }
            }
        a.nearFibLevels = fibNear.size(); a.nearLines = lineNear.size();
        audit_.diagnostics["measurements_examined"] += a.measurements;
        return a;
    }
    void ObserveEpisode(const GeometryMeasurement& m, std::size_t abBirth, std::size_t lineBirth)
    {
        if (episodesStopped_) return;
        const std::string key = m.ab + "|" + m.line + "|" + Number(m.ratio);
        auto it = episodes_.find(key);
        const bool near = m.cohort != Cohort::D;
        if (it == episodes_.end())
        {
            if (episodes_.size() == protocol_.maxEpisodeStates)
            {
                episodesStopped_ = true; audit_.diagnosticsComplete = false;
                ++audit_.diagnostics["episode_capacity_stop"]; return;
            }
            it = episodes_.emplace(key, EpisodeState{!near, 0, abBirth, lineBirth}).first;
            audit_.peakEpisodes = std::max(audit_.peakEpisodes, episodes_.size());
        }
        if (it->second.near == near) return;
        it->second.near = near; ++it->second.sequence;
        ++audit_.diagnostics["interaction_episodes"];
        if (sinks_.episode) sinks_.episode({m, it->second.sequence});
    }
    Covariates Features(std::size_t bar, const TG1A::Candle& c, std::optional<double> atr,
                        const std::vector<TG3::ABStructure>& abs, const std::vector<TG1A::TrendLineCandidate>& lines) const
    {
        Covariates x; x.barsSinceGap = barsSinceGap_;
        const std::time_t time = c.timestamp + 900; std::tm utc{}; gmtime_r(&time, &utc);
        x.utcHour = utc.tm_hour; x.utcWeekday = utc.tm_wday;
        if (!atr || !std::isfinite(*atr) || *atr <= 0) return x;
        const auto finite = [](double v) -> std::optional<double> { return std::isfinite(v) ? std::optional{v} : std::nullopt; };
        x.logAtrClose = finite(std::log(*atr / c.close));
        const auto change = [&](std::size_t n) -> std::optional<double>
            { return history_.size() > n ? finite((c.close - history_[history_.size() - n - 1].close) / *atr) : std::nullopt; };
        x.return4 = change(4); x.return16 = change(16); x.return64 = change(64);
        if (history_.size() >= 64)
        {
            long double total = 0;
            for (std::size_t i = history_.size() - 64; i < history_.size(); ++i) total += history_[i].trueRange;
            if (total > 0) x.logAtrTR64 = finite(std::log(*atr / static_cast<double>(total / 64)));
        }
        long double age = 0, range = 0, slope = 0; std::size_t newly = 0;
        for (const auto& ab : abs)
        { age += bar - ab.identity.availabilityBar; range += (ab.bPrice - ab.aPrice) / *atr; newly += bar - ab.identity.availabilityBar < 4; }
        if (!abs.empty())
        { x.meanABAge = static_cast<double>(age / abs.size()); x.meanABRangeAtr = static_cast<double>(range / abs.size()); x.newABFraction = static_cast<double>(newly) / abs.size(); }
        age = 0; newly = 0;
        for (const auto& line : lines)
        { age += bar - line.creationBar; slope += line.rawPriceSlopePerBar / *atr; newly += bar - line.creationBar < 4; }
        if (!lines.empty())
        { x.meanLineAge = static_cast<double>(age / lines.size()); x.meanLineSlopeAtr = static_cast<double>(slope / lines.size()); x.newLineFraction = static_cast<double>(newly) / lines.size(); }
        return x;
    }
    void Censor(const std::string& reason, std::optional<std::int64_t> time)
    {
        ++audit_.censored; ++audit_.censoring[reason];
        sinks_.outcome({pending_->id, "censored", reason, time, std::nullopt}); pending_.reset();
    }
    void Advance(const TG1A::Candle& c, bool gap)
    {
        if (!pending_) return;
        if (c.timestamp >= Bounds(protocol_.partition).second) { Censor("partition_boundary", std::nullopt); return; }
        if (gap || c.timestamp != pending_->timestamp + static_cast<std::int64_t>(pending_->bars + 1) * 900)
        { Censor("unexpected_timestamp_gap", c.timestamp); return; }
        ++pending_->bars; pending_->high = std::max(pending_->high, c.high); pending_->low = std::min(pending_->low, c.low);
        if (pending_->bars != 4) return;
        Outcome o;
        o.signedAtrChange = (c.close - pending_->close) / pending_->atr;
        o.signedLogReturn = std::log(c.close) - std::log(pending_->close);
        o.favorableAtr = (pending_->high - pending_->close) / pending_->atr;
        o.adverseAtr = (pending_->close - pending_->low) / pending_->atr;
        o.directionalHit = c.close > pending_->close; o.flat = c.close == pending_->close;
        if (!std::isfinite(o.signedAtrChange) || !std::isfinite(o.signedLogReturn) || !std::isfinite(o.favorableAtr) || !std::isfinite(o.adverseAtr))
        { Censor("outcome_numeric_invalid", c.timestamp); return; }
        ++audit_.resolved; sinks_.outcome({pending_->id, "resolved", "", c.timestamp, o}); pending_.reset();
    }
};

class HistoricalEvaluator
{
public:
    HistoricalEvaluator(Protocol p, TG1B::CalibrationConfiguration calibration, TG3::Configuration fibonacci,
                        TG1A::Configuration geometry, TG2::Configuration behavior, std::string detectorFingerprint,
                        Sinks sinks, std::optional<ScoreRange> range = {}, SessionCalendar calendar = {})
        : engine_(p, std::move(detectorFingerprint), std::move(sinks), range, std::move(calendar)),
          integration_(calibration, ResearchFibonacci(std::move(fibonacci)), geometry, behavior, {"AUDCAD", "15m"})
    {
        if (geometry.atrPeriod != 14 || geometry.maxCandidateAgeBars != p.maxLineAgeBars ||
            integration_.Tracker().GetConfiguration().maxABAgeBars != p.maxABAgeBars)
            throw std::invalid_argument("research/detector ATR and age contracts differ");
    }
    void AddCompletedBar(const TG1A::Candle& c)
    {
        engine_.ValidateNext(c); // Including 2025 rejection, before any advancement.
        const auto& before = integration_.Behavior().Classification().Geometry();
        const auto nextBar = before.CompletedBarCount();
        const auto surviving = static_cast<std::size_t>(std::count_if(before.Candidates().begin(), before.Candidates().end(),
            [&](const auto& line) { return nextBar - line.creationBar <= before.GetConfiguration().maxCandidateAgeBars; }));
        const auto abEvictionsBefore = integration_.Tracker().ABCapacityEvictions();
        const auto update = integration_.AddCompletedBar(c);
        const auto& g = integration_.Behavior().Classification().Geometry();
        const auto potential = surviving + update.behaviorUpdate.classificationUpdate.geometryUpdate.newlyCreatedCandidates.size();
        if (potential < g.Candidates().size()) throw std::logic_error("source candidate capacity accounting mismatch");
        engine_.ReportSourceCapacity(integration_.Tracker().ABCapacityEvictions() - abEvictionsBefore,
                                     potential - g.Candidates().size());
        engine_.Add(update.bar, c, g.CurrentAtr(), integration_.ABStructures(), g.Candidates());
    }
    void Finalize() { if (!engine_.IsFinalized()) { engine_.Finalize(); integration_.Finalize(); } }
    const EventEngine& Engine() const { return engine_; }
    const TG3::CausalFibonacciConfluenceIntegration& Integration() const { return integration_; }
    static TG3::Configuration ResearchFibonacci(TG3::Configuration f)
    { f.retracementRatios.assign(kRatios.begin(), kRatios.end()); f.directionalStudyPolicy = TG3::DirectionalStudyPolicy::SourceUTLUpABOnly; return f; }
private:
    EventEngine engine_;
    TG3::CausalFibonacciConfluenceIntegration integration_;
};
} // namespace EA::FibonacciResearch::TrendlineHistorical
