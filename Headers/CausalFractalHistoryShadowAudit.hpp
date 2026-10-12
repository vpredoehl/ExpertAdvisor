#pragma once
// Research-only observer. This interface deliberately has no outcomes, engine,
// mutable detector references, or callback into primary state.
#include "CausalFibonacciConfluenceIntegration.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

namespace EA::FibonacciResearch::HistoryShadow
{
inline constexpr std::string_view kTG3SourceSHA256 =
    "5f2d52a810dd175321ece512d6f4693c92913ec98b1234daed40d3095aeec2fa";
inline constexpr std::string_view kTG1ASourceSHA256 =
    "c6eaa92bfec6c8dde0e67c54920e34d0fcf133c98c8fc5a500c2227214cdf507";
void ValidateFrozenSourceSnapshot(const std::filesystem::path &);
inline constexpr std::string_view kSchema = "causal-fractal-history-shadow-audit-v1";
struct CompletedBarObservation
{
    std::size_t bar;
    TG1A::Candle candle;
    std::optional<double> atr;
    // Eligibility is causal temporal/ATR eligibility, not primary acceptance.
    bool scoring, scheduled, decisionWindowEligible;
    std::span<const TG1A::ConfirmedFractal> confirmed;
    std::span<const TG3::ABIdentity> abBirths;
    std::span<const TG1A::TrendLineCandidate> lineBirths;
    std::span<const TG3::ABStructure> retainedAB;
    std::span<const TG1A::TrendLineCandidate> retainedLines;
    std::size_t totalABCreated, abCapacityEvictions, abAgeExpirations;
};
// Natural identities ignore bar-index aliases; native identities retain them.
struct CandidateIdentityReconciliation
{
    static std::string Natural(const TG3::ABIdentity &);
    static std::string Natural(const TG1A::TrendLineCandidate &);
    static std::string Native(const TG3::ABIdentity &);
};
struct CausalRules
{
    template <class History>
    static std::optional<TG1A::ConfirmedFractal> SelectTG3Anchor(const History &history,
                                                                 const TG1A::ConfirmedFractal &b)
    {
        std::optional<TG1A::ConfirmedFractal> best;
        for (const auto &entry : history)
        {
            const auto &a = [&]() -> const TG1A::ConfirmedFractal &
            {
                if constexpr (requires { entry.fractal; })
                    return entry.fractal;
                else
                    return entry;
            }();
            if (a.anchorBar >= b.anchorBar || a.confirmationBar > b.confirmationBar)
                continue;
            if (b.kind == TG1A::FractalKind::High ? b.price <= a.price : b.price >= a.price)
                continue;
            if (!best || std::tuple(a.anchorBar, a.confirmationBar, a.anchorTimestamp) >
                             std::tuple(best->anchorBar, best->confirmationBar, best->anchorTimestamp))
                best = a;
        }
        return best;
    }
    static bool TG1ATemporallyEligible(std::size_t bar, std::size_t anchor)
    {
        return bar >= anchor && bar - anchor <= 512;
    }
};
using Observer = std::function<void(const CompletedBarObservation &)>;
struct Budgets
{
    std::size_t shadowFractals = 200000; // TG3 has NO age-based retirement.
    std::size_t liveCandidates = 16384;  // Includes same-bar pre-enforcement birth demand.
    std::size_t witnessRecords = 2000000;
    std::size_t removalRecords = 2000000;
    std::size_t diagnosticBytes = 128 * 1024 * 1024;
    std::size_t publicationBufferBytes = 2 * 65536;
    std::size_t decodedPublicationBytes = 1024 * 1024 * 1024;
};
struct Counts
{
    std::size_t insertions = 0, duplicates = 0, capacityRemovals = 0, expirations = 0;
    std::size_t relevantRemovals = 0, peakOccupancy = 0, peakDemand = 0;
    std::optional<std::size_t> firstRemovalBar;
    std::optional<std::int64_t> firstRemovalTimestamp;
};
struct Summary
{
    // Detector TG3/TG1A, kind High/Low, phase full/prefix/scoring.
    std::array<std::array<std::array<Counts, 3>, 2>, 2> history{};
    std::size_t bars = 0, parityChecks = 0, creationLossesTG3 = 0, creationLossesTG1A = 0;
    std::size_t capacityDisplacements = 0, structuralRejections = 0;
    std::size_t survivingLosses = 0, decisionEligibleLosses = 0, contributingLosses = 0;
    std::size_t witnessRecords = 0, removalRecords = 0, peakAccountedBytes = 0;
    bool parity = true, complete = true, exhausted = false;
    std::string failure;
};
enum class PublicationStep
{
    Compress,
    Write,
    Finalize,
    Checksum,
    Marker
};
// Synthetic fault seam, never exposed as a CLI option.
using PublicationHook = std::function<void(PublicationStep)>;
struct Provenance
{
    std::string semanticFingerprint, gitCommit, sourceSnapshotDigest, candleDigest;
    std::filesystem::path parentPrimary;
};
// Actual frozen classes are independent bounded parity mirrors. The restored
// histories and original-cap candidate pools are separate research state.
class Audit
{
  public:
    Audit(TG3::Configuration fibonacci, TG1A::Configuration geometry,
          std::filesystem::path newSidecarDirectory, Budgets budgets = {}, PublicationHook hook = {});
    ~Audit();
    Audit(const Audit &) = delete;
    Audit &operator=(const Audit &) = delete;
    void Observe(const CompletedBarObservation &) noexcept;
    const Summary &GetSummary() const;
    // Finalizes streams, validates parent linkage and checksums. An incomplete
    // diagnostic publishes status/evidence but NEVER AUDIT_COMPLETE.
    bool Complete(const Provenance &);

  private:
    struct State;
    std::unique_ptr<State> state_;
};
void ValidateSidecar(const std::filesystem::path &directory);
} // namespace EA::FibonacciResearch::HistoryShadow
