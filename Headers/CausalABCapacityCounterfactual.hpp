#pragma once
// Research-only: consumes immutable post-primary observations. No outcomes,
// mutable source state, restored fractal histories, or production integration.
#include "CausalFractalHistoryShadowAudit.hpp"

namespace EA::FibonacciResearch::ABCapacity
{
inline constexpr std::string_view kSchema = "causal-ab-capacity-counterfactual-v1";
struct Budgets
{
    std::size_t liveABRecords = 8192;
    std::size_t memoryBytes = 64 * 1024 * 1024;
    std::size_t decodedOutputBytes = 256 * 1024 * 1024;
    // Combined ACTUAL and COUNTERFACTUAL work, not primary pair capacity.
    std::size_t pairsPerDecision = 16 * 1024 * 1024;
    std::size_t totalPairs = 1024 * 1024 * 1024;
};
struct Summary
{
    bool actualComplete = true, counterfactualComplete = true, parity = true;
    std::string failure;
    std::size_t bars = 0, parityChecks = 0, scheduled = 0, reconstructed = 0;
    std::size_t unknown = 0, missing = 0, unpublishedDecisions = 0, primaryParityChecks = 0;
    std::array<std::size_t, 3> evictions{}; // full, prefix, scoring
    std::optional<std::size_t> firstEvictionBar;
    std::optional<std::int64_t> firstEvictionTimestamp;
    std::size_t changedF = 0, changedL = 0, changedI = 0, changedCohort = 0;
    std::size_t victimAgeExpirations = 0;
    std::size_t peakLiveAB = 0, peakMemoryBytes = 0, outputBytes = 0;
    std::size_t peakPairs = 0, totalPairs = 0, decisionRows = 0, candidateRows = 0;
};
// Synthetic fault seams only; never exposed in the CLI.
enum class Step
{
    ActualParity,
    Counterfactual,
    Write,
    Finalize,
    Checksum,
    Marker
};
using Hook = std::function<void(Step)>;
class Audit
{
  public:
    Audit(TG3::Configuration, TG1A::Configuration, HistoryShadow::DecisionAuditConfiguration,
          std::filesystem::path newDirectory, Budgets = {}, Hook = {});
    ~Audit();
    Audit(const Audit &) = delete;
    Audit &operator=(const Audit &) = delete;
    void ObservePrimaryDecision(const HistoryShadow::PrimaryDecisionObservation &) noexcept;
    void Observe(const HistoryShadow::CompletedBarObservation &) noexcept;
    const Summary &GetSummary() const;
    // Parent COMPLETE or INVALID is accepted after integrity validation. This
    // marker certifies this diagnostic only; never promotes parent validity.
    bool Complete(const HistoryShadow::Provenance &) noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};
void ValidateSidecar(const std::filesystem::path &directory, const std::filesystem::path &parentPrimary);

// Independent intervention replay: one continuous native detector state, no
// primary evaluation or publication, and no fractal-history restoration.
struct CapacityVictim
{
    TG3::ABStructure ab;
    std::size_t evictionBar;
    std::int64_t evictionTimestamp;
};
class InterventionReplay
{
  public:
    InterventionReplay(TG3::Configuration, TG1A::Configuration);
    void AddCompletedBar(const TG1A::Candle &);
    const TG3::FibonacciConfluenceTracker &Actual() const { return actual_; }
    const TG1A::CausalFractalTrendLineGeometry &Geometry() const { return geometry_; }
    const std::vector<CapacityVictim> &Victims() const { return victims_; }
    std::size_t ParityChecks() const { return parityChecks_; }

  private:
    TG3::FibonacciConfluenceTracker actual_, mirror_;
    TG1A::CausalFractalTrendLineGeometry geometry_;
    std::vector<CapacityVictim> victims_;
    std::size_t parityChecks_ = 0;
};
struct SubsetResult
{
    std::size_t mask = 0;
    HistoryShadow::DecisionGeometry geometry;
    bool sufficient = false, minimal = false;
};
struct InterventionResult
{
    HistoryShadow::DecisionGeometry actual, complete;
    std::vector<std::string> qualifyingIdentities; // native lexical mask order
    std::vector<SubsetResult> subsets;
    std::size_t pairs = 0;
};
// Checked accounting is exposed for synthetic size_t boundary tests; it does
// not change geometry or detector capacities.
std::size_t RemainingInterventionPairs(std::size_t budget, std::size_t accumulated);
std::size_t AccumulateInterventionPairs(std::size_t budget, std::size_t accumulated, std::size_t eligibleAB,
                                        std::size_t eligibleLines);
// Reuses a single reconstructed decision snapshot. Evaluates every subset with
// the authoritative geometry helper, never by an approximate predicate rule.
// Requires the qualifying list to match ALL contributing live capacity victims.
InterventionResult EvaluateInterventions(std::size_t bar, const TG1A::Candle &, std::optional<double> atr,
                                         std::span<const TG3::ABStructure> actual,
                                         std::span<const TG1A::TrendLineCandidate> lines,
                                         std::span<const CapacityVictim> victims,
                                         std::span<const std::string> qualifying,
                                         std::size_t pairBudget = 64 * 1024 * 1024);
} // namespace EA::FibonacciResearch::ABCapacity
