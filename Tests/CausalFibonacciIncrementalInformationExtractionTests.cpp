#include "../Sources/CausalFibonacciIncrementalInformationExtraction.hpp"

#include <cassert>
#include <cstring>
#include <filesystem>

namespace F = EA::CausalFibonacciIncrementalInformation;
namespace X = EA::CausalFibonacciIncrementalInformation::Extraction;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }
std::ostream& operator<<(std::ostream& output, PriceTP) { return output; }

namespace {
PriceTP At(std::int64_t value) { return PriceTP{std::chrono::seconds{value}}; }

Feature Bar(std::int64_t timestamp, float close = 1.0F,
            float high = 1.0004F, float low = .9996F)
{
    return {close, close, high, low, At(timestamp), 100.0F};
}

Tensor FixtureTensor(std::int64_t start, std::size_t count)
{
    Tensor tensor{"audcadrmp"};
    for (std::size_t index = 0; index < count; ++index)
        tensor.Add(Bar(start + static_cast<std::int64_t>(index) * 900));
    return tensor;
}
}

int main()
{
    // Preserve the direct authoritative target path, including an H4
    // simultaneous high/low threshold hit (Up wins on the frozen <= tie).
    Tensor tensor{"audcadrmp"};
    for (std::size_t index = 0; index < 24; ++index) {
        const auto timestamp = F::kDevelopmentStart + static_cast<std::int64_t>(index) * 900;
        if (index == 11) tensor.Add(Bar(timestamp, 1.0F, 1.002F, .998F));
        else tensor.Add(Bar(timestamp));
    }
    constexpr std::size_t decision = 10;
    const auto directH4 = BuildLookaheadClassInfo(
        tensor, tensor.begin() + decision, 1, 4, X::kTargetThreshold);
    const auto h4 = X::AuthoritativeTargetAudit(
        tensor, decision, 4, F::Partition::Development);
    assert(h4.eligible && h4.assignedClass == directH4.assignedClass);
    assert(h4.assignedClass == 2);
    assert(h4.selectedTargetTimestamp == F::kDevelopmentStart + 11 * 900);
    assert(h4.terminalTimestamp == F::kDevelopmentStart + 14 * 900);

    // A separate no-hit path verifies terminal fallback rather than masking
    // the H4 simultaneous hit with the same future bar.
    const std::int64_t terminalStart = F::kDevelopmentStart + 86400;
    Tensor terminalTensor = FixtureTensor(terminalStart, 24);
    const auto directH6 = BuildLookaheadClassInfo(
        terminalTensor, terminalTensor.begin() + decision, 1, 6, X::kTargetThreshold);
    const auto h6 = X::AuthoritativeTargetAudit(
        terminalTensor, decision, 6, F::Partition::Development);
    assert(h6.eligible && h6.assignedClass == directH6.assignedClass);
    assert(h6.assignedClass == 1);
    assert(h6.selectedTargetTimestamp == terminalStart + 16 * 900);

    const auto physical = X::PhysicalModelInputAt(tensor, decision);
    const auto projected = F::ProjectAuthoritativeLayout9Row(
        "audcadrmp", F::kDevelopmentStart + decision * 900, decision, physical);
    std::array<float, EA::kTG4ProductionPulseModelInputWidth> layout8{};
    const auto raw = MetaNN::LowerAccess(*(tensor.begin() + decision));
    const auto layout8Contract = EA::ResolveModelInputContract(
        EA::kTG4ProductionPulseModelInputWidth, feature_size);
    EA::CopyTensorFeaturesForModelInput(layout8.data(), raw.RawMemory(), layout8Contract);
    EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        decision, layout8.data(), layout8Contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position) { return tensor.RawCloseAtIterator(tensor.begin() + position); });
    for (std::size_t index = 0; index < projected.baseline.size(); ++index)
        assert(projected.baseline[index] == layout8[index]);
    for (std::size_t index = 0; index < projected.fibonacci.size(); ++index)
        assert(projected.fibonacci[index] == raw.RawMemory()[76 + index]);
    assert(projected.baseline[76] == physical[99] && projected.baseline[79] == physical[102]);

    Tensor edge = FixtureTensor(F::kValidationStart - 2 * 900, 12);
    const auto crossing = X::AuthoritativeTargetAudit(edge, 0, 4, F::Partition::Development);
    assert(!crossing.eligible && crossing.exclusionReason == "target_horizon_crosses_partition_boundary");
    assert(crossing.terminalTimestamp >= F::kValidationStart);

    Tensor gapped{"audcadrmp"};
    gapped.Add(Bar(F::kPre2025Start));
    gapped.Add(Bar(F::kPre2025Start + 1800));
    X::AssertCanonicalTensorIdentity(gapped); // Source gaps are not synthesized.
    bool duplicateRejected = false;
    try { gapped.Add(Bar(F::kPre2025Start + 1800)); }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    assert(duplicateRejected);

    const auto missing = X::CensoredTarget("fixture");
    assert(!missing.eligible && missing.exclusionReason == "fixture");
    X::Options options;
    options.codeCommit = "fixture-commit";
    options.calendarSnapshot = {7, "fixture-calendar-content-hash"};
    const F::ArtifactProvenance provenance = X::Provenance(options);
    assert(provenance.economicCalendarSnapshotId == "7");
    assert(provenance.economicCalendarSnapshotSha256 == "fixture-calendar-content-hash");
    assert(provenance.warmupIdentity == "full_history_warmup");
    assert(X::VerifiedFrozenUniverse() == std::vector<std::string>({
        "audcadrmp", "audusdrmp", "eurusdrmp", "gbpusdrmp", "usdcadrmp", "usdjpyrmp"}));
    std::cout << "CausalFibonacciIncrementalInformationExtractionTests passed\n";
}
