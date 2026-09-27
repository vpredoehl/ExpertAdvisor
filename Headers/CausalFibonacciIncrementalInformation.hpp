#ifndef CausalFibonacciIncrementalInformation_hpp
#define CausalFibonacciIncrementalInformation_hpp

// Fixture-first, read-only support for the frozen layout-9 Fibonacci screen.
// This header deliberately has no database, scheduler, worker, or model code.

#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputFeatureSemantics.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace EA::CausalFibonacciIncrementalInformation {

inline constexpr std::string_view kProtocolId =
    "causal-fibonacci-layout9-incremental-information-v2";
inline constexpr std::string_view kProtocolSha256 =
    "9f39886dc46ede11e2322fd8c0afcef6b09505132bfdc243c85ae8b790352c53";
inline constexpr std::string_view kV1SourceArtifactProtocolId =
    "causal-fibonacci-layout9-incremental-information-v1";
inline constexpr std::string_view kV1SourceArtifactProtocolSha256 =
    "f0eae946cd935f1aeb6184d3611204909697f7c47b2c58c8d733121b9e706aaa";
inline constexpr std::size_t kBaselineWidth = kTG4ProductionPulseModelInputWidth;
inline constexpr std::size_t kFibonacciWidth = 23;
static_assert(kBaselineWidth == 80);
static_assert(causal_fibonacci_structural_feature_size == 99);

inline std::string FileSha256(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("fibonacci_protocol_document_unreadable:" + path.string());
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    std::array<char, 8192> bytes{};
    while (in.read(bytes.data(), static_cast<std::streamsize>(bytes.size())) || in.gcount() != 0)
        CC_SHA256_Update(&context, bytes.data(), static_cast<CC_LONG>(in.gcount()));
    if (!in.eof()) throw std::runtime_error("fibonacci_protocol_document_read_failed:" + path.string());
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream out;
    for (unsigned char value : digest)
        out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(value);
    return out.str();
}

inline void VerifyFrozenProtocolDocument(const std::filesystem::path& path)
{
    const std::string observed = FileSha256(path);
    if (observed != kProtocolSha256)
        throw std::runtime_error("fibonacci_protocol_identity_mismatch:expected=" +
            std::string(kProtocolSha256) + ",observed=" + observed);
}

enum class Partition { Development, Validation, Pre2025LockTest, Confirmation2025 };

inline std::string_view PartitionName(Partition value)
{
    switch (value) {
        case Partition::Development: return "development";
        case Partition::Validation: return "validation";
        case Partition::Pre2025LockTest: return "pre2025_lock_test";
        case Partition::Confirmation2025: return "confirmation_2025";
    }
    throw std::logic_error("unknown Fibonacci partition");
}

// UTC epoch boundaries corresponding to the frozen half-open calendar ranges.
inline constexpr std::int64_t kDevelopmentStart = 1262304000;
inline constexpr std::int64_t kValidationStart = 1546300800;
inline constexpr std::int64_t kPre2025Start = 1640995200;
inline constexpr std::int64_t kConfirmationStart = 1735689600;
inline constexpr std::int64_t kProtocolEnd = 1767225600;

inline std::optional<Partition> PartitionFor(std::int64_t timestamp)
{
    if (timestamp < kDevelopmentStart || timestamp >= kProtocolEnd) return std::nullopt;
    if (timestamp < kValidationStart) return Partition::Development;
    if (timestamp < kPre2025Start) return Partition::Validation;
    if (timestamp < kConfirmationStart) return Partition::Pre2025LockTest;
    return Partition::Confirmation2025;
}

inline std::int64_t PartitionEnd(Partition value)
{
    switch (value) {
        case Partition::Development: return kValidationStart;
        case Partition::Validation: return kPre2025Start;
        case Partition::Pre2025LockTest: return kConfirmationStart;
        case Partition::Confirmation2025: return kProtocolEnd;
    }
    throw std::logic_error("unknown Fibonacci partition");
}

struct TargetAudit {
    int assignedClass = 1;
    std::int64_t selectedTargetTimestamp = 0;
    std::int64_t terminalTimestamp = 0;
    float terminalLogReturn = 0.0f;
    bool eligible = false;
    std::string exclusionReason;
};

struct Row {
    std::string symbol;
    std::int64_t decisionTimestamp = 0;
    std::uint64_t sourceRowOrdinal = 0;
    std::array<float, kBaselineWidth> baseline{};
    std::array<float, kFibonacciWidth> fibonacci{};
    TargetAudit h4;
    TargetAudit h6;

    std::string Identity() const {
        return symbol + "|" + std::to_string(decisionTimestamp) + "|" +
            std::to_string(sourceRowOrdinal);
    }
};

inline void ValidateRows(const std::vector<Row>& rows)
{
    std::string previous;
    for (const Row& row : rows) {
        const auto partition = PartitionFor(row.decisionTimestamp);
        if (!partition) throw std::invalid_argument("fibonacci_row_outside_frozen_range:" + row.Identity());
        const std::string key = row.symbol + "|" + std::to_string(row.decisionTimestamp);
        if (!previous.empty() && key <= previous)
            throw std::invalid_argument("fibonacci_rows_not_strictly_sorted_or_duplicate:" + key);
        previous = key;
        for (float value : row.baseline)
            if (!std::isfinite(value)) throw std::invalid_argument("fibonacci_nonfinite_baseline:" + row.Identity());
        for (float value : row.fibonacci)
            if (!std::isfinite(value)) throw std::invalid_argument("fibonacci_nonfinite_feature:" + row.Identity());
        for (const TargetAudit* target : {&row.h4, &row.h6}) {
            if (!target->eligible) continue;
            if (target->assignedClass < 0 || target->assignedClass > 2 ||
                target->selectedTargetTimestamp <= row.decisionTimestamp ||
                target->terminalTimestamp <= row.decisionTimestamp ||
                target->terminalTimestamp >= PartitionEnd(*partition))
                throw std::invalid_argument("fibonacci_target_not_causally_partition_eligible:" + row.Identity());
        }
    }
}

struct FeatureSchema {
    std::vector<ModelInputFeatureSemantic> baseline;
    std::vector<ModelInputFeatureSemantic> fibonacci;
};

inline FeatureSchema FrozenFeatureSchema()
{
    const auto layout8 = ModelInputFeatureSemantics(kBaselineWidth);
    const auto layout9 = ModelInputFeatureSemantics(kCausalFibonacciStructuralModelInputWidth);
    if (layout8.size() != kBaselineWidth || layout9.size() != kBaselineWidth + kFibonacciWidth)
        throw std::logic_error("fibonacci_model_input_schema_width_mismatch");
    FeatureSchema result;
    result.baseline = layout8;
    // Layout-8's last four logical columns are its return suffix.  In physical
    // layout 9 the Fibonacci Tensor fields begin immediately after Tensor 75,
    // before that suffix.  Preserve this distinction in the research schema.
    const std::size_t fibonacciTensorStart = kBaselineWidth - kModelReturnFeatureCount;
    result.fibonacci.assign(layout9.begin() + static_cast<std::ptrdiff_t>(fibonacciTensorStart),
                            layout9.begin() + static_cast<std::ptrdiff_t>(fibonacciTensorStart + kFibonacciWidth));
    return result;
}

struct ArtifactProvenance {
    std::string protocolId;
    std::string protocolSha256;
    std::string codeCommit;
    std::string economicCalendarSnapshotId;
    std::string economicCalendarSnapshotSha256;
    std::string sourceAdapterIdentity;
    std::string sourceQueryIdentity;
    std::string sourcePriceDomain;
    std::string layoutIdentity;
    std::string baselineSchemaIdentity;
    std::string fibonacciSchemaIdentity;
    std::string targetIdentity;
    std::string symbolRangeIdentity;
    std::string warmupIdentity;
};

inline void ValidateArtifactProvenance(const ArtifactProvenance& value)
{
    const bool currentProtocol = value.protocolId == kProtocolId &&
        value.protocolSha256 == kProtocolSha256;
    const bool immutableV1SourceArtifact =
        value.protocolId == kV1SourceArtifactProtocolId &&
        value.protocolSha256 == kV1SourceArtifactProtocolSha256;
    if (!currentProtocol && !immutableV1SourceArtifact)
        throw std::invalid_argument("fibonacci_artifact_protocol_identity_mismatch");
    if (value.codeCommit.empty() || value.economicCalendarSnapshotId.empty() || value.economicCalendarSnapshotSha256.empty())
        throw std::invalid_argument("fibonacci_artifact_missing_economic_calendar_snapshot_identity");
    if (value.sourceAdapterIdentity.empty() || value.sourceQueryIdentity.empty() || value.sourcePriceDomain.empty())
        throw std::invalid_argument("fibonacci_artifact_missing_market_source_provenance");
    if (value.layoutIdentity.empty() || value.baselineSchemaIdentity.empty() ||
        value.fibonacciSchemaIdentity.empty() || value.targetIdentity.empty() ||
        value.symbolRangeIdentity.empty() || value.warmupIdentity.empty())
        throw std::invalid_argument("fibonacci_artifact_missing_frozen_execution_identity");
}

inline ArtifactProvenance FixtureArtifactProvenance()
{
    return {std::string(kProtocolId), std::string(kProtocolSha256), "fixture",
            "snapshot", "digest", "fixture-adapter", "fixture-query", "ask_ohlc",
            "model-input-semantic-layout-v9", "layout8-baseline[80]",
            "layout9-fibonacci-tensor[76..98]",
            "BuildLookaheadClassInfo:window=1:threshold=0.0008:H4,H6",
            "fixture-symbol-range", "full_history_warmup"};
}

struct Scale {
    std::string symbol;
    std::string featureIdentity;
    std::size_t fittedRows = 0;
    std::vector<double> mean;
    std::vector<double> standardDeviation;
    std::vector<bool> fitted;
};

inline Scale FitDevelopmentBaselineScale(const std::string& symbol,
                                         const std::vector<Row>& rows,
                                         const FeatureSchema& schema)
{
    if (schema.baseline.size() != kBaselineWidth) throw std::invalid_argument("fibonacci_baseline_schema_mismatch");
    Scale result; result.symbol = symbol; result.featureIdentity = "layout8-baseline[80]";
    result.mean.assign(kBaselineWidth, 0.0); result.standardDeviation.assign(kBaselineWidth, 0.0);
    result.fitted.assign(kBaselineWidth, false);
    for (const Row& row : rows) if (row.symbol == symbol && PartitionFor(row.decisionTimestamp) == Partition::Development) {
        ++result.fittedRows;
        for (std::size_t i = 0; i < kBaselineWidth; ++i) result.mean[i] += row.baseline[i];
    }
    if (result.fittedRows == 0) throw std::invalid_argument("fibonacci_no_development_rows:" + symbol);
    for (double& mean : result.mean) mean /= static_cast<double>(result.fittedRows);
    for (const Row& row : rows) if (row.symbol == symbol && PartitionFor(row.decisionTimestamp) == Partition::Development)
        for (std::size_t i = 0; i < kBaselineWidth; ++i) {
            const double d = row.baseline[i] - result.mean[i]; result.standardDeviation[i] += d * d;
        }
    for (std::size_t i = 0; i < kBaselineWidth; ++i) {
        if (schema.baseline[i].categorical) { result.mean[i] = 0.0; result.standardDeviation[i] = 1.0; result.fitted[i] = true; continue; }
        result.standardDeviation[i] = std::sqrt(std::max(0.0, result.standardDeviation[i] /
            static_cast<double>(result.fittedRows)));
        result.fitted[i] = result.standardDeviation[i] > 0.0;
        if (!result.fitted[i]) result.standardDeviation[i] = 1.0;
    }
    return result;
}

inline std::vector<double> ApplyBaselineScale(const Row& row, const Scale& scale,
                                              const FeatureSchema& schema)
{
    if (scale.symbol != row.symbol || scale.fittedRows == 0 || schema.baseline.size() != kBaselineWidth)
        throw std::invalid_argument("fibonacci_scale_provenance_mismatch");
    std::vector<double> out;
    for (std::size_t i = 0; i < kBaselineWidth; ++i)
        if (scale.fitted[i]) out.push_back(schema.baseline[i].categorical ? row.baseline[i] :
            (row.baseline[i] - scale.mean[i]) / scale.standardDeviation[i]);
    return out;
}

struct ReconstructionMetrics {
    bool available = false;
    std::string unavailableReason;
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double rSquared = std::numeric_limits<double>::quiet_NaN();
    double logLoss = std::numeric_limits<double>::quiet_NaN();
    double brier = std::numeric_limits<double>::quiet_NaN();
};

struct LbfgsOptions { std::size_t maxIterations = 250; double gradientInfinityTolerance = 1e-8; double relativeObjectiveTolerance = 1e-12; bool captureTrajectory = false; };
enum class LbfgsTerminationReason {
    GradientInfinityTolerance,
    RelativeObjectiveTolerance,
    MaximumIterations,
    ArmijoLineSearchFailure,
    NonFiniteObjectiveOrGradient,
};
inline std::string_view LbfgsTerminationReasonName(LbfgsTerminationReason reason)
{
    switch (reason) {
        case LbfgsTerminationReason::GradientInfinityTolerance: return "gradient_infinity_tolerance";
        case LbfgsTerminationReason::RelativeObjectiveTolerance: return "relative_objective_tolerance";
        case LbfgsTerminationReason::MaximumIterations: return "maximum_iterations";
        case LbfgsTerminationReason::ArmijoLineSearchFailure: return "armijo_line_search_failure";
        case LbfgsTerminationReason::NonFiniteObjectiveOrGradient: return "non_finite_objective_or_gradient";
    }
    throw std::logic_error("unknown_lbfgs_termination_reason");
}
struct LbfgsResult {
    std::vector<double> parameters;
    bool converged = false;
    LbfgsTerminationReason terminationReason = LbfgsTerminationReason::MaximumIterations;
    std::size_t iterations = 0;
    double initialObjective = std::numeric_limits<double>::quiet_NaN();
    double finalObjective = std::numeric_limits<double>::quiet_NaN();
    double finalGradientInfinityNorm = std::numeric_limits<double>::quiet_NaN();
    double finalRelativeObjectiveChange = std::numeric_limits<double>::quiet_NaN();
    double finalAcceptedStepSize = std::numeric_limits<double>::quiet_NaN();
    std::size_t terminatingLineSearchAttempts = 0;
    std::size_t finalHistorySize = 0;
    double finalDirectionalDerivative = std::numeric_limits<double>::quiet_NaN();
    struct TrajectoryPoint {
        std::size_t iterations = 0;
        double objective = std::numeric_limits<double>::quiet_NaN();
        double gradientInfinityNorm = std::numeric_limits<double>::quiet_NaN();
        double relativeObjectiveChange = std::numeric_limits<double>::quiet_NaN();
        double acceptedStepSize = std::numeric_limits<double>::quiet_NaN();
        std::size_t lineSearchAttempts = 0;
        std::size_t historySize = 0;
        double directionalDerivative = std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<TrajectoryPoint> trajectory;
};
inline bool LbfgsFinite(const std::vector<double>& values)
{
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}
inline double LbfgsInfinityNorm(const std::vector<double>& values)
{
    if (!LbfgsFinite(values)) return std::numeric_limits<double>::quiet_NaN();
    double result = 0.0; for (double value : values) result = std::max(result, std::abs(value));
    return result;
}
inline bool LbfgsTrajectoryCadence(std::size_t iterations) noexcept
{
    switch (iterations) {
        case 0: case 1: case 2: case 5: case 10: case 25: case 50: case 100: case 150: case 200: case 250:
        case 300: case 400: case 500: case 750: case 1000: case 1500: case 2000: case 3000: case 4000: case 5000: return true;
        default: return false;
    }
}
template <typename Objective>
inline LbfgsResult OptimizeLbfgs(std::vector<double> parameters, Objective objective,
                                  LbfgsOptions options = {})
{
    // Deterministic full-batch L-BFGS, fixed history/Armijo search and zero
    // caller-supplied initialization.  No random state or row reordering.
    constexpr std::size_t historyLimit = 8;
    std::vector<std::vector<double>> steps, gradients;
    double lastDirectionalDerivative = std::numeric_limits<double>::quiet_NaN();
    auto dot=[](const auto& a,const auto& b){ double r=0; for(std::size_t i=0;i<a.size();++i) r+=a[i]*b[i]; return r; };
    auto [value, gradient] = objective(parameters);
    LbfgsResult result;
    result.initialObjective = value;
    const auto record = [&](std::size_t iterations, double currentValue, const std::vector<double>& currentGradient,
                            double relativeObjectiveChange, double acceptedStepSize, std::size_t lineSearchAttempts,
                            std::size_t historySize, double directionalDerivative, bool force = false) {
        if (!options.captureTrajectory || (!force && !LbfgsTrajectoryCadence(iterations))) return;
        LbfgsResult::TrajectoryPoint point{iterations,currentValue,LbfgsInfinityNorm(currentGradient),relativeObjectiveChange,acceptedStepSize,lineSearchAttempts,historySize,directionalDerivative};
        if (!result.trajectory.empty() && result.trajectory.back().iterations == iterations) result.trajectory.back() = point;
        else result.trajectory.push_back(point);
    };
    const auto finish = [&](std::vector<double> finalParameters, bool converged, LbfgsTerminationReason reason,
                            std::size_t iterations, double finalValue, const std::vector<double>& finalGradient,
                            std::size_t terminatingLineSearchAttempts, std::size_t finalHistorySize,
                            double finalDirectionalDerivative) {
        result.parameters = std::move(finalParameters);
        result.converged = converged;
        result.terminationReason = reason;
        result.iterations = iterations;
        result.finalObjective = finalValue;
        result.finalGradientInfinityNorm = LbfgsInfinityNorm(finalGradient);
        result.terminatingLineSearchAttempts = terminatingLineSearchAttempts;
        result.finalHistorySize = finalHistorySize;
        result.finalDirectionalDerivative = finalDirectionalDerivative;
        return result;
    };
    record(0, value, gradient, result.finalRelativeObjectiveChange, result.finalAcceptedStepSize, 0, 0, std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(value) || !LbfgsFinite(gradient))
        return finish(std::move(parameters), false, LbfgsTerminationReason::NonFiniteObjectiveOrGradient,
                      0, value, gradient, 0, 0, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t iteration=0; iteration<options.maxIterations; ++iteration) {
        const double infinity = LbfgsInfinityNorm(gradient);
        if (infinity <= options.gradientInfinityTolerance) {
            record(iteration, value, gradient, result.finalRelativeObjectiveChange, result.finalAcceptedStepSize,
                   result.terminatingLineSearchAttempts, steps.size(),
                   lastDirectionalDerivative, true);
            return finish(std::move(parameters), true, LbfgsTerminationReason::GradientInfinityTolerance,
                          iteration, value, gradient, 0, steps.size(),
                          lastDirectionalDerivative);
        }
        std::vector<double> q=gradient, alpha(steps.size()), rho(steps.size());
        for(std::size_t n=steps.size(); n-- > 0;) { rho[n]=1.0/dot(gradients[n],steps[n]); alpha[n]=rho[n]*dot(steps[n],q); for(std::size_t i=0;i<q.size();++i) q[i]-=alpha[n]*gradients[n][i]; }
        double gamma=1.0; if(!steps.empty()) gamma=dot(steps.back(),gradients.back())/dot(gradients.back(),gradients.back());
        for(double& x:q) x*=gamma;
        for(std::size_t n=0;n<steps.size();++n) { const double beta=rho[n]*dot(gradients[n],q); for(std::size_t i=0;i<q.size();++i) q[i]+=steps[n][i]*(alpha[n]-beta); }
        for(double& x:q) x=-x;
        const double directional=dot(gradient,q);
        lastDirectionalDerivative = directional;
        if (!LbfgsFinite(q) || !std::isfinite(directional))
            return finish(std::move(parameters), false, LbfgsTerminationReason::NonFiniteObjectiveOrGradient,
                          iteration, value, gradient, 0, steps.size(), directional);
        double rate=1.0; std::vector<double> candidate(parameters.size()), nextGradient; double nextValue=value;
        std::size_t lineSearchAttempts=0; bool nonFiniteCandidate=false;
        while(rate > 1e-12) { ++lineSearchAttempts; for(std::size_t i=0;i<parameters.size();++i) candidate[i]=parameters[i]+rate*q[i]; auto evaluated=objective(candidate); nextValue=evaluated.first; nextGradient=std::move(evaluated.second); if(!std::isfinite(nextValue)||!LbfgsFinite(nextGradient)){nonFiniteCandidate=true;rate*=.5;continue;}if(nextValue <= value + 1e-4*rate*directional) break; rate*=.5; }
        if(rate <= 1e-12) { record(iteration, value, gradient, result.finalRelativeObjectiveChange, result.finalAcceptedStepSize, lineSearchAttempts, steps.size(), directional, true); return finish(std::move(parameters), false, nonFiniteCandidate ? LbfgsTerminationReason::NonFiniteObjectiveOrGradient : LbfgsTerminationReason::ArmijoLineSearchFailure, iteration, value, gradient, lineSearchAttempts, steps.size(), directional); }
        std::vector<double> s(parameters.size()), y(parameters.size()); for(std::size_t i=0;i<parameters.size();++i){s[i]=candidate[i]-parameters[i];y[i]=nextGradient[i]-gradient[i];}
        if(dot(s,y)>1e-14){ if(steps.size()==historyLimit){steps.erase(steps.begin());gradients.erase(gradients.begin());} steps.push_back(std::move(s));gradients.push_back(std::move(y)); }
        const double relativeObjectiveChange=std::abs(value-nextValue)/std::max(1.0,std::abs(value));
        result.finalRelativeObjectiveChange=relativeObjectiveChange;
        result.finalAcceptedStepSize=rate;
        result.terminatingLineSearchAttempts=lineSearchAttempts;
        record(iteration+1, nextValue, nextGradient, relativeObjectiveChange, rate, lineSearchAttempts, steps.size(), directional);
        if(std::abs(value-nextValue) <= options.relativeObjectiveTolerance*std::max(1.0,std::abs(value))) { record(iteration+1, nextValue, nextGradient, relativeObjectiveChange, rate, lineSearchAttempts, steps.size(), directional, true); return finish(std::move(candidate), true, LbfgsTerminationReason::RelativeObjectiveTolerance, iteration+1, nextValue, nextGradient, lineSearchAttempts, steps.size(), directional); }
        parameters=std::move(candidate); gradient=std::move(nextGradient); value=nextValue;
    }
    record(options.maxIterations, value, gradient, result.finalRelativeObjectiveChange, result.finalAcceptedStepSize,
           result.terminatingLineSearchAttempts, steps.size(), lastDirectionalDerivative, true);
    return finish(std::move(parameters), false, LbfgsTerminationReason::MaximumIterations,
                  options.maxIterations, value, gradient, result.terminatingLineSearchAttempts, steps.size(),
                  lastDirectionalDerivative);
}

inline std::vector<double> SolveLinearSystem(std::vector<std::vector<double>> a,
                                             std::vector<double> b)
{
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < n; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
        if (std::abs(a[pivot][col]) < 1e-12) throw std::invalid_argument("fibonacci_singular_ridge_design");
        std::swap(a[col], a[pivot]); std::swap(b[col], b[pivot]);
        const double d = a[col][col];
        for (std::size_t j = col; j < n; ++j) a[col][j] /= d;
        b[col] /= d;
        for (std::size_t row = 0; row < n; ++row) if (row != col) {
            const double factor = a[row][col];
            for (std::size_t j = col; j < n; ++j) a[row][j] -= factor * a[col][j];
            b[row] -= factor * b[col];
        }
    }
    return b;
}

inline ReconstructionMetrics EvaluateRidgeReconstruction(
    const std::vector<std::vector<double>>& developmentX, const std::vector<double>& developmentY,
    const std::vector<std::vector<double>>& holdoutX, const std::vector<double>& holdoutY,
    double lambda = 1.0)
{
    // Frozen convention shared by all harness diagnostics: minimize the
    // unnormalised summed loss plus lambda times the squared non-intercept
    // coefficients.  Lambda is never divided by the number of rows.
    ReconstructionMetrics result;
    if (developmentX.empty() || developmentX.size() != developmentY.size() ||
        holdoutX.size() != holdoutY.size()) { result.unavailableReason = "insufficient_or_misaligned_input"; return result; }
    const std::size_t p = developmentX.front().size() + 1;
    std::vector<std::vector<double>> gram(p, std::vector<double>(p)); std::vector<double> rhs(p);
    for (std::size_t r = 0; r < developmentX.size(); ++r) {
        std::vector<double> x{1.0}; x.insert(x.end(), developmentX[r].begin(), developmentX[r].end());
        if (x.size() != p) { result.unavailableReason = "inconsistent_design_width"; return result; }
        for (std::size_t i = 0; i < p; ++i) { rhs[i] += x[i] * developmentY[r]; for (std::size_t j = 0; j < p; ++j) gram[i][j] += x[i]*x[j]; }
    }
    for (std::size_t i = 1; i < p; ++i) gram[i][i] += lambda;
    const auto beta = SolveLinearSystem(std::move(gram), std::move(rhs));
    if (holdoutY.empty()) { result.unavailableReason = "empty_holdout"; return result; }
    double sumSquared = 0.0, mean = 0.0; for (double y : holdoutY) mean += y; mean /= holdoutY.size();
    double total = 0.0;
    for (std::size_t r = 0; r < holdoutY.size(); ++r) { double prediction = beta[0]; for (std::size_t i = 0; i < holdoutX[r].size(); ++i) prediction += beta[i+1]*holdoutX[r][i]; const double d = holdoutY[r]-prediction; sumSquared += d*d; const double c = holdoutY[r]-mean; total += c*c; }
    result.available = true; result.rmse = std::sqrt(sumSquared / holdoutY.size());
    if (total > 0.0) result.rSquared = 1.0 - sumSquared / total;
    else result.unavailableReason = "zero_variance_holdout_target";
    return result;
}

inline bool IsRecentEventState(const Row& row) noexcept
{
    return row.fibonacci[0] > 0.5f && (row.fibonacci[1] > 0.0f || row.fibonacci[12] > 0.0f);
}

enum class StructuralState { ScaleInvalid, ScaleValidNoRecentEvent, ScaleValidRecentEvent };
inline StructuralState StateFor(const Row& row) noexcept {
    if (row.fibonacci[0] <= 0.5f) return StructuralState::ScaleInvalid;
    return IsRecentEventState(row) ? StructuralState::ScaleValidRecentEvent :
                                     StructuralState::ScaleValidNoRecentEvent;
}

struct Exclusion
{
    std::string symbol;
    std::int64_t decisionTimestamp = 0;
    std::uint64_t sourceRowOrdinal = 0;
    std::string reason;
};

// The production extractor appends in canonical order.  This keeps the
// roughly multi-million-row population out of a duplicate in-memory vector
// while retaining the exact artifact layout used by fixture qualification.
class ArtifactWriter
{
public:
    ArtifactWriter(const std::filesystem::path& directory,
                   const ArtifactProvenance& provenance)
        : directory_(directory), provenance_(provenance)
    {
        ValidateArtifactProvenance(provenance_);
        std::filesystem::create_directories(directory_);
        const FeatureSchema schema = FrozenFeatureSchema();
        schema_.open(directory_ / "feature_schema.csv");
        rows_.open(directory_ / "rows.csv");
        exclusions_.open(directory_ / "exclusions.csv");
        if (!schema_ || !rows_ || !exclusions_)
            throw std::runtime_error("fibonacci_artifact_stream_open_failed");
        schema_ << "logical_family,logical_column,physical_model_input_column,name,categorical\n";
        for (std::size_t i = 0; i < schema.baseline.size(); ++i)
            schema_ << "baseline," << i << ',' << schema.baseline[i].modelInputColumn << ',' << schema.baseline[i].name << ',' << schema.baseline[i].categorical << '\n';
        for (std::size_t i = 0; i < schema.fibonacci.size(); ++i)
            schema_ << "fibonacci," << i << ',' << schema.fibonacci[i].modelInputColumn << ',' << schema.fibonacci[i].name << ',' << schema.fibonacci[i].categorical << '\n';
        rows_ << std::setprecision(9) << "row_identity,symbol,decision_timestamp,source_row_ordinal,partition,h4_class,h4_selected_target_timestamp,h4_terminal_timestamp,h4_terminal_log_return,h4_eligible,h4_exclusion_reason,h6_class,h6_selected_target_timestamp,h6_terminal_timestamp,h6_terminal_log_return,h6_eligible,h6_exclusion_reason";
        for (std::size_t i = 0; i < kBaselineWidth; ++i) rows_ << ",baseline_" << i;
        for (std::size_t i = 0; i < kFibonacciWidth; ++i) rows_ << ",fibonacci_" << i;
        rows_ << '\n';
        exclusions_ << "row_identity,symbol,decision_timestamp,source_row_ordinal,reason\n";
    }

    void Append(const Row& row)
    {
        if (completed_) throw std::logic_error("fibonacci_artifact_append_after_complete");
        ValidateRows({row});
        const std::string key = row.symbol + "|" + std::to_string(row.decisionTimestamp);
        if (!previousKey_.empty() && key <= previousKey_)
            throw std::invalid_argument("fibonacci_rows_not_strictly_sorted_or_duplicate:" + key);
        previousKey_ = key;
        rows_ << Csv(row.Identity()) << ',' << Csv(row.symbol) << ',' << row.decisionTimestamp << ',' << row.sourceRowOrdinal << ',' << PartitionName(*PartitionFor(row.decisionTimestamp))
              << ',' << row.h4.assignedClass << ',' << row.h4.selectedTargetTimestamp << ',' << row.h4.terminalTimestamp << ',' << row.h4.terminalLogReturn << ',' << row.h4.eligible << ',' << Csv(row.h4.exclusionReason)
              << ',' << row.h6.assignedClass << ',' << row.h6.selectedTargetTimestamp << ',' << row.h6.terminalTimestamp << ',' << row.h6.terminalLogReturn << ',' << row.h6.eligible << ',' << Csv(row.h6.exclusionReason);
        for (float value : row.baseline) rows_ << ',' << value;
        for (float value : row.fibonacci) rows_ << ',' << value;
        rows_ << '\n';
    }

    void Exclude(const Exclusion& exclusion)
    {
        if (completed_) throw std::logic_error("fibonacci_artifact_exclude_after_complete");
        exclusions_ << Csv(exclusion.symbol + "|" + std::to_string(exclusion.decisionTimestamp) + "|" + std::to_string(exclusion.sourceRowOrdinal)) << ',' << Csv(exclusion.symbol) << ',' << exclusion.decisionTimestamp << ',' << exclusion.sourceRowOrdinal << ',' << Csv(exclusion.reason) << '\n';
    }

    void Complete()
    {
        if (completed_) throw std::logic_error("fibonacci_artifact_complete_twice");
        schema_.close(); rows_.close(); exclusions_.close();
        if (!schema_ || !rows_ || !exclusions_)
            throw std::runtime_error("fibonacci_artifact_stream_write_failed");
        const std::string schemaHash = FileSha256(directory_ / "feature_schema.csv");
        const std::string rowsHash = FileSha256(directory_ / "rows.csv");
        const std::string exclusionsHash = FileSha256(directory_ / "exclusions.csv");
        const auto manifestPath = directory_ / "manifest.json";
        std::ofstream manifest(manifestPath);
        if (!manifest) throw std::runtime_error("fibonacci_artifact_manifest_open_failed");
        manifest << "{\n\"protocol_id\":\"" << provenance_.protocolId << "\",\n\"protocol_sha256\":\"" << provenance_.protocolSha256
                 << "\",\n\"code_commit\":\"" << provenance_.codeCommit << "\",\n\"economic_calendar_snapshot_id\":\"" << provenance_.economicCalendarSnapshotId
                 << "\",\n\"economic_calendar_snapshot_sha256\":\"" << provenance_.economicCalendarSnapshotSha256
                 << "\",\n\"source_adapter_identity\":\"" << provenance_.sourceAdapterIdentity
                 << "\",\n\"source_query_identity\":\"" << provenance_.sourceQueryIdentity
                 << "\",\n\"source_price_domain\":\"" << provenance_.sourcePriceDomain
                 << "\",\n\"layout_identity\":\"" << provenance_.layoutIdentity
                 << "\",\n\"baseline_schema_identity\":\"" << provenance_.baselineSchemaIdentity
                 << "\",\n\"fibonacci_schema_identity\":\"" << provenance_.fibonacciSchemaIdentity
                 << "\",\n\"target_identity\":\"" << provenance_.targetIdentity
                 << "\",\n\"symbol_range_identity\":\"" << provenance_.symbolRangeIdentity
                 << "\",\n\"warmup_identity\":\"" << provenance_.warmupIdentity
                 << "\",\n\"feature_configuration\":\"causal-fibonacci-layout9-symmetric-structural-v1\",\n\"layout_tensor_width\":99,\n\"baseline_width\":80,\n\"fibonacci_width\":23,\n\"feature_schema_sha256\":\"" << schemaHash
                 << "\",\n\"rows_sha256\":\"" << rowsHash << "\",\n\"exclusions_sha256\":\"" << exclusionsHash << "\"\n}\n";
        manifest.close();
        std::ofstream sums(directory_ / "sha256sums.txt");
        if (!sums) throw std::runtime_error("fibonacci_artifact_sums_open_failed");
        sums << schemaHash << "  feature_schema.csv\n" << rowsHash << "  rows.csv\n" << exclusionsHash << "  exclusions.csv\n" << FileSha256(manifestPath) << "  manifest.json\n";
        completed_ = true;
    }

private:
    static std::string Csv(const std::string& value)
    {
        if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
        std::string escaped = "\"";
        for (char character : value) escaped += character == '\"' ? "\"\"" : std::string(1, character);
        return escaped + "\"";
    }
    std::filesystem::path directory_;
    ArtifactProvenance provenance_;
    std::ofstream schema_, rows_, exclusions_;
    std::string previousKey_;
    bool completed_ = false;
};

inline void WriteArtifact(const std::filesystem::path& directory,
                          const ArtifactProvenance& provenance,
                          const std::vector<Row>& rows,
                          const std::vector<Exclusion>& exclusions = {})
{
    ArtifactWriter writer(directory, provenance);
    for (const Row& row : rows) writer.Append(row);
    for (const Exclusion& exclusion : exclusions) writer.Exclude(exclusion);
    writer.Complete();
}

inline void WriteFixtureArtifact(const std::filesystem::path& directory,
                                 const ArtifactProvenance& provenance,
                                 const std::vector<Row>& rows)
{
    WriteArtifact(directory, provenance, rows);
}

// Map the current authoritative physical layout-9 model input to the frozen
// research order.  The input is made by the normal Tensor/model-input path;
// this adapter deliberately only projects it and never reimplements features.
inline Row ProjectAuthoritativeLayout9Row(const std::string& symbol,
                                          std::int64_t timestamp,
                                          std::uint64_t sourceRowOrdinal,
                                          const std::array<float, kCausalFibonacciStructuralModelInputWidth>& physical)
{
    Row row; row.symbol = symbol; row.decisionTimestamp = timestamp; row.sourceRowOrdinal = sourceRowOrdinal;
    constexpr std::size_t tensorBaseline = kBaselineWidth - kModelReturnFeatureCount;
    for (std::size_t i = 0; i < tensorBaseline; ++i) row.baseline[i] = physical[i];
    for (std::size_t i = 0; i < kFibonacciWidth; ++i) row.fibonacci[i] = physical[tensorBaseline + i];
    for (std::size_t i = 0; i < kModelReturnFeatureCount; ++i) row.baseline[tensorBaseline + i] = physical[tensorBaseline + kFibonacciWidth + i];
    return row;
}

// The extraction adapter supplies this directly from BuildLookaheadClassInfo;
// the core keeps it as a plain audit projection so fixture qualification has no
// dependency on Tensor, MetaNN, or a database connection.
inline TargetAudit ProjectAuthoritativeTargetAudit(int assignedClass,
                                                   std::int64_t selectedTimestamp,
                                                   std::int64_t terminalTimestamp,
                                                   float terminalLogReturn)
{
    return {assignedClass, selectedTimestamp, terminalTimestamp, terminalLogReturn, true, {}};
}

inline double StableSigmoid(double value)
{
    if (value >= 0.0) { const double e = std::exp(-value); return 1.0 / (1.0 + e); }
    const double e = std::exp(value); return e / (1.0 + e);
}

struct LogisticModel { bool available = false; std::string unavailableReason; std::vector<double> coefficients; };

inline LogisticModel FitBinaryLogistic(const std::vector<std::vector<double>>& x,
                                       const std::vector<int>& y, double lambda = 1.0)
{
    LogisticModel result;
    if (x.empty() || x.size() != y.size()) { result.unavailableReason = "insufficient_or_misaligned_input"; return result; }
    const std::size_t p = x.front().size(); bool zero = false, one = false;
    for (std::size_t r = 0; r < x.size(); ++r) {
        if (x[r].size() != p || (y[r] != 0 && y[r] != 1)) { result.unavailableReason = "invalid_binary_design"; return result; }
        zero |= y[r] == 0; one |= y[r] == 1;
    }
    if (!zero || !one) { result.unavailableReason = "development_one_class_binary_target"; return result; }
    const auto optimized = OptimizeLbfgs(std::vector<double>(p + 1, 0.0), [&](const std::vector<double>& b) {
        double loss = 0.0; std::vector<double> gradient(p + 1);
        for (std::size_t r = 0; r < x.size(); ++r) {
            double z = b[0]; for (std::size_t j = 0; j < p; ++j) z += b[j + 1] * x[r][j];
            const double probability = StableSigmoid(z);
            loss += std::max(z, 0.0) - z * y[r] + std::log1p(std::exp(-std::abs(z)));
            const double error = probability - y[r]; gradient[0] += error;
            for (std::size_t j = 0; j < p; ++j) gradient[j + 1] += error * x[r][j];
        }
        for (std::size_t j = 1; j <= p; ++j) { loss += lambda * b[j] * b[j]; gradient[j] += 2.0 * lambda * b[j]; }
        return std::pair{loss, gradient};
    });
    if (!optimized.converged) { result.unavailableReason = "binary_lbfgs_not_converged"; return result; }
    result.available = true; result.coefficients = optimized.parameters; return result;
}

inline ReconstructionMetrics EvaluateBinaryLogistic(const LogisticModel& model,
                                                     const std::vector<std::vector<double>>& x,
                                                     const std::vector<int>& y)
{
    ReconstructionMetrics result;
    if (!model.available) { result.unavailableReason = model.unavailableReason; return result; }
    if (x.empty() || x.size() != y.size()) { result.unavailableReason = "insufficient_or_misaligned_input"; return result; }
    double loss = 0.0, brier = 0.0;
    for (std::size_t r = 0; r < x.size(); ++r) {
        if (x[r].size() + 1 != model.coefficients.size() || (y[r] != 0 && y[r] != 1)) { result.unavailableReason = "invalid_binary_holdout"; return result; }
        double z = model.coefficients[0]; for (std::size_t j = 0; j < x[r].size(); ++j) z += model.coefficients[j + 1] * x[r][j];
        const double p = std::clamp(StableSigmoid(z), 1e-15, 1.0 - 1e-15); loss -= y[r] ? std::log(p) : std::log(1.0 - p); const double d = p - y[r]; brier += d * d;
    }
    result.available = true; result.logLoss = loss / x.size(); result.brier = brier / x.size(); return result;
}

struct MultinomialModel { bool available = false; std::string unavailableReason; std::size_t featureCount = 0; std::vector<double> parameters; };
struct MultinomialMetrics { bool available = false; std::string unavailableReason; double logLoss = std::numeric_limits<double>::quiet_NaN(); double brier = std::numeric_limits<double>::quiet_NaN(); double accuracy = std::numeric_limits<double>::quiet_NaN(); std::vector<std::array<double, 3>> probabilities; std::vector<double> rowLogLoss; std::vector<double> rowBrier; };

inline std::array<double, 3> MultinomialProbabilities(const MultinomialModel& model, const std::vector<double>& x)
{
    if (!model.available || x.size() != model.featureCount) throw std::invalid_argument("fibonacci_multinomial_model_width_mismatch");
    std::array<double, 3> logits{}; double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < 3; ++c) { logits[c] = model.parameters[c * (model.featureCount + 1)]; for (std::size_t j = 0; j < x.size(); ++j) logits[c] += model.parameters[c * (model.featureCount + 1) + j + 1] * x[j]; maximum = std::max(maximum, logits[c]); }
    double sum = 0.0; for (double& value : logits) { value = std::exp(value - maximum); sum += value; } for (double& value : logits) value /= sum; return logits;
}

inline MultinomialModel FitMultinomialDiagnostic(const std::vector<std::vector<double>>& x,
                                                 const std::vector<int>& y, double lambda = 1.0)
{
    MultinomialModel result;
    if (x.empty() || x.size() != y.size()) { result.unavailableReason = "insufficient_or_misaligned_input"; return result; }
    const std::size_t p = x.front().size(); std::array<bool, 3> present{};
    for (std::size_t r = 0; r < x.size(); ++r) { if (x[r].size() != p || y[r] < 0 || y[r] > 2) { result.unavailableReason = "invalid_multinomial_design"; return result; } present[static_cast<std::size_t>(y[r])] = true; }
    if (!present[0] || !present[1] || !present[2]) { result.unavailableReason = "development_missing_directional_class"; return result; }
    const auto optimized = OptimizeLbfgs(std::vector<double>(3 * (p + 1), 0.0), [&](const std::vector<double>& b) {
        double loss = 0.0; std::vector<double> gradient(b.size());
        for (std::size_t r = 0; r < x.size(); ++r) {
            std::array<double, 3> z{}; double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < 3; ++c) { z[c] = b[c * (p + 1)]; for (std::size_t j = 0; j < p; ++j) z[c] += b[c * (p + 1) + j + 1] * x[r][j]; maximum = std::max(maximum, z[c]); }
            double sum = 0.0; for (double& value : z) { value = std::exp(value - maximum); sum += value; } for (double& value : z) value /= sum;
            loss -= std::log(std::max(z[static_cast<std::size_t>(y[r])], 1e-300));
            for (std::size_t c = 0; c < 3; ++c) { const double error = z[c] - (y[r] == static_cast<int>(c)); gradient[c * (p + 1)] += error; for (std::size_t j = 0; j < p; ++j) gradient[c * (p + 1) + j + 1] += error * x[r][j]; }
        }
        for (std::size_t c = 0; c < 3; ++c) for (std::size_t j = 1; j <= p; ++j) { const std::size_t n = c * (p + 1) + j; loss += lambda * b[n] * b[n]; gradient[n] += 2.0 * lambda * b[n]; }
        return std::pair{loss, gradient};
    });
    if (!optimized.converged) { result.unavailableReason = "multinomial_lbfgs_not_converged"; return result; }
    result.available = true; result.featureCount = p; result.parameters = optimized.parameters; return result;
}

inline MultinomialMetrics EvaluateMultinomialDiagnostic(const MultinomialModel& model,
                                                        const std::vector<std::vector<double>>& x,
                                                        const std::vector<int>& y)
{
    MultinomialMetrics result;
    if (!model.available) { result.unavailableReason = model.unavailableReason; return result; }
    if (x.empty() || x.size() != y.size()) { result.unavailableReason = "insufficient_or_misaligned_input"; return result; }
    double loss = 0.0, brier = 0.0, correct = 0.0;
    for (std::size_t r = 0; r < x.size(); ++r) {
        if (y[r] < 0 || y[r] > 2) { result.unavailableReason = "invalid_multinomial_holdout"; return result; }
        const auto p = MultinomialProbabilities(model, x[r]); result.probabilities.push_back(p); const auto label = static_cast<std::size_t>(y[r]);
        const double rowLoss = -std::log(std::max(p[label], 1e-300)); double rowBrier = 0.0; std::size_t best = 0;
        for (std::size_t c = 0; c < 3; ++c) { const double d = p[c] - (c == label ? 1.0 : 0.0); rowBrier += d * d; if (p[c] > p[best]) best = c; }
        rowBrier /= 3.0; loss += rowLoss; brier += rowBrier; correct += best == label; result.rowLogLoss.push_back(rowLoss); result.rowBrier.push_back(rowBrier);
    }
    result.available = true; result.logLoss = loss / x.size(); result.brier = brier / x.size(); result.accuracy = correct / x.size(); return result;
}

struct PairedDiagnosticDelta { bool available = false; std::string unavailableReason; double logLoss = std::numeric_limits<double>::quiet_NaN(); double brier = std::numeric_limits<double>::quiet_NaN(); double accuracy = std::numeric_limits<double>::quiet_NaN(); std::vector<double> rowLogLoss; std::vector<double> rowBrier; };
inline PairedDiagnosticDelta BaselineMinusAugmented(const MultinomialMetrics& baseline, const MultinomialMetrics& augmented)
{
    PairedDiagnosticDelta result;
    if (!baseline.available || !augmented.available || baseline.rowLogLoss.size() != augmented.rowLogLoss.size()) { result.unavailableReason = "paired_diagnostic_unavailable_or_misaligned"; return result; }
    result.available = true; result.logLoss = baseline.logLoss - augmented.logLoss; result.brier = baseline.brier - augmented.brier; result.accuracy = baseline.accuracy - augmented.accuracy;
    for (std::size_t i = 0; i < baseline.rowLogLoss.size(); ++i) { result.rowLogLoss.push_back(baseline.rowLogLoss[i] - augmented.rowLogLoss[i]); result.rowBrier.push_back(baseline.rowBrier[i] - augmented.rowBrier[i]); }
    return result;
}

struct QuantileBins { bool available = false; std::string unavailableReason; std::vector<double> upperEdges; };
inline QuantileBins FitDevelopmentQuantileBins(const std::vector<Row>& rows, const std::string& symbol, std::size_t fibonacciColumn)
{
    QuantileBins result; if (fibonacciColumn == 0 || fibonacciColumn >= kFibonacciWidth) { result.unavailableReason = "binary_or_invalid_fibonacci_column"; return result; }
    std::vector<double> values; for (const Row& row : rows) if (row.symbol == symbol && PartitionFor(row.decisionTimestamp) == Partition::Development) values.push_back(row.fibonacci[fibonacciColumn]);
    std::sort(values.begin(), values.end()); if (values.empty()) { result.unavailableReason = "no_development_values"; return result; }
    for (std::size_t q = 1; q <= 5; ++q) result.upperEdges.push_back(values[(q * values.size() + 4) / 5 - 1]);
    result.upperEdges.erase(std::unique(result.upperEdges.begin(), result.upperEdges.end()), result.upperEdges.end());
    if (result.upperEdges.size() < 2) { result.upperEdges.clear(); result.unavailableReason = "insufficient_distinct_development_values_after_tied_edge_collapse"; return result; }
    result.available = true; return result;
}
inline std::optional<std::size_t> QuantileBinFor(const QuantileBins& bins, double value)
{
    if (!bins.available || !std::isfinite(value)) return std::nullopt;
    return static_cast<std::size_t>(std::lower_bound(bins.upperEdges.begin(), bins.upperEdges.end(), value) - bins.upperEdges.begin());
}
struct OutcomeComposition { std::size_t rows = 0; std::array<std::size_t, 3> classes{}; std::vector<double> terminalReturns; };
inline void AddOutcome(OutcomeComposition& value, const TargetAudit& target) { if (!target.eligible) return; ++value.rows; ++value.classes[static_cast<std::size_t>(target.assignedClass)]; value.terminalReturns.push_back(target.terminalLogReturn); }
inline double Median(std::vector<double> values) { if (values.empty()) return std::numeric_limits<double>::quiet_NaN(); std::sort(values.begin(), values.end()); const std::size_t n = values.size(); return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0; }
inline double Percentile(std::vector<double> values, double q) { if (values.empty()) return std::numeric_limits<double>::quiet_NaN(); std::sort(values.begin(), values.end()); return values[static_cast<std::size_t>(std::ceil(q * values.size())) - 1]; }
inline std::array<OutcomeComposition, 3> StructuralLedger(const std::vector<Row>& rows, const std::string& symbol, Partition partition, bool h6)
{
    std::array<OutcomeComposition, 3> result; for (const Row& row : rows) if (row.symbol == symbol && PartitionFor(row.decisionTimestamp) == partition) { const auto target = h6 ? row.h6 : row.h4; if (!target.eligible) continue; const auto state = static_cast<std::size_t>(StateFor(row)); AddOutcome(result[state], target); } return result;
}

inline double AverageRank(const std::vector<double>& values, std::size_t index)
{
    std::vector<std::pair<double, std::size_t>> ordered; ordered.reserve(values.size()); for (std::size_t i = 0; i < values.size(); ++i) ordered.emplace_back(values[i], i); std::sort(ordered.begin(), ordered.end());
    std::vector<double> ranks(values.size()); for (std::size_t first = 0; first < ordered.size();) { std::size_t last = first + 1; while (last < ordered.size() && ordered[last].first == ordered[first].first) ++last; const double rank = (static_cast<double>(first + 1) + last) / 2.0; for (std::size_t n = first; n < last; ++n) ranks[ordered[n].second] = rank; first = last; } return ranks[index];
}
inline std::vector<double> AverageRanks(const std::vector<double>& values)
{
    std::vector<std::pair<double, std::size_t>> ordered; ordered.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) ordered.emplace_back(values[i], i);
    std::sort(ordered.begin(), ordered.end()); std::vector<double> ranks(values.size());
    for (std::size_t first = 0; first < ordered.size();) { std::size_t last = first + 1; while (last < ordered.size() && ordered[last].first == ordered[first].first) ++last; const double rank = (static_cast<double>(first + 1) + last) / 2.0; for (std::size_t n = first; n < last; ++n) ranks[ordered[n].second] = rank; first = last; }
    return ranks;
}
inline double Pearson(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size() || x.size() < 2) return std::numeric_limits<double>::quiet_NaN(); double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size(), my = std::accumulate(y.begin(), y.end(), 0.0) / y.size(), xy = 0.0, xx = 0.0, yy = 0.0; for (std::size_t i = 0; i < x.size(); ++i) { const double a=x[i]-mx,b=y[i]-my;xy+=a*b;xx+=a*a;yy+=b*b; } return xx > 0.0 && yy > 0.0 ? xy / std::sqrt(xx * yy) : std::numeric_limits<double>::quiet_NaN();
}
inline double Spearman(const std::vector<double>& x, const std::vector<double>& y) { if (x.size()!=y.size()) return std::numeric_limits<double>::quiet_NaN(); return Pearson(AverageRanks(x), AverageRanks(y)); }
inline double PointBiserial(const std::vector<int>& binary, const std::vector<double>& continuous) { std::vector<double> b; b.reserve(binary.size()); for(int x:binary) { if(x!=0&&x!=1) return std::numeric_limits<double>::quiet_NaN(); b.push_back(x); } return Pearson(b,continuous); }
struct AssociationMatrix { std::array<double, kBaselineWidth> scaleValidityPointBiserial{}; std::array<std::array<double, kBaselineWidth>, kFibonacciWidth - 1> spearman{}; std::array<std::size_t, kFibonacciWidth> nearestBaselineProxy{}; };
inline AssociationMatrix CompleteAssociationMatrix(const std::vector<Row>& rows, const std::string& symbol, Partition partition)
{
    AssociationMatrix result; std::vector<const Row*> selected; for(const Row& row:rows) if(row.symbol==symbol&&PartitionFor(row.decisionTimestamp)==partition) selected.push_back(&row);
    for(std::size_t b=0;b<kBaselineWidth;++b){std::vector<int> valid;std::vector<double>x;for(auto row:selected){valid.push_back(row->fibonacci[0]>.5f);x.push_back(row->baseline[b]);}result.scaleValidityPointBiserial[b]=PointBiserial(valid,x);}
    std::array<std::vector<double>, kBaselineWidth> baselineRanks; for (std::size_t b = 0; b < kBaselineWidth; ++b) { std::vector<double> values; values.reserve(selected.size()); for (auto row : selected) values.push_back(row->baseline[b]); baselineRanks[b] = AverageRanks(values); }
    for(std::size_t f=1;f<kFibonacciWidth;++f){std::vector<double> values;values.reserve(selected.size());for(auto row:selected)values.push_back(row->fibonacci[f]);const auto fibonacciRanks=AverageRanks(values);double best=-1.0;for(std::size_t b=0;b<kBaselineWidth;++b){const double value=Pearson(fibonacciRanks,baselineRanks[b]);result.spearman[f-1][b]=value;if(std::isfinite(value)&&std::abs(value)>best){best=std::abs(value);result.nearestBaselineProxy[f]=b;}}} return result;
}

struct MonthlyDelta { std::string calendarMonth; std::size_t rows = 0; double mean = std::numeric_limits<double>::quiet_NaN(); double median = std::numeric_limits<double>::quiet_NaN(); double p10 = std::numeric_limits<double>::quiet_NaN(); double p90 = std::numeric_limits<double>::quiet_NaN(); };
inline std::string CalendarMonthUtc(std::int64_t timestamp) { std::time_t raw = static_cast<std::time_t>(timestamp); std::tm time{}; gmtime_r(&raw, &time); char buffer[8]{}; std::strftime(buffer, sizeof(buffer), "%Y-%m", &time); return buffer; }
inline std::vector<MonthlyDelta> MonthlyDependenceAwareDeltas(const std::vector<std::int64_t>& timestamps, const std::vector<double>& deltas)
{
    if (timestamps.size() != deltas.size()) throw std::invalid_argument("fibonacci_monthly_delta_alignment_mismatch"); std::map<std::string,std::vector<double>> blocks; for(std::size_t i=0;i<timestamps.size();++i)blocks[CalendarMonthUtc(timestamps[i])].push_back(deltas[i]); std::vector<MonthlyDelta> out; for(auto& [month,values]:blocks){MonthlyDelta item;item.calendarMonth=month;item.rows=values.size();item.mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();item.median=Median(values);item.p10=Percentile(values,.1);item.p90=Percentile(values,.9);out.push_back(item);}return out;
}

inline std::string JsonStringField(const std::string& contents, std::string_view name)
{
    const std::string needle = "\"" + std::string(name) + "\":\""; const auto start = contents.find(needle); if(start==std::string::npos) throw std::invalid_argument("fibonacci_artifact_manifest_missing:"+std::string(name)); const auto valueStart=start+needle.size(), end=contents.find('"',valueStart); if(end==std::string::npos) throw std::invalid_argument("fibonacci_artifact_manifest_malformed:"+std::string(name)); return contents.substr(valueStart,end-valueStart);
}
inline void VerifyArtifactDirectory(const std::filesystem::path& directory)
{
    const auto manifestPath=directory/"manifest.json"; std::ifstream in(manifestPath); if(!in) throw std::invalid_argument("fibonacci_artifact_manifest_unreadable"); const std::string manifest((std::istreambuf_iterator<char>(in)),{});
    ArtifactProvenance p; p.protocolId=JsonStringField(manifest,"protocol_id");p.protocolSha256=JsonStringField(manifest,"protocol_sha256");p.codeCommit=JsonStringField(manifest,"code_commit");p.economicCalendarSnapshotId=JsonStringField(manifest,"economic_calendar_snapshot_id");p.economicCalendarSnapshotSha256=JsonStringField(manifest,"economic_calendar_snapshot_sha256");p.sourceAdapterIdentity=JsonStringField(manifest,"source_adapter_identity");p.sourceQueryIdentity=JsonStringField(manifest,"source_query_identity");p.sourcePriceDomain=JsonStringField(manifest,"source_price_domain");p.layoutIdentity=JsonStringField(manifest,"layout_identity");p.baselineSchemaIdentity=JsonStringField(manifest,"baseline_schema_identity");p.fibonacciSchemaIdentity=JsonStringField(manifest,"fibonacci_schema_identity");p.targetIdentity=JsonStringField(manifest,"target_identity");p.symbolRangeIdentity=JsonStringField(manifest,"symbol_range_identity");p.warmupIdentity=JsonStringField(manifest,"warmup_identity");ValidateArtifactProvenance(p);
    for(const auto& [field,file]:std::array<std::pair<const char*,const char*>,3>{{{"feature_schema_sha256","feature_schema.csv"},{"rows_sha256","rows.csv"},{"exclusions_sha256","exclusions.csv"}}})if(JsonStringField(manifest,field)!=FileSha256(directory/file))throw std::invalid_argument("fibonacci_artifact_file_hash_mismatch:"+std::string(file));
    std::ifstream sums(directory / "sha256sums.txt"); if (!sums) throw std::invalid_argument("fibonacci_artifact_sums_unreadable"); std::string digest, filename; std::size_t entries = 0; while (sums >> digest >> filename) { if (digest != FileSha256(directory / filename)) throw std::invalid_argument("fibonacci_artifact_sums_hash_mismatch:" + filename); ++entries; } if (entries != 4) throw std::invalid_argument("fibonacci_artifact_sums_incomplete");
}

struct ConfirmationAuthorization { bool explicitlyAuthorized = false; bool frozenPre2025ResultAvailable = false; bool noRefit = true; std::filesystem::path artifactDirectory; ArtifactProvenance developmentFitProvenance; };
inline void RequireConfirmation2025(const ConfirmationAuthorization& authorization)
{
    if (!authorization.explicitlyAuthorized) throw std::invalid_argument("fibonacci_confirmation_requires_explicit_authorization");
    if (!authorization.frozenPre2025ResultAvailable) throw std::invalid_argument("fibonacci_confirmation_requires_frozen_pre2025_result");
    if (!authorization.noRefit) throw std::invalid_argument("fibonacci_confirmation_refit_forbidden");
    ValidateArtifactProvenance(authorization.developmentFitProvenance); VerifyArtifactDirectory(authorization.artifactDirectory);
}

} // namespace EA::CausalFibonacciIncrementalInformation

#endif
