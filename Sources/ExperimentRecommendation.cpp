#include "ExperimentRecommendation.hpp"

#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::vector<std::string> Split(const std::string& text, char delimiter)
{
    std::vector<std::string> values;
    std::string current;
    std::istringstream input{text};
    while (std::getline(input, current, delimiter))
        values.push_back(current);
    if (!text.empty() && text.back() == delimiter)
        values.emplace_back();
    return values;
}

bool IsAsciiWhitespace(char value)
{
    switch (value)
    {
        case ' ': case '\t': case '\n': case '\r': case '\f': case '\v':
            return true;
        default:
            return false;
    }
}

std::string TrimAsciiWhitespace(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && IsAsciiWhitespace(value[begin])) ++begin;
    size_t end = value.size();
    while (end > begin && IsAsciiWhitespace(value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

bool ParseBoolean(const std::string& key, const std::string& value)
{
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    throw std::invalid_argument(
        "invalid_recommendation_policy_boolean:key=" + key);
}

int ParsePositiveInteger(const std::string& key, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try { parsed = std::stoll(value, &consumed, 10); }
    catch (...) {
        throw std::invalid_argument(
            "invalid_recommendation_policy_integer:key=" + key);
    }
    if (consumed != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "invalid_recommendation_policy_integer:key=" + key);
    return static_cast<int>(parsed);
}

double ParseFiniteDouble(const std::string& key, const std::string& value)
{
    double parsed = 0.0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed,
        std::chars_format::general);
    if (result.ec != std::errc{} ||
        result.ptr != value.data() + value.size() ||
        !std::isfinite(parsed))
        throw std::invalid_argument(
            "invalid_recommendation_policy_number:key=" + key);
    return parsed;
}

std::vector<double> ParseDoubleList(
    const std::string& key,
    const std::string& value)
{
    if (value.empty()) return {};
    std::vector<double> result;
    for (const std::string& rawItem : Split(value, ':'))
    {
        const std::string item = TrimAsciiWhitespace(rawItem);
        if (item.empty())
            throw std::invalid_argument(
                "invalid_recommendation_policy_list:key=" + key);
        result.push_back(ParseFiniteDouble(key, item));
    }
    return result;
}

std::vector<int> ParseIntegerList(
    const std::string& key,
    const std::string& value)
{
    if (value.empty()) return {};
    std::vector<int> result;
    for (const std::string& rawItem : Split(value, ':'))
    {
        const std::string item = TrimAsciiWhitespace(rawItem);
        if (item.empty())
            throw std::invalid_argument(
                "invalid_recommendation_policy_list:key=" + key);
        result.push_back(ParsePositiveInteger(key, item));
    }
    return result;
}

template <typename T>
void SortAndDeduplicate(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

template <typename T, typename Formatter>
std::string SortedJoin(std::vector<T> values, Formatter formatter)
{
    SortAndDeduplicate(values);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) out << ':';
        out << formatter(values[index]);
    }
    return out.str();
}

std::string SortedStringSetText(const std::set<std::string>& values)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    bool first = true;
    for (const std::string& value : values)
    {
        if (!first) out << ':';
        first = false;
        out << value;
    }
    return out.str();
}

std::string StableRecommendationHash(const std::string& canonicalText)
{
    // Canonical text, not this compact accelerator, is authoritative identity.
    // Future persistence must store and compare both after a hash match.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonicalText)
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> encoded{};
    for (size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - 1 - index) * 4);
        encoded[index] = digits[(hash >> shift) & 0x0fU];
    }
    return "fnv1a64:" + std::string(encoded.begin(), encoded.end());
}

std::string OptionalDoubleText(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "NULL";
}

std::string OptionalDateText(const std::optional<std::string>& value)
{
    return value ? CanonicalExperimentDateText(*value) : "NULL";
}

std::string OptionalLongLongText(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

int MatchedInteger(const std::ssub_match& match, int fallback = 0)
{
    return match.matched ? std::stoi(match.str()) : fallback;
}

} // namespace

std::string RecommendationSourceScopeText(RecommendationSourceScope value)
{
    switch (value)
    {
        case RecommendationSourceScope::symbolHorizon: return "symbol_horizon";
        case RecommendationSourceScope::symbol: return "symbol";
        case RecommendationSourceScope::global: return "global";
    }
    throw std::invalid_argument("invalid_recommendation_source_scope");
}

std::optional<RecommendationSourceScope> ParseRecommendationSourceScope(
    const std::string& value)
{
    if (value == "symbol_horizon") return RecommendationSourceScope::symbolHorizon;
    if (value == "symbol") return RecommendationSourceScope::symbol;
    if (value == "global") return RecommendationSourceScope::global;
    return std::nullopt;
}

std::string RecommendationStatusText(RecommendationStatus value)
{
    switch (value)
    {
        case RecommendationStatus::proposed: return "proposed";
        case RecommendationStatus::rejected: return "rejected";
        case RecommendationStatus::expired: return "expired";
        case RecommendationStatus::approved: return "approved";
    }
    throw std::invalid_argument("invalid_recommendation_status");
}

std::optional<RecommendationStatus> ParseRecommendationStatus(
    const std::string& value)
{
    if (value == "proposed") return RecommendationStatus::proposed;
    if (value == "rejected") return RecommendationStatus::rejected;
    if (value == "expired") return RecommendationStatus::expired;
    if (value == "approved") return RecommendationStatus::approved;
    return std::nullopt;
}

std::string RecommendationDuplicateTypeText(RecommendationDuplicateType value)
{
    switch (value)
    {
        case RecommendationDuplicateType::noDuplicate: return "no_duplicate";
        case RecommendationDuplicateType::existingExperiment: return "existing_experiment";
        case RecommendationDuplicateType::activeRecommendation: return "active_recommendation";
        case RecommendationDuplicateType::excludedTerminalExperiment:
            return "excluded_terminal_experiment";
    }
    throw std::invalid_argument("invalid_recommendation_duplicate_type");
}

RecommendationPolicy ParseRecommendationPolicy(const std::string& text)
{
    RecommendationPolicy policy;
    if (TrimAsciiWhitespace(text).empty()) return policy;

    std::set<std::string> seen;
    for (const std::string& rawAssignment : Split(text, ','))
    {
        const std::string assignment = TrimAsciiWhitespace(rawAssignment);
        const size_t equals = assignment.find('=');
        if (equals == std::string::npos)
            throw std::invalid_argument(
                "malformed_recommendation_policy_assignment");
        const std::string key = TrimAsciiWhitespace(
            assignment.substr(0, equals));
        const std::string value = TrimAsciiWhitespace(
            assignment.substr(equals + 1));
        if (key.empty())
            throw std::invalid_argument(
                "malformed_recommendation_policy_assignment");
        if (!seen.insert(key).second)
            throw std::invalid_argument(
                "duplicate_recommendation_policy_key:key=" + key);

        if (key == "enabled") policy.enabled = ParseBoolean(key, value);
        else if (key == "min_leader_score")
            policy.minimumLeaderScore = ParseFiniteDouble(key, value);
        else if (key == "min_infer_accuracy")
            policy.minimumInferenceAccuracy = ParseFiniteDouble(key, value);
        else if (key == "max_predicted_neutral")
            policy.maximumPredictedNeutralProportion =
                value == "null" ? std::nullopt :
                std::optional<double>{ParseFiniteDouble(key, value)};
        else if (key == "top_sources_per_scope")
            policy.topSourcesPerScope = ParsePositiveInteger(key, value);
        else if (key == "max_per_source")
            policy.maximumRecommendationsPerSource = ParsePositiveInteger(key, value);
        else if (key == "max_per_scan")
            policy.maximumRecommendationsPerScan = ParsePositiveInteger(key, value);
        else if (key == "min_evidence_count")
            policy.minimumEvidenceCount = ParsePositiveInteger(key, value);
        else if (key == "allowed_parameters")
        {
            policy.allowedParameters.clear();
            for (const std::string& rawItem : Split(value, ':'))
            {
                const std::string item = TrimAsciiWhitespace(rawItem);
                if (item.empty())
                    throw std::invalid_argument(
                        "invalid_recommendation_policy_list:key=" + key);
                policy.allowedParameters.insert(item);
            }
        }
        else if (key == "core_lr_offsets")
            policy.coreLrOffsets = ParseDoubleList(key, value);
        else if (key == "head_lr_offsets")
            policy.headLrOffsets = ParseDoubleList(key, value);
        else if (key == "label_threshold_offsets")
            policy.labelThresholdOffsets = ParseDoubleList(key, value);
        else if (key == "allow_horizon_changes")
            policy.allowHorizonChanges = ParseBoolean(key, value);
        else if (key == "permitted_horizons")
            policy.permittedHorizons = ParseIntegerList(key, value);
        else if (key == "source_scope")
        {
            const auto scope = ParseRecommendationSourceScope(value);
            if (!scope)
                throw std::invalid_argument("invalid_recommendation_source_scope");
            policy.sourceScope = *scope;
        }
        else if (key == "terminal_experiments_are_duplicates")
            policy.terminalExperimentsAreDuplicates = ParseBoolean(key, value);
        else if (key == "expiration_days")
            policy.expirationDays = value == "null" ? std::nullopt :
                std::optional<int>{ParsePositiveInteger(key, value)};
        else if (key == "policy_version")
            policy.policyVersion = ParsePositiveInteger(key, value);
        else
            throw std::invalid_argument(
                "unsupported_recommendation_policy_key:key=" + key);
    }

    const auto normalizeOffsets = [&](const char* parameter,
                                      const char* key,
                                      std::vector<double>& offsets) {
        if (!policy.allowedParameters.contains(parameter))
        {
            if (seen.contains(key) && !offsets.empty())
                throw std::invalid_argument(
                    std::string{key} + "_requires_allowed_parameter");
            offsets.clear();
        }
    };
    normalizeOffsets(kCoreLrMult, "core_lr_offsets", policy.coreLrOffsets);
    normalizeOffsets(kHeadLrMult, "head_lr_offsets", policy.headLrOffsets);
    normalizeOffsets(
        kLabelThreshold, "label_threshold_offsets",
        policy.labelThresholdOffsets);
    SortAndDeduplicate(policy.coreLrOffsets);
    SortAndDeduplicate(policy.headLrOffsets);
    SortAndDeduplicate(policy.labelThresholdOffsets);
    SortAndDeduplicate(policy.permittedHorizons);
    if (const auto error = ValidateRecommendationPolicy(policy))
        throw std::invalid_argument(*error);
    return policy;
}

std::optional<std::string> ValidateRecommendationPolicy(
    const RecommendationPolicy& policy)
{
    if (!std::isfinite(policy.minimumLeaderScore) ||
        !std::isfinite(policy.minimumInferenceAccuracy))
        return "recommendation_policy_threshold_not_finite";
    if (policy.maximumPredictedNeutralProportion &&
        (!std::isfinite(*policy.maximumPredictedNeutralProportion) ||
         *policy.maximumPredictedNeutralProportion < 0.0 ||
         *policy.maximumPredictedNeutralProportion > 1.0))
        return "maximum_predicted_neutral_out_of_range";
    if (policy.topSourcesPerScope <= 0 ||
        policy.maximumRecommendationsPerSource <= 0 ||
        policy.maximumRecommendationsPerScan <= 0 ||
        policy.minimumEvidenceCount <= 0 || policy.policyVersion <= 0)
        return "recommendation_policy_positive_count_required";
    if (policy.allowedParameters.empty())
        return "recommendation_allowed_parameters_required";
    const std::set<std::string> supported{
        kCoreLrMult, kHeadLrMult, kLabelThreshold, kPredictionHorizon};
    for (const std::string& name : policy.allowedParameters)
        if (!supported.contains(name))
            return "unsupported_recommendation_parameter:name=" + name;
    if (policy.allowedParameters.contains(kCoreLrMult) &&
        policy.coreLrOffsets.empty())
        return "core_lr_offsets_required";
    if (policy.allowedParameters.contains(kHeadLrMult) &&
        policy.headLrOffsets.empty())
        return "head_lr_offsets_required";
    if (policy.allowedParameters.contains(kLabelThreshold) &&
        policy.labelThresholdOffsets.empty())
        return "label_threshold_offsets_required";
    if (!policy.allowedParameters.contains(kCoreLrMult) &&
        !policy.coreLrOffsets.empty())
        return "core_lr_offsets_require_allowed_parameter";
    if (!policy.allowedParameters.contains(kHeadLrMult) &&
        !policy.headLrOffsets.empty())
        return "head_lr_offsets_require_allowed_parameter";
    if (!policy.allowedParameters.contains(kLabelThreshold) &&
        !policy.labelThresholdOffsets.empty())
        return "label_threshold_offsets_require_allowed_parameter";
    if (policy.allowedParameters.contains(kPredictionHorizon) &&
        !policy.allowHorizonChanges)
        return "prediction_horizon_requires_explicit_enablement";
    if (policy.allowHorizonChanges &&
        !policy.allowedParameters.contains(kPredictionHorizon))
        return "horizon_changes_require_prediction_horizon_parameter";
    if (policy.allowHorizonChanges && policy.permittedHorizons.empty())
        return "prediction_horizon_requires_permitted_horizons";
    if (!policy.allowHorizonChanges && !policy.permittedHorizons.empty())
        return "permitted_horizons_require_horizon_changes";
    if (std::any_of(
            policy.permittedHorizons.begin(), policy.permittedHorizons.end(),
            [](int value) { return value <= 0; }))
        return "permitted_horizons_must_be_positive";
    const auto invalidOffset = [](const std::vector<double>& offsets) {
        return std::any_of(offsets.begin(), offsets.end(), [](double value) {
            return !std::isfinite(value) || value == 0.0;
        });
    };
    if (invalidOffset(policy.coreLrOffsets)) return "invalid_core_lr_offset";
    if (invalidOffset(policy.headLrOffsets)) return "invalid_head_lr_offset";
    if (invalidOffset(policy.labelThresholdOffsets))
        return "invalid_label_threshold_offset";
    if (policy.expirationDays && *policy.expirationDays <= 0)
        return "recommendation_expiration_must_be_positive";
    return std::nullopt;
}

std::string CanonicalRecommendationDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("recommendation_identity_nonfinite_number");
    if (value == 0.0) value = 0.0; // Canonicalize IEEE negative zero.
    std::array<char, 128> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general);
    if (result.ec != std::errc{})
        throw std::runtime_error("recommendation_identity_number_format_failed");
    return std::string(buffer.data(), result.ptr);
}

std::string RecommendationCanonicalHash(const std::string& canonicalText)
{
    return StableRecommendationHash(canonicalText);
}

std::optional<RecommendationSemanticConfigurationVersion>
RecommendationSemanticConfigurationVersionFromCanonicalText(
    const std::string& canonicalText)
{
    constexpr std::string_view kV3 =
        "experiment_recommendation_semantic_configuration_v3;";
    constexpr std::string_view kV4 =
        "experiment_recommendation_semantic_configuration_v4;";
    constexpr std::string_view kV5 =
        "experiment_recommendation_semantic_configuration_v5;";
    constexpr std::string_view kV6 =
        "experiment_recommendation_semantic_configuration_v6;";
    constexpr std::string_view kV7 =
        "experiment_recommendation_semantic_configuration_v7;";
    constexpr std::string_view kV8 =
        "experiment_recommendation_semantic_configuration_v8;";
    constexpr std::string_view kV9 =
        "experiment_recommendation_semantic_configuration_v9;";
    if (canonicalText.starts_with(kV3))
        return RecommendationSemanticConfigurationVersion::v3;
    if (canonicalText.starts_with(kV4))
        return RecommendationSemanticConfigurationVersion::v4;
    if (canonicalText.starts_with(kV5))
        return RecommendationSemanticConfigurationVersion::v5;
    if (canonicalText.starts_with(kV6))
        return RecommendationSemanticConfigurationVersion::v6;
    if (canonicalText.starts_with(kV7))
        return RecommendationSemanticConfigurationVersion::v7;
    if (canonicalText.starts_with(kV8))
        return RecommendationSemanticConfigurationVersion::v8;
    if (canonicalText.starts_with(kV9))
        return RecommendationSemanticConfigurationVersion::v9;
    return std::nullopt;
}

std::string RecommendationSemanticConfigurationFromInvocationCanonicalText(
    const std::string& canonicalText)
{
    constexpr std::string_view kPrefix =
        "experiment_recommendation_invocation_v2;";
    constexpr std::string_view kLengthField = "semantic_configuration_length=";
    constexpr std::string_view kSemanticField = ";semantic_configuration=";
    if (!canonicalText.starts_with(kPrefix))
        throw std::invalid_argument(
            "unsupported_recommendation_invocation_configuration_version");
    const std::size_t lengthStart = kPrefix.size();
    if (canonicalText.compare(lengthStart, kLengthField.size(), kLengthField) != 0)
        throw std::invalid_argument(
            "invalid_recommendation_invocation_semantic_configuration_length");
    const std::size_t lengthValueStart = lengthStart + kLengthField.size();
    const std::size_t semanticFieldStart = canonicalText.find(
        kSemanticField, lengthValueStart);
    if (semanticFieldStart == std::string::npos)
        throw std::invalid_argument(
            "invalid_recommendation_invocation_semantic_configuration");
    const std::string lengthText = canonicalText.substr(
        lengthValueStart, semanticFieldStart - lengthValueStart);
    if (lengthText.empty() || !std::all_of(lengthText.begin(), lengthText.end(),
        [](unsigned char c) { return std::isdigit(c); }))
        throw std::invalid_argument(
            "invalid_recommendation_invocation_semantic_configuration_length");
    std::size_t semanticLength = 0;
    try { semanticLength = static_cast<std::size_t>(std::stoull(lengthText)); }
    catch (const std::exception&) {
        throw std::invalid_argument(
            "invalid_recommendation_invocation_semantic_configuration_length");
    }
    const std::size_t semanticStart = semanticFieldStart + kSemanticField.size();
    if (semanticLength > canonicalText.size() - semanticStart)
        throw std::invalid_argument(
            "invalid_recommendation_invocation_semantic_configuration_length");
    return canonicalText.substr(semanticStart, semanticLength);
}

FeatureWarmupScope RecommendationFeatureWarmupScopeFromCanonicalText(
    const std::string& canonicalText)
{
    const auto version =
        RecommendationSemanticConfigurationVersionFromCanonicalText(canonicalText);
    if (!version)
        throw std::invalid_argument(
            "unsupported_recommendation_semantic_configuration_version");
    if (*version != RecommendationSemanticConfigurationVersion::v5 &&
        *version != RecommendationSemanticConfigurationVersion::v6 &&
        *version != RecommendationSemanticConfigurationVersion::v7 &&
        *version != RecommendationSemanticConfigurationVersion::v8 &&
        *version != RecommendationSemanticConfigurationVersion::v9)
        return FeatureWarmupScope::LegacyColdBoundary;
    constexpr std::string_view kField = ";feature_warmup_scope=";
    const std::size_t fieldStart = canonicalText.rfind(kField);
    if (fieldStart == std::string::npos || fieldStart + kField.size() >= canonicalText.size())
        throw std::invalid_argument(
            "recommendation_semantic_configuration_v5_missing_feature_warmup_scope");
    const std::size_t valueStart = fieldStart + kField.size();
    const std::size_t valueEnd = canonicalText.find(';', valueStart);
    return ParseFeatureWarmupScope(canonicalText.substr(
        valueStart, valueEnd == std::string::npos
                        ? std::string::npos : valueEnd - valueStart));
}

Donchian20Mode RecommendationDonchian20ModeFromCanonicalText(
    const std::string& canonicalText)
{
    const auto version =
        RecommendationSemanticConfigurationVersionFromCanonicalText(canonicalText);
    if (!version)
        throw std::invalid_argument(
            "unsupported_recommendation_semantic_configuration_version");
    if (*version == RecommendationSemanticConfigurationVersion::v3)
        return kDefaultDonchian20Mode;
    constexpr std::string_view kField = ";donchian20_mode=";
    const std::size_t fieldStart = canonicalText.find(kField);
    if (fieldStart == std::string::npos || fieldStart + kField.size() >= canonicalText.size())
        throw std::invalid_argument(
            "recommendation_semantic_configuration_missing_donchian20_mode");
    const std::size_t valueStart = fieldStart + kField.size();
    const std::size_t valueEnd = canonicalText.find(';', valueStart);
    return ParseDonchian20Mode(canonicalText.substr(
        valueStart, valueEnd == std::string::npos ? std::string::npos : valueEnd - valueStart));
}

std::size_t RecommendationDonchianLookbackFromCanonicalText(
    const std::string& canonicalText)
{
    const auto version =
        RecommendationSemanticConfigurationVersionFromCanonicalText(canonicalText);
    if (!version)
        throw std::invalid_argument(
            "unsupported_recommendation_semantic_configuration_version");
    if (*version != RecommendationSemanticConfigurationVersion::v6 &&
        *version != RecommendationSemanticConfigurationVersion::v7 &&
        *version != RecommendationSemanticConfigurationVersion::v8 &&
        *version != RecommendationSemanticConfigurationVersion::v9)
        return kDefaultDonchianLookback;
    constexpr std::string_view kField = ";donchian_lookback=";
    const std::size_t fieldStart = canonicalText.find(kField);
    if (fieldStart == std::string::npos || fieldStart + kField.size() >= canonicalText.size())
        throw std::invalid_argument(
            "recommendation_semantic_configuration_v6_missing_donchian_lookback");
    const std::size_t valueStart = fieldStart + kField.size();
    const std::size_t valueEnd = canonicalText.find(';', valueStart);
    return ParseDonchianLookback(canonicalText.substr(
        valueStart, valueEnd == std::string::npos ? std::string::npos : valueEnd - valueStart));
}

std::string RecommendationPolicyCanonicalText(
    const RecommendationPolicy& policy)
{
    if (const auto error = ValidateRecommendationPolicy(policy))
        throw std::invalid_argument(*error);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_policy_v2"
        << ";enabled=" << (policy.enabled ? 1 : 0)
        << ";minimum_leader_score="
        << CanonicalRecommendationDouble(policy.minimumLeaderScore)
        << ";minimum_inference_accuracy="
        << CanonicalRecommendationDouble(policy.minimumInferenceAccuracy)
        << ";maximum_predicted_neutral_proportion="
        << OptionalDoubleText(policy.maximumPredictedNeutralProportion)
        << ";top_sources_per_scope=" << policy.topSourcesPerScope
        << ";maximum_recommendations_per_source="
        << policy.maximumRecommendationsPerSource
        << ";maximum_recommendations_per_scan="
        << policy.maximumRecommendationsPerScan
        << ";minimum_evidence_count=" << policy.minimumEvidenceCount
        << ";allowed_parameters="
        << SortedStringSetText(policy.allowedParameters)
        << ";core_lr_offsets="
        << SortedJoin(policy.coreLrOffsets, CanonicalRecommendationDouble)
        << ";head_lr_offsets="
        << SortedJoin(policy.headLrOffsets, CanonicalRecommendationDouble)
        << ";label_threshold_offsets="
        << SortedJoin(policy.labelThresholdOffsets, CanonicalRecommendationDouble)
        << ";allow_horizon_changes="
        << (policy.allowHorizonChanges ? 1 : 0)
        << ";permitted_horizons="
        << SortedJoin(policy.permittedHorizons,
                      [](int value) { return std::to_string(value); })
        << ";source_scope=" << RecommendationSourceScopeText(policy.sourceScope)
        << ";terminal_experiments_are_duplicates="
        << (policy.terminalExperimentsAreDuplicates ? 1 : 0)
        << ";expiration_days="
        << (policy.expirationDays ? std::to_string(*policy.expirationDays) : "NULL")
        << ";policy_version=" << policy.policyVersion;
    return out.str();
}

std::string RecommendationPolicyHash(const RecommendationPolicy& policy)
{
    return RecommendationCanonicalHash(
        RecommendationPolicyCanonicalText(policy));
}

std::string CanonicalExperimentDateText(const std::string& value)
{
    static const std::regex pattern{
        R"(^([0-9]{4})-([0-9]{2})-([0-9]{2})$)"};
    std::smatch match;
    if (!std::regex_match(value, match, pattern))
        throw std::invalid_argument("invalid_recommendation_date");

    const int yearValue = MatchedInteger(match[1]);
    const unsigned monthValue = static_cast<unsigned>(MatchedInteger(match[2]));
    const unsigned dayValue = static_cast<unsigned>(MatchedInteger(match[3]));
    if (yearValue <= 0)
        throw std::invalid_argument("invalid_recommendation_date");
    const std::chrono::year_month_day date{
        std::chrono::year{yearValue},
        std::chrono::month{monthValue},
        std::chrono::day{dayValue}};
    if (!date.ok())
        throw std::invalid_argument("invalid_recommendation_date");
    return value;
}

std::string EffectiveExperimentConfigurationCanonicalText(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version)
{
    const std::string symbol =
        EA::CanonicalSymbol::Normalize(configuration.symbol);
    if (configuration.predictionHorizon <= 0 ||
        configuration.targetEpochs <= 0)
        throw std::invalid_argument(
            "recommendation_identity_positive_integer_required");
    if (!std::isfinite(configuration.labelThreshold) ||
        (configuration.coreLrMult &&
         !std::isfinite(*configuration.coreLrMult)) ||
        (configuration.headLrMult &&
         !std::isfinite(*configuration.headLrMult)))
        throw std::invalid_argument(
            "recommendation_identity_invalid_numeric_configuration");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    const int versionNumber =
        version == RecommendationSemanticConfigurationVersion::v3 ? 3 :
        version == RecommendationSemanticConfigurationVersion::v4 ? 4 :
        version == RecommendationSemanticConfigurationVersion::v5 ? 5 :
        version == RecommendationSemanticConfigurationVersion::v6 ? 6 :
        version == RecommendationSemanticConfigurationVersion::v7 ? 7 :
        version == RecommendationSemanticConfigurationVersion::v8 ? 8 : 9;
    out << "experiment_recommendation_semantic_configuration_v"
        << versionNumber
        << ";symbol=" << symbol
        << ";prediction_horizon=" << configuration.predictionHorizon
        << ";label_threshold="
        << CanonicalRecommendationDouble(configuration.labelThreshold)
        << ";core_lr_mult=" << OptionalDoubleText(configuration.coreLrMult)
        << ";head_lr_mult=" << OptionalDoubleText(configuration.headLrMult)
        << ";target_epochs=" << configuration.targetEpochs
        << ";train_start_date=" << CanonicalExperimentDateText(
            configuration.trainStartDate)
        << ";train_end_date=" << CanonicalExperimentDateText(
            configuration.trainEndDate)
        << ";infer_start_date=" << OptionalDateText(
            configuration.inferStartDate)
        << ";infer_end_date=" << OptionalDateText(configuration.inferEndDate);
    if (version != RecommendationSemanticConfigurationVersion::v3)
        out << ";donchian20_mode=" << Donchian20ModeText(configuration.donchian20Mode);
    if (version == RecommendationSemanticConfigurationVersion::v5 ||
        version == RecommendationSemanticConfigurationVersion::v6 ||
        version == RecommendationSemanticConfigurationVersion::v7 ||
        version == RecommendationSemanticConfigurationVersion::v8 ||
        version == RecommendationSemanticConfigurationVersion::v9)
        out << ";feature_warmup_scope=" << FeatureWarmupScopeText(
            configuration.featureWarmupScope);
    if (version == RecommendationSemanticConfigurationVersion::v6 ||
        version == RecommendationSemanticConfigurationVersion::v7 ||
        version == RecommendationSemanticConfigurationVersion::v8 ||
        version == RecommendationSemanticConfigurationVersion::v9)
        out << ";donchian_lookback="
            << ValidateDonchianLookback(configuration.donchianLookback);
    return out.str();
}

std::string RecommendationCandidateHash(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version)
{
    return RecommendationCanonicalHash(
        EffectiveExperimentConfigurationCanonicalText(configuration, version));
}

RecommendationCandidateIdentity BuildRecommendationCandidateIdentity(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version)
{
    RecommendationCandidateIdentity identity;
    identity.configuration = configuration;
    identity.configuration.symbol =
        EA::CanonicalSymbol::Normalize(configuration.symbol);
    identity.configuration.trainStartDate =
        CanonicalExperimentDateText(configuration.trainStartDate);
    identity.configuration.trainEndDate =
        CanonicalExperimentDateText(configuration.trainEndDate);
    if (configuration.inferStartDate)
        identity.configuration.inferStartDate =
            CanonicalExperimentDateText(*configuration.inferStartDate);
    if (configuration.inferEndDate)
        identity.configuration.inferEndDate =
            CanonicalExperimentDateText(*configuration.inferEndDate);
    identity.canonicalText =
        EffectiveExperimentConfigurationCanonicalText(identity.configuration, version);
    identity.hash = RecommendationCanonicalHash(identity.canonicalText);
    return identity;
}

std::string ExperimentInvocationCanonicalText(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version)
{
    if (invocation.checkpointInterval <= 0)
        throw std::invalid_argument(
            "recommendation_invocation_checkpoint_interval_must_be_positive");
    if (invocation.resumeModelId && *invocation.resumeModelId <= 0)
        throw std::invalid_argument(
            "recommendation_invocation_resume_model_id_must_be_positive");
    const std::string semantic =
        EffectiveExperimentConfigurationCanonicalText(
            invocation.configuration, version);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_invocation_v2"
        << ";semantic_configuration_length=" << semantic.size()
        << ";semantic_configuration=" << semantic
        << ";checkpoint_interval=" << invocation.checkpointInterval
        << ";resume_model_id=" << OptionalLongLongText(
            invocation.resumeModelId);
    return out.str();
}

std::string ExperimentInvocationHash(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version)
{
    return RecommendationCanonicalHash(
        ExperimentInvocationCanonicalText(invocation, version));
}

RecommendationInvocationIdentity BuildRecommendationInvocationIdentity(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version)
{
    RecommendationInvocationIdentity identity;
    identity.invocation = invocation;
    const RecommendationCandidateIdentity semantic =
        BuildRecommendationCandidateIdentity(invocation.configuration, version);
    identity.invocation.configuration = semantic.configuration;
    identity.canonicalText = ExperimentInvocationCanonicalText(
        identity.invocation, version);
    identity.hash = RecommendationCanonicalHash(identity.canonicalText);
    return identity;
}

} // namespace EA::ExperimentRecommendation
