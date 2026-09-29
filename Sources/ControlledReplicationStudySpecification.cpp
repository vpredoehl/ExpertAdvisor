#include "ControlledReplicationStudySpecification.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::ControlledReplicationStudy
{
namespace
{

std::string Escape(std::string_view value)
{
    std::ostringstream out;
    constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char c : value)
    {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' ||
            c == ':' || c == '/')
            out << static_cast<char>(c);
        else
            out << '%' << hex[c >> 4] << hex[c & 0x0f];
    }
    return out.str();
}

int Hex(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

std::string Unescape(std::string_view value)
{
    std::string out;
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] != '%')
        {
            out.push_back(value[i]);
            continue;
        }
        if (i + 2 >= value.size())
            throw std::invalid_argument("study_specification_bad_escape");
        const int hi = Hex(value[i + 1]);
        const int lo = Hex(value[i + 2]);
        if (hi < 0 || lo < 0)
            throw std::invalid_argument("study_specification_bad_escape");
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
    }
    return out;
}

std::string Frame(std::string_view label, std::string_view value)
{
    std::ostringstream out;
    out << label << ':' << value.size() << ':' << value << ';';
    return out.str();
}

std::string JoinSorted(const std::vector<std::string>& values)
{
    std::vector<std::string> copy = values;
    std::sort(copy.begin(), copy.end());
    std::ostringstream out;
    out << copy.size() << ';';
    for (const auto& value : copy) out << Frame("v", value);
    return out.str();
}

std::string CanonicalDimensionList(
    const std::vector<std::pair<std::string, std::string>>& dimensions)
{
    auto copy = dimensions;
    std::sort(copy.begin(), copy.end());
    std::ostringstream out;
    out << copy.size() << ';';
    for (const auto& [key, value] : copy)
    {
        out << Frame("k", key) << Frame("v", value);
    }
    return out.str();
}

std::string Canonical(const Specification& value)
{
    std::ostringstream out;
    out << "controlled_replication_study_specification_v1;";
    out << Frame("version", std::to_string(value.version));
    out << Frame("study_identifier", value.studyIdentifier);
    out << Frame("study_type", value.studyType);
    out << Frame("intervention_field", value.interventionField);
    out << Frame("control_semantics", value.controlSemantics);
    out << Frame("control_value", value.controlValue);
    out << Frame("treatment_value", value.treatmentValue);
    out << Frame("replication_dimension", value.replicationDimension);
    out << Frame("allowed_context_dimensions",
                 JoinSorted(value.allowedContextDimensions));
    out << Frame("required_configured_identity_fields",
                 JoinSorted(value.requiredConfiguredIdentityFields));
    out << Frame("required_execution_provenance_fields",
                 JoinSorted(value.requiredExecutionProvenanceFields));
    out << Frame("aggregation_policy", value.aggregationPolicy);
    out << Frame("statistical_independence", value.statisticalIndependence);
    out << Frame("subjective_winner", value.subjectiveWinner);
    out << Frame("freeze_timestamp", value.freezeTimestamp);
    out << Frame("exclusions", JoinSorted(value.exclusions));

    std::vector<const ContextSpec*> contexts;
    contexts.reserve(value.contexts.size());
    for (const auto& context : value.contexts) contexts.push_back(&context);
    std::sort(contexts.begin(), contexts.end(),
              [](const ContextSpec* left, const ContextSpec* right)
              { return left->ordinal < right->ordinal; });
    out << Frame("context_count", std::to_string(contexts.size()));
    for (const ContextSpec* context : contexts)
    {
        out << Frame("context_ordinal", context->ordinal)
            << Frame("context_dimensions",
                     CanonicalDimensionList(context->dimensions));
        out << Frame("pair_count", std::to_string(context->pairs.size()));
        for (std::size_t i = 0; i < context->pairs.size(); ++i)
        {
            const auto& pair = context->pairs[i];
            out << Frame("pair_ordinal", std::to_string(i + 1))
                << Frame("arm_a", std::to_string(pair.armA))
                << Frame("arm_b", std::to_string(pair.armB))
                << Frame("requested_seed", std::to_string(pair.requestedSeed));
        }
    }
    return out.str();
}

std::vector<std::string> Split(std::string_view text, char delimiter)
{
    std::vector<std::string> result;
    std::size_t start = 0;
    while (true)
    {
        const std::size_t end = text.find(delimiter, start);
        result.emplace_back(text.substr(start, end == std::string_view::npos
                                                 ? std::string_view::npos
                                                 : end - start));
        if (end == std::string_view::npos) return result;
        start = end + 1;
    }
}

long long Positive(std::string_view text, std::string_view reason)
{
    if (text.empty() || std::any_of(text.begin(), text.end(),
                                    [](unsigned char c) { return !std::isdigit(c); }))
        throw std::invalid_argument(std::string(reason));
    std::size_t consumed = 0;
    const long long value = std::stoll(std::string(text), &consumed);
    if (consumed != text.size() || value <= 0)
        throw std::invalid_argument(std::string(reason));
    return value;
}

unsigned int Seed(std::string_view text)
{
    const long long value = Positive(text, "study_specification_bad_seed");
    if (value > 0xffffffffLL)
        throw std::invalid_argument("study_specification_bad_seed");
    return static_cast<unsigned int>(value);
}

ContextSpec& Context(std::vector<ContextSpec>& contexts, std::string_view ordinal)
{
    auto found = std::find_if(contexts.begin(), contexts.end(),
                              [ordinal](const ContextSpec& context)
                              { return context.ordinal == ordinal; });
    if (found == contexts.end())
        throw std::invalid_argument("study_specification_unknown_context");
    return *found;
}

void RequireUnique(const std::vector<std::string>& values,
                   std::string_view reason)
{
    std::set<std::string> unique(values.begin(), values.end());
    if (unique.size() != values.size())
        throw std::invalid_argument(std::string(reason));
}

} // namespace

std::string Canonicalize(const Specification& specification)
{
    return Canonical(specification);
}

std::string IdentityHash(const Specification& specification)
{
    return EA::TrainingObjective::DeterministicHash(Canonical(specification));
}

std::string Render(const Specification& specification)
{
    std::ostringstream out;
    out << "CONTROLLED_REPLICATION_STUDY_SPECIFICATION\n";
    out << "version=" << specification.version << '\n';
    out << "study_identifier=" << Escape(specification.studyIdentifier) << '\n';
    out << "study_type=" << Escape(specification.studyType) << '\n';
    out << "intervention_field=" << Escape(specification.interventionField) << '\n';
    out << "control_semantics=" << Escape(specification.controlSemantics) << '\n';
    out << "control_value=" << Escape(specification.controlValue) << '\n';
    out << "treatment_value=" << Escape(specification.treatmentValue) << '\n';
    out << "replication_dimension=" << Escape(specification.replicationDimension) << '\n';
    for (const auto& value : specification.allowedContextDimensions)
        out << "allowed_context_dimension=" << Escape(value) << '\n';
    for (const auto& value : specification.requiredConfiguredIdentityFields)
        out << "required_configured_identity_field=" << Escape(value) << '\n';
    for (const auto& value : specification.requiredExecutionProvenanceFields)
        out << "required_execution_provenance_field=" << Escape(value) << '\n';
    out << "aggregation_policy=" << Escape(specification.aggregationPolicy) << '\n';
    out << "statistical_independence=" << Escape(specification.statisticalIndependence) << '\n';
    out << "subjective_winner=" << Escape(specification.subjectiveWinner) << '\n';
    out << "freeze_timestamp=" << Escape(specification.freezeTimestamp) << '\n';
    for (const auto& value : specification.exclusions)
        out << "exclusion=" << Escape(value) << '\n';
    for (const auto& context : specification.contexts)
    {
        out << "context=" << Escape(context.ordinal) << '\n';
        for (const auto& [key, value] : context.dimensions)
            out << "context_dimension=" << Escape(context.ordinal) << '|'
                << Escape(key) << '|' << Escape(value) << '\n';
        for (const auto& pair : context.pairs)
            out << "pair=" << Escape(context.ordinal) << '|'
                << pair.armA << '|' << pair.armB << '|'
                << pair.requestedSeed << '\n';
    }
    out << "identity_hash=" << IdentityHash(specification) << '\n';
    return out.str();
}

Specification Parse(const std::string& artifact)
{
    std::istringstream input(artifact);
    Specification result;
    std::string line;
    bool first = true;
    bool hashSeen = false;
    std::set<std::string> scalarKeys;
    while (std::getline(input, line))
    {
        if (line.empty()) continue;
        if (first)
        {
            if (line != "CONTROLLED_REPLICATION_STUDY_SPECIFICATION")
                throw std::invalid_argument("study_specification_bad_header");
            first = false;
            continue;
        }
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            throw std::invalid_argument("study_specification_malformed_line");
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        const auto addScalar = [&](const std::string& scalar)
        {
            if (!scalarKeys.insert(scalar).second)
                throw std::invalid_argument("study_specification_duplicate_field");
        };
        if (key == "version") { addScalar(key); result.version = static_cast<int>(Positive(value, "study_specification_bad_version")); }
        else if (key == "study_identifier") { addScalar(key); result.studyIdentifier = Unescape(value); }
        else if (key == "study_type") { addScalar(key); result.studyType = Unescape(value); }
        else if (key == "intervention_field") { addScalar(key); result.interventionField = Unescape(value); }
        else if (key == "control_semantics") { addScalar(key); result.controlSemantics = Unescape(value); }
        else if (key == "control_value") { addScalar(key); result.controlValue = Unescape(value); }
        else if (key == "treatment_value") { addScalar(key); result.treatmentValue = Unescape(value); }
        else if (key == "replication_dimension") { addScalar(key); result.replicationDimension = Unescape(value); }
        else if (key == "aggregation_policy") { addScalar(key); result.aggregationPolicy = Unescape(value); }
        else if (key == "statistical_independence") { addScalar(key); result.statisticalIndependence = Unescape(value); }
        else if (key == "subjective_winner") { addScalar(key); result.subjectiveWinner = Unescape(value); }
        else if (key == "freeze_timestamp") { addScalar(key); result.freezeTimestamp = Unescape(value); }
        else if (key == "identity_hash") { addScalar(key); hashSeen = true; result.identityHash = Unescape(value); }
        else if (key == "allowed_context_dimension") result.allowedContextDimensions.push_back(Unescape(value));
        else if (key == "required_configured_identity_field") result.requiredConfiguredIdentityFields.push_back(Unescape(value));
        else if (key == "required_execution_provenance_field") result.requiredExecutionProvenanceFields.push_back(Unescape(value));
        else if (key == "exclusion") result.exclusions.push_back(Unescape(value));
        else if (key == "context")
        {
            const std::string ordinal = Unescape(value);
            if (ordinal.empty() || std::any_of(result.contexts.begin(), result.contexts.end(),
                                               [&](const ContextSpec& c) { return c.ordinal == ordinal; }))
                throw std::invalid_argument("study_specification_duplicate_context");
            result.contexts.push_back(ContextSpec{ordinal, {}, {}});
        }
        else if (key == "context_dimension" || key == "pair")
        {
            const auto fields = Split(value, '|');
            if ((key == "context_dimension" && fields.size() != 3) ||
                (key == "pair" && fields.size() != 4))
                throw std::invalid_argument("study_specification_malformed_membership");
            ContextSpec& context = Context(result.contexts, Unescape(fields[0]));
            if (key == "context_dimension")
                context.dimensions.emplace_back(Unescape(fields[1]), Unescape(fields[2]));
            else
                context.pairs.push_back(PairSpec{Positive(fields[1], "study_specification_bad_arm_id"),
                                                 Positive(fields[2], "study_specification_bad_arm_id"),
                                                 Seed(fields[3])});
        }
        else
            throw std::invalid_argument("study_specification_unknown_field");
    }
    if (first || !hashSeen)
        throw std::invalid_argument("study_specification_missing_identity_hash");
    return result;
}

Specification Load(const std::string& path)
{
    std::ifstream input(std::filesystem::path(path), std::ios::binary);
    if (!input) throw std::invalid_argument("study_specification_artifact_missing");
    std::ostringstream contents;
    contents << input.rdbuf();
    return Parse(contents.str());
}

ValidationResult Validate(const Specification& specification)
{
    ValidationResult result;
    try
    {
        if (specification.version != kSpecificationVersion)
            throw std::invalid_argument("study_specification_unsupported_version");
        if (specification.studyIdentifier.empty())
            throw std::invalid_argument("study_specification_missing_identifier");
        if (specification.studyType != kStudyType)
            throw std::invalid_argument("study_specification_wrong_type");
        if (specification.interventionField != "feature_ablation_mask")
            throw std::invalid_argument("study_specification_wrong_intervention");
        if (specification.controlSemantics != "empty_string")
            throw std::invalid_argument("study_specification_wrong_control_semantics");
        if (specification.interventionField.empty() ||
            specification.controlSemantics.empty() ||
            specification.replicationDimension.empty() ||
            specification.aggregationPolicy.empty() ||
            specification.freezeTimestamp.empty())
            throw std::invalid_argument("study_specification_missing_contract_field");
        if (specification.controlValue != "")
            throw std::invalid_argument("study_specification_control_not_empty");
        if (specification.treatmentValue.empty())
            throw std::invalid_argument("study_specification_missing_treatment");
        if (specification.replicationDimension != "fresh_initialization_seed")
            throw std::invalid_argument("study_specification_unsupported_replication_dimension");
        if (specification.aggregationPolicy !=
            "unweighted_mean_of_context_family_means")
            throw std::invalid_argument("study_specification_unsupported_aggregation");
        if (specification.statisticalIndependence != "not_inferred")
            throw std::invalid_argument("study_specification_statistical_independence_must_be_not_inferred");
        if (specification.subjectiveWinner != "NONE")
            throw std::invalid_argument("study_specification_subjective_winner_forbidden");
        if (specification.freezeTimestamp.back() != 'Z')
            throw std::invalid_argument("study_specification_freeze_timestamp_not_utc");
        RequireUnique(specification.allowedContextDimensions,
                      "study_specification_duplicate_allowed_dimension");
        RequireUnique(specification.requiredConfiguredIdentityFields,
                      "study_specification_duplicate_configured_field");
        RequireUnique(specification.requiredExecutionProvenanceFields,
                      "study_specification_duplicate_provenance_field");
        RequireUnique(specification.exclusions,
                      "study_specification_duplicate_exclusion");
        const auto has = [](const std::vector<std::string>& values,
                            std::string_view expected)
        {
            return std::find(values.begin(), values.end(), expected) !=
                values.end();
        };
        if (!has(specification.requiredConfiguredIdentityFields,
                 "feature_ablation_mask") ||
            !has(specification.requiredConfiguredIdentityFields,
                 "fresh_initialization_seed"))
            throw std::invalid_argument("study_specification_configured_contract_incomplete");
        if (!has(specification.requiredExecutionProvenanceFields,
                 "training_execution_identity") ||
            !has(specification.requiredExecutionProvenanceFields,
                 "producer_worker_attempt_id"))
            throw std::invalid_argument("study_specification_provenance_contract_incomplete");
        const std::set<std::string> knownDimensions{"symbol", "prediction_horizon"};
        for (const auto& dimension : specification.allowedContextDimensions)
            if (!knownDimensions.contains(dimension))
                throw std::invalid_argument("study_specification_undeclared_context_dimension");
        if (specification.contexts.size() < 2)
            throw std::invalid_argument("study_specification_requires_two_contexts");

        std::set<long long> experimentIds;
        std::set<std::pair<long long, long long>> pairs;
        std::set<std::string> ordinals;
        for (const auto& context : specification.contexts)
        {
            if (!ordinals.insert(context.ordinal).second)
                throw std::invalid_argument("study_specification_duplicate_context");
            if (context.pairs.empty())
                throw std::invalid_argument("study_specification_empty_context");
            std::set<std::string> dimensions;
            for (const auto& [key, value] : context.dimensions)
            {
                if (value.empty() || !dimensions.insert(key).second ||
                    !knownDimensions.contains(key))
                    throw std::invalid_argument("study_specification_invalid_context_dimension");
            }
            for (const auto& key : specification.allowedContextDimensions)
                if (!dimensions.contains(key))
                    throw std::invalid_argument("study_specification_context_dimension_missing");
            for (const auto& key : knownDimensions)
                if (!dimensions.contains(key))
                    throw std::invalid_argument("study_specification_context_dimension_missing");
            std::set<unsigned int> seeds;
            for (const auto& pair : context.pairs)
            {
                if (pair.armA <= 0 || pair.armB <= 0 || pair.armA == pair.armB)
                    throw std::invalid_argument("study_specification_invalid_pair");
                if (!experimentIds.insert(pair.armA).second ||
                    !experimentIds.insert(pair.armB).second)
                    throw std::invalid_argument("study_specification_duplicate_experiment_id");
                const auto normalized = std::minmax(pair.armA, pair.armB);
                if (!pairs.insert(normalized).second)
                    throw std::invalid_argument("study_specification_pair_reuse");
                if (!seeds.insert(pair.requestedSeed).second)
                    throw std::invalid_argument("study_specification_duplicate_seed");
            }
        }
        if (specification.identityHash.empty())
            throw std::invalid_argument("study_specification_missing_identity_hash");
        const std::string expected = IdentityHash(specification);
        if (specification.identityHash != expected)
            throw std::invalid_argument("study_specification_identity_hash_mismatch");
        result.valid = true;
        result.reason = "NONE";
        result.canonical = Canonical(specification);
        result.identityHash = expected;
        return result;
    }
    catch (const std::exception& error)
    {
        result.valid = false;
        result.reason = error.what();
        return result;
    }
}

ValidationResult ValidateArtifact(const std::string& artifact)
{
    try
    {
        const Specification specification = Parse(artifact);
        return Validate(specification);
    }
    catch (const std::exception& error)
    {
        ValidationResult result;
        result.reason = error.what();
        return result;
    }
}

EA::ExperimentReplicationComparison::FamilyComparisonCommand
MakeFamilyComparisonCommand(const Specification& specification)
{
    const ValidationResult validation = Validate(specification);
    if (!validation.valid)
        throw std::invalid_argument(validation.reason);
    EA::ExperimentReplicationComparison::FamilyComparisonCommand command;
    for (const auto& context : specification.contexts)
    {
        std::vector<std::pair<long long, long long>> family;
        for (const auto& pair : context.pairs)
            family.emplace_back(pair.armA, pair.armB);
        command.families.push_back(std::move(family));
    }
    return command;
}

} // namespace EA::ControlledReplicationStudy
