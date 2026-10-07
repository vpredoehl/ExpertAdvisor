#include "FeatureAblationFactorialComparison.hpp"

#include "FeatureAblation.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::FeatureAblationFactorialComparison
{
namespace
{

namespace Pair = ExperimentPairComparison;

constexpr std::array<std::string_view, 4> kCellNames {
    "a_on_b_on", "a_off_b_on", "a_on_b_off", "a_off_b_off"};

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

std::string Number(double value)
{
    char buffer[128] {};
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
        throw std::runtime_error("feature_ablation_factorial_number_format_failed");
    return std::string(buffer, converted.ptr);
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

std::string Reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index != 0) output << '|';
        output << MachineText(reasons[index]);
    }
    return output.str();
}

std::vector<std::string_view> Split(std::string_view text, char separator)
{
    std::vector<std::string_view> result;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find(separator, begin);
        result.push_back(text.substr(
            begin, end == std::string_view::npos
                ? std::string_view::npos : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return result;
}

long long ParseId(std::string_view text)
{
    long long value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                        value);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || value <= 0)
        throw std::invalid_argument("feature_ablation_factorial_experiment_id_invalid");
    return value;
}

void ValidateDesign(Design& design)
{
    if (design.factorALabel.empty() || design.factorBLabel.empty())
        throw std::invalid_argument("feature_ablation_factorial_factor_label_empty");
    if (design.factorALabel == design.factorBLabel)
        throw std::invalid_argument("feature_ablation_factorial_factor_labels_equal");
    std::set<std::string> masks;
    for (std::string& mask : design.expectedMasks)
    {
        mask = FeatureAblationMask::CanonicalizeRequestedExpression(mask);
        if (!masks.insert(mask).second)
            throw std::invalid_argument("feature_ablation_factorial_masks_not_distinct");
    }
}

void ValidateSeedCells(const std::vector<SeedCells>& seeds)
{
    if (seeds.empty())
        throw std::invalid_argument("feature_ablation_factorial_seed_cells_empty");
    std::set<long long> allIds;
    for (const SeedCells& cells : seeds)
    {
        std::set<long long> perSeed;
        for (const long long id : cells.experimentIds)
        {
            if (id <= 0)
                throw std::invalid_argument("feature_ablation_factorial_experiment_id_invalid");
            if (!perSeed.insert(id).second)
                throw std::invalid_argument("feature_ablation_factorial_seed_cell_ids_not_distinct");
            if (!allIds.insert(id).second)
                throw std::invalid_argument("feature_ablation_factorial_experiment_id_reused");
        }
    }
}

std::optional<std::string> IdentityValue(
    const Pair::ArmResultSet& arm, std::string_view name)
{
    const auto found = std::find_if(
        arm.scientificIdentity.begin(), arm.scientificIdentity.end(),
        [name](const Pair::IdentityField& field) { return field.name == name; });
    return found == arm.scientificIdentity.end() ? std::nullopt : found->value;
}

bool RequiredFinalEvidence(const Pair::ComparisonResult& comparison)
{
    return comparison.armAFinalModelAvailable &&
        comparison.armATrainingProvenanceAvailable &&
        comparison.armAFinalInferenceAvailable &&
        comparison.armAInferenceProvenanceAvailable &&
        comparison.armAFinalAnalysisAvailable &&
        comparison.armAProfitabilityObservationAvailable &&
        comparison.armBFinalModelAvailable &&
        comparison.armBTrainingProvenanceAvailable &&
        comparison.armBFinalInferenceAvailable &&
        comparison.armBInferenceProvenanceAvailable &&
        comparison.armBFinalAnalysisAvailable &&
        comparison.armBProfitabilityObservationAvailable;
}

Pair::Request FactorialRequest(std::string armALabel, std::string armBLabel)
{
    Pair::Request request;
    request.armALabel = std::move(armALabel);
    request.armBLabel = std::move(armBLabel);
    request.intentionalDifferenceFields = {"feature_ablation_mask"};
    return request;
}

std::optional<double> ArmAValue(const Pair::ComparisonResult& comparison,
                                const Pair::MetricDefinition& definition)
{
    return (comparison.*definition.member).armA;
}

std::optional<double> ArmBValue(const Pair::ComparisonResult& comparison,
                                const Pair::MetricDefinition& definition)
{
    return (comparison.*definition.member).armB;
}

bool IdentityEquivalentExceptSeed(const Pair::ArmResultSet& reference,
                                 const Pair::ArmResultSet& candidate)
{
    std::map<std::string, std::optional<std::string>> referenceMap;
    std::map<std::string, std::optional<std::string>> candidateMap;
    for (const auto& field : reference.scientificIdentity)
        referenceMap.emplace(field.name, field.value);
    for (const auto& field : candidate.scientificIdentity)
        candidateMap.emplace(field.name, field.value);
    referenceMap.erase("fresh_initialization_seed");
    candidateMap.erase("fresh_initialization_seed");
    return referenceMap == candidateMap;
}

void AddReason(std::vector<std::string>& reasons, std::string reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(std::move(reason));
}

} // namespace

Command ParseCommand(std::string_view text)
{
    const std::size_t at = text.find('@');
    if (text.empty() || at == std::string_view::npos ||
        at != text.rfind('@'))
        throw std::invalid_argument("feature_ablation_factorial_argument_invalid");
    const std::string_view members = text.substr(0, at);
    const std::string_view masks = text.substr(at + 1);
    const auto maskParts = Split(masks, '/');
    if (maskParts.size() != 4)
        throw std::invalid_argument("feature_ablation_factorial_requires_four_masks");

    Command command;
    for (std::size_t index = 0; index < 4; ++index)
        command.design.expectedMasks[index] = std::string(maskParts[index]);
    ValidateDesign(command.design);

    for (const std::string_view seedText : Split(members, ';'))
    {
        const auto ids = Split(seedText, ':');
        if (ids.size() != 4)
            throw std::invalid_argument("feature_ablation_factorial_requires_four_cells_per_seed");
        SeedCells cells;
        for (std::size_t index = 0; index < 4; ++index)
            cells.experimentIds[index] = ParseId(ids[index]);
        command.seedCells.push_back(cells);
    }
    ValidateSeedCells(command.seedCells);
    return command;
}

Report Evaluate(const Command& command, const Pair::EvidenceSource& source)
{
    Command validated = command;
    ValidateDesign(validated.design);
    ValidateSeedCells(validated.seedCells);

    Report report;
    report.design = validated.design;
    report.declaredSeedCount = validated.seedCells.size();
    std::array<Pair::ArmResultSet, 4> crossSeedReference;
    bool haveCrossSeedReference = false;

    for (std::size_t ordinal = 0; ordinal < validated.seedCells.size(); ++ordinal)
    {
        const SeedCells& cells = validated.seedCells[ordinal];
        SeedResult seed;
        seed.ordinal = ordinal + 1;
        seed.experimentIds = cells.experimentIds;
        std::array<Pair::ArmResultSet, 4> arms;
        for (std::size_t index = 0; index < 4; ++index)
        {
            const auto evidence = source.Load(cells.experimentIds[index]);
            const std::string actual = FeatureAblationMask::
                CanonicalizeRequestedExpression(
                    evidence.authoritative.configuration.featureAblationMask);
            if (actual != validated.design.expectedMasks[index])
                AddReason(seed.reasons, "declared_feature_ablation_mask_mismatch:" +
                    std::string(kCellNames[index]));
            arms[index] = Pair::MakeArmResultSet(evidence);
        }

        const auto seed11 = IdentityValue(arms[0], "fresh_initialization_seed");
        seed.freshInitializationSeed = seed11;
        if (!seed11) AddReason(seed.reasons, "fresh_initialization_seed_unavailable");
        for (std::size_t index = 1; index < 4; ++index)
            if (!seed11 || IdentityValue(arms[index], "fresh_initialization_seed") != seed11)
                AddReason(seed.reasons, "fresh_initialization_seed_mismatch");

        for (std::size_t index = 1; index < 4; ++index)
        {
            seed.comparisons[index - 1] = Pair::Compare(
                arms[0], arms[index], FactorialRequest(
                    std::string(kCellNames[0]), std::string(kCellNames[index])));
            const auto& comparison = seed.comparisons[index - 1];
            if (!comparison.invalidReasons.empty())
                AddReason(seed.reasons, "pair_invalid_evidence:" +
                    std::string(kCellNames[index]));
            if (!comparison.unexpectedDifferences.empty())
                AddReason(seed.reasons, "pair_unexpected_scientific_identity_difference:" +
                    std::string(kCellNames[index]));
            if (!RequiredFinalEvidence(comparison))
                AddReason(seed.reasons, "completed_evidence_unavailable:" +
                    std::string(kCellNames[index]));
        }

        if (!haveCrossSeedReference)
        {
            crossSeedReference = arms;
            haveCrossSeedReference = true;
        }
        else for (std::size_t index = 0; index < 4; ++index)
            if (!IdentityEquivalentExceptSeed(crossSeedReference[index], arms[index]))
                AddReason(seed.reasons, "cross_seed_scientific_identity_mismatch:" +
                    std::string(kCellNames[index]));

        seed.eligible = seed.reasons.empty();
        if (seed.eligible)
        {
            ++report.eligibleSeedCount;
            for (std::size_t metricIndex = 0;
                 metricIndex < Pair::MetricDefinitions().size(); ++metricIndex)
            {
                const auto& definition = Pair::MetricDefinitions()[metricIndex];
                const auto& against01 = seed.comparisons[0];
                const auto& against10 = seed.comparisons[1];
                const auto& against00 = seed.comparisons[2];
                seed.cell11Metrics[metricIndex] = ArmAValue(against01, definition);
                seed.cell01Metrics[metricIndex] = ArmBValue(against01, definition);
                seed.cell10Metrics[metricIndex] = ArmBValue(against10, definition);
                seed.cell00Metrics[metricIndex] = ArmBValue(against00, definition);
                const auto& y11 = seed.cell11Metrics[metricIndex];
                const auto& y01 = seed.cell01Metrics[metricIndex];
                const auto& y10 = seed.cell10Metrics[metricIndex];
                const auto& y00 = seed.cell00Metrics[metricIndex];
                if (y11 && y01 && y10 && y00)
                {
                    seed.effects[metricIndex] = {
                        ((*y11 - *y01) + (*y10 - *y00)) / 2.0,
                        ((*y11 - *y10) + (*y01 - *y00)) / 2.0,
                        *y11 - *y01 - *y10 + *y00};
                }
            }
        }
        report.seeds.push_back(std::move(seed));
    }

    for (std::size_t metricIndex = 0;
         metricIndex < Pair::MetricDefinitions().size(); ++metricIndex)
    {
        MetricSummary summary;
        summary.name = std::string(Pair::MetricDefinitions()[metricIndex].name);
        summary.eligibleSeedCount = report.eligibleSeedCount;
        double sumA = 0.0, sumB = 0.0, sumInteraction = 0.0;
        for (const SeedResult& seed : report.seeds)
        {
            if (!seed.eligible || !seed.effects[metricIndex].factorAMainEffect)
                continue;
            ++summary.availableSeedCount;
            sumA += *seed.effects[metricIndex].factorAMainEffect;
            sumB += *seed.effects[metricIndex].factorBMainEffect;
            sumInteraction += *seed.effects[metricIndex].interaction;
        }
        if (summary.availableSeedCount != 0)
        {
            const double divisor = static_cast<double>(summary.availableSeedCount);
            summary.descriptiveFactorAMainEffectMean = sumA / divisor;
            summary.descriptiveFactorBMainEffectMean = sumB / divisor;
            summary.descriptiveInteractionMean = sumInteraction / divisor;
        }
        report.metrics.push_back(std::move(summary));
    }
    return report;
}

std::string Render(const Report& report)
{
    std::ostringstream output;
    output << "FEATURE_ABLATION_FACTORIAL_COMPARISON"
           << ",version=1"
           << ",factor_a=" << MachineText(report.design.factorALabel)
           << ",factor_b=" << MachineText(report.design.factorBLabel)
           << ",declared_seed_count=" << report.declaredSeedCount
           << ",eligible_seed_count=" << report.eligibleSeedCount
           << ",effect_convention=on_minus_off"
           << ",statistical_significance=NOT_INFERRED"
           << ",subjective_winner=NONE\n";
    for (std::size_t index = 0; index < 4; ++index)
        output << "FEATURE_ABLATION_FACTORIAL_DECLARED_MASK"
               << ",cell=" << kCellNames[index]
               << ",value=" << std::quoted(report.design.expectedMasks[index]) << '\n';
    for (const SeedResult& seed : report.seeds)
    {
        output << "FEATURE_ABLATION_FACTORIAL_SEED"
               << ",ordinal=" << seed.ordinal
               << ",fresh_initialization_seed="
               << (seed.freshInitializationSeed ? *seed.freshInitializationSeed : "NULL")
               << ",eligible=" << (seed.eligible ? "true" : "false")
               << ",reasons=" << Reasons(seed.reasons);
        for (std::size_t index = 0; index < 4; ++index)
            output << "," << kCellNames[index] << "_experiment_id="
                   << seed.experimentIds[index];
        output << '\n';
        for (std::size_t metricIndex = 0;
             metricIndex < Pair::MetricDefinitions().size(); ++metricIndex)
        {
            const auto& definition = Pair::MetricDefinitions()[metricIndex];
            const Effect& effect = seed.effects[metricIndex];
            output << "FEATURE_ABLATION_FACTORIAL_SEED_METRIC"
                   << ",ordinal=" << seed.ordinal
                   << ",metric=" << definition.name
                   << ",y11=" << OptionalNumber(seed.cell11Metrics[metricIndex])
                   << ",y01=" << OptionalNumber(seed.cell01Metrics[metricIndex])
                   << ",y10=" << OptionalNumber(seed.cell10Metrics[metricIndex])
                   << ",y00=" << OptionalNumber(seed.cell00Metrics[metricIndex])
                   << ",factor_a_main_effect="
                   << OptionalNumber(effect.factorAMainEffect)
                   << ",factor_b_main_effect="
                   << OptionalNumber(effect.factorBMainEffect)
                   << ",interaction=" << OptionalNumber(effect.interaction)
                   << '\n';
        }
    }
    for (const MetricSummary& metric : report.metrics)
        output << "FEATURE_ABLATION_FACTORIAL_METRIC"
               << ",metric=" << metric.name
               << ",eligible_seed_count=" << metric.eligibleSeedCount
               << ",available_seed_count=" << metric.availableSeedCount
               << ",descriptive_factor_a_main_effect_mean="
               << OptionalNumber(metric.descriptiveFactorAMainEffectMean)
               << ",descriptive_factor_b_main_effect_mean="
               << OptionalNumber(metric.descriptiveFactorBMainEffectMean)
               << ",descriptive_interaction_mean="
               << OptionalNumber(metric.descriptiveInteractionMean) << '\n';
    return output.str();
}

int RunCommand(const Command& command, const Pair::EvidenceSource& source,
               std::ostream& output, std::ostream& errors)
{
    try
    {
        output << Render(Evaluate(command, source));
        return 0;
    }
    catch (const Pair::EvidenceUnavailableError& error)
    {
        errors << "FEATURE_ABLATION_FACTORIAL_COMPARISON_LOAD_FAILED"
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "FEATURE_ABLATION_FACTORIAL_COMPARISON_LOAD_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::FeatureAblationFactorialComparison
