#pragma once

// Offline Phase Pocket 3 confirmation freeze.  This is configuration and
// identity only: it contains no source adapter, detector replay, or outcome
// evaluation path.

#include "PocketProspectiveEvaluator.hpp"

namespace EA::Pocket::Prospective::Confirmation
{
inline constexpr std::string_view kStudyId = "pocket-prospective-confirmation-2025-v1";
inline constexpr std::string_view kConfigurationSchema =
    "phase-pocket-4-canonical-confirmation-run-configuration-v1";
inline constexpr std::string_view kPrimaryArtifactSchema =
    "phase-pocket-4-prospective-confirmation-artifact-v1";
inline constexpr std::string_view kDerivedReportSchema =
    "phase-pocket-4-derived-confirmation-report-v1";
inline constexpr std::int64_t kScoringStart = 1735689600; // 2025-01-01T00:00:00Z
inline constexpr std::int64_t kScoringEnd = 1767225600; // 2026-01-01T00:00:00Z
inline constexpr std::int64_t kResolutionEnd = 1767283200; // 2026-01-01T16:00:00Z

struct Configuration final
{
    std::string study, schema, configurationSha256;
    Partition scoringPartition;
    std::int64_t resolutionEnd = 0;
    std::array<std::size_t, 3> lookbacks{};
    std::array<std::size_t, 3> horizons{};
    std::array<SymbolSource, 6> symbols{};
};

inline std::vector<std::pair<std::string, std::string>> Fields()
{
    std::vector<std::pair<std::string, std::string>> fields{
        {"aggregation", "event_weighted_and_equal_symbol_separate"},
        {"bootstrap_block", "utc_calendar_week_confirmation_blocks"},
        {"bootstrap_confidence", "0.95"},
        {"bootstrap_replicates", "2000"},
        {"cadence_seconds", "900"},
        {"censoring", "right_censor_first_unusable_bar_boundary_tail_gap_invalid"},
        {"configuration_schema", std::string(kConfigurationSchema)},
        {"derived_report_schema", std::string(kDerivedReportSchema)},
        {"detector", std::string(::EA::Pocket::Prospective::kDetectorId)},
        {"detector_baseline", "41800f4"},
        {"horizons", "4,16,64"},
        {"lookbacks", "10,15,20"},
        {"ordering", "symbol,lookback,confirmation_timestamp,observation_identity"},
        {"outcomes", "inclusive_touch_and_close;bounded_continuation_and_race"},
        {"partitions", "confirmation:1735689600:1767225600"},
        {"primary_artifact_schema", std::string(kPrimaryArtifactSchema)},
        {"protocol", std::string(::EA::Pocket::Prospective::kProtocolId)},
        {"protocol_document_sha256", std::string(::EA::Pocket::Prospective::kProtocolDocumentSha256)},
        {"resolution_end", "1767283200"},
        {"source_adapter", "postgresql-candlestick-canonical-absolute-half-open-v1"},
        {"source_identity", "canonical_15m_completed_candlestick"},
        {"study", std::string(kStudyId)},
        {"symbols", "AUDCAD:audcadrmp:0.0001,AUDUSD:audusdrmp:0.0001,EURUSD:eurusdrmp:0.0001,GBPUSD:gbpusdrmp:0.0001,USDCAD:usdcadrmp:0.0001,USDJPY:usdjpyrmp:0.01"},
        {"timeframe", "15m_completed"},
        {"warmup_bars", "21"},
    };
    std::sort(fields.begin(), fields.end());
    return fields;
}
inline std::string CanonicalPayload()
{
    std::ostringstream output;
    for (const auto& [key, value] : Fields()) output << key << '=' << value << '\n';
    return output.str();
}
inline std::string CanonicalConfigurationText()
{ return CanonicalPayload() + "configuration_sha256=" + Sha256(CanonicalPayload()) + '\n'; }
inline Configuration FrozenConfiguration()
{
    Configuration output;
    output.study=std::string(kStudyId); output.schema=std::string(kConfigurationSchema);
    output.configurationSha256=Sha256(CanonicalPayload());
    output.scoringPartition={"confirmation", kScoringStart, kScoringEnd}; output.resolutionEnd=kResolutionEnd;
    output.lookbacks=::EA::Pocket::Prospective::kLookbacks; output.horizons=::EA::Pocket::Prospective::kHorizons;
    output.symbols=::EA::Pocket::Prospective::FrozenConfiguration().symbols;
    return output;
}
inline RunConfiguration EvaluatorConfiguration()
{
    const Configuration frozen=FrozenConfiguration();
    RunConfiguration output=::EA::Pocket::Prospective::FrozenConfiguration();
    output.schema=frozen.schema; output.study=frozen.study;
    output.configurationSha256=frozen.configurationSha256;
    output.partitions={{{"confirmation",kScoringStart,kScoringEnd},{"",0,0},{"",0,0}}};
    output.resolutionEnd=kResolutionEnd;
    return output;
}
inline void VerifyPrimaryArtifact(const std::filesystem::path& directory)
{
    ImmutableArtifactWriter::VerifyDirectoryContract(directory, EvaluatorConfiguration(),
        CanonicalConfigurationText(), kPrimaryArtifactSchema);
}
inline Configuration LoadAndValidateConfiguration(const std::filesystem::path& path)
{
    if (ReadTextFile(path) != CanonicalConfigurationText())
        throw std::invalid_argument("POCKET_CONFIRMATION_CONFIGURATION_NONCANONICAL_OR_UNKNOWN_FIELD");
    const Configuration configuration=FrozenConfiguration();
    if (configuration.study != kStudyId || configuration.scoringPartition.start != kScoringStart ||
        configuration.scoringPartition.end != kScoringEnd || configuration.resolutionEnd != kResolutionEnd ||
        configuration.lookbacks != ::EA::Pocket::Prospective::kLookbacks ||
        configuration.horizons != ::EA::Pocket::Prospective::kHorizons ||
        configuration.symbols.size() != 6 || !IsHexSha256(configuration.configurationSha256))
        throw std::invalid_argument("POCKET_CONFIRMATION_CONFIGURATION_FROZEN_CONTRACT_MISMATCH");
    return configuration;
}
inline void ValidateAbsentOutputTarget(const std::filesystem::path& output)
{
    if (output.empty() || std::filesystem::exists(output))
        throw std::invalid_argument("POCKET_CONFIRMATION_OUTPUT_TARGET_EXISTS_OR_EMPTY");
}
inline std::string ValidationSummary(const Configuration& configuration)
{
    return "POCKET_CONFIRMATION_CONFIGURATION_VALID study="+configuration.study+
        ",configuration_sha256="+configuration.configurationSha256+
        ",scoring_start=2025-01-01T00:00:00Z,scoring_end_exclusive=2026-01-01T00:00:00Z"+
        ",resolution_end_exclusive=2026-01-01T16:00:00Z,primary_schema="+std::string(kPrimaryArtifactSchema)+
        ",derived_schema="+std::string(kDerivedReportSchema)+",offline=true\n";
}
} // namespace EA::Pocket::Prospective::Confirmation
