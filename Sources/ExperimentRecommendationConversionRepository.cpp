#include "ExperimentRecommendationConversionRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

void ValidateProposal(const ProposedExperimentSpecification& proposal);

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    const pqxx::field field = row[column];
    if (field.is_null()) return std::nullopt;
    return field.as<Value>();
}

bool IsAsciiBlank(const std::string& value)
{
    for (const unsigned char character : value)
    {
        if (character != ' ' && character != '\t' && character != '\n' &&
            character != '\r' && character != '\f' && character != '\v')
            return false;
    }
    return true;
}

bool ValidBoundedText(const std::string& value, std::size_t maximumBytes)
{
    return !value.empty() && !IsAsciiBlank(value) &&
           value.size() <= maximumBytes;
}

RecommendationMutationParameter ParsePersistedMutationParameter(
    const std::string& value)
{
    if (value == kCoreLrMult)
        return RecommendationMutationParameter::coreLrMult;
    if (value == kHeadLrMult)
        return RecommendationMutationParameter::headLrMult;
    if (value == kLabelThreshold)
        return RecommendationMutationParameter::labelThreshold;
    if (value == kPredictionHorizon)
        return RecommendationMutationParameter::predictionHorizon;
    throw std::runtime_error(
        "invalid_persisted_recommendation_conversion_mutation_parameter");
}

std::string ProposalColumns()
{
    return
        "recommendation_conversion_proposal_id,recommendation_id,"
        "source_experiment_id,conversion_contract_version,changed_parameter,"
        "source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "ranking_snapshot_identity_hash,proposed_symbol,"
        "proposed_prediction_horizon,proposed_label_threshold,"
        "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
        "proposed_train_start_date::text AS proposed_train_start_date,"
        "proposed_train_end_date::text AS proposed_train_end_date,"
        "proposed_infer_start_date::text AS proposed_infer_start_date,"
        "proposed_infer_end_date::text AS proposed_infer_end_date,"
        "proposed_checkpoint_interval,proposed_resume_model_id,"
        "source_invocation_canonical,proposed_invocation_canonical,"
        "conversion_identity_canonical,conversion_identity_hash,"
        "conversion_hash_collision_ordinal,"
        "created_at::text AS created_at";
}

PersistedRecommendationConversionProposal MapProposal(const pqxx::row& row)
{
    PersistedRecommendationConversionProposal persisted;
    persisted.proposalId =
        row["recommendation_conversion_proposal_id"].as<long long>();
    persisted.conversionContractVersion =
        row["conversion_contract_version"].as<int>();
    persisted.hashCollisionOrdinal =
        row["conversion_hash_collision_ordinal"].as<int>();
    ProposedExperimentSpecification& proposal = persisted.proposal;
    proposal.recommendationId = row["recommendation_id"].as<long long>();
    proposal.sourceExperimentId = row["source_experiment_id"].as<long long>();
    proposal.changedParameter = ParsePersistedMutationParameter(
        row["changed_parameter"].as<std::string>());
    proposal.sourceValueCanonical =
        row["source_value_canonical"].as<std::string>();
    proposal.proposedValueCanonical =
        row["proposed_value_canonical"].as<std::string>();
    proposal.recommendationSemanticHash =
        row["recommendation_semantic_hash"].as<std::string>();
    proposal.evaluationIdentityHash =
        row["evaluation_identity_hash"].as<std::string>();
    proposal.evaluationPolicyHash =
        row["evaluation_policy_hash"].as<std::string>();
    proposal.scoringPolicyHash =
        row["scoring_policy_hash"].as<std::string>();
    proposal.reviewAuthorizationHash =
        row["review_authorization_hash"].as<std::string>();
    proposal.rankingSnapshotIdentityHash =
        OptionalValue<std::string>(row, "ranking_snapshot_identity_hash");
    proposal.proposedInvocation.configuration.symbol =
        row["proposed_symbol"].as<std::string>();
    proposal.proposedInvocation.configuration.predictionHorizon =
        row["proposed_prediction_horizon"].as<int>();
    proposal.proposedInvocation.configuration.labelThreshold =
        row["proposed_label_threshold"].as<double>();
    proposal.proposedInvocation.configuration.coreLrMult =
        OptionalValue<double>(row, "proposed_core_lr_mult");
    proposal.proposedInvocation.configuration.headLrMult =
        OptionalValue<double>(row, "proposed_head_lr_mult");
    proposal.proposedInvocation.configuration.targetEpochs =
        row["proposed_target_epochs"].as<int>();
    proposal.proposedInvocation.configuration.trainStartDate =
        row["proposed_train_start_date"].as<std::string>();
    proposal.proposedInvocation.configuration.trainEndDate =
        row["proposed_train_end_date"].as<std::string>();
    proposal.proposedInvocation.configuration.inferStartDate =
        OptionalValue<std::string>(row, "proposed_infer_start_date");
    proposal.proposedInvocation.configuration.inferEndDate =
        OptionalValue<std::string>(row, "proposed_infer_end_date");
    proposal.proposedInvocation.checkpointInterval =
        row["proposed_checkpoint_interval"].as<int>();
    proposal.proposedInvocation.resumeModelId =
        OptionalValue<long long>(row, "proposed_resume_model_id");
    proposal.sourceInvocationCanonical =
        row["source_invocation_canonical"].as<std::string>();
    proposal.proposedInvocationCanonical =
        row["proposed_invocation_canonical"].as<std::string>();
    proposal.conversionIdentityCanonical =
        row["conversion_identity_canonical"].as<std::string>();
    proposal.conversionIdentityHash =
        row["conversion_identity_hash"].as<std::string>();
    persisted.createdAt = row["created_at"].as<std::string>();
    if (persisted.conversionContractVersion !=
        kRecommendationConversionContractVersion)
        throw std::runtime_error(
            "unsupported_persisted_recommendation_conversion_contract");
    try
    {
        ValidateProposal(proposal);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "invalid_persisted_recommendation_conversion_proposal");
    }
    return persisted;
}

void ValidateListArguments(long long identifier, int limit)
{
    if (identifier <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_identifier_must_be_positive");
    if (limit <= 0 || limit > kMaximumRecommendationConversionProposalListLimit)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_list_limit_out_of_range");
}

void ValidateProposal(const ProposedExperimentSpecification& proposal)
{
    if (proposal.recommendationId <= 0 || proposal.sourceExperimentId <= 0)
        throw std::invalid_argument(
            "invalid_recommendation_conversion_proposal_identifier");
    if (!ValidBoundedText(proposal.sourceValueCanonical, 128) ||
        !ValidBoundedText(proposal.proposedValueCanonical, 128) ||
        proposal.sourceValueCanonical == proposal.proposedValueCanonical ||
        !ValidBoundedText(proposal.recommendationSemanticHash, 256) ||
        !ValidBoundedText(proposal.evaluationIdentityHash, 256) ||
        !ValidBoundedText(proposal.evaluationPolicyHash, 256) ||
        !ValidBoundedText(proposal.scoringPolicyHash, 256) ||
        !ValidBoundedText(proposal.reviewAuthorizationHash, 256) ||
        !ValidBoundedText(proposal.sourceInvocationCanonical, 1048576) ||
        !ValidBoundedText(proposal.proposedInvocationCanonical, 1048576) ||
        !ValidBoundedText(proposal.conversionIdentityCanonical, 1048576) ||
        !ValidBoundedText(proposal.conversionIdentityHash, 256) ||
        (proposal.rankingSnapshotIdentityHash &&
         !ValidBoundedText(*proposal.rankingSnapshotIdentityHash, 256)))
        throw std::invalid_argument(
            "invalid_recommendation_conversion_proposal_text");

    const RecommendationInvocationIdentity proposedIdentity =
        BuildRecommendationInvocationIdentity(proposal.proposedInvocation);
    if (!ValidBoundedText(
            proposedIdentity.invocation.configuration.symbol, 64) ||
        proposal.proposedInvocationCanonical != proposedIdentity.canonicalText ||
        proposal.recommendationSemanticHash !=
            BuildRecommendationCandidateIdentity(
                proposedIdentity.invocation.configuration).hash ||
        proposal.conversionIdentityHash !=
            RecommendationCanonicalHash(proposal.conversionIdentityCanonical))
        throw std::invalid_argument(
            "invalid_recommendation_conversion_proposal_identity");

    const std::string sourceField =
        ";source_invocation=" +
        std::to_string(proposal.sourceInvocationCanonical.size()) + ":" +
        proposal.sourceInvocationCanonical;
    const std::string proposedField =
        ";proposed_invocation=" +
        std::to_string(proposal.proposedInvocationCanonical.size()) + ":" +
        proposal.proposedInvocationCanonical;
    if (!proposal.conversionIdentityCanonical.ends_with(
            sourceField + proposedField))
        throw std::invalid_argument(
            "invalid_recommendation_conversion_proposal_canonical_snapshot");
}

std::optional<PersistedRecommendationConversionProposal> FindByCanonical(
    pqxx::transaction_base& transaction,
    const std::string& canonical)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_conversion_proposal "
        "WHERE conversion_identity_canonical=$1 LIMIT 2;",
        pqxx::params{canonical});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "duplicate_persisted_recommendation_conversion_canonical");
    return MapProposal(rows.front());
}

bool HasHashCollision(pqxx::transaction_base& transaction,
                      const ProposedExperimentSpecification& proposal)
{
    return !transaction.exec(
        "SELECT 1 FROM experiment_recommendation_conversion_proposal "
        "WHERE conversion_identity_hash=$1 "
        "AND conversion_identity_canonical<>$2 LIMIT 1;",
        pqxx::params{proposal.conversionIdentityHash,
                     proposal.conversionIdentityCanonical}).empty();
}

bool PersistedMatches(
    const PersistedRecommendationConversionProposal& persisted,
    const ProposedExperimentSpecification& proposal)
{
    const ProposedExperimentSpecification& value = persisted.proposal;
    return persisted.conversionContractVersion ==
               kRecommendationConversionContractVersion &&
           value.sourceExperimentId == proposal.sourceExperimentId &&
           value.recommendationId == proposal.recommendationId &&
           value.proposedInvocationCanonical ==
               proposal.proposedInvocationCanonical &&
           value.sourceInvocationCanonical == proposal.sourceInvocationCanonical &&
           value.changedParameter == proposal.changedParameter &&
           value.sourceValueCanonical == proposal.sourceValueCanonical &&
           value.proposedValueCanonical == proposal.proposedValueCanonical &&
           value.recommendationSemanticHash ==
               proposal.recommendationSemanticHash &&
           value.evaluationIdentityHash == proposal.evaluationIdentityHash &&
           value.evaluationPolicyHash == proposal.evaluationPolicyHash &&
           value.scoringPolicyHash == proposal.scoringPolicyHash &&
           value.reviewAuthorizationHash == proposal.reviewAuthorizationHash &&
           value.conversionIdentityCanonical ==
               proposal.conversionIdentityCanonical &&
           value.conversionIdentityHash == proposal.conversionIdentityHash;
}

RecommendationConversionProposalPersistResult ExistingResult(
    PersistedRecommendationConversionProposal persisted,
    const ProposedExperimentSpecification& proposal)
{
    // Ranking is advisory snapshot metadata and is deliberately ignored for
    // exact conversion identity. All identity-bearing persisted values must
    // still agree before an existing row may be returned.
    if (!PersistedMatches(persisted, proposal))
        throw std::runtime_error(
            "inconsistent_persisted_recommendation_conversion_identity");
    return {RecommendationConversionProposalPersistOutcome::existingIdentical,
            std::move(persisted)};
}

std::vector<PersistedRecommendationConversionProposal> MapProposals(
    const pqxx::result& rows)
{
    std::vector<PersistedRecommendationConversionProposal> proposals;
    proposals.reserve(rows.size());
    for (const pqxx::row& row : rows) proposals.push_back(MapProposal(row));
    return proposals;
}

} // namespace

std::string RecommendationConversionProposalPersistOutcomeText(
    RecommendationConversionProposalPersistOutcome value)
{
    switch (value)
    {
        case RecommendationConversionProposalPersistOutcome::created:
            return "created";
        case RecommendationConversionProposalPersistOutcome::existingIdentical:
            return "existing_identical";
        case RecommendationConversionProposalPersistOutcome::createdWithHashCollision:
            return "created_with_hash_collision";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_proposal_persist_outcome");
}

bool RecommendationConversionProposalSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT to_regclass('experiment_recommendation_conversion_proposal') "
        "IS NOT NULL;").one_row()[0].as<bool>();
}

RecommendationConversionProposalPersistResult
PersistRecommendationConversionProposal(
    pqxx::connection& connection,
    const ProposedExperimentSpecification& proposal)
{
    ValidateProposal(proposal);
    try
    {
        pqxx::work transaction{connection};
        // Serialize only proposals sharing the accelerator hash. This makes
        // collision reporting and same-identity retries deterministic without
        // locking recommendation, experiment, or scheduler-owned rows.
        transaction.exec(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, "
            "7046029254386353131));",
            pqxx::params{proposal.conversionIdentityHash});
        if (auto existing = FindByCanonical(
                transaction, proposal.conversionIdentityCanonical))
        {
            transaction.commit();
            return ExistingResult(std::move(*existing), proposal);
        }
        const bool hashCollision = HasHashCollision(transaction, proposal);
        const int collisionOrdinal = transaction.exec(
            "SELECT coalesce(max(conversion_hash_collision_ordinal),-1)+1 "
            "FROM experiment_recommendation_conversion_proposal WHERE "
            "conversion_identity_hash=$1;",
            pqxx::params{proposal.conversionIdentityHash})
            .one_row()[0].as<int>();
        const auto& invocation = proposal.proposedInvocation;
        const pqxx::row row = transaction.exec(
            "INSERT INTO experiment_recommendation_conversion_proposal ("
            "recommendation_id,source_experiment_id,conversion_contract_version,"
            "changed_parameter,source_value_canonical,proposed_value_canonical,"
            "recommendation_semantic_hash,evaluation_identity_hash,"
            "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
            "ranking_snapshot_identity_hash,proposed_symbol,"
            "proposed_prediction_horizon,proposed_label_threshold,"
            "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
            "proposed_train_start_date,proposed_train_end_date,"
            "proposed_infer_start_date,proposed_infer_end_date,"
            "proposed_checkpoint_interval,proposed_resume_model_id,"
            "source_invocation_canonical,proposed_invocation_canonical,"
            "conversion_identity_canonical,conversion_identity_hash,"
            "conversion_hash_collision_ordinal) VALUES ("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,"
            "$18,$19::date,$20::date,$21::date,$22::date,$23,$24,$25,$26,$27,"
            "$28,$29) "
            "RETURNING " + ProposalColumns() + ";",
            pqxx::params{
                proposal.recommendationId, proposal.sourceExperimentId,
                kRecommendationConversionContractVersion,
                RecommendationMutationParameterText(proposal.changedParameter),
                proposal.sourceValueCanonical, proposal.proposedValueCanonical,
                proposal.recommendationSemanticHash,
                proposal.evaluationIdentityHash, proposal.evaluationPolicyHash,
                proposal.scoringPolicyHash, proposal.reviewAuthorizationHash,
                proposal.rankingSnapshotIdentityHash,
                invocation.configuration.symbol,
                invocation.configuration.predictionHorizon,
                invocation.configuration.labelThreshold,
                invocation.configuration.coreLrMult,
                invocation.configuration.headLrMult,
                invocation.configuration.targetEpochs,
                invocation.configuration.trainStartDate,
                invocation.configuration.trainEndDate,
                invocation.configuration.inferStartDate,
                invocation.configuration.inferEndDate,
                invocation.checkpointInterval, invocation.resumeModelId,
                proposal.sourceInvocationCanonical,
                proposal.proposedInvocationCanonical,
                proposal.conversionIdentityCanonical,
                proposal.conversionIdentityHash,
                collisionOrdinal}).one_row();
        PersistedRecommendationConversionProposal persisted = MapProposal(row);
        transaction.commit();
        return {
            hashCollision
                ? RecommendationConversionProposalPersistOutcome::
                      createdWithHashCollision
                : RecommendationConversionProposalPersistOutcome::created,
            std::move(persisted)};
    }
    catch (const pqxx::unique_violation&)
    {
        pqxx::read_transaction transaction{connection};
        auto existing = FindByCanonical(
            transaction, proposal.conversionIdentityCanonical);
        if (!existing)
            throw std::runtime_error(
                "recommendation_conversion_proposal_unique_conflict");
        return ExistingResult(std::move(*existing), proposal);
    }
}

std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposal(
    pqxx::connection& connection,
    long long proposalId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationConversionProposal(transaction, proposalId);
}

std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposal(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_id_must_be_positive");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{proposalId});
    if (rows.empty()) return std::nullopt;
    return MapProposal(rows.one_row());
}

std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposalByIdentity(
    pqxx::connection& connection,
    const std::string& conversionIdentityCanonical,
    const std::string& conversionIdentityHash)
{
    if (conversionIdentityCanonical.empty() || conversionIdentityHash.empty() ||
        conversionIdentityHash !=
            RecommendationCanonicalHash(conversionIdentityCanonical))
        throw std::invalid_argument(
            "invalid_recommendation_conversion_proposal_identity_lookup");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "conversion_identity_hash=$1 AND conversion_identity_canonical=$2;",
        pqxx::params{conversionIdentityHash, conversionIdentityCanonical});
    if (rows.empty()) return std::nullopt;
    return MapProposal(rows.one_row());
}

std::vector<PersistedRecommendationConversionProposal>
ListRecommendationConversionProposalsByRecommendation(
    pqxx::connection& connection,
    long long recommendationId,
    int limit)
{
    ValidateListArguments(recommendationId, limit);
    pqxx::read_transaction transaction{connection};
    return MapProposals(transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "recommendation_id=$1 ORDER BY recommendation_conversion_proposal_id "
        "ASC LIMIT $2;",
        pqxx::params{recommendationId, limit}));
}

std::vector<PersistedRecommendationConversionProposal>
ListRecommendationConversionProposalsBySourceExperiment(
    pqxx::connection& connection,
    long long sourceExperimentId,
    int limit)
{
    ValidateListArguments(sourceExperimentId, limit);
    pqxx::read_transaction transaction{connection};
    return MapProposals(transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "source_experiment_id=$1 ORDER BY recommendation_conversion_proposal_id "
        "ASC LIMIT $2;",
        pqxx::params{sourceExperimentId, limit}));
}

} // namespace EA::ExperimentRecommendation
