#include "CampaignOperationsRepository.hpp"

#include "ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"
#include "ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

using Campaign = PersistedOperationalCampaign;
using Provenance = PersistedGovernanceProvenanceEvent;
using Authorization = PersistedOperationalAuthorizationEvent;

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    const pqxx::field field = row[column];
    if (field.is_null()) return std::nullopt;
    return field.as<Value>();
}

std::string CampaignColumns()
{
    return
        "operational_campaign_id,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "materialization_member_count,origin_kind,action_kind,"
        "action_contract_version,scope_kind,scope_contract_version,"
        "campaign_contract_version,campaign_identity_canonical,"
        "campaign_identity_hash,created_at::text AS created_at";
}

std::string ProvenanceColumns()
{
    return
        "governance_provenance_event_id,operational_campaign_id,"
        "campaign_identity_canonical,"
        "recommendation_campaign_follow_up_ratification_event_id,"
        "ratification_contract_version,ratification_identity_canonical,"
        "ratification_identity_hash,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,prerequisite_policy,"
        "provenance_contract_version,provenance_identity_canonical,"
        "provenance_identity_hash,created_at::text AS created_at";
}

std::string AuthorizationColumns()
{
    return
        "authorization_event_id,operational_campaign_id,"
        "campaign_identity_canonical,previous_event_id,"
        "previous_event_identity_canonical,previous_event_identity_hash,"
        "chain_version,event_kind,action_kind,action_contract_version,"
        "scope_kind,scope_contract_version,prerequisite_policy,"
        "governance_provenance_event_id,provenance_identity_canonical,"
        "provenance_identity_hash,authorization_role,actor_identity,reason,"
        "to_char(not_before AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') AS not_before_text,"
        "CASE WHEN expires_at IS NULL THEN NULL ELSE "
        "to_char(expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END AS expires_at_text,"
        "authorization_contract_version,authorization_identity_canonical,"
        "authorization_identity_hash,created_at::text AS created_at";
}

void ValidateMaterializationBinding(pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign)
{
    const auto materialization =
        ExperimentRecommendation::FindRecommendationCampaignMaterialization(
            transaction, campaign.materializationId);
    if (!materialization ||
        materialization->contractVersion !=
            campaign.materializationContractVersion ||
        materialization->identityCanonical !=
            campaign.materializationCanonicalText ||
        materialization->identityHash != campaign.materializationIdentityHash ||
        materialization->selectedMemberCount != campaign.memberCount ||
        static_cast<int>(materialization->members.size()) !=
            campaign.memberCount)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_materialization_binding_mismatch");
}

Campaign MapCampaign(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        const OperationalCampaignId campaignId(
            row["operational_campaign_id"].as<long long>());
        OperationalCampaign campaign = BuildOperationalCampaign(
            row["recommendation_campaign_materialization_id"].as<long long>(),
            row["materialization_contract_version"].as<int>(),
            row["materialization_identity_canonical"].as<std::string>(),
            row["materialization_identity_hash"].as<std::string>(),
            row["materialization_member_count"].as<int>(),
            CampaignOriginKindFromText(row["origin_kind"].as<std::string>()),
            OperationalActionKindFromText(
                row["action_kind"].as<std::string>()),
            row["action_contract_version"].as<int>(),
            ScopeKindFromText(row["scope_kind"].as<std::string>()),
            row["scope_contract_version"].as<int>());
        if (campaign.identity.contractVersion() !=
                row["campaign_contract_version"].as<int>() ||
            campaign.identity.canonicalText() !=
                row["campaign_identity_canonical"].as<std::string>() ||
            campaign.identity.hash() !=
                row["campaign_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_campaign_identity_mismatch");
        ValidateMaterializationBinding(transaction, campaign);
        const std::string createdAt = row["created_at"].as<std::string>();
        if (createdAt.empty())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_created_at_invalid");
        return Campaign(campaignId, std::move(campaign), createdAt);
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_campaign:" +
                std::string(error.what()));
    }
}

std::optional<Campaign> FindCampaignByMaterialization(
    pqxx::transaction_base& transaction, long long materializationId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + CampaignColumns() +
            " FROM campaign_operations_campaign WHERE "
            "recommendation_campaign_materialization_id=$1 LIMIT 2;",
        pqxx::params{materializationId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_campaign_duplicate");
    return MapCampaign(transaction, rows.one_row());
}

void ValidateRatificationBinding(pqxx::transaction_base& transaction,
    const GovernanceProvenanceEvent& event,
    const OperationalCampaign& campaign)
{
    const auto ratification = ExperimentRecommendation::
        FindRecommendationCampaignFollowUpProposalRatification(
            transaction, event.ratificationEventId);
    if (!ratification ||
        ratification->ratification.identity.contractVersion !=
            event.ratificationContractVersion ||
        ratification->ratification.identity.canonicalText !=
            event.ratificationCanonicalText ||
        ratification->ratification.identity.hash !=
            event.ratificationIdentityHash ||
        ratification->ratification.reviewEventId != event.reviewEventId ||
        ratification->ratification.reviewContractVersion !=
            event.reviewContractVersion ||
        ratification->ratification.reviewCanonicalText !=
            event.reviewCanonicalText ||
        ratification->ratification.reviewIdentityHash !=
            event.reviewIdentityHash ||
        ratification->ratification.followUpProposalId != event.proposalId ||
        ratification->ratification.proposalContractVersion !=
            event.proposalContractVersion ||
        ratification->ratification.proposalCanonicalText !=
            event.proposalCanonicalText ||
        ratification->ratification.proposalIdentityHash !=
            event.proposalIdentityHash)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_ratification_binding_mismatch");
    const auto proposal = ExperimentRecommendation::
        FindRecommendationCampaignFollowUpProposal(
            transaction, event.proposalId);
    if (!proposal ||
        proposal->proposal.materializationIdentity.materializationId !=
            campaign.materializationId ||
        proposal->proposal.materializationIdentity.contractVersion !=
            campaign.materializationContractVersion ||
        proposal->proposal.materializationIdentity.identityCanonical !=
            campaign.materializationCanonicalText ||
        proposal->proposal.materializationIdentity.identityHash !=
            campaign.materializationIdentityHash ||
        proposal->proposal.materializationIdentity.memberCount !=
            campaign.memberCount)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_ratification_materialization_mismatch");
}

Provenance MapProvenance(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        const GovernanceProvenanceEventId eventId(
            row["governance_provenance_event_id"].as<long long>());
        GovernanceProvenanceEvent event = BuildGovernanceProvenanceEvent(
            OperationalCampaignId(
                row["operational_campaign_id"].as<long long>()),
            row["campaign_identity_canonical"].as<std::string>(),
            row["recommendation_campaign_follow_up_ratification_event_id"]
                .as<long long>(),
            row["ratification_contract_version"].as<int>(),
            row["ratification_identity_canonical"].as<std::string>(),
            row["ratification_identity_hash"].as<std::string>(),
            row["recommendation_campaign_follow_up_proposal_review_event_id"]
                .as<long long>(),
            row["review_contract_version"].as<int>(),
            row["review_identity_canonical"].as<std::string>(),
            row["review_identity_hash"].as<std::string>(),
            row["recommendation_campaign_follow_up_proposal_id"].as<long long>(),
            row["proposal_contract_version"].as<int>(),
            row["proposal_identity_canonical"].as<std::string>(),
            row["proposal_identity_hash"].as<std::string>(),
            PrerequisitePolicyFromText(
                row["prerequisite_policy"].as<std::string>()));
        if (event.identity.contractVersion() !=
                row["provenance_contract_version"].as<int>() ||
            event.identity.canonicalText() !=
                row["provenance_identity_canonical"].as<std::string>() ||
            event.identity.hash() !=
                row["provenance_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_provenance_identity_mismatch");
        const auto campaign = FindOperationalCampaign(
            transaction, event.campaignId);
        if (!campaign || campaign->campaign.identity.canonicalText() !=
                event.campaignCanonicalText)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_provenance_campaign_mismatch");
        ValidateRatificationBinding(transaction, event, campaign->campaign);
        const std::string createdAt = row["created_at"].as<std::string>();
        if (createdAt.empty())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_created_at_invalid");
        return Provenance(eventId, std::move(event), createdAt);
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_provenance:" +
                std::string(error.what()));
    }
}

Authorization MapAuthorization(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        const AuthorizationEventId eventId(
            row["authorization_event_id"].as<long long>());
        const auto previousId = OptionalValue<long long>(row,
            "previous_event_id");
        const auto provenanceId = OptionalValue<long long>(row,
            "governance_provenance_event_id");
        OperationalAuthorizationEvent event =
            BuildOperationalAuthorizationEvent(
                OperationalCampaignId(
                    row["operational_campaign_id"].as<long long>()),
                row["campaign_identity_canonical"].as<std::string>(),
                previousId ? std::optional<AuthorizationEventId>(
                                 AuthorizationEventId(*previousId))
                           : std::nullopt,
                OptionalValue<std::string>(row,
                    "previous_event_identity_canonical"),
                OptionalValue<std::string>(row,
                    "previous_event_identity_hash"),
                row["chain_version"].as<int>(),
                AuthorizationEventKindFromText(
                    row["event_kind"].as<std::string>()),
                OperationalActionKindFromText(
                    row["action_kind"].as<std::string>()),
                row["action_contract_version"].as<int>(),
                ScopeKindFromText(row["scope_kind"].as<std::string>()),
                row["scope_contract_version"].as<int>(),
                PrerequisitePolicyFromText(
                    row["prerequisite_policy"].as<std::string>()),
                provenanceId
                    ? std::optional<GovernanceProvenanceEventId>(
                          GovernanceProvenanceEventId(*provenanceId))
                    : std::nullopt,
                OptionalValue<std::string>(row,
                    "provenance_identity_canonical"),
                OptionalValue<std::string>(row,
                    "provenance_identity_hash"),
                row["authorization_role"].as<std::string>(),
                ActorIdentity(row["actor_identity"].as<std::string>()),
                Reason(row["reason"].as<std::string>()),
                UtcTimestamp(row["not_before_text"].as<std::string>()),
                row["expires_at_text"].is_null()
                    ? std::nullopt
                    : std::optional<UtcTimestamp>(UtcTimestamp(
                          row["expires_at_text"].as<std::string>())));
        if (event.identity.contractVersion() !=
                row["authorization_contract_version"].as<int>() ||
            event.identity.canonicalText() !=
                row["authorization_identity_canonical"].as<std::string>() ||
            event.identity.hash() !=
                row["authorization_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_authorization_identity_mismatch");
        const auto campaign = FindOperationalCampaign(
            transaction, event.campaignId);
        if (!campaign || campaign->campaign.identity.canonicalText() !=
                event.campaignCanonicalText)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_authorization_campaign_mismatch");
        if (event.provenanceEventId)
        {
            const auto provenance = FindGovernanceProvenanceEvent(
                transaction, *event.provenanceEventId);
            if (!provenance ||
                provenance->event.identity.canonicalText() !=
                    *event.provenanceCanonicalText ||
                provenance->event.identity.hash() !=
                    *event.provenanceIdentityHash ||
                provenance->event.campaignId != event.campaignId ||
                provenance->event.prerequisitePolicy !=
                    event.prerequisitePolicy)
                throw Error(ErrorCode::persistenceCorruption,
                    "campaign_operations_authorization_provenance_mismatch");
        }
        if (event.previousEventId)
        {
            const pqxx::result previousRows = transaction.exec(
                "SELECT authorization_identity_canonical,"
                "authorization_identity_hash,operational_campaign_id,"
                "action_kind,action_contract_version,scope_kind,"
                "scope_contract_version,chain_version FROM "
                "campaign_operations_authorization_event WHERE "
                "authorization_event_id=$1;",
                pqxx::params{event.previousEventId->value()});
            if (previousRows.empty())
                throw Error(ErrorCode::persistenceCorruption,
                    "campaign_operations_authorization_predecessor_missing");
            const auto& previous = previousRows.one_row();
            if (previous["authorization_identity_canonical"].as<std::string>() !=
                    *event.previousEventCanonicalText ||
                previous["authorization_identity_hash"].as<std::string>() !=
                    *event.previousEventIdentityHash ||
                previous["operational_campaign_id"].as<long long>() !=
                    event.campaignId.value() ||
                previous["action_kind"].as<std::string>() !=
                    ToText(event.actionKind) ||
                previous["action_contract_version"].as<int>() !=
                    event.actionContractVersion ||
                previous["scope_kind"].as<std::string>() !=
                    ToText(event.scopeKind) ||
                previous["scope_contract_version"].as<int>() !=
                    event.scopeContractVersion ||
                previous["chain_version"].as<int>() + 1 !=
                    event.chainVersion)
                throw Error(ErrorCode::persistenceCorruption,
                    "campaign_operations_authorization_predecessor_mismatch");
        }
        const std::string createdAt = row["created_at"].as<std::string>();
        if (createdAt.empty())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_created_at_invalid");
        return Authorization(eventId, std::move(event), createdAt);
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_authorization:" +
                std::string(error.what()));
    }
}

void InsertAudit(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId,
    std::optional<GovernanceProvenanceEventId> provenanceEventId,
    std::optional<AuthorizationEventId> authorizationEventId,
    const std::string& causeKind, const ActorIdentity& actor,
    const std::string& capability, const Reason& reason,
    std::optional<int> priorVersion, int resultingVersion)
{
    transaction.exec(
        "INSERT INTO campaign_operations_audit_reference_event ("
        "operational_campaign_id,governance_provenance_event_id,"
        "authorization_event_id,cause_kind,actor_identity,capability,reason,"
        "prior_version,resulting_version,outcome,replay_disposition) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,'recorded','recorded');",
        pqxx::params{campaignId.value(),
            provenanceEventId
                ? std::optional<long long>(provenanceEventId->value())
                : std::nullopt,
            authorizationEventId
                ? std::optional<long long>(authorizationEventId->value())
                : std::nullopt,
            causeKind, actor.value(), capability, reason.value(), priorVersion,
            resultingVersion});
}

std::optional<Authorization> FindAuthorizationByNaturalVersion(
    pqxx::transaction_base& transaction,
    const OperationalAuthorizationEvent& event)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + AuthorizationColumns() +
            " FROM campaign_operations_authorization_event WHERE "
            "operational_campaign_id=$1 AND action_kind=$2 AND "
            "action_contract_version=$3 AND scope_kind=$4 AND "
            "scope_contract_version=$5 AND chain_version=$6 LIMIT 2;",
        pqxx::params{event.campaignId.value(), ToText(event.actionKind),
            event.actionContractVersion, ToText(event.scopeKind),
            event.scopeContractVersion, event.chainVersion});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_authorization_duplicate");
    return MapAuthorization(transaction, rows.one_row());
}

} // namespace

PersistedOperationalCampaign::PersistedOperationalCampaign(
    OperationalCampaignId campaignIdValue, OperationalCampaign campaignValue,
    std::string createdAtValue)
    : campaignId(campaignIdValue), campaign(std::move(campaignValue)),
      createdAt(std::move(createdAtValue))
{
}

PersistedGovernanceProvenanceEvent::PersistedGovernanceProvenanceEvent(
    GovernanceProvenanceEventId provenanceEventIdValue,
    GovernanceProvenanceEvent eventValue, std::string createdAtValue)
    : provenanceEventId(provenanceEventIdValue),
      event(std::move(eventValue)), createdAt(std::move(createdAtValue))
{
}

PersistedOperationalAuthorizationEvent::
    PersistedOperationalAuthorizationEvent(
        AuthorizationEventId authorizationEventIdValue,
        OperationalAuthorizationEvent eventValue, std::string createdAtValue)
    : authorizationEventId(authorizationEventIdValue),
      event(std::move(eventValue)), createdAt(std::move(createdAtValue))
{
}

std::string ToText(PersistOutcome outcome)
{
    switch (outcome)
    {
        case PersistOutcome::recorded: return "recorded";
        case PersistOutcome::existingIdentical: return "existing_identical";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_persist_outcome_invalid");
}

bool SchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return SchemaExists(transaction);
}

bool SchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('campaign_operations_campaign') IS NOT NULL "
        "AND to_regclass('campaign_operations_governance_provenance_event') "
        "IS NOT NULL AND "
        "to_regclass('campaign_operations_authorization_event') IS NOT NULL "
        "AND to_regclass('campaign_operations_audit_reference_event') "
        "IS NOT NULL;")
        .one_row()[0]
        .as<bool>();
}

PersistResult<PersistedOperationalCampaign> PersistOperationalCampaign(
    pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign, const ActorIdentity& actor,
    const Reason& reason)
{
    ValidateOperationalCampaign(campaign);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030001));",
        pqxx::params{std::to_string(campaign.materializationId)});
    if (auto existing = FindCampaignByMaterialization(
            transaction, campaign.materializationId))
    {
        if (existing->campaign != campaign)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_campaign_conflict");
        return {PersistOutcome::existingIdentical, std::move(*existing)};
    }
    ValidateMaterializationBinding(transaction, campaign);
    const long long campaignId = transaction.exec(
        "INSERT INTO campaign_operations_campaign ("
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,materialization_member_count,"
        "origin_kind,action_kind,action_contract_version,scope_kind,"
        "scope_contract_version,campaign_contract_version,"
        "campaign_identity_canonical,campaign_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) "
        "RETURNING operational_campaign_id;",
        pqxx::params{campaign.materializationId,
            campaign.materializationContractVersion,
            campaign.materializationCanonicalText,
            campaign.materializationIdentityHash, campaign.memberCount,
            ToText(campaign.originKind), ToText(campaign.actionKind),
            campaign.actionContractVersion, ToText(campaign.scopeKind),
            campaign.scopeContractVersion,
            campaign.identity.contractVersion(),
            campaign.identity.canonicalText(), campaign.identity.hash()})
                                     .one_row()[0]
                                     .as<long long>();
    InsertAudit(transaction, OperationalCampaignId(campaignId), std::nullopt,
        std::nullopt, "campaign_created", actor,
        "campaign_operations_campaign_creator", reason, std::nullopt, 1);
    auto persisted = FindOperationalCampaign(
        transaction, OperationalCampaignId(campaignId));
    if (!persisted)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_campaign_insert_missing");
    return {PersistOutcome::recorded, std::move(*persisted)};
}

std::optional<PersistedOperationalCampaign> FindOperationalCampaign(
    pqxx::connection& connection, OperationalCampaignId campaignId)
{
    pqxx::read_transaction transaction{connection};
    return FindOperationalCampaign(transaction, campaignId);
}

std::optional<PersistedOperationalCampaign> FindOperationalCampaign(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + CampaignColumns() +
            " FROM campaign_operations_campaign WHERE "
            "operational_campaign_id=$1;",
        pqxx::params{campaignId.value()});
    if (rows.empty()) return std::nullopt;
    return MapCampaign(transaction, rows.one_row());
}

std::optional<PersistedOperationalCampaign>
FindOperationalCampaignByMaterialization(pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_materialization_id_invalid");
    return FindCampaignByMaterialization(transaction, materializationId);
}

std::vector<PersistedOperationalCampaign> ListOperationalCampaigns(
    pqxx::transaction_base& transaction, int limit)
{
    if (limit <= 0 || limit > kCampaignOperationsRepositoryMaximumListLimit)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_list_limit_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + CampaignColumns() +
            " FROM campaign_operations_campaign ORDER BY "
            "operational_campaign_id ASC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedOperationalCampaign> campaigns;
    campaigns.reserve(rows.size());
    for (const auto& row : rows)
        campaigns.push_back(MapCampaign(transaction, row));
    return campaigns;
}

PersistResult<PersistedGovernanceProvenanceEvent>
PersistGovernanceProvenanceEvent(pqxx::transaction_base& transaction,
    const GovernanceProvenanceEvent& event, const ActorIdentity& actor,
    const Reason& reason)
{
    ValidateGovernanceProvenanceEvent(event);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030002));",
        pqxx::params{std::to_string(event.campaignId.value()) +
            ";ratification=" + std::to_string(event.ratificationEventId)});
    const auto campaign = FindOperationalCampaign(transaction, event.campaignId);
    if (!campaign || campaign->campaign.identity.canonicalText() !=
            event.campaignCanonicalText)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_provenance_campaign_conflict");
    ValidateRatificationBinding(transaction, event, campaign->campaign);
    if (auto existing = FindGovernanceProvenanceEventByRatification(
            transaction, event.campaignId, event.ratificationEventId))
    {
        if (existing->event != event)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_governance_provenance_conflict");
        return {PersistOutcome::existingIdentical, std::move(*existing)};
    }
    const long long eventId = transaction.exec(
        "INSERT INTO campaign_operations_governance_provenance_event ("
        "operational_campaign_id,campaign_identity_canonical,"
        "recommendation_campaign_follow_up_ratification_event_id,"
        "ratification_contract_version,ratification_identity_canonical,"
        "ratification_identity_hash,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,prerequisite_policy,"
        "provenance_contract_version,provenance_identity_canonical,"
        "provenance_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,"
        "$16,$17,$18) RETURNING governance_provenance_event_id;",
        pqxx::params{event.campaignId.value(), event.campaignCanonicalText,
            event.ratificationEventId, event.ratificationContractVersion,
            event.ratificationCanonicalText, event.ratificationIdentityHash,
            event.reviewEventId, event.reviewContractVersion,
            event.reviewCanonicalText, event.reviewIdentityHash,
            event.proposalId, event.proposalContractVersion,
            event.proposalCanonicalText, event.proposalIdentityHash,
            ToText(event.prerequisitePolicy),
            event.identity.contractVersion(), event.identity.canonicalText(),
            event.identity.hash()})
                                  .one_row()[0]
                                  .as<long long>();
    InsertAudit(transaction, event.campaignId,
        GovernanceProvenanceEventId(eventId), std::nullopt,
        "governance_provenance_recorded", actor,
        kCampaignOperationsAuthorizationRole, reason, std::nullopt, 1);
    auto persisted = FindGovernanceProvenanceEvent(
        transaction, GovernanceProvenanceEventId(eventId));
    if (!persisted)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_provenance_insert_missing");
    return {PersistOutcome::recorded, std::move(*persisted)};
}

std::optional<PersistedGovernanceProvenanceEvent>
FindGovernanceProvenanceEvent(pqxx::transaction_base& transaction,
    GovernanceProvenanceEventId provenanceEventId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProvenanceColumns() +
            " FROM campaign_operations_governance_provenance_event WHERE "
            "governance_provenance_event_id=$1;",
        pqxx::params{provenanceEventId.value()});
    if (rows.empty()) return std::nullopt;
    return MapProvenance(transaction, rows.one_row());
}

std::optional<PersistedGovernanceProvenanceEvent>
FindGovernanceProvenanceEventByRatification(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    long long ratificationEventId)
{
    if (ratificationEventId <= 0)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_ratification_event_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProvenanceColumns() +
            " FROM campaign_operations_governance_provenance_event WHERE "
            "operational_campaign_id=$1 AND "
            "recommendation_campaign_follow_up_ratification_event_id=$2 "
            "LIMIT 2;",
        pqxx::params{campaignId.value(), ratificationEventId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_provenance_duplicate");
    return MapProvenance(transaction, rows.one_row());
}

PersistResult<PersistedOperationalAuthorizationEvent>
PersistOperationalAuthorizationEvent(pqxx::transaction_base& transaction,
    const OperationalAuthorizationEvent& event)
{
    ValidateOperationalAuthorizationEvent(event);
    const std::string lockKey = event.campaignCanonicalText + ";action=" +
        ToText(event.actionKind) + ";scope=" + ToText(event.scopeKind);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030003));",
        pqxx::params{lockKey});
    if (auto existing = FindAuthorizationByNaturalVersion(transaction, event))
    {
        if (existing->event != event)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_authorization_conflict");
        return {PersistOutcome::existingIdentical, std::move(*existing)};
    }
    const auto campaign = FindOperationalCampaign(transaction, event.campaignId);
    if (!campaign || campaign->campaign.identity.canonicalText() !=
            event.campaignCanonicalText)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_authorization_campaign_conflict");
    if (event.provenanceEventId)
    {
        const auto provenance = FindGovernanceProvenanceEvent(
            transaction, *event.provenanceEventId);
        if (!provenance || provenance->event.campaignId != event.campaignId ||
            provenance->event.identity.canonicalText() !=
                *event.provenanceCanonicalText ||
            provenance->event.identity.hash() !=
                *event.provenanceIdentityHash ||
            provenance->event.prerequisitePolicy !=
                event.prerequisitePolicy)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_authorization_provenance_conflict");
    }
    const auto head = FindOperationalAuthorizationHead(transaction,
        event.campaignId, event.actionKind, event.actionContractVersion,
        event.scopeKind, event.scopeContractVersion);
    if ((!head && (event.chainVersion != 1 || event.previousEventId)) ||
        (head && (event.chainVersion != head->event.chainVersion + 1 ||
            !event.previousEventId ||
            *event.previousEventId != head->authorizationEventId ||
            *event.previousEventCanonicalText !=
                head->event.identity.canonicalText() ||
            *event.previousEventIdentityHash != head->event.identity.hash())))
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_authorization_stale_head");
    long long eventId = 0;
    try
    {
        eventId = transaction.exec(
            "INSERT INTO campaign_operations_authorization_event ("
            "operational_campaign_id,campaign_identity_canonical,"
            "previous_event_id,previous_event_identity_canonical,"
            "previous_event_identity_hash,chain_version,event_kind,action_kind,"
            "action_contract_version,scope_kind,scope_contract_version,"
            "prerequisite_policy,governance_provenance_event_id,"
            "provenance_identity_canonical,provenance_identity_hash,"
            "authorization_role,actor_identity,reason,not_before,expires_at,"
            "authorization_contract_version,authorization_identity_canonical,"
            "authorization_identity_hash) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
            "$17,$18,$19::timestamptz,$20::timestamptz,$21,$22,$23) "
            "RETURNING authorization_event_id;",
            pqxx::params{event.campaignId.value(), event.campaignCanonicalText,
                event.previousEventId
                    ? std::optional<long long>(event.previousEventId->value())
                    : std::nullopt,
                event.previousEventCanonicalText,
                event.previousEventIdentityHash, event.chainVersion,
                ToText(event.eventKind), ToText(event.actionKind),
                event.actionContractVersion, ToText(event.scopeKind),
                event.scopeContractVersion, ToText(event.prerequisitePolicy),
                event.provenanceEventId
                    ? std::optional<long long>(event.provenanceEventId->value())
                    : std::nullopt,
                event.provenanceCanonicalText, event.provenanceIdentityHash,
                event.authorizationRole, event.actor.value(),
                event.reason.value(), event.notBefore.value(),
                event.expiresAt
                    ? std::optional<std::string>(event.expiresAt->value())
                    : std::nullopt,
                event.identity.contractVersion(),
                event.identity.canonicalText(), event.identity.hash()})
                      .one_row()[0]
                      .as<long long>();
    }
    catch (const pqxx::unique_violation&)
    {
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_authorization_conflict");
    }
    InsertAudit(transaction, event.campaignId, std::nullopt,
        AuthorizationEventId(eventId), "authorization_recorded", event.actor,
        event.authorizationRole, event.reason,
        event.chainVersion > 1
            ? std::optional<int>(event.chainVersion - 1)
            : std::nullopt,
        event.chainVersion);
    auto persisted = FindOperationalAuthorizationEvent(
        transaction, AuthorizationEventId(eventId));
    if (!persisted)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_authorization_insert_missing");
    return {PersistOutcome::recorded, std::move(*persisted)};
}

std::optional<PersistedOperationalAuthorizationEvent>
FindOperationalAuthorizationEvent(pqxx::transaction_base& transaction,
    AuthorizationEventId authorizationEventId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + AuthorizationColumns() +
            " FROM campaign_operations_authorization_event WHERE "
            "authorization_event_id=$1;",
        pqxx::params{authorizationEventId.value()});
    if (rows.empty()) return std::nullopt;
    return MapAuthorization(transaction, rows.one_row());
}

std::optional<PersistedOperationalAuthorizationEvent>
FindOperationalAuthorizationHead(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion)
{
    if (actionContractVersion != kCampaignOperationsActionContractVersion ||
        scopeContractVersion != kCampaignOperationsScopeContractVersion)
        throw Error(ErrorCode::unsupportedContractVersion,
            "campaign_operations_authorization_contract_unsupported");
    const pqxx::result rows = transaction.exec(
        "SELECT " + AuthorizationColumns() +
            " FROM campaign_operations_authorization_event WHERE "
            "operational_campaign_id=$1 AND action_kind=$2 AND "
            "action_contract_version=$3 AND scope_kind=$4 AND "
            "scope_contract_version=$5 ORDER BY chain_version DESC LIMIT 1;",
        pqxx::params{campaignId.value(), ToText(actionKind),
            actionContractVersion, ToText(scopeKind), scopeContractVersion});
    if (rows.empty()) return std::nullopt;
    return MapAuthorization(transaction, rows.one_row());
}

std::vector<PersistedOperationalAuthorizationEvent>
LoadOperationalAuthorizationChain(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion,
    int limit)
{
    if (limit <= 0 || limit > kCampaignOperationsRepositoryMaximumListLimit)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_list_limit_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + AuthorizationColumns() +
            " FROM campaign_operations_authorization_event WHERE "
            "operational_campaign_id=$1 AND action_kind=$2 AND "
            "action_contract_version=$3 AND scope_kind=$4 AND "
            "scope_contract_version=$5 ORDER BY chain_version ASC LIMIT $6;",
        pqxx::params{campaignId.value(), ToText(actionKind),
            actionContractVersion, ToText(scopeKind), scopeContractVersion,
            limit});
    std::vector<PersistedOperationalAuthorizationEvent> chain;
    chain.reserve(rows.size());
    for (const auto& row : rows)
        chain.push_back(MapAuthorization(transaction, row));
    return chain;
}

} // namespace EA::CampaignOperations
