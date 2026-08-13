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

void RequireCapability(pqxx::transaction_base& transaction,
    const char* capability)
{
    const bool authorized = transaction.exec(
        "SELECT pg_has_role(current_user,$1,'USAGE');",
        pqxx::params{capability})
                                .one_row()[0]
                                .as<bool>();
    if (!authorized)
        throw Error(ErrorCode::authorizationDenied,
            "campaign_operations_capability_denied:" +
                std::string(capability));
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
    RequireCapability(transaction, kCampaignOperationsCampaignCreatorRole);
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
        kCampaignOperationsCampaignCreatorRole, reason, std::nullopt, 1);
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

PersistedBudgetLedgerEntry::PersistedBudgetLedgerEntry(
    BudgetLedgerEntryId budgetLedgerEntryIdValue,
    BudgetLedgerEntry entryValue, std::string createdAtValue)
    : budgetLedgerEntryId(budgetLedgerEntryIdValue),
      entry(std::move(entryValue)), createdAt(std::move(createdAtValue))
{
}

PersistedReservation::PersistedReservation(ReservationId reservationIdValue,
    Reservation reservationValue, ReservationState stateValue,
    int stateVersionValue, std::string createdAtValue,
    std::string updatedAtValue)
    : reservationId(reservationIdValue),
      reservation(std::move(reservationValue)), state(stateValue),
      stateVersion(stateVersionValue), createdAt(std::move(createdAtValue)),
      updatedAt(std::move(updatedAtValue))
{
}

PersistedOperationalRequest::PersistedOperationalRequest(
    OperationalRequestId requestIdValue, OperationalRequest requestValue,
    RequestState stateValue, int stateVersionValue,
    bool productionDispatchEnabledValue, std::string createdAtValue,
    std::string updatedAtValue)
    : requestId(requestIdValue), request(std::move(requestValue)),
      state(stateValue), stateVersion(stateVersionValue),
      productionDispatchEnabled(productionDispatchEnabledValue),
      createdAt(std::move(createdAtValue)),
      updatedAt(std::move(updatedAtValue))
{
}

PersistedReservationEvent::PersistedReservationEvent(
    ReservationEventId reservationEventIdValue, ReservationEvent eventValue,
    std::string createdAtValue)
    : reservationEventId(reservationEventIdValue),
      event(std::move(eventValue)), createdAt(std::move(createdAtValue))
{
}

AcceptedOperationalRequest::AcceptedOperationalRequest(
    PersistedReservation reservationValue,
    PersistedOperationalRequest requestValue,
    PersistedReservationEvent acquisitionEventValue)
    : reservation(std::move(reservationValue)),
      request(std::move(requestValue)),
      acquisitionEvent(std::move(acquisitionEventValue))
{
}

namespace
{

std::string BudgetColumns()
{
    return
        "budget_ledger_entry_id,operational_campaign_id,"
        "campaign_identity_canonical,previous_entry_id,"
        "previous_entry_identity_canonical,previous_entry_identity_hash,"
        "ledger_version,entry_kind,ledger_status,budget_unit,delta,"
        "prior_total,resulting_total,administrator_identity,reason,"
        "budget_contract_version,budget_identity_canonical,"
        "budget_identity_hash,created_at::text AS created_at";
}

std::string ReservationColumns()
{
    return
        "reservation_id,operational_campaign_id,"
        "campaign_identity_canonical,logical_operation_contract_version,"
        "logical_operation_canonical,logical_operation_hash,"
        "authorization_event_id,authorization_identity_canonical,"
        "authorization_identity_hash,budget_ledger_entry_id,"
        "budget_ledger_version,budget_identity_canonical,"
        "budget_identity_hash,action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "scope_kind,scope_contract_version,materialization_member_count,"
        "amount,budget_unit,"
        "CASE WHEN expires_at IS NULL THEN NULL ELSE "
        "to_char(expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END AS expires_at_text,"
        "reservation_contract_version,reservation_identity_canonical,"
        "reservation_identity_hash,reservation_state,state_version,"
        "created_at::text AS created_at,updated_at::text AS updated_at";
}

std::string RequestColumns()
{
    return
        "operational_request_id,operational_campaign_id,"
        "campaign_identity_canonical,logical_operation_contract_version,"
        "logical_operation_canonical,logical_operation_hash,"
        "authorization_event_id,authorization_identity_canonical,"
        "authorization_identity_hash,reservation_id,"
        "reservation_identity_canonical,reservation_identity_hash,"
        "action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "ordered_scope_digest,materialization_member_count,"
        "accepting_actor_identity,reason,prerequisite_policy,"
        "provenance_identity_canonical,provenance_identity_hash,"
        "request_contract_version,request_identity_canonical,"
        "request_identity_hash,request_state,state_version,"
        "production_dispatch_enabled,created_at::text AS created_at,"
        "updated_at::text AS updated_at";
}

std::string ReservationEventColumns()
{
    return
        "reservation_event_id,reservation_id,"
        "reservation_identity_canonical,transition_kind,expected_state,"
        "resulting_state,expected_version,resulting_version,"
        "operational_request_id,request_identity_canonical,amount,"
        "reservation_event_contract_version,"
        "reservation_event_identity_canonical,"
        "reservation_event_identity_hash,created_at::text AS created_at";
}

LogicalOperation MapLogicalOperation(const pqxx::row& row)
{
    LogicalOperation operation = BuildLogicalOperation(
        OperationalCampaignId(
            row["operational_campaign_id"].as<long long>()),
        row["campaign_identity_canonical"].as<std::string>(),
        OperationalActionKindFromText(
            row["action_kind"].as<std::string>()),
        row["action_contract_version"].as<int>(),
        row["recommendation_campaign_materialization_id"].as<long long>(),
        row["materialization_contract_version"].as<int>(),
        row["materialization_identity_canonical"].as<std::string>(),
        row["materialization_identity_hash"].as<std::string>(),
        ScopeKind::completeMaterialization,
        kCampaignOperationsScopeContractVersion);
    if (operation.identity.contractVersion() !=
            row["logical_operation_contract_version"].as<int>() ||
        operation.identity.canonicalText() !=
            row["logical_operation_canonical"].as<std::string>() ||
        operation.identity.hash() !=
            row["logical_operation_hash"].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_logical_operation_identity_mismatch");
    return operation;
}

PersistedBudgetLedgerEntry MapBudgetEntry(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        const auto previousId =
            OptionalValue<long long>(row, "previous_entry_id");
        BudgetLedgerEntry entry = BuildBudgetLedgerEntry(
            OperationalCampaignId(
                row["operational_campaign_id"].as<long long>()),
            row["campaign_identity_canonical"].as<std::string>(),
            previousId
                ? std::optional<BudgetLedgerEntryId>(
                      BudgetLedgerEntryId(*previousId))
                : std::nullopt,
            OptionalValue<std::string>(
                row, "previous_entry_identity_canonical"),
            OptionalValue<std::string>(
                row, "previous_entry_identity_hash"),
            row["ledger_version"].as<int>(),
            BudgetLedgerEntryKindFromText(
                row["entry_kind"].as<std::string>()),
            BudgetLedgerStatusFromText(
                row["ledger_status"].as<std::string>()),
            BudgetUnitFromText(row["budget_unit"].as<std::string>()),
            row["delta"].as<long long>(),
            row["prior_total"].as<long long>(),
            row["resulting_total"].as<long long>(),
            ActorIdentity(
                row["administrator_identity"].as<std::string>()),
            Reason(row["reason"].as<std::string>()));
        if (entry.identity.contractVersion() !=
                row["budget_contract_version"].as<int>() ||
            entry.identity.canonicalText() !=
                row["budget_identity_canonical"].as<std::string>() ||
            entry.identity.hash() !=
                row["budget_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_budget_identity_mismatch");
        const auto campaign =
            FindOperationalCampaign(transaction, entry.campaignId);
        if (!campaign ||
            campaign->campaign.identity.canonicalText() !=
                entry.campaignCanonicalText)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_budget_campaign_mismatch");
        if (entry.previousEntryId)
        {
            const pqxx::result previousRows = transaction.exec(
                "SELECT operational_campaign_id,ledger_version,"
                "budget_identity_canonical,budget_identity_hash "
                "FROM campaign_operations_budget_ledger_entry WHERE "
                "budget_ledger_entry_id=$1;",
                pqxx::params{entry.previousEntryId->value()});
            if (previousRows.empty() ||
                previousRows.one_row()["operational_campaign_id"]
                        .as<long long>() != entry.campaignId.value() ||
                previousRows.one_row()["ledger_version"].as<int>() !=
                    entry.ledgerVersion - 1 ||
                previousRows.one_row()["budget_identity_canonical"]
                        .as<std::string>() !=
                    *entry.previousEntryCanonicalText ||
                previousRows.one_row()["budget_identity_hash"]
                        .as<std::string>() !=
                    *entry.previousEntryIdentityHash)
                throw Error(ErrorCode::persistenceCorruption,
                    "campaign_operations_budget_predecessor_mismatch");
        }
        return PersistedBudgetLedgerEntry(
            BudgetLedgerEntryId(
                row["budget_ledger_entry_id"].as<long long>()),
            std::move(entry), row["created_at"].as<std::string>());
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_budget:" +
                std::string(error.what()));
    }
}

PersistedReservation MapReservation(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        LogicalOperation operation = MapLogicalOperation(row);
        Reservation reservation = BuildReservation(std::move(operation),
            AuthorizationEventId(
                row["authorization_event_id"].as<long long>()),
            row["authorization_identity_canonical"].as<std::string>(),
            row["authorization_identity_hash"].as<std::string>(),
            BudgetLedgerEntryId(
                row["budget_ledger_entry_id"].as<long long>()),
            row["budget_ledger_version"].as<int>(),
            row["budget_identity_canonical"].as<std::string>(),
            row["budget_identity_hash"].as<std::string>(),
            row["materialization_member_count"].as<int>(),
            row["amount"].as<long long>(),
            BudgetUnitFromText(row["budget_unit"].as<std::string>()),
            row["expires_at_text"].is_null()
                ? std::nullopt
                : std::optional<UtcTimestamp>(UtcTimestamp(
                      row["expires_at_text"].as<std::string>())));
        if (reservation.identity.contractVersion() !=
                row["reservation_contract_version"].as<int>() ||
            reservation.identity.canonicalText() !=
                row["reservation_identity_canonical"].as<std::string>() ||
            reservation.identity.hash() !=
                row["reservation_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_reservation_identity_mismatch");
        const auto authorization = FindOperationalAuthorizationEvent(
            transaction, reservation.acceptingAuthorizationEventId);
        const auto budget = FindBudgetLedgerEntry(
            transaction, reservation.budgetLedgerEntryId);
        if (!authorization ||
            authorization->event.identity.canonicalText() !=
                reservation.acceptingAuthorizationCanonicalText ||
            authorization->event.identity.hash() !=
                reservation.acceptingAuthorizationIdentityHash ||
            !budget ||
            budget->entry.ledgerVersion !=
                reservation.budgetLedgerVersion ||
            budget->entry.identity.canonicalText() !=
                reservation.budgetLedgerCanonicalText ||
            budget->entry.identity.hash() !=
                reservation.budgetLedgerIdentityHash)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_reservation_provenance_mismatch");
        return PersistedReservation(
            ReservationId(row["reservation_id"].as<long long>()),
            std::move(reservation),
            ReservationStateFromText(
                row["reservation_state"].as<std::string>()),
            row["state_version"].as<int>(),
            row["created_at"].as<std::string>(),
            row["updated_at"].as<std::string>());
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_reservation:" +
                std::string(error.what()));
    }
}

PersistedOperationalRequest MapRequest(
    pqxx::transaction_base& transaction, const pqxx::row& row)
{
    try
    {
        LogicalOperation operation = MapLogicalOperation(row);
        OperationalRequest request = BuildOperationalRequest(
            std::move(operation),
            AuthorizationEventId(
                row["authorization_event_id"].as<long long>()),
            row["authorization_identity_canonical"].as<std::string>(),
            row["authorization_identity_hash"].as<std::string>(),
            ReservationId(row["reservation_id"].as<long long>()),
            row["reservation_identity_canonical"].as<std::string>(),
            row["reservation_identity_hash"].as<std::string>(),
            row["materialization_member_count"].as<int>(),
            row["ordered_scope_digest"].as<std::string>(),
            ActorIdentity(
                row["accepting_actor_identity"].as<std::string>()),
            Reason(row["reason"].as<std::string>()),
            PrerequisitePolicyFromText(
                row["prerequisite_policy"].as<std::string>()),
            OptionalValue<std::string>(
                row, "provenance_identity_canonical"),
            OptionalValue<std::string>(
                row, "provenance_identity_hash"));
        if (request.identity.contractVersion() !=
                row["request_contract_version"].as<int>() ||
            request.identity.canonicalText() !=
                row["request_identity_canonical"].as<std::string>() ||
            request.identity.hash() !=
                row["request_identity_hash"].as<std::string>())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_request_identity_mismatch");
        const auto reservation =
            FindReservation(transaction, request.reservationId);
        const auto authorization = FindOperationalAuthorizationEvent(
            transaction, request.acceptingAuthorizationEventId);
        if (!reservation ||
            reservation->reservation.logicalOperation !=
                request.logicalOperation ||
            reservation->reservation.acceptingAuthorizationEventId !=
                request.acceptingAuthorizationEventId ||
            reservation->reservation.acceptingAuthorizationCanonicalText !=
                request.acceptingAuthorizationCanonicalText ||
            reservation->reservation.acceptingAuthorizationIdentityHash !=
                request.acceptingAuthorizationIdentityHash ||
            reservation->reservation.identity.canonicalText() !=
                request.reservationCanonicalText ||
            reservation->reservation.identity.hash() !=
                request.reservationIdentityHash)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_request_reservation_mismatch");
        if (!authorization ||
            authorization->event.campaignId !=
                request.logicalOperation.campaignId ||
            authorization->event.campaignCanonicalText !=
                request.logicalOperation.campaignCanonicalText ||
            authorization->event.identity.canonicalText() !=
                request.acceptingAuthorizationCanonicalText ||
            authorization->event.identity.hash() !=
                request.acceptingAuthorizationIdentityHash ||
            authorization->event.actionKind !=
                request.logicalOperation.actionKind ||
            authorization->event.actionContractVersion !=
                request.logicalOperation.actionContractVersion ||
            authorization->event.scopeKind !=
                request.logicalOperation.scopeKind ||
            authorization->event.scopeContractVersion !=
                request.logicalOperation.scopeContractVersion ||
            authorization->event.prerequisitePolicy !=
                request.prerequisitePolicy ||
            authorization->event.provenanceCanonicalText !=
                request.provenanceCanonicalText ||
            authorization->event.provenanceIdentityHash !=
                request.provenanceIdentityHash)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_request_authorization_evidence_mismatch");
        return PersistedOperationalRequest(
            OperationalRequestId(
                row["operational_request_id"].as<long long>()),
            std::move(request),
            RequestStateFromText(
                row["request_state"].as<std::string>()),
            row["state_version"].as<int>(),
            row["production_dispatch_enabled"].as<bool>(),
            row["created_at"].as<std::string>(),
            row["updated_at"].as<std::string>());
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_request:" +
                std::string(error.what()));
    }
}

PersistedReservationEvent MapReservationEvent(
    const pqxx::row& row)
{
    try
    {
        if (ReservationEventKindFromText(
                row["transition_kind"].as<std::string>()) !=
            ReservationEventKind::acquired)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_reservation_event_phase_unsupported");
        ReservationEvent event = BuildReservationAcquisitionEvent(
            ReservationId(row["reservation_id"].as<long long>()),
            row["reservation_identity_canonical"].as<std::string>(),
            OperationalRequestId(
                row["operational_request_id"].as<long long>()),
            row["request_identity_canonical"].as<std::string>(),
            row["amount"].as<long long>());
        if (event.identity.contractVersion() !=
                row["reservation_event_contract_version"].as<int>() ||
            event.identity.canonicalText() !=
                row["reservation_event_identity_canonical"]
                    .as<std::string>() ||
            event.identity.hash() !=
                row["reservation_event_identity_hash"].as<std::string>() ||
            !row["expected_state"].is_null() ||
            ReservationStateFromText(
                row["resulting_state"].as<std::string>()) !=
                event.resultingState ||
            row["expected_version"].as<int>() != event.expectedVersion ||
            row["resulting_version"].as<int>() != event.resultingVersion)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_reservation_event_identity_mismatch");
        return PersistedReservationEvent(
            ReservationEventId(
                row["reservation_event_id"].as<long long>()),
            std::move(event), row["created_at"].as<std::string>());
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::persistenceCorruption,
            "invalid_persisted_campaign_operations_reservation_event:" +
                std::string(error.what()));
    }
}

void InsertBudgetAudit(pqxx::transaction_base& transaction,
    const PersistedBudgetLedgerEntry& persisted)
{
    transaction.exec(
        "INSERT INTO campaign_operations_audit_reference_event ("
        "operational_campaign_id,budget_ledger_entry_id,cause_kind,"
        "actor_identity,capability,reason,prior_version,resulting_version,"
        "outcome,replay_disposition) "
        "VALUES($1,$2,'budget_ledger_recorded',$3,"
        "'campaign_operations_budget_administrator',$4,$5,$6,"
        "'recorded','recorded');",
        pqxx::params{persisted.entry.campaignId.value(),
            persisted.budgetLedgerEntryId.value(),
            persisted.entry.administrator.value(),
            persisted.entry.reason.value(),
            persisted.entry.ledgerVersion > 1
                ? std::optional<int>(persisted.entry.ledgerVersion - 1)
                : std::nullopt,
            persisted.entry.ledgerVersion});
}

void InsertAcceptanceAudit(pqxx::transaction_base& transaction,
    const AcceptedOperationalRequest& accepted)
{
    transaction.exec(
        "INSERT INTO campaign_operations_audit_reference_event ("
        "operational_campaign_id,authorization_event_id,"
        "budget_ledger_entry_id,reservation_id,reservation_event_id,"
        "operational_request_id,cause_kind,actor_identity,capability,reason,"
        "prior_version,resulting_version,outcome,replay_disposition) "
        "VALUES($1,$2,$3,$4,$5,$6,'reservation_request_accepted',$7,"
        "'campaign_operations_request_acceptor',$8,NULL,1,"
        "'recorded','recorded');",
        pqxx::params{
            accepted.request.request.logicalOperation.campaignId.value(),
            accepted.request.request.acceptingAuthorizationEventId.value(),
            accepted.reservation.reservation.budgetLedgerEntryId.value(),
            accepted.reservation.reservationId.value(),
            accepted.acquisitionEvent.reservationEventId.value(),
            accepted.request.requestId.value(),
            accepted.request.request.acceptingActor.value(),
            accepted.request.request.reason.value()});
}

} // namespace

bool BudgetRequestSchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT "
        "to_regclass('campaign_operations_budget_ledger_entry') IS NOT NULL "
        "AND to_regclass('campaign_operations_reservation') IS NOT NULL "
        "AND to_regclass('campaign_operations_operational_request') IS NOT NULL "
        "AND to_regclass('campaign_operations_reservation_event') IS NOT NULL "
        "AND to_regclass('campaign_operations_budget_status_v1') IS NOT NULL "
        "AND to_regclass('campaign_operations_request_status_v1') IS NOT NULL;")
        .one_row()[0]
        .as<bool>();
}

void LockOperationalAuthorizationDomain(
    pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign)
{
    const std::string lockKey = campaign.identity.canonicalText() +
        ";action=" + ToText(campaign.actionKind) +
        ";scope=" + ToText(campaign.scopeKind);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030003));",
        pqxx::params{lockKey});
}

void LockBudgetDomain(pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign)
{
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030004));",
        pqxx::params{campaign.identity.canonicalText()});
}

PersistedOperationalCampaign LockOperationalCampaign(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    transaction.exec(
        "SELECT lock_campaign_operations_campaign($1);",
        pqxx::params{campaignId.value()});
    const pqxx::result rows = transaction.exec(
        "SELECT " + CampaignColumns() +
            " FROM campaign_operations_campaign WHERE "
            "operational_campaign_id=$1;",
        pqxx::params{campaignId.value()});
    if (rows.empty())
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    return MapCampaign(transaction, rows.one_row());
}

UtcTimestamp CurrentDatabaseTime(pqxx::transaction_base& transaction)
{
    return UtcTimestamp(transaction.exec(
        "SELECT to_char(transaction_timestamp() AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');")
            .one_row()[0]
            .as<std::string>());
}

std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerEntry(
    pqxx::transaction_base& transaction,
    BudgetLedgerEntryId budgetLedgerEntryId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + BudgetColumns() +
            " FROM campaign_operations_budget_ledger_entry WHERE "
            "budget_ledger_entry_id=$1;",
        pqxx::params{budgetLedgerEntryId.value()});
    if (rows.empty()) return std::nullopt;
    return MapBudgetEntry(transaction, rows.one_row());
}

std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerHead(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + BudgetColumns() +
            " FROM campaign_operations_budget_ledger_entry WHERE "
            "operational_campaign_id=$1 ORDER BY ledger_version DESC LIMIT 1;",
        pqxx::params{campaignId.value()});
    if (rows.empty()) return std::nullopt;
    return MapBudgetEntry(transaction, rows.one_row());
}

std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerEntryByVersion(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    int ledgerVersion)
{
    if (ledgerVersion <= 0)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_budget_version_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + BudgetColumns() +
            " FROM campaign_operations_budget_ledger_entry WHERE "
            "operational_campaign_id=$1 AND ledger_version=$2;",
        pqxx::params{campaignId.value(), ledgerVersion});
    if (rows.empty()) return std::nullopt;
    return MapBudgetEntry(transaction, rows.one_row());
}

BudgetAccounting LoadBudgetAccounting(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const auto head = FindBudgetLedgerHead(transaction, campaignId);
    if (!head)
        throw Error(ErrorCode::budgetDenied,
            "campaign_operations_budget_not_found");
    const pqxx::row row = transaction.exec(
        "SELECT coalesce(sum(amount),0)::bigint AS ever_reserved,"
        "coalesce(sum(amount) FILTER (WHERE reservation_state='committed'),"
        "0)::bigint AS committed,"
        "coalesce(sum(amount) FILTER (WHERE reservation_state IN "
        "('released','expired')),0)::bigint AS released_or_expired,"
        "count(*)::bigint AS reservation_count,"
        "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM "
        "campaign_operations_reservation_event event WHERE "
        "event.reservation_id=reservation.reservation_id AND "
        "event.transition_kind='acquired'))::bigint AS acquired_count,"
        "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM "
        "campaign_operations_operational_request request WHERE "
        "request.reservation_id=reservation.reservation_id))::bigint "
        "AS request_count FROM campaign_operations_reservation reservation "
        "WHERE operational_campaign_id=$1;",
        pqxx::params{campaignId.value()})
                              .one_row();
    if (row["reservation_count"].as<long long>() !=
            row["acquired_count"].as<long long>() ||
        row["reservation_count"].as<long long>() !=
            row["request_count"].as<long long>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reservation_reconciliation_required");
    return CalculateBudgetAccounting(head->entry.resultingTotal,
        row["ever_reserved"].as<long long>(),
        row["committed"].as<long long>(),
        row["released_or_expired"].as<long long>(), head->entry.status);
}

PersistResult<PersistedBudgetLedgerEntry> PersistBudgetLedgerEntry(
    pqxx::transaction_base& transaction, const BudgetLedgerEntry& entry)
{
    ValidateBudgetLedgerEntry(entry);
    RequireCapability(
        transaction, "campaign_operations_budget_administrator");
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        " 1179402835030004));",
        pqxx::params{entry.campaignCanonicalText});
    const auto campaign = LockOperationalCampaign(
        transaction, entry.campaignId);
    if (campaign.campaign.identity.canonicalText() !=
        entry.campaignCanonicalText)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_budget_campaign_conflict");
    if (const auto existing = FindBudgetLedgerEntryByVersion(
            transaction, entry.campaignId, entry.ledgerVersion))
    {
        if (existing->entry != entry)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_budget_conflict");
        return {PersistOutcome::existingIdentical, std::move(*existing)};
    }
    const auto head = FindBudgetLedgerHead(transaction, entry.campaignId);
    if ((!head && (entry.ledgerVersion != 1 || entry.previousEntryId)) ||
        (head && (entry.ledgerVersion != head->entry.ledgerVersion + 1 ||
            !entry.previousEntryId ||
            *entry.previousEntryId != head->budgetLedgerEntryId ||
            *entry.previousEntryCanonicalText !=
                head->entry.identity.canonicalText() ||
            *entry.previousEntryIdentityHash !=
                head->entry.identity.hash() ||
            entry.priorTotal != head->entry.resultingTotal)))
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_budget_stale_head");
    long long committed = 0;
    long long held = 0;
    if (head)
    {
        const auto accounting =
            LoadBudgetAccounting(transaction, entry.campaignId);
        committed = accounting.committed;
        held = accounting.held;
    }
    if ((entry.entryKind == BudgetLedgerEntryKind::amend &&
            (!head || head->entry.status != BudgetLedgerStatus::active)) ||
        (entry.entryKind == BudgetLedgerEntryKind::revoke &&
            (!head || head->entry.status != BudgetLedgerStatus::active ||
                entry.resultingTotal != committed + held)) ||
        (entry.entryKind == BudgetLedgerEntryKind::supersede &&
            (!head || head->entry.status != BudgetLedgerStatus::revoked)) ||
        entry.resultingTotal < committed + held)
        throw Error(ErrorCode::budgetDenied,
            "campaign_operations_budget_transition_denied");
    const long long entryId = transaction.exec(
        "INSERT INTO campaign_operations_budget_ledger_entry ("
        "operational_campaign_id,campaign_identity_canonical,"
        "previous_entry_id,previous_entry_identity_canonical,"
        "previous_entry_identity_hash,ledger_version,entry_kind,"
        "ledger_status,budget_unit,delta,prior_total,resulting_total,"
        "administrator_identity,reason,budget_contract_version,"
        "budget_identity_canonical,budget_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,"
        "$16,$17) RETURNING budget_ledger_entry_id;",
        pqxx::params{entry.campaignId.value(), entry.campaignCanonicalText,
            entry.previousEntryId
                ? std::optional<long long>(entry.previousEntryId->value())
                : std::nullopt,
            entry.previousEntryCanonicalText,
            entry.previousEntryIdentityHash, entry.ledgerVersion,
            ToText(entry.entryKind), ToText(entry.status), ToText(entry.unit),
            entry.delta, entry.priorTotal, entry.resultingTotal,
            entry.administrator.value(), entry.reason.value(),
            entry.identity.contractVersion(), entry.identity.canonicalText(),
            entry.identity.hash()})
                                  .one_row()[0]
                                  .as<long long>();
    auto persisted = FindBudgetLedgerEntry(
        transaction, BudgetLedgerEntryId(entryId));
    if (!persisted)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_budget_insert_missing");
    InsertBudgetAudit(transaction, *persisted);
    return {PersistOutcome::recorded, std::move(*persisted)};
}

std::optional<PersistedReservation> FindReservation(
    pqxx::transaction_base& transaction, ReservationId reservationId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReservationColumns() +
            " FROM campaign_operations_reservation WHERE reservation_id=$1;",
        pqxx::params{reservationId.value()});
    if (rows.empty()) return std::nullopt;
    return MapReservation(transaction, rows.one_row());
}

std::optional<PersistedOperationalRequest> FindOperationalRequest(
    pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + RequestColumns() +
            " FROM campaign_operations_operational_request WHERE "
            "operational_request_id=$1;",
        pqxx::params{requestId.value()});
    if (rows.empty()) return std::nullopt;
    return MapRequest(transaction, rows.one_row());
}

std::optional<PersistedOperationalRequest>
FindOperationalRequestByCampaignAction(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    OperationalActionKind actionKind, int actionContractVersion)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + RequestColumns() +
            " FROM campaign_operations_operational_request WHERE "
            "operational_campaign_id=$1 AND action_kind=$2 AND "
            "action_contract_version=$3;",
        pqxx::params{campaignId.value(), ToText(actionKind),
            actionContractVersion});
    if (rows.empty()) return std::nullopt;
    return MapRequest(transaction, rows.one_row());
}

std::optional<PersistedReservationEvent> FindReservationEvent(
    pqxx::transaction_base& transaction,
    ReservationEventId reservationEventId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReservationEventColumns() +
            " FROM campaign_operations_reservation_event WHERE "
            "reservation_event_id=$1;",
        pqxx::params{reservationEventId.value()});
    if (rows.empty()) return std::nullopt;
    return MapReservationEvent(rows.one_row());
}

std::optional<PersistedReservationEvent> FindReservationAcquisitionEvent(
    pqxx::transaction_base& transaction, ReservationId reservationId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReservationEventColumns() +
            " FROM campaign_operations_reservation_event WHERE "
            "reservation_id=$1 AND transition_kind='acquired';",
        pqxx::params{reservationId.value()});
    if (rows.empty()) return std::nullopt;
    return MapReservationEvent(rows.one_row());
}

std::optional<CampaignBudgetStatusProjection>
FindCampaignBudgetStatusProjection(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT operational_campaign_id,budget_ledger_entry_id,"
        "ledger_version,ledger_status,coalesce(granted,0)::bigint AS granted,"
        "ever_reserved,committed,released_or_expired,held,unallocated,"
        "reservable,budget_identity_hash,accounting_consistent "
        "FROM campaign_operations_budget_status_v1 WHERE "
        "operational_campaign_id=$1;",
        pqxx::params{campaignId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    CampaignBudgetStatusProjection value;
    value.campaignId = row["operational_campaign_id"].as<long long>();
    value.budgetLedgerEntryId =
        OptionalValue<long long>(row, "budget_ledger_entry_id");
    value.ledgerVersion = OptionalValue<int>(row, "ledger_version");
    const auto status = OptionalValue<std::string>(row, "ledger_status");
    if (status) value.status = BudgetLedgerStatusFromText(*status);
    value.granted = row["granted"].as<long long>();
    value.everReserved = row["ever_reserved"].as<long long>();
    value.committed = row["committed"].as<long long>();
    value.releasedOrExpired =
        row["released_or_expired"].as<long long>();
    value.held = row["held"].as<long long>();
    value.unallocated = row["unallocated"].as<long long>();
    value.reservable = row["reservable"].as<long long>();
    value.budgetIdentityHash =
        OptionalValue<std::string>(row, "budget_identity_hash");
    value.accountingConsistent = row["accounting_consistent"].as<bool>();
    return value;
}

std::optional<OperationalRequestStatusProjection>
FindOperationalRequestStatusProjection(pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT operational_request_id,operational_campaign_id,"
        "request_state,request_state_version,request_identity_hash,"
        "production_dispatch_enabled,reservation_id,reservation_state,"
        "reservation_state_version,amount,budget_unit,"
        "CASE WHEN expires_at IS NULL THEN NULL ELSE "
        "to_char(expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END AS expires_at_text,"
        "reservation_identity_hash,authorization_event_id,"
        "budget_ledger_entry_id,budget_ledger_version,evidence_consistent "
        "FROM campaign_operations_request_status_v1 WHERE "
        "operational_request_id=$1;",
        pqxx::params{requestId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    OperationalRequestStatusProjection value;
    value.requestId = row["operational_request_id"].as<long long>();
    value.campaignId = row["operational_campaign_id"].as<long long>();
    value.requestState =
        RequestStateFromText(row["request_state"].as<std::string>());
    value.requestStateVersion = row["request_state_version"].as<int>();
    value.requestIdentityHash =
        row["request_identity_hash"].as<std::string>();
    value.productionDispatchEnabled =
        row["production_dispatch_enabled"].as<bool>();
    value.reservationId = row["reservation_id"].as<long long>();
    value.reservationState =
        ReservationStateFromText(
            row["reservation_state"].as<std::string>());
    value.reservationStateVersion =
        row["reservation_state_version"].as<int>();
    value.amount = row["amount"].as<long long>();
    value.budgetUnit =
        BudgetUnitFromText(row["budget_unit"].as<std::string>());
    value.expiresAt =
        OptionalValue<std::string>(row, "expires_at_text");
    value.reservationIdentityHash =
        row["reservation_identity_hash"].as<std::string>();
    value.authorizationEventId =
        row["authorization_event_id"].as<long long>();
    value.budgetLedgerEntryId =
        row["budget_ledger_entry_id"].as<long long>();
    value.budgetLedgerVersion =
        row["budget_ledger_version"].as<int>();
    value.evidenceConsistent = row["evidence_consistent"].as<bool>();
    return value;
}

PersistResult<AcceptedOperationalRequest> PersistAcceptedOperationalRequest(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    const ActorIdentity& actor, const Reason& reason,
    const std::optional<UtcTimestamp>& expiresAt)
{
    RequireCapability(transaction, "campaign_operations_request_acceptor");
    const auto initialCampaign =
        FindOperationalCampaign(transaction, campaignId);
    if (!initialCampaign)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    LockOperationalAuthorizationDomain(
        transaction, initialCampaign->campaign);
    LockBudgetDomain(transaction, initialCampaign->campaign);
    const auto campaign =
        LockOperationalCampaign(transaction, campaignId);
    if (campaign.campaign != initialCampaign->campaign)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_campaign_changed");
    const LogicalOperation operation =
        BuildLogicalOperation(campaignId, campaign.campaign);
    const auto authorization = FindOperationalAuthorizationHead(transaction,
        campaignId, operation.actionKind, operation.actionContractVersion,
        operation.scopeKind, operation.scopeContractVersion);
    const auto budget = FindBudgetLedgerHead(transaction, campaignId);

    if (auto existing = FindOperationalRequestByCampaignAction(
            transaction, campaignId, operation.actionKind,
            operation.actionContractVersion))
    {
        const auto reservation =
            FindReservation(transaction, existing->request.reservationId);
        const auto acquisition = reservation
            ? FindReservationAcquisitionEvent(
                  transaction, reservation->reservationId)
            : std::nullopt;
        if (!reservation || !acquisition ||
            existing->request.logicalOperation != operation ||
            existing->request.acceptingActor != actor ||
            existing->request.reason != reason ||
            reservation->reservation.expiresAt != expiresAt ||
            acquisition->event.reservationId !=
                reservation->reservationId ||
            acquisition->event.reservationCanonicalText !=
                reservation->reservation.identity.canonicalText() ||
            acquisition->event.requestId != existing->requestId ||
            acquisition->event.requestCanonicalText !=
                existing->request.identity.canonicalText() ||
            acquisition->event.amount != reservation->reservation.amount ||
            !authorization ||
            existing->request.acceptingAuthorizationEventId !=
                authorization->authorizationEventId ||
            existing->request.acceptingAuthorizationCanonicalText !=
                authorization->event.identity.canonicalText() ||
            existing->request.acceptingAuthorizationIdentityHash !=
                authorization->event.identity.hash() ||
            existing->request.prerequisitePolicy !=
                authorization->event.prerequisitePolicy ||
            existing->request.provenanceCanonicalText !=
                authorization->event.provenanceCanonicalText ||
            existing->request.provenanceIdentityHash !=
                authorization->event.provenanceIdentityHash ||
            !budget ||
            reservation->reservation.budgetLedgerEntryId !=
                budget->budgetLedgerEntryId ||
            reservation->reservation.budgetLedgerVersion !=
                budget->entry.ledgerVersion ||
            reservation->reservation.budgetLedgerCanonicalText !=
                budget->entry.identity.canonicalText() ||
            reservation->reservation.budgetLedgerIdentityHash !=
                budget->entry.identity.hash())
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_logical_operation_payload_conflict");
        return {PersistOutcome::existingIdentical,
            AcceptedOperationalRequest(
                *reservation, std::move(*existing), *acquisition)};
    }

    const UtcTimestamp databaseTime = CurrentDatabaseTime(transaction);
    if (!authorization ||
        !IsAuthorizationEffectiveAt(authorization->event, databaseTime))
        throw Error(ErrorCode::authorizationDenied,
            "campaign_operations_authorization_denied");
    if (expiresAt && !(*expiresAt > databaseTime))
        throw Error(ErrorCode::invalidReservation,
            "campaign_operations_reservation_expiry_invalid");
    if (!budget || budget->entry.status != BudgetLedgerStatus::active)
        throw Error(ErrorCode::budgetDenied,
            "campaign_operations_budget_inactive");
    const auto accounting = LoadBudgetAccounting(transaction, campaignId);
    if (campaign.campaign.memberCount > accounting.reservable)
        throw Error(ErrorCode::budgetDenied,
            "insufficient_materialized_member_dispatch_units");

    Reservation reservation = BuildReservation(operation,
        authorization->authorizationEventId,
        authorization->event.identity.canonicalText(),
        authorization->event.identity.hash(), budget->budgetLedgerEntryId,
        budget->entry.ledgerVersion,
        budget->entry.identity.canonicalText(), budget->entry.identity.hash(),
        campaign.campaign.memberCount, campaign.campaign.memberCount,
        BudgetUnit::materializedMemberDispatch, expiresAt);
    const long long reservationId = transaction.exec(
        "INSERT INTO campaign_operations_reservation ("
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "budget_ledger_entry_id,budget_ledger_version,"
        "budget_identity_canonical,budget_identity_hash,action_kind,"
        "action_contract_version,recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,scope_kind,scope_contract_version,"
        "materialization_member_count,amount,budget_unit,expires_at,"
        "reservation_contract_version,reservation_identity_canonical,"
        "reservation_identity_hash,reservation_state,state_version) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
        "$17,$18,$19,$20,$21,$22,$23,$24::timestamptz,$25,$26,$27,"
        "'held',1) RETURNING reservation_id;",
        pqxx::params{campaignId.value(),
            campaign.campaign.identity.canonicalText(),
            operation.identity.contractVersion(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization->authorizationEventId.value(),
            authorization->event.identity.canonicalText(),
            authorization->event.identity.hash(),
            budget->budgetLedgerEntryId.value(),
            budget->entry.ledgerVersion,
            budget->entry.identity.canonicalText(),
            budget->entry.identity.hash(), ToText(operation.actionKind),
            operation.actionContractVersion, operation.materializationId,
            operation.materializationContractVersion,
            operation.materializationCanonicalText,
            operation.materializationIdentityHash,
            ToText(operation.scopeKind), operation.scopeContractVersion,
            campaign.campaign.memberCount, campaign.campaign.memberCount,
            ToText(BudgetUnit::materializedMemberDispatch),
            expiresAt
                ? std::optional<std::string>(expiresAt->value())
                : std::nullopt,
            reservation.identity.contractVersion(),
            reservation.identity.canonicalText(),
            reservation.identity.hash()})
                                        .one_row()[0]
                                        .as<long long>();
    OperationalRequest request = BuildOperationalRequest(operation,
        authorization->authorizationEventId,
        authorization->event.identity.canonicalText(),
        authorization->event.identity.hash(), ReservationId(reservationId),
        reservation.identity.canonicalText(), reservation.identity.hash(),
        campaign.campaign.memberCount,
        campaign.campaign.materializationIdentityHash, actor, reason,
        authorization->event.prerequisitePolicy,
        authorization->event.provenanceCanonicalText,
        authorization->event.provenanceIdentityHash);
    const long long requestId = transaction.exec(
        "INSERT INTO campaign_operations_operational_request ("
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "reservation_id,reservation_identity_canonical,"
        "reservation_identity_hash,action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,ordered_scope_digest,"
        "materialization_member_count,accepting_actor_identity,reason,"
        "prerequisite_policy,provenance_identity_canonical,"
        "provenance_identity_hash,request_contract_version,"
        "request_identity_canonical,request_identity_hash,request_state,"
        "state_version) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,"
        "$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,"
        "'ready',1) RETURNING operational_request_id;",
        pqxx::params{campaignId.value(),
            campaign.campaign.identity.canonicalText(),
            operation.identity.contractVersion(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization->authorizationEventId.value(),
            authorization->event.identity.canonicalText(),
            authorization->event.identity.hash(), reservationId,
            reservation.identity.canonicalText(),
            reservation.identity.hash(), ToText(operation.actionKind),
            operation.actionContractVersion, operation.materializationId,
            operation.materializationContractVersion,
            operation.materializationCanonicalText,
            operation.materializationIdentityHash,
            campaign.campaign.materializationIdentityHash,
            campaign.campaign.memberCount, actor.value(), reason.value(),
            ToText(authorization->event.prerequisitePolicy),
            authorization->event.provenanceCanonicalText,
            authorization->event.provenanceIdentityHash,
            request.identity.contractVersion(),
            request.identity.canonicalText(), request.identity.hash()})
                                    .one_row()[0]
                                    .as<long long>();
    ReservationEvent event = BuildReservationAcquisitionEvent(
        ReservationId(reservationId), reservation.identity.canonicalText(),
        OperationalRequestId(requestId), request.identity.canonicalText(),
        reservation.amount);
    const long long eventId = transaction.exec(
        "INSERT INTO campaign_operations_reservation_event ("
        "reservation_id,reservation_identity_canonical,transition_kind,"
        "expected_state,resulting_state,expected_version,resulting_version,"
        "operational_request_id,request_identity_canonical,amount,"
        "reservation_event_contract_version,"
        "reservation_event_identity_canonical,"
        "reservation_event_identity_hash) "
        "VALUES($1,$2,'acquired',NULL,'held',0,1,$3,$4,$5,$6,$7,$8) "
        "RETURNING reservation_event_id;",
        pqxx::params{reservationId, reservation.identity.canonicalText(),
            requestId, request.identity.canonicalText(), reservation.amount,
            event.identity.contractVersion(), event.identity.canonicalText(),
            event.identity.hash()})
                                  .one_row()[0]
                                  .as<long long>();
    auto persistedReservation =
        FindReservation(transaction, ReservationId(reservationId));
    auto persistedRequest =
        FindOperationalRequest(transaction, OperationalRequestId(requestId));
    auto persistedEvent =
        FindReservationEvent(transaction, ReservationEventId(eventId));
    if (!persistedReservation || !persistedRequest || !persistedEvent)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_acceptance_insert_missing");
    AcceptedOperationalRequest accepted(
        std::move(*persistedReservation), std::move(*persistedRequest),
        std::move(*persistedEvent));
    InsertAcceptanceAudit(transaction, accepted);
    return {PersistOutcome::recorded, std::move(accepted)};
}

} // namespace EA::CampaignOperations
