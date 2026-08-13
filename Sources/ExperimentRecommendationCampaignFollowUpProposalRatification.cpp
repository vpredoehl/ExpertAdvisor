#include "ExperimentRecommendationCampaignFollowUpProposalRatification.hpp"

#include "ExperimentRecommendation.hpp"

#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

bool IsAsciiAlphanumeric(unsigned char character)
{
    return (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9');
}

bool IsValidGovernanceActorIdentity(const std::string& value)
{
    if (value.empty() ||
        value.size() >
            kMaximumRecommendationCampaignFollowUpProposalRatifierIdentityBytes ||
        !IsAsciiAlphanumeric(static_cast<unsigned char>(value.front())))
        return false;
    for (const unsigned char character : value)
    {
        if (!IsAsciiAlphanumeric(character) && character != '.' &&
            character != '_' && character != '@' && character != ':' &&
            character != '/' && character != '+' && character != '-')
            return false;
    }
    return true;
}

bool IsValidUtf8(const std::string& value)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(value.data());
    std::size_t index = 0;
    while (index < value.size())
    {
        const unsigned char first = bytes[index];
        if (first <= 0x7fU)
        {
            ++index;
            continue;
        }
        std::size_t continuationCount = 0;
        unsigned int codePoint = 0;
        if (first >= 0xc2U && first <= 0xdfU)
        {
            continuationCount = 1;
            codePoint = first & 0x1fU;
        }
        else if (first >= 0xe0U && first <= 0xefU)
        {
            continuationCount = 2;
            codePoint = first & 0x0fU;
        }
        else if (first >= 0xf0U && first <= 0xf4U)
        {
            continuationCount = 3;
            codePoint = first & 0x07U;
        }
        else return false;
        if (index + continuationCount >= value.size()) return false;
        for (std::size_t offset = 1; offset <= continuationCount; ++offset)
        {
            const unsigned char next = bytes[index + offset];
            if ((next & 0xc0U) != 0x80U) return false;
            codePoint = (codePoint << 6U) | (next & 0x3fU);
        }
        if ((continuationCount == 2 && codePoint < 0x800U) ||
            (continuationCount == 3 && codePoint < 0x10000U) ||
            (codePoint >= 0xd800U && codePoint <= 0xdfffU) ||
            codePoint > 0x10ffffU)
            return false;
        index += continuationCount + 1;
    }
    return true;
}

bool IsValidBasis(const std::string& value)
{
    if (value.empty() ||
        value.size() >
            kMaximumRecommendationCampaignFollowUpProposalRatificationBasisBytes ||
        !IsValidUtf8(value))
        return false;
    bool containsNonWhitespace = false;
    for (const unsigned char character : value)
    {
        if (character == 0U || character == 0x7fU ||
            (character < 0x20U && character != '\t' && character != '\n' &&
                character != '\r'))
            return false;
        if (character >= 0x80U ||
            (character != ' ' && character != '\t' && character != '\n' &&
                character != '\r'))
            containsNonWhitespace = true;
    }
    return containsNonWhitespace;
}

std::string LengthPrefixed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

bool ReviewIdentityMatchesPayload(
    int reviewContractVersion,
    const std::string& reviewCanonicalText,
    const std::string& reviewIdentityHash,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
    const std::string& reviewerIdentity)
{
    const std::string marker =
        ";reviewer=" + LengthPrefixed(reviewerIdentity) + ";reason=";
    std::size_t markerPosition = reviewCanonicalText.find(marker);
    while (markerPosition != std::string::npos)
    {
        const std::size_t lengthPosition = markerPosition + marker.size();
        const std::size_t colonPosition =
            reviewCanonicalText.find(':', lengthPosition);
        if (colonPosition != std::string::npos &&
            colonPosition > lengthPosition)
        {
            std::size_t reasonLength = 0U;
            bool validLength = true;
            for (std::size_t index = lengthPosition;
                 index < colonPosition; ++index)
            {
                const unsigned char character =
                    static_cast<unsigned char>(reviewCanonicalText[index]);
                if (character < '0' || character > '9')
                {
                    validLength = false;
                    break;
                }
                const std::size_t digit = character - '0';
                if (reasonLength >
                    (std::numeric_limits<std::size_t>::max() - digit) / 10U)
                {
                    validLength = false;
                    break;
                }
                reasonLength = reasonLength * 10U + digit;
            }
            const std::size_t reasonPosition = colonPosition + 1U;
            if (validLength && reasonLength ==
                    reviewCanonicalText.size() - reasonPosition)
            {
                try
                {
                    const auto review =
                        BuildRecommendationCampaignFollowUpProposalReview(
                            reviewContractVersion, followUpProposalId,
                            proposalContractVersion, proposalCanonicalText,
                            proposalIdentityHash, reviewDecision,
                            reviewerIdentity,
                            reviewCanonicalText.substr(reasonPosition));
                    if (review.identity.canonicalText == reviewCanonicalText &&
                        review.identity.hash == reviewIdentityHash)
                        return true;
                }
                catch (const std::exception&)
                {
                }
            }
        }
        markerPosition = reviewCanonicalText.find(
            marker, markerPosition + 1U);
    }
    return false;
}

std::string RatificationCanonicalText(
    long long reviewEventId,
    int reviewContractVersion,
    const std::string& reviewCanonicalText,
    const std::string& reviewIdentityHash,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
    const std::string& reviewerIdentity,
    const std::string& ratificationAuthorityRole,
    RecommendationCampaignFollowUpProposalRatificationDecision decision,
    const std::string& ratifierIdentity,
    const std::string& ratificationBasis)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output
        << "experiment_recommendation_campaign_follow_up_proposal_ratification_v1"
        << ";ratification_contract_version="
        << kRecommendationCampaignFollowUpProposalRatificationContractVersion
        << ";read_only=true"
           ";database_free=true"
           ";persistent=false"
           ";governance_ratification=true"
           ";ratifies_advancement_to_next_separately_controlled_phase=true"
           ";eligible_review_required=true"
           ";separation_of_duties_required=true"
           ";phase_6e_capability_granted=false"
           ";activated=false"
           ";execution_authorized=false"
           ";follow_up_authorized=false"
           ";queued=false"
           ";scheduled=false"
           ";scheduler_started=false"
           ";scheduler_signaled=false"
           ";workers_started=false"
           ";experiments_created=false"
           ";experiments_modified=false"
           ";campaign_success_declared=false"
        << ";review_event_id=" << reviewEventId
        << ";review_contract_version=" << reviewContractVersion
        << ";review_canonical=" << LengthPrefixed(reviewCanonicalText)
        << ";review_identity_hash=" << LengthPrefixed(reviewIdentityHash)
        << ";review_decision="
        << RecommendationCampaignFollowUpProposalReviewDecisionText(
               reviewDecision)
        << ";reviewer=" << LengthPrefixed(reviewerIdentity)
        << ";follow_up_proposal_id=" << followUpProposalId
        << ";proposal_contract_version=" << proposalContractVersion
        << ";proposal_canonical=" << LengthPrefixed(proposalCanonicalText)
        << ";proposal_identity_hash=" << LengthPrefixed(proposalIdentityHash)
        << ";ratification_authority_role="
        << LengthPrefixed(ratificationAuthorityRole)
        << ";ratification_decision="
        << RecommendationCampaignFollowUpProposalRatificationDecisionText(decision)
        << ";ratifier=" << LengthPrefixed(ratifierIdentity)
        << ";basis=" << LengthPrefixed(ratificationBasis);
    return output.str();
}

} // namespace

void ValidateRecommendationCampaignFollowUpProposalRatifierIdentity(
    const std::string& ratifierIdentity)
{
    if (!IsValidGovernanceActorIdentity(ratifierIdentity))
        throw std::invalid_argument("ratifier_identity_invalid");
}

void ValidateRecommendationCampaignFollowUpProposalRatificationBasis(
    const std::string& ratificationBasis)
{
    if (!IsValidBasis(ratificationBasis))
        throw std::invalid_argument("ratification_basis_invalid");
}

struct RecommendationCampaignFollowUpProposalRatificationBuilder
{
    static RecommendationCampaignFollowUpProposalRatification Build(
        int ratificationContractVersion,
        long long reviewEventId,
        int reviewContractVersion,
        const std::string& reviewCanonicalText,
        const std::string& reviewIdentityHash,
        long long followUpProposalId,
        int proposalContractVersion,
        const std::string& proposalCanonicalText,
        const std::string& proposalIdentityHash,
        RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
        std::string reviewerIdentity,
        std::string ratificationAuthorityRole,
        RecommendationCampaignFollowUpProposalRatificationDecision decision,
        std::string ratifierIdentity,
        std::string ratificationBasis)
    {
        if (ratificationContractVersion !=
            kRecommendationCampaignFollowUpProposalRatificationContractVersion)
            throw std::invalid_argument("ratification_contract_unsupported");
        if (reviewEventId <= 0)
            throw std::invalid_argument("persisted_review_event_id_invalid");
        if (reviewContractVersion !=
            kRecommendationCampaignFollowUpProposalReviewContractVersion)
            throw std::invalid_argument("review_contract_unsupported");
        if (reviewCanonicalText.empty() ||
            reviewCanonicalText.size() >
                kMaximumRecommendationCampaignFollowUpProposalReviewCanonicalTextBytes ||
            reviewCanonicalText.find('\0') != std::string::npos ||
            reviewIdentityHash != RecommendationCanonicalHash(
                reviewCanonicalText))
            throw std::invalid_argument("review_identity_invalid");
        (void)RecommendationCampaignFollowUpProposalReviewDecisionText(
            reviewDecision);
        if (reviewDecision !=
            RecommendationCampaignFollowUpProposalReviewDecision::approved)
            throw std::invalid_argument("review_not_eligible_for_ratification");
        if (!IsValidGovernanceActorIdentity(reviewerIdentity))
            throw std::invalid_argument("reviewer_identity_invalid");
        if (followUpProposalId <= 0)
            throw std::invalid_argument("persisted_proposal_id_invalid");
        if (proposalContractVersion !=
            kRecommendationCampaignFollowUpProposalContractVersion)
            throw std::invalid_argument("proposal_contract_unsupported");
        if (proposalCanonicalText.empty() ||
            proposalCanonicalText.size() >
                kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes ||
            proposalCanonicalText.find('\0') != std::string::npos ||
            proposalIdentityHash != RecommendationCanonicalHash(
                proposalCanonicalText))
            throw std::invalid_argument("proposal_identity_invalid");
        if (!ReviewIdentityMatchesPayload(reviewContractVersion,
                reviewCanonicalText, reviewIdentityHash, followUpProposalId,
                proposalContractVersion, proposalCanonicalText,
                proposalIdentityHash, reviewDecision, reviewerIdentity))
            throw std::invalid_argument("review_identity_invalid");
        (void)RecommendationCampaignFollowUpProposalRatificationDecisionText(
            decision);
        if (ratificationAuthorityRole !=
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole)
            throw std::invalid_argument(
                "ratification_authority_role_invalid");
        ValidateRecommendationCampaignFollowUpProposalRatifierIdentity(
            ratifierIdentity);
        if (ratifierIdentity == reviewerIdentity)
            throw std::invalid_argument(
                "ratification_separation_of_duties_violation");
        ValidateRecommendationCampaignFollowUpProposalRatificationBasis(
            ratificationBasis);

        std::string canonicalText = RatificationCanonicalText(reviewEventId,
            reviewContractVersion, reviewCanonicalText, reviewIdentityHash,
            followUpProposalId, proposalContractVersion,
            proposalCanonicalText, proposalIdentityHash, reviewDecision,
            reviewerIdentity, ratificationAuthorityRole, decision,
            ratifierIdentity, ratificationBasis);
        if (canonicalText.size() >
            kMaximumRecommendationCampaignFollowUpProposalRatificationCanonicalTextBytes)
            throw std::invalid_argument("ratification_canonical_size_exceeded");
        RecommendationCampaignFollowUpProposalRatificationIdentity identity(
            ratificationContractVersion, canonicalText,
            RecommendationCanonicalHash(canonicalText));
        return RecommendationCampaignFollowUpProposalRatification(
            std::move(identity), reviewEventId, reviewContractVersion,
            reviewCanonicalText, reviewIdentityHash, followUpProposalId,
            proposalContractVersion, proposalCanonicalText,
            proposalIdentityHash, reviewDecision, std::move(reviewerIdentity),
            std::move(ratificationAuthorityRole), decision,
            std::move(ratifierIdentity), std::move(ratificationBasis));
    }
};

RecommendationCampaignFollowUpProposalRatificationIdentity::
    RecommendationCampaignFollowUpProposalRatificationIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
}

RecommendationCampaignFollowUpProposalRatification::
    RecommendationCampaignFollowUpProposalRatification(
        RecommendationCampaignFollowUpProposalRatificationIdentity identityValue,
        long long reviewEventIdValue,
        int reviewContractVersionValue,
        std::string reviewCanonicalTextValue,
        std::string reviewIdentityHashValue,
        long long followUpProposalIdValue,
        int proposalContractVersionValue,
        std::string proposalCanonicalTextValue,
        std::string proposalIdentityHashValue,
        RecommendationCampaignFollowUpProposalReviewDecision reviewDecisionValue,
        std::string reviewerIdentityValue,
        std::string ratificationAuthorityRoleValue,
        RecommendationCampaignFollowUpProposalRatificationDecision decisionValue,
        std::string ratifierIdentityValue,
        std::string ratificationBasisValue)
    : identity(std::move(identityValue)),
      reviewEventId(reviewEventIdValue),
      reviewContractVersion(reviewContractVersionValue),
      reviewCanonicalText(std::move(reviewCanonicalTextValue)),
      reviewIdentityHash(std::move(reviewIdentityHashValue)),
      followUpProposalId(followUpProposalIdValue),
      proposalContractVersion(proposalContractVersionValue),
      proposalCanonicalText(std::move(proposalCanonicalTextValue)),
      proposalIdentityHash(std::move(proposalIdentityHashValue)),
      reviewDecision(reviewDecisionValue),
      reviewerIdentity(std::move(reviewerIdentityValue)),
      ratificationAuthorityRole(std::move(ratificationAuthorityRoleValue)),
      decision(decisionValue),
      ratifierIdentity(std::move(ratifierIdentityValue)),
      ratificationBasis(std::move(ratificationBasisValue))
{
}

RecommendationCampaignFollowUpProposalRatification
BuildRecommendationCampaignFollowUpProposalRatification(
    int ratificationContractVersion,
    long long reviewEventId,
    int reviewContractVersion,
    const std::string& reviewCanonicalText,
    const std::string& reviewIdentityHash,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
    std::string reviewerIdentity,
    std::string ratificationAuthorityRole,
    RecommendationCampaignFollowUpProposalRatificationDecision decision,
    std::string ratifierIdentity,
    std::string ratificationBasis)
{
    return RecommendationCampaignFollowUpProposalRatificationBuilder::Build(
        ratificationContractVersion, reviewEventId, reviewContractVersion,
        reviewCanonicalText, reviewIdentityHash, followUpProposalId,
        proposalContractVersion, proposalCanonicalText, proposalIdentityHash,
        reviewDecision, std::move(reviewerIdentity),
        std::move(ratificationAuthorityRole), decision,
        std::move(ratifierIdentity), std::move(ratificationBasis));
}

RecommendationCampaignFollowUpProposalRatification
BuildRecommendationCampaignFollowUpProposalRatification(
    long long reviewEventId,
    const RecommendationCampaignFollowUpProposalReview& review,
    std::string ratifierIdentity,
    std::string ratificationBasis)
{
    ValidateRecommendationCampaignFollowUpProposalReview(review);
    return BuildRecommendationCampaignFollowUpProposalRatification(
        kRecommendationCampaignFollowUpProposalRatificationContractVersion,
        reviewEventId, review.identity.contractVersion,
        review.identity.canonicalText, review.identity.hash,
        review.followUpProposalId, review.proposalContractVersion,
        review.proposalCanonicalText, review.proposalIdentityHash,
        review.decision, review.reviewerIdentity,
        kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
        RecommendationCampaignFollowUpProposalRatificationDecision::ratified,
        std::move(ratifierIdentity), std::move(ratificationBasis));
}

void ValidateRecommendationCampaignFollowUpProposalRatification(
    const RecommendationCampaignFollowUpProposalRatification& ratification)
{
    const auto rebuilt = BuildRecommendationCampaignFollowUpProposalRatification(
        ratification.identity.contractVersion, ratification.reviewEventId,
        ratification.reviewContractVersion, ratification.reviewCanonicalText,
        ratification.reviewIdentityHash, ratification.followUpProposalId,
        ratification.proposalContractVersion, ratification.proposalCanonicalText,
        ratification.proposalIdentityHash, ratification.reviewDecision,
        ratification.reviewerIdentity, ratification.ratificationAuthorityRole,
        ratification.decision, ratification.ratifierIdentity,
        ratification.ratificationBasis);
    if (rebuilt != ratification)
        throw std::invalid_argument("ratification_identity_mismatch");
}

std::string RecommendationCampaignFollowUpProposalRatificationDecisionText(
    RecommendationCampaignFollowUpProposalRatificationDecision decision)
{
    switch (decision)
    {
        case RecommendationCampaignFollowUpProposalRatificationDecision::ratified:
            return "ratified";
    }
    throw std::invalid_argument("ratification_decision_invalid");
}

RecommendationCampaignFollowUpProposalRatificationDecision
RecommendationCampaignFollowUpProposalRatificationDecisionFromText(
    const std::string& text)
{
    if (text == "ratified")
        return RecommendationCampaignFollowUpProposalRatificationDecision::ratified;
    throw std::invalid_argument("ratification_decision_invalid");
}

} // namespace EA::ExperimentRecommendation
