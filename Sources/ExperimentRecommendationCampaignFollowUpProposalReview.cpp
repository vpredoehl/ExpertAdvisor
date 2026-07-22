#include "ExperimentRecommendationCampaignFollowUpProposalReview.hpp"

#include "ExperimentRecommendation.hpp"

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

bool IsValidReviewerIdentity(const std::string& value)
{
    if (value.empty() ||
        value.size() >
            kMaximumRecommendationCampaignFollowUpProposalReviewerIdentityBytes ||
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

bool IsValidReasonText(const std::string& value)
{
    if (value.empty() ||
        value.size() >
            kMaximumRecommendationCampaignFollowUpProposalReviewReasonTextBytes ||
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

std::string ReviewCanonicalText(
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision decision,
    const std::string& reviewerIdentity,
    const std::string& reasonText)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output
        << "experiment_recommendation_campaign_follow_up_proposal_review_v1"
        << ";review_contract_version="
        << kRecommendationCampaignFollowUpProposalReviewContractVersion
        << ";read_only=true"
           ";database_free=true"
           ";persistent=false"
           ";administrative_review=true"
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
        << ";follow_up_proposal_id=" << followUpProposalId
        << ";proposal_contract_version=" << proposalContractVersion
        << ";proposal_canonical=" << LengthPrefixed(proposalCanonicalText)
        << ";proposal_identity_hash="
        << LengthPrefixed(proposalIdentityHash)
        << ";decision=" << RecommendationCampaignFollowUpProposalReviewDecisionText(
               decision)
        << ";reviewer=" << LengthPrefixed(reviewerIdentity)
        << ";reason=" << LengthPrefixed(reasonText);
    return output.str();
}

} // namespace

struct RecommendationCampaignFollowUpProposalReviewBuilder
{
    static RecommendationCampaignFollowUpProposalReview Build(
        int reviewContractVersion,
        long long followUpProposalId,
        int proposalContractVersion,
        const std::string& proposalCanonicalText,
        const std::string& proposalIdentityHash,
        RecommendationCampaignFollowUpProposalReviewDecision decision,
        std::string reviewerIdentity,
        std::string reasonText)
    {
        if (reviewContractVersion !=
            kRecommendationCampaignFollowUpProposalReviewContractVersion)
            throw std::invalid_argument("review_contract_unsupported");
        if (followUpProposalId <= 0)
            throw std::invalid_argument("persisted_proposal_id_invalid");
        if (proposalContractVersion !=
            kRecommendationCampaignFollowUpProposalContractVersion)
            throw std::invalid_argument("proposal_contract_unsupported");
        if (proposalCanonicalText.empty() ||
            proposalCanonicalText.size() >
                kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes ||
            proposalCanonicalText.find('\0') != std::string::npos ||
            proposalIdentityHash !=
                RecommendationCanonicalHash(proposalCanonicalText))
            throw std::invalid_argument("proposal_identity_invalid");
        (void)RecommendationCampaignFollowUpProposalReviewDecisionText(
            decision);
        if (!IsValidReviewerIdentity(reviewerIdentity))
            throw std::invalid_argument("reviewer_identity_invalid");
        if (!IsValidReasonText(reasonText))
            throw std::invalid_argument("review_reason_invalid");

        std::string canonicalText = ReviewCanonicalText(followUpProposalId,
            proposalContractVersion, proposalCanonicalText,
            proposalIdentityHash, decision, reviewerIdentity, reasonText);
        if (canonicalText.size() >
            kMaximumRecommendationCampaignFollowUpProposalReviewCanonicalTextBytes)
            throw std::invalid_argument("review_canonical_size_exceeded");
        RecommendationCampaignFollowUpProposalReviewIdentity identity(
            reviewContractVersion, canonicalText,
            RecommendationCanonicalHash(canonicalText));
        return RecommendationCampaignFollowUpProposalReview(
            std::move(identity), followUpProposalId,
            proposalContractVersion, proposalCanonicalText,
            proposalIdentityHash, decision, std::move(reviewerIdentity),
            std::move(reasonText));
    }
};

RecommendationCampaignFollowUpProposalReviewIdentity::
    RecommendationCampaignFollowUpProposalReviewIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
}

RecommendationCampaignFollowUpProposalReview::
    RecommendationCampaignFollowUpProposalReview(
        RecommendationCampaignFollowUpProposalReviewIdentity identityValue,
        long long followUpProposalIdValue,
        int proposalContractVersionValue,
        std::string proposalCanonicalTextValue,
        std::string proposalIdentityHashValue,
        RecommendationCampaignFollowUpProposalReviewDecision decisionValue,
        std::string reviewerIdentityValue,
        std::string reasonTextValue)
    : identity(std::move(identityValue)),
      followUpProposalId(followUpProposalIdValue),
      proposalContractVersion(proposalContractVersionValue),
      proposalCanonicalText(std::move(proposalCanonicalTextValue)),
      proposalIdentityHash(std::move(proposalIdentityHashValue)),
      decision(decisionValue),
      reviewerIdentity(std::move(reviewerIdentityValue)),
      reasonText(std::move(reasonTextValue))
{
}

RecommendationCampaignFollowUpProposalReview
BuildRecommendationCampaignFollowUpProposalReview(
    int reviewContractVersion,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision decision,
    std::string reviewerIdentity,
    std::string reasonText)
{
    return RecommendationCampaignFollowUpProposalReviewBuilder::Build(
        reviewContractVersion, followUpProposalId, proposalContractVersion,
        proposalCanonicalText, proposalIdentityHash, decision,
        std::move(reviewerIdentity), std::move(reasonText));
}

RecommendationCampaignFollowUpProposalReview
BuildRecommendationCampaignFollowUpProposalReview(
    long long followUpProposalId,
    const RecommendationCampaignFollowUpProposal& proposal,
    RecommendationCampaignFollowUpProposalReviewDecision decision,
    std::string reviewerIdentity,
    std::string reasonText)
{
    return BuildRecommendationCampaignFollowUpProposalReview(
        kRecommendationCampaignFollowUpProposalReviewContractVersion,
        followUpProposalId, proposal.identity.contractVersion,
        proposal.identity.canonicalText, proposal.identity.hash, decision,
        std::move(reviewerIdentity), std::move(reasonText));
}

void ValidateRecommendationCampaignFollowUpProposalReview(
    const RecommendationCampaignFollowUpProposalReview& review)
{
    const auto rebuilt = BuildRecommendationCampaignFollowUpProposalReview(
        review.identity.contractVersion, review.followUpProposalId,
        review.proposalContractVersion, review.proposalCanonicalText,
        review.proposalIdentityHash, review.decision,
        review.reviewerIdentity, review.reasonText);
    if (rebuilt != review)
        throw std::invalid_argument("review_identity_mismatch");
}

std::string RecommendationCampaignFollowUpProposalReviewDecisionText(
    RecommendationCampaignFollowUpProposalReviewDecision decision)
{
    switch (decision)
    {
        case RecommendationCampaignFollowUpProposalReviewDecision::approved:
            return "approved";
        case RecommendationCampaignFollowUpProposalReviewDecision::rejected:
            return "rejected";
    }
    throw std::invalid_argument("review_decision_invalid");
}

RecommendationCampaignFollowUpProposalReviewDecision
RecommendationCampaignFollowUpProposalReviewDecisionFromText(
    const std::string& text)
{
    if (text == "approved")
        return RecommendationCampaignFollowUpProposalReviewDecision::approved;
    if (text == "rejected")
        return RecommendationCampaignFollowUpProposalReviewDecision::rejected;
    throw std::invalid_argument("review_decision_invalid");
}

} // namespace EA::ExperimentRecommendation
