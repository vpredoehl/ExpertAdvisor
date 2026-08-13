#include "CampaignOperationsManager.hpp"

#include <sstream>
#include <stdexcept>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

} // namespace

ManagerRequestOperationIdentity BuildManagerRequestOperationIdentity(
    const std::string& requestIdentityCanonical, int expectedRequestVersion)
{
    if (requestIdentityCanonical.empty() ||
        requestIdentityCanonical.size() > kCampaignOperationsCanonicalMaximumBytes)
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_manager_request_canonical_invalid");
    if (expectedRequestVersion <= 0)
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_manager_expected_request_version_invalid");

    std::ostringstream source;
    source << kCampaignOperationsManagerOperationPrefix
           << ";request_identity_canonical="
           << Framed(requestIdentityCanonical)
           << ";expected_request_version=" << expectedRequestVersion;
    const auto sourceIdentity = CanonicalIdentity::Create(1, source.str());
    return {sourceIdentity,
        "mgr-v1:" + sourceIdentity.hash() + ":" +
            std::to_string(expectedRequestVersion)};
}

void ValidateManagerRequestOperationIdentity(
    const ManagerRequestOperationIdentity& identity,
    const std::string& requestIdentityCanonical, int expectedRequestVersion)
{
    const auto expected = BuildManagerRequestOperationIdentity(
        requestIdentityCanonical, expectedRequestVersion);
    if (identity.source != expected.source ||
        identity.operationKey != expected.operationKey)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_manager_operation_identity_mismatch");
}

} // namespace EA::CampaignOperations
