#pragma once

#include "CampaignOperations.hpp"
#include "CampaignOperationsDispatch.hpp"

#include <string>
#include <utility>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsManagerOperationContractVersion = 1;
inline constexpr int kCampaignOperationsManagerMaximumRunOnceLimit =
    kCampaignOperationsMaximumDispatchBatchSize;
inline constexpr char kCampaignOperationsManagerOperationPrefix[] =
    "campaign_operations_manager_request_operation_v1";
inline constexpr char kCampaignOperationsManagerActor[] =
    "campaign_operations_manager";

struct ManagerRequestOperationIdentity final
{
    const CanonicalIdentity source;
    const std::string operationKey;

    ManagerRequestOperationIdentity(CanonicalIdentity sourceValue,
        std::string operationKeyValue)
        : source(std::move(sourceValue)),
          operationKey(std::move(operationKeyValue))
    {
    }

    ManagerRequestOperationIdentity(const ManagerRequestOperationIdentity&) =
        default;
    ManagerRequestOperationIdentity(ManagerRequestOperationIdentity&&) =
        default;
    ManagerRequestOperationIdentity& operator=(
        const ManagerRequestOperationIdentity&) = delete;
    ManagerRequestOperationIdentity& operator=(
        ManagerRequestOperationIdentity&&) = delete;
};

ManagerRequestOperationIdentity BuildManagerRequestOperationIdentity(
    const std::string& requestIdentityCanonical,
    int expectedRequestVersion);

void ValidateManagerRequestOperationIdentity(
    const ManagerRequestOperationIdentity& identity,
    const std::string& requestIdentityCanonical,
    int expectedRequestVersion);

} // namespace EA::CampaignOperations
