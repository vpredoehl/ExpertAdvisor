#include "CampaignOperationsManager.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace CO = EA::CampaignOperations;

int main()
{
    const std::string request = "request-\xC2\xB5";
    const auto identity = CO::BuildManagerRequestOperationIdentity(request, 7);
    assert(identity.source.canonicalText() ==
        "campaign_operations_manager_request_operation_v1;"
        "request_identity_canonical=10:request-\xC2\xB5;"
        "expected_request_version=7");
    assert(identity.source.hash() == "fnv1a64:4f1c6a7f9ed711f5");
    assert(identity.operationKey ==
        "mgr-v1:fnv1a64:4f1c6a7f9ed711f5:7");
    std::cout << "H3_SOURCE=" << identity.source.canonicalText() << '\n'
              << "H3_HASH=" << identity.source.hash() << '\n'
              << "H3_KEY=" << identity.operationKey << '\n';
    const auto same = CO::BuildManagerRequestOperationIdentity(request, 7);
    assert(same.source == identity.source);
    assert(same.operationKey == identity.operationKey);
    assert(CO::BuildManagerRequestOperationIdentity(request + "x", 7)
               .operationKey != identity.operationKey);
    assert(CO::BuildManagerRequestOperationIdentity(request, 8)
               .operationKey != identity.operationKey);
    bool rejected = false;
    try
    {
        (void)CO::BuildManagerRequestOperationIdentity(request, 0);
    }
    catch (const CO::Error& error)
    {
        rejected = error.code() == CO::ErrorCode::invalidOperationalRequest;
    }
    assert(rejected);
    std::cout << "H3_IDENTITY_VECTORS_PASS\n";
}
