#include "../Headers/ExperimentCurrentOperation.hpp"

#include <cassert>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using namespace EA::ExperimentLifecycle;

int main()
{
    for (const std::string operation : {"train", "infer", "analyze"})
    {
        assert(IsCanonicalCurrentOperation(operation));
        assert(CanonicalCurrentOperationValue(operation) == operation);
        assert(RequireCanonicalCurrentOperationForPhase(operation) == operation);
        assert(CurrentOperationForStatus(operation, operation) == operation);
    }

    assert(!IsCanonicalCurrentOperation("training"));
    assert(!IsCanonicalCurrentOperation("inference"));
    assert(!IsCanonicalCurrentOperation("analysis"));
    assert(CanonicalCurrentOperationValue("training") ==
           std::optional<std::string>{"train"});
    assert(CanonicalCurrentOperationValue("inference") ==
           std::optional<std::string>{"infer"});
    assert(CanonicalCurrentOperationValue("analysis") ==
           std::optional<std::string>{"analyze"});

    assert(NormalizePersistedCurrentOperation(
               "training") == "train");
    assert(NormalizePersistedCurrentOperation(
               "inference") == "infer");
    assert(NormalizePersistedCurrentOperation(
               "analysis") == "analyze");
    assert(NormalizePersistedCurrentOperation(
               "cancel_checkpoint_restart_pending") ==
           "cancel_checkpoint_restart_pending");
    assert(NormalizePersistedCurrentOperation("unexpected") == "unexpected");
    assert(NormalizeOptionalPersistedCurrentOperation(std::nullopt) ==
           std::nullopt);
    assert(NormalizeOptionalPersistedCurrentOperation(
               std::optional<std::string>{"training"}) ==
           std::optional<std::string>{"train"});
    assert(CurrentOperationForStatus(std::nullopt, "train") == "train");
    assert(CurrentOperationForStatus(std::nullopt, "infer") == "infer");
    assert(CurrentOperationForStatus(std::nullopt, "analyze") == "analyze");
    assert(CurrentOperationForStatus(std::nullopt, "done") == "unknown");
    assert(CurrentOperationForStatus(
               std::optional<std::string>{"checkpoint_stopped"}, "analyze") ==
           "invalid(checkpoint_stopped)");
    assert(CurrentOperationForStatus(
               std::optional<std::string>{"unexpected"}, "train") ==
           "invalid(unexpected)");

    bool rejectedDone = false;
    try
    {
        (void)RequireCanonicalCurrentOperationForPhase("done");
    }
    catch (const std::invalid_argument&)
    {
        rejectedDone = true;
    }
    assert(rejectedDone);

    bool rejectedUnknown = false;
    try
    {
        (void)RequireCanonicalCurrentOperationForPhase("training");
    }
    catch (const std::invalid_argument&)
    {
        rejectedUnknown = true;
    }
    assert(rejectedUnknown);

    std::cout << "ExperimentCurrentOperationTests passed\n";
    return 0;
}
