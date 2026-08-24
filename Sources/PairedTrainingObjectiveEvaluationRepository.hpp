#pragma once

#include "PairedTrainingObjectiveEvaluation.hpp"

#include <stdexcept>
#include <string>

#include <pqxx/pqxx>

namespace EA::PairedTrainingObjectiveEvaluation
{

enum class EvidenceLoadErrorKind
{
    ExperimentNotFound,
    ProvenanceContractFailure,
    AmbiguousEvidence
};

class EvidenceLoadError final : public std::runtime_error
{
public:
    EvidenceLoadError(EvidenceLoadErrorKind kind, std::string reason);
    EvidenceLoadErrorKind kind() const noexcept { return kind_; }
    const std::string& reason() const noexcept { return reason_; }

private:
    EvidenceLoadErrorKind kind_;
    std::string reason_;
};

// Loads one arm entirely through caller-owned read-only transaction scope.
// Missing final artifacts on an otherwise unambiguous row are represented as
// absent ArmEvidence optionals for the pure evaluator's INCOMPLETE result.
ArmEvidence LoadAuthoritativeArmEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId);

} // namespace EA::PairedTrainingObjectiveEvaluation
