#include "ControlledReplicationStudySpecification.hpp"

#include <cassert>
#include <string>

using namespace EA::ControlledReplicationStudy;

Specification Fixture()
{
    Specification specification;
    specification.studyIdentifier = "tg4-cross-context-fixture";
    specification.interventionField = "feature_ablation_mask";
    specification.controlSemantics = "empty_string";
    specification.controlValue = "";
    specification.treatmentValue =
        "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,tg4_source_tg3_confluent";
    specification.replicationDimension = "fresh_initialization_seed";
    specification.allowedContextDimensions = {"symbol"};
    specification.requiredConfiguredIdentityFields = {
        "symbol", "prediction_horizon", "feature_ablation_mask",
        "fresh_initialization_seed", "training_objective_hash",
        "inference_start", "inference_end"};
    specification.requiredExecutionProvenanceFields = {
        "training_execution_identity", "inference_execution_identity",
        "producer_worker_attempt_id"};
    specification.aggregationPolicy =
        "unweighted_mean_of_context_family_means";
    specification.freezeTimestamp = "2026-09-28T10:00:00Z";
    specification.contexts = {
        {"1", {{"symbol", "CADCHF"}, {"prediction_horizon", "4"}},
         {{676, 677, 44}, {678, 679, 45}, {680, 681, 46}}},
        {"2", {{"symbol", "AUDCAD"}, {"prediction_horizon", "4"}},
         {{670, 671, 44}, {672, 673, 45}, {674, 675, 46}}}};
    specification.identityHash = IdentityHash(specification);
    return specification;
}

int main()
{
    const Specification original = Fixture();
    const std::string artifact = Render(original);
    const Specification parsed = Parse(artifact);
    const auto valid = Validate(parsed);
    assert(valid.valid);
    assert(valid.identityHash == original.identityHash);
    assert(Canonicalize(parsed) == Canonicalize(original));
    assert(Render(parsed) == artifact);

    auto duplicate = parsed;
    duplicate.contexts[1].pairs[0].armA = duplicate.contexts[0].pairs[0].armA;
    assert(!Validate(duplicate).valid);

    auto reused = parsed;
    reused.contexts[1].pairs[0].armA = reused.contexts[0].pairs[0].armB;
    assert(!Validate(reused).valid);

    auto reversed = parsed;
    reversed.contexts[0].pairs[0].armA = 677;
    reversed.contexts[0].pairs[0].armB = 676;
    assert(!Validate(reversed).valid);

    auto wrongIntervention = parsed;
    wrongIntervention.interventionField = "training_objective";
    assert(!Validate(wrongIntervention).valid);

    auto duplicateSeed = parsed;
    duplicateSeed.contexts[0].pairs[1].requestedSeed = 44;
    assert(!Validate(duplicateSeed).valid);

    auto undeclared = parsed;
    undeclared.contexts[0].dimensions.push_back({"timeframe", "H4"});
    assert(!Validate(undeclared).valid);

    auto badVersion = parsed;
    badVersion.version = 99;
    assert(!Validate(badVersion).valid);

    std::string badHash = artifact;
    const std::string marker = "identity_hash=";
    const std::size_t offset = badHash.find(marker);
    assert(offset != std::string::npos);
    badHash.replace(offset + marker.size(), parsed.identityHash.size(),
                    "fnv1a64:0000000000000000");
    assert(!ValidateArtifact(badHash).valid);

    const auto command = MakeFamilyComparisonCommand(parsed);
    assert(command.families.size() == 2);
    assert(command.families[0].size() == 3);
    assert(command.families[1].size() == 3);
    assert(command.families[0][0] == std::make_pair(676LL, 677LL));
    assert(command.families[1][0] == std::make_pair(670LL, 671LL));

    assert(parsed.statisticalIndependence == "not_inferred");
    assert(parsed.subjectiveWinner == "NONE");
    return 0;
}
