#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
scheduler_archive="${products}/libSchedulerCore.a"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_production_scheduler_link.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

test -f "${scheduler_archive}"
test -f "${products}/libProfitabilityCore.a"
test -f "${products}/libMetaNN.a"
test -f "${products}/libMetalBuffer.a"

read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -Wno-deprecated-declarations -Wno-unused-function \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_compile_flags[@]}" \
    "${repo_root}/Tests/ProductionSchedulerCompositionLinkHarness.cpp" \
    "${repo_root}/Sources/LstmRuntimeLogging.cpp" \
    "${repo_root}/Sources/CheckpointPolicy.cpp" \
    "${repo_root}/Sources/ContinuationPolicy.cpp" \
    "${repo_root}/Sources/ContinuationPolicyInheritance.cpp" \
    "${repo_root}/Sources/ContinuationPolicyPersistence.cpp" \
    "${repo_root}/Sources/EconomicEventRepository.cpp" \
    "${repo_root}/Sources/GlobalExperimentControl.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Sources/RunMetadata.cpp" \
    "${scheduler_archive}" \
    "${products}/libProfitabilityCore.a" \
    "${products}/libMetaNN.a" \
    "${products}/libMetalBuffer.a" \
    "${pqxx_link_flags[@]}" \
    -framework Metal -framework Foundation \
    -Wl,-map,"${test_dir}/ProductionSchedulerCompositionLinkHarness.map" \
    -o "${test_dir}/ProductionSchedulerCompositionLinkHarness"

"${test_dir}/ProductionSchedulerCompositionLinkHarness"

link_map="${test_dir}/ProductionSchedulerCompositionLinkHarness.map"
required_scheduler_objects=(
    ProductionSchedulerDaemon.o
    SchedulerEngine.o
    SemanticWorkerRegistry.o
    PostgresSchedulerRepository.o
    WorkerProcessController.o
    SchedulerAuthorityService.o
    ContinuationOrchestrationService.o
    WorkerAttemptLifecycleService.o
    CheckpointAnalysisOrchestrationService.o
    CheckpointEvaluationService.o
    SchedulerCycleService.o
    FinalExperimentDispatchService.o
    SchedulerChildCompletionService.o
)
for object in "${required_scheduler_objects[@]}"; do
    rg -Fq "libSchedulerCore.a(${object})" "${link_map}"
done

if rg -Fq 'libSchedulerCore.a(ExperimentScheduler.o)' "${link_map}"; then
    printf '%s\n' "ExperimentScheduler.o was unexpectedly extracted" >&2
    exit 1
fi

forbidden_symbols=(
    'EA::ExperimentScheduler::RunSchedulerDaemon('
    'EA::ExperimentScheduler::IsExperimentSchedulerCommand('
    'EA::ExperimentScheduler::RunExperimentSchedulerCli('
    'EA::SchedulerCore::RegisterSchedulerWorkerAttempt('
)
linked_symbols="${test_dir}/linked-symbols.txt"
nm "${test_dir}/ProductionSchedulerCompositionLinkHarness" | c++filt >"${linked_symbols}"
for symbol in "${forbidden_symbols[@]}"; do
    if rg -Fq "${symbol}" "${linked_symbols}"; then
        printf 'forbidden legacy compatibility symbol linked: %s\n' \
            "${symbol}" >&2
        exit 1
    fi
done

printf '%s\n' \
    "ProductionSchedulerCompositionLinkHarness passed; real production composition linked without ExperimentScheduler.o"
