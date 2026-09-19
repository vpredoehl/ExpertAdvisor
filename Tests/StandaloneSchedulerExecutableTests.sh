#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
build_dir="${repo_root}/DerivedData/ExpertAdvisor/Build/Intermediates.noindex/ExpertAdvisor.build/Release/lstm-scheduler.build"
binary="${products}/lstm-scheduler"
link_map="${build_dir}/lstm-scheduler.map"

test -x "${binary}"
test -f "${link_map}"

required_objects=(
    SchedulerMain.o
    SchedulerDaemonCli.o
    SchedulerDaemonConfiguration.o
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
for object in "${required_objects[@]}"; do
    rg -Fq "${object}" "${link_map}"
done

if rg -Fq 'libSchedulerCore.a(ExperimentScheduler.o)' "${link_map}"; then
    printf '%s\n' 'ExperimentScheduler.o was unexpectedly extracted' >&2
    exit 1
fi

forbidden_symbols=(
    'EA::ExperimentScheduler::RunSchedulerDaemon('
    'EA::ExperimentScheduler::IsExperimentSchedulerCommand('
    'EA::ExperimentScheduler::RunExperimentSchedulerCli('
    'EA::SchedulerCore::RegisterSchedulerWorkerAttempt('
)
linked_symbols="$(mktemp "${TMPDIR:-/tmp}/ea_standalone_scheduler_symbols.XXXXXX")"
trap 'rm -f "${linked_symbols}"' EXIT
nm "${binary}" | c++filt >"${linked_symbols}"
for symbol in "${forbidden_symbols[@]}"; do
    if rg -Fq "${symbol}" "${linked_symbols}"; then
        printf 'forbidden legacy compatibility symbol linked: %s\n' \
            "${symbol}" >&2
        exit 1
    fi
done

printf '%s\n' \
    'StandaloneSchedulerExecutableTests passed; actual target linked without ExperimentScheduler.o'
