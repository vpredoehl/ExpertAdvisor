#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
configuration="${EA_BUILD_CONFIGURATION:-Release}"
products="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/${configuration}"
build_dir="${repo_root}/DerivedData/ExpertAdvisor/Build/Intermediates.noindex/ExpertAdvisor.build/${configuration}/lstm-scheduler.build"
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

cli_output="$(mktemp "${TMPDIR:-/tmp}/ea_standalone_scheduler_cli.XXXXXX")"
trap 'rm -f "${linked_symbols}" "${cli_output}"' EXIT

expect_recovery_argument_error() {
    local expected="$1"
    shift

    set +e
    "${binary}" "$@" >"${cli_output}" 2>&1
    local rc=$?
    set -e

    test "${rc}" = "1"
    rg -Fq "Argument error: ${expected}" "${cli_output}"
}

# Both supported experiment-ID forms must be recognized by the standalone
# adapter before daemon parsing, and the confirmation modes are exclusive.
expect_recovery_argument_error \
    '--recover-failed-inference requires exactly one of --dry-run or --yes' \
    --recover-failed-inference 17
expect_recovery_argument_error \
    '--recover-failed-inference requires exactly one of --dry-run or --yes' \
    --recover-failed-inference=17
expect_recovery_argument_error \
    '--recover-failed-inference requires exactly one of --dry-run or --yes' \
    --recover-failed-inference=17 --dry-run --yes

expect_recovery_argument_error \
    '--recover-failed-inference requires a positive integer' \
    --recover-failed-inference=malformed --dry-run
expect_recovery_argument_error \
    '--recover-failed-inference requires a positive integer' \
    --recover-failed-inference=0 --dry-run
expect_recovery_argument_error \
    '--recover-failed-inference requires a positive integer' \
    --recover-failed-inference=-1 --dry-run
expect_recovery_argument_error \
    '--recover-failed-inference specified more than once' \
    --recover-failed-inference=17 --recover-failed-inference=18 --dry-run
expect_recovery_argument_error \
    "unknown historical failed-inference recovery option '--unknown'" \
    --recover-failed-inference=17 --dry-run --unknown

printf '%s\n' \
    'StandaloneSchedulerExecutableTests passed; actual target linked without ExperimentScheduler.o'
