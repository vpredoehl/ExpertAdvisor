#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
read_model="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.cpp"
read_header="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.hpp"
observer_cli="${repo_root}/Sources/SchedulerCore/SchedulerObserverCli.cpp"
status_service="${repo_root}/Sources/SchedulerCore/SchedulerStatusService.cpp"
observation_source="${repo_root}/Sources/SchedulerCore/SchedulerOperationalObservation.cpp"
observation_header="${repo_root}/Sources/SchedulerCore/SchedulerOperationalObservation.hpp"
control_source="${repo_root}/Sources/GlobalExperimentControl.cpp"
observer_main="${repo_root}/LSTM/ObserverMain.cpp"
project="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"

rg -Fq 'pqxx::read_transaction' "${read_model}"
rg -Fq 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;' "${read_model}"
rg -Fq 'withReadOnlySnapshot' "${read_model}"
rg -Fq 'PrintObserverSchedulerStatus(readModel, output, error)' "${observer_cli}"
rg -Fq 'PrintObserverExperimentStatus(' "${observer_cli}"
rg -Fq 'PrintObserverSchedulerEvidence(readModel, output, error)' "${observer_cli}"
rg -Fq 'PrintObserverExperimentEvidence(' "${observer_cli}"
rg -Fq 'PrintObserverComparisonEvidence' "${observer_cli}"
rg -Fq 'PrintSchedulerStatusFromTransaction' "${status_service}"
rg -Fq 'PrintCompactExperimentStatusFromTransaction' "${status_service}"
rg -Fq '#include "SchedulerOperationalObservation.hpp"' "${status_service}"
rg -Fq 'CreateNativeProcessObserver()' "${status_service}"
rg -Fq 'ClassifySchedulerWorkers(' "${observation_source}"
rg -Fq 'SummarizeSchedulerWorkers(' "${observation_source}"
rg -Fq 'LoadControlSnapshot(' "${observation_source}"
rg -Fq 'class ProcessObserver' "${observation_header}"

if rg -n '\b(CreateNativeProcessOperations|ProcessOperations|SignalProcessGroup|WaitForProcessGroupExit|AcquireCoordinationLock|RunCommand)\b|\b(SIGSTOP|SIGCONT|SIGTERM|SIGKILL)\b|\b(INSERT|UPDATE|DELETE)\b' \
    "${observation_header}" "${observation_source}"; then
    printf '%s\n' 'shared observer implementation exposes mutation or signaling authority' >&2
    exit 1
fi
if rg -Fq '#include "GlobalExperimentControl.hpp"' "${status_service}" ||
    rg -Fq '#include "GlobalExperimentControl.hpp"' "${observation_source}"; then
    printf '%s\n' 'observer status dependency reaches the administrative control surface' >&2
    exit 1
fi
rg -Fq 'CreateNativeProcessOperations()' "${control_source}"
rg -Fq 'SignalProcessGroup(' "${control_source}"
if rg -n '^(const char\* ToString\((IdentityResult|ProcessExecutionState)|std::vector<SchedulerWorkerClassification> ClassifySchedulerWorkers|SchedulerWorkerClassificationSummary SummarizeSchedulerWorkers|ControlSnapshot LoadControlSnapshot)' \
    "${control_source}"; then
    printf '%s\n' 'observation primitive remains duplicated in administrative implementation' >&2
    exit 1
fi

observer_scheduler_render="$(sed -n '/int PrintObserverSchedulerStatus(/,/^}/p' "${status_service}")"
observer_experiment_render="$(sed -n '/int PrintObserverExperimentStatus(/,/^}/p' "${status_service}")"
if ! printf '%s\n' "${observer_scheduler_render}" | rg -Fq 'PrintSchedulerStatusFromTransaction('; then
    printf '%s\n' 'observer scheduler path does not use shared status rendering' >&2
    exit 1
fi
if ! printf '%s\n' "${observer_experiment_render}" | rg -Fq 'PrintCompactExperimentStatusFromTransaction('; then
    printf '%s\n' 'observer experiment path does not use shared compact status rendering' >&2
    exit 1
fi

if rg -Fq 'count(*)' "${read_model}" || rg -Fq 'ObservedSchedulerSummary' "${read_header}"; then
    printf '%s\n' 'provisional aggregate-count observation path remains' >&2
    exit 1
fi

rg -Fq 'operation != "scheduler" && operation != "experiment"' "${observer_cli}"
rg -Fq 'if (operation == "evidence")' "${observer_cli}"
rg -Fq 'evidenceKind != "scheduler" && evidenceKind != "experiment"' "${observer_cli}"
rg -Fq 'PrintObserverSchedulerEvidence' "${read_header}"
rg -Fq 'PrintObserverExperimentEvidence' "${read_header}"
rg -Fq 'PrintObserverComparisonEvidence' "${read_header}"

public_surface="$(sed -n '/public:/,/private:/p' "${read_header}")"
if printf '%s\n' "${public_surface}" | rg -n '\b(pqxx|withReadOnlySnapshot|claim|finalize|signal|reconcile|queue|preempt)\b'; then
    printf '%s\n' 'read model exposes transaction or mutation authority publicly' >&2
    exit 1
fi

if rg -n '^(#include .*SchedulerAdmissionService|#include .*WorkerControlService|#include .*SchedulerAuthorityService|#include .*WorkerAttemptLifecycleService)|\b(SchedulerServiceComposition|RunExperimentSchedulerCli|popen)\b' \
    "${observer_cli}" "${read_header}" "${read_model}"; then
    printf '%s\n' 'observer boundary reaches a mutation/control dependency' >&2
    exit 1
fi

if rg -n '\bSchedulerServiceComposition\b' "${status_service}"; then
    printf '%s\n' 'shared status service still instantiates mutable scheduler composition' >&2
    exit 1
fi

rg -Fq '#include "SchedulerCore/SchedulerObserverCli.hpp"' "${observer_main}"
rg -Fq 'RunSchedulerObserverCli' "${observer_main}"
rg -Fq 'SchedulerObserverCli.cpp in Sources' "${project}"
rg -Fq 'SchedulerOperationalReadModel.cpp in Sources' "${project}"
rg -Fq 'SchedulerOperationalObservation.cpp in Sources' "${project}"

observer_target="$(sed -n '/0F1A004C33A0000100AAA001 \/\* lstm-observer \*\//,/^\t\t};/p' "${project}")"
observer_sources="$(sed -n '/0F1A004B33A0000100AAA001 \/\* Sources \*\//,/^\t\t};/p' "${project}")"
if ! printf '%s\n' "${observer_sources}" | rg -Fq '0F1A004833A0000100AAA001 /* ObserverMain.cpp in Sources */'; then
    printf '%s\n' 'observer target does not use the thin ObserverMain entry point' >&2
    exit 1
fi
if ! printf '%s\n' "${observer_target}" | rg -Fq '0F1A005133A0000100AAA001 /* PBXTargetDependency */'; then
    printf '%s\n' 'observer target does not depend on SchedulerCore' >&2
    exit 1
fi
if printf '%s\n' "${observer_target}" | rg -n 'ProfitabilityCore|StrategyEvaluationCore|Metal|provenance|SchedulerMain'; then
    printf '%s\n' 'observer target inherits an unrelated runtime dependency' >&2
    exit 1
fi

observer_frameworks="$(sed -n '/0F1A004A33A0000100AAA001 \/\* Frameworks \*\//,/^\t\t};/p' "${project}")"
if ! printf '%s\n' "${observer_frameworks}" | rg -Fq 'libSchedulerCore.a in Frameworks'; then
    printf '%s\n' 'observer target does not link SchedulerCore' >&2
    exit 1
fi
if printf '%s\n' "${observer_frameworks}" | rg -n 'ProfitabilityCore|StrategyEvaluationCore|Metal|Foundation'; then
    printf '%s\n' 'observer target links an unrelated framework or core library' >&2
    exit 1
fi

printf '%s\n' 'SchedulerOperationalObservationBoundaryTests passed'
