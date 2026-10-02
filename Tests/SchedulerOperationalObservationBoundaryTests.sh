#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
read_model="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.cpp"
read_header="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.hpp"
observer_cli="${repo_root}/Sources/SchedulerCore/SchedulerObserverCli.cpp"
status_service="${repo_root}/Sources/SchedulerCore/SchedulerStatusService.cpp"
observer_main="${repo_root}/LSTM/ObserverMain.cpp"
project="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"

rg -Fq 'pqxx::read_transaction' "${read_model}"
rg -Fq 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;' "${read_model}"
rg -Fq 'withReadOnlySnapshot' "${read_model}"
rg -Fq 'PrintObserverSchedulerStatus(readModel, output, error)' "${observer_cli}"
rg -Fq 'PrintObserverExperimentStatus(' "${observer_cli}"
rg -Fq 'PrintSchedulerStatusFromTransaction' "${status_service}"
rg -Fq 'PrintCompactExperimentStatusFromTransaction' "${status_service}"

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
