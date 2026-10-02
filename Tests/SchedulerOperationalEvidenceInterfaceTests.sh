#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
observer_cli="${repo_root}/Sources/SchedulerCore/SchedulerObserverCli.cpp"
read_model="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.cpp"
read_header="${repo_root}/Sources/SchedulerCore/SchedulerOperationalReadModel.hpp"
status_service="${repo_root}/Sources/SchedulerCore/SchedulerStatusService.cpp"
observation_header="${repo_root}/Sources/SchedulerCore/SchedulerOperationalObservation.hpp"
investigator="${repo_root}/Scripts/ExpertAdvisorInvestigator.py"

# The command grammar has an additive, closed evidence namespace.  It is not
# a generic observer verb or a control surface.
rg -Fq 'if (operation == "evidence")' "${observer_cli}"
rg -Fq 'evidenceKind != "scheduler" && evidenceKind != "experiment"' "${observer_cli}"
rg -Fq 'evidence experiment requires EXPERIMENT_ID' "${observer_cli}"
rg -Fq 'EXPERIMENT_ID must be a positive integer' "${observer_cli}"
rg -Fq 'PrintObserverSchedulerEvidence' "${observer_cli}"
rg -Fq 'PrintObserverExperimentEvidence' "${observer_cli}"

# Evidence is captured through the private read-only snapshot bridge, then
# serialized from typed records.  It must not call a legacy text renderer.
rg -Fq 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;' "${read_model}"
rg -Fq 'struct SchedulerOperationalEvidence' "${status_service}"
rg -Fq 'struct ExperimentOperationalEvidence' "${status_service}"
rg -Fq 'WriteJsonSchedulerEvidence' "${status_service}"
rg -Fq 'WriteJsonExperimentEvidence' "${status_service}"
rg -Fq 'expertadvisor-operational-evidence-v1' "${status_service}"
rg -Fq '\"process_observation\":' "${status_service}"
rg -Fq '\"durable\":' "${status_service}"

scheduler_evidence_render="$(sed -n '/int PrintObserverSchedulerEvidence(/,/^}/p' "${status_service}")"
experiment_evidence_render="$(sed -n '/int PrintObserverExperimentEvidence(/,/^}/p' "${status_service}")"
if printf '%s\n' "${scheduler_evidence_render}" | rg -n 'PrintSchedulerStatusFromTransaction|PrintCompactExperimentStatusFromTransaction'; then
    printf '%s\n' 'evidence renderer parses a human status report' >&2
    exit 1
fi
if printf '%s\n' "${experiment_evidence_render}" | rg -n 'PrintSchedulerStatusFromTransaction|PrintCompactExperimentStatusFromTransaction'; then
    printf '%s\n' 'evidence renderer parses a human status report' >&2
    exit 1
fi

public_surface="$(sed -n '/public:/,/private:/p' "${read_header}")"
if printf '%s\n' "${public_surface}" | rg -n '\b(pqxx|withReadOnlySnapshot|claim|finalize|signal|reconcile|queue|preempt)\b'; then
    printf '%s\n' 'evidence read model exposes transaction or mutation authority publicly' >&2
    exit 1
fi
if rg -n '\b(CreateNativeProcessOperations|ProcessOperations|SignalProcessGroup|WaitForProcessGroupExit|AcquireCoordinationLock|RunCommand)\b|\b(SIGSTOP|SIGCONT|SIGTERM|SIGKILL)\b|\b(INSERT|UPDATE|DELETE)\b' \
    "${observer_cli}" "${read_header}" "${read_model}" "${observation_header}"; then
    printf '%s\n' 'evidence observer exposes mutation or process-signaling authority' >&2
    exit 1
fi

# The repository investigator remains a separate checked-in-only producer.
if rg -n 'ps -|pqxx|LSTM_DB_|lstm-observer|subprocess.*LSTM_Release' "${investigator}"; then
    printf '%s\n' 'repository investigator crossed into live operational evidence' >&2
    exit 1
fi

# A committed observer build may opt into a read-only database smoke test. This
# keeps the source boundary test usable without a local PostgreSQL service,
# while making JSON parsing and the closed runtime grammar regression-testable.
if [[ -n "${LSTM_OBSERVER_EVIDENCE_BINARY:-}" ]]; then
    experiment_id="${LSTM_OBSERVER_EVIDENCE_EXPERIMENT_ID:-688}"
    tmp="$(mktemp -d /tmp/ea_observer_evidence.XXXXXX)"
    trap 'rm -rf "${tmp}"' EXIT
    "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence scheduler >"${tmp}/scheduler.json"
    "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence experiment "${experiment_id}" >"${tmp}/experiment.json"
    python3 - "${tmp}/scheduler.json" "${tmp}/experiment.json" "${experiment_id}" <<'PY'
import json
import sys
from pathlib import Path
scheduler = json.loads(Path(sys.argv[1]).read_text())
experiment = json.loads(Path(sys.argv[2]).read_text())
assert scheduler['schema'] == 'expertadvisor-operational-evidence-v1'
assert scheduler['kind'] == 'scheduler'
assert set(('durable', 'process_observation')) <= set(scheduler)
assert experiment['schema'] == 'expertadvisor-operational-evidence-v1'
assert experiment['kind'] == 'experiment'
assert experiment['durable']['experiment']['experiment_id'] == int(sys.argv[3])
assert set(('durable', 'process_observation')) <= set(experiment)
PY
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence queue >/dev/null 2>&1; then
        printf '%s\n' 'unsupported evidence operation unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence experiment 0 >/dev/null 2>&1; then
        printf '%s\n' 'invalid evidence experiment identifier unexpectedly succeeded' >&2
        exit 1
    fi
fi

printf '%s\n' 'SchedulerOperationalEvidenceInterfaceTests passed'
