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
rg -Fq 'evidenceKind != "scheduler" && evidenceKind != "experiment" &&' "${observer_cli}"
rg -Fq 'evidenceKind != "inference" && evidenceKind != "profitability"' "${observer_cli}"
rg -Fq 'requires EXPERIMENT_ID' "${observer_cli}"
rg -Fq 'evidence inference' "${observer_cli}"
rg -Fq 'evidence profitability' "${observer_cli}"
rg -Fq 'EXPERIMENT_ID must be a positive integer' "${observer_cli}"
rg -Fq 'PrintObserverSchedulerEvidence' "${observer_cli}"
rg -Fq 'PrintObserverExperimentEvidence' "${observer_cli}"
rg -Fq 'PrintObserverInferenceEvidence' "${observer_cli}"
rg -Fq 'PrintObserverProfitabilityEvidence' "${observer_cli}"
rg -Fq 'evidence comparison LEFT_EXPERIMENT_ID RIGHT_EXPERIMENT_ID' "${observer_cli}"
rg -Fq 'evidenceKind != "comparison"' "${observer_cli}"
rg -Fq 'LEFT_EXPERIMENT_ID and RIGHT_EXPERIMENT_ID must differ' "${observer_cli}"
rg -Fq 'PrintObserverComparisonEvidence' "${observer_cli}"

# Evidence is captured through the private read-only snapshot bridge, then
# serialized from typed records.  It must not call a legacy text renderer.
rg -Fq 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;' "${read_model}"
rg -Fq 'struct SchedulerOperationalEvidence' "${status_service}"
rg -Fq 'struct ExperimentOperationalEvidence' "${status_service}"
rg -Fq 'struct InferenceOperationalEvidence' "${status_service}"
rg -Fq 'struct ProfitabilityOperationalEvidence' "${status_service}"
rg -Fq 'struct ComparisonOperationalEvidence' "${status_service}"
rg -Fq 'struct FinalInferenceAssociation' "${status_service}"
rg -Fq 'struct FinalProfitabilityAssociation' "${status_service}"
rg -Fq 'struct ComparisonComparabilityEvidence' "${status_service}"
rg -Fq 'WriteJsonSchedulerEvidence' "${status_service}"
rg -Fq 'WriteJsonExperimentEvidence' "${status_service}"
rg -Fq 'WriteJsonInferenceEvidence' "${status_service}"
rg -Fq 'WriteJsonProfitabilityEvidence' "${status_service}"
rg -Fq 'expertadvisor-operational-evidence-v1' "${status_service}"
rg -Fq 'expertadvisor-operational-evidence-v2' "${status_service}"
rg -Fq 'expertadvisor-operational-evidence-v3' "${status_service}"
rg -Fq '\"kind\":\"comparison\"' "${status_service}"
rg -Fq '\"delta_convention\":\"right_minus_left\"' "${status_service}"
rg -Fq '\"controlled_comparison\":false' "${status_service}"
rg -Fq '\"process_observation\":' "${status_service}"
rg -Fq '\"durable\":' "${status_service}"
rg -Fq 'kExperimentAssociatedInferenceCte' "${status_service}"
rg -Fq 'r.parent_experiment_id=$1' "${status_service}"
rg -Fq 'ce.checkpoint_epoch=r.checkpoint_epoch' "${status_service}"
rg -Fq 'ce.checkpoint_model_id=r.model_id' "${status_service}"
rg -Fq 'r.id=o.inference_eval_result_id' "${status_service}"
rg -Fq 'std::string{kExperimentAssociatedInferenceCte}' "${status_service}"
rg -Fq "WHERE inference_scope='final' ORDER BY id" "${status_service}"
rg -Fq 'not_selectable_final_inference_' "${status_service}"
rg -Fq 'ORDER BY profitability_observation_id' "${status_service}"
rg -Fq 'IS NOT DISTINCT FROM for nullable facts' "${status_service}"
rg -Fq 'gross_positive_terminal_horizon_log_return_sum' "${status_service}"
rg -Fq 'aggregate_terminal_horizon_log_return_sum' "${status_service}"
if rg -n 'inference_eval_result\.experiment_id|r\.experiment_id' "${status_service}"; then
    printf '%s\n' 'inference evidence assumes a nonexistent experiment_id column' >&2
    exit 1
fi

# The configured identity catalog is shared with the established generic
# experiment-pair comparator; V3 consumes the catalog rather than owning a
# second list of scientific-equivalence field names.
identity_header="${repo_root}/Sources/ExperimentComparisonIdentity.hpp"
pair_comparator="${repo_root}/Sources/ExperimentPairComparison.cpp"
rg -Fq 'kConfiguredScientificIdentityFields' "${identity_header}"
rg -Fq 'HasConfiguredScientificIdentityFields' "${identity_header}"
rg -Fq 'ExperimentComparisonIdentity.hpp' "${pair_comparator}"
rg -Fq 'HasConfiguredScientificIdentityFields' "${pair_comparator}"
rg -Fq 'kConfiguredScientificIdentityFields' "${status_service}"

scheduler_evidence_render="$(sed -n '/int PrintObserverSchedulerEvidence(/,/^}/p' "${status_service}")"
experiment_evidence_render="$(sed -n '/int PrintObserverExperimentEvidence(/,/^}/p' "${status_service}")"
inference_evidence_render="$(sed -n '/int PrintObserverInferenceEvidence(/,/^}/p' "${status_service}")"
profitability_evidence_render="$(sed -n '/int PrintObserverProfitabilityEvidence(/,/^}/p' "${status_service}")"
if printf '%s\n' "${scheduler_evidence_render}" | rg -n 'PrintSchedulerStatusFromTransaction|PrintCompactExperimentStatusFromTransaction'; then
    printf '%s\n' 'evidence renderer parses a human status report' >&2
    exit 1
fi
if printf '%s\n' "${experiment_evidence_render}" | rg -n 'PrintSchedulerStatusFromTransaction|PrintCompactExperimentStatusFromTransaction'; then
    printf '%s\n' 'evidence renderer parses a human status report' >&2
    exit 1
fi
if printf '%s\n' "${inference_evidence_render}${profitability_evidence_render}" | rg -n 'PrintSchedulerStatusFromTransaction|PrintCompactExperimentStatusFromTransaction'; then
    printf '%s\n' 'V2 evidence renderer parses a human status report' >&2
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
    "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence inference "${experiment_id}" >"${tmp}/inference.json"
    "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence profitability "${experiment_id}" >"${tmp}/profitability.json"
    python3 - "${tmp}/scheduler.json" "${tmp}/experiment.json" "${tmp}/inference.json" "${tmp}/profitability.json" "${experiment_id}" <<'PY'
import json
import sys
from pathlib import Path
scheduler = json.loads(Path(sys.argv[1]).read_text())
experiment = json.loads(Path(sys.argv[2]).read_text())
inference = json.loads(Path(sys.argv[3]).read_text())
profitability = json.loads(Path(sys.argv[4]).read_text())
assert scheduler['schema'] == 'expertadvisor-operational-evidence-v1'
assert scheduler['kind'] == 'scheduler'
assert set(('durable', 'process_observation')) <= set(scheduler)
assert experiment['schema'] == 'expertadvisor-operational-evidence-v1'
assert experiment['kind'] == 'experiment'
assert experiment['durable']['experiment']['experiment_id'] == int(sys.argv[5])
assert set(('durable', 'process_observation')) <= set(experiment)
assert inference['schema'] == 'expertadvisor-operational-evidence-v2'
assert inference['kind'] == 'inference'
assert inference['requested_experiment_id'] == int(sys.argv[5])
assert isinstance(inference['durable']['evaluations'], list)
assert profitability['schema'] == 'expertadvisor-operational-evidence-v2'
assert profitability['kind'] == 'profitability'
assert profitability['requested_experiment_id'] == int(sys.argv[5])
assert isinstance(profitability['durable']['observations'], list)

for evaluation in inference['durable']['evaluations']:
    assert isinstance(evaluation['id'], int)
    assert isinstance(evaluation['model_id'], int)
    assert evaluation['inference_scope'] in ('final', 'checkpoint')
    assert evaluation['accept_model'] in (True, False, None)
    for name in ('completed_epochs', 'accuracy', 'reject_reason', 'pred_down',
                 'pred_neutral', 'pred_up', 'checkpoint_eval_id',
                 'parent_experiment_id', 'checkpoint_epoch',
                 'producer_worker_attempt_id'):
        assert name in evaluation
    if evaluation['inference_scope'] == 'final':
        assert evaluation['checkpoint_eval_id'] is None
        assert evaluation['parent_experiment_id'] is None
        assert evaluation['checkpoint_epoch'] is None

for observation in profitability['durable']['observations']:
    assert isinstance(observation['profitability_observation_id'], int)
    assert isinstance(observation['inference_eval_result_id'], int)
    assert isinstance(observation['prediction_count'], int)
    assert isinstance(observation['actionable_count'], int)
    assert isinstance(
        observation['aggregate_terminal_horizon_log_return_sum'], (int, float))
    assert observation['inference_scope'] in ('final', 'checkpoint')
    assert 'metric_definition_canonical' not in observation
    assert 'observation_identity_canonical' not in observation
    if observation['inference_scope'] == 'final':
        assert observation['checkpoint_eval_id'] is None

assert inference['durable']['evaluations'] == sorted(
    inference['durable']['evaluations'],
    key=lambda row: (0 if row['inference_scope'] == 'final' else 1,
                     -1 if row['checkpoint_eval_id'] is None else row['checkpoint_eval_id'],
                     row['id']))
assert profitability['durable']['observations'] == sorted(
    profitability['durable']['observations'],
    key=lambda row: (0 if row['inference_scope'] == 'final' else 1,
                     -1 if row['checkpoint_eval_id'] is None else row['checkpoint_eval_id'],
                     row['profitability_observation_id']))
PY
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence queue >/dev/null 2>&1; then
        printf '%s\n' 'unsupported evidence operation unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence experiment 0 >/dev/null 2>&1; then
        printf '%s\n' 'invalid evidence experiment identifier unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence inference 0 >/dev/null 2>&1; then
        printf '%s\n' 'invalid inference evidence identifier unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence profitability missing >/dev/null 2>&1; then
        printf '%s\n' 'invalid profitability evidence identifier unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence inference 9223372036854775807 >/dev/null 2>&1; then
        printf '%s\n' 'nonexistent inference experiment unexpectedly succeeded' >&2
        exit 1
    fi
    if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence profitability 9223372036854775807 >/dev/null 2>&1; then
        printf '%s\n' 'nonexistent profitability experiment unexpectedly succeeded' >&2
        exit 1
    fi
    if [[ -n "${LSTM_OBSERVER_EVIDENCE_EMPTY_EXPERIMENT_ID:-}" ]]; then
        "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence inference \
            "${LSTM_OBSERVER_EVIDENCE_EMPTY_EXPERIMENT_ID}" >"${tmp}/empty-inference.json"
        "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence profitability \
            "${LSTM_OBSERVER_EVIDENCE_EMPTY_EXPERIMENT_ID}" >"${tmp}/empty-profitability.json"
        python3 - "${tmp}/empty-inference.json" "${tmp}/empty-profitability.json" <<'PY'
import json
import sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text())['durable']['evaluations'] == []
assert json.loads(Path(sys.argv[2]).read_text())['durable']['observations'] == []
PY
    fi
    if [[ -n "${LSTM_OBSERVER_COMPARISON_LEFT_ID:-}" && -n "${LSTM_OBSERVER_COMPARISON_RIGHT_ID:-}" ]]; then
        "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence comparison \
            "${LSTM_OBSERVER_COMPARISON_LEFT_ID}" \
            "${LSTM_OBSERVER_COMPARISON_RIGHT_ID}" >"${tmp}/comparison.json"
        python3 - "${tmp}/comparison.json" "${LSTM_OBSERVER_COMPARISON_LEFT_ID}" "${LSTM_OBSERVER_COMPARISON_RIGHT_ID}" <<'PY'
import json
import math
import sys
from pathlib import Path
evidence = json.loads(Path(sys.argv[1]).read_text())
assert evidence['schema'] == 'expertadvisor-operational-evidence-v3'
assert evidence['kind'] == 'comparison'
assert evidence['requested'] == {
    'left_experiment_id': int(sys.argv[2]),
    'right_experiment_id': int(sys.argv[3]),
    'scope': 'final', 'roles': ['left', 'right']}
assert evidence['durable']['left']['side'] == 'left'
assert evidence['durable']['right']['side'] == 'right'
assert evidence['comparability']['controlled_comparison'] is False
assert evidence['comparability']['declared_controlled_pair_provenance'] == 'not_available'
assert evidence['derived']['delta_convention'] == 'right_minus_left'
for arm in ('left', 'right'):
    final = evidence['durable'][arm]['final_inference']
    assert final['candidate_evaluations'] == sorted(
        final['candidate_evaluations'], key=lambda row: row['id'])
    assert all(row['inference_scope'] == 'final'
               and row['checkpoint_eval_id'] is None
               and row['parent_experiment_id'] is None
               for row in final['candidate_evaluations'])
    profits = evidence['durable'][arm]['final_profitability']['candidate_observations']
    assert profits == sorted(profits,
                             key=lambda row: (row['inference_eval_result_id'],
                                              row['profitability_observation_id']))
for group in ('inference_metrics', 'profitability_metrics',
              'profitability_percentage_metrics'):
    for delta in evidence['derived'][group]:
        for key in ('left', 'right', 'right_minus_left', 'relative_to_absolute_left'):
            assert delta[key] is None or math.isfinite(delta[key])
        if delta['left'] is not None and delta['right'] is not None:
            assert delta['right_minus_left'] == delta['right'] - delta['left']
        if delta['left'] == 0:
            assert delta['relative_to_absolute_left'] is None
PY
        if "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence comparison 0 1 >/dev/null 2>&1 || \
           "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence comparison 1 1 >/dev/null 2>&1 || \
           "${LSTM_OBSERVER_EVIDENCE_BINARY}" evidence comparison 1 >/dev/null 2>&1; then
            printf '%s\n' 'invalid comparison grammar unexpectedly succeeded' >&2
            exit 1
        fi
    fi
fi

printf '%s\n' 'SchedulerOperationalEvidenceInterfaceTests passed'
