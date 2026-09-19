#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
daemon="${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
configuration="${repo_root}/Sources/SchedulerCore/SchedulerDaemonConfiguration.cpp"
legacy="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

# ANALYZE is role-routed independently of semantic worker selection.
rg -U -q 'BuildAnalyzeCommand[\s\S]*argv\.push_back\(options\.analyzeWorkerExecutablePath\)' "${daemon}"
rg -U -q 'selectedWorkerExecutable =\n        phase == "analyze"\n            \? options\.analyzeWorkerExecutablePath' "${daemon}"
rg -q 'ResolveAnalyzeWorkerExecutablePath' "${daemon}"
rg -q '"lstm-analyze-worker"' "${daemon}"
rg -q -- '--analyze-worker' "${configuration}"

# The managed command keeps the Phase 22B contract. The attempt ID is appended
# only after canonical executable identity matches the reserved attempt.
rg -U -q 'BuildAnalyzeCommand[\s\S]*--analyze-experiment[\s\S]*--auto-generate-reports[\s\S]*--experiment-report-dir' "${daemon}"
rg -U -q 'ValidateAndCanonicalizeWorkerExecutable[\s\S]*attempt\.canonicalExecutablePath[\s\S]*--scheduler-worker-attempt-id' "${daemon}"

# Existing compatibility and generic persisted-attempt reconciliation remain.
rg -q 'AnalyzeExperimentById' "${legacy}"
rg -q 'RecoverOrphanedRunningExperiments' "${daemon}"
rg -q 'canonicalExecutablePath' "${daemon}"

# No analysis artifact is added to semantic-worker selection.
if rg -q 'analyzeWorkerExecutablePath' "${repo_root}/Sources/SchedulerCore/SemanticWorkerRegistry.cpp"; then
    echo "analyze worker leaked into semantic worker registry" >&2
    exit 1
fi

printf '%s\n' "SchedulerAnalyzeWorkerRoutingTests passed"
