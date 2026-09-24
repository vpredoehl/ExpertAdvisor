#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

function_body="$(sed -n '/bool ApplyStructuredInferenceMetrics/,/^}/p' "${source_file}")"

# Final-inference identity follows the established repository tolerance.  The
# production representation mismatch (0.0007999999797903001 versus
# 0.00079999998) is inside the boundary; a difference strictly greater than
# 1e-7 is not.  The predicate keeps exact matches as a subset.
rg -Fq 'AND abs(threshold_logret - $4) <= 1e-7 ' <<<"${function_body}"
python3 - <<'PY'
threshold = 0.0007999999797903001
within = 0.00079999998
outside = threshold + 1.0000001e-7
tolerance = 1e-7
assert abs(threshold - within) <= tolerance
assert not abs(threshold - outside) <= tolerance
assert abs(threshold - threshold) <= tolerance
PY

# A selected structured row overrides log-derived final metrics and thereby
# supplies the values persisted to experiment_analysis_result.
for assignment in \
    'metrics.inferAccuracy = rows[0][0].as<double>();' \
    'metrics.acceptModel = rows[0][1].as<bool>();' \
    'metrics.completedEpochs = rows[0][3].as<int>();' \
    'metrics.acceptAccuracy = metrics.inferAccuracy;' \
    'metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;'
do
    rg -Fq "${assignment}" <<<"${function_body}"
done

printf '%s\n' 'StructuredAnalysisInferenceThresholdTests passed'
