#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
main="${root}/LSTM/main.cpp"
cpp="${root}/Sources/RuntimeLogging.cpp"
hpp="${root}/Sources/RuntimeLogging.hpp"
test -f "${cpp}"; test -f "${hpp}"
for symbol in LogSummary LogDiagnostic DiagnosticOut ScopedDiagnosticCoutSilencer; do
  rg -q "${symbol}" "${cpp}" "${hpp}"
done
! rg -q '^bool LogSummary\(|^bool LogDiagnostic\(|^std::ostream& DiagnosticOut\(|^class ScopedDiagnosticCoutSilencer' "${main}"
rg -q 'RuntimeLogging.hpp' "${main}"
rg -q 'InstallModelRuntimeValidationDiagnostics\(' "${cpp}" "${main}"
! rg -q 'RuntimeLogging' "${root}/Sources/ModelRuntimeValidation.cpp" "${root}/Sources/ModelRuntimeValidation.hpp"
! rg -q 'EvalLabelConfig|PrintClassificationProofDiagnostics|PrintPhase2TensorDiagnostics|PrintRuntimeLrConfig' "${cpp}" "${hpp}"
rg -q 'struct EvalLabelConfig' "${main}"
rg -q 'void PrintClassificationProofDiagnostics' "${main}"
rg -q 'void PrintPhase2TensorDiagnostics' "${main}"
rg -q 'void PrintRuntimeLrConfig' "${main}"
printf '%s\n' 'RuntimeLoggingStructuralTests passed'
