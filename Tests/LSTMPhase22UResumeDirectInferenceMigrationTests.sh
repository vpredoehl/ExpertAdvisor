#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"
io_file="${repo_root}/Headers/PgModelIO.hpp"

# These assertions deliberately inspect the real production entry point.  They
# guard the two Phase 22U call chains against a later reintroduction of the
# pre-Tensor database loader rather than testing a fabricated detached value.
rg -U -q 'resumeRead\.exec\(\n[[:space:]]*"SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;"\);\n[[:space:]]*selectedModelMaterialization =[\s\S]{0,500}ReadPersistedModelMaterialization\([\s\S]{0,1200}resumeRead\.commit\(\);' "${main_file}"
rg -U -q 'directMaterializationRead\.exec\(\n[[:space:]]*"SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;"\);[\s\S]{0,1200}ReadPersistedModelMaterialization\([\s\S]{0,1200}directMaterializationRead\.commit\(\);' "${main_file}"
rg -U -q 'if \(selectedModelMaterialization\.has_value\(\)\)[\s\S]{0,700}ApplyPersistedModelMaterialization\(' "${main_file}"
rg -q 'PrintMaterializedModelConfigValidation\(' "${main_file}"
rg -q 'const DBIO::PgModelIO::PersistedModelMaterialization\*' "${main_file}"
rg -q 'The applier deliberately accepts no transaction or connection' "${io_file}"

printf '%s\n' 'LSTMPhase22UResumeDirectInferenceMigrationTests passed'
