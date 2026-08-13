#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_DIR="${SCRIPT_DIR}/Database/migrations"

LSTM_DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
LSTM_DB_NAME="${LSTM_DB_NAME:-LSTM}"
DEFAULT_ADMIN_USER="${USER:-vjp}"
LSTM_DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${DEFAULT_ADMIN_USER}}"

PSQL=(psql -q -v ON_ERROR_STOP=1 -h "${LSTM_DB_HOST}" -U "${LSTM_DB_ADMIN_USER}" -d "${LSTM_DB_NAME}")

if [[ ! -d "${MIGRATION_DIR}" ]]; then
    echo "MIGRATION_ERROR,reason=missing_migration_dir,path=${MIGRATION_DIR}" >&2
    exit 1
fi

"${PSQL[@]}" <<'SQL'
SET client_min_messages TO warning;
CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    filename text NOT NULL,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
SQL

applied=0
skipped=0

shopt -s nullglob
for migration in "${MIGRATION_DIR}"/*.sql; do
    filename="$(basename "${migration}")"
    version="${filename%%_*}"
    if [[ "${version}" == "${filename}" ]]; then
        version="${filename%.sql}"
    fi

    checksum="$(shasum -a 256 "${migration}" | awk '{print $1}')"
    escaped_version="${version//\'/\'\'}"
    existing_checksum="$("${PSQL[@]}" -At \
        -c "SELECT checksum FROM schema_migrations WHERE version = '${escaped_version}';")"

    if [[ -n "${existing_checksum}" ]]; then
        if [[ "${existing_checksum}" != "${checksum}" ]]; then
            echo "MIGRATION_ERROR,version=${version},filename=${filename},reason=checksum_mismatch" >&2
            exit 1
        fi
        echo "MIGRATION_SKIP,version=${version},filename=${filename}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "MIGRATION_APPLY,version=${version},filename=${filename}"
    tmp_sql="$(mktemp)"
    {
        echo "SET client_min_messages TO warning;"
        echo "BEGIN;"
        cat "${migration}"
        printf "\nINSERT INTO schema_migrations (version, filename, checksum) VALUES ('%s', '%s', '%s');\n" \
            "${version//\'/\'\'}" \
            "${filename//\'/\'\'}" \
            "${checksum//\'/\'\'}"
        echo "COMMIT;"
    } > "${tmp_sql}"

    "${PSQL[@]}" -f "${tmp_sql}"
    rm -f "${tmp_sql}"
    applied=$((applied + 1))
done

echo "MIGRATION_DONE,applied=${applied},skipped=${skipped},database=${LSTM_DB_NAME},host=${LSTM_DB_HOST}"
