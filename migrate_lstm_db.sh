#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_DIR="${SCRIPT_DIR}/Database/migrations"

LSTM_DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
LSTM_DB_NAME="${LSTM_DB_NAME:-LSTM}"
DEFAULT_ADMIN_USER="${USER:-vjp}"
LSTM_DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${DEFAULT_ADMIN_USER}}"

PSQL=(psql -X -q -v ON_ERROR_STOP=1 -h "${LSTM_DB_HOST}" -U "${LSTM_DB_ADMIN_USER}" -d "${LSTM_DB_NAME}")

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
tmp_body=""
tmp_sql=""

cleanup_temporary_sql() {
    [[ -z "${tmp_body}" ]] || rm -f -- "${tmp_body}"
    [[ -z "${tmp_sql}" ]] || rm -f -- "${tmp_sql}"
}
trap cleanup_temporary_sql EXIT

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
    tmp_body="$(mktemp)"
    tmp_sql="$(mktemp)"

    # Some historical migrations include their own whole-file transaction
    # wrapper.  Preserve the checked-in bytes (and therefore their checksum),
    # but remove only a matched outer BEGIN/COMMIT pair for execution so the
    # runner remains the sole owner of the migration transaction.
    awk '
        function sql_line(line, value) {
            value = line
            sub(/^[[:space:]]*/, "", value)
            sub(/[[:space:]]*--.*$/, "", value)
            sub(/[[:space:]]*$/, "", value)
            return toupper(value)
        }
        {
            lines[NR] = $0
            value = sql_line($0)
            if (value == "") {
                next
            }
            if (first_sql_line == 0) {
                first_sql_line = NR
            }
            last_sql_line = NR
        }
        END {
            first_sql = sql_line(lines[first_sql_line])
            last_sql = sql_line(lines[last_sql_line])
            if (first_sql ~ /^BEGIN([[:space:]]+(WORK|TRANSACTION))?[[:space:]]*;$/ &&
                last_sql ~ /^(COMMIT|END)([[:space:]]+(WORK|TRANSACTION))?[[:space:]]*;$/) {
                omit[first_sql_line] = 1
                omit[last_sql_line] = 1
            }
            for (line_number = 1; line_number <= NR; ++line_number) {
                if (!omit[line_number]) {
                    print lines[line_number]
                }
            }
        }
    ' "${migration}" > "${tmp_body}"

    # Any transaction-control statement left after outer-wrapper
    # normalization could escape the runner-owned transaction.  Track
    # top-level lexical state and fail closed on remaining transaction control
    # rather than risking a partial commit.  A bare END; is ambiguous: it is
    # transaction control at statement level, but closes a SQL CASE expression
    # while a CASE is open.
    if ! awk '
        function transaction_control(line, value) {
            value = line
            sub(/^[[:space:]]*/, "", value)
            sub(/[[:space:]]*--.*$/, "", value)
            sub(/[[:space:]]*$/, "", value)
            value = toupper(value)
            return value ~ /^BEGIN([[:space:]].*)?;$/ ||
                   value ~ /^START[[:space:]]+TRANSACTION([[:space:]].*)?;$/ ||
                   value ~ /^COMMIT([[:space:]]+(WORK|TRANSACTION))?([[:space:]]+AND[[:space:]]+(NO[[:space:]]+)?CHAIN)?[[:space:]]*;$/ ||
                   value ~ /^END[[:space:]]+(WORK|TRANSACTION)([[:space:]]+AND[[:space:]]+(NO[[:space:]]+)?CHAIN)?[[:space:]]*;$/ ||
                   value ~ /^END[[:space:]]+AND[[:space:]]+(NO[[:space:]]+)?CHAIN[[:space:]]*;$/ ||
                   (case_depth == 0 && value ~ /^END[[:space:]]*;$/) ||
                   value ~ /^(COMMIT|ROLLBACK)[[:space:]]+PREPARED([[:space:]].*)?;$/ ||
                   value ~ /^(ROLLBACK|ABORT)([[:space:]].*)?;$/ ||
                   value ~ /^PREPARE[[:space:]]+TRANSACTION([[:space:]].*)?;$/
        }

        # Count CASE/END tokens outside literals and comments.  This is not a
        # SQL parser; it only resolves the one otherwise ambiguous whole-line
        # statement, END;.  Other transaction-control spellings are never
        # accepted merely because a CASE is open.
        function update_case_depth(line, i, line_length, word, remainder) {
            line_length = length(line)
            for (i = 1; i <= line_length;) {
                if (dollar_quote != "") {
                    if (substr(line, i, length(dollar_quote)) == dollar_quote) {
                        i += length(dollar_quote)
                        dollar_quote = ""
                    } else {
                        ++i
                    }
                    continue
                }
                if (block_comment_depth > 0) {
                    if (substr(line, i, 2) == "/*") {
                        ++block_comment_depth
                        i += 2
                    } else if (substr(line, i, 2) == "*/") {
                        --block_comment_depth
                        i += 2
                    } else {
                        ++i
                    }
                    continue
                }
                if (single_quote) {
                    if (substr(line, i, 1) == "\\") {
                        i += 2
                    } else if (substr(line, i, 1) == "\047") {
                        if (substr(line, i + 1, 1) == "\047") {
                            i += 2
                        } else {
                            single_quote = 0
                            ++i
                        }
                    } else {
                        ++i
                    }
                    continue
                }
                if (double_quote) {
                    if (substr(line, i, 1) == "\042") {
                        if (substr(line, i + 1, 1) == "\042") {
                            i += 2
                        } else {
                            double_quote = 0
                            ++i
                        }
                    } else {
                        ++i
                    }
                    continue
                }

                if (substr(line, i, 2) == "--") {
                    return
                }
                if (substr(line, i, 2) == "/*") {
                    ++block_comment_depth
                    i += 2
                    continue
                }
                if (substr(line, i, 1) == "\047") {
                    single_quote = 1
                    ++i
                    continue
                }
                if (substr(line, i, 1) == "\042") {
                    double_quote = 1
                    ++i
                    continue
                }

                remainder = substr(line, i)
                if (match(remainder, /^\$[A-Za-z_][A-Za-z0-9_]*\$|^\$\$/)) {
                    dollar_quote = substr(remainder, 1, RLENGTH)
                    i += RLENGTH
                    continue
                }
                if (match(remainder, /^[A-Za-z_][A-Za-z0-9_$]*/)) {
                    word = toupper(substr(remainder, 1, RLENGTH))
                    if (word == "CASE") {
                        ++case_depth
                    } else if (word == "END" && case_depth > 0) {
                        --case_depth
                    }
                    i += RLENGTH
                    continue
                }
                ++i
            }
        }
        {
            if (dollar_quote == "" && !single_quote &&
                block_comment_depth == 0 && transaction_control($0)) {
                exit 1
            }
            update_case_depth($0)
        }
    ' "${tmp_body}"; then
        echo "MIGRATION_ERROR,version=${version},filename=${filename},reason=unsupported_transaction_control" >&2
        exit 1
    fi

    {
        echo "SET client_min_messages TO warning;"
        echo "BEGIN;"
        cat "${tmp_body}"
        printf "\nINSERT INTO schema_migrations (version, filename, checksum) VALUES ('%s', '%s', '%s');\n" \
            "${version//\'/\'\'}" \
            "${filename//\'/\'\'}" \
            "${checksum//\'/\'\'}"
        echo "COMMIT;"
    } > "${tmp_sql}"

    "${PSQL[@]}" -f "${tmp_sql}"
    cleanup_temporary_sql
    tmp_body=""
    tmp_sql=""
    applied=$((applied + 1))
done

echo "MIGRATION_DONE,applied=${applied},skipped=${skipped},database=${LSTM_DB_NAME},host=${LSTM_DB_HOST}"
