#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/Build/bea_economic_event_import_repository_tests"
BIN="$BUILD_DIR/BeaEconomicEventImportRepositoryTests"
CLI_BIN="$BUILD_DIR/EconomicEventImportCliHarness"
DB_NAME="ea_economic_calendar_phase3_bea_001"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
FIXTURES="$ROOT/Tests/fixtures/economic_calendar/bea"
mkdir -p "$BUILD_DIR"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi

COMMON_SOURCES=(
    "$ROOT/Common/HistoricalFxTimestamp.cpp"
    "$ROOT/Sources/BeaEconomicReleaseAdapter.cpp"
    "$ROOT/Sources/CensusEconomicReleaseAdapter.cpp"
    "$ROOT/Sources/DolEtaWeeklyClaimsAdapter.cpp"
    "$ROOT/Sources/EconomicEventImportValidation.cpp"
    "$ROOT/Sources/EconomicEventImportRepository.cpp"
    "$ROOT/Sources/EconomicEventImportService.cpp"
)

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "${COMMON_SOURCES[@]}" \
    "$ROOT/Tests/BeaEconomicEventImportRepositoryTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$BIN"

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "${COMMON_SOURCES[@]}" \
    "$ROOT/Tests/EconomicEventImportCliHarness.cpp" \
    "${PQXX_LIBS[@]}" -o "$CLI_BIN"

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" "$DB_NAME" >/dev/null
}
trap cleanup EXIT

if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    echo "Refusing to reuse existing disposable database: $DB_NAME" >&2
    exit 1
fi

createdb --host="$DB_HOST" --username="$DB_USER" --template=template0 "$DB_NAME"
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" -f "$ROOT/Database/migrations/072_economic_event.sql" >/dev/null

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" "$BIN"

ROWS_BEFORE="$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event;')"
DRY_RUN_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest.tsv" --dry-run)"
grep -Fq 'ECONOMIC_EVENT_IMPORT_SUMMARY,agency=bea,mode=dry-run,inserted=6,unchanged=0,updated=0,rejected=0' \
    <<< "$DRY_RUN_OUTPUT"
ROWS_AFTER="$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event;')"
test "$ROWS_BEFORE" = "$ROWS_AFTER"

FIRST_APPLY_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest.tsv" --apply)"
grep -Fq 'ECONOMIC_EVENT_IMPORT_SUMMARY,agency=bea,mode=apply,inserted=6,unchanged=0,updated=0,rejected=0' \
    <<< "$FIRST_APPLY_OUTPUT"
SECOND_APPLY_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest.tsv" --apply)"
grep -Fq 'ECONOMIC_EVENT_IMPORT_SUMMARY,agency=bea,mode=apply,inserted=0,unchanged=6,updated=0,rejected=0' \
    <<< "$SECOND_APPLY_OUTPUT"

if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea --manifest="$FIXTURES/manifest.tsv"; then
    echo "CLI unexpectedly accepted missing mode" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events census \
    --manifest="$FIXTURES/manifest.tsv" --dry-run; then
    echo "CLI unexpectedly accepted unsupported agency" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea --dry-run; then
    echo "CLI unexpectedly accepted missing manifest" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest.tsv" --dry-run --apply; then
    echo "CLI unexpectedly accepted multiple modes" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest_bad_hash.tsv" --dry-run; then
    echo "CLI unexpectedly accepted hash mismatch" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest_malformed.tsv" --dry-run; then
    echo "CLI unexpectedly accepted malformed manifest" >&2
    exit 1
fi
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$CLI_BIN" --import-economic-events bea \
    --manifest="$FIXTURES/manifest_unsupported_type.tsv" --dry-run; then
    echo "CLI unexpectedly accepted unsupported artifact type" >&2
    exit 1
fi

test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT count(*) FROM economic_event WHERE source_agency = 'DOL_ETA';")" = "2"

echo "DISPOSABLE_DATABASE_USED=$DB_NAME"
cleanup
trap - EXIT
echo "DISPOSABLE_DATABASE_DROPPED=true"
