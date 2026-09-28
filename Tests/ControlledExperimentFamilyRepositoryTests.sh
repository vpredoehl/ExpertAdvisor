#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; work="$(mktemp -d /tmp/ea_controlled_family.XXXXXX)"; pg=/opt/homebrew/opt/postgresql@17/bin; data="$work/data"; socket="$work/socket"; db="ea_controlled_family_${$}"; port="$((59000 + ($$ % 500)))"; started=false
trap 's=$?; if $started; then "$pg/pg_ctl" -D "$data" -m immediate -w stop >/dev/null || true; fi; rm -rf "$work"; exit $s' EXIT
mkdir -p "$socket"; "$pg/initdb" -D "$data" --username=pqxx --auth=trust --no-locale >/dev/null; "$pg/pg_ctl" -D "$data" -o "-F -k $socket -p $port -h ''" -w start >/dev/null; started=true; "$pg/createdb" -h "$socket" -p "$port" -U pqxx "$db"
psql=("$pg/psql" -X -v ON_ERROR_STOP=1 -q -h "$socket" -p "$port" -U pqxx -d "$db")
"${psql[@]}" <<'SQL'
CREATE TABLE economic_calendar_snapshot (economic_calendar_snapshot_id bigint PRIMARY KEY);
INSERT INTO economic_calendar_snapshot VALUES (1);
CREATE TABLE experiment (experiment_id bigint PRIMARY KEY);
INSERT INTO experiment VALUES (1);
SQL
"${psql[@]}" -f "$root/Database/migrations/098_controlled_experiment_family_foundation.sql"
"${psql[@]}" -Atqc "SELECT controlled_experiment_family_member_id IS NULL FROM experiment WHERE experiment_id=1" | grep -qx t
clang++ -std=c++20 -Wall -Wextra -Werror -I"$root/Sources" "$root/Tests/ControlledExperimentFamilyRepositoryTests.cpp" "$root/Sources/ControlledExperimentFamilyRepository.cpp" -I/opt/homebrew/opt/libpqxx@7.10.1/include -L/opt/homebrew/opt/libpqxx@7.10.1/lib -L/opt/homebrew/opt/libpq/lib -lpqxx -lpq -o "$work/test"
EA_CONTROLLED_FAMILY_DB_HOST="$socket" EA_CONTROLLED_FAMILY_DB_PORT="$port" EA_CONTROLLED_FAMILY_DB_NAME="$db" "$work/test"
echo CONTROLLED_EXPERIMENT_FAMILY_REPOSITORY_TESTS=passed
