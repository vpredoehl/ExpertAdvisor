#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
postgres_bin="/opt/homebrew/opt/postgresql@17/bin"
runtime_dir="$(mktemp -d "/tmp/ea_replication_lock.XXXXXX")"
cluster_dir="${runtime_dir}/pgdata"
socket_dir="${runtime_dir}/socket"
database_name="ea_replication_materializer_lock_${$}_${RANDOM}"
port="$((57000 + (RANDOM % 1000)))"
server_started=false

cleanup() {
    local status=$?
    if [[ "${server_started}" == true ]]; then
        "${postgres_bin}/pg_ctl" -D "${cluster_dir}" -m immediate -w stop \
            >/dev/null || true
    fi
    rm -rf "${runtime_dir}"
    exit "${status}"
}
trap cleanup EXIT

if [[ ! "${database_name}" =~ ^ea_replication_materializer_lock_[A-Za-z0-9_]+$ ]]; then
    printf '%s\n' 'unsafe disposable database name' >&2
    exit 1
fi

mkdir -p "${socket_dir}"
"${postgres_bin}/initdb" -D "${cluster_dir}" --username=pqxx \
    --auth=trust --no-locale >/dev/null
if ! "${postgres_bin}/pg_ctl" -D "${cluster_dir}" \
    -l "${runtime_dir}/postgres.log" \
    -o "-F -k ${socket_dir} -p ${port} -h ''" -w start >/dev/null; then
    cat "${runtime_dir}/postgres.log" >&2
    exit 1
fi
server_started=true
"${postgres_bin}/createdb" -h "${socket_dir}" -p "${port}" -U pqxx \
    "${database_name}"

psql_target=("${postgres_bin}/psql" -X -q -h "${socket_dir}" -p "${port}" \
    -U pqxx -d "${database_name}")
"${psql_target[@]}" -c \
    'CREATE TABLE experiment(experiment_id bigint PRIMARY KEY, identity text);'

# A generic pre-existing writer holds ROW EXCLUSIVE while inserting. The
# materializer's SHARE ROW EXCLUSIVE request must wait for it, then the first
# SERIALIZABLE read after lock acquisition must observe the committed row.
"${psql_target[@]}" -c \
    "BEGIN; INSERT INTO experiment VALUES(1,'equivalent'); SELECT pg_sleep(2); COMMIT;" \
    >"${runtime_dir}/writer.out" 2>"${runtime_dir}/writer.err" &
writer_pid=$!
sleep 0.5
observed=$("${psql_target[@]}" -At -c \
    "BEGIN ISOLATION LEVEL SERIALIZABLE; LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE; SELECT count(*) FROM experiment WHERE identity='equivalent'; COMMIT;")
wait "${writer_pid}"
[[ "${observed}" == "1" ]]

# Two materializer-shaped transactions cannot both pass the protected
# equivalence check. The first inserts while holding the table lock; the second
# waits, sees the equivalent, and conditionally inserts zero rows.
"${psql_target[@]}" -c \
    "BEGIN ISOLATION LEVEL SERIALIZABLE; LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE; INSERT INTO experiment VALUES(2,'wave'); SELECT pg_sleep(2); COMMIT;" \
    >"${runtime_dir}/first.out" 2>"${runtime_dir}/first.err" &
first_pid=$!
sleep 0.5
second_inserted=$("${psql_target[@]}" -At -c \
    "BEGIN ISOLATION LEVEL SERIALIZABLE; LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE; WITH inserted AS (INSERT INTO experiment SELECT 3,'wave' WHERE NOT EXISTS (SELECT 1 FROM experiment WHERE identity='wave') RETURNING 1) SELECT count(*) FROM inserted; COMMIT;")
wait "${first_pid}"
[[ "${second_inserted}" == "0" ]]
[[ "$("${psql_target[@]}" -At -c \
    "SELECT count(*) FROM experiment WHERE identity='wave';")" == "1" ]]

command_source="${repo_root}/Sources/ExperimentReplicationMaterializationCommand.cpp"
lock_line="$(rg -n 'LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE' \
    "${command_source}" | cut -d: -f1)"
run_line="$(rg -n 'RunMaterializationInTransaction' "${command_source}" | \
    cut -d: -f1)"
[[ -n "${lock_line}" && -n "${run_line}" ]]
(( lock_line < run_line ))

printf '%s\n' 'Experiment replication materialization concurrency tests passed'
