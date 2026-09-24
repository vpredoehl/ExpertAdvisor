#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; work="$(mktemp -d "${TMPDIR:-/tmp}/ea_migration095.XXXXXX")"
pgdata="$work/pgdata"; socket="$work/socket"; db="ea_migration095_${$}"; port="$((57000 + ($$ % 1000)))"; postgres=/opt/homebrew/opt/postgresql@17/bin; started=false; created=false
trap 's=$?; if $created; then "$postgres/dropdb" -h "$socket" -p "$port" -U pqxx "$db" || true; fi; if $started; then "$postgres/pg_ctl" -D "$pgdata" -m fast -w stop >/dev/null || true; fi; rm -rf "$work"; exit $s' EXIT
mkdir -p "$socket"; "$postgres/initdb" -D "$pgdata" --username=pqxx --auth=trust --no-locale >/dev/null; "$postgres/pg_ctl" -D "$pgdata" -o "-F -k $socket -p $port -h ''" -w start >/dev/null; started=true; "$postgres/createdb" -h "$socket" -p "$port" -U pqxx "$db"; created=true
psql=("$postgres/psql" -X -v ON_ERROR_STOP=1 -q -h "$socket" -p "$port" -U pqxx -d "$db")
"${psql[@]}" <<'SQL'
CREATE TABLE experiment (experiment_id bigint PRIMARY KEY);
CREATE TABLE experiment_checkpoint_eval (checkpoint_eval_id bigint PRIMARY KEY);
CREATE TABLE experiment_scheduler_worker_attempt (worker_attempt_id bigint PRIMARY KEY, experiment_id bigint NOT NULL REFERENCES experiment, worker_kind text NOT NULL, lifecycle_phase text NOT NULL, capacity_class text NOT NULL, checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval);
CREATE TABLE model (model_id bigint PRIMARY KEY, experiment_id bigint NOT NULL REFERENCES experiment);
CREATE TABLE inference_eval_result (id bigint PRIMARY KEY, model_id bigint NOT NULL REFERENCES model, inference_scope text NOT NULL, checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval);
INSERT INTO experiment VALUES (1),(2); INSERT INTO experiment_checkpoint_eval VALUES (10),(11); INSERT INTO model VALUES (1,1),(2,2);
INSERT INTO experiment_scheduler_worker_attempt VALUES
 (100,1,'experiment','train','train',NULL),(101,2,'experiment','train','train',NULL),(102,1,'checkpoint_infer','train','train',NULL),
 (103,1,'experiment','infer','train',NULL),(104,1,'experiment','train','infer',NULL),
 (200,1,'experiment','infer','infer',NULL),(201,1,'checkpoint_infer','infer','infer',10),(202,1,'checkpoint_infer','infer','infer',11),(203,1,'checkpoint_infer','infer','infer',NULL),(204,1,'experiment','infer','infer',10);
SQL
"${psql[@]}" -f "$root/Database/migrations/096_scientific_execution_provenance.sql"
has(){ "${psql[@]}" -At -c "$1" | grep -qx t; }
has "SELECT count(*)=7 FROM information_schema.columns WHERE table_name='experiment_scheduler_worker_attempt' AND column_name IN ('semantic_layout_version','model_input_width','semantic_worker_role','source_commit','executable_sha256','runtime_identity','canonical_manifest_path')"
has "SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='model'::regclass AND contype='f' AND confdeltype='r')"
has "SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='inference_eval_result'::regclass AND contype='f' AND confdeltype='r')"
sql(){ "${psql[@]}" -c "$1"; }; reject(){ if sql "$1" >/dev/null 2>&1; then echo "expected rejection: $1" >&2; exit 1; fi; }
sql "UPDATE model SET producer_worker_attempt_id=100 WHERE model_id=1"
reject "UPDATE model SET producer_worker_attempt_id=101 WHERE model_id=1"
reject "UPDATE model SET producer_worker_attempt_id=102 WHERE model_id=1"
reject "UPDATE model SET producer_worker_attempt_id=103 WHERE model_id=1"
reject "UPDATE model SET producer_worker_attempt_id=104 WHERE model_id=1"
sql "INSERT INTO inference_eval_result VALUES (1,1,'final',NULL)"; sql "UPDATE inference_eval_result SET producer_worker_attempt_id=200 WHERE id=1"
reject "UPDATE inference_eval_result SET inference_scope='checkpoint' WHERE id=1"
reject "UPDATE inference_eval_result SET model_id=2 WHERE id=1"
sql "UPDATE inference_eval_result SET model_id=1 WHERE id=1"
sql "INSERT INTO inference_eval_result VALUES (2,1,'final',NULL)"; reject "UPDATE inference_eval_result SET producer_worker_attempt_id=201 WHERE id=2"
sql "INSERT INTO inference_eval_result VALUES (7,1,'final',NULL)"; reject "UPDATE inference_eval_result SET producer_worker_attempt_id=204 WHERE id=7"
sql "INSERT INTO inference_eval_result VALUES (3,1,'checkpoint',10)"; sql "UPDATE inference_eval_result SET producer_worker_attempt_id=201 WHERE id=3"
reject "UPDATE inference_eval_result SET inference_scope='final',checkpoint_eval_id=NULL WHERE id=3"
reject "UPDATE inference_eval_result SET checkpoint_eval_id=11 WHERE id=3"
sql "UPDATE inference_eval_result SET checkpoint_eval_id=10 WHERE id=3"
sql "INSERT INTO inference_eval_result VALUES (4,1,'checkpoint',11)"; reject "UPDATE inference_eval_result SET producer_worker_attempt_id=201 WHERE id=4"
sql "INSERT INTO inference_eval_result VALUES (5,2,'final',NULL)"; reject "UPDATE inference_eval_result SET producer_worker_attempt_id=200 WHERE id=5"
sql "INSERT INTO inference_eval_result VALUES (6,1,'other',NULL)"; reject "UPDATE inference_eval_result SET producer_worker_attempt_id=200 WHERE id=6"
reject "DELETE FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=100"
echo SCIENTIFIC_EXECUTION_PROVENANCE_MIGRATION_096_DISPOSABLE_TESTS=passed
