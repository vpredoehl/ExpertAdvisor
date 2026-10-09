#!/usr/bin/env python3
"""Restore genuine archived evidence and run at most one managed ANALYZE worker.

Default is offline preparation only. --single-worker enables a disposable
SCRAM cluster and the existing requeue-analysis/scheduler workflows. Never
connects to production, rewrites scientific input, or runs TRAIN/INFER.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time

from AnalyzeResourceQualificationPreflight import Collector, ROOT, PG_RESTORE, copy_rows

if not __debug__:
    raise RuntimeError("qualification safety assertions require Python without -O")

PG = PG_RESTORE.parent
DEFAULT_ARCHIVE = ROOT / "DerivedData/ExpertAdvisor/Phase24R/backup/LSTM-pre-cutover.dump"
ROUTING = ROOT / "DerivedData/ExpertAdvisor/Phase24R/publication-rollback-rehearsal"
PRODUCTS = ROOT / "DerivedData/ExpertAdvisor/Build/Products/Release"
OBSERVER = ROOT / "DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native18/GlobalExperimentControlProcessTests"
TERMINAL = {"completed", "failed", "launch_failed", "abandoned"}


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def copy_sql(table, rows):
    if not rows:
        return ""
    columns = list(rows[0])
    assert all(list(row) == columns for row in rows)
    return (f"COPY public.{table} ({', '.join(columns)}) FROM stdin;\n" +
            "\n".join("\t".join(row[column] for column in columns) for row in rows) +
            "\n\\.\n")


def verify_analysis(actual, expected):
    """Require every scientific field to match; only update timestamps may differ."""
    for key, raw in expected.items():
        if key in ("created_at", "updated_at"):
            continue
        value = actual[key]
        if raw == "\\N":
            assert value is None, key
        elif isinstance(value, (int, float)):
            assert math.isfinite(float(value)) and abs(float(value) - float(raw)) < 1e-12, key
        else:
            assert value == raw, key


def select_fixture_registry(registry, fixture):
    """Keep genuine current routes and the exact two historical producers."""
    attempts = {x["worker_attempt_id"]: x for x in fixture["historical_attempts"]}
    selected = [w for w in registry["workers"] if w["worker_rule"] == "current"]
    assert len(selected) == 2 and {w["worker_role"] for w in selected} == {"train", "infer"}
    assert all(w["semantic_layout"] == registry["current_layout"] for w in selected)
    assert set(fixture["producer_worker_attempt_ids"]) == {"train", "infer"}
    for role, producer in fixture["producer_worker_attempt_ids"].items():
        attempt = attempts[producer]
        assert attempt["lifecycle_phase"] == role and attempt["experiment_id"] == fixture["experiment_id"]
        matches = [w for w in registry["workers"] if
                   w["worker_role"] == role and
                   w["semantic_layout"] == int(attempt["semantic_layout_version"]) and
                   w["model_input_width"] == int(attempt["model_input_width"]) and
                   w["source_commit"] == attempt["source_commit"] and
                   w["sha256"] == attempt["executable_sha256"] and
                   w["runtime_identity"] == attempt["runtime_identity"]]
        worker, = matches
        if worker not in selected:
            selected.append(worker)
    runtime_ids = {w["runtime_identity"] for w in selected}
    runtimes = [r for r in registry["runtimes"] if r["identity"] in runtime_ids]
    assert {r["identity"] for r in runtimes} == runtime_ids
    return {**registry, "workers": selected, "runtimes": runtimes}


def verify_analysis_source(analysis, experiment, model, inference):
    """Bind the original comparison baseline to this exact final evidence."""
    assert analysis["experiment_id"] == experiment["experiment_id"]
    assert analysis["model_id"] == model["model_id"] == inference["model_id"]
    assert analysis["analysis_scope"] == "final" and analysis["checkpoint_eval_id"] == "\\N"
    assert analysis["analysis_status"] == "completed"
    for name in ("symbol", "prediction_horizon", "target_epochs"):
        assert analysis[name] == experiment[name]
    assert analysis["completed_epochs"] == inference["completed_epochs"]
    assert float(analysis["infer_accuracy"]) == float(inference["accuracy"])


def select_snapshot_invocations(rows, attempts, leases):
    """Close terminal attempt/lease FKs without changing ownership records."""
    required = {a["scheduler_invocation_id"] for a in attempts}
    for attempt in attempts:
        for name in ("observed_by_scheduler_invocation_id", "reconciled_by_scheduler_invocation_id"):
            if attempt[name] not in ("", "\\N"):
                required.add(attempt[name])
    for lease in leases:
        owner = lease["owner_scheduler_invocation_id"]
        if owner not in ("", "\\N"):
            required.add(owner)
    selected = [r for r in rows if r["scheduler_invocation_id"] in required]
    assert {r["scheduler_invocation_id"] for r in selected} == required
    assert all(r["ended_at"] != "\\N" for r in selected), "live invocation cannot enter historical snapshot"
    return selected


def verify_chain(experiment, model, inference, attempts, train, infer):
    eid, mid = experiment["experiment_id"], model["model_id"]
    assert experiment["status"] == "completed" and experiment["phase"] == "done"
    assert experiment["worker_pid"] == "\\N"
    assert experiment["active_scheduler_worker_attempt_id"] == "\\N"
    assert experiment["last_model_id"] == mid and model["experiment_id"] == eid
    assert model["parent_model_id"] == "\\N", "ancestry closure requires separate review"
    assert inference["model_id"] == mid and inference["status"] == "completed"
    assert inference["inference_scope"] == "final" and inference["checkpoint_eval_id"] == "\\N"
    assert inference["symbol"] == experiment["symbol"]
    assert inference["prediction_horizon"] == experiment["prediction_horizon"]
    assert abs(float(inference["threshold_logret"]) - float(experiment["c_next_threshold"])) <= 1e-7
    for name in ("from_date", "to_date"):
        assert inference[name][:10] == experiment["infer_start" if name == "from_date" else "infer_end"][:10]
    assert inference["completed_epochs"] == experiment["target_epochs"]
    assert math.isfinite(float(inference["accuracy"]))
    for phase, producer, text in (
            ("train", model["producer_worker_attempt_id"], train),
            ("infer", inference["producer_worker_attempt_id"], infer)):
        attempt = attempts[producer]
        assert attempt["experiment_id"] == eid
        assert attempt["worker_kind"] == "experiment"
        assert attempt["lifecycle_phase"] == phase and attempt["capacity_class"] == phase
        assert attempt["lifecycle_state"] == "completed" and attempt["exit_code"] == "0"
        for field in ("worker_pid", "worker_process_group_id", "worker_process_start_identity",
                      "canonical_executable_path", "source_commit", "executable_sha256",
                      "runtime_identity", "canonical_manifest_path", "registered_at"):
            assert attempt[field] not in ("", "\\N"), field
        assert f"worker_attempt_id={producer},experiment_id={eid}," in text
        assert f"phase={phase},pid={attempt['worker_pid']}" in text
        assert f"--scheduler-experiment-id={eid}" in attempt["command_line"]
        assert f"--scheduler-worker-attempt-id={producer}" in attempt["command_line"]
        assert attempt["model_input_width"] == experiment["model_input_width"]
        assert attempt["semantic_layout_version"] == experiment["model_input_semantic_layout_version"]
    assert f"Saved model with model_id={mid}" in train
    assert train.count("EPOCH_3CLASS_ACCURACY") == int(experiment["target_epochs"])
    assert (f"MODEL_INPUT_IDENTITY_ACTIVE,experiment_id={eid},model_input_width={experiment['model_input_width']},"
            f"semantic_layout={experiment['model_input_semantic_layout_version']}") in train
    assert (f"TRAINING_OBJECTIVE_ACTIVE,experiment_id={eid},objective_id={experiment['training_objective_id']},"
            f"objective_hash={experiment['training_objective_hash']}") in train
    if "TRAIN_FEATURE_ABLATION_ACTIVE," in train:
        assert f"persisted_mask={experiment['feature_ablation_mask']},effective_mask={experiment['feature_ablation_mask']}," in train
    for text in (train, infer):
        assert (f"ECONOMIC_CALENDAR_CORPUS,behavior=immutable_snapshot,snapshot_id={experiment['economic_calendar_snapshot_id']},"
                f"content_hash={experiment['economic_calendar_snapshot_hash']}") in text
    assert f"--fresh-initialization-seed={experiment['fresh_initialization_seed']}" in attempts[model["producer_worker_attempt_id"]]["command_line"]
    assert f"stage=persistence_transaction_committed,model_id={mid}," in infer
    assert f"stage=managed_application_success_return,model_id={mid}," in infer


def stream_matrix(collector, archive, model_id):
    """Decode with a deadline and bounded buffers; retain exact selected rows."""
    argv = [str(PG_RESTORE), "--data-only", "--schema=public", "--table=matrix", "--file=-", str(archive)]
    started = time.monotonic()
    target = collector.out / "matrix.copy.sql"
    errors = collector.out / "matrix-decode.stderr"
    count = total = 0
    inside = False
    with errors.open("w") as stderr, target.open("wb") as output:
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr, env=collector.env)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        buffer = b""
        try:
            while selector.get_map():
                if time.monotonic() - started > 90:
                    raise TimeoutError("offline matrix decode exceeded 90 seconds")
                for key, _ in selector.select(0.25):
                    chunk = os.read(key.fd, 1024 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()
                    assert len(buffer) < 1024 * 1024
                    for line in lines:
                        if line.startswith(b"COPY public.matrix ("):
                            output.write(line + b"\n")
                            inside = True
                        elif inside and line == b"\\.":
                            output.write(b"\\.\n")
                            inside = False
                        elif inside:
                            total += 1
                            if line.split(b"\t", 1)[0] == model_id.encode():
                                output.write(line + b"\n")
                                count += 1
            assert child.wait(timeout=3) == 0 and count > 0
        finally:
            selector.close()
            child.stdout.close()
            if child.poll() is None:
                # This is our read-only pg_restore child, never a managed worker.
                child.kill()
                child.wait(timeout=3)
            receipt = {"argv": argv, "exit": child.returncode, "timeout_seconds": 90,
                       "elapsed_seconds": time.monotonic() - started,
                       "selected_rows": count, "decoded_rows": total}
            (collector.out / "matrix-decode.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target


def prepare(collector, archive, logs_root, eid):
    assert archive.resolve().is_relative_to(ROOT), "use an existing development backup"
    assert archive.stat().st_size < 1024 ** 3
    archive_hash = sha(archive)
    toc = collector.run("archive-toc", [PG_RESTORE, "--list", archive], 30)
    tables = {}
    for table in ("experiment", "model", "inference_eval_result", "experiment_analysis_result",
                  "experiment_scheduler_worker_attempt", "experiment_scheduler_invocation",
                  "experiment_scheduler_protocol", "experiment_scheduler_lease",
                  "experiment_global_control", "schema_migrations", "experiment_scheduler_phase_policy"):
        raw = collector.run("archive-" + table, [PG_RESTORE, "--data-only", "--schema=public",
                            "--table=" + table, "--file=-", archive], 30)
        assert raw is not None
        tables[table] = copy_rows(raw, table)
    experiment, = [x for x in tables["experiment"] if x["experiment_id"] == eid]
    model, = [x for x in tables["model"] if x["model_id"] == experiment["last_model_id"]]
    inference, = [x for x in tables["inference_eval_result"] if x["model_id"] == model["model_id"]
                  and x["inference_scope"] == "final" and x["status"] == "completed"]
    attempts = {x["worker_attempt_id"]: x for x in tables["experiment_scheduler_worker_attempt"]}
    log_records, texts = [], {}
    for phase in ("train", "infer"):
        relative = Path(experiment[phase + "_log_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        source = (logs_root / relative).resolve()
        assert source.is_relative_to(logs_root.resolve())
        before = source.stat()
        assert before.st_size < 32 * 1024 ** 2
        raw = source.read_bytes()
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        destination = collector.out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        destination.chmod(0o400)
        texts[phase] = raw.decode()
        log_records.append({"source": str(source), "private": str(destination),
                            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    verify_chain(experiment, model, inference, attempts, texts["train"], texts["infer"])
    selected_attempts = [x for x in attempts.values() if x["experiment_id"] == eid]
    assert all(x["lifecycle_state"] in TERMINAL for x in selected_attempts)
    assert all(x["checkpoint_eval_id"] == "\\N" for x in selected_attempts)
    for producer in (model["producer_worker_attempt_id"], inference["producer_worker_attempt_id"]):
        attempt = attempts[producer]
        assert sha(Path(attempt["canonical_executable_path"])) == attempt["executable_sha256"]
        manifest = json.loads(Path(attempt["canonical_manifest_path"]).read_text())
        assert manifest["source_commit"] == attempt["source_commit"]
        assert manifest["sha256"] == attempt["executable_sha256"]
        assert manifest["semantic_layout"] == int(attempt["semantic_layout_version"])
        assert manifest["model_input_width"] == int(attempt["model_input_width"])
        assert attempt["lifecycle_phase"] in manifest["capabilities"]
    tables["experiment_scheduler_invocation"] = select_snapshot_invocations(
        tables["experiment_scheduler_invocation"], selected_attempts, tables["experiment_scheduler_lease"])
    tables["experiment"], tables["model"] = [experiment], [model]
    tables["inference_eval_result"] = [inference]
    tables["experiment_scheduler_worker_attempt"] = selected_attempts
    tables["experiment_analysis_result"] = [x for x in tables["experiment_analysis_result"]
                                            if x["experiment_id"] == eid and x["analysis_scope"] == "final"]
    assert len(tables["experiment_analysis_result"]) == 1
    verify_analysis_source(tables["experiment_analysis_result"][0], experiment, model, inference)
    assert tables["experiment_scheduler_protocol"][0]["cutover_state"] == "complete"
    assert tables["experiment_scheduler_lease"][0]["authority_state"] == "released"
    assert tables["experiment_global_control"][0]["desired_state"] == "running"
    assert tables["experiment_global_control"][0]["active_request_id"] == "\\N"
    # Restore complete genuine reference tables so snapshot FK/hash meaning is preserved.
    for table in ("economic_calendar_snapshot", "economic_calendar_snapshot_event",
                  "economic_calendar_snapshot_consensus", "economic_calendar_snapshot_release_actual",
                  "economic_calendar_snapshot_first_release_actual", "economic_event",
                  "economic_event_consensus", "economic_event_release_actual", "economic_event_actual_observation"):
        raw = collector.run("archive-" + table, [PG_RESTORE, "--data-only", "--schema=public",
                            "--table=" + table, "--file=-", archive], 30)
        assert raw is not None
        tables[table] = copy_rows(raw, table)
    for section in ("pre-data", "post-data"):
        raw = collector.run("schema-" + section, [PG_RESTORE, "--section=" + section,
                            "--no-owner", "--no-acl", "--file=-", archive], 30)
        assert raw is not None
        (collector.out / (section + ".sql")).write_text(raw)
    (collector.out / "data.sql").write_text("".join(copy_sql(t, rows) for t, rows in tables.items()))
    matrix = stream_matrix(collector, archive, model["model_id"])
    metadata = copy_rows(matrix.read_text(), "matrix")
    grouped = defaultdict(list)
    for row in metadata:
        assert math.isfinite(float(row["value"]))
        grouped[row["param_name"]].append(row)
    for rows in grouped.values():
        assert len(rows) == int(rows[0]["n_rows"]) * int(rows[0]["n_cols"])
        assert len({(x["row_idx"], x["col_idx"]) for x in rows}) == len(rows)
    values = lambda name: [float(x["value"]) for x in sorted(grouped[name], key=lambda x: (int(x["row_idx"]), int(x["col_idx"])))]
    text = lambda name: "".join(chr(round(x)) for x in values(name))
    assert text("train_symbol_meta") == experiment["symbol"]
    assert values("model_meta") == [1, float(experiment["model_input_width"]), 64]
    assert text("training_objective_hash_meta") == experiment["training_objective_hash"]
    assert text("training_objective_canonical_meta") == experiment["training_objective_canonical"]
    assert values("model_input_semantics_meta") == [1, float(experiment["model_input_semantic_layout_version"])]
    assert text("feature_warmup_scope_meta") == experiment["feature_warmup_scope"]
    assert text("donchian20_mode_meta") == experiment["donchian20_mode"]
    assert int(text("donchian_lookback_meta")) == int(experiment["donchian_lookback"])
    assert text("train_range_meta") == experiment["train_start"][:10] + "|" + experiment["train_end"][:10]
    config = values("train_config_meta")
    assert config[1] == float(experiment["prediction_horizon"])
    assert abs(config[2] - float(experiment["c_next_threshold"])) <= 1e-7
    assert config[3] == float(inference["window_size"]) and config[4] == float(inference["label_rule_id"])
    assert config[10:13] == [float(experiment["target_epochs"]), float(experiment["core_lr_mult"]), float(experiment["head_lr_mult"])]
    assert values("target_meta")[0] == float(inference["target_type"])
    assert int(grouped["param"][0]["n_rows"]) - 64 == int(experiment["model_input_width"])
    assert int(grouped["param"][0]["n_cols"]) == 64 * 4
    for field in ("economic_calendar_snapshot_id", "economic_calendar_snapshot_hash"):
        assert model[field] == experiment[field]
    sequences = "\n".join(line for line in toc.splitlines() if "SEQUENCE SET " in line)
    (collector.out / "sequences.list").write_text(sequences + "\n")
    raw = collector.run("sequences", [PG_RESTORE, "--data-only", "--use-list=" + str(collector.out / "sequences.list"), "--file=-", archive], 30)
    assert raw is not None
    (collector.out / "sequences.sql").write_text(raw)
    receipt = {"archive": str(archive), "archive_sha256": archive_hash, "experiment_id": eid,
               "model_id": model["model_id"], "inference_id": inference["id"],
               "producer_worker_attempt_ids": {"train": model["producer_worker_attempt_id"],
                                               "infer": inference["producer_worker_attempt_id"]},
               "historical_attempts": selected_attempts, "logs": log_records,
               "matrix_rows": len(metadata), "matrix_parameters": sorted(grouped),
               "inference": inference, "expected_analysis": tables["experiment_analysis_result"][0],
               "scientific_chain": "PASS", "database_isolation": "NOT_YET_VALIDATED"}
    (collector.out / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--logs-root", type=Path, default=Path("/Volumes/Developer SSD/ExpertAdvisor"))
    parser.add_argument("--experiment", default="746")
    parser.add_argument("--single-worker", action="store_true")
    parser.add_argument("--instrument-resources", action="store_true",
                        help="require adequate real worker/resource measurements")
    args = parser.parse_args()
    if args.instrument_resources and not args.single_worker:
        parser.error("--instrument-resources requires --single-worker")
    collector = Collector(args.output)
    (collector.out / "qualification-source.py").write_bytes(Path(__file__).read_bytes())
    receipt = prepare(collector, args.archive, args.logs_root, args.experiment)
    if args.single_worker:
        from AnalyzeHistoricalFixtureWorker import qualify
        qualify(collector, receipt, instrument=args.instrument_resources)
    print(json.dumps({"fixture": "PASS", "output": str(collector.out)}))


if __name__ == "__main__":
    main()
