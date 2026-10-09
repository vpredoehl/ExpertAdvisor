#!/usr/bin/env python3
"""Execute the production diagnostic blocks on CPU matrices with counted reads/copies.

No database, worker executable, or Metal command is used. Source extraction keeps
the test attached to the worker lambda and the exact preclip snapshot policy.
"""
from pathlib import Path
import subprocess
import tempfile
import shlex


ROOT = Path(__file__).resolve().parents[1]


def between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


worker = (ROOT / "Sources/TrainingWorkerApplication.cpp").read_text()
lstm = (ROOT / "LSTM/LSTM.cpp").read_text()
worker_block = between(worker, "                        auto l2 =", "                    } );")
worker_block = worker_block.replace("MetaNN::Evaluate(m)", "CountEvaluate(m)")
snapshot_block = between(lstm, "        static size_t s_phase3ClipFullDiagCount =",
                         "        // For UpNeutralDownReturn")
snapshot_block = snapshot_block.replace("NNUtils::DeepCopyMatrix(", "CountDeepCopy(")
assert lstm.count("s_phase3ClipFullDiagCount++") == 1
assert "std::move(z_f_logits)" in lstm
assert "DeepCopyMatrix(z_f_" not in lstm
assert lstm.index("phase3ScaleMatrixInPlace(d_headDirB_f, invN)") < lstm.index(
    "static size_t s_phase3ClipFullDiagCount") < lstm.index(
    'ClipMatrixInPlace(d_param_f, LSTM_GRAD_CLIP_THRESHOLD, "d_param")')

with tempfile.TemporaryDirectory(prefix="ea_phase25a_cpu_") as directory:
    test_dir = Path(directory)
    (test_dir / "WorkerDiagnosticBlock.inc").write_text(worker_block)
    (test_dir / "GradientSnapshotBlock.inc").write_text(snapshot_block)
    metann = (ROOT / "MetaNN/MetaNN").resolve()
    includes = ["-I" + str(path) for path in [metann, *sorted({
        path.parent for path in (metann / "MetaNN").rglob("*.h")})]]
    pqxx_flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "libpqxx"], text=True))
    pqxx_libraries = shlex.split(subprocess.check_output(
        ["pkg-config", "--libs", "libpqxx"], text=True))
    subprocess.run([
        "xcrun", "--sdk", "macosx", "clang++", "-std=c++20",
        "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
        "-Wno-ignored-qualifiers", "-Wno-unused-but-set-variable",
        "-I" + str(ROOT / "Headers"),
        *includes, *pqxx_flags, "-I" + directory,
        str(ROOT / "Tests/LSTMTrainingDiagnosticOverheadTests.cpp"),
        str(ROOT / "Sources/LstmRuntimeLogging.cpp"),
        *pqxx_libraries,
        "-o", str(test_dir / "tests"),
    ], check=True)
    subprocess.run([str(test_dir / "tests")], check=True)
