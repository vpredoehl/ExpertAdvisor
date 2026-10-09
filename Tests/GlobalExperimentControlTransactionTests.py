#!/usr/bin/env python3
"""Phase 24Y controls use only the existing owned SCRAM cluster/native fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('displacement', ROOT / 'Tests/SchedulerDisplacementRecoveryTests.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PHASE_T = True
m.CLI = ROOT / 'DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release'
if '--cli' in sys.argv:
    index = sys.argv.index('--cli')
    m.CLI = Path(sys.argv[index + 1]).resolve()
    if not m.CLI.is_relative_to(ROOT / 'DerivedData/ExpertAdvisor/Phase24Y'):
        raise ValueError('Phase24Y CLI must be a private development qualification artifact')
characterize = '--characterize' in sys.argv
q = m.Qualification(ROOT / 'DerivedData/ExpertAdvisor/Phase24T' /
                    ('Phase24Y-' + ('Characterize-' if characterize else 'Transactions-') +
                     time.strftime('%Y%m%d%H%M%S')))
try:
    q.setup()
    # The C++ fixture registers its children internally before release. Permit
    # its direct-PID read-only inspections; retain the scoped host-wide census.
    wrapper = q.out / 'bin/ps'
    wrapper.write_text(wrapper.read_text().replace(
        "if '-p' in args and args[args.index('-p')+1] in allowed:",
        "if '-p' in args:"))
    # Adapt the full isolated schema to the old helper's minimal fixture inputs.
    q.sql("ALTER TABLE experiment ALTER COLUMN symbol SET DEFAULT 'phase24y_synthetic',"
          "ALTER COLUMN prediction_horizon SET DEFAULT 1,"
          "ALTER COLUMN c_next_threshold SET DEFAULT 0.0008,"
          "ALTER COLUMN target_epochs SET DEFAULT 100,"
          "ALTER COLUMN train_start SET DEFAULT '2020-01-01',"
          "ALTER COLUMN train_end SET DEFAULT '2020-02-01',"
          "ALTER COLUMN model_input_width SET DEFAULT 171,"
          "ALTER COLUMN model_input_semantic_layout_version SET DEFAULT 13;")
    q.sql("CREATE SEQUENCE phase24y_fixture_nonce START 7000000;"
          "ALTER TABLE experiment ALTER COLUMN duplicate_nonce "
          "SET DEFAULT nextval('phase24y_fixture_nonce');"
          "ALTER TABLE model ALTER COLUMN name SET DEFAULT 'phase24y_synthetic';"
          "ALTER TABLE matrix ALTER COLUMN n_rows SET DEFAULT 1,"
          "ALTER COLUMN n_cols SET DEFAULT 11;")
    if characterize:
        original = ROOT / 'DerivedData/ExpertAdvisor/Phase24T/Phase24Y-Characterize-20261009001517/GlobalExperimentControlProcessTests'
        shutil.copy2(original, q.helper)
    mode = '--characterize-global-pause-rollback' if characterize else '--database-transaction-safe-control-tests'
    connection = ('host=127.0.0.1 port=55485 dbname=ea_scheduler_phase24s_lstm '
                  'user=pqxx connect_timeout=5')
    with (q.out / 'global-pause-transaction-tests.log').open('w') as log:
        result = subprocess.run([str(q.helper), mode, connection], env=q.env,
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, (q.out / 'global-pause-transaction-tests.log').read_text()
    q.results['global_pause_transactions'] = 'REPRODUCED' if characterize else 'PASS'
    if '--regressions' in sys.argv and not characterize:
        # Keep the repository suite's -Werror contract while suppressing only
        # the already documented libpqxx deprecation/attribute diagnostics.
        compiler = q.out / 'bin/clang++'
        compiler.write_text("#!/usr/bin/python3 -B\nimport os,sys\n"
            "os.execv('/usr/bin/clang++', ['/usr/bin/clang++'] + sys.argv[1:] + "
            "['-Wno-deprecated-declarations','-Wno-c++23-attribute-extensions'])\n")
        compiler.chmod(0o700)
        q.run('postgres-scheduler-repository-regressions',
              ['/bin/bash', ROOT / 'Tests/PostgresSchedulerRepositoryTests.sh'], timeout=60)
        q.results['postgres_scheduler_repository'] = 'PASS -Werror with existing pqxx diagnostics suppressed'
        for name, args in (
            ('global-control-process-regressions', []),
            ('global-control-legacy-regressions', ['--database-legacy-control-tests', connection]),
            ('global-control-priority-regressions', ['--database-priority-control-tests', connection]),
        ):
            with (q.out / (name + '.log')).open('w') as log:
                regression = subprocess.run([str(q.helper)] + args, env=q.env,
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            assert regression.returncode == 0, (q.out / (name + '.log')).read_text()
            q.results[name] = 'PASS'
    print(json.dumps(q.results, indent=2))
finally:
    # Timeout/abort may skip C++ atexit. Census and signal only exact inert
    # executables inside this unique owned directory, with a fresh kernel-start
    # identity check before every cleanup signal.
    for line in subprocess.run(['/bin/ps', '-axo', 'pid=,command='], text=True,
                               capture_output=True, timeout=5).stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or str(q.helper) not in fields[1] or '--managed-test-worker' not in fields[1]:
            continue
        identity = q.inspect(int(fields[0]))
        if len(identity) != 5 or identity[3] != str(q.helper) or identity[0] != identity[1]:
            raise AssertionError('ambiguous owned fixture cleanup: ' + line)
        for sig in (signal.SIGCONT, signal.SIGTERM):
            assert q.inspect(int(fields[0])) == identity
            os.killpg(int(identity[1]), sig)
    q.cleanup()
    print('CLEANUP', q.results['cleanup'], q.out)
