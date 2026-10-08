#!/usr/bin/env python3
"""Bounded private-SCRAM regressions for post-abort compensation control races.

Uses only inert, positively identified Phase24S/T fixtures; never production.
--characterize records the old unsafe SIGCONT before the compensation fix.
"""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('displacement', ROOT / 'Tests/SchedulerDisplacementRecoveryTests.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PHASE_T = True
m.CLI = ROOT / 'DerivedData/ExpertAdvisor/Phase24U/build/LSTM_Release'
characterize = '--characterize' in sys.argv
output = ROOT / 'DerivedData/ExpertAdvisor/Phase24T' / ('Phase24U-Rollback-' + time.strftime('%Y%m%d%H%M%S'))
q = m.Qualification(output)


def rollback(name, actor=None):
    q.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE'] = '1'
    q.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY'] = ('capacity_rollback_pause_race' if actor else 'capacity_after_sigstop_before_db')
    log_path = q.out / (name + '.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(list(map(str, q.scheduler_args(0, 1))), env=q.env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        q.register_pid(process.pid)
        record = {'process': process, 'log': log, 'identity': []}
        q.daemons.append(record)
        try:
            q.wait(lambda: bool(q.inspect(process.pid)), 'rollback scheduler identity', seconds=3)
            record['identity'] = q.inspect(process.pid)
            if actor:
                q.wait(lambda: 'SCHEDULER_CAPACITY_TEST_ROLLBACK_UNLOCKED' in log_path.read_text(),
                       'transaction abort handoff', seconds=8)
                actor()
            assert process.wait(timeout=12) != 0
        finally:
            if process.poll() is None:
                q.stop_daemon(record)
            q.daemons.remove(record)
            q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_ENABLE', None)
            q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY', None)
    text = log_path.read_text()
    assert 'injected_capacity_db_failure_after_sigstop' in text
    q.snapshot(name)
    return text


def pair(base):
    q.launch(base, priority='normal', detached=True)
    q.launch(base + 1, layout=13, priority='low', status='pending', origin='preemption', detached=True)
    q.signal_worker(base + 1, signal.SIGCONT)


try:
    q.setup()
    if not characterize:
        pair(997100)
        text = rollback('ordinary-rollback')
        assert 'restored=1' in text
        q.expect(997101, 'running', False)
        q.cycle('ordinary-rollback-retry')
        q.expect(997101, 'pending', True)
        q.results['ordinary_compensation'] = 'PASS exact original worker restored; retry enforces cap'
        q.retire()

    pair(997200)
    identities = {eid: w['identity'] for eid, w in q.workers.items()}
    def pause():
        q.run('supported-pause-during-abort', [m.CLI, '--pause-all-experiments', '--yes'], timeout=10)
        assert q.sql("SELECT desired_state FROM experiment_global_control;") == 'paused'
        q.expect(997201, 'paused', True)
    text = rollback('operator-pause-after-abort', pause)
    q.expect(997201, 'paused', not characterize)
    assert all(q.inspect(q.workers[eid]['pid']) == identity for eid, identity in identities.items())
    assert q.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=997201;') == '1'
    if characterize:
        assert 'restored=1' in text
        q.results['counterexample'] = 'REPRODUCED: after supported global pause commits, old compensation resumes paused exact worker'
    else:
        assert 'restored=0' in text
        q.results['operator_pause_race'] = 'PASS fresh global control withholds SIGCONT; PID and attempt retained'
    q.run('supported-private-resume', [m.CLI, '--resume-all-experiments', '--yes'], timeout=10)
    q.retire()

    if not characterize:
        pair(997300)
        def mismatched_identity():
            q.sql("UPDATE experiment_scheduler_worker_attempt SET worker_process_start_identity='mismatched-start' WHERE worker_attempt_id=1997301;")
        text = rollback('changed-start-after-abort', mismatched_identity)
        assert 'restored=0' in text
        assert q.state(q.workers[997301]['pid']).startswith('T')
        q.results['changed_identity'] = 'PASS mismatched persisted start refuses compensation without unsafe SIGCONT'
        q.retire()

        pair(997400)
        def lost_fence():
            q.sql('UPDATE experiment_scheduler_lease SET fencing_token=fencing_token+1;')
        text = rollback('lost-fence-after-abort', lost_fence)
        assert 'restored=0' in text
        assert q.state(q.workers[997401]['pid']).startswith('T')
        q.results['lost_fence'] = 'PASS stale scheduler fence withholds compensation'
    print(json.dumps(q.results, indent=2))
finally:
    q.cleanup()
    print('CLEANUP', q.results['cleanup'], q.out)
