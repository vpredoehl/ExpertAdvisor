#!/usr/bin/env python3
"""Bounded native scheduler regression; owns its SCRAM PostgreSQL cluster.

No ambient PG settings, production schema dumps, real model execution or
publication. Registry artifacts are copies of the inert managed-process fixture.
Protocol completion is test fixture setup, not a protocol-cutover qualification.
Usage: python3 Tests/SchedulerDisplacementRecoveryTests.py [evidence-directory]
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PG = Path('/opt/homebrew/opt/postgresql@17/bin')
CLI = ROOT / 'DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release'
PHASE_T = '--phase24t' in sys.argv
if PHASE_T:
    sys.argv.remove('--phase24t')
    CLI = ROOT / 'DerivedData/ExpertAdvisor/Phase24T/build/LSTM_Release'
ACTIVE = "('reserved','spawned','running','observed','identity_ambiguous')"


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


class Qualification:
    def __init__(self, output):
        self.out = output.resolve()
        area = ROOT / 'DerivedData/ExpertAdvisor' / ('Phase24T' if PHASE_T else 'Phase24S')
        if not self.out.is_relative_to(area):
            raise ValueError('evidence must be inside the phase Rollover DerivedData area')
        self.out.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.data = self.out / 'pgdata'
        self.helper = self.out / 'GlobalExperimentControlProcessTests'
        self.workers = {}
        self.daemons = []
        self.started = False
        self.results = {}
        self.sequence = 0
        # initdb selects a free TCP port only at start; a collision fails closed.
        # libpq never falls back to 5432 or a Unix socket.
        self.port = '55485'
        self.env = {
            'PATH': str(PG) + ':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
            'LC_ALL': 'C', 'LANG': 'C',
            'PGHOST': '127.0.0.1', 'PGPORT': self.port,
            'PGUSER': 'phase24s_admin', 'PGDATABASE': 'postgres',
            'PGPASSFILE': str(self.out / 'pgpass'), 'PGCONNECT_TIMEOUT': '5',
            'LSTM_DB_HOST': '127.0.0.1', 'LSTM_DB_NAME': 'ea_scheduler_phase24s_lstm',
            'FOREX_DB_HOST': '127.0.0.1', 'FOREX_DB_NAME': 'ea_phase24s_forex',
        }
        (self.out / 'environment.json').write_text(json.dumps(self.env, indent=2))
        self.env['TMPDIR'] = str(self.out / 'tmp')
        (self.out / 'tmp').mkdir()
        # Native recovery includes a host-wide legacy scheduler search. Scope
        # that census to registered fixtures; never inspect production PIDs.
        self.allowed = self.out / 'owned-pids'
        self.allowed.write_text('')
        self.env['EA_TEST_OWNED_PIDS'] = str(self.allowed)
        tools = self.out / 'bin'
        tools.mkdir()
        wrapper = tools / 'ps'
        wrapper.write_text('''#!/usr/bin/python3 -B
import os,subprocess,sys
args=sys.argv[1:]
allowed=set(open(os.environ['EA_TEST_OWNED_PIDS']).read().split())
if args and args[0]=='-axo':
 for pid in sorted(allowed):
  r=subprocess.run(['/bin/ps','-p',pid,'-o',args[1]],capture_output=True,text=True,timeout=3)
  sys.stdout.write(r.stdout)
 sys.exit(0)
if '-p' in args and args[args.index('-p')+1] in allowed:
 sys.exit(subprocess.run(['/bin/ps']+args,timeout=3).returncode)
sys.exit(1)
''')
        wrapper.chmod(0o700)
        self.env['PATH'] = str(tools) + ':' + self.env['PATH']
        (self.out / 'environment.json').write_text(json.dumps(self.env, indent=2))

    def register_pid(self, pid):
        with self.allowed.open('a') as f:
            f.write(str(pid) + '\n')

    def run(self, name, argv, sql=None, timeout=30, expected=0):
        self.sequence += 1
        result = subprocess.run(list(map(str, argv)), input=sql, env=self.env,
                                cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        (self.out / f'{self.sequence:03d}-{name}.log').write_text(result.stdout + result.stderr)
        with (self.out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'name': name, 'argv': list(map(str, argv)),
                                'exit': result.returncode, 'timeout': timeout}) + '\n')
        if expected is not None and result.returncode != expected:
            raise AssertionError(f'{name}: exit {result.returncode}\n' +
                                 (result.stdout + result.stderr)[-4000:])
        return result

    def sql(self, sql, db='ea_scheduler_phase24s_lstm'):
        assert db in ('postgres', 'ea_scheduler_phase24s_lstm', 'ea_phase24s_forex')
        return self.run('sql', [PG / 'psql', '-X', '-Atq', '-v', 'ON_ERROR_STOP=1',
                               '-d', db], sql=sql).stdout.strip()

    def setup(self):
        password = secrets.token_hex(32)
        runtime_password = secrets.token_hex(32)
        pw = self.out / 'init-password'
        pw.write_text(password)
        pw.chmod(0o600)
        passfile = self.out / 'pgpass'
        passfile.write_text(f'127.0.0.1:{self.port}:*:phase24s_admin:{password}\n'
                            f'127.0.0.1:{self.port}:*:pqxx:{runtime_password}\n')
        passfile.chmod(0o600)
        self.run('initdb', [PG / 'initdb', '-D', self.data, '-U', 'phase24s_admin',
                           '--pwfile=' + str(pw), '--auth=scram-sha-256',
                           '--encoding=UTF8', '--locale=C'])
        with (self.data / 'postgresql.conf').open('a') as f:
            f.write("\nlisten_addresses='127.0.0.1'\nport=55485\n"
                    "unix_socket_directories=''\nmax_connections=16\nshared_buffers='32MB'\n")
        self.run('start-private-postgres', [PG / 'pg_ctl', '-D', self.data,
                                           '-l', self.out / 'postgres.log', '-w', 'start'])
        self.started = True
        identity = self.sql("SELECT current_database(),current_user,inet_server_addr(),"
                            "inet_server_port(),current_setting('data_directory');"
                            "SELECT system_identifier FROM pg_control_system();", 'postgres')
        assert identity.startswith('postgres|phase24s_admin|127.0.0.1|55485|')
        assert str(self.data) in identity
        self.results['isolation'] = identity
        # Private fixture superuser is disposable, with a new random password.
        # This is not a production runtime grant or a copied production role.
        self.sql('CREATE ROLE pqxx LOGIN SUPERUSER PASSWORD ' + quote(runtime_password) + ';'
                 'CREATE DATABASE ea_scheduler_phase24s_lstm; CREATE DATABASE ea_phase24s_forex;', 'postgres')
        schema = (ROOT / 'Database/LSTM_schema.sql').read_text()
        marker = '-- Durable declared controlled-experiment family foundation.'
        assert marker in schema
        self.run('checked-in-schema', [PG / 'psql', '-X', '-q', '-v', 'ON_ERROR_STOP=1',
                                      '-d', 'ea_scheduler_phase24s_lstm'],
                 sql=schema.replace(marker, 'SET search_path TO public;\n' + marker, 1))
        for prefix in ('046', '051', '052', '071', '078', '086', '093', '099'):
            migration, = (ROOT / 'Database/migrations').glob(prefix + '_*.sql')
            self.run('migration-' + prefix, [PG / 'psql', '-X', '-q', '-v',
                                            'ON_ERROR_STOP=1', '-d', 'ea_scheduler_phase24s_lstm',
                                            '-f', migration])
        compile_flags = self.run('pqxx-cflags', ['/opt/homebrew/bin/pkg-config', '--cflags', 'libpqxx']).stdout.split()
        link_flags = self.run('pqxx-libs', ['/opt/homebrew/bin/pkg-config', '--libs', 'libpqxx']).stdout.split()
        sources = ['Tests/GlobalExperimentControlProcessTests.cpp',
                   'Sources/GlobalExperimentControl.cpp', 'Sources/CheckpointPolicy.cpp',
                   'Sources/SchedulerCore/CheckpointEvaluationService.cpp',
                   'Sources/SchedulerCore/PostgresSchedulerRepository.cpp',
                   'Sources/SchedulerCore/SchedulerRepository.cpp',
                   'Sources/SchedulerCore/SchedulerPolicy.cpp',
                   'Sources/SchedulerCore/SchedulerOperationalObservation.cpp']
        self.run('apple-clang-process-fixture', ['/usr/bin/clang++', '-std=c++20', '-O0',
                 '-Wno-deprecated-declarations', '-Wno-c++23-attribute-extensions',
                 '-I' + str(ROOT / 'Headers'), '-I' + str(ROOT / 'Sources')] + compile_flags +
                 [ROOT / s for s in sources] + link_flags + ['-o', self.helper], timeout=60)
        self.registry()
        pending = self.run('pending-protocol-rejected', self.scheduler_args(0, 0), expected=3)
        assert 'SCHEDULER_PROTOCOL_BARRIER_REJECTED' in pending.stdout + pending.stderr
        assert self.sql('SELECT count(*) FROM experiment_scheduler_invocation;') == '0'
        self.sql("UPDATE experiment_scheduler_protocol SET required_generation=52,"
                 "cutover_state='complete',cutover_completed_at=clock_timestamp(),"
                 "cutover_completed_by='isolated-regression-fixture',"
                 "cutover_executable_path='/test/LSTM_Release',"
                 "cutover_process_evidence='private-cluster-test-fixture',failure_diagnostic=NULL;"
                 "INSERT INTO experiment_global_control(singleton,desired_state) VALUES(true,'running') "
                 "ON CONFLICT(singleton) DO UPDATE SET desired_state='running';")
        self.results['pending_protocol_admission'] = 'PASS (zero invocations)'
        self.run('ownership-postgres-contracts', ['/bin/bash', ROOT /
                 'Tests/SchedulerOwnershipIntegrationTests.sh'], timeout=45)
        self.results['ownership_postgres_contracts'] = 'PASS on same independent cluster; migration test database dropped'

    def registry(self):
        root = self.out / 'registry'
        payload = b'isolated inert process fixture; never loaded by Metal\n'
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {'schema_version': 1, 'storage': 'immutable', 'resources': [
            {'built_identity': b, 'runtime_name': n, 'sha256': digest}
            for b, n in [('default.metallib', 'default.metallib'),
                         ('MetaNN_metal.metallib', 'MetaNN.metallib')]]}
        encoded = json.dumps(manifest).encode()
        runtime = hashlib.sha256(encoded).hexdigest()
        directory = root / 'runtime' / runtime
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_bytes(encoded)
        for resource in manifest['resources']:
            (directory / resource['runtime_name']).write_bytes(payload)
        binary_hash = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        self.artifacts = {}
        entries = []
        for layout, width, role in [(9, 103, 'infer'), (13, 171, 'infer'), (13, 171, 'train')]:
            commit = str(layout % 10) * 40
            relative = Path(f'layout{layout}') / role / commit / binary_hash
            directory = root / relative
            directory.mkdir(parents=True)
            executable = 'lstm-' + role + '-worker'
            shutil.copy2(self.helper, directory / executable)
            caps = ['infer'] if role == 'infer' else ['train', 'train_feature_ablation_v1']
            artifact = {'schema_version': 2, 'semantic_layout': layout, 'storage': 'immutable',
                        'model_input_width': width, 'source_commit': commit, 'sha256': binary_hash,
                        'executable_identity': executable, 'worker_role': role, 'capabilities': caps}
            (directory / 'manifest.json').write_text(json.dumps(artifact))
            for resource in manifest['resources']:
                (directory / resource['runtime_name']).symlink_to(
                    os.path.relpath(root / 'runtime' / runtime / resource['runtime_name'], directory))
            entries.append({'semantic_layout': layout, 'model_input_width': width,
                            'worker_role': role, 'artifact_manifest_schema_version': 2,
                            'worker_rule': 'current' if layout == 13 else 'historical',
                            'source_commit': commit, 'sha256': binary_hash,
                            'executable': str(relative / executable),
                            'manifest': str(relative / 'manifest.json'),
                            'runtime_identity': runtime, 'capabilities': caps, 'selection_priority': 0})
            self.artifacts[layout, role] = directory / executable
        self.registry_path = root / 'registry.json'
        self.registry_path.write_text(json.dumps({'schema_version': 5, 'current_layout': 13,
            'runtimes': [{'identity': runtime, 'directory': 'runtime/' + runtime,
                          'manifest': 'runtime/' + runtime + '/manifest.json'}], 'workers': entries}))

    def inspect(self, pid):
        result = subprocess.run([str(self.helper), '--inspect-managed-test-process=' + str(pid)],
                                env=self.env, text=True, capture_output=True, timeout=5)
        return result.stdout.strip().split('|') if result.returncode == 0 else []

    def state(self, pid):
        return subprocess.run(['/bin/ps', '-o', 'state=', '-p', str(pid)],
                              text=True, capture_output=True, timeout=5).stdout.strip()

    def wait(self, predicate, label, seconds=4):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError('bounded wait failed: ' + label)

    def signal_worker(self, eid, sig):
        worker = self.workers[eid]
        if worker['process'] is not None and worker['process'].poll() is not None:
            return
        observed = self.inspect(worker['pid'])
        if not observed or self.state(worker['pid']).startswith('Z'):
            return
        assert observed == worker['identity'], 'refuse signal: exact fixture identity changed'
        assert Path(observed[3]).is_relative_to(self.out)
        os.killpg(worker['pid'], sig)
        with (self.out / 'signals.jsonl').open('a') as f:
            f.write(json.dumps({'fixture_experiment': eid, 'identity': observed, 'signal': int(sig)}) + '\n')

    def launch(self, eid, phase='infer', layout=9, priority='normal', status='running',
               origin='none', detached=False):
        attempt = eid + 1000000
        ready = (self.out / f'{eid}.ready').open('wb')
        args = [str(self.artifacts[layout, phase]), '--managed-test-worker', '--self-session',
                '--' + phase, '--scheduler-experiment-id=' + str(eid),
                '--scheduler-worker-attempt-id=' + str(attempt), '--ready-fd=' + str(ready.fileno())]
        if PHASE_T:
            args.append('--managed-test-max-seconds=180')
        if detached:
            # Dedicated launcher exits; only this new fixture is reparented to PID 1.
            code = ('import subprocess,sys; p=subprocess.Popen(sys.argv[2:],'
                    'pass_fds=(int(sys.argv[1]),),stdout=subprocess.DEVNULL,'
                    'stderr=subprocess.DEVNULL); print(p.pid)')
            launcher = subprocess.run(['/usr/bin/python3', '-c', code, str(ready.fileno())] + args,
                                      env=self.env, pass_fds=(ready.fileno(),), capture_output=True,
                                      text=True, timeout=5)
            assert launcher.returncode == 0, launcher.stderr
            pid = int(launcher.stdout)
            process = None
        else:
            process = subprocess.Popen(args, env=self.env, pass_fds=(ready.fileno(),),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = process.pid
        ready.close()
        self.register_pid(pid)
        self.wait(lambda: (self.out / f'{eid}.ready').stat().st_size > 0, 'worker ready')
        identity = self.inspect(pid)
        assert len(identity) == 5 and identity[0] == identity[1] == str(pid), identity
        self.workers[eid] = {'pid': pid, 'identity': identity, 'process': process}
        stopped = status in ('pending', 'paused')
        if stopped:
            self.signal_worker(eid, signal.SIGSTOP)
            self.wait(lambda: self.state(pid).startswith('T'), 'SIGSTOP')
        width = 103 if layout == 9 else 171
        self.sql("SET expertadvisor.scheduler_protocol_generation='52';"
                 "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,"
                 "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,"
                 "infer_start,infer_end,status,phase,current_operation,duplicate_nonce,scheduler_priority,"
                 "resume_requested,scheduler_resume_origin,worker_pid,worker_process_group_id,"
                 "worker_process_start_identity,worker_executable,worker_command_line,worker_control_state,"
                 "worker_started_at,model_input_width,model_input_semantic_layout_version) VALUES(" +
                 f"{eid},'synthetic{eid}',1,0.0008,1,1,20,20,'2020-01-01','2020-02-01',"
                 f"'2020-02-01','2020-03-01',{quote(status)},{quote(phase)},{quote(phase)},{eid},"
                 f"{quote(priority)},{str(status == 'pending').lower()},{quote(origin)},{pid},{pid},"
                 f"{quote(identity[2])},{quote(identity[3])},{quote(identity[4])},"
                 f"{quote('paused' if stopped else 'running')},clock_timestamp(),{width},{layout});"
                 "INSERT INTO experiment_scheduler_worker_attempt(worker_attempt_id,launch_attempt_identity,"
                 "experiment_id,worker_kind,lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,"
                 "worker_pid,worker_process_group_id,worker_process_start_identity,canonical_executable_path,"
                 "command_line,command_identity,spawned_at,registered_at) VALUES(" +
                 f"{attempt},'phase24s-{attempt}',{eid},'experiment',{quote(phase)},{quote(phase)},"
                 f"'prior_scheduler_observed',{quote('stopped' if stopped else 'running')},{pid},{pid},"
                 f"{quote(identity[2])},{quote(identity[3])},{quote(identity[4])},"
                 f"'experiment:{eid}:{phase}',clock_timestamp(),clock_timestamp());"
                 f"UPDATE experiment SET active_scheduler_worker_attempt_id={attempt} WHERE experiment_id={eid};")
        if phase == 'infer':
            self.sql(f"WITH m AS (INSERT INTO model(experiment_id,name,comment) VALUES({eid},"
                     f"'fixture-{eid}','synthetic header only') RETURNING model_id) "
                     f"UPDATE experiment SET last_model_id=m.model_id FROM m WHERE experiment_id={eid};")
        if detached:
            ppid = subprocess.run(['/bin/ps', '-o', 'ppid=', '-p', str(pid)],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
            assert ppid == '1', ppid

    def scheduler_args(self, train, infer, once=True):
        args = [CLI, '--schedule-experiments', '--max-train-procs=' + str(train),
                '--max-infer-procs=' + str(infer), '--max-analyze-procs=0',
                '--scheduler-log-dir=' + str(self.out / 'logs')]
        if once:
            args.append('--scheduler-once')
        if hasattr(self, 'registry_path'):
            args += ['--semantic-worker-registry=' + str(self.registry_path),
                     '--analyze-worker=' + str(self.helper)]
        return args

    def cycle(self, name, train=0, infer=1, extra=()):
        r = self.run(name, self.scheduler_args(train, infer) + list(extra))
        self.snapshot(name)
        return r.stdout + r.stderr

    def snapshot(self, name):
        rows = self.sql("SELECT e.experiment_id,e.status,e.phase,e.scheduler_priority,e.resume_requested,"
                        "e.scheduler_resume_origin,e.worker_pid,e.active_scheduler_worker_attempt_id,"
                        "a.lifecycle_state,a.reconciliation_result,a.scheduler_invocation_id,"
                        "a.observed_by_scheduler_invocation_id FROM experiment e LEFT JOIN "
                        "experiment_scheduler_worker_attempt a ON a.worker_attempt_id="
                        "e.active_scheduler_worker_attempt_id ORDER BY e.experiment_id;")
        states = {str(e): {'identity': self.inspect(w['pid']), 'state': self.state(w['pid'])}
                  for e, w in self.workers.items()}
        (self.out / (name + '.json')).write_text(json.dumps({'rows': rows, 'processes': states}, indent=2))

    def expect(self, eid, status, os_stopped=None):
        assert self.sql(f'SELECT status FROM experiment WHERE experiment_id={eid};') == status
        if os_stopped is not None:
            assert self.state(self.workers[eid]['pid']).startswith('T') == os_stopped

    def capacity(self, phase='infer'):
        return int(self.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt '
                            f'WHERE capacity_class={quote(phase)} AND lifecycle_state IN {ACTIVE};'))

    def start_daemon(self, name):
        log = (self.out / (name + '.log')).open('w')
        args = self.scheduler_args(0, 1, once=False) + ['--scheduler-poll-seconds=1']
        process = subprocess.Popen(list(map(str, args)), env=self.env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'process': process, 'log': log, 'identity': []}
        self.daemons.append(record)
        self.register_pid(process.pid)
        self.wait(lambda: bool(self.inspect(process.pid)), 'daemon identity')
        record['identity'] = self.inspect(process.pid)
        self.wait(lambda: self.sql("SELECT count(*) FROM experiment_scheduler_invocation "
                  f"WHERE process_pid={process.pid} AND status='owner';") == '1', 'daemon authority')
        self.wait(lambda: 'SCHEDULER_START,' in (self.out / (name + '.log')).read_text(),
                  'daemon startup completed', seconds=5)
        (self.out / (name + '-identity.json')).write_text(json.dumps(record['identity']))
        with (self.out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'name': name, 'argv': list(map(str, args)),
                                'pid': process.pid, 'type': 'owned-daemon'}) + '\n')
        return record

    def stop_daemon(self, record, sig=signal.SIGTERM):
        process = record['process']
        if process.poll() is None:
            assert self.inspect(process.pid) == record['identity']
            assert record['identity'][3] == str(CLI.resolve())
            os.kill(process.pid, sig)  # Never signal its worker groups.
            with (self.out / 'signals.jsonl').open('a') as f:
                f.write(json.dumps({'private_scheduler': record['identity'], 'signal': int(sig)}) + '\n')
            process.wait(timeout=5)
        record['log'].close()

    def retire(self):
        for eid in self.workers:
            self.signal_worker(eid, signal.SIGCONT)
            self.signal_worker(eid, signal.SIGTERM)
        for eid,worker in self.workers.items():
            if worker['process'] is not None:
                worker['process'].wait(timeout=4)
            else:
                self.wait(lambda: not self.inspect(worker['pid']) or
                          self.state(worker['pid']).startswith('Z'), 'detached exit')
            with (self.out/'cleanup-workers.jsonl').open('a') as receipt:
                receipt.write(json.dumps({'experiment':eid,'identity':worker['identity'],
                    'result':'direct child reaped' if worker['process'] is not None else 'detached fixture missing or exited zombie',
                    'timeout_seconds':4})+'\n')
        self.workers.clear()
        self.sql("SET expertadvisor.scheduler_protocol_generation='52';"
                 "UPDATE experiment SET status='failed',resume_requested=false,scheduler_resume_origin='none',"
                 "worker_pid=NULL,worker_process_group_id=NULL,worker_process_start_identity=NULL,"
                 "worker_executable=NULL,worker_command_line=NULL,active_scheduler_worker_attempt_id=NULL,"
                 "worker_control_state='running' WHERE active_scheduler_worker_attempt_id IS NOT NULL;"
                 "UPDATE experiment_scheduler_worker_attempt SET lifecycle_state='abandoned',"
                 "completed_at=clock_timestamp() WHERE lifecycle_state IN " + ACTIVE[:-1] + ",'stopped');")

    def tests(self):
        for phase in ('infer', 'train'):
            base = 995000 if phase == 'infer' else 995100
            layout = 9 if phase == 'infer' else 13
            self.launch(base, phase, layout, 'low', detached=True)
            self.launch(base + 1, phase, 13, 'high', 'pending', 'operator')
            self.launch(base + 2, phase, layout, 'normal', 'pending', 'operator')
            self.launch(base + 3, phase, 13, 'high', 'paused')
            paused = self.sql(f'SELECT row_to_json(e) FROM experiment e WHERE experiment_id={base+3};')
            capacities = {'train': int(phase == 'train'), 'infer': int(phase == 'infer')}
            output = self.cycle(phase + '-displacement', **capacities)
            assert f'victim_experiment_id={base}' in output
            if phase == 'infer':
                assert 'model_input_semantic_layout_version=13,worker_semantic_layout_version=13' in output
                assert 'model_input_semantic_layout_version=9,worker_semantic_layout_version=9' in output
            self.expect(base, 'pending', True)
            self.expect(base + 1, 'running', False)
            self.expect(base + 2, 'pending', True)
            assert self.capacity(phase) == 1
            self.cycle(phase + '-restart-reconcile', extra=['--recover-orphans-only'], **capacities)
            self.expect(base, 'pending', True)
            self.signal_worker(base + 1, signal.SIGTERM)
            self.workers[base + 1]['process'].wait(timeout=4)
            self.cycle(phase + '-normal-before-low', **capacities)
            assert self.sql(f'SELECT status FROM experiment WHERE experiment_id={base+1};') == 'failed'
            self.expect(base + 2, 'running', False)
            self.expect(base, 'pending', True)
            self.signal_worker(base + 2, signal.SIGTERM)
            self.workers[base + 2]['process'].wait(timeout=4)
            self.cycle(phase + '-eventual-low-resume', **capacities)
            self.expect(base, 'running', False)
            assert self.capacity(phase) == 1
            assert self.sql(f'SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id BETWEEN {base} AND {base+3};') == '4'
            assert paused == self.sql(f'SELECT row_to_json(e) FROM experiment e WHERE experiment_id={base+3};')
            self.results[phase + '_delayed_same_attempt_recovery'] = 'PASS; detached PPID=1, normal resumes before low, paused row identical'
            self.retire()

        # An out-of-band stop remains capacity-consuming: conservative and
        # visible, rather than silently allowing a second active process.
        self.launch(995200)
        self.signal_worker(995200, signal.SIGSTOP)
        self.cycle('external-stop')
        assert self.capacity() == 1
        self.expect(995200, 'running', True)
        self.signal_worker(995200, signal.SIGCONT)
        self.cycle('external-resume')
        self.expect(995200, 'running', False)
        self.results['external_stop'] = 'PASS conservative accounting; automatic SIGCONT not implemented for unrecorded stop'
        self.retire()

        # PID reuse is deterministically represented by wrong start identity;
        # do not try to force the host PID allocator or signal an unrelated PID.
        self.launch(995210, priority='low')
        self.launch(995211, layout=13, priority='high', status='pending', origin='operator')
        self.sql("UPDATE experiment_scheduler_worker_attempt SET worker_process_start_identity='stale-reused-pid' WHERE experiment_id=995210;")
        self.cycle('pid-reuse-fail-closed')
        assert self.sql("SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE experiment_id=995210;") == 'identity_ambiguous'
        self.expect(995210, 'running', False)
        self.expect(995211, 'pending', True)
        assert self.capacity() == 1
        self.results['stale_pid_identity'] = 'PASS; ambiguous identity consumes slot, no signal or duplicate launch'
        self.retire()

        self.launch(995215, priority='low')
        self.launch(995216, layout=13, priority='high', status='pending', origin='operator')
        self.sql("UPDATE experiment SET worker_process_start_identity='conflicting-lifecycle-identity' WHERE experiment_id=995215;")
        self.cycle('conflicting-lifecycle-identity')
        self.expect(995215, 'running', False)
        self.expect(995216, 'pending', True)
        self.results['conflicting_lifecycle_identity'] = 'PASS; exact-attempt guard refuses preemption'
        self.retire()

        # Result fixture models an already-committed final inference. Restart
        # must recover it once, without a replacement attempt or new result.
        self.launch(995220)
        self.sql("INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,threshold_logret,"
                 "window_size,label_rule_id,target_type,from_date,to_date,completed_epochs,accuracy,accept_model,"
                 "status,completed_at,inference_scope) SELECT last_model_id,symbol,prediction_horizon,"
                 "c_next_threshold,30,1,0,infer_start::date::text,infer_end::date::text,target_epochs,0.5,false,"
                 "'completed',clock_timestamp(),'final' FROM experiment WHERE experiment_id=995220;")
        self.signal_worker(995220, signal.SIGTERM)
        self.workers[995220]['process'].wait(timeout=4)
        for i in range(3):
            self.cycle(f'result-recovery-{i}')
        self.expect(995220, 'pending')
        assert self.sql('SELECT phase FROM experiment WHERE experiment_id=995220;') == 'analyze'
        assert self.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=995220;') == '1'
        assert self.sql('SELECT count(*) FROM inference_eval_result;') == '1'
        assert self.sql("SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE experiment_id=995220;") == 'completed'
        self.results['completed_result_recovery'] = 'PASS; three invocations, one attempt and one result'
        self.retire()

        self.launch(995250, phase='train', layout=13, priority='low')
        self.launch(995251, phase='train', layout=13, priority='high', status='pending', origin='operator')
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE'] = '1'
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY'] = 'preemption_after_sigstop_before_db'
        try:
            result = self.run('preemption-rollback-compensation', self.scheduler_args(1, 0), expected=None)
        finally:
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE']
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY']
        assert result.returncode != 0
        assert 'SCHEDULER_PREEMPTION_ROLLBACK_COMPENSATION' in result.stdout + result.stderr
        assert 'restored=1' in result.stdout + result.stderr
        self.expect(995250, 'running', False)
        self.expect(995251, 'pending', True)
        assert self.capacity('train') == 1
        self.results['preemption_rollback'] = 'PASS; private DB failure after SIGSTOP restores exact original worker'
        self.cycle('preempted-train-missing-setup', train=1, infer=0)
        self.expect(995250, 'pending', True)
        self.signal_worker(995250, signal.SIGKILL)
        self.workers[995250]['process'].wait(timeout=4)
        self.cycle('missing-preempted-train-no-checkpoint', train=0, infer=0, extra=['--recover-orphans-only'])
        self.expect(995250, 'failed')
        assert self.sql('SELECT error_message FROM experiment WHERE experiment_id=995250;') == 'preempted_worker_missing_no_valid_checkpoint'
        assert self.sql('SELECT active_scheduler_worker_attempt_id IS NULL FROM experiment WHERE experiment_id=995250;') == 't'
        self.results['missing_train_no_checkpoint'] = 'PASS explicit terminal failure; no fresh training restart'
        self.retire()

        # An external SIGCONT can violate a configured cap independently of
        # scheduler admission. Record actual behavior; do not assume repair.
        self.launch(995230)
        self.launch(995231, layout=13, status='pending', origin='preemption')
        self.signal_worker(995231, signal.SIGCONT)
        self.cycle('external-over-cap')
        self.results['external_over_cap'] = {'active_attempts': self.capacity(),
            'os_active': sum(not self.state(w['pid']).startswith('T') for w in self.workers.values()),
            'new_attempts': int(self.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id IN (995230,995231);')) - 2}
        assert self.results['external_over_cap'] == {
            'active_attempts': 1 if PHASE_T else 2, 'os_active': 1 if PHASE_T else 2, 'new_attempts': 0}
        self.retire()

        if PHASE_T:
            self.capacity_tests()

        # Real owner crash; advance only the private fixture's lease clock to
        # avoid a long wall-clock expiry wait. The live worker stays detached.
        self.launch(995240, detached=True)
        if PHASE_T:
            self.launch(995241,layout=13,priority='low',status='pending',origin='preemption',detached=True)
        worker_identity = self.workers[995240]['identity']
        daemon = self.start_daemon('owner-before-crash')
        owner, fence = self.sql('SELECT owner_scheduler_invocation_id,fencing_token '
                                'FROM experiment_scheduler_lease;').split('|')
        self.sql('UPDATE experiment_scheduler_worker_attempt SET ownership_origin=\'scheduler_launch\','
                 f'scheduler_invocation_id={quote(owner)},scheduler_fencing_token={fence} '
                 "WHERE experiment_id IN (995240,995241);")
        rival = self.run('live-owner-rejected', self.scheduler_args(0, 1), expected=3)
        assert 'SCHEDULER_OWNERSHIP_REJECTED' in rival.stdout + rival.stderr
        self.stop_daemon(daemon, signal.SIGKILL)
        if PHASE_T:self.signal_worker(995241,signal.SIGCONT)
        assert self.inspect(self.workers[995240]['pid']) == worker_identity
        fresh = self.run('dead-owner-fresh-lease-rejected', self.scheduler_args(0, 1), expected=3)
        assert 'reason=owner_lease_valid' in fresh.stdout + fresh.stderr
        self.sql("UPDATE experiment_scheduler_lease SET expires_at=clock_timestamp()-interval '1 second';")
        self.cycle('dead-owner-expired-takeover')
        if PHASE_T:
            self.expect(995241,'pending',True)
            assert self.capacity()==1
            assert self.sql('SELECT scheduler_invocation_id FROM experiment_scheduler_worker_attempt WHERE experiment_id=995241;')==owner
        assert self.inspect(self.workers[995240]['pid']) == worker_identity
        assert self.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=995240;') == '1'
        assert self.sql('SELECT scheduler_invocation_id FROM experiment_scheduler_worker_attempt WHERE experiment_id=995240;') == owner
        assert self.sql('SELECT observed_by_scheduler_invocation_id FROM experiment_scheduler_worker_attempt WHERE experiment_id=995240;') != owner
        assert int(self.sql('SELECT fencing_token FROM experiment_scheduler_lease;')) > int(fence)
        self.results['owner_crash_restart'] = 'PASS; fresh lease rejected, expired dead owner replaced, original attempt/launch owner retained'

        # Corrupt only a test lease token; an old context must lose authority
        # at its next heartbeat and must leave the existing worker alone.
        daemon = self.start_daemon('stale-fence-owner')
        self.sql('UPDATE experiment_scheduler_lease SET fencing_token=fencing_token+1;')
        self.wait(lambda: daemon['process'].poll() is not None, 'stale fence shutdown', seconds=5)
        self.stop_daemon(daemon)
        log = (self.out / 'stale-fence-owner.log').read_text()
        assert 'SCHEDULER_OWNERSHIP_LOST' in log
        assert self.inspect(self.workers[995240]['pid']) == worker_identity
        assert self.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=995240;') == '1'
        self.results['stale_fencing_token'] = 'PASS; native heartbeat stops stale owner without signaling worker'
        self.retire()

    def capacity_tests(self):
        # Resume after reservation but before parent exec permission. Only the
        # private scheduler's existing test failpoint facility delays this gate.
        self.launch(995280,priority='low',status='pending',origin='preemption')
        self.launch(995281,layout=13,priority='high',status='pending',origin='operator')
        self.signal_worker(995281,signal.SIGCONT)
        self.signal_worker(995281,signal.SIGTERM)
        self.workers[995281]['process'].wait(timeout=4)
        self.sql("SET expertadvisor.scheduler_protocol_generation='52';"
                 "UPDATE experiment_scheduler_worker_attempt SET lifecycle_state='abandoned' WHERE experiment_id=995281;"
                 "UPDATE experiment SET active_scheduler_worker_attempt_id=NULL,worker_pid=NULL,worker_process_group_id=NULL,"
                 "worker_process_start_identity=NULL,worker_executable=NULL,worker_command_line=NULL,"
                 "worker_control_state='running',resume_requested=false,scheduler_resume_origin='none' WHERE experiment_id=995281;")
        actor=[]
        def resume_at_gate():
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    r=subprocess.run([str(PG/'psql'),'-X','-Atq','-d','ea_scheduler_phase24s_lstm','-c',
                       "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=995281 AND lifecycle_state IN ('reserved','spawned');"],
                       env=self.env,capture_output=True,text=True,timeout=2)
                    if r.returncode==0 and r.stdout.strip()=='1':
                        self.signal_worker(995280,signal.SIGCONT)
                        actor.append('verified fixture resumed after reservation')
                        return
                    time.sleep(0.02)
                actor.append('reservation not observed')
            except Exception as error: actor.append(str(error))
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE']='1'
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY']='capacity_before_child_gate'
        observer=threading.Thread(target=resume_at_gate)
        observer.start()
        try:
            result=self.run('external-resume-before-child-gate',self.scheduler_args(0,1),expected=None)
        finally:
            observer.join(timeout=7)
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE']
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY']
        assert not observer.is_alive() and actor==['verified fixture resumed after reservation'],actor
        (self.out/'gate-race-actor.json').write_text(json.dumps(actor))
        assert result.returncode!=0
        assert 'scheduler_capacity_changed_before_child_gate' in result.stdout+result.stderr
        assert 'SCHEDULER_CHILD_LAUNCHED' not in result.stdout+result.stderr
        self.cycle('child-gate-race-recovery')
        self.expect(995280,'running',False)
        assert self.capacity()==1
        assert self.sql("SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=995280;")=='1'
        self.results['child_gate_race']='PASS externally resumed exact fixture blocks new child exec; original PID/attempt recovered'
        self.retire()

        # Failure after a verified SIGSTOP must roll back the durable state and
        # restore only the exact victim; the next normal cycle corrects excess.
        self.launch(995270,priority='normal')
        self.launch(995271,layout=13,priority='low',status='pending',origin='preemption')
        self.signal_worker(995271,signal.SIGCONT)
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE']='1'
        self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY']='capacity_after_sigstop_before_db'
        try:
            result=self.run('capacity-rollback-compensation',self.scheduler_args(0,1),expected=None)
        finally:
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE']
            del self.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY']
        assert result.returncode!=0
        assert 'injected_capacity_db_failure_after_sigstop' in result.stdout+result.stderr
        assert 'restored=1' in result.stdout+result.stderr
        self.expect(995270,'running',False)
        self.expect(995271,'running',False)
        self.cycle('capacity-rollback-retry')
        self.expect(995270,'running',False)
        self.expect(995271,'pending',True)
        assert self.capacity()==1
        self.results['capacity_rollback']='PASS exact victim restored after failed persistence; bounded retry corrected cap'
        self.retire()

        # Existing victim order: low loses before high; equal priority loses
        # newest start first. Stop/resume each original attempt repeatedly.
        for phase, base in [('infer', 996000), ('train', 996100)]:
            layout = 9 if phase == 'infer' else 13
            self.launch(base, phase, layout, 'normal', detached=True)
            self.launch(base+1, phase, 13, 'high', 'pending', 'preemption', detached=True)
            self.launch(base+2, phase, 13, 'low', 'pending', 'preemption')
            self.launch(base+3, phase, 13, 'high', 'paused')
            paused = self.sql(f'SELECT row_to_json(e) FROM experiment e WHERE experiment_id={base+3};')
            identities = {e:w['identity'] for e,w in self.workers.items()}
            caps = {'train': int(phase=='train'), 'infer': int(phase=='infer')}
            for eid in (base+1,base+2,base+3):
                self.signal_worker(eid, signal.SIGCONT)
            for i in range(3):
                self.cycle(f'{phase}-multiple-cap-correction-{i}', **caps)
                assert self.capacity(phase)==1
                self.expect(base+1,'running',False)
                self.expect(base,'pending',True)
                self.expect(base+2,'pending',True)
                self.expect(base+3,'paused',True)
            assert paused == self.sql(f'SELECT row_to_json(e) FROM experiment e WHERE experiment_id={base+3};')
            self.signal_worker(base+1,signal.SIGCONT)
            self.signal_worker(base,signal.SIGCONT)
            self.cycle(phase+'-repeated-external-resume',**caps)
            self.expect(base,'pending',True)
            # Remove only high-priority fixture, then ordinary policy admits
            # normal before low, without replacing either detached identity.
            self.signal_worker(base+1,signal.SIGTERM)
            self.wait(lambda:not self.inspect(self.workers[base+1]['pid']), 'high detached exit')
            self.cycle(phase+'-capacity-readmission',**caps)
            self.expect(base,'running',False)
            assert self.inspect(self.workers[base]['pid'])==identities[base]
            assert self.sql(f'SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id BETWEEN {base} AND {base+3};')=='4'
            self.results[phase+'_capacity_correction']='PASS unequal priority, multiple external resumes, operator pause, same detached PID/attempt readmission'
            self.retire()

        # Both independent caps, equal-priority deterministic order, zero caps.
        for eid,phase,layout in [(996200,'infer',9),(996201,'infer',13),
                                 (996202,'train',13),(996203,'train',13)]:
            self.launch(eid,phase,layout,status='running' if eid%2==0 else 'pending',origin='preemption' if eid%2 else 'none')
            if eid%2:self.signal_worker(eid,signal.SIGCONT)
        self.cycle('both-phases-cap-one',train=1,infer=1)
        for eid in (996200,996202): self.expect(eid,'running',False)
        for eid in (996201,996203): self.expect(eid,'pending',True)
        assert self.capacity('train')==self.capacity('infer')==1
        for eid in (996201,996203):self.signal_worker(eid,signal.SIGCONT)
        self.cycle('both-phases-cap-zero',train=0,infer=0)
        for eid in (996200,996201,996202,996203):self.expect(eid,'pending',True)
        assert self.capacity('train')==self.capacity('infer')==0
        self.cycle('zero-caps-idempotent',train=0,infer=0)
        self.cycle('both-phases-readmission',train=1,infer=1)
        assert self.capacity('train')==self.capacity('infer')==1
        self.results['independent_zero_caps']='PASS both phases, equal-priority oldest survivor, zero cap, later readmission'
        self.retire()

        # Identity uncertainty must not become a signal target. Known excess
        # may be safely stopped; the ambiguous worker remains accounted for.
        self.launch(996210,priority='low')
        self.launch(996211,layout=13,status='pending',origin='preemption')
        self.signal_worker(996211,signal.SIGCONT)
        self.sql("UPDATE experiment_scheduler_worker_attempt SET worker_process_start_identity='reused-pid' WHERE experiment_id=996210;")
        self.cycle('ambiguous-excess-fail-closed')
        self.expect(996210,'running',False)
        self.expect(996211,'pending',True)
        assert self.sql('SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE experiment_id=996210;')=='identity_ambiguous'
        self.results['ambiguous_excess']='PASS unknown identity never signaled, capacity retained, verified excess stopped'
        self.retire()

        # An in-flight checkpoint/control transition is never a stop target,
        # even if this means retaining it ahead of a higher-priority worker.
        self.launch(996220,priority='low')
        self.launch(996221,layout=13,priority='high',status='pending',origin='preemption')
        self.sql('UPDATE experiment SET stop_after_checkpoint_epoch=100 WHERE experiment_id=996220;')
        self.signal_worker(996221,signal.SIGCONT)
        self.cycle('in-flight-capacity-victim-excluded')
        self.expect(996220,'running',False)
        self.expect(996221,'pending',True)
        self.results['in_flight_exclusion']='PASS checkpoint transition retained; eligible excess safely stopped'
        self.retire()

        self.launch(996230,layout=13,status='paused')
        self.sql('UPDATE experiment SET stop_after_checkpoint_epoch=100 WHERE experiment_id=996230;')
        self.signal_worker(996230,signal.SIGCONT)
        log=self.cycle('paused-in-flight-control-deferred',train=0,infer=0)
        assert 'unsafe_transition_or_unobserved_owner' in log
        assert 'SCHEDULER_CAPACITY_ADMISSION_BLOCKED' in log
        self.expect(996230,'paused',False)
        self.results['paused_in_flight_deferred']='PASS explicit diagnostic, no ambiguous control signal or admission; operator recovery required'
        self.retire()

    def cleanup(self):
        errors = []
        for daemon in self.daemons:
            try:
                self.stop_daemon(daemon)
            except Exception as e:
                errors.append(str(e))
        try:
            if self.workers:
                self.retire()
        except Exception as e:
            errors.append(str(e))
        if self.started:
            try:
                # pg_ctl targets the owned, destination-verified data directory.
                self.run('stop-private-postgres', [PG / 'pg_ctl', '-D', self.data, '-m', 'fast', '-w', 'stop'])
                self.started = False
                shutil.rmtree(self.data)
            except Exception as e:
                errors.append(str(e))
        for name in ('pgpass', 'init-password'):
            (self.out / name).unlink(missing_ok=True)
        self.results['cleanup'] = 'PASS' if not errors else errors
        (self.out / 'results.json').write_text(json.dumps(self.results, indent=2))
        if errors:
            raise AssertionError('cleanup failed: ' + repr(errors))


if __name__ == '__main__':
    default = ROOT / 'DerivedData/ExpertAdvisor' / ('Phase24T' if PHASE_T else 'Phase24S') / time.strftime('Run%Y%m%d%H%M%S')
    qualification = Qualification(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
    try:
        qualification.setup()
        qualification.tests()
        print(json.dumps(qualification.results, indent=2))
    finally:
        qualification.cleanup()
