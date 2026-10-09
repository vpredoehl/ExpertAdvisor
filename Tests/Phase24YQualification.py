#!/usr/bin/env python3
"""Development-only Phase 24Y build and regression driver; never publishes.

The private relink follows the retained Phase24T/U qualification recipe. It is
not an authorizable Release build: dirty source is recorded in the manifest,
and the repository's clean-tree provenance generator is left unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'DerivedData/ExpertAdvisor/Phase24Y/FinalQualification'
BUILD = OUT / 'build'
CLI = BUILD / 'LSTM_Release'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(name, args, timeout=90, expected=0, env=None):
    OUT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout)
    (OUT / (name + '.log')).write_text(result.stdout + result.stderr)
    with (OUT / 'commands.jsonl').open('a') as receipt:
        receipt.write(json.dumps({'name': name, 'argv': list(map(str, args)),
                                 'exit': result.returncode, 'timeout': timeout}) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode != expected:
        raise AssertionError(name + '\n' + (result.stdout + result.stderr)[-4000:])
    return result


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    retained = ROOT / 'DerivedData/ExpertAdvisor/Phase24T'
    inputs = retained / 'BuildInputs'
    assert inputs.is_dir()
    args = ['/usr/bin/clang++', '-std=c++20', '-O2', '-g',
            '-target', 'arm64-apple-macos27.0', '-Wall', '-Wextra',
            '-Wno-deprecated-declarations', '-Wno-c++23-attribute-extensions',
            '-I/opt/homebrew/opt/libpqxx@7.10.1/include',
            '-I/opt/homebrew/opt/libpq/include', '-isystem', str(inputs),
            '-I' + str(ROOT / 'Headers'), '-I' + str(ROOT / 'Sources'),
            '-I' + str(ROOT / 'Sources/SchedulerCore')]
    for directory in sorted(inputs.rglob('*')):
        if directory.is_dir():
            args += ['-isystem', str(directory)]
    sources = ['Sources/GlobalExperimentControl.cpp',
               'Sources/SchedulerCore/ReconciliationService.cpp',
               'Sources/SchedulerCore/PostgresSchedulerRepository.cpp',
               'Sources/SchedulerCore/ProductionSchedulerDaemon.cpp']
    for source in sources:
        name = Path(source).stem
        run('compile-' + name, args + ['-c', ROOT / source, '-o', BUILD / (name + '.o')])
    release = ROOT / 'DerivedData/ExpertAdvisor/Build/Products/Release'
    archive = BUILD / 'libSchedulerCore.a'
    shutil.copy2(release / 'libSchedulerCore.a', archive)
    run('archive', ['/usr/bin/ar', '-r', archive, BUILD / 'ReconciliationService.o',
                    BUILD / 'PostgresSchedulerRepository.o', BUILD / 'ProductionSchedulerDaemon.o'])
    original = ROOT / ('DerivedData/ExpertAdvisor/Build/Intermediates.noindex/'
                       'ExpertAdvisor.build/Release/LSTM Release.build/Objects-normal/'
                       'arm64/LSTM_Release.LinkFileList')
    objects = [str(BUILD / 'GlobalExperimentControl.o') if item.endswith('/GlobalExperimentControl.o')
               else item for item in original.read_text().splitlines()]
    filelist = BUILD / 'LSTM_Release.LinkFileList'
    filelist.write_text('\n'.join(objects) + '\n')
    link = json.loads((retained / 'cli-link.command.json').read_text())
    command = []
    index = 0
    while index < len(link):
        value = link[index]
        if value in ('-target', '-filelist', '-o'):
            replacement = {'-target': 'arm64-apple-macos27.0',
                           '-filelist': str(filelist), '-o': str(CLI)}[value]
            command += [value, replacement]
            index += 2
        elif value == '-lSchedulerCore':
            command.append(str(archive))
            index += 1
        elif value == '-Xlinker' and link[index + 1] in ('-object_path_lto', '-dependency_info'):
            command += link[index:index + 3] + [str(BUILD / Path(link[index + 3]).name)]
            index += 4
        else:
            command.append(value)
            index += 1
    run('link', command)
    run('sign', ['/usr/bin/codesign', '-s', '-', '--force', CLI])
    run('verify-signature', ['/usr/bin/codesign', '--verify', '--strict', CLI])
    run('file', ['/usr/bin/file', CLI])
    manifest = {'qualification_only': True, 'publishable_release': False,
                'recipe': 'Phase24T/U private object/archive relink',
                'artifact': str(CLI), 'sha256': digest(CLI),
                'head': run('head', ['/usr/bin/git', 'rev-parse', 'HEAD']).stdout.strip(),
                'status': run('status', ['/usr/bin/git', 'status', '--short']).stdout,
                'sources': {source: digest(ROOT / source) for source in sources},
                'scheduler_archive_sha256': digest(archive),
                'reused_objects': {item: digest(item) for item in objects},
                'provenance_limitation': 'Reused Release objects retain baseline provenance; '
                    'the dirty source hashes in this manifest are the qualification identity.'}
    (OUT / 'artifact.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PRIVATE QUALIFICATION ARTIFACT', CLI, manifest['sha256'], flush=True)


def offline():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(OUT / 'tmp'))
    (OUT / 'tmp').mkdir(parents=True, exist_ok=True)
    names = ['GlobalExperimentControlTests', 'OrphanedRunningExperimentReconciliationServiceTests',
             'SchedulerPhasePriorityTests', 'SchedulerCoreBoundaryTests',
             'SchedulerOrchestrationServiceTests', 'SchedulerAuthorityServiceTests',
             'SchedulerCycleServiceTests', 'SchedulerChildCompletionServiceTests',
             'SchedulerOperationalObservationBoundaryTests', 'SchedulerTrainingWorkerRoutingTests',
             'SchedulerAnalyzeWorkerRoutingTests', 'SchedulerInternalSeamTests',
             'SchedulerDaemonConfigurationTests', 'SchedulerRuntimeConfigValidationStructuralTests',
             'WorkerAttemptLifecycleServiceTests', 'SchedulerCanonicalPathTests', 'SchedulerZeroWorkerLimitsTests']
    for name in names:
        run(name, ['/bin/bash', ROOT / 'Tests' / (name + '.sh')], env=env)
    run('semantic-admission-compile', ['/usr/bin/clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
        '-IHeaders', '-ISources', 'Tests/SchedulerSemanticAdmissionTests.cpp', '-o', OUT / 'semantic-admission'])
    run('semantic-admission', [OUT / 'semantic-admission'])
    run('terminal-adapter', ['python3', '-B', 'Tests/RepositoryAgentTerminalTests.py'], env=env)


def native(faults_only=False):
    assert CLI.is_file()
    spec = importlib.util.spec_from_file_location('displacement', ROOT / 'Tests/SchedulerDisplacementRecoveryTests.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PHASE_T = True
    module.CLI = CLI
    q = module.Qualification(ROOT / 'DerivedData/ExpertAdvisor/Phase24T' /
                             ('Phase24Y-FinalNative-' + time.strftime('%Y%m%d%H%M%S')))
    try:
        q.setup()
        # All established displacement/capacity/ownership/checkpoint recovery
        # scenarios run through this exact private artifact and inert workers.
        if not faults_only:
            q.tests()
        for index, boundary in enumerate(('stopped_admission_before_sigcont',
                'stopped_admission_fence_loss_before_sigcont',
                'stopped_admission_sigcont_failure', 'stopped_admission_after_sigcont',
                'stopped_admission_before_commit')):
            eid = 998500 + index
            q.launch(eid, phase='infer', status='pending', origin='operator')
            q.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE'] = '1'
            q.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY'] = boundary
            result = q.run(boundary, q.scheduler_args(0, 1), expected=None)
            q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_ENABLE')
            q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY')
            if boundary.endswith('sigcont_failure'):
                assert result.returncode == 0
            else:
                assert result.returncode != 0
            q.snapshot(boundary)
            q.expect(eid, 'pending', boundary in (
                'stopped_admission_before_sigcont', 'stopped_admission_sigcont_failure',
                'stopped_admission_fence_loss_before_sigcont'))
            assert q.sql(f'SELECT lifecycle_state FROM experiment_scheduler_worker_attempt '
                         f'WHERE experiment_id={eid};') == 'stopped'
            assert q.sql(f'SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id={eid};') == '1'
            lease_active = q.sql("SELECT authority_state='active' AND "
                "expires_at>clock_timestamp() FROM experiment_scheduler_lease WHERE singleton;") == 't'
            if result.returncode != 0 and lease_active:
                # A crashed owner retains its lease until expiry. A competing
                # invocation must first fail closed; only the private fixture
                # clock is advanced to exercise eventual takeover promptly.
                rejected = q.run(boundary + '-live-lease-reject',
                                 q.scheduler_args(0, 1), expected=3)
                assert 'owner_lease_valid' in rejected.stdout
                q.sql("UPDATE experiment_scheduler_lease SET "
                      "expires_at=clock_timestamp()-interval '1 second';")
            q.cycle(boundary + '-recover', train=0, infer=1)
            q.expect(eid, 'running', False)
            assert q.sql(f'SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id={eid};') == '1'
            q.results[boundary] = 'PASS exact original worker/attempt recovered without duplicate'
            q.retire()
        q.launch(998510, phase='infer', status='pending', origin='operator')
        identity = q.workers[998510]['identity']
        q.sql("UPDATE experiment_scheduler_worker_attempt SET "
              "worker_process_start_identity='phase24y-reused-pid' WHERE experiment_id=998510;"
              "UPDATE experiment SET worker_process_start_identity='phase24y-reused-pid' "
              "WHERE experiment_id=998510;")
        q.cycle('resume-reused-pid-refused', train=0, infer=1)
        q.expect(998510, 'pending', True)
        assert q.inspect(q.workers[998510]['pid']) == identity
        assert q.sql('SELECT lifecycle_state FROM experiment_scheduler_worker_attempt '
                     'WHERE experiment_id=998510;') == 'identity_ambiguous'
        assert q.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt '
                     'WHERE experiment_id=998510;') == '1'
        q.results['resume_reused_pid'] = 'PASS no SIGCONT/replacement; ambiguous identity retained for operator review'
        q.retire()
        q.launch(998511, phase='infer', status='pending', origin='operator')
        q.env['EA_SCHEDULER_OWNERSHIP_TEST_ENABLE'] = '1'
        q.env['EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY'] = 'stopped_admission_after_sigcont'
        result = q.run('resume-exit-after-sigcont', q.scheduler_args(0, 1), expected=None)
        q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_ENABLE')
        q.env.pop('EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY')
        assert result.returncode != 0
        q.expect(998511, 'pending', False)
        q.sql("INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,threshold_logret,"
              "window_size,label_rule_id,target_type,from_date,to_date,completed_epochs,accuracy,accept_model,"
              "status,completed_at,inference_scope) SELECT last_model_id,symbol,prediction_horizon,"
              "c_next_threshold,30,1,0,infer_start::date::text,infer_end::date::text,target_epochs,0.5,false,"
              "'completed',clock_timestamp(),'final' FROM experiment WHERE experiment_id=998511;")
        q.signal_worker(998511, signal.SIGTERM)
        q.workers[998511]['process'].wait(timeout=4)
        for index in range(2):
            q.cycle('resume-exit-result-recovery-' + str(index), train=0, infer=1)
        assert q.sql('SELECT phase FROM experiment WHERE experiment_id=998511;') == 'analyze'
        assert q.sql('SELECT lifecycle_state FROM experiment_scheduler_worker_attempt '
                     'WHERE experiment_id=998511;') == 'completed'
        assert q.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt '
                     'WHERE experiment_id=998511;') == '1'
        q.results['resume_worker_exit'] = 'PASS committed inference recovered once after SIGCONT/rollback/exit; no repeat dispatch'
        q.retire()
        print(json.dumps(q.results, indent=2))
    finally:
        q.cleanup()
        name = 'native-fault-results.json' if faults_only else 'native-results.json'
        (OUT / name).write_text(json.dumps(q.results, indent=2) + '\n')


def priority():
    spec = importlib.util.spec_from_file_location('capacity', ROOT / 'Tests/SchedulerCapacityPriorityQualification.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.m.CLI = CLI
    q = module.CapacityQualification(ROOT / 'DerivedData/ExpertAdvisor/Phase24T' /
            ('Phase24Y-FinalPriority-' + time.strftime('%Y%m%d%H%M%S')))
    try:
        q.setup()
        q.tests()
        # Retire the intentionally unadmitted nineteenth ANALYZE fixture before
        # starting an independent displacement scenario. Otherwise FIFO picks
        # that older normal-priority candidate instead of this case's worker.
        q.sql("UPDATE experiment SET status='failed' WHERE experiment_id=981018 "
              "AND status='pending' AND active_scheduler_worker_attempt_id IS NULL;")
        q.launch(998700, phase='train', layout=13, priority='low')
        identity = q.workers[998700]['identity']
        q.queue_analyze(998701, priority='normal')
        q.cycle_w('normal-analyze-displaces-low-train', train=1, infer=0, analyze=1)
        q.expect(998700, 'pending', True)
        assert q.sql('SELECT status FROM experiment WHERE experiment_id=998701;') == 'running'
        assert q.inspect(q.workers[998700]['pid']) == identity
        assert q.sql('SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=998700;') == '1'
        q.results['normal_analyze_low_train'] = 'PASS existing cross-phase displacement; exact TRAIN attempt retained'
        print(json.dumps(q.results, indent=2))
    finally:
        q.cleanup()
        (OUT / 'priority-results.json').write_text(json.dumps(q.results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'offline', 'native', 'priority'))
    parser.add_argument('--faults-only', action='store_true',
                        help='run only the isolated native resume failure matrix')
    arguments = parser.parse_args()
    if arguments.phase == 'native':
        native(arguments.faults_only)
    else:
        assert not arguments.faults_only
        globals()[arguments.phase]()
