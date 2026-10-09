#!/usr/bin/env python3
"""Offline policy/adapter tests; --native adds small macOS sandbox fixtures.

No MLX imports, model loads, production DB access, or active-worker controls.
"""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent')
os.environ['EXPERTADVISOR_REPOSITORY_ROOT'] = str(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from Scripts.RepositoryAgentTerminal.terminal_policy import (
    AuditLog, CommandExecutor, CommandPolicy, Plan, Rejected, GIT, TEST_FILES, TEST_HASHES, PRODUCTION, clean_path)
from Scripts.RepositoryAgentTerminal.mcp_adapter import make_server, QwenTerminalBridge, parse_command_proposal

NATIVE = '--native' in sys.argv
if NATIVE:
    sys.argv.remove('--native')


class CommandProposalParserTests(unittest.TestCase):
    proposal = '{"operation": "git_status"}'

    def test_plain_json_accepted(self):
        self.assertEqual(parse_command_proposal(self.proposal), {'operation': 'git_status'})

    def test_json_fenced_response_accepted(self):
        self.assertEqual(parse_command_proposal('```json\n' + self.proposal + '\n```'),
                         {'operation': 'git_status'})

    def test_unlabeled_fenced_json_accepted(self):
        self.assertEqual(parse_command_proposal('```\n' + self.proposal + '\n```'),
                         {'operation': 'git_status'})

    def test_prose_surrounding_json_rejected(self):
        for proposal in (self.proposal, '```json\n' + self.proposal + '\n```'):
            for raw in ('Here is the command:\n' + proposal, proposal + '\nDone.'):
                with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                    parse_command_proposal(raw)

    def test_multiple_objects_rejected(self):
        payload = self.proposal + '\n' + self.proposal
        for raw in (payload, '```json\n' + payload + '\n```'):
            with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                parse_command_proposal(raw)

    def test_malformed_json_rejected(self):
        for raw in ('{"operation":', '```json\n{"operation":\n```'):
            with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                parse_command_proposal(raw)

    def test_multiple_code_blocks_and_unsupported_fences_rejected(self):
        block = '```json\n' + self.proposal + '\n```'
        for raw in (block + '\n' + block, '```python\n' + self.proposal + '\n```',
                    '```json\n' + self.proposal, self.proposal + '\n```'):
            with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                parse_command_proposal(raw)

    def test_nonobject_json_rejected(self):
        for raw in ('[]', 'null', '```json\n[]\n```'):
            with self.subTest(raw=raw), self.assertRaises(Rejected):
                parse_command_proposal(raw)


class TerminalTests(unittest.TestCase):
    def setUp(self):
        area = ROOT / 'DerivedData/ExpertAdvisor/RepositoryAgentTerminal'
        area.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=area)
        self.root = Path(self.temporary.name).resolve()
        (self.root / 'Sources').mkdir()
        (self.root / 'Sources/example.cpp').write_text('first\nsecond needle\nthird\n')
        self.policy = CommandPolicy(self.root)
        self.audit = AuditLog(self.root / 'audit', self.root)
        self.executor = CommandExecutor(self.policy, self.audit)

    def tearDown(self):
        self.audit.close()
        self.temporary.cleanup()

    def request(self, operation='git_status', **fields):
        return {'operation': operation, **fields}

    def reject(self, request):
        with patch.object(self.executor, '_run', side_effect=AssertionError('must not spawn')):
            result = self.executor.execute(request, proposer='qwen')
        self.assertEqual(result['status'], 'rejected')
        self.assertIsNone(result['process'])
        return result

    def events(self):
        return [json.loads(line) for line in (self.audit.directory / 'audit.jsonl').read_text().splitlines()]

    def test_approved_readonly_catalog_and_explicit_arrays(self):
        for request in [self.request(), self.request('git_diff'), self.request('git_log'),
                        self.request('git_show', path='Sources/example.cpp', revision='HEAD'),
                        self.request('read_file', path='Sources/example.cpp', start=1, end=2),
                        self.request('search_file', path='Sources/example.cpp', pattern='needle')]:
            plan = self.policy.validate(request)
            self.assertTrue(Path(plan.argv[0]).is_absolute())
            self.assertIsInstance(plan.argv, tuple)
            self.assertFalse(plan.isolated_test)

    def test_reject_destructive_and_unrestricted_operations(self):
        for op in ['rm', 'shell', 'git_push', 'git_merge', 'git_reset', 'git_clean', 'git_checkout',
                   'psql', 'schedule', 'signal', 'curl', 'python', 'xcodebuild']:
            self.reject(self.request(op))

    def test_reject_executable_args_env_and_model_approval(self):
        for field, value in [('executable', '/bin/sh'), ('arguments', ['-c', 'rm -rf /']),
                             ('environment', {'PGHOST': 'localhost'}), ('approved', True), ('timeout', 100000)]:
            self.reject(self.request(**{field: value}))

    def test_reject_production_paths_and_roots(self):
        self.reject(self.request('read_file', path=str(PRODUCTION / 'Sources/x.cpp')))
        with self.assertRaises(Rejected): CommandPolicy(PRODUCTION)

    def test_reject_traversal_absolute_hidden_and_working_directory(self):
        for name in ['../x', 'Sources/../AGENTS.md', '/etc/passwd', 'Sources//example.cpp',
                     './Sources/example.cpp', '.git/config', 'Sources/.env']:
            self.reject(self.request('read_file', path=name))
        self.reject(self.request(working_directory='/tmp'))
        self.reject(self.request(working_directory='Sources'))

    def test_reject_shell_injection(self):
        for value in ['x;touch /tmp/x', 'x|cat', 'x&&id', 'x>out', '$(id)', '`id`', 'x\ny', 'x\\y']:
            self.reject(self.request('search_file', path='Sources/example.cpp', pattern=value))

    def test_reject_symlink_file_and_directory(self):
        (self.root / 'Sources/escape.cpp').symlink_to('/etc/passwd')
        self.reject(self.request('read_file', path='Sources/escape.cpp'))
        (self.root / 'Sources/escape').symlink_to(PRODUCTION / 'Sources', target_is_directory=True)
        self.reject(self.request('read_file', path='Sources/escape/file.cpp'))

    def test_reject_internal_symlinks_and_hardlinks(self):
        (self.root / 'Sources/alias.cpp').symlink_to(self.root / 'Sources/example.cpp')
        self.reject(self.request('read_file', path='Sources/alias.cpp'))
        os.link(self.root / 'Sources/example.cpp', self.root / 'Sources/hard.cpp')
        self.reject(self.request('read_file', path='Sources/hard.cpp'))

    def test_reject_read_bounds_and_revision_injection(self):
        for fields in [{'start': 0}, {'start': True}, {'start': 3, 'end': 2}, {'start': 1, 'end': 501}]:
            self.reject(self.request('read_file', path='Sources/example.cpp', **fields))
        for revision in ['--exec=id', 'HEAD:path', 'HEAD~1', 'main', 'abc;id']:
            self.reject(self.request('git_show', path='Sources/example.cpp', revision=revision))

    def test_default_test_denial(self):
        self.reject(self.request('run_test', test_id='scheduler_phase_priority'))
        self.policy.tests_enabled = True
        self.reject(self.request('run_test', test_id='AnalyzeHistoricalFixtureQualification'))

    def test_hash_pinned_test_source_closure(self):
        for name in TEST_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        self.policy.tests_enabled = True
        plan = self.policy.validate(self.request('run_test', test_id='scheduler_phase_priority'))
        self.assertTrue(plan.isolated_test)
        self.assertEqual(len(plan.targets), 5)
        (self.root / TEST_FILES[0]).write_text('changed test')
        self.reject(self.request('run_test', test_id='scheduler_phase_priority'))

    def test_audit_complete_for_denial_and_success(self):
        denial = self.reject(self.request('shell'))
        with patch.object(self.executor, '_run', return_value={'status': 'completed', 'exit_code': 0}):
            success = self.executor.execute(self.request())
        events = self.events()
        self.assertEqual([e['event'] for e in events], ['requested', 'result', 'requested', 'authorized', 'result'])
        self.assertEqual([e['sequence'] for e in events], list(range(1, 6)))
        self.assertTrue(all(e['utc'] for e in events))
        self.assertEqual(events[1]['result']['request_id'], denial['request_id'])
        self.assertEqual(events[-1]['result']['request_id'], success['request_id'])

    def test_reject_oversized_and_nonobject_requests(self):
        self.reject(['git_status'])
        result = self.reject({'operation': 'git_status', 'payload': 'x' * 9000})
        self.assertIn('8192', result['error'])
        self.assertIsNone(self.events()[-2]['request'])
        self.assertTrue(self.events()[-2]['request_sha256'])

    def test_existing_audit_and_symlinked_audit_rejected(self):
        with self.assertRaises(FileExistsError): AuditLog(self.root / 'audit', self.root)
        (self.root / 'linked').symlink_to(self.audit.directory, target_is_directory=True)
        with self.assertRaises(Rejected): AuditLog(self.root / 'linked/other', self.root)

    def test_no_unsandboxed_fallback(self):
        with patch('Scripts.RepositoryAgentTerminal.terminal_policy.sys.platform', 'linux'):
            with patch('subprocess.Popen', side_effect=AssertionError('no execution')):
                result = self.executor.execute(self.request())
        self.assertIn('sandbox is required', result['error'])

    def test_real_worktree_git_metadata_is_readonly_narrow_exception(self):
        roots = CommandPolicy(ROOT).git_metadata()
        self.assertNotIn(PRODUCTION, roots)
        self.assertIn(PRODUCTION / '.git/objects', roots)
        (self.root / '.git').symlink_to(PRODUCTION / '.git', target_is_directory=True)
        with self.assertRaises(Rejected): self.policy.git_metadata()

    def server(self):
        return make_server(CommandExecutor(CommandPolicy(ROOT), self.audit))

    def call(self, server, name, arguments):
        return server._handle_request({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                                      'params': {'name': name, 'arguments': arguments}})['result']

    def test_existing_mcp_analysis_compatibility(self):
        from Tools.RepositoryAgent.repository_agent_mcp import TOOLS as ORIGINAL
        server = self.server()
        self.assertEqual(list(server._tools[:len(ORIGINAL)]), ORIGINAL)
        cap = self.call(server, 'capabilities', {})['structuredContent']
        self.assertEqual(cap['protocol'], 'expertadvisor.repository.readonly.v1')
        self.assertIn('shell', cap['forbidden_capabilities'])
        read = self.call(server, 'read', {'file': 'Sources/SchedulerCore/SchedulerPolicy.hpp', 'start': 1, 'end': 5})
        self.assertFalse(read['isError'])
        self.assertIn('read_only', cap)
        self.assertFalse(self.call(server, 'terminal_capabilities', {})['structuredContent']['tests_enabled'])
        self.assertTrue(self.call(server, 'shell', {'command': 'id'})['isError'])
        self.assertNotIn('mlx_lm', sys.modules)

    def test_inspection_cannot_switch_to_tests_and_test_tool_closed(self):
        server = self.server()
        for name, request in [('terminal_inspect', self.request('run_test', test_id='scheduler_phase_priority')),
                              ('terminal_test', self.request()), ('terminal_test', {'test_id': 'x', 'environment': {}})]:
            self.assertTrue(self.call(server, name, request)['isError'])

    def test_adapter_cannot_start_for_production(self):
        with patch.dict(os.environ, {'EXPERTADVISOR_REPOSITORY_ROOT': str(PRODUCTION)}):
            with self.assertRaises(Rejected): self.server()

    def bridge(self, responses):
        runtime = types.SimpleNamespace(loaded=True, model=object(), tokenizer=object(),
                     model_name='mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit')
        interface = types.SimpleNamespace(_claim_runtime=runtime)
        received = []
        def generate(model, tokenizer, messages, max_tokens):
            self.assertIs(model, runtime.model)
            self.assertIs(tokenizer, runtime.tokenizer)
            self.assertLessEqual(max_tokens, 160)
            received.append(messages[-1]['content'])
            return responses.pop(0)
        return QwenTerminalBridge(interface, self.executor, generate), received

    def test_model_request_result_exchange_reuses_runtime_with_mock(self):
        request = self.request('read_file', path='Sources/example.cpp', start=1, end=2)
        plain = json.dumps(request)
        for raw in (plain, '```json\n' + plain + '\n```', '```\n' + plain + '\n```'):
            with self.subTest(raw=raw):
                bridge, received = self.bridge([raw, 'Received exit 0.'])
                with patch.object(self.executor, '_run', return_value={'status': 'completed', 'exit_code': 0, 'stdout': 'first\nsecond needle\n'}):
                    result = bridge.exchange('Read the approved lines.', [request])
                self.assertEqual(result['result']['exit_code'], 0)
                self.assertIn('first', received[1])
                self.assertIn('STRUCTURED TERMINAL RESULT', received[1])
                self.assertTrue(result['runtime_reused'])
                self.assertEqual(self.events()[-1]['event'], 'qwen_result_delivered')
                self.assertEqual([e for e in self.events() if e['event'] == 'qwen_command_proposed'][-1]['raw'], raw)

    def test_model_output_never_grants_permission(self):
        for request in (self.request('shell'), self.request(environment={'PGHOST': 'localhost'})):
            plain = json.dumps(request)
            for raw in (plain, '```json\n' + plain + '\n```', '```\n' + plain + '\n```'):
                with self.subTest(raw=raw):
                    bridge, _ = self.bridge([raw])
                    with patch.object(self.executor, '_run', side_effect=AssertionError('no spawn')):
                        with self.assertRaises(Rejected): bridge.exchange('Inspect', [self.request()])
                    self.assertEqual(self.events()[-1]['event'], 'qwen_unoffered_request_rejected')

    def test_bridge_does_not_load_second_model(self):
        bridge = QwenTerminalBridge(types.SimpleNamespace(_claim_runtime=None), self.executor)
        with self.assertRaises(Rejected): bridge.exchange('Inspect', [self.request()])
        self.assertNotIn('mlx_lm', sys.modules)

    def test_qwen_malformed_and_wrong_model_fail_closed(self):
        bridge, _ = self.bridge(['not JSON'])
        with self.assertRaises(json.JSONDecodeError): bridge.exchange('Inspect', [self.request()])
        self.assertEqual(self.events()[-1]['event'], 'qwen_command_proposed')
        bridge.interface._claim_runtime.model_name = 'unapproved model'
        with self.assertRaises(Rejected): bridge.exchange('Inspect', [self.request()])

    def test_qwen_mcp_schema_and_unloaded_runtime(self):
        server = self.server()
        good = {'task': 'Inspect status', 'offered_commands': [self.request()]}
        result = self.call(server, 'qwen_terminal_inspect', good)
        self.assertTrue(result['isError'])
        self.assertIn('no second loader', result['structuredContent']['error'])
        self.assertTrue(self.call(server, 'qwen_terminal_inspect', {**good, 'environment': {}})['isError'])

    def test_changed_executable_hash_fails_before_spawn(self):
        self.policy.executables['/usr/bin/sed'] = 'changed'
        self.reject(self.request('read_file', path='Sources/example.cpp'))

    def test_oversized_source_rejected_before_spawn(self):
        with (self.root / 'Sources/example.cpp').open('wb') as file:
            file.truncate(8 * 1024 * 1024 + 1)
        self.reject(self.request('read_file', path='Sources/example.cpp'))

    def test_failed_audit_prevents_execution(self):
        with patch.object(self.audit, 'write', side_effect=OSError('audit unavailable')):
            with patch('subprocess.Popen', side_effect=AssertionError('must not spawn')):
                with self.assertRaises(OSError): self.executor.execute(self.request())

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_isolated_test_compile_and_execution_are_distinct(self):
        for name in TEST_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.policy.tests_enabled = True
        result = self.executor.execute(self.request('run_test', test_id='scheduler_phase_priority'))
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['compile_result']['exit_code'], 0)
        launches = [e for e in self.events() if e['event'] == 'launch']
        self.assertEqual(len(launches), 2)
        self.assertNotEqual(launches[0]['argv'][2], launches[1]['argv'][2])
        exits = [e for e in self.events() if e['event'] == 'process_exited']
        self.assertEqual(len(exits), 2)
        self.assertTrue(all(e['result']['exit_code'] == 0 for e in exits))

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_approved_read_search_and_failure(self):
        result = self.executor.execute(self.request('read_file', path='Sources/example.cpp', start=1, end=2))
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['stdout'], 'first\nsecond needle\n')
        absent = self.executor.execute(self.request('search_file', path='Sources/example.cpp', pattern='absent'))
        self.assertEqual((absent['status'], absent['exit_code']), ('failed', 1), absent)
        events = self.events()
        self.assertTrue(any(e['event'] == 'process_started' for e in events))
        self.assertTrue(all('utc' in e for e in events))

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_output_limit(self):
        (self.root / 'Sources/example.cpp').write_text('x' * 100000)
        result = self.executor.execute(self.request('read_file', path='Sources/example.cpp', start=1, end=1))
        self.assertEqual(result['status'], 'completed', result)
        self.assertTrue(result['output_truncated'])
        self.assertEqual(result['retained_output_bytes'], 65536)
        self.assertGreater(result['observed_output_bytes']['stdout'], 65536)

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_timeout_and_owned_cleanup(self):
        # Trusted unit fixture only, never an admitted model/public command.
        plan = Plan('unit_timeout', ('/bin/sleep', '5'), self.root, (), timeout=0.05)
        with patch.object(self.policy, 'validate', return_value=plan):
            result = self.executor.execute(self.request())
        self.assertTrue(result['timed_out'], result)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['exit_code'], -9)
        with self.assertRaises(ProcessLookupError): os.getpgid(result['process']['pid'])
        self.assertTrue(any(e['event'] == 'cleanup_signal' for e in self.events()))

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_git_fixture(self):
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root), 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
        subprocess.run([str(GIT), 'init', str(self.root)], check=True, capture_output=True, env=env)
        result = self.executor.execute(self.request())
        self.assertEqual(result['status'], 'completed', result)
        self.assertIn('Sources/', result['stdout'])

    @unittest.skipUnless(NATIVE, 'enable --native for isolated macOS execution')
    def test_native_network_and_write_denial_in_private_fixture(self):
        # Fixed fixture verifies kernel denial without touching a production path.
        outside = self.root / 'outside.txt'
        secret = self.root.parent / (self.root.name + '-private-fixture-secret.txt')
        secret.write_text('private fixture secret')
        try:
            profile = __import__('Scripts.RepositoryAgentTerminal.terminal_policy', fromlist=['sandbox_profile']).sandbox_profile
            sb = profile([Path('/System'), Path('/usr/lib'), Path('/usr/bin'), Path('/bin'), Path('/Applications/Xcode.app'), Path('/dev/urandom')], [self.audit.directory])
            fixture = 'import socket,pathlib; failures=0\nfor action in [lambda:socket.create_connection(("127.0.0.1",9),timeout=0.1),lambda:pathlib.Path(' + repr(str(outside)) + ').write_text("x"),lambda:pathlib.Path(' + repr(str(secret)) + ').read_text()]:\n try: action()\n except PermissionError: failures+=1\nprint(failures)\nassert failures==3\n'
            r = subprocess.run(['/usr/bin/sandbox-exec', '-p', sb, '/Applications/Xcode.app/Contents/Developer/usr/bin/python3', '-I', '-B', '-c', fixture],
                               capture_output=True, text=True, timeout=10, env={'PATH': '/usr/bin:/bin'})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('3', r.stdout)
            self.assertFalse(outside.exists())
        finally:
            secret.unlink()


if __name__ == '__main__':
    unittest.main()
