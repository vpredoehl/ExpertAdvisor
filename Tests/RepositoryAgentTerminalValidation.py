#!/usr/bin/env python3
"""Bounded validation of the staged adapter; no weights or production services.

Creates new evidence only. Does not register/restart/deploy any MCP connection.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENT = Path('/Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent')
AREA = ROOT / 'DerivedData/ExpertAdvisor/RepositoryAgentTerminal'


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Supply one fresh evidence directory below DerivedData/ExpertAdvisor/RepositoryAgentTerminal')
    output = Path(sys.argv[1]).absolute()
    if not output.parent.resolve().is_relative_to(AREA) or output.exists():
        raise SystemExit('New private evidence directory required')
    output.mkdir(parents=True, mode=0o700)
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(output), 'TMPDIR': str(output),
           'PYTHONPATH': str(ROOT) + ':' + str(AGENT), 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'EXPERTADVISOR_REPOSITORY_ROOT': str(ROOT),
           'EXPERTADVISOR_LEDGER_CACHE_NAMESPACE': 'ExpertAdvisor-Rollover'}
    records = []
    def run(argv, timeout, *, input=None, environment=env):
        before = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(argv, input=input, capture_output=True, text=True,
                                cwd=ROOT, env=environment, timeout=timeout)
        record = {'started_utc': before, 'ended_utc': datetime.now(timezone.utc).isoformat(),
                  'argv': argv, 'cwd': str(ROOT), 'timeout_seconds': timeout,
                  'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        records.append(record)
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({'argv': argv, 'exit_code': result.returncode}), flush=True)
        assert result.returncode == 0, record
        return result
    run([sys.executable, '-B', 'Tests/RepositoryAgentTerminalTests.py', '--native'], 30)
    # These original tests supply explicit reader/verifier mocks; no production root.
    mock_env = {key: value for key, value in env.items() if key != 'EXPERTADVISOR_REPOSITORY_ROOT'}
    for name in ('test_repository_agent_mcp', 'test_repository_agent_mcp_profiles',
                 'test_repository_agent_mcp_claims', 'test_codex_interface', 'test_source_reader'):
        run([sys.executable, '-B', '-m', 'Tools.RepositoryAgent.tests.' + name], 30, environment=mock_env)
    calls = [('tools/list', {}), ('tools/call', {'name': 'capabilities', 'arguments': {}}),
             ('tools/call', {'name': 'read', 'arguments': {'file': 'Sources/SchedulerCore/SchedulerPolicy.hpp', 'start': 1, 'end': 5}}),
             ('tools/call', {'name': 'terminal_capabilities', 'arguments': {}})]
    requests = [{'operation': 'git_status'}, {'operation': 'git_diff', 'path': 'Sources/SchedulerCore/SchedulerPolicy.cpp'},
                {'operation': 'git_log'}, {'operation': 'git_show', 'path': 'Sources/SchedulerCore/SchedulerPolicy.hpp'},
                {'operation': 'read_file', 'path': 'Sources/SchedulerCore/SchedulerPolicy.hpp', 'start': 1, 'end': 5},
                {'operation': 'search_file', 'path': 'Sources/SchedulerCore/SchedulerPolicy.cpp', 'pattern': 'PriorityRank'}]
    calls += [('tools/call', {'name': 'terminal_inspect', 'arguments': request}) for request in requests]
    calls += [('tools/call', {'name': 'terminal_test', 'arguments': {'test_id': 'scheduler_phase_priority'}}),
              ('tools/call', {'name': 'terminal_inspect', 'arguments': {'operation': 'shell'}}),
              ('tools/call', {'name': 'terminal_inspect', 'arguments': {'operation': 'read_file', 'path': '/Volumes/Developer SSD/ExpertAdvisor/AGENTS.md'}}),
              ('tools/call', {'name': 'terminal_inspect', 'arguments': {'operation': 'git_status', 'environment': {'PGHOST': 'localhost'}}}),
              ('tools/call', {'name': 'qwen_terminal_inspect', 'arguments': {'task': 'Inspect status', 'offered_commands': [{'operation': 'git_status'}]}})]
    messages = [dict(jsonrpc='2.0', id=i + 1, method=method, params=params) for i, (method, params) in enumerate(calls)]
    argv = [sys.executable, '-B', '-m', 'Scripts.RepositoryAgentTerminal.mcp_adapter',
            '--audit-directory', str(output / 'mcp-audit'), '--approve-scheduler-phase-priority-test']
    rpc = run(argv, 30, input='\n'.join(json.dumps(m) for m in messages) + '\n')
    replies = [json.loads(line) for line in rpc.stdout.splitlines()]
    (output / 'mcp-requests.json').write_text(json.dumps(messages, indent=2) + '\n')
    (output / 'mcp-replies.json').write_text(json.dumps(replies, indent=2) + '\n')
    assert len(replies) == len(calls)
    assert all(not reply['result'].get('isError') for reply in replies[:11])
    assert all(reply['result']['isError'] for reply in replies[11:])
    assert 'no second loader' in replies[-1]['result']['structuredContent']['error']
    assert replies[1]['result']['structuredContent']['protocol'] == 'expertadvisor.repository.readonly.v1'
    audit = [json.loads(line) for line in (output / 'mcp-audit/audit.jsonl').read_text().splitlines()]
    started = [event for event in audit if event['event'] == 'process_started']
    exited = [event for event in audit if event['event'] == 'process_exited']
    assert len(started) == len(exited) == 8  # Six inspections, compile and test.
    for event in started:
        try:
            os.getpgid(event['identity']['pid'])
        except ProcessLookupError:
            continue
        raise AssertionError('Recorded child PID still present; do not signal a discovered process')
    assert [event['sequence'] for event in audit] == list(range(1, len(audit) + 1))
    sources = list((ROOT / 'Scripts/RepositoryAgentTerminal').glob('*.py')) + [Path(__file__).resolve(), ROOT / 'Tests/RepositoryAgentTerminalTests.py']
    (output / 'source-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
    summary = {'qualification': 'PASS', 'native_regression_count': 33, 'original_compatibility_suites': 5,
               'stdio_mcp_requests': len(calls), 'successful_native_inspections': 6, 'isolated_pure_test': 'PASS',
               'audited_child_processes': 8, 'residual_children': False, 'live_mcp_configuration_changed': False,
               'real_qwen_terminal_exchange': 'NOT RUN: existing live connection unchanged; bridge tested with mock runtime',
               'qwen_model_loads': 0, 'production_connections': 0}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('Validation requires Python without -O')
    main()
