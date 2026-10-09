"""Host-owned command policy. No model output grants authority or selects a shell."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import uuid

ROLLOVER = Path('/Volumes/Developer SSD/ExpertAdvisor-Rollover')
PRODUCTION = Path('/Volumes/Developer SSD/ExpertAdvisor')
GIT = Path('/Applications/Xcode.app/Contents/Developer/usr/bin/git')
CLANG = Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++')
SDK = Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk')
SANDBOX = Path('/usr/bin/sandbox-exec')
READ_ROOTS = ('Sources', 'Headers', 'LSTM', 'Tests', 'Scripts', 'docs')
READ_FILES = ('AGENTS.md', 'ExpertAdvisor.xcodeproj/project.pbxproj')
MAX_INPUT_BYTES = 8 * 1024 * 1024
META = re.compile(r'[\x00-\x1f\x7f;|&<>`$\\]')
TEST_FILES = ('Tests/SchedulerPhasePriorityTests.cpp',
              'Sources/SchedulerCore/SchedulerPolicy.cpp',
              'Sources/SchedulerCore/SchedulerPolicy.hpp',
              'Sources/SchedulerCore/SchedulerPhasePriority.hpp',
              'Sources/SchedulerCore/SchedulerPhasePriorityService.hpp')
# Filled from reviewed source, never from model requests or current source at runtime.
TEST_HASHES = {'Tests/SchedulerPhasePriorityTests.cpp': '1880f5e3450c56d739e8b0a12aad3f751eb7b2a88b3d57ef2b0b2e97eeb05836', 'Sources/SchedulerCore/SchedulerPolicy.cpp': '5140b64d47b8fea1806202bda9b5f1317322d5bae1f3bcf2761a0eba398b5fc0', 'Sources/SchedulerCore/SchedulerPolicy.hpp': '8ac737ceeda39c7a4238f35829e9096a1f45c29181a3d76215bf32c5929fb4ef', 'Sources/SchedulerCore/SchedulerPhasePriority.hpp': '0cb03537f607e938e48d24ec4af3dd86f074a26022d3e7db46b4ae7bc6f53bd0', 'Sources/SchedulerCore/SchedulerPhasePriorityService.hpp': '95096409ae1d70871a9b803538dc0b20e7f6a35232219f2cbd9f6ef0bdd63399'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Rejected(ValueError):
    pass


def text(value, name, maximum=1024):
    if not isinstance(value, str) or not value or len(value) > maximum or META.search(value):
        raise Rejected(f'invalid {name}')
    return value


def exact(request, required, optional=()):
    if not isinstance(request, dict) or set(request) - set(required) - set(optional) or not set(required) <= set(request):
        raise Rejected('unexpected or missing request fields')


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise Rejected(f'invalid {name}')
    return value


def clean_path(root, relative, *, directory=False):
    relative = text(relative, 'repository path')
    parts = relative.split('/')
    if relative.startswith('/') or any(part in ('', '.', '..') or part.startswith('.') for part in parts):
        raise Rejected('path must be a normalized repository-relative path')
    cursor = root
    for part in parts:
        cursor /= part
        mode = cursor.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise Rejected('symlink paths are forbidden')
    resolved = cursor.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.is_relative_to(PRODUCTION):
        raise Rejected('resolved path outside approved repository')
    if directory != resolved.is_dir() or not directory and not resolved.is_file():
        raise Rejected('unsupported filesystem target')
    if not directory and resolved.stat().st_nlink != 1:
        raise Rejected('multiply-linked files are forbidden')
    return resolved


def read_source_bytes(root, relative):
    """Anchor every component with openat/O_NOFOLLOW before copying test data."""
    parts = relative.split('/')
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_INPUT_BYTES:
                raise Rejected('unsupported or oversized test input')
            data = source.read(MAX_INPUT_BYTES + 1)
            if len(data) > MAX_INPUT_BYTES:
                raise Rejected('test input grew beyond bound')
            return data
    finally:
        os.close(parent)


@dataclass(frozen=True)
class Plan:
    operation: str
    argv: tuple[str, ...]
    cwd: Path
    targets: tuple[str, ...]
    timeout: float = 10.0
    output_limit: int = 65536
    isolated_test: bool = False


class CommandPolicy:
    def __init__(self, root=ROLLOVER, *, tests_enabled=False):
        self.root = Path(root).resolve(strict=True)
        if self.root.is_relative_to(PRODUCTION) or not self.root.is_dir():
            raise Rejected('production repository cannot be a terminal root')
        self.tests_enabled = tests_enabled
        self.executables = {str(p): digest(p.read_bytes()) for p in (GIT, Path('/usr/bin/sed'), Path('/usr/bin/grep'), CLANG, SANDBOX) if p.is_file()}

    def file(self, value):
        path = clean_path(self.root, value)
        if value not in READ_FILES and value.split('/')[0] not in READ_ROOTS:
            raise Rejected('file is outside approved source/evidence roots')
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise Rejected('source file exceeds 8 MiB input limit')
        return path

    def validate(self, request):
        exact(request, ('operation',), ('path', 'pattern', 'start', 'end', 'revision', 'test_id', 'working_directory'))
        op = text(request['operation'], 'operation', 64)
        if request.get('working_directory', '.') != '.':
            raise Rejected('working directory is fixed to the approved root')
        allowed = {'git_status': (), 'git_diff': ('path',), 'git_log': (), 'git_show': ('path', 'revision'),
                   'read_file': ('path', 'start', 'end'), 'search_file': ('path', 'pattern'), 'run_test': ('test_id',)}
        if op not in allowed:
            raise Rejected('operation not approved')
        exact(request, ('operation',), (*allowed[op], 'working_directory'))
        targets = []
        path = None
        if 'path' in request:
            path = self.file(request['path'])
            targets.append(str(path))
        git_flags = ('--no-pager', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
                     '-c', 'diff.external=', '-c', 'credential.helper=', '-c', 'core.untrackedCache=false')
        if op == 'git_status':
            argv = (str(GIT), *git_flags, 'status', '--short')
        elif op == 'git_diff':
            argv = (str(GIT), *git_flags, 'diff', '--no-ext-diff', '--no-textconv', '--',
                    *( (request['path'],) if path else ('Sources', 'Headers', 'LSTM', 'Tests', 'Scripts', 'docs', 'AGENTS.md')))
        elif op == 'git_log':
            argv = (str(GIT), *git_flags, 'log', '-10', '--format=%H %s')
        elif op == 'git_show':
            if path is None:
                raise Rejected('git_show requires an approved file')
            revision = request.get('revision', 'HEAD')
            if not isinstance(revision, str) or not re.fullmatch(r'HEAD|[0-9a-f]{40}', revision):
                raise Rejected('revision must be HEAD or a full commit hash')
            argv = (str(GIT), *git_flags, 'show', '--no-ext-diff', '--no-textconv', revision + ':' + request['path'])
        elif op == 'read_file':
            if path is None:
                raise Rejected('read_file requires a path')
            start = integer(request.get('start', 1), 'start', 1, 1000000)
            end = integer(request.get('end', start + 199), 'end', start, start + 499)
            argv = ('/usr/bin/sed', '-n', f'{start},{end}p;{end}q', str(path))
        elif op == 'search_file':
            if path is None:
                raise Rejected('search_file requires a path')
            pattern = text(request.get('pattern'), 'pattern', 256)
            argv = ('/usr/bin/grep', '-F', '-n', '-m', '100', '--', pattern, str(path))
        else:
            if not self.tests_enabled or request.get('test_id') != 'scheduler_phase_priority':
                raise Rejected('test execution requires startup approval of the fixed test catalog')
            for name in TEST_FILES:
                original = self.file(name)
                if digest(read_source_bytes(self.root, name)) != TEST_HASHES.get(name):
                    raise Rejected('approved test dependency hash changed')
                targets.append(str(original))
            return Plan(op, (), self.root, tuple(targets), timeout=60.0, isolated_test=True)
        executable = Path(argv[0])
        if str(executable) not in self.executables or digest(executable.read_bytes()) != self.executables[str(executable)]:
            raise Rejected('executable identity changed or unavailable')
        return Plan(op, argv, self.root, tuple(targets))

    def git_metadata(self):
        """Read-only exception for this linked worktree's required Git metadata."""
        git = self.root / '.git'
        if git.is_symlink():
            raise Rejected('symlinked git directory')
        if git.is_dir():
            return (git.resolve(),)
        if self.root != ROLLOVER or not git.is_file():
            raise Rejected('unrecognized Git worktree metadata')
        expected = PRODUCTION / '.git/worktrees/ExpertAdvisor-Rollover'
        if git.read_text().strip() != 'gitdir: ' + str(expected) or expected.resolve() != expected:
            raise Rejected('Git worktree binding changed')
        common = PRODUCTION / '.git'
        if common.resolve() != common or (expected / 'commondir').read_text().strip() != '../..':
            raise Rejected('Git common-directory binding changed')
        return (expected, common / 'objects', common / 'refs', common / 'logs', common / 'config',
                common / 'packed-refs', common / 'shallow', common / 'HEAD', common / 'info')


class AuditLog:
    def __init__(self, directory, root):
        directory = Path(directory)
        if not directory.is_absolute() or not directory.parent.resolve().is_relative_to(root):
            raise Rejected('audit directory must be under approved repository')
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.parent.resolve() != directory.parent:
            raise Rejected('symlinked audit directory')
        directory.mkdir(mode=0o700, exist_ok=False)
        self.directory = directory
        self.fd = os.open(directory / 'audit.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence = 0

    def write(self, request_id, event, **fields):
        self.sequence += 1
        payload = json.dumps({'utc': utc(), 'sequence': self.sequence, 'request_id': request_id, 'event': event, **fields}, sort_keys=True) + '\n'
        data = payload.encode()
        offset = 0
        while offset < len(data):
            offset += os.write(self.fd, data[offset:])
        os.fsync(self.fd)

    def close(self):
        os.close(self.fd)


def sandbox_profile(read_roots, write_roots):
    quoted = lambda p: json.dumps(str(p))
    readers = ' '.join('(subpath ' + quoted(p) + ')' if Path(p).is_dir() else '(literal ' + quoted(p) + ')' for p in read_roots)
    return ('(version 1)(deny default)'
            '(allow process-fork)(allow process-exec)(allow sysctl-read)(allow mach-lookup)'
            '(allow process-info* (target self))(allow signal (target self))'
            '(allow file-read-metadata)'
            # macOS 27 dyld/libignition opens / as an openat root (Apple's
            # dyld-support.sb). This literal does not authorize its descendants.
            '(allow file-read* (literal "/"))'
            '(allow file-read* ' + readers + ')'
            '(allow file-map-executable ' + readers + ')'
            '(allow file-write* ' + ' '.join('(subpath ' + quoted(p) + ')' for p in write_roots) + ' (literal "/dev/null"))')


class CommandExecutor:
    def __init__(self, policy, audit):
        self.policy, self.audit = policy, audit

    def execute(self, request, *, proposer='controller'):
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        result = {'request_id': request_id, 'status': 'rejected', 'exit_code': None,
                  'stdout': '', 'stderr': '', 'timed_out': False, 'output_truncated': False,
                  'process': None, 'argv': [], 'working_directory': str(self.policy.root)}
        encoded = json.dumps(request, sort_keys=True).encode()
        self.audit.write(request_id, 'requested', proposer=proposer,
                         request=request if len(encoded) <= 8192 else None,
                         request_bytes=len(encoded), request_sha256=digest(encoded))
        try:
            if len(encoded) > 8192:
                raise Rejected('request exceeds 8192-byte admission limit')
            plan = self.policy.validate(request)
            self.audit.write(request_id, 'authorized', plan={'operation': plan.operation,
                             'argv': plan.argv, 'cwd': str(plan.cwd), 'targets': plan.targets,
                             'test_input_sha256': TEST_HASHES if plan.isolated_test else {},
                             'timeout_seconds': plan.timeout, 'output_limit_bytes': plan.output_limit,
                             'isolated_test': plan.isolated_test})
            if plan.isolated_test:
                result.update(self._test(plan, request_id))
            else:
                result.update(self._run(plan, request_id))
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        result['duration_seconds'] = time.monotonic() - started
        self.audit.write(request_id, 'result', result=result)
        return result

    def _test(self, plan, request_id):
        scratch = self.audit.directory / request_id
        scratch.mkdir(mode=0o700)
        for target in plan.targets:
            original = Path(target)
            relative = original.relative_to(self.policy.root)
            data = read_source_bytes(self.policy.root, str(relative))
            if digest(data) != TEST_HASHES.get(str(relative)):
                raise Rejected('test input changed before snapshot')
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o400)
        if digest(CLANG.read_bytes()) != self.policy.executables.get(str(CLANG)):
            raise Rejected('compiler identity changed')
        if not SDK.is_dir() or not SDK.resolve().is_relative_to(Path('/Applications/Xcode.app')):
            raise Rejected('approved macOS SDK missing or escaped')
        argv = (str(CLANG), '-isysroot', str(SDK), '-std=c++20', '-Wall', '-Wextra', '-Werror', '-I' + str(scratch / 'Sources'),
                str(scratch / TEST_FILES[0]), str(scratch / TEST_FILES[1]), '-o', str(scratch / 'test'))
        compiled = self._run(Plan(plan.operation, argv, scratch, plan.targets, 50.0, plan.output_limit, True), request_id)
        if compiled['status'] != 'completed' or compiled['exit_code'] != 0:
            return {**compiled, 'test_stage': 'compile'}
        executed = self._run(Plan(plan.operation, (str(scratch / 'test'),), scratch, (), 10.0, plan.output_limit, True), request_id)
        return {**executed, 'test_stage': 'execute', 'compile_result': compiled}

    def _run(self, plan, request_id):
        if sys.platform != 'darwin' or not SANDBOX.is_file():
            raise Rejected('macOS filesystem/network sandbox is required; no unsandboxed fallback')
        if digest(SANDBOX.read_bytes()) != self.policy.executables.get(str(SANDBOX)):
            raise Rejected('sandbox executable identity changed')
        scratch = self.audit.directory / (request_id + '-runtime-' + uuid.uuid4().hex[:8])
        scratch.mkdir(mode=0o700, exist_ok=False)
        readers = [Path('/System'), Path('/usr/lib'), Path('/usr/share'), Path('/usr/bin'), Path('/bin'),
                   Path('/Applications/Xcode.app'), Path('/Library/Developer'), Path('/dev/null'), Path('/dev/urandom'), scratch]
        readers += [plan.cwd] if plan.isolated_test else [self.policy.root]
        if plan.argv[0] == str(GIT):
            readers += list(self.policy.git_metadata())
        profile = sandbox_profile(readers, [scratch, plan.cwd] if plan.isolated_test else [scratch])
        profile_path = scratch / 'sandbox.sb'
        profile_path.write_text(profile)
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(scratch), 'TMPDIR': str(scratch),
               'LC_ALL': 'C', 'LANG': 'C', 'GIT_OPTIONAL_LOCKS': '0',
               'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
               'GIT_PAGER': 'cat', 'GIT_TERMINAL_PROMPT': '0', 'PYTHONDONTWRITEBYTECODE': '1'}
        argv = [str(SANDBOX), '-f', str(profile_path), *plan.argv]
        self.audit.write(request_id, 'launch', argv=argv, environment=env, cwd=str(plan.cwd),
                         profile_sha256=digest(profile.encode()), timeout_seconds=plan.timeout,
                         resolved_executable=str(Path(plan.argv[0]).resolve(strict=True)),
                         executable_sha256=digest(Path(plan.argv[0]).read_bytes()),
                         sandbox_executable_sha256=self.policy.executables[str(SANDBOX)])
        begin = time.monotonic()
        child = subprocess.Popen(argv, cwd=plan.cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                                 start_new_session=True, close_fds=True)
        identity = {'pid': child.pid, 'process_group_id': child.pid, 'launch_monotonic_ns': time.monotonic_ns(),
                    'identity_method': 'retained unreaped Popen child/session; verified process group before cleanup'}
        outputs = {'stdout': bytearray(), 'stderr': bytearray()}
        counts = {'stdout': 0, 'stderr': 0}
        retained = 0
        timed_out = False
        selector = selectors.DefaultSelector()
        try:
            self.audit.write(request_id, 'process_started', identity=identity)
            for name, pipe in (('stdout', child.stdout), ('stderr', child.stderr)):
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = plan.timeout - (time.monotonic() - begin)
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    counts[key.data] += len(chunk)
                    kept = chunk[:max(0, plan.output_limit - retained)]
                    outputs[key.data].extend(kept)
                    retained += len(kept)
            if not timed_out:
                try:
                    child.wait(timeout=max(0.001, plan.timeout - (time.monotonic() - begin)))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            # Do not poll/reap before timeout cleanup: retained PID cannot be reused.
            if child.returncode is None:
                if os.getpgid(child.pid) != child.pid:
                    raise RuntimeError('owned child session identity changed; no signal sent')
                try:
                    self.audit.write(request_id, 'cleanup_signal', identity=identity, signal=int(signal.SIGKILL))
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Retained direct child exited; still reap its handle.
                    child.wait(timeout=5)
            selector.close()
            child.stdout.close()
            child.stderr.close()
        result = {'status': 'timeout' if timed_out else ('completed' if child.returncode == 0 else 'failed'),
                'exit_code': child.returncode, 'stdout': outputs['stdout'].decode('utf-8', 'replace'),
                'stderr': outputs['stderr'].decode('utf-8', 'replace'), 'timed_out': timed_out,
                'output_truncated': sum(counts.values()) > retained, 'observed_output_bytes': counts,
                'retained_output_bytes': retained, 'process': identity, 'argv': list(plan.argv),
                'working_directory': str(plan.cwd), 'execution_duration_seconds': time.monotonic() - begin}
        self.audit.write(request_id, 'process_exited', result=result)
        return result
