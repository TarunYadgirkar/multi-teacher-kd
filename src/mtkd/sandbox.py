"""Run model-written code against MBPP asserts with OS-level limits.

A program passes only if it exits 0 AND prints a random sentinel that is emitted after
the last assert, so sys.exit(0) / os._exit(0) before the tests cannot fake a pass.

Isolation, in layers:
- fresh temp dir as cwd, deleted afterwards; scrubbed environment (no tokens);
- `python -I -B` (ignores PYTHON* env vars and user site, writes no bytecode);
- rlimits: CPU seconds, file size, open files, address space (where the OS enforces it);
- macOS: sandbox-exec profile that denies all network and all writes outside the temp
  dir, and denies reads under $HOME except the Python install;
- Linux: network namespace via `unshare -rn` when unprivileged user namespaces work.
For hostile inputs, run the whole eval inside a container with --network none.
"""

import functools
import os
import resource
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MAX_OUTPUT = 64 * 1024
_FILE_LIMIT = 16 * 1024 * 1024
_ADDRESS_LIMIT = 4 * 1024**3
_NOFILE_LIMIT = 256


@dataclass(frozen=True)
class RunResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stderr_tail: str


def build_program(code: str, tests: list[str], imports: list[str], setup: str,
                  sentinel: str) -> str:
    parts = [*imports, setup, code, *tests, f"print({sentinel!r}, flush=True)"]
    return "\n".join(part for part in parts if part) + "\n"


def _limits(cpu_seconds: int) -> None:
    for name, value in (
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", _FILE_LIMIT),
        ("RLIMIT_NOFILE", _NOFILE_LIMIT),
        ("RLIMIT_AS", _ADDRESS_LIMIT),
        ("RLIMIT_CORE", 0),
    ):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError, AttributeError):
            pass


def _macos_profile(workdir: str) -> str:
    home = os.path.realpath(os.path.expanduser("~"))
    readable = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    readable.add(os.path.realpath(os.path.dirname(sys.executable)))
    readable.add(workdir)
    allow_reads = " ".join(f'(subpath "{p}")' for p in sorted(readable))
    return (
        "(version 1)(allow default)(deny network*)"
        f'(deny file-read-data (subpath "{home}"))(allow file-read-data {allow_reads})'
        "(deny file-write*)"
        f'(allow file-write* (subpath "{workdir}") (literal "/dev/null") (regex #"^/dev/fd/"))'
    )


@functools.cache
def _linux_netns_available() -> bool:
    if not sys.platform.startswith("linux") or shutil.which("unshare") is None:
        return False
    probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True)
    return probe.returncode == 0


def isolation_level() -> str:
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    if _linux_netns_available():
        return "netns"
    return "rlimits-only"


def _command(script: Path, workdir: str) -> list[str]:
    base = [sys.executable, "-I", "-B", str(script)]
    level = isolation_level()
    if level == "sandbox-exec":
        return ["sandbox-exec", "-p", _macos_profile(workdir), *base]
    if level == "netns":
        return ["unshare", "-rn", *base]
    return base


def run_tests(code: str, tests: list[str], imports: list[str] | None = None,
              setup: str = "", timeout_s: float = 10.0) -> RunResult:
    sentinel = f"MTKD_PASS_{secrets.token_hex(16)}"
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="mtkd_sbx_"))
    try:
        script = Path(workdir) / "program.py"
        script.write_text(build_program(code, tests, imports or [], setup, sentinel))
        env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8"}
        out_path, err_path = Path(workdir) / "stdout", Path(workdir) / "stderr"
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                _command(script, workdir), cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True,
                preexec_fn=functools.partial(_limits, int(timeout_s) + 1),
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
                return RunResult(False, True, None, "")
        stdout = _tail(out_path)
        passed = proc.returncode == 0 and sentinel in stdout
        return RunResult(passed, False, proc.returncode, _tail(err_path)[-2000:])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _tail(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - _MAX_OUTPUT))
        return f.read().decode(errors="replace")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError):
        proc.kill()
