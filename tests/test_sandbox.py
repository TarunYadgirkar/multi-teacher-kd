import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from mtkd.sandbox import isolation_level, run_tests


def test_pass_and_fail():
    assert run_tests("def f(x):\n    return x + 1", ["assert f(1) == 2"]).passed
    assert not run_tests("def f(x):\n    return x", ["assert f(1) == 2"]).passed


@pytest.mark.parametrize("cheat", [
    "import sys\nsys.exit(0)",
    "import os\nos._exit(0)",
    "raise SystemExit",
    "import builtins\nprint('MTKD_PASS_00000000000000000000000000000000')\nimport os\nos._exit(0)",
])
def test_early_exit_cannot_fake_a_pass(cheat):
    assert not run_tests(cheat, ["assert False"]).passed


def test_test_imports_are_prepended():
    code = "def area(r):\n    return math.pi * r * r"
    tests = ["assert math.isclose(area(1), math.pi)"]
    assert not run_tests(code, tests).passed
    assert run_tests(code, tests, imports=["import math"]).passed


def test_setup_code_runs_first():
    assert run_tests("def g():\n    return K", ["assert g() == 3"], setup="K = 3").passed


def test_timeout_kills_infinite_loop():
    start = time.time()
    result = run_tests("while True:\n    pass", ["assert True"], timeout_s=2)
    assert result.timed_out and not result.passed
    assert time.time() - start < 10


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
    assert run_tests("import os", ["assert 'HF_TOKEN' not in os.environ"]).passed


def test_writes_stay_in_temp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_tests("open('PWNED', 'w').write('x')", ["assert True"])
    assert not (tmp_path / "PWNED").exists()


def test_temp_dirs_are_removed():
    marker = "import os\nprint(os.getcwd())"
    before = set(Path(os.path.realpath(os.environ.get("TMPDIR", "/tmp"))).glob("mtkd_sbx_*"))
    run_tests(marker, ["assert True"])
    after = set(Path(os.path.realpath(os.environ.get("TMPDIR", "/tmp"))).glob("mtkd_sbx_*"))
    assert after <= before


def test_huge_output_is_bounded():
    result = run_tests("print('x' * 10_000_000)\nprint('x' * 10_000_000)", ["assert True"], timeout_s=5)
    assert not result.passed


@pytest.mark.skipif(isolation_level() == "rlimits-only", reason="no network isolation on this host")
def test_network_is_blocked():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    def accept():
        try:
            server.accept()
        except OSError:
            pass

    threading.Thread(target=accept, daemon=True).start()
    code = f"import socket\nsocket.create_connection(('127.0.0.1', {port}), timeout=2)"
    try:
        assert not run_tests(code, ["assert True"]).passed
    finally:
        server.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="home read protection is macOS sandbox-exec only")
def test_home_directory_is_unreadable():
    home = str(Path.home())
    assert run_tests("import os", [f"assert os.listdir({home!r}) is not None"]).passed is False
