"""Who/what/where metadata written into every result file."""

import datetime
import platform
import subprocess
from importlib.metadata import version


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _chip() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            pass
    return platform.processor() or platform.machine()


def provenance(device: str) -> dict:
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown"),
        "versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft", "datasets")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "chip": _chip(),
        "device": device,
        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
