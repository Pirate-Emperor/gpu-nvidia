#!/usr/bin/env python3
"""Read-only audit of whether the current node can upgrade the host NVIDIA driver."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return 127, str(exc)
    return result.returncode, result.stdout.strip()

def main() -> int:
    uid = os.getuid()
    euid = os.geteuid()
    driver = run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv"]
    )[1]

    sudo_available = shutil.which("sudo") is not None
    sudo_passwordless = False
    sudo_policy = ""
    if sudo_available:
        code, _ = run(["sudo", "-n", "true"])
        sudo_passwordless = code == 0
        policy_code, policy_output = run(["sudo", "-n", "-l"])
        if policy_code == 0:
            sudo_policy = policy_output
        else:
            sudo_policy = "sudo -n -l requires credentials or is not permitted"

    package_manager = next(
        (name for name in ("apt-get", "dnf", "yum", "zypper") if shutil.which(name)),
        None,
    )

    print(json.dumps({
        "uid": uid,
        "euid": euid,
        "is_root": uid == 0 or euid == 0,
        "driver_query": driver,
        "sudo_available": sudo_available,
        "sudo_passwordless": sudo_passwordless,
        "sudo_policy": sudo_policy,
        "package_manager": package_manager,
        "nvcc": shutil.which("nvcc"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "home": str(Path.home()),
        "warning": "Read-only audit. No install, upgrade, unload, or reboot operation is performed.",
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
