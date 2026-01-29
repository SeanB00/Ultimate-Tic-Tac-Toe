import subprocess
import sys
import os
from pathlib import Path

# ===== CONFIG =====
PROJECT_DIR = Path.home() / "projects" / "Ultimate-Tic-Tac-Toe"
VENV_DIR = PROJECT_DIR / "venv"
PYTHON = "python3"
# ==================

def sh_bash(cmd: str):
    print(f"> {cmd}")
    subprocess.run(
        ["bash", "-lc", cmd],
        check=True
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 linux_runner.py <script.py> [args...]")
        sys.exit(1)

    script = sys.argv[1]
    args = sys.argv[2:]

    os.chdir(PROJECT_DIR)

    # Create venv if missing
    if not VENV_DIR.exists():
        sh_bash(f"{PYTHON} -m venv venv")

    # Activate venv, install deps, run script
    cmd = (
        "source venv/bin/activate && "
        "pip install --upgrade pip && "
        "pip install numpy && "
        "pip install -r torch && "
        f"{PYTHON} {script} {' '.join(args)}"
    )

    sh_bash(cmd)

if __name__ == "__main__":
    main()
