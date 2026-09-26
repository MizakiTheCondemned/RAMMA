"""
RAMMA start.

Compiled to RAMMA.exe by build_start_exe.bat. It does one thing: run
ramma.py with the venv's Python, from the folder the exe sits in.

No setup, no installing, no checks — the environment is assumed to be
there already. If it isn't, the launcher says which piece is missing
rather than failing silently.
"""

import os
import subprocess
import sys

APP  = "ramma.py"
VENV = "venv"


def base_dir():
    """The folder holding the exe (frozen) or this script."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def main():
    root = base_dir()
    os.chdir(root)

    py  = os.path.join(root, VENV,
                       "Scripts" if os.name == "nt" else "bin",
                       "python.exe" if os.name == "nt" else "python")
    app = os.path.join(root, APP)

    if not os.path.isfile(app):
        print(f"{APP} is not in {root}")
        print("Keep RAMMA.exe in the same folder as the app.")
        input("\nPress Enter to close...")
        return 1

    if not os.path.isfile(py):
        print(f"No virtual environment found at {os.path.join(root, VENV)}")
        print("Set one up first (RAMMA.bat setup), then use this launcher.")
        input("\nPress Enter to close...")
        return 1

    # Hand over to the app. Its console output — the [BS-RoFormer] and
    # [Karaoke] lines — comes through this window.
    code = subprocess.call([py, app])

    if code != 0:
        print()
        print(f"RAMMA exited with code {code} — the traceback is above.")
        input("\nPress Enter to close...")
    return code


if __name__ == "__main__":
    sys.exit(main())
