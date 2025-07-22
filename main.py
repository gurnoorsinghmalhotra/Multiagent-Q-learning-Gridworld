# main.py
"""
Entry‑point menu for the Q‑Learning / DQN GridWorld project.

Options:
    1 – Train agents (calls train.cli_main)
    2 – Simulate / animate (calls simulate.cli_main)
    3 – Exit
"""

import subprocess
import sys
from pathlib import Path

def run_sub(command: list[str]):
    """Run a sub‑process (same Python) and stream output."""
    python = sys.executable
    proc   = subprocess.Popen([python, *command])
    proc.communicate()

def menu() -> str:
    print("\nQ‑Learning GridWorld")
    print("────────────────────")
    print("1  Train agents")
    print("2  Simulate / animate trained agents")
    print("3  Exit\n")
    return input("> ").strip()

def ensure_runs_dir():
    Path("runs").mkdir(exist_ok=True)

if __name__ == "__main__":
    ensure_runs_dir()

    while True:
        choice = menu()

        if choice == "1":
            # Example: train 5×5 grid for 10 000 episodes
            cmd = ["train.py",
                   "--episodes", "10000",
                   "--grid-size", "5",
                   "--n-agents", "1",
                   "--out-dir", "runs"]
            print("\n[main] Launching training…\n")
            run_sub(cmd)

        elif choice == "2":
            ckpt = Path("runs/checkpoint_final.pt")
            if not ckpt.exists():
                print("\n[main] ERROR: runs/checkpoint_final.pt not found.  Train first!\n")
                continue

            # Example: simulate 3 episodes and create GIF in assets/
            cmd = ["simulate.py",
                   "--checkpoint", str(ckpt),
                   "--episodes", "3",
                   "--grid-size", "5",
                   "--n-agents", "1",
                   "--gif", "assets/agent_demo.gif"]
            print("\n[main] Launching simulation…\n")
            run_sub(cmd)

        elif choice == "3":
            print("Good‑bye!")
            break

        else:
            print("Please choose 1, 2, or 3.")
