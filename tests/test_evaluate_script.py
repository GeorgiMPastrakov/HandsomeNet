import subprocess
import sys


def test_evaluate_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--model" in completed.stdout
