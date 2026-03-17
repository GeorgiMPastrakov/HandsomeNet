import subprocess
import sys


def test_training_plan_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_training_plan.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "baseline_short" in completed.stdout
    assert "handsomenet_short" in completed.stdout
