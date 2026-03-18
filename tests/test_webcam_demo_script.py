import subprocess
import sys


def test_webcam_demo_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/webcam_demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--camera-index" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--roi-expansion" in completed.stdout
    assert "--tracker-grace-frames" in completed.stdout
