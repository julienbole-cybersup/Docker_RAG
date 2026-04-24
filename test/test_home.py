import subprocess
import sys

def test_home_runs():
    result = subprocess.run(
        [sys.executable, "home.py", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
