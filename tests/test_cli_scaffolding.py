"""
Test file for CLI scaffolding functionality.
"""

import subprocess
import sys
import os

# Add the current directory to Python path so we can import nanocord
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


def test_help_commands():
    """Test that all new help commands work without error."""

    # Test main help - run the actual CLI entry point
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test dataset help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "dataset", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test dataset cpt help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "dataset", "cpt", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test dataset sft help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "dataset", "sft", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test train help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "train", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test train cpt help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "train", "cpt", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test train sft help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "train", "sft", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test bot help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "bot", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test bot register help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "bot", "register", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0

    # Test pipeline help
    result = subprocess.run([sys.executable, "src/nanocord/cli.py", "pipeline", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0


if __name__ == "__main__":
    test_help_commands()
    print("All CLI help tests passed!")