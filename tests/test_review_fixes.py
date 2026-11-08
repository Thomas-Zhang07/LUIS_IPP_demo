"""Regression tests for issues found in the post-cleanup review."""

import subprocess
import sys
from pathlib import Path

import pytest

import onoff
import transducer_onoff_demo
import transducer_onoff_steer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module", [onoff, transducer_onoff_demo, transducer_onoff_steer]
)
def test_uart_achievable_low_frequency_is_accepted(module):
    # The UART tops out near 7.4 Hz, so rates below the old 20 Hz floor must be
    # selectable.
    assert module.parse_frequency("2") == 2.0
    assert module.parse_frequency("0.1") == 0.1
    with pytest.raises(ValueError):
        module.parse_frequency("0.01")


def test_onoff_dry_run_accepts_two_hz():
    result = subprocess.run(
        [sys.executable, "onoff.py", "--dry-run", "--frequency", "2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script,args",
    [
        ("single_point_hover.py", ["--dry-run"]),
        ("single_point_vortex.py", ["--dry-run"]),
        ("transducer_phase_steer.py", ["--dry-run"]),
        ("transducer_phase_steer_2d.py", ["--dry-run"]),
    ],
)
def test_headless_prompt_fails_cleanly(script, args):
    result = subprocess.run(
        [sys.executable, script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Interactive input is unavailable" in result.stderr
    assert "Traceback" not in result.stderr
