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


class _JitterClock:
    """Clock whose sleeps always overshoot by a fixed amount."""

    def __init__(self, jitter=0.1):
        self.now = 0.0
        self.jitter = jitter

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds) + self.jitter


class _RecordingConnection:
    def __init__(self, clock, starts):
        self.clock = clock
        self.starts = starts
        self.dtr = None
        self.rts = None

    def write(self, data):
        self.starts.append(self.clock.now)
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


def test_rotation_deadlines_do_not_accumulate_jitter():
    import trajectory

    clock = _JitterClock()
    starts = []

    def factory(*args, **kwargs):
        return _RecordingConnection(clock, starts)

    summary = trajectory.run_rotation(
        30.0,
        mode="two",
        rotation_period_s=10.0,
        port="COM9",
        max_updates=4,
        serial_factory=factory,
        clock=clock,
        sleep=clock.sleep,
        output_fn=lambda *_: None,
    )

    assert summary["frames_sent"] == 4
    deltas = [later - earlier for earlier, later in zip(starts, starts[1:])]
    # The first interval includes the settle overshoot; later intervals stay
    # on the 5 s epoch grid instead of growing by the 0.1 s jitter each time.
    assert deltas[0] == pytest.approx(5.1, abs=0.06)
    assert deltas[1] == pytest.approx(5.0, abs=0.06)
    assert deltas[2] == pytest.approx(5.0, abs=0.06)
