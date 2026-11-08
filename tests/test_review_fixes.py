"""Regression tests for issues found in the post-cleanup review."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import onoff
import transducer_onoff_demo
import transducer_onoff_steer
import trajectory

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


def test_vortex_uses_array_center_for_off_center_targets():
    import acoustics

    result = acoustics.vortex_phases((0.05, 0.05, 0.05), 1)
    np.testing.assert_allclose(result.center, acoustics.array_center())

    radii, angles = acoustics.polar_coordinates(center=result.center)
    np.testing.assert_allclose(result.radii, radii)
    np.testing.assert_allclose(result.angles, angles)


def test_stream_duration_does_not_oversleep(fake_clock, fake_serial_factory):
    connection = fake_serial_factory()
    sent = trajectory.stream_timed_frames(
        lambda index: b"frame",
        interval_s=10.0,
        connection=connection,
        duration_s=0.1,
        max_frames=2,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )
    assert sent == 1
    assert fake_clock.now == pytest.approx(0.1, abs=1e-6)


def test_toggle_duration_does_not_oversleep(fake_serial_factory, fake_clock):
    sent = trajectory.run_toggle(
        port="COM9",
        on_frame=b"ON",
        off_frame=b"OFF",
        frequency_hz=0.1,
        duration_s=0.1,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert sent == 1
    assert fake_clock.now == pytest.approx(0.4, abs=1e-6)  # 0.3 settle + 0.1


def test_bit_pattern_duration_does_not_oversleep(fake_serial_factory, fake_clock):
    sent = trajectory.run_bit_pattern(
        port="COM9",
        pattern=[1],
        bit_seconds=10.0,
        duration_s=0.1,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert sent == 1
    assert fake_clock.now == pytest.approx(0.4, abs=1e-6)  # 0.3 settle + 0.1


def test_rotation_duration_does_not_oversleep(fake_serial_factory, fake_clock):
    summary = trajectory.run_rotation(
        30.0,
        mode="two",
        rotation_period_s=10.0,
        port="COM9",
        duration_s=0.1,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert summary["frames_sent"] == 1
    assert fake_clock.now == pytest.approx(0.4, abs=1e-6)  # 0.3 settle + 0.1


@pytest.mark.parametrize("script", ["onoff.py", "transducer_onoff_demo.py"])
def test_invalid_frequency_fails_cleanly(script):
    result = subprocess.run(
        [sys.executable, script, "--dry-run", "--frequency", "30000"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Frequency must be between" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_rotation_timing_fails_cleanly():
    result = subprocess.run(
        [
            sys.executable,
            "transducer_phase_steer_2d.py",
            "--dry-run",
            "--rotate",
            "--beta",
            "30",
            "--mode",
            "two",
            "--rotation-period",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Rotation period" in result.stderr
    assert "Traceback" not in result.stderr


def test_rotation_rejects_both_poles():
    for beta in (0.0, 180.0):
        with pytest.raises(ValueError):
            trajectory.run_rotation(beta, dry_run=True)


def test_cli_rotation_rejects_beta_180():
    result = subprocess.run(
        [
            sys.executable,
            "transducer_phase_steer_2d.py",
            "--dry-run",
            "--rotate",
            "--beta",
            "180",
            "--mode",
            "two",
            "--rotation-period",
            "0.2",
            "--duration",
            "0.1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 2
    assert "beta=0 or beta=180" in result.stderr


def test_readme_tables_have_consistent_column_counts():
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    expected = None
    for line in lines:
        if line.startswith("|"):
            columns = len(re.findall(r"(?<!\\)\|", line))
            if expected is None:
                expected = columns
            assert columns == expected, f"ragged table row: {line}"
        else:
            expected = None
