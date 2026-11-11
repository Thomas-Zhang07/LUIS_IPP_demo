"""End-to-end CLI checks: flags preserved, dry runs need no hardware."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EXPECTED_FLAGS = {
    "single_point_hover.py": {
        "--detect", "--x", "--y", "--z", "--port", "--dry-run", "--baud",
        "--frequency", "--sound-speed",
    },
    "single_point_vortex.py": {
        "--detect", "--x", "--y", "--z", "--charge", "--port", "--dry-run",
        "--baud", "--frequency", "--sound-speed",
    },
    "single_point_line.py": {
        "--detect", "--x", "--y", "--z", "--port", "--period", "--points",
        "--dry-run", "--baud", "--frequency", "--sound-speed",
    },
    "single_point_circle.py": {
        "--detect", "--x", "--y", "--z", "--port", "--period", "--points",
        "--dry-run", "--baud", "--frequency", "--sound-speed",
    },
    "transducer_all_on.py": {"--port", "--dry-run", "--list-ports"},
    "transducer_onoff_demo.py": {
        "--port", "--frequency", "--duration", "--dry-run", "--list-ports",
    },
    "onoff.py": {
        "--port", "--frequency", "--duration", "--dry-run", "--list-ports",
    },
    "bit_pattern_onoff.py": {
        "--port", "--pattern", "--bit-seconds", "--duration", "--dry-run",
        "--list-ports",
    },
    "transducer_onoff_steer.py": {
        "--port", "--frequency", "--duration", "--angle", "--dry-run",
        "--list-ports", "--carrier-frequency", "--sound-speed",
    },
    "transducer_phase_steer.py": {
        "--port", "--list-ports", "--angle", "--dry-run", "--frequency",
        "--sound-speed",
    },
    "transducer_phase_steer_2d.py": {
        "--port", "--list-ports", "--angle", "--legacy-angle", "--alpha",
        "--beta", "--dry-run", "--rotate", "--mode", "--rotation-period",
        "--update-interval", "--duration", "--plot-phase-rotation",
        "--plot-array-factor", "--frequency", "--sound-speed",
    },
    "mode_control.py": {"--port", "--mode", "--frame", "--dry-run", "--list-ports"},
    "directional_speaker.py": {
        "--alpha", "--beta", "--frequency", "--sound-speed",
        "--density-window", "--keep-fraction", "--seed", "--dither-bins",
        "--optimize-sweeps", "--null", "--null-weight", "--calibration",
        "--samples", "--compare", "--plot", "--port", "--baud", "--dry-run",
        "--list-ports",
    },
}

DRY_RUN_CASES = [
    ("single_point_hover.py", ["--dry-run", "--x", "0.075", "--y", "0.075", "--z", "0.05"]),
    ("single_point_vortex.py", ["--dry-run", "--x", "0.075", "--y", "0.075", "--z", "0.05", "--charge", "1"]),
    ("single_point_line.py", ["--dry-run", "--x", "0.075", "--y", "0.075", "--z", "0.05"]),
    ("single_point_circle.py", ["--dry-run", "--x", "0.075", "--y", "0.075", "--z", "0.05"]),
    ("transducer_all_on.py", ["--dry-run"]),
    ("onoff.py", ["--dry-run", "--frequency", "100"]),
    ("bit_pattern_onoff.py", ["--dry-run", "--pattern", "10110001", "--bit-seconds", "0.1", "--duration", "0.01"]),
    ("transducer_onoff_demo.py", ["--dry-run", "--frequency", "100"]),
    ("transducer_onoff_steer.py", ["--dry-run", "--angle", "45", "--frequency", "100"]),
    ("transducer_phase_steer.py", ["--dry-run", "--angle", "45"]),
    ("transducer_phase_steer_2d.py", ["--dry-run", "--alpha", "0", "--beta", "30"]),
    (
        "transducer_phase_steer_2d.py",
        ["--dry-run", "--rotate", "--beta", "30", "--mode", "two",
         "--rotation-period", "0.2", "--update-interval", "0.01",
         "--duration", "0.1"],
    ),
    ("mode_control.py", ["--dry-run"]),
    (
        "directional_speaker.py",
        ["--dry-run", "--alpha", "0", "--beta", "20", "--samples", "41"],
    ),
    (
        "directional_speaker.py",
        [
            "--dry-run", "--alpha", "0", "--beta", "20",
            "--density-window", "hann", "--keep-fraction", "0.7",
            "--dither-bins", "1", "--optimize-sweeps", "1",
            "--samples", "41",
        ],
    ),
]


def _run(*args, timeout=60):
    return subprocess.run(
        [PYTHON, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("script", sorted(EXPECTED_FLAGS))
def test_help_exits_zero_and_preserves_flags(script):
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    for flag in EXPECTED_FLAGS[script]:
        assert flag in result.stdout, f"{script} lost {flag}"


@pytest.mark.parametrize("script,args", DRY_RUN_CASES)
def test_dry_run_works_without_hardware_or_prompts(script, args):
    result = _run(script, *args)
    assert result.returncode == 0, result.stderr


def test_circle_help_states_two_second_default():
    result = _run("single_point_circle.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "default 2.0" in result.stdout


def test_directional_speaker_compare_prints_table():
    result = _run("directional_speaker.py", "--compare", "--samples", "41")
    assert result.returncode == 0, result.stderr
    assert "Strategy" in result.stdout
    assert "uniform + optimize" in result.stdout
    assert "hann 70%" in result.stdout


def test_python_O_still_raises_validation_errors():
    result = _run(
        "-O",
        "-c",
        "import protocol; protocol.build_frame([0] * 255)",
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
