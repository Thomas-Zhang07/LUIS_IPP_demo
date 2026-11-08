"""Tests for the mode-control command script."""

import subprocess
import sys
from pathlib import Path

import pytest

import mode_control

ROOT = Path(__file__).resolve().parents[1]


def test_mode_payload_layout():
    payload = mode_control.build_mode_payload("c4")
    assert len(payload) == 259
    assert payload[0] == 0xC9
    assert payload[1] == 0xFE
    assert payload[2:-1] == bytes([0x10]) * 256
    assert payload[-1] == 0xFD


def test_all_modes_map_to_expected_bytes():
    assert mode_control.MODE_COMMANDS == {
        "normal": 0xC8,
        "c4": 0xC9,
        "little-star": 0xCA,
        "200hz": 0xCB,
    }


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        mode_control.build_mode_payload("nope")


def test_import_has_no_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import mode_control"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_dry_run_needs_no_port():
    result = subprocess.run(
        [sys.executable, "mode_control.py", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "0xC9" in result.stdout
    assert "[DRY RUN]" in result.stdout
