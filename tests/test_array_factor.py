"""Tests for theoretical array-factor and sidelobe analysis."""

import ast
from pathlib import Path

import numpy as np
import pytest

import array_factor

ROOT = Path(__file__).resolve().parents[1]


def test_physical_phase_grid_orientation():
    values = (np.arange(256) % 16).astype(np.uint8)
    x_coordinates, y_coordinates, grid = array_factor.physical_phase_grid(values)

    assert grid.shape == (16, 16)
    assert len(x_coordinates) == len(y_coordinates) == 16
    # Logical index is row * 16 + column; grid rows are physical y.
    assert grid[0, 0] == values[0]
    assert grid[3, 5] == values[3 * 16 + 5]
    assert grid[15, 15] == values[255]
    np.testing.assert_allclose(x_coordinates, np.arange(16) * 0.0105)
    np.testing.assert_allclose(y_coordinates, np.arange(16) * 0.0105)


def test_print_physical_phase_map():
    messages = []
    array_factor.print_physical_phase_map(
        np.zeros(256, dtype=int), output=messages.append
    )
    text = "\n".join(messages)
    assert "Physical x-y phase map" in text
    assert "y=" in text


def test_array_factor_shape_and_normalized_peak():
    azimuth, off_axis, levels = array_factor.compute_array_factor_pattern(
        np.zeros(256, dtype=int), azimuth_samples=91, off_axis_samples=91
    )
    assert azimuth.shape == off_axis.shape == levels.shape == (91, 91)
    assert levels.max() == pytest.approx(0.0)
    peak = np.unravel_index(np.argmax(levels), levels.shape)
    assert off_axis[peak] == pytest.approx(0.0)


def test_array_factor_rejects_invalid_bins():
    bad = np.zeros(256, dtype=int)
    bad[0] = 16
    with pytest.raises(ValueError):
        array_factor.compute_array_factor_pattern(bad)
    with pytest.raises(ValueError):
        array_factor.compute_array_factor_pattern(np.zeros(255, dtype=int))
    with pytest.raises(ValueError):
        array_factor.compute_array_factor_pattern(
            np.zeros(256, dtype=int), azimuth_samples=1
        )


def test_evaluate_peak_sidelobe_broadside_returns_expected_keys():
    result = array_factor.evaluate_peak_sidelobe(
        0.0, 0.0, np.zeros(256, dtype=int), direction_cosine_samples=51
    )
    for key in (
        "sll_db",
        "main_lobe_peak",
        "main_lobe_peak_azimuth_deg",
        "main_lobe_peak_beta_deg",
        "sidelobe_peak",
        "sidelobe_azimuth_deg",
        "sidelobe_beta_deg",
        "likely_grating_lobe",
        "main_lobe_exclusion_deg",
        "main_lobe_half_width_ux",
        "main_lobe_half_width_uy",
    ):
        assert key in result
    assert result["main_lobe_peak"] > 0
    assert result["sll_db"] <= 0.0


def test_sidelobe_rejects_back_hemisphere():
    with pytest.raises(ValueError):
        array_factor.evaluate_peak_sidelobe(
            0.0, 120.0, np.zeros(256, dtype=int)
        )


def test_array_factor_rejects_invalid_reference_center():
    with pytest.raises(ValueError):
        array_factor.compute_array_factor_pattern(
            np.zeros(256, dtype=int), reference_center=(0.0, 0.0)
        )


def test_dead_beta_functions_are_removed():
    for name in (
        "rank_beta_candidates",
        "optimize_beta",
        "plot_beta_optimization",
    ):
        assert not hasattr(array_factor, name)

    tree = ast.parse(
        (ROOT / "transducer_phase_steer_2d.py").read_text(encoding="utf-8")
    )
    defined = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "rank_beta_candidates",
        "optimize_beta",
        "plot_beta_optimization",
    ):
        assert name not in defined
