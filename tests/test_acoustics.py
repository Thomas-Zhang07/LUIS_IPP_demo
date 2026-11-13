"""Tests for shared acoustic math."""

import math

import numpy as np
import pytest

import acoustics
from emitters import EMITTERS_POS, N_EMITTERS

TARGET = (0.075, 0.075, 0.05)


def test_default_wavelength_is_343_over_40000():
    assert acoustics.acoustic_wavelength() == pytest.approx(343.0 / 40000.0)
    assert acoustics.DEFAULT_FREQUENCY_HZ == 40000.0
    assert acoustics.DEFAULT_SOUND_SPEED_MPS == 343.0


def test_focus_phases_match_legacy_formula_at_same_wavelength():
    wavelength = acoustics.acoustic_wavelength()
    distances = np.linalg.norm(EMITTERS_POS - np.array(TARGET), axis=1)
    legacy = 2.0 * np.pi * (np.ceil(distances / wavelength) - distances / wavelength)

    new = acoustics.focus_phases(TARGET)
    np.testing.assert_allclose(
        np.mod(new, acoustics.TWO_PI),
        np.mod(legacy, acoustics.TWO_PI),
        atol=1e-12,
    )


def test_focus_phase_bins_are_in_range():
    bins = acoustics.focus_phase_bins(TARGET)
    assert bins.shape == (N_EMITTERS,)
    assert bins.dtype == np.uint8
    assert bins.max() <= acoustics.PHASE_BINS - 1


def test_focus_rejects_bad_target():
    with pytest.raises(ValueError):
        acoustics.focus_phases((0.1, 0.2))
    with pytest.raises(ValueError):
        acoustics.focus_phases((float("nan"), 0.0, 0.0))


def test_quantizer_snaps_floating_point_boundaries():
    # A phase one ULP below an exact bin boundary must land in that bin.
    phase = np.nextafter(acoustics.TWO_PI * 8 / acoustics.PHASE_BINS, 0.0)
    assert acoustics.quantize_phase_bins([phase])[0] == 8

    # Clearly inside a bin it must not snap.
    phase = acoustics.TWO_PI * 7.75 / acoustics.PHASE_BINS
    assert acoustics.quantize_phase_bins([phase])[0] == 7


def test_quantizer_wraps_full_circle_to_zero():
    bins = acoustics.quantize_phase_bins([0.0, acoustics.TWO_PI, 4.0 * acoustics.TWO_PI])
    assert bins.tolist() == [0, 0, 0]


def test_quantizer_snaps_near_2pi_to_bin_zero_not_last_bin():
    # One ULP below 2*pi snaps to the 2*pi boundary, which wraps to bin 0.
    phase = np.nextafter(acoustics.TWO_PI, 0.0)
    assert acoustics.quantize_phase_bins([phase])[0] == 0

    # Clearly below the boundary must remain the last bin.
    phase = acoustics.TWO_PI * 15.5 / acoustics.PHASE_BINS
    assert acoustics.quantize_phase_bins([phase])[0] == 15


def test_quantizer_rejects_nonfinite_phases():
    with pytest.raises(ValueError):
        acoustics.quantize_phase_bins([0.0, float("inf")])


def test_quantizer_validates_bin_count():
    for bad in (0, -1, True, 1.5, "16"):
        with pytest.raises(ValueError):
            acoustics.quantize_phase_bins([0.0], num_bins=bad)


def test_vortex_total_is_focus_plus_charge_times_angle():
    result = acoustics.vortex_phases(TARGET, topological_charge=2)
    np.testing.assert_allclose(
        result.total,
        np.mod(result.focus + 2.0 * result.angles, acoustics.TWO_PI),
    )
    np.testing.assert_allclose(result.center, acoustics.array_center())


def test_vortex_rejects_noninteger_charge():
    for bad in (1.5, True, "1", None):
        with pytest.raises(ValueError):
            acoustics.vortex_phases(TARGET, topological_charge=bad)


def test_polar_coordinates_are_centered_and_symmetric():
    radii, angles = acoustics.polar_coordinates()
    center = acoustics.array_center()
    assert radii.shape == (N_EMITTERS,)
    assert angles.shape == (N_EMITTERS,)
    np.testing.assert_allclose(
        np.mean(EMITTERS_POS[:, :2] - center[:2], axis=0), [0.0, 0.0], atol=1e-12
    )
    assert radii.min() == pytest.approx(math.hypot(0.00525, 0.00525))
    assert radii.max() == pytest.approx(math.hypot(0.07875, 0.07875))


def test_legacy_angle_to_spherical_boundaries():
    assert acoustics.legacy_angle_to_spherical(0.0) == (0.0, 90.0)
    assert acoustics.legacy_angle_to_spherical(45.0) == (0.0, 45.0)
    assert acoustics.legacy_angle_to_spherical(90.0) == (0.0, 0.0)
    assert acoustics.legacy_angle_to_spherical(135.0) == (180.0, 45.0)
    assert acoustics.legacy_angle_to_spherical(180.0) == (180.0, 90.0)
    with pytest.raises(ValueError):
        acoustics.legacy_angle_to_spherical(-1.0)


def test_direction_conventions():
    np.testing.assert_allclose(acoustics.direction_from_angles(0, 90), [1, 0, 0])
    np.testing.assert_allclose(acoustics.direction_from_angles(90, 90), [0, 1, 0])
    np.testing.assert_allclose(acoustics.direction_from_angles(180, 90), [-1, 0, 0])
    np.testing.assert_allclose(acoustics.direction_from_angles(0, 0), [0, 0, 1])
    np.testing.assert_allclose(acoustics.direction_from_angles(0, 180), [0, 0, -1])


def test_line_steering_equals_legacy_2d_steering():
    for angle in (0.0, 30.0, 45.0, 90.0, 135.0, 180.0):
        line = acoustics.line_steering_phase_bins(angle)
        azimuth, off_axis = acoustics.legacy_angle_to_spherical(angle)
        two_d = acoustics.steering_phase_bins(azimuth, off_axis)
        np.testing.assert_array_equal(line, two_d)


def test_broadside_steering_is_zero_phase():
    bins = acoustics.steering_phase_bins(37.0, 0.0)
    assert np.all(bins == 0)


def test_acoustic_parameters_reject_nonpositive_values():
    with pytest.raises(ValueError):
        acoustics.acoustic_wavelength(0.0)
    with pytest.raises(ValueError):
        acoustics.acoustic_wavelength(40000.0, 0.0)
    with pytest.raises(ValueError):
        acoustics.acoustic_wavelength(float("nan"))
    with pytest.raises(ValueError):
        acoustics.validate_acoustic_params(True, 343.0)


def test_emitter_position_validation():
    with pytest.raises(ValueError):
        acoustics.validate_emitter_positions(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        bad = EMITTERS_POS.copy()
        bad[0, 0] = float("nan")
        acoustics.validate_emitter_positions(bad)


def test_steering_rejects_invalid_reference_center():
    with pytest.raises(ValueError):
        acoustics.steering_phase_bins(0.0, 30.0, reference_center=(0.0, 0.0))
