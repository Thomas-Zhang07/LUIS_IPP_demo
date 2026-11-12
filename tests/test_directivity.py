"""Tests for host-side directivity shaping and beam analysis."""

import json

import numpy as np
import pytest

import acoustics
import directivity
from emitters import EMITTERS_POS, N_EMITTERS, SPACING
from protocol import OFF_VALUE


def _steering_vector(direction, frequency_hz=40000.0, sound_speed=343.0):
    positions = np.asarray(EMITTERS_POS, dtype=float)
    centered = positions - positions.mean(axis=0)
    wave_number = acoustics.TWO_PI * frequency_hz / sound_speed
    return np.exp(1j * wave_number * (centered @ np.asarray(direction, dtype=float)))


def test_phase_bin_validation_accepts_on_and_off():
    bins = np.arange(N_EMITTERS) % 16
    assert directivity.validate_phase_bins(bins).shape == (N_EMITTERS,)
    off = np.zeros(N_EMITTERS, dtype=int)
    off[0] = OFF_VALUE
    assert directivity.validate_phase_bins(off)[0] == OFF_VALUE
    with pytest.raises(ValueError):
        directivity.validate_phase_bins(off, allow_off=False)
    with pytest.raises(ValueError):
        directivity.validate_phase_bins(np.zeros(N_EMITTERS - 1, dtype=int))
    with pytest.raises(ValueError):
        directivity.validate_phase_bins(np.full(N_EMITTERS, 17))


def test_active_mask_validation():
    mask = np.ones(N_EMITTERS, dtype=bool)
    assert directivity.validate_active_mask(mask).all()
    with pytest.raises(ValueError):
        directivity.validate_active_mask(np.zeros(N_EMITTERS, dtype=bool))
    with pytest.raises(ValueError):
        directivity.validate_active_mask(np.ones(N_EMITTERS, dtype=int))


def test_aperture_weights_shapes_and_taper():
    flat = directivity.aperture_weights("none")
    hann = directivity.aperture_weights("hann")
    gaussian = directivity.aperture_weights("gaussian")
    for weights in (flat, hann, gaussian):
        assert weights.shape == (N_EMITTERS,)
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)
    center_index = 7 * 16 + 7
    corner_index = 0
    assert hann[center_index] > hann[corner_index]
    assert gaussian[center_index] > gaussian[corner_index]
    assert gaussian[center_index] > 0.9
    with pytest.raises(ValueError):
        directivity.aperture_weights("triangle")


def test_amplitude_code_mapping_matches_firmware():
    assert directivity.duty_width_from_code(0) == 8
    assert directivity.duty_width_from_code(15) == 8
    assert [directivity.duty_width_from_code(c) for c in range(1, 15)] == [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
    ]
    assert directivity.amplitude_from_code(15) == pytest.approx(1.0)
    assert directivity.amplitude_from_code(2) == pytest.approx(
        np.sin(np.pi / 16)
    )
    with pytest.raises(ValueError):
        directivity.duty_width_from_code(16)


def test_amplitude_code_for_weight_round_trip():
    assert directivity.amplitude_code_for_weight(0.0) == OFF_VALUE
    assert directivity.amplitude_code_for_weight(-0.1) == OFF_VALUE
    code = directivity.amplitude_code_for_weight(1.0)
    assert directivity.duty_width_from_code(code) == 8
    # Every positive weight gets a code whose width is the nearest one.
    for weight in (0.2, 0.4, 0.6, 0.8):
        code = directivity.amplitude_code_for_weight(weight)
        assert 1 <= code <= 15
        assert directivity.amplitude_from_code(code) > 0.0
    with pytest.raises(ValueError):
        directivity.amplitude_code_for_weight(1.5)


def test_amplitude_codes_from_weights_validates_and_maps_zero_to_off():
    weights = directivity.aperture_weights("hann")
    codes = directivity.amplitude_codes_from_weights(weights)
    assert codes.shape == (N_EMITTERS,)
    assert np.all((codes >= 0) & (codes <= OFF_VALUE))
    assert np.any(codes == OFF_VALUE)
    with pytest.raises(ValueError):
        directivity.amplitude_codes_from_weights(np.zeros(N_EMITTERS - 1))
    bad = np.zeros(N_EMITTERS)
    bad[0] = 1.5
    with pytest.raises(ValueError):
        directivity.amplitude_codes_from_weights(bad)


def test_metrics_with_amplitude_taper_lowers_sidelobes_but_not_grating():
    analyzer = directivity.BeamAnalyzer(0.0, 0.0, samples=61)
    bins = np.zeros(N_EMITTERS, dtype=int)
    codes = directivity.amplitude_codes_from_weights(
        directivity.aperture_weights("hann")
    )
    flat = analyzer.metrics(bins)
    tapered = analyzer.metrics(bins, amplitude_codes=codes)
    assert tapered["near_sidelobe_db"] < flat["near_sidelobe_db"] - 2.0
    assert tapered["main_lobe_gain_db"] < flat["main_lobe_gain_db"] - 2.0
    # Per-element weights scale the main and grating lobes equally.
    assert tapered["grating_lobe_db"] == pytest.approx(
        flat["grating_lobe_db"], abs=0.5
    )


def test_optimizer_accepts_fixed_amplitude_codes():
    analyzer = directivity.BeamAnalyzer(0.0, 0.0, samples=41)
    baseline = acoustics.steering_phase_bins(0.0, 0.0)
    codes = directivity.amplitude_codes_from_weights(
        directivity.aperture_weights("gaussian")
    )
    before = analyzer.metrics(baseline, amplitude_codes=codes)["near_sidelobe_db"]
    optimized, info = analyzer.optimize(
        baseline, max_sweeps=2, seed=0, amplitude_codes=codes
    )
    after = analyzer.metrics(optimized, amplitude_codes=codes)["near_sidelobe_db"]
    assert info["sweeps_run"] >= 1
    assert after <= before + 1e-9
    assert np.all(optimized <= 15)


def test_thinned_mask_is_deterministic_and_exact():
    first = directivity.thinned_mask("hann", 0.7, seed=5)
    second = directivity.thinned_mask("hann", 0.7, seed=5)
    other = directivity.thinned_mask("hann", 0.7, seed=6)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert int(first.sum()) == round(0.7 * N_EMITTERS)
    full = directivity.thinned_mask("hann", 1.0)
    assert full.all()
    assert directivity.thinned_mask("none", 0.5, seed=0).sum() == 128
    with pytest.raises(ValueError):
        directivity.thinned_mask("none", 0.0)
    with pytest.raises(ValueError):
        directivity.thinned_mask("none", 1.5)


def test_dither_phase_bins_is_bounded_and_keeps_off():
    bins = acoustics.steering_phase_bins(0.0, 20.0)
    bins[3] = OFF_VALUE
    dithered = directivity.dither_phase_bins(bins, 2, seed=7)
    repeated = directivity.dither_phase_bins(bins, 2, seed=7)
    assert np.array_equal(dithered, repeated)
    assert dithered[3] == OFF_VALUE
    active = bins != OFF_VALUE
    delta = (dithered[active].astype(int) - bins[active].astype(int)) % 16
    assert np.all((delta <= 2) | (delta >= 14))
    unchanged = directivity.dither_phase_bins(bins, 0)
    assert np.array_equal(unchanged, bins)
    with pytest.raises(ValueError):
        directivity.dither_phase_bins(bins, -1)
    with pytest.raises(ValueError):
        directivity.dither_phase_bins(bins, 9)


def test_apply_calibration_round_trip():
    bins = acoustics.steering_phase_bins(10.0, 15.0)
    offsets = (np.arange(N_EMITTERS) * 7) % 16
    calibrated = directivity.apply_calibration_bins(bins, offsets)
    restored = directivity.apply_calibration_bins(
        calibrated, (-offsets) % 16
    )
    assert np.array_equal(restored, bins)
    with pytest.raises(ValueError):
        directivity.apply_calibration_bins(bins, np.zeros(N_EMITTERS - 1, dtype=int))
    bad = np.zeros(N_EMITTERS, dtype=int)
    bad[0] = OFF_VALUE
    with pytest.raises(ValueError):
        directivity.apply_calibration_bins(bins, bad)


def test_load_calibration_file(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(list(range(16)) * 16), encoding="utf-8")
    assert directivity.load_calibration_file(plain).shape == (N_EMITTERS,)

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"phase_offsets_bins": list(range(16)) * 16}),
        encoding="utf-8",
    )
    assert directivity.load_calibration_file(wrapped).shape == (N_EMITTERS,)

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([0, 1, 2]), encoding="utf-8")
    with pytest.raises(ValueError):
        directivity.load_calibration_file(broken)
    with pytest.raises(ValueError):
        directivity.load_calibration_file(tmp_path / "missing.json")


def test_grating_lobe_threshold_and_directions():
    assert directivity.spatial_aliasing_frequency_hz() == pytest.approx(34300.0)
    lobes = directivity.grating_lobe_directions(0.0, 0.0)
    assert len(lobes) == 4
    step = 343.0 / 40000.0 / SPACING
    for lobe in lobes:
        assert abs(abs(lobe.ux) + abs(lobe.uy) - step) < 1e-9
        assert lobe.uz == pytest.approx(np.sqrt(1.0 - step**2))
    # Below c/spacing there is no broadside grating lobe.
    assert directivity.grating_lobe_directions(
        0.0, 0.0, frequency_hz=30000.0
    ) == ()


def test_grating_lobe_level_is_invariant_under_any_weighting():
    """The core physics limit: per-element weights cannot change a grating lobe."""
    target = acoustics.direction_from_angles(0.0, 20.0)
    target_vector = _steering_vector(target)
    grid = directivity.grating_lobe_directions(0.0, 20.0)

    rng = np.random.default_rng(3)
    patterns = [
        np.exp(1j * acoustics.steering_phase_bins(0.0, 20.0) * acoustics.TWO_PI / 16),
        np.exp(1j * rng.integers(0, 16, N_EMITTERS) * acoustics.TWO_PI / 16),
    ]
    thinned = patterns[0].copy()
    thinned[rng.random(N_EMITTERS) < 0.5] = 0.0
    patterns.append(thinned)

    for weights in patterns:
        main = abs(target_vector @ weights)
        for lobe in grid:
            grating_vector = _steering_vector((lobe.ux, lobe.uy, lobe.uz))
            assert abs(grating_vector @ weights) == pytest.approx(main, rel=1e-9)


def test_metrics_baseline_broadside():
    analyzer = directivity.BeamAnalyzer(0.0, 0.0, samples=61)
    metrics = analyzer.metrics(np.zeros(N_EMITTERS, dtype=int))
    assert metrics["main_lobe_amplitude"] == pytest.approx(256.0)
    assert metrics["main_lobe_gain_db"] == pytest.approx(0.0)
    assert metrics["active_elements"] == 256
    # The grating lobe dominates the unrestricted sidelobe metric.
    assert metrics["peak_sidelobe_db"] > -1.0
    assert metrics["grating_lobe_db"] > -1.0
    assert metrics["near_sidelobe_db"] < -10.0


def test_analyzer_rejects_back_hemisphere_and_bad_grid():
    with pytest.raises(ValueError):
        directivity.BeamAnalyzer(0.0, 120.0)
    with pytest.raises(ValueError):
        directivity.BeamAnalyzer(0.0, 0.0, samples=4)


def test_optimizer_improves_near_sidelobe_and_respects_mask():
    analyzer = directivity.BeamAnalyzer(0.0, 0.0, samples=41)
    baseline = acoustics.steering_phase_bins(0.0, 0.0)
    before = analyzer.metrics(baseline)["near_sidelobe_db"]
    optimized, info = analyzer.optimize(baseline, max_sweeps=3, seed=0)
    after = analyzer.metrics(optimized)["near_sidelobe_db"]
    assert after < before - 1.0
    assert info["sweeps_run"] >= 1
    assert np.all(optimized <= 15)

    mask = directivity.thinned_mask("hann", 0.7, seed=0)
    thinned = baseline.copy()
    thinned[~mask] = OFF_VALUE
    shaped, info = analyzer.optimize(thinned, mask, max_sweeps=2, seed=0)
    assert np.all(shaped[~mask] == OFF_VALUE)
    assert np.all(shaped[mask] <= 15)
    assert info["changed_elements"] >= 0


def test_optimizer_reports_null_directions():
    analyzer = directivity.BeamAnalyzer(0.0, 20.0, samples=61)
    baseline = acoustics.steering_phase_bins(0.0, 20.0)
    _, info = analyzer.optimize(
        baseline,
        max_sweeps=1,
        seed=0,
        null_directions=[(180.0, 40.0)],
        null_weight=2.0,
    )
    assert len(info["null_directions"]) == 1
    note = info["null_directions"][0]
    assert note["azimuth_deg"] == 180.0
    assert note["beta_deg"] == 40.0
    assert "in_main_lobe" in note and "in_grating_lobe" in note
