"""Host-side directivity shaping for the directional-speaker demo.

The hardware frame gives each transducer one of 16 phase bins (0..15) or OFF
(``0x10``).  With the duty-weighting firmware the high nibble of each payload
byte also selects one of eight pulse widths, giving coarse per-element
amplitude control. Within those limits this module provides the host-side
algorithms that change the far-field pattern:

* :func:`thinned_mask` -- binary amplitude shaping by keeping a windowed,
  seeded subset of the elements (uniform random, Hann or Gaussian density).
* :func:`aperture_weights` / :func:`amplitude_codes_from_weights` -- continuous
  taper weights (Hann or Gaussian) encoded as firmware duty codes.
* :func:`dither_phase_bins` -- deterministic per-element phase dithering.
* :meth:`BeamAnalyzer.optimize` -- coordinate descent over the 16 phase bins,
  minimizing the peak sidelobe outside the main lobe and the predicted
  grating-lobe regions, with optional null constraints.
* :func:`apply_calibration_bins` / :func:`load_calibration_file` -- per-channel
  phase-error pre-compensation.

Hard physical limit
-------------------
For a periodic lattice with identical elements, the array factor at a grating
lobe equals the array factor at the main lobe *for any per-element phase or
amplitude (including ON/OFF) pattern*, because the grating condition removes
the phase differences and both lobes scale with the same weighted sum.
No host algorithm can therefore lower the grating lobes of this 10 mm-pitch,
40 kHz array (they sit near +/-59 degrees from broadside). Only the element
directivity, a carrier below ``sound_speed / spacing`` (34.3 kHz for 10 mm), or
spacing <= lambda/2 can. The algorithms here improve the non-grating sidelobe
floor and can place user-selected nulls; they cannot narrow the ~3 degree main
lobe either, which is set by the aperture.

All angles are in degrees. Positions are meters and use the same coordinate
system as :mod:`emitters`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

import acoustics
from acoustics import DEFAULT_FREQUENCY_HZ, DEFAULT_SOUND_SPEED_MPS, TWO_PI
from emitters import EMITTERS_POS, N_EMITTERS, SPACING
from protocol import MAX_AMPLITUDE_CODE, OFF_VALUE, PHASE_BINS

DEFAULT_SAMPLES = 121
DEFAULT_MAIN_LOBE_SAFETY_FACTOR = 1.5
DEFAULT_GAUSSIAN_SIGMA = 0.35
APERTURE_WINDOWS = ("none", "hann", "gaussian")
_TINY = 1e-12


class GratingLobe(NamedTuple):
    """One predicted grating-lobe direction and its lattice orders."""

    ux: float
    uy: float
    uz: float
    order_x: int
    order_y: int


def validate_phase_bins(values, *, allow_off: bool = True, label: str = "phase bins"):
    """Return a validated 256-element integer array of phase bins."""
    array = np.asarray(values)
    if array.shape != (N_EMITTERS,):
        raise ValueError(f"{label} must contain exactly {N_EMITTERS} values.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must be integers.")
    upper = OFF_VALUE if allow_off else PHASE_BINS - 1
    if np.any((array < 0) | (array > upper)):
        raise ValueError(f"{label} must be between 0 and {upper}.")
    return array.astype(np.uint8, copy=False)


def validate_amplitude_codes(values, *, allow_off: bool = True) -> np.ndarray:
    """Return a validated 256-element integer array of amplitude codes.

    Codes ``0..15`` select a pulse width; :data:`OFF_VALUE` mutes a channel
    (used for zero weights).  ``allow_off=False`` rejects the mute code.
    """
    array = np.asarray(values)
    if array.shape != (N_EMITTERS,):
        raise ValueError(f"Amplitude codes must contain exactly {N_EMITTERS} values.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Amplitude codes must be integers.")
    upper = OFF_VALUE if allow_off else MAX_AMPLITUDE_CODE
    if np.any((array < 0) | (array > upper)):
        raise ValueError(f"Amplitude codes must be between 0 and {upper}.")
    return array.astype(np.int16, copy=False)


def validate_active_mask(active_mask):
    """Return a validated boolean mask with at least one active element."""
    mask = np.asarray(active_mask)
    if mask.shape != (N_EMITTERS,) or mask.dtype != bool:
        raise ValueError(
            f"Active mask must be a boolean array of {N_EMITTERS} values."
        )
    if not mask.any():
        raise ValueError("At least one emitter must be active.")
    return mask


def spatial_aliasing_frequency_hz(
    *,
    spacing: float = SPACING,
    sound_speed: float = DEFAULT_SOUND_SPEED_MPS,
) -> float:
    """Carrier above which the first grating order is visible near broadside.

    ``sound_speed / spacing`` is about 34.3 kHz for the 10 mm grid. Below it,
    a broadside beam has no spatial grating lobe; steering near endfire can
    still create one.
    """
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Spacing must be finite and greater than 0.")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("Sound speed must be finite and greater than 0.")
    return float(sound_speed) / float(spacing)


def _lattice_spacing(positions: np.ndarray) -> tuple[float, float]:
    spacings = []
    for axis in (0, 1):
        coordinates = np.unique(positions[:, axis])
        if len(coordinates) < 2:
            raise ValueError("Emitter coordinates must vary along x and y.")
        spacings.append(float(np.median(np.diff(coordinates))))
    return spacings[0], spacings[1]


def grating_lobe_directions(
    azimuth_degrees: float,
    off_axis_degrees: float,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
    max_order: int = 2,
    max_radius: float = 1.0,
) -> tuple[GratingLobe, ...]:
    """Return predicted grating-lobe centers inside the +z hemisphere.

    A grating lobe appears at direction cosines
    ``u = u0 + (m * lambda / dx, n * lambda / dy)``. Only centers with
    ``|u| <= max_radius`` and ``u_z >= 0`` are returned. The default
    ``max_radius=1`` keeps physical directions; a slightly larger radius is
    useful for masking the skirt of a grating lobe whose center lies just
    outside visible space.
    """
    acoustics.validate_acoustic_params(frequency_hz, sound_speed)
    acoustics.validate_steering_angles(azimuth_degrees, off_axis_degrees)
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer.")
    if not math.isfinite(max_radius) or max_radius < 1.0:
        raise ValueError("max_radius must be finite and at least 1.")

    positions = acoustics.validate_emitter_positions(emitters_pos)
    spacing_x, spacing_y = _lattice_spacing(positions)
    wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
    step_x = wavelength / spacing_x
    step_y = wavelength / spacing_y
    target = acoustics.direction_from_angles(azimuth_degrees, off_axis_degrees)

    lobes = []
    for order_x in range(-max_order, max_order + 1):
        for order_y in range(-max_order, max_order + 1):
            if order_x == 0 and order_y == 0:
                continue
            ux = float(target[0] + order_x * step_x)
            uy = float(target[1] + order_y * step_y)
            radius_squared = ux * ux + uy * uy
            if radius_squared > max_radius * max_radius + 1e-12:
                continue
            uz = math.sqrt(max(0.0, 1.0 - radius_squared))
            lobes.append(GratingLobe(ux, uy, uz, order_x, order_y))
    return tuple(lobes)


def aperture_weights(
    window: str = "hann",
    *,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return a radial aperture weight in ``[0, 1]`` for every emitter.

    ``none`` is flat, ``hann`` is ``0.5 * (1 + cos(pi * r))`` and ``gaussian``
    is ``exp(-0.5 * (r / sigma)^2)``, where ``r`` is the distance from the
    array center normalized to the farthest emitter.
    """
    if window not in APERTURE_WINDOWS:
        raise ValueError(
            f"window must be one of {', '.join(APERTURE_WINDOWS)}."
        )
    if not math.isfinite(gaussian_sigma) or gaussian_sigma <= 0.0:
        raise ValueError("gaussian_sigma must be finite and greater than 0.")

    positions = acoustics.validate_emitter_positions(emitters_pos)
    offsets = positions[:, :2] - positions[:, :2].mean(axis=0)
    radius = np.linalg.norm(offsets, axis=1)
    largest = float(radius.max())
    if largest <= 0.0:
        raise ValueError("Emitter positions must not all share one point.")
    normalized = radius / largest

    if window == "none":
        return np.ones(N_EMITTERS, dtype=float)
    if window == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * normalized))
    return np.exp(-0.5 * (normalized / gaussian_sigma) ** 2)


# Amplitude weighting through the firmware's duty-cycle control.
#
# A payload byte's high nibble is an amplitude code.  The FPGA turns it into
# a pulse width of 1..8 steps (of the 16-step, 40 kHz period), centred on the
# phase bin, so the fundamental pressure amplitude relative to the legacy
# 50 % pulse is sin(pi * width / 16).
MIN_DUTY_AMPLITUDE = math.sin(math.pi / 16)  # width 1: about 0.195


def duty_width_from_code(code: int) -> int:
    """Return the firmware pulse width (1..8) for an amplitude code.

    Code 0 is the legacy "no amplitude information" full-scale pulse and
    code 15 is also full scale; 1..14 map onto widths 1..8.  This mirrors
    the ``pulse_width`` function in ``AllChannels.vhd``.
    """
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
        raise ValueError("amplitude code must be an integer.")
    code = int(code)
    if not 0 <= code <= 15:
        raise ValueError("amplitude code must be between 0 and 15.")
    if code == 0 or code >= 15:
        return 8
    return (code + 1) // 2


def amplitude_from_code(code: int) -> float:
    """Return the fundamental amplitude relative to full scale (0..1)."""
    return math.sin(math.pi * duty_width_from_code(code) / 16.0)


def amplitude_code_for_weight(weight: float) -> int:
    """Map a relative pressure amplitude ``0..1`` to a firmware code.

    Zero (or negative) weight returns :data:`OFF_VALUE`; a positive weight
    is snapped to the nearest representable width.  The firmware's smallest
    non-zero width already gives ``sin(pi/16) ~ 0.195``, so quieter non-zero
    weights round up to that level.
    """
    if isinstance(weight, bool) or not isinstance(
        weight, (int, float, np.integer, np.floating)
    ):
        raise ValueError("weight must be a number.")
    value = float(weight)
    if not math.isfinite(value):
        raise ValueError("weight must be finite.")
    if value <= 0.0:
        return OFF_VALUE
    if value > 1.0:
        raise ValueError("weight must not exceed 1.")

    width = math.floor((16.0 / math.pi) * math.asin(min(value, 1.0)) + 0.5)
    width = max(1, min(8, int(width)))
    code = math.floor((15.0 * width / 8.0) + 0.5)
    return max(1, min(15, int(code)))


def amplitude_codes_from_weights(weights) -> np.ndarray:
    """Return 256 amplitude codes for per-element relative weights.

    Weights must be finite and in ``0..1``.  A zero weight becomes
    :data:`OFF_VALUE`, which both mutes the channel on the wire and removes
    it from the active mask in :class:`BeamAnalyzer`.
    """
    values = np.asarray(weights, dtype=float)
    if values.shape != (N_EMITTERS,):
        raise ValueError(f"Amplitude weights must contain {N_EMITTERS} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Amplitude weights must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Amplitude weights must be between 0 and 1.")
    return np.asarray(
        [amplitude_code_for_weight(value) for value in values], dtype=int
    )


def thinned_mask(
    window: str = "none",
    keep_fraction: float = 1.0,
    *,
    seed: int = 0,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return an active-element mask with a windowed density taper.

    Exactly ``round(keep_fraction * 256)`` elements are selected without
    replacement, with probability proportional to the radial window. ``none``
    therefore gives uniform random thinning, while ``hann``/``gaussian`` keep
    the center and thin the edges. The same seed gives the same mask.
    """
    if not isinstance(keep_fraction, (int, float, np.integer, np.floating)):
        raise ValueError("keep_fraction must be a number.")
    fraction = float(keep_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("keep_fraction must be greater than 0 and at most 1.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    count = min(N_EMITTERS, max(1, int(round(fraction * N_EMITTERS))))
    mask = np.zeros(N_EMITTERS, dtype=bool)
    if count == N_EMITTERS:
        mask[:] = True
        return mask

    weights = aperture_weights(
        window, gaussian_sigma=gaussian_sigma, emitters_pos=emitters_pos
    )
    probabilities = np.maximum(weights, _TINY)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(N_EMITTERS, size=count, replace=False, p=probabilities)
    mask[chosen] = True
    return mask


def dither_phase_bins(
    phase_bins,
    max_offset_bins: int = 1,
    *,
    seed: int = 0,
    active_mask=None,
) -> np.ndarray:
    """Add a deterministic random phase offset in ``[-k, +k]`` bins.

    Dithering decorrelates quantization error between channels. It does not
    remove grating lobes (see the module docstring) and normally costs some
    main-lobe gain; it is provided for pattern-error experiments.
    """
    if isinstance(max_offset_bins, bool) or not isinstance(
        max_offset_bins, (int, np.integer)
    ):
        raise ValueError("max_offset_bins must be an integer.")
    offset_limit = int(max_offset_bins)
    if not 0 <= offset_limit <= PHASE_BINS // 2:
        raise ValueError(f"max_offset_bins must be between 0 and {PHASE_BINS // 2}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    bins = validate_phase_bins(phase_bins).copy()
    active = (
        bins != OFF_VALUE
        if active_mask is None
        else validate_active_mask(active_mask) & (bins != OFF_VALUE)
    )
    if offset_limit == 0:
        return bins

    rng = np.random.default_rng(seed)
    offsets = rng.integers(-offset_limit, offset_limit + 1, size=N_EMITTERS)
    bins[active] = (bins[active].astype(int) + offsets[active]) % PHASE_BINS
    return bins


def apply_calibration_bins(phase_bins, offsets_bins, *, active_mask=None) -> np.ndarray:
    """Add per-channel phase offsets (in bins) and wrap modulo 16."""
    bins = validate_phase_bins(phase_bins).copy()
    offsets = validate_phase_bins(
        offsets_bins, allow_off=False, label="calibration offsets"
    )
    active = (
        bins != OFF_VALUE
        if active_mask is None
        else validate_active_mask(active_mask) & (bins != OFF_VALUE)
    )
    bins[active] = (bins[active].astype(int) + offsets[active]) % PHASE_BINS
    return bins


def load_calibration_file(path) -> np.ndarray:
    """Load 256 phase offsets in bins from a JSON file.

    Accepted shapes: a plain list of 256 integers, or an object with an
    ``offsets_bins`` (or ``phase_offsets_bins``) key. Offsets are in bins and
    must be in ``0..15``.
    """
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read calibration file {source}: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"Calibration file {source} is not UTF-8 text.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Calibration file {source} is not valid JSON.") from error

    if isinstance(data, dict):
        for key in ("offsets_bins", "phase_offsets_bins"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                "Calibration JSON must contain an 'offsets_bins' list."
            )
    return validate_phase_bins(data, allow_off=False, label="calibration offsets")


class BeamAnalyzer:
    """Ideal far-field model plus a phase-only sidelobe optimizer.

    The model uses isotropic elements and ideal phase bins, matching
    :mod:`array_factor`. It is a design tool, not a measurement. Grating-lobe
    regions are predicted from the lattice and excluded from the optimizer's
    objective because their level is invariant under per-element weighting.

    Args:
        azimuth_degrees: beam azimuth, from +x toward +y.
        off_axis_degrees: off-axis angle from +z; must be in ``0..90``.
        samples: direction-cosine grid samples per axis.
    """

    def __init__(
        self,
        azimuth_degrees: float,
        off_axis_degrees: float,
        *,
        frequency_hz=DEFAULT_FREQUENCY_HZ,
        sound_speed=DEFAULT_SOUND_SPEED_MPS,
        emitters_pos=EMITTERS_POS,
        samples: int = DEFAULT_SAMPLES,
        main_lobe_safety_factor: float = DEFAULT_MAIN_LOBE_SAFETY_FACTOR,
        grating_order: int = 2,
    ):
        acoustics.validate_acoustic_params(frequency_hz, sound_speed)
        acoustics.validate_steering_angles(azimuth_degrees, off_axis_degrees)
        if not 0.0 <= off_axis_degrees <= 90.0:
            raise ValueError(
                "Beam analysis is limited to the +z front hemisphere (beta 0..90)."
            )
        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
            raise ValueError("samples must be an integer.")
        if int(samples) < 5:
            raise ValueError("samples must be at least 5.")
        if (
            not math.isfinite(main_lobe_safety_factor)
            or main_lobe_safety_factor <= 0.0
        ):
            raise ValueError("main_lobe_safety_factor must be greater than 0.")

        self.frequency_hz = float(frequency_hz)
        self.sound_speed = float(sound_speed)
        self.wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
        self.azimuth_degrees = float(azimuth_degrees)
        self.off_axis_degrees = float(off_axis_degrees)
        self.direction = acoustics.direction_from_angles(
            azimuth_degrees, off_axis_degrees
        )

        positions = acoustics.validate_emitter_positions(emitters_pos)
        self._positions = positions
        self._center = positions.mean(axis=0)
        centered = positions - self._center
        wave_number = TWO_PI / self.wavelength

        self._target_vector = np.exp(1j * wave_number * (centered @ self.direction))

        samples = int(samples)
        u_values = np.linspace(-1.0, 1.0, samples)
        ux_grid, uy_grid = np.meshgrid(u_values, u_values)
        valid = ux_grid**2 + uy_grid**2 <= 1.0
        uz_valid = np.sqrt(np.maximum(0.0, 1.0 - ux_grid**2 - uy_grid**2))
        directions = np.column_stack(
            (ux_grid[valid], uy_grid[valid], uz_valid[valid])
        )

        self._full_ux = ux_grid
        self._full_uy = uy_grid
        self._full_valid = valid
        self._dirs = directions
        self._steering_matrix = np.exp(
            1j * wave_number * (centered @ directions.T)
        ).T

        aperture_x = float(np.ptp(positions[:, 0]))
        aperture_y = float(np.ptp(positions[:, 1]))
        if aperture_x <= 0.0 or aperture_y <= 0.0:
            raise ValueError("Array apertures in x and y must be greater than 0.")
        self.main_lobe_half_width_ux = (
            float(main_lobe_safety_factor) * self.wavelength / aperture_x
        )
        self.main_lobe_half_width_uy = (
            float(main_lobe_safety_factor) * self.wavelength / aperture_y
        )
        self.main_lobe_exclusion_deg = math.degrees(
            math.asin(
                min(
                    1.0,
                    math.hypot(
                        self.main_lobe_half_width_ux,
                        self.main_lobe_half_width_uy,
                    ),
                )
            )
        )

        ux_valid = directions[:, 0]
        uy_valid = directions[:, 1]
        main_mask = (
            np.abs(ux_valid - self.direction[0]) <= self.main_lobe_half_width_ux
        ) & (
            np.abs(uy_valid - self.direction[1]) <= self.main_lobe_half_width_uy
        )
        if not np.any(~main_mask):
            raise ValueError(
                "Direction grid is too coarse: it contains only main-lobe points."
            )

        grating_mask = np.zeros(directions.shape[0], dtype=bool)
        self.grating_lobes = ()
        if grating_order > 0:
            kept = []
            skirt_margin = math.hypot(
                self.main_lobe_half_width_ux, self.main_lobe_half_width_uy
            )
            for lobe in grating_lobe_directions(
                azimuth_degrees,
                off_axis_degrees,
                frequency_hz=frequency_hz,
                sound_speed=sound_speed,
                emitters_pos=positions,
                max_order=int(grating_order),
                max_radius=1.0 + skirt_margin,
            ):
                box = (
                    np.abs(ux_valid - lobe.ux) <= self.main_lobe_half_width_ux
                ) & (
                    np.abs(uy_valid - lobe.uy) <= self.main_lobe_half_width_uy
                )
                if np.any(box):
                    grating_mask |= box
                    if lobe.ux**2 + lobe.uy**2 <= 1.0 + 1e-12:
                        kept.append(lobe)
            self.grating_lobes = tuple(kept)

        self._main_mask = main_mask
        self._sidelobe_mask = ~main_mask
        self._near_sidelobe_mask = (~main_mask) & (~grating_mask)
        if not np.any(self._near_sidelobe_mask):
            self._near_sidelobe_mask = self._sidelobe_mask
        self._grating_mask = grating_mask

    @property
    def lattice_spacing_m(self) -> float:
        """Median emitter spacing in meters."""
        spacing_x, spacing_y = _lattice_spacing(self._positions)
        return 0.5 * (spacing_x + spacing_y)

    @property
    def spatial_aliasing_frequency_hz(self) -> float:
        """Carrier above which a broadside beam has a visible grating lobe."""
        return spatial_aliasing_frequency_hz(
            spacing=self.lattice_spacing_m, sound_speed=self.sound_speed
        )

    def _amplitude_vector(self, amplitude_codes=None) -> np.ndarray:
        if amplitude_codes is None:
            return np.ones(N_EMITTERS, dtype=float)
        codes = validate_amplitude_codes(amplitude_codes)
        amplitudes = np.zeros(N_EMITTERS, dtype=float)
        for index, code in enumerate(codes):
            if code != OFF_VALUE:
                amplitudes[index] = amplitude_from_code(int(code))
        return amplitudes

    def _weight_vector(self, phase_bins, active_mask=None, amplitude_codes=None):
        bins = validate_phase_bins(phase_bins)
        amplitudes = self._amplitude_vector(amplitude_codes)
        active = (bins != OFF_VALUE) & (amplitudes > 0.0)
        if active_mask is not None:
            active &= validate_active_mask(active_mask)
        if not np.any(active):
            raise ValueError("At least one emitter must be active with a phase bin.")
        weights = np.zeros(N_EMITTERS, dtype=complex)
        weights[active] = amplitudes[active] * np.exp(
            1j * bins[active].astype(float) * TWO_PI / PHASE_BINS
        )
        return weights, active

    def field(self, phase_bins, active_mask=None, amplitude_codes=None) -> np.ndarray:
        """Return the complex array factor on the direction grid."""
        weights, _ = self._weight_vector(phase_bins, active_mask, amplitude_codes)
        return self._steering_matrix @ weights

    def _target_amplitude(self, weights) -> float:
        return float(abs(self._target_vector @ weights))

    def _peak(self, ratio, mask):
        if not np.any(mask):
            return float("-inf"), float("nan"), float("nan")
        index = int(np.argmax(np.where(mask, ratio, -np.inf)))
        direction = self._dirs[index]
        azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        beta = math.degrees(math.acos(max(-1.0, min(1.0, direction[2]))))
        return 20.0 * math.log10(max(float(ratio[index]), _TINY)), azimuth, beta

    def metrics(self, phase_bins, active_mask=None, amplitude_codes=None) -> dict:
        """Return idealized pattern metrics for one frame.

        ``peak_sidelobe_db`` includes the grating-lobe regions and is
        therefore dominated by them on this lattice. ``near_sidelobe_db``
        excludes them and is the level the host algorithms can change.
        ``amplitude_codes`` optionally applies per-element duty weighting.
        """
        weights, active = self._weight_vector(phase_bins, active_mask, amplitude_codes)
        main = self._target_amplitude(weights)
        ratio = np.abs(self._steering_matrix @ weights) / max(main, _TINY)

        peak_sidelobe_db, _, _ = self._peak(ratio, self._sidelobe_mask)
        near_db, near_azimuth, near_beta = self._peak(
            ratio, self._near_sidelobe_mask
        )
        grating_db, grating_azimuth, grating_beta = self._peak(
            ratio, self._grating_mask
        )
        return {
            "main_lobe_amplitude": main,
            "main_lobe_gain_db": 20.0 * math.log10(max(main / N_EMITTERS, _TINY)),
            "active_elements": int(np.count_nonzero(active)),
            "peak_sidelobe_db": peak_sidelobe_db,
            "near_sidelobe_db": near_db,
            "near_sidelobe_azimuth_deg": near_azimuth,
            "near_sidelobe_beta_deg": near_beta,
            "grating_lobe_db": grating_db,
            "grating_lobe_azimuth_deg": grating_azimuth,
            "grating_lobe_beta_deg": grating_beta,
            "main_lobe_exclusion_deg": self.main_lobe_exclusion_deg,
        }

    def _null_indices(self, null_directions) -> tuple[np.ndarray, list]:
        if not null_directions:
            return np.empty(0, dtype=int), []
        indices = []
        notes = []
        for item in null_directions:
            if len(item) != 2:
                raise ValueError("Each null direction must be (azimuth, beta).")
            azimuth, beta = float(item[0]), float(item[1])
            acoustics.validate_steering_angles(azimuth, beta)
            if not 0.0 <= beta <= 90.0:
                raise ValueError("Null directions must be in the front hemisphere.")
            direction = acoustics.direction_from_angles(azimuth, beta)
            dots = self._dirs @ direction
            index = int(np.argmax(dots))
            indices.append(index)
            in_main = bool(self._main_mask[index])
            in_grating = bool(self._grating_mask[index])
            notes.append(
                {
                    "azimuth_deg": azimuth,
                    "beta_deg": beta,
                    "grid_azimuth_deg": math.degrees(
                        math.atan2(self._dirs[index, 1], self._dirs[index, 0])
                    ),
                    "grid_beta_deg": math.degrees(
                        math.acos(
                            max(-1.0, min(1.0, float(self._dirs[index, 2])))
                        )
                    ),
                    "in_main_lobe": in_main,
                    "in_grating_lobe": in_grating,
                }
            )
        return np.asarray(indices, dtype=int), notes

    def optimize(
        self,
        phase_bins,
        active_mask=None,
        *,
        max_sweeps: int = 4,
        seed: int = 0,
        null_directions=None,
        null_weight: float = 1.0,
        amplitude_codes=None,
    ) -> tuple[np.ndarray, dict]:
        """Minimize the near-sidelobe peak by coordinate descent on phase bins.

        Only elements that are active and carry a phase bin are changed; OFF
        elements stay OFF. Each element is offered all 16 phase bins, so the
        result is directly encodable in a frame.

        Args:
            max_sweeps: maximum passes over the active elements.
            seed: shuffles the element order; same seed gives the same result.
            null_directions: optional ``(azimuth, beta)`` pairs to null.
            null_weight: penalty weight added for the deepest requested null.
            amplitude_codes: optional per-element duty codes.  The amplitude
                is held fixed while the phase is optimized.
        """
        if isinstance(max_sweeps, bool) or not isinstance(
            max_sweeps, (int, np.integer)
        ):
            raise ValueError("max_sweeps must be an integer.")
        sweeps_limit = int(max_sweeps)
        if sweeps_limit < 0:
            raise ValueError("max_sweeps must be non-negative.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not math.isfinite(null_weight) or null_weight < 0.0:
            raise ValueError("null_weight must be a finite non-negative number.")

        bins = validate_phase_bins(phase_bins).copy()
        weights, active = self._weight_vector(bins, active_mask, amplitude_codes)
        amplitudes = self._amplitude_vector(amplitude_codes)
        active_indices = np.flatnonzero(active)
        null_indices, null_notes = self._null_indices(null_directions)
        effective_null_weight = float(null_weight) if null_indices.size else 0.0

        field = self._steering_matrix @ weights
        main = complex(self._target_vector @ weights)
        near_indices = np.flatnonzero(self._near_sidelobe_mask)
        candidate_phases = np.exp(
            1j * np.arange(PHASE_BINS, dtype=float) * TWO_PI / PHASE_BINS
        )

        def scores(fields, mains):
            denominator = np.maximum(np.abs(mains), _TINY)[:, None]
            ratios = np.abs(fields[:, near_indices]) / denominator
            peak = ratios.max(axis=1)
            energy = np.mean(ratios * ratios, axis=1)
            if effective_null_weight and null_indices.size:
                null_ratios = np.abs(fields[:, null_indices]) / denominator
                peak = peak + effective_null_weight * null_ratios.max(axis=1)
            return peak, energy

        current_peaks, current_energies = scores(field[None, :], np.array([main]))
        current_peak = float(current_peaks[0])
        current_energy = float(current_energies[0])
        initial_null_db = []
        if null_indices.size:
            initial_null_ratio = np.abs(field[null_indices]) / max(abs(main), _TINY)
            initial_null_db = 20.0 * np.log10(
                np.maximum(initial_null_ratio, _TINY)
            )
        sweeps_run = 0
        changed_total = 0
        if sweeps_limit == 0:
            for note, level in zip(null_notes, initial_null_db):
                note["initial_db"] = float(level)
                note["achieved_db"] = float(level)
            return bins, {
                "sweeps_run": 0,
                "changed_elements": 0,
                "null_directions": null_notes,
            }

        rng = np.random.default_rng(seed)
        for sweep in range(sweeps_limit):
            changed = 0
            for index in rng.permutation(active_indices):
                steering_column = self._steering_matrix[:, index]
                old_phase = weights[index] / amplitudes[index]
                delta = amplitudes[index] * (candidate_phases - old_phase)
                candidate_fields = (
                    field[None, :] + delta[:, None] * steering_column[None, :]
                )
                candidate_mains = main + delta * self._target_vector[index]
                peaks, energies = scores(candidate_fields, candidate_mains)
                best = np.lexsort((energies, peaks))[0]
                better = peaks[best] < current_peak - 1e-9 or (
                    abs(peaks[best] - current_peak) <= 1e-9
                    and energies[best] < current_energy - 1e-12
                )
                if better:
                    field = candidate_fields[best]
                    main = complex(candidate_mains[best])
                    weights[index] = amplitudes[index] * candidate_phases[best]
                    bins[index] = best
                    current_peak = float(peaks[best])
                    current_energy = float(energies[best])
                    changed += 1
            sweeps_run = sweep + 1
            changed_total += changed
            if changed == 0:
                break

        if null_indices.size:
            final_null_ratio = np.abs(field[null_indices]) / max(abs(main), _TINY)
            final_null_db = 20.0 * np.log10(np.maximum(final_null_ratio, _TINY))
            for note, start, end in zip(null_notes, initial_null_db, final_null_db):
                note["initial_db"] = float(start)
                note["achieved_db"] = float(end)

        return bins, {
            "sweeps_run": sweeps_run,
            "changed_elements": changed_total,
            "null_directions": null_notes,
        }


def plot_pattern(
    analyzer: BeamAnalyzer,
    phase_bins,
    active_mask=None,
    *,
    title: str | None = None,
    amplitude_codes=None,
) -> None:
    """Plot the ideal pattern in azimuth/beta coordinates; matplotlib optional."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to plot beam patterns.") from error

    weights, _ = analyzer._weight_vector(phase_bins, active_mask, amplitude_codes)
    main = analyzer._target_amplitude(weights)
    ratio = np.abs(analyzer._steering_matrix @ weights) / max(main, _TINY)
    levels = 20.0 * np.log10(np.maximum(ratio, 1e-6))

    full = np.full(analyzer._full_valid.shape, np.nan)
    full[analyzer._full_valid] = levels
    azimuth = np.degrees(np.arctan2(analyzer._full_uy, analyzer._full_ux))
    radial_squared = np.maximum(
        0.0, 1.0 - analyzer._full_ux**2 - analyzer._full_uy**2
    )
    beta = np.degrees(np.arccos(np.clip(np.sqrt(radial_squared), -1.0, 1.0)))

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = axis.pcolormesh(
        azimuth,
        beta,
        full,
        shading="auto",
        cmap="viridis",
        vmin=-40.0,
        vmax=0.0,
    )
    axis.plot(
        analyzer.azimuth_degrees,
        analyzer.off_axis_degrees,
        marker="x",
        color="red",
        markersize=8,
        label="Requested direction",
    )
    axis.set_xlabel("Azimuth (degrees, from +x toward +y)")
    axis.set_ylabel("Off-axis angle from +z (degrees)")
    axis.set_title(title or "Idealized beam pattern (not a measurement)")
    figure.colorbar(image, ax=axis, label="Normalized array factor (dB)")
    axis.legend()
    plt.show()
