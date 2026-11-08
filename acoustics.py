"""Shared acoustic math for focusing, vortex and beam steering.

All coordinates are meters. The array lies in the z=0 plane with its origin
at the bottom-left emitter (see :mod:`emitters`).

Propagation convention: phases are computed as ``-k * r`` and wrapped to
``[0, 2*pi)``, which assumes ``exp(i*(omega*t - k*r))``. The firmware's
phase-bin mapping and physical beam direction still require hardware
verification.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from emitters import EMITTERS_POS, N_EMITTERS
from protocol import PHASE_BINS

DEFAULT_FREQUENCY_HZ = 40000.0
DEFAULT_SOUND_SPEED_MPS = 343.0
TWO_PI = 2.0 * math.pi


def _positive_finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and greater than 0.")
    return number


def validate_acoustic_params(
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
) -> None:
    """Validate carrier frequency and sound speed."""
    _positive_finite(frequency_hz, "Carrier frequency")
    _positive_finite(sound_speed, "Sound speed")


def acoustic_wavelength(
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
) -> float:
    """Return the wavelength in meters for the given medium and carrier."""
    validate_acoustic_params(frequency_hz, sound_speed)
    return float(sound_speed) / float(frequency_hz)


def validate_emitter_positions(emitters_pos=EMITTERS_POS) -> np.ndarray:
    """Return validated emitter coordinates with shape (256, 3)."""
    positions = np.asarray(emitters_pos, dtype=float)
    if positions.shape != (N_EMITTERS, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"Expected finite emitter positions with shape ({N_EMITTERS}, 3)."
        )
    return positions


def array_center(emitters_pos=EMITTERS_POS) -> np.ndarray:
    """Compute the geometric center from the actual emitter coordinates."""
    return validate_emitter_positions(emitters_pos).mean(axis=0)


def _target_position(target_pos) -> np.ndarray:
    target = np.asarray(target_pos, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Target position must contain three finite coordinates.")
    return target


def quantize_phase_bins(phases, num_bins: int = PHASE_BINS) -> np.ndarray:
    """Map continuous phases to bins, snapping floating-point boundary noise.

    Values within a few ULPs of an exact bin boundary are snapped to that
    boundary before flooring, which keeps mathematically identical phases
    from landing in different bins.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, (int, np.integer)):
        if not (isinstance(num_bins, float) and float(num_bins).is_integer()):
            raise ValueError("Number of phase bins must be a positive integer.")
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("Number of phase bins must be a positive integer.")

    phase_values = np.asarray(phases, dtype=float)
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("Phases must contain only finite values.")

    wrapped = np.mod(phase_values, TWO_PI)
    scaled = wrapped * bins / TWO_PI
    nearest_boundaries = np.rint(scaled)
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(scaled), 1.0))
    scaled = np.where(
        np.abs(scaled - nearest_boundaries) <= tolerance,
        nearest_boundaries,
        scaled,
    )
    phase_bins = np.mod(np.floor(scaled).astype(int), bins)
    return phase_bins.astype(np.uint8)


def focus_phases(
    target_pos,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return wrapped focusing phases for one target point.

    ``phase = mod(-2*pi*f*d/c, 2*pi)`` where ``d`` is the emitter-target
    distance. This is equivalent to the legacy ``ceil(d/lambda) - d/lambda``
    formula when the same wavelength is used.
    """
    validate_acoustic_params(frequency_hz, sound_speed)
    positions = validate_emitter_positions(emitters_pos)
    target = _target_position(target_pos)
    distances = np.linalg.norm(positions - target, axis=1)
    return np.mod(-TWO_PI * float(frequency_hz) * distances / float(sound_speed), TWO_PI)


def focus_phase_bins(
    target_pos,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return one 0..15 phase bin per emitter for a focus point."""
    phases = focus_phases(
        target_pos,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
        emitters_pos=emitters_pos,
    )
    return quantize_phase_bins(phases)


def polar_coordinates(emitters_pos=EMITTERS_POS, center=None):
    """Return (radii, angles) of every emitter around the XY center."""
    positions = validate_emitter_positions(emitters_pos)
    if center is None:
        center = array_center(positions)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Reference center must contain three finite coordinates.")
    offsets = positions[:, :2] - center[:2]
    radii = np.linalg.norm(offsets, axis=1)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    return radii, angles


class VortexPhases(NamedTuple):
    """Phase components of a single-point vortex."""

    total: np.ndarray
    focus: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    rotation_offsets: np.ndarray
    center: np.ndarray


def vortex_phases(
    target_pos,
    topological_charge: int = 1,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
) -> VortexPhases:
    """Return focus phase plus ``charge * polar_angle`` for every emitter."""
    if isinstance(topological_charge, bool) or not isinstance(
        topological_charge, (int, np.integer)
    ):
        raise ValueError("topological_charge must be an integer.")

    center = array_center(emitters_pos)
    focus = focus_phases(
        target_pos,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
        emitters_pos=emitters_pos,
    )
    radii, angles = polar_coordinates(emitters_pos, center)
    rotation_offsets = float(topological_charge) * angles
    total = np.mod(focus + rotation_offsets, TWO_PI)
    return VortexPhases(total, focus, radii, angles, rotation_offsets, center)


def vortex_phase_bins(
    target_pos,
    topological_charge: int = 1,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return one phase bin per emitter for a single-point vortex."""
    return quantize_phase_bins(
        vortex_phases(
            target_pos,
            topological_charge=topological_charge,
            frequency_hz=frequency_hz,
            sound_speed=sound_speed,
            emitters_pos=emitters_pos,
        ).total
    )


def validate_steering_angles(azimuth_degrees: float, off_axis_degrees: float) -> None:
    """Validate periodic azimuth in [-360, 360] and off-axis in [0, 180]."""
    if not math.isfinite(azimuth_degrees) or not math.isfinite(off_axis_degrees):
        raise ValueError("Angles must be finite numbers.")
    if not -360.0 <= azimuth_degrees <= 360.0:
        raise ValueError("Azimuth must be between -360 and 360 degrees.")
    if not 0.0 <= off_axis_degrees <= 180.0:
        raise ValueError("Off-axis angle must be between 0 and 180 degrees.")


def direction_from_angles(azimuth_degrees: float, off_axis_degrees: float) -> np.ndarray:
    """Return the unit direction for the documented spherical convention.

    Azimuth is measured in the x-y plane from +x toward +y; off-axis angle is
    measured from +z. ``beta=0`` points along +z.
    """
    validate_steering_angles(azimuth_degrees, off_axis_degrees)
    azimuth = math.radians(azimuth_degrees % 360.0)
    off_axis = math.radians(off_axis_degrees)
    direction = np.array(
        [
            math.sin(off_axis) * math.cos(azimuth),
            math.sin(off_axis) * math.sin(azimuth),
            math.cos(off_axis),
        ],
        dtype=float,
    )
    direction[np.isclose(direction, 0.0, atol=1e-15)] = 0.0
    return direction


def legacy_angle_to_spherical(angle_degrees: float) -> tuple[float, float]:
    """Convert the legacy x-z angle to (azimuth, off-axis) degrees.

    0 degrees points along +x, 90 along +z, and 180 along -x.
    """
    if (
        isinstance(angle_degrees, bool)
        or not math.isfinite(angle_degrees)
        or not 0.0 <= angle_degrees <= 180.0
    ):
        raise ValueError("Legacy angle must be between 0 and 180 degrees.")
    if angle_degrees <= 90.0:
        return 0.0, 90.0 - angle_degrees
    return 180.0, angle_degrees - 90.0


def steering_phases(
    azimuth_degrees: float,
    off_axis_degrees: float,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
    reference_center=None,
) -> np.ndarray:
    """Return wrapped phases that steer the array toward one direction."""
    validate_acoustic_params(frequency_hz, sound_speed)
    validate_steering_angles(azimuth_degrees, off_axis_degrees)
    positions = validate_emitter_positions(emitters_pos)

    if reference_center is None:
        center = array_center(positions)
    else:
        center = np.asarray(reference_center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Reference center must contain three finite coordinates.")

    direction = direction_from_angles(azimuth_degrees, off_axis_degrees)
    projected = (positions - center) @ direction
    return np.mod(
        -TWO_PI * float(frequency_hz) * projected / float(sound_speed),
        TWO_PI,
    )


def steering_phase_bins(
    azimuth_degrees: float,
    off_axis_degrees: float,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
    reference_center=None,
) -> np.ndarray:
    """Return one phase bin per emitter for a steered beam."""
    return quantize_phase_bins(
        steering_phases(
            azimuth_degrees,
            off_axis_degrees,
            frequency_hz=frequency_hz,
            sound_speed=sound_speed,
            emitters_pos=emitters_pos,
            reference_center=reference_center,
        )
    )


def line_steering_phase_bins(
    angle_degrees: float,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
) -> np.ndarray:
    """Return bins for the legacy x-z beam angle (0=+x, 90=+z, 180=-x)."""
    azimuth, off_axis = legacy_angle_to_spherical(angle_degrees)
    return steering_phase_bins(
        azimuth,
        off_axis,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
        emitters_pos=emitters_pos,
    )
