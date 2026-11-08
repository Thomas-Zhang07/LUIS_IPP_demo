"""Theoretical array-factor and sidelobe analysis (diagnostic only).

These models assume ideal phase bins (2*pi*k/16) and isotropic elements. They
are not measurements and do not predict the complete acoustic field, which
also depends on the firmware's phase table and element directivity.
"""

from __future__ import annotations

import math

import numpy as np

import acoustics
from emitters import EMITTERS_POS, N_EMITTERS, N_SIDE, SPACING
from protocol import PHASE_BINS

DEFAULT_MAIN_LOBE_SAFETY_FACTOR = 1.5


def physical_phase_grid(phase_values, emitters_pos=EMITTERS_POS):
    """Arrange logical phases into rows indexed by physical y and x."""
    positions = acoustics.validate_emitter_positions(emitters_pos)
    values = np.asarray(phase_values)
    if values.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} phase values.")

    x_coordinates = np.unique(positions[:, 0])
    y_coordinates = np.unique(positions[:, 1])
    if len(x_coordinates) != N_SIDE or len(y_coordinates) != N_SIDE:
        raise ValueError("Emitter coordinates do not form the expected 16x16 grid.")

    grid = np.empty((N_SIDE, N_SIDE), dtype=values.dtype)
    x_indices = np.searchsorted(x_coordinates, positions[:, 0])
    y_indices = np.searchsorted(y_coordinates, positions[:, 1])
    grid[y_indices, x_indices] = values
    return x_coordinates, y_coordinates, grid


def print_physical_phase_map(
    phase_values, emitters_pos=EMITTERS_POS, *, output=print
) -> None:
    """Print phase bins in physical x-y order."""
    x_coordinates, y_coordinates, grid = physical_phase_grid(
        phase_values, emitters_pos
    )
    output("\nPhysical x-y phase map (bin 0..15):")
    output(
        "       "
        + " ".join(f"{coordinate * 1000:5.0f}" for coordinate in x_coordinates)
    )
    for row_index in range(N_SIDE - 1, -1, -1):
        row_values = " ".join(f"{int(value):5d}" for value in grid[row_index])
        output(f"y={y_coordinates[row_index] * 1000:5.0f} mm {row_values}")


def compute_array_factor_pattern(
    phase_bins,
    *,
    frequency_hz=acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed=acoustics.DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
    reference_center=None,
    azimuth_samples: int = 361,
    off_axis_samples: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized ideal array-factor levels in dB."""
    values = np.asarray(phase_bins)
    if values.shape != (N_EMITTERS,) or np.any(
        (values < 0) | (values >= PHASE_BINS)
    ):
        raise ValueError(
            "Array-factor input must contain 256 phase bins in the range 0..15."
        )
    if azimuth_samples < 2 or off_axis_samples < 2:
        raise ValueError("Array-factor grids must have at least two samples per angle.")

    positions = acoustics.validate_emitter_positions(emitters_pos)
    center = (
        acoustics.array_center(positions)
        if reference_center is None
        else np.asarray(reference_center, dtype=float)
    )
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Reference center must contain three finite coordinates.")

    azimuth_values = np.linspace(-180.0, 180.0, azimuth_samples)
    off_axis_values = np.linspace(0.0, 180.0, off_axis_samples)
    azimuth_grid, off_axis_grid = np.meshgrid(azimuth_values, off_axis_values)
    azimuth_radians = np.deg2rad(azimuth_grid.ravel())
    off_axis_radians = np.deg2rad(off_axis_grid.ravel())
    directions = np.column_stack(
        (
            np.sin(off_axis_radians) * np.cos(azimuth_radians),
            np.sin(off_axis_radians) * np.sin(azimuth_radians),
            np.cos(off_axis_radians),
        )
    )

    wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
    wave_number = acoustics.TWO_PI / wavelength
    centered_positions = positions - center
    ideal_bin_phases = values.astype(float) * acoustics.TWO_PI / PHASE_BINS
    element_phase = (
        wave_number * (directions @ centered_positions.T) + ideal_bin_phases
    )
    field = np.exp(1j * element_phase).sum(axis=1)
    magnitude = np.abs(field).reshape(azimuth_grid.shape)
    normalized = magnitude / np.max(magnitude)
    levels_db = 20.0 * np.log10(np.maximum(normalized, 1e-6))
    return azimuth_grid, off_axis_grid, levels_db


def _array_factor_on_direction_cosine_grid(
    phase_bins,
    emitters_pos,
    direction_cosine_samples: int,
    *,
    frequency_hz,
    sound_speed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a planar Cartesian array factor efficiently on a square grid."""
    positions = acoustics.validate_emitter_positions(emitters_pos)
    x_coordinates = np.unique(positions[:, 0])
    y_coordinates = np.unique(positions[:, 1])
    center = acoustics.array_center(positions)
    if len(x_coordinates) != N_SIDE or len(y_coordinates) != N_SIDE:
        raise ValueError(
            "Array-factor evaluation requires the existing 16x16 Cartesian grid."
        )
    if not np.allclose(positions[:, 2], center[2]):
        raise ValueError("Array-factor evaluation requires a planar array with constant z.")

    x_indices = np.searchsorted(x_coordinates, positions[:, 0])
    y_indices = np.searchsorted(y_coordinates, positions[:, 1])
    if len(set(zip(x_indices.tolist(), y_indices.tolist()))) != N_EMITTERS:
        raise ValueError("Emitter coordinates must form a complete Cartesian grid.")

    phase_values = np.asarray(phase_bins)
    if phase_values.shape != (N_EMITTERS,) or np.any(
        (phase_values < 0) | (phase_values >= PHASE_BINS)
    ):
        raise ValueError(
            "Array-factor input must contain 256 phase bins in the range 0..15."
        )
    if direction_cosine_samples < 11:
        raise ValueError("Direction-cosine grid must have at least 11 samples per axis.")

    _, _, phases_by_y_x = physical_phase_grid(phase_values, positions)
    weights_x_y = np.exp(1j * phases_by_y_x * acoustics.TWO_PI / PHASE_BINS).T
    x_centered = x_coordinates - center[0]
    y_centered = y_coordinates - center[1]
    u_values = np.linspace(-1.0, 1.0, direction_cosine_samples)
    wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
    wave_number = acoustics.TWO_PI / wavelength
    x_steering = np.exp(
        1j * wave_number * u_values[:, None] * x_centered[None, :]
    )
    y_steering = np.exp(
        1j * wave_number * u_values[:, None] * y_centered[None, :]
    )
    field = (x_steering @ weights_x_y @ y_steering.T).T
    return u_values, u_values, field, center


def evaluate_peak_sidelobe(
    azimuth_degrees: float,
    off_axis_degrees: float,
    phase_bins,
    *,
    frequency_hz=acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed=acoustics.DEFAULT_SOUND_SPEED_MPS,
    emitters_pos=EMITTERS_POS,
    direction_cosine_samples: int = 101,
    main_lobe_exclusion_deg: float | None = None,
) -> dict:
    """Measure peak sidelobe outside the main-lobe exclusion region."""
    acoustics.validate_steering_angles(azimuth_degrees, off_axis_degrees)
    if not 0.0 <= off_axis_degrees <= 90.0:
        raise ValueError(
            "Sidelobe analysis is limited to the +z front hemisphere (beta 0..90)."
        )
    positions = acoustics.validate_emitter_positions(emitters_pos)

    ux_values, uy_values, field, center = _array_factor_on_direction_cosine_grid(
        phase_bins,
        emitters_pos,
        direction_cosine_samples,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
    )
    ux_grid, uy_grid = np.meshgrid(ux_values, uy_values)
    valid_directions = ux_grid**2 + uy_grid**2 <= 1.0
    target_direction = acoustics.direction_from_angles(
        azimuth_degrees, off_axis_degrees
    )
    uz_grid = np.sqrt(np.maximum(0.0, 1.0 - ux_grid**2 - uy_grid**2))
    dot_products = (
        ux_grid * target_direction[0]
        + uy_grid * target_direction[1]
        + uz_grid * target_direction[2]
    )

    wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
    aperture_x = float(np.ptp(positions[:, 0]))
    aperture_y = float(np.ptp(positions[:, 1]))
    if aperture_x <= 0.0 or aperture_y <= 0.0:
        raise ValueError("Array apertures in x and y must be greater than 0.")
    if main_lobe_exclusion_deg is None:
        half_width_ux = DEFAULT_MAIN_LOBE_SAFETY_FACTOR * wavelength / aperture_x
        half_width_uy = DEFAULT_MAIN_LOBE_SAFETY_FACTOR * wavelength / aperture_y
        main_lobe_exclusion_deg = math.degrees(
            math.asin(min(1.0, math.hypot(half_width_ux, half_width_uy)))
        )
        main_lobe = (
            valid_directions
            & (np.abs(ux_grid - target_direction[0]) <= half_width_ux)
            & (np.abs(uy_grid - target_direction[1]) <= half_width_uy)
        )
    else:
        if (
            not math.isfinite(main_lobe_exclusion_deg)
            or not 0.0 < main_lobe_exclusion_deg < 90.0
        ):
            raise ValueError(
                "Main-lobe exclusion angle must be between 0 and 90 degrees."
            )
        half_width_ux = math.sin(math.radians(main_lobe_exclusion_deg))
        half_width_uy = half_width_ux
        main_lobe = valid_directions & (
            dot_products >= math.cos(math.radians(main_lobe_exclusion_deg))
        )

    sidelobes = valid_directions & ~main_lobe
    magnitude = np.abs(field)
    target_amplitude = abs(
        np.exp(
            1j
            * (
                np.asarray(phase_bins, dtype=float) * acoustics.TWO_PI / PHASE_BINS
                + acoustics.TWO_PI
                / wavelength
                * ((positions - center) @ target_direction)
            )
        ).sum()
    )
    if np.any(main_lobe):
        main_peak_index = np.unravel_index(
            np.argmax(np.where(main_lobe, magnitude, -np.inf)), magnitude.shape
        )
        sampled_main_peak = float(magnitude[main_peak_index])
        sampled_main_ux = float(ux_grid[main_peak_index])
        sampled_main_uy = float(uy_grid[main_peak_index])
        sampled_main_uz = math.sqrt(
            max(0.0, 1.0 - sampled_main_ux**2 - sampled_main_uy**2)
        )
        sampled_main_azimuth = math.degrees(
            math.atan2(sampled_main_uy, sampled_main_ux)
        )
        sampled_main_beta = math.degrees(math.acos(sampled_main_uz))
    else:
        sampled_main_peak = 0.0
        sampled_main_azimuth = azimuth_degrees
        sampled_main_beta = off_axis_degrees

    if target_amplitude > sampled_main_peak:
        main_peak = target_amplitude
        main_peak_azimuth = azimuth_degrees
        main_peak_beta = off_axis_degrees
    else:
        main_peak = sampled_main_peak
        main_peak_azimuth = sampled_main_azimuth
        main_peak_beta = sampled_main_beta

    if not np.any(sidelobes):
        return {
            "sll_db": float("-inf"),
            "main_lobe_peak": main_peak,
            "main_lobe_peak_azimuth_deg": main_peak_azimuth,
            "main_lobe_peak_beta_deg": main_peak_beta,
            "sidelobe_peak": 0.0,
            "sidelobe_azimuth_deg": float("nan"),
            "sidelobe_beta_deg": float("nan"),
            "likely_grating_lobe": False,
            "main_lobe_exclusion_deg": main_lobe_exclusion_deg,
            "main_lobe_half_width_ux": half_width_ux,
            "main_lobe_half_width_uy": half_width_uy,
        }

    sidelobe_magnitudes = np.where(sidelobes, magnitude, -np.inf)
    peak_row, peak_column = np.unravel_index(
        np.argmax(sidelobe_magnitudes), magnitude.shape
    )
    sidelobe_peak = float(magnitude[peak_row, peak_column])
    ratio = sidelobe_peak / max(main_peak, np.finfo(float).tiny)
    sll_db = 20.0 * math.log10(max(ratio, np.finfo(float).tiny))
    sidelobe_ux = float(ux_grid[peak_row, peak_column])
    sidelobe_uy = float(uy_grid[peak_row, peak_column])
    target_ux, target_uy = target_direction[:2]
    spacing = float(
        np.mean(
            [
                np.median(np.diff(np.unique(positions[:, axis])))
                for axis in (0, 1)
            ]
        )
    )
    grating_step = wavelength / spacing
    order_x = round((sidelobe_ux - target_ux) / grating_step)
    order_y = round((sidelobe_uy - target_uy) / grating_step)
    residual = math.hypot(
        sidelobe_ux - target_ux - order_x * grating_step,
        sidelobe_uy - target_uy - order_y * grating_step,
    )
    grid_spacing = 2.0 / (direction_cosine_samples - 1)
    likely_grating = (order_x != 0 or order_y != 0) and residual <= max(
        2.0 * grid_spacing, 0.03
    )
    sidelobe_azimuth = math.degrees(math.atan2(sidelobe_uy, sidelobe_ux))
    sidelobe_beta = math.degrees(
        math.acos(
            math.sqrt(max(0.0, 1.0 - sidelobe_ux**2 - sidelobe_uy**2))
        )
    )
    return {
        "sll_db": sll_db,
        "main_lobe_peak": main_peak,
        "main_lobe_peak_azimuth_deg": main_peak_azimuth,
        "main_lobe_peak_beta_deg": main_peak_beta,
        "sidelobe_peak": sidelobe_peak,
        "sidelobe_azimuth_deg": sidelobe_azimuth,
        "sidelobe_beta_deg": sidelobe_beta,
        "likely_grating_lobe": bool(likely_grating),
        "main_lobe_exclusion_deg": main_lobe_exclusion_deg,
        "main_lobe_half_width_ux": half_width_ux,
        "main_lobe_half_width_uy": half_width_uy,
    }


def plot_array_factor(
    phase_bins,
    target_azimuth_degrees: float,
    target_off_axis_degrees: float,
    *,
    frequency_hz=acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed=acoustics.DEFAULT_SOUND_SPEED_MPS,
) -> None:
    """Display the ideal theoretical array factor; matplotlib is optional."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Install matplotlib to use --plot-array-factor."
        ) from error

    azimuth_grid, off_axis_grid, levels_db = compute_array_factor_pattern(
        phase_bins,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
    )
    peak_index = np.unravel_index(np.argmax(levels_db), levels_db.shape)
    peak_azimuth = azimuth_grid[peak_index]
    peak_off_axis = off_axis_grid[peak_index]
    wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
    peak_direction = acoustics.direction_from_angles(peak_azimuth, peak_off_axis)
    print(
        "Theoretical sampled array-factor peak: "
        f"azimuth={peak_azimuth:.1f} deg, off-axis={peak_off_axis:.1f} deg, "
        f"direction=({peak_direction[0]:.3f}, {peak_direction[1]:.3f}, "
        f"{peak_direction[2]:.3f})"
    )
    print(
        "The planar array factor is front/back ambiguous; this is not a physical "
        "measurement. Ideal bin mapping is assumed for this plot."
    )
    print(
        f"Wavelength={wavelength * 1000:.3f} mm; nominal spacing="
        f"{SPACING * 1000:.1f} mm ({SPACING / wavelength:.3f} wavelengths)."
    )

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = axis.pcolormesh(
        azimuth_grid,
        off_axis_grid,
        levels_db,
        shading="auto",
        cmap="viridis",
        vmin=-40.0,
        vmax=0.0,
    )
    axis.plot(
        target_azimuth_degrees,
        target_off_axis_degrees,
        marker="x",
        color="red",
        markersize=8,
        label="Requested direction",
    )
    axis.set_xlabel("Azimuth (degrees, from +x toward +y)")
    axis.set_ylabel("Off-axis angle from +z (degrees)")
    axis.set_title("Theoretical Idealized Array Factor (not a measurement)")
    figure.colorbar(image, ax=axis, label="Normalized array factor (dB)")
    axis.legend()
    plt.show()
