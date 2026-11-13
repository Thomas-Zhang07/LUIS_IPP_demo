"""Two-board geometry, frame routing and combined beam analysis.

The two 16x16 boards are coplanar and side by side along x.  Board A occupies
``x = 0 .. 150 mm``, and board B is shifted by half a lattice pitch in both x
and y:

    x_B(c) = 150 mm + gap + 5 mm + 10 mm * c
    y_B(r) = 5 mm + 10 mm * r

The half-pitch offset cancels the axial grating lobes that a single board has
near 59 degrees, while the broadside fields of the two boards add coherently.

This module is the host counterpart of the secondary-board firmware:
``0xC0`` enables board A (the primary) and ``0xC1`` enables board B (the
secondary).  Each board receives its own 258-byte frame built in the same
wire order as the single-board demo (``emitters.EMITTERS_ORDER``).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

import acoustics
import emitters
import protocol
from directivity import amplitude_code_for_weight, amplitude_from_code

DEFAULT_GAP_M = 0.000
DEFAULT_OFFSET_X_M = 0.005
DEFAULT_OFFSET_Y_M = 0.005

PRIMARY_SELECT_BYTE = 0xC0
SECONDARY_SELECT_BYTE = 0xC1

PHASE_BINS = protocol.PHASE_BINS
OFF_VALUE = protocol.OFF_VALUE
TWO_PI = 2.0 * math.pi

# Nominal broadside grating-lobe angle of the 10 mm lattice at 40 kHz.
GRATING_BETA_DEG = math.degrees(
    math.asin(
        (acoustics.DEFAULT_SOUND_SPEED_MPS / acoustics.DEFAULT_FREQUENCY_HZ)
        / emitters.SPACING
    )
)


def _bessel_j1(x):
    """Bessel J1 for a numpy array (power series, no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    term = x / 2.0
    total = term.copy()
    for m in range(1, 40):
        term = term * (-(x * x) / 4.0) / (m * (m + 1))
        total = total + term
        if np.max(np.abs(term)) < 1e-14:
            break
    return total


def piston_directivity(beta_degrees, diameter_m, wavelength_m):
    """Pressure directivity of a circular piston in a baffle.

    Returns ``2*J1(x)/x`` with ``x = pi*diameter*sin(beta)/wavelength``,
    normalised to 1 on axis.  This is the same for every transducer, so it
    only scales each observation direction, not the per-element weights.
    """
    beta = np.asarray(beta_degrees, dtype=float)
    x = (math.pi * float(diameter_m) / float(wavelength_m)) * np.sin(
        np.radians(beta)
    )
    x = np.maximum(x, 1e-12)
    return 2.0 * _bessel_j1(x) / x


@dataclass(frozen=True)
class BoardGeometry:
    """Positions of the two boards in one coordinate frame (meters).

    Args:
        gap_m: clear extra gap between the two PCB edges (element-center
            distance across the seam minus the 5 mm offset).
        offset_x_m: board B shift along x relative to the nominal lattice.
        offset_y_m: board B shift along y.
        spacing_m: lattice pitch, 10 mm on this hardware.
        side: emitters per side, 16.
    """

    gap_m: float = DEFAULT_GAP_M
    offset_x_m: float = DEFAULT_OFFSET_X_M
    offset_y_m: float = DEFAULT_OFFSET_Y_M
    spacing_m: float = emitters.SPACING
    side: int = emitters.N_SIDE
    frequency_hz: float = acoustics.DEFAULT_FREQUENCY_HZ
    sound_speed: float = acoustics.DEFAULT_SOUND_SPEED_MPS

    def __post_init__(self) -> None:
        for value, label in (
            (self.gap_m, "gap"),
            (self.offset_x_m, "offset_x"),
            (self.offset_y_m, "offset_y"),
            (self.spacing_m, "spacing"),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{label} must be a number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite.")
        if self.gap_m < 0.0:
            raise ValueError("gap must be non-negative.")
        if self.spacing_m <= 0.0:
            raise ValueError("spacing must be positive.")
        if self.side < 1:
            raise ValueError("side must be at least 1.")
        acoustics.validate_acoustic_params(self.frequency_hz, self.sound_speed)

    @property
    def n_per_board(self) -> int:
        return self.side * self.side

    @property
    def board_b_origin(self) -> np.ndarray:
        return np.array(
            [
                (self.side - 1) * self.spacing_m
                + self.gap_m
                + self.offset_x_m,
                self.offset_y_m,
                0.0,
            ]
        )

    def _grid(self, origin: np.ndarray) -> np.ndarray:
        positions = np.zeros((self.n_per_board, 3), dtype=float)
        for row in range(self.side):
            for col in range(self.side):
                index = row * self.side + col
                positions[index] = (
                    origin[0] + col * self.spacing_m,
                    origin[1] + row * self.spacing_m,
                    0.0,
                )
        return positions

    def board_a_positions(self) -> np.ndarray:
        return self._grid(np.zeros(3))

    def board_b_positions(self) -> np.ndarray:
        return self._grid(self.board_b_origin)

    def positions(self) -> np.ndarray:
        """Return the 2*N logical positions: board A then board B."""
        return np.vstack((self.board_a_positions(), self.board_b_positions()))

    @property
    def reference_center(self) -> np.ndarray:
        return self.positions().mean(axis=0)

    @property
    def aperture_m(self) -> tuple[float, float]:
        positions = self.positions()
        return (
            float(np.ptp(positions[:, 0])),
            float(np.ptp(positions[:, 1])),
        )

    def summary(self) -> str:
        aperture_x, aperture_y = self.aperture_m
        return (
            f"board A x=0..{(self.side - 1) * self.spacing_m * 1e3:.0f} mm, "
            f"board B origin x={self.board_b_origin[0] * 1e3:.1f} mm "
            f"y={self.board_b_origin[1] * 1e3:.1f} mm, "
            f"gap={self.gap_m * 1e3:.1f} mm, "
            f"offset=({self.offset_x_m * 1e3:.1f}, "
            f"{self.offset_y_m * 1e3:.1f}) mm, "
            f"combined aperture {aperture_x * 1e3:.0f} x "
            f"{aperture_y * 1e3:.0f} mm"
        )


def board_phase_bins(
    geometry: BoardGeometry,
    *,
    azimuth_degrees: float = 0.0,
    off_axis_degrees: float = 0.0,
    focus_z_m: float | None = None,
    focus_xy_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (board A bins, board B bins) in each board's logical order.

    With ``focus_z_m`` the phases focus a spherical wave on a point at that
    height; otherwise they steer a plane wave.  Both boards use the same
    global reference center for steering so their relative phase is the
    physical path difference, not an arbitrary per-board constant.
    """
    if focus_z_m is not None:
        x, y = (0.0, 0.0) if focus_xy_m is None else focus_xy_m
        target = (float(x), float(y), float(focus_z_m))
        bins_a = acoustics.focus_phase_bins(
            target,
            frequency_hz=geometry.frequency_hz,
            sound_speed=geometry.sound_speed,
            emitters_pos=geometry.board_a_positions(),
        )
        bins_b = acoustics.focus_phase_bins(
            target,
            frequency_hz=geometry.frequency_hz,
            sound_speed=geometry.sound_speed,
            emitters_pos=geometry.board_b_positions(),
        )
        return bins_a, bins_b

    kwargs = dict(
        frequency_hz=geometry.frequency_hz,
        sound_speed=geometry.sound_speed,
        reference_center=geometry.reference_center,
    )
    bins_a = acoustics.steering_phase_bins(
        azimuth_degrees,
        off_axis_degrees,
        emitters_pos=geometry.board_a_positions(),
        **kwargs,
    )
    bins_b = acoustics.steering_phase_bins(
        azimuth_degrees,
        off_axis_degrees,
        emitters_pos=geometry.board_b_positions(),
        **kwargs,
    )
    return bins_a, bins_b


def _window_1d(u: np.ndarray, window: str, *, sigma: float, tukey_alpha: float):
    if window == "none":
        return np.ones_like(u)
    if window == "hann":
        return 0.5 - 0.5 * np.cos(TWO_PI * u)
    if window == "gaussian":
        if not 0.0 < sigma <= 1.0:
            raise ValueError("Gaussian sigma must be in (0, 1].")
        return np.exp(-0.5 * ((u - 0.5) / sigma) ** 2)
    if window == "blackman":
        return (
            0.42
            - 0.5 * np.cos(TWO_PI * u)
            + 0.08 * np.cos(2.0 * TWO_PI * u)
        )
    if window == "tukey":
        if not 0.0 <= tukey_alpha <= 1.0:
            raise ValueError("Tukey alpha must be in [0, 1].")
        w = np.ones_like(u)
        left = u < tukey_alpha / 2.0
        right = u > 1.0 - tukey_alpha / 2.0
        w[left] = 0.5 * (
            1.0 + np.cos(math.pi * (2.0 * u[left] / tukey_alpha - 1.0))
        )
        w[right] = 0.5 * (
            1.0
            + np.cos(
                math.pi * (2.0 * u[right] / tukey_alpha - 2.0 / tukey_alpha + 1.0)
            )
        )
        return w
    raise ValueError(
        "window must be one of none, hann, gaussian, blackman, tukey."
    )


def amplitude_window_weights(
    geometry: BoardGeometry,
    window: str = "none",
    *,
    gaussian_sigma: float = 0.35,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """Return a separable 2-D taper over the full 512-element envelope."""
    positions = geometry.positions()
    x = positions[:, 0]
    y = positions[:, 1]
    ux = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
    uy = (y - y.min()) / max(float(np.ptp(y)), 1e-12)
    wx = _window_1d(
        ux, window, sigma=gaussian_sigma, tukey_alpha=tukey_alpha
    )
    wy = _window_1d(
        uy, window, sigma=gaussian_sigma, tukey_alpha=tukey_alpha
    )
    return np.clip(wx * wy, 0.0, 1.0)


def amplitude_codes_from_weights(weights) -> np.ndarray:
    """Map 512 (or 2N) relative weights to firmware duty codes."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise ValueError("Weights must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Weights must be between 0 and 1.")
    return np.asarray(
        [amplitude_code_for_weight(float(value)) for value in values],
        dtype=int,
    )


def board_payload(select_byte: int, frame: bytes) -> bytes:
    """Prefix one board frame with its board-select byte."""
    if select_byte not in (PRIMARY_SELECT_BYTE, SECONDARY_SELECT_BYTE):
        raise ValueError("Board-select byte must be 0xC0 or 0xC1.")
    return bytes([select_byte]) + bytes(frame)


def split_by_board(values) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size % 2 != 0:
        raise ValueError("Expected an even, one-dimensional per-element array.")
    half = array.size // 2
    return array[:half], array[half:]


def build_payloads(
    geometry: BoardGeometry,
    phases_a,
    phases_b,
    codes_a=None,
    codes_b=None,
) -> tuple[bytes, bytes]:
    """Build the two board payloads (select byte + 258-byte frame)."""
    n = geometry.n_per_board
    if codes_a is None:
        codes_a = np.zeros(n, dtype=int)
    if codes_b is None:
        codes_b = np.zeros(n, dtype=int)
    frame_a = protocol.build_amplitude_frame(
        np.asarray(phases_a), np.asarray(codes_a),
        transmit_order=emitters.EMITTERS_ORDER,
    )
    frame_b = protocol.build_amplitude_frame(
        np.asarray(phases_b), np.asarray(codes_b),
        transmit_order=emitters.EMITTERS_ORDER,
    )
    return (
        board_payload(PRIMARY_SELECT_BYTE, frame_a),
        board_payload(SECONDARY_SELECT_BYTE, frame_b),
    )


class BoardLink:
    """Two serial ports, one per board, written in parallel.

    ``serial_factory`` and ``sleep`` are resolved at call time so tests can
    inject fakes, exactly like :func:`protocol.open_serial`.
    """

    def __init__(
        self,
        port_a: str,
        port_b: str,
        *,
        baudrate: int = protocol.BAUD_RATE,
        timeout: float = 2.0,
        write_timeout: float = 2.0,
        settle_s: float = 0.3,
        serial_factory=None,
        sleep=None,
    ):
        if not port_a or not port_b:
            raise ValueError("Two serial ports are required.")
        if port_a == port_b:
            raise ValueError("Board A and board B need different ports.")
        self.port_a = port_a
        self.port_b = port_b
        self.baudrate = int(baudrate)
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.settle_s = settle_s
        self._factory = serial_factory
        self._sleep = sleep
        self._connection_a = None
        self._connection_b = None

    def __enter__(self) -> "BoardLink":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._connection_a is not None or self._connection_b is not None:
            raise RuntimeError("BoardLink is already open.")
        stack = [
            protocol.open_serial(
                self.port_a,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
                settle_s=self.settle_s,
                serial_factory=self._factory,
                sleep=self._sleep,
            ),
            protocol.open_serial(
                self.port_b,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
                settle_s=self.settle_s,
                serial_factory=self._factory,
                sleep=self._sleep,
            ),
        ]
        try:
            self._connection_a = stack[0].__enter__()
            self._connection_b = stack[1].__enter__()
        except Exception:
            for context in stack:
                try:
                    context.__exit__(None, None, None)
                except Exception:
                    pass
            self._connection_a = self._connection_b = None
            raise
        self._contexts = stack

    def close(self) -> None:
        contexts = getattr(self, "_contexts", ())
        for context in contexts:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
        self._contexts = ()
        self._connection_a = self._connection_b = None

    def send_trim(self, steps: int) -> None:
        """Send the fine-trim command to board B (the secondary) only."""
        if self._connection_b is None:
            raise RuntimeError("BoardLink is not open.")
        payload = protocol.build_trim_command(steps)
        written = self._connection_b.write(payload)
        self._connection_b.flush()
        if written != len(payload):
            raise OSError(
                f"Incomplete trim write: sent {written} of {len(payload)} bytes."
            )

    def send(self, payload_a: bytes, payload_b: bytes) -> None:
        """Write both payloads concurrently.

        A barrier makes the two writes start together.  The frames are still
        staged on each FPGA until its own 0xFD trailer, so the remaining skew
        is the difference between the two UARTs, not the host loop.
        """
        if self._connection_a is None or self._connection_b is None:
            raise RuntimeError("BoardLink is not open.")
        payload_a = bytes(payload_a)
        payload_b = bytes(payload_b)
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker(connection, payload):
            try:
                barrier.wait(timeout=2.0)
                written = connection.write(payload)
                connection.flush()
                if written != len(payload):
                    raise OSError(
                        f"Incomplete write: sent {written} of {len(payload)} bytes."
                    )
            except BaseException as error:  # propagate to the caller
                errors.append(error)

        thread_a = threading.Thread(
            target=worker, args=(self._connection_a, payload_a), daemon=True
        )
        thread_b = threading.Thread(
            target=worker, args=(self._connection_b, payload_b), daemon=True
        )
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
        if errors:
            raise errors[0]


class CombinedBeamAnalyzer:
    """Ideal far-field model for the 512-element two-board aperture.

    Unlike the single-board :class:`directivity.BeamAnalyzer` this model
    keeps the whole visible hemisphere in the sidelobe mask: the half-pitch
    offset is precisely what cancels the axial grating lobes, so they must
    not be excluded from the objective.
    """

    def __init__(
        self,
        positions,
        *,
        azimuth_degrees: float = 0.0,
        off_axis_degrees: float = 0.0,
        frequency_hz: float = acoustics.DEFAULT_FREQUENCY_HZ,
        sound_speed: float = acoustics.DEFAULT_SOUND_SPEED_MPS,
        samples: int = 121,
        main_lobe_safety_factor: float = 1.5,
        element_diameter_m: float | None = 0.010,
        side_start_deg: float = 60.0,
    ):
        acoustics.validate_acoustic_params(frequency_hz, sound_speed)
        acoustics.validate_steering_angles(azimuth_degrees, off_axis_degrees)
        if not 0.0 <= off_axis_degrees <= 90.0:
            raise ValueError("Analysis is limited to the +z hemisphere.")
        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
            raise ValueError("samples must be an integer.")
        if int(samples) < 5:
            raise ValueError("samples must be at least 5.")

        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")
        if self.positions.shape[0] < 2 or not np.all(np.isfinite(self.positions)):
            raise ValueError("positions must contain at least two finite points.")

        self.frequency_hz = float(frequency_hz)
        self.sound_speed = float(sound_speed)
        self.wavelength = acoustics.acoustic_wavelength(frequency_hz, sound_speed)
        self.azimuth_degrees = float(azimuth_degrees)
        self.off_axis_degrees = float(off_axis_degrees)
        self.direction = acoustics.direction_from_angles(
            azimuth_degrees, off_axis_degrees
        )
        if element_diameter_m is not None and element_diameter_m <= 0.0:
            raise ValueError("element_diameter_m must be positive or None.")
        self.element_diameter_m = element_diameter_m
        if not 0.0 < float(side_start_deg) <= 90.0:
            raise ValueError("side_start_deg must be in (0, 90].")
        self.side_start_deg = float(side_start_deg)

        self._center = self.positions.mean(axis=0)
        self._centered = self.positions - self._center
        wave_number = TWO_PI / self.wavelength
        self._k = wave_number
        self._target_vector = np.exp(
            1j * wave_number * (self._centered @ self.direction)
        )
        if self.element_diameter_m is not None:
            beta_target = math.degrees(
                math.acos(max(-1.0, min(1.0, self.direction[2])))
            )
            self._target_vector = self._target_vector * float(
                piston_directivity(
                    beta_target, self.element_diameter_m, self.wavelength
                )
            )

        samples = int(samples)
        u_values = np.linspace(-1.0, 1.0, samples)
        ux_grid, uy_grid = np.meshgrid(u_values, u_values)
        valid = ux_grid**2 + uy_grid**2 <= 1.0
        uz_valid = np.sqrt(np.maximum(0.0, 1.0 - ux_grid**2 - uy_grid**2))
        directions = np.column_stack(
            (ux_grid[valid], uy_grid[valid], uz_valid[valid])
        )
        self._dirs = directions
        self._steering_matrix = np.exp(
            1j * wave_number * (self._centered @ directions.T)
        ).T
        if self.element_diameter_m is not None:
            beta_grid = np.degrees(
                np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
            )
            self._steering_matrix = self._steering_matrix * piston_directivity(
                beta_grid, self.element_diameter_m, self.wavelength
            )[:, None]

        aperture_x = float(np.ptp(self.positions[:, 0]))
        aperture_y = float(np.ptp(self.positions[:, 1]))
        if aperture_x <= 0.0 or aperture_y <= 0.0:
            raise ValueError("Array apertures in x and y must be positive.")
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
        self._main_mask = (
            np.abs(ux_valid - self.direction[0])
            <= self.main_lobe_half_width_ux
        ) & (
            np.abs(uy_valid - self.direction[1])
            <= self.main_lobe_half_width_uy
        )
        if not np.any(~self._main_mask):
            raise ValueError("Direction grid contains only main-lobe points.")
        self._sidelobe_mask = ~self._main_mask
        beta_all = np.degrees(
            np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
        )
        self._far_mask = (~self._main_mask) & (
            beta_all >= self.side_start_deg - 1e-12
        )
        if not np.any(self._far_mask):
            self._far_mask = self._sidelobe_mask

    # -- weights -------------------------------------------------------
    def _weights(self, phase_bins, amplitude_codes=None, active_mask=None):
        bins = np.asarray(phase_bins)
        if bins.shape != (self.positions.shape[0],):
            raise ValueError(
                f"Expected {self.positions.shape[0]} phase bins, got {bins.shape}."
            )
        if not np.issubdtype(bins.dtype, np.integer):
            raise ValueError("Phase bins must be integers.")
        if np.any((bins < 0) | (bins > OFF_VALUE)):
            raise ValueError("Phase bins must be 0..15 or 0x10 (OFF).")
        if amplitude_codes is None:
            codes = np.zeros(self.positions.shape[0], dtype=int)
        else:
            codes = np.asarray(amplitude_codes)
            if codes.shape != bins.shape:
                raise ValueError("Amplitude codes must match the phase shape.")
            if not np.issubdtype(codes.dtype, np.integer):
                raise ValueError("Amplitude codes must be integers.")
            if np.any((codes < 0) | (codes > OFF_VALUE)):
                raise ValueError("Amplitude codes must be 0..15 or 0x10.")
        amplitudes = np.zeros_like(codes, dtype=float)
        for index, code in enumerate(codes):
            if code != OFF_VALUE:
                amplitudes[index] = amplitude_from_code(int(code))
        active = (bins != OFF_VALUE) & (amplitudes > 0.0)
        if active_mask is not None:
            mask = np.asarray(active_mask)
            if mask.shape != bins.shape or mask.dtype != bool:
                raise ValueError("active_mask must be boolean with one value per element.")
            active &= mask
        if not np.any(active):
            raise ValueError("At least one element must be active.")
        weights = np.zeros(self.positions.shape[0], dtype=complex)
        weights[active] = amplitudes[active] * np.exp(
            1j * np.asarray(bins[active], dtype=float) * TWO_PI / PHASE_BINS
        )
        return weights, active, amplitudes

    def field(self, phase_bins, amplitude_codes=None, active_mask=None):
        weights, _, _ = self._weights(phase_bins, amplitude_codes, active_mask)
        return self.field_of_weights(weights)

    def field_of_weights(self, weights):
        return self._steering_matrix @ weights

    def field_at(self, phase_bins, azimuth_degrees, off_axis_degrees,
                 amplitude_codes=None):
        weights, _, _ = self._weights(phase_bins, amplitude_codes)
        direction = acoustics.direction_from_angles(
            azimuth_degrees, off_axis_degrees
        )
        vector = np.exp(1j * self._k * (self._centered @ direction))
        if self.element_diameter_m is not None:
            vector = vector * float(
                piston_directivity(
                    off_axis_degrees, self.element_diameter_m, self.wavelength
                )
            )
        return complex(vector @ weights)

    # -- metrics -------------------------------------------------------
    def metrics(self, phase_bins, amplitude_codes=None, active_mask=None) -> dict:
        weights, active, _ = self._weights(
            phase_bins, amplitude_codes, active_mask
        )
        main = complex(self._target_vector @ weights)
        ratio = np.abs(self._steering_matrix @ weights) / max(abs(main), 1e-30)

        masked = np.where(self._sidelobe_mask, ratio, -np.inf)
        peak_index = int(np.argmax(masked))
        peak = float(masked[peak_index])
        direction = self._dirs[peak_index]
        peak_azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        peak_beta = math.degrees(math.acos(max(-1.0, min(1.0, direction[2]))))

        far_masked = np.where(self._far_mask, ratio, -np.inf)
        far_index = int(np.argmax(far_masked))
        far_peak = float(far_masked[far_index])
        far_direction = self._dirs[far_index]
        far_azimuth = math.degrees(
            math.atan2(far_direction[1], far_direction[0])
        )
        far_beta = math.degrees(
            math.acos(max(-1.0, min(1.0, far_direction[2])))
        )

        grating_levels = {}
        for azimuth in (0.0, 90.0, 180.0, 270.0):
            level = self.field_at(
                phase_bins, azimuth, GRATING_BETA_DEG, amplitude_codes
            )
            grating_levels[azimuth] = 20.0 * math.log10(
                max(abs(level) / max(abs(main), 1e-30), 1e-12)
            )

        return {
            "main_lobe_amplitude": abs(main),
            "main_lobe_gain_db": 20.0 * math.log10(
                max(abs(main) / self.positions.shape[0], 1e-30)
            ),
            "active_elements": int(np.count_nonzero(active)),
            "peak_sidelobe_db": 20.0 * math.log10(max(peak, 1e-12)),
            "peak_sidelobe_azimuth_deg": peak_azimuth,
            "peak_sidelobe_beta_deg": peak_beta,
            "far_sidelobe_db": 20.0 * math.log10(max(far_peak, 1e-12)),
            "far_sidelobe_azimuth_deg": far_azimuth,
            "far_sidelobe_beta_deg": far_beta,
            "side_start_deg": self.side_start_deg,
            "grating_lobe_db": grating_levels,
            "main_lobe_exclusion_deg": self.main_lobe_exclusion_deg,
        }

    def level_at(self, phase_bins, azimuth, beta, amplitude_codes=None) -> float:
        weights, _, _ = self._weights(phase_bins, amplitude_codes)
        main = complex(self._target_vector @ weights)
        level = self.field_at(phase_bins, azimuth, beta, amplitude_codes)
        return 20.0 * math.log10(max(abs(level) / max(abs(main), 1e-30), 1e-12))

    def half_power_width_deg(
        self, phase_bins, amplitude_codes=None, *, axis: str = "beta",
        step_deg: float = 0.2,
    ) -> float:
        """Return the -3 dB width of the main lobe on one angular cut."""
        weights, _, _ = self._weights(phase_bins, amplitude_codes)
        main = abs(complex(self._target_vector @ weights))
        if main <= 0.0:
            return float("nan")

        def amplitude_at(azimuth, beta):
            direction = acoustics.direction_from_angles(azimuth, beta)
            vector = np.exp(1j * self._k * (self._centered @ direction))
            if self.element_diameter_m is not None:
                vector = vector * float(
                    piston_directivity(
                        beta, self.element_diameter_m, self.wavelength
                    )
                )
            return abs(vector @ weights) / main

        target_azimuth = self.azimuth_degrees
        target_beta = self.off_axis_degrees
        if axis == "beta":
            values = np.arange(0.0, 90.0 + step_deg, step_deg)
            curve = [amplitude_at(target_azimuth, beta) for beta in values]
            center = int(np.argmin(np.abs(values - target_beta)))
        elif axis == "azimuth":
            values = np.arange(target_azimuth - 90.0,
                               target_azimuth + 90.0 + step_deg, step_deg)
            curve = [amplitude_at(azimuth, target_beta) for azimuth in values]
            center = int(np.argmin(np.abs(values - target_azimuth)))
        else:
            raise ValueError("axis must be 'beta' or 'azimuth'.")

        threshold = 1.0 / math.sqrt(2.0)
        left = center
        while left > 0 and curve[left - 1] >= threshold:
            left -= 1
        right = center
        while right < len(curve) - 1 and curve[right + 1] >= threshold:
            right += 1
        return float(values[right] - values[left])

    # -- optimizer -----------------------------------------------------
    def optimize(
        self,
        phase_bins,
        amplitude_codes=None,
        *,
        max_sweeps: int = 4,
        seed: int = 0,
        null_directions=None,
        null_weight: float = 1.0,
        side_start_deg: float | None = None,
        near_weight: float = 0.5,
    ):
        """Coordinate descent on phase bins, peak sidelobe objective.

        Amplitude codes are held fixed.  ``null_directions`` are optional
        ``(azimuth, beta)`` pairs that receive an extra penalty, on top of
        the peak-sidelobe objective.
        """
        if isinstance(max_sweeps, bool) or not isinstance(
            max_sweeps, (int, np.integer)
        ):
            raise ValueError("max_sweeps must be an integer.")
        if max_sweeps < 0:
            raise ValueError("max_sweeps must be non-negative.")
        if not math.isfinite(null_weight) or null_weight < 0.0:
            raise ValueError("null_weight must be finite and non-negative.")

        bins = np.asarray(phase_bins).copy()
        weights, active, amplitudes = self._weights(bins, amplitude_codes)
        active_indices = np.flatnonzero(active)

        null_indices = np.empty(0, dtype=int)
        if null_directions:
            indices = []
            for item in null_directions:
                if len(item) != 2:
                    raise ValueError("Each null direction must be (azimuth, beta).")
                azimuth, beta = float(item[0]), float(item[1])
                acoustics.validate_steering_angles(azimuth, beta)
                if not 0.0 <= beta <= 90.0:
                    raise ValueError("Nulls must be in the front hemisphere.")
                direction = acoustics.direction_from_angles(azimuth, beta)
                indices.append(int(np.argmax(self._dirs @ direction)))
            null_indices = np.asarray(indices, dtype=int)
        weight_null = float(null_weight) if null_indices.size else 0.0

        field = self._steering_matrix @ weights
        main = complex(self._target_vector @ weights)
        if not math.isfinite(near_weight) or near_weight < 0.0:
            raise ValueError("near_weight must be finite and non-negative.")
        beta_all = np.degrees(
            np.arccos(np.clip(self._dirs[:, 2], -1.0, 1.0))
        )
        if side_start_deg is None:
            side_mask = self._sidelobe_mask
            near_mask = np.zeros_like(side_mask)
        else:
            if not 0.0 < float(side_start_deg) <= 90.0:
                raise ValueError("side_start_deg must be in (0, 90].")
            side_mask = (~self._main_mask) & (
                beta_all >= float(side_start_deg) - 1e-12
            )
            near_mask = (~self._main_mask) & (
                beta_all < float(side_start_deg) - 1e-12
            )
            if not np.any(side_mask):
                side_mask = self._sidelobe_mask
        side_indices = np.flatnonzero(side_mask)
        near_indices = np.flatnonzero(near_mask)
        candidate_phases = np.exp(
            1j * np.arange(PHASE_BINS, dtype=float) * TWO_PI / PHASE_BINS
        )

        def score(fields, mains):
            denominator = np.maximum(np.abs(mains), 1e-30)[:, None]
            ratios = np.abs(fields[:, side_indices]) / denominator
            peak = ratios.max(axis=1)
            energy = np.mean(ratios * ratios, axis=1)
            if near_indices.size and near_weight:
                near_ratios = np.abs(fields[:, near_indices]) / denominator
                peak = peak + float(near_weight) * near_ratios.max(axis=1)
            if weight_null and null_indices.size:
                null_ratios = np.abs(fields[:, null_indices]) / denominator
                peak = peak + weight_null * null_ratios.max(axis=1)
            return peak, energy

        current_peak, current_energy = score(
            field[None, :], np.array([main])
        )
        current_peak = float(current_peak[0])
        current_energy = float(current_energy[0])

        rng = np.random.default_rng(seed)
        sweeps_run = 0
        changed_total = 0
        for sweep in range(int(max_sweeps)):
            changed = 0
            for index in rng.permutation(active_indices):
                column = self._steering_matrix[:, index]
                old_phase = weights[index] / amplitudes[index]
                delta = amplitudes[index] * (candidate_phases - old_phase)
                candidate_fields = field[None, :] + delta[:, None] * column[None, :]
                candidate_mains = main + delta * self._target_vector[index]
                peaks, energies = score(candidate_fields, candidate_mains)
                best = int(np.lexsort((energies, peaks))[0])
                better = peaks[best] < current_peak - 1e-12 or (
                    abs(peaks[best] - current_peak) <= 1e-12
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

        return bins, {
            "sweeps_run": sweeps_run,
            "changed_elements": changed_total,
        }
