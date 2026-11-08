#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Steer a planar 16x16 ultrasonic array with azimuth and off-axis angles.

Azimuth is measured in the x-y plane from +x toward +y. Off-axis angle is
measured from +z. The phase sign follows the existing one-dimensional script
and assumes an exp(i*omega*t - i*k*r) propagation convention. The firmware's
phase-bin mapping and the physical beam direction must be independently
verified.
"""

import argparse
import contextlib
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

if __package__ in (None, ""):
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

from array_factor import (
    physical_phase_grid,
    plot_array_factor,
    print_physical_phase_map,
)
from emitters import EMITTERS_ORDER, EMITTERS_POS, N_EMITTERS

BAUDRATE = 38400
HEADER = 0xFE
FOOTER = 0xFD
PHASE_BINS = 16
SPEED_OF_SOUND = 343.0
DEFAULT_CARRIER_FREQUENCY = 40000.0
TWO_PI = 2.0 * math.pi
ROTATION_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)
TWO_DIRECTION_ANGLES_DEG = (0.0, 180.0)


def validate_angles(azimuth_degrees: float, off_axis_degrees: float) -> None:
    """Validate periodic azimuth in [-360, 360] and off-axis angle in [0, 180]."""
    if not math.isfinite(azimuth_degrees) or not math.isfinite(off_axis_degrees):
        raise ValueError("Angles must be finite numbers.")
    if not -360.0 <= azimuth_degrees <= 360.0:
        raise ValueError("Azimuth must be between -360 and 360 degrees.")
    if not 0.0 <= off_axis_degrees <= 180.0:
        raise ValueError("Off-axis angle must be between 0 and 180 degrees.")


def compute_direction(azimuth_degrees: float, off_axis_degrees: float) -> np.ndarray:
    """Return the unit direction for the documented spherical convention."""
    validate_angles(azimuth_degrees, off_axis_degrees)
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
    """Convert the legacy x-z angle to (azimuth, off-axis) degrees."""
    if not math.isfinite(angle_degrees) or not 0.0 <= angle_degrees <= 180.0:
        raise ValueError("Legacy angle must be between 0 and 180 degrees.")
    if angle_degrees <= 90.0:
        return 0.0, 90.0 - angle_degrees
    return 180.0, angle_degrees - 90.0


def validate_emitter_positions(emitters_pos: np.ndarray = EMITTERS_POS) -> np.ndarray:
    """Return validated emitter coordinates in meters."""
    positions = np.asarray(emitters_pos, dtype=float)
    if positions.shape != (N_EMITTERS, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(f"Expected finite emitter positions with shape ({N_EMITTERS}, 3).")
    return positions


def get_array_center(emitters_pos: np.ndarray = EMITTERS_POS) -> np.ndarray:
    """Compute the geometric center from the actual emitter coordinates."""
    return validate_emitter_positions(emitters_pos).mean(axis=0)


def compute_projected_distances(
    azimuth_degrees: float,
    off_axis_degrees: float,
    emitters_pos: np.ndarray = EMITTERS_POS,
    reference_center: np.ndarray | None = None,
) -> np.ndarray:
    """Project centered emitter coordinates onto the requested unit direction."""
    positions = validate_emitter_positions(emitters_pos)
    center = get_array_center(positions) if reference_center is None else np.asarray(
        reference_center, dtype=float
    )
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Reference center must contain three finite coordinates in meters.")
    direction = compute_direction(azimuth_degrees, off_axis_degrees)
    return (positions - center) @ direction


def compute_continuous_steering_phases(
    azimuth_degrees: float,
    off_axis_degrees: float,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY,
    sound_speed: float = SPEED_OF_SOUND,
    emitters_pos: np.ndarray = EMITTERS_POS,
    reference_center: np.ndarray | None = None,
) -> np.ndarray:
    """Compute wrapped ideal phases before hardware quantization.

    The negative sign preserves the current script's convention. It assumes
    fields propagate as exp(i*omega*t - i*k*r); firmware phase semantics may
    use a different lead/lag convention and require physical verification.
    """
    if not math.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0.0:
        raise ValueError("Carrier frequency must be finite and greater than 0 Hz.")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("Sound speed must be finite and greater than 0 m/s.")
    projected_distances = compute_projected_distances(
        azimuth_degrees,
        off_axis_degrees,
        emitters_pos=emitters_pos,
        reference_center=reference_center,
    )
    return np.mod(
        -TWO_PI * carrier_frequency_hz * projected_distances / sound_speed,
        TWO_PI,
    )


def quantize_phase_bins(phases: np.ndarray, num_bins: int = PHASE_BINS) -> np.ndarray:
    """Apply floor quantization, snapping only floating-point boundary noise."""
    if isinstance(num_bins, bool) or int(num_bins) != num_bins or num_bins <= 0:
        raise ValueError("Number of phase bins must be a positive integer.")
    phase_values = np.asarray(phases, dtype=float)
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("Phases must contain only finite values.")

    wrapped_phases = np.mod(phase_values, TWO_PI)
    scaled_values = wrapped_phases * int(num_bins) / TWO_PI
    nearest_boundaries = np.rint(scaled_values)
    boundary_tolerance = 4.0 * np.spacing(np.maximum(np.abs(scaled_values), 1.0))
    scaled_values = np.where(
        np.abs(scaled_values - nearest_boundaries) <= boundary_tolerance,
        nearest_boundaries,
        scaled_values,
    )
    phase_bins = np.floor(scaled_values).astype(int)
    return np.clip(phase_bins, 0, int(num_bins) - 1).astype(np.uint8)


def compute_steering_phase_bins(
    azimuth_degrees: float,
    off_axis_degrees: float,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY,
    sound_speed: float = SPEED_OF_SOUND,
    emitters_pos: np.ndarray = EMITTERS_POS,
    reference_center: np.ndarray | None = None,
) -> np.ndarray:
    """Return one 0..15 phase bin per emitter in logical emitter order."""
    phases = compute_continuous_steering_phases(
        azimuth_degrees,
        off_axis_degrees,
        carrier_frequency_hz=carrier_frequency_hz,
        sound_speed=sound_speed,
        emitters_pos=emitters_pos,
        reference_center=reference_center,
    )
    return quantize_phase_bins(phases)


def compute_legacy_phase_bins(
    angle_degrees: float,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY,
    sound_speed: float = SPEED_OF_SOUND,
    emitters_pos: np.ndarray = EMITTERS_POS,
) -> np.ndarray:
    """Preserve the old 0..180 degree x-z steering interface."""
    azimuth_degrees, off_axis_degrees = legacy_angle_to_spherical(angle_degrees)
    return compute_steering_phase_bins(
        azimuth_degrees,
        off_axis_degrees,
        carrier_frequency_hz=carrier_frequency_hz,
        sound_speed=sound_speed,
        emitters_pos=emitters_pos,
    )


def validate_transmit_order(transmit_order: np.ndarray = EMITTERS_ORDER) -> np.ndarray:
    """Validate that hardware order is a permutation of logical indices."""
    order = np.asarray(transmit_order)
    if order.shape != (N_EMITTERS,) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError(f"Transmit order must contain {N_EMITTERS} integer indices.")
    if not np.array_equal(np.sort(order), np.arange(N_EMITTERS)):
        raise ValueError("EMITTERS_ORDER must be a permutation of all emitter indices.")
    return order.astype(int, copy=False)


def build_phase_frame(phase_values: np.ndarray | list[int]) -> bytes:
    """Build FE + 256 reordered phase bytes + FD, preserving the UART protocol."""
    values = np.asarray(phase_values)
    if values.shape != (N_EMITTERS,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Expected exactly {N_EMITTERS} integer phase values.")
    if np.any((values < 0) | (values >= PHASE_BINS)):
        raise ValueError(f"Phase values must be between 0 and {PHASE_BINS - 1}.")

    order = validate_transmit_order()
    payload = values[order].astype(np.uint8).tobytes()
    return bytes([HEADER]) + payload + bytes([FOOTER])


def get_rotation_angles(mode: str = "four") -> tuple[float, ...]:
    """Return the fixed azimuth sequence for the selected rotation mode."""
    if mode == "two":
        return TWO_DIRECTION_ANGLES_DEG
    if mode == "four":
        return ROTATION_ANGLES_DEG
    raise ValueError("Rotation mode must be 'two' or 'four'.")


def alpha_at_elapsed_time(
    elapsed_time_s: float,
    rotation_period_s: float,
    mode: str = "four",
) -> float:
    """Return the active fixed azimuth based on elapsed time and mode."""
    if not math.isfinite(elapsed_time_s) or elapsed_time_s < 0.0:
        raise ValueError("Elapsed time must be finite and nonnegative.")
    if not math.isfinite(rotation_period_s) or rotation_period_s <= 0.0:
        raise ValueError("Rotation period must be finite and greater than 0 seconds.")
    rotation_angles = get_rotation_angles(mode)
    sector_duration_s = rotation_period_s / len(rotation_angles)
    sector_index = int((elapsed_time_s % rotation_period_s) / sector_duration_s)
    return rotation_angles[min(sector_index, len(rotation_angles) - 1)]


def calculate_rotation_timing(
    rotation_period_s: float = 10.0,
    update_interval_s: float = 0.10,
    measured_send_time_s: float = 0.0,
    baudrate: int = BAUDRATE,
    frame_bytes: int = 258,
    safety_margin_s: float = 0.005,
    mode: str = "four",
) -> dict[str, float]:
    """Estimate safe per-angle dwell time and slow the cycle if needed."""
    direction_count = len(get_rotation_angles(mode))
    values = (rotation_period_s, update_interval_s, measured_send_time_s, safety_margin_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Timing values must be finite.")
    if rotation_period_s <= 0.0 or update_interval_s <= 0.0:
        raise ValueError("Rotation period and update interval must be greater than 0.")
    if measured_send_time_s < 0.0 or safety_margin_s < 0.0:
        raise ValueError("Measured send time and safety margin cannot be negative.")
    if baudrate <= 0 or frame_bytes <= 0:
        raise ValueError("Baud rate and frame size must be positive.")

    wire_time_s = frame_bytes * 10.0 / baudrate
    effective_update_interval_s = max(
        update_interval_s,
        rotation_period_s / direction_count,
        wire_time_s + safety_margin_s,
        measured_send_time_s + safety_margin_s,
    )
    effective_period_s = max(
        rotation_period_s,
        effective_update_interval_s * direction_count,
    )
    return {
        "wire_time_s": wire_time_s,
        "effective_update_interval_s": effective_update_interval_s,
        "effective_rotation_period_s": effective_period_s,
        "effective_angular_speed_deg_s": 360.0 / effective_period_s,
    }


def select_serial_port() -> str:
    """List detected serial ports and prompt for their displayed index."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports detected. Connect the device and try again.")
    print("\nAvailable serial ports:")
    for index, port_info in enumerate(ports):
        print(f"[{index}] {port_info.device} ({port_info.description or 'No description'})")
    while True:
        try:
            index = int(input("Select port number: ").strip())
        except ValueError:
            print("Please enter a valid port number.")
            continue
        if 0 <= index < len(ports):
            return ports[index].device
        print(f"Invalid selection. Choose a number from 0 to {len(ports) - 1}.")


def send_phase_frame(port: str, frame: bytes) -> None:
    """Send one static phase frame with the existing serial configuration."""
    with serial.Serial(
        port,
        BAUDRATE,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=2,
        rtscts=False,
        dsrdtr=False,
    ) as connection:
        connection.dtr = False
        connection.rts = False
        time.sleep(0.3)
        written = connection.write(frame)
        connection.flush()
        if written != len(frame):
            raise serial.SerialException(
                f"Incomplete frame write: sent {written} of {len(frame)} bytes."
            )


def plot_rotation_phase_samples(off_axis_degrees: float, mode: str = "four") -> None:
    """Plot physical phase maps for each selected rotation direction."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to plot phase diagnostics.") from exc

    rotation_angles = get_rotation_angles(mode)
    rows = math.ceil(len(rotation_angles) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    image = None
    for axis, alpha_degrees in zip(axes, rotation_angles):
        phase_bins = compute_steering_phase_bins(alpha_degrees, off_axis_degrees)
        x_coordinates, y_coordinates, phase_grid = physical_phase_grid(phase_bins)
        image = axis.imshow(
            phase_grid,
            origin="lower",
            extent=(x_coordinates[0] * 1000, x_coordinates[-1] * 1000,
                    y_coordinates[0] * 1000, y_coordinates[-1] * 1000),
            interpolation="nearest",
            vmin=0,
            vmax=PHASE_BINS - 1,
            cmap="twilight",
        )
        axis.set_title(f"alpha = {alpha_degrees:.0f} deg, beta = {off_axis_degrees:.1f} deg")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_aspect("equal")
    for axis in axes[len(rotation_angles):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes.tolist(), label="Phase bin (logical physical grid)")
    plt.show()


def _write_complete_frame(connection, frame: bytes) -> None:
    """Write and flush exactly one complete protocol frame."""
    written = connection.write(frame)
    connection.flush()
    if written != len(frame):
        raise serial.SerialException(
            f"Incomplete frame write: sent {written} of {len(frame)} bytes."
        )


def rotate_beam(
    beta_deg: float,
    rotation_period_s: float = 10.0,
    update_interval_s: float = 0.10,
    *,
    mode: str = "four",
    port: str | None = None,
    dry_run: bool = False,
    stop_event=None,
    duration_s: float | None = None,
    max_updates: int | None = None,
    serial_connection_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> dict[str, float | int]:
    """Cycle through fixed alpha states at fixed beta until stopped.

    Serial is opened once and reused. The first full frame establishes alpha=0;
    subsequent azimuths use monotonic elapsed time. Mode "two" uses 0/180
    degrees; mode "four" uses 0/90/180/270 degrees. Slow writes increase the
    safe dwell and full-cycle period. Use Ctrl+C or stop_event to stop.
    """
    validate_angles(0.0, beta_deg)
    rotation_angles = get_rotation_angles(mode)
    if beta_deg == 0.0:
        raise ValueError("Beta must be greater than 0 degrees for azimuth rotation.")
    if not math.isfinite(rotation_period_s) or rotation_period_s <= 0.0:
        raise ValueError("Rotation period must be finite and greater than 0 seconds.")
    if not math.isfinite(update_interval_s) or update_interval_s <= 0.0:
        raise ValueError("Update interval must be finite and greater than 0 seconds.")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("Duration must be finite and greater than 0 seconds.")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be at least 1.")
    if not dry_run and not port:
        raise ValueError("A serial port is required unless dry_run=True.")

    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    stop_event = threading.Event() if stop_event is None else stop_event
    timing = calculate_rotation_timing(rotation_period_s, update_interval_s, mode=mode)
    output_fn(
        f"Rotation preview: beta={beta_deg:.2f} deg, requested period={rotation_period_s:.3f} s, "
        f"angle dwell={timing['effective_update_interval_s']:.3f} s, "
        f"effective period={timing['effective_rotation_period_s']:.3f} s, "
        f"speed={timing['effective_angular_speed_deg_s']:.2f} deg/s, "
        f"estimated wire time={timing['wire_time_s'] * 1000:.2f} ms."
    )
    sequence = " -> ".join(f"{angle:g}" for angle in rotation_angles + (rotation_angles[0],))
    output_fn(f"Angle sequence ({mode}): {sequence} degrees.")
    output_fn("Press Ctrl+C to stop." if not dry_run else "Dry-run mode: no serial port will be opened.")

    connection_factory = serial.Serial if serial_connection_factory is None else serial_connection_factory
    connection_context = None
    connection = None
    if not dry_run:
        connection_context = connection_factory(
            port,
            BAUDRATE,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=2,
            rtscts=False,
            dsrdtr=False,
        )

    frames_sent = 0
    effective_period_s = timing["effective_rotation_period_s"]
    effective_interval_s = timing["effective_update_interval_s"]
    rotation_epoch = None
    next_deadline = None

    try:
        if dry_run:
            connection_scope = contextlib.nullcontext(None)
        else:
            connection_scope = connection_context
        with connection_scope as connection:
            if connection is not None:
                connection.dtr = False
                connection.rts = False
                sleep(0.3)

            while not stop_event.is_set():
                if frames_sent >= 1:
                    if max_updates is not None and frames_sent >= max_updates:
                        break
                    now = clock()
                    if duration_s is not None and rotation_epoch is not None:
                        if now - rotation_epoch >= duration_s:
                            break
                    wait_s = max(0.0, next_deadline - now)
                    if wait_s:
                        sleep(wait_s)
                    if stop_event.is_set():
                        break

                frame_started = clock()
                if frames_sent == 0:
                    alpha_degrees = rotation_angles[0]
                else:
                    elapsed_s = max(0.0, frame_started - rotation_epoch)
                    if duration_s is not None and elapsed_s >= duration_s:
                        break
                    alpha_degrees = alpha_at_elapsed_time(elapsed_s, effective_period_s, mode)

                phase_bins = compute_steering_phase_bins(alpha_degrees, beta_deg)
                frame = build_phase_frame(phase_bins)
                if dry_run:
                    payload = frame[1:-1]
                    output_fn(
                        f"[DRY RUN] frame={frames_sent + 1}, alpha={alpha_degrees:.3f} deg, "
                        f"beta={beta_deg:.3f} deg, bytes={len(frame)}, "
                        f"header=0x{frame[0]:02X}, first_phase_bytes={payload[:8].hex(' ')}, "
                        f"... last_phase_byte=0x{payload[-1]:02X}, footer=0x{frame[-1]:02X}"
                    )
                else:
                    _write_complete_frame(connection, frame)
                frame_finished = clock()
                send_duration_s = 0.0 if dry_run else max(0.0, frame_finished - frame_started)
                frames_sent += 1

                updated_timing = calculate_rotation_timing(
                    rotation_period_s,
                    update_interval_s,
                    measured_send_time_s=send_duration_s,
                    mode=mode,
                )
                if updated_timing["effective_rotation_period_s"] > effective_period_s:
                    if rotation_epoch is not None:
                        cycle_fraction = (
                            max(0.0, frame_finished - rotation_epoch) % effective_period_s
                        ) / effective_period_s
                        rotation_epoch = frame_finished - (
                            cycle_fraction * updated_timing["effective_rotation_period_s"]
                        )
                    effective_period_s = updated_timing["effective_rotation_period_s"]
                    output_fn(
                        "Frame transmission exceeded the estimated cadence; "
                        f"slowing rotation to {360.0 / effective_period_s:.2f} deg/s."
                    )
                effective_interval_s = updated_timing["effective_update_interval_s"]
                if frames_sent == 1:
                    rotation_epoch = frame_started
                    output_fn(
                        f"Effective timing: interval={effective_interval_s:.3f} s, "
                        f"period={effective_period_s:.3f} s, "
                        f"speed={360.0 / effective_period_s:.2f} deg/s."
                    )
                next_deadline = frame_started + effective_interval_s
    except KeyboardInterrupt:
        output_fn("Rotation stopped by user.")

    summary = {
        "frames_sent": frames_sent,
        "effective_update_interval_s": effective_interval_s,
        "effective_rotation_period_s": effective_period_s,
        "effective_angular_speed_deg_s": 360.0 / effective_period_s,
    }
    output_fn(
        f"Rotation ended after {frames_sent} frames; effective speed="
        f"{summary['effective_angular_speed_deg_s']:.2f} deg/s."
    )
    return summary


def list_serial_ports() -> None:
    """Print the available serial ports and return."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports detected. Connect the device and try again.")
        return
    print("\nAvailable serial ports:")
    for index, port_info in enumerate(ports):
        print(f"[{index}] {port_info.device} ({port_info.description or 'No description'})")


def prompt_rotation_choice() -> bool:
    """Ask whether to enter rotation mode when no mode was selected."""
    while True:
        answer = input("Enable directional rotation? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer y or n.")


def prompt_rotation_mode() -> str:
    """Ask whether the rotation should use two or four fixed directions."""
    while True:
        answer = input("Select rotation mode: two (0/180) or four (0/90/180/270) [four]: ").strip().lower()
        if answer in ("", "4", "four"):
            return "four"
        if answer in ("2", "two"):
            return "two"
        print("Please enter two or four.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Steer or cycle a planar 16x16 array through two or four directions at a selected beta."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    angle_group = parser.add_mutually_exclusive_group()
    angle_group.add_argument(
        "--angle",
        "--legacy-angle",
        dest="legacy_angle",
        type=float,
        help="Legacy x-z steering angle in degrees (0..180)",
    )
    angle_group.add_argument("--alpha", type=float, help="Azimuth in degrees (-360..360)")
    parser.add_argument("--beta", type=float, help="Off-axis angle from +z in degrees (0..180)")
    parser.add_argument("--dry-run", action="store_true", help="Compute frame without transmission")
    parser.add_argument("--rotate", action="store_true", help="Cycle azimuth through fixed directions")
    parser.add_argument("--mode", choices=("two", "four"), help="Rotation mode: two uses 0/180; four uses 0/90/180/270")
    parser.add_argument("--rotation-period", type=float, default=10.0, help="Requested revolution period in seconds")
    parser.add_argument("--update-interval", type=float, default=0.10, help="Minimum requested dwell time per angle in seconds")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; omit for Ctrl+C")
    parser.add_argument("--plot-phase-rotation", action="store_true", help="Plot phase maps at alpha 0, 90, 180, and 270 degrees")
    parser.add_argument(
        "--plot-array-factor",
        action="store_true",
        help="Plot the idealized theoretical far-field array factor",
    )
    args = parser.parse_args(argv)

    if args.list_ports:
        list_serial_ports()
        return

    if not args.rotate and args.alpha is None and args.legacy_angle is None:
        try:
            args.rotate = prompt_rotation_choice()
            if args.rotate and args.mode is None:
                args.mode = prompt_rotation_mode()
            if args.rotate and args.beta is None:
                args.beta = float(input("Select fixed beta in degrees (0..180): ").strip())
            elif not args.rotate and args.beta is not None:
                args.alpha = float(input("Select static azimuth alpha in degrees (-360..360): ").strip())
        except (ValueError, KeyboardInterrupt) as exc:
            print(f"Invalid steering selection: {exc}")
            return

    if args.rotate:
        if args.alpha is not None or args.legacy_angle is not None:
            parser.error("--rotate sweeps alpha automatically; do not specify --alpha or --angle.")
        if args.beta is None:
            parser.error("--rotate requires an explicitly selected --beta.")
        mode = args.mode or prompt_rotation_mode()
        beta_degrees = args.beta

        try:
            validate_angles(0.0, beta_degrees)
        except ValueError as exc:
            parser.error(str(exc))
        if beta_degrees == 0.0:
            parser.error("Rotating azimuth is undefined at beta=0; choose beta > 0 degrees.")

        if args.plot_phase_rotation:
            try:
                plot_rotation_phase_samples(beta_degrees, mode)
            except RuntimeError as exc:
                parser.error(str(exc))
        if args.plot_array_factor:
            try:
                initial_bins = compute_steering_phase_bins(0.0, beta_degrees)
                plot_array_factor(initial_bins, 0.0, beta_degrees)
            except RuntimeError as exc:
                parser.error(str(exc))

        port = args.port
        if not args.dry_run and port is None:
            try:
                port = select_serial_port()
            except (RuntimeError, KeyboardInterrupt) as exc:
                print(f"Cannot select a serial port: {exc}")
                return
        dry_run_duration = args.duration
        if args.dry_run and dry_run_duration is None:
            dry_run_duration = calculate_rotation_timing(
                args.rotation_period,
                args.update_interval,
                mode=mode,
            )["effective_rotation_period_s"]
            print("Dry-run defaults to one estimated revolution; use --duration to change it.")
        try:
            rotate_beam(
                beta_degrees,
                rotation_period_s=args.rotation_period,
                update_interval_s=args.update_interval,
                mode=mode,
                port=port,
                dry_run=args.dry_run,
                duration_s=dry_run_duration,
            )
        except (ValueError, serial.SerialException, OSError, RuntimeError) as exc:
            print(f"Rotation failed: {exc}")
        return

    if args.plot_phase_rotation:
        parser.error("--plot-phase-rotation requires --rotate.")
    if args.legacy_angle is not None and args.beta is not None:
        parser.error("--beta cannot be combined with the legacy --angle option.")
    if args.alpha is not None and args.beta is None:
        parser.error("--beta is required when --alpha is provided.")
    if args.alpha is None and args.beta is not None and args.legacy_angle is None:
        parser.error("--alpha is required when --beta is provided.")

    port = args.port
    if not args.dry_run and port is None:
        try:
            port = select_serial_port()
        except (RuntimeError, KeyboardInterrupt) as exc:
            print(f"Cannot select a serial port: {exc}")
            return

    try:
        if args.legacy_angle is not None:
            target_azimuth, target_off_axis = legacy_angle_to_spherical(args.legacy_angle)
        elif args.alpha is not None:
            target_azimuth, target_off_axis = args.alpha, args.beta
        else:
            target_azimuth = float(input("Azimuth alpha in degrees (-180..180): ").strip())
            target_off_axis = float(input("Off-axis beta in degrees (0..180): ").strip())
        validate_angles(target_azimuth, target_off_axis)
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"Invalid steering angles: {exc}")
        return

    phases = compute_continuous_steering_phases(target_azimuth, target_off_axis)
    phase_bins = quantize_phase_bins(phases)
    frame = build_phase_frame(phase_bins)
    center = get_array_center()
    wavelength = SPEED_OF_SOUND / DEFAULT_CARRIER_FREQUENCY
    direction = compute_direction(target_azimuth, target_off_axis)

    print(f"Azimuth alpha: {target_azimuth:.2f} deg")
    print(f"Off-axis beta: {target_off_axis:.2f} deg")
    print(
        "Requested unit direction: "
        f"({direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f})"
    )
    print(f"Array center from emitter coordinates: {center.tolist()} m")
    print(f"Wavelength: {wavelength * 1000:.3f} mm")
    print_physical_phase_map(phase_bins)
    print(f"UART frame size: {len(frame)} bytes (header + 256 phases + footer)")
    print("Phase-bin physical mapping and phase sign require hardware or measurement verification.")

    if args.plot_array_factor:
        plot_array_factor(phase_bins, target_azimuth, target_off_axis)
    if args.dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    print(f"Sending phase frame to {port}...")
    try:
        send_phase_frame(port, frame)
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        return
    print("Phase frame sent. No ON/OFF command or OFF frame was sent.")


if __name__ == "__main__":
    main()