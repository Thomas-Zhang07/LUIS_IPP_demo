# -*- coding: utf-8 -*-
"""
Circular Acoustic Focus Demo
============================

Continuously move a single acoustic focus around a 3 cm diameter circle in the
XY plane, centered at a user-specified position.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# Resolve the demo modules relative to this file.  This keeps the script
# importable/runnable even when the caller's current directory is elsewhere.
DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from emitters import EMITTERS_ORDER, N_EMITTERS
from serial_utils import select_serial_port
from single_point_hover import (
    BAUD,
    PHASE_BINS,
    build_uart_frame,
    compute_phases_for_focus,
    discretize_phases_to_bins,
    print_phase_top_view,
)

# Requested circle diameter is 3 cm, so the trajectory radius is 1.5 cm.
CIRCLE_RADIUS = 0.015
DEFAULT_POINT_COUNT = 48
DEFAULT_PERIOD = 2.0
DEFAULT_CENTER = (0.075, 0.075, 0.05)


def generate_circle_points(center, point_count=DEFAULT_POINT_COUNT):
    """Return equally spaced target positions along a horizontal circle."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain three finite coordinates")
    if isinstance(point_count, (bool, np.bool_)) or not isinstance(
        point_count, (int, np.integer)
    ):
        raise ValueError("point_count must be an integer")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")

    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.repeat(center[np.newaxis, :], point_count, axis=0)
    points[:, 0] += CIRCLE_RADIUS * np.cos(angles)
    points[:, 1] += CIRCLE_RADIUS * np.sin(angles)
    return points


def make_focus_frame(target_pos):
    """Compute a UART focus frame for one point on the circle."""
    phases = compute_phases_for_focus(target_pos)
    phase_values = discretize_phases_to_bins(phases, num_bins=PHASE_BINS)
    return build_uart_frame(
        phase_values,
        transmit_order=np.asarray(EMITTERS_ORDER, dtype=int),
    )


def run_circle(center, port=None, baudrate=BAUD, period=DEFAULT_PERIOD,
               point_count=DEFAULT_POINT_COUNT, dry_run=False):
    """Run the circular focus trajectory, or describe it without hardware."""
    if isinstance(period, (bool, np.bool_)) or not np.isfinite(period) or period <= 0:
        raise ValueError("period must be a positive finite number")
    if isinstance(baudrate, (bool, np.bool_)) or not isinstance(
        baudrate, (int, np.integer)
    ) or baudrate <= 0:
        raise ValueError("baudrate must be a positive integer")
    points = generate_circle_points(center, point_count)
    center = np.asarray(center, dtype=float)

    print(
        f"Circle center: ({center[0]:.4f}, {center[1]:.4f}, "
        f"{center[2]:.4f}) m"
    )
    print(f"Diameter: {2 * CIRCLE_RADIUS:.3f} m (3 cm), XY plane")
    print(f"Direction: counterclockwise; period: {period:.2f} s")
    print(f"Samples per circle: {point_count}")

    if dry_run:
        first_target = points[0]
        frame = make_focus_frame(first_target)
        print(
            f"First target: ({first_target[0]:.4f}, "
            f"{first_target[1]:.4f}, {first_target[2]:.4f}) m"
        )
        print_phase_top_view(
            discretize_phases_to_bins(
                compute_phases_for_focus(first_target), num_bins=PHASE_BINS
            )
        )
        print(f"\nUART frame size: {len(frame)} bytes")
        print("\n[DRY RUN] No frames sent.")
        return

    if port is None:
        raise ValueError("port is required unless dry_run is enabled")

    interval = period / point_count
    try:
        with serial.Serial(port, baudrate, timeout=2, write_timeout=2) as ser:
            ser.dtr = False
            ser.rts = False
            time.sleep(0.5)
            print(f"Sending circle trajectory to {port}. Press Ctrl+C to stop.")
            start_time = time.monotonic()
            frame_index = 0
            try:
                while True:
                    target = points[frame_index % point_count]
                    ser.write(make_focus_frame(target))
                    ser.flush()

                    frame_index += 1
                    next_deadline = start_time + frame_index * interval
                    delay = next_deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            except KeyboardInterrupt:
                # Stop transmitting without changing the board's current state.
                # The serial context closes here; no all-off frame is sent.
                print("\nStopped; no all-off frame sent.")
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error


def main():
    parser = argparse.ArgumentParser(
        description="Continuously move an acoustic focus around a 3 cm diameter circle."
    )
    parser.add_argument("--detect", action="store_true",
                        help="List available serial ports and exit.")
    parser.add_argument("--x", type=float, default=None,
                        help="Circle center X coordinate (meters).")
    parser.add_argument("--y", type=float, default=None,
                        help="Circle center Y coordinate (meters).")
    parser.add_argument("--z", type=float, default=None,
                        help="Circle center Z coordinate (meters).")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g., COM3).")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD,
                        help="Seconds per circle (default 5).")
    parser.add_argument("--points", type=int, default=DEFAULT_POINT_COUNT,
                        help="Samples per circle (default 48).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Describe the trajectory without sending frames.")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"Baud rate (default {BAUD}).")
    args = parser.parse_args()

    if args.detect:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports detected.")
            return
        print("\nAvailable serial ports:")
        for index, port in enumerate(ports):
            print(f"  [{index}] {port.device:<10} - {port.description}")
        return

    if not np.isfinite(args.period) or args.period <= 0:
        parser.error("--period must be a positive finite number")
    if args.points < 3:
        parser.error("--points must be at least 3")
    if args.baud <= 0:
        parser.error("--baud must be positive")

    coordinates = (args.x, args.y, args.z)
    if all(value is not None for value in coordinates):
        center = coordinates
    elif all(value is None for value in coordinates):
        default_msg = "Press Enter to use default (0.075, 0.075, 0.05)"
        user_input = input(
            f"\nEnter circle center 'x y z' (meters), or {default_msg}: "
        ).strip()
        if not user_input:
            center = DEFAULT_CENTER
        else:
            try:
                values = user_input.split()
                if len(values) != 3:
                    parser.error("center must contain exactly three numbers")
                center = tuple(map(float, values))
            except ValueError:
                parser.error("center coordinates must be numbers")
    else:
        parser.error("provide all of --x, --y, and --z, or none")

    try:
        generate_circle_points(center, args.points)
    except ValueError as error:
        parser.error(str(error))

    port = args.port
    if not args.dry_run and port is None:
        try:
            port = select_serial_port(baudrate=args.baud)
        except RuntimeError as error:
            parser.error(str(error))

    try:
        run_circle(
            center,
            port=port,
            baudrate=args.baud,
            period=args.period,
            point_count=args.points,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
