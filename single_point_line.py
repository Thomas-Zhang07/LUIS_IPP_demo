# -*- coding: utf-8 -*-
"""
Axial Line Acoustic Focus Demo (Along X-Axis)
=============================================

Continuously move a single acoustic focus along the X-axis.
The Y and Z coordinates are fixed based on user input.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# Resolve the demo modules relative to this file.
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

# Total span along the X-axis (e.g., +/- 1.5 cm from center, total 3 cm)
X_SPAN = 0.03
DEFAULT_POINT_COUNT = 48
DEFAULT_PERIOD = 2.0
DEFAULT_CENTER = (0.075, 0.075, 0.05)


def generate_line_points(center, point_count=DEFAULT_POINT_COUNT):
    """Return equally spaced target positions along the X-axis."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain three finite coordinates")
    if isinstance(point_count, (bool, np.bool_)) or not isinstance(
        point_count, (int, np.integer)
    ):
        raise ValueError("point_count must be an integer")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")

    # Calculate minimum and maximum X coordinates
    x_min = center[0] - X_SPAN / 2.0
    x_max = center[0] + X_SPAN / 2.0

    # Generate equally spaced X coordinates
    x_coords = np.linspace(x_min, x_max, point_count)
    
    # Copy the center coordinates to all points, then update X
    points = np.repeat(center[np.newaxis, :], point_count, axis=0)
    points[:, 0] = x_coords
    
    return points


def make_focus_frame(target_pos):
    """Compute a UART focus frame for one point on the line."""
    phases = compute_phases_for_focus(target_pos)
    phase_values = discretize_phases_to_bins(phases, num_bins=PHASE_BINS)
    return build_uart_frame(
        phase_values,
        transmit_order=np.asarray(EMITTERS_ORDER, dtype=int),
    )


def run_line(center, port=None, baudrate=BAUD, period=DEFAULT_PERIOD,
             point_count=DEFAULT_POINT_COUNT, dry_run=False):
    """Run the X-axis focus trajectory, or describe it without hardware."""
    if isinstance(period, (bool, np.bool_)) or not np.isfinite(period) or period <= 0:
        raise ValueError("period must be a positive finite number")
    if isinstance(baudrate, (bool, np.bool_)) or not isinstance(
        baudrate, (int, np.integer)
    ) or baudrate <= 0:
        raise ValueError("baudrate must be a positive integer")
    
    points = generate_line_points(center, point_count)
    center = np.asarray(center, dtype=float)

    print(
        f"Line center (Y, Z fixed): ({center[0]:.4f}, {center[1]:.4f}, "
        f"{center[2]:.4f}) m"
    )
    print(f"X-Span: {X_SPAN * 100:.1f} cm (from X={points[0,0]:.4f} to X={points[-1,0]:.4f})")
    print(f"Direction: Ping-Pong along X-axis; period: {period:.2f} s")
    print(f"Samples per sweep (one-way): {point_count}")

    if dry_run:
        first_target = points[0]
        frame = make_focus_frame(first_target)
        print(
            f"First target (X_min): ({first_target[0]:.4f}, "
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
            print(f"Sending X-axis trajectory to {port}. Press Ctrl+C to stop.")
            
            frame_index = 0
            # Ping-Pong state variable
            direction = 1
            
            try:
                while True:
                    target = points[frame_index]
                    ser.write(make_focus_frame(target))
                    ser.flush()

                    # Update index for Ping-Pong movement
                    frame_index += direction
                    if frame_index >= point_count - 1:
                        frame_index = point_count - 1
                        direction = -1  # Reverse at X_max
                    elif frame_index <= 0:
                        frame_index = 0
                        direction = 1   # Reverse at X_min

                    # Sleep for a constant interval to maintain uniform speed
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                # Stop transmitting without changing the board's current state.
                print("\nStopped; no all-off frame sent.")
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error


def main():
    parser = argparse.ArgumentParser(
        description="Continuously move an acoustic focus along the X-axis."
    )
    parser.add_argument("--detect", action="store_true",
                        help="List available serial ports and exit.")
    parser.add_argument("--x", type=float, default=None,
                        help="Center X coordinate (meters).")
    parser.add_argument("--y", type=float, default=None,
                        help="Center Y coordinate (meters).")
    parser.add_argument("--z", type=float, default=None,
                        help="Center Z coordinate (meters).")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g., COM3).")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD,
                        help="Seconds per one-way sweep (default 2.0).")
    parser.add_argument("--points", type=int, default=DEFAULT_POINT_COUNT,
                        help="Samples per one-way sweep (default 48).")
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
            f"\nEnter center 'x y z' (meters), or {default_msg}: "
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
        generate_line_points(center, args.points)
    except ValueError as error:
        parser.error(str(error))

    port = args.port
    if not args.dry_run and port is None:
        try:
            port = select_serial_port(baudrate=args.baud)
        except RuntimeError as error:
            parser.error(str(error))

    try:
        run_line(
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