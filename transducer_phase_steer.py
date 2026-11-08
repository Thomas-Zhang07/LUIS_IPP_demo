#!/usr/bin/env python3
"""Steer a continuously driven array with one angle in the x-z plane.

The 2D array is treated as a 1D array along x, centered on the y-parallel axis
x=0.075 m. One 0..180 degree angle selects the beam direction in the x-z
cross-section: 0 points along +x, 90 along +z (the array normal), and 180
along -x. One static 16-level phase frame is sent; no ON/OFF or OFF frame is
sent.
"""

import argparse
import math
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from acoustics import line_steering_phase_bins, validate_acoustic_params  # noqa: E402
from cli_utils import add_acoustic_args, print_phase_top_view  # noqa: E402
from protocol import BAUD_RATE, build_frame, write_frame_to_port  # noqa: E402
from serial_utils import print_available_ports, select_serial_port  # noqa: E402


def parse_angle_input(raw_value: str) -> float:
    """Parse and validate the beam angle in degrees."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError("角度必须是数字。") from error
    if not math.isfinite(value):
        raise ValueError("角度必须是有限值。")
    if not 0.0 <= value <= 180.0:
        raise ValueError("角度必须在 0 到 180 度之间。")
    return value


def prompt_for_angle() -> float:
    """Prompt for the single beam angle in Chinese."""
    while True:
        raw_angle = input(
            "\n请输入波束角度（0 到 180 度；0=沿+x，90=沿+z，180=沿-x）： "
        ).strip()
        try:
            return parse_angle_input(raw_angle)
        except ValueError as error:
            print(f"{error} 请重新输入。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steer the transducer array with one 0..180 degree beam angle."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Scan and list available serial ports, then exit",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Beam angle in degrees (0..180); if omitted, prompt interactively",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and show frame without sending"
    )
    add_acoustic_args(parser)
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    port = None
    if not args.dry_run:
        if args.port:
            port = args.port
        else:
            try:
                port = select_serial_port(baudrate=BAUD_RATE)
            except (RuntimeError, KeyboardInterrupt) as error:
                if isinstance(error, KeyboardInterrupt):
                    print("\nPort selection cancelled.")
                else:
                    print(f"Cannot select a serial port: {error}")
                return

    try:
        angle_degrees = (
            parse_angle_input(str(args.angle))
            if args.angle is not None
            else prompt_for_angle()
        )
    except (ValueError, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            print("\n角度输入已取消。")
            return
        parser.error(str(error))

    try:
        validate_acoustic_params(args.frequency, args.sound_speed)
        phases = line_steering_phase_bins(
            angle_degrees,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
        )
        frame = build_frame(phases)
    except ValueError as error:
        parser.error(str(error))

    print(f"波束角度: {angle_degrees:.2f} 度")
    print("角度约定: 0° 沿+x，90° 沿+z（阵列法线），180° 沿-x")
    print(
        f"相位计算中心轴: x=0.075 m，载波频率: {args.frequency:.0f} Hz，"
        f"声速: {args.sound_speed:.1f} m/s"
    )
    print_phase_top_view(phases, title="16x16 phase map (0..15; top view):")
    print(f"UART frame size: {len(frame)} bytes (header + 256 phases + footer)")

    if args.dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    print(f"Sending phase frame to {port}...")
    try:
        write_frame_to_port(port, frame, baudrate=BAUD_RATE)
    except (OSError, RuntimeError) as error:
        print(f"Serial error: {error}")
        return
    print("Phase frame sent. No ON/OFF toggle or OFF frame was sent.")


if __name__ == "__main__":
    main()
