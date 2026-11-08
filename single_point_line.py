#!/usr/bin/env python3
"""Axial Line Acoustic Focus Demo (Along X-Axis).

Continuously move a single acoustic focus along the X axis. The Y and Z
coordinates are fixed based on user input.
"""

import argparse
import math
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from cli_utils import (  # noqa: E402
    add_acoustic_args,
    add_baud_arg,
    add_dry_run_arg,
    add_port_arg,
    add_target_args,
    read_target_triplet,
)
from serial_utils import print_available_ports, resolve_serial_port  # noqa: E402
from trajectory import (  # noqa: E402
    DEFAULT_LINE_SPAN,
    DEFAULT_PERIOD_S,
    DEFAULT_POINT_COUNT,
    generate_line_points,
    run_line,
)

DEFAULT_CENTER = (0.075, 0.075, 0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Continuously move an acoustic focus along the X-axis."
    )
    parser.add_argument(
        "--detect", action="store_true", help="List available serial ports and exit."
    )
    add_target_args(parser)
    add_port_arg(parser)
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PERIOD_S,
        help=f"Seconds per one-way sweep (default {DEFAULT_PERIOD_S}).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINT_COUNT,
        help=f"Samples per one-way sweep (default {DEFAULT_POINT_COUNT}).",
    )
    add_dry_run_arg(
        parser, help_text="Describe the trajectory without sending frames."
    )
    add_baud_arg(parser)
    add_acoustic_args(parser)
    args = parser.parse_args()

    if args.detect:
        print_available_ports()
        return

    if not math.isfinite(args.period) or args.period <= 0:
        parser.error("--period must be a positive finite number")
    if args.points < 3:
        parser.error("--points must be at least 3")
    if args.baud <= 0:
        parser.error("--baud must be positive")

    try:
        center = read_target_triplet(
            (args.x, args.y, args.z),
            default=DEFAULT_CENTER,
            prompt=(
                "\nEnter center 'x y z' (meters), or press Enter to use "
                "default (0.075, 0.075, 0.05): "
            ),
            option_label="center",
        )
        generate_line_points(
            center, span=DEFAULT_LINE_SPAN, point_count=args.points
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        port = resolve_serial_port(
            args.port, dry_run=args.dry_run, baudrate=args.baud
        )
    except (RuntimeError, KeyboardInterrupt) as error:
        parser.error(str(error))

    try:
        run_line(
            center,
            port=port,
            baudrate=args.baud,
            period=args.period,
            point_count=args.points,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
