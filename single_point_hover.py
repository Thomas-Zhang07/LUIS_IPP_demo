#!/usr/bin/env python3
"""Single-Point Acoustic Focus Demo.

This script demonstrates how to create and send an acoustic focus at a single
target position using a 16x16 ultrasonic transducer array.

Features:
  - 16-level phase control (0..15, with 16 = OFF).
  - Interactive serial port selection.
  - Interactive or command-line position input.
  - Prints phase values and UART frame for verification.
  - Configurable carrier frequency and speed of sound.

Dependencies: numpy, pyserial, and modules from the demo folder.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from acoustics import (  # noqa: E402
    DEFAULT_FREQUENCY_HZ,
    DEFAULT_SOUND_SPEED_MPS,
    focus_phases,
    quantize_phase_bins,
    validate_acoustic_params,
)
from cli_utils import (  # noqa: E402
    add_acoustic_args,
    add_baud_arg,
    add_dry_run_arg,
    add_port_arg,
    add_target_args,
    print_frame,
    print_phase_top_view,
    read_target_triplet,
    run_cli,
)
from emitters import EMITTERS_ORDER  # noqa: E402
from protocol import (  # noqa: E402
    BAUD_RATE,
    PHASE_BINS,
    build_frame,
    open_serial,
    send_frame,
)
from serial_utils import print_available_ports, resolve_serial_port  # noqa: E402

BAUD = BAUD_RATE  # backward-compatible name used by the trajectory scripts
DEFAULT_TARGET = (0.075, 0.075, 0.0)


def send_focus_frame_to_port(
    port,
    target_x,
    target_y,
    target_z,
    verbose=True,
    dry_run=False,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
    baudrate=BAUD_RATE,
):
    """Compute and send (or describe) one acoustic focus frame."""
    target_pos = np.array([target_x, target_y, target_z], dtype=float)

    phases = focus_phases(
        target_pos, frequency_hz=frequency_hz, sound_speed=sound_speed
    )
    values = quantize_phase_bins(phases)

    if verbose:
        print(
            f"\nTarget position: "
            f"({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m"
        )
        print(f"Phase discretization: 0..{PHASE_BINS - 1} levels")
        print_phase_top_view(values)

    frame = build_frame(values, transmit_order=EMITTERS_ORDER)

    if verbose:
        print_frame(frame)

    if not dry_run:
        with open_serial(port, baudrate) as connection:
            send_frame(connection, frame)
        print(f"\nFrame sent to {port}")
    else:
        print("\n[DRY RUN] Frame NOT sent.")

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Single-point acoustic focus demo (16-level phase control)."
    )
    parser.add_argument(
        "--detect", action="store_true", help="List available serial ports and exit."
    )
    add_target_args(parser)
    add_port_arg(parser)
    add_dry_run_arg(parser)
    add_baud_arg(parser)
    add_acoustic_args(parser)
    args = parser.parse_args()

    if args.detect:
        print_available_ports()
        return

    try:
        validate_acoustic_params(args.frequency, args.sound_speed)
    except ValueError as error:
        parser.error(str(error))

    try:
        target_x, target_y, target_z = read_target_triplet(
            (args.x, args.y, args.z),
            default=DEFAULT_TARGET,
            prompt=(
                "\nEnter target position 'x y z' (meters), or press Enter to "
                "use default (0.075, 0.075, 0.0): "
            ),
            option_label="target",
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        port = resolve_serial_port(
            args.port, dry_run=args.dry_run, baudrate=args.baud
        )
    except (RuntimeError, KeyboardInterrupt) as error:
        print(f"Port selection failed: {error}")
        return

    try:
        send_focus_frame_to_port(
            port,
            target_x,
            target_y,
            target_z,
            verbose=True,
            dry_run=args.dry_run,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
            baudrate=args.baud,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    run_cli(main)
