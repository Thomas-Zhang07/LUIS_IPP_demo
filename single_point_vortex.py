#!/usr/bin/env python3
"""Single-Point Acoustic Vortex Demo.

Compute and send a single-point acoustic vortex for a 16x16 transducer array.
The phase is the sum of:

  1. Static focus phase at the target point.
  2. Rotational offset phase: topological_charge * polar_angle.

Coordinates are in meters. The array coordinate origin and emitter order are
shared with demo/emitters.py.
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
    quantize_phase_bins,
    validate_acoustic_params,
    vortex_phases,
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
)
from emitters import EMITTERS_ORDER  # noqa: E402
from protocol import (  # noqa: E402
    BAUD_RATE,
    build_frame,
    open_serial,
    send_frame,
)
from serial_utils import print_available_ports, resolve_serial_port  # noqa: E402

DEFAULT_TARGET = (0.075, 0.075, 0.05)


def send_vortex_frame_to_port(
    port,
    target_x,
    target_y,
    target_z,
    topological_charge=1,
    baudrate=BAUD_RATE,
    verbose=True,
    dry_run=False,
    *,
    frequency_hz=DEFAULT_FREQUENCY_HZ,
    sound_speed=DEFAULT_SOUND_SPEED_MPS,
):
    """Compute and send (or describe) a single-point vortex frame."""
    target_pos = np.array([target_x, target_y, target_z], dtype=float)
    result = vortex_phases(
        target_pos,
        topological_charge=topological_charge,
        frequency_hz=frequency_hz,
        sound_speed=sound_speed,
    )
    values = quantize_phase_bins(result.total)

    if verbose:
        print(
            f"\nTarget position: "
            f"({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m"
        )
        print(
            f"Array center (mean): "
            f"({result.center[0]:.4f}, {result.center[1]:.4f}, "
            f"{result.center[2]:.4f}) m"
        )
        print(f"Topological charge: {topological_charge}")
        print("Phase formula: focus_phase + topological_charge * polar_angle")
        print_phase_top_view(
            values,
            title="16x16 vortex phase map (top view, y increases upward):",
        )
        print("\nPhase ranges (radians):")
        print(
            f"  focus phase:    "
            f"{result.focus.min():.4f} .. {result.focus.max():.4f}"
        )
        print(
            f"  center distance:{result.radii.min():.4f} .. "
            f"{result.radii.max():.4f} m"
        )
        print(
            f"  polar angle:    "
            f"{result.angles.min():.4f} .. {result.angles.max():.4f}"
        )
        print(
            f"  rotation offset:{result.rotation_offsets.min():.4f} .. "
            f"{result.rotation_offsets.max():.4f}"
        )

    frame = build_frame(values, transmit_order=EMITTERS_ORDER)

    if verbose:
        print_frame(frame)

    if dry_run:
        print("\n[DRY RUN] Frame NOT sent.")
    else:
        with open_serial(port, baudrate) as connection:
            send_frame(connection, frame)
        print(f"\nFrame sent to {port}")

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Single-point acoustic vortex demo (16-level phase control)."
    )
    parser.add_argument(
        "--detect", action="store_true", help="List available serial ports and exit."
    )
    add_target_args(parser)
    parser.add_argument(
        "--charge",
        type=int,
        default=1,
        help="Vortex topological charge, e.g. 1 or -1 (default 1).",
    )
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

    if args.x is not None and args.y is not None and args.z is not None:
        target_x, target_y, target_z = args.x, args.y, args.z
        topological_charge = args.charge
    else:
        try:
            target_x, target_y, target_z = read_target_triplet(
                (args.x, args.y, args.z),
                default=DEFAULT_TARGET,
                prompt=(
                    "\nEnter vortex target 'x y z' (m), or press Enter for "
                    "default (0.075, 0.075, 0.05): "
                ),
                option_label="vortex target",
            )
        except ValueError as error:
            parser.error(str(error))

        direction = input(
            "Direction [ccw=counter-clockwise, cw=clockwise] "
            f"(default {'ccw' if args.charge >= 0 else 'cw'}): "
        ).strip().lower()
        if not direction:
            topological_charge = args.charge
        elif direction in (
            "ccw",
            "counter-clockwise",
            "counterclockwise",
            "逆时针",
        ):
            topological_charge = abs(args.charge) or 1
        elif direction in ("cw", "clockwise", "顺时针"):
            topological_charge = -(abs(args.charge) or 1)
        else:
            print("Invalid direction: use ccw/cw (逆时针/顺时针).")
            return

    try:
        port = resolve_serial_port(
            args.port, dry_run=args.dry_run, baudrate=args.baud
        )
    except (RuntimeError, KeyboardInterrupt) as error:
        print(f"Port selection failed: {error}")
        return

    try:
        send_vortex_frame_to_port(
            port,
            target_x,
            target_y,
            target_z,
            topological_charge=topological_charge,
            baudrate=args.baud,
            verbose=True,
            dry_run=args.dry_run,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
