#!/usr/bin/env python3
"""Toggle the transducer output with a single-angle phase gradient.

The ON frame carries 16-level carrier phase values (0..15); the OFF frame uses
0x10 for every emitter. The 2D array is treated as a 1D array along x,
centered on the y-parallel axis x=0.075 m. One angle in the x-z cross-section
sets the phase gradient: 0 degrees points along +x, 90 along +z (the array
normal), and 180 along -x. The ON/OFF frequency controls the envelope toggle.

At 38400 baud, sending one 258-byte frame takes about 67 ms (8N1), so a
host-driven ON/OFF cycle cannot reach the 20 Hz lower end of the audible
range. The requested toggle rate is retained for compatibility, but the limit
is reported before sending.
"""

import argparse
import math
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from acoustics import (  # noqa: E402
    line_steering_phase_bins,
    validate_acoustic_params,
)
from cli_utils import add_acoustic_args, print_phase_top_view, run_cli  # noqa: E402
from protocol import (  # noqa: E402
    BAUD_RATE,
    build_frame,
    build_off_frame,
    maximum_toggle_frequency_hz,
)
from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from trajectory import run_toggle  # noqa: E402

MIN_FREQUENCY_HZ = 0.1
MAX_FREQUENCY_HZ = 20000.0


def parse_angle_input(raw_value: str) -> float:
    """Parse and validate the beam angle in degrees."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError("Angle must be a number.") from error
    if not math.isfinite(value):
        raise ValueError("Angle must be finite.")
    if not 0.0 <= value <= 180.0:
        raise ValueError("Angle must be between 0 and 180 degrees.")
    return value


def prompt_for_angle() -> float:
    """Prompt for the single beam angle."""
    while True:
        raw_angle = input(
            "\nEnter beam angle (0 to 180 degrees; 0=+x, 90=+z, 180=-x): "
        ).strip()
        try:
            return parse_angle_input(raw_angle)
        except ValueError as error:
            print(f"{error} Please try again.", flush=True)


def parse_frequency(raw: str) -> float:
    try:
        frequency = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid frequency: {raw!r}") from error
    if not math.isfinite(frequency) or not (
        MIN_FREQUENCY_HZ <= frequency <= MAX_FREQUENCY_HZ
    ):
        raise ValueError("Frequency must be between 0.1 Hz and 20000 Hz.")
    return frequency


def prompt_for_frequency() -> float:
    while True:
        try:
            return parse_frequency(
                input("Enter target frequency in Hz (0.1 to 20000): ").strip()
            )
        except ValueError as error:
            print(f"{error} Please try again.")


def prompt_for_duration():
    while True:
        raw = input(
            "Enter run duration in seconds (leave empty for continuous): "
        ).strip()
        if not raw:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            print("Invalid duration. Please enter a number or leave it empty.")
            continue
        if not math.isfinite(seconds) or seconds <= 0.0:
            print("Duration must be a finite number greater than 0 seconds.")
            continue
        return seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toggle all transducers ON/OFF with a single-angle beam direction."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument(
        "--frequency", type=float, help="ON/OFF toggle frequency in Hz (0.1..20000)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Stop after this many seconds; omit for continuous mode",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Beam angle in degrees (0..180); if omitted, prompt interactively",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and print frames without opening a port"
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="Print available serial ports and exit"
    )
    add_acoustic_args(parser, frequency_flag="--carrier-frequency")
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    if args.dry_run and args.frequency is None:
        parser.error("--frequency is required with --dry-run")
    if args.dry_run and args.angle is None:
        parser.error("--angle is required with --dry-run")

    try:
        validate_acoustic_params(args.carrier_frequency, args.sound_speed)
        angle_degrees = (
            parse_angle_input(str(args.angle))
            if args.angle is not None
            else prompt_for_angle()
        )
        frequency_hz = (
            parse_frequency(str(args.frequency))
            if args.frequency is not None
            else prompt_for_frequency()
        )
    except (ValueError, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            print("\nCancelled.")
            return
        parser.error(str(error))

    if args.duration is not None:
        duration_seconds = args.duration
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            parser.error("Duration must be a finite number greater than 0 seconds.")
    elif args.dry_run:
        duration_seconds = None
    else:
        duration_seconds = prompt_for_duration()

    try:
        phases = line_steering_phase_bins(
            angle_degrees,
            frequency_hz=args.carrier_frequency,
            sound_speed=args.sound_speed,
        )
        on_frame = build_frame(phases)
        off_frame = build_off_frame()
    except ValueError as error:
        parser.error(str(error))

    print(f"Beam angle: {angle_degrees:.2f} degrees")
    print("Angle convention: 0 degrees along +x, 90 along +z, 180 along -x")
    print(
        f"Phase center axis: x=0.075 m; carrier frequency: "
        f"{args.carrier_frequency:.0f} Hz; sound speed: {args.sound_speed:.1f} m/s"
    )
    print(f"ON/OFF toggle frequency: {frequency_hz:.2f} Hz")
    print_phase_top_view(phases, title="16x16 phase map (0..15; top view):")
    print(f"ON frame size: {len(on_frame)} bytes; OFF frame size: {len(off_frame)} bytes")
    max_frequency = maximum_toggle_frequency_hz()
    print(
        f"UART wire-time limit: about {max_frequency:.2f} Hz maximum "
        "(ideal, excluding software overhead)."
    )

    if args.dry_run:
        run_toggle(
            port=None,
            on_frame=on_frame,
            off_frame=off_frame,
            frequency_hz=frequency_hz,
            duration_s=duration_seconds,
            dry_run=True,
        )
        return

    if args.port:
        port = args.port
    else:
        try:
            port = select_serial_port(baudrate=BAUD_RATE)
        except (RuntimeError, KeyboardInterrupt) as error:
            print(f"Cannot select a serial port: {error}")
            return

    print(f"Using port: {port}")
    try:
        run_toggle(
            port=port,
            on_frame=on_frame,
            off_frame=off_frame,
            frequency_hz=frequency_hz,
            duration_s=duration_seconds,
            send_off_on_exit=True,
            timestamp=True,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(f"Serial error: {error}")


if __name__ == "__main__":
    run_cli(main)
