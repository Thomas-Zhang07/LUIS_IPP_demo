#!/usr/bin/env python3
"""Toggle all 256 transducers ON/OFF at a user-selected frequency.

The UART protocol is ``0xFE + 256 payload bytes + 0xFD``; the payload value
0x00 enables the array and 0x10 disables it. The serial port is opened once
per run, so the ON/OFF cadence is not limited by port reconnection.
"""

import argparse
import math
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from protocol import BAUD_RATE, build_off_frame, build_on_frame  # noqa: E402
from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from trajectory import run_toggle  # noqa: E402

MIN_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 20000.0


def parse_frequency(raw: str) -> float:
    """Parse and validate the requested toggle frequency in Hz."""
    try:
        frequency = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid frequency: {raw!r}") from error
    if not math.isfinite(frequency) or not (
        MIN_FREQUENCY_HZ <= frequency <= MAX_FREQUENCY_HZ
    ):
        raise ValueError("Frequency must be between 20 Hz and 20000 Hz.")
    return frequency


def prompt_for_frequency() -> float:
    while True:
        try:
            return parse_frequency(
                input("Enter target frequency in Hz (20 to 20000): ").strip()
            )
        except ValueError as error:
            print(f"{error}. Please try again.")


def prompt_for_duration():
    raw = input(
        "Enter run duration in seconds (leave empty for continuous): "
    ).strip()
    if raw == "":
        return None
    try:
        seconds = float(raw)
    except ValueError:
        print("Invalid duration, using continuous mode.")
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        print("Duration must be > 0; using continuous mode.")
        return None
    return seconds


def main():
    parser = argparse.ArgumentParser(
        description="Toggle all 256 transducers ON/OFF at a user-selected frequency."
    )
    parser.add_argument(
        "--port", type=str, help="Serial port to use directly, e.g. COM3"
    )
    parser.add_argument(
        "--frequency", type=float, help="Target frequency in Hz (20..20000)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Stop after this many seconds; omit for continuous mode",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Describe the run without opening a port"
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="Print available serial ports and exit"
    )
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    if args.dry_run:
        if args.frequency is None:
            parser.error("--frequency is required with --dry-run")
        port = None
    else:
        if args.port:
            port = args.port
        else:
            try:
                port = select_serial_port(baudrate=BAUD_RATE)
            except (RuntimeError, KeyboardInterrupt) as error:
                print(f"Port selection failed: {error}")
                return

    try:
        frequency_hz = (
            parse_frequency(str(args.frequency))
            if args.frequency is not None
            else prompt_for_frequency()
        )
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    if args.duration is not None:
        duration_seconds = args.duration
    elif args.dry_run:
        duration_seconds = None
    else:
        duration_seconds = prompt_for_duration()

    if not args.dry_run:
        print(f"\nUsing port: {port}")
    print(f"Frequency: {frequency_hz:.2f} Hz")
    print(f"Half-period: {0.5 / frequency_hz:.6f} s")
    if duration_seconds is None:
        print("Duration: continuous (Ctrl+C to stop)")
    else:
        print(f"Duration: {duration_seconds:.3f} s")

    try:
        run_toggle(
            port=port,
            on_frame=build_on_frame(),
            off_frame=build_off_frame(),
            frequency_hz=frequency_hz,
            duration_s=duration_seconds,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(f"Serial error: {error}")


if __name__ == "__main__":
    main()
