#!/usr/bin/env python3
"""Drive all 256 transducers using an 8-bit binary pattern.

Example:
    10110001

Each bit selects the state of the whole array:
    1 -> ON  (send full-frame value 0x00)
    0 -> OFF (send full-frame value 0x10)

The protocol is ``0xFE + 256 payload bytes + 0xFD``. The serial port is opened
once per run instead of once per bit.
"""

import argparse
import math
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from trajectory import run_bit_pattern  # noqa: E402


def parse_bit_pattern(raw: str):
    clean = raw.strip().replace(" ", "").replace(",", "")
    if len(clean) != 8:
        raise ValueError("Pattern must contain exactly 8 bits.")
    if any(character not in "01" for character in clean):
        raise ValueError("Pattern can only contain 0 and 1 characters.")
    return [1 if character == "1" else 0 for character in clean]


def prompt_for_pattern():
    while True:
        raw = input("Enter 8-bit binary pattern, e.g. 10110001: ").strip()
        try:
            return parse_bit_pattern(raw)
        except ValueError as error:
            print(f"{error}. Please try again.")


def parse_bit_seconds(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Bit duration must be a number.") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Time per bit must be greater than 0.")
    return value


def prompt_for_bit_seconds():
    while True:
        raw = input(
            "Enter time for each bit in seconds (for example 0.1): "
        ).strip()
        try:
            return parse_bit_seconds(raw)
        except ValueError as error:
            print(f"{error}. Please try again.")


def prompt_for_duration():
    raw = input("Run for how many seconds? Leave empty for continuous mode: ").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        print("Invalid duration; using continuous mode.")
        return None
    if not math.isfinite(value) or value <= 0:
        print("Duration must be positive; using continuous mode.")
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Repeat an 8-bit ON/OFF pattern across the full 256-transducer array."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument(
        "--pattern", type=str, help="8-bit binary pattern to repeat, e.g. 10110001"
    )
    parser.add_argument(
        "--bit-seconds", type=float, help="How long each bit lasts in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Run for this many seconds; omit for continuous mode",
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
        if args.pattern is None:
            parser.error("--pattern is required with --dry-run")
        if args.bit_seconds is None:
            parser.error("--bit-seconds is required with --dry-run")
        port = None
    else:
        if args.port:
            port = args.port
        else:
            try:
                port = select_serial_port()
            except (RuntimeError, KeyboardInterrupt) as error:
                print(f"Port selection failed: {error}")
                return

    try:
        pattern = (
            parse_bit_pattern(args.pattern)
            if args.pattern is not None
            else prompt_for_pattern()
        )
        bit_seconds = (
            parse_bit_seconds(args.bit_seconds)
            if args.bit_seconds is not None
            else prompt_for_bit_seconds()
        )
    except ValueError as error:
        parser.error(str(error))

    if args.duration is not None:
        duration_seconds = args.duration
    elif args.dry_run:
        duration_seconds = None
    else:
        duration_seconds = prompt_for_duration()

    if not args.dry_run:
        print(f"\nUsing port: {port}")
    print(f"Pattern: {''.join(str(bit) for bit in pattern)}")
    print(f"Each bit lasts {bit_seconds:.3f} seconds")
    if duration_seconds is None:
        print("Duration: continuous (Ctrl+C to stop)")
    else:
        print(f"Total duration: {duration_seconds:.3f} s")

    try:
        run_bit_pattern(
            port=port,
            pattern=pattern,
            bit_seconds=bit_seconds,
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
