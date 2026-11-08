#!/usr/bin/env python3
"""Toggle all 256 transducers on/off at a user-selected audible frequency.

This demo works with the project UART protocol already used in the repo:
- 0xFE header
- 256 bytes of payload
- 0xFD footer

The payload is the same value for all 256 transducers. A value of 0x00 is the
active ON state, while 0x10 is the OFF state.
"""

import argparse
import sys
import time
from pathlib import Path

import serial

# Ensure the demo folder itself is importable when this script is run directly.
if __package__ in (None, ""):
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

from serial_utils import select_serial_port

BAUDRATE = 38400
N_EMITTERS = 256
HEADER = 0xFE
FOOTER = 0xFD
ON_VALUE = 0x00
OFF_VALUE = 0x10


def build_uniform_frame(value: int, n_emitters: int = N_EMITTERS) -> bytes:
    """Build a UART frame with the same value for all emitters."""
    if not 0 <= value <= 255:
        raise ValueError("value must be an integer in the range 0..255")
    payload = bytes([value & 0xFF]) * n_emitters
    return bytes([HEADER]) + payload + bytes([FOOTER])


def parse_frequency(raw: str) -> float:
    """Parse and validate the requested frequency in the human-hearing range."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid frequency: {raw!r}") from exc

    if not (20.0 <= value <= 20000.0):
        raise ValueError("Frequency must be between 20 Hz and 20000 Hz.")
    return value


def run_toggle_cycle(port: str, frequency_hz: float, duration_seconds: float | None = None) -> None:
    """Continuously toggle all 256 transducers between ON and OFF at frequency_hz."""
    half_period = 0.5 / frequency_hz
    on_frame = build_uniform_frame(ON_VALUE)
    off_frame = build_uniform_frame(OFF_VALUE)

    with serial.Serial(
        port,
        BAUDRATE,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=2,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)

        deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
        cycle_index = 0

        print(f"\nStarting toggle cycle on {port}")
        print(f"Target frequency: {frequency_hz:.2f} Hz")
        print(f"Half-period: {half_period:.6f} s")
        print("Press Ctrl+C to stop.\n")

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print("Duration reached; stopping.")
                break

            ser.write(on_frame)
            ser.flush()
            print(f"Cycle {cycle_index + 1}: ON  ({time.strftime('%H:%M:%S')})")
            time.sleep(half_period)

            if deadline is not None and time.monotonic() >= deadline:
                print("Duration reached; stopping.")
                break

            ser.write(off_frame)
            ser.flush()
            print(f"Cycle {cycle_index + 1}: OFF ({time.strftime('%H:%M:%S')})")
            time.sleep(half_period)
            cycle_index += 1


def prompt_for_frequency() -> float:
    while True:
        try:
            raw = input("Enter target frequency in Hz (20 to 20000): ").strip()
            return parse_frequency(raw)
        except ValueError as exc:
            print(f"{exc}. Please try again.")


def prompt_for_duration() -> float | None:
    while True:
        raw = input("Enter run duration in seconds (leave empty for continuous): ").strip()
        if raw == "":
            return None
        try:
            seconds = float(raw)
        except ValueError:
            print("Invalid duration. Please enter a number or leave it empty.")
            continue

        if seconds <= 0:
            print("Duration must be > 0 seconds.")
            continue
        return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Toggle all 256 transducers ON/OFF at a user-selected frequency.")
    parser.add_argument("--port", type=str, help="Serial port to use, e.g. COM3 or /dev/rfcomm0")
    parser.add_argument("--frequency", type=float, help="Target frequency in Hz (20..20000)")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; omit for continuous mode")
    parser.add_argument("--list-ports", action="store_true", help="Print available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                print("No serial ports detected.")
                return
            print("Available serial ports:")
            for i, p in enumerate(ports):
                print(f"  [{i}] {p.device:<12}  -  {p.description or 'No description'}")
            return
        except Exception as exc:
            print(f"Unable to list serial ports: {exc}")
            return

    # Always ask for port selection when no port is explicitly supplied.
    if args.port:
        port = args.port
    else:
        try:
            print("\nChoose a port by number or by the exact port name shown in the list.")
            port = select_serial_port(baudrate=BAUDRATE)
        except Exception as exc:
            print(f"Cannot select a serial port: {exc}")
            return

    if args.frequency is not None:
        frequency_hz = args.frequency
    else:
        frequency_hz = prompt_for_frequency()

    if args.duration is not None:
        duration_seconds = args.duration
    else:
        duration_seconds = prompt_for_duration()

    if not (20.0 <= frequency_hz <= 20000.0):
        print("Frequency must be between 20 and 20000 Hz.")
        return

    print(f"Using port: {port}")
    try:
        run_toggle_cycle(port, frequency_hz=frequency_hz, duration_seconds=duration_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
