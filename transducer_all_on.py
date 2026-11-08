#!/usr/bin/env python3
"""Turn all 256 transducers on with a uniform phase frame.

The UART protocol used by the other demo files is::

    0xFE + 256 payload bytes + 0xFD

Values 0..15 enable a transducer at a phase bin; value 0x10 disables it.
This demo sends phase 0 to every transducer. Ctrl+C stops the host program
without sending an all-off frame, so the board keeps its last state.
"""

import argparse
import sys
import time
from pathlib import Path

import serial

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from protocol import (  # noqa: E402
    BAUD_RATE,
    build_on_frame,
    open_serial,
    send_frame,
)
from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from cli_utils import run_cli  # noqa: E402

ON_PHASE = 0


def run(port: str, dry_run: bool = False) -> None:
    """Send the all-on frame continuously until interrupted."""
    on_frame = build_on_frame()
    print(f"ON frame size: {len(on_frame)} bytes")
    print(f"All transducers enabled at phase {ON_PHASE}.")
    if dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    try:
        with open_serial(port) as connection:
            send_frame(connection, on_frame)
            print(f"All transducers are ON on {port}. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        # Stop without sending an all-off frame; the board keeps its last state.
        print("\nStopped; no all-off frame sent.")
    except serial.SerialException as error:
        raise RuntimeError(
            f"无法打开串口 {port} ({error})。请确认串口未被其他程序占用。"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn all transducers on.")
    parser.add_argument("--port", help="Serial port, e.g. COM9")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build frames without opening a port"
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="List available serial ports and exit"
    )
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    port = args.port
    if not args.dry_run and port is None:
        try:
            port = select_serial_port(baudrate=BAUD_RATE)
        except (RuntimeError, KeyboardInterrupt) as error:
            parser.error(str(error))

    try:
        run(port, dry_run=args.dry_run)
    except RuntimeError as error:
        parser.error(str(error))


if __name__ == "__main__":
    run_cli(main)
