#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
import serial
import serial.tools.list_ports

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from emitters import EMITTERS_ORDER, N_EMITTERS  # noqa: E402
from serial_utils import select_serial_port  # noqa: E402

BAUDRATE = 38400
HEADER = 0xFE
FOOTER = 0xFD
OFF_VALUE = 0x10
ON_PHASE = 0


def build_frame(values) -> bytes:
    """Build a validated UART frame in hardware transmit order."""
    payload = np.asarray(values, dtype=int)
    if payload.shape != (N_EMITTERS,):
        raise ValueError(f"Expected {N_EMITTERS} transducer values.")
    if np.any((payload < 0) | (payload > OFF_VALUE)):
        raise ValueError(f"Values must be between 0 and {OFF_VALUE}.")
    order = np.asarray(EMITTERS_ORDER, dtype=int)
    if order.shape != (N_EMITTERS,) or not np.array_equal(
        np.sort(order), np.arange(N_EMITTERS)
    ):
        raise ValueError("EMITTERS_ORDER must contain every emitter exactly once.")
    return bytes([HEADER]) + payload[order].astype(np.uint8).tobytes() + bytes([FOOTER])


def build_on_frame() -> bytes:
    """Return a frame that enables every transducer at phase 0."""
    return build_frame([ON_PHASE] * N_EMITTERS)


def build_off_frame() -> bytes:
    """Return the standard all-off frame."""
    return build_frame([OFF_VALUE] * N_EMITTERS)


def run(port: str, dry_run: bool = False) -> None:
    """Send the all-on frame continuously until interrupted."""
    on_frame = build_on_frame()
    print(f"ON frame size: {len(on_frame)} bytes")
    print(f"All transducers enabled at phase {ON_PHASE}.")
    if dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    try:
        with serial.Serial(
            port, BAUDRATE, bytesize=8, parity="N", stopbits=1,
            timeout=2, write_timeout=2, rtscts=False, dsrdtr=False,
        ) as ser:
            ser.dtr = False
            ser.rts = False
            time.sleep(0.3)
            ser.write(on_frame)
            ser.flush()
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
    parser.add_argument("--dry-run", action="store_true", help="Build frames without opening a port")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports detected.")
        else:
            for index, info in enumerate(ports):
                print(f"[{index}] {info.device:<10} - {info.description or 'No description'}")
        return

    port = args.port
    if not args.dry_run and port is None:
        try:
            port = select_serial_port(baudrate=BAUDRATE)
        except (RuntimeError, KeyboardInterrupt) as error:
            parser.error(str(error))
    try:
        run(port, dry_run=args.dry_run)
    except RuntimeError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
