#!/usr/bin/env python3
"""Send a mode-selection byte followed by a phase frame.

The command byte is outside the 258-byte frame protocol; it is sent first,
followed by a whole-array frame. The modes gate the 40 kHz drive on the FPGA,
so the frame must leave the transducers on (phase 0) for any modulation to be
audible:

    --frame on    all-on frame, ``0xFE + 256 x 0x00 + 0xFD`` (default)
    --frame off   all-off frame, ``0xFE + 256 x 0x10 + 0xFD``
    --frame keep  mode byte only; the last latched phase pattern stays

The mode labels are inherited from the original scratch script:

    0xC8  200  Normal mode (no modulation)
    0xC9  201  C4
    0xCA  202  Little star
    0xCB  203  200 Hz
"""

import argparse
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from protocol import (  # noqa: E402
    BAUD_RATE,
    build_off_frame,
    build_on_frame,
    open_serial,
    send_frame,
)
from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from cli_utils import run_cli  # noqa: E402

MODE_COMMANDS = {
    "normal": 0xC8,
    "c4": 0xC9,
    "little-star": 0xCA,
    "200hz": 0xCB,
}
DEFAULT_MODE = "c4"
FRAME_CHOICES = ("on", "off", "keep")
DEFAULT_FRAME = "on"


def build_mode_payload(mode: str = DEFAULT_MODE, frame: str = DEFAULT_FRAME) -> bytes:
    """Return the command byte followed by the requested whole-array frame.

    ``frame="keep"`` sends only the command byte, leaving whatever phase
    pattern is already latched on the board untouched.
    """
    try:
        command = MODE_COMMANDS[mode]
    except KeyError as error:
        raise ValueError(f"Unknown mode {mode!r}.") from error

    if frame == "on":
        return bytes([command]) + build_on_frame()
    if frame == "off":
        return bytes([command]) + build_off_frame()
    if frame == "keep":
        return bytes([command])
    raise ValueError(f"Unknown frame {frame!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a mode-selection byte followed by a whole-array phase frame."
    )
    parser.add_argument("--port", help="Serial port, e.g. COM9")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_COMMANDS),
        default=DEFAULT_MODE,
        help=f"Mode command to send (default {DEFAULT_MODE}, 0xC9)",
    )
    parser.add_argument(
        "--frame",
        choices=FRAME_CHOICES,
        default=DEFAULT_FRAME,
        help=(
            "Whole-array frame sent after the mode byte: 'on' (default) keeps "
            "the transducers driven so modulation is audible, 'off' mutes the "
            "array, 'keep' leaves the latched phase pattern untouched"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the payload without opening a port"
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="List serial ports and exit"
    )
    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    payload = build_mode_payload(args.mode, args.frame)
    command = payload[0]
    print(f"Mode: {args.mode} (0x{command:02X})")
    if args.frame == "keep":
        print("Payload: 1 byte (mode command only; phase pattern kept)")
    else:
        print(
            f"Payload: {len(payload)} bytes "
            f"(1 command + 258-byte {args.frame.upper()} frame)"
        )

    if args.dry_run:
        print(payload.hex(" "))
        print("[DRY RUN] No serial data sent.")
        return

    port = args.port
    if port is None:
        try:
            port = select_serial_port(baudrate=BAUD_RATE)
        except (RuntimeError, KeyboardInterrupt) as error:
            print(f"Cannot select a serial port: {error}")
            return

    try:
        with open_serial(port) as connection:
            send_frame(connection, payload)
    except OSError as error:
        print(f"Serial error: {error}")
        return

    print("done")


if __name__ == "__main__":
    run_cli(main)
