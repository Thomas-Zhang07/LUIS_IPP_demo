"""A/B the phase-correction table ordering without reflashing the FPGA.

The FPGA always adds its stored table (`PHASE_CORRECTION[wire]`) to the phase
byte you send. This script pre-subtracts the stored table and adds a desired
table, so the *output* phase is the desired one:

    sent[k] = desired[k] - stored[k]   (mod 16)

Then measure the on-axis level at broadside (all phases equal). The correct
table ordering gives the highest coherent sum.

Examples
--------
    # output phases follow the table exactly as stored in fpga_code
    python table_test.py --port COM6 --mode 200hz --desired stored

    # output phases follow the table permuted logical -> wire
    python table_test.py --port COM6 --mode 200hz --desired permuted

    # no correction (all output phases 0)
    python table_test.py --port COM6 --mode 200hz --desired zero

Run the three back-to-back with the dB meter fixed at broadside.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import emitters  # noqa: E402
import protocol  # noqa: E402
import serial  # noqa: E402
from wire_probe import MODE_COMMANDS, read_tables  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--baud", type=int, default=protocol.BAUD_RATE)
    p.add_argument("--desired", choices=("stored", "permuted", "zero"),
                   default="stored",
                   help="Output phase table (default stored)")
    p.add_argument("--mode", choices=sorted(MODE_COMMANDS), default="200hz",
                   help="Audio mode byte (default 200hz; use normal for CW)")
    p.add_argument("--keep", action="store_true", help="Do not send the mode byte")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        import serial.tools.list_ports as ports
        for p in ports.comports():
            print(f"{p.device}: {p.description}")
        return 0

    tables = read_tables()
    if "primary" not in tables:
        print("primary table not found in fpga_code/src/Distribute.vhd", file=sys.stderr)
        return 2
    stored = tables["primary"]                     # what the FPGA currently adds
    permuted = [stored[int(i)] for i in emitters.EMITTERS_ORDER]

    desired = {
        "stored": stored,
        "permuted": permuted,
        "zero": [0] * 256,
    }[args.desired]

    sent = [(d - s) % 16 for d, s in zip(desired, stored)]
    frame = bytes([protocol.HEADER]) + bytes(sent) + bytes([protocol.FOOTER])

    print(f"FPGA stored table = primary (wire order), {sum(1 for v in stored if v)} ones")
    print(f"desired output    = {args.desired}, {sum(1 for v in desired if v)} ones")
    print(f"frame 258 B; mode {args.mode} (0x{MODE_COMMANDS[args.mode]:02X})")
    if args.dry_run:
        print("first 16 sent bytes:", sent[:16])
        print("[dry run]")
        return 0
    if not args.port:
        print("--port is required (or --dry-run)", file=sys.stderr)
        return 2

    try:
        with protocol.open_serial(args.port, args.baud) as conn:
            if not args.keep:
                conn.write(bytes([MODE_COMMANDS[args.mode]]))
                conn.flush()
            protocol.send_bytes(conn, frame)
        print(f"sent desired={args.desired} to {args.port}")
    except (serial.SerialException, OSError) as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
