#!/usr/bin/env python3
"""Temporarily cancel each board's FPGA phase table for a two-board scope test.

The firmware drives  drive = (command + PHASE_CORRECTION[wire]) mod 16.
To make the FPGA output exactly the commanded phase, pre-subtract that board's
own table:
    command = (desired - table[wire]) mod 16

The primary and secondary do NOT share a table: the primary's active table is
`boardsPrimary`, the secondary's is `boardsSecondary`, and they differ at 134
of 256 channels (0 vs 8 bins). Each table compensates its own board's
transducer phase, so each payload must use its own table.

This sends an all-on frame with every channel commanded to phase 0 *after*
correction, to both boards:
    primary port   -> select 0xC0, primary table pre-subtracted
    secondary port -> select 0xC1, secondary table pre-subtracted

With both tables cancelled, the FPGA drive is phase 0 on every channel on both
boards. If the driver pads were 180 deg apart only because of the tables, they
should now line up.

Examples:
    python table_cancel_test.py --dry-run
    python table_cancel_test.py --port-a COM7 --port-b COM6
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

import protocol  # noqa: E402

PRIMARY_DISTRIBUTE = DEMO.parent / "fpga" / "primary" / "src" / "Distribute.vhd"
SECONDARY_DISTRIBUTE = DEMO.parent / "fpga" / "secondary" / "src" / "Distribute.vhd"


def read_table(path):
    """Return the 256-entry ACTIVE phase table from a Distribute.vhd.

    Commented-out candidate tables are skipped; the first uncommented
    256-value line after the PHASE_CORRECTION declaration is the active one.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    started = False
    for line in text.splitlines():
        if "PHASE_CORRECTION : T_PHASE_CORRECTION :=" in line:
            started = True
            continue
        if not started:
            continue
        stripped = line.strip()
        if stripped.startswith("--") or stripped.count(",") < 100:
            continue
        values = [int(v) for v in re.findall(r"\d+", stripped)][:256]
        if len(values) == 256:
            return values
    raise ValueError(f"no active 256-entry table found in {path}")


def build_payload(table, desired=0):
    """All channels ON (amplitude 0 = full) with the table pre-subtracted."""
    phases = [(desired - t) % 16 for t in table]
    return bytes([protocol.HEADER]) + bytes(phases) + bytes([protocol.FOOTER])


def send(port, select, mode, payload, repeats=2):
    with protocol.open_serial(port) as conn:
        conn.write(bytes([mode]))
        conn.flush()
        for _ in range(repeats):
            conn.write(bytes([select]) + payload)
            conn.flush()
    return len(payload) * repeats + 1


def describe(name, path, table, payload):
    print(f"{name} table: {path}")
    print(f"  first 16: {table[:16]}")
    print(f"  entries of 8 (180 deg): {sum(1 for v in table if v == 8)} / 256")
    print(f"  pre-subtracted command, first 16: {list(payload[1:17])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port-a", help="primary port (select 0xC0)")
    p.add_argument("--port-b", help="secondary port (select 0xC1)")
    p.add_argument("--primary-table-file", default=str(PRIMARY_DISTRIBUTE))
    p.add_argument("--secondary-table-file", default=str(SECONDARY_DISTRIBUTE))
    p.add_argument("--mode", type=int, default=0xC8,
                   help="0xC8 normal (default), 0xC9 C4, 0xCA Little Star, 0xCB 200 Hz")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    table_a = read_table(args.primary_table_file)
    table_b = read_table(args.secondary_table_file)
    payload_a = build_payload(table_a)
    payload_b = build_payload(table_b)

    describe("primary", args.primary_table_file, table_a, payload_a)
    describe("secondary", args.secondary_table_file, table_b, payload_b)
    differing = sum(1 for x, y in zip(table_a, table_b) if (y - x) % 16)
    print(f"tables differ on {differing} / 256 channels")
    print(f"frame: {len(payload_a)} bytes each, all channels ON, output phase 0")

    if args.dry_run:
        print("[dry run] nothing sent")
        return 0
    if not args.port_a or not args.port_b:
        print("both --port-a and --port-b are required (or use --dry-run)",
              file=sys.stderr)
        return 2
    n_a = send(args.port_a, 0xC0, args.mode, payload_a)
    n_b = send(args.port_b, 0xC1, args.mode, payload_b)
    print(f"sent {n_a} B to {args.port_a} (0xC0) and {n_b} B to {args.port_b} (0xC1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
