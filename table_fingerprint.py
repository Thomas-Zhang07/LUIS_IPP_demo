"""Print per-group phase-correction fingerprints for oscilloscope checks.

The FPGA adds ``PHASE_CORRECTION[wire]`` to every phase byte it receives:

    drive[wire] = (command + table[wire]) mod 16

With ``command = 0`` the drive of every channel is exactly the stored table
entry: 0 deg for entry 0, 180 deg for entry 8.  This script prints, for one
shift-register group, what each candidate table predicts for the eight slots,
so a two-probe measurement can say *which* table is really on the board.

The candidates are parsed from both source trees:

* ``primary raw`` / ``secondary raw``  -- the tables as imported from the
  teammate (``fpga_code/src/Distribute.vhd``).
* ``primary perm`` / ``secondary perm`` -- the same tables permuted to wire
  order with ``emitters.EMITTERS_ORDER`` (the active table in
  ``work_fpga_code/src/Distribute.vhd`` at the time of writing).
* ``zero`` -- no correction.

Relative phase (each slot against slot Q0 of the same group) is the quantity
that is independent of the driver polarity, so use the ``rel`` rows when
triggering on Q0.

Examples
--------
    python table_fingerprint.py                    # rank the best groups
    python table_fingerprint.py --group 9 --group 10 --group 27
    python table_fingerprint.py --group 27 --command 8
    python table_fingerprint.py --wire 64
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import emitters  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "fpga_code": ROOT / "fpga_code" / "src" / "Distribute.vhd",
    "work_fpga_code": ROOT / "work_fpga_code" / "src" / "Distribute.vhd",
}

ZERO_NAMES = ("zero",)


def parse_file(path: Path):
    """Return [(values, commented)] for every 256-value table line."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count(",") < 100:
            continue
        values = [int(v) for v in re.findall(r"\d+", stripped)][:256]
        if len(values) == 256:
            out.append((values, stripped.startswith("--")))
    return out


def load_candidates():
    """Return {name: table}, {source: active-table-name} and warnings."""
    raw_primary = raw_secondary = None
    active_values = {}
    for source, path in SOURCES.items():
        tables = parse_file(path)
        if tables:
            active_values[source] = tables[0][0]
        for values, _commented in tables:
            if len(set(values)) == 1 and values[0] == 0:
                continue
            if values[:2] == [8, 8]:
                raw_secondary = raw_secondary or values
            elif values[:6] == [0, 0, 8, 8, 8, 8]:
                raw_primary = raw_primary or values

    candidates = {}
    if raw_primary is not None:
        candidates["primary raw"] = raw_primary
    if raw_secondary is not None:
        candidates["secondary raw"] = raw_secondary

    order = emitters.EMITTERS_ORDER
    for name, table in list(candidates.items()):
        candidates[name.replace(" raw", " perm")] = [table[order[k]] for k in range(256)]
    candidates["zero"] = [0] * 256

    # Deduplicate while keeping the first name.
    unique = {}
    seen = set()
    for name, table in candidates.items():
        key = tuple(table)
        if key not in seen:
            seen.add(key)
            unique[name] = table
    candidates = unique

    active = {}
    for source, values in active_values.items():
        key = tuple(values)
        match = next((name for name, table in candidates.items() if tuple(table) == key), None)
        active[source] = match or "unrecognised"

    return candidates, active


def distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def group_score(candidates, group):
    wires = range(group * 8, group * 8 + 8)
    tables = [[table[w] for w in wires] for table in candidates.values()]
    # Sum over every pair: high means all candidate patterns are far apart.
    return sum(distance(a, b) for i, a in enumerate(tables) for b in tables[i + 1:])


def phase_text(values):
    return " ".join("180" if v else "  0" for v in values)


def show_group(candidates, group, command):
    wires = list(range(group * 8, group * 8 + 8))
    print(f"\nPCB group {group + 1}  (DATA[{group}], wires {wires[0]}..{wires[-1]}, Q0..Q7)")
    print("expected drive with command =", command,
          "(absolute | relative to Q0)")
    for name, table in candidates.items():
        absolute = [(table[w] + command) % 16 for w in wires]
        relative = [(v - absolute[0]) % 16 for v in absolute]
        print(f"  {name:16s} abs: {phase_text(absolute)}   rel: {phase_text(relative)}")
    flags = []
    for w in wires:
        flags.append(f"--wire {w}={command}")
    print("  (if the measured Q0..Q7 sequence is a row read right-to-left, the "
          "PCB Q numbering is reversed vs the byte slots)")
    print("  drive only this group (all others OFF):")
    print("    python wire_probe.py --port COM6 --mode normal --all 16 \\")
    print("        " + " ".join(flags))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--group", type=int, action="append", default=[],
                        metavar="N", help="PCB group 1..32 (repeatable)")
    parser.add_argument("--command", type=int, default=0,
                        help="phase byte sent to every slot (default 0)")
    parser.add_argument("--wire", type=int, action="append", default=[],
                        metavar="W", help="show one wire address (repeatable)")
    parser.add_argument("--rank", type=int, default=8, metavar="N",
                        help="show the N most discriminating groups (default 8)")
    args = parser.parse_args(argv)

    if not 0 <= args.command <= 15:
        parser.error("--command must be 0..15")

    candidates, active = load_candidates()
    if not candidates:
        print("No correction tables found in the source trees.", file=sys.stderr)
        return 2

    print("candidate tables")
    for name in candidates:
        print(f"  {name}")
    print("\nactive table in each source tree (first uncommented table):")
    for source, name in active.items():
        print(f"  {source:16s} -> {name}")

    if args.wire:
        for wire in sorted(args.wire):
            group, slot = divmod(wire, 8)
            print(f"\nwire {wire}  (group {group + 1}, Q{slot})")
            for name, table in candidates.items():
                value = table[wire]
                print(f"  {name:16s} {value:3d} deg -> drive with command "
                      f"{args.command}: {(value + args.command) % 16 * 22.5:.1f} deg")

    for group in args.group:
        if not 1 <= group <= 32:
            parser.error("--group must be 1..32")
        show_group(candidates, group - 1, args.command)

    if args.rank and not args.group and not args.wire:
        ranked = sorted(
            ((group_score(candidates, g), g) for g in range(32)),
            reverse=True,
        )
        print("\nmost discriminating groups (sum of pairwise slot differences):")
        for score, group in ranked[: args.rank]:
            print(f"  group {group + 1:2d} (wires {group * 8:3d}..{group * 8 + 7:3d}): "
                  f"score {score}")
        print("\nRe-run with --group N to print the expected patterns.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
