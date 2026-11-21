"""Send phases by WIRE address (shift-register group/slot), not logical emitter.

The FPGA applies PHASE_CORRECTION at the wire byte index (address = group*8 +
slot). This script lets you drive exact wire addresses for oscilloscope tests.

Examples
--------
    # normal mode (40 kHz CW), all wire addresses phase 0 -> table still applies
    python wire_probe.py --port COM6 --all 0 --mode normal

    # one 74HC595 output ON at phase 0, everything else OFF
    python wire_probe.py --port COM6 --all 16 --wire 5=0 --mode normal

    # compare two slots of the same group: slot 0 phase 0, slot 1 phase 8
    python wire_probe.py --port COM6 --all 16 --wire 40=0 --wire 41=8 --mode normal

    # same two addresses, but pre-subtract the active table so the FPGA output
    # phase is exactly what you asked for (needs --table primary|secondary|zero)
    python wire_probe.py --port COM6 --all 16 --wire 40=0 --wire 41=8 \
        --post --table primary --mode normal

    # all eight channels of PCB group 27 ON at phase 0, everything else OFF
    python wire_probe.py --port COM6 --group 27 --mode normal

Wire address k -> group g = k // 8, slot s = k % 8.
Phase values are 0..15; 16 means OFF (0x10).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402
import serial  # noqa: E402

MODE_COMMANDS = {"normal": 0xC8, "c4": 0xC9, "little-star": 0xCA, "200hz": 0xCB}


def find_distribute_files():
    """Locate every Distribute.vhd across the project's renamed working trees.

    The raw primary/secondary tables may be active in one tree and commented
    out in another, so read_tables() scans all of them.
    """
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / name / "src" / "Distribute.vhd"
        for name in ("fpga_code", "work_fpga_code",
                     "latest_work_fpga_code", "fpga_code_bak")
    ]
    candidates.append(root / "ws" / "teammate_pkg" / "fpga_code" / "src" / "Distribute.vhd")
    found = [candidate for candidate in candidates if candidate.is_file()]
    return found or candidates[:1]


DISTRIBUTE = find_distribute_files()[0]


def read_tables(path=None):
    """Return {'primary': [...], 'secondary': [...], 'zero': [...]}.

    Without ``path`` all known source trees are scanned and the first match
    for each table wins.  The raw primary table is often only present as a
    commented line in a tree whose active table has been permuted.
    """
    paths = [Path(path)] if path else find_distribute_files()
    result = {}
    for source in paths:
        text = source.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.count(",") < 100:
                continue
            values = [int(v) for v in re.findall(r"\d+", line)][:256]
            if len(values) != 256:
                continue
            if sum(values) == 0:
                result.setdefault("zero", values)
            elif values[:2] == [8, 8]:
                result.setdefault("secondary", values)
            elif values[:3] == [0, 0, 8]:
                result.setdefault("primary", values)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--baud", type=int, default=protocol.BAUD_RATE)
    p.add_argument("--all", type=int, default=0, help="Phase for all wire addresses (default 0; 16=OFF)")
    p.add_argument("--wire", action="append", default=[], metavar="ADDR=PHASE",
                   help="Set one wire address (repeatable)")
    p.add_argument("--group", type=int, metavar="N",
                   help="Drive all eight slots of PCB group N (1..32), others OFF")
    p.add_argument("--group-phase", type=int, default=0,
                   help="Phase for --group (default 0; 16=OFF)")
    p.add_argument("--steer", metavar="ALPHA,BETA",
                   help="Send the 2D steering pattern from acoustics.steering_phase_bins")
    p.add_argument("--print-bins", action="store_true",
                   help="With --steer, print the wire-order bins for all 32 groups")
    p.add_argument("--mode", choices=sorted(MODE_COMMANDS), default="normal",
                   help="Audio mode byte sent first (default normal)")
    p.add_argument("--keep", action="store_true", help="Do not send a mode byte")
    p.add_argument("--post", action="store_true",
                   help="Pre-subtract the table so the FPGA post-correction phase equals ADDR value")
    p.add_argument("--table", choices=("primary", "secondary", "zero"), default="primary",
                   help="Table assumed for --post (default primary)")
    p.add_argument("--table-file", metavar="VHD",
                   help="Distribute.vhd to read tables from (auto-detected by default)")
    p.add_argument("--lookup", metavar="ROW,COL",
                   help="Print logical/wire address and table values for a physical emitter")
    p.add_argument("--scan", metavar="START:END",
                   help="Cycle one wire address at a time (e.g. 0:7), others OFF")
    p.add_argument("--dwell", type=float, default=3.0,
                   help="Seconds per address in --scan (default 3)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        import serial.tools.list_ports as ports
        for p in ports.comports():
            print(f"{p.device}: {p.description}")
        return 0

    if args.lookup:
        import emitters
        try:
            r, c = (int(x) for x in args.lookup.split(","))
        except ValueError:
            print("--lookup expects ROW,COL (0..15)", file=sys.stderr)
            return 2
        logical = r * 16 + c
        wire = int(emitters.EMITTERS_ORDER.index(logical))
        tables = read_tables(args.table_file)
        print(f"physical row={r} col={c} -> logical index {logical}")
        print(f"wire address {wire} (PCB group {wire//8 + 1}, slot {wire%8}; "
              f"byte-group index {wire//8})")
        for name in ("primary", "secondary", "zero"):
            if name in tables:
                print(f"  {name:9s} correction = {tables[name][wire]}")
        print(f"command phase output = requested + correction; "
              f"requested 0 -> drive phase {'8 (180 deg)' if tables.get('primary',[0]*256)[wire] else '0'}")
        return 0

    if args.scan:
        try:
            first, last = (int(x) for x in args.scan.split(":"))
        except ValueError:
            print("--scan expects START:END (e.g. 0:7)", file=sys.stderr)
            return 2
        if not 0 <= first <= last <= 255:
            print("--scan range must be 0..255 and START <= END", file=sys.stderr)
            return 2
        if args.dry_run:
            for k in range(first, last + 1):
                print(f"wire {k:3d} (PCB group {k//8 + 1:2d}, slot {k%8}) ON")
            return 0
        if not args.port:
            print("--port is required (or --dry-run)", file=sys.stderr)
            return 2
        try:
            with protocol.open_serial(args.port, args.baud) as conn:
                if not args.keep:
                    conn.write(bytes([MODE_COMMANDS[args.mode]]))
                    conn.flush()
                for k in range(first, last + 1):
                    phases = [16] * 256
                    phases[k] = 0
                    frame = bytes([protocol.HEADER]) + bytes(phases) + bytes([protocol.FOOTER])
                    protocol.send_bytes(conn, frame)
                    print(f"wire {k:3d} (PCB group {k//8 + 1:2d}, slot {k%8}) ON  "
                          f"[{k - first + 1}/{last - first + 1}]", flush=True)
                    time.sleep(args.dwell)
        except (serial.SerialException, OSError) as error:
            print(f"Serial error: {error}", file=sys.stderr)
            return 2
        print("scan done; the last wire is still ON")
        return 0

    if not 0 <= args.all <= 16:
        print("--all must be 0..16", file=sys.stderr)
        return 2

    if args.steer:
        if args.group is not None:
            print("--steer cannot be combined with --group", file=sys.stderr)
            return 2
        try:
            alpha_s, beta_s = args.steer.split(",", 1)
            alpha, beta = float(alpha_s), float(beta_s)
        except ValueError:
            print("--steer expects ALPHA,BETA in degrees", file=sys.stderr)
            return 2
        try:
            import acoustics
            import emitters
            logical = acoustics.steering_phase_bins(alpha, beta)
            phases = [int(logical[emitters.EMITTERS_ORDER[k]]) for k in range(256)]
        except (ValueError, ImportError) as error:
            print(f"--steer error: {error}", file=sys.stderr)
            return 2
        display_base = None
        print(f"steering alpha={alpha:g} beta={beta:g}: desired FPGA output bins (wire order)")
        if args.print_bins:
            for g in range(32):
                print(f"  PCB group {g + 1:2d}: " +
                      " ".join(f"{phases[g * 8 + s]:2d}" for s in range(8)))
        print("  (--post makes the FPGA output these bins; without --post it adds the table)")
    elif args.group is not None:
        if not 1 <= args.group <= 32:
            print("--group must be 1..32", file=sys.stderr)
            return 2
        if not 0 <= args.group_phase <= 16:
            print("--group-phase must be 0..16", file=sys.stderr)
            return 2
        phases = [16] * 256
        for k in range((args.group - 1) * 8, args.group * 8):
            phases[k] = args.group_phase
        display_base = 16
        print(f"PCB group {args.group}: slots 0..7 = phase {args.group_phase}; all others OFF")
    else:
        phases = [args.all] * 256
        display_base = args.all

    for item in args.wire:
        try:
            addr_s, phase_s = item.split("=", 1)
            addr, phase = int(addr_s), int(phase_s)
        except ValueError:
            print(f"bad --wire {item!r}; use ADDR=PHASE", file=sys.stderr)
            return 2
        if not 0 <= addr <= 255 or not 0 <= phase <= 16:
            print(f"--wire {item}: address 0..255, phase 0..16", file=sys.stderr)
            return 2
        phases[addr] = phase

    table = None
    if args.post:
        tables = read_tables(args.table_file)
        source = args.table_file or DISTRIBUTE
        if args.table not in tables:
            print(f"table {args.table} not found in {source}", file=sys.stderr)
            return 2
        table = tables[args.table]
        phases = [16 if v == 16 else (v - table[i]) % 16 for i, v in enumerate(phases)]

    # Values sent on the wire (FPGA adds its own table to these).
    print("wire addresses set:")
    explicit = {int(x.split('=')[0]) for x in args.wire}
    shown = 0
    for k, v in enumerate(phases):
        if k not in explicit:
            if display_base is None or v == display_base:
                continue
        print(f"  addr {k:3d} (PCB group {k//8 + 1:2d}, slot {k%8}) = {v}")
        shown += 1
    if shown == 0:
        active = sum(1 for v in phases if v != 16)
        print(f"  (256 bins, {active} active; use --print-bins with --steer to list them)")
    frame = protocol.build_frame_raw(phases) if hasattr(protocol, "build_frame_raw") else None
    if frame is None:
        frame = bytes([protocol.HEADER]) + bytes(phases) + bytes([protocol.FOOTER])

    if args.dry_run:
        print(f"frame {len(frame)} B [dry run]")
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
        print(f"sent {len(frame)} B (wire order, no logical reordering)")
    except (serial.SerialException, OSError) as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
