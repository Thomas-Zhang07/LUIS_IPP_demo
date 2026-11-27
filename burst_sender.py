#!/usr/bin/env python3
"""Send isolated, low-output acoustic bursts to the two-board array.

The FPGA exposes no fast acoustic mute, so a burst is made by gating the two
latched PhaseLine banks:

* ``swap`` (default): prime both banks OFF, load the ON pattern, then toggle
  with ``0xFE, 0xFD`` pairs.  The FE deasserts the swap line so the FD gives
  a rising edge; each pair is about 0.5 ms at 38.4 kBaud.
* ``frame``: send a full ON frame, dwell, then a full OFF frame.  This uses
  only documented frame semantics but the minimum burst is one frame time
  (~67 ms).

Every ON frame uses a low amplitude code by default (code 2, one-tick width,
about -14 dB).  A higher code needs ``--allow-high-amplitude``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import dual_board  # noqa: E402
import protocol  # noqa: E402
from directivity import amplitude_from_code, duty_width_from_code  # noqa: E402
from emitters import N_EMITTERS  # noqa: E402

GATE_SWAP = "swap"
GATE_FRAME = "frame"
GATES = (GATE_SWAP, GATE_FRAME)

# FE deasserts the swap line, FD gives the rising edge.
TOGGLE = bytes([protocol.HEADER, protocol.FOOTER])
PRIMARY_SELECT = 0xC0
SECONDARY_SELECT = 0xC1

DEFAULT_AMPLITUDE_CODE = 2  # width 1, about -14 dB
MAX_DEFAULT_WIDTH = duty_width_from_code(DEFAULT_AMPLITUDE_CODE)  # 1


@dataclass(frozen=True)
class BurstStep:
    """One write to both boards followed by a delay."""

    payload_a: bytes
    payload_b: bytes
    delay_s: float


def build_on_frame(amplitude_code: int = DEFAULT_AMPLITUDE_CODE) -> bytes:
    """Return a uniform all-on frame at ``amplitude_code`` (phase 0)."""
    if not 0 <= int(amplitude_code) <= protocol.MAX_AMPLITUDE_CODE:
        raise ValueError(
            f"amplitude_code must be 0..{protocol.MAX_AMPLITUDE_CODE}."
        )
    phases = np.zeros(N_EMITTERS, dtype=int)
    codes = np.full(N_EMITTERS, int(amplitude_code), dtype=int)
    frame = protocol.build_amplitude_frame(phases, codes)
    return protocol.validate_channel_frame(frame)


def _selected(frame: bytes) -> tuple[bytes, bytes]:
    return (
        bytes([PRIMARY_SELECT]) + frame,
        bytes([SECONDARY_SELECT]) + frame,
    )


def _frame_step(frame: bytes, delay_s: float) -> BurstStep:
    payload_a, payload_b = _selected(frame)
    return BurstStep(payload_a, payload_b, float(delay_s))


def _toggle_step(delay_s: float) -> BurstStep:
    return BurstStep(TOGGLE, TOGGLE, float(delay_s))


def build_burst_plan(
    gate: str,
    amplitude_code: int = DEFAULT_AMPLITUDE_CODE,
    repeats: int = 1,
    dwell_s: float = 0.004,
    gap_s: float = 0.1,
    allow_high_amplitude: bool = False,
) -> list[BurstStep]:
    """Return the ordered steps for one gated burst run.

    Two OFF frames prime both banks.  ``frame`` then alternates full ON and
    OFF frames; ``swap`` loads one ON frame and toggles the banks with
    ``FE, FD`` pairs.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {', '.join(GATES)}.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if dwell_s < 0.0 or gap_s < 0.0:
        raise ValueError("dwell and gap must be non-negative.")
    if (
        not allow_high_amplitude
        and duty_width_from_code(amplitude_code) > MAX_DEFAULT_WIDTH
    ):
        raise ValueError(
            f"amplitude code {amplitude_code} is wider than the low-output "
            f"default (width {MAX_DEFAULT_WIDTH}); pass --allow-high-amplitude "
            "to use it. Codes 0 and 15 are full scale."
        )
    on_frame = build_on_frame(amplitude_code)
    off_frame = protocol.validate_channel_frame(protocol.build_off_frame())

    steps = [_frame_step(off_frame, 0.0), _frame_step(off_frame, 0.0)]
    if gate == GATE_FRAME:
        for _ in range(repeats):
            steps.append(_frame_step(on_frame, dwell_s))
            steps.append(_frame_step(off_frame, gap_s))
        return steps

    steps.append(_frame_step(on_frame, dwell_s))
    for repeat in range(repeats):
        steps.append(_toggle_step(gap_s))
        if repeat < repeats - 1:
            steps.append(_toggle_step(dwell_s))
    return steps


def run_burst(link, steps, sleep=time.sleep) -> None:
    """Write every step to both boards, then sleep its delay."""
    for step in steps:
        link.send(step.payload_a, step.payload_b)
        sleep(step.delay_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gate", choices=GATES, default=GATE_SWAP,
        help="swap = FE+FD bank toggle (~0.5 ms edges); "
             "frame = full ON/OFF frames (~67 ms minimum)",
    )
    parser.add_argument(
        "--amplitude-code", type=int, default=DEFAULT_AMPLITUDE_CODE,
        help=f"ON-frame amplitude code 0..15; 0 and 15 are full scale "
             f"(default {DEFAULT_AMPLITUDE_CODE} = width 1)",
    )
    parser.add_argument(
        "--allow-high-amplitude", action="store_true",
        help="permit an amplitude code above the low-output default",
    )
    parser.add_argument("--dwell-ms", type=float, default=4.0,
                        help="burst length after the ON edge (default 4 ms)")
    parser.add_argument("--gap-ms", type=float, default=100.0,
                        help="off time between repeats (default 100 ms)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of bursts (default 1)")
    parser.add_argument("--port-a", help="primary board serial port")
    parser.add_argument("--port-b", help="secondary board serial port")
    parser.add_argument("--trim", type=int, default=None,
                        help="send this secondary trim before the burst")
    parser.add_argument("--baud", type=int, default=protocol.BAUD_RATE)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without opening a port")
    parser.add_argument("--list-ports", action="store_true",
                        help="list serial ports and exit")
    return parser


def _relative_amplitude_db(code: int) -> float:
    return 20.0 * math.log10(max(amplitude_from_code(code), 1e-12))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        from serial_utils import print_available_ports

        print_available_ports()
        return 0

    try:
        steps = build_burst_plan(
            args.gate,
            amplitude_code=args.amplitude_code,
            repeats=args.repeats,
            dwell_s=args.dwell_ms / 1000.0,
            gap_s=args.gap_ms / 1000.0,
            allow_high_amplitude=args.allow_high_amplitude,
        )
    except ValueError as error:
        print(f"Cannot build the burst plan: {error}", file=sys.stderr)
        return 2

    if args.trim is not None and not 0 <= args.trim <= 255:
        print("--trim must be 0..255.", file=sys.stderr)
        return 2

    frames = sum(
        1 for step in steps
        if step.payload_a[:1] in (bytes([PRIMARY_SELECT]), bytes([SECONDARY_SELECT]))
    )
    toggles = sum(1 for step in steps if step.payload_a == TOGGLE)
    print(f"gate: {args.gate}, repeats: {args.repeats}")
    print(
        f"amplitude code: {args.amplitude_code} "
        f"({_relative_amplitude_db(args.amplitude_code):+.1f} dB vs the "
        "full-scale code)"
    )
    if duty_width_from_code(args.amplitude_code) > MAX_DEFAULT_WIDTH:
        print(
            "WARNING: full-scale/high output selected; keep the physical "
            "inhibit until the approved capture."
        )
    print(f"dwell: {args.dwell_ms:g} ms, gap: {args.gap_ms:g} ms")
    print(f"steps: {len(steps)} ({frames} full frames, {toggles} toggles)")
    print(
        "timing is host-limited; take the start reference from the first "
        "carrier edge after silence and verify it on the scope."
    )
    print(
        "keep the physical output inhibit until an approved capture; the ON "
        "frame requests the selected pulse code, not a calibrated level."
    )

    if args.dry_run:
        print("[DRY RUN] no serial data sent.")
        return 0

    if not (args.port_a and args.port_b):
        print(
            "Both --port-a and --port-b are required (or use --dry-run).",
            file=sys.stderr,
        )
        return 2

    try:
        link = dual_board.BoardLink(args.port_a, args.port_b, baudrate=args.baud)
        with link:
            if args.trim is not None:
                link.send_trim(args.trim)
                print(f"trim {args.trim}")
            run_burst(link, steps)
    except OSError as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
