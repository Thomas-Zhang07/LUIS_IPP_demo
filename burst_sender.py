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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import protocol  # noqa: E402
from emitters import N_EMITTERS  # noqa: E402

GATE_SWAP = "swap"
GATE_FRAME = "frame"
GATES = (GATE_SWAP, GATE_FRAME)

# FE deasserts the swap line, FD gives the rising edge.
TOGGLE = bytes([protocol.HEADER, protocol.FOOTER])
PRIMARY_SELECT = 0xC0
SECONDARY_SELECT = 0xC1

DEFAULT_AMPLITUDE_CODE = 2  # width 1, about -14 dB
MAX_DEFAULT_AMPLITUDE_CODE = 2


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
    if amplitude_code > MAX_DEFAULT_AMPLITUDE_CODE and not allow_high_amplitude:
        raise ValueError(
            f"amplitude code {amplitude_code} is above the low-output default "
            f"({MAX_DEFAULT_AMPLITUDE_CODE}); pass --allow-high-amplitude to "
            "use it."
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
