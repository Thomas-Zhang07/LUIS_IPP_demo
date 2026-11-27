# Burst / gate host tool — design

2026-11-27. Short design for the measurement-plan burst prerequisite.

## Intent

The two-board measurement run sheet needs an isolated, low-output acoustic
burst and a usable burst-start reference. The documented FPGA build exposes no
fast acoustic mute:

- `0x96` / `0x97` (shift clocks) and `0xC0` / `0xC1` (board enable) drive
  LEDs 3 and 2 only in the generated netlist (`QuadrupleBuffer.hier_info`).
- The board-enable branch stops channel *writes*; it does not mute the latched
  output.
- The mode commands (`0xC9`/`0xCA`/`0xCB`) select music tones, not gates.

So the burst must be produced host-side by swapping the two `PhaseLine` banks.

## Goals

- A host CLI that emits isolated ON → dwell → OFF bursts on both boards.
- Two gate mechanisms, selectable:
  - `swap` (default): `0xFE` deasserts the swap line, `0xFD` swaps banks;
    ~0.5 ms edges, so a 4 ms burst is feasible.
  - `frame` (fallback): full ON and OFF frames; minimum burst ≈ one frame
    time (~67 ms at 38.4 kBaud).
- Low output by default: the ON frame uses a low amplitude code (width 1,
  ≈−14 dB) rather than full width. A higher code needs an explicit flag.
- Dry-run output, byte-level tests with fake serial, and clear limitations.

## Non-goals

- No FPGA/RTL change and no dedicated electrical start strobe.
- No acoustic validation; bench timing and echo behaviour must be measured.
- No guarantee for a shared UART bus (separate ports are the documented path).

## Constraints

- Whole frame is 258 bytes at 38.4 kBaud → ~67 ms.
- A swap needs a rising edge: `0xFE` must precede each `0xFD`, because
  consecutive `0xFD` bytes leave the swap line asserted.
- The trim register resets to 127 on power-up; the operator re-applies the
  approved value.
- Keep the physical output inhibit until an approved capture.

## Approaches considered

1. **Frame-latch only** — send ON frame, dwell, OFF frame. Uses documented
   frame semantics; minimum burst ~67 ms.
2. **Bank-toggle only** — load ON, then `FE, FD` per transition. Fast edges;
   relies on bare command bytes and needs bench validation.
3. **Both, default bank-toggle** — chosen. Keeps the conservative frame path
   available if the bench finds bare-command toggling unreliable.

## Design

New `demo/burst_sender.py`:

- `--port-a`, `--port-b`, `--dry-run`, `--list-ports`.
- `--gate {swap,frame}` default `swap`.
- `--dwell-ms` (burst length after the ON edge), `--gap-ms` (off time between
  repeats), `--repeats`.
- `--amplitude-code` default `2` (width 1 ≈ −14 dB); reject `0x10`.
- `--allow-high-amplitude` required for any pulse width wider than code 2
  (including codes 0 and 15, both full scale), so full-width drive is never
  the accidental default.
- Sequence (`swap`): send the OFF frame twice to prime both banks OFF, send
  the ON frame, then per repeat send `FE, FD` to go OFF after the dwell and
  `FE, FD` to go ON for the next repeat.
- Sequence (`frame`): send the OFF frame twice, then per repeat send the ON
  frame, sleep the dwell, send the OFF frame.
- Print the frame count, gate mode, requested dwell/gap, and the caveat that
  timing is host-limited and the start reference is the first carrier edge
  after silence.

Tests (`tests/test_burst_sender.py`) use a fake serial connection and assert
byte order, the `FE`-before-`FD` pairing, repeat count, both ports, the
low-output default, and that every full frame passes the channel-frame lint.

## Open / bench

- Validate that `FE, FD` reliably toggles the banks on both installed builds.
- Measure edge jitter, minimum burst length and the usable pre-echo window.
- Decide whether the first carrier edge is an adequate start reference or a
  one-shot strobe is required (RTL change).
