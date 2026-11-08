# LUIS IPP Demo — 16×16 Ultrasonic Array

Python demos for a 256-transducer (16×16) ultrasonic array driven over UART.
The scripts cover single-point focusing, vortex beams, moving focal points,
beam steering, array-factor analysis, and simple ON/OFF patterns.

## Hardware and protocol

- **Array:** 16×16 = 256 transducers, 10 mm spacing, lying in the z = 0 plane.
  The coordinate origin is the bottom-left emitter; x and y increase to the
  right and upward.
- **Carrier:** 40 kHz by default, with an assumed speed of sound of 343 m/s
  (20 °C). Scripts that compute phases accept `--frequency`/`--sound-speed`
  (or `--carrier-frequency` on `transducer_onoff_steer.py`).
- **UART:** 38400 baud, 8N1.
- **Frame:** `0xFE` + 256 payload bytes + `0xFD` (258 bytes).
  Payload values `0..15` select a phase bin; `0x10` disables a transducer.
- **Transmit order:** the 256 payload bytes are not in logical row/column
  order. `emitters.py` holds the geometry and the hardware mapping; treat it
  as the hardware contract.

At 38400 baud one frame takes about **67 ms** on the wire. A full-frame ON/OFF
toggle therefore tops out near **7.4 Hz**. The ON/OFF scripts accept rates from
0.1 Hz to 20000 Hz for compatibility, but only rates below about 7.4 Hz are
achievable by resending full frames; they print this limit and warn when a
request exceeds it.

## Requirements

- Python 3.10 or newer
- `numpy`, `pyserial`
- `pytest` to run the test suite
- `matplotlib` (optional) for `--plot-array-factor` and
  `--plot-phase-rotation`

```bash
python -m pip install -r requirements.txt
# optional plotting support
python -m pip install matplotlib
```

## Quick start

Dry runs compute frames and print phase maps without touching hardware:

```bash
python single_point_hover.py --dry-run --x 0.075 --y 0.075 --z 0.05
python single_point_vortex.py --dry-run --x 0.075 --y 0.075 --z 0.05 --charge 1
python single_point_circle.py --dry-run --x 0.075 --y 0.075 --z 0.05
python transducer_phase_steer_2d.py --dry-run --alpha 0 --beta 30
python mode_control.py --dry-run
```

Send to a board by adding `--port COM3` (or run without `--port` to pick from a
list of detected ports):

```bash
python single_point_hover.py --x 0.075 --y 0.075 --z 0.05 --port COM3
python single_point_vortex.py --x 0.075 --y 0.075 --z 0.05 --charge -1 --port COM3
python transducer_phase_steer.py --angle 45 --port COM3
```

## Scripts

### Modules

| File | Purpose |
|------|---------|
| `emitters.py` | Grid geometry, logical indices, hardware transmit order. |
| `protocol.py` | Frame constants, validation, frame builders, serial transport, wire-time math. |
| `acoustics.py` | Wavelength, focus/vortex/steering phases, phase quantization. |
| `cli_utils.py` | Shared argparse groups, target prompting, phase-map display. |
| `trajectory.py` | Circle/line point generation, deadline-based streaming, ON/OFF, bit patterns, rotation. |
| `array_factor.py` | Theoretical array-factor and sidelobe analysis, plotting. |
| `serial_utils.py` | Port discovery and selection. |

### Focus and trajectories

| Script | Behavior | Notable flags |
|--------|----------|---------------|
| `single_point_hover.py` | One focus frame at a target point. | `--x --y --z`, `--detect`, `--dry-run`, `--baud`, `--frequency`, `--sound-speed` |
| `single_point_vortex.py` | Focus plus a topological-charge phase (vortex). Interactive runs accept `ccw`/`cw` (逆时针/顺时针). | adds `--charge` |
| `single_point_line.py` | Moves the focus back and forth along x over a fixed 3 cm span. | `--period` (default 2 s), `--points` (default 48) |
| `single_point_circle.py` | Moves the focus around a 3 cm **diameter** circle (1.5 cm radius) in the XY plane. | `--period` (default 2 s), `--points` (default 48) |

Default targets differ: `single_point_hover.py` prompts with
`(0.075, 0.075, 0.0)`, while vortex/line/circle use
`(0.075, 0.075, 0.05)`.

Ctrl+C stops line and circle streaming **without** sending an all-off frame;
the board keeps its last state.

### ON/OFF and patterns

| Script | Behavior | Notable flags |
|--------|----------|---------------|
| `transducer_all_on.py` | Sends an all-on frame (phase 0) and holds it. Ctrl+C keeps the state. | `--port`, `--dry-run`, `--list-ports` |
| `transducer_onoff_demo.py` | Toggles the whole array ON/OFF at a requested rate, with timestamps. | `--frequency`, `--duration`, `--dry-run` |
| `onoff.py` | Same toggle, one serial connection per run. | `--frequency`, `--duration`, `--dry-run` |
| `bit_pattern_onoff.py` | Repeats an 8-bit pattern (`1` = ON, `0` = OFF). | `--pattern`, `--bit-seconds`, `--duration`, `--dry-run` |
| `mode_control.py` | Sends a mode command byte (`0xC8`/`0xC9`/`0xCA`/`0xCB`) followed by an all-off frame. | `--mode`, `--port`, `--dry-run`, `--list-ports` |

The `--frequency` on these scripts is the ON/OFF envelope rate, not the
carrier. `transducer_onoff_steer.py` also sends an OFF frame before exiting.

### Beam steering

| Script | Behavior | Flags |
|--------|----------|-------|
| `transducer_phase_steer.py` | Static steering with one angle in the x–z plane: 0° = +x, 90° = +z (array normal), 180° = −x. | `--angle`, `--frequency`, `--sound-speed`, `--dry-run` |
| `transducer_onoff_steer.py` | Toggles an angle-steered ON frame against an all-off frame. | `--angle`, `--frequency` (toggle), `--carrier-frequency`, `--sound-speed`, `--duration`, `--dry-run` |
| `transducer_phase_steer_2d.py` | Static 2D steering, rotation through two or four azimuths, and optional plots. | `--alpha`, `--beta`, `--angle`/`--legacy-angle`, `--rotate`, `--mode {two,four}`, `--rotation-period`, `--update-interval`, `--duration`, `--plot-array-factor`, `--plot-phase-rotation`, `--dry-run` |

Steering conventions:

- **Azimuth** `alpha` is measured in the x–y plane from +x toward +y
  (`-360..360`).
- **Off-axis** `beta` is measured from +z (`0..180`). `beta = 0` is broadside.
- The 2D array is treated as a 1D array along x for the single-angle legacy
  steering; the phase center is x = 0.075 m.
- The phase sign assumes `exp(i·(ωt − k·r))`. **The firmware's phase-bin
  mapping and the physical beam direction still require hardware
  verification.**

## Notes and limitations

- **Phase sign (hardware-verify):** focusing computes `−k·d` while plane-wave
  steering computes `−k·(r·u)`. Under the documented `exp(i(ωt − k·r))`
  convention, a converging focus requires `+k·d`, so the legacy focus sign is
  inverted relative to steering. Both formulas are preserved, but they cannot
  both be correct for one firmware bin-to-phase sign. Verify focus and
  steering separately on hardware.
- **Vortex center:** the vortex phase is computed around the mean array
  center, not the target's transverse position, so an off-axis vortex spirals
  about the array axis. This preserves legacy behavior.
- **No all-off frame on interrupt** for most scripts: line, circle, hover,
  vortex, onoff, bit pattern and the demo scripts leave the board in its last
  state. `transducer_onoff_steer.py` is the exception and sends OFF on exit.
- **ON/OFF rate:** full-frame toggling is limited to about 7.4 Hz by the UART.
- **Phase resolution:** 16 bins (0..15), plus `0x10` for off.
- **Quantization** snaps values within a few ULPs of an exact bin boundary so
  mathematically identical phases do not land in different bins.
- **Array-factor plots are theoretical** idealized models (isotropic
  elements, ideal bin phases). They are not measurements.
- **Interactive vs. command line:** if coordinates/angles are omitted and the
  script would need them, it prompts. `--dry-run` never prompts for a serial
  port; scripts whose dry-run also needs other parameters state that in their
  error messages.

## Tuning physics

Wavelength is derived from:

```text
lambda = sound_speed / frequency
```

Defaults are 343 m/s and 40 kHz → 8.575 mm. The focus phase for an emitter at
distance `d` from the target is:

```text
phase = mod(-2*pi*f*d / c, 2*pi)
```

Steering phases project centered emitter positions onto the requested unit
direction and use the same `-k·r` convention.

## Tests

The suite runs without hardware: serial and time are faked.

```bash
python -m pytest -q
python -m compileall -q .
```

Coverage includes the 258-byte frame layout and transmit-order permutation,
focus/steering math and legacy-formula equivalence, quantization boundary
handling, deadline scheduling without drift, one-open-per-run serial behavior,
dry-run safety, mode-control payloads, and CLI flag preservation.

## Repository layout

```text
emitters.py                    hardware geometry and transmit order
protocol.py                    UART frame protocol and serial transport
acoustics.py                   physics: wavelength, phases, quantization
cli_utils.py                   shared CLI and display helpers
trajectory.py                  timed streaming, toggles, rotation
array_factor.py                theoretical analysis and plotting
serial_utils.py                port discovery and selection
single_point_*.py              focus, vortex, line, circle demos
transducer_*.py                all-on, ON/OFF, steering demos
onoff.py, bit_pattern_onoff.py ON/OFF and pattern generators
mode_control.py                mode command + all-off frame
tests/                         hardware-free pytest suite
```

See `CHANGELOG.md` for changes.
