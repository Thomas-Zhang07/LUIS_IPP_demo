# LUIS IPP Demo — 16×16 Ultrasonic Array

Python demos for a 256-transducer (16×16) ultrasonic array driven over UART.
The scripts cover single-point focusing, vortex beams, moving focal points,
beam steering, directional-speaker beam shaping, array-factor analysis, and
simple ON/OFF patterns.

## Hardware and protocol

- **Array:** 16×16 = 256 transducers, 10 mm spacing, lying in the z = 0 plane.
  The coordinate origin is the bottom-left emitter; x and y increase to the
  right and upward.
- **Carrier:** 40 kHz by default, with an assumed speed of sound of 343 m/s
  (20 °C). Scripts that compute phases accept `--frequency`/`--sound-speed`
  (or `--carrier-frequency` on `transducer_onoff_steer.py`).
- **UART:** 38400 baud, 8N1.
- **Frame:** `0xFE` + 256 payload bytes + `0xFD` (258 bytes).
  With the legacy/phase-only firmware a payload value `0..15` selects a phase
  bin and `0x10` disables a transducer.  With the duty-weighting firmware the
  low nibble is the phase bin, the high nibble is an amplitude code (`0` keeps
  the legacy full-scale pulse), and `0x10` is still OFF.  Phase-only frames
  remain valid because their high nibble is zero.
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
python directional_speaker.py --dry-run --alpha 0 --beta 0
```

Compare the host-side beam-shaping algorithms on one steering direction:

```bash
python directional_speaker.py --compare --alpha 0 --beta 0
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
| `directivity.py` | Density thinning, phase dithering, phase-only sidelobe optimization, calibration, and beam analysis. |
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
| `mode_control.py` | Sends a mode command byte (`0xC8`/`0xC9`/`0xCA`/`0xCB`) followed by a whole-array frame (all-on by default). | `--mode`, `--frame {on,off,keep}`, `--port`, `--dry-run`, `--list-ports` |

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

### Directional-speaker beam shaping

`directional_speaker.py` steers the beam and then applies host-side algorithms
to the phase-only frame. It reports the ideal pattern before and after, and
`--compare` prints the trade-off of every strategy for the current direction.

| Strategy | Host control | Measured effect (ideal model, broadside) |
|----------|--------------|------------------------------------------|
| Baseline steering | 16 phase bins | Near sidelobe ≈ −16 dB, grating ≈ 0 dB |
| Density thinning | ON/OFF (Hann/Gaussian) | Near sidelobe ≈ −17 dB, main gain ≈ −3 dB |
| Phase dithering | ±1–2 bins per element | Near sidelobe ≈ −15 dB, main gain ≈ −0.5 dB |
| Phase-only optimization | 16 bins per element | Near sidelobe ≈ −19 dB, main gain ≈ −1 dB |
| Taper + optimization | ON/OFF + 16 bins | Near sidelobe ≈ −20 dB, main gain ≈ −3 to −4 dB |
| Duty amplitude (Hann) | high nibble, 8 widths | Near sidelobe ≈ −21 dB (β=20°) / −28 dB (broadside), main gain ≈ −7.6 dB |
| Duty amplitude + optimization | high nibble + 16 bins | Near sidelobe ≈ −26 dB (β=20°) / −30 dB (broadside), main gain ≈ −8 dB |

The numbers are idealized (isotropic elements, ideal bins) and depend on the
analysis grid; reproduce them with `directional_speaker.py --compare`. They
describe the **non-grating** sidelobe floor, not the grating lobes.

**Hard limit.** With a periodic lattice and identical elements, the array
factor at a grating-lobe direction equals the main-lobe array factor for *any*
per-element phase or amplitude (including ON/OFF) pattern — the grating
condition removes every phase difference and both lobes scale with the same
weighted sum. At 10 mm pitch and 40 kHz (`d/λ = 1.166`), broadside grating
lobes sit near ±59°; no host algorithm can lower them. The same limit applies
to the ≈3° main-lobe width, which is set by the 150 mm aperture. Host
algorithms shape the non-grating sidelobes and can place user-selected nulls
(`--null AZ:BETA`), but the real levers for grating lobes are:

- the element directivity (a 10 mm piston attenuates ±59° by roughly 15 dB),
- a carrier below `c / spacing` = 34.3 kHz (no broadside grating lobe),
- spacing ≤ λ/2 (≤ 4.3 mm), or an aperiodic element layout.

Per-channel phase errors are the other practical limiter. Measure them and
pass the 256 offsets (in bins, 0..15) through `--calibration offsets.json`;
calibration is applied after shaping and is not included in the ideal-model
numbers.

Usage:

```bash
# Uniform steering, no shaping
python directional_speaker.py --dry-run --alpha 0 --beta 0

# Hann density taper plus phase-only optimization
python directional_speaker.py --dry-run --alpha 0 --beta 20 \
    --density-window hann --keep-fraction 0.7 --optimize-sweeps 4

# Null a nearby direction while steering
python directional_speaker.py --dry-run --alpha 0 --beta 0 \
    --optimize-sweeps 4 --null 90:40 --null-weight 2

# Best measured host-side setting: Gaussian 80% taper + 5 optimizer sweeps
python directional_speaker.py --dry-run --alpha 0 --beta 20 \
    --density-window gaussian --keep-fraction 0.8 \
    --optimize-sweeps 5 --seed 1

# Send the shaped frame
python directional_speaker.py --alpha 0 --beta 20 --port COM3
```

The best setting lowers the near sidelobe from −15.61 dB to −20.54 dB at
β = 20° (and from −16.11 dB to −21.03 dB broadside) for about 2.8 dB of
main-lobe gain. The grating lobes stay within 0.1 dB of the baseline, as the
lattice requires.

### Duty-cycle amplitude weighting

`--amplitude-window hann|gaussian` encodes a radial aperture taper into the
high nibble of each payload byte. `--amplitude-gaussian-sigma` sets the
Gaussian width. This requires the duty-weighting FPGA firmware; the legacy
firmware masks the high nibble, so the same frame degrades gracefully to a
phase-only pattern.

The host maps a desired relative pressure weight `a` to a pulse width
`w = round((16/pi) * asin(a))` in `1..8` steps of the 16-step carrier period,
then to the firmware code. A width-`w` pulse has fundamental amplitude
`sin(pi*w/16)`, so the smallest non-zero weight is about 0.20; zero weights
are sent as OFF. Codes 0 and 15 are both full scale, and a phase-only frame
(high nibble 0) is bit-for-bit unchanged.

```bash
# Hann amplitude taper (duty firmware)
python directional_speaker.py --dry-run --alpha 0 --beta 0 --amplitude-window hann

# Amplitude taper plus phase optimization
python directional_speaker.py --dry-run --alpha 0 --beta 20 \
    --amplitude-window gaussian --amplitude-gaussian-sigma 0.35 \
    --optimize-sweeps 4
```

The Hann taper lowers the near sidelobe from −15.61 to −21.29 dB at β = 20°
(and from −16.92 to −28.04 dB broadside) for about 7.6 dB of main-lobe gain.
The four corner emitters have zero Hann weight and are sent OFF, leaving 252
active elements. Combined with phase optimization it reaches about −26 dB
(β = 20°) / −30 dB (broadside) for about 8 dB of main-lobe gain. Amplitude
weighting does not change the grating-lobe ratio; see the hard limit above.

Thinning, dithering and the optimizer are deterministic for a given `--seed`.
The optimizer changes only ON elements; it never re-enables a thinned element.
`--dither-bins` is included for experiments but measured to raise the
non-grating floor and cost main-lobe gain, so it is not recommended on its own.

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
- **Audio modes gate the drive:** the FPGA's C4, Little-star and 200 Hz modes
  chop the 40 kHz drive, so they are audible only while the transducers are
  driven. `mode_control.py` therefore sends an all-on frame (phase 0) by
  default; `--frame off` mutes the array and `--frame keep` leaves the last
  latched phase pattern (e.g. a steering pattern) in place.
- **ON/OFF rate:** full-frame toggling is limited to about 7.4 Hz by the UART.
- **Phase resolution:** 16 bins (0..15), plus `0x10` for off.
- **Amplitude resolution:** the duty firmware gives 8 pulse widths (1..8 of
  the 16 carrier steps), so the fundamental amplitude steps are
  `sin(pi*w/16)` — about 0.20 to 1.0. It is coarse but real; per-channel DACs
  would be needed for true continuous control.
- **Quantization** snaps values within a few ULPs of an exact bin boundary so
  mathematically identical phases do not land in different bins.
- **Array-factor plots are theoretical** idealized models (isotropic
  elements, ideal bin phases). They are not measurements.
- **Directional-speaker algorithms cannot remove grating lobes.** At 10 mm
  pitch and 40 kHz the lattice fixes them near ±59° for broadside; every
  per-element phase/ON-OFF pattern leaves their level unchanged. The host code
  optimizes the non-grating sidelobes and nulls instead. See
  "Directional-speaker beam shaping" above.
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
directivity.py                 thinning, duty amplitude, dithering, phase optimization, calibration
serial_utils.py                port discovery and selection
single_point_*.py              focus, vortex, line, circle demos
transducer_*.py                all-on, ON/OFF, steering demos
directional_speaker.py         steering + beam shaping + comparison
onoff.py, bit_pattern_onoff.py ON/OFF and pattern generators
mode_control.py                mode command + phase frame
wire_probe.py                  wire-address probe for oscilloscope tests
table_test.py                  phase-table checks
table_fingerprint.py           phase-table fingerprint tool
orbit_test.py                  orbit diagnostic
tests/                         hardware-free pytest suite
```

See `CHANGELOG.md` for changes. See `docs/demos.md` for design notes on the
three demo experiences (levitation, directional speaker, sandbox/touch), the
planned two-PCB architecture, and what each demo still needs.
