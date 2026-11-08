# Three Demo Experiences — Design Notes

This document describes the three demo experiences the project is built around,
how the current host code implements them, and what the planned two-PCB
hardware (one 16×16 bottom array, one 16×16 top array) still needs.

It is a planning document, not a description of a finished system. Each section
separates **what exists in the code today** from **what is inferred or still
required**.

## Summary

| # | Experience | Primary scripts | Host code today | Critical dependency |
|---|------------|-----------------|-----------------|---------------------|
| 1 | **Acoustic levitation** — a small bead hovers and moves | `single_point_hover.py`, `single_point_line.py`, `single_point_circle.py` | Single-array focus and moving focus implemented | Phase sign; two-array phase coherence |
| 2 | **Directional speaker** — sound audible only inside a beam | `single_point_vortex.py`, `transducer_phase_steer*.py`, `mode_control.py` | Phase shaping and steering only; no audio in the host | FPGA audio modes / carrier modulation |
| 3 | **Sandbox / touch** — visible sand response or a tactile point | `single_point_circle.py` | Moving focus implemented; no amplitude modulation | Amplitude modulation; faster field updates |

Shared numbers used throughout:

| Quantity | Value |
|----------|-------|
| Carrier | 40 kHz (default) |
| Speed of sound | 343 m/s (default) |
| Wavelength λ | 8.575 mm |
| λ/2 (standing-wave node spacing) | 4.2875 mm |
| Phase resolution (1 bin) | 22.5° = 1.5625 µs = λ/16 ≈ 0.536 mm of path |
| Frame | 258 bytes (`0xFE` + 256 + `0xFD`), 8N1 |
| Frame time at 38400 baud | ≈ 67.2 ms → ≈ 14.9 frames/s |
| ON/OFF limit | 2 frames per cycle → ≈ 7.44 Hz |
| Grid | 16×16 = 256 emitters, 10 mm pitch, z = 0 plane |

Two physical consequences follow from the grid:

- **10 mm pitch > λ/2 = 4.29 mm.** The array is under-sampled for far-field
  steering, so grating lobes are expected; the sidelobe analysis code already
  flags them. Normal-incidence focusing/standing waves are less affected.
- **The host updates the field slowly** compared with the carrier. The FPGA
  holds the last frame and generates the 40 kHz drive continuously; the host
  only moves the setpoint, at most ~15 times per second.

---

## Demo 1 — Single-point focus: acoustic levitation

### The experience

A small expanded-polystyrene bead hovers at a point above the array and can be
moved along a line or around a circle.

### The physics

For each emitter at distance `d` from the target, the host computes a phase so
that all wavefronts arrive in phase at the target:

```text
phase = -2*pi*f*d / c   (mod 2*pi)
```

That creates a pressure maximum at the focus. The acoustic radiation force
pushes a dense particle toward the pressure maximum (or toward a pressure node
in a standing wave), creating a stable trap.

A single-sided focus gives a trap, but the axial (vertical) restoring force is
weak. The planned **top array** changes this fundamentally: two opposing arrays
create a standing wave with pressure nodes every **λ/2 = 4.29 mm**, giving much
stiffer 3D trapping and allowing multiple particles at different heights.

### How the code implements it today

| Piece | Where |
|-------|-------|
| Focus phase per emitter | `acoustics.focus_phases`, `focus_phase_bins` |
| 16-level quantization | `acoustics.quantize_phase_bins` |
| Frame construction | `protocol.build_frame` |
| One-shot focus | `single_point_hover.py` |
| Moving focus along x | `single_point_line.py` (3 cm span, ping-pong) |
| Moving focus around a circle | `single_point_circle.py` (1.5 cm radius, 48 points, 2 s) |

The line and circle demos are the same trap, moved. The maximum motion update
rate is ~15 frames/s, so a 48-point circle takes at least ~3.2 s of wire time
even though the demo requests 2 s.

### What the two-PCB design requires

- **Geometry:** bottom array at z = 0 facing +z; top array at z = H facing −z.
  H should be chosen relative to λ/2 (e.g. a multiple of 4.29 mm) and is part of
  the trap design.
- **Coherence:** the two carriers must share a phase reference. Independent
  oscillators will drift, moving the standing wave and the bead. One controller
  driving both arrays, or a shared 40 kHz clock/sync line, is required.
- **Control modes:** either
  - **standing wave** — both arrays emit continuously; move the bead by shifting
    the relative phase between arrays (one 22.5° bin moves the nodes by
    λ/32 ≈ 0.27 mm), or
  - **dual focus** — both arrays focus on the same point, giving a stronger
    trap that can be moved by moving the focus.
- **Per-array mapping:** the top PCB's x/y orientation and wiring order may be
  rotated or mirrored relative to the bottom. That needs its own coordinate
  transform and transmit order.
- **Calibration:** the focus phase sign is still unverified (see the README
  caveat). The bead itself is the definitive test: if it traps, the sign is
  correct; if not, invert the phase bins and retry.

### Status

| Item | Status |
|------|--------|
| Single-array focus + moving focus | Implemented |
| Two-array geometry and protocol | Not implemented |
| Coherent dual-array drive | Hardware decision required |
| Phase-sign calibration | Unverified; documented |
| Relative-phase calibration | Not implemented |

---

## Demo 2 — Vortex: directional speaker

### The experience

Sound is audible only inside a narrow beam — a "personal audio" zone that can
be aimed at one listener without filling the room.

### The physics

A directional speaker uses a **parametric acoustic array**: a 40 kHz carrier is
amplitude-modulated with the audio signal. Air nonlinearity demodulates the
carrier along the beam, producing audible sound that stays confined to the
beam's path. The host must therefore control **amplitude**, not only phase.

A vortex adds a helical phase `charge × polar_angle` on top of the focus. The
wavefront twists and the beam has an on-axis pressure null (a phase
singularity). That is a distinctive acoustic beam, but by itself it produces no
sound; audio still requires modulation of the carrier.

### How the code implements it today

| Piece | Where |
|-------|-------|
| Helical phase | `acoustics.vortex_phases` (`focus + charge × angle`) |
| Static vortex | `single_point_vortex.py` |
| Beam steering | `transducer_phase_steer*.py`, `transducer_onoff_steer.py` |
| Mode selection | `mode_control.py` (`0xC8`–`0xCB`) |

The mode labels in `mode_control.py` — **C4**, **Little star**, **200 Hz** — look
like musical/audio modes (`C4` is a note, "Little star" is a melody). This
strongly suggests the FPGA generates the sound internally, and the host only
selects a mode and, optionally, steers the beam. That is an inference from the
labels, not verified behavior.

### What is missing

- **Audio modulation is not in the host code.** The host cannot stream audio:
  it sends full 258-byte frames at ~15 frames/s. The FPGA must generate or
  modulate the audio.
- **The vortex may not be the right beam for a speaker.** An on-axis null means
  the center of the beam is quiet. If the speaker demo uses a vortex, the
  listening point would be off-axis; if it uses the steered beam, the vortex is
  a separate demonstration.
- **Two arrays:** the bottom/top layout is a levitation geometry (opposing
  beams). A speaker demo would normally use one array, or both aimed the same
  way; the fixed facing-each-other layout would create a standing wave between
  them. Confirm which array(s) the speaker demo uses.

### Status

| Item | Status |
|------|--------|
| Vortex phase pattern | Implemented |
| Beam steering | Implemented |
| Audio generation / modulation | Inferred FPGA capability; unverified |
| Host audio streaming | Not possible on the current link |
| Combined mode + steering command | Not implemented |

---

## Demo 3 — Circle: sandbox / sense of touch

### The experience

A moving focal point creates a visible response on a sand surface, or a
localized tactile sensation when it is moved over a hand — a "touch without
contact" demo.

### The physics

- **Sand:** acoustic radiation pressure at the surface can move grains and reveal
  the focal pattern. For this, the focus must be at the sand surface (z near 0),
  not the 5 cm default used by the circle demo.
- **Touch:** ultrasonic haptics focuses on the skin and **modulates the focal
  amplitude** at roughly 100–500 Hz while moving the point. The modulation is
  what is felt. The host currently updates at ~15 frames/s, far below that, so
  the FPGA would have to provide the amplitude modulation.

### How the code implements it today

`single_point_circle.py` moves a focus around a 3 cm-diameter circle:
1.5 cm radius, 48 points, 2 s default, in the XY plane, at the center's z. The
same machinery in `trajectory.run_circle` could drive other paths.

### What is missing

- **Amplitude modulation.** The wire payload is phase-only (`0`–`15` = phase,
  `0x10` = off). There is no per-frame or per-element amplitude control.
- **Update rate.** A 48-point circle at 2 s asks for 24 updates/s; the link
  provides ~15/s, so the real revolution is longer. For haptics, modulation is
  much faster still and must live on the FPGA.
- **Path shaping.** The radius and line span are fixed in the CLIs. A raster or
  Lissajous path would demonstrate a surface/area better than a small circle.
- **Focus surface.** The circle's z is settable, but the default is 5 cm above
  the array. Sand or a hand requires a different height.

### Status

| Item | Status |
|------|--------|
| Moving focus trajectory | Implemented |
| Configurable radius/span | Not exposed in the CLI |
| Raster/Lissajous paths | Not implemented |
| Amplitude modulation (haptics) | Not implemented; likely FPGA-side |
| Update-rate warning | Not implemented |

---

## The planned two-PCB architecture

```text
        top PCB 16×16, facing down
        ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
             H  (ideally n × λ/2 = n × 4.29 mm)
        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
        bottom PCB 16×16, facing up
```

### Why two arrays help

- **Levitation:** much stiffer axial trapping; a true standing-wave trap; multiple
  nodes at different heights; particle height can be tuned by relative phase.
- **Acoustic power:** roughly double the pressure for the same drive.
- **Holography:** 512 emitters give more control over the field (multiple traps,
  shapes), at the cost of more computation and a larger frame.

### Open hardware questions

1. **One controller or two independent boards?** One FPGA driving both arrays is
   the simplest way to guarantee phase coherence.
2. **Shared clock/sync?** If there are two boards, they must share a 40 kHz
   reference; otherwise the standing wave drifts and the bead will not stay put.
3. **Separation H** — fixed or adjustable? It sets which heights are nodes.
4. **Top PCB orientation** — same x/y axes, rotated 180°, or mirrored? This
   determines the top array's coordinate transform and transmit order.
5. **Link topology** — two serial ports (one per PCB) or one link with an
   extended frame? Two ports keep ~15 updates/s per array; one 512-byte frame
   halves the update rate but keeps a single connection.
6. **First control mode** — standing wave (plane waves + relative phase) or dual
   focus (both arrays focus on the particle)?

### Frame and timing implications

| Topology | Bytes per update | Update rate (ideal) |
|----------|------------------|---------------------|
| One array (today) | 258 | ~14.9 /s |
| Two arrays, two ports | 258 each | ~14.9 /s each |
| Two arrays, one 512-channel frame | 514 | ~7.5 /s |

Phase coherence is a **clock** property, not a UART property: as long as both
arrays hold their frames, the relative phase is set by their carrier clocks.
The serial link only determines how quickly the setpoints can change.

---

## What is real vs. what is assumed

| Capability | Evidence | Confidence |
|------------|----------|------------|
| 258-byte phase frame, 16 bins, `0x10` off | Code + protocol | Verified by tests |
| Focus/steering phase math | Code + tests | Verified, except hardware sign |
| Vortex phase pattern | Code + tests | Verified as a pattern |
| ON/OFF and steering | Code + tests | Verified, except hardware behavior |
| FPGA generates 40 kHz carrier and phases | Protocol implies it | High |
| FPGA audio modes (C4, Little Star, 200 Hz) | `mode_control.py` labels | Inferred, **confirm on hardware** |
| Amplitude modulation for haptics | Not in code | Unknown; likely FPGA-side |
| Two-array coherence | Not designed yet | Open |

---

## Recommended next steps

1. **Resolve the phase sign on hardware** with the levitation test: focus a bead
   and see whether it traps; if not, invert the bins and repeat. This is the
   highest-value measurement.
2. **Confirm the FPGA audio modes** by running `mode_control.py --mode c4`,
   `little-star`, `200hz` and listening for sound. If they work, the speaker
   demo's audio path exists; the host only needs mode + steering.
3. **Decide the two-array controller/clock architecture** before writing the
   dual-array host code. Without coherence there is no stable standing wave.
4. **Extend the host for two arrays** — combined geometry, per-array transmit
   order, and either two ports or a 514-byte frame.
5. **Add field calibration** — per-array phase sign and the relative phase
   between arrays, measured at the midpoint.
6. **Improve the motion demos** — expose `--radius`/`--span`, add raster or
   Lissajous paths, and warn when the requested cadence is below the 67 ms wire
   time.
7. **Confirm the speaker beam** — decide whether the directional speaker uses
   the vortex (off-axis listening) or the steered beam, and whether it uses one
   array or both.

## Safety

Two opposing arrays create a strong standing wave. Keep hands, ears, and eyes
out of the focal region; run high-amplitude demos enclosed and never unattended.
