# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project aims to follow
semantic versioning once releases are tagged.

## [Unreleased]

### Fixed

- **Array pitch corrected to 10.5 mm.**  The board pick-place/Gerber data give
  16 x 16 elements at 10.5000 mm pitch (157.5 mm span); the previous 10 mm
  value was the transducer body diameter.  `emitters.SPACING` and all geometry,
  steering and grating predictions now use the measured pitch.  The broadside
  grating lobe moves from 59.04 deg to **54.75 deg**, and the spatial-aliasing
  carrier from 34.3 kHz to 32.7 kHz.
- **Two-board geometry from the measured PCB.**  `BoardGeometry` now takes the
  element-centre `seam_m` directly (default 36.75 mm) and a 5.25 mm y offset,
  matching the acrylic fixture: 351.75 x 162.75 mm combined aperture, 9.16 mm
  edge gap.  The old `gap_m`/`offset_x_m` parameters and `--gap`/`--offset-x`
  flags are replaced by `--seam`/`--offset-y`.

### Added

- `dual_board.py`: 512-element two-board geometry (5 mm x/y half-pitch
  offset, configurable gap), per-board frame routing with the primary
  `0xC0` / secondary `0xC1` select bytes, a parallel two-port `BoardLink`,
  full-aperture Hann/Gaussian/Blackman/Tukey tapers, and a
  `CombinedBeamAnalyzer` that keeps the whole hemisphere in the sidelobe
  objective and reports the split grating lobes beside the 59° null.
- `dual_directional_speaker.py`: steering or spherical-wave focusing for
  both boards, optional phase optimization, predicted pattern report, and a
  `--dry-run` that builds both frames without serial ports.
- `tests/test_dual_board.py`: geometry, frame routing, offset nulling, the
  split-lobe physics, taper codes, parallel link writes, and CLI dry-run.
- Side-leak analysis for the two-board model: optional circular-piston
  directivity (10 mm default), a `far_sidelobe_db` metric for the region above
  `--side-start`, and a far-region optimizer with a `--near-weight` guard.  The
  CLI reports both the above-60 peak and the overall peak so the trade against
  main-beam gain is visible.
- Shared `protocol.py`: validated frame builders, uniform ON/OFF frames,
  wire-time calculations, a consistent serial context manager, and
  partial-write-checked sends.
- `serial_utils.resolve_serial_port()`: dry-run-aware port resolution that
  never prompts when no port is needed.
- Shared `acoustics.py`: focus, vortex, steering and quantization math with
  configurable carrier frequency and sound speed.
- Shared `cli_utils.py`: argparse groups, target prompting, and phase-map /
  frame display used by the demo CLIs.
- Shared `trajectory.py`: circle/line point generators, ping-pong indexing,
  absolute-deadline frame streaming, and single-connection ON/OFF and
  bit-pattern runners.
- Shared `array_factor.py`: physical phase grid, theoretical array factor,
  sidelobe evaluation, and plotting.
- Test tooling: `requirements.txt`, `pytest.ini`, and shared pytest fixtures
  with hardware-free serial and clock fakes.
- Hardware-free pytest suite (138 tests) covering the frame protocol,
  transmit order, acoustics and quantization, streaming deadlines, dry-run
  safety, mode control, README consistency, and CLI flag preservation.
- `docs/demos.md`: design notes for the three demo experiences, the planned
  two-PCB architecture, and the hardware-dependent open questions.
- `directivity.py`: host-side beam-shaping algorithms for the phase-only
  frame — windowed density thinning (flat/Hann/Gaussian), deterministic phase
  dithering, phase-only sidelobe optimization by coordinate descent with
  optional null constraints, per-channel calibration offsets, grating-lobe
  prediction, and idealized pattern metrics that separate near sidelobes from
  the invariant grating lobes.
- `directional_speaker.py`: steered directional-speaker CLI that combines
  those algorithms, reports baseline-versus-shaped metrics, and compares every
  strategy with `--compare`. It has `--dry-run`, `--calibration`, `--null`,
  `--plot` and the shared port/baud flags.
- Tests for the directivity module (mask determinism, dither bounds,
  calibration round-trip, grating-lobe invariance, optimizer improvement and
  mask handling) and end-to-end CLI coverage for the new script.

### Changed

- Frequency parsing and prompting now live in `cli_utils.py` and are shared
  by the three toggle scripts instead of being copy-pasted.
- Fixed the `--mode two|four` cell that broke the README table layout, and
  added a test that guards table column consistency.
- Documented the pre-existing focus/steering phase-sign conflict and the
  array-centered vortex behavior in `acoustics.py` and the README, and
  corrected the transmit-order direction in the `protocol.build_frame`
  docstring. These are hardware-dependent behaviors preserved from the
  original scripts, not silently changed.
- Port discovery and selection now live in one implementation in
  `serial_utils.py`, with injectable input/output functions.
- `transducer_all_on.py` now uses the shared protocol and serial modules;
  its flags and frame bytes are unchanged.
- `single_point_line.py` and `single_point_circle.py` now use the shared
  trajectory runner. The line trajectory follows absolute deadlines instead
  of accumulating `sleep()` drift (D12), and both accept
  `--frequency`/`--sound-speed`. The circle `--help` now correctly states the
  2.0 s default (D7).

### Fixed

- Rotation now rejects `beta=0` and `beta=180`; azimuth is undefined at both
  poles, and `beta=180` previously emitted identical frames while claiming to
  rotate.
- Invalid `--frequency` values and invalid rotation timing now produce a
  standard argparse error and exit status 2 instead of a traceback.
- Timed runs now clamp waits to the remaining `--duration`, so a short run
  no longer sleeps for a full interval (or several seconds) before stopping.
- Rotation scheduling now anchors each next-frame deadline to the rotation
  epoch instead of the previous frame start, so scheduling jitter does not
  accumulate over long runs.
- Headless runs that need interactive input now exit with a clear message
  and status 2 instead of raising an unhandled `EOFError`.
- ON/OFF scripts now accept rates from 0.1 Hz (previously the 20 Hz floor
  rejected every rate the UART can actually achieve); they still warn above
  the ~7.44 Hz full-frame limit.
- Phase quantization now wraps a boundary value snapped at 2π to bin 0
  instead of clipping it to bin 15, which had introduced a nearly 2π phase
  error on affected transducers.
- Wavelength is now derived consistently from carrier frequency and sound
  speed (defaults 40 kHz / 343 m/s) instead of 0.00865 m in the focus scripts
  and 343/40000 m in the steering scripts. Focus and vortex frame bytes shift
  by about 0.87% (D6).
- `--dry-run` in `single_point_hover.py` and `single_point_vortex.py` no
  longer prompts for or requires a serial port (D4).
- The focus scripts now apply `--baud` to the serial connection (previously
  it only affected port selection), and use the shared frame validator
  instead of an `assert` (D11).
- `onoff.py` and `bit_pattern_onoff.py` now open the serial port once per run
  instead of once per half-cycle/bit, add `--dry-run`, and warn when the
  requested rate exceeds the ~7.44 Hz / ~67 ms wire-time limit (D5).
- `transducer_onoff_demo.py` now uses the shared toggle runner and adds
  `--dry-run`; its once-per-run port behavior is unchanged.
- `transducer_onoff_steer.py` now uses the shared steering and toggle code.
  It adds `--carrier-frequency`/`--sound-speed` (leaving `--frequency` as the
  ON/OFF toggle rate) and keeps sending an OFF frame before exit.
- Removed 168 lines of unreachable beta-optimization code
  (`rank_beta_candidates`, `optimize_beta`, `plot_beta_optimization`) from
  `transducer_phase_steer_2d.py` (D9).
- `transducer_phase_steer.py` now uses the shared acoustics, protocol and
  serial modules, and adds `--frequency`/`--sound-speed`.
- `transducer_phase_steer_2d.py` is reduced from 1,242 to 331 lines: steering
  math, frame building, rotation timing and plotting now come from the shared
  modules. All existing flags, including the `--angle`/`--legacy-angle` alias
  and both plot flags, are preserved; `--frequency`/`--sound-speed` are added.
- `mode_control.py` is now an import-safe CLI with `--port`, `--mode`,
  `--dry-run` and `--list-ports` instead of executing on import with a
  hardcoded `COM9` (D13). The mode command byte and all-off frame are
  unchanged and documented.
- Converted `onoff.py`, `single_point_hover.py`, and
  `single_point_vortex.py` from GBK to UTF-8. `onoff.py` previously raised
  `SyntaxError` on Python 3 and now runs; Chinese comments and messages are
  preserved (D1, D2).
- Rewrote `README.md` from actual behavior: repaired the corrupted
  characters, corrected the circle to a 1.5 cm radius / 3 cm diameter and a
  2 s default, corrected the Ctrl+C all-off claim, and documented every
  script, flag, protocol constraint and physics default (D3, D8, D15).
- `mode_control.py` now sends an all-on frame after the mode byte by default.
  The FPGA's audio modes gate the 40 kHz drive, so the previous all-off frame
  left every channel silent; `--frame off` restores the old payload and
  `--frame keep` sends only the mode byte, preserving an existing steering
  pattern.
