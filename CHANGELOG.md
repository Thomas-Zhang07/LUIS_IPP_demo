# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project aims to follow
semantic versioning once releases are tagged.

## [Unreleased]

### Added

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
- Test tooling: `requirements.txt`, `pytest.ini`, and shared pytest fixtures
  with hardware-free serial and clock fakes.
- `docs/cleanup-progress.md` to track the repository cleanup.

### Changed

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

### Fixed

- Converted `onoff.py`, `single_point_hover.py`, and
  `single_point_vortex.py` from GBK to UTF-8. `onoff.py` previously raised
  `SyntaxError` on Python 3 and now runs; Chinese comments and messages are
  preserved (D1, D2).
