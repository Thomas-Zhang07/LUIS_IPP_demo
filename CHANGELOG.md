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
- Test tooling: `requirements.txt`, `pytest.ini`, and shared pytest fixtures
  with hardware-free serial and clock fakes.
- `docs/cleanup-progress.md` to track the repository cleanup.

### Changed

- Port discovery and selection now live in one implementation in
  `serial_utils.py`, with injectable input/output functions.

### Fixed

- Converted `onoff.py`, `single_point_hover.py`, and
  `single_point_vortex.py` from GBK to UTF-8. `onoff.py` previously raised
  `SyntaxError` on Python 3 and now runs; Chinese comments and messages are
  preserved (D1, D2).
