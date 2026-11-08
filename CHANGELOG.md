# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project aims to follow
semantic versioning once releases are tagged.

## [Unreleased]

### Added

- Test tooling: `requirements.txt`, `pytest.ini`, and shared pytest fixtures
  with hardware-free serial and clock fakes.
- `docs/cleanup-progress.md` to track the repository cleanup.

### Changed

### Fixed

- Converted `onoff.py`, `single_point_hover.py`, and
  `single_point_vortex.py` from GBK to UTF-8. `onoff.py` previously raised
  `SyntaxError` on Python 3 and now runs; Chinese comments and messages are
  preserved (D1, D2).
