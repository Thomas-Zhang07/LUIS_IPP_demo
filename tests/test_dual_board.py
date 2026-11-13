"""Hardware-free tests for the two-board geometry, frames and analyzer."""

import math

import numpy as np
import pytest

import dual_board
from emitters import EMITTERS_ORDER


def test_geometry_has_512_positions_and_half_pitch_offset():
    geometry = dual_board.BoardGeometry(gap_m=0.0)
    positions = geometry.positions()
    assert positions.shape == (512, 3)

    board_a = geometry.board_a_positions()
    board_b = geometry.board_b_positions()
    assert np.allclose(board_a[0], (0.0, 0.0, 0.0))
    assert np.allclose(board_a[255], (0.150, 0.150, 0.0))
    # With no gap, the seam is exactly the 5 mm half-pitch offset.
    assert np.allclose(board_b[0], (0.155, 0.005, 0.0))
    assert board_b[1, 0] - board_b[0, 0] == pytest.approx(0.010)
    assert board_b[16, 1] - board_b[0, 1] == pytest.approx(0.010)


def test_gap_only_changes_the_seam():
    tight = dual_board.BoardGeometry(gap_m=0.0).positions()
    loose = dual_board.BoardGeometry(gap_m=0.020).positions()
    assert np.allclose(tight[:256], loose[:256])
    assert np.allclose(loose[256:, 0] - tight[256:, 0], 0.020)
    assert np.allclose(loose[256:, 1], tight[256:, 1])


def test_geometry_rejects_bad_values():
    with pytest.raises(ValueError):
        dual_board.BoardGeometry(gap_m=-0.001)
    with pytest.raises(ValueError):
        dual_board.BoardGeometry(offset_x_m=float("nan"))


def test_phase_bins_fit_and_payloads_carry_select_byte():
    geometry = dual_board.BoardGeometry()
    bins_a, bins_b = dual_board.board_phase_bins(
        geometry, azimuth_degrees=0.0, off_axis_degrees=0.0
    )
    assert bins_a.shape == bins_b.shape == (256,)
    payload_a, payload_b = dual_board.build_payloads(geometry, bins_a, bins_b)

    assert len(payload_a) == len(payload_b) == 259
    assert payload_a[0] == dual_board.PRIMARY_SELECT_BYTE == 0xC0
    assert payload_b[0] == dual_board.SECONDARY_SELECT_BYTE == 0xC1
    assert payload_a[1] == payload_b[1] == 0xFE
    assert payload_a[-1] == payload_b[-1] == 0xFD


def test_payload_phase_order_matches_emitters_order():
    geometry = dual_board.BoardGeometry()
    bins_a = np.arange(256, dtype=int) % 16
    bins_b = (np.arange(256, dtype=int) + 4) % 16
    payload_a, payload_b = dual_board.build_payloads(geometry, bins_a, bins_b)
    expected_a = bins_a[EMITTERS_ORDER]
    expected_b = bins_b[EMITTERS_ORDER]
    # Uniform full-scale codes are 0xF0; the low nibble is the phase.
    assert [byte & 0x0F for byte in payload_a[2:-1]] == list(expected_a)
    assert [byte & 0x0F for byte in payload_b[2:-1]] == list(expected_b)


def test_half_pitch_offset_nulls_the_exact_grating_direction():
    shifted = dual_board.BoardGeometry(gap_m=0.0)
    analyzer = dual_board.CombinedBeamAnalyzer(
        shifted.positions(), samples=121, element_diameter_m=None
    )
    phases = np.zeros(512, dtype=int)
    level = analyzer.level_at(phases, 0.0, dual_board.GRATING_BETA_DEG)
    assert level < -20.0

    unshifted = dual_board.BoardGeometry(
        gap_m=0.0, offset_x_m=0.0, offset_y_m=0.0
    )
    reference = dual_board.CombinedBeamAnalyzer(
        unshifted.positions(), samples=121, element_diameter_m=None
    )
    reference_level = reference.level_at(phases, 0.0, dual_board.GRATING_BETA_DEG)
    assert reference_level > -6.0


def test_offset_leaves_split_lobes_beside_the_null():
    """Document the physics: the null is narrow, the lobe energy splits."""
    geometry = dual_board.BoardGeometry(gap_m=0.0)
    analyzer = dual_board.CombinedBeamAnalyzer(
        geometry.positions(), samples=121, element_diameter_m=None
    )
    phases = np.zeros(512, dtype=int)
    # The exact grating direction is nulled, but a few degrees away the two
    # board halves no longer cancel.
    level = analyzer.level_at(phases, 0.0, 61.0)
    assert level > -10.0
    metrics = analyzer.metrics(phases)
    assert metrics["peak_sidelobe_db"] > -10.0
    assert metrics["grating_lobe_db"][0.0] < -20.0


def test_amplitude_window_codes_are_valid_and_taper_edges():
    geometry = dual_board.BoardGeometry()
    weights = dual_board.amplitude_window_weights(geometry, "hann")
    codes = dual_board.amplitude_codes_from_weights(weights)
    assert codes.shape == (512,)
    assert np.all((codes >= 0) & (codes <= 15) | (codes == 0x10))
    positions = geometry.positions()
    # Corners of the combined envelope are silent, the middle is full scale.
    xs, ys = positions[:, 0], positions[:, 1]
    corner = (xs == xs.max()) & (ys == ys.max())
    assert all(code == 0x10 for code in codes[corner])
    center_index = int(np.argmin(np.abs(xs - xs.mean()) + np.abs(ys - ys.mean())))
    assert codes[center_index] != 0x10


def test_window_values():
    u = np.array([0.0, 0.5, 1.0])
    assert np.allclose(dual_board._window_1d(
        u, "hann", sigma=0.35, tukey_alpha=0.5), [0.0, 1.0, 0.0])
    blackman = dual_board._window_1d(
        u, "blackman", sigma=0.35, tukey_alpha=0.5)
    assert np.allclose(blackman, [0.0, 1.0, 0.0], atol=1e-12)
    tukey = dual_board._window_1d(
        u, "tukey", sigma=0.35, tukey_alpha=0.5)
    assert np.allclose(tukey, [0.0, 1.0, 0.0])


class _FakeConnection:
    def __init__(self, port):
        self.port = port
        self.writes = []
        self.flushed = 0
        self.closed = False

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


class _FakeFactory:
    def __init__(self):
        self.connections = []

    def __call__(self, port, *args, **kwargs):
        connection = _FakeConnection(port)
        self.connections.append(connection)
        return connection


def test_board_link_writes_both_ports():
    factory = _FakeFactory()
    payload_a = b"\xc0" + bytes([0xFE]) + bytes(256) + bytes([0xFD])
    payload_b = b"\xc1" + bytes([0xFE]) + bytes(256) + bytes([0xFD])
    link = dual_board.BoardLink(
        "COM6", "COM7", serial_factory=factory, sleep=lambda _: None
    )
    with link:
        link.send(payload_a, payload_b)
    assert [connection.port for connection in factory.connections] == ["COM6", "COM7"]
    for connection, payload in zip(factory.connections, (payload_a, payload_b)):
        assert connection.writes == [payload]
        assert connection.flushed == 1
        assert connection.closed


def test_board_link_rejects_same_port():
    with pytest.raises(ValueError):
        dual_board.BoardLink("COM6", "COM6")


def test_grating_beta_is_the_lattice_alias():
    expected = math.degrees(
        math.asin((343.0 / 40000.0) / 0.010)
    )
    assert dual_board.GRATING_BETA_DEG == pytest.approx(expected)


def test_cli_dry_run_builds_both_frames(capsys):
    import dual_directional_speaker as cli

    assert cli.main(["--dry-run", "--samples", "31"]) == 0
    output = capsys.readouterr().out
    assert "board A payload: 259 bytes" in output
    assert "board B payload: 259 bytes" in output
    assert "select 0xC0" in output
    assert "select 0xC1" in output
    assert "[DRY RUN] no serial data sent." in output


def test_cli_requires_both_ports_without_dry_run(capsys):
    import dual_directional_speaker as cli

    assert cli.main(["--samples", "21"]) == 2
    error = capsys.readouterr().err
    assert "board-a-port" in error and "board-b-port" in error


def test_trim_command_bytes():
    import protocol

    assert protocol.build_trim_command(0) == b"\xe0\x5a\x00"
    assert protocol.build_trim_command(127) == b"\xe0\x5a\x7f"
    assert protocol.build_trim_command(255) == b"\xe0\x5a\xff"
    for bad in (-1, 256, True, 1.5):
        with pytest.raises(ValueError):
            protocol.build_trim_command(bad)


def test_board_link_sends_trim_only_to_secondary():
    factory = _FakeFactory()
    link = dual_board.BoardLink(
        "COM6", "COM7", serial_factory=factory, sleep=lambda _: None
    )
    with link:
        link.send_trim(200)
    connection_a, connection_b = factory.connections
    assert connection_a.writes == []
    assert connection_b.writes == [b"\xe0\x5a\xc8"]


def test_parse_trim_sweep():
    import dual_directional_speaker as cli

    assert cli.parse_trim_sweep("0:255:128") == [0, 128]
    assert cli.parse_trim_sweep("120:130:5") == [120, 125, 130]
    for bad in ("1:2", "0:10:0", "0:300:10", "a:b:c"):
        with pytest.raises(ValueError):
            cli.parse_trim_sweep(bad)


def test_cli_dry_run_reports_trim(capsys):
    import dual_directional_speaker as cli

    assert cli.main(["--dry-run", "--samples", "21", "--trim", "127"]) == 0
    output = capsys.readouterr().out
    assert "trim command: e0 5a 7f" in output


def test_piston_directivity_lowers_the_far_lobes():
    geometry = dual_board.BoardGeometry(gap_m=0.0)
    isotropic = dual_board.CombinedBeamAnalyzer(
        geometry.positions(), samples=121, element_diameter_m=None
    )
    piston = dual_board.CombinedBeamAnalyzer(
        geometry.positions(), samples=121, element_diameter_m=0.010
    )
    phases = np.zeros(512, dtype=int)
    iso = isotropic.metrics(phases)
    real = piston.metrics(phases)
    assert iso["far_sidelobe_db"] > -6.0
    assert real["far_sidelobe_db"] < iso["far_sidelobe_db"] - 10.0
    assert piston.level_at(phases, 0.0, dual_board.GRATING_BETA_DEG) < -20.0


def test_far_region_optimizer_trades_near_sidelobes():
    geometry = dual_board.BoardGeometry(gap_m=0.0)
    analyzer = dual_board.CombinedBeamAnalyzer(geometry.positions(), samples=61)
    phases = np.zeros(512, dtype=int)
    before = analyzer.metrics(phases)
    optimized, info = analyzer.optimize(
        phases, max_sweeps=1, seed=1, side_start_deg=60.0, near_weight=0.5
    )
    after = analyzer.metrics(optimized)
    assert info["changed_elements"] > 0
    assert after["far_sidelobe_db"] < before["far_sidelobe_db"] - 0.5
    # The near guard keeps the main beam from collapsing.
    assert after["main_lobe_gain_db"] > -10.0
