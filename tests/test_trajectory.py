"""Tests for trajectory generation and timed streaming."""

import itertools

import numpy as np
import pytest

import trajectory


def test_circle_points_have_expected_radius_and_center():
    center = (0.075, 0.075, 0.05)
    points = trajectory.generate_circle_points(center, radius=0.015, point_count=8)
    assert points.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2] - np.array(center[:2]), axis=1), 0.015)
    np.testing.assert_allclose(points[:, 2], 0.05)
    np.testing.assert_allclose(points[0], [0.09, 0.075, 0.05])


def test_circle_point_validation():
    with pytest.raises(ValueError):
        trajectory.generate_circle_points((0.0, 0.0, 0.0), point_count=2)
    with pytest.raises(ValueError):
        trajectory.generate_circle_points((0.0, 0.0, 0.0), point_count=5.0)
    with pytest.raises(ValueError):
        trajectory.generate_circle_points((float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        trajectory.generate_circle_points((0.0, 0.0, 0.0), radius=0.0)


def test_line_points_span_endpoints_and_hold_y_z():
    center = (0.075, 0.075, 0.05)
    points = trajectory.generate_line_points(center, span=0.03, point_count=5)
    np.testing.assert_allclose(points[:, 0], [0.06, 0.0675, 0.075, 0.0825, 0.09])
    np.testing.assert_allclose(points[:, 1], 0.075)
    np.testing.assert_allclose(points[:, 2], 0.05)


def test_ping_pong_order_matches_legacy_sequence():
    sequence = list(itertools.islice(trajectory.ping_pong_indices(4), 10))
    assert sequence == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]


def test_stream_uses_absolute_deadlines_without_drift(fake_clock, fake_serial_factory):
    connection = fake_serial_factory()
    start_times = []

    def factory(index):
        start_times.append(fake_clock.now)
        fake_clock.advance(0.02)  # simulate a slow write
        return b"frame-%d" % index

    sent = trajectory.stream_timed_frames(
        factory,
        interval_s=0.1,
        connection=connection,
        max_frames=5,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )

    assert sent == 5
    assert start_times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert len(connection.writes) == 5


def test_stream_dry_run_sends_nothing(fake_clock, fake_serial_factory):
    connection = fake_serial_factory()
    sent = trajectory.stream_timed_frames(
        lambda index: b"frame",
        interval_s=0.1,
        connection=connection,
        dry_run=True,
        max_frames=3,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )
    assert sent == 3
    assert connection.writes == []


def test_stream_requires_connection_without_dry_run(fake_clock):
    with pytest.raises(ValueError):
        trajectory.stream_timed_frames(
            lambda index: b"frame",
            interval_s=0.1,
            clock=fake_clock,
            sleep=fake_clock.sleep,
        )


def test_stream_duration_stops_on_schedule(fake_clock, fake_serial_factory):
    connection = fake_serial_factory()
    sent = trajectory.stream_timed_frames(
        lambda index: b"frame",
        interval_s=0.1,
        connection=connection,
        duration_s=0.25,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )
    assert sent == 3


def test_toggle_opens_serial_once_and_settles_once(fake_serial_factory, fake_clock):
    sent = trajectory.run_toggle(
        port="COM9",
        on_frame=b"ON",
        off_frame=b"OFF",
        frequency_hz=10.0,
        max_frames=2,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )

    assert sent == 2
    assert len(fake_serial_factory.instances) == 1
    assert fake_serial_factory.last.writes == [b"ON", b"OFF"]
    # settle (0.3 s) once + one half period (0.05 s), not a settle per frame
    assert fake_clock.now == pytest.approx(0.35)


def test_toggle_duration_stops(fake_serial_factory, fake_clock):
    sent = trajectory.run_toggle(
        port="COM9",
        on_frame=b"ON",
        off_frame=b"OFF",
        frequency_hz=10.0,
        duration_s=0.25,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert sent == 5
    assert fake_serial_factory.last.writes[-1] == b"ON"


def test_toggle_sends_off_on_exit_when_requested(fake_serial_factory, fake_clock):
    messages = []
    sent = trajectory.run_toggle(
        port="COM9",
        on_frame=b"ON",
        off_frame=b"OFF",
        frequency_hz=10.0,
        duration_s=0.25,
        send_off_on_exit=True,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=messages.append,
    )
    assert sent == 5
    assert fake_serial_factory.last.writes[-1] == b"OFF"
    assert any("Sent OFF frame" in message for message in messages)


def test_bit_pattern_sequence_and_wire_time_warning(fake_serial_factory, fake_clock):
    messages = []
    sent = trajectory.run_bit_pattern(
        port="COM9",
        pattern=[1, 0, 1],
        bit_seconds=0.01,
        max_frames=4,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=messages.append,
    )
    assert sent == 4
    payloads = [write[1] for write in fake_serial_factory.last.writes]
    assert payloads == [0x00, 0x10, 0x00, 0x00]
    assert any("Warning" in message for message in messages)


def test_bit_pattern_rejects_bad_input():
    with pytest.raises(ValueError):
        trajectory.run_bit_pattern(pattern=[], bit_seconds=0.1, dry_run=True)
    with pytest.raises(ValueError):
        trajectory.run_bit_pattern(pattern=[1, 2], bit_seconds=0.1, dry_run=True)
    with pytest.raises(ValueError):
        trajectory.run_bit_pattern(pattern=[1], bit_seconds=0.0, dry_run=True)


def _exploding_factory(*args, **kwargs):
    raise AssertionError("serial must not be opened during a dry run")


def test_run_circle_dry_run_does_not_open_serial(fake_clock):
    sent = trajectory.run_circle(
        (0.075, 0.075, 0.05),
        dry_run=True,
        serial_factory=_exploding_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert sent == 0


def test_run_line_dry_run_does_not_open_serial(fake_clock):
    sent = trajectory.run_line(
        (0.075, 0.075, 0.05),
        dry_run=True,
        serial_factory=_exploding_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert sent == 0


def test_toggle_and_bit_pattern_dry_run_do_not_open_serial():
    assert (
        trajectory.run_toggle(
            on_frame=b"ON",
            off_frame=b"OFF",
            frequency_hz=10.0,
            dry_run=True,
            serial_factory=_exploding_factory,
            output_fn=lambda *_: None,
        )
        == 0
    )
    assert (
        trajectory.run_bit_pattern(
            pattern=[1, 0],
            bit_seconds=0.1,
            dry_run=True,
            serial_factory=_exploding_factory,
            output_fn=lambda *_: None,
        )
        == 0
    )


def test_rotation_angles_modes():
    assert trajectory.get_rotation_angles("two") == (0.0, 180.0)
    assert trajectory.get_rotation_angles("four") == (0.0, 90.0, 180.0, 270.0)
    with pytest.raises(ValueError):
        trajectory.get_rotation_angles("three")


def test_alpha_at_elapsed_time_quantizes_to_sectors():
    assert trajectory.alpha_at_elapsed_time(0.0, 10.0, "four") == 0.0
    assert trajectory.alpha_at_elapsed_time(2.5, 10.0, "four") == 90.0
    assert trajectory.alpha_at_elapsed_time(5.0, 10.0, "four") == 180.0
    assert trajectory.alpha_at_elapsed_time(7.5, 10.0, "four") == 270.0
    assert trajectory.alpha_at_elapsed_time(9.99, 10.0, "four") == 270.0
    assert trajectory.alpha_at_elapsed_time(10.0, 10.0, "four") == 0.0
    with pytest.raises(ValueError):
        trajectory.alpha_at_elapsed_time(-1.0, 10.0)
    with pytest.raises(ValueError):
        trajectory.alpha_at_elapsed_time(0.0, 0.0)


def test_calculate_rotation_timing():
    timing = trajectory.calculate_rotation_timing(10.0, 0.1, mode="four")
    assert timing["wire_time_s"] == pytest.approx(258 * 10 / 38400)
    assert timing["effective_update_interval_s"] == pytest.approx(2.5)
    assert timing["effective_rotation_period_s"] == pytest.approx(10.0)
    assert timing["effective_angular_speed_deg_s"] == pytest.approx(36.0)

    slow = trajectory.calculate_rotation_timing(10.0, 5.0, mode="four")
    assert slow["effective_update_interval_s"] == pytest.approx(5.0)
    assert slow["effective_rotation_period_s"] == pytest.approx(20.0)
    assert slow["effective_angular_speed_deg_s"] == pytest.approx(18.0)

    with pytest.raises(ValueError):
        trajectory.calculate_rotation_timing(0.0, 0.1)


def test_run_rotation_dry_run_opens_nothing(fake_clock):
    summary = trajectory.run_rotation(
        30.0,
        mode="two",
        dry_run=True,
        max_updates=2,
        serial_factory=_exploding_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert summary["frames_sent"] == 2
    assert summary["effective_update_interval_s"] == pytest.approx(5.0)


def test_run_rotation_opens_serial_once(fake_serial_factory, fake_clock):
    summary = trajectory.run_rotation(
        30.0,
        mode="two",
        port="COM9",
        max_updates=2,
        serial_factory=fake_serial_factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert summary["frames_sent"] == 2
    assert len(fake_serial_factory.instances) == 1
    writes = fake_serial_factory.last.writes
    assert len(writes) == 2
    assert all(len(frame) == 258 for frame in writes)


def test_run_rotation_slows_when_writes_exceed_cadence(fake_clock):
    class SlowConnection:
        def __init__(self):
            self.writes = []
            self.dtr = None
            self.rts = None

        def write(self, data):
            fake_clock.advance(4.0)
            self.writes.append(bytes(data))
            return len(data)

        def flush(self):
            pass

        def close(self):
            pass

    connections = []

    def factory(*args, **kwargs):
        connection = SlowConnection()
        connections.append(connection)
        return connection

    summary = trajectory.run_rotation(
        30.0,
        mode="four",
        port="COM9",
        max_updates=2,
        serial_factory=factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        output_fn=lambda *_: None,
    )
    assert summary["frames_sent"] == 2
    assert len(connections) == 1
    assert summary["effective_rotation_period_s"] > 10.0


def test_run_rotation_validates_beta_and_port():
    with pytest.raises(ValueError):
        trajectory.run_rotation(0.0, dry_run=True)
    with pytest.raises(ValueError):
        trajectory.run_rotation(30.0, dry_run=False, port=None)
