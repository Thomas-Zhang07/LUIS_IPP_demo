"""Tests for the shared UART protocol layer."""

import numpy as np
import pytest
import serial

import protocol
from emitters import EMITTERS_ORDER, N_EMITTERS


def _values():
    return (np.arange(N_EMITTERS, dtype=int) % protocol.PHASE_BINS)


def test_frame_layout_and_size():
    frame = protocol.build_frame(_values())
    assert len(frame) == protocol.FRAME_SIZE == 258
    assert frame[0] == protocol.HEADER
    assert frame[-1] == protocol.FOOTER


def test_payload_is_reordered_into_transmit_order():
    values = _values()
    frame = protocol.build_frame(values)
    expected = values[np.asarray(EMITTERS_ORDER)].astype(np.uint8).tobytes()
    assert frame[1:-1] == expected


def test_build_frame_accepts_plain_python_list():
    frame = protocol.build_frame([0] * N_EMITTERS)
    assert len(frame) == 258
    assert frame[1:-1] == bytes(N_EMITTERS)


def test_build_frame_rejects_wrong_length():
    with pytest.raises(ValueError):
        protocol.build_frame(np.zeros(N_EMITTERS - 1, dtype=int))


def test_build_frame_rejects_non_integer_values():
    with pytest.raises(ValueError):
        protocol.build_frame(np.zeros(N_EMITTERS, dtype=float))


def test_build_frame_rejects_out_of_range_values():
    values = np.zeros(N_EMITTERS, dtype=int)
    values[3] = protocol.OFF_VALUE + 1
    with pytest.raises(ValueError):
        protocol.build_frame(values)


def test_allow_off_flag_controls_0x10():
    values = np.zeros(N_EMITTERS, dtype=int)
    values[0] = protocol.OFF_VALUE
    # Allowed by default (ON/OFF capable), rejected for phase-only frames.
    assert len(protocol.build_frame(values)) == 258
    with pytest.raises(ValueError):
        protocol.build_frame(values, allow_off=False)


def test_invalid_transmit_order_raises():
    bad_order = list(EMITTERS_ORDER)
    bad_order[1] = bad_order[0]  # duplicate => not a permutation
    with pytest.raises(ValueError):
        protocol.build_frame(np.zeros(N_EMITTERS, dtype=int), transmit_order=bad_order)


def test_on_off_uniform_frames():
    assert protocol.build_on_frame() == (
        bytes([protocol.HEADER]) + bytes([protocol.ON_VALUE]) * N_EMITTERS
        + bytes([protocol.FOOTER])
    )
    assert protocol.build_off_frame() == (
        bytes([protocol.HEADER]) + bytes([protocol.OFF_VALUE]) * N_EMITTERS
        + bytes([protocol.FOOTER])
    )


def test_build_uniform_frame_validates_value():
    with pytest.raises(ValueError):
        protocol.build_uniform_frame(-1)
    with pytest.raises(ValueError):
        protocol.build_uniform_frame(256)
    with pytest.raises(ValueError):
        protocol.build_uniform_frame(True)


def test_send_frame_raises_on_partial_write():
    class ShortWriter:
        def write(self, data):
            return len(data) - 1

        def flush(self):
            pass

    with pytest.raises(serial.SerialException):
        protocol.send_bytes(ShortWriter(), b"abcdef")


def test_open_serial_uses_consistent_settings(fake_serial_factory, fake_clock):
    with protocol.open_serial(
        "COM9", serial_factory=fake_serial_factory, sleep=fake_clock.sleep
    ) as connection:
        assert connection is fake_serial_factory.last

    assert len(fake_serial_factory.instances) == 1
    connection = fake_serial_factory.last
    assert connection.settings == {
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 2.0,
        "write_timeout": 2.0,
        "rtscts": False,
        "dsrdtr": False,
    }
    assert connection.dtr is False
    assert connection.rts is False
    assert connection.closed is True
    assert fake_clock.now == pytest.approx(0.3)


def test_open_serial_closes_on_error(fake_serial_factory, fake_clock):
    with pytest.raises(RuntimeError):
        with protocol.open_serial(
            "COM9", serial_factory=fake_serial_factory, sleep=fake_clock.sleep
        ):
            raise RuntimeError("boom")
    assert fake_serial_factory.last.closed is True


def test_open_serial_requires_port(fake_serial_factory):
    with pytest.raises(ValueError):
        with protocol.open_serial("", serial_factory=fake_serial_factory):
            pass


def test_write_frame_to_port_sends_once(fake_serial_factory, fake_clock):
    protocol.write_frame_to_port(
        "COM9",
        b"abc",
        serial_factory=fake_serial_factory,
        sleep=fake_clock.sleep,
    )
    assert len(fake_serial_factory.instances) == 1
    assert fake_serial_factory.last.writes == [b"abc"]
    assert fake_serial_factory.last.closed is True


def test_amplitude_payload_byte_layout_and_off_collision():
    assert protocol.amplitude_payload_byte(5, 0) == 0x05
    assert protocol.amplitude_payload_byte(5, 8) == 0x85
    assert protocol.amplitude_payload_byte(0, 15) == 0xF0
    # 0x10 is the OFF code, so code 1 / phase 0 is bumped to code 2.
    assert protocol.amplitude_payload_byte(0, 1) == 0x20
    with pytest.raises(ValueError):
        protocol.amplitude_payload_byte(16, 0)
    with pytest.raises(ValueError):
        protocol.amplitude_payload_byte(0, 16)


def test_build_amplitude_frame_layout_and_order():
    phases = _values()
    codes = (np.arange(N_EMITTERS, dtype=int) % 15) + 1
    frame = protocol.build_amplitude_frame(phases, codes)
    assert len(frame) == protocol.FRAME_SIZE
    assert frame[0] == protocol.HEADER and frame[-1] == protocol.FOOTER

    order = np.asarray(EMITTERS_ORDER)
    expected = bytearray()
    for phase, code in zip(phases[order], codes[order]):
        expected.append(protocol.amplitude_payload_byte(int(phase), int(code)))
    assert frame[1:-1] == bytes(expected)


def test_build_amplitude_frame_off_and_legacy_code():
    phases = np.zeros(N_EMITTERS, dtype=int)
    codes = np.zeros(N_EMITTERS, dtype=int)
    assert protocol.build_amplitude_frame(phases, codes)[1:-1] == bytes(N_EMITTERS)

    phases[0] = protocol.OFF_VALUE
    codes[0] = 15
    frame = protocol.build_amplitude_frame(phases, codes)
    assert frame[1 + EMITTERS_ORDER.index(0)] == protocol.OFF_VALUE

    codes[1] = protocol.OFF_VALUE  # zero weight mutes even with a phase bin
    frame = protocol.build_amplitude_frame(phases, codes)
    assert frame[1 + EMITTERS_ORDER.index(1)] == protocol.OFF_VALUE


def test_build_amplitude_frame_validates_inputs():
    phases = np.zeros(N_EMITTERS, dtype=int)
    codes = np.zeros(N_EMITTERS, dtype=int)
    with pytest.raises(ValueError):
        protocol.build_amplitude_frame(phases[:-1], codes)
    with pytest.raises(ValueError):
        protocol.build_amplitude_frame(phases, codes[:-1])
    with pytest.raises(ValueError):
        protocol.build_amplitude_frame(phases.astype(float), codes)
    bad_codes = codes.copy()
    bad_codes[0] = 17
    with pytest.raises(ValueError):
        protocol.build_amplitude_frame(phases, bad_codes)
    off = phases.copy()
    off[0] = protocol.OFF_VALUE
    with pytest.raises(ValueError):
        protocol.build_amplitude_frame(off, codes, allow_off=False)


def test_wire_time_and_max_toggle_frequency():
    expected_wire = 258 * 10 / 38400
    assert protocol.wire_time_s() == pytest.approx(expected_wire)
    assert protocol.maximum_toggle_frequency_hz() == pytest.approx(
        1.0 / (2.0 * expected_wire)
    )
    assert protocol.maximum_toggle_frequency_hz() == pytest.approx(7.4418, rel=1e-3)
