"""Shared UART protocol for the 16x16 ultrasonic transducer array.

Wire format (258 bytes):

    0xFE + 256 payload bytes in hardware transmit order + 0xFD

Payload values 0..15 select a phase bin; 0x10 disables a transducer.
This module owns wire semantics only; it knows nothing about acoustics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import serial

from emitters import EMITTERS_ORDER, N_EMITTERS

BAUD_RATE = 38400
HEADER = 0xFE
FOOTER = 0xFD
PHASE_BINS = 16
ON_VALUE = 0x00
OFF_VALUE = 0x10
FRAME_SIZE = N_EMITTERS + 2
BITS_PER_BYTE = 10  # 8N1: start + 8 data + stop


def validate_transmit_order(transmit_order=None) -> np.ndarray:
    """Return the hardware transmit order, validating it is a permutation."""
    order = np.asarray(
        EMITTERS_ORDER if transmit_order is None else transmit_order
    )
    if order.shape != (N_EMITTERS,) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError(
            f"Transmit order must contain {N_EMITTERS} integer indices."
        )
    if not np.array_equal(np.sort(order), np.arange(N_EMITTERS)):
        raise ValueError(
            "Transmit order must be a permutation of all emitter indices."
        )
    return order.astype(int, copy=False)


def build_frame(
    phase_values,
    *,
    transmit_order=None,
    allow_off: bool = True,
) -> bytes:
    """Build one protocol frame from logical-order transducer values.

    Args:
        phase_values: 256 integer values in logical emitter order.
        transmit_order: permutation mapping wire position -> logical index
            (EMITTERS_ORDER[k] is the logical emitter sent at position k).
        allow_off: when False, only phase bins 0..15 are accepted.

    Returns:
        A 258-byte frame ready to write to the serial port.
    """
    values = np.asarray(phase_values)
    if values.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} transducer values.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Transducer values must be integers.")

    upper = OFF_VALUE if allow_off else PHASE_BINS - 1
    if np.any((values < 0) | (values > upper)):
        raise ValueError(f"Transducer values must be between 0 and {upper}.")

    order = validate_transmit_order(transmit_order)
    payload = values[order].astype(np.uint8).tobytes()
    return bytes([HEADER]) + payload + bytes([FOOTER])


def build_uniform_frame(value: int) -> bytes:
    """Build a frame that applies the same value to every transducer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("value must be an integer in the range 0..255")
    if not 0 <= int(value) <= 0xFF:
        raise ValueError("value must be an integer in the range 0..255")
    payload = bytes([int(value)]) * N_EMITTERS
    return bytes([HEADER]) + payload + bytes([FOOTER])


def build_on_frame() -> bytes:
    """Return the all-on frame (phase 0 for every transducer)."""
    return build_uniform_frame(ON_VALUE)


def build_off_frame() -> bytes:
    """Return the all-off frame (0x10 for every transducer)."""
    return build_uniform_frame(OFF_VALUE)


def wire_time_s(
    frame_bytes: int = FRAME_SIZE,
    baudrate: int = BAUD_RATE,
    bits_per_byte: int = BITS_PER_BYTE,
) -> float:
    """Ideal time to transmit one frame, excluding software overhead."""
    if frame_bytes <= 0 or baudrate <= 0 or bits_per_byte <= 0:
        raise ValueError("Frame size, baud rate and bits per byte must be positive.")
    return frame_bytes * bits_per_byte / baudrate


def maximum_toggle_frequency_hz(
    baudrate: int = BAUD_RATE,
    frame_bytes: int = FRAME_SIZE,
) -> float:
    """Upper bound for a full-frame ON/OFF toggle on this link."""
    return 1.0 / (2.0 * wire_time_s(frame_bytes, baudrate))


@contextmanager
def open_serial(
    port,
    baudrate: int = BAUD_RATE,
    *,
    timeout: float = 2.0,
    write_timeout: float = 2.0,
    settle_s: float = 0.3,
    serial_factory=None,
    sleep=None,
):
    """Open the array's serial port with the project's standard settings.

    The factory and sleep function are resolved at call time so tests can
    substitute fakes. Settling (DTR/RTS low, then a short delay) happens once
    per connection.
    """
    if not port:
        raise ValueError("A serial port is required.")

    factory = serial.Serial if serial_factory is None else serial_factory
    sleeper = time.sleep if sleep is None else sleep

    connection = factory(
        port,
        baudrate,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=timeout,
        write_timeout=write_timeout,
        rtscts=False,
        dsrdtr=False,
    )
    try:
        connection.dtr = False
        connection.rts = False
        if settle_s > 0:
            sleeper(settle_s)
        yield connection
    finally:
        connection.close()


def send_bytes(connection, data: bytes) -> int:
    """Write and flush one buffer, raising on an incomplete write."""
    data = bytes(data)
    written = connection.write(data)
    connection.flush()
    if written != len(data):
        raise serial.SerialException(
            f"Incomplete write: sent {written} of {len(data)} bytes."
        )
    return len(data)


def send_frame(connection, frame: bytes) -> int:
    """Alias for :func:`send_bytes` used with protocol frames."""
    return send_bytes(connection, frame)


def write_frame_to_port(
    port,
    frame: bytes,
    *,
    baudrate: int = BAUD_RATE,
    settle_s: float = 0.3,
    serial_factory=None,
    sleep=None,
) -> None:
    """Open the port, send exactly one frame, and close it."""
    with open_serial(
        port,
        baudrate,
        settle_s=settle_s,
        serial_factory=serial_factory,
        sleep=sleep,
    ) as connection:
        send_frame(connection, frame)
