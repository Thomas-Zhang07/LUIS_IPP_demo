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

# Amplitude-capable frames use the high nibble of each payload byte.
# Code 0 keeps the legacy full-scale pulse; codes 1..15 select a narrower,
# centred pulse.  See the "Amplitude weighting" section of the README.
AMPLITUDE_CODES = 16
MAX_AMPLITUDE_CODE = AMPLITUDE_CODES - 1


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


def amplitude_payload_byte(phase_bin: int, amplitude_code: int) -> int:
    """Return the wire byte for one phase bin and amplitude code.

    The payload byte is ``amplitude_code << 4 | phase_bin``.  ``0x10`` is
    reserved for OFF, so the single colliding combination (amplitude code 1,
    phase 0) is bumped to code 2; both select the same one-step pulse width
    in the firmware.  A code of :data:`OFF_VALUE` mutes the channel.
    """
    if isinstance(phase_bin, bool) or not isinstance(phase_bin, (int, np.integer)):
        raise ValueError("phase_bin must be an integer.")
    if isinstance(amplitude_code, bool) or not isinstance(
        amplitude_code, (int, np.integer)
    ):
        raise ValueError("amplitude_code must be an integer.")
    phase_bin = int(phase_bin)
    amplitude_code = int(amplitude_code)
    if not 0 <= phase_bin <= PHASE_BINS - 1:
        raise ValueError(f"phase_bin must be between 0 and {PHASE_BINS - 1}.")
    if not 0 <= amplitude_code <= MAX_AMPLITUDE_CODE:
        raise ValueError(
            f"amplitude_code must be between 0 and {MAX_AMPLITUDE_CODE}."
        )
    if amplitude_code == 1 and phase_bin == 0:
        amplitude_code = 2
    return (amplitude_code << 4) | phase_bin


def build_amplitude_frame(
    phase_values,
    amplitude_codes,
    *,
    transmit_order=None,
    allow_off: bool = True,
) -> bytes:
    """Build a frame carrying both phase bins and per-channel amplitude.

    Args:
        phase_values: 256 integer phase bins (and OFF if ``allow_off``).
        amplitude_codes: 256 integer codes ``0..15`` (0 = legacy full
            scale) or :data:`OFF_VALUE` to mute the channel.
        transmit_order: permutation mapping wire position -> logical index.
        allow_off: when False, OFF phase bins are rejected.

    The firmware emits a centred pulse whose width is selected by the code,
    so the fundamental amplitude follows ``sin(pi * width / 16)``.  Use
    :func:`directivity.amplitude_codes_from_weights` to turn physical
    weights into codes.  The OFF byte mutes the channel regardless of its
    amplitude code.
    """
    phases = np.asarray(phase_values)
    codes = np.asarray(amplitude_codes)
    if phases.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} phase values.")
    if codes.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} amplitude codes.")
    if not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("Phase bins must be integers.")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("Amplitude codes must be integers.")

    upper = OFF_VALUE if allow_off else PHASE_BINS - 1
    if np.any((phases < 0) | (phases > upper)):
        raise ValueError(f"Phase bins must be between 0 and {upper}.")
    if np.any((codes < 0) | (codes > OFF_VALUE)):
        raise ValueError(
            f"Amplitude codes must be between 0 and {MAX_AMPLITUDE_CODE} "
            f"(or {OFF_VALUE} for OFF)."
        )

    order = validate_transmit_order(transmit_order)
    payload = bytearray()
    for phase, code in zip(phases[order], codes[order]):
        phase = int(phase)
        code = int(code)
        if phase == OFF_VALUE or code == OFF_VALUE:
            payload.append(OFF_VALUE)
        else:
            payload.append(amplitude_payload_byte(phase, code))
    return bytes([HEADER]) + bytes(payload) + bytes([FOOTER])


TRIM_PREFIX = 0xE0
TRIM_MAGIC = 0x5A


def build_trim_command(steps: int) -> bytes:
    """Return the secondary fine-trim command ``0xE0 0x5A <steps>``.

    The firmware only recognises this sequence between frames (after
    ``0xFD`` and before the next ``0xFE``).  Each step delays the secondary
    carrier by 20 ns, i.e. 0.288 degrees at 40 kHz; 0..255 covers about
    +/-36 degrees around the default value of 127.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("trim steps must be an integer.")
    steps = int(steps)
    if not 0 <= steps <= 255:
        raise ValueError("trim steps must be between 0 and 255.")
    return bytes([TRIM_PREFIX, TRIM_MAGIC, steps])


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
