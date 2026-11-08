"""Timed frame streaming and trajectory generation.

All streaming uses absolute deadlines anchored to the first frame, so a slow
send does not accumulate drift. Serial is opened once per run.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time

import numpy as np
import serial

import acoustics
import protocol
from cli_utils import print_phase_top_view

DEFAULT_CIRCLE_RADIUS = 0.015
DEFAULT_LINE_SPAN = 0.03
DEFAULT_POINT_COUNT = 48
DEFAULT_PERIOD_S = 2.0
ROTATION_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)
TWO_DIRECTION_ANGLES_DEG = (0.0, 180.0)


def _validate_center(center) -> np.ndarray:
    position = np.asarray(center, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("center must contain three finite coordinates")
    return position


def _validate_point_count(point_count: int) -> int:
    if isinstance(point_count, bool) or not isinstance(
        point_count, (int, np.integer)
    ):
        raise ValueError("point_count must be an integer")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")
    return int(point_count)


def _validate_positive(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def generate_circle_points(
    center,
    *,
    radius: float = DEFAULT_CIRCLE_RADIUS,
    point_count: int = DEFAULT_POINT_COUNT,
) -> np.ndarray:
    """Return equally spaced target positions along a horizontal circle."""
    center = _validate_center(center)
    radius = _validate_positive(radius, "radius")
    point_count = _validate_point_count(point_count)

    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.repeat(center[np.newaxis, :], point_count, axis=0)
    points[:, 0] += radius * np.cos(angles)
    points[:, 1] += radius * np.sin(angles)
    return points


def generate_line_points(
    center,
    *,
    span: float = DEFAULT_LINE_SPAN,
    point_count: int = DEFAULT_POINT_COUNT,
) -> np.ndarray:
    """Return equally spaced target positions along the X axis."""
    center = _validate_center(center)
    span = _validate_positive(span, "span")
    point_count = _validate_point_count(point_count)

    x_min = center[0] - span / 2.0
    x_max = center[0] + span / 2.0
    points = np.repeat(center[np.newaxis, :], point_count, axis=0)
    points[:, 0] = np.linspace(x_min, x_max, point_count)
    return points


def ping_pong_indices(point_count: int):
    """Yield the legacy ping-pong index sequence: 0..N-1..0..N-1."""
    point_count = _validate_point_count(point_count)
    index = 0
    direction = 1
    while True:
        yield index
        index += direction
        if index >= point_count - 1:
            index = point_count - 1
            direction = -1
        elif index <= 0:
            index = 0
            direction = 1


def stream_timed_frames(
    frame_factory,
    *,
    interval_s: float,
    connection=None,
    dry_run: bool = False,
    max_frames: int | None = None,
    duration_s: float | None = None,
    stop_event=None,
    on_stop=None,
    clock=None,
    sleep=None,
    output_fn=None,
) -> int:
    """Send frames on an absolute schedule and return the number sent.

    Frame ``n`` is due at ``start + n * interval_s``. The first frame is sent
    immediately. ``frame_factory(n)`` supplies frame *n*. In dry-run mode no
    data is written, but frames are still built and counted.
    """
    interval_s = _validate_positive(interval_s, "interval")
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(
            max_frames, (int, np.integer)
        ) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        max_frames = int(max_frames)
    if duration_s is not None:
        duration_s = _validate_positive(duration_s, "duration")
    if not dry_run and connection is None:
        raise ValueError("A serial connection is required unless dry_run is enabled")

    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    stop_event = threading.Event() if stop_event is None else stop_event

    started = clock()
    frames_sent = 0

    try:
        while True:
            if stop_event.is_set():
                break
            if max_frames is not None and frames_sent >= max_frames:
                break
            if duration_s is not None and clock() - started >= duration_s:
                break

            if frames_sent:
                deadline = started + frames_sent * interval_s
                wait_s = deadline - clock()
                if wait_s > 0:
                    sleep(wait_s)
                if stop_event.is_set():
                    break
                if duration_s is not None and clock() - started >= duration_s:
                    break

            frame = frame_factory(frames_sent)
            if not dry_run:
                protocol.send_frame(connection, frame)
            frames_sent += 1
    except KeyboardInterrupt:
        if output_fn is not None:
            output_fn("Stopped.")
    finally:
        if on_stop is not None and not dry_run:
            on_stop(connection)

    return frames_sent


def _focus_frames(
    points: np.ndarray,
    *,
    frequency_hz: float,
    sound_speed: float,
) -> list[bytes]:
    return [
        protocol.build_frame(
            acoustics.focus_phase_bins(
                point,
                frequency_hz=frequency_hz,
                sound_speed=sound_speed,
            )
        )
        for point in points
    ]


def run_circle(
    center,
    *,
    port: str | None = None,
    baudrate: int = protocol.BAUD_RATE,
    period: float = DEFAULT_PERIOD_S,
    point_count: int = DEFAULT_POINT_COUNT,
    radius: float = DEFAULT_CIRCLE_RADIUS,
    frequency_hz: float = acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed: float = acoustics.DEFAULT_SOUND_SPEED_MPS,
    dry_run: bool = False,
    serial_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> int:
    """Run the circular focus trajectory, or describe it without hardware."""
    period = _validate_positive(period, "period")
    _validate_positive(baudrate, "baudrate")
    acoustics.validate_acoustic_params(frequency_hz, sound_speed)

    points = generate_circle_points(
        center, radius=radius, point_count=point_count
    )
    center = np.asarray(center, dtype=float)

    output_fn(f"Circle center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) m")
    output_fn(f"Diameter: {2 * radius:.3f} m (XY plane)")
    output_fn(f"Direction: counterclockwise; period: {period:.2f} s")
    output_fn(f"Samples per circle: {point_count}")

    frames = _focus_frames(points, frequency_hz=frequency_hz, sound_speed=sound_speed)
    interval = period / point_count

    if dry_run:
        first = points[0]
        output_fn(
            f"First target: ({first[0]:.4f}, {first[1]:.4f}, {first[2]:.4f}) m"
        )
        print_phase_top_view(
            acoustics.focus_phase_bins(
                first, frequency_hz=frequency_hz, sound_speed=sound_speed
            ),
            output=output_fn,
        )
        output_fn(f"\nUART frame size: {len(frames[0])} bytes")
        output_fn("\n[DRY RUN] No frames sent.")
        return 0

    if port is None:
        raise ValueError("port is required unless dry_run is enabled")

    try:
        with protocol.open_serial(
            port, baudrate, serial_factory=serial_factory, sleep=sleep
        ) as connection:
            output_fn(f"Sending circle trajectory to {port}. Press Ctrl+C to stop.")
            return stream_timed_frames(
                lambda index: frames[index % point_count],
                interval_s=interval,
                connection=connection,
                clock=clock,
                sleep=sleep,
                output_fn=output_fn,
            )
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error


def run_line(
    center,
    *,
    port: str | None = None,
    baudrate: int = protocol.BAUD_RATE,
    period: float = DEFAULT_PERIOD_S,
    point_count: int = DEFAULT_POINT_COUNT,
    span: float = DEFAULT_LINE_SPAN,
    frequency_hz: float = acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed: float = acoustics.DEFAULT_SOUND_SPEED_MPS,
    dry_run: bool = False,
    serial_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> int:
    """Run the X-axis ping-pong focus trajectory, or describe it."""
    period = _validate_positive(period, "period")
    _validate_positive(baudrate, "baudrate")
    acoustics.validate_acoustic_params(frequency_hz, sound_speed)

    points = generate_line_points(center, span=span, point_count=point_count)
    center = np.asarray(center, dtype=float)

    output_fn(
        f"Line center (Y, Z fixed): ({center[0]:.4f}, {center[1]:.4f}, "
        f"{center[2]:.4f}) m"
    )
    output_fn(
        f"X-Span: {span * 100:.1f} cm (from X={points[0, 0]:.4f} "
        f"to X={points[-1, 0]:.4f})"
    )
    output_fn(f"Direction: ping-pong along X; period: {period:.2f} s")
    output_fn(f"Samples per sweep (one-way): {point_count}")

    frames = _focus_frames(points, frequency_hz=frequency_hz, sound_speed=sound_speed)
    interval = period / point_count

    if dry_run:
        first = points[0]
        output_fn(
            f"First target (X_min): ({first[0]:.4f}, {first[1]:.4f}, "
            f"{first[2]:.4f}) m"
        )
        print_phase_top_view(
            acoustics.focus_phase_bins(
                first, frequency_hz=frequency_hz, sound_speed=sound_speed
            ),
            output=output_fn,
        )
        output_fn(f"\nUART frame size: {len(frames[0])} bytes")
        output_fn("\n[DRY RUN] No frames sent.")
        return 0

    if port is None:
        raise ValueError("port is required unless dry_run is enabled")

    indices = ping_pong_indices(point_count)

    try:
        with protocol.open_serial(
            port, baudrate, serial_factory=serial_factory, sleep=sleep
        ) as connection:
            output_fn(f"Sending X-axis trajectory to {port}. Press Ctrl+C to stop.")
            return stream_timed_frames(
                lambda _index: frames[next(indices)],
                interval_s=interval,
                connection=connection,
                clock=clock,
                sleep=sleep,
                output_fn=output_fn,
            )
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error


def _warn_toggle_limit(frequency_hz: float, output_fn) -> None:
    limit = protocol.maximum_toggle_frequency_hz()
    if frequency_hz > limit:
        output_fn(
            f"Warning: {frequency_hz:.2f} Hz exceeds the UART full-frame limit "
            f"of about {limit:.2f} Hz; frames cannot be sent that fast."
        )


def run_toggle(
    *,
    port: str | None = None,
    on_frame: bytes,
    off_frame: bytes,
    frequency_hz: float,
    duration_s: float | None = None,
    dry_run: bool = False,
    send_off_on_exit: bool = False,
    timestamp: bool = False,
    max_frames: int | None = None,
    stop_event=None,
    serial_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> int:
    """Toggle between two frames on one serial connection.

    Returns the number of half-cycles (frames) sent. The port is opened once;
    the settle delay happens once, not per half-cycle.
    """
    frequency_hz = _validate_positive(frequency_hz, "frequency")
    if duration_s is not None:
        duration_s = _validate_positive(duration_s, "duration")
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(
            max_frames, (int, np.integer)
        ) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        max_frames = int(max_frames)

    half_period = 0.5 / frequency_hz
    _warn_toggle_limit(frequency_hz, output_fn)

    if dry_run:
        output_fn("[DRY RUN] No serial data sent.")
        return 0
    if port is None:
        raise ValueError("port is required unless dry_run is enabled")

    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    stop_event = threading.Event() if stop_event is None else stop_event

    frames_sent = 0
    state = "OFF"

    try:
        with protocol.open_serial(
            port, serial_factory=serial_factory, sleep=sleep
        ) as connection:
            started = clock()
            try:
                while True:
                    if stop_event.is_set():
                        break
                    if max_frames is not None and frames_sent >= max_frames:
                        break
                    if duration_s is not None and clock() - started >= duration_s:
                        break

                    if frames_sent:
                        deadline = started + frames_sent * half_period
                        wait_s = deadline - clock()
                        if wait_s > 0:
                            sleep(wait_s)
                        if stop_event.is_set():
                            break
                        if (
                            duration_s is not None
                            and clock() - started >= duration_s
                        ):
                            break

                    is_on = frames_sent % 2 == 0
                    protocol.send_frame(connection, on_frame if is_on else off_frame)
                    state = "ON" if is_on else "OFF"
                    cycle = frames_sent // 2 + 1
                    moment = f" ({time.strftime('%H:%M:%S')})" if timestamp else ""
                    output_fn(f"Cycle {cycle}: {state}{moment}")
                    frames_sent += 1
            except KeyboardInterrupt:
                output_fn("\nStopped by user.")
            finally:
                if send_off_on_exit and state != "OFF":
                    protocol.send_frame(connection, off_frame)
                    output_fn("Sent OFF frame before closing the serial port.")
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error

    return frames_sent


def run_bit_pattern(
    *,
    port: str | None = None,
    pattern,
    bit_seconds: float,
    duration_s: float | None = None,
    dry_run: bool = False,
    max_frames: int | None = None,
    stop_event=None,
    serial_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> int:
    """Repeat an 8-bit ON/OFF pattern on one serial connection."""
    bits = list(pattern)
    if not bits or any(
        isinstance(bit, bool)
        or not isinstance(bit, (int, np.integer))
        or bit not in (0, 1)
        for bit in bits
    ):
        raise ValueError("pattern must contain at least one 0/1 bit")
    bit_seconds = _validate_positive(bit_seconds, "bit_seconds")
    if duration_s is not None:
        duration_s = _validate_positive(duration_s, "duration")
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(
            max_frames, (int, np.integer)
        ) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        max_frames = int(max_frames)

    if bit_seconds < protocol.wire_time_s():
        output_fn(
            f"Warning: {bit_seconds:.3f} s per bit is shorter than the "
            f"~{protocol.wire_time_s():.3f} s needed to send one 258-byte frame."
        )

    frames = {
        0: protocol.build_off_frame(),
        1: protocol.build_on_frame(),
    }

    if dry_run:
        output_fn("[DRY RUN] No serial data sent.")
        return 0
    if port is None:
        raise ValueError("port is required unless dry_run is enabled")

    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    stop_event = threading.Event() if stop_event is None else stop_event

    frames_sent = 0

    try:
        with protocol.open_serial(
            port, serial_factory=serial_factory, sleep=sleep
        ) as connection:
            started = clock()
            try:
                while True:
                    if stop_event.is_set():
                        break
                    if max_frames is not None and frames_sent >= max_frames:
                        break
                    if duration_s is not None and clock() - started >= duration_s:
                        break

                    if frames_sent:
                        deadline = started + frames_sent * bit_seconds
                        wait_s = deadline - clock()
                        if wait_s > 0:
                            sleep(wait_s)
                        if stop_event.is_set():
                            break
                        if (
                            duration_s is not None
                            and clock() - started >= duration_s
                        ):
                            break

                    bit = bits[frames_sent % len(bits)]
                    protocol.send_frame(connection, frames[bit])
                    output_fn(f"Step {frames_sent + 1}: bit={bit} ({'ON' if bit else 'OFF'})")
                    frames_sent += 1
            except KeyboardInterrupt:
                output_fn("\nStopped by user.")
    except serial.SerialException as error:
        raise RuntimeError(
            f"Failed to open port {port} ({error}). Please check connection."
        ) from error

    return frames_sent


def get_rotation_angles(mode: str = "four") -> tuple[float, ...]:
    """Return the fixed azimuth sequence for the selected rotation mode."""
    if mode == "two":
        return TWO_DIRECTION_ANGLES_DEG
    if mode == "four":
        return ROTATION_ANGLES_DEG
    raise ValueError("Rotation mode must be 'two' or 'four'.")


def alpha_at_elapsed_time(
    elapsed_time_s: float,
    rotation_period_s: float,
    mode: str = "four",
) -> float:
    """Return the active fixed azimuth based on elapsed time and mode."""
    if not math.isfinite(elapsed_time_s) or elapsed_time_s < 0.0:
        raise ValueError("Elapsed time must be finite and nonnegative.")
    if not math.isfinite(rotation_period_s) or rotation_period_s <= 0.0:
        raise ValueError("Rotation period must be finite and greater than 0 seconds.")
    rotation_angles = get_rotation_angles(mode)
    sector_duration_s = rotation_period_s / len(rotation_angles)
    sector_index = int((elapsed_time_s % rotation_period_s) / sector_duration_s)
    return rotation_angles[min(sector_index, len(rotation_angles) - 1)]


def calculate_rotation_timing(
    rotation_period_s: float = 10.0,
    update_interval_s: float = 0.10,
    measured_send_time_s: float = 0.0,
    baudrate: int = protocol.BAUD_RATE,
    frame_bytes: int = protocol.FRAME_SIZE,
    safety_margin_s: float = 0.005,
    mode: str = "four",
) -> dict[str, float]:
    """Estimate safe per-angle dwell time and slow the cycle if needed."""
    direction_count = len(get_rotation_angles(mode))
    values = (
        rotation_period_s,
        update_interval_s,
        measured_send_time_s,
        safety_margin_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Timing values must be finite.")
    if rotation_period_s <= 0.0 or update_interval_s <= 0.0:
        raise ValueError("Rotation period and update interval must be greater than 0.")
    if measured_send_time_s < 0.0 or safety_margin_s < 0.0:
        raise ValueError("Measured send time and safety margin cannot be negative.")
    if baudrate <= 0 or frame_bytes <= 0:
        raise ValueError("Baud rate and frame size must be positive.")

    wire_time_s = protocol.wire_time_s(frame_bytes, baudrate)
    effective_update_interval_s = max(
        update_interval_s,
        rotation_period_s / direction_count,
        wire_time_s + safety_margin_s,
        measured_send_time_s + safety_margin_s,
    )
    effective_period_s = max(
        rotation_period_s,
        effective_update_interval_s * direction_count,
    )
    return {
        "wire_time_s": wire_time_s,
        "effective_update_interval_s": effective_update_interval_s,
        "effective_rotation_period_s": effective_period_s,
        "effective_angular_speed_deg_s": 360.0 / effective_period_s,
    }


def run_rotation(
    beta_deg: float,
    *,
    rotation_period_s: float = 10.0,
    update_interval_s: float = 0.10,
    mode: str = "four",
    port: str | None = None,
    dry_run: bool = False,
    stop_event=None,
    duration_s: float | None = None,
    max_updates: int | None = None,
    frequency_hz: float = acoustics.DEFAULT_FREQUENCY_HZ,
    sound_speed: float = acoustics.DEFAULT_SOUND_SPEED_MPS,
    serial_factory=None,
    clock=None,
    sleep=None,
    output_fn=print,
) -> dict[str, float | int]:
    """Cycle through fixed alpha states at fixed beta until stopped.

    Serial is opened once and reused. Slow writes increase the safe dwell and
    the full-cycle period.
    """
    acoustics.validate_steering_angles(0.0, beta_deg)
    rotation_angles = get_rotation_angles(mode)
    if beta_deg == 0.0:
        raise ValueError("Beta must be greater than 0 degrees for azimuth rotation.")
    if not math.isfinite(rotation_period_s) or rotation_period_s <= 0.0:
        raise ValueError("Rotation period must be finite and greater than 0 seconds.")
    if not math.isfinite(update_interval_s) or update_interval_s <= 0.0:
        raise ValueError("Update interval must be finite and greater than 0 seconds.")
    if duration_s is not None and (
        not math.isfinite(duration_s) or duration_s <= 0.0
    ):
        raise ValueError("Duration must be finite and greater than 0 seconds.")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be at least 1.")
    if not dry_run and not port:
        raise ValueError("A serial port is required unless dry_run=True.")

    acoustics.validate_acoustic_params(frequency_hz, sound_speed)

    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    stop_event = threading.Event() if stop_event is None else stop_event
    timing = calculate_rotation_timing(
        rotation_period_s, update_interval_s, mode=mode
    )
    output_fn(
        f"Rotation preview: beta={beta_deg:.2f} deg, "
        f"requested period={rotation_period_s:.3f} s, "
        f"angle dwell={timing['effective_update_interval_s']:.3f} s, "
        f"effective period={timing['effective_rotation_period_s']:.3f} s, "
        f"speed={timing['effective_angular_speed_deg_s']:.2f} deg/s, "
        f"estimated wire time={timing['wire_time_s'] * 1000:.2f} ms."
    )
    sequence = " -> ".join(
        f"{angle:g}" for angle in rotation_angles + (rotation_angles[0],)
    )
    output_fn(f"Angle sequence ({mode}): {sequence} degrees.")
    output_fn(
        "Press Ctrl+C to stop."
        if not dry_run
        else "Dry-run mode: no serial port will be opened."
    )

    frames_sent = 0
    effective_period_s = timing["effective_rotation_period_s"]
    effective_interval_s = timing["effective_update_interval_s"]
    rotation_epoch = None
    next_deadline = None

    try:
        if dry_run:
            connection_scope = contextlib.nullcontext(None)
        else:
            connection_scope = protocol.open_serial(
                port, serial_factory=serial_factory, sleep=sleep
            )
        with connection_scope as connection:
            while not stop_event.is_set():
                if frames_sent >= 1:
                    if max_updates is not None and frames_sent >= max_updates:
                        break
                    now = clock()
                    if duration_s is not None and rotation_epoch is not None:
                        if now - rotation_epoch >= duration_s:
                            break
                    wait_s = max(0.0, next_deadline - now)
                    if wait_s:
                        sleep(wait_s)
                    if stop_event.is_set():
                        break

                frame_started = clock()
                if frames_sent == 0:
                    alpha_degrees = rotation_angles[0]
                else:
                    elapsed_s = max(0.0, frame_started - rotation_epoch)
                    if duration_s is not None and elapsed_s >= duration_s:
                        break
                    alpha_degrees = alpha_at_elapsed_time(
                        elapsed_s, effective_period_s, mode
                    )

                phase_bins = acoustics.steering_phase_bins(
                    alpha_degrees,
                    beta_deg,
                    frequency_hz=frequency_hz,
                    sound_speed=sound_speed,
                )
                frame = protocol.build_frame(phase_bins)
                if dry_run:
                    payload = frame[1:-1]
                    output_fn(
                        f"[DRY RUN] frame={frames_sent + 1}, "
                        f"alpha={alpha_degrees:.3f} deg, "
                        f"beta={beta_deg:.3f} deg, bytes={len(frame)}, "
                        f"header=0x{frame[0]:02X}, "
                        f"first_phase_bytes={payload[:8].hex(' ')}, "
                        f"... last_phase_byte=0x{payload[-1]:02X}, "
                        f"footer=0x{frame[-1]:02X}"
                    )
                else:
                    protocol.send_frame(connection, frame)
                frame_finished = clock()
                send_duration_s = (
                    0.0 if dry_run else max(0.0, frame_finished - frame_started)
                )
                frames_sent += 1

                updated_timing = calculate_rotation_timing(
                    rotation_period_s,
                    update_interval_s,
                    measured_send_time_s=send_duration_s,
                    mode=mode,
                )
                if (
                    updated_timing["effective_rotation_period_s"]
                    > effective_period_s
                ):
                    if rotation_epoch is not None:
                        cycle_fraction = (
                            max(0.0, frame_finished - rotation_epoch)
                            % effective_period_s
                        ) / effective_period_s
                        rotation_epoch = frame_finished - (
                            cycle_fraction
                            * updated_timing["effective_rotation_period_s"]
                        )
                    effective_period_s = updated_timing[
                        "effective_rotation_period_s"
                    ]
                    output_fn(
                        "Frame transmission exceeded the estimated cadence; "
                        f"slowing rotation to "
                        f"{360.0 / effective_period_s:.2f} deg/s."
                    )
                effective_interval_s = updated_timing[
                    "effective_update_interval_s"
                ]
                if frames_sent == 1:
                    rotation_epoch = frame_started
                    output_fn(
                        f"Effective timing: interval={effective_interval_s:.3f} s, "
                        f"period={effective_period_s:.3f} s, "
                        f"speed={360.0 / effective_period_s:.2f} deg/s."
                    )
                if rotation_epoch is None:
                    rotation_epoch = frame_started
                elapsed_from_epoch = max(0.0, frame_finished - rotation_epoch)
                steps = (
                    int(math.floor(elapsed_from_epoch / effective_interval_s)) + 1
                )
                next_deadline = rotation_epoch + steps * effective_interval_s
    except KeyboardInterrupt:
        output_fn("Rotation stopped by user.")

    summary = {
        "frames_sent": frames_sent,
        "effective_update_interval_s": effective_interval_s,
        "effective_rotation_period_s": effective_period_s,
        "effective_angular_speed_deg_s": 360.0 / effective_period_s,
    }
    output_fn(
        f"Rotation ended after {frames_sent} frames; effective speed="
        f"{summary['effective_angular_speed_deg_s']:.2f} deg/s."
    )
    return summary
