#!/usr/bin/env python3
"""Toggle the transducer_onoff output with a single-angle phase gradient.

The ON frame carries 16-level carrier phase values (0..15); the OFF frame
uses 0x10 for every emitter, matching transducer_onoff_demo.py. The 2D array
is treated as a 1D array along x, centered on the y-parallel axis x=0.075 m.
One angle in the x-z cross-section sets the phase gradient: 0 degrees points
along +x, 90 along +z (the array normal), and 180 along -x. The ON/OFF
frequency controls the envelope toggle.

At 38400 baud, sending one 258-byte frame takes at least 67 ms (8N1), so a
host-driven ON/OFF cycle cannot reach the 20 Hz lower end of the audible range.
The requested toggle rate is retained for compatibility, but the limitation
is reported before sending.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

if __package__ in (None, ""):
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

from emitters import EMITTERS_ORDER, EMITTERS_POS, N_EMITTERS
from serial_utils import select_serial_port

BAUDRATE = 38400
HEADER = 0xFE
FOOTER = 0xFD
OFF_VALUE = 0x10
PHASE_BINS = 16
SPEED_OF_SOUND = 343.0
DEFAULT_CARRIER_FREQUENCY = 40000.0


def parse_angle_input(raw_value: str) -> float:
    """Parse and validate the beam angle in degrees."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("Angle must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError("Angle must be finite.")
    if not 0.0 <= value <= 180.0:
        raise ValueError("Angle must be between 0 and 180 degrees.")
    return value


def prompt_for_angle() -> float:
    """Prompt for the single beam angle."""
    while True:
        raw_angle = input(
            "\nEnter beam angle (0 to 180 degrees; 0=+x, 90=+z, 180=-x): "
        ).strip()
        try:
            return parse_angle_input(raw_angle)
        except ValueError as exc:
            print(f"{exc} Please try again.", flush=True)


def compute_steering_phase_bins(
    angle_degrees: float,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY,
    sound_speed: float = SPEED_OF_SOUND,
    emitters_pos: np.ndarray = EMITTERS_POS,
) -> np.ndarray:
    """Return phases for an x-z beam around the x=0.075 m center axis."""
    values = (angle_degrees, carrier_frequency_hz, sound_speed)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Angle, carrier frequency, and sound speed must be finite.")
    if not 0.0 <= angle_degrees <= 180.0:
        raise ValueError("Angle must be between 0 and 180 degrees.")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("Carrier frequency must be greater than 0 Hz.")
    if sound_speed <= 0.0:
        raise ValueError("Sound speed must be greater than 0 m/s.")

    positions = np.asarray(emitters_pos, dtype=float)
    if positions.shape != (N_EMITTERS, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(f"Expected finite emitter positions with shape ({N_EMITTERS}, 3).")

    direction_x = math.cos(math.radians(angle_degrees))
    if math.isclose(direction_x, 0.0, abs_tol=1e-12):
        direction_x = 0.0

    # Treat the array as a line along x; y does not contribute to the phase.
    projected_distance = (positions[:, 0] - 0.075) * direction_x
    phases = np.mod(
        -2.0 * np.pi * carrier_frequency_hz * projected_distance / sound_speed,
        2.0 * np.pi,
    )
    phase_bins = np.floor(phases * PHASE_BINS / (2.0 * np.pi)).astype(int)
    return np.clip(phase_bins, 0, PHASE_BINS - 1).astype(np.uint8)


def build_frame(values: np.ndarray | list[int]) -> bytes:
    """Build a complete 0xFE + 256-byte payload + 0xFD UART frame."""
    payload_values = np.asarray(values, dtype=int)
    if payload_values.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} transducer values.")
    if np.any((payload_values < 0) | (payload_values > OFF_VALUE)):
        raise ValueError(f"Transducer values must be between 0 and {OFF_VALUE}.")

    transmit_order = np.asarray(EMITTERS_ORDER, dtype=int)
    if transmit_order.shape != (N_EMITTERS,) or not np.array_equal(
        np.sort(transmit_order), np.arange(N_EMITTERS)
    ):
        raise ValueError("EMITTERS_ORDER must be a permutation of all emitter indices.")

    payload = payload_values[transmit_order].astype(np.uint8).tobytes()
    return bytes([HEADER]) + payload + bytes([FOOTER])


def print_phase_top_view(phase_values: np.ndarray) -> None:
    """Print phase bins in physical 16x16 layout (y increases upward)."""
    print("\n16x16 phase map (0..15; top view):")
    print("      " + " ".join(f"x={col * 10:02d}" for col in range(16)))
    print("      " + "---" * 16)
    for row in range(15, -1, -1):
        row_values = phase_values[row * 16 : (row + 1) * 16]
        print(f"y={row * 10:02d} | " + " ".join(f"{int(value):02d}" for value in row_values))


def parse_frequency(raw: str) -> float:
    try:
        frequency = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid frequency: {raw!r}") from exc
    if not math.isfinite(frequency) or not 20.0 <= frequency <= 20000.0:
        raise ValueError("Frequency must be between 20 Hz and 20000 Hz.")
    return frequency


def prompt_for_frequency() -> float:
    while True:
        try:
            return parse_frequency(input("Enter target frequency in Hz (20 to 20000): ").strip())
        except ValueError as exc:
            print(f"{exc} Please try again.")


def prompt_for_duration() -> float | None:
    while True:
        raw = input("Enter run duration in seconds (leave empty for continuous): ").strip()
        if not raw:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            print("Invalid duration. Please enter a number or leave it empty.")
            continue
        if not math.isfinite(seconds) or seconds <= 0.0:
            print("Duration must be a finite number greater than 0 seconds.")
            continue
        return seconds


def build_off_frame() -> bytes:
    """Build the transducer_onoff OFF frame (0x10 for every emitter)."""
    return build_frame([OFF_VALUE] * N_EMITTERS)


def maximum_toggle_frequency_hz() -> float:
    """Ideal upper bound from serial wire time, excluding software overhead."""
    frame_seconds = (N_EMITTERS + 2) * 10.0 / BAUDRATE
    return 1.0 / (2.0 * frame_seconds)


def run_toggle_cycle(
    port: str,
    on_frame: bytes,
    frequency_hz: float,
    duration_seconds: float | None = None,
) -> None:
    """Toggle the phase-steered ON frame and uniform OFF frame over one UART."""
    half_period = 0.5 / frequency_hz
    off_frame = build_off_frame()
    deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
    current_state = "OFF"
    cycle_index = 0

    with serial.Serial(
        port,
        BAUDRATE,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=2,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)

        print(f"\nStarting phase-steered ON/OFF cycle on {port}")
        print(f"Target toggle frequency: {frequency_hz:.2f} Hz")
        print(f"Half-period: {half_period:.6f} s")
        print("Press Ctrl+C to stop.\n")

        try:
            while deadline is None or time.monotonic() < deadline:
                ser.write(on_frame)
                ser.flush()
                current_state = "ON"
                print(f"Cycle {cycle_index + 1}: ON  ({time.strftime('%H:%M:%S')})")
                time.sleep(half_period)

                if deadline is not None and time.monotonic() >= deadline:
                    break

                ser.write(off_frame)
                ser.flush()
                current_state = "OFF"
                print(f"Cycle {cycle_index + 1}: OFF ({time.strftime('%H:%M:%S')})")
                time.sleep(half_period)
                cycle_index += 1
        finally:
            if current_state != "OFF":
                ser.write(off_frame)
                ser.flush()
                print("Sent OFF frame before closing the serial port.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toggle all transducers ON/OFF with a single-angle beam direction."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument("--frequency", type=float, help="ON/OFF toggle frequency in Hz (20..20000)")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; omit for continuous mode")
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Beam angle in degrees (0..180); if omitted, prompt interactively",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute and print frames without opening a port")
    parser.add_argument("--list-ports", action="store_true", help="Print available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports detected.")
            return
        print("Available serial ports:")
        for index, port_info in enumerate(ports):
            print(f"  [{index}] {port_info.device:<12} - {port_info.description or 'No description'}")
        return

    try:
        angle_degrees = (
            parse_angle_input(str(args.angle))
            if args.angle is not None
            else prompt_for_angle()
        )
        if args.frequency is None:
            frequency_hz = prompt_for_frequency()
        else:
            frequency_hz = parse_frequency(str(args.frequency))

        duration_seconds = args.duration
        if duration_seconds is None and not args.dry_run:
            duration_seconds = prompt_for_duration()
        elif duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds <= 0.0
        ):
            raise ValueError("Duration must be a finite number greater than 0 seconds.")

        phases = compute_steering_phase_bins(angle_degrees)
    except (ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nCancelled.")
        else:
            parser.error(str(exc))
        return

    on_frame = build_frame(phases)
    print(f"Beam angle: {angle_degrees:.2f} degrees")
    print("Angle convention: 0 degrees along +x, 90 along +z, 180 along -x")
    print(f"Phase center axis: x=0.075 m; carrier frequency: {DEFAULT_CARRIER_FREQUENCY:.0f} Hz")
    print(f"ON/OFF toggle frequency: {frequency_hz:.2f} Hz")
    print_phase_top_view(phases)
    print(f"ON frame size: {len(on_frame)} bytes; OFF frame size: {len(build_off_frame())} bytes")
    max_frequency = maximum_toggle_frequency_hz()
    print(
        f"UART wire-time limit: about {max_frequency:.2f} Hz maximum "
        "(ideal, excluding software overhead)."
    )
    if frequency_hz > max_frequency:
        print("Warning: the requested ON/OFF frequency cannot be reached by sending full frames at this baud rate.")

    if args.dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    if args.port:
        port = args.port
    else:
        try:
            port = select_serial_port(baudrate=BAUDRATE)
        except (RuntimeError, KeyboardInterrupt) as exc:
            print(f"Cannot select a serial port: {exc}")
            return

    print(f"Using port: {port}")
    try:
        run_toggle_cycle(port, on_frame, frequency_hz, duration_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")


if __name__ == "__main__":
    main()
