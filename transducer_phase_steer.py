#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Steer a continuously driven array with one angle in the x-z plane.

The 2D array is treated as a 1D array along x, centered on the y-parallel axis
x=0.075 m. One 0..180 degree angle selects the beam direction in the x-z
cross-section: 0 points along +x, 90 along +z (the array normal), and 180
along -x. One static 16-level phase frame is sent; no ON/OFF or OFF frame is
sent.
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

BAUDRATE = 38400
HEADER = 0xFE
FOOTER = 0xFD
PHASE_BINS = 16
SPEED_OF_SOUND = 343.0
DEFAULT_CARRIER_FREQUENCY = 40000.0


def parse_angle_input(raw_value: str) -> float:
    """Parse and validate the beam angle in degrees."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("角度必须是数字。") from exc
    if not math.isfinite(value):
        raise ValueError("角度必须是有限值。")
    if not 0.0 <= value <= 180.0:
        raise ValueError("角度必须在 0 到 180 度之间。")
    return value


def prompt_for_angle() -> float:
    """Prompt for the single beam angle in Chinese."""
    while True:
        raw_angle = input(
            "\n请输入波束角度（0 到 180 度；0=沿+x，90=沿+z，180=沿-x）： "
        ).strip()
        try:
            return parse_angle_input(raw_angle)
        except ValueError as exc:
            print(f"{exc} 请重新输入。", flush=True)


def select_serial_port() -> str:
    """List detected serial ports and prompt for their displayed index."""
    print("Scanning serial ports...", flush=True)
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports detected. Connect the device and try again.")

    print("\nAvailable serial ports:", flush=True)
    for index, port_info in enumerate(ports):
        print(
            f"[{index}] {port_info.device} "
            f"({port_info.description or 'No description'})",
            flush=True,
        )

    while True:
        raw = input("\nSelect port number: ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("Please enter a valid port number.", flush=True)
            continue
        if 0 <= index < len(ports):
            selected_port = ports[index].device
            print(f"Selected {selected_port}", flush=True)
            return selected_port
        print(f"Invalid selection. Choose a number from 0 to {len(ports) - 1}.", flush=True)


def compute_steering_phase_bins(
    angle_degrees: float,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY,
    sound_speed: float = SPEED_OF_SOUND,
    emitters_pos: np.ndarray = EMITTERS_POS,
) -> np.ndarray:
    """Return phases for a beam in the x-z plane around the x=0.075 m axis."""
    parameters = (angle_degrees, carrier_frequency_hz, sound_speed)
    if not all(math.isfinite(value) for value in parameters):
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

    angle = math.radians(angle_degrees)
    direction_x = math.cos(angle)
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


def build_phase_frame(phase_values: np.ndarray | list[int]) -> bytes:
    """Build a 258-byte frame with phases arranged in hardware transmit order."""
    values = np.asarray(phase_values, dtype=int)
    if values.shape != (N_EMITTERS,):
        raise ValueError(f"Expected exactly {N_EMITTERS} phase values.")
    if np.any((values < 0) | (values >= PHASE_BINS)):
        raise ValueError(f"Phase values must be between 0 and {PHASE_BINS - 1}.")

    order = np.asarray(EMITTERS_ORDER, dtype=int)
    if order.shape != (N_EMITTERS,) or not np.array_equal(
        np.sort(order), np.arange(N_EMITTERS)
    ):
        raise ValueError("EMITTERS_ORDER must be a permutation of all emitter indices.")

    payload = values[order].astype(np.uint8).tobytes()
    return bytes([HEADER]) + payload + bytes([FOOTER])


def print_phase_map(phase_values: np.ndarray) -> None:
    """Print phase values in physical 16x16 layout (y increases upward)."""
    print("\n16x16 phase map (0..15; top view):")
    print("      " + " ".join(f"x={col * 10:02d}" for col in range(16)))
    print("      " + "---" * 16)
    for row in range(15, -1, -1):
        row_values = phase_values[row * 16 : (row + 1) * 16]
        print(f"y={row * 10:02d} | " + " ".join(f"{int(value):02d}" for value in row_values))


def send_phase_frame(port: str, frame: bytes) -> None:
    """Send one static phase frame; leave the controller in its resulting state."""
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
        written = ser.write(frame)
        ser.flush()
        if written != len(frame):
            raise serial.SerialException(
                f"Incomplete frame write: sent {written} of {len(frame)} bytes."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steer the transducer array with one 0..180 degree beam angle."
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Scan and list available serial ports, then exit",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Beam angle in degrees (0..180); if omitted, prompt interactively",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute and show frame without sending")
    args = parser.parse_args()

    if args.list_ports:
        print("Scanning serial ports...", flush=True)
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports detected. Connect the device and try again.", flush=True)
            return
        print("\nAvailable serial ports:", flush=True)
        for index, port_info in enumerate(ports):
            print(
                f"[{index}] {port_info.device} "
                f"({port_info.description or 'No description'})",
                flush=True,
            )
        return

    port = None
    if not args.dry_run:
        if args.port:
            port = args.port
        else:
            try:
                # Prompt before printing the phase map so the port choices are
                # immediately visible and not pushed out of the terminal view.
                port = select_serial_port()
            except (RuntimeError, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    print("\nPort selection cancelled.")
                else:
                    print(f"Cannot select a serial port: {exc}")
                return

    try:
        angle_degrees = (
            parse_angle_input(str(args.angle))
            if args.angle is not None
            else prompt_for_angle()
        )
    except (ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\n角度输入已取消。")
            return
        parser.error(str(exc))

    try:
        phases = compute_steering_phase_bins(angle_degrees)
        frame = build_phase_frame(phases)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"波束角度: {angle_degrees:.2f} 度")
    print("角度约定: 0° 沿+x，90° 沿+z（阵列法线），180° 沿-x")
    print(f"相位计算中心轴: x=0.075 m，载波频率: {DEFAULT_CARRIER_FREQUENCY:.0f} Hz")
    print_phase_map(phases)
    print(f"UART frame size: {len(frame)} bytes (header + 256 phases + footer)")

    if args.dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    print(f"Sending phase frame to {port}...")
    try:
        send_phase_frame(port, frame)
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        return
    print("Phase frame sent. No ON/OFF toggle or OFF frame was sent.")


if __name__ == "__main__":
    main()