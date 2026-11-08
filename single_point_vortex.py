# -*- coding: gbk -*-
"""
Single-Point Acoustic Vortex Demo
=================================

Compute and send a single-point acoustic vortex for a 16x16 transducer array.
The phase is the sum of:
  1. Static focus phase at the target point.
  2. Rotational offset phase: topological_charge * polar_angle.

Coordinates are in meters. The array coordinate origin and emitter order are
shared with demo/emitters.py.
"""

import argparse
import time

import numpy as np
import serial
import serial.tools.list_ports

from emitters import EMITTERS_POS, EMITTERS_ORDER, N_EMITTERS
from serial_utils import select_serial_port

WAVELENGTH = 0.00865
PHASE_BINS = 16
BAUD = 38400


def detect_serial_ports():
    """Return all available serial ports."""
    return list(serial.tools.list_ports.comports())


def print_available_ports():
    """Print all available serial ports."""
    ports = detect_serial_ports()
    if not ports:
        print("No serial ports detected.")
        return
    print("\nAvailable serial ports:")
    for index, port in enumerate(ports):
        print(f"  [{index}] {port.device:<10} - {port.description}")
    print()


def compute_focus_phases(target_pos, emitters_pos=EMITTERS_POS):
    """Compute the static focusing phase for every emitter."""
    target_pos = np.asarray(target_pos, dtype=float)
    distances = np.linalg.norm(emitters_pos - target_pos, axis=1)
    wavelengths_count = distances / WAVELENGTH
    fractional_parts = np.ceil(wavelengths_count) - wavelengths_count
    return 2.0 * np.pi * fractional_parts


def compute_array_center(emitters_pos=EMITTERS_POS):
    """Compute the array center as the mean of all emitter positions."""
    emitters_pos = np.asarray(emitters_pos, dtype=float)
    if emitters_pos.ndim != 2 or emitters_pos.shape[1] != 3:
        raise ValueError("emitters_pos must have shape (N, 3)")
    return np.mean(emitters_pos, axis=0)


def compute_polar_coordinates(emitters_pos=EMITTERS_POS, center=None):
    """Return radius and polar angle for every emitter around the center."""
    emitters_pos = np.asarray(emitters_pos, dtype=float)
    if center is None:
        center = compute_array_center(emitters_pos)
    center = np.asarray(center, dtype=float)
    offsets = emitters_pos[:, :2] - center[:2]
    radii = np.linalg.norm(offsets, axis=1)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    return radii, angles


def compute_vortex_phases(target_pos, topological_charge=1,
                          emitters_pos=EMITTERS_POS):
    """Compute focus phase plus the vortex rotational phase offset.

    The rotational offset is l * theta, where theta is the emitter polar angle
    around the mean array center and l is the topological charge.
    """
    if not isinstance(topological_charge, (int, np.integer)):
        raise ValueError("topological_charge must be an integer")

    center = compute_array_center(emitters_pos)
    focus_phases = compute_focus_phases(target_pos, emitters_pos)
    center_distances, polar_angles = compute_polar_coordinates(emitters_pos, center)
    rotation_offsets = topological_charge * polar_angles
    total_phases = np.mod(focus_phases + rotation_offsets, 2.0 * np.pi)
    return total_phases, focus_phases, center_distances, polar_angles, rotation_offsets, center


def discretize_phases_to_bins(phases, num_bins=PHASE_BINS):
    """Map continuous phases to integer bins 0..num_bins-1."""
    phases = np.asarray(phases, dtype=float)
    phases_mod = np.mod(phases, 2.0 * np.pi)
    values = (phases_mod * num_bins / (2.0 * np.pi)).astype(int)
    values = np.clip(values, 0, num_bins - 1)
    return values.astype(np.uint8)


def print_phase_top_view(phase_values):
    """Print phase bins in physical 16x16 top-view order."""
    print("\n16x16 vortex phase map (top view, y increases upward):")
    print("      " + " ".join(f"x={col * 10:02d}" for col in range(16)))
    print("      " + "---" * 16)
    for row in range(15, -1, -1):
        row_values = [int(phase_values[row * 16 + col]) for col in range(16)]
        print(f"y={row * 10:02d} | " + " ".join(f"{value:02d}" for value in row_values))


def build_uart_frame(phase_values, transmit_order=None):
    """Build 0xFE + 256 phase bytes + 0xFD UART frame."""
    phase_values = np.asarray(phase_values, dtype=np.uint8)
    if phase_values.size != N_EMITTERS:
        raise ValueError(f"Expected {N_EMITTERS} phase values")
    if transmit_order is not None:
        body = phase_values[np.asarray(transmit_order, dtype=int)]
    else:
        body = phase_values
    return bytes([0xFE]) + bytes(body.tolist()) + bytes([0xFD])


def send_frame(port, frame, baudrate=BAUD):
    """Send one UART frame to the selected port."""
    with serial.Serial(port, baudrate, timeout=2) as ser:
        ser.dtr = False
        ser.rts = False
        time.sleep(0.5)
        ser.write(frame)
        ser.flush()
    print(f"\nFrame sent to {port}")


def send_vortex_frame_to_port(port, target_x, target_y, target_z,
                              topological_charge=1, baudrate=BAUD,
                              verbose=True, dry_run=False):
    """Compute and optionally send a single-point vortex frame."""
    target_pos = np.array([target_x, target_y, target_z], dtype=float)
    phases, focus_phases, center_distances, polar_angles, rotation_offsets, center = compute_vortex_phases(
        target_pos,
        topological_charge=topological_charge,
    )
    values = discretize_phases_to_bins(phases)

    if verbose:
        print(f"\nTarget position: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m")
        print(f"Array center (mean): ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) m")
        print(f"Topological charge: {topological_charge}")
        print("Phase formula: focus_phase + topological_charge * polar_angle")
        print_phase_top_view(values)
        print("\nPhase ranges (radians):")
        print(f"  focus phase:    {focus_phases.min():.4f} .. {focus_phases.max():.4f}")
        print(f"  center distance:{center_distances.min():.4f} .. {center_distances.max():.4f} m")
        print(f"  polar angle:    {polar_angles.min():.4f} .. {polar_angles.max():.4f}")
        print(f"  rotation offset:{rotation_offsets.min():.4f} .. {rotation_offsets.max():.4f}")

    frame = build_uart_frame(values, transmit_order=EMITTERS_ORDER)
    if verbose:
        print(f"\nUART Frame (hex): {frame.hex(' ')}")
        print(f"Frame size: {len(frame)} bytes (header + 256 body + footer)")

    if dry_run:
        print("\n[DRY RUN] Frame NOT sent.")
    else:
        send_frame(port, frame, baudrate=baudrate)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Single-point acoustic vortex demo (16-level phase control)."
    )
    parser.add_argument("--detect", action="store_true",
                        help="List available serial ports and exit.")
    parser.add_argument("--x", type=float, default=None,
                        help="Target X coordinate in meters.")
    parser.add_argument("--y", type=float, default=None,
                        help="Target Y coordinate in meters.")
    parser.add_argument("--z", type=float, default=None,
                        help="Target Z coordinate in meters.")
    parser.add_argument("--charge", type=int, default=1,
                        help="Vortex topological charge, e.g. 1 or -1 (default 1).")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port, e.g. COM3.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute frame but do not send.")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"Baud rate (default {BAUD}).")
    args = parser.parse_args()

    if args.detect:
        print_available_ports()
        return

    if args.port:
        port = args.port
        print(f"Using port: {port}")
    else:
        try:
            port = select_serial_port(baudrate=args.baud)
        except Exception as error:
            print(f"Port selection failed: {error}")
            return

    if args.x is not None and args.y is not None and args.z is not None:
        target_x, target_y, target_z = args.x, args.y, args.z
        topological_charge = args.charge
    else:
        user_input = input(
            "\nEnter vortex target 'x y z' (m), "
            "or press Enter for default (0.075, 0.075, 0.05): "
        ).strip()
        if not user_input:
            target_x, target_y, target_z = 0.075, 0.075, 0.05
        else:
            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input: expected three numbers.")
                return
            try:
                target_x, target_y, target_z = map(float, parts)
            except ValueError:
                print("Invalid input: could not parse coordinates.")
                return

        direction = input(
            "Direction [ccw=counter-clockwise, cw=clockwise] "
            f"(default {'ccw' if args.charge >= 0 else 'cw'}): "
        ).strip().lower()
        if not direction:
            topological_charge = args.charge
        elif direction in ("ccw", "counter-clockwise", "counterclockwise", "逆时针"):
            topological_charge = abs(args.charge) or 1
        elif direction in ("cw", "clockwise", "顺时针"):
            topological_charge = -(abs(args.charge) or 1)
        else:
            print("Invalid direction: use ccw/cw (逆时针/顺时针).")
            return

    send_vortex_frame_to_port(
        port,
        target_x,
        target_y,
        target_z,
        topological_charge=topological_charge,
        baudrate=args.baud,
        verbose=True,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
