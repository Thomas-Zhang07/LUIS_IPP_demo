"""
Single-Point Acoustic Focus Demo
=================================

This script demonstrates how to create and send acoustic focus at a single
target position using a 16x16 ultrasonic transducer array.

Features:
  - 16-level phase control (0..15, with 16 = OFF).
  - Interactive serial port selection.
  - Interactive or command-line position input.
  - Prints phase values and UART frame for verification.

Dependencies: numpy, pyserial, and modules from the test folder.
"""

import sys
import time
import argparse
import numpy as np
import serial
import serial.tools.list_ports

# Import emitters and serial_utils from local demo folder.
from emitters import EMITTERS_POS, EMITTERS_ORDER, N_EMITTERS
from serial_utils import select_serial_port

# Acoustic parameters
WAVELENGTH = 0.00865  # meters (air ~25C, ~40 kHz)
PHASE_BINS = 16       # discrete phase levels: 0..15
BAUD = 38400


def detect_serial_ports():
    """Detect and list all available serial ports.

    Returns:
        list: List of (device, description) tuples.
    """
    ports = list(serial.tools.list_ports.comports())
    return ports


def print_available_ports():
    """Print all available serial ports in a user-friendly format."""
    ports = detect_serial_ports()
    if not ports:
        print("No serial ports detected.")
        return
    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device:<10} - {port.description}")
    print()


def compute_phases_for_focus(target_pos, emitters_pos=EMITTERS_POS):
    """Compute continuous phase values (radians) for single-point focus.

    Args:
        target_pos: array-like (3,) = [x, y, z] target position in meters.
        emitters_pos: (N, 3) array of transducer positions.

    Returns:
        phases: (N,) array of continuous phase values (radians) for each emitter.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    distances = np.linalg.norm(emitters_pos - target_pos, axis=1)
    wavelengths_count = distances / WAVELENGTH
    fractional_parts = np.ceil(wavelengths_count) - wavelengths_count
    phases = 2.0 * np.pi * fractional_parts
    return phases


def discretize_phases_to_bins(phases, num_bins=PHASE_BINS):
    """Map continuous phases [0, 2π) to discrete bins 0..num_bins-1.

    Args:
        phases: (N,) array of continuous phase values (radians).
        num_bins: number of discrete bins (default 16).

    Returns:
        values: (N,) array of integers in range [0, num_bins-1].
    """
    phases = np.asarray(phases, dtype=float)
    phases_mod = np.mod(phases, 2.0 * np.pi)
    values = (phases_mod * num_bins / (2.0 * np.pi)).astype(int)  # floor
    values = np.clip(values, 0, num_bins - 1)
    return values.astype(np.uint8)


def print_phase_top_view(phase_values):
    """Print phase bins as a 16x16 top-view physical layout."""
    print("\n16x16 phase map (top view, y increases upward):")
    print("      " + " ".join(f"x={col * 10:02d}" for col in range(16)))
    print("      " + "---" * 16)

    for row in range(15, -1, -1):
        row_values = [int(phase_values[row * 16 + col]) for col in range(16)]
        print(f"y={row * 10:02d} | " + " ".join(f"{value:02d}" for value in row_values))


def build_uart_frame(phase_values, transmit_order=None):
    """Build UART frame for transmission.

    Frame format: 0xFE (header) + 256 phase bytes + 0xFD (footer)

    Args:
        phase_values: (N,) array of phase values (0..15).
        transmit_order: optional order list; if provided, rearranges bytes.

    Returns:
        frame: bytes object ready to send.
    """
    phase_values = np.asarray(phase_values, dtype=np.uint8)
    assert phase_values.size == N_EMITTERS

    if transmit_order is not None:
        transmit_order = np.asarray(transmit_order, dtype=int)
        body = phase_values[transmit_order]
    else:
        body = phase_values

    frame = bytes([0xFE]) + bytes(body.tolist()) + bytes([0xFD])
    return frame


def send_focus_frame_to_port(port, target_x, target_y, target_z,
                              verbose=True, dry_run=False):
    """Compute and send acoustic focus frame.

    Args:
        port: serial port device name (e.g., 'COM3').
        target_x, target_y, target_z: target position in meters.
        verbose: if True, print phase values and frame details.
        dry_run: if True, compute but do NOT send (useful for testing).

    Returns:
        frame: the UART frame that was (or would be) sent.
    """
    target_pos = np.array([target_x, target_y, target_z], dtype=float)

    # Compute phases
    phases = compute_phases_for_focus(target_pos)
    values = discretize_phases_to_bins(phases, num_bins=PHASE_BINS)

    if verbose:
        print(f"\nTarget position: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m")
        print(f"Phase discretization: 0..{PHASE_BINS - 1} levels")
        print_phase_top_view(values)

    # Build frame
    frame = build_uart_frame(values, transmit_order=np.array(EMITTERS_ORDER, dtype=int))

    if verbose:
        print(f"\nUART Frame (hex): {frame.hex(' ')}")
        print(f"Frame size: {len(frame)} bytes (header + 256 body + footer)")

    # Send
    if not dry_run:
        with serial.Serial(port, BAUD, timeout=2) as ser:
            ser.dtr = False
            ser.rts = False
            time.sleep(0.5)
            ser.write(frame)
            ser.flush()
        print(f"\nFrame sent to {port}")
    else:
        print("\n[DRY RUN] Frame NOT sent.")

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Single-point acoustic focus demo (16-level phase control)."
    )
    parser.add_argument("--detect", action="store_true",
                        help="List available serial ports and exit.")
    parser.add_argument("--x", type=float, default=None,
                        help="Target X coordinate (meters). If not set, prompt user.")
    parser.add_argument("--y", type=float, default=None,
                        help="Target Y coordinate (meters). If not set, prompt user.")
    parser.add_argument("--z", type=float, default=None,
                        help="Target Z coordinate (meters). If not set, prompt user.")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g., COM3). If not set, list and prompt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute frame but do not send.")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"Baud rate (default {BAUD}).")

    args = parser.parse_args()

    # Handle --detect option
    if args.detect:
        print_available_ports()
        return

    # Select port
    if args.port:
        port = args.port
        print(f"Using port: {port}")
    else:
        try:
            port = select_serial_port(baudrate=args.baud)
        except Exception as e:
            print(f"Port selection failed: {e}")
            return

    # Get target position
    if args.x is not None and args.y is not None and args.z is not None:
        target_x, target_y, target_z = args.x, args.y, args.z
        print(f"Using target position from CLI: ({target_x}, {target_y}, {target_z})")
    else:
        # Interactive input
        default_msg = "Press Enter to use default (0.075, 0.075, 0.0)"
        user_input = input(f"\nEnter target position 'x y z' (meters), or {default_msg}: ").strip()
        if not user_input:
            target_x, target_y, target_z = 0.075, 0.075, 0.0
            print(f"Using default: ({target_x}, {target_y}, {target_z})")
        else:
            try:
                parts = user_input.split()
                if len(parts) != 3:
                    print("Invalid input: expected three numbers.")
                    return
                target_x, target_y, target_z = map(float, parts)
            except ValueError:
                print("Invalid input: could not parse coordinates.")
                return

    # Send focus frame
    send_focus_frame_to_port(port, target_x, target_y, target_z,
                              verbose=True, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
