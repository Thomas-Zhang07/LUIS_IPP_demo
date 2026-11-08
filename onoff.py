#!/usr/bin/env python3
"""Send a full 0xFE + 256 bytes + 0xFD ON/OFF frame to the selected serial port.

This version is intentionally implemented in the demo folder and prints the same
serial-port selection flow as the project examples, including:

Scanning serial ports...
Available serial ports:
[0] COM8 (蓝牙链接上的标准串行 (COM8))
...
Select port number:
"""

import argparse
import sys
import time

import serial
import serial.tools.list_ports

BAUD = 38400
N_EMITTERS = 256
HEADER = 0xFE
FOOTER = 0xFD
ON_VALUE = 0x00
OFF_VALUE = 0x10


def list_ports():
    """Return all serial port objects."""
    return list(serial.tools.list_ports.comports())


def print_port_list():
    """Print the serial-port list in the same style used by the project."""
    ports = list_ports()
    print("Scanning serial ports...", flush=True)

    if not ports:
        print("No serial ports detected. Please connect the device and try again.", flush=True)
        return None

    print("\nAvailable serial ports:", flush=True)
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device} ({port.description})", flush=True)
    return ports


def select_serial_port():
    """Prompt the user to choose a port by index."""
    ports = print_port_list()
    if not ports:
        raise RuntimeError("No serial ports detected.")

    while True:
        try:
            idx = int(input("\nSelect port number: ").strip())
            if 0 <= idx < len(ports):
                selected_port = ports[idx].device
                print(f"Selected {selected_port}")
                return selected_port
            print("Invalid selection.")
        except ValueError:
            print("Please enter a valid integer.")
        except Exception as exc:
            print(f"Cannot select port: {exc}")


def build_uniform_frame(value):
    """Build a uniform UART frame of length 258: FE + 256 body + FD."""
    payload = bytes([value & 0xFF]) * N_EMITTERS
    return bytes([HEADER]) + payload + bytes([FOOTER])


def send_frame(port, frame):
    """Open the serial port and send the exact frame bytes."""
    with serial.Serial(
        port,
        BAUD,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=2,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)
        ser.write(frame)
        ser.flush()
        print(f"Sent frame ({len(frame)} bytes): {frame.hex(' ')}")


def parse_frequency(value):
    """Validate audible frequency input in Hz."""
    try:
        freq = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid frequency: {value!r}") from exc
    if not (20.0 <= freq <= 20000.0):
        raise ValueError("Frequency must be between 20 and 20000 Hz.")
    return freq


def prompt_for_frequency():
    while True:
        raw = input("Enter target frequency in Hz (20 to 20000): ").strip()
        try:
            return parse_frequency(raw)
        except ValueError as exc:
            print(f"{exc}. Please try again.")


def prompt_for_duration():
    raw = input("Enter run duration in seconds (leave empty for continuous): ").strip()
    if raw == "":
        return None
    try:
        seconds = float(raw)
    except ValueError:
        print("Invalid duration, using continuous mode.")
        return None
    if seconds <= 0:
        print("Duration must be > 0; using continuous mode.")
        return None
    return seconds


def toggle_on_off(port, frequency_hz, duration_seconds=None):
    """Swing the entire 256-transducer array ON and OFF at the target frequency."""
    half_period = 0.5 / frequency_hz
    on_frame = build_uniform_frame(ON_VALUE)
    off_frame = build_uniform_frame(OFF_VALUE)

    deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
    cycle_index = 0

    print(f"\nUsing port: {port}")
    print(f"Frequency: {frequency_hz:.2f} Hz")
    print(f"Half-period: {half_period:.6f} s")
    print("Press Ctrl+C to stop.\n")

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            print("Duration reached; stopping.")
            break

        send_frame(port, on_frame)
        print(f"Cycle {cycle_index + 1}: ON")
        time.sleep(half_period)

        if deadline is not None and time.monotonic() >= deadline:
            print("Duration reached; stopping.")
            break

        send_frame(port, off_frame)
        print(f"Cycle {cycle_index + 1}: OFF")
        time.sleep(half_period)
        cycle_index += 1


def main():
    parser = argparse.ArgumentParser(description="Toggle all 256 transducers ON/OFF at a user-selected frequency.")
    parser.add_argument("--port", type=str, help="Serial port to use directly, e.g. COM3 or /dev/rfcomm0")
    parser.add_argument("--frequency", type=float, help="Target frequency in Hz (20..20000)")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; omit for continuous mode")
    parser.add_argument("--list-ports", action="store_true", help="Print available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_ports()
        if not ports:
            print("No serial ports detected.")
            return
        print_port_list()
        return

    try:
        port = args.port if args.port else select_serial_port()
    except Exception as exc:
        print(f"Port selection failed: {exc}")
        return

    try:
        frequency_hz = args.frequency if args.frequency is not None else prompt_for_frequency()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    duration_seconds = args.duration if args.duration is not None else prompt_for_duration()

    try:
        toggle_on_off(port, frequency_hz, duration_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")


if __name__ == "__main__":
    main()
