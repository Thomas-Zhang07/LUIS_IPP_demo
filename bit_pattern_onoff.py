#!/usr/bin/env python3
"""Drive all 256 transducers using an 8-bit binary pattern.

Example:
    10110001

This script will repeat the pattern over time. Each 8-bit value is interpreted as:
    1 -> ON  (send full-frame value 0x00)
    0 -> OFF (send full-frame value 0x10)

The array is toggled as a whole, not per transducer, using the send_frame
protocol already used in this project:
    0xFE + 256 payload bytes + 0xFD
"""

import argparse
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
    return list(serial.tools.list_ports.comports())


def print_port_list():
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
    ports = print_port_list()
    if not ports:
        raise RuntimeError("No serial ports detected.")

    while True:
        try:
            idx = int(input("\nSelect port number: ").strip())
            if 0 <= idx < len(ports):
                selected = ports[idx].device
                print(f"Selected {selected}")
                return selected
            print("Invalid selection.")
        except ValueError:
            print("Please enter a valid integer.")
        except Exception as exc:
            print(f"Cannot select port: {exc}")


def build_uniform_frame(value):
    payload = bytes([value & 0xFF]) * N_EMITTERS
    return bytes([HEADER]) + payload + bytes([FOOTER])


def parse_bit_pattern(raw):
    clean = raw.strip().replace(" ", "").replace(",", "")
    if len(clean) != 8:
        raise ValueError("Pattern must contain exactly 8 bits.")
    if any(ch not in "01" for ch in clean):
        raise ValueError("Pattern can only contain 0 and 1 characters.")
    return [1 if ch == "1" else 0 for ch in clean]


def prompt_for_pattern():
    while True:
        raw = input("Enter 8-bit binary pattern, e.g. 10110001: ").strip()
        try:
            return parse_bit_pattern(raw)
        except ValueError as exc:
            print(f"{exc}. Please try again.")


def prompt_for_bit_seconds():
    while True:
        raw = input("Enter time for each bit in seconds (for example 0.1): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number of seconds.")
            continue
        if value <= 0:
            print("Time per bit must be greater than 0.")
            continue
        return value


def prompt_for_duration():
    raw = input("Run for how many seconds? Leave empty for continuous mode: ").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        print("Invalid duration; using continuous mode.")
        return None
    if value <= 0:
        print("Duration must be positive; using continuous mode.")
        return None
    return value


def send_frame(port, frame):
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


def play_pattern(port, pattern, bit_seconds, duration_seconds=None):
    """Repeat the user pattern endlessly or for a fixed length of time."""
    frames = {
        0: build_uniform_frame(OFF_VALUE),
        1: build_uniform_frame(ON_VALUE),
    }

    deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
    step = 0

    print(f"\nUsing port: {port}")
    print(f"Pattern: {''.join('1' if bit else '0' for bit in pattern)}")
    print(f"Each bit lasts {bit_seconds:.3f} seconds")
    if duration_seconds is None:
        print("Looping continuously. Press Ctrl+C to stop.\n")
    else:
        print(f"Total duration: {duration_seconds:.3f} s\n")

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            print("Duration reached; stopping.")
            break

        bit_value = pattern[step % len(pattern)]
        frame = frames[bit_value]
        send_frame(port, frame)
        print(f"Step {step + 1}: bit={bit_value} ({'ON' if bit_value else 'OFF'})")
        time.sleep(bit_seconds)
        step += 1


def main():
    parser = argparse.ArgumentParser(description="Repeat an 8-bit ON/OFF pattern across the full 256-transducer array.")
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3 or /dev/rfcomm0")
    parser.add_argument("--pattern", type=str, help="8-bit binary pattern to repeat, e.g. 10110001")
    parser.add_argument("--bit-seconds", type=float, help="How long each bit lasts in seconds")
    parser.add_argument("--duration", type=float, help="Run for this many seconds; omit for continuous mode")
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
        pattern = parse_bit_pattern(args.pattern) if args.pattern else prompt_for_pattern()
    except ValueError as exc:
        print(f"{exc}")
        return

    if args.bit_seconds is not None:
        bit_seconds = args.bit_seconds
    else:
        bit_seconds = prompt_for_bit_seconds()

    duration_seconds = args.duration if args.duration is not None else prompt_for_duration()

    try:
        play_pattern(port, pattern, bit_seconds, duration_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")


if __name__ == "__main__":
    main()
