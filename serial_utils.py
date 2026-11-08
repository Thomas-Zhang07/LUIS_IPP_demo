"""Serial port discovery and selection shared by every demo script.

This is the only port-selection implementation in the project. Selection
accepts either the list index or the exact device name (case-insensitive),
and the input/output functions are injectable for tests and for CLI wrappers.
"""

from __future__ import annotations

import serial.tools.list_ports

BAUD_RATE = 38400


def list_available_ports():
    """Return all available serial ports as a list of port info objects."""
    return list(serial.tools.list_ports.comports())


def print_available_ports(*, output=print) -> bool:
    """Print the detected ports. Return True if at least one was found."""
    ports = list_available_ports()
    if not ports:
        output("No serial ports detected. Connect the device and try again.")
        return False

    output("\nAvailable serial ports:")
    for index, port in enumerate(ports):
        description = port.description or "No description"
        output(f"  [{index}] {port.device:<12}  -  {description}")
    return True


def select_serial_port(baudrate=BAUD_RATE, *, input_fn=input, output=print) -> str:
    """Prompt for a port by index or device name and return its device string.

    Args:
        baudrate: accepted for backward compatibility; selection does not use it.
        input_fn: prompt function, injectable for tests.
        output: message sink, injectable for tests.

    Raises:
        RuntimeError: if no serial ports are detected.
    """
    ports = list_available_ports()
    if not ports:
        raise RuntimeError(
            "No serial ports detected. Connect the device and try again."
        )

    output("\nAvailable serial ports:")
    output(
        "Type the number in brackets, or type the exact port name "
        "(for example COM5 /dev/rfcomm0)."
    )
    for index, port in enumerate(ports):
        description = port.description or "No description"
        output(f"  [{index}] {port.device:<12}  -  {description}")

    while True:
        raw = input_fn("Select port: ").strip()
        if not raw:
            output("Selection cannot be empty. Try again.")
            continue

        if raw.isdigit():
            index = int(raw)
            if 0 <= index < len(ports):
                return ports[index].device
            output(f"Index out of range. Choose a number from 0 to {len(ports) - 1}.")
            continue

        for port in ports:
            if port.device.lower() == raw.lower():
                return port.device
        output(f"Port '{raw}' was not found. Please choose one of the listed ports.")


def resolve_serial_port(
    port,
    *,
    dry_run: bool = False,
    baudrate=BAUD_RATE,
    input_fn=input,
    output=print,
) -> str | None:
    """Return the port to use, or None when a dry run needs no port.

    An explicitly supplied port is returned without prompting. Dry runs with
    no explicit port return None; interactive runs prompt via
    :func:`select_serial_port`.
    """
    if port:
        return port
    if dry_run:
        return None
    return select_serial_port(baudrate=baudrate, input_fn=input_fn, output=output)
