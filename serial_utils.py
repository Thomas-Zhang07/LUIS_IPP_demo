import serial.tools.list_ports


def list_available_ports():
    """Return all available serial ports as a list of port info objects."""
    return list(serial.tools.list_ports.comports())


def select_serial_port(baudrate=38400):
    """List available serial ports and prompt the user to select one.

    Returns the device string (e.g., 'COM3').
    """
    ports = list_available_ports()

    if not ports:
        raise RuntimeError("No serial ports detected. Connect the device and try again.")

    print("\nAvailable serial ports:")
    print("Type the number in brackets, or type the exact port name (for example COM5 /dev/rfcomm0).")
    for i, p in enumerate(ports):
        description = p.description or "No description"
        print(f"  [{i}] {p.device:<12}  -  {description}")

    while True:
        raw = input("Select port: ").strip()
        if not raw:
            print("Selection cannot be empty. Try again.")
            continue

        # Support both numeric index and direct device name.
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(ports):
                return ports[idx].device
            print(f"Index out of range. Choose a number from 0 to {len(ports) - 1}.")
            continue

        for p in ports:
            if p.device.lower() == raw.lower():
                return p.device
        print(f"Port '{raw}' was not found. Please choose one of the listed ports.")
