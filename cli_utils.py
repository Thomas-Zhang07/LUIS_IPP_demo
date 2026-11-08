"""Shared argparse and console-output helpers for the demo scripts."""

from __future__ import annotations

import acoustics
import protocol
from emitters import N_SIDE


def add_port_arg(parser) -> None:
    parser.add_argument(
        "--port", type=str, default=None, help="Serial port (e.g., COM3)."
    )


def add_baud_arg(parser, *, default: int = protocol.BAUD_RATE) -> None:
    parser.add_argument(
        "--baud", type=int, default=default, help=f"Baud rate (default {default})."
    )


def add_dry_run_arg(parser, *, help_text: str = "Compute frames but do not send.") -> None:
    parser.add_argument("--dry-run", action="store_true", help=help_text)


def add_target_args(parser) -> None:
    parser.add_argument("--x", type=float, default=None, help="Target X coordinate (meters).")
    parser.add_argument("--y", type=float, default=None, help="Target Y coordinate (meters).")
    parser.add_argument("--z", type=float, default=None, help="Target Z coordinate (meters).")


def add_acoustic_args(
    parser,
    *,
    frequency_flag: str = "--frequency",
    sound_speed_flag: str = "--sound-speed",
) -> None:
    parser.add_argument(
        frequency_flag,
        type=float,
        default=acoustics.DEFAULT_FREQUENCY_HZ,
        help=f"Carrier frequency in Hz (default {acoustics.DEFAULT_FREQUENCY_HZ:.0f}).",
    )
    parser.add_argument(
        sound_speed_flag,
        type=float,
        default=acoustics.DEFAULT_SOUND_SPEED_MPS,
        help=f"Speed of sound in m/s (default {acoustics.DEFAULT_SOUND_SPEED_MPS:.0f}).",
    )


def read_target_triplet(
    values,
    *,
    default,
    prompt: str,
    option_label: str = "target",
    input_fn=None,
):
    """Return an (x, y, z) target from CLI values or an interactive prompt.

    All three values must be supplied on the command line, or none. When none
    are supplied, pressing Enter selects ``default``.
    """
    ask = input if input_fn is None else input_fn
    x, y, z = values

    if all(value is not None for value in values):
        return (x, y, z)
    if all(value is None for value in values):
        raw = ask(prompt).strip()
        if not raw:
            return tuple(default)
        parts = raw.split()
        if len(parts) != 3:
            raise ValueError(f"{option_label} must contain exactly three numbers.")
        try:
            return tuple(float(part) for part in parts)
        except ValueError as error:
            raise ValueError(f"{option_label} coordinates must be numbers.") from error
    raise ValueError("Provide all of --x, --y, and --z, or none of them.")


def print_phase_top_view(
    phase_values,
    *,
    title: str = "16x16 phase map (top view, y increases upward):",
    output=print,
) -> None:
    """Print logical phase bins as a 16x16 physical-layout top view."""
    output(f"\n{title}")
    output("      " + " ".join(f"x={column * 10:02d}" for column in range(N_SIDE)))
    output("      " + "---" * N_SIDE)
    for row in range(N_SIDE - 1, -1, -1):
        row_values = [
            int(phase_values[row * N_SIDE + column]) for column in range(N_SIDE)
        ]
        output(f"y={row * 10:02d} | " + " ".join(f"{value:02d}" for value in row_values))


def print_frame(frame: bytes, *, output=print) -> None:
    """Print a frame in hex and its size."""
    output(f"\nUART Frame (hex): {frame.hex(' ')}")
    output(f"Frame size: {len(frame)} bytes (header + 256 body + footer)")
