"""Orbit the beam around the array at a fixed beta, with an audible tone on.

Sends one steering frame per angle and dwells there, so you can read a dB meter
(or listen) at fixed positions. The FPGA's audio mode is set once at the start
and is not touched by the steering frames, so the tone stays on.

Examples
--------
    # 8 directions, 3 s each, C4 tone, beta = 15 deg
    python orbit_test.py --port COM6 --mode c4 --beta 15 --step 45 --dwell 3

    # only the four cardinal directions, 6 s each, two revolutions
    python orbit_test.py --port COM6 --mode 200hz --beta 15 --step 90 --dwell 6 --cycles 2

    # counter-clockwise, no tone change, keep the current pattern at start
    python orbit_test.py --port COM6 --keep --direction ccw

    # check the frames without touching the board
    python orbit_test.py --dry-run --step 45

The array frame: x = columns (right), y = rows (up), z = out of the array.
alpha = 0 points +x (right), 90 = +y (up), 180 = -x (left), 270 = -y (down).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acoustics  # noqa: E402
import protocol  # noqa: E402
import serial  # noqa: E402

MODE_COMMANDS = {"normal": 0xC8, "c4": 0xC9, "little-star": 0xCA, "200hz": 0xCB}
CARDINAL = {0: "right (+x)", 90: "up (+y)", 180: "left (-x)", 270: "down (-y)"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orbit the beam at a fixed beta with an audible tone.",
    )
    p.add_argument("--port", help="Serial port, e.g. COM6")
    p.add_argument("--list-ports", action="store_true", help="List ports and exit")
    p.add_argument("--baud", type=int, default=protocol.BAUD_RATE)
    p.add_argument("--mode", choices=sorted(MODE_COMMANDS), default="c4",
                   help="Audio mode set once at start (default c4)")
    p.add_argument("--keep", action="store_true",
                   help="Do not send the mode byte or an all-on frame at start")
    p.add_argument("--beta", type=float, default=15.0,
                   help="Off-axis angle from +z in degrees (default 15)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Starting azimuth alpha in degrees (default 0)")
    p.add_argument("--step", type=float, default=45.0,
                   help="Azimuth step in degrees (default 45; use 90 for cardinals)")
    p.add_argument("--dwell", type=float, default=3.0,
                   help="Seconds to hold each angle (default 3)")
    p.add_argument("--cycles", type=int, default=1,
                   help="Number of revolutions (default 1; 0 = until Ctrl+C)")
    p.add_argument("--direction", choices=("cw", "ccw"), default="cw",
                   help="cw = increasing alpha, ccw = decreasing (default cw)")
    p.add_argument("--frequency", type=float, default=40000.0,
                   help="Carrier frequency in Hz (default 40000)")
    p.add_argument("--sound-speed", type=float, default=343.0,
                   help="Speed of sound in m/s (default 343)")
    p.add_argument("--off-at-end", action="store_true",
                   help="Send an all-off frame when the orbit finishes")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the angles/frames without opening the port")
    return p


def direction_label(alpha: float) -> str:
    a = int(round(alpha)) % 360
    return CARDINAL.get(a, f"{a:3d} deg")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        import serial.tools.list_ports as ports
        for p in ports.comports():
            print(f"{p.device}: {p.description}")
        return 0

    if args.dry_run:
        for alpha in orbit_angles(args):
            print(f"alpha={alpha:6.1f}  beta={args.beta:5.1f}  ({direction_label(alpha)})"
                  f"  [dry run]")
        return 0

    if not args.port:
        print("A serial port is required (or use --list-ports).", file=sys.stderr)
        return 2

    try:
        with protocol.open_serial(args.port, args.baud) as conn:
            if not args.keep:
                cmd = MODE_COMMANDS[args.mode]
                protocol.send_bytes(conn, bytes([cmd]) + protocol.build_on_frame())
                print(f"mode {args.mode} (0x{cmd:02X}) + all-on frame sent", flush=True)
                time.sleep(0.2)

            t0 = time.monotonic()
            try:
                for alpha in orbit_angles(args):
                    bins = acoustics.steering_phase_bins(
                        alpha, args.beta,
                        frequency_hz=args.frequency,
                        sound_speed=args.sound_speed,
                    )
                    frame = protocol.build_frame(bins)
                    protocol.send_bytes(conn, frame)
                    t = time.monotonic() - t0
                    print(f"[{t:6.1f}s] alpha={alpha:6.1f}  beta={args.beta:5.1f}"
                          f"  {direction_label(alpha):10s}  frame {len(frame)} B",
                          flush=True)
                    time.sleep(args.dwell)
            except KeyboardInterrupt:
                print("\nstopped (Ctrl+C); the board holds the last frame", flush=True)
                return 0

            if args.off_at_end:
                protocol.send_bytes(conn, protocol.build_off_frame())
                print("all-off frame sent", flush=True)
            else:
                print("orbit finished; the board holds the last frame", flush=True)
    except (serial.SerialException, OSError) as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Argument error: {error}", file=sys.stderr)
        return 2
    return 0


def orbit_angles(args):
    """Yield azimuths for the requested revolutions."""
    sign = 1.0 if args.direction == "cw" else -1.0
    if args.step == 0:
        raise ValueError("--step must be non-zero")
    steps = max(1, int(round(360.0 / abs(args.step))))
    one_turn = [(args.start + sign * i * abs(args.step)) % 360.0
                for i in range(steps)]
    cycles = args.cycles if args.cycles > 0 else None
    done = 0
    while cycles is None or done < cycles:
        for alpha in one_turn:
            yield alpha
        done += 1


if __name__ == "__main__":
    raise SystemExit(main())
