#!/usr/bin/env python3
"""Drive both 16x16 boards as one 512-element directional array.

Board A (primary) and board B (secondary) are coplanar, side by side along x,
with board B shifted by half a lattice pitch (5 mm) in x and y.  Each board
gets its own 258-byte frame in the usual wire order; the board-select byte is
``0xC0`` for the primary and ``0xC1`` for the secondary.

Examples
--------
    # Planar broadside, no serial ports touched
    python dual_directional_speaker.py --dry-run

    # Send a broadside beam to both boards
    python dual_directional_speaker.py --board-a-port COM6 --board-b-port COM7

    # Steer 20 degrees off broadside with a Hann taper over the full aperture
    python dual_directional_speaker.py --dry-run --beta 20 \
        --amplitude-window hann

    # Near-field focus at 1.5 m on the axis
    python dual_directional_speaker.py --dry-run --focus-z 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import acoustics  # noqa: E402
import dual_board  # noqa: E402
import protocol  # noqa: E402
from serial_utils import print_available_ports  # noqa: E402

REPORT_BETAS = (0.0, 10.0, 20.0, 40.0, 50.0, 59.0, 65.0, 70.0, 90.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--board-a-port", help="Primary board serial port, e.g. COM6")
    parser.add_argument("--board-b-port", help="Secondary board serial port, e.g. COM7")
    parser.add_argument("--list-ports", action="store_true", help="List ports and exit")
    parser.add_argument("--dry-run", action="store_true", help="Build and report only")
    parser.add_argument("--baud", type=int, default=protocol.BAUD_RATE)

    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Azimuth in degrees, from +x toward +y (default 0)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Off-axis angle from +z in degrees (default 0)")
    parser.add_argument("--focus-z", type=float, default=None,
                        help="Focus distance in meters; omit for plane-wave steering")
    parser.add_argument("--focus-x", type=float, default=0.0,
                        help="Focus x coordinate in meters (with --focus-z)")
    parser.add_argument("--focus-y", type=float, default=0.0,
                        help="Focus y coordinate in meters (with --focus-z)")

    parser.add_argument("--gap", type=float, default=dual_board.DEFAULT_GAP_M,
                        help="Clear gap between the boards in meters (default 0.010)")
    parser.add_argument("--offset-x", type=float,
                        default=dual_board.DEFAULT_OFFSET_X_M,
                        help="Board B x offset in meters (default 0.005)")
    parser.add_argument("--offset-y", type=float,
                        default=dual_board.DEFAULT_OFFSET_Y_M,
                        help="Board B y offset in meters (default 0.005)")

    parser.add_argument("--amplitude-window",
                        choices=("none", "hann", "gaussian", "blackman", "tukey"),
                        default="none",
                        help="2-D taper over the full 512-element aperture")
    parser.add_argument("--amplitude-gaussian-sigma", type=float, default=0.35,
                        help="Gaussian taper sigma in (0, 1] (default 0.35)")
    parser.add_argument("--tukey-alpha", type=float, default=0.5,
                        help="Tukey taper alpha in [0, 1] (default 0.5)")

    parser.add_argument("--optimize-sweeps", type=int, default=0,
                        help="Phase-only sidelobe optimization sweeps (default 0)")
    parser.add_argument("--samples", type=int, default=81,
                        help="Direction-grid samples per axis (default 81)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null", action="append", default=[], metavar="AZ:BETA",
                        help="Extra null direction; repeatable")
    parser.add_argument("--null-weight", type=float, default=1.0)
    return parser


def parse_nulls(values):
    nulls = []
    for value in values:
        try:
            azimuth, beta = value.split(":", 1)
        except ValueError as error:
            raise ValueError(f"Null {value!r} must look like AZ:BETA.") from error
        nulls.append((float(azimuth), float(beta)))
    return nulls


def build_plan(args):
    geometry = dual_board.BoardGeometry(
        gap_m=args.gap,
        offset_x_m=args.offset_x,
        offset_y_m=args.offset_y,
    )
    bins_a, bins_b = dual_board.board_phase_bins(
        geometry,
        azimuth_degrees=args.alpha,
        off_axis_degrees=args.beta,
        focus_z_m=args.focus_z,
        focus_xy_m=(args.focus_x, args.focus_y),
    )

    weights = dual_board.amplitude_window_weights(
        geometry,
        args.amplitude_window,
        gaussian_sigma=args.amplitude_gaussian_sigma,
        tukey_alpha=args.tukey_alpha,
    )
    codes = dual_board.amplitude_codes_from_weights(weights)
    codes_a, codes_b = dual_board.split_by_board(codes)

    analyzer = dual_board.CombinedBeamAnalyzer(
        geometry.positions(),
        azimuth_degrees=args.alpha,
        off_axis_degrees=args.beta,
        samples=args.samples,
    )
    phases = np.concatenate((bins_a, bins_b))
    if args.optimize_sweeps:
        phases, info = analyzer.optimize(
            phases,
            codes,
            max_sweeps=args.optimize_sweeps,
            seed=args.seed,
            null_directions=parse_nulls(args.null),
            null_weight=args.null_weight,
        )
        bins_a, bins_b = dual_board.split_by_board(phases)
    else:
        info = {"sweeps_run": 0, "changed_elements": 0}

    payload_a, payload_b = dual_board.build_payloads(
        geometry, bins_a, bins_b, codes_a, codes_b
    )
    return geometry, analyzer, phases, codes, payload_a, payload_b, info


def report(geometry, analyzer, phases, codes, info, args) -> None:
    metrics = analyzer.metrics(phases, codes)
    print("geometry:")
    print(f"  {geometry.summary()}")
    print(f"  elements: {geometry.positions().shape[0]}")
    print(
        f"steering: alpha={args.alpha:g} deg, beta={args.beta:g} deg"
        + (
            f", focus=({args.focus_x:g}, {args.focus_y:g}, "
            f"{args.focus_z:g}) m"
            if args.focus_z is not None
            else ""
        )
    )
    print(f"taper: {args.amplitude_window}")
    if info["sweeps_run"]:
        print(
            f"optimizer: {info['sweeps_run']} sweeps, "
            f"{info['changed_elements']} phase changes"
        )
    print(
        f"active elements: {metrics['active_elements']} / "
        f"{geometry.positions().shape[0]}"
    )
    width = analyzer.half_power_width_deg(phases, codes)
    print(f"main lobe -3 dB width (beta cut): {width:.2f} deg")
    print(
        f"peak sidelobe: {metrics['peak_sidelobe_db']:.1f} dB at "
        f"az {metrics['peak_sidelobe_azimuth_deg']:.0f}, "
        f"beta {metrics['peak_sidelobe_beta_deg']:.1f}"
    )
    print("levels (az = steering azimuth):")
    for beta in REPORT_BETAS:
        print(
            f"  beta {beta:5.1f}: "
            f"{analyzer.level_at(phases, args.alpha, beta, codes):6.1f} dB"
        )
    print("59 deg grating-lobe directions:")
    for azimuth, level in sorted(metrics["grating_lobe_db"].items()):
        print(f"  az {azimuth:5.1f}: {level:6.1f} dB")
    print(
        "note: the half-pitch offset nulls the exact 59 deg direction; the "
        "remaining grating energy appears as a split lobe on either side. "
        "peak_sidelobe_db is the honest no-leak number."
    )
    print(
        "ideal isotropic model, not a measurement; element directivity and "
        "the near field will lower the absolute levels."
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        print_available_ports()
        return 0

    try:
        geometry, analyzer, phases, codes, payload_a, payload_b, info = build_plan(
            args
        )
    except (ValueError, RuntimeError) as error:
        print(f"Cannot build the dual-board plan: {error}", file=sys.stderr)
        return 2

    print(f"board A payload: {len(payload_a)} bytes (select 0xC0)")
    print(f"board B payload: {len(payload_b)} bytes (select 0xC1)")
    print(
        "wire time per board: "
        f"{protocol.wire_time_s(len(payload_a), args.baud) * 1e3:.1f} ms"
    )
    report(geometry, analyzer, phases, codes, info, args)

    if args.dry_run:
        print("board A first bytes:", payload_a[:12].hex(" "))
        print("board B first bytes:", payload_b[:12].hex(" "))
        print("[DRY RUN] no serial data sent.")
        return 0

    if not args.board_a_port or not args.board_b_port:
        print(
            "Both --board-a-port and --board-b-port are required to transmit "
            "(or use --dry-run).",
            file=sys.stderr,
        )
        return 2

    try:
        with dual_board.BoardLink(
            args.board_a_port, args.board_b_port, baudrate=args.baud
        ) as link:
            link.send(payload_a, payload_b)
    except OSError as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
