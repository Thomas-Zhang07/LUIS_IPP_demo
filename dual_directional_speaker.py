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
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import acoustics  # noqa: E402
import dual_board  # noqa: E402
import protocol  # noqa: E402
from serial_utils import print_available_ports  # noqa: E402

REPORT_BETAS = (0.0, 10.0, 20.0, 40.0, 50.0, 59.0, 61.0, 65.0, 70.0, 90.0)


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
                        help="Clear extra gap between the boards in meters (default 0.000)")
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

    parser.add_argument("--side-start", type=float, default=60.0, metavar="DEG",
                        help="Start of the side-leak region to report and optimize (default 60; 0 = all sidelobes)")
    parser.add_argument("--element-diameter-mm", type=float, default=10.0,
                        help="Transducer piston diameter for directivity (0 = isotropic, default 10)")
    parser.add_argument("--near-weight", type=float, default=0.5,
                        help="Penalty on the below---side-start sidelobes while optimizing (default 0.5)")
    parser.add_argument("--trim", type=int, default=None, metavar="N",
                        help="Secondary fine trim 0..255 (20 ns/step; default 127)")
    parser.add_argument("--trim-sweep", default=None, metavar="START:STOP:STEP",
                        help="Sweep the secondary trim after sending the beam")
    parser.add_argument("--trim-dwell", type=float, default=2.0, metavar="S",
                        help="Dwell time per sweep point in seconds (default 2)")
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


def parse_trim_sweep(value):
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError as error:
        raise ValueError(f"Trim sweep {value!r} must look like START:STOP:STEP.") from error
    if len(parts) != 3:
        raise ValueError(f"Trim sweep {value!r} must look like START:STOP:STEP.")
    start, stop, step = parts
    if step <= 0:
        raise ValueError("Trim sweep step must be positive.")
    if not (0 <= start <= 255 and 0 <= stop <= 255):
        raise ValueError("Trim sweep values must be 0..255.")
    return list(range(start, stop + 1, step))


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

    if args.side_start < 0.0 or args.side_start > 90.0:
        raise ValueError("--side-start must be 0..90.")
    analyzer = dual_board.CombinedBeamAnalyzer(
        geometry.positions(),
        azimuth_degrees=args.alpha,
        off_axis_degrees=args.beta,
        samples=args.samples,
        element_diameter_m=(
            None if args.element_diameter_mm <= 0 else args.element_diameter_mm / 1000.0
        ),
        side_start_deg=args.side_start if args.side_start > 0 else 60.0,
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
            side_start_deg=args.side_start if args.side_start > 0 else None,
            near_weight=args.near_weight,
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
    print(f"main-lobe gain: {metrics['main_lobe_gain_db']:.1f} dB")
    print(
        f"peak sidelobe: {metrics['peak_sidelobe_db']:.1f} dB at "
        f"az {metrics['peak_sidelobe_azimuth_deg']:.0f}, "
        f"beta {metrics['peak_sidelobe_beta_deg']:.1f}"
    )
    print(
        f"side leak above {metrics['side_start_deg']:g} deg: "
        f"{metrics['far_sidelobe_db']:.1f} dB at "
        f"az {metrics['far_sidelobe_azimuth_deg']:.0f}, "
        f"beta {metrics['far_sidelobe_beta_deg']:.1f}"
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
        "far_sidelobe_db is the peak above --side-start; the optimizer "
        "targets that region, so watch main_lobe_gain_db and the near peak."
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

    try:
        sweep = parse_trim_sweep(args.trim_sweep) if args.trim_sweep else None
    except ValueError as error:
        print(f"Invalid trim sweep: {error}", file=sys.stderr)
        return 2
    if args.trim is not None and not 0 <= args.trim <= 255:
        print("--trim must be 0..255.", file=sys.stderr)
        return 2
    if args.trim is not None and sweep is not None:
        print("Use either --trim or --trim-sweep, not both.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("board A first bytes:", payload_a[:12].hex(" "))
        print("board B first bytes:", payload_b[:12].hex(" "))
        if args.trim is not None:
            print("trim command:", protocol.build_trim_command(args.trim).hex(" "))
        if sweep is not None:
            print(
                "trim sweep:", " ".join(str(value) for value in sweep),
                f"({args.trim_dwell:g} s each)",
            )
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
            if sweep is not None:
                link.send(payload_a, payload_b)
                for value in sweep:
                    link.send_trim(value)
                    print(f"trim {value:3d}")
                    time.sleep(args.trim_dwell)
            else:
                if args.trim is not None:
                    link.send_trim(args.trim)
                    print(f"trim {args.trim:3d}")
                link.send(payload_a, payload_b)
    except OSError as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
