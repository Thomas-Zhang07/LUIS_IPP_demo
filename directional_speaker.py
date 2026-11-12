#!/usr/bin/env python3
"""Directional-speaker beam shaping for the 16x16 array.

Computes a steered phase frame and optionally applies the host-side
directivity algorithms available through the phase-only protocol:

* windowed density thinning (binary ON/OFF amplitude shaping),
* per-element duty-cycle amplitude tapering (Hann/Gaussian, duty firmware),
* deterministic phase dithering,
* phase-only sidelobe optimization with optional nulls,
* per-channel phase calibration offsets.

The script reports the idealized pattern before and after shaping, including
the grating lobes predicted by the 10 mm lattice. Those grating lobes are a
property of the periodic lattice: no per-element phase or ON/OFF pattern can
lower them (see :mod:`directivity`). The algorithms improve the non-grating
sidelobe floor; the ~3 degree main lobe is set by the aperture.

Examples:

    python directional_speaker.py --dry-run --alpha 0 --beta 20
    python directional_speaker.py --dry-run --alpha 0 --beta 20 \
        --density-window hann --keep-fraction 0.7 --optimize-sweeps 4
    python directional_speaker.py --compare --alpha 0 --beta 0
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

from acoustics import (  # noqa: E402
    acoustic_wavelength,
    steering_phase_bins,
    validate_acoustic_params,
    validate_steering_angles,
)
from array_factor import print_physical_phase_map  # noqa: E402
from cli_utils import (  # noqa: E402
    add_acoustic_args,
    add_baud_arg,
    add_dry_run_arg,
    add_port_arg,
    run_cli,
)
from directivity import (  # noqa: E402
    APERTURE_WINDOWS,
    BeamAnalyzer,
    amplitude_codes_from_weights,
    apply_calibration_bins,
    aperture_weights,
    dither_phase_bins,
    duty_width_from_code,
    load_calibration_file,
    plot_pattern,
    thinned_mask,
)
from emitters import N_EMITTERS, SPACING  # noqa: E402
from protocol import (  # noqa: E402
    OFF_VALUE,
    build_amplitude_frame,
    build_frame,
    write_frame_to_port,
)
from serial_utils import print_available_ports, resolve_serial_port  # noqa: E402


def parse_null_direction(value: str):
    """Parse ``AZIMUTH:BETA`` (or ``AZIMUTH,BETA``) into two floats."""
    parts = value.replace(",", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "null direction must be written as AZIMUTH:BETA"
        )
    try:
        azimuth, beta = float(parts[0]), float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "null direction angles must be numbers"
        ) from error
    if not (math.isfinite(azimuth) and math.isfinite(beta)):
        raise argparse.ArgumentTypeError("null direction angles must be finite")
    return azimuth, beta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Steer the directional speaker and optionally shape its beam with "
            "thinning, duty-cycle amplitude tapering, phase dithering, "
            "phase-only optimization or calibration."
        )
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="Azimuth in degrees (default 0)."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Off-axis angle from +z in degrees, 0..90 (default 0).",
    )
    add_acoustic_args(parser)
    parser.add_argument(
        "--density-window",
        choices=APERTURE_WINDOWS,
        default="none",
        help=(
            "Binary amplitude window used when thinning elements "
            "(default none: flat)."
        ),
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=1.0,
        help="Fraction of the 256 elements kept ON (default 1.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for thinning and dithering."
    )
    parser.add_argument(
        "--amplitude-window",
        choices=APERTURE_WINDOWS,
        default="none",
        help=(
            "Per-element duty-cycle amplitude taper (default none). "
            "Requires the duty-weighting FPGA firmware."
        ),
    )
    parser.add_argument(
        "--amplitude-gaussian-sigma",
        type=float,
        default=0.35,
        help="Gaussian sigma for --amplitude-window gaussian (default 0.35).",
    )
    parser.add_argument(
        "--dither-bins",
        type=int,
        default=0,
        help="Random per-element phase dither in +/- bins (default 0).",
    )
    parser.add_argument(
        "--optimize-sweeps",
        type=int,
        default=0,
        help="Phase-only coordinate-descent sweeps; 0 disables (default 0).",
    )
    parser.add_argument(
        "--null",
        action="append",
        type=parse_null_direction,
        default=None,
        metavar="AZ:BETA",
        help="Add a null direction; repeat for several (optimizer only).",
    )
    parser.add_argument(
        "--null-weight",
        type=float,
        default=1.0,
        help="Penalty weight for requested nulls (default 1.0).",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="JSON file with 256 phase offsets in bins (0..15).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=121,
        help="Direction-cosine samples per axis for analysis (default 121).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a table comparing the shaping strategies and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the final ideal pattern (requires matplotlib).",
    )
    add_port_arg(parser)
    add_baud_arg(parser)
    add_dry_run_arg(parser)
    parser.add_argument(
        "--list-ports", action="store_true", help="List serial ports and exit."
    )
    return parser.parse_args(argv)


def _measure(analyzer, bins, mask, amplitude_codes=None):
    return analyzer.metrics(bins, active_mask=mask, amplitude_codes=amplitude_codes)


def _format_row(name, metrics):
    return (
        f"{name:<22s} {metrics['active_elements']:>6d}  "
        f"{metrics['near_sidelobe_db']:>8.2f}  "
        f"{metrics['grating_lobe_db']:>8.2f}  "
        f"{metrics['main_lobe_gain_db']:>+9.2f}"
    )


def print_comparison(args, analyzer):
    """Print the measured trade-off of every host-side strategy."""
    baseline = steering_phase_bins(
        args.alpha,
        args.beta,
        frequency_hz=args.frequency,
        sound_speed=args.sound_speed,
    )
    uniform = np.ones(N_EMITTERS, dtype=bool)
    hann = thinned_mask("hann", 0.7, seed=args.seed)
    gauss = thinned_mask("gaussian", 0.7, seed=args.seed)
    amp_hann = amplitude_codes_from_weights(aperture_weights("hann"))
    amp_gauss = amplitude_codes_from_weights(
        aperture_weights("gaussian", gaussian_sigma=args.amplitude_gaussian_sigma)
    )

    rows = [
        ("uniform", baseline, uniform, None),
        (
            "dither +/-1 bin",
            dither_phase_bins(baseline, 1, seed=args.seed),
            uniform,
            None,
        ),
        ("amplitude hann", baseline, uniform, amp_hann),
        ("amplitude gaussian", baseline, uniform, amp_gauss),
    ]
    for name, mask in (("hann 70%", hann), ("gaussian 70%", gauss)):
        thinned = baseline.copy()
        thinned[~mask] = OFF_VALUE
        rows.append((name, thinned, mask, None))

    optimized, _ = analyzer.optimize(
        baseline, uniform, max_sweeps=max(args.optimize_sweeps, 3), seed=args.seed
    )
    rows.append(("uniform + optimize", optimized, uniform, None))
    amp_optimized, _ = analyzer.optimize(
        baseline,
        uniform,
        max_sweeps=max(args.optimize_sweeps, 3),
        seed=args.seed,
        amplitude_codes=amp_hann,
    )
    rows.append(("amplitude hann + opt", amp_optimized, uniform, amp_hann))
    for name, mask in (("hann 70% + optimize", hann), ("gaussian 70% + optimize", gauss)):
        thinned = baseline.copy()
        thinned[~mask] = OFF_VALUE
        shaped, _ = analyzer.optimize(
            thinned,
            mask,
            max_sweeps=max(args.optimize_sweeps, 3),
            seed=args.seed,
        )
        rows.append((name, shaped, mask, None))

    print(
        "Strategy               Active  Near SLL     Grating  Main gain\n"
        "                              (dB)          (dB)      (dB)"
    )
    for name, bins, mask, codes in rows:
        print(_format_row(name, _measure(analyzer, bins, mask, codes)))
    print()
    print(
        "Near SLL excludes the grating-lobe regions. Grating lobes are fixed by "
        "the\n10 mm lattice: no per-element phase or amplitude pattern can lower them."
    )


def build_shaped_bins(args, analyzer):
    """Build the shaped frame (before calibration) and the active mask."""
    bins = steering_phase_bins(
        args.alpha,
        args.beta,
        frequency_hz=args.frequency,
        sound_speed=args.sound_speed,
    )
    if args.density_window != "none" or args.keep_fraction < 1.0:
        mask = thinned_mask(
            args.density_window, args.keep_fraction, seed=args.seed
        )
    else:
        mask = np.ones(N_EMITTERS, dtype=bool)
    if not np.all(mask):
        bins[~mask] = OFF_VALUE
    if args.dither_bins:
        bins = dither_phase_bins(
            bins, args.dither_bins, seed=args.seed, active_mask=mask
        )

    amplitude_codes = None
    if args.amplitude_window != "none":
        amplitude_codes = amplitude_codes_from_weights(
            aperture_weights(
                args.amplitude_window,
                gaussian_sigma=args.amplitude_gaussian_sigma,
            )
        )

    optimize_info = {"sweeps_run": 0, "changed_elements": 0}
    if args.optimize_sweeps:
        bins, optimize_info = analyzer.optimize(
            bins,
            active_mask=mask,
            max_sweeps=args.optimize_sweeps,
            seed=args.seed,
            null_directions=args.null,
            null_weight=args.null_weight,
            amplitude_codes=amplitude_codes,
        )
    return bins, mask, optimize_info, amplitude_codes


def main(argv=None):
    args = parse_args(argv)

    if args.list_ports:
        print_available_ports()
        return

    try:
        validate_acoustic_params(args.frequency, args.sound_speed)
        validate_steering_angles(args.alpha, args.beta)
        if not 0.0 <= args.beta <= 90.0:
            raise ValueError("--beta must be in 0..90 degrees for this demo.")
        if not 0.0 < args.keep_fraction <= 1.0:
            raise ValueError("--keep-fraction must be greater than 0 and at most 1.")
        if not 0 <= args.dither_bins <= 8:
            raise ValueError("--dither-bins must be in 0..8.")
        if args.optimize_sweeps < 0:
            raise ValueError("--optimize-sweeps must be non-negative.")
        if args.null and args.optimize_sweeps == 0:
            raise ValueError("--null requires --optimize-sweeps greater than 0.")
        if not math.isfinite(args.amplitude_gaussian_sigma) or args.amplitude_gaussian_sigma <= 0:
            raise ValueError("--amplitude-gaussian-sigma must be greater than 0.")
    except ValueError as error:
        print(f"Invalid argument: {error}", file=sys.stderr)
        return 2

    try:
        analyzer = BeamAnalyzer(
            args.alpha,
            args.beta,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
            samples=args.samples,
        )
    except ValueError as error:
        print(f"Invalid argument: {error}", file=sys.stderr)
        return 2

    wavelength = acoustic_wavelength(args.frequency, args.sound_speed)
    print(f"Directional speaker: alpha={args.alpha:.2f} deg, beta={args.beta:.2f} deg")
    print(
        f"Carrier {args.frequency:.0f} Hz, c={args.sound_speed:.0f} m/s, "
        f"lambda={wavelength * 1000:.3f} mm, spacing={SPACING * 1000:.1f} mm "
        f"({SPACING / wavelength:.3f} lambda)"
    )
    threshold = analyzer.spatial_aliasing_frequency_hz
    if analyzer.grating_lobes:
        centers = ", ".join(
            f"(az={math.degrees(math.atan2(lobe.uy, lobe.ux)):.1f}, "
            f"beta={math.degrees(math.acos(max(-1.0, min(1.0, lobe.uz)))):.1f})"
            for lobe in analyzer.grating_lobes
        )
        print(
            f"Predicted grating lobes: {centers}"
        )
        print(
            f"  c/spacing = {threshold:.0f} Hz. The carrier is above it, so these "
            "lobes are\n  set by the lattice and cannot be lowered by any host-side "
            "phase/amplitude pattern."
        )
    else:
        print(f"No grating lobe near this direction (c/spacing = {threshold:.0f} Hz).")

    if args.compare:
        print()
        print_comparison(args, analyzer)
        return

    baseline = steering_phase_bins(
        args.alpha,
        args.beta,
        frequency_hz=args.frequency,
        sound_speed=args.sound_speed,
    )
    shaped, mask, optimize_info, amplitude_codes = build_shaped_bins(args, analyzer)

    baseline_metrics = analyzer.metrics(baseline)
    shaped_metrics = analyzer.metrics(shaped, mask, amplitude_codes=amplitude_codes)

    print()
    print(
        f"{'':<22s} {'Near SLL':>10s} {'Grating':>10s} {'Main gain':>11s} "
        f"{'Active':>7s}"
    )
    print(
        f"{'baseline':<22s} {baseline_metrics['near_sidelobe_db']:>9.2f}  "
        f"{baseline_metrics['grating_lobe_db']:>9.2f}  "
        f"{baseline_metrics['main_lobe_gain_db']:>+10.2f}  "
        f"{baseline_metrics['active_elements']:>7d}"
    )
    print(
        f"{'shaped':<22s} {shaped_metrics['near_sidelobe_db']:>9.2f}  "
        f"{shaped_metrics['grating_lobe_db']:>9.2f}  "
        f"{shaped_metrics['main_lobe_gain_db']:>+10.2f}  "
        f"{shaped_metrics['active_elements']:>7d}"
    )
    print(
        f"Change: near sidelobe "
        f"{shaped_metrics['near_sidelobe_db'] - baseline_metrics['near_sidelobe_db']:+.2f} dB, "
        f"main gain "
        f"{shaped_metrics['main_lobe_gain_db'] - baseline_metrics['main_lobe_gain_db']:+.2f} dB"
    )
    if amplitude_codes is not None:
        muted = int(np.count_nonzero(amplitude_codes == OFF_VALUE))
        active_codes = amplitude_codes[amplitude_codes != OFF_VALUE]
        widths = sorted({duty_width_from_code(int(code)) for code in active_codes})
        print(
            f"Amplitude taper: {args.amplitude_window} "
            f"({muted} zero-weight channel(s) OFF; duty widths {widths}). "
            "Requires the duty-weighting FPGA firmware."
        )
    if optimize_info["sweeps_run"]:
        print(
            f"Phase-only optimizer: {optimize_info['sweeps_run']} sweep(s), "
            f"{optimize_info['changed_elements']} element update(s)."
        )
        for note in optimize_info.get("null_directions", []):
            warning = ""
            if note["in_main_lobe"]:
                warning = " (inside the main lobe: not a valid null)"
            elif note["in_grating_lobe"]:
                warning = " (inside a grating lobe: cannot be nulled)"
            print(
                f"  null requested at az={note['azimuth_deg']:.1f}, "
                f"beta={note['beta_deg']:.1f} -> grid "
                f"az={note['grid_azimuth_deg']:.1f}, "
                f"beta={note['grid_beta_deg']:.1f}: "
                f"{note['initial_db']:.1f} -> {note['achieved_db']:.1f} dB"
                f"{warning}"
            )
    if args.calibration:
        print(
            "Calibration offsets are applied last. They correct unmodelled "
            "channel errors,\n  so the ideal-model numbers above do not include "
            "them."
        )

    print_physical_phase_map(shaped)
    print("Physical map values are phase bins; 16 means OFF (0x10).")
    if amplitude_codes is not None:
        print("Amplitude codes are carried separately in the high nibble.")

    if args.plot:
        try:
            plot_pattern(analyzer, shaped, mask, amplitude_codes=amplitude_codes)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 2

    final_bins = shaped
    if args.calibration:
        try:
            offsets = load_calibration_file(args.calibration)
        except ValueError as error:
            print(f"Calibration error: {error}", file=sys.stderr)
            return 2
        final_bins = apply_calibration_bins(shaped, offsets, active_mask=mask)

    if amplitude_codes is None:
        frame = build_frame(final_bins)
    else:
        frame = build_amplitude_frame(final_bins, amplitude_codes)

    if args.dry_run:
        print(f"Frame size: {len(frame)} bytes. [DRY RUN] No serial data sent.")
        return

    try:
        port = resolve_serial_port(args.port, dry_run=False, baudrate=args.baud)
    except (RuntimeError, KeyboardInterrupt) as error:
        print(f"Cannot select a serial port: {error}", file=sys.stderr)
        return 2
    if port is None:
        return

    try:
        write_frame_to_port(port, frame, baudrate=args.baud)
    except OSError as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 2
    print(f"Frame sent to {port} ({len(frame)} bytes).")


if __name__ == "__main__":
    run_cli(main)
