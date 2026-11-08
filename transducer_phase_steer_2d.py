#!/usr/bin/env python3
"""Steer a planar 16x16 ultrasonic array with azimuth and off-axis angles.

Azimuth is measured in the x-y plane from +x toward +y. Off-axis angle is
measured from +z. The phase sign assumes an exp(i*omega*t - i*k*r)
propagation convention. The firmware's phase-bin mapping and the physical
beam direction must be independently verified.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    demo_dir = str(Path(__file__).resolve().parent)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

from acoustics import (  # noqa: E402
    acoustic_wavelength,
    array_center,
    direction_from_angles,
    legacy_angle_to_spherical,
    steering_phase_bins,
    validate_acoustic_params,
    validate_steering_angles,
)
from array_factor import (  # noqa: E402
    plot_array_factor,
    plot_rotation_phase_samples,
    print_physical_phase_map,
)
from cli_utils import add_acoustic_args  # noqa: E402
from protocol import BAUD_RATE, build_frame, write_frame_to_port  # noqa: E402
from serial_utils import print_available_ports, select_serial_port  # noqa: E402
from trajectory import calculate_rotation_timing, run_rotation  # noqa: E402


def prompt_rotation_choice() -> bool:
    """Ask whether to enter rotation mode when no mode was selected."""
    while True:
        answer = input("Enable directional rotation? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer y or n.")


def prompt_rotation_mode() -> str:
    """Ask whether the rotation should use two or four fixed directions."""
    while True:
        answer = input(
            "Select rotation mode: two (0/180) or four (0/90/180/270) [four]: "
        ).strip().lower()
        if answer in ("", "4", "four"):
            return "four"
        if answer in ("2", "two"):
            return "two"
        print("Please enter two or four.")


def _select_port(port, dry_run):
    if dry_run:
        return port
    if port:
        return port
    try:
        return select_serial_port()
    except (RuntimeError, KeyboardInterrupt) as error:
        print(f"Cannot select a serial port: {error}")
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Steer or cycle a planar 16x16 array through two or four "
            "directions at a selected beta."
        )
    )
    parser.add_argument("--port", type=str, help="Serial port, e.g. COM3")
    parser.add_argument(
        "--list-ports", action="store_true", help="List serial ports and exit"
    )
    angle_group = parser.add_mutually_exclusive_group()
    angle_group.add_argument(
        "--angle",
        "--legacy-angle",
        dest="legacy_angle",
        type=float,
        help="Legacy x-z steering angle in degrees (0..180)",
    )
    angle_group.add_argument(
        "--alpha", type=float, help="Azimuth in degrees (-360..360)"
    )
    parser.add_argument(
        "--beta", type=float, help="Off-axis angle from +z in degrees (0..180)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute frame without transmission"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="Cycle azimuth through fixed directions"
    )
    parser.add_argument(
        "--mode",
        choices=("two", "four"),
        help="Rotation mode: two uses 0/180; four uses 0/90/180/270",
    )
    parser.add_argument(
        "--rotation-period",
        type=float,
        default=10.0,
        help="Requested revolution period in seconds",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.10,
        help="Minimum requested dwell time per angle in seconds",
    )
    parser.add_argument(
        "--duration", type=float, help="Stop after this many seconds; omit for Ctrl+C"
    )
    parser.add_argument(
        "--plot-phase-rotation",
        action="store_true",
        help="Plot phase maps at alpha 0, 90, 180, and 270 degrees",
    )
    parser.add_argument(
        "--plot-array-factor",
        action="store_true",
        help="Plot the idealized theoretical far-field array factor",
    )
    add_acoustic_args(parser)
    args = parser.parse_args(argv)

    if args.list_ports:
        print_available_ports()
        return

    if not args.rotate and args.alpha is None and args.legacy_angle is None:
        try:
            args.rotate = prompt_rotation_choice()
            if args.rotate and args.mode is None:
                args.mode = prompt_rotation_mode()
            if args.rotate and args.beta is None:
                args.beta = float(
                    input("Select fixed beta in degrees (0..180): ").strip()
                )
            elif not args.rotate and args.beta is not None:
                args.alpha = float(
                    input(
                        "Select static azimuth alpha in degrees (-360..360): "
                    ).strip()
                )
        except (ValueError, KeyboardInterrupt) as error:
            print(f"Invalid steering selection: {error}")
            return

    try:
        validate_acoustic_params(args.frequency, args.sound_speed)
    except ValueError as error:
        parser.error(str(error))

    if args.rotate:
        if args.alpha is not None or args.legacy_angle is not None:
            parser.error(
                "--rotate sweeps alpha automatically; do not specify --alpha or --angle."
            )
        if args.beta is None:
            parser.error("--rotate requires an explicitly selected --beta.")
        mode = args.mode or prompt_rotation_mode()
        beta_degrees = args.beta

        try:
            validate_steering_angles(0.0, beta_degrees)
        except ValueError as error:
            parser.error(str(error))
        if beta_degrees == 0.0:
            parser.error(
                "Rotating azimuth is undefined at beta=0; choose beta > 0 degrees."
            )

        if args.plot_phase_rotation:
            try:
                plot_rotation_phase_samples(
                    beta_degrees,
                    mode,
                    frequency_hz=args.frequency,
                    sound_speed=args.sound_speed,
                )
            except RuntimeError as error:
                parser.error(str(error))
        if args.plot_array_factor:
            try:
                initial_bins = steering_phase_bins(
                    0.0,
                    beta_degrees,
                    frequency_hz=args.frequency,
                    sound_speed=args.sound_speed,
                )
                plot_array_factor(
                    initial_bins,
                    0.0,
                    beta_degrees,
                    frequency_hz=args.frequency,
                    sound_speed=args.sound_speed,
                )
            except RuntimeError as error:
                parser.error(str(error))

        port = _select_port(args.port, args.dry_run)
        if not args.dry_run and port is None:
            return

        dry_run_duration = args.duration
        if args.dry_run and dry_run_duration is None:
            dry_run_duration = calculate_rotation_timing(
                args.rotation_period,
                args.update_interval,
                mode=mode,
            )["effective_rotation_period_s"]
            print(
                "Dry-run defaults to one estimated revolution; use --duration to change it."
            )
        try:
            run_rotation(
                beta_degrees,
                rotation_period_s=args.rotation_period,
                update_interval_s=args.update_interval,
                mode=mode,
                port=port,
                dry_run=args.dry_run,
                duration_s=dry_run_duration,
                frequency_hz=args.frequency,
                sound_speed=args.sound_speed,
            )
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Rotation failed: {error}")
        return

    if args.plot_phase_rotation:
        parser.error("--plot-phase-rotation requires --rotate.")
    if args.legacy_angle is not None and args.beta is not None:
        parser.error("--beta cannot be combined with the legacy --angle option.")
    if args.alpha is not None and args.beta is None:
        parser.error("--beta is required when --alpha is provided.")
    if args.alpha is None and args.beta is not None and args.legacy_angle is None:
        parser.error("--alpha is required when --beta is provided.")

    port = _select_port(args.port, args.dry_run)
    if not args.dry_run and port is None:
        return

    try:
        if args.legacy_angle is not None:
            target_azimuth, target_off_axis = legacy_angle_to_spherical(
                args.legacy_angle
            )
        elif args.alpha is not None:
            target_azimuth, target_off_axis = args.alpha, args.beta
        else:
            target_azimuth = float(
                input("Azimuth alpha in degrees (-180..180): ").strip()
            )
            target_off_axis = float(
                input("Off-axis beta in degrees (0..180): ").strip()
            )
        validate_steering_angles(target_azimuth, target_off_axis)
    except (ValueError, KeyboardInterrupt) as error:
        print(f"Invalid steering angles: {error}")
        return

    try:
        phase_bins = steering_phase_bins(
            target_azimuth,
            target_off_axis,
            frequency_hz=args.frequency,
            sound_speed=args.sound_speed,
        )
        frame = build_frame(phase_bins)
    except ValueError as error:
        parser.error(str(error))

    center = array_center()
    wavelength = acoustic_wavelength(args.frequency, args.sound_speed)
    direction = direction_from_angles(target_azimuth, target_off_axis)

    print(f"Azimuth alpha: {target_azimuth:.2f} deg")
    print(f"Off-axis beta: {target_off_axis:.2f} deg")
    print(
        "Requested unit direction: "
        f"({direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f})"
    )
    print(f"Array center from emitter coordinates: {center.tolist()} m")
    print(f"Wavelength: {wavelength * 1000:.3f} mm")
    print_physical_phase_map(phase_bins)
    print(f"UART frame size: {len(frame)} bytes (header + 256 phases + footer)")
    print(
        "Phase-bin physical mapping and phase sign require hardware or "
        "measurement verification."
    )

    if args.plot_array_factor:
        try:
            plot_array_factor(
                phase_bins,
                target_azimuth,
                target_off_axis,
                frequency_hz=args.frequency,
                sound_speed=args.sound_speed,
            )
        except RuntimeError as error:
            parser.error(str(error))
    if args.dry_run:
        print("[DRY RUN] No serial data sent.")
        return

    print(f"Sending phase frame to {port}...")
    try:
        write_frame_to_port(port, frame, baudrate=BAUD_RATE)
    except OSError as error:
        print(f"Serial error: {error}")
        return
    print("Phase frame sent. No ON/OFF command or OFF frame was sent.")


if __name__ == "__main__":
    main()
