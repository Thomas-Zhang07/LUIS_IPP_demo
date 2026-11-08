# Demo �� Single-Point Acoustic Focus

Single-point acoustic focus demo using 16-level phase control.

## Usage

### Interactive Mode
```bash
python single_point_hover.py
```
Select serial port, then enter target position (x, y, z in meters), or press Enter for default.

### Command-Line Mode
```bash
python single_point_hover.py --x 0.075 --y 0.1 --z 0.05 --port COM3
```
Specify position and port directly (non-interactive).

### Dry-Run (No Hardware)
```bash
python single_point_hover.py --dry-run
```
Compute and display the 256 phase values as a 16x16 top-view physical layout without sending.

## Arguments

| Argument | Description |
|----------|-------------|
| `--x FLOAT` | Target X coordinate (meters). |
| `--y FLOAT` | Target Y coordinate (meters). |
| `--z FLOAT` | Target Z coordinate (meters). |
| `--port STRING` | Serial port (e.g., COM3). If not set, prompts user. |
| `--dry-run` | Compute but do NOT send frame. |
| `--baud INT` | Baud rate (default 38400). |
| `--help` | Show help message. |

## Example

```bash
# Focus at (7.5 cm, 10 cm, 5 cm)
python single_point_hover.py --x 0.075 --y 0.1 --z 0.05 --port COM3
```

## Single-Point Vortex

Use `single_point_vortex.py` to add a rotational phase around the array center.
The default target is `(0.075, 0.075, 0.05)` meters, and the default topological
charge is `+1`.

```bash
# Send a positive-charge vortex at the array center, 5 cm above it
python single_point_vortex.py --x 0.075 --y 0.075 --z 0.05 --charge 1 --port COM3

# Reverse the vortex rotation direction
python single_point_vortex.py --x 0.075 --y 0.075 --z 0.05 --charge -1 --port COM3

# Compute and print the 16x16 phase map without sending
python single_point_vortex.py --dry-run
```

The vortex phase is calculated as:

```text
total_phase = focus_phase + topological_charge * polar_angle
```

The polar angle is calculated around the mean position of all 256 emitters.
In interactive mode, after entering the target coordinates, enter `ccw` or
`��ʱ��` for positive topological charge, or `cw` or `˳ʱ��` for negative
topological charge. Press Enter to keep the default direction.

## Circular Focus

Use `single_point_circle.py` to continuously move a single acoustic focus
counterclockwise around a circle in the XY plane. Enter the circle center; the
radius is fixed at 3 cm and the center's Z coordinate is kept constant. Press
Ctrl+C to stop; the script then sends an all-off frame.

```bash
# Prompt for the center and serial port
python single_point_circle.py

# Circle around (7.5 cm, 7.5 cm, 5 cm) using COM3
python single_point_circle.py --x 0.075 --y 0.075 --z 0.05 --port COM3

# One trajectory preview without hardware
python single_point_circle.py --x 0.075 --y 0.075 --z 0.05 --dry-run
```

By default, each circle takes 5 seconds and uses 48 focus positions. Adjust
these with `--period` and `--points` if needed.

Output:
```
Target position: (0.0750, 0.1000, 0.0500) m
Phase discretization: 0..15 levels

16x16 phase map (top view, y increases upward):
	x=00 x=10 x=20 ... x=150
	-----------------------------------------------
y=150 | 05 06 06 ... 05
 ...
y=00  | 05 06 06 ... 05

UART Frame (hex): fe 05 06 06 05 05 05 06 05 06 05 06 06 05 06 05 06 ... fd
Frame size: 258 bytes (header + 256 body + footer)

Frame sent to COM3
```
