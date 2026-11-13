import numpy as np

# Emitter layout for a 16x16 grid (256 emitters).
# Coordinates are (x, y, z). Grid lies in the x-y plane with z=0 (height = 0).
# Origin (0,0,0) is at the LEFT-BOTTOM corner of the array.
# Spacing between neighboring centres is 10.5 mm (measured from the board's
# pick-place data: 10 mm transducer bodies with a 0.5 mm gap).
N_SIDE = 16
N_EMITTERS = N_SIDE * N_SIDE

SPACING = 0.0105  # meters (10.5 mm, measured)

# Build positions: rows (r) from 0..15 bottom->top (vertical axis),
# cols (c) from 0..15 left->right (horizontal axis).
# Logical index order used for EMITTERS_POS: index = r * N_SIDE + c
EMITTERS_POS = np.zeros((N_EMITTERS, 3), dtype=float)
for r in range(N_SIDE):
    for c in range(N_SIDE):
        idx = r * N_SIDE + c
        x = c * SPACING        # horizontal (column)
        y = r * SPACING        # vertical (row)
        z = 0.0                # height fixed at 0
        EMITTERS_POS[idx] = [x, y, z]

# Compute EMITTERS_ORDER according to the hardware mapping you described:
# - 256 transducers grouped into 32 shift-register groups of 8 bits each.
# - Groups are arranged in 4 rows x 8 columns. Group (gr, gc) uses a 4x2 block
#   of transducers: rows gr*4 .. gr*4+3 and cols gc*2 .. gc*2+1.
# - Group numbering starts at left-bottom (group 1) and increases left->right,
#   then bottom->top. So left-bottom is group 1, right-top is group 32.
# - Within each 4x2 group, the eight bit positions map to local coordinates as:
#   bit1 -> (r0,c0)
#   bit2 -> (r2,c0)
#   bit3 -> (r3,c0)
#   bit4 -> (r1,c1)
#   bit5 -> (r0,c1)
#   bit6 -> (r3,c1)
#   bit7 -> (r2,c1)
#   bit8 -> (r1,c0)
#   where local r counts from bottom (0) to top (3), c: 0=left,1=right.
# - The transmit order is: for group in groups (group1..group32 in the layout order),
#   send bit1..bit8 in that order.

EMITTERS_ORDER = []  # list of logical indices (length 256), EMITTERS_ORDER[k] = logical index sent at position k

# group rows: 0..3 bottom->top; group cols: 0..7 left->right
for gr in range(4):
    for gc in range(8):
        # for group number visualization: group_num = gc + gr*8 + 1
        base_row = gr * 4
        base_col = gc * 2
        # local mapping as specified
        local_positions = [
            (0, 0),  # bit1
            (2, 0),  # bit2
            (3, 0),  # bit3
            (1, 1),  # bit4
            (0, 1),  # bit5
            (3, 1),  # bit6
            (2, 1),  # bit7
            (1, 0),  # bit8
        ]
        for (lr, lc) in local_positions:
            r = base_row + lr
            c = base_col + lc
            if not (0 <= r < N_SIDE and 0 <= c < N_SIDE):
                raise ValueError(f"Computed out-of-range emitter at group {gr},{gc} local {lr},{lc}")
            logical_idx = r * N_SIDE + c
            EMITTERS_ORDER.append(int(logical_idx))

assert len(EMITTERS_ORDER) == N_EMITTERS


def get_position_by_logical_index(logical_idx):
    """Get (x, y, z) position for a given logical index (0..255)."""
    if not (0 <= logical_idx < N_EMITTERS):
        raise ValueError(f"Logical index must be 0..{N_EMITTERS - 1}, got {logical_idx}")
    return EMITTERS_POS[logical_idx]


def get_position_by_transmit_order(transmit_idx):
    """Get (x, y, z) position for a given transmit order position (0..255).

    transmit_idx = 0 is the first byte sent, transmit_idx = 1 is the second, etc.
    """
    if not (0 <= transmit_idx < N_EMITTERS):
        raise ValueError(f"Transmit index must be 0..{N_EMITTERS - 1}, got {transmit_idx}")
    logical_idx = EMITTERS_ORDER[transmit_idx]
    return EMITTERS_POS[logical_idx]


def get_grid_info_by_logical_index(logical_idx):
    """Get grid coordinates (row, col) and physical position for a logical index."""
    if not (0 <= logical_idx < N_EMITTERS):
        raise ValueError(f"Logical index must be 0..{N_EMITTERS - 1}, got {logical_idx}")
    row = logical_idx // N_SIDE
    col = logical_idx % N_SIDE
    pos = EMITTERS_POS[logical_idx]
    return {"row": row, "col": col, "x": pos[0], "y": pos[1], "z": pos[2]}
