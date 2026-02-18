import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# -----------------------
# Parameters
# -----------------------
GRID_ROWS = 5
GRID_COLS = 5

# Visual sizing to match nodes_three_panels styling
CIRCLE_DIAMETER_PX = 95  # circles drawn with 25 px diameter
LABEL_FONT_SIZE = 3
ARROW_COLOR = "#DB5050"
ARROW_LINEWIDTH = 2
ARROW_HEAD_SCALE = 5
GLOBAL_MARGIN_FACTOR = 3  # how many radii to pad axes by

SPACING = 1

P = 0.01        # probability of arrow from any A-node to any B-node
SEED = 3        # set to None for non-deterministic arrows

AREA_OFFSET = 8.0   # horizontal distance between Area A and Area B

AREA_A_NAME = "Area A"
AREA_B_NAME = "Area B"

if SEED is not None:
    random.seed(SEED)

# -----------------------
# Base positions for a 5x5 grid
# -----------------------
base_positions = {}
for idx in range(GRID_ROWS * GRID_COLS):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    x = col * SPACING
    y = -row * SPACING
    base_positions[idx] = (x, y)

# -----------------------
# Shifted positions for each area
# -----------------------
positions_A = {}
positions_B = {}

# We'll centre Area A around x=0 by subtracting half the width,
# and then put Area B AREA_OFFSET units to the right of Area A.
# But easiest: take base as Area A, then Area B = base + AREA_OFFSET.
for idx, (bx, by) in base_positions.items():
    positions_A[idx] = (bx, by)
    positions_B[idx] = (bx + AREA_OFFSET, by)

# Collect all coordinates to set nice global limits later
all_xs = []
all_ys = []

# -----------------------
# Helper: draw one area (circles, labels, border, name)
# -----------------------
def draw_area(positions, area_name):
    # Draw circles and labels
    for idx, (x, y) in positions.items():
        circle = Circle(
            (x, y),
            RADIUS,
            fill=False,
            linewidth=0.8,
            edgecolor="black",
        )
        ax.add_patch(circle)

        if idx < 23:
            label = str(idx + 1)
        elif idx == 23:
            label = "..."
        else:
            label = "n"

        ax.text(x, y, label, ha="center", va="center", fontsize=LABEL_FONT_SIZE)
        all_xs.append(x)
        all_ys.append(y)

    # Compute bounds for border
    xs = [x for (x, y) in positions.values()]
    ys = [y for (x, y) in positions.values()]

    margin = RADIUS * 2
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
    width, height = xmax - xmin, ymax - ymin

    # Rounded border
    border = FancyBboxPatch(
        (xmin, ymin),
        width,
        height,
        boxstyle="round,pad=0.3,rounding_size=0.4",
        linewidth=0.8,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(border)

    # Area label under the area
    label_y = ymin - RADIUS * 3  # a bit below the border
    label_x = 0.5 * (xmin + xmax)
    ax.text(
        label_x,
        label_y,
        area_name,
        ha="center",
        va="center",
        fontsize=8,
    )
    all_xs.append(label_x)
    all_ys.append(label_y)

    return xmin, xmax, ymin, ymax


# -----------------------
# Create figure and axes
# -----------------------
fig_width = 8
fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=300)
# Fix axes position so we can size elements predictably
ax.set_position([0.02, 0.1, 0.96, 0.8])

# Compute desired radius in data units so circles render at CIRCLE_DIAMETER_PX.
axes_bbox = ax.get_position()
fig_width_px = fig.get_size_inches()[0] * fig.dpi
axis_width_px = fig_width_px * axes_bbox.width

all_positions_x = []
for pos in positions_A.values():
    all_positions_x.append(pos[0])
for pos in positions_B.values():
    all_positions_x.append(pos[0])

base_range_x = max(all_positions_x) - min(all_positions_x)

desired_diameter_px = CIRCLE_DIAMETER_PX
RADIUS = (desired_diameter_px * base_range_x) / (
    2 * axis_width_px - 6 * desired_diameter_px
)

# Draw both areas
bounds_A = draw_area(positions_A, AREA_A_NAME)
bounds_B = draw_area(positions_B, AREA_B_NAME)

# -----------------------
# Draw arrows from Area A to Area B
# -----------------------
for i in range(GRID_ROWS * GRID_COLS):      # nodes in Area A (0..24)
    sx, sy = positions_A[i]
    for j in range(GRID_ROWS * GRID_COLS):  # nodes in Area B (0..24)
        tx, ty = positions_B[j]

        if random.random() < P:
            dx = tx - sx
            dy = ty - sy
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            offset = RADIUS * 1.05
            start_x = sx + dx * offset / dist
            start_y = sy + dy * offset / dist
            end_x = tx - dx * offset / dist
            end_y = ty - dy * offset / dist

            arrow = FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="->",
                linewidth=ARROW_LINEWIDTH,
                mutation_scale=ARROW_HEAD_SCALE,
                color=ARROW_COLOR,
            )
            ax.add_patch(arrow)

# -----------------------
# Global aesthetics
# -----------------------
xmin_global = min(all_xs) - RADIUS * GLOBAL_MARGIN_FACTOR
xmax_global = max(all_xs) + RADIUS * GLOBAL_MARGIN_FACTOR
ymin_global = min(all_ys) - RADIUS * GLOBAL_MARGIN_FACTOR
ymax_global = max(all_ys) + RADIUS * GLOBAL_MARGIN_FACTOR

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlim(xmin_global, xmax_global)
ax.set_ylim(ymin_global, ymax_global)
ax.axis("off")

# plt.savefig("bipartite_two_areas.pdf")
# plt.savefig("bipartite_two_areas.svg")
plt.savefig("bipartite_two_areas.png", dpi=300)

plt.show()
