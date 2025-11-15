import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# -----------------------
# Parameters
# -----------------------
GRID_ROWS = 5
GRID_COLS = 5
RADIUS = 0.35
SPACING = 2.0

P = 0.01        # probability of arrow from any A-node to any B-node
SEED = None        # set to None for non-deterministic arrows

AREA_OFFSET = 12.0   # horizontal distance between Area A and Area B

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
            linewidth=2,
            edgecolor="black",
        )
        ax.add_patch(circle)

        if idx < 23:
            label = str(idx + 1)
        elif idx == 23:
            label = "..."
        else:
            label = "n"

        ax.text(x, y, label, ha="center", va="center", fontsize=10)
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
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(border)

    # Area label under the area
    label_y = ymin - 1.2  # a bit below the border
    label_x = 0.5 * (xmin + xmax)
    ax.text(
        label_x,
        label_y,
        area_name,
        ha="center",
        va="center",
        fontsize=12,
    )
    all_xs.append(label_x)
    all_ys.append(label_y)

    return xmin, xmax, ymin, ymax


# -----------------------
# Create figure and axes
# -----------------------
fig_width = 10
fig, ax = plt.subplots(figsize=(fig_width, 6))

# Draw both areas
bounds_A = draw_area(positions_A, AREA_A_NAME)
bounds_B = draw_area(positions_B, AREA_B_NAME)

# -----------------------
# Draw arrows from Area A to Area B
# -----------------------
arrow_color = "#CC0000"  # nice red

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
                linewidth=1.3,
                mutation_scale=12,
                color=arrow_color,
            )
            ax.add_patch(arrow)

# -----------------------
# Global aesthetics
# -----------------------
xmin_global = min(all_xs) - RADIUS * 3
xmax_global = max(all_xs) + RADIUS * 3
ymin_global = min(all_ys) - RADIUS * 3
ymax_global = max(all_ys) + RADIUS * 3

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlim(xmin_global, xmax_global)
ax.set_ylim(ymin_global, ymax_global)
ax.axis("off")

plt.tight_layout()
# plt.savefig("bipartite_two_areas.pdf")
# plt.savefig("bipartite_two_areas.svg")
plt.savefig("bipartite_two_areas.png", dpi=300)

plt.show()
