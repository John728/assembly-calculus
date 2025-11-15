import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# -----------------------
# Global parameters
# -----------------------
GRID_ROWS = 5
GRID_COLS = 5
RADIUS = 0.35
SPACING = 2.0

P = 0.50      # probability of drawing an arrow from the chosen node
SEED = None      # set to None for non-deterministic arrows

N_PANELS = 3          # number of copies horizontally
PANEL_OFFSET = 12.0   # horizontal offset between panels
START_OFFSET = 3.0    # NEW: shift everything right so first panel isn't cut off

# For panels 1, 2, 3, show arrows from nodes 1, 2, 3 respectively (1-based)
ARROW_SOURCE_NODES = [0, 1, 2]  # 0 -> node 1, etc.

if SEED is not None:
    random.seed(SEED)

# -----------------------
# Base grid positions for one panel
# -----------------------
base_positions = {}
for idx in range(GRID_ROWS * GRID_COLS):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    x = col * SPACING
    y = -row * SPACING
    base_positions[idx] = (x, y)

# -----------------------
# Create figure
# -----------------------
fig_width = 6 + (N_PANELS - 1) * 4
fig, ax = plt.subplots(figsize=(fig_width, 6))

all_xs = []
all_ys = []

# -----------------------
# Helper: draw one panel
# -----------------------
def draw_panel(panel_idx, x_offset, source_node_idx):
    """Draw one grid panel at horizontal offset x_offset with arrows from source_node_idx."""
    panel_positions = {}
    for idx, (bx, by) in base_positions.items():
        x = bx + x_offset
        y = by
        panel_positions[idx] = (x, y)
        all_xs.append(x)
        all_ys.append(y)

    # Circles & labels
    for idx, (x, y) in panel_positions.items():
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

    # Arrows from selected node to nodes 1..23
    sx, sy = panel_positions[source_node_idx]
    arrow_color = "#CC0000"

    for target in range(23):
        if target == source_node_idx:
            continue

        if random.random() < P:
            tx, ty = panel_positions[target]
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
                linewidth=1.4,
                mutation_scale=12,
                color=arrow_color,
            )
            ax.add_patch(arrow)

    # Rounded border
    xs = [x for (x, y) in panel_positions.values()]
    ys = [y for (x, y) in panel_positions.values()]

    margin = RADIUS * 2
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
    width, height = xmax - xmin, ymax - ymin

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

    return xmin, xmax, ymin, ymax


# -----------------------
# Draw all panels
# -----------------------
panel_bounds = []
for panel_idx in range(N_PANELS):
    # NOTE: we now add START_OFFSET so the first panel isn't flush with the left
    x_offset = START_OFFSET + panel_idx * PANEL_OFFSET
    source_node_idx = ARROW_SOURCE_NODES[panel_idx]
    bounds = draw_panel(panel_idx, x_offset, source_node_idx)
    panel_bounds.append(bounds)

# -----------------------
# Ellipsis at the end
# -----------------------
_, last_xmax, last_ymin, last_ymax = panel_bounds[-1]
ellipsis_x = last_xmax + 2.5
ellipsis_y = 0.5 * (last_ymin + last_ymax)

ax.text(
    ellipsis_x,
    ellipsis_y,
    "⋯",
    fontsize=28,
    ha="center",
    va="center",
)

all_xs.append(ellipsis_x)
all_ys.append(ellipsis_y)

# -----------------------
# Figure aesthetics
# -----------------------
# Slightly more generous padding here too
xmin_global = min(all_xs) - RADIUS * 3
xmax_global = max(all_xs) + RADIUS * 3
ymin_global = min(all_ys) - RADIUS * 3
ymax_global = max(all_ys) + RADIUS * 3

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlim(xmin_global, xmax_global)
ax.set_ylim(ymin_global, ymax_global)
ax.axis("off")

plt.tight_layout()
# plt.savefig("nodes_three_panels.pdf")
# plt.savefig("nodes_three_panels.svg")
plt.savefig("nodes_three_panels.png", dpi=300)

plt.show()
