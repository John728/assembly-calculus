import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# -----------------------
# Global parameters
# -----------------------
GRID_ROWS = 5
GRID_COLS = 5

# Visual sizing
CIRCLE_DIAMETER_PX = 85  # circles drawn with 25 px diameter
LABEL_FONT_SIZE = 5
ARROW_COLOR = "#DB5050"
ARROW_LINEWIDTH = 1
ARROW_HEAD_SCALE = 5
GLOBAL_MARGIN_FACTOR = 3  # how many radii to pad axes by

SPACING = 2.0

P = 0.30      # probability of drawing an arrow from the chosen node
SEED = 1      # set to None for non-deterministic arrows

N_PANELS = 3          # number of copies horizontally
PANEL_OFFSET = 18.0   # horizontal offset between panels (default gap)
FIRST_PANEL_GAP = 12.0  # tighter spacing between panels 1 and 2
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
fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
# Fix axes position so we can size elements predictably
ax.set_position([0.02, 0.1, 0.96, 0.8])

# Compute desired radius in data units so circles render at CIRCLE_DIAMETER_PX.
axes_bbox = ax.get_position()
fig_width_px = fig.get_size_inches()[0] * fig.dpi
axis_width_px = fig_width_px * axes_bbox.width

# base range ignoring margins
all_panel_xs = []
all_panel_ys = []

# Pre-compute custom offsets so the first gap is reduced while others stay the same.
panel_offsets = []
current_offset = START_OFFSET
for panel_idx in range(N_PANELS):
    panel_offsets.append(current_offset)
    gap = FIRST_PANEL_GAP if panel_idx == 0 else PANEL_OFFSET
    current_offset += gap

for panel_idx in range(N_PANELS):
    x_offset = panel_offsets[panel_idx]
    for idx, (bx, by) in base_positions.items():
        x = bx + x_offset
        y = by
        all_panel_xs.append(x)
        all_panel_ys.append(y)

base_range_x = max(all_panel_xs) - min(all_panel_xs)

desired_diameter_px = CIRCLE_DIAMETER_PX
RADIUS = (desired_diameter_px * base_range_x) / (
    2 * axis_width_px - 6 * desired_diameter_px
)

all_xs = []
all_ys = []

# -----------------------
# Helper: draw one panel
# -----------------------
def draw_panel(panel_idx, x_offset, source_node_idx, fully_connected=False):
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

        ax.text(x, y, label, ha="center", va="center", fontsize=3)

    active_nodes = list(range(23))
    sources = active_nodes if fully_connected else [source_node_idx]

    for src_idx in sources:
        sx, sy = panel_positions[src_idx]

        for target in active_nodes:
            if target == src_idx:
                continue

            if fully_connected or random.random() < P:
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
                    linewidth=ARROW_LINEWIDTH,
                    mutation_scale=ARROW_HEAD_SCALE,
                    color=ARROW_COLOR,
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
        linewidth=0.8,
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
    x_offset = panel_offsets[panel_idx]
    source_node_idx = ARROW_SOURCE_NODES[panel_idx]
    is_last = panel_idx == N_PANELS - 1
    bounds = draw_panel(
        panel_idx,
        x_offset,
        source_node_idx,
        fully_connected=is_last,
    )
    panel_bounds.append(bounds)

# -----------------------
# Labels beneath panels
# -----------------------
panel_text = [
    "Connections for neuron 0",
    "Connections for neuron 1",
    "Showing fully connected network",
]

for (xmin, xmax, ymin, ymax), label in zip(panel_bounds, panel_text):
    text_x = 0.5 * (xmin + xmax)
    text_y = ymin - RADIUS * 2
    ax.text(text_x, text_y, label, ha="center", va="center", fontsize=LABEL_FONT_SIZE)
    all_xs.append(text_x)
    all_ys.append(text_y)

# -----------------------
# Symbols between panels
# -----------------------
if len(panel_bounds) >= 2:
    first_bounds = panel_bounds[0]
    second_bounds = panel_bounds[1]
    plus_x = 0.5 * (first_bounds[1] + second_bounds[0])
    plus_y = 0.5 * (first_bounds[2] + first_bounds[3])
    ax.text(plus_x, plus_y, "+", ha="center", va="center", fontsize=20)
    all_xs.append(plus_x)
    all_ys.append(plus_y)

if len(panel_bounds) >= 3:
    second_bounds = panel_bounds[1]
    third_bounds = panel_bounds[2]
    plus_eq_x = 0.5 * (second_bounds[1] + third_bounds[0])
    plus_eq_y = 0.5 * (second_bounds[2] + second_bounds[3])
    ax.text(plus_eq_x, plus_eq_y, "+ ... =", ha="center", va="center", fontsize=20)
    all_xs.append(plus_eq_x)
    all_ys.append(plus_eq_y)

# -----------------------
# Figure aesthetics
# -----------------------
# Slightly more generous padding here too
xmin_global = min(all_xs) - RADIUS * GLOBAL_MARGIN_FACTOR
xmax_global = max(all_xs) + RADIUS * GLOBAL_MARGIN_FACTOR
ymin_global = min(all_ys) - RADIUS * GLOBAL_MARGIN_FACTOR
ymax_global = max(all_ys) + RADIUS * GLOBAL_MARGIN_FACTOR

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlim(xmin_global, xmax_global)
ax.set_ylim(ymin_global, ymax_global)
ax.axis("off")

# plt.savefig("nodes_three_panels.pdf")
# plt.savefig("nodes_three_panels.svg")
plt.savefig("nodes_three_panels.png", dpi=300)

plt.show()
