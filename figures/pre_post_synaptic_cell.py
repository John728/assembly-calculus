import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# -----------------------
# Parameters
# -----------------------
RADIUS = 0.35
PRE_X = 0.0
POST_X = 3.0
POST_Y_OFFSETS = [-1.0, 0.0, 1.0]  # three postsynaptic cells
TIME_STEPS = ["t₀", "t₁", "t₂", "t₃"]

# Line widths for the Hebbian (middle) synapse over time
hebb_widths = [1.2, 2.0, 3.0, 4.0]
other_width = 1.0

# Colors
active_fill = "#FFCC66"   # light yellow/orange for active cells
inactive_fill = "white"
outline = "black"
hebb_color = "#CC0000"    # red for the strengthening synapse
other_color = "#555555"   # grey for non-strengthening ones

fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

for idx, ax in enumerate(axes):
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------
    # Neuron positions
    # -----------------------
    pre_pos = (PRE_X, 0.0)
    post_positions = [(POST_X, y) for y in POST_Y_OFFSETS]

    # -----------------------
    # Determine activity pattern
    # Here: pre + middle post are active at all time steps
    # (you can tweak this if you want changing activity)
    # -----------------------
    pre_active = True
    # index 1 is the "Hebbian" postsynaptic neuron
    post_active_flags = [False, True, False]

    # -----------------------
    # Draw pre neuron
    # -----------------------
    pre_circle = Circle(
        pre_pos,
        RADIUS,
        edgecolor=outline,
        facecolor=active_fill if pre_active else inactive_fill,
        linewidth=2,
    )
    ax.add_patch(pre_circle)
    ax.text(
        pre_pos[0],
        pre_pos[1] + 1.2,
        "Presynaptic",
        ha="center",
        va="center",
        fontsize=9,
    )

    # -----------------------
    # Draw post neurons
    # -----------------------
    for (x, y), active in zip(post_positions, post_active_flags):
        c = Circle(
            (x, y),
            RADIUS,
            edgecolor=outline,
            facecolor=active_fill if active else inactive_fill,
            linewidth=2,
        )
        ax.add_patch(c)

    ax.text(
        POST_X,
        POST_Y_OFFSETS[0] - 1.2,
        "Postsynaptic cells",
        ha="center",
        va="center",
        fontsize=9,
    )

    # -----------------------
    # Draw synapses
    # -----------------------
    for j, (x_post, y_post) in enumerate(post_positions):
        # Choose style: middle synapse is Hebbian
        if j == 1:
            lw = hebb_widths[idx]
            color = hebb_color
        else:
            lw = other_width
            color = other_color

        arrow = FancyArrowPatch(
            (pre_pos[0] + RADIUS * 1.05, pre_pos[1]),   # start just outside pre cell
            (x_post - RADIUS * 1.05, y_post),           # end just before post cell
            arrowstyle="-|>",
            linewidth=lw,
            mutation_scale=12,
            color=color,
        )
        ax.add_patch(arrow)

    # -----------------------
    # Time label under panel
    # -----------------------
    ax.text(
        (PRE_X + POST_X) / 2,
        POST_Y_OFFSETS[-1] + 1.4,
        f"Time: {TIME_STEPS[idx]}",
        ha="center",
        va="center",
        fontsize=11,
    )

# Adjust global layout
plt.tight_layout()
# plt.savefig("hebbian_over_time.pdf")
# plt.savefig("hebbian_over_time.svg")
plt.savefig("hebbian_over_time.png", dpi=300)

plt.show()
