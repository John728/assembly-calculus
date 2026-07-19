from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

depth = np.linspace(0, 10, 300)
reach = depth
hold = depth + 2.5

fig, axis = plt.subplots(figsize=(5.8, 3.15))
axis.fill_between(depth, 0, reach, color="#D9D9D9")
axis.fill_between(depth, reach, hold, color="#56B4E9", alpha=0.55)
axis.fill_between(depth, hold, 13, color="#E69F00", alpha=0.30)
axis.plot(depth, reach, color="#0072B2", lw=1.8)
axis.plot(depth, hold, color="#D55E00", lw=1.4, ls="--")
axis.text(7.4, 3.1, "Unreachable", color="0.25", ha="center")
axis.text(7.4, 8.5, "Reached and readable", color="#005B96", ha="center")
axis.text(7.4, 11.5, "Beyond holding window", color="#8A5200", ha="center")
axis.text(3.0, 3.35, "$t=\\kappa L$", color="#005B96", rotation=39, ha="center")
axis.set(xlim=(0, 10), ylim=(0, 13), xlabel="Logical depth $L$", ylabel="Readout update $t$")
fig.tight_layout()

output = Path(__file__).resolve().parent
fig.savefig(output / "reach_survive_hold.pdf", bbox_inches="tight")
fig.savefig(output / "reach_survive_hold.png", bbox_inches="tight")
