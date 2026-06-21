import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

ASSETS_DIR = "/home/johnh/Documents/Notes/University/Elec/ELEC4951/Theory/assets"

# ---------------------------------------------------------
# Plot: Theoretical Predictions for Settling vs Collapse
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

time_steps = np.arange(0, 11)
o_star = 0.8 # target overlap
o_0 = 0.2    # initial overlap

# Scenario 1: Fast Settling (rho = 0.5)
overlap_fast = o_star - (o_star - o_0) * (0.5**time_steps)
# Scenario 2: Slow Settling (rho = 0.85)
overlap_slow = o_star - (o_star - o_0) * (0.85**time_steps)
# Scenario 3: Collapse / Divergence (rho = 1.2) - bounded at 0
overlap_collapse = o_star - (o_star - o_0) * (1.2**time_steps)
overlap_collapse = np.clip(overlap_collapse, 0.1, 1.0) # noise floor

ax.plot(time_steps, overlap_fast, 'o-', color='#2ca02c', linewidth=2, label=r'Fast Settling ($\rho = 0.5$)')
ax.plot(time_steps, overlap_slow, 's-', color='#1f77b4', linewidth=2, label=r'Slow Settling ($\rho = 0.85$)')
ax.plot(time_steps, overlap_collapse, 'X-', color='#d62728', linewidth=2, label=r'Collapse ($\rho > 1$)')

ax.axhline(o_star, color='k', linestyle='--', label=r'Target Overlap ($o^*$)')
ax.axhline(0.1, color='gray', linestyle=':', label='Noise Floor')

ax.set_xlabel(r'Internal Time Steps ($t$)')
ax.set_ylabel(r'Assembly Overlap ($o_y$)')
ax.set_title(r'Theoretical Predictions: Eigenvalue ($\rho$) Dynamics')
ax.legend()
ax.set_ylim(0, 1.0)
ax.set_xticks(time_steps)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'ch5_theoretical_settling.png'), dpi=300)
plt.close()

print("Prediction plot generated!")
