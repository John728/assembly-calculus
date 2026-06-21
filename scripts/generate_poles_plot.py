import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

ASSETS_DIR = "/home/johnh/Documents/Notes/University/Elec/ELEC4951/Theory/assets"

# ---------------------------------------------------------
# Plot: Jacobian Poles and Oscillations (Chapter 6)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left Subplot: Complex Plane with Poles
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle ($|\\rho|=1$)')
ax1.axhline(0, color='gray', linewidth=1)
ax1.axvline(0, color='gray', linewidth=1)

# Stable poles (e.g. 0.8)
ax1.plot([0.8], [0], 'o', color='#1f77b4', markersize=10, label='Stable Settling ($\\rho = 0.8$)')
# Oscillating unstable pole (e.g. -1.1)
ax1.plot([-1.1], [0], 'X', color='#ff7f0e', markersize=12, label='Oscillatory Pathology ($\\rho = -1.1$)')

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.set_title('Jacobian Eigenvalues (Poles) in Complex Plane')
ax1.legend(loc='upper right')

# Right Subplot: Time Domain Simulation
time_steps = np.arange(0, 15)
stable_resp = 0.8**time_steps * 0.5 + 0.5
osc_resp = (-1.1)**time_steps * 0.1 + 0.5
# clip oscillation for realism
osc_resp = np.clip(osc_resp, 0, 1)

ax2.plot(time_steps, stable_resp, 'o-', color='#1f77b4', linewidth=2, label='Stable Settling')
ax2.plot(time_steps, osc_resp, 'X-', color='#ff7f0e', linewidth=2, label='Limit Cycle Oscillation')
ax2.set_xlabel('Internal Time Steps ($r$)')
ax2.set_ylabel('Assembly Overlap ($o_y$)')
ax2.set_title('Resulting Temporal Dynamics')
ax2.legend()
ax2.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'jacobian_poles_oscillation.png'), dpi=300)
plt.close()

print("Pole plot generated!")
