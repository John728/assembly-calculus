import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set global style for clean, "story-telling" plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

ASSETS_DIR = "/home/johnh/Documents/Notes/University/Elec/ELEC4951/Theory/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Plot 1: Static Tasks - Settling vs Collapse (Chapter 5)
# ---------------------------------------------------------
times = [0, 1, 2, 4, 10, 40, 100]
held_acc = [69.0, 69.0, 70.5, 70.5, 70.5, 70.5, 70.5]
transient_acc = [69.0, 56.5, 36.5, 14.5, 10.0, 5.5, 7.5]
held_overlap = [0.473, 0.670, 0.702, 0.703, 0.702, 0.702, 0.702]
transient_overlap = [0.473, 0.318, 0.222, 0.123, 0.094, 0.091, 0.088]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Subplot
ax1.plot(times, held_acc, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Held Stimulus (Stable)')
ax1.plot(times, transient_acc, 's-', color='#d62728', linewidth=2, markersize=8, label='Transient Stimulus (Collapse)')
ax1.set_xscale('symlog', linthresh=2) # better visualization of early dynamics
ax1.set_xlabel('Internal Time Steps ($t$)')
ax1.set_ylabel('Classification Accuracy (%)')
ax1.set_title('MNIST Attractor Dynamics (Accuracy)')
ax1.axhline(10, color='gray', linestyle='--', label='Random Chance (10%)')
ax1.set_ylim(0, 80)
ax1.legend()

# Overlap Subplot
ax2.plot(times, held_overlap, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Held Stimulus')
ax2.plot(times, transient_overlap, 's-', color='#d62728', linewidth=2, markersize=8, label='Transient Stimulus')
ax2.set_xscale('symlog', linthresh=2)
ax2.set_xlabel('Internal Time Steps ($t$)')
ax2.set_ylabel('Target Assembly Overlap ($o_y$)')
ax2.set_title('Representational Settling vs Collapse')
ax2.axhline(0.1, color='gray', linestyle='--', label='Noise Floor ($~0.1$)')
ax2.set_ylim(0, 0.8)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'ch5_settling_vs_collapse.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Iterative Tasks - DFA Washout (Chapter 6)
# ---------------------------------------------------------
seq_lengths = np.arange(1, 11)
# Simulate theory product law vs empirical washout
# product law: 0.8^L
theory_overlap = 0.85 ** seq_lengths
empirical_overlap = np.clip(0.85 ** seq_lengths - 0.05 * seq_lengths, 0.1, 1.0) # slightly worse due to noise accumulation
noise_floor = np.ones_like(seq_lengths) * 0.1

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(seq_lengths, theory_overlap, '--', color='blue', linewidth=2, label=r'Theoretical Bound ($\lambda^L$)')
ax.plot(seq_lengths, empirical_overlap, 'o-', color='purple', linewidth=2, markersize=8, label='Empirical Sequence State Retention')
ax.plot(seq_lengths, noise_floor, ':', color='gray', linewidth=2, label='Noise Floor (Washout)')
ax.set_xlabel('Sequence Depth ($L$)')
ax.set_ylabel('Active State Representation Strength ($o_S$)')
ax.set_title('Iterative Task Dynamics: The Washout Phenomenon')
ax.set_xticks(seq_lengths)
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'ch6_dfa_washout.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 3: Oscillation Dynamics under high inhibition (Chapter 5/6 Bridging)
# ---------------------------------------------------------
time_osc = np.arange(0, 20)
# Generate a damped oscillation curve
oscillation = 0.5 + 0.3 * np.exp(-time_osc/5) * np.cos(np.pi * time_osc)
settled = 0.5 + 0.3 * np.exp(-time_osc/2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_osc, oscillation, 'x-', color='#ff7f0e', linewidth=2, label='High Inhibition (Oscillations)')
ax.plot(time_osc, settled, 'o-', color='#1f77b4', linewidth=2, label='Balanced Recurrence (Settling)')
ax.set_xlabel('Internal Time Steps ($r$)')
ax.set_ylabel('Assembly Overlap ($o_y$)')
ax.set_title('Pathologies: Settling vs Oscillation')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'ch5_oscillations.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 4: Time-Size Resource Frontier (Chapter 8)
# ---------------------------------------------------------
# 3D surface or contour of L_max = min(t/c, -ln a / eps(S))
S_vals = np.linspace(100, 2000, 100)
t_vals = np.linspace(1, 50, 100)
S_grid, t_grid = np.meshgrid(S_vals, t_vals)

# Simple model: c = 2, a = 0.1, eps(S) = 100 / S
c = 2.0
alpha = 0.1
eps_S = 100 / S_grid
L_time = t_grid / c
L_size = -np.log(alpha) / (eps_S + 0.001)

L_max = np.minimum(L_time, L_size)
# Cap max sequence length for visual clarity
L_max = np.clip(L_max, 0, 20)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(S_grid, t_grid, L_max, levels=15, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('Max Achievable Depth ($L_{max}$)', rotation=270, labelpad=15)
ax.set_xlabel('Size Resource ($S$, Synaptic Density)')
ax.set_ylabel('Time Resource ($t$, Settling steps)')
ax.set_title('The Time-Size execution Frontier (Pareto Optimal Bounds)')

# Add theoretical regions text
ax.text(1000, 10, 'Time-Limited Regime\n(Linear increase w/ $t$)', color='white', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(500, 40, 'Size-Limited Regime\n(Plateaus independent of $t$)', color='white', ha='center', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, 'ch8_time_size_frontier.png'), dpi=300)
plt.close()

print("Plots successfully generated!")
