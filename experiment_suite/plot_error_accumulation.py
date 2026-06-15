import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Paths
CODE_OUT_DIR = Path("Theory/assets")
CODE_OUT_DIR.mkdir(parents=True, exist_ok=True)

NOTES_OUT_DIR = Path("/home/johnh/Documents/Notes/University/Elec/ELEC4951/Theory/assets")
NOTES_OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = Path("results/pointer/theory/raw_results.csv")
if not csv_path.exists():
    print("Pointer theory CSV not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Add path_correct column: 1 if first_error_index is NaN (meaning no errors), 0 otherwise
df["path_correct"] = df["first_error_index"].isna().astype(int)

# Filter for c = 1 diagonal: t = L
df_c1 = df[(df["c"] == 1) & (df["t"] == df["L"])].copy()

# Unique L values (1, 2, 3)
L_values = sorted(df_c1["L"].unique())

empirical_accs = []
n_trials_list = []
ci_lows = []
ci_highs = []

def wilson_ci(count, nobs, alpha=0.05):
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = 1.96  # For 95% CI
    denom = 1 + z**2 / nobs
    center = (p + z**2 / (2 * nobs)) / denom
    spread = z * np.sqrt(p * (1 - p) / nobs + z**2 / (4 * nobs**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

for L in L_values:
    sub = df_c1[df_c1["L"] == L]
    n_trials = len(sub)
    n_correct = int(sub["path_correct"].sum())
    emp_acc = sub["path_correct"].mean()
    
    ci_low, ci_high = wilson_ci(n_correct, n_trials)
    
    empirical_accs.append(emp_acc)
    n_trials_list.append(n_trials)
    ci_lows.append(ci_low)
    ci_highs.append(ci_high)

# Fit epsilon_hat at L = 1
emp_acc_L1 = empirical_accs[L_values.index(1)]
epsilon_hat = 1.0 - emp_acc_L1

# Compute predictions
pred_accs = [(1.0 - epsilon_hat)**L for L in L_values]

# Create CSV summary
summary_df = pd.DataFrame({
    "L": L_values,
    "n_trials": n_trials_list,
    "empirical_acc": empirical_accs,
    "ci_low": ci_lows,
    "ci_high": ci_highs,
    "pred_acc": pred_accs,
    "epsilon_hat": [epsilon_hat] * len(L_values),
    "c": [1] * len(L_values)
})

# Save CSV
summary_df.to_csv(CODE_OUT_DIR / "pointer_error_accumulation.csv", index=False)
summary_df.to_csv(NOTES_OUT_DIR / "pointer_error_accumulation.csv", index=False)

# Compute fit errors
mae = np.mean(np.abs(np.array(empirical_accs) - np.array(pred_accs)))
rmse = np.sqrt(np.mean((np.array(empirical_accs) - np.array(pred_accs))**2))

print(f"Fitted epsilon_hat (at L=1): {epsilon_hat:.6f}")
print(f"Fit MAE: {mae:.6f}")
print(f"Fit RMSE: {rmse:.6f}")

# Compute j-hop conditional error rates (Nice extra)
# We look at L=3 trajectories to see the conditional error rates step-by-step
df_L3 = df[(df["c"] == 1) & (df["t"] == 3) & (df["L"] == 3)]
tot_L3 = len(df_L3)
err_counts = df_L3["first_error_index"].value_counts()
err_1 = int(err_counts.get(1.0, 0))
err_2 = int(err_counts.get(2.0, 0))
err_3 = int(err_counts.get(3.0, 0))

eps_1 = err_1 / tot_L3
eps_2 = err_2 / (tot_L3 - err_1) if (tot_L3 - err_1) > 0 else 0
eps_3 = err_3 / (tot_L3 - err_1 - err_2) if (tot_L3 - err_1 - err_2) > 0 else 0

print(f"Per-hop conditional error rates (from L=3 trials):")
print(f"  epsilon_1 (hop 1): {eps_1:.6f}")
print(f"  epsilon_2 (hop 2): {eps_2:.6f}")
print(f"  epsilon_3 (hop 3): {eps_3:.6f}")

# Generate Plot (16:9 ratio)
plt.figure(figsize=(10.67, 6))

# Smooth curve for predicted accuracy
L_smooth = np.linspace(1, 3, 100)
pred_smooth = (1.0 - epsilon_hat)**L_smooth
plt.plot(L_smooth, pred_smooth, "-", color="red", linewidth=2.5, 
         label=f"Prediction: $(1 - \\hat{{\\epsilon}})^L$ ($\\hat{{\\epsilon}}$ = {epsilon_hat:.3f})")

# Error bars for empirical accuracy
# yerr needs to be a 2xN array of lower and upper limits relative to the mean value
yerr_low = np.array(empirical_accs) - np.array(ci_lows)
yerr_high = np.array(ci_highs) - np.array(empirical_accs)
yerr = np.vstack([yerr_low, yerr_high])

plt.errorbar(L_values, empirical_accs, yerr=yerr, fmt="o", color="blue", 
             ecolor="blue", elinewidth=2, capsize=6, markersize=10, 
             label="Empirical path accuracy (mean $\\pm$ 95% CI)")

plt.xlabel("Pointer-chain depth L")
plt.ylabel("Path accuracy")
plt.title("Pointer Chasing: Empirical vs Predicted Path Accuracy")
plt.xticks(L_values)
plt.ylim(-0.05, 1.05)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save PNG and SVG
plt.savefig(CODE_OUT_DIR / "pointer_error_accumulation.png", dpi=200)
plt.savefig(CODE_OUT_DIR / "pointer_error_accumulation.svg", format="svg")
plt.savefig(NOTES_OUT_DIR / "pointer_error_accumulation.png", dpi=200)
plt.savefig(NOTES_OUT_DIR / "pointer_error_accumulation.svg", format="svg")
plt.close()

print("Plots generated and saved successfully!")
