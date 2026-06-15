import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

OUT_DIR = Path("Theory/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = Path("results/mnist/probes/t100_held/raw_results.csv")
if not csv_path.exists():
    print("MNIST t100 held CSV not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Coerce columns
df["t"] = pd.to_numeric(df["t"])
df["correct"] = df["correct"].astype(float) # True/False -> 1/0
df["margin"] = pd.to_numeric(df["margin"])
df["correct_overlap"] = pd.to_numeric(df["correct_overlap"])
df["strongest_wrong_overlap"] = pd.to_numeric(df["strongest_wrong_overlap"])

# Filter for t in [0, 10]
df_zoom = df[df["t"] <= 10].copy()

# Aggregate functions
def _seed_aggregated_accuracy(sub_df):
    if "seed" not in sub_df.columns:
        return pd.DataFrame(sub_df.groupby("t", as_index=False)["correct"].agg(
            accuracy="mean", se=lambda x: x.std() / np.sqrt(len(x))
        ))
    per_seed = pd.DataFrame(sub_df.groupby(["seed", "t"], as_index=False)["correct"].mean())
    return pd.DataFrame(per_seed.groupby("t", as_index=False)["correct"].agg(
        accuracy="mean", se=lambda x: x.std() / np.sqrt(len(x))
    ))

def _seed_aggregated_margin(sub_df):
    if "seed" not in sub_df.columns:
        return pd.DataFrame(sub_df.groupby("t", as_index=False).agg(
            margin_mean=("margin", "mean"),
            margin_q10=("margin", lambda x: x.quantile(0.1)),
        ))
    per_seed = pd.DataFrame(
        sub_df.groupby(["seed", "t"], as_index=False).agg(
            margin_mean=("margin", "mean"),
            margin_q10=("margin", lambda x: x.quantile(0.1)),
        )
    )
    return pd.DataFrame(per_seed.groupby("t", as_index=False).agg(
        margin_mean=("margin_mean", "mean"),
        margin_q10=("margin_q10", "mean"),
        se=("margin_mean", lambda x: x.std() / np.sqrt(len(x))),
    ))

def _seed_aggregated_overlap(sub_df):
    if "seed" not in sub_df.columns:
        return pd.DataFrame(sub_df.groupby("t", as_index=False).agg(
            correct_overlap=("correct_overlap", "mean"),
            strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        ))
    per_seed = pd.DataFrame(
        sub_df.groupby(["seed", "t"], as_index=False).agg(
            correct_overlap=("correct_overlap", "mean"),
            strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        )
    )
    return pd.DataFrame(per_seed.groupby("t", as_index=False).agg(
        correct_overlap=("correct_overlap", "mean"),
        strongest_wrong_overlap=("strongest_wrong_overlap", "mean"),
        se_correct=("correct_overlap", lambda x: x.std() / np.sqrt(len(x))),
        se_wrong=("strongest_wrong_overlap", lambda x: x.std() / np.sqrt(len(x))),
    ))

acc_df = _seed_aggregated_accuracy(df_zoom)
mar_df = _seed_aggregated_margin(df_zoom)
ov_df = _seed_aggregated_overlap(df_zoom)

# Has standard error?
has_se = "se" in acc_df.columns

# 1. Zoomed Accuracy vs t (t <= 10)
plt.figure(figsize=(8.5, 5.2))
if has_se:
    plt.errorbar(acc_df["t"], acc_df["accuracy"], yerr=acc_df["se"],
                 fmt="o-", capsize=4, capthick=1.5, linewidth=2, color="blue", label="Accuracy")
else:
    sns.lineplot(data=acc_df, x="t", y="accuracy", marker="o", color="blue")
plt.title("MNIST: Accuracy vs t (Zoomed-in t = 0 to 10)")
plt.xlabel("t")
plt.ylabel("Accuracy")
plt.xticks(range(0, 11))
plt.tight_layout()
plt.savefig(OUT_DIR / "mnist_accuracy_zoom.png", dpi=200)
plt.close()

# 2. Zoomed Margin/Overlap vs t (t <= 10)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

# Left: overlaps
if has_se and "se_correct" in ov_df.columns:
    ax1.errorbar(ov_df["t"], ov_df["correct_overlap"], yerr=ov_df["se_correct"],
                 fmt="o-", capsize=3, label=r"$o_y(t)$ Correct Overlap", linewidth=2, color="C0")
    ax1.errorbar(ov_df["t"], ov_df["strongest_wrong_overlap"], yerr=ov_df["se_wrong"],
                 fmt="s--", capsize=3, label=r"$\max_{z\neq y} o_z(t)$ Strongest Wrong", linewidth=2, color="C1")
else:
    ax1.plot(ov_df["t"], ov_df["correct_overlap"], "o-", label=r"$o_y(t)$", color="C0")
    ax1.plot(ov_df["t"], ov_df["strongest_wrong_overlap"], "s--", label=r"$\max_{z\neq y} o_z(t)$", color="C1")
ax1.set_title("Overlap vs t")
ax1.set_xlabel("t")
ax1.set_ylabel("Overlap")
ax1.set_xticks(range(0, 11))
ax1.legend()

# Right: margin
ax2.plot(mar_df["t"], mar_df["margin_mean"], "o-", label=r"$\mathbb{E}[m_y(t)]$ Mean Margin", linewidth=2, color="C2")
if "margin_q10" in mar_df.columns:
    ax2.plot(mar_df["t"], mar_df["margin_q10"], "s--", label=r"$Q_{0.1}[m_y(t)]$ Lower Quantile", linewidth=2, color="C3")
if "se" in mar_df.columns:
    ax2.fill_between(mar_df["t"],
                     mar_df["margin_mean"] - mar_df["se"],
                     mar_df["margin_mean"] + mar_df["se"],
                     color="C2", alpha=0.2, label="±1 SE")
ax2.set_title("Margin vs t")
ax2.set_xlabel("t")
ax2.set_ylabel("Margin")
ax2.set_xticks(range(0, 11))
ax2.legend()

fig.suptitle("MNIST: Overlap and Margin vs t (Zoomed-in t = 0 to 10)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "mnist_margin_zoom.png", dpi=200)
plt.close(fig)

print("Zoomed MNIST plots generated successfully!")
