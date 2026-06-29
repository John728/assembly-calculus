import os
import ast
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

def generate_dfa_plots():
    dfa_csv = Path("outputs/experiments/dfa-ac/raw_results.csv")
    if not dfa_csv.exists():
        print("DFA CSV not found!")
        return
    df = pd.read_csv(dfa_csv)
    
    # Group by c and L
    grouped = df.groupby(["L", "c"])["path_accuracy"].mean().reset_index()
    
    # 1. dfa_path_accuracy.png: Path Accuracy vs c
    plt.figure(figsize=(8, 5))
    for L_val in sorted(grouped["L"].unique()):
        sub = grouped[grouped["L"] == L_val].sort_values("c")
        plt.plot(sub["c"], sub["path_accuracy"], marker="o", label=f"L={L_val}")
    plt.xlabel("Steps per Symbol (c)")
    plt.ylabel("Mean Path Accuracy")
    plt.title("DFA State Tracking: Path Accuracy vs Steps per Symbol (c)")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Sequence Length")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dfa_path_accuracy.png", dpi=150)
    plt.close()
    
    # 2. dfa_path_accuracy_vs_L.png: Path Accuracy vs L
    plt.figure(figsize=(8, 5))
    for c_val in sorted(grouped["c"].unique()):
        sub = grouped[grouped["c"] == c_val].sort_values("L")
        plt.plot(sub["L"], sub["path_accuracy"], marker="s", label=f"c={c_val}")
    plt.xlabel("Sequence Length (L)")
    plt.ylabel("Mean Path Accuracy")
    plt.title("DFA State Tracking: Path Accuracy vs Sequence Length")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Steps per Symbol (c)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dfa_path_accuracy_vs_L.png", dpi=150)
    plt.close()
    
    # 3. dfa_first_error_histogram.png
    first_err = df["first_error_index"].dropna()
    if not first_err.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(first_err.astype(int), bins=range(int(first_err.max()) + 2), align="left",
                 edgecolor="black", color="skyblue", alpha=0.8)
        plt.xlabel("Sequence Index of First Error")
        plt.ylabel("Count")
        plt.title("DFA State Tracking: First Error Index Distribution")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "dfa_first_error_histogram.png", dpi=150)
        plt.close()

def generate_bs_plots():
    bs_csv = Path("outputs/experiments/bs-ac/raw_results.csv")
    if not bs_csv.exists():
        print("BS CSV not found!")
        return
    df = pd.read_csv(bs_csv)
    
    # 1. bs_example_trajectory.png (Shaded interval boundaries)
    long_df = df[(df["N"] == 16) & (df["c"] == 1)]
    example_row = None
    for _, row in long_df.iterrows():
        try:
            target_trace = ast.literal_eval(row["target"])
            if len(target_trace) > 3:
                example_row = row
                break
        except:
            continue
            
    if example_row is not None:
        target_trace = ast.literal_eval(example_row["target"])
        pred_trace = ast.literal_eval(example_row["prediction"])
        
        def reverse_map(state_id, N):
            num_states = N * (N + 1) // 2 + 1
            terminal_state = num_states - 1
            if state_id == terminal_state or state_id == -1:
                return None
            rem = state_id
            for a in range(N):
                cnt = N - a
                if rem < cnt:
                    return a, a + rem
                rem -= cnt
            return None

        def trace_to_intervals(trace, N):
            intervals = []
            for state_id in trace:
                res = reverse_map(state_id, N)
                if res is not None:
                    intervals.append(res)
                else:
                    if intervals:
                        prev_a, prev_b = intervals[-1]
                        m = (prev_a + prev_b) // 2
                        intervals.append((m, m))
                    else:
                        intervals.append((0, 0))
            return intervals

        true_intervals = trace_to_intervals(target_trace, 16)
        pred_intervals = trace_to_intervals(pred_trace, 16)
        
        steps = range(len(true_intervals))
        true_a = [inv[0] for inv in true_intervals]
        true_b = [inv[1] for inv in true_intervals]
        pred_a = [inv[0] for inv in pred_intervals]
        pred_b = [inv[1] for inv in pred_intervals]
        
        # Target index is the final converged index
        target_idx = true_a[-1]
        
        plt.figure(figsize=(9, 5))
        # Fill between boundaries for ground truth
        plt.fill_between(steps, true_a, true_b, color="skyblue", alpha=0.5, label="Ground Truth Interval Space")
        # Plot predicted boundaries
        plt.plot(steps, pred_a, color="red", linestyle="--", marker="x", label="AC Lower Bound (a)")
        plt.plot(steps, pred_b, color="darkred", linestyle="--", marker="o", label="AC Upper Bound (b)")
        # Plot horizontal line for target index
        plt.axhline(target_idx, color="green", linestyle=":", linewidth=2.5, label=f"Target Index (idx={target_idx})")
        
        plt.xlabel("Search Step")
        plt.ylabel("Array Index Space")
        plt.title(f"Binary Search: Shaded Interval Shrinkage (N=16, c={example_row['c']})")
        plt.xticks(steps)
        plt.yticks(range(0, 17, 2))
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "bs_example_trajectory.png", dpi=150)
        plt.close()

    # 2. bs_path_accuracy_vs_L.png
    grouped = df.groupby(["N", "c"])["path_accuracy"].mean().reset_index()
    grouped["L"] = np.ceil(np.log2(grouped["N"])).astype(int)
    
    plt.figure(figsize=(8, 5))
    for c_val in sorted(grouped["c"].unique()):
        sub = grouped[grouped["c"] == c_val].sort_values("L")
        plt.plot(sub["L"], sub["path_accuracy"], marker="o", label=f"c={c_val} (steps/comp)")
    plt.xlabel("Nominal Search Depth (L = ceil(log2 N))")
    plt.ylabel("Mean Path Accuracy")
    plt.title("Binary Search: Path Accuracy vs Search Depth")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.xticks(sorted(grouped["L"].unique()))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bs_path_accuracy_vs_L.png", dpi=150)
    plt.close()
    
    # 3. bs_accuracy_vs_t_by_L.png
    plt.figure(figsize=(8, 5))
    for N_val in sorted(grouped["N"].unique()):
        sub = grouped[grouped["N"] == N_val].sort_values("c")
        plt.plot(sub["c"], sub["path_accuracy"], marker="s", label=f"N={N_val}")
    plt.xlabel("Steps per Comparison Step (c)")
    plt.ylabel("Mean Path Accuracy")
    plt.title("Binary Search: Path Accuracy vs Steps per Comparison")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Array Size (N)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bs_accuracy_vs_t_by_L.png", dpi=150)
    plt.close()

def generate_pointer_plots():
    pointer_csv = Path("results/pointer/seen_theory/raw_results.csv")
    if not pointer_csv.exists():
        print("Pointer seen CSV not found!")
        return
    df = pd.read_csv(pointer_csv)
    
    # Ensure correct columns and types
    df["L"] = df["L"].astype(int)
    df["t"] = df["t"].astype(int)
    if "c" not in df.columns:
        df["c"] = 1
    df["c"] = pd.to_numeric(df["c"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    c_values = sorted(df["c"].unique())
    primary_c = 1 if 1 in c_values else int(c_values[0])
    
    # Group and aggregate. Legacy filenames are filtered to one c value; do not
    # average transition costs together.
    grouped_all = df.groupby(["c", "L", "t"])["accuracy"].mean().reset_index()
    grouped = grouped_all[grouped_all["c"] == primary_c].copy()
    
    # 1. Heatmap Acc(K,t)
    unique_L = sorted(grouped["L"].unique())
    unique_t = sorted(grouped["t"].unique())
    heatmap_data = np.full((len(unique_L), len(unique_t)), np.nan)
    for i, L_val in enumerate(unique_L):
        mask = grouped["L"] == L_val
        for j, t_val in enumerate(unique_t):
            val = grouped.loc[mask & (grouped["t"] == t_val), "accuracy"]
            if len(val) > 0:
                heatmap_data[i, j] = val.iloc[0]
                
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(range(len(unique_t)))
    ax.set_xticklabels([str(t) for t in unique_t])
    ax.set_yticks(range(len(unique_L)))
    ax.set_yticklabels([str(L) for L in unique_L])
    ax.set_xlabel("Execution Time Budget (t)")
    ax.set_ylabel("Hop Depth (K)")
    ax.set_title(f"Pointer Chasing: Accuracy Heatmap Acc(K, t), c={primary_c}")
    plt.colorbar(im, ax=ax, label="Mean Accuracy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pointer_heatmap.png", dpi=150)
    fig.savefig(OUT_DIR / "pointer_accuracy_heatmap_L_t.png", dpi=150)
    plt.close(fig)
    
    # 2. pointer_accuracy_vs_t_by_L.png
    plt.figure(figsize=(8, 5))
    for L_val in unique_L:
        sub = grouped[grouped["L"] == L_val].sort_values("t")
        plt.plot(sub["t"], sub["accuracy"], marker="o", label=f"K={L_val}")
    plt.xlabel("Execution Time Budget (t)")
    plt.ylabel("Mean Accuracy")
    plt.title(f"Pointer Chasing: Accuracy vs Time Budget, c={primary_c}")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Hop Depth (K)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pointer_accuracy_vs_t_by_L.png", dpi=150)
    plt.close()
    
    # 3. pointer_accuracy_vs_L_by_t.png
    plt.figure(figsize=(8, 5))
    for t_val in [1, 2, 3, 4, 6]:
        if t_val in unique_t:
            sub = grouped[grouped["t"] == t_val].sort_values("L")
            plt.plot(sub["L"], sub["accuracy"], marker="s", label=f"t={t_val}")
    plt.xlabel("Hop Depth (K)")
    plt.ylabel("Mean Accuracy")
    plt.title(f"Pointer Chasing: Accuracy vs Hop Depth, c={primary_c}")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Time Budget (t)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pointer_accuracy_vs_L_by_t.png", dpi=150)
    plt.close()
    
    # 4. pointer_path_accuracy_vs_L.png
    if "path_accuracy" in df.columns:
        path_grouped_all = df.groupby(["c", "L", "t"])["path_accuracy"].mean().reset_index()
        path_grouped = path_grouped_all[path_grouped_all["c"] == primary_c].copy()
        plt.figure(figsize=(8, 5))
        for t_val in [1, 2, 3, 4, 6]:
            if t_val in unique_t:
                sub = path_grouped[path_grouped["t"] == t_val].sort_values("L")
                plt.plot(sub["L"], sub["path_accuracy"], marker="o", label=f"t={t_val}")
        plt.xlabel("Hop Depth (K)")
        plt.ylabel("Mean Path Accuracy")
        plt.title(f"Pointer Chasing: Path Accuracy vs Hop Depth, c={primary_c}")
        plt.ylim(-0.05, 1.05)
        plt.legend(title="Time Budget (t)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pointer_path_accuracy_vs_L.png", dpi=150)
        plt.close()
        
    # 5. pointer_first_error_histogram.png
    if "first_error_index" in df.columns:
        first_err = df[df["c"] == primary_c]["first_error_index"].dropna()
        if not first_err.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(first_err.astype(int), bins=range(int(first_err.max()) + 2), align="left",
                     edgecolor="black", color="purple", alpha=0.8)
            plt.xlabel("First Error Hop Index")
            plt.ylabel("Count")
            plt.title(f"Pointer Chasing: First Error Hop Index Distribution, c={primary_c}")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "pointer_first_error_histogram.png", dpi=150)
            plt.close()
            
    # 6. pointer_shortcut_ablation.png & pointer_shortcuts.png
    # Only generate this figure from an explicit shortcut/operator experiment.
    label_column = None
    for candidate in ("shortcut_operator", "shortcut_label", "operator"):
        if candidate in df.columns:
            label_column = candidate
            break
    if label_column is not None:
        shortcut_df = df[df[label_column].notna()].copy()
    else:
        shortcut_df = pd.DataFrame()
    if not shortcut_df.empty:
        shortcut_grouped = shortcut_df.groupby([label_column, "t"])["accuracy"].mean().reset_index()
        plt.figure(figsize=(8, 6))
        for label in sorted(shortcut_grouped[label_column].astype(str).unique()):
            sub = shortcut_grouped[shortcut_grouped[label_column].astype(str) == label].sort_values("t")
            plt.plot(sub["t"], sub["accuracy"], marker="s", label=label)
        plt.axhline(0.95, color="gray", linestyle="--", alpha=0.7)
        plt.title("Time-Size Tradeoff: Explicit Shortcuts vs Execution Time")
        plt.xlabel("Execution Time Budget (t)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pointer_shortcut_ablation.png", dpi=150)
        plt.savefig(OUT_DIR / "pointer_shortcuts.png", dpi=150)
        plt.close()

    # 7. Fit-Then-Predict Protocol plot (pointer_fit_predict.png)
    df_fit = df[df["L"].isin([1, 2, 3])].copy()
    if not df_fit.empty:
        obs_stats = df_fit.groupby(["L", "c"])["path_accuracy"].agg(["mean", "sem"]).reset_index()
        
        plt.figure(figsize=(9, 6))
        colors = ["blue", "orange", "green", "red"]
        unique_cs = sorted(obs_stats["c"].unique())
        
        for idx, c_val in enumerate(unique_cs):
            c_data = obs_stats[obs_stats["c"] == c_val].sort_values("L")
            c_color = colors[idx % len(colors)]
            
            # Observed curve
            plt.errorbar(c_data["L"], c_data["mean"], yerr=c_data["sem"],
                         fmt="-o", color=c_color, capsize=4, capthick=1.5, linewidth=2,
                         label=f"Observed (c={c_val})")
            
            # Predict curve: (1 - epsilon)^L
            p_L1 = c_data[c_data["L"] == 1]["mean"].values
            if len(p_L1) > 0:
                p_L1 = p_L1[0]
                pred_L = [1, 2, 3]
                pred_acc = [p_L1, p_L1**2, p_L1**3]
                plt.plot(pred_L, pred_acc, linestyle="--", color=c_color, alpha=0.7,
                         label=f"Predicted (c={c_val})")
                         
        plt.xlabel("Hop Depth (K)")
        plt.ylabel("Path Accuracy")
        plt.title("Pointer Chasing: Fit-Then-Predict Protocol Validation")
        plt.xticks([1, 2, 3])
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pointer_fit_predict.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    generate_dfa_plots()
    generate_bs_plots()
    generate_pointer_plots()
    print("All plots generated successfully!")
