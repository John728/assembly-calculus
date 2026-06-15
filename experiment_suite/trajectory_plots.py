import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_trajectories():
    # DFA trajectories not stored in csv, skipping
    pass

    # 2. Binary Search Trajectories
    bs_csv = Path("outputs/experiments/bs-ac/raw_results.csv")
    if bs_csv.exists():
        df = pd.read_csv(bs_csv)
        df["target"] = df["target"].apply(ast.literal_eval)
        df["prediction"] = df["prediction"].apply(ast.literal_eval)
        
        long_df = df[df["N"] >= 16]
        if not long_df.empty:
            example = long_df.iloc[0]
            true_traj = example["target"]
            pred_traj = example["prediction"]
            
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(true_traj)), true_traj, marker='o', label="Ground Truth Path", linewidth=3, alpha=0.6)
            plt.plot(range(len(pred_traj)), pred_traj, marker='x', label="AC Prediction Path", linestyle="--", linewidth=2, color="red")
            plt.title(f"Binary Search Interval Trajectory (N={example['N']}, Target Path Length={len(true_traj)-1})")
            plt.xlabel("Search Depth Step")
            plt.ylabel("Interval State ID")
            plt.legend()
            plt.tight_layout()
            
            out_dir = Path("outputs/experiments/bs-ac/plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / "bs_example_trajectory.png", dpi=150)
            plt.close()

if __name__ == "__main__":
    plot_trajectories()
