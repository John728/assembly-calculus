import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx

from config import K_train_max, K_TEST_VALS

PLOT_STYLE = {
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(PLOT_STYLE)

FAMILY_COLORS = {'MLP': '#E15759', 'RNN': '#4E79A7', 'GNN': '#59A14F', 'Transformer': '#F28E2B', 'SSM': '#B07AA1', 'UT': '#9370DB', 'AC': '#76B7B2'}

def save_fig(path_no_ext, save_dir):
    plt.savefig(os.path.join(save_dir, path_no_ext + '.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_pointer_graph(p, start_node=0, num_hops=100, save_dir='outputs'):
    """
    Plots a directed graph tracing the first `num_hops` starting from `start_node`.
    """
    if hasattr(p, 'tolist'): p = p.tolist()
    G = nx.DiGraph()
    current = start_node
    path = [current]
    for _ in range(min(num_hops, len(p))):
        next_node = p[current]
        G.add_edge(current, next_node)
        current = next_node
        path.append(current)
        if current == start_node: break
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, edge_color='gray', arrows=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='orange', node_size=800)
    plt.title(f"Pointer Chasing - {len(path)-1} Hops")
    save_fig('plot_graph_hops', save_dir)

def plot_hero_surface(df, save_dir='outputs'):
    """Plot 1: Hero Plot - Accuracy Heatmap/Surface over (n, k) per Model Family."""
    families = df['family'].unique()
    sns.set_theme(style="white", context="talk")
    
    for family in families:
        sub_df = df[df['family'] == family]
        # Aggregate across seeds
        agg = sub_df.groupby(['N', 'k'])['accuracy'].mean().reset_index()
        pivot = agg.pivot(index='N', columns='k', values='accuracy')
        
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1.0, annot=True, fmt=".2f",
                         cbar_kws={'label': 'Accuracy'})
        
        # Add contour at 95%
        # We need to interpolate for a smooth contour, but let's stick to discrete for now
        # or just highlight cells >= 0.95
        
        ax.set_title(f"Hero Plot: {family} Accuracy Surface", fontweight='bold')
        ax.set_xlabel("Hop Count (k)")
        ax.set_ylabel("List Length (n)")
        plt.tight_layout()
        save_fig(f"plot_01_{family}_hero_surface", save_dir)

def plot_hop_scaling(df, save_dir='outputs'):
    """Plot 2: Hop Scaling Curves - Accuracy vs Hops (Fixed n)."""
    n_vals = sorted(df['N'].unique())
    sns.set_theme(style="whitegrid", context="talk")
    
    ncols = 2
    nrows = (len(n_vals) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=True)
    axes = axes.flatten()
    
    for i, n in enumerate(n_vals):
        sub_df = df[df['N'] == n]
        sns.lineplot(data=sub_df, x='k', y='accuracy', hue='family', marker='o', ax=axes[i], palette=FAMILY_COLORS)
        axes[i].set_title(f"n = {n}", fontweight='bold')
        axes[i].set_xlabel("Hops (k)")
        axes[i].set_ylabel("Accuracy")
        axes[i].axvline(x=K_train_max, color='gray', linestyle='--', alpha=0.5)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle("Plot 2: Hop Scaling Curves (Fixed n)", fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig("plot_02_hop_scaling", save_dir)

def plot_generalization_shock(df, save_dir='outputs'):
    """Plot 3: Generalization Shock Plot - Train (k <= K), Test (k > K)."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    
    # We'll use a specific N for this plot, e.g., the largest one
    n_max = df['N'].max()
    sub_df = df[df['N'] == n_max]
    
    sns.lineplot(data=sub_df, x='k', y='accuracy', hue='family', marker='o', palette=FAMILY_COLORS)
    plt.axvline(x=K_train_max, color='black', linestyle='--', label=f'Train Limit (k={K_train_max})')
    plt.title(f"Generalization Shock (N={n_max})", fontweight='bold')
    plt.xlabel("Hop Count (k)")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig("plot_03_generalization_shock", save_dir)

def plot_time_vs_hops(df, save_dir='outputs'):
    """Plot 4: Time Buys Hops Plot - AC/UT specific Heatmap (Internal Time vs Hops)."""
    # For AC, 'inference_steps' varies with 'k' in my implementation
    # but we can filter for AC family.
    ac_df = df[df['family'] == 'AC']
    if ac_df.empty:
        return
        
    agg = ac_df.groupby(['inference_steps', 'k'])['accuracy'].mean().reset_index()
    pivot = agg.pivot(index='k', columns='inference_steps', values='accuracy')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1.0, annot=True, fmt=".2f")
    plt.title("AC: Internal Time Steps (t) vs Hops (k)", fontweight='bold')
    plt.xlabel("Internal Time Steps (t)")
    plt.ylabel("Hops (k)")
    plt.tight_layout()
    save_fig("plot_04_time_buys_hops", save_dir)

def plot_phase_boundary(df, save_dir='outputs'):
    """Plot 5: Phase Boundary Plot - Max hop (k*) such that accuracy >= 95% vs n."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    
    threshold = 0.95
    results = []
    
    for (family, n), group in df.groupby(['family', 'N']):
        # Group by k and take mean accuracy
        k_agg = group.groupby('k')['accuracy'].mean().sort_index(ascending=False)
        # Find largest k with acc >= 0.95
        k_star = 0
        for k, acc in k_agg.items():
            if acc >= threshold:
                k_star = k
                break
        results.append({'family': family, 'N': n, 'k_star': k_star})
        
    res_df = pd.DataFrame(results)
    sns.lineplot(data=res_df, x='N', y='k_star', hue='family', marker='s', palette=FAMILY_COLORS)
    plt.title(f"Phase Boundary (k* at {int(threshold*100)}% Acc)", fontweight='bold')
    plt.xlabel("List Length (n)")
    plt.ylabel("Max Hop Capability (k*)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig("plot_05_phase_boundary", save_dir)

def plot_failure_mode(df, save_dir='outputs'):
    """Plot 6: Failure Mode Visualization - Cyclic Distance Histogram."""
    sns.set_theme(style="whitegrid", context="talk")
    hard_df = df[df['k'] > K_train_max]
    if hard_df.empty: return
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=hard_df, x='family', y='mean_cyclic_dist', palette=FAMILY_COLORS)
    plt.title("Failure Mode: Prediction Distance to Target (k > Train Max)", fontweight='bold')
    plt.xlabel("Model Family")
    plt.ylabel("Mean Cyclic Distance")
    plt.tight_layout()
    save_fig("plot_06_failure_mode", save_dir)

def plot_compute_tradeoff(df, save_dir='outputs'):
    """Plot 8: Runtime/Compute Tradeoff - Runtime vs Accuracy at hard setting."""
    sns.set_theme(style="whitegrid", context="talk")
    hard_k = df['k'].max()
    sub_df = df[df['k'] == hard_k]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sub_df, x='runtime_ms', y='accuracy', hue='family', style='family', s=100, palette=FAMILY_COLORS)
    plt.title(f"Compute Tradeoff (at k={hard_k})", fontweight='bold')
    plt.xlabel("Runtime (ms) per batch")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig("plot_08_compute_tradeoff", save_dir)

def plot_scaling_law(df, save_dir='outputs'):
    """Plot 9: Model Scaling Law - Parameters vs Max Hop k*."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    
    threshold = 0.95
    results = []
    
    for (family, params), group in df.groupby(['family', 'params']):
        k_agg = group.groupby('k')['accuracy'].mean().sort_index(ascending=False)
        k_star = 0
        for k, acc in k_agg.items():
            if acc >= threshold:
                k_star = k
                break
        results.append({'family': family, 'params': params, 'k_star': k_star})
        
    res_df = pd.DataFrame(results)
    sns.lineplot(data=res_df, x='params', y='k_star', hue='family', marker='o', palette=FAMILY_COLORS)
    plt.xscale('log')
    plt.title(f"Model Scaling Law (k* at {int(threshold*100)}% Acc)", fontweight='bold')
    plt.xlabel("Number of Parameters (log scale)")
    plt.ylabel("Max Hop Capability (k*)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig("plot_09_scaling_law", save_dir)

def plot_paper_ready(df, save_dir='outputs'):
    """A beautiful paper-ready figure layout (Fig 1: Heatmaps, Fig 2: Shock, Fig 3: AC Time, Fig 4: Phase)."""
    sns.set_theme(style="whitegrid", context="paper")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 4)
    
    # Top Row: Heatmaps for 3 families (MLP, Transformer, AC)
    families = ['MLP', 'Transformer', 'AC']
    for i, fam in enumerate(families):
        if fam not in df['family'].unique(): continue
        ax = fig.add_subplot(gs[0, i])
        sub_df = df[df['family'] == fam]
        agg = sub_df.groupby(['N', 'k'])['accuracy'].mean().reset_index()
        pivot = agg.pivot(index='N', columns='k', values='accuracy')
        sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1.0, ax=ax, cbar=(i==2))
        ax.set_title(f"{fam} Accuracy")
        
    # Bottom Row: The rest
    # 2. Generalization Shock
    ax1 = fig.add_subplot(gs[1, 0])
    n_max = df['N'].max()
    sub_df = df[df['N'] == n_max]
    sns.lineplot(data=sub_df, x='k', y='accuracy', hue='family', ax=ax1, palette=FAMILY_COLORS, legend=False)
    ax1.axvline(x=K_train_max, color='black', ls='--')
    ax1.set_title("Generalization Shock")
    
    # 3. AC Time vs Hops
    ax2 = fig.add_subplot(gs[1, 1:3])
    ac_df = df[df['family'] == 'AC']
    if not ac_df.empty:
        agg = ac_df.groupby(['inference_steps', 'k'])['accuracy'].mean().reset_index()
        pivot = agg.pivot(index='k', columns='inference_steps', values='accuracy')
        sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1.0, ax=ax2)
        ax2.set_title("AC: Internal Time vs Hops")
        
    # 4. Phase Boundary
    ax3 = fig.add_subplot(gs[1, 3])
    threshold = 0.95
    results = []
    for (family, n), group in df.groupby(['family', 'N']):
        k_agg = group.groupby('k')['accuracy'].mean().sort_index(ascending=False)
        k_star = 0
        for k, acc in k_agg.items():
            if acc >= threshold: k_star = k; break
        results.append({'family': family, 'N': n, 'k_star': k_star})
    res_df = pd.DataFrame(results)
    sns.lineplot(data=res_df, x='N', y='k_star', hue='family', ax=ax3, palette=FAMILY_COLORS, legend=False)
    ax3.set_title("Phase Boundary k*(n)")
    
    fig.suptitle("Assembly Calculus & Recurring Models: Evaluation Suite", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig("plot_paper_ready_figure", save_dir)

def plot_all_figures(df, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    plot_hero_surface(df, save_dir)
    plot_hop_scaling(df, save_dir)
    plot_generalization_shock(df, save_dir)
    plot_time_vs_hops(df, save_dir)
    plot_phase_boundary(df, save_dir)
    plot_failure_mode(df, save_dir)
    plot_compute_tradeoff(df, save_dir)
    plot_scaling_law(df, save_dir)
    plot_paper_ready(df, save_dir)
