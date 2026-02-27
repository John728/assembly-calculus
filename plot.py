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

FAMILY_COLORS = {'MLP': '#E15759', 'RNN': '#4E79A7', 'GNN': '#59A14F', 'Transformer': '#F28E2B'}

def save_fig(path_no_ext, save_dir):
    plt.savefig(os.path.join(save_dir, path_no_ext + '.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_pointer_graph(p, start_node=0, num_hops=100, save_dir='outputs'):
    """
    Plots a directed graph tracing the first `num_hops` starting from `start_node`.
    p is a list or 1D torch.Tensor representing the permutation.
    """
    if hasattr(p, 'tolist'):
        p = p.tolist()
        
    G = nx.DiGraph()
    
    # Trace the path to build the graph nodes and edges
    current = start_node
    path = [current]
    for _ in range(min(num_hops, len(p))):
        next_node = p[current]
        G.add_edge(current, next_node)
        current = next_node
        path.append(current)
        if current == start_node:
            # We hit a cycle and wrapped back around, stop tracing.
            break

    plt.figure(figsize=(10, 8))
    
    # Layout algorithm for the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw graph
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='skyblue', 
            node_size=600, 
            edge_color='gray', 
            linewidths=1, 
            font_size=10, 
            font_weight='bold', 
            arrows=True, 
            arrowstyle='-|>,head_length=0.7,head_width=0.4')
            
    # Highlight the start node in orange
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='orange', node_size=800)
    
    plt.title(f"Pointer Chasing - First {len(path)-1} Hops from Node {start_node}", fontsize=14, fontweight='bold', pad=15)
    
    save_fig(f'plot_graph_hops', save_dir)


def plot_heatmaps(df, save_dir='outputs'):
    families = df['family'].unique()
    
    for family in families:
        sub_df = df[df['family'] == family]
        
        # Replace 'name' with 'params' formatted for display
        sub_df = sub_df.copy()
        sub_df['label'] = sub_df['params'].apply(lambda x: f"{x/1000:.1f}K Params" if x >= 1000 else f"{x} Params")
        
        agg_df = sub_df.groupby(['label', 'k', 'params'])['accuracy'].mean().reset_index()
        pivot_df = agg_df.pivot(index='label', columns='k', values='accuracy')
        # Sort by actual 'params' size, not alphabetical label
        param_map = agg_df[['label', 'params']].drop_duplicates().set_index('label')['params'].to_dict()
        pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: param_map[x]))
        
        n_rows = pivot_df.shape[0]
        # Cap figure height so it fits on an A4 page (around 11 inches max)
        fig_h = min(11.0, max(2.5, n_rows * 0.3 + 1.5))
        plt.figure(figsize=(12, fig_h))
        
        # Annotate cells if small enough
        do_annot = (pivot_df.size <= 60)
        annot_fmt = '.2f' if do_annot else False
        ax = sns.heatmap(
            pivot_df, cmap='RdYlGn', vmin=0, vmax=1.0,
            annot=do_annot, fmt=annot_fmt if do_annot else '',
            linewidths=0.4 if do_annot else 0,
            cbar_kws={'label': 'Exact-Match Accuracy', 'shrink': 0.8},
            annot_kws={'size': 8}
        )
        ax.invert_yaxis()
        
        # Mark the train / test boundary
        if K_train_max in K_TEST_VALS:
            train_col = list(pivot_df.columns).index(K_train_max)
            ax.axvline(x=train_col + 1, color='dodgerblue', linewidth=2.5, linestyle='--', label=f'Train boundary (k={K_train_max})')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        
        ax.set_title(f"{family} — Accuracy Heatmap (k-hop extrapolation)", fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel("Hop Count (k)", fontsize=11)
        ax.set_ylabel("Model Configuration", fontsize=11)
        plt.tight_layout()
        save_fig(f'plot_A1_{family}_heatmap', save_dir)


def plot_phase_transition(df, save_dir='outputs'):
    families = sorted(df['family'].unique())
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    
    palette = sns.color_palette('tab10', n_colors=10)
    
    for i, family in enumerate(families):
        sub_df = df[df['family'] == family].copy()
        
        # Replace 'name' with param size
        sub_df['label'] = sub_df['params'].apply(lambda x: f"{x/1000:.1f}K Params" if x >= 1000 else f"{x} Params")
        
        # We need to process names in param order to keep legend sorted nicely
        param_to_label = sub_df[['params', 'label']].drop_duplicates().sort_values('params')
        
        for j, (_, row) in enumerate(param_to_label.iterrows()):
            grp = sub_df[sub_df['params'] == row['params']].groupby('k')['accuracy'].mean().reset_index()
            axes[i].plot(grp['k'], grp['accuracy'], marker='o', linewidth=2,
                        label=row['label'], color=palette[j % 10])
        
        axes[i].axvline(x=K_train_max, color='crimson', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Train max (k={K_train_max})')
        axes[i].axhline(y=0.9, color='dimgray', linestyle=':', linewidth=1.5, label='90% threshold')
        axes[i].set_title(family, fontsize=13, fontweight='bold')
        axes[i].set_xscale('log')
        xticks = [t for t in K_TEST_VALS if t <= max(K_TEST_VALS)]
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticks, fontsize=8, rotation=45)
        axes[i].set_xlabel("Hop Count (k)", fontsize=11)
        axes[i].set_ylim(-0.02, 1.05)
        
        # Move legend outside the plot to handle many models, use multiple columns if needed
        num_models = len(param_to_label)
        ncol = max(1, num_models // 15)
        axes[i].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=ncol)
        axes[i].grid(True, alpha=0.2)
    
    axes[0].set_ylabel("Exact-Match Accuracy", fontsize=12)
    fig.suptitle("Plot A3: Phase Transition — Accuracy vs. k", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('plot_A3_phase_transition', save_dir)
