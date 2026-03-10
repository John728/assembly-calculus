import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os
import gc
from tqdm.auto import tqdm
import seaborn as sns

from config import N_LIST, K_train_max, K_TEST_VALS, NUM_CONFIGS, T_STEPS
from dataset import FixedPermutationDataset, collate_dict
from models import (
    mlp_configs, transformer_configs, gnn_configs, ssm_configs, ut_configs, rnn_configs, ac_configs,
    create_model, count_parameters
)
from plot import plot_pointer_graph
# Ensure reproducibility and set nice seaborn styles
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid", context="talk")

os.makedirs("outputs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

print(f"PyTorch Version: {torch.__version__}")
# Automatically use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_full_cycle(n):
    """
    Generates a random permutation that consists of a single cycle of length n.
    """
    nodes = np.random.permutation(n)
    cycle = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        cycle[nodes[i]] = nodes[i + 1]
    cycle[nodes[-1]] = nodes[0]
    return torch.from_numpy(cycle).long()

def train_model(model, train_loader, val_loader, K_train_max, device, num_epochs=10, lr=3e-4):
    """
    Train the model using the provided loaders.
    """
    if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch['y']).sum().item()
                val_total += batch['y'].size(0)
        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return model

def evaluate_k(model, p, size=2000, fixed_k=1, device='cpu'):
    """Evaluate a model on a fixed permutation p for a specific hop count k."""
    model.eval()
    test_ds = FixedPermutationDataset(p=p, size=size, k_range=(fixed_k, fixed_k), fixed_k=fixed_k)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_dict)
    correct = 0
    total = 0
    t0 = time.time()
    all_steps = []
    
    # For cyclic distance (Plot 6)
    # We need the permutation to compute distance from prediction to true target
    p_np = p.cpu().numpy() if hasattr(p, 'cpu') else np.array(p)
    distances = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, steps = model(batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['y']).sum().item()
            total += batch['y'].size(0)
            all_steps.append(steps)
            
            # Compute distances for Plot 13 (Failure Mode)
            # Distance from preds[j] to batch['y'][j] along the permutation cycle
            pred_nodes = preds.cpu().numpy()
            true_nodes = batch['y'].cpu().numpy()
            
            for j in range(len(pred_nodes)):
                if pred_nodes[j] == true_nodes[j]:
                    distances.append(0)
                else:
                    # Find distance in cycle
                    d = 0
                    curr = pred_nodes[j]
                    found = False
                    for _ in range(len(p_np)):
                        curr = p_np[curr]
                        d += 1
                        if curr == true_nodes[j]:
                            found = True
                            break
                    distances.append(d if found else len(p_np))

    runtime = (time.time() - t0) * 1000 / max(len(test_loader), 1)
    acc = correct / max(total, 1)
    mean_steps = np.mean(all_steps) if all_steps else 0
    mean_dist = np.mean(distances) if distances else 0
    return acc, runtime, mean_steps, mean_dist

if __name__ == '__main__':
    RUN_EXPERIMENTS = True
    IS_TEST_RUN = False

    if RUN_EXPERIMENTS:
        if IS_TEST_RUN:
            TRAIN_SAMPLES = 1000
            VAL_SAMPLES   = 100
            TEST_SAMPLES  = 200
            EPOCHS        = 2
            SEEDS         = 1
            N_LIST_RUN    = [16, 32]
        else:
            TRAIN_SAMPLES = 10000 
            VAL_SAMPLES   = 1000   
            TEST_SAMPLES  = 1000   
            EPOCHS        = 10      
            SEEDS         = 1      
            N_LIST_RUN    = N_LIST

        results = []

        for n_val in N_LIST_RUN:
            print(f"\n--- Experiments for N={n_val} ---")
            
            experiments = []
            for conf in mlp_configs: experiments.append(('MLP', conf))
            for conf in transformer_configs: experiments.append(('Transformer', conf))
            for conf in gnn_configs: experiments.append(('GNN', conf))
            for conf in ssm_configs: experiments.append(('SSM', conf))
            for conf in ut_configs: experiments.append(('UT', conf))
            for conf in rnn_configs: experiments.append(('RNN', conf))
            for conf in ac_configs(): experiments.append(('AC', conf))

            for seed in range(SEEDS):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                for family, conf in tqdm(experiments, desc=f"N={n_val}, Seed {seed}"):
                    model = create_model(family, conf, N=n_val)
                    model.to(device)
                    n_params = count_parameters(model)

                    p_tensor = generate_full_cycle(n_val)
                    train_k_max = K_train_max

                    weight_path = os.path.join("weights", f"{family}_{conf['name']}_N{n_val}_s{seed}.pt")
                    if os.path.exists(weight_path) and family != 'AC': # Don't cache AC as it's purely recurrent/simulated
                        model.load_state_dict(torch.load(weight_path, map_location=device))
                    elif family != 'AC':
                        train_ds = FixedPermutationDataset(p=p_tensor, size=TRAIN_SAMPLES, k_range=(1, train_k_max))
                        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_dict)
                        val_ds = FixedPermutationDataset(p=p_tensor, size=VAL_SAMPLES, k_range=(train_k_max, train_k_max), fixed_k=train_k_max)
                        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_dict)
                        model = train_model(model, train_loader, val_loader, train_k_max, device, num_epochs=EPOCHS)
                        torch.save(model.state_dict(), weight_path)

                    for k_val in K_TEST_VALS:
                        # Special case for AC/UT: we might want to sweep internal time steps T
                        # but for now we follow the 'k' hops requirement.
                        acc, runtime_ms, steps, mean_dist = evaluate_k(model, p=p_tensor, size=TEST_SAMPLES, fixed_k=k_val, device=device)
                        results.append({
                            'run_id': f"{family}_{conf['name']}_N{n_val}_s{seed}",
                            'seed': seed,
                            'family': family,
                            'name': conf['name'],
                            'N': n_val,
                            'k': k_val,
                            'K_train': train_k_max,
                            'params': n_params,
                            'inference_steps': steps,
                            'accuracy': acc,
                            'runtime_ms': runtime_ms,
                            'mean_cyclic_dist': mean_dist
                        })
                    
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

        df = pd.DataFrame(results)
        df.to_csv('outputs/results.csv', index=False)
        print("Experiments completed. Saved to outputs/results.csv")

    else:
        df = pd.read_csv('outputs/results.csv')

    print("Generating figures...")
    from plot import plot_all_figures
    plot_all_figures(df)
    print("Done! All plots saved in 'outputs/'")
