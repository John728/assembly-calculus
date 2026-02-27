import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os
from tqdm.auto import tqdm
import seaborn as sns

from config import N, K_train_max, K_TEST_VALS
from dataset import FixedPermutationDataset, collate_dict
from models import mlp_configs, transformer_configs, create_model, count_parameters
from plot import plot_heatmaps, plot_phase_transition, plot_pointer_graph

# Ensure reproducibility and set nice seaborn styles
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid", context="talk")

os.makedirs("outputs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

print(f"PyTorch Version: {torch.__version__}")
# Force CPU due to sm_61 incompatibility with GTX 1070
device = torch.device('cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, K_train_max, device, num_epochs=10, lr=3e-4):
    """
    Train the model using the provided loaders.
    """
    model.to(device)
    # If the model has no trainable parameters, skip the training loop
    if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Simple LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['y']).sum().item()
            total += batch['y'].size(0)
        scheduler.step()
        # Validation
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
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_dict)
    correct = 0
    total = 0
    t0 = time.time()
    mean_steps = 0
    batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, steps = model(batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['y']).sum().item()
            total += batch['y'].size(0)
            mean_steps += steps
            batches += 1
    runtime = (time.time() - t0) * 1000 / max(len(test_loader), 1)
    acc = correct / max(total, 1)
    return acc, runtime, (mean_steps / batches)

if __name__ == '__main__':
    RUN_EXPERIMENTS = True
    IS_TEST_RUN = False

    if RUN_EXPERIMENTS:
        if IS_TEST_RUN:
            TRAIN_SAMPLES = 5000
            VAL_SAMPLES   = 500
            TEST_SAMPLES  = 1000
            EPOCHS        = 5
            SEEDS         = 1
        else:
            TRAIN_SAMPLES = 20000 
            VAL_SAMPLES   = 2000   
            TEST_SAMPLES  = 2000   
            EPOCHS        = 10      
            SEEDS         = 1      

        results = []

        experiments = []
        for conf in mlp_configs:
            experiments.append(('MLP', conf))
        for conf in transformer_configs:
            experiments.append(('Transformer', conf))

        for seed in range(SEEDS):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            for family, conf in tqdm(experiments, desc=f"Seed {seed}"):
                print(f"Training {family} - {conf['name']}")
                model = create_model(family, conf, N=N)
                n_params = count_parameters(model)

                p_tensor = torch.randperm(N)
                train_k_max = K_train_max

                # Only plot the graph once based on the first dataset generation to show an example graph
                if family == 'MLP' and conf['name'] == 'MLP-01' and seed == 0:
                    plot_pointer_graph(p_tensor, start_node=0, num_hops=100)

                # Check for cached weights
                weight_path = os.path.join("weights", f"{family}_{conf['name']}_s{seed}.pt")
                if os.path.exists(weight_path):
                    print(f"Skipping training, loaded cached weights from {weight_path}")
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                else:
                    train_ds = FixedPermutationDataset(p=p_tensor, size=TRAIN_SAMPLES, k_range=(1, train_k_max))
                    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_dict)

                    val_ds = FixedPermutationDataset(p=p_tensor, size=VAL_SAMPLES, k_range=(train_k_max, train_k_max), fixed_k=train_k_max)
                    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_dict)

                    model = train_model(model, train_loader, val_loader, train_k_max, device, num_epochs=EPOCHS)
                    torch.save(model.state_dict(), weight_path)

                for k_val in K_TEST_VALS:
                    acc, runtime_ms, steps = evaluate_k(model, p=p_tensor, size=TEST_SAMPLES, fixed_k=k_val, device=device)
                    results.append({
                        'run_id': f"{family}_{conf['name']}_s{seed}",
                        'seed': seed,
                        'family': family,
                        'name': conf['name'],
                        'N': N,
                        'k': k_val,
                        'K_train': train_k_max,
                        'params': n_params,
                        'layers': conf.get('layers', 1),
                        'hidden_dim': conf.get('hidden_dim'),
                        'inference_steps': steps,
                        'accuracy': acc,
                        'runtime_ms': runtime_ms
                    })

        # Collect results into DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv('outputs/results.csv', index=False)
        print("Experiments completed. Saved to outputs/results.csv")

    else:
        # If RUN_EXPERIMENTS is False, simply load existing results
        df = pd.read_csv('outputs/results.csv')

    print("Generating figures...")
    plot_heatmaps(df)
    plot_phase_transition(df)
    print("Done! All plots saved in 'outputs/' as .png")
