"""Theory-compliant MNIST notebook-architecture runner.

Implements all §6 requirements: multi-seed, recurrent on/off, held/cue-only,
true trajectories, overlap trajectories, and all required measurements.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import convolve

ROOT = Path(__file__).resolve().parents[2]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def _k_cap(input, cap_size):
    output = np.zeros_like(input)
    if len(input.shape) == 1:
        idx = np.argsort(input)[-cap_size:]
        output[idx] = 1
    else:
        idx = np.argsort(input, axis=-1)[:, -cap_size:]
        np.put_along_axis(output, idx, 1, axis=-1)
    return output


def run_mnist_nb_full(
    data_dir="data/mnist",
    seed=42,
    t_max=10,
    train_limit=5000,
    test_limit=2000,
    n_neurons=2000,
    cap_size=200,
    sparsity=0.1,
    beta=1.0,
    n_rounds=5,
    recurrent=True,
    stimulus_mode="held",
    model_name="MNIST-NB-Full",
):
    rng = np.random.default_rng(seed)

    # --- Load data ---
    train_path = Path(data_dir) / "mnist_train.csv"
    test_path = Path(data_dir) / "mnist_test.csv"
    if not train_path.exists():
        from pyac.tasks.mnist.data import load_mnist_split
        train_i = load_mnist_split(data_dir, "train")
        test_i = load_mnist_split(data_dir, "test")
        train_data = np.column_stack([train_i.labels, (train_i.images.reshape(len(train_i.images), -1) * 255).astype(np.uint8)])
        test_data = np.column_stack([test_i.labels, (test_i.images.reshape(len(test_i.images), -1) * 255).astype(np.uint8)])
    else:
        train_data = np.loadtxt(str(train_path), delimiter=',')
        test_data = np.loadtxt(str(test_path), delimiter=',')

    train_imgs = train_data[:, 1:].astype(np.float64)
    train_labels = train_data[:, 0].astype(np.int64)
    test_imgs = test_data[:, 1:].astype(np.float64)
    test_labels = test_data[:, 0].astype(np.int64)

    n_in = 784
    n_examples_per_class = min(train_limit, max(1, (train_labels == 0).sum()))
    n_examples_total = n_examples_per_class * 10

    # --- Build connectivity ---
    if recurrent:
        mask_w = (rng.random((n_neurons, n_neurons)) < sparsity) & np.logical_not(np.eye(n_neurons, dtype=bool))
        W = np.ones((n_neurons, n_neurons)) * mask_w
        W /= W.sum(axis=0)
    else:
        # Recurrent off: use zero-weight identity matrix (no recurrent contribution)
        W = np.zeros((n_neurons, n_neurons))

    mask_a = rng.random((n_in, n_neurons)) < sparsity
    A = np.ones((n_in, n_neurons)) * mask_a
    A /= A.sum(axis=0)

    # --- Pre-process examples per class ---
    examples = np.zeros((10, n_examples_per_class, n_in))
    for i in range(10):
        class_imgs = train_imgs[train_labels == i][:n_examples_per_class]
        examples[i] = _k_cap(
            convolve(class_imgs.reshape(-1, 28, 28), np.ones((1, 3, 3)), mode='same').reshape(-1, 28 * 28),
            cap_size,
        )

    # --- Training ---
    W = np.ones_like(W) * (mask_w if recurrent else np.zeros_like(W, dtype=bool))
    if recurrent:
        W /= W.sum(axis=0, keepdims=True)
    A = np.ones_like(A) * mask_a
    A /= A.sum(axis=0, keepdims=True)
    bias = np.zeros(n_neurons)
    b = -1
    activations = np.zeros((10, n_rounds, n_neurons))
    for i in range(10):
        act_h = np.zeros(n_neurons)
        for j in range(n_rounds):
            input_vec = examples[i, j % n_examples_per_class]
            act_h_new = _k_cap(act_h @ W + input_vec @ A + bias, cap_size)
            activations[i, j] = act_h_new.copy()
            A[(input_vec > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
            if recurrent:
                W[(act_h > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
            act_h = act_h_new
        bias[act_h > 0] += b
        A /= A.sum(axis=0, keepdims=True)
        if recurrent:
            W /= W.sum(axis=0, keepdims=True)

    # --- Build disjoint class assemblies (greedy) ---
    idx = np.full(n_neurons, -1, dtype=int)
    act = activations[:, -1].copy()
    for i, j in enumerate(range(10)):
        idx[i * cap_size:(i + 1) * cap_size] = act[j].argsort()[-cap_size:][::-1]
        act[:, idx[i * cap_size:(i + 1) * cap_size]] = -1
    r_indices = np.arange(n_neurons)
    r_indices[idx[idx > -1]] = -1
    idx[(i + 1) * cap_size:] = np.unique(r_indices)[1:]

    # --- Evaluate on test set with t sweep ---
    actual_test = min(test_limit, len(test_imgs))
    test_img_sub = test_imgs[:actual_test]
    test_lbl_sub = test_labels[:actual_test]

    test_examples = _k_cap(
        convolve(test_img_sub.reshape(-1, 28, 28), np.ones((1, 3, 3)), mode='same').reshape(-1, 28 * 28),
        cap_size,
    )

    t_values = list(range(t_max + 1))
    rows = []

    for t in t_values:
        # Step through t+1 recurrent steps, recording trajectory at each step
        test_acts = np.zeros((actual_test, n_neurons))
        ext = test_examples.copy()  # held stimulus

        trajectory_all = []  # decoded class at each step, per image
        overlap_all = []     # overlap vector at each step, per image

        for step in range(t + 1):
            if stimulus_mode == "cue_only" and step > 0:
                # After first step, stimulus is removed
                test_acts = _k_cap(test_acts @ W + bias, cap_size)
            else:
                test_acts = _k_cap(test_acts @ W + ext @ A + bias, cap_size)

            # Decode per image at this step
            step_preds = []
            step_overlaps = []
            for i in range(actual_test):
                response = np.array([
                    test_acts[i, idx[j * cap_size:(j + 1) * cap_size]].sum() / cap_size
                    for j in range(10)
                ])
                step_preds.append(int(np.argmax(response)))
                step_overlaps.append(response.tolist())
            trajectory_all.append(step_preds)
            overlap_all.append(step_overlaps)

        # Write per-example rows with full trajectory
        for i in range(actual_test):
            traj = [trajectory_all[s][i] for s in range(t + 1)]
            ov_traj = [overlap_all[s][i] for s in range(t + 1)]
            final_overlaps = np.array(overlap_all[-1][i])
            pred = traj[-1]
            target = int(test_lbl_sub[i])
            correct = pred == target
            correct_overlap = float(final_overlaps[target])
            wrong_mask = np.ones(10, dtype=bool)
            wrong_mask[target] = False
            strongest_wrong = float(final_overlaps[wrong_mask].max()) if wrong_mask.any() else 0.0
            margin = correct_overlap - strongest_wrong

            rows.append({
                "experiment": "mnist",
                "seed": seed,
                "theta_id": f"nb_n{n_neurons}_k{cap_size}_beta{beta}_r{n_rounds}",
                "n": n_neurons,
                "k": cap_size,
                "p": sparsity,
                "beta": beta,
                "t": t,
                "instance_id": i,
                "target": target,
                "prediction": pred,
                "correct": correct,
                "overlaps": json.dumps(final_overlaps.tolist()),
                "correct_overlap": correct_overlap,
                "strongest_wrong_overlap": strongest_wrong,
                "margin": margin,
                "trajectory": json.dumps(traj),
                "overlap_trajectory": json.dumps(ov_traj),
                "stimulus_mode": stimulus_mode,
                "recurrent": recurrent,
                "plasticity_on": False,
                "family": "MNIST_NB",
                "model_name": model_name,
                "suite": "mnist-nb",
                "list_type": "MNIST",
                "N": 10,
                "num_train_lists": n_examples_total,
                "num_test_lists": actual_test,
                "k_train_min": 1,
                "k_train_max": 1,
                "k_test": t,
                "accuracy": 1.0 if correct else 0.0,
                "internal_steps": t,
                "params": None,
                "runtime_ms": None,
            })

    return rows
