from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_suite.jobs import ExperimentJob
from experiment_suite.mlp_baseline import ExplicitMLP, RandomListPermutationDataset, collate_dict, eval_model, generate_unique_lists, train_model
from experiment_suite.schema import standardize_mlp_row


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_mlp_job(job: ExperimentJob) -> list[dict[str, object]]:
    _set_seed(job.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_lists = generate_unique_lists(job.condition.num_train_lists + job.condition.num_test_lists, job.condition.N)
    train_lists = all_lists[: job.condition.num_train_lists]
    test_lists = all_lists[job.condition.num_train_lists :]

    model_values = job.model.values
    samples_per_list_train = int(model_values.get("samples_per_list_train", 50))
    samples_per_list_eval = int(model_values.get("samples_per_list_eval", 50))
    batch_size = int(model_values.get("batch_size", 128))
    lr = float(model_values.get("lr", 1e-3))
    epochs = int(model_values.get("epochs", 10))
    hidden_dim = int(model_values.get("hidden_dim", 64))
    layers = int(model_values.get("layers", 2))

    model = ExplicitMLP(
        N=job.condition.N,
        K_max=max(job.condition.k_train_max, job.condition.k_test_max),
        num_layers=layers,
        hidden_dim=hidden_dim,
    ).to(device)

    train_args = type(
        "TrainArgs",
        (),
        {"lr": lr, "epochs": epochs},
    )()

    ds_train = RandomListPermutationDataset(
        train_lists,
        samples_per_list=samples_per_list_train,
        k_range=(job.condition.k_train_min, job.condition.k_train_max),
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_dict)
    train_model(model, loader_train, train_args, device)

    eval_lists = train_lists if job.condition.list_type == "Seen" else test_lists
    if not eval_lists:
        raise ValueError(f"No evaluation lists available for list_type={job.condition.list_type}")

    params = sum(p.numel() for p in model.parameters())
    rows: list[dict[str, object]] = []
    for hop in range(job.condition.k_test_min, job.condition.k_test_max + 1):
        ds_eval = RandomListPermutationDataset(
            eval_lists,
            samples_per_list=samples_per_list_eval,
            k_range=(hop, hop),
            fixed_k=hop,
        )
        loader_eval = DataLoader(ds_eval, batch_size=batch_size, collate_fn=collate_dict)
        accuracy = eval_model(model, loader_eval, device)
        raw_row = {
            "List Type": job.condition.list_type,
            "Model": str(model_values.get("model_name", job.model.model_name)),
            "Layers": layers,
            "Dim": hidden_dim,
            "Params": params,
            "k": hop,
            "Accuracy": accuracy,
            "LR": lr,
            "Epochs": epochs,
        }
        rows.append(
            standardize_mlp_row(
                raw_row,
                suite=job.suite_name,
                seed=job.seed,
                N=job.condition.N,
                num_train_lists=job.condition.num_train_lists,
                num_test_lists=job.condition.num_test_lists,
                k_train_min=job.condition.k_train_min,
                k_train_max=job.condition.k_train_max,
            )
        )
    return rows
