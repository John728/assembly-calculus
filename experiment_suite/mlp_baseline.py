from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ExplicitMLP(nn.Module):
    def __init__(self, N: int, K_max: int, num_layers: int, hidden_dim: int):
        super().__init__()
        self.N = N
        self.K_max = K_max

        in_dim = (N * N) + N + K_max
        layers: list[nn.Module] = []
        proj_dim = max(hidden_dim, in_dim * 2)
        layers.append(nn.Linear(in_dim, proj_dim))
        layers.append(nn.GELU())

        curr_dim = proj_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, N))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        p = batch["p"]
        s = batch["s"]
        k = batch["k"]
        batch_size = p.size(0)

        p_one_hot = F.one_hot(p, num_classes=self.N).float()
        p_flat = p_one_hot.view(batch_size, -1)
        s_one_hot = F.one_hot(s, num_classes=self.N).float()
        k_zero_indexed = torch.clamp(k - 1, min=0, max=self.K_max - 1)
        k_one_hot = F.one_hot(k_zero_indexed, num_classes=self.K_max).float()
        x = torch.cat([p_flat, s_one_hot, k_one_hot], dim=-1)
        return self.mlp(x), 1


def generate_full_cycle(n: int) -> torch.Tensor:
    nodes = np.random.permutation(n)
    cycle = np.zeros(n, dtype=np.int64)
    for idx in range(n - 1):
        cycle[nodes[idx]] = nodes[idx + 1]
    cycle[nodes[-1]] = nodes[0]
    return torch.from_numpy(cycle).long()


def generate_unique_lists(num_lists: int, n: int) -> list[torch.Tensor]:
    lists: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    while len(lists) < num_lists:
        pointer = generate_full_cycle(n)
        key = tuple(pointer.tolist())
        if key in seen:
            continue
        seen.add(key)
        lists.append(pointer)
    return lists


class RandomListPermutationDataset(Dataset[dict[str, int | torch.Tensor]]):
    def __init__(self, lists: list[torch.Tensor], samples_per_list: int, k_range: tuple[int, int], fixed_k: int | None = None):
        super().__init__()
        self.lists = lists
        self.samples_per_list = samples_per_list
        self.k_range = k_range
        self.fixed_k = fixed_k
        self.N = len(lists[0])
        self.size = len(self.lists) * self.samples_per_list

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, int | torch.Tensor]:
        del idx
        possible_ks = list(range(self.k_range[0], self.k_range[1] + 1))
        list_idx = torch.randint(0, len(self.lists), (1,)).item()
        pointer = self.lists[list_idx]
        start = torch.randint(0, self.N, (1,)).item()
        if self.fixed_k is not None:
            hops = self.fixed_k
        else:
            hops = possible_ks[torch.randint(0, len(possible_ks), (1,)).item()]

        x = start
        for _ in range(hops):
            x = pointer[x].item()

        return {"p": pointer, "s": start, "k": hops, "y": x}


def collate_dict(batch: list[dict[str, int | torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "p": torch.stack([item["p"] for item in batch if isinstance(item["p"], torch.Tensor)]),
        "s": torch.tensor([int(item["s"]) for item in batch]),
        "k": torch.tensor([int(item["k"]) for item in batch]),
        "y": torch.tensor([int(item["y"]) for item in batch]),
    }


def train_model(model: nn.Module, train_loader, args, device: torch.device) -> nn.Module:
    model.train()
    model_lr = args.lr
    if sum(p.numel() for p in model.parameters()) > 1_000_000:
        model_lr = min(model_lr, 1e-4)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for _ in range(args.epochs):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def eval_model(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = model(batch)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch["y"]).sum().item())
            total += int(batch["y"].size(0))
    return correct / max(total, 1)
