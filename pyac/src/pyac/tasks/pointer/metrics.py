from __future__ import annotations

import numpy as np

from pyac.core.network import Network

from pyac.tasks.pointer.protocol import PointerTask, evaluate_seen_lists


def accuracy_vs_hop(
    network: Network,
    task: PointerTask,
    lists: list[np.ndarray],
    k_values: list[int],
    samples_per_list: int,
    rng: np.random.Generator,
    model_name: str = "AC-Seen",
    settle_steps: int = 1,
) -> list[dict[str, int | float | str]]:
    records: list[dict[str, int | float | str]] = []
    for hop in k_values:
        accuracy = evaluate_seen_lists(
            network,
            task,
            lists,
            samples_per_list=samples_per_list,
            k=hop,
            rng=rng,
            settle_steps=settle_steps,
        )
        records.append(
            {
                "List Type": "Seen",
                "Model": model_name,
                "Num Lists": len(lists),
                "N": task.list_length,
                "k": int(hop),
                "Accuracy": float(accuracy),
                "Internal Steps": int(settle_steps + hop),
            }
        )
    return records
