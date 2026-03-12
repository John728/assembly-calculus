from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def generate_full_cycle(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        raise ValueError("n must be > 1")

    nodes = rng.permutation(n)
    cycle = np.zeros(n, dtype=np.int64)
    for idx in range(n - 1):
        cycle[nodes[idx]] = nodes[idx + 1]
    cycle[nodes[-1]] = nodes[0]
    return cycle


def generate_unique_lists(num_lists: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    if num_lists <= 0:
        raise ValueError("num_lists must be > 0")

    lists: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(lists) < num_lists:
        pointer = generate_full_cycle(n, rng)
        key = tuple(int(value) for value in pointer.tolist())
        if key in seen:
            continue
        seen.add(key)
        lists.append(pointer)
    return lists


def follow_pointer(pointer: Sequence[int] | np.ndarray, start: int, hops: int) -> int:
    if hops < 0:
        raise ValueError("hops must be >= 0")

    pointer_arr = np.asarray(pointer, dtype=np.int64)
    current = int(start)
    for _ in range(hops):
        current = int(pointer_arr[current])
    return current


def sample_pointer_examples(
    lists: Sequence[Sequence[int] | np.ndarray],
    samples_per_list: int,
    k: int,
    rng: np.random.Generator,
) -> list[dict[str, int | np.ndarray]]:
    if samples_per_list <= 0:
        raise ValueError("samples_per_list must be > 0")

    records: list[dict[str, int | np.ndarray]] = []
    for list_idx, pointer in enumerate(lists):
        pointer_arr = np.asarray(pointer, dtype=np.int64)
        n = int(pointer_arr.shape[0])
        for _ in range(samples_per_list):
            start = int(rng.integers(0, n))
            target = follow_pointer(pointer_arr, start=start, hops=k)
            records.append(
                {
                    "list_idx": list_idx,
                    "pointer": pointer_arr,
                    "start": start,
                    "k": k,
                    "target": target,
                }
            )
    return records
