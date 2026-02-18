from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import override

import numpy as np
from scipy.sparse import csr_matrix

IndexArray = np.ndarray
Context = Mapping[str, object]


def _to_index_array(values: object) -> IndexArray:
    return np.asarray(values, dtype=np.int64)


class DynamicsStrategy(ABC):
    @abstractmethod
    def dynamics_contribution(
        self,
        area_name: str,
        activations: IndexArray,
        context: Context,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        area_name: str,
        firing: IndexArray,
        total_input: np.ndarray,
        plasticity: float,
        context: Context,
    ) -> None:
        raise NotImplementedError


class FeedforwardStrategy(DynamicsStrategy):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n: int = n

    @override
    def dynamics_contribution(
        self,
        area_name: str,
        activations: IndexArray,
        context: Context,
    ) -> np.ndarray:
        return np.zeros(self.n, dtype=np.float64)

    @override
    def update_state(
        self,
        area_name: str,
        firing: IndexArray,
        total_input: np.ndarray,
        plasticity: float,
        context: Context,
    ) -> None:
        return


class RecurrentStrategy(DynamicsStrategy):
    def __init__(self, recurrent_weights: csr_matrix):
        self.recurrent_weights: csr_matrix = recurrent_weights

    @override
    def dynamics_contribution(
        self,
        area_name: str,
        activations: IndexArray,
        context: Context,
    ) -> np.ndarray:
        active = _to_index_array(activations)
        return self.recurrent_weights[active].sum(axis=0).A1

    @override
    def update_state(
        self,
        area_name: str,
        firing: IndexArray,
        total_input: np.ndarray,
        plasticity: float,
        context: Context,
    ) -> None:
        pre = _to_index_array(context.get("pre_activations", firing))
        post = _to_index_array(firing)
        if pre.size == 0 or post.size == 0:
            return
        self.recurrent_weights[np.ix_(pre, post)] *= 1.0 + plasticity


class RefractedStrategy(DynamicsStrategy):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.bias: np.ndarray = np.zeros(n, dtype=np.float64)

    @override
    def dynamics_contribution(
        self,
        area_name: str,
        activations: IndexArray,
        context: Context,
    ) -> np.ndarray:
        return -self.bias

    @override
    def update_state(
        self,
        area_name: str,
        firing: IndexArray,
        total_input: np.ndarray,
        plasticity: float,
        context: Context,
    ) -> None:
        fired = _to_index_array(firing)
        if fired.size == 0:
            return
        self.bias[fired] += total_input[fired] * plasticity
