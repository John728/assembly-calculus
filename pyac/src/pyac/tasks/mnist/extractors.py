from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from pyac.core.network import Network
from pyac.core.rng import spawn_rngs
from pyac.core.types import AreaSpec, NetworkSpec
from pyac.tasks.mnist.encoders import COLT2022Encoder


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _as_flat_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape == (28, 28):
        return arr.reshape(784)
    if arr.shape == (784,):
        return arr
    raise ValueError(f"image must be shape (28, 28) or (784,), got {arr.shape}")


def _as_square_image(image: np.ndarray) -> np.ndarray:
    flat = _as_flat_image(image)
    return flat.reshape(28, 28)


def _stack_images(images: list[np.ndarray]) -> np.ndarray:
    return np.vstack([_as_flat_image(image) for image in images]).astype(np.float64, copy=False)


def _indices_to_stimulus(indices: np.ndarray, n: int) -> np.ndarray:
    stimulus = np.zeros(n, dtype=np.float64)
    if indices.size > 0:
        stimulus[indices] = 1.0
    return stimulus


def _reset_network_activations(network: Network) -> None:
    for area_name in network.area_names:
        network.activations[area_name] = np.array([], dtype=np.int64)


def _snapshot_inhibition(network: Network) -> dict[str, bool]:
    return {
        area_name: network.inhibition_state.is_inhibited(area_name)
        for area_name in network.area_names
    }


def _restore_inhibition(network: Network, snapshot: dict[str, bool]) -> None:
    for area_name in network.area_names:
        if snapshot.get(area_name, False):
            network.inhibit(area_name)
        else:
            network.disinhibit(area_name)


def _set_only_area_active(network: Network, area_name: str) -> None:
    for candidate in network.area_names:
        if candidate == area_name:
            network.disinhibit(candidate)
        else:
            network.inhibit(candidate)


class FeatureExtractor(ABC):
    @abstractmethod
    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_features(self) -> int:
        raise NotImplementedError


class LinearExtractor(FeatureExtractor):
    def __init__(self, m: int):
        _validate_positive("m", m)
        self.m = m
        self.weights: np.ndarray | None = None

    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        _ = images
        _ = labels
        weights = rng.standard_normal((784, self.m)).astype(np.float64)
        self.weights = weights

    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        _ = rng
        if self.weights is None:
            raise RuntimeError("must call fit() before transform()")
        X = _stack_images(images)
        return X @ self.weights

    @property
    def n_features(self) -> int:
        return self.m


class NonlinearExtractor(FeatureExtractor):
    def __init__(self, m: int):
        _validate_positive("m", m)
        self.m = m
        self.weights: np.ndarray | None = None
        self.theta: np.ndarray | None = None

    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        _ = labels
        X = _stack_images(images)
        weights = rng.binomial(1, 0.5, size=(784, self.m)).astype(np.float64)
        self.weights = weights
        mean_pixel = float(X.mean())
        self.theta = mean_pixel * self.weights.sum(axis=0)

    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        _ = rng
        if self.weights is None or self.theta is None:
            raise RuntimeError("must call fit() before transform()")
        X = _stack_images(images)
        return (X @ self.weights > self.theta).astype(np.float64)

    @property
    def n_features(self) -> int:
        return self.m


class LargeAreaExtractor(FeatureExtractor):
    def __init__(
        self,
        m: int,
        beta: float = 1.0,
        t_internal: int = 5,
        n_examples_per_class: int = 5,
    ):
        _validate_positive("m", m)
        _validate_positive("t_internal", t_internal)
        _validate_positive("n_examples_per_class", n_examples_per_class)

        self.m = m
        self.beta = beta
        self.t_internal = t_internal
        self.n_examples_per_class = n_examples_per_class
        self.area_name = "features"

        self.encoder: COLT2022Encoder | None = None
        self.network: Network | None = None

    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        encoder_rng, network_rng, train_rng = spawn_rngs(rng, 3)

        k = max(1, self.m // 10)
        self.encoder = COLT2022Encoder(n_neurons=self.m, k_stimulus=k, rng=encoder_rng)
        self.network = Network(
            NetworkSpec(
                areas=[
                    AreaSpec(
                        name=self.area_name,
                        n=self.m,
                        k=k,
                        dynamics_type="refracted",
                    )
                ],
                fibers=[],
                beta=self.beta,
            ),
            network_rng,
        )

        labels_arr = np.asarray(labels, dtype=np.int64)
        for digit in range(10):
            class_indices = np.flatnonzero(labels_arr == digit)
            if class_indices.size == 0:
                continue
            selected = train_rng.choice(
                class_indices,
                size=min(self.n_examples_per_class, class_indices.size),
                replace=False,
            )

            for idx in selected:
                image = _as_square_image(images[int(idx)])
                stimulus_indices = self.encoder.encode(image, train_rng)
                stimulus = _indices_to_stimulus(stimulus_indices, self.m)
                for _ in range(self.t_internal):
                    self.network.step(
                        external_stimuli={self.area_name: stimulus},
                        plasticity_on=True,
                    )

            self.network.normalize(self.area_name)

    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        if self.encoder is None or self.network is None:
            raise RuntimeError("must call fit() before transform()")

        features = np.zeros((len(images), self.m), dtype=np.float64)
        for row, image_raw in enumerate(images):
            _reset_network_activations(self.network)
            image = _as_square_image(image_raw)
            stimulus_indices = self.encoder.encode(image, rng)
            stimulus = _indices_to_stimulus(stimulus_indices, self.m)
            for _ in range(self.t_internal):
                self.network.step(
                    external_stimuli={self.area_name: stimulus},
                    plasticity_on=False,
                )

            fired = self.network.activations[self.area_name]
            if fired.size > 0:
                features[row, fired] = 1.0
        return features

    @property
    def n_features(self) -> int:
        return self.m


class RandomAssemblyExtractor(FeatureExtractor):
    def __init__(
        self,
        m: int,
        beta: float = 1.0,
        t_internal: int = 5,
        n_examples_per_class: int = 5,
    ):
        _validate_positive("m", m)
        _validate_positive("t_internal", t_internal)
        _validate_positive("n_examples_per_class", n_examples_per_class)
        if m % 100 != 0:
            raise ValueError("m must be a multiple of 100")

        self.m = m
        self.n_areas = m // 100
        self.beta = beta
        self.t_internal = t_internal
        self.n_examples_per_class = n_examples_per_class

        self.area_names = [f"area_{idx}" for idx in range(self.n_areas)]
        self.encoder: COLT2022Encoder | None = None
        self.network: Network | None = None

    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        encoder_rng, network_rng, train_rng = spawn_rngs(rng, 3)

        self.encoder = COLT2022Encoder(n_neurons=100, k_stimulus=10, rng=encoder_rng)
        areas = [
            AreaSpec(name=area_name, n=100, k=10, dynamics_type="refracted")
            for area_name in self.area_names
        ]
        self.network = Network(NetworkSpec(areas=areas, fibers=[], beta=self.beta), network_rng)

        labels_arr = np.asarray(labels, dtype=np.int64)
        for area_name in self.area_names:
            class_order = train_rng.permutation(10)
            for digit in class_order:
                class_indices = np.flatnonzero(labels_arr == digit)
                if class_indices.size == 0:
                    continue
                selected = train_rng.choice(
                    class_indices,
                    size=min(self.n_examples_per_class, class_indices.size),
                    replace=False,
                )

                inhibited = _snapshot_inhibition(self.network)
                try:
                    _set_only_area_active(self.network, area_name)
                    for idx in selected:
                        image = _as_square_image(images[int(idx)])
                        stimulus_indices = self.encoder.encode(image, train_rng)
                        stimulus = _indices_to_stimulus(stimulus_indices, 100)
                        for _ in range(self.t_internal):
                            self.network.step(
                                external_stimuli={area_name: stimulus},
                                plasticity_on=True,
                            )
                finally:
                    _restore_inhibition(self.network, inhibited)

                self.network.normalize(area_name)

    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        if self.encoder is None or self.network is None:
            raise RuntimeError("must call fit() before transform()")

        features = np.zeros((len(images), self.n_features), dtype=np.float64)
        for row, image_raw in enumerate(images):
            _reset_network_activations(self.network)
            image = _as_square_image(image_raw)
            stimulus_indices = self.encoder.encode(image, rng)
            stimulus = _indices_to_stimulus(stimulus_indices, 100)

            for area_idx, area_name in enumerate(self.area_names):
                inhibited = _snapshot_inhibition(self.network)
                try:
                    _set_only_area_active(self.network, area_name)
                    for _ in range(self.t_internal):
                        self.network.step(
                            external_stimuli={area_name: stimulus},
                            plasticity_on=False,
                        )
                finally:
                    _restore_inhibition(self.network, inhibited)

                fired = self.network.activations[area_name]
                if fired.size > 0:
                    start = area_idx * 100
                    features[row, start + fired] = 1.0

        return features

    @property
    def n_features(self) -> int:
        return self.m


class SplitAssemblyExtractor(FeatureExtractor):
    def __init__(
        self,
        m: int,
        beta: float = 1.0,
        t_internal: int = 5,
        n_examples_per_class: int = 5,
    ):
        _validate_positive("m", m)
        _validate_positive("t_internal", t_internal)
        _validate_positive("n_examples_per_class", n_examples_per_class)
        if m % 10 != 0:
            raise ValueError("m must be a multiple of 10")
        if m < 100:
            raise ValueError("m must be >= 100")

        self.m = m
        self.n_per_area = m // 10
        self.k_per_area = max(1, m // 100)
        self.beta = beta
        self.t_internal = t_internal
        self.n_examples_per_class = n_examples_per_class
        self.area_names = [f"digit_{digit}" for digit in range(10)]

        self.encoder: COLT2022Encoder | None = None
        self.network: Network | None = None

    def fit(self, images: list[np.ndarray], labels: np.ndarray, rng: Generator) -> None:
        encoder_rng, network_rng, train_rng = spawn_rngs(rng, 3)

        self.encoder = COLT2022Encoder(
            n_neurons=self.n_per_area,
            k_stimulus=self.k_per_area,
            rng=encoder_rng,
        )
        areas = [
            AreaSpec(
                name=area_name,
                n=self.n_per_area,
                k=self.k_per_area,
                dynamics_type="refracted",
            )
            for area_name in self.area_names
        ]
        self.network = Network(NetworkSpec(areas=areas, fibers=[], beta=self.beta), network_rng)

        labels_arr = np.asarray(labels, dtype=np.int64)
        for digit in range(10):
            area_name = self.area_names[digit]
            class_indices = np.flatnonzero(labels_arr == digit)
            if class_indices.size == 0:
                continue
            selected = train_rng.choice(
                class_indices,
                size=min(self.n_examples_per_class, class_indices.size),
                replace=False,
            )

            inhibited = _snapshot_inhibition(self.network)
            try:
                _set_only_area_active(self.network, area_name)
                for idx in selected:
                    image = _as_square_image(images[int(idx)])
                    stimulus_indices = self.encoder.encode(image, train_rng)
                    stimulus = _indices_to_stimulus(stimulus_indices, self.n_per_area)
                    for _ in range(self.t_internal):
                        self.network.step(
                            external_stimuli={area_name: stimulus},
                            plasticity_on=True,
                        )
            finally:
                _restore_inhibition(self.network, inhibited)

            self.network.normalize(area_name)

    def transform(self, images: list[np.ndarray], rng: Generator) -> np.ndarray:
        if self.encoder is None or self.network is None:
            raise RuntimeError("must call fit() before transform()")

        features = np.zeros((len(images), self.m), dtype=np.float64)
        for row, image_raw in enumerate(images):
            _reset_network_activations(self.network)
            image = _as_square_image(image_raw)
            stimulus_indices = self.encoder.encode(image, rng)
            stimulus = _indices_to_stimulus(stimulus_indices, self.n_per_area)

            for area_idx, area_name in enumerate(self.area_names):
                inhibited = _snapshot_inhibition(self.network)
                try:
                    _set_only_area_active(self.network, area_name)
                    for _ in range(self.t_internal):
                        self.network.step(
                            external_stimuli={area_name: stimulus},
                            plasticity_on=False,
                        )
                finally:
                    _restore_inhibition(self.network, inhibited)

                fired = self.network.activations[area_name]
                if fired.size > 0:
                    start = area_idx * self.n_per_area
                    features[row, start + fired] = 1.0

        return features

    @property
    def n_features(self) -> int:
        return self.m
