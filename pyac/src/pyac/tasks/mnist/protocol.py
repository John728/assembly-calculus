from __future__ import annotations

import numpy as np
from numpy.random import Generator

from pyac.core.network import Network
from pyac.core.rng import make_rng
from pyac.core.types import Assembly
from pyac.measures.overlap import assembly_overlap
from pyac.tasks.mnist.encoders import MNISTEncoder


def _validate_area_exists(network: Network, area_name: str) -> None:
    if area_name not in network.area_names:
        raise ValueError(f"unknown area name: {area_name}")


def _validate_images(images: np.ndarray) -> None:
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"images must have shape (n, 28, 28), got {images.shape}")


def _validate_labels(labels: np.ndarray, n_images: int) -> None:
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (n,), got {labels.shape}")
    if labels.shape[0] != n_images:
        raise ValueError("labels length must match number of images")

    label_set = set(np.unique(labels).tolist())
    if not label_set.issubset(set(range(10))):
        raise ValueError("labels must contain only digits 0-9")


def _stimulus_from_indices(area_n: int, indices: np.ndarray) -> np.ndarray:
    stimulus = np.zeros(area_n, dtype=np.float64)
    if indices.size > 0:
        stimulus[indices] = 1.0
    return stimulus


def train_assemblies(
    network: Network,
    area_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    encoder: MNISTEncoder,
    t_per_image: int = 5,
    homeostasis_between_classes: bool = True,
    rng: Generator | None = None,
) -> dict[int, Assembly]:
    _validate_area_exists(network, area_name)
    _validate_images(images)
    _validate_labels(labels, images.shape[0])

    if t_per_image <= 0:
        raise ValueError("t_per_image must be > 0")

    class_labels = set(np.unique(labels).tolist())
    missing = set(range(10)) - class_labels
    if missing:
        raise ValueError(f"labels must include all classes 0-9, missing {sorted(missing)}")

    effective_rng = make_rng(0) if rng is None else rng
    area_n = network.areas_by_name[area_name].n
    assemblies: dict[int, Assembly] = {}

    for class_label in range(10):
        class_images = images[labels == class_label]
        for image in class_images:
            stimulus_indices = encoder.encode(image, effective_rng)
            stimulus = _stimulus_from_indices(area_n, stimulus_indices)
            for _ in range(t_per_image):
                _ = network.step(
                    external_stimuli={area_name: stimulus},
                    plasticity_on=True,
                )

        if homeostasis_between_classes:
            network.normalize(area_name)

        assemblies[class_label] = network.get_assembly(area_name)

    return assemblies


def extract_features(
    network: Network,
    area_name: str,
    images: np.ndarray,
    encoder: MNISTEncoder,
    assemblies: dict[int, Assembly],
    rng: Generator,
) -> np.ndarray:
    _validate_area_exists(network, area_name)
    _validate_images(images)

    ordered_labels = sorted(assemblies.keys())
    if not ordered_labels:
        raise ValueError("assemblies must not be empty")

    area_n = network.areas_by_name[area_name].n
    features = np.zeros((images.shape[0], len(ordered_labels)), dtype=np.float64)

    for image_idx, image in enumerate(images):
        stimulus_indices = encoder.encode(image, rng)
        stimulus = _stimulus_from_indices(area_n, stimulus_indices)
        _ = network.step(external_stimuli={area_name: stimulus}, plasticity_on=False)
        activation = network.get_assembly(area_name)

        for class_idx, class_label in enumerate(ordered_labels):
            features[image_idx, class_idx] = assembly_overlap(
                activation,
                assemblies[class_label],
            )

    return features


def classify(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
) -> dict[str, float | int]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features_train, labels_train)

    return {
        "train_accuracy": float(clf.score(features_train, labels_train)),
        "test_accuracy": float(clf.score(features_test, labels_test)),
        "n_train": int(labels_train.shape[0]),
        "n_test": int(labels_test.shape[0]),
    }
