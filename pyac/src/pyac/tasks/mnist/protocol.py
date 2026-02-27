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


def train_disjoint_assemblies(
    network: Network,
    stimulus_area_name: str,
    target_area_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    encoder: MNISTEncoder,
    n_rounds: int = 5,
    bias_penalty: float = -1.0,
    input_multiplier: float = 5.0,
    rng: Generator | None = None,
) -> tuple[dict[int, Assembly], np.ndarray]:
    """
    Train disjoint assemblies class-by-class using a bias penalty.
    """
    _validate_area_exists(network, target_area_name)
    _validate_area_exists(network, stimulus_area_name)
    _validate_images(images)
    _validate_labels(labels, images.shape[0])

    effective_rng = make_rng(0) if rng is None else rng
    area_n = network.areas_by_name[target_area_name].n
    assemblies: dict[int, Assembly] = {}
    
    activations = np.zeros((10, n_rounds, area_n))
    bias = np.zeros(area_n)

    for class_label in range(10):
        class_images = images[labels == class_label][:n_rounds]
        
        for a_name in network.area_names:
            network.activations[a_name] = np.array([], dtype=np.int64)

        for j in range(len(class_images)):
            image = class_images[j]
            stimulus_indices = encoder.encode(image, effective_rng)
            stimulus = _stimulus_from_indices(network.areas_by_name[stimulus_area_name].n, stimulus_indices) * input_multiplier
            
            _ = network.step(
                external_stimuli={stimulus_area_name: stimulus},
                plasticity_on=True,
                biases={target_area_name: bias}
            )

            act_h_idx = network.activations[target_area_name].copy()
            act_h_new = np.zeros(area_n)
            act_h_new[act_h_idx] = 1.0
            activations[class_label, j] = act_h_new

        bias[network.activations[target_area_name]] += bias_penalty

        network.weights[(target_area_name, target_area_name)].setdiag(0)
        network.weights[(target_area_name, target_area_name)].eliminate_zeros()
        network.normalize(target_area_name)

        assemblies[class_label] = network.get_assembly(target_area_name)

    return assemblies, activations


def generate_rollout_tensor(
    network: Network,
    stimulus_area_name: str,
    target_area_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    encoder: MNISTEncoder,
    n_rounds: int,
    n_examples: int,
    input_multiplier: float = 5.0,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    Generate the legacy readout tensor from a trained network.
    """
    _validate_area_exists(network, target_area_name)
    _validate_area_exists(network, stimulus_area_name)
    _validate_images(images)
    _validate_labels(labels, images.shape[0])

    effective_rng = make_rng(0) if rng is None else rng
    area_n = network.areas_by_name[target_area_name].n
    
    outputs = np.zeros((10, n_rounds + 1, n_examples, area_n), dtype=np.uint8)
    
    for class_label in range(10):
        class_images = images[labels == class_label][:n_examples]
        for ex in range(len(class_images)):
            image = class_images[ex]
            stimulus_indices = encoder.encode(image, effective_rng)
            stimulus = _stimulus_from_indices(network.areas_by_name[stimulus_area_name].n, stimulus_indices) * input_multiplier
            
            for a_name in network.area_names:
                network.activations[a_name] = np.array([], dtype=np.int64)
                
            for j in range(n_rounds):
                _ = network.step(external_stimuli={stimulus_area_name: stimulus}, plasticity_on=False)
                prev_idx = network.activations[target_area_name].copy()
                outputs[class_label, j + 1, ex, prev_idx] = 1
                
    return outputs
