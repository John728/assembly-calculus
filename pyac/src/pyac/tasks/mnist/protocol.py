from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyac.core.network import Network
from pyac.core.types import AreaSpec, Assembly, FiberSpec, NetworkSpec
from pyac.measures.overlap import class_overlap_vector, correct_class_margin
from pyac.tasks.mnist.encoding import PixelAssemblyEncoder


@dataclass
class MnistTask:
    encoder: PixelAssemblyEncoder
    area_map: dict[str, str]
    class_assemblies: dict[int, Assembly] = field(default_factory=dict)
    n: int = 0
    k: int = 0
    p: float = 0.0
    beta: float = 0.0
    presentation_rounds: int = 0
    settle_steps: int = 0


def build_mnist_network(
    n: int,
    k: int,
    p: float,
    beta: float,
    rng: np.random.Generator,
    *,
    encoder: PixelAssemblyEncoder | None = None,
) -> tuple[Network, MnistTask]:
    encoder = encoder or PixelAssemblyEncoder(rng=rng, area_name="X")
    if encoder.area_name != "X":
        raise ValueError("encoder area_name must be 'X'")

    sensory_n = encoder.num_pixels * encoder.neurons_per_pixel
    sensory_k = encoder.active_pixels * encoder.neurons_per_pixel

    spec = NetworkSpec(
        areas=[
            AreaSpec(name="X", n=sensory_n, k=sensory_k, dynamics_type="feedforward"),
            AreaSpec(name="Y", n=n, k=k, p_recurrent=p, dynamics_type="recurrent"),
        ],
        fibers=[FiberSpec(src="X", dst="Y", p_fiber=p)],
        beta=beta,
        step_order=["X", "Y"],
    )
    network = Network(spec=spec, rng=rng)
    task = MnistTask(
        encoder=encoder,
        area_map={"sensory": "X", "coding": "Y"},
        n=n,
        k=k,
        p=p,
        beta=beta,
    )
    return network, task


def train_mnist_assemblies(
    network: Network,
    task: MnistTask,
    images: np.ndarray,
    labels: np.ndarray,
    presentation_rounds: int = 1,
    settle_steps: int = 1,
) -> None:
    if presentation_rounds <= 0:
        raise ValueError("presentation_rounds must be > 0")
    if settle_steps <= 0:
        raise ValueError("settle_steps must be > 0")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    labels_array = np.asarray(labels)
    if not np.issubdtype(labels_array.dtype, np.integer) or np.any(
        (labels_array < 0) | (labels_array > 9)
    ):
        raise ValueError("labels must be MNIST digits in 0..9")

    sensory_area = task.area_map["sensory"]
    coding_area = task.area_map["coding"]
    sensory_n = network.areas_by_name[sensory_area].n
    coding_area_spec = network.areas_by_name[coding_area]
    label_counts: dict[int, np.ndarray] = {}

    for _ in range(presentation_rounds):
        for image, label in zip(images, labels):
            sensory_assembly = task.encoder.encode(image)
            if sensory_assembly.area_name != sensory_area:
                raise ValueError(
                    f"encoded assembly belongs to '{sensory_assembly.area_name}', "
                    f"expected '{sensory_area}'"
                )
            stimulus = np.zeros(sensory_n, dtype=np.float64)
            stimulus[sensory_assembly.indices] = 1.0

            network.activations[sensory_area] = np.array([], dtype=np.int64)
            network.activations[coding_area] = np.array([], dtype=np.int64)

            for _ in range(settle_steps):
                network.step(
                    external_stimuli={sensory_area: stimulus},
                    plasticity_on=True,
                )

            digit = int(label)
            counts = label_counts.setdefault(
                digit, np.zeros(coding_area_spec.n, dtype=np.int64)
            )
            counts[network.activations[coding_area]] += 1

    task.class_assemblies.clear()
    for digit, counts in label_counts.items():
        prototype_indices = np.argsort(-counts, kind="stable")[: coding_area_spec.k]
        task.class_assemblies[digit] = Assembly(
            area_name=coding_area,
            indices=prototype_indices.astype(np.int64, copy=False),
        )

    task.presentation_rounds = presentation_rounds
    task.settle_steps = settle_steps


def _reset_mnist_evaluation_state(network: Network, task: MnistTask) -> None:
    for area_name in (task.area_map["sensory"], task.area_map["coding"]):
        network.activations[area_name] = np.array([], dtype=np.int64)
    network.step_count = 0


def _mnist_stimulus(network: Network, task: MnistTask, image: np.ndarray) -> np.ndarray:
    sensory_area = task.area_map["sensory"]
    sensory_assembly = task.encoder.encode(image)
    if sensory_assembly.area_name != sensory_area:
        raise ValueError(
            f"encoded assembly belongs to '{sensory_assembly.area_name}', "
            f"expected '{sensory_area}'"
        )

    stimulus = np.zeros(network.areas_by_name[sensory_area].n, dtype=np.float64)
    stimulus[sensory_assembly.indices] = 1.0
    return stimulus


def _mnist_overlap_vector(active: Assembly, task: MnistTask) -> list[float]:
    overlaps = class_overlap_vector(active, task.class_assemblies, num_classes=10)
    return overlaps.tolist()


def decode_mnist_class(active: Assembly, task: MnistTask) -> int:
    overlaps = class_overlap_vector(active, task.class_assemblies, num_classes=10)
    return int(np.argmax(overlaps))


def _validate_mnist_evaluation_inputs(task: MnistTask, target: int) -> int:
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or not 0 <= int(target) <= 9:
        raise ValueError("target must be an MNIST digit in 0..9")

    missing = set(range(10)) - set(task.class_assemblies)
    if missing:
        raise ValueError("MNIST evaluation requires class assemblies for all 10 digits")

    coding_area = task.area_map["coding"]
    for digit in range(10):
        assembly = task.class_assemblies[digit]
        if assembly.area_name != coding_area or assembly.indices.size == 0:
            raise ValueError(
                "MNIST evaluation requires non-empty class assemblies in the coding area"
            )

    return int(target)


def evaluate_mnist_example(
    network: Network,
    task: MnistTask,
    image: np.ndarray,
    target: int,
    instance_id: object,
    t: int,
    stimulus_mode: str = "held",
) -> dict[str, object]:
    if t < 0:
        raise ValueError("t must be >= 0")
    if stimulus_mode not in {"held", "transient"}:
        raise ValueError("stimulus_mode must be 'held' or 'transient'")
    target_digit = _validate_mnist_evaluation_inputs(task, target)

    sensory_area = task.area_map["sensory"]
    coding_area = task.area_map["coding"]
    stimulus = _mnist_stimulus(network, task, image)
    _reset_mnist_evaluation_state(network, task)

    trajectory: list[int] = []
    overlap_trajectory: list[list[float]] = []

    for step_index in range(t + 1):
        stimuli = None
        if stimulus_mode == "held" or step_index == 0:
            stimuli = {sensory_area: stimulus}

        network.step(external_stimuli=stimuli, plasticity_on=False)
        active = network.get_assembly(coding_area)
        trajectory.append(decode_mnist_class(active, task))
        overlap_trajectory.append(_mnist_overlap_vector(active, task))

    final_overlaps = np.asarray(overlap_trajectory[-1], dtype=np.float64)
    margin = correct_class_margin(final_overlaps, target_digit)
    prediction = trajectory[-1]

    row = {
        "experiment": "mnist",
        "seed": getattr(task, "seed", None),
        "theta_id": getattr(task, "theta_id", None),
        "n": task.n,
        "k": task.k,
        "p": task.p,
        "beta": task.beta,
        "t": t,
        "instance_id": instance_id,
        "target": target_digit,
        "prediction": prediction,
        "correct": prediction == target_digit,
        "overlaps": final_overlaps.tolist(),
        "correct_overlap": margin.correct_overlap,
        "strongest_wrong_overlap": margin.strongest_wrong_overlap,
        "margin": margin.margin,
        "trajectory": trajectory,
        "overlap_trajectory": overlap_trajectory,
        "stimulus_mode": stimulus_mode,
        "plasticity_on": False,
    }
    if hasattr(task, "task_seed"):
        row["task_seed"] = getattr(task, "task_seed")
    return row


def evaluate_mnist_t_sweep(
    network: Network,
    task: MnistTask,
    images: np.ndarray,
    labels: np.ndarray,
    t_values: list[int],
    instance_ids: list[object] | None = None,
    stimulus_mode: str = "held",
) -> list[dict[str, object]]:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if instance_ids is not None and len(instance_ids) != len(images):
        raise ValueError("instance_ids must have the same length as images")

    ids = instance_ids if instance_ids is not None else list(range(len(images)))
    rows: list[dict[str, object]] = []
    for image, label, instance_id in zip(images, labels, ids):
        for t in t_values:
            rows.append(
                evaluate_mnist_example(
                    network,
                    task,
                    image,
                    label,
                    instance_id=instance_id,
                    t=int(t),
                    stimulus_mode=stimulus_mode,
                )
            )
    return rows
