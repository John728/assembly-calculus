from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class MnistSplit:
    images: np.ndarray
    labels: np.ndarray


_FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def _open_idx(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _require_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / f"{filename}.gz"
    if path.exists():
        return path

    path = data_dir / filename
    if path.exists():
        return path

    raise FileNotFoundError(f"MNIST file not found: {filename}(.gz) in {data_dir}")


def _read_images(path: Path) -> np.ndarray:
    with _open_idx(path) as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid MNIST image header in {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic in {path}: {magic}")

        payload = handle.read()
    expected = count * rows * cols
    if len(payload) != expected:
        raise ValueError(
            f"Invalid MNIST image payload length in {path}: "
            f"expected {expected}, found {len(payload)}"
        )

    images = np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, cols)
    return images.astype(np.float64) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid MNIST label header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic in {path}: {magic}")

        payload = handle.read()
    if len(payload) != count:
        raise ValueError(
            f"Invalid MNIST label payload length in {path}: "
            f"expected {count}, found {len(payload)}"
        )

    return np.frombuffer(payload, dtype=np.uint8).astype(int)


def load_mnist_split(data_dir: str | Path, split: str) -> MnistSplit:
    try:
        image_filename, label_filename = _FILES[split]
    except KeyError as exc:
        raise ValueError(f"Unknown MNIST split: {split}") from exc

    data_path = Path(data_dir)
    image_path = _require_file(data_path, image_filename)
    label_path = _require_file(data_path, label_filename)
    images = _read_images(image_path)
    labels = _read_labels(label_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"MNIST image/label count mismatch: {images.shape[0]} images, "
            f"{labels.shape[0]} labels"
        )
    return MnistSplit(images=images, labels=labels)
