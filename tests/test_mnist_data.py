from __future__ import annotations

import gzip
import struct

import numpy as np
import pytest


def _write_idx_images(path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(
            struct.pack(
                ">IIII", 2051, images.shape[0], images.shape[1], images.shape[2]
            )
        )
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, labels.shape[0]))
        handle.write(labels.astype(np.uint8).tobytes())


def test_load_mnist_idx_reads_real_file_format(tmp_path):
    from pyac.tasks.mnist.data import load_mnist_split

    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    labels = np.array([3, 7], dtype=np.uint8)
    _write_idx_images(tmp_path / "train-images-idx3-ubyte.gz", images)
    _write_idx_labels(tmp_path / "train-labels-idx1-ubyte.gz", labels)

    split = load_mnist_split(tmp_path, split="train")

    assert split.images.shape == (2, 28, 28)
    assert split.images.dtype == np.float64
    assert split.labels.tolist() == [3, 7]
    assert split.images.max() <= 1.0


def test_load_mnist_split_fails_when_real_files_missing(tmp_path):
    from pyac.tasks.mnist.data import load_mnist_split

    with pytest.raises(FileNotFoundError, match="MNIST"):
        load_mnist_split(tmp_path, split="train")
