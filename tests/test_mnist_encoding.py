from __future__ import annotations

import numpy as np
import pytest


def test_pixel_encoder_is_deterministic_and_sparse():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

    rng = np.random.default_rng(1)
    encoder = PixelAssemblyEncoder(
        num_pixels=4, neurons_per_pixel=3, active_pixels=2, rng=rng
    )
    image = np.array([[0.1, 0.9], [0.7, 0.2]], dtype=np.float64)

    first = encoder.encode(image)
    second = encoder.encode(image)

    assert first.indices.tolist() == second.indices.tolist()
    assert first.area_name == "X"
    assert first.indices.size == 6


def test_pixel_encoder_uses_intensity_rank_not_label_information():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

    encoder = PixelAssemblyEncoder(
        num_pixels=4,
        neurons_per_pixel=2,
        active_pixels=1,
        rng=np.random.default_rng(2),
    )
    image = np.array([[0.0, 0.1], [0.2, 1.0]], dtype=np.float64)

    assembly = encoder.encode(image)
    expected_pool = set(encoder.pixel_indices[3].tolist())

    assert set(assembly.indices.tolist()) == expected_pool


def test_pixel_encoder_rejects_wrong_pixel_count():
    from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

    encoder = PixelAssemblyEncoder(num_pixels=4, neurons_per_pixel=2, active_pixels=1)

    with pytest.raises(ValueError, match="4 pixels"):
        encoder.encode(np.zeros((3, 3), dtype=np.float64))
