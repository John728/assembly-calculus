from __future__ import annotations

import numpy as np

from pyac.core.types import Assembly


class PixelAssemblyEncoder:
    """Encode image intensities into fixed sensory assemblies."""

    def __init__(
        self,
        num_pixels: int = 28 * 28,
        neurons_per_pixel: int = 8,
        active_pixels: int = 64,
        rng: np.random.Generator | None = None,
        area_name: str = "X",
    ) -> None:
        if num_pixels <= 0:
            raise ValueError("num_pixels must be > 0")
        if neurons_per_pixel <= 0:
            raise ValueError("neurons_per_pixel must be > 0")
        if active_pixels <= 0 or active_pixels > num_pixels:
            raise ValueError("active_pixels must be in [1, num_pixels]")

        self.num_pixels = num_pixels
        self.neurons_per_pixel = neurons_per_pixel
        self.active_pixels = active_pixels
        self.area_name = area_name

        indices = np.arange(num_pixels * neurons_per_pixel, dtype=np.int64)
        if rng is not None:
            indices = rng.permutation(indices)
        self.pixel_indices = indices.reshape(num_pixels, neurons_per_pixel)

    def encode(self, image: np.ndarray) -> Assembly:
        flat = np.asarray(image).reshape(-1)
        if flat.size != self.num_pixels:
            raise ValueError(f"image must contain {self.num_pixels} pixels")

        active_pixel_indices = np.argsort(-flat, kind="stable")[: self.active_pixels]
        indices = np.unique(self.pixel_indices[active_pixel_indices].reshape(-1))
        return Assembly(area_name=self.area_name, indices=indices)
