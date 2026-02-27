"""
MNIST encoder module: convert 28x28 images to neuron index stimuli.

Provides pluggable ABC architecture for different encoding strategies:
- MNISTEncoder: Abstract base class defining encode interface
- COLT2022Encoder: Random projection encoder (COLT 2022 paper approach)
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator


class MNISTEncoder(ABC):
    """
    Abstract base class for MNIST image encoders.
    
    Encoders convert 28x28 pixel images into sets of neuron indices
    for use as stimuli in assembly calculus networks.
    """

    @abstractmethod
    def encode(self, image: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Encode 28x28 image into neuron indices.
        
        Args:
            image: 28x28 float array (grayscale pixel intensities)
            rng: Explicit RNG for stochastic encoding (if needed)
            
        Returns:
            1D int64 array of unique neuron indices in [0, n_neurons)
        """
        pass


class COLT2022Encoder(MNISTEncoder):
    """
    Random projection encoder following COLT 2022 paper approach.
    
    Projects 784-pixel image through random Gaussian matrix to n_neurons
    activations, then selects top k_stimulus neurons as the stimulus.
    
    Args:
        n_neurons: Total neurons in target area
        k_stimulus: Number of neurons to activate (sparsity parameter)
        rng: RNG for initializing projection matrix
    """

    def __init__(self, n_neurons: int, k_stimulus: int, rng: Generator):
        if n_neurons <= 0:
            raise ValueError("n_neurons must be > 0")
        if k_stimulus <= 0:
            raise ValueError("k_stimulus must be > 0")
        if k_stimulus > n_neurons:
            raise ValueError("k_stimulus must be <= n_neurons")

        self.n_neurons = n_neurons
        self.k_stimulus = k_stimulus

        self.projection_matrix = rng.normal(
            0.0, 1.0, size=(n_neurons, 784)
        ).astype(np.float64)

    def encode(self, image: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Encode image via random projection and top-k selection.
        
        Process:
        1. Flatten 28x28 image to 784 pixels
        2. Normalize: pixels / (sum + epsilon) to avoid div-by-zero
        3. Project: activations = projection_matrix @ normalized_pixels
        4. Select top k_stimulus neurons via argpartition
        5. Sort indices (Assembly convention)
        
        Args:
            image: 28x28 float array
            rng: RNG parameter (unused in this encoder, kept for interface)
            
        Returns:
            Sorted int64 array of k_stimulus unique neuron indices
        """
        if image.shape != (28, 28):
            raise ValueError(f"image must be (28, 28), got {image.shape}")

        pixels = image.flatten().astype(np.float64)

        pixel_sum = pixels.sum()
        if pixel_sum > 0.0:
            normalized_pixels = pixels / pixel_sum
        else:
            normalized_pixels = pixels / (pixel_sum + 1e-10)

        activations = self.projection_matrix @ normalized_pixels

        if self.k_stimulus == self.n_neurons:
            top_k_indices = np.arange(self.n_neurons, dtype=np.int64)
        else:
            top_k_indices = np.argpartition(-activations, self.k_stimulus)[
                : self.k_stimulus
            ]

        sorted_indices = np.sort(top_k_indices).astype(np.int64)

        return sorted_indices


class KCapSmoothedEncoder(MNISTEncoder):
    """
    Smoothing encoder following the legacy NEMO/AC approach.
    
    Applies a 3x3 uniform convolution filter to the image to spread
    activation, then uses k-cap to select the top active neurons.
    
    Args:
        cap_size: Number of neurons to activate.
    """

    def __init__(self, cap_size: int):
        if cap_size <= 0:
            raise ValueError("cap_size must be > 0")
        if cap_size > 784:
            raise ValueError("cap_size must be <= 784")

        self.cap_size = cap_size

    def encode(self, image: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Encode image via 3x3 smoothing and top-k selection.
        
        Process:
        1. Apply 3x3 uniform convolution.
        2. Flatten to 784 pixels.
        3. Select top cap_size neurons via argpartition.
        4. Sort indices (Assembly convention).
        """
        from scipy.signal import convolve
        
        if image.shape != (28, 28):
            raise ValueError(f"image must be (28, 28), got {image.shape}")

        smooth = convolve(image.reshape(1, 28, 28), np.ones((1, 3, 3)), mode='same').reshape(784)
        
        if self.cap_size == 784:
            top_k_indices = np.arange(784, dtype=np.int64)
        else:
            top_k_indices = np.argpartition(-smooth, self.cap_size)[: self.cap_size]

        sorted_indices = np.sort(top_k_indices).astype(np.int64)
        return sorted_indices

