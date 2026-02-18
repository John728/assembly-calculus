"""
IPC basis functions using Legendre polynomials.

Provides functions for:
- legendre_basis: Evaluate Legendre polynomial P_degree(x)
- generate_input_signal: Generate uniform [-1, 1] input signal
- generate_targets: Generate target functions as products of delayed Legendre evaluations
"""

import numpy as np
from numpy.random import Generator
from scipy.special import eval_legendre


def legendre_basis(degree: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Legendre polynomial P_degree at points x.

    Uses scipy.special.eval_legendre for numerically stable evaluation.

    Args:
        degree: Polynomial degree (non-negative integer).
        x: Points at which to evaluate; typically in [-1, 1].

    Returns:
        Array of Legendre polynomial values with dtype float64.
    """
    return eval_legendre(degree, x).astype(np.float64)


def generate_input_signal(length: int, rng: Generator) -> np.ndarray:
    """
    Generate uniform random signal in [-1, 1].

    Args:
        length: Number of samples.
        rng: Explicit np.random.Generator for reproducibility (G2 compliance).

    Returns:
        1D array of uniform random values in [-1, 1] with dtype float64.
    """
    return rng.uniform(-1, 1, size=length).astype(np.float64)


def generate_targets(
    input_signal: np.ndarray,
    max_degree: int,
    max_delay: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]]:
    """
    Generate target functions as products of delayed Legendre evaluations.

    For each combination of degrees d_0, d_1, ... and delays τ_0, τ_1, ...,
    generates target z_t = ∏_i P_{d_i}(u_{t-τ_i}).

    Args:
        input_signal: 1D array of input values (typically from generate_input_signal).
        max_degree: Maximum polynomial degree (0 to max_degree inclusive).
        max_delay: Maximum delay in steps (1 to max_delay inclusive).

    Returns:
        List of tuples (degrees, delays, target_array) where:
        - degrees: tuple of polynomial degrees
        - delays: tuple of delays in steps
        - target_array: 1D numpy array of target values (float64)
    """
    targets = []

    if max_delay <= 0:
        return targets

    signal_length = len(input_signal)

    # Generate single-degree targets (unary products)
    for degree in range(max_degree + 1):
        for delay in range(1, max_delay + 1):
            if delay >= signal_length:
                continue
            delayed_signal = input_signal[delay:]
            target = legendre_basis(degree, delayed_signal)
            targets.append(((degree,), (delay,), target))

    # Generate multi-degree targets (pairwise products)
    for degree1 in range(max_degree + 1):
        for degree2 in range(degree1, max_degree + 1):
            for delay1 in range(1, max_delay + 1):
                for delay2 in range(1, max_delay + 1):
                    if degree1 == degree2 and delay1 >= delay2:
                        continue
                    max_d = max(delay1, delay2)
                    if max_d >= signal_length:
                        continue
                    target_len = signal_length - max_d
                    p1 = legendre_basis(degree1, input_signal[delay1:delay1 + target_len])
                    p2 = legendre_basis(degree2, input_signal[delay2:delay2 + target_len])
                    target = p1 * p2
                    targets.append(((degree1, degree2), (delay1, delay2), target))

    return targets
