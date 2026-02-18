"""
RNG management for pyac.

Provides deterministic, reproducible random number generation.
- make_rng: Create a seeded Generator
- spawn_rngs: Create independent child Generators from a parent
- get_entropy: Extract entropy from a Generator for artifact logging

Enforces:
- G2 (No global RNG): Each rng object is explicit, no module-level default_rng()
- Determinism: Same seed → identical random sequences
- Independence: Spawned RNGs produce non-overlapping sequences
"""

import numpy as np
from typing import Union, Optional


def make_rng(
    seed: Union[int, np.random.SeedSequence, None] = None,
) -> np.random.Generator:
    """
    Create a seeded numpy Generator backed by PCG64.

    Enforces deterministic, reproducible random number generation:
    - Same seed always produces identical sequences
    - No global state or side effects
    - Each call returns a fresh, independent Generator

    Args:
        seed: Seed for the Generator.
            - int: Converted to SeedSequence(seed)
            - SeedSequence: Used directly to initialize Generator
            - None: Creates SeedSequence() with random OS entropy

    Returns:
        np.random.Generator backed by PCG64 bit generator.

    Examples:
        >>> rng = make_rng(42)  # Deterministic from int seed
        >>> rng.random()  # Returns same value across calls with seed=42
        0.77395605...

        >>> rng = make_rng()  # Random entropy from OS
        >>> rng.random()  # Different value each call

        >>> seed_seq = np.random.SeedSequence(999)
        >>> rng = make_rng(seed_seq)  # Explicit SeedSequence
    """
    # Handle different seed types
    if seed is None:
        # Create SeedSequence with random entropy from OS
        seed_seq = np.random.SeedSequence()
    elif isinstance(seed, int):
        # Convert int to SeedSequence
        seed_seq = np.random.SeedSequence(seed)
    elif isinstance(seed, np.random.SeedSequence):
        # Use provided SeedSequence directly
        seed_seq = seed
    else:
        raise TypeError(
            f"seed must be int, SeedSequence, or None, got {type(seed)}"
        )

    # Create Generator backed by PCG64
    return np.random.Generator(np.random.PCG64(seed_seq))


def spawn_rngs(
    parent: Union[np.random.SeedSequence, np.random.Generator],
    n: int,
) -> list[np.random.Generator]:
    """
    Spawn n independent child Generators from a parent.

    Creates n new, independent random number streams suitable for
    parallel execution or multiple stochastic processes. Children
    are guaranteed to have non-overlapping state.

    Args:
        parent: Parent SeedSequence or Generator.
            - Generator: Extracts parent.bit_generator.seed_seq
            - SeedSequence: Used directly
        n: Number of child Generators to spawn.

    Returns:
        List of n independent np.random.Generator objects backed by PCG64.

    Examples:
        >>> parent = make_rng(42)
        >>> children = spawn_rngs(parent, 3)
        >>> len(children)
        3
        >>> [c.random() for c in children]  # All different values
        [0.77..., 0.44..., 0.99...]
    """
    # Extract SeedSequence from Generator if needed
    if isinstance(parent, np.random.Generator):
        seed_seq = parent.bit_generator.seed_seq
    elif isinstance(parent, np.random.SeedSequence):
        seed_seq = parent
    else:
        raise TypeError(
            f"parent must be SeedSequence or Generator, got {type(parent)}"
        )

    # Spawn n child SeedSequences
    child_seqs = seed_seq.spawn(n)

    # Create Generator for each child SeedSequence
    return [np.random.Generator(np.random.PCG64(seq)) for seq in child_seqs]


def get_entropy(rng: np.random.Generator) -> Union[int, tuple]:
    """
    Extract entropy from a Generator for artifact logging.

    Returns the underlying entropy used to initialize the Generator's
    SeedSequence. Useful for recording the random seed in artifacts
    for reproducibility.

    Args:
        rng: A np.random.Generator instance.

    Returns:
        The entropy tuple/int from rng.bit_generator.seed_seq.entropy.

    Examples:
        >>> rng = make_rng(42)
        >>> entropy = get_entropy(rng)
        >>> entropy is not None
        True
    """
    return rng.bit_generator.seed_seq.entropy
