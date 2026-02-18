"""Plotting utilities (stubs for post-MVP implementation)."""

from __future__ import annotations


def plot_overlap_trace(
    overlaps: list[float],
    title: str = "Assembly Overlap",
    ax=None,
) -> None:
    """
    Plot overlap trace over time (STUB).

    Args:
        overlaps: List of overlap values (Jaccard similarity) over time steps.
        title: Plot title.
        ax: Matplotlib axes object (optional).

    Raises:
        NotImplementedError: Viz is a post-MVP feature.
    """
    _ = overlaps
    _ = title
    _ = ax
    raise NotImplementedError("Viz is a post-MVP feature. Use matplotlib directly for now.")


def plot_ipc_capacity(
    capacities: dict,
    title: str = "IPC Capacity",
    ax=None,
) -> None:
    """
    Plot IPC capacity vs polynomial degree (STUB).

    Args:
        capacities: Dict mapping polynomial degree to capacity value.
        title: Plot title.
        ax: Matplotlib axes object (optional).

    Raises:
        NotImplementedError: Viz is a post-MVP feature.
    """
    _ = capacities
    _ = title
    _ = ax
    raise NotImplementedError("Viz is a post-MVP feature. Use matplotlib directly for now.")


def plot_accuracy_vs_features(
    results: list[dict],
    title: str = "Accuracy vs Features",
    ax=None,
) -> None:
    """
    Plot accuracy vs feature count for multiple extractors (STUB).

    Args:
        results: List of dicts with keys: extractor, n_features, accuracy.
        title: Plot title.
        ax: Matplotlib axes object (optional).

    Raises:
        NotImplementedError: Viz is a post-MVP feature.
    """
    _ = results
    _ = title
    _ = ax
    raise NotImplementedError("Viz is a post-MVP feature. Use matplotlib directly for now.")


def plot_dfa_accuracy(
    results: list[dict],
    title: str = "DFA Accuracy",
    ax=None,
) -> None:
    """
    Plot DFA accuracy vs presentations (STUB).

    Args:
        results: List of dicts with keys: n_presentations, accuracy.
        title: Plot title.
        ax: Matplotlib axes object (optional).

    Raises:
        NotImplementedError: Viz is a post-MVP feature.
    """
    _ = results
    _ = title
    _ = ax
    raise NotImplementedError("Viz is a post-MVP feature. Use matplotlib directly for now.")


def plot_convergence(
    traces: dict[str, list[float]],
    title: str = "Convergence",
    ax=None,
) -> None:
    """
    Plot convergence traces for multiple areas (STUB).

    Args:
        traces: Dict mapping area name to list of convergence values over time.
        title: Plot title.
        ax: Matplotlib axes object (optional).

    Raises:
        NotImplementedError: Viz is a post-MVP feature.
    """
    _ = traces
    _ = title
    _ = ax
    raise NotImplementedError("Viz is a post-MVP feature. Use matplotlib directly for now.")
