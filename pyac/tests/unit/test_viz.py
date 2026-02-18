from __future__ import annotations

import pytest

from pyac.viz.plotting import (
    plot_accuracy_vs_features,
    plot_convergence,
    plot_dfa_accuracy,
    plot_ipc_capacity,
    plot_overlap_trace,
)


def test_all_viz_stubs_raise_not_implemented() -> None:
    """All viz stubs raise NotImplementedError."""
    stubs = [
        (plot_overlap_trace, [[0.1, 0.2, 0.3]]),
        (plot_ipc_capacity, [{}]),
        (plot_accuracy_vs_features, [[]]),
        (plot_dfa_accuracy, [[]]),
        (plot_convergence, [{}]),
    ]
    
    for fn, args in stubs:
        with pytest.raises(NotImplementedError, match="post-MVP"):
            fn(*args)


def test_all_viz_stubs_have_docstrings() -> None:
    """All viz stubs have docstrings."""
    stubs = [
        plot_overlap_trace,
        plot_ipc_capacity,
        plot_accuracy_vs_features,
        plot_dfa_accuracy,
        plot_convergence,
    ]
    
    for fn in stubs:
        assert fn.__doc__ is not None, f"{fn.__name__} missing docstring"
        assert "STUB" in fn.__doc__ or "stub" in fn.__doc__, f"{fn.__name__} docstring missing STUB marker"
