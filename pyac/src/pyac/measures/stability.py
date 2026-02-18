"""
Assembly stability and convergence trace measures.

Tracks overlap convergence and detects stabilization.
"""

from typing import TYPE_CHECKING

import numpy as np

from pyac.core.operations import project
from pyac.measures.overlap import assembly_overlap

if TYPE_CHECKING:
    from pyac.core.network import Network


def convergence_trace(
    network: "Network",
    src_area: str,
    dst_area: str,
    stimulus: np.ndarray,
    t_steps: int,
) -> list[float]:
    if t_steps < 1:
        raise ValueError("t_steps must be positive")
    
    overlaps: list[float] = []
    prev_assembly = None
    
    for step_idx in range(t_steps):
        current_assembly = project(
            network,
            src_area=src_area,
            dst_area=dst_area,
            stimulus=stimulus if step_idx == 0 else None,
            t_internal=1,
            plasticity_on=True,
            clamp_src=True,
        )
        
        if prev_assembly is None:
            overlaps.append(1.0)
        else:
            overlap = assembly_overlap(prev_assembly, current_assembly)
            overlaps.append(overlap)
        
        prev_assembly = current_assembly
    
    return overlaps


def is_stable(
    overlaps: list[float],
    threshold: float = 0.95,
    window: int = 5,
) -> bool:
    if not overlaps:
        return False
    
    start_idx = max(0, len(overlaps) - window)
    recent_overlaps = overlaps[start_idx:]
    
    return all(ov >= threshold for ov in recent_overlaps)
