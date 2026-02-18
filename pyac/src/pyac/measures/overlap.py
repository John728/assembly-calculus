"""
Assembly overlap and intersection measures.

Provides Jaccard similarity and raw intersection metrics for assemblies.
"""

import numpy as np

from pyac.core.types import Assembly


def _validate_same_area(asm_a: Assembly, asm_b: Assembly) -> None:
    if asm_a.area_name != asm_b.area_name:
        raise ValueError(f"area name mismatch: {asm_a.area_name} != {asm_b.area_name}")


def assembly_intersection_size(asm_a: Assembly, asm_b: Assembly) -> int:
    _validate_same_area(asm_a, asm_b)
    
    if asm_a.indices.size == 0 or asm_b.indices.size == 0:
        return 0
    
    intersection = np.intersect1d(asm_a.indices, asm_b.indices)
    return int(intersection.size)


def assembly_overlap(asm_a: Assembly, asm_b: Assembly) -> float:
    _validate_same_area(asm_a, asm_b)
    
    if asm_a.indices.size == 0 and asm_b.indices.size == 0:
        return 0.0
    
    if asm_a.indices.size == 0 or asm_b.indices.size == 0:
        return 0.0
    
    intersection = np.intersect1d(asm_a.indices, asm_b.indices)
    union = np.union1d(asm_a.indices, asm_b.indices)
    
    if union.size == 0:
        return 0.0
    
    return float(intersection.size) / float(union.size)
