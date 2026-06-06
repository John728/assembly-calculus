"""
Assembly overlap and intersection measures.

Provides Jaccard similarity, raw intersection, and class prototype metrics
for assemblies.
"""

from dataclasses import dataclass

import numpy as np

from pyac.core.types import Assembly


@dataclass(frozen=True)
class ClassMargin:
    correct_overlap: float
    strongest_wrong_overlap: float
    margin: float


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


def class_overlap_vector(
    active: Assembly, prototypes: dict[int, Assembly], num_classes: int
) -> np.ndarray:
    overlaps = np.zeros(num_classes, dtype=float)

    for class_idx in range(num_classes):
        prototype = prototypes.get(class_idx)
        if prototype is None or prototype.indices.size == 0:
            continue

        overlaps[class_idx] = assembly_intersection_size(active, prototype) / float(
            prototype.indices.size
        )

    return overlaps


def correct_class_margin(overlaps: np.ndarray, target: int) -> ClassMargin:
    correct_overlap = float(overlaps[target])
    wrong_overlaps = np.delete(overlaps, target)
    strongest_wrong_overlap = float(wrong_overlaps.max()) if wrong_overlaps.size else 0.0

    return ClassMargin(
        correct_overlap=correct_overlap,
        strongest_wrong_overlap=strongest_wrong_overlap,
        margin=correct_overlap - strongest_wrong_overlap,
    )
