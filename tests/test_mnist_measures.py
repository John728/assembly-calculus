from __future__ import annotations

import numpy as np


def test_class_overlap_vector_and_margin_are_measured_from_assemblies():
    from pyac.core.types import Assembly
    from pyac.measures.overlap import class_overlap_vector, correct_class_margin

    active = Assembly("Y", np.array([0, 1, 2, 10]))
    prototypes = {
        0: Assembly("Y", np.array([0, 1, 2, 3])),
        1: Assembly("Y", np.array([10, 11, 12, 13])),
    }

    overlaps = class_overlap_vector(active, prototypes, num_classes=2)
    margin = correct_class_margin(overlaps, target=0)

    assert overlaps.tolist() == [0.75, 0.25]
    assert margin.correct_overlap == 0.75
    assert margin.strongest_wrong_overlap == 0.25
    assert margin.margin == 0.5
