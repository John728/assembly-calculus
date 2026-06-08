from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def _make_fake_rollout_trace(
    start_node: int,
    hops: int,
    internal_steps: int,
    decoded_sequence: list[int],
) -> dict[str, object]:
    return {
        "start_node": start_node,
        "hops": hops,
        "intermediate_nodes": [int(d) for d in decoded_sequence[1:]],
        "current_state_nodes": [int(d) for d in decoded_sequence],
        "cur_nodes": [int(d) for d in decoded_sequence],
        "src_nodes": [int(d) for d in decoded_sequence],
        "dst_nodes": [int(d) for d in decoded_sequence],
        "final_prediction": int(decoded_sequence[-1]),
        "external_cue_count": 1,
        "internal_steps": internal_steps,
        "hop_ctrl_states": [max(hops, 0)] * (len(decoded_sequence)),
        "hop_ctrl_source": "network",
        "controller_mode": "internal",
        "mechanism_route": "CUR->SRC->DST->CUR",
        "write_mode": "plastic",
        "write_steps": 3,
    }


def test_evaluate_proper_unseen_per_instance_returns_theory_rows() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    pointers = [pointer]

    # Correct decoded sequence: [start=0, 1, 2, 3] (length 4 = t+1 for t=3)
    decoded = [0, 1, 2, 3]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=3, internal_steps=3, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=3,
            time_budget=3,
            samples_per_list=1,
            rng=np.random.default_rng(42),
            theta_id="theta-001",
        )

    assert len(rows) == 1
    row = rows[0]

    # Theory schema fields
    assert row["experiment"] == "pointer_chasing"
    assert row["pointer_variant"] == "proper_unseen"
    assert row["theta_id"] == "theta-001"
    assert row["N"] == 6
    assert row["L"] == 3
    assert row["t"] == 3
    assert row["c"] == 1  # original PC advances one symbolic transition per update
    assert row["readout_step"] == 3
    assert row["completed_hops"] == 3
    assert isinstance(row["instance_id"], str)
    assert row["start_node"] == 0
    assert row["target"] == 3  # 0->1->2->3
    assert row["prediction"] == 3
    assert row["correct"] is True

    # Trajectories
    true_traj = row["true_trajectory"]
    assert isinstance(true_traj, list)
    assert len(true_traj) == 4  # L+1
    assert true_traj == [0, 1, 2, 3]

    decoded_traj = row["trajectory"]
    assert isinstance(decoded_traj, list)
    assert len(decoded_traj) == 4
    assert decoded_traj == [0, 1, 2, 3]

    # Path accuracy: all 4 states match
    assert row["path_accuracy"] == 1.0

    # First error: none
    assert row["first_error_index"] is None

    # Proper-unseen evaluation writes the episode plastically, then freezes rollout.
    assert row["plasticity_on"] is True
    assert row["episodic_write_plasticity_on"] is True
    assert row["rollout_plasticity_on"] is False


def test_evaluate_proper_unseen_per_instance_detects_path_errors() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer = np.array([1, 2, 3, 0, 5, 4], dtype=np.int64)
    pointers = [pointer]

    # Decoded: [0, 1, 2, 5, 5] (wrong at hop 3, internal steps=4)
    decoded = [0, 1, 2, 5, 5]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=3, internal_steps=4, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=3,
            time_budget=4,
            samples_per_list=1,
            rng=np.random.default_rng(42),
            theta_id="theta-002",
        )

    assert len(rows) == 1
    row = rows[0]

    assert row["L"] == 3
    assert row["t"] == 4
    assert row["c"] == 1
    assert row["readout_step"] == 3
    assert row["completed_hops"] == 3

    # True: 0->1->2->3
    assert row["target"] == 3
    assert row["true_trajectory"] == [0, 1, 2, 3]

    # Decoded at hop boundaries (c=1): steps 0,1,2,3 -> states [0,1,2,5]
    decoded_traj = row["trajectory"]
    assert len(decoded_traj) == 4
    assert decoded_traj == [0, 1, 2, 5]

    # path_accuracy: 3/4 correct (states at positions 0,1,2 match, position 3 is wrong)
    assert row["path_accuracy"] == 0.75

    # first_error_index = 3 (the failed hop)
    assert row["first_error_index"] == 3

    # Final prediction is the last decoded state (at hop L)
    assert row["prediction"] == 5
    assert row["correct"] is False


def test_evaluate_proper_unseen_per_instance_handles_t_less_than_L_as_incomplete_execution() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer = np.array([1, 2, 3, 0, 5, 4], dtype=np.int64)
    pointers = [pointer]

    # Original PC advances one transition per recurrent update. With t=1 and L=3,
    # the model has executed only the first transition, so the future path states are missing.
    decoded = [0, 1]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=3, internal_steps=1, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=3,
            time_budget=1,
            samples_per_list=1,
            rng=np.random.default_rng(42),
        )

    assert len(rows) == 1
    row = rows[0]

    assert row["L"] == 3
    assert row["t"] == 1
    assert row["c"] == 1
    assert row["readout_step"] == 1
    assert row["completed_hops"] == 1

    assert row["target"] == 3  # 0->1->2->3
    assert row["true_trajectory"] == [0, 1, 2, 3]
    assert row["prediction"] == 1  # current state after the available one-step execution
    assert row["correct"] is False

    # path_accuracy: states 0 and 1 are correct; future required states are missing and count wrong.
    assert row["trajectory"] == [0, 1, None, None]
    assert row["path_accuracy"] == 0.5  # 2/4

    assert row["first_error_index"] == 2


def test_evaluate_proper_unseen_per_instance_uses_budget_without_forcing_overshoot() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    pointers = [pointer]

    # t=5 gives more budget than needed for L=2. The Theory experiment measures
    # whether the target depth can be reached within the budget, so readout is at hop L,
    # not after forced overshooting to M^5(s0).
    decoded = [0, 1, 2, 3, 4, 5]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=2, internal_steps=5, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=2,
            time_budget=5,
            samples_per_list=1,
            rng=np.random.default_rng(42),
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["L"] == 2
    assert row["t"] == 5
    assert row["c"] == 1
    assert row["readout_step"] == 2
    assert row["completed_hops"] == 2
    assert row["target"] == 2
    assert row["prediction"] == 2
    assert row["correct"] is True
    assert row["true_trajectory"] == [0, 1, 2]
    assert row["trajectory"] == [0, 1, 2]
    assert row["path_accuracy"] == 1.0
    assert row["first_error_index"] is None


def test_evaluate_seen_per_instance_uses_original_recurrent_step_budget() -> None:
    from pyac.tasks.pointer.protocol import PointerTask, evaluate_seen_per_instance

    task = PointerTask(
        num_lists=1,
        list_length=5,
        assembly_size=8,
        area_map={"input": "input", "state": "state"},
        token_to_key=[],
        input_assemblies={},
        state_assemblies={},
    )
    pointer = np.array([1, 2, 3, 4, 0], dtype=np.int64)

    with patch(
        "pyac.tasks.pointer.protocol.rollout_seen_pointer_sequence",
        return_value=[0, 1, 2, 3, 4, 0],
    ):
        rows = evaluate_seen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=[pointer],
            hops=3,
            time_budget=5,
            samples_per_list=1,
            rng=np.random.default_rng(42),
            theta_id="seen-theta",
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment"] == "pointer_chasing"
    assert row["pointer_variant"] == "seen"
    assert row["theta_id"] == "seen-theta"
    assert row["N"] == 5
    assert row["L"] == 3
    assert row["t"] == 5
    assert row["c"] == 1
    assert row["completed_hops"] == 3
    assert row["readout_step"] == 3
    assert row["target"] == 3
    assert row["prediction"] == 3
    assert row["correct"] is True
    assert row["true_trajectory"] == [0, 1, 2, 3]
    assert row["trajectory"] == [0, 1, 2, 3]
    assert row["path_accuracy"] == 1.0
    assert row["first_error_index"] is None


def test_evaluate_seen_per_instance_marks_missing_future_states_wrong() -> None:
    from pyac.tasks.pointer.protocol import PointerTask, evaluate_seen_per_instance

    task = PointerTask(
        num_lists=1,
        list_length=5,
        assembly_size=8,
        area_map={"input": "input", "state": "state"},
        token_to_key=[],
        input_assemblies={},
        state_assemblies={},
    )
    pointer = np.array([1, 2, 3, 4, 0], dtype=np.int64)

    with patch(
        "pyac.tasks.pointer.protocol.rollout_seen_pointer_sequence",
        return_value=[0, 1],
    ):
        rows = evaluate_seen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=[pointer],
            hops=3,
            time_budget=1,
            samples_per_list=1,
            rng=np.random.default_rng(42),
        )

    row = rows[0]
    assert row["completed_hops"] == 1
    assert row["readout_step"] == 1
    assert row["target"] == 3
    assert row["prediction"] == 1
    assert row["correct"] is False
    assert row["true_trajectory"] == [0, 1, 2, 3]
    assert row["trajectory"] == [0, 1, None, None]
    assert row["path_accuracy"] == 0.5
    assert row["first_error_index"] == 2


def test_evaluate_proper_unseen_per_instance_respects_samples_per_list() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=1,
        assembly_size=4,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer = np.array([0], dtype=np.int64)
    pointers = [pointer]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=2, internal_steps=2, decoded_sequence=[0, 1, 2]),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=2,
            time_budget=2,
            samples_per_list=3,
            rng=np.random.default_rng(42),
        )

    assert len(rows) == 3  # 1 list * 3 samples
    instance_ids = {row["instance_id"] for row in rows}
    assert len(instance_ids) == 3  # each instance has unique id

    for row in rows:
        assert row["N"] == 1
        assert row["L"] == 2
        assert row["t"] == 2
        assert row["c"] == 1
        assert row["start_node"] is not None


def test_evaluate_seen_per_instance_uses_unique_ids_when_start_repeats() -> None:
    from pyac.tasks.pointer.protocol import PointerTask, evaluate_seen_per_instance

    task = PointerTask(
        num_lists=1,
        list_length=1,
        assembly_size=8,
        area_map={"input": "input", "state": "state"},
        token_to_key=[],
        input_assemblies={},
        state_assemblies={},
    )
    pointer = np.array([0], dtype=np.int64)

    with patch(
        "pyac.tasks.pointer.protocol.rollout_seen_pointer_sequence",
        return_value=[0, 0],
    ):
        rows = evaluate_seen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=[pointer],
            hops=1,
            time_budget=1,
            samples_per_list=3,
            rng=np.random.default_rng(42),
        )

    assert len(rows) == 3
    assert len({row["instance_id"] for row in rows}) == 3


def test_evaluate_proper_unseen_per_instance_multiple_lists() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=4,
        assembly_size=4,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "loop": "loop", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )

    pointer_a = np.array([1, 2, 3, 0], dtype=np.int64)
    pointer_b = np.array([2, 3, 0, 1], dtype=np.int64)
    pointers = [pointer_a, pointer_b]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=2, internal_steps=2, decoded_sequence=[0, 1, 2]),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=pointers,
            hops=2,
            time_budget=2,
            samples_per_list=2,
            rng=np.random.default_rng(42),
        )

    assert len(rows) == 4  # 2 lists * 2 samples


def test_evaluate_proper_unseen_per_instance_with_c_equals_2() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )
    pointer = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)

    # c=2: each pointer transition costs 2 updates. L=2, t=4 budget.
    decoded = [0, 0, 1, 1, 2, 2]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=2, internal_steps=4, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=[pointer],
            hops=2,
            time_budget=4,
            c=2,
            samples_per_list=1,
            rng=np.random.default_rng(42),
            theta_id="c2-test",
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["L"] == 2
    assert row["t"] == 4
    assert row["c"] == 2
    assert row["readout_step"] == 4
    assert row["completed_hops"] == 2
    assert row["target"] == 2
    assert row["prediction"] == 2
    assert row["correct"] is True
    assert row["true_trajectory"] == [0, 1, 2]
    assert row["trajectory"] == [0, 1, 2]
    assert row["path_accuracy"] == 1.0
    assert row["first_error_index"] is None


def test_evaluate_proper_unseen_per_instance_c_equals_4_partial_budget() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        evaluate_proper_unseen_per_instance,
    )

    task = ProperUnseenPointerTask(
        list_length=6,
        assembly_size=8,
        area_map={"cur": "cur", "src": "src", "dst": "dst", "readout": "readout"},
        node_assemblies={},
        hop_assemblies={},
        memory_fiber=("src", "dst"),
        episodic_baseline=None,
        controller_fibers=[],
    )
    pointer = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)

    # c=4, L=2, t=5 budget: only completes 1 hop (floor(5/4)=1), need 8 for full.
    decoded = [0, 0, 0, 0, 1, 1]

    with patch(
        "pyac.tasks.pointer.proper_unseen_protocol.rollout_proper_unseen_pointer",
        return_value=_make_fake_rollout_trace(start_node=0, hops=2, internal_steps=5, decoded_sequence=decoded),
    ):
        rows = evaluate_proper_unseen_per_instance(
            network=None,  # type: ignore[arg-type]
            task=task,
            pointers=[pointer],
            hops=2,
            time_budget=5,
            c=4,
            samples_per_list=1,
            rng=np.random.default_rng(42),
        )

    row = rows[0]
    assert row["c"] == 4
    assert row["completed_hops"] == 1
    assert row["readout_step"] == 4
    assert row["prediction"] == 1
    assert row["correct"] is False
    assert row["true_trajectory"] == [0, 1, 2]
    assert row["trajectory"] == [0, 1, None]  # only first hop boundary sampled
    assert row["path_accuracy"] == 2 / 3
    assert row["first_error_index"] == 2
