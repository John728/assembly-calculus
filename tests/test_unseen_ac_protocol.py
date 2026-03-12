from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def test_reset_episode_memory_clears_written_bindings() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.unseen_protocol import (
        build_unseen_pointer_network,
        episodic_binding_mass,
        reset_episode_memory,
        write_list_episode,
    )

    pointer = np.asarray([1, 2, 3, 0], dtype=np.int64)
    network, task = build_unseen_pointer_network(list_length=4, assembly_size=8, density=0.5, plasticity=0.25, rng=make_rng(7))

    before = episodic_binding_mass(network, task)
    write_list_episode(network, task, pointer, write_rounds=2)
    after_write = episodic_binding_mass(network, task)
    reset_episode_memory(network, task)
    after_reset = episodic_binding_mass(network, task)

    assert before == 0.0
    assert after_write > 0.0
    assert after_reset == 0.0


def test_unseen_one_hop_lookup_is_above_chance() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.unseen_protocol import (
        build_unseen_pointer_network,
        evaluate_unseen_one_hop,
    )

    root_rng = make_rng(11)
    list_rng, net_rng, eval_rng = spawn_rngs(root_rng, 3)
    lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_unseen_pointer_network(list_length=6, assembly_size=10, density=0.5, plasticity=0.25, rng=net_rng)

    accuracy = evaluate_unseen_one_hop(network, task, lists, samples_per_list=8, rng=eval_rng)

    assert accuracy >= 0.4


def test_unseen_multi_hop_rollout_tracks_intermediate_states() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.unseen_protocol import (
        build_unseen_pointer_network,
        rollout_unseen_pointer,
    )

    pointer = np.asarray([2, 4, 1, 0, 3], dtype=np.int64)
    network, task = build_unseen_pointer_network(list_length=5, assembly_size=10, density=0.5, plasticity=0.25, rng=make_rng(13))

    trace = rollout_unseen_pointer(network, task, pointer, start_node=0, hops=3)

    assert trace["intermediate_nodes"] == [2, 1, 4]
    assert trace["final_prediction"] == 4


def test_write_list_episode_reports_plastic_write_steps() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.unseen_protocol import build_unseen_pointer_network, write_list_episode

    pointer = np.asarray([1, 2, 3, 0], dtype=np.int64)
    network, task = build_unseen_pointer_network(list_length=4, assembly_size=8, density=0.5, plasticity=0.25, rng=make_rng(17))

    stats = write_list_episode(network, task, pointer, write_rounds=2)

    assert stats["write_mode"] == "plastic"
    assert stats["write_steps"] == 8


def test_closed_loop_rollout_uses_single_initial_cue_only() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.unseen_protocol import build_unseen_pointer_network, rollout_unseen_pointer

    pointer = np.asarray([1, 2, 3, 0], dtype=np.int64)
    network, task = build_unseen_pointer_network(list_length=4, assembly_size=8, density=0.5, plasticity=0.25, rng=make_rng(19))

    trace = rollout_unseen_pointer(network, task, pointer, start_node=0, hops=3)

    assert trace["external_cue_count"] == 1


def test_closed_loop_rollout_tracks_internal_current_state() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.unseen_protocol import build_unseen_pointer_network, rollout_unseen_pointer

    pointer = np.asarray([2, 4, 1, 0, 3], dtype=np.int64)
    network, task = build_unseen_pointer_network(list_length=5, assembly_size=10, density=0.5, plasticity=0.25, rng=make_rng(23))

    trace = rollout_unseen_pointer(network, task, pointer, start_node=0, hops=3)

    assert trace["current_state_nodes"][:4] == [0, 2, 1, 4]
