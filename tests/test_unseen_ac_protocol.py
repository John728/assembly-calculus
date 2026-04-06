from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))





def test_proper_unseen_protocol_exposes_training_entrypoint() -> None:
    from pyac.tasks.pointer.proper_unseen_protocol import (
        ProperUnseenPointerTask,
        build_proper_unseen_pointer_network,
        train_proper_unseen_controller,
    )

    assert ProperUnseenPointerTask is not None
    assert callable(build_proper_unseen_pointer_network)
    assert callable(train_proper_unseen_controller)


def test_proper_unseen_rollout_contract_uses_single_external_cue() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        rollout_proper_unseen_pointer,
        train_proper_unseen_controller,
    )

    root_rng = make_rng(31)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    training_lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(list_length=6, assembly_size=8, density=0.35, plasticity=0.2, rng=net_rng)

    train_proper_unseen_controller(
        network,
        task,
        training_lists=training_lists,
        k_values=[1, 2, 3],
        episodes=2,
        rng=train_rng,
    )
    trace = rollout_proper_unseen_pointer(network, task, training_lists[0], start_node=0, hops=3, internal_steps=3)

    assert trace["external_cue_count"] == 1
    assert trace["controller_mode"] == "internal"


def test_proper_unseen_network_uses_canonical_cur_src_dst_loop_areas() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import build_proper_unseen_pointer_network

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(33),
    )

    assert task.area_map == {
        "cur": "cur",
        "src": "src",
        "dst": "dst",
        "loop": "loop",
        "readout": "readout",
    }
    cur = network.areas_by_name[task.area_map["cur"]]
    loop = network.areas_by_name[task.area_map["loop"]]
    assert cur.dynamics_type == "recurrent"
    assert loop.dynamics_type == "recurrent"
    assert cur.p_recurrent > 0.0
    assert loop.p_recurrent > 0.0
    assert task.memory_fiber == ("src", "dst")
    assert ("cur", "src") in task.controller_fibers
    assert ("dst", "cur") in task.controller_fibers


def test_stage0_reports_stable_identity_aligned_assemblies() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        evaluate_identity_alignment,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=5,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(34),
    )

    diagnostics = evaluate_identity_alignment(network, task)

    assert diagnostics["num_nodes"] == 5
    assert diagnostics["areas_checked"] == ["cur", "src", "dst"]
    assert diagnostics["mean_self_overlap"] > 0


def test_episode_memory_write_binds_src_to_dst_for_current_pointer() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        probe_episode_memory,
        reset_proper_episode_memory,
        write_unseen_episode,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=4,
        assembly_size=8,
        density=0.4,
        plasticity=0.25,
        rng=make_rng(36),
    )
    pointer = np.asarray([1, 2, 3, 0], dtype=np.int64)

    write_unseen_episode(network, task, pointer, write_rounds=2)
    probe = probe_episode_memory(network, task, src_node=2)

    assert probe["predicted_dst_node"] == 3
    reset_proper_episode_memory(network, task)
    probe_after_reset = probe_episode_memory(network, task, src_node=2)
    assert probe_after_reset["memory_active"] is False


def test_proper_unseen_rollout_reports_hop_control_states() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        rollout_proper_unseen_pointer,
        train_proper_unseen_controller,
    )

    root_rng = make_rng(35)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    training_lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(list_length=6, assembly_size=8, density=0.35, plasticity=0.2, rng=net_rng)

    train_proper_unseen_controller(
        network,
        task,
        training_lists=training_lists,
        k_values=[1, 2, 3],
        episodes=2,
        rng=train_rng,
    )
    trace = rollout_proper_unseen_pointer(network, task, training_lists[0], start_node=0, hops=3, internal_steps=3)

    hop_ctrl_states = trace["hop_ctrl_states"]
    assert isinstance(hop_ctrl_states, list)
    assert len(hop_ctrl_states) == 4
    assert hop_ctrl_states[0] == 3
    assert trace["hop_ctrl_source"] == "network"


def test_proper_unseen_rollout_counts_hops_down_with_t_equals_k() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        rollout_proper_unseen_pointer,
        train_proper_unseen_controller,
    )

    root_rng = make_rng(37)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    pointer = generate_unique_lists(4, 6, list_rng)[0]
    network, task = build_proper_unseen_pointer_network(list_length=6, assembly_size=12, density=0.45, plasticity=0.3, rng=net_rng)

    train_proper_unseen_controller(
        network,
        task,
        training_lists=[pointer],
        k_values=[1, 2, 3],
        episodes=12,
        rng=train_rng,
    )
    trace = rollout_proper_unseen_pointer(network, task, pointer, start_node=0, hops=3, internal_steps=3)

    assert trace["hop_ctrl_states"] == [3, 2, 1, 0]


def test_proper_unseen_training_records_query_supervision_and_updates_only_controller() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_proper_unseen_controller,
    )

    root_rng = make_rng(41)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    training_lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(list_length=6, assembly_size=8, density=0.35, plasticity=0.2, rng=net_rng)

    before_controller = {
        fiber: float(network.weights[fiber].sum())
        for fiber in task.controller_fibers
    }
    before_episodic = float(network.weights[task.memory_fiber].sum())

    history = train_proper_unseen_controller(
        network,
        task,
        training_lists=training_lists,
        k_values=[1, 2, 3],
        episodes=3,
        rng=train_rng,
    )

    assert len(history) == 3
    first = history[0]
    assert "start_node" in first
    assert "target_node" in first
    assert "final_prediction" in first
    assert "episode_accuracy" in first
    assert "controller_update_steps" in first
    assert "path_nodes" in first
    assert 0.0 <= float(first["episode_accuracy"]) <= 1.0
    assert int(first["controller_update_steps"]) >= (2 * int(first["query_hops"])) + 1

    after_controller = {
        fiber: float(network.weights[fiber].sum())
        for fiber in task.controller_fibers
    }
    after_episodic = float(network.weights[task.memory_fiber].sum())

    assert any(after_controller[fiber] != before_controller[fiber] for fiber in task.controller_fibers)
    assert after_episodic == before_episodic


def test_controller_query_primitive_transfers_cur_to_src() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        probe_query_primitive,
        train_query_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(43),
    )

    train_query_primitive(network, task)
    result = probe_query_primitive(network, task, node_idx=3)

    assert result["predicted_src_node"] == 3
    assert result["correct"] is True


def test_query_primitive_generalizes_across_node_vocab() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        evaluate_query_primitive,
        train_query_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(45),
    )

    train_query_primitive(network, task, rounds=8)
    report = evaluate_query_primitive(network, task)

    assert report["accuracy"] >= 0.8


def test_controller_writeback_primitive_transfers_dst_to_cur() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        probe_writeback_primitive,
        train_writeback_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(47),
    )

    train_writeback_primitive(network, task)
    result = probe_writeback_primitive(network, task, node_idx=4)

    assert result["predicted_cur_node"] == 4
    assert result["correct"] is True


def test_writeback_primitive_generalizes_across_node_vocab() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        evaluate_writeback_primitive,
        train_writeback_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(49),
    )

    train_writeback_primitive(network, task, rounds=8)
    report = evaluate_writeback_primitive(network, task)

    assert report["accuracy"] >= 0.8


def test_mechanism_trace_reports_cur_src_dst_cur_route() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        rollout_proper_unseen_pointer,
        train_proper_unseen_controller,
    )

    root_rng = make_rng(53)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    training_lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=net_rng,
    )

    train_proper_unseen_controller(
        network,
        task,
        training_lists=training_lists,
        k_values=[1, 2, 3],
        episodes=2,
        rng=train_rng,
    )
    trace = rollout_proper_unseen_pointer(
        network,
        task,
        training_lists[0],
        start_node=0,
        hops=3,
        internal_steps=3,
    )

    assert "cur_nodes" in trace
    assert "src_nodes" in trace
    assert "dst_nodes" in trace
    assert trace["mechanism_route"] == "CUR->SRC->DST->CUR"


def test_one_hop_composition_uses_current_episode_memory() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_query_primitive,
        train_writeback_primitive,
        evaluate_one_hop_composition,
    )

    root_rng = make_rng(109)
    list_rng, net_rng = spawn_rngs(root_rng, 2)
    pointers = generate_unique_lists(8, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=10,
        density=0.4,
        plasticity=0.25,
        rng=net_rng,
    )
    train_query_primitive(network, task, rounds=8)
    train_writeback_primitive(network, task, rounds=8)

    report = evaluate_one_hop_composition(network, task, test_lists=pointers[4:])
    assert report["accuracy"] >= 0.5
    assert report["memory_dependent"] is True

