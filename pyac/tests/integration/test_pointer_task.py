from __future__ import annotations

import importlib

from pyac.core.rng import make_rng, spawn_rngs


def test_pointer_data_generates_unique_full_cycles() -> None:
    data = importlib.import_module("pyac.tasks.pointer.data")

    rng = make_rng(123)
    lists = data.generate_unique_lists(num_lists=4, n=5, rng=rng)

    assert len(lists) == 4
    assert len({tuple(pointer.tolist()) for pointer in lists}) == 4

    for pointer in lists:
        assert sorted(pointer.tolist()) == [0, 1, 2, 3, 4]


def test_follow_pointer_matches_manual_rollout() -> None:
    data = importlib.import_module("pyac.tasks.pointer.data")

    pointer = [2, 4, 1, 0, 3]

    assert data.follow_pointer(pointer, start=0, hops=0) == 0
    assert data.follow_pointer(pointer, start=0, hops=1) == 2
    assert data.follow_pointer(pointer, start=0, hops=2) == 1
    assert data.follow_pointer(pointer, start=0, hops=3) == 4


def test_build_pointer_network_has_expected_areas() -> None:
    protocol = importlib.import_module("pyac.tasks.pointer.protocol")

    rng = make_rng(7)
    network, task = protocol.build_pointer_network(
        num_lists=3,
        list_length=4,
        assembly_size=6,
        density=0.35,
        plasticity=0.2,
        rng=rng,
    )

    assert task.num_tokens == 12
    assert task.area_map == {"input": "input", "state": "state"}
    assert set(network.area_names) == {"input", "state"}
    assert network.areas_by_name["input"].k == 6
    assert network.areas_by_name["state"].k == 6


def test_pointer_training_improves_seen_list_accuracy() -> None:
    data = importlib.import_module("pyac.tasks.pointer.data")
    protocol = importlib.import_module("pyac.tasks.pointer.protocol")

    root_rng = make_rng(11)
    list_rng, net_rng, eval_rng = spawn_rngs(root_rng, 3)
    lists = data.generate_unique_lists(num_lists=2, n=5, rng=list_rng)

    network, task = protocol.build_pointer_network(
        num_lists=len(lists),
        list_length=5,
        assembly_size=8,
        density=0.5,
        plasticity=0.3,
        rng=net_rng,
    )

    before = protocol.evaluate_seen_lists(
        network,
        task,
        lists,
        samples_per_list=20,
        k=3,
        rng=eval_rng,
    )

    protocol.train_node_assemblies(network, task, presentation_rounds=6, settle_steps=4)
    protocol.train_seen_transitions(
        network,
        task,
        lists,
        transition_rounds=10,
        association_steps=3,
    )

    after = protocol.evaluate_seen_lists(
        network,
        task,
        lists,
        samples_per_list=20,
        k=3,
        rng=make_rng(19),
    )

    assert after >= before
    assert after >= 0.6


def test_decode_is_restricted_to_the_active_list() -> None:
    protocol = importlib.import_module("pyac.tasks.pointer.protocol")

    rng = make_rng(31)
    network, task = protocol.build_pointer_network(
        num_lists=2,
        list_length=4,
        assembly_size=6,
        density=0.4,
        plasticity=0.2,
        rng=rng,
    )

    final_assembly = task.state_assemblies[(1, 2)]

    assert protocol._decode_node(task, 1, final_assembly) == 2
    assert protocol._decode_node(task, 0, final_assembly) != 2


def test_pointer_accuracy_vs_hop_returns_expected_records() -> None:
    data = importlib.import_module("pyac.tasks.pointer.data")
    metrics = importlib.import_module("pyac.tasks.pointer.metrics")
    protocol = importlib.import_module("pyac.tasks.pointer.protocol")

    root_rng = make_rng(23)
    list_rng, net_rng, eval_rng = spawn_rngs(root_rng, 3)
    lists = data.generate_unique_lists(num_lists=2, n=4, rng=list_rng)
    network, task = protocol.build_pointer_network(
        num_lists=len(lists),
        list_length=4,
        assembly_size=8,
        density=0.55,
        plasticity=0.25,
        rng=net_rng,
    )
    protocol.train_node_assemblies(network, task, presentation_rounds=5, settle_steps=4)
    protocol.train_seen_transitions(
        network,
        task,
        lists,
        transition_rounds=8,
        association_steps=3,
    )

    records = metrics.accuracy_vs_hop(
        network,
        task,
        lists,
        k_values=[1, 2, 3, 4],
        samples_per_list=12,
        rng=eval_rng,
        model_name="AC-Seen",
    )

    assert [record["k"] for record in records] == [1, 2, 3, 4]
    assert all(record["List Type"] == "Seen" for record in records)
    assert all(record["Model"] == "AC-Seen" for record in records)
    assert all(0.0 <= record["Accuracy"] <= 1.0 for record in records)
