from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYAC_SRC = ROOT / "pyac" / "src"
if str(PYAC_SRC) not in sys.path:
    sys.path.insert(0, str(PYAC_SRC))


def test_record_rollout_trace_has_expected_fields() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists, train_node_assemblies, train_seen_transitions
    from pyac.tasks.pointer.protocol import build_pointer_network
    from pyac.tasks.pointer.trace import record_rollout_trace

    root_rng = make_rng(5)
    list_rng, net_rng = spawn_rngs(root_rng, 2)
    lists = generate_unique_lists(2, 6, list_rng)
    network, task = build_pointer_network(num_lists=2, list_length=6, assembly_size=8, density=0.2, plasticity=0.25, rng=net_rng)

    train_node_assemblies(network, task, presentation_rounds=2, settle_steps=2)
    train_seen_transitions(network, task, lists, transition_rounds=3, association_steps=2)

    trace = record_rollout_trace(network, task, lists, list_idx=0, start_node=0, hops=3, settle_steps=1)

    assert "steps" in trace
    assert "final_prediction" in trace
    assert "target_node" in trace
    assert "assembly_spans" in trace
    assert "expected_edges" in trace
    assert "assembly_weight_matrix" in trace
    assert trace["steps"]
    first_step = trace["steps"][0]
    assert "active_neurons" in first_step
    assert "active_assemblies" in first_step
    assert isinstance(first_step["active_neurons"], list)
    assert isinstance(first_step["active_assemblies"], list)





def test_record_rollout_trace_supports_proper_unseen_pointer_tasks() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_proper_unseen_controller,
    )
    from pyac.tasks.pointer.trace import record_rollout_trace

    root_rng = make_rng(37)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    training_lists = generate_unique_lists(4, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(list_length=6, assembly_size=8, density=0.35, plasticity=0.2, rng=net_rng)
    train_proper_unseen_controller(network, task, training_lists=training_lists, k_values=[1, 2], episodes=2, rng=train_rng)

    trace = record_rollout_trace(network, task, training_lists, list_idx=0, start_node=0, hops=2)

    assert trace["steps"]
    assert trace["external_cue_count"] == 1
