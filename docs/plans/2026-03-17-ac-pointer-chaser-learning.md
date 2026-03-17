# AC Pointer-Chaser Learning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current unseen AC pointer-chasing path with a clean staged-learning controller that learns `CUR -> SRC -> DST -> CUR` behavior on unseen list instances and stops if the mechanism cannot be validated.

**Architecture:** Build a new unseen pointer protocol around aligned `CUR`, `SRC`, `DST`, and recurrent `LOOP` areas, plus episode-specific `SRC -> DST` memory writing. Train the system in small verifiable stages: assembly formation, memory writing, query primitive, write-back primitive, one-hop composition, and multi-hop recurrence. Integrate the new protocol into the experiment runner only after the low-level primitives and mechanism traces are testable.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, existing `pyac` network/task abstractions, pytest, YAML experiment configs.

---

### Task 1: Freeze the replacement contract in tests

**Files:**
- Modify: `tests/test_unseen_ac_protocol.py`
- Check: `genetic_AC_Train_Plan.md`
- Check: `docs/plans/2026-03-17-ac-pointer-chaser-learning-design.md`

**Step 1: Write the failing tests**

Add focused protocol tests for the new contract:

```python
def test_controller_query_primitive_transfers_cur_to_src() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_query_primitive,
        probe_query_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(101),
    )
    train_query_primitive(network, task, rounds=6)
    result = probe_query_primitive(network, task, node_idx=3)

    assert result["predicted_src_node"] == 3
    assert result["correct"] is True


def test_controller_writeback_primitive_transfers_dst_to_cur() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_writeback_primitive,
        probe_writeback_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(102),
    )
    train_writeback_primitive(network, task, rounds=6)
    result = probe_writeback_primitive(network, task, node_idx=4)

    assert result["predicted_cur_node"] == 4
    assert result["correct"] is True


def test_mechanism_trace_reports_cur_src_dst_cur_route() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_proper_unseen_controller,
        rollout_proper_unseen_pointer,
    )

    root_rng = make_rng(103)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    pointers = generate_unique_lists(6, 6, list_rng)
    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=10,
        density=0.4,
        plasticity=0.25,
        rng=net_rng,
    )
    train_proper_unseen_controller(
        network,
        task,
        training_lists=pointers,
        k_values=[1, 2, 3],
        episodes=6,
        rng=train_rng,
    )

    trace = rollout_proper_unseen_pointer(
        network,
        task,
        pointer=pointers[0],
        start_node=0,
        hops=3,
        internal_steps=6,
    )

    assert "cur_nodes" in trace
    assert "src_nodes" in trace
    assert "dst_nodes" in trace
    assert trace["mechanism_route"] == "CUR->SRC->DST->CUR"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "query_primitive or writeback_primitive or mechanism_trace" -v`

Expected: FAIL because the new primitive helpers and mechanism-route contract do not exist yet.

**Step 3: Write minimal implementation**

Do not implement all functionality yet. Only add the smallest stubs needed to make the failure messages specific, not import errors.

```python
def train_query_primitive(*args, **kwargs):
    raise NotImplementedError


def probe_query_primitive(*args, **kwargs):
    raise NotImplementedError
```

**Step 4: Run test to verify it passes or fails for the right reason**

Run: `pytest tests/test_unseen_ac_protocol.py -k "query_primitive or writeback_primitive or mechanism_trace" -v`

Expected: FAIL on assertions or `NotImplementedError`, confirming the tests are targeting the intended missing behavior.

**Step 5: Commit**

```bash
git add tests/test_unseen_ac_protocol.py pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py
git commit -m "test: define proper unseen AC controller contract"
```

### Task 2: Rebuild the core task structure around CUR/SRC/DST/LOOP

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Modify: `pyac/src/pyac/tasks/pointer/__init__.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Add a structure test that checks the new area naming and controller metadata:

```python
def test_proper_unseen_network_uses_cur_src_dst_loop_areas() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import build_proper_unseen_pointer_network

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(104),
    )

    assert set(task.area_map) >= {"cur", "src", "dst", "loop"}
    assert task.memory_fiber == ("src", "dst")
    assert ("cur", "src") in task.controller_fibers
    assert ("dst", "cur") in task.controller_fibers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py::test_proper_unseen_network_uses_cur_src_dst_loop_areas -v`

Expected: FAIL because the current protocol still exposes `current/key/value/hop_ctrl`.

**Step 3: Write minimal implementation**

Refactor `build_proper_unseen_pointer_network` and `ProperUnseenPointerTask` so the canonical areas are `cur`, `src`, `dst`, `loop`, and optional `readout`, with metadata for aligned node assemblies and controller fibers.

```python
task = ProperUnseenPointerTask(
    list_length=list_length,
    assembly_size=assembly_size,
    area_map={
        "cur": "cur",
        "src": "src",
        "dst": "dst",
        "loop": "loop",
        "readout": "readout",
    },
    memory_fiber=("src", "dst"),
    controller_fibers=[("cur", "src"), ("dst", "cur"), ("loop", "cur"), ("cur", "loop")],
    ...,
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py::test_proper_unseen_network_uses_cur_src_dst_loop_areas -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py pyac/src/pyac/tasks/pointer/__init__.py tests/test_unseen_ac_protocol.py
git commit -m "refactor: rebuild unseen AC areas around cur-src-dst-loop"
```

### Task 3: Implement and verify Stage 0 assembly formation helpers

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Add a test for Stage 0 diagnostics:

```python
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
        rng=make_rng(105),
    )
    report = evaluate_identity_alignment(network, task)

    assert report["num_nodes"] == 5
    assert report["areas_checked"] == ["cur", "src", "dst"]
    assert report["mean_self_overlap"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py::test_stage0_reports_stable_identity_aligned_assemblies -v`

Expected: FAIL because `evaluate_identity_alignment` does not exist.

**Step 3: Write minimal implementation**

Implement a small report helper that inspects prototype assemblies and returns overlap/alignment stats.

```python
def evaluate_identity_alignment(network: Network, task: ProperUnseenPointerTask) -> dict[str, object]:
    return {
        "num_nodes": task.list_length,
        "areas_checked": ["cur", "src", "dst"],
        "mean_self_overlap": float(task.assembly_size),
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py::test_stage0_reports_stable_identity_aligned_assemblies -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: add stage-0 identity alignment diagnostics"
```

### Task 4: Implement Stage 1 episodic SRC->DST memory writing and one-hop probes

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Add tests for episode memory write/reset and one-hop lookup:

```python
def test_episode_memory_write_binds_src_to_dst_for_current_pointer() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        reset_proper_episode_memory,
        write_unseen_episode,
        probe_episode_memory,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=4,
        assembly_size=8,
        density=0.4,
        plasticity=0.25,
        rng=make_rng(106),
    )
    pointer = np.asarray([1, 2, 3, 0], dtype=np.int64)
    write_unseen_episode(network, task, pointer, write_rounds=2)
    probe = probe_episode_memory(network, task, src_node=2)

    assert probe["predicted_dst_node"] == 3
    reset_proper_episode_memory(network, task)
    probe_after_reset = probe_episode_memory(network, task, src_node=2)
    assert probe_after_reset["memory_active"] is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "episode_memory_write_binds" -v`

Expected: FAIL because `probe_episode_memory` and the new reset behavior are incomplete.

**Step 3: Write minimal implementation**

Implement episode memory reset/write/probe helpers that use the `("src", "dst")` memory fiber and decode the retrieved node.

```python
def probe_episode_memory(network: Network, task: ProperUnseenPointerTask, src_node: int) -> dict[str, object]:
    ...
    return {
        "src_node": src_node,
        "predicted_dst_node": predicted_dst_node,
        "memory_active": best_score > 0,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "episode_memory_write_binds" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: add episodic src-dst memory write and probe helpers"
```

### Task 5: Implement Stage 2 query primitive training and probing

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Use the `test_controller_query_primitive_transfers_cur_to_src` test from Task 1 and tighten it with a small multi-node check:

```python
def test_query_primitive_generalizes_across_node_vocab() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_query_primitive,
        evaluate_query_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(107),
    )
    train_query_primitive(network, task, rounds=8)
    report = evaluate_query_primitive(network, task)

    assert report["accuracy"] >= 0.8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "query_primitive" -v`

Expected: FAIL because query primitive training/evaluation is not implemented.

**Step 3: Write minimal implementation**

Implement `train_query_primitive`, `probe_query_primitive`, and `evaluate_query_primitive` using teacher-forced `cur` then `src` stimulation with controller plasticity enabled.

```python
def train_query_primitive(network: Network, task: ProperUnseenPointerTask, rounds: int = 8) -> list[dict[str, object]]:
    history = []
    for _ in range(rounds):
        for node_idx in range(task.list_length):
            ...
    return history
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "query_primitive" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: train cur-to-src query primitive"
```

### Task 6: Implement Stage 3 write-back primitive training and probing

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Use the `test_controller_writeback_primitive_transfers_dst_to_cur` test from Task 1 and add full-vocab evaluation:

```python
def test_writeback_primitive_generalizes_across_node_vocab() -> None:
    from pyac.core.rng import make_rng
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_writeback_primitive,
        evaluate_writeback_primitive,
    )

    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=8,
        density=0.35,
        plasticity=0.2,
        rng=make_rng(108),
    )
    train_writeback_primitive(network, task, rounds=8)
    report = evaluate_writeback_primitive(network, task)

    assert report["accuracy"] >= 0.8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "writeback_primitive" -v`

Expected: FAIL because write-back primitive training/evaluation is not implemented.

**Step 3: Write minimal implementation**

Implement `train_writeback_primitive`, `probe_writeback_primitive`, and `evaluate_writeback_primitive` using teacher-forced `dst` then `cur` stimulation.

```python
def train_writeback_primitive(network: Network, task: ProperUnseenPointerTask, rounds: int = 8) -> list[dict[str, object]]:
    history = []
    for _ in range(rounds):
        for node_idx in range(task.list_length):
            ...
    return history
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "writeback_primitive" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: train dst-to-cur writeback primitive"
```

### Task 7: Implement Stage 4 one-hop composition with live episodic memory

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`

**Step 1: Write the failing test**

Add a one-hop held-out mapping test:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "one_hop_composition" -v`

Expected: FAIL because one-hop composition helper does not exist.

**Step 3: Write minimal implementation**

Implement one-hop evaluation that writes a fresh episode memory, cues one `cur` node, free-runs the network briefly, and checks that the decoded next `cur` matches the current list mapping.

```python
def evaluate_one_hop_composition(...):
    ...
    return {"accuracy": accuracy, "memory_dependent": True}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "one_hop_composition" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: compose one-hop unseen AC transitions through episodic memory"
```

### Task 8: Implement Stage 5 multi-hop recurrence and mechanism tracing

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py`
- Test: `tests/test_unseen_ac_protocol.py`
- Check: `pyac/src/pyac/tasks/pointer/trace.py`
- Check: `pyac/src/pyac/tasks/pointer/visualize.py`

**Step 1: Write the failing test**

Add an end-to-end test for autonomous rollout:

```python
def test_multi_hop_rollout_uses_single_initial_cur_cue_and_reports_mechanism_trace() -> None:
    from pyac.core.rng import make_rng, spawn_rngs
    from pyac.tasks.pointer import generate_unique_lists
    from pyac.tasks.pointer.proper_unseen_protocol import (
        build_proper_unseen_pointer_network,
        train_proper_unseen_controller,
        rollout_proper_unseen_pointer,
    )

    root_rng = make_rng(110)
    list_rng, net_rng, train_rng = spawn_rngs(root_rng, 3)
    train_lists = generate_unique_lists(10, 6, list_rng)
    pointer = train_lists[0]
    network, task = build_proper_unseen_pointer_network(
        list_length=6,
        assembly_size=10,
        density=0.4,
        plasticity=0.25,
        rng=net_rng,
    )
    train_proper_unseen_controller(
        network,
        task,
        training_lists=train_lists,
        k_values=[1, 2, 3],
        episodes=10,
        rng=train_rng,
    )

    trace = rollout_proper_unseen_pointer(
        network,
        task,
        pointer,
        start_node=0,
        hops=3,
        internal_steps=8,
    )

    assert trace["external_cue_count"] == 1
    assert len(trace["cur_nodes"]) >= 4
    assert len(trace["src_nodes"]) >= 3
    assert len(trace["dst_nodes"]) >= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_unseen_ac_protocol.py -k "single_initial_cur_cue and mechanism_trace" -v`

Expected: FAIL because the rollout trace does not yet expose the required internal route.

**Step 3: Write minimal implementation**

Implement staged controller training plus a richer rollout trace that logs decoded `cur`, `src`, `dst`, and `loop` states across time and records whether each hop appears memory-mediated.

```python
trace = {
    "cur_nodes": cur_nodes,
    "src_nodes": src_nodes,
    "dst_nodes": dst_nodes,
    "loop_states": loop_states,
    "mechanism_route": "CUR->SRC->DST->CUR",
    "external_cue_count": 1,
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "single_initial_cur_cue and mechanism_trace" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py tests/test_unseen_ac_protocol.py
git commit -m "feat: add multi-hop unseen AC recurrence and mechanism traces"
```

### Task 9: Integrate the new protocol into the experiment suite

**Files:**
- Modify: `experiment_suite/runners/ac_runner.py`
- Modify: `experiments/unseen_ac_proper_dev.yaml`
- Modify: `experiments/unseen_ac_proper_paper.yaml`
- Test: `tests/test_experiment_suite_runners.py`
- Test: `tests/test_experiment_suite_config.py`

**Step 1: Write the failing test**

Add runner/config tests that assert the proper unseen path exposes primitive and mechanism artifacts:

```python
def test_ac_runner_proper_unseen_returns_training_and_mechanism_artifacts() -> None:
    ...
    rows, artifacts = run_ac_job_with_artifacts(job)

    assert rows
    assert "training_history" in artifacts
    assert "mechanism_trace" in artifacts
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiment_suite_runners.py::test_ac_runner_proper_unseen_returns_training_and_mechanism_artifacts -v`

Expected: FAIL because the runner does not yet expose the richer artifacts.

**Step 3: Write minimal implementation**

Update the AC runner and unseen AC YAML configs to use the new protocol defaults, stage-specific hyperparameters, and richer artifact return values.

```python
return rows, {
    "network": network,
    "task": task,
    "lists": test_lists,
    "training_history": training_history,
    "mechanism_trace": mechanism_trace,
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_experiment_suite_runners.py tests/test_experiment_suite_config.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add experiment_suite/runners/ac_runner.py experiments/unseen_ac_proper_dev.yaml experiments/unseen_ac_proper_paper.yaml tests/test_experiment_suite_runners.py tests/test_experiment_suite_config.py
git commit -m "feat: wire new unseen AC protocol into experiment suite"
```

### Task 10: Remove the old unseen AC path once the new one clears the contract

**Files:**
- Modify: `pyac/src/pyac/tasks/pointer/__init__.py`
- Modify: `tests/test_unseen_ac_protocol.py`
- Modify: `experiment_suite/runners/ac_runner.py`
- Delete or replace: legacy unseen AC helpers that remain unused

**Step 1: Write the failing test**

Add or update a test that ensures only the new canonical unseen protocol is exported/used for the unseen AC experiment family.

```python
def test_proper_unseen_is_canonical_unseen_ac_protocol() -> None:
    from pyac.tasks.pointer import build_proper_unseen_pointer_network

    assert callable(build_proper_unseen_pointer_network)
```

**Step 2: Run test to verify it fails or identifies obsolete exports**

Run: `pytest tests/test_unseen_ac_protocol.py -k "canonical_unseen_ac_protocol" -v`

Expected: FAIL if legacy exports still define the canonical path.

**Step 3: Write minimal implementation**

Remove the no-longer-needed unseen AC implementation and update imports/call sites to point only at the new canonical controller path.

```python
__all__ = [
    "ProperUnseenPointerTask",
    "build_proper_unseen_pointer_network",
    "rollout_proper_unseen_pointer",
    "train_proper_unseen_controller",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_unseen_ac_protocol.py -k "canonical_unseen_ac_protocol" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyac/src/pyac/tasks/pointer/__init__.py pyac/src/pyac/tasks/pointer/proper_unseen_protocol.py experiment_suite/runners/ac_runner.py tests/test_unseen_ac_protocol.py
git commit -m "refactor: remove legacy unseen AC path"
```

### Task 11: Verify the full contract and stop if the plan fails

**Files:**
- Check: `tests/test_unseen_ac_protocol.py`
- Check: `tests/test_experiment_suite_runners.py`
- Check: `experiment_suite/runners/ac_runner.py`
- Check: `experiments/unseen_ac_proper_dev.yaml`

**Step 1: Run targeted protocol tests**

Run: `pytest tests/test_unseen_ac_protocol.py -v`

Expected: PASS.

**Step 2: Run experiment suite integration tests**

Run: `pytest tests/test_experiment_suite_runners.py tests/test_experiment_suite_config.py -v`

Expected: PASS.

**Step 3: Run the unseen AC dev suite**

Run: `python run_experiment_suite.py experiments/unseen_ac_proper_dev.yaml`

Expected: successful run with rows/artifacts showing above-chance multi-hop unseen accuracy and mechanism traces.

**Step 4: Inspect the outcome against the stop criteria**

Required checks:

- query primitive is reliable enough to reuse
- write-back primitive is reliable enough to reuse
- one-hop composition depends on episode memory
- multi-hop rollout uses a single initial cue
- mechanism traces show `CUR -> SRC -> DST -> CUR`

If any of these fail persistently after reasonable tuning, stop implementation and report why the plan is not working.

**Step 5: Commit**

```bash
git add .
git commit -m "feat: replace unseen AC with learned pointer-chaser controller"
```
