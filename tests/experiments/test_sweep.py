"""
Tests for experiment sweep harness.

TDD: Write tests FIRST, then implement sweep.py and artifacts.py.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime


def test_sweep_imports():
    """Test that sweep module and functions can be imported."""
    from pyac_experiments.sweep import sweep
    from pyac_experiments.artifacts import Artifact, save_artifacts, load_artifacts
    
    assert callable(sweep)
    assert callable(save_artifacts)
    assert callable(load_artifacts)


def test_artifact_creation():
    """Test Artifact dataclass can be created with all fields."""
    from pyac_experiments.artifacts import Artifact
    
    artifact = Artifact(
        params={"n": 100, "k": 10},
        seed=42,
        entropy=(42,),
        metrics={"accuracy": 0.95, "loss": 0.05},
        pyac_version="0.1.0",
        timestamp="2026-01-01T00:00:00"
    )
    
    assert artifact.params == {"n": 100, "k": 10}
    assert artifact.seed == 42
    assert artifact.entropy == (42,)
    assert artifact.metrics["accuracy"] == 0.95
    assert artifact.pyac_version == "0.1.0"
    assert artifact.timestamp == "2026-01-01T00:00:00"


def test_sweep_count():
    """Test sweep produces correct number of artifacts (grid_size × n_seeds)."""
    from pyac_experiments.sweep import sweep
    
    def dummy_run(params, seed):
        return {"sum": params["x"] + params["y"]}
    
    param_grid = {"x": [1, 2], "y": [10, 20, 30]}
    results = sweep(param_grid, dummy_run, n_seeds=3, base_seed=42)
    
    # 2 x-values × 3 y-values × 3 seeds = 18 artifacts
    assert len(results) == 18


def test_sweep_includes_all_param_combinations():
    """Test sweep covers all parameter combinations."""
    from pyac_experiments.sweep import sweep
    
    def capture_params(params, seed):
        return {"params_copy": params.copy()}
    
    param_grid = {"a": [1, 2], "b": [10, 20]}
    results = sweep(param_grid, capture_params, n_seeds=2, base_seed=100)
    
    # Extract unique parameter combinations
    param_combos = set()
    for artifact in results:
        combo = tuple(sorted(artifact.params.items()))
        param_combos.add(combo)
    
    # Should have 2 × 2 = 4 unique parameter combinations
    assert len(param_combos) == 4
    assert (("a", 1), ("b", 10)) in param_combos
    assert (("a", 1), ("b", 20)) in param_combos
    assert (("a", 2), ("b", 10)) in param_combos
    assert (("a", 2), ("b", 20)) in param_combos


def test_sweep_uses_different_seeds():
    """Test sweep uses n_seeds different seeds for each param combination."""
    from pyac_experiments.sweep import sweep
    
    def capture_seed(params, seed):
        return {"seed_copy": seed}
    
    param_grid = {"x": [1]}
    results = sweep(param_grid, capture_seed, n_seeds=5, base_seed=42)
    
    # Should have 5 artifacts with different seeds
    assert len(results) == 5
    seeds = [artifact.seed for artifact in results]
    assert len(set(seeds)) == 5  # All seeds unique


def test_sweep_determinism():
    """Test sweep is deterministic: same params+seed → identical metrics."""
    from pyac_experiments.sweep import sweep
    from pyac.core.rng import make_rng
    
    def stochastic_metric(params, seed):
        rng = make_rng(seed)
        return {"random_value": float(rng.random())}
    
    param_grid = {"n": [100]}
    
    # Run sweep twice with same base_seed
    results1 = sweep(param_grid, stochastic_metric, n_seeds=3, base_seed=999)
    results2 = sweep(param_grid, stochastic_metric, n_seeds=3, base_seed=999)
    
    assert len(results1) == 3
    assert len(results2) == 3
    
    # Same seed should produce same metrics
    for i in range(3):
        assert results1[i].seed == results2[i].seed
        assert results1[i].metrics["random_value"] == results2[i].metrics["random_value"]


def test_artifact_has_entropy():
    """Test Artifact captures RNG entropy for reproducibility."""
    from pyac_experiments.sweep import sweep
    from pyac.core.rng import make_rng, get_entropy
    
    def capture_entropy(params, seed):
        rng = make_rng(seed)
        return {"dummy": 1}
    
    param_grid = {"x": [1]}
    results = sweep(param_grid, capture_entropy, n_seeds=2, base_seed=42)
    
    # All artifacts should have entropy
    for artifact in results:
        assert artifact.entropy is not None
        assert isinstance(artifact.entropy, tuple) or isinstance(artifact.entropy, int)


def test_artifact_has_version_and_timestamp():
    """Test Artifact includes pyac version and timestamp."""
    from pyac_experiments.sweep import sweep
    
    def dummy_run(params, seed):
        return {"metric": 1.0}
    
    param_grid = {"x": [1]}
    results = sweep(param_grid, dummy_run, n_seeds=1, base_seed=42)
    
    artifact = results[0]
    assert artifact.pyac_version == "0.1.0"
    assert artifact.timestamp is not None
    assert isinstance(artifact.timestamp, str)
    # Timestamp should be ISO format
    datetime.fromisoformat(artifact.timestamp)  # Should not raise


def test_save_artifacts_creates_file():
    """Test save_artifacts creates JSON file."""
    from pyac_experiments.artifacts import Artifact, save_artifacts
    
    artifact = Artifact(
        params={"n": 100},
        seed=42,
        entropy=(42,),
        metrics={"acc": 0.95},
        pyac_version="0.1.0",
        timestamp="2026-01-01T00:00:00"
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        save_artifacts([artifact], path)
        assert Path(path).exists()
        
        # Verify it's valid JSON
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_artifacts_reconstructs():
    """Test load_artifacts reconstructs Artifact objects from JSON."""
    from pyac_experiments.artifacts import Artifact, save_artifacts, load_artifacts
    
    original = Artifact(
        params={"n": 100, "k": 10},
        seed=42,
        entropy=(42, 123),
        metrics={"acc": 0.95, "loss": 0.05},
        pyac_version="0.1.0",
        timestamp="2026-01-01T00:00:00"
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        save_artifacts([original], path)
        loaded = load_artifacts(path)
        
        assert len(loaded) == 1
        artifact = loaded[0]
        
        assert isinstance(artifact, Artifact)
        assert artifact.params == original.params
        assert artifact.seed == original.seed
        assert artifact.entropy == original.entropy
        assert artifact.metrics == original.metrics
        assert artifact.pyac_version == original.pyac_version
        assert artifact.timestamp == original.timestamp
    finally:
        Path(path).unlink(missing_ok=True)


def test_artifacts_json_roundtrip():
    """Test artifacts survive JSON save/load roundtrip."""
    from pyac_experiments.sweep import sweep
    from pyac_experiments.artifacts import save_artifacts, load_artifacts
    
    def metric_fn(params, seed):
        return {"result": params["x"] * 2}
    
    param_grid = {"x": [1, 2, 3]}
    original_artifacts = sweep(param_grid, metric_fn, n_seeds=2, base_seed=42)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        save_artifacts(original_artifacts, path)
        loaded_artifacts = load_artifacts(path)
        
        assert len(loaded_artifacts) == len(original_artifacts)
        
        for orig, loaded in zip(original_artifacts, loaded_artifacts):
            assert loaded.params == orig.params
            assert loaded.seed == orig.seed
            assert loaded.entropy == orig.entropy
            assert loaded.metrics == orig.metrics
            assert loaded.pyac_version == orig.pyac_version
            assert loaded.timestamp == orig.timestamp
    finally:
        Path(path).unlink(missing_ok=True)


def test_sweep_with_complex_metrics():
    """Test sweep handles complex metric dictionaries."""
    from pyac_experiments.sweep import sweep
    
    def complex_metric(params, seed):
        return {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1": 0.95,
            "confusion_matrix": [[10, 2], [1, 15]]
        }
    
    param_grid = {"model": ["A", "B"]}
    results = sweep(param_grid, complex_metric, n_seeds=1, base_seed=42)
    
    assert len(results) == 2
    for artifact in results:
        assert "accuracy" in artifact.metrics
        assert "confusion_matrix" in artifact.metrics
        assert artifact.metrics["confusion_matrix"] == [[10, 2], [1, 15]]


def test_sweep_single_param_single_seed():
    """Test sweep works with single parameter and single seed."""
    from pyac_experiments.sweep import sweep
    
    def simple_metric(params, seed):
        return {"value": 1.0}
    
    param_grid = {"x": [100]}
    results = sweep(param_grid, simple_metric, n_seeds=1, base_seed=42)
    
    assert len(results) == 1
    assert results[0].params == {"x": 100}
    assert results[0].metrics == {"value": 1.0}


def test_sweep_passes_correct_params_to_run_fn():
    """Test sweep passes correct parameters to run function."""
    from pyac_experiments.sweep import sweep
    
    captured = []
    
    def capture_run(params, seed):
        captured.append((params.copy(), seed))
        return {"dummy": 1}
    
    param_grid = {"a": [1, 2], "b": [10]}
    sweep(param_grid, capture_run, n_seeds=1, base_seed=42)
    
    # Should have 2 param combinations × 1 seed = 2 calls
    assert len(captured) == 2
    
    # Check parameters were passed correctly
    params_list = [params for params, _ in captured]
    assert {"a": 1, "b": 10} in params_list
    assert {"a": 2, "b": 10} in params_list
