from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_experiment_suite.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_experiment_suite", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_experiment_suite_writes_outputs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: suite-demo\n"
        "output_dir: " + str(tmp_path / "outputs") + "\n"
        "seeds: [1]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "    N: 8\n"
        "    num_train_lists: 2\n"
        "    num_test_lists: 0\n"
        "    k_train_min: 1\n"
        "    k_train_max: 2\n"
        "    k_test_min: 1\n"
        "    k_test_max: 2\n"
        "models:\n"
        "  MLP:\n"
        "    - model_name: Tiny-MLP\n"
        "      layers: 2\n"
        "      hidden_dim: 32\n"
        "      epochs: 1\n"
        "      batch_size: 16\n"
        "      lr: 0.001\n"
        "      samples_per_list_train: 4\n"
        "      samples_per_list_eval: 4\n",
        encoding="utf-8",
    )

    module = _load_module()
    out_dir = module.run_suite(config_path=cfg_path)

    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "plots" / "accuracy_vs_hop_seen_mlp.png").exists()
    assert (out_dir / "plots" / "seen_mlp_heatmap.png").exists()
    assert (out_dir / "plots" / "size_tradeoff_seen_mlp.png").exists()


def test_run_experiment_suite_writes_family_local_ac_trace_outputs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: seen-ac-demo\n"
        "output_dir: " + str(tmp_path / "outputs" / "seen-ac-demo") + "\n"
        "seeds: [1]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "    N: 8\n"
        "    num_train_lists: 2\n"
        "    num_test_lists: 0\n"
        "    k_train_min: 1\n"
        "    k_train_max: 2\n"
        "    k_test_min: 1\n"
        "    k_test_max: 6\n"
        "models:\n"
        "  AC:\n"
        "    - model_name: Tiny-AC\n"
        "      assembly_size: 8\n"
        "      density: 0.2\n"
        "      plasticity: 0.25\n"
        "      presentation_rounds: 2\n"
        "      transition_rounds: 3\n"
        "      association_steps: 2\n"
        "      samples_per_list_eval: 4\n"
        "trace_plots:\n"
        "  enabled: true\n"
        "  list_idx: 0\n"
        "  start_node: 0\n"
        "  hops: 6\n",
        encoding="utf-8",
    )

    module = _load_module()
    out_dir = module.run_suite(config_path=cfg_path)

    trace_dir = out_dir / "trace_plots"
    assert (trace_dir / "assembly_heatmap.png").exists()
    assert (trace_dir / "assembly_bars_over_time.png").exists()
    assert (trace_dir / "assembly_connectivity_graph.png").exists()
    assert (trace_dir / "assembly_weight_matrix.png").exists()
