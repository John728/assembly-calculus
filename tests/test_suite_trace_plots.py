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


def test_suite_trace_plot_hook_writes_expected_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "suite.yaml"
    output_dir = tmp_path / "outputs"
    cfg_path.write_text(
        "suite_name: ac-trace-demo\n"
        f"output_dir: {output_dir}\n"
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
        "  hops: 2\n",
        encoding="utf-8",
    )

    module = _load_module()
    out_dir = module.run_suite(config_path=cfg_path)
    trace_dir = out_dir / "trace_plots"

    assert (trace_dir / "assembly_bars_over_time.png").exists()
    assert (trace_dir / "assembly_heatmap.png").exists()
    assert (trace_dir / "assembly_connectivity_graph.png").exists()
    assert (trace_dir / "assembly_weight_matrix.png").exists()
    assert (trace_dir / "pointer_reference.png").exists()
