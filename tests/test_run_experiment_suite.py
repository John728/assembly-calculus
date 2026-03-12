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
