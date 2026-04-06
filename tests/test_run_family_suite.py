from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_family_suite.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_family_suite", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_config_path_maps_family_and_scale() -> None:
    module = _load_module()

    assert module.resolve_config_path("seen_mlp", "dev") == ROOT / "experiments" / "seen_mlp_dev.yaml"
    assert module.resolve_config_path("unseen_ac", "paper") == ROOT / "experiments" / "unseen_ac_paper.yaml"


def test_resolve_config_path_rejects_unknown_family() -> None:
    module = _load_module()

    try:
        module.resolve_config_path("bad_family", "dev")
    except ValueError as exc:
        assert "family" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown family")


def test_resolve_config_path_rejects_unknown_scale() -> None:
    module = _load_module()

    try:
        module.resolve_config_path("seen_mlp", "bad_scale")
    except ValueError as exc:
        assert "scale" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown scale")


def test_main_supports_dry_run(capsys) -> None:
    module = _load_module()

    exit_code = module.main(["seen_mlp", "dev", "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_experiment_suite.py --config" in captured.out
    assert "experiments/seen_mlp_dev.yaml" in captured.out


def test_nci_readme_uses_python_launcher_examples() -> None:
    readme = (ROOT / "scripts" / "nci" / "README.md").read_text(encoding="utf-8")

    assert "venv/bin/python run_family_suite.py seen_mlp dev" in readme
    assert "venv/bin/python run_family_suite.py seen_mlp paper" in readme
    assert "run_family_dev.sh" not in readme
    assert "run_family_paper.sh" not in readme
