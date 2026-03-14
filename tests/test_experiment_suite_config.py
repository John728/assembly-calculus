from __future__ import annotations

from pathlib import Path


def test_expand_jobs_produces_family_seed_combinations(tmp_path: Path) -> None:
    from experiment_suite.config import load_suite_config
    from experiment_suite.jobs import expand_jobs

    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: demo\n"
        "output_dir: outputs/demo\n"
        "seeds: [1, 2]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "    N: 8\n"
        "    num_train_lists: 4\n"
        "    num_test_lists: 0\n"
        "  - list_type: Unseen\n"
        "    N: 8\n"
        "    num_train_lists: 4\n"
        "    num_test_lists: 2\n"
        "models:\n"
        "  MLP:\n"
        "    - model_name: MLP-01\n"
        "      layers: 2\n"
        "    - model_name: MLP-02\n"
        "      layers: 4\n"
        "  AC:\n"
        "    - model_name: AC-Seen\n"
        "      assembly_size: 12\n",
        encoding="utf-8",
    )

    config = load_suite_config(cfg_path)
    jobs = expand_jobs(config)

    assert config.suite_name == "demo"
    assert config.output_dir == "outputs/demo"
    assert len(jobs) == 12
    assert {job.family for job in jobs} == {"MLP", "AC"}
    assert {job.seed for job in jobs} == {1, 2}
    assert {job.condition.list_type for job in jobs} == {"Seen", "Unseen"}
    assert {job.model.model_name for job in jobs} == {"MLP-01", "MLP-02", "AC-Seen"}


def test_load_suite_config_preserves_condition_fields(tmp_path: Path) -> None:
    from experiment_suite.config import load_suite_config

    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: seen-small\n"
        "output_dir: outputs/seen-small\n"
        "seeds: [7]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "    N: 16\n"
        "    num_train_lists: 8\n"
        "    num_test_lists: 2\n"
        "    k_train_min: 1\n"
        "    k_train_max: 4\n"
        "    k_test_min: 1\n"
        "    k_test_max: 8\n"
        "models:\n"
        "  AC:\n"
        "    - model_name: AC-Seen\n",
        encoding="utf-8",
    )

    config = load_suite_config(cfg_path)

    assert len(config.conditions) == 1
    condition = config.conditions[0]
    assert condition.list_type == "Seen"
    assert condition.N == 16
    assert condition.num_train_lists == 8
    assert condition.num_test_lists == 2
    assert condition.k_train_min == 1
    assert condition.k_train_max == 4
    assert condition.k_test_min == 1
    assert condition.k_test_max == 8


def test_load_suite_config_defaults_output_dir(tmp_path: Path) -> None:
    from experiment_suite.config import load_suite_config

    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: no-output-dir\n"
        "seeds: [3]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "    N: 8\n"
        "    num_train_lists: 2\n"
        "    num_test_lists: 0\n"
        "models:\n"
        "  AC:\n"
        "    - model_name: AC-Seen\n",
        encoding="utf-8",
    )

    config = load_suite_config(cfg_path)

    assert config.output_dir == "outputs/no-output-dir"


def test_load_suite_config_rejects_boolean_seed(tmp_path: Path) -> None:
    from experiment_suite.config import load_suite_config

    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text(
        "suite_name: bad-seed\n"
        "seeds: [true]\n"
        "conditions:\n"
        "  - list_type: Seen\n"
        "models:\n"
        "  AC:\n"
        "    - model_name: AC-Seen\n",
        encoding="utf-8",
    )

    try:
        load_suite_config(cfg_path)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("Expected ValueError for boolean seed")


def test_paper_family_configs_use_canonical_output_roots() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    config_expectations = {
        root / "experiments" / "seen_mlp.yaml": "outputs/experiments/seen-mlp",
        root / "experiments" / "unseen_mlp.yaml": "outputs/experiments/unseen-mlp",
        root / "experiments" / "seen_ac.yaml": "outputs/experiments/seen-ac",
        root / "experiments" / "unseen_ac.yaml": "outputs/experiments/unseen-ac",
    }

    for config_path, expected_output in config_expectations.items():
        config = load_suite_config(config_path)
        assert config.output_dir == expected_output


def test_paper_ac_family_configs_share_core_model_surface() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    seen_ac = load_suite_config(root / "experiments" / "seen_ac.yaml")
    unseen_ac = load_suite_config(root / "experiments" / "unseen_ac.yaml")

    seen_model_keys = set(seen_ac.models["AC"][0].values.keys())
    unseen_model_keys = set(unseen_ac.models["AC"][0].values.keys())

    assert seen_model_keys == unseen_model_keys
    assert {model.model_name for model in seen_ac.models["AC"]} == {"AC-01", "AC-02"}
    assert {model.model_name for model in unseen_ac.models["AC"]} == {"AC-01", "AC-02"}


def test_paper_mlp_family_configs_use_calibrated_story_ranges() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    seen_mlp = load_suite_config(root / "experiments" / "seen_mlp.yaml")
    unseen_mlp = load_suite_config(root / "experiments" / "unseen_mlp.yaml")

    seen_condition = seen_mlp.conditions[0]
    unseen_condition = unseen_mlp.conditions[0]

    assert seen_condition.N == 24
    assert seen_condition.k_train_max == 4
    assert seen_condition.k_test_max == 12
    assert unseen_condition.N == 12
    assert unseen_condition.k_train_max == 4
    assert unseen_condition.k_test_max == 10
    assert {model.model_name for model in seen_mlp.models["MLP"]} == {"MLP-01", "MLP-02", "MLP-03"}
    assert {model.model_name for model in unseen_mlp.models["MLP"]} == {"MLP-01", "MLP-02", "MLP-03"}


def test_paper_ac_family_configs_enable_canonical_trace_examples() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    seen_ac = load_suite_config(root / "experiments" / "seen_ac.yaml")
    unseen_ac = load_suite_config(root / "experiments" / "unseen_ac.yaml")

    assert seen_ac.trace_plots is not None
    assert unseen_ac.trace_plots is not None
    assert seen_ac.trace_plots["enabled"] is True
    assert unseen_ac.trace_plots["enabled"] is True
    assert int(seen_ac.trace_plots["hops"]) == 4
    assert int(unseen_ac.trace_plots["hops"]) == 4
