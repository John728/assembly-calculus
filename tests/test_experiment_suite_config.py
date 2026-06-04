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


def test_kept_unseen_configs_use_expected_output_roots() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    config_expectations = {
        root / "experiments" / "unseen_mlp_dev.yaml": "outputs/experiments/unseen-mlp-dev",
        root / "experiments" / "unseen_mlp_paper.yaml": "outputs/experiments/unseen-mlp-paper",
        root / "experiments" / "unseen_ac_proper_dev.yaml": "outputs/experiments/unseen-ac-proper-dev",
        root / "experiments" / "unseen_ac_proper_paper.yaml": "outputs/experiments/unseen-ac-proper-paper",
    }

    for config_path, expected_output in config_expectations.items():
        config = load_suite_config(config_path)
        assert config.output_dir == expected_output


def test_unseen_mlp_dev_and_paper_configs_preserve_story_ranges() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    dev_cfg = load_suite_config(root / "experiments" / "unseen_mlp_dev.yaml")
    paper_cfg = load_suite_config(root / "experiments" / "unseen_mlp_paper.yaml")

    dev_condition = dev_cfg.conditions[0]
    paper_condition = paper_cfg.conditions[0]

    assert dev_condition.list_type == "Unseen"
    assert paper_condition.list_type == "Unseen"
    assert dev_condition.N == paper_condition.N
    assert dev_condition.k_train_max == paper_condition.k_train_max
    assert paper_condition.k_test_max >= dev_condition.k_test_max
    assert len(paper_cfg.seeds) >= len(dev_cfg.seeds)
    assert {model.model_name for model in dev_cfg.models["MLP"]}
    assert {model.model_name for model in paper_cfg.models["MLP"]}


def test_proper_unseen_ac_configs_enable_trace_examples() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    dev_cfg = load_suite_config(root / "experiments" / "unseen_ac_proper_dev.yaml")
    paper_cfg = load_suite_config(root / "experiments" / "unseen_ac_proper_paper.yaml")

    assert dev_cfg.trace_plots is not None
    assert paper_cfg.trace_plots is not None
    assert dev_cfg.trace_plots["enabled"] is True
    assert paper_cfg.trace_plots["enabled"] is True
    assert str(dev_cfg.trace_plots["hops"]) == "4"
    assert str(paper_cfg.trace_plots["hops"]) == "4"


def test_only_kept_dev_and_paper_configs_exist() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    expected = {
        "unseen_mlp": ("Unseen", "MLP", "outputs/experiments/unseen-mlp-dev", "outputs/experiments/unseen-mlp-paper"),
        "unseen_ac_proper": ("Unseen", "AC", "outputs/experiments/unseen-ac-proper-dev", "outputs/experiments/unseen-ac-proper-paper"),
    }

    for stem, (list_type, family, dev_output, paper_output) in expected.items():
        dev_cfg = load_suite_config(root / "experiments" / f"{stem}_dev.yaml")
        paper_cfg = load_suite_config(root / "experiments" / f"{stem}_paper.yaml")

        assert dev_cfg.conditions[0].list_type == list_type
        assert paper_cfg.conditions[0].list_type == list_type
        assert set(dev_cfg.models.keys()) == {family}
        assert set(paper_cfg.models.keys()) == {family}
        assert dev_cfg.output_dir == dev_output
        assert paper_cfg.output_dir == paper_output


def test_proper_unseen_ac_dev_and_paper_configs_use_fixed_time_budgets() -> None:
    from experiment_suite.config import load_suite_config

    root = Path(__file__).resolve().parents[1]
    dev_cfg = load_suite_config(root / "experiments" / "unseen_ac_proper_dev.yaml")
    paper_cfg = load_suite_config(root / "experiments" / "unseen_ac_proper_paper.yaml")

    assert dev_cfg.conditions[0].list_type == "Unseen"
    assert paper_cfg.conditions[0].list_type == "Unseen"
    assert dev_cfg.output_dir == "outputs/experiments/unseen-ac-proper-dev"
    assert paper_cfg.output_dir == "outputs/experiments/unseen-ac-proper-paper"
    assert all(model.values["protocol_variant"] == "proper_unseen" for model in dev_cfg.models["AC"])
    assert all(model.values["protocol_variant"] == "proper_unseen" for model in paper_cfg.models["AC"])
    assert all(not bool(model.values.get("t_equals_k", True)) for model in dev_cfg.models["AC"])
    assert all(not bool(model.values.get("t_equals_k", True)) for model in paper_cfg.models["AC"])
    assert all(len(model.values.get("time_budgets", [])) == 1 for model in dev_cfg.models["AC"])
    assert all(len(model.values.get("time_budgets", [])) == 1 for model in paper_cfg.models["AC"])
