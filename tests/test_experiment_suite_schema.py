from __future__ import annotations


def test_standardize_rows_aligns_family_specific_outputs() -> None:
    from experiment_suite.schema import standardize_ac_row, standardize_mlp_row

    mlp_row = {
        "List Type": "Seen",
        "Model": "MLP-01",
        "Layers": 2,
        "Dim": 64,
        "Params": 194432,
        "k": 4,
        "Accuracy": 1.0,
    }
    ac_row = {
        "List Type": "Seen",
        "Model": "AC-Seen",
        "N": 16,
        "Num Lists": 8,
        "k": 4,
        "Accuracy": 1.0,
        "Internal Steps": 5,
        "Assembly Size": 16,
        "Density": 0.15,
        "Plasticity": 0.25,
        "Transition Rounds": 12,
        "Association Steps": 2,
    }

    std_mlp = standardize_mlp_row(
        mlp_row,
        suite="demo",
        seed=1,
        N=16,
        num_train_lists=8,
        num_test_lists=0,
        k_train_min=1,
        k_train_max=4,
    )
    std_ac = standardize_ac_row(
        ac_row,
        suite="demo",
        seed=1,
        num_train_lists=8,
        num_test_lists=0,
        k_train_min=1,
        k_train_max=4,
    )

    assert std_mlp["family"] == "MLP"
    assert std_ac["family"] == "AC"
    assert std_mlp["k_test"] == 4
    assert std_ac["k_test"] == 4
    assert std_mlp["internal_steps"] is None
    assert std_ac["internal_steps"] == 5
    assert std_mlp["params"] == 194432
    assert std_ac["assembly_size"] == 16


def test_standardized_rows_share_core_schema_keys() -> None:
    from experiment_suite.schema import CORE_RESULT_FIELDS, standardize_ac_row, standardize_mlp_row

    mlp_row = {"List Type": "Seen", "Model": "MLP-01", "k": 2, "Accuracy": 0.5}
    ac_row = {"List Type": "Seen", "Model": "AC-Seen", "k": 2, "Accuracy": 0.75, "Internal Steps": 3}

    std_mlp = standardize_mlp_row(mlp_row, suite="demo", seed=2, N=8, num_train_lists=4, num_test_lists=0, k_train_min=1, k_train_max=2)
    std_ac = standardize_ac_row(ac_row, suite="demo", seed=2, N=8, num_train_lists=4, num_test_lists=0, k_train_min=1, k_train_max=2)

    assert set(CORE_RESULT_FIELDS).issubset(std_mlp.keys())
    assert set(CORE_RESULT_FIELDS).issubset(std_ac.keys())
    assert std_mlp["suite"] == std_ac["suite"] == "demo"
    assert std_mlp["seed"] == std_ac["seed"] == 2
    assert std_mlp["list_type"] == std_ac["list_type"] == "Seen"


def test_standardize_ac_row_requires_N() -> None:
    from experiment_suite.schema import standardize_ac_row

    ac_row = {"List Type": "Seen", "Model": "AC-Seen", "k": 2, "Accuracy": 0.75, "Internal Steps": 3}

    try:
        standardize_ac_row(ac_row, suite="demo", seed=2, num_train_lists=4, num_test_lists=0, k_train_min=1, k_train_max=2)
    except ValueError as exc:
        assert "requires N" in str(exc)
    else:
        raise AssertionError("Expected ValueError when N is missing")
