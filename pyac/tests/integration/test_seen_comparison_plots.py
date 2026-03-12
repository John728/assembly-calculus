from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "plot_seen_ac_vs_mlp.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_seen_ac_vs_mlp", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_seen_comparison_df_normalizes_mlp_and_ac(tmp_path: Path) -> None:
    import pandas as pd

    mlp_csv = tmp_path / "mlp.csv"
    mlp_csv.write_text(
        "List Type,Model,Layers,Dim,Params,k,Accuracy\n"
        "Seen,MLP-01,2,64,194432,1,1.0\n"
        "Seen,MLP-01,2,64,194432,5,0.0024\n",
        encoding="utf-8",
    )

    ac_csv = tmp_path / "ac.csv"
    ac_csv.write_text(
        "List Type,Model,Num Lists,N,k,Accuracy,Internal Steps,Assembly Size,Density,Plasticity,Transition Rounds,Association Steps\n"
        "Seen,AC-Seen,8,16,1,1.0,2,16,0.15,0.25,12,2\n"
        "Seen,AC-Seen,8,16,5,1.0,6,16,0.15,0.25,12,2\n",
        encoding="utf-8",
    )

    module = _load_module()
    df = module.build_seen_comparison_df(mlp_csv, ac_csv)

    assert list(df["family"].astype(str)) == ["MLP", "MLP", "AC", "AC"]
    assert list(df["list_type"]) == ["Seen", "Seen", "Seen", "Seen"]
    assert list(df["k"]) == [1, 5, 1, 5]
    assert list(df["accuracy"]) == [1.0, 0.0024, 1.0, 1.0]
    assert pd.isna(df["internal_steps"].iloc[0])
    assert pd.isna(df["internal_steps"].iloc[1])
    assert list(df["internal_steps"].iloc[2:]) == [2, 6]


def test_generate_seen_comparison_plots_writes_expected_pngs(tmp_path: Path) -> None:
    mlp_csv = tmp_path / "mlp.csv"
    mlp_csv.write_text(
        "List Type,Model,Layers,Dim,Params,k,Accuracy\n"
        "Seen,MLP-01,2,64,194432,1,1.0\n"
        "Seen,MLP-01,2,64,194432,2,1.0\n"
        "Seen,MLP-01,2,64,194432,3,1.0\n"
        "Seen,MLP-01,2,64,194432,4,1.0\n"
        "Seen,MLP-01,2,64,194432,5,0.0\n",
        encoding="utf-8",
    )

    ac_csv = tmp_path / "ac.csv"
    ac_csv.write_text(
        "List Type,Model,Num Lists,N,k,Accuracy,Internal Steps,Assembly Size,Density,Plasticity,Transition Rounds,Association Steps\n"
        "Seen,AC-Seen,8,16,1,1.0,2,16,0.15,0.25,12,2\n"
        "Seen,AC-Seen,8,16,2,1.0,3,16,0.15,0.25,12,2\n"
        "Seen,AC-Seen,8,16,3,1.0,4,16,0.15,0.25,12,2\n"
        "Seen,AC-Seen,8,16,4,1.0,5,16,0.15,0.25,12,2\n"
        "Seen,AC-Seen,8,16,5,1.0,6,16,0.15,0.25,12,2\n",
        encoding="utf-8",
    )

    module = _load_module()
    out_dir = tmp_path / "plots"
    paths = module.generate_seen_comparison_plots(mlp_csv, ac_csv, out_dir)

    expected = {
        "accuracy_vs_hop_seen.png",
        "accuracy_vs_hop_seen_best_mlp_vs_ac.png",
        "mlp_accuracy_heatmap_seen.png",
        "ac_accuracy_heatmap_seen.png",
        "mlp_size_tradeoff_seen.png",
        "max_solved_hop_seen.png",
        "ac_time_vs_hop.png",
        "paper_panel_seen_comparison.png",
    }
    assert {path.name for path in paths} == expected
    assert all(path.exists() for path in paths)
