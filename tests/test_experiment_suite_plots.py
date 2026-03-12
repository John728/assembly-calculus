from __future__ import annotations

from pathlib import Path


def test_generate_seen_suite_plots_from_standardized_results(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_seen_suite_plots

    raw_results = tmp_path / "raw_results.csv"
    raw_results.write_text(
        "suite,seed,family,model_name,list_type,N,num_train_lists,num_test_lists,k_train_min,k_train_max,k_test,accuracy,internal_steps,params,runtime_ms,layers,hidden_dim,lr,epochs,assembly_size,density,plasticity,transition_rounds,association_steps\n"
        "demo,1,MLP,MLP-01,Seen,16,8,0,1,4,1,1.0,,194432,,2,64,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-02,Seen,16,8,0,1,4,1,1.0,,6264961,,4,1395,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-01,Seen,16,8,0,1,4,5,0.0,,194432,,2,64,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-02,Seen,16,8,0,1,4,5,0.65,,6264961,,4,1395,0.001,10,,,,,\n"
        "demo,1,AC,AC-Seen,Seen,16,8,0,1,4,1,1.0,2,,,,,,,16,0.15,0.25,12,2\n"
        "demo,1,AC,AC-Wide,Seen,16,8,0,1,4,1,1.0,2,,,,,,,24,0.2,0.3,16,3\n"
        "demo,1,AC,AC-Seen,Seen,16,8,0,1,4,5,1.0,6,,,,,,,16,0.15,0.25,12,2\n"
        "demo,1,AC,AC-Wide,Seen,16,8,0,1,4,5,1.0,6,,,,,,,24,0.2,0.3,16,3\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "plots"
    paths = generate_seen_suite_plots(raw_results, out_dir)

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


def test_generate_unseen_suite_plots_from_standardized_results(tmp_path: Path) -> None:
    from experiment_suite.plots import generate_unseen_suite_plots

    raw_results = tmp_path / "raw_results.csv"
    raw_results.write_text(
        "suite,seed,family,model_name,list_type,N,num_train_lists,num_test_lists,k_train_min,k_train_max,k_test,accuracy,internal_steps,params,runtime_ms,layers,hidden_dim,lr,epochs,assembly_size,density,plasticity,transition_rounds,association_steps\n"
        "demo,1,MLP,MLP-01,Unseen,16,8,4,1,4,1,0.09,,194432,,2,64,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-02,Unseen,16,8,4,1,4,1,0.14,,6264961,,4,1395,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-01,Unseen,16,8,4,1,4,5,0.03,,194432,,2,64,0.001,10,,,,,\n"
        "demo,1,MLP,MLP-02,Unseen,16,8,4,1,4,5,0.05,,6264961,,4,1395,0.001,10,,,,,\n"
        "demo,1,AC,AC-Unseen,Unseen,16,8,4,1,4,1,0.65,1,,,,,,,16,0.5,0.25,,\n"
        "demo,1,AC,AC-Unseen-Wide,Unseen,16,8,4,1,4,1,0.8,1,,,,,,,24,0.5,0.25,,\n"
        "demo,1,AC,AC-Unseen,Unseen,16,8,4,1,4,5,0.3,5,,,,,,,16,0.5,0.25,,\n"
        "demo,1,AC,AC-Unseen-Wide,Unseen,16,8,4,1,4,5,0.45,5,,,,,,,24,0.5,0.25,,\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "plots"
    paths = generate_unseen_suite_plots(raw_results, out_dir)

    expected = {
        "unseen_accuracy_vs_hop.png",
        "mlp_accuracy_heatmap_unseen.png",
        "ac_accuracy_heatmap_unseen.png",
        "ac_time_sweep_unseen.png",
        "unseen_size_vs_time_tradeoff.png",
    }
    assert {path.name for path in paths} == expected
    assert all(path.exists() for path in paths)
