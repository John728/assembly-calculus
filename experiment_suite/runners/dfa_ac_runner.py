from __future__ import annotations

import numpy as np

from experiment_suite.jobs import ExperimentJob
from pyac.tasks.dfa.dfa_protocol import build_dfa_network, evaluate_dfa_sequence, train_dfa_transitions

def run_dfa_ac_job(job: ExperimentJob) -> list[dict[str, object]]:
    rng = np.random.default_rng(job.seed)
    
    n_states = int(job.condition.extra.get("num_states", 5))
    n_symbols = int(job.condition.extra.get("num_symbols", 2))
    seq_lengths = job.condition.extra.get("sequence_lengths", [10])
    
    assembly_size = int(job.model.values.get("assembly_size", 32))
    density = float(job.model.values.get("density", 1.0))
    plasticity = float(job.model.values.get("plasticity", 0.1))
    samples_per_sequence = int(job.model.values.get("samples_per_sequence", 10))
    c_values = job.model.values.get("c_values", [1])
    
    network, task = build_dfa_network(
        n_states=n_states,
        n_symbols=n_symbols,
        assembly_size=assembly_size,
        density=density,
        plasticity=plasticity,
        rng=rng,
    )
    
    rounds = int(job.model.values.get("rounds", 15))
    train_dfa_transitions(network, task, rounds=rounds, rng=rng)
    
    rows = []
    
    for L in seq_lengths:
        for _ in range(samples_per_sequence):
            start_state = int(rng.integers(0, n_states))
            sequence = [int(rng.integers(0, n_symbols)) for _ in range(L)]
            
            for c in c_values:
                result = evaluate_dfa_sequence(network, task, sequence, start_state=start_state, c=c)
                rows.append({
                    "family": job.family,
                    "model_name": job.model.model_name,
                    "seed": job.seed,
                    "list_type": "DFA",
                    "experiment": "dfa",
                    "t": c,
                    "c": c,
                    "L": L,
                    "k_test": L,
                    "accuracy": result["path_accuracy"],
                    "correct": result["correct"],
                    "path_accuracy": result["path_accuracy"],
                    "first_error_index": result["first_error_index"] if result["first_error_index"] is not None else L,
                    "target": result["target"],
                    "prediction": result["prediction"],
                })
                
    return rows
