from __future__ import annotations

import numpy as np

from experiment_suite.jobs import ExperimentJob
from pyac.tasks.binary_search.bs_protocol import build_bs_network, evaluate_bs_sequence
from pyac.tasks.dfa.dfa_protocol import train_dfa_transitions

def run_bs_ac_job(job: ExperimentJob) -> list[dict[str, object]]:
    rng = np.random.default_rng(job.seed)
    
    array_sizes = job.condition.extra.get("array_sizes", [16])
    
    assembly_size = int(job.model.values.get("assembly_size", 32))
    density = float(job.model.values.get("density", 1.0))
    plasticity = float(job.model.values.get("plasticity", 0.1))
    samples_per_size = int(job.model.values.get("samples_per_size", 10))
    c_values = job.model.values.get("c_values", [1])
    
    rows = []
    
    for N in array_sizes:
        network, task = build_bs_network(
            N=N,
            assembly_size=assembly_size,
            density=density,
            plasticity=plasticity,
            rng=rng,
        )
        
        rounds = int(job.model.values.get("rounds", 15))
        train_dfa_transitions(network, task, rounds=rounds, rng=rng)
        
        for _ in range(samples_per_size):
            # Generate a sorted array of size N
            A = sorted(rng.integers(0, 100, size=N))
            
            # Pick a target element
            idx = int(rng.integers(0, N))
            x = A[idx]
            
            for c in c_values:
                result = evaluate_bs_sequence(network, task, A, x, start_a=0, start_b=N-1, c=c)
                rows.append({
                    "family": job.family,
                    "model_name": job.model.model_name,
                    "seed": job.seed,
                    "list_type": "BinarySearch",
                    "experiment": "binary_search",
                    "t": c,
                    "c": c,
                    "N": N,
                    "k_test": int(np.ceil(np.log2(N))), # nominal depth
                    "L": int(np.ceil(np.log2(N))),      # nominal depth
                    "accuracy": result["path_accuracy"],
                    "correct": result["correct"],
                    "path_accuracy": result["path_accuracy"],
                    "first_error_index": result["first_error_index"] if result["first_error_index"] is not None else int(np.ceil(np.log2(N))),
                    "target": result["target"],
                    "prediction": result["prediction"],
                })
                
    return rows
