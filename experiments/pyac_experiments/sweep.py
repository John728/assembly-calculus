import itertools
from datetime import datetime
from typing import Callable, Any

from pyac.core.rng import make_rng, get_entropy
from .artifacts import Artifact


def sweep(
    param_grid: dict[str, list[Any]],
    run_fn: Callable[[dict[str, Any], int], dict[str, Any]],
    n_seeds: int = 5,
    base_seed: int = 42,
) -> list[Artifact]:
    artifacts = []
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    parent_rng = make_rng(base_seed)
    seeds = parent_rng.integers(0, 2**31, size=n_seeds)
    
    for param_combo in itertools.product(*param_values):
        params = dict(zip(param_names, param_combo))
        
        for seed in seeds:
            seed_int = int(seed)
            rng = make_rng(seed_int)
            entropy = get_entropy(rng)
            
            metrics = run_fn(params, seed_int)
            
            artifact = Artifact(
                params=params,
                seed=seed_int,
                entropy=entropy,
                metrics=metrics,
                pyac_version="0.1.0",
                timestamp=datetime.now().isoformat(),
            )
            artifacts.append(artifact)
    
    return artifacts
