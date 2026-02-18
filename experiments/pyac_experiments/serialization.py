"""Network serialization module (save/load via pickle)."""

import pickle
from pathlib import Path

from pyac.core.network import Network


def save_network(network: Network, path: str) -> None:
    """Save network to file using pickle.
    
    Captures full state: weights, activations, strategies, spec, RNG state.
    
    Args:
        network: Network instance to save
        path: File path where pickle will be written
    """
    with open(path, "wb") as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_network(path: str) -> Network:
    """Load network from pickle file.
    
    Restores full network state including weights, activations, RNG state.
    
    Args:
        path: File path to load from
        
    Returns:
        Restored Network instance
        
    Raises:
        FileNotFoundError: If path does not exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Network file not found: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)
