import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyac.tasks.dfa.dfa_protocol import build_dfa_network, train_dfa_transitions, _clear_activations, _stimulus, _decode_state

def plot_dfa_neural_dynamics():
    # 1. Setup a simple DFA
    n_states = 5
    n_symbols = 2
    L = 10
    rng = np.random.default_rng(42)
    
    network, task = build_dfa_network(
        n_states=n_states,
        n_symbols=n_symbols,
        assembly_size=32,
        density=1.0,
        plasticity=0.1,
        rng=rng,
    )
    
    # Train it perfectly to see it running "normally" and beautifully
    train_dfa_transitions(network, task, rounds=-1, rng=rng)
    
    # Run a sequence
    start_state = 0
    sequence = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
    
    sym_area = task.area_map["sym"]
    cur_area = task.area_map["cur"]
    hidden_area = task.area_map["hidden"]
    sym_n = network.areas_by_name[sym_area].n
    
    _clear_activations(network)
    network.activations[cur_area] = task.state_assemblies[start_state].indices.copy()
    
    true_state = start_state
    
    # Record overlaps at each micro-step!
    overlaps = []
    
    def record_overlap():
        # Overlap of cur area with all states
        act = network.activations.get(cur_area, np.array([]))
        ov = []
        for s in range(n_states):
            proto = task.state_assemblies[s].indices
            ov.append(len(np.intersect1d(act, proto)) / len(proto))
        overlaps.append(ov)
        
    record_overlap()
    
    for x in sequence:
        true_state = task.delta[(true_state, x)]
        x_stim = _stimulus(sym_n, task.sym_assemblies[x].indices)
        
        # We need 3 steps for sym & cur -> hidden -> dst -> cur
        for step_idx in range(3):
            stimuli = {sym_area: x_stim} if step_idx == 0 else None
            network.step(external_stimuli=stimuli, plasticity_on=False)
            record_overlap()
            
    overlaps = np.array(overlaps).T # shape: (n_states, Time)
    
    plt.figure(figsize=(10, 4))
    im = plt.imshow(overlaps, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label="Assembly Overlap Fraction")
    plt.title(f"AC Neural Dynamics: State Assembly Activation over Time (DFA, L={L})")
    plt.xlabel("Internal Network Update Steps (t)")
    plt.ylabel("Prototype State ID")
    plt.yticks(range(n_states))
    
    # Draw vertical lines to separate sequence symbols
    for i in range(L):
        plt.axvline(x=i*3 + 0.5, color='white', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    
    out_dir = Path("Theory/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "ac_neural_dynamics.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_dfa_neural_dynamics()
