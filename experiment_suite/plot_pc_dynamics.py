import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyac.tasks.pointer.protocol import build_pointer_network, train_node_assemblies, train_seen_transitions, _reset_network
from pyac.tasks.pointer.data import generate_unique_lists

def plot_pc_neural_dynamics():
    # 1. Setup a simple Pointer Chasing Task
    num_lists = 1
    n_nodes = 5
    K = 4 # Hop Depth
    rng = np.random.default_rng(42)
    
    # 2. Generate Memory Array (the pointers)
    pointers_list = generate_unique_lists(num_lists, n_nodes, rng)
    pointers = pointers_list[0]
    
    # Let's print out the pointers for the markdown document
    print("Memory Array Table:")
    for i in range(n_nodes):
        print(f"Node {i} -> Node {pointers[i]}")
    
    # 3. Build Network
    network, task = build_pointer_network(
        num_lists=num_lists,
        list_length=n_nodes,
        assembly_size=32,
        density=1.0,
        plasticity=0.1,
        rng=rng,
    )
    
    # 4. Train node representations
    train_node_assemblies(network, task)
    
    # 5. Train the transitions
    train_seen_transitions(network, task, pointers_list, steps_per_hop=1, transition_rounds=12)
    
    start_node = 0
    cur_area = task.area_map["state"]
    input_area = task.area_map["input"]
    
    _reset_network(network)
    network.inhibit(input_area)
    
    # Create stimulus
    state_n = network.areas_by_name[cur_area].n
    start_proto = task.state_assemblies[(0, start_node)].indices
    state_stimulus = np.zeros(state_n)
    state_stimulus[start_proto] = 10.0
    
    # Settle step
    network.step(external_stimuli={cur_area: state_stimulus}, plasticity_on=False)
    
    # Record overlaps
    overlaps = []
    
    def record_overlap():
        act = network.activations.get(cur_area, np.array([]))
        ov = []
        for s in range(n_nodes):
            proto = task.state_assemblies[(0, s)].indices
            ov.append(len(np.intersect1d(act, proto)) / len(proto))
        overlaps.append(ov)
        
    record_overlap()
    
    for step in range(K):
        # Time step t
        network.step(external_stimuli=None, plasticity_on=False)
        record_overlap()
            
    overlaps = np.array(overlaps).T # shape: (n_nodes, Time)
    
    plt.figure(figsize=(8, 4))
    im = plt.imshow(overlaps, aspect='auto', cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label="Assembly Overlap Fraction")
    plt.title(f"AC Neural Dynamics: Pointer Chasing Navigation (t={K})")
    plt.xlabel("Internal Execution Time (t)")
    plt.ylabel("Node ID")
    plt.yticks(range(n_nodes))
    
    # Draw vertical lines to separate steps
    for i in range(K):
        plt.axvline(x=i + 0.5, color='white', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    
    out_dir = Path("Theory/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pc_neural_dynamics.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_pc_neural_dynamics()
