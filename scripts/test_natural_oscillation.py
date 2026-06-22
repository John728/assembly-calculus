import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyac/src"))
from pyac.core.network import Network
from pyac.core.types import NetworkSpec, AreaSpec, FiberSpec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_natural_oscillation():
    rng = np.random.default_rng(42)
    n = 1000
    k = 50
    p = 0.1
    beta = 0.5 # High plasticity to learn the sequence quickly
    
    spec = NetworkSpec(
        areas=[
            AreaSpec(name="X", n=100, k=10, dynamics_type="feedforward"),
            AreaSpec(name="Y", n=n, k=k, p_recurrent=p, dynamics_type="recurrent")
        ],
        fibers=[FiberSpec(src="X", dst="Y", p_fiber=p)],
        beta=beta,
        step_order=["X", "Y"],
    )
    
    network = Network(spec=spec, rng=rng)
    
    # Create two distinct stimuli
    stimulus_1 = np.zeros(100)
    stimulus_1[:10] = 1.0
    
    stimulus_2 = np.zeros(100)
    stimulus_2[10:20] = 1.0
    
    print("Training the network on an alternating sequence (Stim 1 -> Stim 2 -> Stim 1...)")
    # We train by alternating the stimuli.
    # Because plasticity links pre (t-1) to post (t),
    # Stim 1 -> Stim 2 strengthens A -> B
    # Stim 2 -> Stim 1 strengthens B -> A
    
    for epoch in range(10):
        # Sequence: 1, 2, 1, 2
        for _ in range(5):
            network.step(external_stimuli={"X": stimulus_1}, plasticity_on=True)
            network.step(external_stimuli={"X": stimulus_2}, plasticity_on=True)
            
        network.normalize("Y")
        
    print("Training complete. Finding the assemblies for Stim 1 and Stim 2...")
    # Find assembly A
    network.activations["Y"] = np.array([], dtype=np.int64)
    network.step(external_stimuli={"X": stimulus_1}, plasticity_on=False)
    assembly_A = network.activations["Y"].copy()
    
    # Find assembly B
    network.activations["Y"] = np.array([], dtype=np.int64)
    network.step(external_stimuli={"X": stimulus_2}, plasticity_on=False)
    assembly_B = network.activations["Y"].copy()
    
    intersection = len(np.intersect1d(assembly_A, assembly_B))
    print(f"Assembly A size: {len(assembly_A)}, Assembly B size: {len(assembly_B)}, Overlap: {intersection}")
    
    print("Testing autonomous dynamics (Transient Stimulus 1 -> Remove Stimulus)")
    
    # Kickoff with Stimulus 1
    network.activations["Y"] = np.array([], dtype=np.int64)
    network.step(external_stimuli={"X": stimulus_1}, plasticity_on=False)
    
    overlaps_A = []
    overlaps_B = []
    
    # Now remove stimulus and let recurrent dynamics run
    for t in range(15):
        network.step(external_stimuli={"X": np.zeros(100)}, plasticity_on=False)
        active = network.activations["Y"]
        
        o_A = len(np.intersect1d(active, assembly_A)) / k
        o_B = len(np.intersect1d(active, assembly_B)) / k
        
        overlaps_A.append(o_A)
        overlaps_B.append(o_B)
        
        print(f"t={t}: Overlap A = {o_A:.2f}, Overlap B = {o_B:.2f}")

    # Create Plot
    plt.figure(figsize=(9, 5))
    time_steps = range(15)
    plt.plot(time_steps, overlaps_A, 'X-', color='#1f77b4', linewidth=2, markersize=8, label='Overlap with Assembly A')
    plt.plot(time_steps, overlaps_B, 'o-', color='#d62728', linewidth=2, markersize=8, label='Overlap with Assembly B')
    plt.title('Natural Limit Cycle via Temporal Alternation (Unsupervised AC)')
    plt.xlabel('Time Step ($t$)')
    plt.ylabel('Overlap ($o$)')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = os.path.join(os.path.dirname(__file__), "../Theory/assets/natural_temporal_oscillation.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved natural oscillation plot to {out_path}")

if __name__ == "__main__":
    run_natural_oscillation()
