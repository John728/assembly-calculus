import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyac/src"))
from pyac.core.network import Network
from pyac.core.types import NetworkSpec, AreaSpec, Assembly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_empirical_oscillation():
    rng = np.random.default_rng(42)
    n = 1000
    k = 100
    
    spec = NetworkSpec(
        areas=[AreaSpec(name="Y", n=n, k=k, p_recurrent=1.0, dynamics_type="recurrent")],
        fibers=[],
        beta=0.0,
        step_order=["Y"],
    )
    
    network = Network(spec=spec, rng=rng)
    
    assembly_A = np.arange(0, k)
    assembly_B = np.arange(k, 2*k)
    
    W = np.zeros((n, n), dtype=np.float64)
    
    # A strongly excites B
    for i in assembly_A:
        for j in assembly_B:
            W[i, j] = 10.0
            
    # B strongly excites A
    for i in assembly_B:
        for j in assembly_A:
            W[i, j] = 10.0
            
    from scipy.sparse import csr_matrix
    network.weights[("Y", "Y")] = csr_matrix(W)
    network.strategies["Y"].recurrent_weights = network.weights[("Y", "Y")]
    
    network.activations["Y"] = np.array(assembly_A, dtype=np.int64)
    
    overlaps_A = [1.0] # at start
    
    print("Running AC Oscillation...")
    for t in range(10):
        network.step(plasticity_on=False)
        active = network.activations["Y"]
        
        o_A = len(np.intersect1d(active, assembly_A)) / k
        overlaps_A.append(o_A)
        print(f"t={t+1}: Overlap with A = {o_A}")
        
    o_star = 0.5
    empirical_rhos = []
    for t in range(len(overlaps_A) - 1):
        err_t = overlaps_A[t] - o_star
        err_next = overlaps_A[t+1] - o_star
        if abs(err_t) > 0.01:
            rho = err_next / err_t
            empirical_rhos.append(rho)
            
    avg_rho = np.mean(empirical_rhos)
    print(f"\nEmpirical Jacobian Eigenvalue (Pole) rho: {avg_rho:.2f}")
    
    plt.figure(figsize=(8, 5))
    time_steps = range(0, 11)
    plt.plot(time_steps, overlaps_A, 'X-', color='#ff7f0e', linewidth=2, markersize=10, label='Empirical AC Overlap')
    plt.axhline(o_star, color='k', linestyle='--', label='Theoretical Fixed Point')
    plt.title(f'Practical Limit Cycle Oscillation (Empirical $\\rho \\approx {avg_rho:.1f}$)')
    plt.xlabel('Time Step ($t$)')
    plt.ylabel('Overlap with Assembly A ($o_A$)')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    
    out_path = os.path.join(os.path.dirname(__file__), "../Theory/assets/practical_oscillation.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved empirical plot to {out_path}")

if __name__ == "__main__":
    run_empirical_oscillation()
