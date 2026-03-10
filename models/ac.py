import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add pyac to path if not already there
repo_root = Path(__file__).parent.parent
pyac_src = repo_root / 'pyac' / 'src'
if str(pyac_src) not in sys.path:
    sys.path.insert(0, str(pyac_src))

from pyac.core.network import Network
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec, Assembly
from pyac.core.rng import make_rng

class PointerAC(nn.Module):
    """
    Assembly Calculus (AC) wrapper for the Pointer-Chasing task.
    This model uses a biologically-inspired assembly network.
    It represents nodes as disjoint assemblies and 'learns' the permutation
    by strengthening connections between assemblies that fire in sequence.
    """
    def __init__(self, N, cap_size=30, density=0.2, plasticity=0.1, seed=42):
        super().__init__()
        self.N = N
        self.cap_size = cap_size
        self.rng = make_rng(seed)
        
        # Define areas: input area and main memory area
        # We represent each node 0..N-1 as an assembly in 'memory'
        self.spec = NetworkSpec(
            areas=[
                AreaSpec(name='input', n=N * cap_size, k=cap_size, dynamics_type='feedforward'),
                AreaSpec(name='memory', n=N * cap_size, k=cap_size, dynamics_type='recurrent', p_recurrent=density),
            ],
            fibers=[
                FiberSpec(src='input', dst='memory', p_fiber=density),
            ],
            beta=plasticity,
        )
        self.net = Network(self.spec, self.rng)
        
        # Pre-defined assemblies for each node
        self.node_assemblies = []
        for i in range(N):
            # Disjoint assemblies: node i uses neurons [i*cap, (i+1)*cap)
            indices = np.arange(i * cap_size, (i + 1) * cap_size, dtype=np.int64)
            self.node_assemblies.append(Assembly(area_name='memory', indices=indices))
            
    def encode_permutation(self, p_tensor):
        """
        Hard-code or 'train' the permutation into the recurrent weights of the memory area.
        In a real AC setting, this would be done by associative learning.
        Here we directly set weights to simulate a learned permutation.
        """
        p = p_tensor.tolist()
        # For each node i, we want it to excite node p[i]
        # Recurrent weights: memory -> memory
        rec_weights = self.net.weights[('memory', 'memory')]
        
        # We zero out existing recurrent weights first for a clean encode
        rec_weights.data.fill(0.0)
        
        for i in range(self.N):
            src_indices = self.node_assemblies[i].indices
            dst_indices = self.node_assemblies[p[i]].indices
            # Set weights between source assembly and destination assembly
            # This is a sparse matrix (CSR), so we need to be careful with efficiency
            # But for small N, we can just iterate
            for src in src_indices:
                # Find which dst indices this src node is connected to via the fiber
                # Then set those weights to 1.0 (or some value)
                row_start = rec_weights.indptr[src]
                row_end = rec_weights.indptr[src + 1]
                for idx in range(row_start, row_end):
                    col = rec_weights.indices[idx]
                    if col in dst_indices:
                        rec_weights.data[idx] = 1.0
        
        self.net.normalize('memory')

    def forward(self, batch):
        """
        Perform pointer chasing.
        batch contains 'p', 's', 'k'.
        Returns:
            logits: (B, N) distribution over nodes
            steps: internal time steps spent
        """
        # Note: pyac is currently CPU-based/numpy-based, so this is a bit slow for large batches
        # and doesn't support backprop (AC is a different paradigm).
        # We'll process each item in the batch.
        
        p = batch['p'][0] # All items in batch usually share the same p for FixedPermutationDataset
        self.encode_permutation(p)
        
        s_batch = batch['s'].tolist()
        k_batch = batch['k'].tolist()
        B = len(s_batch)
        
        logits = torch.zeros(B, self.N)
        
        for i in range(B):
            s = s_batch[i]
            k = k_batch[i]
            
            # Reset network state
            for area in self.net.area_names:
                self.net.activations[area] = np.array([], dtype=np.int64)
            
            # 1. Project start node assembly into memory
            start_indices = self.node_assemblies[s].indices
            stim = np.zeros(self.N * self.cap_size)
            # In 'input' area, we just activate the assembly corresponding to node s
            # Let's assume input area also has disjoint assemblies for nodes
            stim[s * self.cap_size : (s + 1) * self.cap_size] = 10.0 # High stimulus
            
            # One step to move from input to memory
            self.net.step(external_stimuli={'input': stim}, plasticity_on=False)
            
            # 2. Run k steps to follow pointers
            # Each step in the network roughly corresponds to one hop if weights are strong
            for _ in range(k):
                self.net.step(plasticity_on=False)
            
            # 3. Readout: which assembly is most active?
            # Since we use disjoint assemblies and k-cap, one should be exactly active
            final_indices = self.net.activations['memory']
            # Compute overlap with each node assembly
            overlaps = []
            for n_idx in range(self.N):
                node_indices = self.node_assemblies[n_idx].indices
                overlap = len(np.intersect1d(final_indices, node_indices))
                overlaps.append(overlap)
            
            logits[i, np.argmax(overlaps)] = 1.0
            
        return logits.to(batch['p'].device), max(k_batch)

def ac_configs():
    # AC doesn't really have 'configs' in the same way, but we can vary density/plasticity
    return [{'name': 'AC-Standard', 'cap_size': 30, 'density': 0.2, 'plasticity': 0.1}]
