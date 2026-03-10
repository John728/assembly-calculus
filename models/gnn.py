import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePointerEncoder
from config import NUM_CONFIGS

class PointerGNN(nn.Module):
    def __init__(self, N, d_model, num_layers, hidden_dim):
        """
        Baseline C: Graph Neural Network.
        Each node i in the graph has exactly one outgoing edge to p[i].
        We perform message passing for 'num_layers' steps.
        """
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        self.N = N
        self.num_layers = num_layers
        
        # We use a simple Gated Graph Neural Network-like update
        self.message_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.update_gate = nn.GRUCell(d_model, d_model)
        
        self.combiner = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, N)
        
    def forward(self, batch):
        # p: (B, N)
        # s: (B,)
        # k: (B,)
        p, s, k = batch['p'], batch['s'], batch['k']
        mem, q_s, q_k = self.encoder(p, s, k)
        
        B = p.size(0)
        # Initial node states H: (B, N, d_model)
        # We use the embedded indices as initial states
        H = self.encoder.emb_idx(torch.arange(self.N, device=p.device).expand(B, self.N))
        
        # Message passing for a fixed number of layers
        # Note: This is "non-recurrent" in the sense that the number of layers is fixed,
        # but the weights are shared across steps (standard GNN).
        for _ in range(self.num_layers):
            # Each node i receives a message from its predecessor.
            # In our case, the edge is i -> p[i]. 
            # So node p[i] receives a message from i.
            # We want to propagate information 'forward' along the edges.
            
            # messages[batch, p[i]] = H[batch, i]
            # This is a bit tricky with batching. 
            # Let's use scatter to send messages:
            # We want H_new[p[i]] = f(H[i])
            
            # Compute messages from each node: (B, N, d_model)
            msgs = self.message_net(H)
            
            # Scatter messages to the targets p[i]
            # p: (B, N), msgs: (B, N, d_model)
            # We need to expand p to match msgs dimensions for scatter
            target_indices = p.unsqueeze(-1).expand(-1, -1, H.size(-1))
            
            # Aggregate messages (since it's a permutation, each node has exactly one incoming edge)
            agg_msgs = torch.zeros_like(msgs)
            agg_msgs.scatter_(1, target_indices, msgs)
            
            # Update node states using GRU-like gate
            H_flat = H.view(-1, H.size(-1))
            agg_msgs_flat = agg_msgs.view(-1, agg_msgs.size(-1))
            H_new_flat = self.update_gate(agg_msgs_flat, H_flat)
            H = H_new_flat.view(B, self.N, -1)
            
        # After message passing, extract the state of the start node 's'
        # s: (B,)
        # We also want to incorporate q_k (the hop count query)
        
        # Gather state of start node s
        # s_indices: (B, 1, d_model)
        s_indices = s.view(B, 1, 1).expand(-1, 1, H.size(-1))
        s_state = H.gather(1, s_indices).squeeze(1)
        
        # Combine with hop count embedding
        final_rep = torch.cat([s_state, q_k], dim=-1)
        combined = F.gelu(self.combiner(final_rep))
        logits = self.head(combined)
        
        return logits, 1 # One-shot prediction

gnn_configs = []
_tiny_layers, _tiny_dim = 2, 64
_xl_layers, _xl_dim = 8, 512 # GNNs are often smaller per layer
for i in range(NUM_CONFIGS):
    frac = i / max(1, NUM_CONFIGS - 1)
    layers = int(round(_tiny_layers + frac * (_xl_layers - _tiny_layers)))
    hidden_dim = int(round(_tiny_dim + frac * (_xl_dim - _tiny_dim)))
    d_model = max(16, hidden_dim // 2)
    gnn_configs.append({
        'name': f'GNN-{i+1:02d}',
        'layers': layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model
    })
