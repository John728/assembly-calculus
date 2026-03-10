import torch
import torch.nn as nn
from .base import BasePointerEncoder
from config import NUM_CONFIGS

class PointerUniversalTransformer(nn.Module):
    def __init__(self, N, d_model, nhead, hidden_dim, max_steps=100):
        """
        Recursive Model: Universal Transformer.
        A single Transformer layer is applied iteratively.
        The number of iterations can be dynamic based on k.
        """
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        self.N = N
        self.max_steps = max_steps
        
        # Positional embedding for [q_s, q_k, p_0...p_N]
        self.pos_emb = nn.Parameter(torch.randn(1, N + 2, d_model))
        
        # Timestep embedding (optional, but standard for UT)
        # We'll use a simple learned embedding for the current iteration step
        self.step_emb = nn.Embedding(max_steps + 1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
            activation='gelu'
        )
        # Only ONE layer, which we will reuse
        self.transformer_layer = encoder_layer
        
        self.head = nn.Linear(d_model, N)
        
    def forward(self, batch):
        p, s, k = batch['p'], batch['s'], batch['k']
        mem, q_s, q_k = self.encoder(p, s, k)
        
        B = mem.size(0)
        q_s = q_s.unsqueeze(1)
        q_k = q_k.unsqueeze(1)
        
        # Initial state: (B, N+2, d_model)
        x = torch.cat([q_s, q_k, mem], dim=1)
        x = x + self.pos_emb
        
        # Determine number of steps
        # We take the maximum k in the batch and iterate up to that.
        # This allows the model to compute as much as the longest hop count needs.
        num_steps = k.max().item()
        num_steps = min(num_steps, self.max_steps)
        
        # We'll use a loop to apply the layer num_steps times.
        # For a truly 'dynamic' depth per example, we could use ACT or 
        # simply mask out updates for examples that have already reached their k.
        # But here, we'll just run it k_max times to see if it generalizes.
        
        for t in range(num_steps):
            # Create a mask for samples that still need hops (B, 1, 1)
            # A sample with k=5 should be updated when t=0,1,2,3,4
            mask = (k > t).view(B, 1, 1).float()
            
            # Standard UT adds a step embedding at each iteration
            # so the model knows 'when' it is.
            t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)
            s_emb = self.step_emb(t_tensor).unsqueeze(1) # (B, 1, d_model)
            
            # Add step embedding to all tokens before layer application
            x_step = x + s_emb
            
            # Apply the shared layer and update only if not "done"
            new_x = self.transformer_layer(x_step)
            x = x * (1 - mask) + new_x * mask
            
        # Pool from the first token (q_s)
        out_pool = x[:, 0, :]
        logits = self.head(out_pool)
        
        return logits, num_steps

ut_configs = []
for i in range(NUM_CONFIGS):
    frac = i / max(1, NUM_CONFIGS - 1)
    # Scale parameters similarly to transformer
    layers = 1 # Always 1 shared layer
    hidden_dim = int(round(64 + frac * (1024 - 64)))
    d_model = max(16, hidden_dim // 4)
    nhead = 4
    
    d_model = int(d_model * 1.5)
    hidden_dim = int(hidden_dim * 1.5)
    d_model = max(1, d_model // nhead) * nhead
    
    ut_configs.append({
        'name': f'UT-{i+1:02d}',
        'layers': layers, # In UT context, this is 1 physical layer
        'hidden_dim': hidden_dim,
        'd_model': d_model,
        'nhead': nhead
    })
