import torch
import torch.nn as nn
from .base import BasePointerEncoder

class PointerMLP(nn.Module):
    def __init__(self, N, d_model, num_layers, hidden_dim):
        """Baseline A: Fixed depth Feedforward Network."""
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        
        # Flattened sequence + query_s + query_k
        in_dim = (N * d_model) + (d_model * 2)
        
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim
            
        layers.append(nn.Linear(curr_dim, N)) # Output logits
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, batch):
        mem, q_s, q_k = self.encoder(batch['p'], batch['s'], batch['k'])
        
        B = mem.size(0)
        # Flatten memory
        mem_flat = mem.view(B, -1)
        
        # Combine
        x = torch.cat([mem_flat, q_s, q_k], dim=-1)
        
        logits = self.mlp(x)
        return logits, 1 # Steps = 1 for MLP

mlp_configs = []
_tiny_layers, _tiny_dim = 2, 64
_xl_layers, _xl_dim = 8, 1024
for i in range(50):
    # Interpolate between Tiny (2, 64) and XL (8, 1024)
    frac = i / 49.0
    layers = int(round(_tiny_layers + frac * (_xl_layers - _tiny_layers)))
    hidden_dim = int(round(_tiny_dim + frac * (_xl_dim - _tiny_dim)))
    # Ensure d_model scales somewhat proportionally, e.g. hidden_dim // 4
    d_model = max(16, hidden_dim // 4)
    mlp_configs.append({
        'name': f'MLP-{i+1:02d}',
        'layers': layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model
    })


