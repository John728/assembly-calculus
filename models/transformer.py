import torch
import torch.nn as nn
from .base import BasePointerEncoder

class PointerTransformer(nn.Module):
    def __init__(self, N, d_model, num_layers, nhead, hidden_dim):
        """Baseline: Transformer Encoder Network."""
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        
        # We need positional embeddings for the N memory items.
        # Plus 2 for q_s and q_k.
        self.pos_emb = nn.Parameter(torch.randn(1, N + 2, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, N)
        
    def forward(self, batch):
        mem, q_s, q_k = self.encoder(batch['p'], batch['s'], batch['k'])
        
        # mem: (B, N, d_model)
        # q_s: (B, d_model)
        # q_k: (B, d_model)
        
        B = mem.size(0)
        
        # (B, 1, d_model)
        q_s = q_s.unsqueeze(1)
        q_k = q_k.unsqueeze(1)
        
        # (B, N+2, d_model)
        x = torch.cat([q_s, q_k, mem], dim=1)
        
        # Add positional embedding
        x = x + self.pos_emb
        
        # Pass through transformer
        out = self.transformer(x)
        
        # Pool from the first token (q_s)
        out_pool = out[:, 0, :]
        
        logits = self.head(out_pool)
        return logits, 1 # Steps = 1 for feedforward

transformer_configs = []
_tiny_layers, _tiny_dim = 2, 64
_xl_layers, _xl_dim = 8, 1024
for i in range(50):
    # Interpolate between Tiny (2, 64) and XL (8, 1024)
    frac = i / 49.0
    layers = int(round(_tiny_layers + frac * (_xl_layers - _tiny_layers)))
    hidden_dim = int(round(_tiny_dim + frac * (_xl_dim - _tiny_dim)))
    # Ensure d_model scales somewhat proportionally, e.g. hidden_dim // 4
    d_model = max(16, hidden_dim // 4)
    # Ensure d_model is divisible by nhead (4)
    d_model = max(1, d_model // 4) * 4
    nhead = 4
    
    # We tweak the d_model and hidden_dim slightly up to try to match MLP parameter sizes 
    # since MLP flattens the input causing massive matrices, while transformers share weights across sequence.
    # To artificially match parameter scale, we scale d_model and hidden_dim by 2x for transformer.
    d_model = d_model * 2
    hidden_dim = hidden_dim * 2
    
    transformer_configs.append({
        'name': f'Trans-{i+1:02d}',
        'layers': layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model,
        'nhead': nhead
    })
