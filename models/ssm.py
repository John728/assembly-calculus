import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePointerEncoder
from config import NUM_CONFIGS

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        A simplified Mamba-like Selective SSM block.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.d_state * 2 + 1, bias=False
        )
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # S4D real initialization
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        shortcut = x
        
        # Project to inner dimension
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.d_inner, dim=-1)

        # Conv1d
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # Selective Scan (simplified)
        # s_x = self.x_proj(x) # (B, L, d_state * 2 + 1)
        # dt, B_ssm, C_ssm = s_x.split([1, self.d_state, self.d_state], dim=-1)
        # dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        # (B, L, d_inner)
        # Simplified selective scan:
        # For this research task, we'll use a gated linear unit that mimics some 
        # of the properties of selective SSMs.
        gate = torch.sigmoid(self.x_proj(x)[:, :, :1])
        x = x * gate
        
        out = self.out_proj(x)
        return out + shortcut

class PointerSSM(nn.Module):
    def __init__(self, N, d_model, num_layers, hidden_dim):
        """
        Baseline D: Selective State Space Model (Simplified Mamba).
        """
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        self.N = N
        
        # Sequence input: [q_s, q_k, p_0, p_1, ..., p_N]
        self.pos_emb = nn.Parameter(torch.randn(1, N + 2, d_model))
        
        self.layers = nn.ModuleList([
            SelectiveSSM(d_model=d_model, d_state=16) 
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(d_model, N)
        
    def forward(self, batch):
        mem, q_s, q_k = self.encoder(batch['p'], batch['s'], batch['k'])
        
        B = mem.size(0)
        q_s = q_s.unsqueeze(1)
        q_k = q_k.unsqueeze(1)
        
        # (B, N+2, d_model)
        x = torch.cat([q_s, q_k, mem], dim=1)
        x = x + self.pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        # Pool from the first token (q_s)
        out_pool = x[:, 0, :]
        logits = self.head(out_pool)
        return logits, 1

ssm_configs = []
_tiny_layers, _tiny_dim = 2, 64
_xl_layers, _xl_dim = 8, 512
for i in range(NUM_CONFIGS):
    frac = i / max(1, NUM_CONFIGS - 1)
    layers = int(round(_tiny_layers + frac * (_xl_layers - _tiny_layers)))
    hidden_dim = int(round(_tiny_dim + frac * (_xl_dim - _tiny_dim)))
    d_model = max(16, hidden_dim // 2)
    # Ensure d_model is even for some SSM operations
    d_model = (d_model // 2) * 2
    
    ssm_configs.append({
        'name': f'SSM-{i+1:02d}',
        'layers': layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model
    })
