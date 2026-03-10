import torch
import torch.nn as nn
from .base import BasePointerEncoder
from config import NUM_CONFIGS

class PointerRNN(nn.Module):
    def __init__(self, N, d_model, num_layers, hidden_dim):
        """Baseline B: Recurrent Neural Network (LSTM/GRU)."""
        super().__init__()
        self.encoder = BasePointerEncoder(N, d_model)
        
        # We'll treat the permutation as a sequence of length N
        # and use the query (s, k) to initialize or influence the RNN
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # We need a way to incorporate the query_s and query_k.
        # Let's use them to initialize the hidden state or concatenate them to the output.
        # For simplicity, let's project the combined query to the initial hidden/cell state.
        self.query_to_h = nn.Linear(d_model * 2, hidden_dim * num_layers)
        self.query_to_c = nn.Linear(d_model * 2, hidden_dim * num_layers)
        
        self.out = nn.Linear(hidden_dim, N)
        
    def forward(self, batch):
        mem, q_s, q_k = self.encoder(batch['p'], batch['s'], batch['k'])
        
        B = mem.size(0)
        
        # Combine query components
        q = torch.cat([q_s, q_k], dim=-1) # (B, d_model * 2)
        
        # Initialize hidden states from query
        h0 = self.query_to_h(q).view(self.rnn.num_layers, B, self.rnn.hidden_size)
        c0 = self.query_to_c(q).view(self.rnn.num_layers, B, self.rnn.hidden_size)
        
        # Process the "permutation sequence"
        # Each index i in the sequence is the embedding of the value p[i]
        output, (hn, cn) = self.rnn(mem, (h0, c0))
        
        # Use the last hidden state of the top layer
        logits = self.out(hn[-1])
        
        return logits, 1 # Steps = 1 for simple RNN

rnn_configs = []
_tiny_layers, _tiny_dim = 1, 64
_xl_layers, _xl_dim = 4, 512
for i in range(NUM_CONFIGS):
    # Interpolate between Tiny (1, 64) and XL (4, 512)
    # Fewer layers than MLP because RNN is deeper in time
    frac = i / max(1, NUM_CONFIGS - 1)
    layers = int(round(_tiny_layers + frac * (_xl_layers - _tiny_layers)))
    hidden_dim = int(round(_tiny_dim + frac * (_xl_dim - _tiny_dim)))
    d_model = max(16, hidden_dim // 4)
    rnn_configs.append({
        'name': f'RNN-{i+1:02d}',
        'layers': layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model
    })
