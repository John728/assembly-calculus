import torch
import torch.nn as nn

class BasePointerEncoder(nn.Module):
    def __init__(self, N, d_model):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        # CRITICAL FIX: Tie the embeddings!
        # The start state `s`, the position `pos`, and the value `p` all represent 
        # indices in the same domain 0..N-1. By using the same embedding matrix, 
        # the model doesn't have to relearn the mapping from scratch 3 times.
        self.emb_idx = nn.Embedding(N, d_model)
        
        # We embed k using a sine/cosine or learned up to max expected k (e.g. 200)
        self.emb_k   = nn.Embedding(256, d_model)
        
    def forward(self, p, s, k):
        """
        p: (B, N) int indices
        s: (B,) int start state
        k: (B,) int hop count
        
        Returns:
            memory: (B, N, d_model) representation of the permutation
            query: (B, d_model * 2) concatenation of start and k embeddings
        """
        mem = self.emb_idx(p)
        
        # Query
        q_s = self.emb_idx(s)
        q_k = self.emb_k(k)
        
        return mem, q_s, q_k
