import torch
from torch.utils.data import Dataset

class FixedPermutationDataset(Dataset):
    """
    A dataset where all examples share the **same permutation** p.  This
    formulation isolates the challenge of repeated function composition:
    models must learn to apply a fixed map p multiple times.  The domain
    size N is inferred from the length of p.  Start indices and hop counts
    are sampled per example.  Hop counts can be drawn uniformly from a
    range or fixed to a specific value.  Labels are computed exactly by
    repeated application of p.
    """

    def __init__(self, p, size, k_range=(1, 10), fixed_k=None):
        super().__init__()
        self.p = p.clone() if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.long)
        self.size = size
        self.N = self.p.numel()
        self.k_range = k_range
        self.fixed_k = fixed_k

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Always use the same permutation
        p = self.p
        # Sample start state uniformly
        s = torch.randint(0, self.N, (1,)).item()
        # Determine hop count
        if self.fixed_k is not None:
            k = self.fixed_k
        else:
            k = torch.randint(self.k_range[0], self.k_range[1] + 1, (1,)).item()
        # Compute y by applying p k times
        x = s
        for _ in range(k):
            x = p[x].item()
        y = x
        # Return p as a tensor so collate stacks it across the batch
        return {
            'p': p,
            's': s,
            'k': k,
            'y': y,
        }

def collate_dict(batch):
    return {
        'p': torch.stack([item['p'] for item in batch]),
        's': torch.tensor([item['s'] for item in batch]),
        'k': torch.tensor([item['k'] for item in batch]),
        'y': torch.tensor([item['y'] for item in batch])
    }
