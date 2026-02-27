from .mlp import PointerMLP, mlp_configs
from .transformer import PointerTransformer, transformer_configs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(family, config, N=64):
    if family == 'MLP':
        return PointerMLP(N=N, d_model=config['d_model'], num_layers=config['layers'], hidden_dim=config['hidden_dim'])
    elif family == 'Transformer':
        return PointerTransformer(N=N, d_model=config['d_model'], num_layers=config['layers'], nhead=config.get('nhead', 4), hidden_dim=config['hidden_dim'])
    else:
        raise ValueError(f"Unknown family: {family}")
