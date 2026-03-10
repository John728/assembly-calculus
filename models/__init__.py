from .mlp import PointerMLP, mlp_configs
from .transformer import PointerTransformer, transformer_configs
from .gnn import PointerGNN, gnn_configs
from .ssm import PointerSSM, ssm_configs
from .universal_transformer import PointerUniversalTransformer, ut_configs
from .rnn import PointerRNN, rnn_configs
from .ac import PointerAC, ac_configs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(family, config, N=64):
    if family == 'MLP':
        return PointerMLP(N=N, d_model=config['d_model'], num_layers=config['layers'], hidden_dim=config['hidden_dim'])
    elif family == 'Transformer':
        return PointerTransformer(N=N, d_model=config['d_model'], num_layers=config['layers'], nhead=config.get('nhead', 4), hidden_dim=config['hidden_dim'])
    elif family == 'GNN':
        return PointerGNN(N=N, d_model=config['d_model'], num_layers=config['layers'], hidden_dim=config['hidden_dim'])
    elif family == 'SSM':
        return PointerSSM(N=N, d_model=config['d_model'], num_layers=config['layers'], hidden_dim=config['hidden_dim'])
    elif family == 'UT':
        return PointerUniversalTransformer(N=N, d_model=config['d_model'], nhead=config.get('nhead', 4), hidden_dim=config['hidden_dim'])
    elif family == 'RNN':
        return PointerRNN(N=N, d_model=config['d_model'], num_layers=config['layers'], hidden_dim=config['hidden_dim'])
    elif family == 'AC':
        return PointerAC(N=N, cap_size=config.get('cap_size', 30), density=config.get('density', 0.2), plasticity=config.get('plasticity', 0.1))
    else:
        raise ValueError(f"Unknown family: {family}")
