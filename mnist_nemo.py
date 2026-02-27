import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
repo_root = Path(__file__).parent
pyac_src = repo_root / 'pyac' / 'src'
if str(pyac_src) not in sys.path:
    sys.path.insert(0, str(pyac_src))

from pyac.core.network import Network
from pyac.core.types import AreaSpec, FiberSpec, NetworkSpec
from pyac.core.rng import make_rng

# Import new PyAC modules refactored from legacy script
from pyac.tasks.mnist.data import load_mnist
from pyac.tasks.mnist.encoders import KCapSmoothedEncoder
from pyac.tasks.mnist.protocol import train_disjoint_assemblies, generate_rollout_tensor
from pyac.tasks.mnist.metrics import evaluate_softmax, evaluate_voting


def main():
    rng = make_rng(42)
    print("Loading MNIST Dataset...")
    x_train, y_train, x_test, y_test = load_mnist('./data/mnist')
    
    n_in = 784
    n_neurons = 2000
    cap_size = 200
    sparsity = 0.1
    n_rounds = 5
    beta = 1e0
    n_train_examples = 4000
    n_test_examples = 1000
    n_total_examples = n_train_examples + n_test_examples

    spec = NetworkSpec(
        areas=[
            AreaSpec('input', n=n_in, k=cap_size, dynamics_type='feedforward'),
            AreaSpec('class', n=n_neurons, k=cap_size, p_recurrent=sparsity, dynamics_type='recurrent')
        ],
        fibers=[FiberSpec('input', 'class', sparsity)],
        beta=beta,
    )
    net = Network(spec, make_rng(42))
    encoder = KCapSmoothedEncoder(cap_size=cap_size)

    # Provide all necessary examples for readout generation
    train_subset = []
    labels_subset = []
    for c in range(10):
        class_imgs = x_train[y_train == c].reshape(-1, 28, 28)
        selected = class_imgs[:n_total_examples]
        train_subset.append(selected)
        labels_subset.append(np.full(len(selected), c))
    
    train_subset = np.concatenate(train_subset)
    labels_subset = np.concatenate(labels_subset)

    print("Training Disjoint Assemblies...")
    assemblies, activations = train_disjoint_assemblies(
        network=net,
        stimulus_area_name='input',
        target_area_name='class',
        images=train_subset,
        labels=labels_subset,
        encoder=encoder,
        n_rounds=n_rounds,
        bias_penalty=-1.0,
        input_multiplier=5.0,
        rng=rng
    )

    print("Generating Rollout Tensor...")
    outputs = generate_rollout_tensor(
        network=net,
        stimulus_area_name='input',
        target_area_name='class',
        images=train_subset,
        labels=labels_subset,
        encoder=encoder,
        n_rounds=n_rounds,
        n_examples=n_total_examples,
        input_multiplier=5.0,
        rng=rng
    )

    print("Evaluating Softmax Classification...")
    softmax_results = evaluate_softmax(
        outputs, 
        n_train_per_class=n_train_examples, 
        n_test_per_class=n_test_examples, 
        rng=rng
    )
    print(f"Softmax Train Accuracy: {softmax_results['train_accuracy']:.2%}")
    print(f"Softmax Test Accuracy: {softmax_results['test_accuracy']:.2%}")

    print("Evaluating Voting Classification...")
    voting_results = evaluate_voting(
        outputs, 
        cap_size=cap_size, 
        n_train_per_class=n_train_examples
    )
    print(f"Voting Train Accuracy: {voting_results['train_accuracy']:.2%}")
    print(f"Voting Test Accuracy: {voting_results['test_accuracy']:.2%}")

if __name__ == "__main__":
    main()
