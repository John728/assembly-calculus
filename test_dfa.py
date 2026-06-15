import numpy as np
from pyac.tasks.dfa.dfa_protocol import build_dfa_network, train_dfa_transitions, evaluate_dfa_sequence

def main():
    rng = np.random.default_rng(42)
    network, task = build_dfa_network(
        n_states=5,
        n_symbols=2,
        assembly_size=32,
        density=0.2,
        plasticity=0.1,
        rng=rng
    )
    
    print("Training...")
    train_dfa_transitions(network, task, rounds=24, rng=rng)
    
    print("Evaluating...")
    # Generate a random sequence
    seq = [int(rng.integers(0, 2)) for _ in range(10)]
    result = evaluate_dfa_sequence(network, task, seq, start_state=0, c=1)
    
    print(f"Target: {result['target']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Correct: {result['correct']}")
    print(f"Path accuracy: {result['path_accuracy']}")

if __name__ == '__main__':
    main()
