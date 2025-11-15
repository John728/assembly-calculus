import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brain import k_cap, idx_to_vec, FFArea, RecurrentArea, RandomChoiceArea, ScaffoldNetwork, FSMNetwork, PFANetwork, AttentionArea

rng = np.random.default_rng(729)

n_symbol_neurons = 1000
n_state_neurons = 500
n_arc_neurons = 5000
cap_size = 70
density = 0.2
plasticity = 1e-1

fsm_net = FSMNetwork(n_symbol_neurons, n_state_neurons, n_arc_neurons, cap_size, density, plasticity)

n_symbols = 10 + 1
n_states = 3 + 2

symbols = np.arange(n_symbols * cap_size).reshape(n_symbols, cap_size)
states = np.arange(n_states * cap_size).reshape(n_states, cap_size)

transition_list = []
for mod in range(3):
    for digit in range(10):
        transition_list += [[mod, digit, (mod + digit) % 3]]
transition_list += [[0, 10, 3], [1, 10, 4], [2, 10, 4]]

n_presentations = 15
for i in range(n_presentations):
    for j, transition in enumerate(transition_list):
        fsm_net.train(symbols[transition[1]], states[transition[0]], states[transition[2]])

n_trials = 500
max_len = 6
per_len = np.zeros(max_len, dtype=int)
per_len_correct = np.zeros(max_len, dtype=int)
correct = 0

for _ in range(n_trials):
    L = int(rng.integers(1, max_len + 1))
    digits = rng.integers(0, 10, size=L)
    gt_accept = (digits.sum() % 3 == 0)
    seq = list(digits) + [10]

    fsm_net.inhibit()
    fsm_net.state_area.fire(states[0], update=False)
    for s in seq:
        fsm_net.forward(symbols[s], update=False)
    final_dense = fsm_net.read(dense=True)
    final_overlap = final_dense @ idx_to_vec(states, n_state_neurons).T / cap_size
    pred_state = int(np.argmax(final_overlap))
    pred_accept = (pred_state == 3)

    correct += int(pred_accept == gt_accept)
    per_len[L - 1] += 1
    per_len_correct[L - 1] += int(pred_accept == gt_accept)

overall_acc = correct / n_trials
per_len_acc = per_len_correct / np.maximum(per_len, 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].bar(np.arange(1, max_len + 1), per_len_acc)
ax[0].set_xlabel('Sequence length (digits)')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim(0, 1.05)
ax[0].set_title('Per-length accuracy')

ax[1].bar([0], [overall_acc], width=0.6)
ax[1].set_xticks([0])
ax[1].set_xticklabels(['Overall'])
ax[1].set_ylim(0, 1.05)
ax[1].set_title(f'Overall acc = {overall_acc:.2f}')

for a in ax:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

print(f'Overall accuracy over {n_trials} sequences: {overall_acc:.3f}')
