import numpy as np

from pyac.tasks.dfa.dfa_protocol import build_dfa_network, train_dfa_transitions

def get_interval_id(a, b, N):
    # Map valid (a, b) to an integer ID, plus one ID for "found" state
    if a > b:
        return -1 # Invalid
    # Lexicographical index for 0 <= a <= b < N
    # For a given a, there are N-a valid b's: b in [a, N-1]
    # Sum of (N-i) for i from 0 to a-1 is a*N - a*(a-1)/2
    return int(a * N - a * (a - 1) // 2 + (b - a))

def build_bs_network(N, assembly_size=32, density=1.0, plasticity=0.1, rng=None):
    num_states = N * (N + 1) // 2 + 1 # +1 for the terminal/found state
    num_symbols = 3 # 0: < (left), 1: > (right), 2: = (found)
    
    network, task = build_dfa_network(
        n_states=num_states,
        n_symbols=num_symbols,
        assembly_size=assembly_size,
        density=density,
        plasticity=plasticity,
        rng=rng
    )
    
    # Redefine the transition function for Binary Search
    terminal_state = num_states - 1
    
    def bs_transition(state_id, symbol_id):
        if state_id == terminal_state:
            return terminal_state
        
        # Reverse map state_id to (a, b)
        a_val, b_val = -1, -1
        rem = state_id
        for a in range(N):
            cnt = N - a
            if rem < cnt:
                a_val = a
                b_val = a + rem
                break
            rem -= cnt
            
        if a_val == -1 or a_val > b_val:
            return terminal_state # Invalid state -> terminal
            
        m = (a_val + b_val) // 2
        
        if symbol_id == 0: # <
            new_a, new_b = a_val, m - 1
        elif symbol_id == 1: # >
            new_a, new_b = m + 1, b_val
        else: # ==
            new_a, new_b = m, m # Found! Or we could transition to terminal
            return terminal_state
            
        if new_a > new_b:
            return terminal_state # Not found terminal state
            
        return get_interval_id(new_a, new_b, N)
        
    task.transition = bs_transition
    
    # Overwrite the randomly generated delta with the BS transitions
    for s in range(num_states):
        for x in range(num_symbols):
            task.delta[(s, x)] = bs_transition(s, x)
            
    return network, task

from pyac.tasks.dfa.dfa_protocol import build_dfa_network, train_dfa_transitions, _clear_activations, _stimulus, _decode_state

def evaluate_bs_sequence(network, task, A, x, start_a=0, start_b=None, c=1):
    N = len(A)
    if start_b is None:
        start_b = N - 1
        
    max_steps = int(np.ceil(np.log2(N))) + 1
    
    current_state = get_interval_id(start_a, start_b, N)
    terminal_state = N * (N + 1) // 2
    
    # Ground truth
    true_states = [current_state]
    a, b = start_a, start_b
    for _ in range(max_steps):
        if a > b:
            true_states.append(terminal_state)
            break
        m = (a + b) // 2
        if x < A[m]:
            a, b = a, m - 1
        elif x > A[m]:
            a, b = m + 1, b
        else:
            true_states.append(terminal_state)
            break
        if a <= b:
            true_states.append(get_interval_id(a, b, N))
        else:
            true_states.append(terminal_state)
            
    target_len = len(true_states) - 1
            
    sym_area = task.area_map["sym"]
    cur_area = task.area_map["cur"]
    sym_n = network.areas_by_name[sym_area].n
    
    _clear_activations(network)
    network.activations[cur_area] = task.state_assemblies[current_state].indices.copy()
    
    pred_states = [current_state]
    
    for step in range(target_len):
        if current_state == terminal_state:
            sym = 2
        else:
            a_val, b_val = -1, -1
            rem = current_state
            for i in range(N):
                cnt = N - i
                if rem < cnt:
                    a_val = i
                    b_val = i + rem
                    break
                rem -= cnt
            
            if a_val == -1 or a_val > b_val:
                sym = 2 # invalid
            else:
                m = (a_val + b_val) // 2
                if x < A[m]:
                    sym = 0
                elif x > A[m]:
                    sym = 1
                else:
                    sym = 2
                    
        x_stim = _stimulus(sym_n, task.sym_assemblies[sym].indices)
        
        for step_idx in range(c):
            stimuli = {sym_area: x_stim} if step_idx == 0 else None
            network.step(external_stimuli=stimuli, plasticity_on=False)
            
        pred_state = _decode_state(task, network.activations[cur_area])
        pred_states.append(pred_state)
        current_state = pred_state
        
    path_acc = sum(1 for p, t in zip(pred_states, true_states) if p == t) / len(true_states)
    first_error = next((i for i, (p, t) in enumerate(zip(pred_states, true_states)) if p != t), None)
    
    return {
        "correct": pred_states[-1] == true_states[-1],
        "path_accuracy": path_acc,
        "first_error_index": first_error,
        "target": true_states,
        "prediction": pred_states,
    }

