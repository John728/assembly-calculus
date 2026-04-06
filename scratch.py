import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/johnh/Documents/assembly-calculus")
sys.path.insert(0, str(ROOT / "pyac" / "src"))

from pyac.core.rng import make_rng, spawn_rngs
from pyac.tasks.pointer import generate_unique_lists
from pyac.tasks.pointer.proper_unseen_protocol import (
    build_proper_unseen_pointer_network,
    train_query_primitive,
    train_writeback_primitive,
    rollout_proper_unseen_pointer,
    write_unseen_episode,
)

root_rng = make_rng(109)
list_rng, net_rng = spawn_rngs(root_rng, 2)
pointers = generate_unique_lists(8, 6, list_rng)
test_lists = pointers[4:]

network, task = build_proper_unseen_pointer_network(
    list_length=6,
    assembly_size=10,
    density=0.4,
    plasticity=0.25,
    rng=net_rng,
)

train_query_primitive(network, task, rounds=8)
train_writeback_primitive(network, task, rounds=8)

pointer = test_lists[0]
write_unseen_episode(network, task, np.asarray(pointer, dtype=np.int64), write_rounds=2)

print("Pointer mapping:", pointer)

# Try different internal_steps values
for steps in [1, 2, 3, 4, 5]:
    trace = rollout_proper_unseen_pointer(network, task, np.asarray(pointer, dtype=np.int64), start_node=0, hops=1, internal_steps=steps)
    print(f"\n--- internal_steps = {steps} ---")
    print("Start node:", 0, "Target node:", pointer[0])
    print("CUR trace:", trace["cur_nodes"])
    print("SRC trace:", trace["src_nodes"])
    print("DST trace:", trace["dst_nodes"])
    print("Final prediction:", trace["final_prediction"])
