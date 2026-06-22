import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyac/src"))
from pyac.core.network import Network
from pyac.core.types import NetworkSpec, AreaSpec, FiberSpec, Assembly
from pyac.tasks.mnist.protocol import MnistTask, _mnist_stimulus, decode_mnist_class
from pyac.tasks.mnist.encoding import PixelAssemblyEncoder

def run_supervised_forcing():
    rng = np.random.default_rng(42)
    n = 1000
    k = 100
    p = 0.1
    beta = 0.1
    
    encoder = PixelAssemblyEncoder(rng=rng, area_name="X")
    sensory_n = encoder.num_pixels * encoder.neurons_per_pixel
    sensory_k = encoder.active_pixels * encoder.neurons_per_pixel
    
    spec = NetworkSpec(
        areas=[
            AreaSpec(name="X", n=sensory_n, k=sensory_k, dynamics_type="feedforward"),
            AreaSpec(name="Y", n=n, k=k, p_recurrent=p, dynamics_type="recurrent"),
        ],
        fibers=[FiberSpec(src="X", dst="Y", p_fiber=p)],
        beta=beta,
        step_order=["X", "Y"],
    )
    
    network = Network(spec=spec, rng=rng)
    task = MnistTask(encoder=encoder, area_map={"sensory": "X", "coding": "Y"}, n=n, k=k, p=p, beta=beta)
    
    # Generate random orthogonal targets
    print("Generating fixed random assemblies...")
    assigned = set()
    for d in range(10):
        available = list(set(range(n)) - assigned)
        if len(available) < k:
            available = list(range(n))
        indices = rng.choice(available, size=k, replace=False)
        assigned.update(indices)
        task.class_assemblies[d] = Assembly(area_name="Y", indices=np.array(indices, dtype=np.int64))
        
    print("Generating random dummy data for test...")
    images = rng.integers(0, 256, size=(500, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=500)
    
    print("Training via SUPERVISED FORCING...")
    for img, lbl in zip(images, labels):
        target_indices = task.class_assemblies[lbl].indices
        stimulus = _mnist_stimulus(network, task, img)
        
        # Clear activations
        network.activations["X"] = np.array([], dtype=np.int64)
        network.activations["Y"] = np.array([], dtype=np.int64)
        
        # Force activation by applying massive bias to target indices
        bias = np.zeros(n, dtype=np.float64)
        bias[target_indices] = 1000000.0
        
        for _ in range(2): # Settle steps
            network.step(external_stimuli={"X": stimulus}, plasticity_on=True, biases={"Y": bias})
            network.normalize("Y")

    print("Evaluating...")
    correct = 0
    for img, lbl in zip(images[:100], labels[:100]):
        stimulus = _mnist_stimulus(network, task, img)
        network.activations["X"] = np.array([], dtype=np.int64)
        network.activations["Y"] = np.array([], dtype=np.int64)
        
        for _ in range(2):
            network.step(external_stimuli={"X": stimulus}, plasticity_on=False)
            
        pred = decode_mnist_class(network.get_assembly("Y"), task)
        if pred == lbl:
            correct += 1
            
    print(f"Supervised Accuracy: {correct}%")

if __name__ == "__main__":
    run_supervised_forcing()
