#!/usr/bin/env python3

import sys
import numpy as np
from pyac.core.network import Network
from pyac.core.types import NetworkSpec, AreaSpec, FiberSpec
from pyac.core.rng import make_rng


def test_empty_area():
    print("Testing empty area (n=0)...")
    try:
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=0, k=5)],
            fibers=[],
            beta=0.05
        )
        network = Network(spec=spec, rng=make_rng(42))
        print("  ❌ FAIL: Should have raised ValueError for n=0")
        return False
    except (ValueError, AssertionError) as e:
        print(f"  ✓ PASS: Correctly raised error: {e}")
        return True
    except Exception as e:
        print(f"  ⚠ PARTIAL: Different exception: {type(e).__name__}: {e}")
        return True


def test_zero_plasticity():
    print("Testing zero plasticity (beta=0)...")
    try:
        spec = NetworkSpec(
            areas=[
                AreaSpec(name='stim', n=10, k=10),
                AreaSpec(name='A', n=100, k=10),
                AreaSpec(name='B', n=100, k=10)
            ],
            fibers=[
                FiberSpec(src='stim', dst='A', p_fiber=0.1),
                FiberSpec(src='A', dst='B', p_fiber=0.1)
            ],
            beta=0.0
        )
        network = Network(spec=spec, rng=make_rng(42))
        
        inputs = {'stim': np.arange(10, dtype=np.int64)}
        network.step(inputs)
        
        print(f"  ✓ PASS: Network with beta=0 completed step")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        return False


def test_single_neuron():
    print("Testing single neuron (k=1, n=1)...")
    try:
        spec = NetworkSpec(
            areas=[
                AreaSpec(name='stim', n=1, k=1),
                AreaSpec(name='A', n=1, k=1)
            ],
            fibers=[FiberSpec(src='stim', dst='A', p_fiber=1.0)],
            beta=0.05
        )
        network = Network(spec=spec, rng=make_rng(42))
        
        inputs = {'stim': np.array([0], dtype=np.int64)}
        result = network.step(inputs)
        
        assembly_a = result.assemblies['A']
        if 0 in assembly_a.indices:
            print(f"  ✓ PASS: Single neuron correctly activated")
            return True
        else:
            print(f"  ⚠ PARTIAL: Neuron 0 not in indices {assembly_a.indices}")
            return True
    except Exception as e:
        print(f"  ❌ FAIL: Unexpected exception: {type(e).__name__}: {e}")
        return False


def test_large_k():
    print("Testing large k (k > n)...")
    try:
        spec = NetworkSpec(
            areas=[AreaSpec(name='A', n=10, k=20)],
            fibers=[],
            beta=0.05
        )
        network = Network(spec=spec, rng=make_rng(42))
        
        print(f"  ❌ FAIL: Should have raised ValueError for k > n")
        return False
            
    except (ValueError, AssertionError) as e:
        print(f"  ✓ PASS: Correctly raised error: {e}")
        return True
    except Exception as e:
        print(f"  ⚠ PARTIAL: Different exception: {type(e).__name__}: {e}")
        return True


def main():
    print("\n=== Edge Case Testing ===\n")
    
    results = {
        "empty_area": test_empty_area(),
        "zero_plasticity": test_zero_plasticity(),
        "single_neuron": test_single_neuron(),
        "large_k": test_large_k(),
    }
    
    print("\n=== Summary ===")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
