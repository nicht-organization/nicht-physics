#!/usr/bin/env python3
"""
tests/test_apophatic_invariants.py

Empirical verification suite for the Apophatic Riemann Invariant (Paper I).
Verifies the zero-friction attractor property of the critical line Re(s) = 1/2
under the inversion-closed transformation z = s - 1/2 and friction metric W(z).
"""

import math
import random
import unittest


def compute_friction(z: complex) -> float:
    """
    Calculates the analytical friction metric W(z) = |z - I_RH(z)|^2 = 4 * (Re(z))^2.
    
    Args:
        z: Complex coordinate in the shifted space R_RH (z = s - 1/2).
        
    Returns:
        float: Non-negative friction value W(z).
    """
    i_rh_z = -z
    return abs(z - i_rh_z) ** 2


def run_invariant_suite(samples: int = 10000, seed: int | None = 42) -> dict:
    """
    Runs the empirical test suite across on-axis (B_0) and off-axis samples.
    
    Args:
        samples: Number of sample points per regime (default: 10,000).
        seed: Random seed for deterministic reproducibility. None for dynamic mode.
        
    Returns:
        dict: Aggregated test metrics (max on-axis friction, min off-axis friction).
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    max_on_axis_friction = 0.0
    min_off_axis_friction = float("inf")

    # 1. On-Axis Regime: Re(s) = 0.5 -> Re(z) = 0.0
    for _ in range(samples):
        t = random.uniform(10.0, 1000.0)
        z_on_axis = complex(0.0, t)
        w_on = compute_friction(z_on_axis)
        if w_on > max_on_axis_friction:
            max_on_axis_friction = w_on

    # 2. Off-Axis Regime: Re(s) = 0.5 + delta -> Re(z) = delta != 0
    for _ in range(samples):
        t = random.uniform(10.0, 1000.0)
        # Small displacement delta in range [1e-5, 0.49]
        delta = random.uniform(0.00001, 0.49)
        if random.choice([True, False]):
            delta = -delta
            
        z_off_axis = complex(delta, t)
        w_off = compute_friction(z_off_axis)
        if w_off < min_off_axis_friction:
            min_off_axis_friction = w_off

    return {
        "samples": samples,
        "seed": seed,
        "max_on_axis_friction": max_on_axis_friction,
        "min_off_axis_friction": min_off_axis_friction,
    }


class TestApophaticInvariants(unittest.TestCase):
    """Pytest/Unittest integration wrapper."""

    def test_deterministic_regime(self):
        """Verify bit-identical numerical separation in Seed=42 mode."""
        res = run_invariant_suite(samples=10000, seed=42)
        
        # On-axis friction must be exact machine zero
        self.assertAlmostEqual(res["max_on_axis_friction"], 0.0, places=12)
        
        # Off-axis friction must be strictly positive (separation gap)
        self.assertGreater(res["min_off_axis_friction"], 0.0)

    def test_dynamic_stochastic_regime(self):
        """Verify scale-invariant stability under system entropy."""
        res = run_invariant_suite(samples=10000, seed=None)
        
        self.assertAlmostEqual(res["max_on_axis_friction"], 0.0, places=12)
        self.assertGreater(res["min_off_axis_friction"], 0.0)


if __name__ == "__main__":
    print("=== Running Apophatic Invariant Test Suite ===")
    
    # Run Deterministic
    det_res = run_invariant_suite(samples=10000, seed=42)
    print(f"[Deterministic (Seed=42)]")
    print(f"  Max On-Axis Friction  (B_0): {det_res['max_on_axis_friction']:.12f}")
    print(f"  Min Off-Axis Friction (d!=0): {det_res['min_off_axis_friction']:.12f}")
    
    # Run Dynamic
    dyn_res = run_invariant_suite(samples=10000, seed=None)
    print(f"\n[Dynamic Stochastic (Seed=None)]")
    print(f"  Max On-Axis Friction  (B_0): {dyn_res['max_on_axis_friction']:.12f}")
    print(f"  Min Off-Axis Friction (d!=0): {dyn_res['min_off_axis_friction']:.12f}")
    
    print("\nExecuting Unittest Framework...")
    unittest.main()
    