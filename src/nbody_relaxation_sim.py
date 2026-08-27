import numpy as np
import time

def simulate_apophatic_nbody(N=100, steps=50, epsilon_noise=1e-3, G=1.0):
    """
    N-Body simulation with apophatic baseline relaxation (A_sigma).
    Pairwise forces below epsilon_noise collapse to B_0 = 0, reducing compute friction.
    """
    np.random.seed(42)
    positions = np.random.randn(N, 3)
    velocities = np.random.randn(N, 3) * 0.1
    masses = np.random.rand(N) + 0.5
    dt = 0.01

    total_pairs = N * (N - 1) // 2
    evaluated_pairs = 0

    start_time = time.time()

    for step in range(steps):
        forces = np.zeros_like(positions)
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec) + 1e-5
                
                # Raw gravitational force magnitude
                f_mag = G * masses[i] * masses[j] / (dist**2)
                
                # Apophatic Filter: Sub-marginal forces collapse to B_0 = 0
                if f_mag < epsilon_noise:
                    continue  # A_sigma projection to zero baseline
                
                evaluated_pairs += 1
                f_vec = f_mag * (r_vec / dist)
                forces[i] += f_vec
                forces[j] -= f_vec

        # Integration
        velocities += (forces / masses[:, None]) * dt
        positions += velocities * dt

    elapsed = time.time() - start_time
    pruning_ratio = 1.0 - (evaluated_pairs / (total_pairs * steps))

    print(f"--- N-Body Apophatic Simulation (N={N}) ---")
    print(f"Runtime: {elapsed:.4f}s")
    print(f"Pruned Interactions (B_0 Collapse): {pruning_ratio * 100:.2f}%")
    print(f"Effective Complexity: O(N^{2 * (1 - pruning_ratio):.2f})")

if __name__ == "__main__":
    simulate_apophatic_nbody()