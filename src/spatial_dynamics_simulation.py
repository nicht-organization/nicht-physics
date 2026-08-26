import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# Styling für klare, bildschöne Publikations-Grafiken
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def run_spatial_simulation(
    mode="multi",           # "multi", "pure_tft_c", "pure_tft_d", "pure_tft_rand"
    tft_mode="optimistic",  # "optimistic" (TFT_DC) or "defensive" (TFT_DD)
    grid_size=64,
    steps=250,
    seed=42,
    save_path=None,
    show_plot=False,
):
    np.random.seed(seed)
    total_nodes = grid_size * grid_size

    # Strategy Codes:
    # 0 = Apophatic Cooperator (B0, PA -> 0)
    # 1 = Exploitative Defector (D, PA >> 0)
    # 2 = Tit-For-Tat (TFT)
    # 3 = Pseudo-Random (sigma-Noise)
    
    if mode == "multi":
        strategies = np.random.choice([0, 1, 2, 3], size=(grid_size, grid_size))
        if tft_mode == "defensive":
            last_opponent_move = np.ones((grid_size, grid_size))
        else:
            last_opponent_move = np.zeros((grid_size, grid_size))
    else:
        strategies = np.full((grid_size, grid_size), 2)
        if mode == "pure_tft_c":
            last_opponent_move = np.zeros((grid_size, grid_size))
        elif mode == "pure_tft_d":
            last_opponent_move = np.ones((grid_size, grid_size))
        elif mode == "pure_tft_rand":
            last_opponent_move = np.random.choice([0, 1], size=(grid_size, grid_size))

    resources = np.ones((grid_size, grid_size)) * 100.0
    energy = np.ones((grid_size, grid_size)) * 10.0

    history_b0 = []
    history_d = []
    history_tft = []
    history_rand = []
    history_resources = []
    history_friction = []

    R_REGEN = 0.08
    P_A_PENALTY = 0.35  # Thermal Friction (Wei) levied on Defection moves

    for t in range(steps):
        # 1. Resource Regeneration
        resources += R_REGEN * resources * (1.0 - resources / 100.0)

        # 2. Action Dynamics
        moves = np.zeros((grid_size, grid_size))
        moves[strategies == 1] = 1
        moves[strategies == 2] = last_opponent_move[strategies == 2]
        moves[strategies == 3] = np.random.choice(
            [0, 1], size=np.sum(strategies == 3)
        )

        is_d_move = moves == 1
        is_c_move = moves == 0

        energy[is_c_move] += np.minimum(resources[is_c_move], 1.0)
        resources[is_c_move] -= np.minimum(resources[is_c_move], 1.0)

        energy[is_d_move] += np.minimum(resources[is_d_move], 3.5)
        resources[is_d_move] -= np.minimum(resources[is_d_move], 3.5)

        friction_loss = np.sum(is_d_move) * P_A_PENALTY * 10.0
        energy[is_d_move] -= P_A_PENALTY * 10.0
        energy -= 1.0  # Baseline metabolic friction

        last_opponent_move = np.roll(moves, shift=1, axis=0)

        history_b0.append(np.sum(strategies == 0))
        history_d.append(np.sum(strategies == 1))
        history_tft.append(np.sum(strategies == 2))
        history_rand.append(np.sum(strategies == 3))
        history_resources.append(np.mean(resources))
        history_friction.append(friction_loss)

        # 3. Selection & Reproduction
        dead = energy <= 0
        if np.any(dead):
            survivors = energy > 0
            if np.any(survivors):
                surv_strats = strategies[survivors]
                counts = [np.sum(surv_strats == s) for s in range(4)]
                total = sum(counts)
                if total > 0:
                    probs = [c / total for c in counts]
                    new_s = np.random.choice(
                        [0, 1, 2, 3], size=np.sum(dead), p=probs
                    )
                    mutate = np.random.rand(len(new_s)) < 0.01
                    new_s[mutate] = np.random.choice(
                        [0, 1, 2, 3], size=np.sum(mutate)
                    )
                    strategies[dead] = new_s
                    energy[dead] = 5.0

    # Large & readable Figure Output
    fig, ax1 = plt.subplots(figsize=(12, 6))

    if mode == "multi":
        mode_label = r"$\text{TFT}_{DC}$ (Optimistic)" if tft_mode == "optimistic" else r"$\text{TFT}_{DD}$ (Defensive)"
        ax1.plot(history_b0, label=r"Apophatic Baseline ($B_0, P_A \to 0$)", color="#1f77b4", linewidth=2.5)
        ax1.plot(history_d, label=r"Exploitative Defector ($P_A \gg 0$)", color="#d62728", linestyle="--", linewidth=2.0)
        ax1.plot(history_tft, label=f"Tit-for-Tat ({mode_label})", color="#ff7f0e", linestyle="-.", linewidth=2.0)
        ax1.plot(history_rand, label=r"Pseudo-Random ($\sigma$-Noise)", color="#9467bd", linestyle=":", linewidth=2.0)
        
        # Highlight 87.5% Gaussian Quantile Threshold
        quantile_val = total_nodes * 0.875
        ax1.axhline(y=quantile_val, color="gray", linestyle="--", alpha=0.6, label=r"Gaussian Quantile Bound ($\approx 87.5\%$)")
        
        title_str = f"Multi-Strategy Spatial Dynamics [{tft_mode.capitalize()} Start]"

    elif mode == "pure_tft_c":
        ax1.plot(history_tft, label=r"Pure $\text{TFT}_{DC}$ Population", color="#ff7f0e", linewidth=2.5)
        title_str = r"Pure Spatial Dynamics: 100% $\text{TFT}_{DC}$ (Full Cooperative Start)"
    elif mode == "pure_tft_d":
        ax1.plot(history_tft, label=r"Pure $\text{TFT}_{DD}$ Population", color="#d62728", linewidth=2.5)
        title_str = r"Pure Spatial Dynamics: 100% $\text{TFT}_{DD}$ (Full Defensive Start)"
    elif mode == "pure_tft_rand":
        ax1.plot(history_tft, label=r"Pure Random TFT Population", color="#8c564b", linewidth=2.5)
        title_str = "Pure Spatial Dynamics: 100% Random-Initialized TFT"

    ax1.set_xlabel("Generations / Time Steps (t)", labelpad=10)
    ax1.set_ylabel("Population Count (Nodes)", labelpad=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="center right", framealpha=0.9)

    # Second axis for Wei Friction Spike
    ax2 = ax1.twinx()
    ax2.plot(
        history_friction,
        color="#8c1515",
        linestyle=":",
        alpha=0.4,
        linewidth=1.5,
        label=r"Thermal Friction Spike ($W\acute{e}i$)",
    )
    ax2.set_ylabel(r"Thermal Friction Spikes ($W\acute{e}i$)", color="#8c1515", labelpad=10)

    plt.title(title_str, pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[{mode} / {tft_mode}] High-res plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatial Dynamics Simulation Scenarios")
    parser.add_argument("--steps", type=int, default=250, help="Number of generations")
    parser.add_argument("--out_dir", type=str, default="latex/images", help="Output directory")
    args = parser.parse_args()

    # Run all 5 scenarios
    run_spatial_simulation("multi", "optimistic", steps=args.steps, save_path=os.path.join(args.out_dir, "multi_selection-tftdc.png"))
    run_spatial_simulation("multi", "defensive", steps=args.steps, save_path=os.path.join(args.out_dir, "multi_selection-tftdd.png"))
    run_spatial_simulation("pure_tft_c", steps=args.steps, save_path=os.path.join(args.out_dir, "pure_tft_cooperative_start.png"))
    run_spatial_simulation("pure_tft_d", steps=args.steps, save_path=os.path.join(args.out_dir, "pure_tft_defensive_start.png"))
    run_spatial_simulation("pure_tft_rand", steps=args.steps, save_path=os.path.join(args.out_dir, "pure_tft_random_start.png"))