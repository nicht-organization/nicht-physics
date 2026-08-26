#ci
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def run_spatial_simulation(
    tft_mode="optimistic",
    grid_size=64,
    steps=250,
    seed=42,
    save_path=None,
    show_plot=False,
):
    np.random.seed(seed)

    # Strategy Codes:
    # 0 = Apophatic Cooperator (B0, PA -> 0)
    # 1 = Exploitative Defector (D, PA >> 0)
    # 2 = Tit-For-Tat (TFT)
    # 3 = Pseudo-Random (sigma-Noise)
    strategies = np.random.choice([0, 1, 2, 3], size=(grid_size, grid_size))

    # Initial Memory Posture for TFT
    # "optimistic" (TFT_DC) -> starts with C (0)
    # "defensive"  (TFT_DD) -> starts with D (1), triggering immediate Wei spike at t=0
    if tft_mode == "defensive":
        last_opponent_move = np.ones((grid_size, grid_size))
    else:
        last_opponent_move = np.zeros((grid_size, grid_size))

    resources = np.ones((grid_size, grid_size)) * 100.0
    energy = np.ones((grid_size, grid_size)) * 10.0

    # History Tracking
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

        # 2. Determine Action Dynamics
        moves = np.zeros((grid_size, grid_size))
        moves[strategies == 1] = 1
        moves[strategies == 2] = last_opponent_move[strategies == 2]
        moves[strategies == 3] = np.random.choice(
            [0, 1], size=np.sum(strategies == 3)
        )

        is_d_move = moves == 1
        is_c_move = moves == 0

        # Energy & Resource Transfers
        energy[is_c_move] += np.minimum(resources[is_c_move], 1.0)
        resources[is_c_move] -= np.minimum(resources[is_c_move], 1.0)

        energy[is_d_move] += np.minimum(resources[is_d_move], 3.5)
        resources[is_d_move] -= np.minimum(resources[is_d_move], 3.5)

        # Apply Thermal Friction Loss (Wei = lambda * PA) to all Defection actions
        friction_loss = np.sum(is_d_move) * P_A_PENALTY * 10.0
        energy[is_d_move] -= P_A_PENALTY * 10.0
        energy -= 1.0  # Metabolic baseline friction

        # Update TFT memory for spatial neighborhood transmission
        last_opponent_move = np.roll(moves, shift=1, axis=0)

        # Metrics Tracking
        history_b0.append(np.sum(strategies == 0))
        history_d.append(np.sum(strategies == 1))
        history_tft.append(np.sum(strategies == 2))
        history_rand.append(np.sum(strategies == 3))
        history_resources.append(np.mean(resources))
        history_friction.append(friction_loss)

        # 3. Selection & Neighbor Reproduction
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

                    # Mutation Rate (1%)
                    mutate = np.random.rand(len(new_s)) < 0.01
                    new_s[mutate] = np.random.choice(
                        [0, 1, 2, 3], size=np.sum(mutate)
                    )

                    strategies[dead] = new_s
                    energy[dead] = 5.0

    # Plotting Figure Output
    fig, ax1 = plt.subplots(figsize=(10, 5))

    mode_label = (
        r"$\text{TFT}_{DC}$ (Optimistic)"
        if tft_mode == "optimistic"
        else r"$\text{TFT}_{DD}$ (Defensive)"
    )

    ax1.plot(
        history_b0,
        label=r"Apophatic Baseline ($B_0, P_A \to 0$)",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        history_d,
        label=r"Exploitative Defector ($P_A \gg 0$)",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )
    ax1.plot(
        history_tft,
        label=f"Tit-for-Tat ({mode_label})",
        color="orange",
        linestyle="-.",
        linewidth=1.5,
    )
    ax1.plot(
        history_rand,
        label=r"Pseudo-Random ($\sigma$-Noise)",
        color="purple",
        linestyle=":",
        linewidth=1.5,
    )

    ax1.set_xlabel("Generations / Time Steps (t)")
    ax1.set_ylabel("Population Count")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(
        history_friction,
        color="darkred",
        linestyle=":",
        alpha=0.5,
        label=r"Thermal Friction Spike ($Wéi$)",
    )
    ax2.set_ylabel(r"Thermal Friction Spikes ($Wéi$)", color="darkred")

    title_str = f"Multi-Strategy Spatial Dynamics: Thermal Friction Selection [{tft_mode.capitalize()} Start]"
    plt.title(title_str)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[{tft_mode}] Plot successfully saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Spatial Apophatic Dynamics Simulation"
    )
    parser.add_argument(
        "--steps", type=int, default=250, help="Number of generations"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="latex/images",
        help="Directory to save output figures",
    )
    args = parser.parse_args()

    # Generate Optimistic Plot (TFT_DC) -> latex/images/multi_strategy_friction_selection-tftdc.png
    run_spatial_simulation(
        tft_mode="optimistic",
        steps=args.steps,
        save_path=os.path.join(
            args.out_dir, "multi_strategy_friction_selection-tftdc.png"
        ),
    )

    # Generate Defensive Plot (TFT_DD) -> latex/images/multi_strategy_friction_selection-tftdd.png
    run_spatial_simulation(
        tft_mode="defensive",
        steps=args.steps,
        save_path=os.path.join(
            args.out_dir, "multi_strategy_friction_selection-tftdd.png"
        ),
    )
