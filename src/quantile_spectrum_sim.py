import os
import numpy as np
import matplotlib.pyplot as plt

def generate_quantile_spectrum_plot(output_path="../images/quantile_spectrum.png"):
    # Sicherstellen, dass das Zielverzeichnis existiert
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    quantiles = np.linspace(0.01, 0.99, 200)
    
    # Thermische Reibung Wei(Q) & Informationsertrag I(Q)
    thermal_friction = np.exp(4.0 * quantiles) - 1.0
    info_yield = 1.0 - np.exp(-6.0 * quantiles)
    
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300, facecolor='#09090b')
    ax1.set_facecolor('#09090b')

    ax1.plot(quantiles * 100, thermal_friction, color='#ef4444', label=r'Thermal Friction $\text{W\'ei}(Q)$', linewidth=2)
    ax1.plot(quantiles * 100, info_yield * 10, color='#3b82f6', label=r'Information Yield $I(Q) \times 10$', linewidth=2)
    ax1.set_xlabel('Quantile Spectrum ($Q_0 \to Q_{100}$)', color='#a1a1aa')
    ax1.set_ylabel('Friction / Yield Scale', color='#a1a1aa')
    
    # Exakte Fixierung auf die 87.5% Suffizienzgrenze (Gauß-Varianz-Grenze)
    sufficiency_q = 87.5
    ax1.axvline(sufficiency_q, color='#f59e0b', linestyle=':', label=f'Sufficiency Peak (~{sufficiency_q:.1f}%)')

    ax1.grid(True, alpha=0.15)
    plt.title(r"Apophatic Quantile Spectrum: $I(Q)$ vs. Thermal Friction $\text{W\'ei}(Q)$", color='#f4f4f5')
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), facecolor='#18181b', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Plot erfolgreich gespeichert unter {output_path}")

if __name__ == "__main__":
    generate_quantile_spectrum_plot()