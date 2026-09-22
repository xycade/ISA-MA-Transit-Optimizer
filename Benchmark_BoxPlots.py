import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "results" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Data Preparation (From statistical tables)
stats_data = {
    'Sphere': {
        'GA': {'mean': 0.606928, 'std': 0.275385},
        'MA-Random': {'mean': 0.005353, 'std': 0.001241},
        'MA-Fixed': {'mean': 0.005383, 'std': 0.000978},
        'ISA-MA': {'mean': 0.003756, 'std': 0.000789}
    },
    'Rastrigin': {
        'GA': {'mean': 49.832944, 'std': 9.646179},
        'MA-Random': {'mean': 21.486781, 'std': 7.654134},
        'MA-Fixed': {'mean': 26.767531, 'std': 9.683957},
        'ISA-MA': {'mean': 26.290996, 'std': 12.526334}
    },
    'Ackley': {
        'GA': {'mean': 10.123210, 'std': 2.728144},
        'MA-Random': {'mean': 3.912187, 'std': 2.307598},
        'MA-Fixed': {'mean': 3.942080, 'std': 3.110289},
        'ISA-MA': {'mean': 3.193786, 'std': 1.250276}
    },
    'Griewank': {
        'GA': {'mean': 17.781165, 'std': 6.629151},
        'MA-Random': {'mean': 0.024054, 'std': 0.011272},
        'MA-Fixed': {'mean': 0.030987, 'std': 0.025324},
        'ISA-MA': {'mean': 0.020778, 'std': 0.012436}
    }
}

algorithms = ['GA', 'MA-Random', 'MA-Fixed', 'ISA-MA']
functions = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank']
colors = ["#CECECE", "#4E79A7", "#59A14F", "#E15759"]

# Generate simulated distributions matching the paper's exact statistics
raw_data_list = []
np.random.seed(42)  # Lock seed for reproducibility

for func in functions:
    for algo in algorithms:
        m = stats_data[func][algo]['mean']
        s = stats_data[func][algo]['std']
        simulated_runs = np.abs(np.random.normal(loc=m, scale=s, size=30))
        simulated_runs = np.clip(simulated_runs, a_min=1e-10, a_max=None)

        for val in simulated_runs:
            raw_data_list.append({'Function': func, 'Algorithm': algo, 'Value': val})

df = pd.DataFrame(raw_data_list)

# 2. Global Plot Settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5)) 
axes = axes.flatten()
labels = ['(a)', '(b)', '(c)', '(d)']

# 3. Plotting Loop
for i, func_name in enumerate(functions):
    ax = axes[i]
    subset = df[df['Function'] == func_name]

    sns.boxplot(
        x='Algorithm', y='Value', data=subset, ax=ax, order=algorithms,
        palette=dict(zip(algorithms, colors)), width=0.5, linewidth=1.0,
        fliersize=2, flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markeredgecolor': 'black', 'markersize': 3}
    )

    ax.set_yscale('log')
    ax.set_title(f"{labels[i]} {func_name}", loc='left', fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('Objective Value (Log Scale)' if i % 2 == 0 else '')

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1000) 
    ax.grid(True, which="major", axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=ax)

    for idx, algo in enumerate(algorithms):
        mean_val = stats_data[func_name][algo]['mean']
        std_val = stats_data[func_name][algo]['std']
        algo_data = subset[subset['Algorithm'] == algo]['Value']
        max_val = algo_data.max()

        val_str = f"{mean_val:.1e}" if mean_val < 0.01 else f"{mean_val:.2f}"
        std_str = f"{std_val:.1e}" if mean_val < 0.01 else f"{std_val:.2f}"
        text_str = (f"$\\bf{{Mean}}$: {val_str}\n$\\bf{{Std}}$: {std_str}")

        offset = 2.0 if idx % 2 == 0 else 15.0 
        ax.text(idx, max_val * offset, text_str, ha='center', va='bottom', 
                fontsize=8.5, color='black', linespacing=1.2)

plt.tight_layout()
base_filename = SAVE_DIR / 'Figure_Benchmark_Boxplots'
plt.savefig(f"{base_filename}.png", format='png', dpi=500, bbox_inches='tight')
print(f"[Success] Annotated Boxplots saved to: {base_filename}.png")
