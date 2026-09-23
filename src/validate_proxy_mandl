# ==============================================================================
# validate_proxy_mandl: Validates Fitness CV as a proxy for Structural Diversity
# Generates 100 synthetic populations and calculates Pearson correlation.
# ==============================================================================

import os
import sys
import importlib.util
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path

# --- Dynamic Path Routing ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dynamically Import ISA_MA.py from the src folder
isama_path = BASE_DIR / "src" / "ISA_MA.py"
spec = importlib.util.spec_from_file_location("isama", isama_path)
isama = importlib.util.module_from_spec(spec)
sys.modules["isama"] = isama
spec.loader.exec_module(isama)

# Override config for Mandl explicitly
isama.NUM_ROUTES = 4
isama.MAX_TOTAL_FLEET = 99
isama.MAX_STOPS_PER_ROUTE = 15
isama.MIN_STOPS_PER_ROUTE = 3
isama.MIN_FREQUENCY = 2
isama.MAX_FREQUENCY = 99
isama.ALPHA_UNMET_DEMAND = 10000
isama.BUS_CAPACITY = 50
isama.MAX_TRANSFERS = 2
isama.TRANSFER_PENALTY_TIME = 5

DATA_PATH = BASE_DIR / "data" / "mandl"

# ==========================================
# Helper Functions for Structural Distance
# ==========================================
def extract_edges(individual):
    """Extracts unique, undirected road segments (edges) from an individual's routes."""
    edges = set()
    for route in individual.get('routes', []):
        if not route: continue
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edges.add(tuple(sorted((u, v))))
    return edges

def calc_population_topological_distance(population):
    """Calculates the average pairwise Edge Jaccard Distance for the population."""
    pop_edges = [extract_edges(ind) for ind in population]
    distances = []
    pop_size = len(pop_edges)

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            set1 = pop_edges[i]
            set2 = pop_edges[j]
            if not set1 and not set2:
                dist = 0.0
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                dist = 1.0 - (intersection / union) 
            distances.append(dist)
    return np.mean(distances) if distances else 0.0

# ==========================================
# Main Validation Routine
# ==========================================
def run_validation():
    # Lock the random seed to guarantee reproducible results
    random.seed(42)
    np.random.seed(42)

    print("Loading Mandl network data...")
    G, node_df, od_df, sp_len, sp, stop_nodes_set = isama.load_data(DATA_PATH)

    print("Generating a pool of 80 valid heuristic individuals...")
    pool_individuals, pool_fitnesses = [], []

    for _ in tqdm(range(80), desc="Evaluating Pool"):
        ind = isama.create_heuristic_individual(od_df, sp, sp_len, G, stop_nodes_set)
        freq = isama.calculate_demand_responsive_frequencies(ind, od_df, sp_len, stop_nodes_set)
        ind['frequencies'] = freq
        valid_ind = isama.repair_and_validate_individual(ind, G, sp_len, sp, stop_nodes_set)
        precomputed = isama._prepare_network_data_for_individual(valid_ind, sp_len, stop_nodes_set)
        metrics = isama.calculate_performance_metrics(precomputed, od_df)
        pool_individuals.append(valid_ind)
        pool_fitnesses.append(metrics['fitness'])

    print("Assembling 100 populations with varying diversity to test correlation...")
    pop_size = 30
    results = []

    for i in tqdm(range(100), desc="Analyzing Populations"):
        clone_count = random.randint(0, 28)
        random_count = pop_size - clone_count

        base_idx = random.randint(0, len(pool_individuals) - 1)
        base_ind, base_fit = pool_individuals[base_idx], pool_fitnesses[base_idx]

        current_pop = [copy.deepcopy(base_ind) for _ in range(clone_count)]
        current_fits = [base_fit for _ in range(clone_count)]

        random_indices = random.sample(range(len(pool_individuals)), random_count)
        for idx in random_indices:
            current_pop.append(copy.deepcopy(pool_individuals[idx]))
            current_fits.append(pool_fitnesses[idx])

        mean_fit = np.mean(current_fits)
        std_fit = np.std(current_fits)
        fitness_cv = (std_fit / mean_fit) if mean_fit > 0 else 0.0
        topo_distance = calc_population_topological_distance(current_pop)

        results.append({
            "Population_ID": i + 1,
            "Clone_Ratio": clone_count / pop_size,
            "Fitness_CV": fitness_cv,
            "Topological_Distance": topo_distance
        })

    df_results = pd.DataFrame(results)
    r_val, p_val = pearsonr(df_results['Fitness_CV'], df_results['Topological_Distance'])
    
    print("\n" + "=" * 50)
    print(f"Validation Complete (Mandl)!")
    print(f"Pearson Correlation (R): {r_val:.4f}")
    print(f"P-value: {p_val:.4e}")
    print("=" * 50)

    # Plotting (Publication-Ready)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(3.5, 2.6))
    x_vals = df_results['Fitness_CV'].to_numpy()
    y_vals = df_results['Topological_Distance'].to_numpy()

    plt.scatter(x_vals, y_vals, alpha=0.7, color='blue', edgecolor='k', s=15)
    m, b = np.polyfit(x_vals, y_vals, 1)
    plt.plot(x_vals, m * x_vals + b, color='red', linewidth=1.5, label=f'Trend (R={r_val:.2f})')

    plt.title("Diversity Proxy (Mandl)", fontsize=11, fontweight='bold')
    plt.xlabel("Phenotypic Diversity (Fitness CV)", fontsize=10)
    plt.ylabel("Genotypic Diversity (Jaccard)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout(pad=0.5)

    plot_path = OUTPUT_DIR / "fitness_cv_correlation_mandl.png"
    csv_path = OUTPUT_DIR / "correlation_data_mandl.csv"
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    df_results.to_csv(csv_path, index=False)
    
    print(f"\nPlot saved to: {plot_path}")
    print(f"Data saved to: {csv_path}")

if __name__ == '__main__':
    run_validation()
