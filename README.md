# ISA-MA: Interpretable State-Aware Memetic Algorithm for BNDFS

This repository contains the official implementation, benchmark datasets, and raw run-level experimental results of the paper:
*"Beyond the Black Box: An Inherently Interpretable State-Aware Reinforcement-Learning Framework for Transit Network Design"*.

## Key Features
- **Interpretable RL Controller:** Uses tabular Q-learning to map evolutionary health states to operator selection.
- **Modular Operator Toolbox:** Supports fine-tuning for small networks and restructuring for large-scale urban grids.
- **Visual Auditability:** Automatically generates Q-table heatmaps and selection dynamics for decision transparency.
- **Rigorous Reproducibility (Updated):** Includes controlled baselines (MA-AOS), proxy validation scripts, locked random seeds, and raw 30-run data for exact statistical verification.

## Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas networkx matplotlib numpy scipy tqdm seaborn

## Project Structure
src/: Directory containing all source code (ISA_MA.py, MA_AOS.py, Statistical_Analysis.py, etc.).
data/: Directory containing benchmark datasets (mandl/, mumford0/).
results/: Directory containing raw run-level objective values (30 independent runs) and generated plots.
reproduce_results.py: The master guide script for reproducing the paper's tables and figures.  

## Quick Start
1. Ensure your dataset folders are placed in the data/ directory.
2. To see all available reproduction commands, run the master script:
-  python reproduce_results.py
3. To run the main algorithm directly, use:
-  python src/ISA_MA.py

Follow the interactive prompt:
Select your benchmark network (1: Mandl, 2: Mumford0).
Choose the algorithm mode (4: ISA-MA).
## Reproducibility
To address peer review feedback and ensure absolute scientific transparency:
- **Bug Fixes:** Path and BASE_DIR logic have been explicitly defined to ensure out-of-the-box execution on any machine.
- **Statistics Verification (Table 6):** Run python src/Statistical_Analysis.py to instantly calculate and verify the exact p-values (Mann-Whitney U), Effect Sizes (r), and 95% Confidence Intervals reported in the manuscript using the raw data in the results/ folder.
- **Proxy Validation (Appendix D):** Run python src/validate_proxy_mandl.py to reproduce the phenotypic vs. genotypic Jaccard distance correlation analysis.
## Contact
For any questions regarding the implementation, please contact the author at: xyc1477@163.com
