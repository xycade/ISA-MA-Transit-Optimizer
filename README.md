# ISA-MA: Interpretable State-Aware Memetic Algorithm for BNDFS

This repository contains the official implementation of the paper: 
*"Beyond the Black Box: An Inherently Interpretable State-Aware Reinforcement Learning Framework for Trustworthy Transit Network Design"*.

##  Key Features
- **Interpretable RL Controller:** Uses tabular Q-learning to map evolutionary health states to operator selection.
- **Modular Operator Toolbox:** Supports fine-tuning for small networks and restructuring for large-scale urban grids.
- **Visual Auditability:** Automatically generates Q-table heatmaps and selection dynamics for decision transparency.

##  Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas networkx matplotlib numpy tqdm seaborn

##  Project Structure
main.py: The main entry point for the algorithm.
data/: Directory containing benchmark datasets (mandl/, mumford0/).
output/: Directory where results (CSV/PNG) will be saved automatically.

##  Quick Start
1.Place your dataset folders into the data/ directory.
2.Run the main script: python main.py
3.Follow the interactive prompt:
Select your benchmark network (1: Mandl, 2: Mumford0).
Choose the algorithm mode (4: MA-RL).

##  Reproducibility
The framework reproduces the results reported in Table 5 and Table 6 of the manuscript.
All visualization files (fitness curves, Q-table heatmaps) are saved in the output/ folder after the run.

##  Contact
For any questions regarding the implementation, please contact the author at: xyc1477@163.com