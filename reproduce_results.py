import os
import sys

def main():
    print("=========================================================")
    print("     ISA-MA Reproducibility & Evaluation Guide           ")
    print("=========================================================")
    print("Welcome! This script outlines how to reproduce the tables, ")
    print("figures, and statistical tests presented in the manuscript.\n")
    
    print("⚠️  NOTE: Running full 30-run evaluations on the Mumford0 ")
    print("network may take several hours. For immediate verification, ")
    print("we have provided the raw run-level output data in the ")
    print("'/results' directory.\n")

    print("---------------------------------------------------------")
    print(" 1. Verify Statistical Significance & Confidence Intervals")
    print("---------------------------------------------------------")
    print("To instantly reproduce the exact p-values, effect sizes (r),")
    print("and 95% CIs reported in Table 6 using the provided raw data:")
    print(" -> Command: python src/Statistical_Analysis.py\n")

    print("---------------------------------------------------------")
    print(" 2. Re-run Algorithms (ISA-MA & MA-AOS Baseline)")
    print("---------------------------------------------------------")
    print("To run the optimization algorithms from scratch:")
    print(" -> Command: python src/ISA-MA.py")
    print(" -> Command: python src/MA-AOS.py\n")

    print("---------------------------------------------------------")
    print(" 3. Reproduce Figure 5 & Appendix E (Continuous Benchmarks)")
    print("---------------------------------------------------------")
    print("To generate the boxplots for Sphere, Rastrigin, Ackley, etc.:")
    print(" -> Command: python src/Benchmark_BoxPlots.py\n")

    print("---------------------------------------------------------")
    print(" 4. Reproduce Appendix D (Diversity Proxy Validation)")
    print("---------------------------------------------------------")
    print("To generate the Jaccard distance vs. Fitness CV plots (Fig D1):")
    print(" -> Command: python src/validate_proxy_mandl.py")
    print(" -> Command: python src/validate_proxy_mumford0.py\n")

    print("=========================================================")
    print("All generated plots and CSVs will be saved to the '/results' folder.")
    print("=========================================================")

if __name__ == "__main__":
    main()
