#!/usr/bin/env python3
# ============================================================
# src/aml_pipeline/eda_general.py
#
# Exploratory Data Analysis (EDA) script for the AML dataset.
# Computes:
#   1. Class imbalance across all temporal splits
#   2. Degree distribution (in-degree, out-degree, total degree)
#      for the account nodes based on the graph structure.
#
# Run: python -m src.aml_pipeline.eda_general
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import PREPROCESSED_DIR, AML_OUTPUT_DIR

def run_general_eda():
    print("=" * 60)
    print("  GENERAL EDA: Class Imbalance & Degree Distribution")
    print("=" * 60)

    splits = ["train", "val", "test"]
    all_edges = []
    
    print("\n[1] Class Imbalance Analysis (Chronological Split)")
    print(f"{'Split':<10} | {'Total Edges':>12} | {'Fraud Edges':>12} | {'Fraud Rate (%)':>14}")
    print("-" * 55)
    
    for split in splits:
        path = os.path.join(PREPROCESSED_DIR, f"edges_{split}.csv")
        if not os.path.exists(path):
            print(f"  [!] Missing {path}")
            continue
            
        df = pd.read_csv(path)
        all_edges.append(df)
        
        total = len(df)
        fraud = df["laundering_label"].sum()
        rate = (fraud / total) * 100
        
        print(f"{split:<10} | {total:>12,} | {fraud:>12,} | {rate:>13.4f}%")

    if not all_edges:
        return

    # Combine all edges for degree distribution
    full_graph = pd.concat(all_edges, ignore_index=True)
    
    print("\n[2] Degree Distribution Analysis")
    
    out_degree = full_graph.groupby("from_account").size()
    in_degree = full_graph.groupby("to_account").size()
    
    # Fill missing accounts with 0
    all_accounts = set(out_degree.index).union(set(in_degree.index))
    out_degree = out_degree.reindex(list(all_accounts), fill_value=0)
    in_degree = in_degree.reindex(list(all_accounts), fill_value=0)
    
    total_degree = out_degree + in_degree
    
    print(f"Total Unique Active Accounts: {len(all_accounts):,}")
    
    def print_stats(name, s):
        print(f"\n  {name} Statistics:")
        print(f"    Min    : {s.min():,}")
        print(f"    Max    : {s.max():,}")
        print(f"    Mean   : {s.mean():.2f}")
        print(f"    Median : {s.median():.2f}")
        print(f"    99th % : {np.percentile(s, 99):.2f}")

    print_stats("Out-Degree (Transactions Sent)", out_degree)
    print_stats("In-Degree (Transactions Received)", in_degree)
    print_stats("Total Degree (All Transactions)", total_degree)
    
    # Save a degree distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(total_degree[total_degree <= 100], bins=50, alpha=0.7, color='blue')
    plt.title("Degree Distribution (Truncated at 100)")
    plt.xlabel("Total Degree")
    plt.ylabel("Frequency (Number of Accounts)")
    plt.grid(axis='y', alpha=0.75)
    
    plot_dir = os.path.join(AML_OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "degree_distribution.png")
    plt.savefig(plot_path)
    print(f"\n  [+] Saved degree distribution plot to: {plot_path}")

if __name__ == "__main__":
    run_general_eda()
