# ============================================================
# src/aml_pipeline/cycle_features.py
#
# Phase 3: Cycle-Aware Component A
# N-step Random Walk Cycle Return Probability Feature Extractor
#
# Mathematical formulation:
#   Given directed transition matrix P = D_out^-1 * A:
#   For edge (u -> v), cycle return probability after k steps is:
#     R_k(u, v) = [P^k]_{v, u}
#   where R_1 = 2-hop loop (u -> v -> u)
#         R_2 = 3-hop loop (u -> v -> w -> u)
#         R_3 = 4-hop loop (u -> v -> w1 -> w2 -> u)
#         R_4 = 5-hop loop (u -> v -> w1 -> w2 -> w3 -> u)
#
# Zero Data-Leakage Protocol:
#   Transition probabilities for training edges use ONLY training topology (Days 1-8).
#   Validation and test edges use historical topology up to their respective windows.
#
# Output:
#   outputs/aml/graph/aml_graph_cycle.pt
# ============================================================

import os
import sys
import time
import json
import torch
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import GRAPH_DIR, PREPROCESSED_DIR

CYCLE_FEAT_COLS = [
    "rw_cycle_prob_2hop",
    "rw_cycle_prob_3hop",
    "rw_cycle_prob_4hop",
    "rw_cycle_prob_5hop",
    "rw_cycle_discounted",
    "has_cycle_2hop",
    "has_cycle_3hop",
    "has_cycle_4hop",
]

def build_transition_matrix(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> sp.csr_matrix:
    """Constructs row-stochastic random walk transition matrix P = D_out^-1 * A."""
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float32)
    out_degree[out_degree == 0] = 1.0  # avoid division by zero
    weights = np.ones(len(src), dtype=np.float32) / out_degree[src]
    return sp.csr_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))

def extract_entries(matrix: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray, chunk_size: int = 500_000) -> np.ndarray:
    """Extracts matrix entries [rows[i], cols[i]] in memory-efficient chunks."""
    res = np.zeros(len(rows), dtype=np.float32)
    for i in range(0, len(rows), chunk_size):
        r_chunk = rows[i:i + chunk_size]
        c_chunk = cols[i:i + chunk_size]
        res[i:i + chunk_size] = np.array(matrix[r_chunk, c_chunk]).flatten()
    return res

def compute_cycle_features_for_edges(
    num_nodes: int,
    topo_src: np.ndarray,
    topo_dst: np.ndarray,
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
) -> np.ndarray:
    """
    Computes k-step cycle return probabilities for eval edges using topology graph.
    Returns array of shape [len(eval_src), 8].
    """
    # 1. Build transition matrix on topology
    P = build_transition_matrix(num_nodes, topo_src, topo_dst)
    
    # 2. Compute powers P^2, P^3, P^4
    P2 = P.dot(P)
    P3 = P2.dot(P)
    P4 = P3.dot(P)
    
    # 3. Extract return probabilities [P^k]_{dst, src} for eval edges
    r1 = extract_entries(P, eval_dst, eval_src)
    r2 = extract_entries(P2, eval_dst, eval_src)
    r3 = extract_entries(P3, eval_dst, eval_src)
    r4 = extract_entries(P4, eval_dst, eval_src)
    
    # 4. Compute discounted cumulative cycle probability
    # gamma = 0.8: R1 + 0.8*R2 + 0.64*R3 + 0.512*R4
    r_disc = r1 + 0.8 * r2 + 0.64 * r3 + 0.512 * r4
    
    # 5. Binary indicators
    has_2 = (r1 > 0).astype(np.float32)
    has_3 = (r2 > 0).astype(np.float32)
    has_4 = (r3 > 0).astype(np.float32)
    
    cycle_matrix = np.column_stack([
        r1, r2, r3, r4,
        r_disc,
        has_2, has_3, has_4
    ])
    return cycle_matrix

def extract_and_augment_graph():
    print("=" * 65)
    print("  PHASE 3, COMPONENT A: N-STEP RANDOM WALK CYCLE FEATURE EXTRACTION")
    print("=" * 65)
    
    base_graph_path = os.path.join(GRAPH_DIR, "aml_graph.pt")
    out_graph_path = os.path.join(GRAPH_DIR, "aml_graph_cycle.pt")
    
    if not os.path.exists(base_graph_path):
        print(f"[!] Error: {base_graph_path} not found.")
        return
        
    print(f"  Loading base graph: {base_graph_path}...")
    t0 = time.time()
    data = torch.load(base_graph_path, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Total Edges: {data.edge_index.size(1):,}")
    
    N = data.num_nodes
    src = data.edge_index[0].numpy().copy()
    dst = data.edge_index[1].numpy().copy()
    
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    print("\n[1/3] Computing Cycle Features for Training Split (Days 1-8)...")
    # Topology strictly Days 1-8
    train_src = src[train_mask]
    train_dst = dst[train_mask]
    cycle_feats_train = compute_cycle_features_for_edges(
        N, train_src, train_dst, train_src, train_dst
    )
    print(f"  ✅ Train cycle features computed: shape {cycle_feats_train.shape}")
    
    print("\n[2/3] Computing Cycle Features for Validation Split (Day 9)...")
    # Historical topology up to Day 9 (Train + Val)
    hist_val_mask = train_mask | val_mask
    val_src = src[val_mask]
    val_dst = dst[val_mask]
    cycle_feats_val = compute_cycle_features_for_edges(
        N, src[hist_val_mask], dst[hist_val_mask], val_src, val_dst
    )
    print(f"  ✅ Val cycle features computed: shape {cycle_feats_val.shape}")
    
    print("\n[3/3] Computing Cycle Features for Test Split (Days 10-18)...")
    # Full historical topology for test
    test_src = src[test_mask]
    test_dst = dst[test_mask]
    cycle_feats_test = compute_cycle_features_for_edges(
        N, src, dst, test_src, test_dst
    )
    print(f"  ✅ Test cycle features computed: shape {cycle_feats_test.shape}")
    
    # Assemble full cycle features in exact original edge order
    all_cycle_feats = np.zeros((len(src), 8), dtype=np.float32)
    all_cycle_feats[train_mask] = cycle_feats_train
    all_cycle_feats[val_mask] = cycle_feats_val
    all_cycle_feats[test_mask] = cycle_feats_test
    
    # Statistical Summary
    y = data.y.numpy()
    print("\n" + "-" * 65)
    print("  CYCLE FEATURE DISCRIMINATIVE POWER (FRAUD VS LEGIT)")
    print("-" * 65)
    for idx, col in enumerate(CYCLE_FEAT_COLS):
        legit_mean = all_cycle_feats[y == 0, idx].mean()
        fraud_mean = all_cycle_feats[y == 1, idx].mean()
        print(f"  {col:<26}: Legit = {legit_mean:8.5f} | Fraud = {fraud_mean:8.5f}")
    print("-" * 65)
    
    # Augment edge_attr
    orig_edge_attr = data.edge_attr
    cycle_tensor = torch.tensor(all_cycle_feats, dtype=torch.float)
    augmented_edge_attr = torch.cat([orig_edge_attr, cycle_tensor], dim=1)
    
    print(f"\n  Original Edge Feature Dim : {orig_edge_attr.size(1)}")
    print(f"  Cycle-Aware Feature Dim    : {cycle_tensor.size(1)}")
    print(f"  New Augmented Edge Dim     : {augmented_edge_attr.size(1)}")
    
    # Clone data object and update edge attributes
    cycle_data = data.clone()
    cycle_data.edge_attr = augmented_edge_attr
    
    # Update split edge attributes
    cycle_data.train_edge_attr = augmented_edge_attr[data.train_mask]
    cycle_data.val_edge_attr = augmented_edge_attr[data.val_mask]
    cycle_data.test_edge_attr = augmented_edge_attr[data.test_mask]
    
    # Save augmented graph
    print(f"\n  Saving cycle-augmented graph to: {out_graph_path}...")
    torch.save(cycle_data, out_graph_path)
    
    # Save metadata
    meta_path = os.path.join(GRAPH_DIR, "graph_metadata_cycle.json")
    metadata = {
        "graph_type": "homogeneous_cycle_augmented",
        "dataset": "IBM HI-Small AML",
        "num_nodes": N,
        "node_feat_dim": int(data.x.size(1)),
        "base_edge_feat_dim": int(orig_edge_attr.size(1)),
        "cycle_feat_dim": len(CYCLE_FEAT_COLS),
        "total_edge_feat_dim": int(augmented_edge_attr.size(1)),
        "cycle_feat_cols": CYCLE_FEAT_COLS,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  ✅ Saved metadata to {meta_path}")
    print(f"  Total time elapsed: {time.time() - t0:.2f}s")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    extract_and_augment_graph()
