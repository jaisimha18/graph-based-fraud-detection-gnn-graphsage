# ============================================================
# src/aml_pipeline/build_graph.py
#
# Homogeneous Account–Account Graph Builder (AML)
#
# Reads preprocessed CSVs → builds PyG Data object
#
# Graph structure (HOMOGENEOUS):
#   Nodes : accounts  [~518K × 8 features]
#   Edges : (account → account) directed money transfers
#           train → edges_train_smote.csv
#           val   → edges_val.csv
#           test  → edges_test.csv
#
# Output:
#   outputs/aml/graph/aml_graph.pt          ← PyG Data
#   outputs/aml/graph/graph_metadata.json   ← stats & dims
#   outputs/aml/graph/node_mappings.json    ← account → index maps
#
# Run: python -m src.aml_pipeline.build_graph
# ============================================================

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aml_pipeline.aml_config import AML_OUTPUT_DIR, PREPROCESSED_DIR, GRAPH_DIR

os.makedirs(GRAPH_DIR, exist_ok=True)

# ── Edge feature columns (14 features) ───────────────────────
EDGE_FEAT_COLS = [
    "amount_received_log", "amount_paid_log", "amount_diff_log",
    "is_cross_bank", "is_self_transfer", "is_cross_currency",
    "receiving_currency_enc", "payment_format_enc",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_night", "is_weekend",
]

# ── Account feature columns (8 features) ─────────────────────
ACCOUNT_FEAT_COLS = [
    "bank_id", "entity_type_enc", "is_crypto_bank",
    "out_txn_count", "in_txn_count",
    "out_avg_amount", "in_avg_amount",
    "unique_counterparties",
]


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load account node features
# ─────────────────────────────────────────────────────────────

def load_account_nodes() -> tuple:
    """
    Load account_features.csv → account node feature matrix.

    Returns:
        account_feat_tensor : torch.FloatTensor [N_accounts × 8]
        account_id_to_idx   : dict {account_hex_id → node_index}
    """
    print("\n[1/3] Loading Account Nodes...")
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "account_features.csv"))

    # Build account_id → node_index mapping
    account_id_to_idx = {
        str(aid): idx for idx, aid in enumerate(df["account_id"].values)
    }

    # Select feature columns
    feat_cols = [c for c in ACCOUNT_FEAT_COLS if c in df.columns]
    feat_matrix = df[feat_cols].fillna(0).values.astype(np.float32)

    print(f"  ✅ Account nodes : {feat_matrix.shape[0]:,} nodes  |  {feat_matrix.shape[1]} features")
    return torch.tensor(feat_matrix, dtype=torch.float), account_id_to_idx


# ─────────────────────────────────────────────────────────────
# STEP 2 — Load edges for a split (vectorized)
# ─────────────────────────────────────────────────────────────

def load_edges_fast(
    csv_path: str,
    account_id_to_idx: dict,
    split_name: str,
    chunk_size: int = 1_000_000,
) -> tuple:
    """
    Vectorized edge loader for AML homogeneous graph.

    Returns:
        edge_index : LongTensor  [2 × E]  (from_idx, to_idx)
        edge_attr  : FloatTensor [E × 14]
        edge_label : LongTensor  [E]      (0=legit, 1=laundering)
    """
    print(f"\n[Loading {split_name} edges] {csv_path}")

    all_chunks = []

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False),
        desc=f"  {split_name}"
    ):
        df = chunk.copy()

        # Drop synthetic SMOTE edges (no real account mapping)
        if "is_synthetic" in df.columns:
            real_mask = df["is_synthetic"] != 1
            df = df[real_mask]

        if df.empty:
            continue

        # Map account IDs → node indices
        df["from_idx"] = df["from_account"].astype(str).map(account_id_to_idx)
        df["to_idx"]   = df["to_account"].astype(str).map(account_id_to_idx)

        # Drop rows where mapping failed (unknown account)
        df = df.dropna(subset=["from_idx", "to_idx"])
        df["from_idx"] = df["from_idx"].astype(int)
        df["to_idx"]   = df["to_idx"].astype(int)

        # Fill missing edge features with 0
        for col in EDGE_FEAT_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[EDGE_FEAT_COLS] = df[EDGE_FEAT_COLS].fillna(0.0)

        all_chunks.append(df)

    if not all_chunks:
        print(f"  ⚠ No edges found for {split_name}")
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, len(EDGE_FEAT_COLS)), dtype=torch.float),
            torch.zeros(0, dtype=torch.long),
        )

    combined = pd.concat(all_chunks, ignore_index=True)

    edge_index = torch.tensor(
        [combined["from_idx"].values, combined["to_idx"].values],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        combined[EDGE_FEAT_COLS].values,
        dtype=torch.float,
    )
    edge_label = torch.tensor(
        combined["laundering_label"].values,
        dtype=torch.long,
    )

    n_launder = edge_label.sum().item()
    n_total = edge_index.shape[1]
    print(f"  ✅ Edges loaded    : {n_total:,}")
    print(f"     Laundering      : {n_launder:,}  ({n_launder/max(n_total,1)*100:.4f}%)")
    print(f"     edge_index      : {list(edge_index.shape)}")
    print(f"     edge_attr       : {list(edge_attr.shape)}")

    return edge_index, edge_attr, edge_label


# ─────────────────────────────────────────────────────────────
# STEP 3 — Assemble PyG Data & Save
# ─────────────────────────────────────────────────────────────

def build_graph() -> Data:
    print("\n" + "=" * 60)
    print("  HOMOGENEOUS GRAPH CONSTRUCTION")
    print("  Account–Account AML Detection")
    print("=" * 60)

    # ── Node features ─────────────────────────────────────────
    account_x, account_id_to_idx = load_account_nodes()

    # ── Edge data per split ───────────────────────────────────
    splits = {
        "train": os.path.join(PREPROCESSED_DIR, "edges_train_smote.csv"),
        "val":   os.path.join(PREPROCESSED_DIR, "edges_val.csv"),
        "test":  os.path.join(PREPROCESSED_DIR, "edges_test.csv"),
    }

    # Fallback: if SMOTE file doesn't exist, use raw train
    if not os.path.exists(splits["train"]):
        print("  ⚠ edges_train_smote.csv not found, using edges_train.csv")
        splits["train"] = os.path.join(PREPROCESSED_DIR, "edges_train.csv")

    print("\n[2/3] Loading Edges (all splits)...")
    edge_data = {}
    for split_name, csv_path in splits.items():
        ei, ea, el = load_edges_fast(csv_path, account_id_to_idx, split_name)
        edge_data[split_name] = (ei, ea, el)

    # ── Combine all edges into one graph ──────────────────────
    print("\n[3/3] Assembling PyG Data graph...")

    # Concatenate all splits for the unified graph
    all_edge_index = torch.cat(
        [edge_data[s][0] for s in ["train", "val", "test"]], dim=1
    )
    all_edge_attr = torch.cat(
        [edge_data[s][1] for s in ["train", "val", "test"]], dim=0
    )
    all_edge_label = torch.cat(
        [edge_data[s][2] for s in ["train", "val", "test"]], dim=0
    )

    # Build split masks
    n_train = edge_data["train"][0].shape[1]
    n_val   = edge_data["val"][0].shape[1]
    n_test  = edge_data["test"][0].shape[1]
    n_total = n_train + n_val + n_test

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask   = torch.zeros(n_total, dtype=torch.bool)
    test_mask  = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    # ── Build PyG Data object ─────────────────────────────────
    data = Data(
        x=account_x,
        edge_index=all_edge_index,
        edge_attr=all_edge_attr,
        y=all_edge_label,
    )
    data.num_nodes = account_x.shape[0]
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # Also store per-split edge data for convenience
    data.train_edge_index = edge_data["train"][0]
    data.train_edge_attr  = edge_data["train"][1]
    data.train_edge_label = edge_data["train"][2]
    data.val_edge_index   = edge_data["val"][0]
    data.val_edge_attr    = edge_data["val"][1]
    data.val_edge_label   = edge_data["val"][2]
    data.test_edge_index  = edge_data["test"][0]
    data.test_edge_attr   = edge_data["test"][1]
    data.test_edge_label  = edge_data["test"][2]

    # ── Graph metadata ────────────────────────────────────────
    metadata = {
        "graph_type": "homogeneous",
        "dataset": "IBM HI-Small AML",
        "node_type": "account",
        "num_nodes": int(account_x.shape[0]),
        "node_feat_dim": int(account_x.shape[1]),
        "node_feat_cols": ACCOUNT_FEAT_COLS,
        "edge_feat_dim": int(all_edge_attr.shape[1]),
        "edge_feat_cols": EDGE_FEAT_COLS,
        "splits": {},
    }
    for split_name, (ei, ea, el) in edge_data.items():
        n_launder = int(el.sum().item())
        n_edges = int(ei.shape[1])
        metadata["splits"][split_name] = {
            "num_edges": n_edges,
            "num_laundering": n_launder,
            "laundering_rate": round(n_launder / max(n_edges, 1) * 100, 4),
        }
    metadata["total_edges"] = n_total
    metadata["total_laundering"] = int(all_edge_label.sum().item())

    # ── Save ──────────────────────────────────────────────────
    graph_path    = os.path.join(GRAPH_DIR, "aml_graph.pt")
    metadata_path = os.path.join(GRAPH_DIR, "graph_metadata.json")
    mapping_path  = os.path.join(GRAPH_DIR, "node_mappings.json")

    torch.save(data, graph_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save node mappings (account_hex_id → node_index)
    with open(mapping_path, "w") as f:
        json.dump(account_id_to_idx, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ AML GRAPH BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Graph structure (homogeneous):")
    print(f"    Account nodes  : {account_x.shape[0]:,}  (feat_dim={account_x.shape[1]})")
    print(f"    Total edges    : {n_total:,}  (feat_dim={all_edge_attr.shape[1]})")
    for split_name, (ei, ea, el) in edge_data.items():
        n_l = int(el.sum().item())
        n_e = int(ei.shape[1])
        print(f"    {split_name:<6} edges   : {n_e:>12,}  laundering={n_l:,} ({n_l/max(n_e,1)*100:.4f}%)")

    print(f"\n  Saved files:")
    print(f"    {graph_path}")
    print(f"    {metadata_path}")
    print(f"    {mapping_path}")
    print(f"\n  Load later with:")
    print(f"    import torch")
    print(f"    data = torch.load('outputs/aml/graph/aml_graph.pt')")
    print("=" * 60)

    return data


if __name__ == "__main__":
    data = build_graph()
    print("\n  Graph object preview:")
    print(data)
