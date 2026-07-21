# ============================================================
# src/graph_construction/build_graph.py
#
# Bipartite User–Merchant Graph Builder
#
# Reads preprocessed CSVs → builds PyG HeteroData object
# Preserves full semantic meaning of transactions
#
# Graph structure:
#   Nodes : user     [2000   × 10 features]
#           merchant [90133  ×  6 features]
#   Edges : (user, transacts, merchant)
#           train → edges_train_smote.csv  [~18.9M edges]
#           val   → edges_val.csv          [~1.7M  edges]
#           test  → edges_test.csv         [~3.8M  edges]
#
# Output:
#   outputs/graph/hetero_graph.pt       ← PyG HeteroData
#   outputs/graph/graph_metadata.json   ← stats & dims
#   outputs/graph/node_mappings.json    ← id → index maps
#
# Run: python -m src.graph_construction.build_graph
# ============================================================

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import OUTPUT_DIR

PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
GRAPH_DIR        = os.path.join(OUTPUT_DIR, "graph")
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── Edge feature columns (21 features, same order as preprocessing) ──
EDGE_FEAT_COLS = [
    "amount_log",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_night",
    "chip_enc",
    "has_any_error",
    "err_insufficient_balance", "err_bad_pin", "err_technical_glitch",
    "err_bad_card_number", "err_bad_expiration", "err_bad_cvv", "err_bad_zipcode",
    "dark_web_enc", "credit_limit_f", "has_chip_enc",
    "card_type_enc", "card_brand_enc",
]

# ── Merchant feature columns ──────────────────────────────────
MERCHANT_FEAT_COLS = [
    "txn_count", "fraud_rate", "avg_amount",
    "std_amount", "unique_users", "mcc_enc",
]

# ── User feature columns ──────────────────────────────────────
USER_FEAT_COLS = [
    "Current Age", "years_to_retirement", "gender_enc",
    "FICO Score", "Yearly Income - Person", "Total Debt",
    "debt_to_income_ratio", "Num Credit Cards", "Latitude", "Longitude",
]


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load node feature matrices
# ─────────────────────────────────────────────────────────────

def load_user_nodes() -> tuple:
    """
    Load user_features.csv → user node feature matrix.

    Returns:
        user_feat_tensor : torch.FloatTensor [2000 × 10]
        user_id_to_idx   : dict {user_int_id → node_index}
    """
    print("\n[1/4] Loading User Nodes...")
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "user_features.csv"))

    # user_id = row index (0 … 1999), matches transactions['User']
    user_id_to_idx = {int(uid): int(uid) for uid in df["user_id"].values}

    # Select feature columns (normalized)
    feat_cols = [c for c in USER_FEAT_COLS if c in df.columns]
    feat_matrix = df[feat_cols].fillna(0).values.astype(np.float32)

    print(f"  ✅ User nodes   : {feat_matrix.shape[0]:,} nodes  |  {feat_matrix.shape[1]} features")
    return torch.tensor(feat_matrix, dtype=torch.float), user_id_to_idx


def load_merchant_nodes() -> tuple:
    """
    Load merchant_features.csv → merchant node feature matrix.

    Returns:
        merch_feat_tensor : torch.FloatTensor [N_merchants × 6]
        merch_id_to_idx   : dict {merchant_int_id → node_index}
    """
    print("\n[2/4] Loading Merchant Nodes...")
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "merchant_features.csv"))

    # merchant_id is the integer Merchant Name from transactions.csv
    merch_id_to_idx = {
        int(mid): idx for idx, mid in enumerate(df["merchant_id"].values)
    }

    feat_cols = [c for c in MERCHANT_FEAT_COLS if c in df.columns]
    feat_matrix = df[feat_cols].fillna(0).values.astype(np.float32)

    print(f"  ✅ Merchant nodes: {feat_matrix.shape[0]:,} nodes  |  {feat_matrix.shape[1]} features")
    return torch.tensor(feat_matrix, dtype=torch.float), merch_id_to_idx


# ─────────────────────────────────────────────────────────────
# STEP 2 — Load edges for a split
# ─────────────────────────────────────────────────────────────

def load_edges(
    csv_path: str,
    user_id_to_idx: dict,
    merch_id_to_idx: dict,
    split_name: str,
    chunk_size: int = 1_000_000,
) -> tuple:
    """
    Load preprocessed edge CSV → (edge_index, edge_attr, edge_label).

    Handles 18M+ rows via chunked reading.
    Synthetic edges (user_id == -1) are kept for training but
    skipped from edge_index (they have no real node mapping).

    Returns:
        edge_index : LongTensor  [2 × E]  (user_node_idx, merchant_node_idx)
        edge_attr  : FloatTensor [E × 21]
        edge_label : LongTensor  [E]      (0=legit, 1=fraud)
    """
    print(f"\n[Loading {split_name} edges] {csv_path}")

    src_list   = []
    dst_list   = []
    attr_list  = []
    label_list = []

    skipped_user = 0
    skipped_merch = 0
    skipped_synthetic = 0
    total = 0

    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
    for chunk in tqdm(reader, desc=f"  {split_name}"):
        total += len(chunk)

        for _, row in chunk.iterrows():
            uid  = int(row["user_id"])
            mid  = int(row["merchant_id"])

            # Skip synthetic SMOTE edges (no real node mapping)
            if uid == -1 or mid == -1:
                skipped_synthetic += 1
                # Still include their features for label counting
                # but they cannot be in edge_index (no real node)
                continue

            u_idx = user_id_to_idx.get(uid)
            m_idx = merch_id_to_idx.get(mid)

            if u_idx is None:
                skipped_user += 1
                continue
            if m_idx is None:
                skipped_merch += 1
                continue

            src_list.append(u_idx)
            dst_list.append(m_idx)
            label_list.append(int(row["fraud_label"]))
            attr_list.append(
                [float(row.get(c, 0.0) or 0.0) for c in EDGE_FEAT_COLS]
            )

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list,            dtype=torch.float)
    edge_label = torch.tensor(label_list,           dtype=torch.long)

    n_fraud = edge_label.sum().item()
    print(f"  ✅ Edges         : {edge_index.shape[1]:,}")
    print(f"     Fraud         : {n_fraud:,}  ({n_fraud/edge_index.shape[1]*100:.3f}%)")
    print(f"     Skipped synthetic: {skipped_synthetic:,}")
    print(f"     Skipped unknown user/merchant: {skipped_user+skipped_merch:,}")
    print(f"     Edge feature dim : {edge_attr.shape[1]}")

    return edge_index, edge_attr, edge_label


# ─────────────────────────────────────────────────────────────
# STEP 3 — Vectorized fast loader (replaces row-by-row above)
# ─────────────────────────────────────────────────────────────

def load_edges_fast(
    csv_path: str,
    user_id_to_idx: dict,
    merch_id_to_idx: dict,
    split_name: str,
    chunk_size: int = 1_000_000,
) -> tuple:
    """
    Vectorized edge loader — much faster than row-by-row.
    Uses pandas map() instead of Python loops.
    """
    print(f"\n[Loading {split_name} edges] {csv_path}")

    all_chunks = []

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False),
        desc=f"  {split_name}"
    ):
        df = chunk.copy()

        # Drop synthetic SMOTE edges (no real node)
        real_mask = (df["user_id"] != -1) & (df["merchant_id"] != -1)
        df = df[real_mask]

        if df.empty:
            continue

        # Map IDs → node indices
        df["u_idx"] = df["user_id"].map(user_id_to_idx)
        df["m_idx"] = df["merchant_id"].map(merch_id_to_idx)

        # Drop rows where mapping failed (unknown node)
        df = df.dropna(subset=["u_idx", "m_idx"])
        df["u_idx"] = df["u_idx"].astype(int)
        df["m_idx"] = df["m_idx"].astype(int)

        # Fill missing edge features with 0
        for col in EDGE_FEAT_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[EDGE_FEAT_COLS] = df[EDGE_FEAT_COLS].fillna(0.0)

        all_chunks.append(df)

    combined = pd.concat(all_chunks, ignore_index=True)

    edge_index = torch.tensor(
        [combined["u_idx"].values, combined["m_idx"].values],
        dtype=torch.long
    )
    edge_attr  = torch.tensor(
        combined[EDGE_FEAT_COLS].values,
        dtype=torch.float
    )
    edge_label = torch.tensor(
        combined["fraud_label"].values,
        dtype=torch.long
    )

    n_fraud = edge_label.sum().item()
    n_total = edge_index.shape[1]
    print(f"  ✅ Edges loaded  : {n_total:,}")
    print(f"     Fraud         : {n_fraud:,}  ({n_fraud/n_total*100:.4f}%)")
    print(f"     edge_index    : {list(edge_index.shape)}")
    print(f"     edge_attr     : {list(edge_attr.shape)}")
    print(f"     edge_label    : {list(edge_label.shape)}")

    return edge_index, edge_attr, edge_label


# ─────────────────────────────────────────────────────────────
# STEP 4 — Assemble HeteroData & Save
# ─────────────────────────────────────────────────────────────

def build_graph() -> HeteroData:
    print("\n" + "="*60)
    print("  BIPARTITE GRAPH CONSTRUCTION")
    print("  User–Merchant Fraud Detection")
    print("="*60)

    # ── Node features ─────────────────────────────────────────
    user_x,    user_id_to_idx  = load_user_nodes()
    merch_x,   merch_id_to_idx = load_merchant_nodes()

    # ── Edge data per split ───────────────────────────────────
    splits = {
        "train": os.path.join(PREPROCESSED_DIR, "edges_train_smote.csv"),
        "val"  : os.path.join(PREPROCESSED_DIR, "edges_val.csv"),
        "test" : os.path.join(PREPROCESSED_DIR, "edges_test.csv"),
    }

    print("\n[3/4] Loading Edges (all splits)...")
    edge_data = {}
    for split_name, csv_path in splits.items():
        ei, ea, el = load_edges_fast(
            csv_path, user_id_to_idx, merch_id_to_idx, split_name
        )
        edge_data[split_name] = (ei, ea, el)

    # ── Build HeteroData ──────────────────────────────────────
    print("\n[4/4] Assembling HeteroData graph...")
    data = HeteroData()

    # Node feature matrices
    data["user"].x     = user_x      # [2000, 10]
    data["merchant"].x = merch_x     # [N_merchants, 6]

    # Store node counts explicitly (useful for model init)
    data["user"].num_nodes     = user_x.shape[0]
    data["merchant"].num_nodes = merch_x.shape[0]

    # Edges per split — stored as separate edge types
    # Convention: (src_type, relation, dst_type)
    rel_map = {
        "train": ("user", "train_txn", "merchant"),
        "val"  : ("user", "val_txn",   "merchant"),
        "test" : ("user", "test_txn",  "merchant"),
    }
    for split_name, (ei, ea, el) in edge_data.items():
        rel = rel_map[split_name]
        data[rel].edge_index = ei
        data[rel].edge_attr  = ea
        data[rel].edge_label = el
        data[rel].num_edges  = ei.shape[1]

    # ── Graph metadata ────────────────────────────────────────
    metadata = {
        "node_types": {
            "user": {
                "num_nodes": int(user_x.shape[0]),
                "feat_dim" : int(user_x.shape[1]),
                "feat_cols": USER_FEAT_COLS,
            },
            "merchant": {
                "num_nodes": int(merch_x.shape[0]),
                "feat_dim" : int(merch_x.shape[1]),
                "feat_cols": MERCHANT_FEAT_COLS,
            },
        },
        "edge_types": {},
    }
    for split_name, (ei, ea, el) in edge_data.items():
        n_fraud = int(el.sum().item())
        n_total = int(ei.shape[1])
        metadata["edge_types"][split_name] = {
            "relation"   : list(rel_map[split_name]),
            "num_edges"  : n_total,
            "num_fraud"  : n_fraud,
            "fraud_rate" : round(n_fraud / n_total * 100, 4),
            "feat_dim"   : int(ea.shape[1]),
            "feat_cols"  : EDGE_FEAT_COLS,
        }

    # ── Save graph ────────────────────────────────────────────
    graph_path    = os.path.join(GRAPH_DIR, "hetero_graph.pt")
    metadata_path = os.path.join(GRAPH_DIR, "graph_metadata.json")
    mapping_path  = os.path.join(GRAPH_DIR, "node_mappings.json")

    torch.save(data, graph_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save node mappings (for inference: map new user/merchant to node index)
    mappings = {
        "user_id_to_node_idx"    : {str(k): v for k, v in user_id_to_idx.items()},
        "merchant_id_to_node_idx": {str(k): v for k, v in merch_id_to_idx.items()},
    }
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅ GRAPH BUILT SUCCESSFULLY")
    print("="*60)
    print(f"\n  Graph structure:")
    print(f"    User nodes     : {user_x.shape[0]:,}  (feat_dim={user_x.shape[1]})")
    print(f"    Merchant nodes : {merch_x.shape[0]:,}  (feat_dim={merch_x.shape[1]})")
    for split_name, (ei, ea, el) in edge_data.items():
        n_f = int(el.sum().item())
        n_t = int(ei.shape[1])
        print(f"    {split_name:<6} edges   : {n_t:>12,}  fraud={n_f:,} ({n_f/n_t*100:.3f}%)")

    print(f"\n  Saved files:")
    print(f"    {graph_path}")
    print(f"    {metadata_path}")
    print(f"    {mapping_path}")
    print("\n  Load later with:")
    print("    import torch")
    print("    data = torch.load('outputs/graph/hetero_graph.pt')")
    print("="*60)

    return data


if __name__ == "__main__":
    data = build_graph()
    print("\n  Graph object preview:")
    print(data)
