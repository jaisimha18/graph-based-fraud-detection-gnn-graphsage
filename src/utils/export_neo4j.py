# ============================================================
# src/utils/export_neo4j.py
#
# Exports graph data as Neo4j-compatible CSV files
# These can be imported into Neo4j Desktop / Neo4j Aura
# and visualized interactively as a graph
#
# Output (in outputs/neo4j/):
#   nodes_users.csv        ← User nodes for Neo4j
#   nodes_merchants.csv    ← Merchant nodes for Neo4j
#   edges_transactions.csv ← Transaction edges for Neo4j
#   import_commands.txt    ← Exact Cypher commands to paste in Neo4j
#
# NOTE: Full 24M edges can't be visualized in Neo4j meaningfully.
#       We export a SAMPLE (configurable). Default = 50K edges
#       including ALL fraud edges so they are visible.
#
# Run: python -m src.utils.export_neo4j
# ============================================================

import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import OUTPUT_DIR

PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
NEO4J_DIR        = os.path.join(OUTPUT_DIR, "neo4j")
os.makedirs(NEO4J_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Export User Nodes
# ─────────────────────────────────────────────────────────────

def export_user_nodes() -> pd.DataFrame:
    print("\n[1/3] Exporting User Nodes...")

    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "user_features.csv"))

    neo4j_df = pd.DataFrame()
    neo4j_df["userId:ID"]          = df["user_id"].astype(int)
    neo4j_df["ficoScore:int"]      = df["FICO Score"].round(3)
    neo4j_df["yearlyIncome:float"] = df["Yearly Income - Person"].round(3)
    neo4j_df["totalDebt:float"]    = df["Total Debt"].round(3)
    neo4j_df["age:int"]            = df["Current Age"].round(3)
    neo4j_df["gender"]             = df["gender_enc"].map({1: "Male", 0: "Female"})
    neo4j_df["numCards:int"]       = df["Num Credit Cards"].round(3)
    neo4j_df["latitude:float"]     = df["Latitude"].round(4)
    neo4j_df["longitude:float"]    = df["Longitude"].round(4)
    neo4j_df[":LABEL"]             = "User"

    path = os.path.join(NEO4J_DIR, "nodes_users.csv")
    neo4j_df.to_csv(path, index=False)
    print(f"  ✅ {len(neo4j_df):,} user nodes → {path}")
    return neo4j_df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Export Merchant Nodes
# ─────────────────────────────────────────────────────────────

def export_merchant_nodes() -> pd.DataFrame:
    print("\n[2/3] Exporting Merchant Nodes...")

    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "merchant_features_raw.csv"))

    neo4j_df = pd.DataFrame()
    neo4j_df["merchantId:ID"]      = df["merchant_id"].astype(int)
    neo4j_df["mcc:int"]            = df["top_mcc"].astype(int)
    neo4j_df["txnCount:int"]       = df["txn_count"].astype(int)
    neo4j_df["fraudCount:int"]     = df["fraud_count"].astype(int)
    neo4j_df["fraudRate:float"]    = df["fraud_rate"].round(6)
    neo4j_df["avgAmount:float"]    = df["avg_amount"].round(2)
    neo4j_df["stdAmount:float"]    = df["std_amount"].round(2)
    neo4j_df["uniqueUsers:int"]    = df["unique_users"].astype(int)
    neo4j_df[":LABEL"]             = "Merchant"

    path = os.path.join(NEO4J_DIR, "nodes_merchants.csv")
    neo4j_df.to_csv(path, index=False)
    print(f"  ✅ {len(neo4j_df):,} merchant nodes → {path}")
    return neo4j_df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Export Transaction Edges (sampled)
# ─────────────────────────────────────────────────────────────

def export_edges(
    n_legit_sample: int = 50_000,
    n_fraud_sample: int = "all",   # 'all' = include every fraud edge
    split: str = "train",
):
    """
    Export transaction edges as Neo4j relationship CSV.

    Args:
        n_legit_sample : How many legit edges to sample (for manageable viz)
        n_fraud_sample : 'all' = export all fraud edges (important!)
        split          : 'train' | 'val' | 'test'
    """
    print(f"\n[3/3] Exporting Transaction Edges (split={split})...")

    edge_file = os.path.join(PREPROCESSED_DIR, f"edges_{split}.csv")
    print(f"  Loading {edge_file}...")

    # Read in chunks to handle large file
    fraud_chunks = []
    legit_chunks = []
    legit_collected = 0
    CHUNK = 500_000

    for chunk in pd.read_csv(edge_file, chunksize=CHUNK, low_memory=False):
        # All fraud rows
        fraud_chunk = chunk[chunk["fraud_label"] == 1]
        if not fraud_chunk.empty:
            fraud_chunks.append(fraud_chunk)

        # Sample legit rows
        if legit_collected < n_legit_sample:
            legit_chunk = chunk[chunk["fraud_label"] == 0]
            needed = n_legit_sample - legit_collected
            if len(legit_chunk) > needed:
                legit_chunk = legit_chunk.sample(n=needed, random_state=42)
            legit_chunks.append(legit_chunk)
            legit_collected += len(legit_chunk)

    fraud_df = pd.concat(fraud_chunks, ignore_index=True) if fraud_chunks else pd.DataFrame()
    legit_df = pd.concat(legit_chunks, ignore_index=True) if legit_chunks else pd.DataFrame()
    combined = pd.concat([fraud_df, legit_df], ignore_index=True)

    print(f"  Legit edges sampled : {len(legit_df):,}")
    print(f"  Fraud edges (all)   : {len(fraud_df):,}")
    print(f"  Total edges export  : {len(combined):,}")

    # ── Build Neo4j relationship format ───────────────────────
    neo4j_df = pd.DataFrame()
    neo4j_df[":START_ID"]          = combined["user_id"].astype(int)
    neo4j_df[":END_ID"]            = combined["merchant_id"].astype(int)
    neo4j_df[":TYPE"]              = combined["fraud_label"].map(
                                        {0: "TRANSACTED", 1: "FRAUD_TRANSACTED"}
                                    )
    neo4j_df["isFraud:boolean"]    = combined["fraud_label"].astype(bool)
    neo4j_df["amountLog:float"]    = combined["amount_log"].round(4)
    neo4j_df["isWeekend:boolean"]  = combined["is_weekend"].astype(bool)
    neo4j_df["isNight:boolean"]    = combined["is_night"].astype(bool)
    neo4j_df["chipUsed:int"]       = combined["chip_enc"].astype(int)
    neo4j_df["hasError:boolean"]   = combined["has_any_error"].astype(bool)
    neo4j_df["darkWeb:boolean"]    = combined["dark_web_enc"].astype(bool)

    # Filter to only nodes that exist (user_id != -1 = no synthetic)
    neo4j_df = neo4j_df[
        (neo4j_df[":START_ID"] >= 0) &
        (neo4j_df[":END_ID"] >= 0)
    ]

    path = os.path.join(NEO4J_DIR, "edges_transactions.csv")
    neo4j_df.to_csv(path, index=False)
    print(f"  ✅ {len(neo4j_df):,} edges → {path}")
    return neo4j_df


# ─────────────────────────────────────────────────────────────
# STEP 4 — Write Neo4j Import Instructions
# ─────────────────────────────────────────────────────────────

def write_import_instructions(n_users, n_merchants, n_edges):
    path = os.path.join(NEO4J_DIR, "import_commands.txt")

    content = f"""
=============================================================
  NEO4J IMPORT GUIDE — Graph-Based Fraud Detection
=============================================================

FILES TO IMPORT (copy all 3 to Neo4j import folder):
  nodes_users.csv        ({n_users:,} User nodes)
  nodes_merchants.csv    ({n_merchants:,} Merchant nodes)
  edges_transactions.csv ({n_edges:,} Transaction edges)

=============================================================
  OPTION 1: Neo4j Desktop (Recommended for Visualization)
=============================================================

Step 1: Install Neo4j Desktop from https://neo4j.com/download/
Step 2: Create a new project → Add Database → Create Local DB
Step 3: Copy the 3 CSV files to the DB's "import" folder
        (Click the 3 dots on DB → Open Folder → Import)
Step 4: Start the database
Step 5: Open Neo4j Browser and run these Cypher commands:

-- Load User Nodes --
LOAD CSV WITH HEADERS FROM 'file:///nodes_users.csv' AS row
CREATE (:User {{
  userId:       toInteger(row['userId:ID']),
  ficoScore:    toFloat(row['ficoScore:int']),
  yearlyIncome: toFloat(row['yearlyIncome:float']),
  totalDebt:    toFloat(row['totalDebt:float']),
  age:          toInteger(row['age:int']),
  gender:       row['gender'],
  numCards:     toInteger(row['numCards:int']),
  latitude:     toFloat(row['latitude:float']),
  longitude:    toFloat(row['longitude:float'])
}});

-- Load Merchant Nodes --
LOAD CSV WITH HEADERS FROM 'file:///nodes_merchants.csv' AS row
CREATE (:Merchant {{
  merchantId:  toInteger(row['merchantId:ID']),
  mcc:         toInteger(row['mcc:int']),
  txnCount:    toInteger(row['txnCount:int']),
  fraudCount:  toInteger(row['fraudCount:int']),
  fraudRate:   toFloat(row['fraudRate:float']),
  avgAmount:   toFloat(row['avgAmount:float']),
  uniqueUsers: toInteger(row['uniqueUsers:int'])
}});

-- Create Indexes (run before loading edges) --
CREATE INDEX user_id_idx FOR (u:User) ON (u.userId);
CREATE INDEX merchant_id_idx FOR (m:Merchant) ON (m.merchantId);

-- Load Edges --
LOAD CSV WITH HEADERS FROM 'file:///edges_transactions.csv' AS row
MATCH (u:User     {{userId:     toInteger(row[':START_ID'])}})
MATCH (m:Merchant {{merchantId: toInteger(row[':END_ID'])}})
CREATE (u)-[:TRANSACTED {{
  isFraud:   row['isFraud:boolean'] = 'True',
  amountLog: toFloat(row['amountLog:float']),
  isWeekend: row['isWeekend:boolean'] = 'True',
  isNight:   row['isNight:boolean'] = 'True',
  chipUsed:  toInteger(row['chipUsed:int']),
  hasError:  row['hasError:boolean'] = 'True',
  darkWeb:   row['darkWeb:boolean'] = 'True'
}}]->(m);

=============================================================
  VISUALIZATION QUERIES (run in Neo4j Browser)
=============================================================

-- 1. See the full subgraph (limit 100 for speed) --
MATCH (u:User)-[t:TRANSACTED]->(m:Merchant)
RETURN u, t, m LIMIT 100

-- 2. See ONLY fraud transactions --
MATCH (u:User)-[t:TRANSACTED]->(m:Merchant)
WHERE t.isFraud = true
RETURN u, t, m LIMIT 200

-- 3. Find users with most fraud --
MATCH (u:User)-[t:TRANSACTED]->(m:Merchant)
WHERE t.isFraud = true
RETURN u.userId, COUNT(t) AS fraud_count
ORDER BY fraud_count DESC LIMIT 20

-- 4. Find highest-risk merchants --
MATCH (u:User)-[t:TRANSACTED]->(m:Merchant)
WHERE t.isFraud = true
RETURN m.merchantId, m.fraudRate, COUNT(t) AS fraud_txns
ORDER BY fraud_txns DESC LIMIT 20

-- 5. Bipartite neighborhood of 1 user --
MATCH (u:User {{userId: 0}})-[t:TRANSACTED]->(m:Merchant)
RETURN u, t, m LIMIT 50

=============================================================
  OPTION 2: Neo4j Aura (Cloud, Free Tier)
=============================================================
1. Go to https://neo4j.com/cloud/platform/aura-graph-database/
2. Create free AuraDB instance
3. Use the same Cypher commands above in the online browser

=============================================================
"""

    with open(path, "w") as f:
        f.write(content)
    print(f"\n  ✅ Instructions saved → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all(n_legit=50_000, split="train"):
    print("\n" + "="*60)
    print("  NEO4J CSV EXPORT — Graph-Based Fraud Detection")
    print("="*60)

    user_df   = export_user_nodes()
    merch_df  = export_merchant_nodes()
    edge_df   = export_edges(n_legit_sample=n_legit, split=split)

    write_import_instructions(
        n_users=len(user_df),
        n_merchants=len(merch_df),
        n_edges=len(edge_df)
    )

    print("\n" + "="*60)
    print("  ✅ NEO4J EXPORT COMPLETE")
    print("="*60)
    print(f"\n  Files saved in: outputs/neo4j/")
    print(f"  ├── nodes_users.csv          ({len(user_df):,} rows)")
    print(f"  ├── nodes_merchants.csv       ({len(merch_df):,} rows)")
    print(f"  ├── edges_transactions.csv    ({len(edge_df):,} rows)")
    print(f"  └── import_commands.txt       ← Step-by-step guide")
    print(f"\n  Open import_commands.txt and follow the steps!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_legit", type=int, default=50_000,
                        help="Number of legit edges to sample (default: 50000)")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    args = parser.parse_args()
    run_all(n_legit=args.n_legit, split=args.split)
