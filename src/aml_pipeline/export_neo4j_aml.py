# ============================================================
# src/aml_pipeline/export_neo4j_aml.py
#
# Exports AML graph data as Neo4j-compatible CSV files
# These can be imported into Neo4j Desktop / Neo4j Aura
# and visualized interactively as a graph
#
# Output (in outputs/aml/neo4j/):
#   nodes_accounts.csv         ← Account nodes for Neo4j
#   edges_transactions_aml.csv ← Transaction edges for Neo4j
#   import_commands_aml.txt    ← Exact Cypher commands to paste in Neo4j
#
# Run: python -m src.aml_pipeline.export_neo4j_aml
# ============================================================

import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aml_pipeline.aml_config import AML_OUTPUT_DIR

PREPROCESSED_DIR = os.path.join(AML_OUTPUT_DIR, "preprocessed")
NEO4J_DIR        = os.path.join(AML_OUTPUT_DIR, "neo4j")
os.makedirs(NEO4J_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Export Account Nodes
# ─────────────────────────────────────────────────────────────

def export_account_nodes() -> pd.DataFrame:
    print("\n[1/2] Exporting Account Nodes...")

    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "account_features.csv"))

    neo4j_df = pd.DataFrame()
    neo4j_df["accountId:ID"]               = df["account_id"].astype(str)
    neo4j_df["bankId:int"]                 = df["bank_id"].astype(int)
    neo4j_df["entityType:int"]             = df["entity_type_enc"].astype(int)
    neo4j_df["isCrypto:boolean"]           = df["is_crypto_bank"].astype(bool)
    
    # These features are normalized (StandardScaler) in account_features.csv
    neo4j_df["outTxnCount:float"]          = df["out_txn_count"].round(4)
    neo4j_df["inTxnCount:float"]           = df["in_txn_count"].round(4)
    neo4j_df["outAvgAmount:float"]         = df["out_avg_amount"].round(4)
    neo4j_df["inAvgAmount:float"]          = df["in_avg_amount"].round(4)
    neo4j_df["uniqueCounterparties:float"] = df["unique_counterparties"].round(4)
    neo4j_df[":LABEL"]                     = "Account"

    path = os.path.join(NEO4J_DIR, "nodes_accounts.csv")
    neo4j_df.to_csv(path, index=False)
    print(f"  ✅ {len(neo4j_df):,} account nodes → {path}")
    return neo4j_df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Export Transaction Edges (sampled)
# ─────────────────────────────────────────────────────────────

def export_edges(
    n_legit_sample: int = 50_000,
    split: str = "train",
):
    print(f"\n[2/2] Exporting Transaction Edges (split={split})...")

    edge_file = os.path.join(PREPROCESSED_DIR, f"edges_{split}.csv")
    print(f"  Loading {edge_file}...")

    # Read the full edge file
    df = pd.read_csv(edge_file)
    
    # All laundering edges
    fraud_df = df[df["laundering_label"] == 1]
    
    # Sample legit edges
    legit_df = df[df["laundering_label"] == 0]
    if len(legit_df) > n_legit_sample:
        legit_df = legit_df.sample(n=n_legit_sample, random_state=42)
        
    combined = pd.concat([fraud_df, legit_df], ignore_index=True)

    print(f"  Legit edges sampled : {len(legit_df):,}")
    print(f"  Laundering edges    : {len(fraud_df):,}")
    print(f"  Total edges export  : {len(combined):,}")

    # Build Neo4j relationship format
    neo4j_df = pd.DataFrame()
    neo4j_df[":START_ID"]               = combined["from_account"].astype(str)
    neo4j_df[":END_ID"]                 = combined["to_account"].astype(str)
    neo4j_df[":TYPE"]                   = "TRANSACTED"
    
    neo4j_df["isLaundering:boolean"]    = combined["laundering_label"].astype(bool)
    neo4j_df["amountLog:float"]         = combined["amount_paid_log"].round(4)
    neo4j_df["isCrossBank:boolean"]     = combined["is_cross_bank"].astype(bool)
    neo4j_df["isCrossCurrency:boolean"] = combined["is_cross_currency"].astype(bool)
    neo4j_df["isNight:boolean"]         = combined["is_night"].astype(bool)
    neo4j_df["isWeekend:boolean"]       = combined["is_weekend"].astype(bool)

    path = os.path.join(NEO4J_DIR, "edges_transactions_aml.csv")
    neo4j_df.to_csv(path, index=False)
    print(f"  ✅ {len(neo4j_df):,} edges → {path}")
    return neo4j_df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Write Neo4j Import Instructions
# ─────────────────────────────────────────────────────────────

def write_import_instructions(n_accounts, n_edges):
    path = os.path.join(NEO4J_DIR, "import_commands_aml.txt")

    content = f"""
=============================================================
  NEO4J IMPORT GUIDE — AML Fraud Detection
=============================================================

FILES TO IMPORT (copy both to Neo4j import folder):
  nodes_accounts.csv         ({n_accounts:,} Account nodes)
  edges_transactions_aml.csv ({n_edges:,} Transaction edges)

=============================================================
  HOW TO IMPORT (Neo4j Desktop / Neo4j Aura)
=============================================================

1. Copy the CSV files to the DB's "import" folder.
2. Start the database.
3. Open Neo4j Browser and run these Cypher commands sequentially:

-- Load Account Nodes --
LOAD CSV WITH HEADERS FROM 'file:///nodes_accounts.csv' AS row
CREATE (:Account {{
  accountId:            row['accountId:ID'],
  bankId:               toInteger(row['bankId:int']),
  entityType:           toInteger(row['entityType:int']),
  isCrypto:             row['isCrypto:boolean'] = 'True',
  outTxnCount:          toFloat(row['outTxnCount:float']),
  inTxnCount:           toFloat(row['inTxnCount:float']),
  outAvgAmount:         toFloat(row['outAvgAmount:float']),
  inAvgAmount:          toFloat(row['inAvgAmount:float']),
  uniqueCounterparties: toFloat(row['uniqueCounterparties:float'])
}});

-- Create Indexes (Crucial for fast edge loading) --
CREATE INDEX account_id_idx FOR (a:Account) ON (a.accountId);

-- Load Transaction Edges --
LOAD CSV WITH HEADERS FROM 'file:///edges_transactions_aml.csv' AS row
MATCH (from:Account {{accountId: row[':START_ID']}})
MATCH (to:Account {{accountId: row[':END_ID']}})
CREATE (from)-[:TRANSACTED {{
  isLaundering:    row['isLaundering:boolean'] = 'True',
  amountLog:       toFloat(row['amountLog:float']),
  isCrossBank:     row['isCrossBank:boolean'] = 'True',
  isCrossCurrency: row['isCrossCurrency:boolean'] = 'True',
  isNight:         row['isNight:boolean'] = 'True',
  isWeekend:       row['isWeekend:boolean'] = 'True'
}}]->(to);

=============================================================
  VISUALIZATION QUERIES
=============================================================

-- 1. See a general subgraph (limit 100 for speed) --
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
RETURN a, t, b LIMIT 100

-- 2. See ONLY money laundering transactions --
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WHERE t.isLaundering = true
RETURN a, t, b LIMIT 200

-- 3. Find accounts involved in the most laundering transactions (outgoing) --
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WHERE t.isLaundering = true
RETURN a.accountId, COUNT(t) AS laundering_count
ORDER BY laundering_count DESC LIMIT 20

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
    print("  NEO4J CSV EXPORT — AML Graph Dataset")
    print("="*60)

    acc_df  = export_account_nodes()
    edge_df = export_edges(n_legit_sample=n_legit, split=split)

    write_import_instructions(
        n_accounts=len(acc_df),
        n_edges=len(edge_df)
    )

    print("\n" + "="*60)
    print("  ✅ NEO4J EXPORT COMPLETE")
    print("="*60)
    print(f"\n  Files saved in: {NEO4J_DIR}/")
    print(f"  ├── nodes_accounts.csv         ({len(acc_df):,} rows)")
    print(f"  ├── edges_transactions_aml.csv ({len(edge_df):,} rows)")
    print(f"  └── import_commands_aml.txt    ← Cypher import commands")
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
