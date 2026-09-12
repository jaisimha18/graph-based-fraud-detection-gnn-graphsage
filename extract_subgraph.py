import pandas as pd
import os

# Load the full dataset
data_path = "AML_DATASET/HI-Small_Trans.csv"
print("Loading dataset...")
df = pd.read_csv(data_path)

# Find a fraudulent transaction
fraud_txns = df[df['Is Laundering'] == 1]
if fraud_txns.empty:
    print("No fraud found!")
    exit()

# Pick a specific target account involved in fraud
target_account = fraud_txns.iloc[0]['Account']
print(f"Target fraud account: {target_account}")

# HOP 1: Find all transactions involving the target account
hop1_mask = (df['Account'] == target_account) | (df['Account.1'] == target_account)
hop1_txns = df[hop1_mask]

# Get all accounts involved in Hop 1
hop1_accounts = set(hop1_txns['Account']).union(set(hop1_txns['Account.1']))

# HOP 2: Find all transactions involving ANY of the Hop 1 accounts
hop2_mask = df['Account'].isin(hop1_accounts) | df['Account.1'].isin(hop1_accounts)
hop2_txns = df[hop2_mask]

# Save to CSV
out_path = "sample_trans_5_subgraph.csv"
hop2_txns.to_csv(out_path, index=False)
print(f"Saved {len(hop2_txns)} transactions (2-hop subgraph) to {out_path}")
print(f"Actual Fraud in this subgraph: {hop2_txns['Is Laundering'].sum()}")
