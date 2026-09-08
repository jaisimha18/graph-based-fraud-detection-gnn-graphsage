import pandas as pd
import glob
import os

print("Loading full dataset...")
full_df = pd.read_csv("AML_DATASET/HI-Small_Trans.csv")

sample_files = glob.glob("sample_trans*.csv")

for file in sample_files:
    # Skip if it's already a huge subgraph
    if file == "sample_trans_5_subgraph.csv":
        continue
        
    print(f"\nEnriching {file}...")
    df = pd.read_csv(file)
    
    # Get seed accounts
    seed_accounts = set(df['Account']).union(set(df['Account.1']))
    
    # Find all transactions involving these accounts in the full dataset
    mask = full_df['Account'].isin(seed_accounts) | full_df['Account.1'].isin(seed_accounts)
    connected_txns = full_df[mask]
    
    # We want to add context, but not overwhelm the browser. 
    # Let's cap the added context to 200 transactions.
    # We prioritize fraud transactions if any exist in the neighborhood, then sample legitimate ones.
    
    fraud_context = connected_txns[connected_txns['Is Laundering'] == 1]
    legit_context = connected_txns[connected_txns['Is Laundering'] == 0]
    
    # We already have the original df, so we exclude those to avoid duplicates
    # We can just drop duplicates after concatenating
    
    # Sample up to 200 legitimate context transactions
    if len(legit_context) > 150:
        legit_context = legit_context.sample(n=150, random_state=42)
        
    enriched_df = pd.concat([df, fraud_context, legit_context]).drop_duplicates()
    
    # Save back to the file
    enriched_df.to_csv(file, index=False)
    
    print(f"  Original size: {len(df)}")
    print(f"  Enriched size: {len(enriched_df)}")
    print(f"  Total Fraud in enriched file: {enriched_df['Is Laundering'].sum()}")

print("\nDone enriching all sample files!")
