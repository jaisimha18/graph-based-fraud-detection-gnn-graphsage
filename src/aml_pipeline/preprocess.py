# ============================================================
# src/aml_pipeline/preprocess.py
#
# AML Graph-aware preprocessing pipeline
#
# What this does:
#   1. Load & enrich HI-Small_accounts.csv   → account node features
#   2. Process HI-Small_Trans.csv in chunks  → per-account aggregates
#                                             → clean edge attributes
#   3. Temporal split (Sept 1-12 / 13-15 / 16-18)
#   4. Save preprocessed artefacts to outputs/aml/preprocessed/
#
# Graph structure:
#   Nodes : accounts (single type — homogeneous graph)
#   Edges : account → account (money transfers)
#   Label : Is Laundering (0/1)
#
# Run: python -m src.aml_pipeline.preprocess
# ============================================================

import os, sys, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aml_pipeline.aml_config import (
    AML_DATA_DIR, AML_OUTPUT_DIR,
    TRANSACTIONS_CSV, ACCOUNTS_CSV,
    PREPROCESSED_DIR, CHUNK_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)

os.makedirs(PREPROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def cyclical(series: pd.Series, period: int):
    """Sine/cosine encoding for periodic features."""
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def extract_entity_type(name: str) -> str:
    """Extract base entity type from names like 'Corporation #12345'."""
    return re.sub(r'\s*#\d+$', '', str(name)).strip()


# ─────────────────────────────────────────────────────────────
# STEP 1 — ACCOUNT NODES (static features from accounts.csv)
# ─────────────────────────────────────────────────────────────

def preprocess_accounts() -> pd.DataFrame:
    """
    Load HI-Small_accounts.csv and build account node feature table.

    Columns in accounts.csv:
      Bank Name, Bank ID, Account Number, Entity ID, Entity Name

    Derived features:
      - bank_id           : integer bank identifier
      - entity_type_enc   : LabelEncoded entity type (6 classes)
      - is_crypto_bank    : 1 if bank name contains 'Crypto'/'Crytpo'

    Returns DataFrame with one row per unique account.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing Account Nodes")
    print("=" * 60)

    df = pd.read_csv(ACCOUNTS_CSV)
    print(f"  Loaded: {len(df):,} accounts, {df.shape[1]} columns")

    # ── Entity type encoding ──────────────────────────────────
    df["entity_type"] = df["Entity Name"].apply(extract_entity_type)
    le_entity = LabelEncoder()
    df["entity_type_enc"] = le_entity.fit_transform(df["entity_type"])
    print(f"  Entity types: {dict(zip(le_entity.classes_, le_entity.transform(le_entity.classes_)))}")

    # ── Crypto bank flag ──────────────────────────────────────
    # Note: dataset has typo "Crytpo" in some entries
    df["is_crypto_bank"] = df["Bank Name"].str.contains(
        r"Crypto|Crytpo", case=False, na=False
    ).astype(np.int8)
    print(f"  Crypto bank accounts: {df['is_crypto_bank'].sum():,}")

    # ── Bank ID (already integer) ─────────────────────────────
    df["bank_id"] = df["Bank ID"].astype(int)

    # ── Account key = Account Number (hex string, unique per account)
    # This is what we match in transactions
    df["account_id"] = df["Account Number"]

    # ── Select static features ────────────────────────────────
    static_df = df[["account_id", "bank_id", "entity_type_enc", "is_crypto_bank"]].copy()
    static_df = static_df.drop_duplicates(subset=["account_id"]).reset_index(drop=True)

    print(f"  Unique accounts: {len(static_df):,}")
    print(f"  Static features: bank_id, entity_type_enc, is_crypto_bank")

    return static_df


# ─────────────────────────────────────────────────────────────
# STEP 2 — ACCOUNT AGGREGATES FROM TRANSACTIONS (train only)
# ─────────────────────────────────────────────────────────────

def compute_account_aggregates() -> pd.DataFrame:
    """
    Chunked pass over HI-Small_Trans.csv.
    Computes per-account stats from ALL transactions.

    NOTE: Since we use random stratified splitting (not temporal),
    aggregates are computed from the full dataset before split
    assignment. This is valid because node features are based on
    the global graph structure, not on label-dependent information.

    Stats per account:
      out_txn_count   : outgoing transactions
      in_txn_count    : incoming transactions
      out_avg_amount  : average outgoing amount
      in_avg_amount   : average incoming amount
      unique_counterparties : unique accounts transacted with
      laundering_rate : fraction of transactions flagged as laundering
    """
    print("\n" + "=" * 60)
    print("STEP 2: Computing Account Aggregates")
    print("=" * 60)

    # Accumulators: account_id -> stats
    out_stats = {}   # outgoing stats
    in_stats = {}    # incoming stats
    counterparties = {}  # account_id -> set of counterparty accounts

    total_rows = 0

    for chunk in tqdm(
        pd.read_csv(TRANSACTIONS_CSV, chunksize=CHUNK_SIZE, low_memory=False),
        desc="  Aggregation pass"
    ):
        total_rows += len(chunk)

        # Process all rows (no temporal filtering — random split later)
        for _, row in chunk.iterrows():
            from_acc = row["Account"]
            to_acc = row["Account.1"]
            amt = float(row["Amount Paid"])
            is_launder = int(row["Is Laundering"])

            # Outgoing stats
            if from_acc not in out_stats:
                out_stats[from_acc] = {"n": 0, "sum_a": 0.0, "launder": 0}
            out_stats[from_acc]["n"] += 1
            out_stats[from_acc]["sum_a"] += amt
            out_stats[from_acc]["launder"] += is_launder

            # Incoming stats
            if to_acc not in in_stats:
                in_stats[to_acc] = {"n": 0, "sum_a": 0.0, "launder": 0}
            in_stats[to_acc]["n"] += 1
            in_stats[to_acc]["sum_a"] += float(row["Amount Received"])
            in_stats[to_acc]["launder"] += is_launder

            # Counterparties
            if from_acc not in counterparties:
                counterparties[from_acc] = set()
            counterparties[from_acc].add(to_acc)
            if to_acc not in counterparties:
                counterparties[to_acc] = set()
            counterparties[to_acc].add(from_acc)

    print(f"  Total rows scanned : {total_rows:,}")

    # Build aggregate dataframe — union of all accounts seen
    all_accounts = set(out_stats.keys()) | set(in_stats.keys())
    print(f"  Unique accounts in transactions: {len(all_accounts):,}")

    rows = []
    for acc in all_accounts:
        o = out_stats.get(acc, {"n": 0, "sum_a": 0.0, "launder": 0})
        i = in_stats.get(acc, {"n": 0, "sum_a": 0.0, "launder": 0})
        total_txn = o["n"] + i["n"]
        total_launder = o["launder"] + i["launder"]
        rows.append({
            "account_id": acc,
            "out_txn_count": o["n"],
            "in_txn_count": i["n"],
            "out_avg_amount": o["sum_a"] / o["n"] if o["n"] > 0 else 0.0,
            "in_avg_amount": i["sum_a"] / i["n"] if i["n"] > 0 else 0.0,
            "unique_counterparties": len(counterparties.get(acc, set())),
            "laundering_rate": total_launder / total_txn if total_txn > 0 else 0.0,
        })

    agg_df = pd.DataFrame(rows)
    print(f"  Aggregate features computed for {len(agg_df):,} accounts")

    return agg_df


# ─────────────────────────────────────────────────────────────
# STEP 3 — MERGE STATIC + AGGREGATE → FINAL ACCOUNT FEATURES
# ─────────────────────────────────────────────────────────────

def build_account_features(
    static_df: pd.DataFrame,
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge static account info with transaction-derived aggregates.
    Normalize all numeric features.
    Save to account_features.csv.

    Final feature vector per account (8 dims):
      bank_id, entity_type_enc, is_crypto_bank,
      out_txn_count, in_txn_count,
      out_avg_amount, in_avg_amount,
      unique_counterparties
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building Account Feature Matrix")
    print("=" * 60)

    # Merge on account_id
    merged = static_df.merge(agg_df, on="account_id", how="left")

    # Fill accounts with no transactions (they exist in accounts.csv but not in train)
    fill_cols = [
        "out_txn_count", "in_txn_count",
        "out_avg_amount", "in_avg_amount",
        "unique_counterparties", "laundering_rate",
    ]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0)

    print(f"  Merged accounts: {len(merged):,}")
    print(f"  Accounts with transactions: {(merged['out_txn_count'] > 0).sum():,}")
    print(f"  Accounts without transactions: {(merged['out_txn_count'] == 0).sum():,}")

    # ── Feature columns (8 dims) ─────────────────────────────
    feat_cols = [
        "bank_id", "entity_type_enc", "is_crypto_bank",
        "out_txn_count", "in_txn_count",
        "out_avg_amount", "in_avg_amount",
        "unique_counterparties",
    ]

    # Normalize
    scaler = StandardScaler()
    feat_matrix = scaler.fit_transform(merged[feat_cols].astype(float).values)
    feat_df = pd.DataFrame(feat_matrix, columns=feat_cols)
    feat_df["account_id"] = merged["account_id"].values

    # Save
    feat_df.to_csv(os.path.join(PREPROCESSED_DIR, "account_features.csv"), index=False)

    print(f"  ✅ Account feature matrix: {feat_matrix.shape}  (dim={len(feat_cols)})")
    print(f"     Saved → {os.path.join(PREPROCESSED_DIR, 'account_features.csv')}")

    return feat_df


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRANSACTION EDGES (clean, encode, stratified split)
# ─────────────────────────────────────────────────────────────

def preprocess_transactions():
    """
    Process HI-Small_Trans.csv in chunks.
    Encode edge features and split using random stratified sampling.

    NOTE: Temporal split doesn't work for this dataset because
    99.9% of transactions fall on days 1-10. Days 11-18 contain
    almost exclusively laundering edges, producing unusable
    val/test sets with ~60% laundering rate.

    Instead, we use random stratified split (80/10/10) that
    preserves the ~0.10% laundering rate across all splits.

    Edge features (14 dims):
      amount_received_log, amount_paid_log, amount_diff_log,
      is_cross_bank, is_self_transfer, is_cross_currency,
      receiving_currency_enc, payment_format_enc,
      hour_sin, hour_cos, dow_sin, dow_cos,
      is_night, is_weekend

    Saves:
      edges_train.csv, edges_val.csv, edges_test.csv
    """
    print("\n" + "=" * 60)
    print("STEP 4: Preprocessing Transaction Edges")
    print("=" * 60)
    print(f"  Split strategy: Random Stratified ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})")

    # ── Fit label encoders on full dataset first ─────────────
    print("  Fitting encoders...")
    currencies = set()
    formats = set()
    for chunk in pd.read_csv(TRANSACTIONS_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        currencies.update(chunk["Receiving Currency"].dropna().unique())
        currencies.update(chunk["Payment Currency"].dropna().unique())
        formats.update(chunk["Payment Format"].dropna().unique())

    le_currency = LabelEncoder()
    le_currency.fit(sorted(currencies))
    le_format = LabelEncoder()
    le_format.fit(sorted(formats))

    print(f"  Currencies ({len(le_currency.classes_)}): {list(le_currency.classes_)}")
    print(f"  Payment formats ({len(le_format.classes_)}): {list(le_format.classes_)}")

    # ── Pass 1: Encode all edges into one buffer ──────────────
    all_edges = []

    chunk_id = 0
    for chunk in tqdm(
        pd.read_csv(TRANSACTIONS_CSV, chunksize=CHUNK_SIZE, low_memory=False),
        desc="  Edge preprocessing"
    ):
        chunk_id += 1
        df = chunk.copy()

        # ── Parse timestamp ───────────────────────────────────
        dt = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek

        # ── Account IDs ───────────────────────────────────────
        df["from_account"] = df["Account"]
        df["to_account"] = df["Account.1"]

        # ── Amount features ───────────────────────────────────
        df["amount_received_log"] = np.log1p(
            df["Amount Received"].astype(float).clip(lower=0)
        ).astype(np.float32)
        df["amount_paid_log"] = np.log1p(
            df["Amount Paid"].astype(float).clip(lower=0)
        ).astype(np.float32)
        df["amount_diff_log"] = (
            df["amount_received_log"] - df["amount_paid_log"]
        ).astype(np.float32)

        # ── Binary flags ──────────────────────────────────────
        df["is_cross_bank"] = (
            df["From Bank"].astype(str) != df["To Bank"].astype(str)
        ).astype(np.int8)
        df["is_self_transfer"] = (
            df["Account"] == df["Account.1"]
        ).astype(np.int8)
        df["is_cross_currency"] = (
            df["Receiving Currency"] != df["Payment Currency"]
        ).astype(np.int8)

        # ── Categorical encodings ─────────────────────────────
        df["receiving_currency_enc"] = le_currency.transform(
            df["Receiving Currency"].fillna("US Dollar")
        ).astype(np.int8)
        df["payment_format_enc"] = le_format.transform(
            df["Payment Format"].fillna("Cheque")
        ).astype(np.int8)

        # ── Temporal features ─────────────────────────────────
        sin_h, cos_h = cyclical(df["hour"], 24)
        df["hour_sin"] = sin_h
        df["hour_cos"] = cos_h
        sin_d, cos_d = cyclical(df["dow"], 7)
        df["dow_sin"] = sin_d
        df["dow_cos"] = cos_d
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
        df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)

        # ── Label ─────────────────────────────────────────────
        df["laundering_label"] = df["Is Laundering"].astype(np.int8)

        # ── Select output columns ─────────────────────────────
        edge_cols = [
            "from_account", "to_account",
            "amount_received_log", "amount_paid_log", "amount_diff_log",
            "is_cross_bank", "is_self_transfer", "is_cross_currency",
            "receiving_currency_enc", "payment_format_enc",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "is_night", "is_weekend",
            "laundering_label",
        ]
        all_edges.append(df[edge_cols])

    # ── Combine all edges ─────────────────────────────────────
    print("\n  Combining all edges...")
    full_df = pd.concat(all_edges, ignore_index=True)
    print(f"  Total edges: {len(full_df):,}")
    print(f"  Laundering : {full_df['laundering_label'].sum():,} ({full_df['laundering_label'].mean()*100:.4f}%)")

    # ── Random stratified split ───────────────────────────────
    print(f"\n  Performing stratified split ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_df, val_test_df = train_test_split(
        full_df,
        test_size=val_test_ratio,
        random_state=42,
        stratify=full_df["laundering_label"],
    )

    # Second split: val vs test
    test_relative = TEST_RATIO / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_relative,
        random_state=42,
        stratify=val_test_df["laundering_label"],
    )

    # ── Save ──────────────────────────────────────────────────
    for sp, sp_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(PREPROCESSED_DIR, f"edges_{sp}.csv")
        sp_df.to_csv(path, index=False)

    # ── Summary ───────────────────────────────────────────────
    print("\n  Edge split summary:")
    print(f"  {'Split':<8} {'Edges':>12} {'Laundering':>12} {'Rate':>10}")
    print(f"  {'-'*46}")
    for sp, sp_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_l = sp_df["laundering_label"].sum()
        n_t = len(sp_df)
        rate = n_l / n_t * 100
        print(f"  {sp:<8} {n_t:>12,} {n_l:>12,} {rate:>9.4f}%")

    print(f"\n  ✅ Saved → {PREPROCESSED_DIR}/edges_[train|val|test].csv")

    # Free memory
    del full_df, train_df, val_df, test_df, val_test_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "=" * 60)
    print("  AML GRAPH-AWARE PREPROCESSING PIPELINE")
    print("  Homogeneous Account–Account Graph")
    print("=" * 60)
    print(f"  Dataset: IBM HI-Small AML")
    print(f"  Splits:  Stratified Random ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})")
    print(f"  Output → {PREPROCESSED_DIR}")

    # Step 1: Static account features
    static_df = preprocess_accounts()

    # Step 2: Transaction-derived aggregates (train only — uses full data
    #          since aggregates are computed before split assignment)
    agg_df = compute_account_aggregates()

    # Step 3: Merge & normalize account features
    _ = build_account_features(static_df, agg_df)

    # Step 4: Transaction edge preprocessing + stratified split
    preprocess_transactions()

    print("\n" + "=" * 60)
    print("  ✅ ALL AML PREPROCESSING DONE")
    print(f"  Files saved in: {PREPROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

