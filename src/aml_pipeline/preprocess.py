# ============================================================
# src/aml_pipeline/preprocess.py
#
# AML Graph-aware preprocessing pipeline
#
# What this does:
#   1. Load & enrich HI-Small_accounts.csv   → account node features
#   2. Process HI-Small_Trans.csv in chunks  → per-account aggregates
#                                             → clean edge attributes
#   3. Chronological split:
#        Train : Sept  1 –  8   (days 1-8)
#        Val   : Sept  9        (day 9)
#        Test  : Sept 10 – 18   (days 10-18)
#   4. Save preprocessed artefacts to outputs/aml/preprocessed/
#
# Data-leakage prevention:
#   • Account aggregates are computed from TRAINING transactions only.
#   • StandardScaler is fit on TRAINING accounts only, then applied
#     to all accounts (train + val + test).
#   • laundering_rate is NOT used as a node feature — it directly
#     encodes the label and would leak target information.
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
    TRAIN_END_DAY, VAL_DAY, TEST_START_DAY,
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


def _parse_day(timestamp_series: pd.Series) -> pd.Series:
    """Extract day-of-month from Timestamp column (format: YYYY/MM/DD HH:MM)."""
    return pd.to_datetime(timestamp_series, format="%Y/%m/%d %H:%M").dt.day


def assign_split(day_series: pd.Series) -> pd.Series:
    """
    Assign chronological split based on day-of-month.
      Train : day <= TRAIN_END_DAY   (Sept 1-8)
      Val   : day == VAL_DAY         (Sept 9)
      Test  : day >= TEST_START_DAY  (Sept 10-18)
    """
    conditions = [
        day_series <= TRAIN_END_DAY,
        day_series == VAL_DAY,
        day_series >= TEST_START_DAY,
    ]
    choices = ["train", "val", "test"]
    return pd.Series(
        np.select(conditions, choices, default="test"),
        index=day_series.index,
    )


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
# STEP 2 — ACCOUNT AGGREGATES FROM TRAINING TRANSACTIONS ONLY
# ─────────────────────────────────────────────────────────────

def compute_account_aggregates() -> pd.DataFrame:
    """
    Chunked pass over HI-Small_Trans.csv.
    Computes per-account stats using ONLY TRAINING transactions
    (Sept 1 – TRAIN_END_DAY) to prevent data leakage.

    Stats per account:
      out_txn_count         : outgoing transactions (train only)
      in_txn_count          : incoming transactions (train only)
      out_avg_amount        : average outgoing amount (train only)
      in_avg_amount         : average incoming amount (train only)
      unique_counterparties : unique accounts transacted with (train only)

    NOTE: laundering_rate is intentionally EXCLUDED — it directly
    encodes the target label and would cause label leakage even if
    restricted to training data (the model would learn to rely on a
    pre-computed fraud signal rather than graph structure).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Computing Account Aggregates (TRAIN period only)")
    print("=" * 60)
    print(f"  Training window: Sept 1 – Sept {TRAIN_END_DAY}")

    # Accumulators: account_id -> stats
    out_stats = {}   # outgoing stats
    in_stats = {}    # incoming stats
    counterparties = {}  # account_id -> set of counterparty accounts

    total_rows = 0
    train_rows = 0

    for chunk in tqdm(
        pd.read_csv(TRANSACTIONS_CSV, chunksize=CHUNK_SIZE, low_memory=False),
        desc="  Aggregation pass"
    ):
        total_rows += len(chunk)

        # Parse day and filter to training period only
        day = _parse_day(chunk["Timestamp"])
        train_mask = day <= TRAIN_END_DAY
        train_chunk = chunk[train_mask]
        train_rows += len(train_chunk)

        if train_chunk.empty:
            continue

        for _, row in train_chunk.iterrows():
            from_acc = row["Account"]
            to_acc = row["Account.1"]
            amt = float(row["Amount Paid"])

            # Outgoing stats
            if from_acc not in out_stats:
                out_stats[from_acc] = {"n": 0, "sum_a": 0.0}
            out_stats[from_acc]["n"] += 1
            out_stats[from_acc]["sum_a"] += amt

            # Incoming stats
            if to_acc not in in_stats:
                in_stats[to_acc] = {"n": 0, "sum_a": 0.0}
            in_stats[to_acc]["n"] += 1
            in_stats[to_acc]["sum_a"] += float(row["Amount Received"])

            # Counterparties
            if from_acc not in counterparties:
                counterparties[from_acc] = set()
            counterparties[from_acc].add(to_acc)
            if to_acc not in counterparties:
                counterparties[to_acc] = set()
            counterparties[to_acc].add(from_acc)

    print(f"  Total rows scanned : {total_rows:,}")
    print(f"  Training rows used : {train_rows:,}")

    # Build aggregate dataframe — union of all accounts seen in training
    all_accounts = set(out_stats.keys()) | set(in_stats.keys())
    print(f"  Unique accounts in training transactions: {len(all_accounts):,}")

    rows = []
    for acc in all_accounts:
        o = out_stats.get(acc, {"n": 0, "sum_a": 0.0})
        i = in_stats.get(acc, {"n": 0, "sum_a": 0.0})
        rows.append({
            "account_id": acc,
            "out_txn_count": o["n"],
            "in_txn_count": i["n"],
            "out_avg_amount": o["sum_a"] / o["n"] if o["n"] > 0 else 0.0,
            "in_avg_amount": i["sum_a"] / i["n"] if i["n"] > 0 else 0.0,
            "unique_counterparties": len(counterparties.get(acc, set())),
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

    IMPORTANT: StandardScaler is fit on TRAINING-period accounts
    only (those that appear in agg_df) and then applied to all
    accounts — this prevents val/test information from influencing
    the scaling parameters.

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

    # Fill accounts with no training transactions
    fill_cols = [
        "out_txn_count", "in_txn_count",
        "out_avg_amount", "in_avg_amount",
        "unique_counterparties",
    ]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0)

    print(f"  Merged accounts: {len(merged):,}")
    print(f"  Accounts with train transactions: {(merged['out_txn_count'] > 0).sum():,}")
    print(f"  Accounts without train transactions: {(merged['out_txn_count'] == 0).sum():,}")

    # ── Feature columns (8 dims) ─────────────────────────────
    feat_cols = [
        "bank_id", "entity_type_enc", "is_crypto_bank",
        "out_txn_count", "in_txn_count",
        "out_avg_amount", "in_avg_amount",
        "unique_counterparties",
    ]

    # Fit scaler on training-period accounts only (those with aggregates),
    # then transform ALL accounts to prevent data leakage.
    train_account_ids = set(agg_df["account_id"].values)
    train_mask = merged["account_id"].isin(train_account_ids)

    scaler = StandardScaler()
    scaler.fit(merged.loc[train_mask, feat_cols].astype(float).values)
    feat_matrix = scaler.transform(merged[feat_cols].astype(float).values)

    feat_df = pd.DataFrame(feat_matrix, columns=feat_cols)
    feat_df["account_id"] = merged["account_id"].values

    # Save
    feat_df.to_csv(os.path.join(PREPROCESSED_DIR, "account_features.csv"), index=False)

    print(f"  ✅ Account feature matrix: {feat_matrix.shape}  (dim={len(feat_cols)})")
    print(f"     Scaler fit on {train_mask.sum():,} training accounts")
    print(f"     Saved → {os.path.join(PREPROCESSED_DIR, 'account_features.csv')}")

    return feat_df


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRANSACTION EDGES (clean, encode, chronological split)
# ─────────────────────────────────────────────────────────────

def preprocess_transactions():
    """
    Process HI-Small_Trans.csv in chunks.
    Encode edge features and split chronologically:
      Train : Sept  1 –  8  (day <= TRAIN_END_DAY)
      Val   : Sept  9       (day == VAL_DAY)
      Test  : Sept 10 – 18  (day >= TEST_START_DAY)

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
    print(f"  Split strategy: Chronological")
    print(f"    Train : Sept 1 – {TRAIN_END_DAY}")
    print(f"    Val   : Sept {VAL_DAY}")
    print(f"    Test  : Sept {TEST_START_DAY} – 18")

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

    # ── Process edges in chunks, route to split buffers ───────
    split_buffers = {"train": [], "val": [], "test": []}
    counters = {
        "train": {"total": 0, "launder": 0},
        "val":   {"total": 0, "launder": 0},
        "test":  {"total": 0, "launder": 0},
    }

    chunk_id = 0
    for chunk in tqdm(
        pd.read_csv(TRANSACTIONS_CSV, chunksize=CHUNK_SIZE, low_memory=False),
        desc="  Edge preprocessing"
    ):
        chunk_id += 1
        df = chunk.copy()

        # ── Parse timestamp ───────────────────────────────────
        dt = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
        df["day"] = dt.dt.day
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek

        # ── Assign chronological split ────────────────────────
        df["split"] = assign_split(df["day"])

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

        # ── Route to split buffers ────────────────────────────
        for sp in ["train", "val", "test"]:
            sub = df.loc[df["split"] == sp, edge_cols]
            if not sub.empty:
                split_buffers[sp].append(sub)
                counters[sp]["total"] += len(sub)
                counters[sp]["launder"] += sub["laundering_label"].sum()

    # ── Concatenate and save each split ───────────────────────
    for sp in ["train", "val", "test"]:
        if split_buffers[sp]:
            sp_df = pd.concat(split_buffers[sp], ignore_index=True)
        else:
            sp_df = pd.DataFrame(columns=edge_cols)
        path = os.path.join(PREPROCESSED_DIR, f"edges_{sp}.csv")
        sp_df.to_csv(path, index=False)

    # ── Summary ───────────────────────────────────────────────
    total_edges = sum(c["total"] for c in counters.values())
    print(f"\n  Total edges: {total_edges:,}")
    print(f"\n  Chronological edge split summary:")
    print(f"  {'Split':<8} {'Days':<12} {'Edges':>12} {'Laundering':>12} {'Rate':>10}")
    print(f"  {'-'*58}")
    day_ranges = {
        "train": f"Sept 1-{TRAIN_END_DAY}",
        "val":   f"Sept {VAL_DAY}",
        "test":  f"Sept {TEST_START_DAY}-18",
    }
    for sp in ["train", "val", "test"]:
        n_l = counters[sp]["launder"]
        n_t = counters[sp]["total"]
        rate = n_l / n_t * 100 if n_t > 0 else 0.0
        print(f"  {sp:<8} {day_ranges[sp]:<12} {n_t:>12,} {n_l:>12,} {rate:>9.4f}%")

    print(f"\n  ✅ Saved → {PREPROCESSED_DIR}/edges_[train|val|test].csv")

    # Free memory
    del split_buffers


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "=" * 60)
    print("  AML GRAPH-AWARE PREPROCESSING PIPELINE")
    print("  Homogeneous Account–Account Graph")
    print("=" * 60)
    print(f"  Dataset: IBM HI-Small AML")
    print(f"  Split:   Chronological (Train: Sept 1-{TRAIN_END_DAY} | "
          f"Val: Sept {VAL_DAY} | Test: Sept {TEST_START_DAY}-18)")
    print(f"  Output → {PREPROCESSED_DIR}")

    # Step 1: Static account features
    static_df = preprocess_accounts()

    # Step 2: Transaction-derived aggregates (TRAINING period only)
    agg_df = compute_account_aggregates()

    # Step 3: Merge & normalize account features
    #   (scaler fit on training accounts only → no leakage)
    _ = build_account_features(static_df, agg_df)

    # Step 4: Transaction edge preprocessing + chronological split
    preprocess_transactions()

    print("\n" + "=" * 60)
    print("  ✅ ALL AML PREPROCESSING DONE")
    print(f"  Files saved in: {PREPROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
