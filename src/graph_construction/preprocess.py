# ============================================================
# src/graph_construction/preprocess.py
#
# Graph-aware data preprocessing pipeline
#
# What this does:
#   1. Clean & encode user_details.csv  → user node features
#   2. Clean & encode cards.csv         → card lookup table
#   3. Process transactions.csv chunks  → merchant aggregates
#                                       → clean edge attributes
#   4. Save preprocessed artefacts to  outputs/preprocessed/
#
# Run: python -m src.graph_construction.preprocess
# ============================================================

import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    DATA_DIR, OUTPUT_DIR, CHUNK_SIZE,
    TRAIN_YEAR_MAX, VAL_YEAR, TEST_YEAR_MIN
)

PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def strip_dollar(series: pd.Series) -> pd.Series:
    """Remove '$' and ',' → float."""
    return series.astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)


def encode_fraud(series: pd.Series) -> pd.Series:
    """'Yes' → 1, 'No' → 0."""
    return (series.str.strip().str.lower() == "yes").astype(np.int8)


def encode_chip(series: pd.Series) -> pd.Series:
    """Swipe → 0, Chip → 1, Online → 2."""
    m = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
    return series.map(m).fillna(0).astype(np.int8)


def cyclical(series: pd.Series, period: int):
    """Sine/cosine encoding for periodic features."""
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def encode_errors(series: pd.Series) -> pd.DataFrame:
    """
    Errors? column is multi-label (e.g. 'Bad PIN,Insufficient Balance').
    Returns 6 binary columns, one per error type.
    NaN → all zeros (no error).
    """
    error_types = [
        "Insufficient Balance", "Bad PIN", "Technical Glitch",
        "Bad Card Number", "Bad Expiration", "Bad CVV",
        "Bad Zipcode"
    ]
    result = {}
    for err in error_types:
        col_name = "err_" + err.lower().replace(" ", "_")
        result[col_name] = series.astype(str).str.contains(err, na=False).astype(np.int8)
    result["has_any_error"] = (series.notna() & (series.astype(str) != "nan")).astype(np.int8)
    return pd.DataFrame(result)


# ─────────────────────────────────────────────────────────────
# STEP 1 — USER NODES
# ─────────────────────────────────────────────────────────────

def preprocess_users() -> pd.DataFrame:
    """
    Clean user_details.csv and build user node feature matrix.

    Nulls:
      - Apartment: 1472 nulls → not used (dropped)
    Returns DataFrame indexed by User (0-based int matching transactions.csv).
    """
    print("\n" + "="*60)
    print("STEP 1: Preprocessing User Nodes")
    print("="*60)

    df = pd.read_csv(os.path.join(DATA_DIR, "user_details.csv"))
    print(f"  Loaded: {df.shape[0]} users, {df.shape[1]} columns")

    # ── Monetary columns → float ──────────────────────────────
    for col in ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]:
        df[col] = strip_dollar(df[col])

    # ── Derived features ──────────────────────────────────────
    df["debt_to_income_ratio"] = (
        df["Total Debt"] / df["Yearly Income - Person"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)

    df["years_to_retirement"] = (
        df["Retirement Age"] - df["Current Age"]
    ).astype(np.float32)

    # ── Encode gender ─────────────────────────────────────────
    # Male → 1, Female → 0
    df["gender_enc"] = (df["Gender"].str.strip() == "Male").astype(np.int8)

    # ── User index = row position (matches transactions['User']) ──
    # user_details rows are already 0-indexed (Person names only for reference)
    df["user_id"] = df.index.astype(int)

    # ── Select & normalize feature columns ───────────────────
    feat_cols = [
        "Current Age",            # age
        "years_to_retirement",    # financial horizon
        "gender_enc",             # gender binary
        "FICO Score",             # creditworthiness
        "Yearly Income - Person", # income
        "Total Debt",             # debt
        "debt_to_income_ratio",   # derived risk signal
        "Num Credit Cards",       # card exposure
        "Latitude",               # geo
        "Longitude",              # geo
    ]

    scaler = StandardScaler()
    feat_matrix = scaler.fit_transform(df[feat_cols].astype(float).fillna(0))
    feat_df = pd.DataFrame(feat_matrix, columns=feat_cols)
    feat_df["user_id"] = df["user_id"].values
    feat_df["person_name"] = df["Person"].values   # keep for reference

    # ── Save ──────────────────────────────────────────────────
    feat_df.to_csv(os.path.join(PREPROCESSED_DIR, "user_features.csv"), index=False)
    print(f"  ✅ User feature matrix: {feat_matrix.shape}  (dim={len(feat_cols)})")
    print(f"     Saved → outputs/preprocessed/user_features.parquet")

    # ── Stats ─────────────────────────────────────────────────
    print(f"     Gender dist: {df['Gender'].value_counts().to_dict()}")
    print(f"     FICO range : {df['FICO Score'].min()} – {df['FICO Score'].max()}")
    print(f"     Age range  : {df['Current Age'].min()} – {df['Current Age'].max()}")

    return feat_df


# ─────────────────────────────────────────────────────────────
# STEP 2 — CARD LOOKUP TABLE
# ─────────────────────────────────────────────────────────────

def preprocess_cards() -> pd.DataFrame:
    """
    Clean cards.csv and build card lookup table for edge enrichment.
    Keyed on (User, Card_index).
    """
    print("\n" + "="*60)
    print("STEP 2: Preprocessing Cards")
    print("="*60)

    df = pd.read_csv(os.path.join(DATA_DIR, "cards.csv"))
    print(f"  Loaded: {df.shape[0]} cards, {df.shape[1]} columns")

    # ── Credit Limit → float ──────────────────────────────────
    df["credit_limit_f"] = strip_dollar(df["Credit Limit"])

    # ── Binary flags ──────────────────────────────────────────
    df["has_chip_enc"]   = (df["Has Chip"].str.strip().str.upper() == "YES").astype(np.int8)
    df["dark_web_enc"]   = (df["Card on Dark Web"].str.strip().str.lower() == "yes").astype(np.int8)

    # ── Card type encoding ────────────────────────────────────
    card_type_map = {"Debit": 0, "Credit": 1, "Debit (Prepaid)": 2}
    df["card_type_enc"] = df["Card Type"].map(card_type_map).fillna(3).astype(np.int8)

    # ── Card brand encoding ───────────────────────────────────
    brand_le = LabelEncoder()
    df["card_brand_enc"] = brand_le.fit_transform(df["Card Brand"].astype(str))

    # ── Expiry → months until expiry from reference 2022-01 ──
    def months_until_expiry(exp_str):
        try:
            m, y = exp_str.strip().split("/")
            return max(int(y) * 12 + int(m) - (2022 * 12 + 1), 0)
        except Exception:
            return 0

    df["months_to_expiry"] = df["Expires"].apply(months_until_expiry).astype(np.float32)

    # ── Select lookup columns ─────────────────────────────────
    lookup_cols = [
        "User", "CARD INDEX",
        "credit_limit_f", "has_chip_enc", "dark_web_enc",
        "card_type_enc", "card_brand_enc", "months_to_expiry",
        "Cards Issued", "Year PIN last Changed"
    ]
    lookup_df = df[lookup_cols].rename(columns={"CARD INDEX": "Card"})

    # Normalize numeric card features
    num_cols = ["credit_limit_f", "months_to_expiry", "Cards Issued", "Year PIN last Changed"]
    scaler = StandardScaler()
    lookup_df[num_cols] = scaler.fit_transform(lookup_df[num_cols].astype(float))

    lookup_df.to_csv(os.path.join(PREPROCESSED_DIR, "card_lookup.csv"), index=False)
    print(f"  ✅ Card lookup table: {lookup_df.shape}")
    print(f"     Dark web cards    : {df['dark_web_enc'].sum()} / {len(df)}")
    print(f"     Has chip          : {df['has_chip_enc'].sum()} / {len(df)}")
    print(f"     Card types        : {df['Card Type'].value_counts().to_dict()}")
    print(f"     Saved → outputs/preprocessed/card_lookup.parquet")

    return lookup_df


# ─────────────────────────────────────────────────────────────
# STEP 3 — MERCHANT AGGREGATES  (train split only, no leakage)
# ─────────────────────────────────────────────────────────────

def compute_merchant_aggregates() -> pd.DataFrame:
    """
    Single chunked pass over transactions.csv.
    Computes per-merchant stats from TRAIN rows only (Year <= TRAIN_YEAR_MAX).

    Stats per merchant:
      txn_count, fraud_count, fraud_rate,
      avg_amount, std_amount, unique_users,
      top_mcc (most common MCC)
    """
    print("\n" + "="*60)
    print("STEP 3: Computing Merchant Aggregates (train set only)")
    print("="*60)

    # Accumulators
    stats = {}   # merchant_id (int) → dict

    total_rows = 0
    train_rows = 0

    for chunk in tqdm(
        pd.read_csv(
            os.path.join(DATA_DIR, "transactions.csv"),
            chunksize=CHUNK_SIZE,
            low_memory=False,
            dtype={"Merchant Name": int, "MCC": int, "User": int, "Card": int}
        ),
        desc="  Pass 1/2 – merchant aggregation"
    ):
        total_rows += len(chunk)
        train_chunk = chunk[chunk["Year"].astype(int) <= TRAIN_YEAR_MAX].copy()
        train_rows += len(train_chunk)

        if train_chunk.empty:
            continue

        # Clean amount
        train_chunk["amount_f"] = (
            train_chunk["Amount"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
        train_chunk["fraud_int"] = encode_fraud(train_chunk["Is Fraud?"])

        grouped = train_chunk.groupby("Merchant Name")
        for merchant_id, grp in grouped:
            if merchant_id not in stats:
                stats[merchant_id] = {
                    "n": 0, "sum_a": 0.0, "sum_sq_a": 0.0,
                    "fraud": 0, "users": set(), "mcc_list": []
                }
            s = stats[merchant_id]
            s["n"]         += len(grp)
            s["sum_a"]     += grp["amount_f"].sum()
            s["sum_sq_a"]  += (grp["amount_f"] ** 2).sum()
            s["fraud"]     += grp["fraud_int"].sum()
            s["users"].update(grp["User"].tolist())
            s["mcc_list"].extend(grp["MCC"].tolist())

    print(f"  Total rows scanned : {total_rows:,}")
    print(f"  Train rows used    : {train_rows:,}")
    print(f"  Unique merchants   : {len(stats):,}")

    # Build dataframe
    rows = []
    for mid, s in stats.items():
        n    = s["n"]
        mean = s["sum_a"] / n if n > 0 else 0
        var  = max((s["sum_sq_a"] / n) - mean**2, 0) if n > 0 else 0
        # Most common MCC
        from collections import Counter
        top_mcc = Counter(s["mcc_list"]).most_common(1)[0][0] if s["mcc_list"] else 0
        rows.append({
            "merchant_id":   int(mid),
            "txn_count":     n,
            "fraud_count":   s["fraud"],
            "fraud_rate":    s["fraud"] / n if n > 0 else 0.0,
            "avg_amount":    mean,
            "std_amount":    float(var ** 0.5),
            "unique_users":  len(s["users"]),
            "top_mcc":       int(top_mcc),
        })

    merch_df = pd.DataFrame(rows).sort_values("merchant_id").reset_index(drop=True)

    # MCC encoding
    le = LabelEncoder()
    merch_df["mcc_enc"] = le.fit_transform(merch_df["top_mcc"].astype(str))

    # Normalize numerical merchant features
    num_cols = ["txn_count", "fraud_rate", "avg_amount", "std_amount", "unique_users"]
    scaler = StandardScaler()
    merch_df_norm = merch_df.copy()
    merch_df_norm[num_cols] = scaler.fit_transform(merch_df[num_cols].astype(float))

    merch_df_norm.to_csv(os.path.join(PREPROCESSED_DIR, "merchant_features.csv"), index=False)

    # Also save raw (un-normalized) for inspection
    merch_df.to_csv(os.path.join(PREPROCESSED_DIR, "merchant_features_raw.csv"), index=False)

    print(f"\n  Merchant stats:")
    print(f"     Fraud rate (avg) : {merch_df['fraud_rate'].mean()*100:.3f}%")
    print(f"     Avg txn count    : {merch_df['txn_count'].mean():.1f}")
    print(f"     Unique MCCs      : {merch_df['top_mcc'].nunique()}")
    print(f"  ✅ Saved → outputs/preprocessed/merchant_features.parquet")

    return merch_df_norm


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRANSACTION EDGES (clean, encode, split)
# ─────────────────────────────────────────────────────────────

def preprocess_transactions(card_lookup: pd.DataFrame):
    """
    Second pass over transactions.csv.
    Cleans and encodes all edge features.
    Saves separate parquet files per split:
      edges_train.parquet, edges_val.parquet, edges_test.parquet

    Each row = one edge with columns:
      user_id, merchant_id,
      amount_log, hour_sin, hour_cos, dow_sin, dow_cos,
      is_weekend, is_night, chip_enc,
      has_any_error, [7 binary error cols],
      dark_web_enc, credit_limit_f, has_chip_enc,
      card_type_enc, card_brand_enc,
      fraud_label, split
    """
    print("\n" + "="*60)
    print("STEP 4: Preprocessing Transaction Edges")
    print("="*60)

    card_feat_cols = [
        "dark_web_enc", "credit_limit_f", "has_chip_enc",
        "card_type_enc", "card_brand_enc"
    ]

    split_buffers = {"train": [], "val": [], "test": []}
    counters      = {"train": {"total": 0, "fraud": 0},
                     "val":   {"total": 0, "fraud": 0},
                     "test":  {"total": 0, "fraud": 0}}

    chunk_id = 0
    for chunk in tqdm(
        pd.read_csv(
            os.path.join(DATA_DIR, "transactions.csv"),
            chunksize=CHUNK_SIZE,
            low_memory=False,
            dtype={"Merchant Name": int, "MCC": int, "User": int, "Card": int}
        ),
        desc="  Pass 2/2 – edge preprocessing"
    ):
        chunk_id += 1
        df = chunk.copy()

        # ── Determine split ───────────────────────────────────
        year = df["Year"].astype(int)
        conditions = [
            year <= TRAIN_YEAR_MAX,
            year == VAL_YEAR,
            year >= TEST_YEAR_MIN
        ]
        choices = ["train", "val", "test"]
        df["split"] = np.select(conditions, choices, default="test")

        # ── Amount → log-scale float ──────────────────────────
        df["amount_f"] = (
            df["Amount"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
        df["amount_log"] = np.log1p(df["amount_f"]).astype(np.float32)

        # ── Temporal features ─────────────────────────────────
        df["hour"] = df["Time"].astype(str).str.split(":").str[0].astype(int)
        sin_h, cos_h = cyclical(df["hour"], 24)
        df["hour_sin"] = sin_h
        df["hour_cos"] = cos_h

        # Day of week via pandas datetime
        df["date"] = pd.to_datetime(
            df[["Year","Month","Day"]].rename(columns={"Year":"year","Month":"month","Day":"day"}),
            errors="coerce"
        )
        df["dow"] = df["date"].dt.dayofweek.fillna(0).astype(int)
        sin_d, cos_d = cyclical(df["dow"], 7)
        df["dow_sin"]    = sin_d
        df["dow_cos"]    = cos_d
        df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
        df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)

        # ── Chip encoding ─────────────────────────────────────
        df["chip_enc"] = encode_chip(df["Use Chip"])

        # ── Error encoding ────────────────────────────────────
        err_df = encode_errors(df["Errors?"])
        df     = pd.concat([df, err_df], axis=1)

        # ── Fraud label ───────────────────────────────────────
        df["fraud_label"] = encode_fraud(df["Is Fraud?"])

        # ── Card features (vectorized merge) ──────────────────
        card_lookup_sub = card_lookup[['User', 'Card'] + card_feat_cols].copy()
        df = df.merge(card_lookup_sub, on=['User', 'Card'], how='left', suffixes=('', '_card'))
        for feat in card_feat_cols:
            if feat + '_card' in df.columns:
                df[feat] = df[feat + '_card'].fillna(0)
                df.drop(columns=[feat + '_card'], inplace=True)
            else:
                df[feat] = df.get(feat, pd.Series(0, index=df.index)).fillna(0)

        # ── Final edge columns ────────────────────────────────
        edge_cols = [
            "User", "Merchant Name",
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
            "fraud_label", "split"
        ]
        df_out = df[edge_cols].rename(columns={"User": "user_id", "Merchant Name": "merchant_id"})

        # ── Route to split buffers ────────────────────────────
        for sp in ["train", "val", "test"]:
            sub = df_out[df_out["split"] == sp].drop(columns=["split"])
            if not sub.empty:
                split_buffers[sp].append(sub)
                counters[sp]["total"] += len(sub)
                counters[sp]["fraud"] += sub["fraud_label"].sum()

        # Write out buffer every 20 chunks to avoid RAM blowout
        if chunk_id % 20 == 0:
            _flush_buffers(split_buffers, append=(chunk_id > 20))

    # Final flush
    _flush_buffers(split_buffers, append=(chunk_id > 20))

    # ── Summary ───────────────────────────────────────────────
    print("\n  Edge split summary:")
    print(f"  {'Split':<8} {'Edges':>12} {'Fraud':>10} {'Fraud %':>10}")
    print(f"  {'-'*44}")
    for sp, c in counters.items():
        if c["total"] > 0:
            rate = c["fraud"] / c["total"] * 100
            print(f"  {sp:<8} {c['total']:>12,} {c['fraud']:>10,} {rate:>9.3f}%")

    print("\n  ✅ Saved → outputs/preprocessed/edges_[train|val|test].parquet")


def _flush_buffers(buffers: dict, append: bool = False):
    """Write accumulated dataframes to CSV files."""
    for sp, chunks in buffers.items():
        if not chunks:
            continue
        path = os.path.join(PREPROCESSED_DIR, f"edges_{sp}.csv")
        combined = pd.concat(chunks, ignore_index=True)
        if append and os.path.exists(path):
            combined.to_csv(path, mode='a', header=False, index=False)
        else:
            combined.to_csv(path, index=False)
        buffers[sp] = []   # clear buffer


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*60)
    print("  GRAPH-AWARE DATA PREPROCESSING PIPELINE")
    print("  Bipartite User–Merchant Fraud Detection")
    print("="*60)
    print(f"  Splits: Train (Year≤{TRAIN_YEAR_MAX}) | Val ({VAL_YEAR}) | Test (Year≥{TEST_YEAR_MIN})")
    print(f"  Output → {PREPROCESSED_DIR}")

    user_feats  = preprocess_users()
    card_lookup = preprocess_cards()
    _           = compute_merchant_aggregates()
    preprocess_transactions(card_lookup)

    print("\n" + "="*60)
    print("  ✅ ALL PREPROCESSING DONE")
    print(f"  Files saved in: {PREPROCESSED_DIR}")
    print("="*60)


if __name__ == "__main__":
    run_all()
