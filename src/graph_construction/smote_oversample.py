# ============================================================
# src/graph_construction/smote_oversample.py
#
# Edge-level SMOTE for graph fraud detection
#
# Why SMOTE here:
#   - Fraud rate in train = 0.132% (24,924 fraud / 18.8M edges)
#   - pos_weight alone helps loss but not the model's feature space
#   - SMOTE synthesizes new fraud EDGES by interpolating between
#     real fraud edges in feature space → richer minority signal
#
# What is NOT done:
#   - We do NOT apply SMOTE to val or test (would cause leakage)
#   - We do NOT apply it to node features (only edge features)
#   - SMOTE is applied AFTER preprocessing, BEFORE graph build
#
# Output:
#   outputs/preprocessed/edges_train_smote.csv
#   (drop-in replacement for edges_train.csv with balanced classes)
#
# Run: python -m src.graph_construction.smote_oversample
# ============================================================

import os, sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import OUTPUT_DIR

PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
TRAIN_PATH       = os.path.join(PREPROCESSED_DIR, "edges_train.csv")
OUTPUT_PATH      = os.path.join(PREPROCESSED_DIR, "edges_train_smote.csv")

# ── Edge feature columns used for SMOTE ──────────────────────
# These are all numerical — SMOTE interpolates between them
SMOTE_FEAT_COLS = [
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

# ID columns to preserve but NOT use in SMOTE
ID_COLS = ["user_id", "merchant_id"]
LABEL_COL = "fraud_label"


def load_train_edges() -> pd.DataFrame:
    """Load train edges CSV. Expected size: ~18.8M rows."""
    print(f"\nLoading train edges: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"  Loaded : {len(df):,} rows")
    c = Counter(df[LABEL_COL])
    print(f"  Class 0 (legit) : {c[0]:,}")
    print(f"  Class 1 (fraud) : {c[1]:,}")
    print(f"  Fraud rate      : {c[1]/len(df)*100:.4f}%")
    return df


def apply_smote(
    df: pd.DataFrame,
    strategy: str = "borderline",
    target_ratio: float = 0.05,
    random_state: int = 42,
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Apply SMOTE to oversample fraud edges on the training set.

    Args:
        df           : Full training edge dataframe
        strategy     : 'standard' | 'borderline' | 'smotetomek'
                       - standard    : basic SMOTE (k-NN interpolation)
                       - borderline  : focuses on borderline fraud samples (recommended)
                       - smotetomek  : SMOTE + Tomek link cleaning
        target_ratio : Desired fraud / total ratio after resampling
                       0.05 = 5% fraud (from 0.13%)
        random_state : For reproducibility
        chunk_size   : SMOTE is applied on chunks of legit + all fraud
                       to avoid running out of RAM on 18M rows

    Returns:
        Resampled DataFrame with synthetic fraud edges appended.
    """
    print(f"\nApplying {strategy.upper()} SMOTE")
    print(f"  Target fraud ratio : {target_ratio*100:.1f}%")
    print(f"  Random state       : {random_state}")

    # ── Separate fraud and legit ──────────────────────────────
    fraud_df = df[df[LABEL_COL] == 1].copy()
    legit_df = df[df[LABEL_COL] == 0].copy()

    n_fraud = len(fraud_df)
    n_legit = len(legit_df)
    print(f"  Fraud edges        : {n_fraud:,}")
    print(f"  Legit edges        : {n_legit:,}")

    # ── Determine how many synthetic fraud samples needed ─────
    # target_ratio = n_fraud_final / (n_legit + n_fraud_final)
    # → n_fraud_final = target_ratio * n_legit / (1 - target_ratio)
    n_fraud_target = int(target_ratio * n_legit / (1 - target_ratio))
    n_synthetic    = max(0, n_fraud_target - n_fraud)
    sampling_ratio = n_fraud_target / n_legit   # ratio for imblearn

    print(f"  Fraud needed       : {n_fraud_target:,}")
    print(f"  Synthetic to gen   : {n_synthetic:,}")

    if n_synthetic == 0:
        print("  ⚠ Already at target ratio — no SMOTE needed")
        return df

    # ── SMOTE runs on chunks to manage RAM ───────────────────
    # Strategy: subsample legit, keep all fraud, run SMOTE on that
    # Then scale synthetic samples to match full dataset
    print(f"\n  Running SMOTE on feature space...")

    X_fraud = fraud_df[SMOTE_FEAT_COLS].fillna(0).values.astype(np.float32)
    y_fraud = fraud_df[LABEL_COL].values

    # Use a representative legit sample (4x fraud = fast, representative)
    legit_sample = legit_df.sample(
        n=min(len(legit_df), n_fraud * 100),
        random_state=random_state
    )
    X_legit = legit_sample[SMOTE_FEAT_COLS].fillna(0).values.astype(np.float32)
    y_legit = legit_sample[LABEL_COL].values

    X_combined = np.vstack([X_legit, X_fraud])
    y_combined = np.concatenate([y_legit, y_fraud])

    print(f"  SMOTE input size   : {len(X_combined):,} samples")
    print(f"  Class dist before  : {dict(Counter(y_combined))}")

    # ── Choose SMOTE variant ──────────────────────────────────
    k_neighbors = min(5, n_fraud - 1)   # safety for small minority class

    if strategy == "borderline":
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_ratio,
            k_neighbors=k_neighbors,
            random_state=random_state,
            kind="borderline-1"
        )
    elif strategy == "smotetomek":
        sampler = SMOTETomek(
            sampling_strategy=sampling_ratio,
            random_state=random_state,
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_ratio,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

    X_resampled, y_resampled = sampler.fit_resample(X_combined, y_combined)

    print(f"  Class dist after   : {dict(Counter(y_resampled))}")

    # ── Extract only the NEW synthetic fraud rows ─────────────
    n_orig = len(X_combined)
    X_synthetic = X_resampled[n_orig:]   # rows added by SMOTE
    y_synthetic = y_resampled[n_orig:]

    # All synthetic rows should be fraud
    assert all(y_synthetic == 1), "Unexpected non-fraud in synthetic rows"

    print(f"  Synthetic rows     : {len(X_synthetic):,}")

    # ── Build synthetic DataFrame with placeholder IDs ────────
    # Synthetic edges get user_id / merchant_id = -1 (flagged as synthetic)
    synthetic_df = pd.DataFrame(X_synthetic, columns=SMOTE_FEAT_COLS)
    synthetic_df[LABEL_COL]    = 1
    synthetic_df["user_id"]    = -1    # synthetic — no real user
    synthetic_df["merchant_id"] = -1  # synthetic — no real merchant
    synthetic_df["is_synthetic"] = 1

    # ── Tag originals as non-synthetic ────────────────────────
    df["is_synthetic"] = 0

    # ── Combine: original + synthetic fraud ───────────────────
    result = pd.concat([df, synthetic_df], ignore_index=True)

    # Final stats
    final_c = Counter(result[LABEL_COL])
    print(f"\n  ── Final dataset ──────────────────────────")
    print(f"  Total edges    : {len(result):,}")
    print(f"  Legit (0)      : {final_c[0]:,}")
    print(f"  Fraud (1)      : {final_c[1]:,}")
    print(f"  Fraud rate     : {final_c[1]/len(result)*100:.3f}%")
    print(f"  Synthetic flag : {result['is_synthetic'].sum():,} synthetic edges added")

    return result


def run_smote(strategy: str = "borderline", target_ratio: float = 0.05):
    """
    Full SMOTE pipeline:
      1. Load train edges
      2. Apply SMOTE (borderline by default)
      3. Save to edges_train_smote.csv
    """
    print("\n" + "="*60)
    print("  SMOTE OVERSAMPLING — Fraud Edge Augmentation")
    print("="*60)
    print(f"  Strategy     : {strategy}")
    print(f"  Target ratio : {target_ratio*100:.0f}% fraud")

    # Load
    train_df = load_train_edges()

    # Apply SMOTE
    balanced_df = apply_smote(
        train_df,
        strategy=strategy,
        target_ratio=target_ratio,
        random_state=42,
    )

    # Save
    balanced_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✅ Saved → {OUTPUT_PATH}")
    print(f"     Size: {len(balanced_df):,} rows")

    # Also save a summary report
    report_path = os.path.join(PREPROCESSED_DIR, "smote_report.txt")
    with open(report_path, "w") as f:
        c = Counter(balanced_df[LABEL_COL])
        f.write("SMOTE Oversampling Report\n")
        f.write("="*40 + "\n")
        f.write(f"Strategy         : {strategy}\n")
        f.write(f"Target ratio     : {target_ratio*100:.1f}%\n")
        f.write(f"Total edges      : {len(balanced_df):,}\n")
        f.write(f"Legit (0)        : {c[0]:,}\n")
        f.write(f"Fraud (1)        : {c[1]:,}\n")
        f.write(f"Achieved ratio   : {c[1]/len(balanced_df)*100:.3f}%\n")
        f.write(f"Synthetic edges  : {balanced_df['is_synthetic'].sum():,}\n")
    print(f"  ✅ Report   → {report_path}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SMOTE oversampling for fraud edges")
    parser.add_argument("--strategy",     default="borderline",
                        choices=["standard", "borderline", "smotetomek"],
                        help="SMOTE variant to use")
    parser.add_argument("--target_ratio", default=0.05, type=float,
                        help="Target fraud ratio after resampling (default: 0.05 = 5%%)")
    args = parser.parse_args()
    run_smote(strategy=args.strategy, target_ratio=args.target_ratio)
