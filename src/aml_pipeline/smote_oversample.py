# ============================================================
# src/aml_pipeline/smote_oversample.py
#
# Edge-level SMOTE for AML (Anti-Money Laundering) graph
#
# Why SMOTE here:
#   - Laundering rate in train ≈ 0.10% (highly imbalanced)
#   - SMOTE synthesizes new laundering EDGES by interpolating
#     between real laundering edges in feature space
#
# What is NOT done:
#   - We do NOT apply SMOTE to val or test (would cause leakage)
#   - We do NOT apply it to node features (only edge features)
#   - SMOTE is applied AFTER preprocessing, BEFORE graph build
#
# Output:
#   outputs/aml/preprocessed/edges_train_smote.csv
#
# Run: python -m src.aml_pipeline.smote_oversample
# ============================================================

import os, sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aml_pipeline.aml_config import AML_OUTPUT_DIR, PREPROCESSED_DIR

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(PREPROCESSED_DIR, "edges_train.csv")
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, "edges_train_smote.csv")

# ── Edge feature columns used for SMOTE ──────────────────────
# These are all numerical — SMOTE interpolates between them
SMOTE_FEAT_COLS = [
    "amount_received_log", "amount_paid_log", "amount_diff_log",
    "is_cross_bank", "is_self_transfer", "is_cross_currency",
    "receiving_currency_enc", "payment_format_enc",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_night", "is_weekend",
]

# ID columns to preserve but NOT use in SMOTE
ID_COLS = ["from_account", "to_account"]
LABEL_COL = "laundering_label"


def load_train_edges() -> pd.DataFrame:
    """Load train edges CSV."""
    print(f"\nLoading train edges: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"  Loaded : {len(df):,} rows")
    c = Counter(df[LABEL_COL])
    print(f"  Class 0 (legit)      : {c[0]:,}")
    print(f"  Class 1 (laundering) : {c[1]:,}")
    print(f"  Laundering rate      : {c[1]/len(df)*100:.4f}%")
    return df


def apply_smote(
    df: pd.DataFrame,
    strategy: str = "borderline",
    target_ratio: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply SMOTE to oversample laundering edges on the training set.

    Args:
        df           : Full training edge dataframe
        strategy     : 'standard' | 'borderline' | 'smotetomek'
        target_ratio : Desired laundering / total ratio after resampling
                       0.05 = 5% laundering (from ~0.10%)
        random_state : For reproducibility

    Returns:
        Resampled DataFrame with synthetic laundering edges appended.
    """
    print(f"\nApplying {strategy.upper()} SMOTE")
    print(f"  Target laundering ratio : {target_ratio*100:.1f}%")
    print(f"  Random state            : {random_state}")

    # ── Separate laundering and legit ─────────────────────────
    launder_df = df[df[LABEL_COL] == 1].copy()
    legit_df   = df[df[LABEL_COL] == 0].copy()

    n_launder = len(launder_df)
    n_legit   = len(legit_df)
    print(f"  Laundering edges : {n_launder:,}")
    print(f"  Legit edges      : {n_legit:,}")

    # ── Determine how many synthetic samples needed ───────────
    n_launder_target = int(target_ratio * n_legit / (1 - target_ratio))
    n_synthetic      = max(0, n_launder_target - n_launder)
    sampling_ratio   = n_launder_target / n_legit

    print(f"  Laundering needed  : {n_launder_target:,}")
    print(f"  Synthetic to gen   : {n_synthetic:,}")

    if n_synthetic == 0:
        print("  ⚠ Already at target ratio — no SMOTE needed")
        return df

    # ── SMOTE on feature space ────────────────────────────────
    print(f"\n  Running SMOTE on feature space...")

    X_launder = launder_df[SMOTE_FEAT_COLS].fillna(0).values.astype(np.float32)
    y_launder = launder_df[LABEL_COL].values

    # Use a representative legit sample (100x laundering = fast, representative)
    legit_sample = legit_df.sample(
        n=min(len(legit_df), n_launder * 100),
        random_state=random_state,
    )
    X_legit = legit_sample[SMOTE_FEAT_COLS].fillna(0).values.astype(np.float32)
    y_legit = legit_sample[LABEL_COL].values

    X_combined = np.vstack([X_legit, X_launder])
    y_combined = np.concatenate([y_legit, y_launder])

    print(f"  SMOTE input size   : {len(X_combined):,} samples")
    print(f"  Class dist before  : {dict(Counter(y_combined))}")

    # ── Choose SMOTE variant ──────────────────────────────────
    k_neighbors = min(5, n_launder - 1)

    if strategy == "borderline":
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_ratio,
            k_neighbors=k_neighbors,
            random_state=random_state,
            kind="borderline-1",
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

    # ── Extract only the NEW synthetic laundering rows ────────
    n_orig = len(X_combined)
    X_synthetic = X_resampled[n_orig:]
    y_synthetic = y_resampled[n_orig:]

    assert all(y_synthetic == 1), "Unexpected non-laundering in synthetic rows"

    print(f"  Synthetic rows     : {len(X_synthetic):,}")

    # ── Build synthetic DataFrame with placeholder IDs ────────
    synthetic_df = pd.DataFrame(X_synthetic, columns=SMOTE_FEAT_COLS)
    synthetic_df[LABEL_COL]       = 1
    synthetic_df["from_account"]  = "SYNTHETIC"
    synthetic_df["to_account"]    = "SYNTHETIC"
    synthetic_df["is_synthetic"]  = 1

    # ── Tag originals as non-synthetic ────────────────────────
    df["is_synthetic"] = 0

    # ── Combine: original + synthetic laundering ──────────────
    result = pd.concat([df, synthetic_df], ignore_index=True)

    # Final stats
    final_c = Counter(result[LABEL_COL])
    print(f"\n  ── Final dataset ──────────────────────────")
    print(f"  Total edges        : {len(result):,}")
    print(f"  Legit (0)          : {final_c[0]:,}")
    print(f"  Laundering (1)     : {final_c[1]:,}")
    print(f"  Laundering rate    : {final_c[1]/len(result)*100:.3f}%")
    print(f"  Synthetic edges    : {result['is_synthetic'].sum():,}")

    return result


def run_smote(strategy: str = "borderline", target_ratio: float = 0.05):
    """
    Full SMOTE pipeline:
      1. Load train edges
      2. Apply SMOTE
      3. Save to edges_train_smote.csv
    """
    print("\n" + "=" * 60)
    print("  SMOTE OVERSAMPLING — AML Laundering Edge Augmentation")
    print("=" * 60)
    print(f"  Strategy     : {strategy}")
    print(f"  Target ratio : {target_ratio*100:.0f}% laundering")

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

    # Save a summary report
    report_path = os.path.join(PREPROCESSED_DIR, "smote_report.txt")
    with open(report_path, "w") as f:
        c = Counter(balanced_df[LABEL_COL])
        f.write("AML SMOTE Oversampling Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Strategy         : {strategy}\n")
        f.write(f"Target ratio     : {target_ratio*100:.1f}%\n")
        f.write(f"Total edges      : {len(balanced_df):,}\n")
        f.write(f"Legit (0)        : {c[0]:,}\n")
        f.write(f"Laundering (1)   : {c[1]:,}\n")
        f.write(f"Achieved ratio   : {c[1]/len(balanced_df)*100:.3f}%\n")
        f.write(f"Synthetic edges  : {balanced_df['is_synthetic'].sum():,}\n")
    print(f"  ✅ Report   → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SMOTE oversampling for AML laundering edges")
    parser.add_argument("--strategy", default="borderline",
                        choices=["standard", "borderline", "smotetomek"],
                        help="SMOTE variant to use")
    parser.add_argument("--target_ratio", default=0.05, type=float,
                        help="Target laundering ratio after resampling (default: 0.05 = 5%%)")
    args = parser.parse_args()
    run_smote(strategy=args.strategy, target_ratio=args.target_ratio)
