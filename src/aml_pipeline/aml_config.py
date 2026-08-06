# ============================================================
# src/aml_pipeline/aml_config.py
#
# Configuration for AML (Anti-Money Laundering) pipeline
# IBM HI-Small dataset
# ============================================================
import os

ROOT_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AML_DATA_DIR   = os.path.join(ROOT_DIR, "AML_DATASET")
AML_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "aml")

# ── Input files ──────────────────────────────────────────────
TRANSACTIONS_CSV = os.path.join(AML_DATA_DIR, "HI-Small_Trans.csv")
ACCOUNTS_CSV     = os.path.join(AML_DATA_DIR, "HI-Small_accounts.csv")
PATTERNS_TXT     = os.path.join(AML_DATA_DIR, "HI-Small_Patterns.txt")

# ── Output directories ──────────────────────────────────────
PREPROCESSED_DIR = os.path.join(AML_OUTPUT_DIR, "preprocessed")
GRAPH_DIR        = os.path.join(AML_OUTPUT_DIR, "graph")

# ── Processing ───────────────────────────────────────────────
CHUNK_SIZE = 500_000

# ── Split strategy ───────────────────────────────────────────
# Strict chronological split on September 2022:
#   Train : Sept  1 –  8  (days 1-8)
#   Val   : Sept  9       (day 9)
#   Test  : Sept 10 – 18  (days 10-18)
TRAIN_END_DAY = 8     # last day (inclusive) for training
VAL_DAY       = 9     # single validation day
TEST_START_DAY = 10   # first day (inclusive) for testing

# ── Feature dimensions ──────────────────────────────────────
ACCOUNT_FEAT_DIM = 8    # account node features
EDGE_FEAT_DIM    = 14   # transaction edge features

# ── Model (same hyperparams as credit card pipeline) ────────
HIDDEN_DIM       = 128
OUTPUT_DIM       = 64
SAGE_AGGR        = "mean"
NUM_SAGE_LAYERS  = 2
DROPOUT          = 0.3

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE       = 1024
NUM_NEIGHBORS    = [15, 10]
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
EPOCHS           = 2
EARLY_STOP_PAT   = 10
LR_SCHED_PAT     = 5

# ── Evaluation ───────────────────────────────────────────────
THRESHOLD_SWEEP  = [i / 100 for i in range(5, 96, 5)]
SEED             = 42
