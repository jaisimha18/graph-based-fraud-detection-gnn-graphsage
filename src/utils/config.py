# ============================================================
# src/utils/config.py — All hyperparameters and paths
# ============================================================
import os

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")

TRANSACTIONS_CSV = os.path.join(DATA_DIR, "transactions.csv")
USER_DETAILS_CSV = os.path.join(DATA_DIR, "user_details.csv")
CARDS_CSV        = os.path.join(DATA_DIR, "cards.csv")
GRAPH_SAVE_PATH  = os.path.join(OUTPUT_DIR, "hetero_graph.pt")

# ── Graph splits (temporal) ──────────────────────────────────
CHUNK_SIZE    = 500_000
TRAIN_YEAR_MAX = 2016
VAL_YEAR       = 2017
TEST_YEAR_MIN  = 2018

# ── Feature dims ────────────────────────────────────────────
USER_FEAT_DIM     = 10
MERCHANT_FEAT_DIM = 6
EDGE_FEAT_DIM     = 23

# ── Model ────────────────────────────────────────────────────
HIDDEN_DIM      = 128
OUTPUT_DIM      = 64
SAGE_AGGR       = "mean"
NUM_SAGE_LAYERS = 2
DROPOUT         = 0.3

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE    = 1024
NUM_NEIGHBORS = [15, 10]
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
EARLY_STOP_PAT = 10
LR_SCHED_PAT   = 5

# ── Evaluation ───────────────────────────────────────────────
THRESHOLD_SWEEP = [i / 100 for i in range(5, 96, 5)]
SEED = 42
