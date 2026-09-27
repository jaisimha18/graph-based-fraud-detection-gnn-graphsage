# Progress Report — AML GraphSAGE Baseline Pipeline Fixes

**Project:** Graph-Based Fraud Detection using GNN (GraphSAGE)  
**Dataset:** IBM HI-Small Anti-Money Laundering (AML) Synthetic Dataset  
**Date:** August 4, 2026  

---

## Overview

This report documents four critical improvements made to the AML GraphSAGE baseline pipeline. These changes address methodological issues (data leakage, non-temporal splitting) and add new evaluation capabilities (pattern-level recall, cycle EDA) that strengthen the experimental rigor of the project.

---

## 1. Strict Chronological Train/Val/Test Split

### Problem

The original pipeline used **random stratified splitting** (80/10/10) via `sklearn.train_test_split()`. This is inappropriate for a temporal fraud detection task because:

- It allows the model to train on **future transactions** and be tested on **past ones** — an unrealistic scenario in production.
- It breaks the temporal dependency structure inherent in money laundering patterns, where transactions in a laundering chain occur sequentially over days.
- Published AML literature (e.g., Weber et al., 2019 — the IBM AML dataset paper) recommends temporal evaluation.

### Solution

Implemented a **strict chronological split** based on the day-of-month in September 2022:

| Split | Days | Date Range | Purpose |
|-------|------|------------|---------|
| **Train** | 1 – 8 | Sept 1 – Sept 8 | Model learns from historical patterns |
| **Val** | 9 | Sept 9 | Threshold tuning & early stopping |
| **Test** | 10 – 18 | Sept 10 – Sept 18 | Final held-out evaluation on future data |

### Files Changed

- **`src/aml_pipeline/aml_config.py`** — Replaced `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` with `TRAIN_END_DAY = 8`, `VAL_DAY = 9`, `TEST_START_DAY = 10`.
- **`src/aml_pipeline/preprocess.py`** — Rewrote `preprocess_transactions()` to assign splits by parsing the `Timestamp` column and routing edges by day instead of random shuffling.

### Rationale

This split simulates a realistic deployment scenario: the model is trained on the first 8 days and must detect laundering on days it has never seen. This is the standard protocol for temporal fraud detection benchmarks.

---

## 2. Data Leakage Fix — Training-Only Node Aggregates

### Problem

The original pipeline had **three sources of data leakage**:

| Leakage Source | Severity | Explanation |
|---|---|---|
| Account aggregates computed from **all** transactions | **High** | The model's node features included statistics (transaction counts, average amounts) derived from val/test transactions — information the model should not have access to at training time. |
| `StandardScaler` fit on **all** accounts | **Medium** | The scaler's mean/std parameters were influenced by val/test data, subtly leaking distributional information. |
| `laundering_rate` used as a node feature | **Critical** | This feature directly encodes the target label (`Is Laundering`). Even if restricted to training data, it gives the model a pre-computed fraud signal — the model learns to threshold this feature rather than learn from graph structure. |

### Solution

Three targeted fixes:

#### Fix A — Training-Only Aggregates
`compute_account_aggregates()` now filters transactions to `day <= TRAIN_END_DAY` before computing any statistics. Only Sept 1–8 transactions contribute to node features.

```python
# BEFORE: processed ALL transactions
for _, row in chunk.iterrows():  # no filtering

# AFTER: only training-period transactions
train_mask = day <= TRAIN_END_DAY
train_chunk = chunk[train_mask]
for _, row in train_chunk.iterrows():
```

#### Fix B — Training-Only Scaler
`StandardScaler` is now fit exclusively on accounts that appear in the training period:

```python
# BEFORE: fit on all accounts
scaler.fit_transform(merged[feat_cols])

# AFTER: fit on training accounts, transform all
scaler.fit(merged.loc[train_mask, feat_cols])
feat_matrix = scaler.transform(merged[feat_cols])
```

#### Fix C — Removed `laundering_rate`
The `laundering_rate` feature has been completely removed from the node feature vector. The final feature set is 8 dimensions:

| Feature | Description | Type |
|---------|-------------|------|
| `bank_id` | Integer bank identifier | Static |
| `entity_type_enc` | LabelEncoded entity type (6 classes) | Static |
| `is_crypto_bank` | Binary flag for crypto banks | Static |
| `out_txn_count` | Outgoing transaction count (train only) | Aggregate |
| `in_txn_count` | Incoming transaction count (train only) | Aggregate |
| `out_avg_amount` | Average outgoing amount (train only) | Aggregate |
| `in_avg_amount` | Average incoming amount (train only) | Aggregate |
| `unique_counterparties` | Unique transacting partners (train only) | Aggregate |

### Files Changed

- **`src/aml_pipeline/preprocess.py`** — All three fixes applied in `compute_account_aggregates()` and `build_account_features()`.

---

## 3. Pattern-Level Recall Metric

### Problem

The original evaluation pipeline only reported **edge-level metrics** (ROC-AUC, PR-AUC, F1). However, in real-world AML, the relevant question is:

> *"How many complete laundering schemes did the model detect?"*

A model could flag many individual suspicious edges but still miss entire laundering patterns. **Pattern-level recall** is the standard metric for this in the AML literature.

### Solution

Added two new functions to `src/training/evaluate_aml.py`:

#### `parse_patterns(patterns_path)`
Parses `HI-Small_Patterns.txt` into structured pattern objects. Each pattern contains:
- Pattern type (CYCLE, FAN-OUT, FAN-IN, SCATTER-GATHER, GATHER-SCATTER, STACK, BIPARTITE, RANDOM)
- Description/parameters
- List of transactions with (from_account, to_account) pairs

#### `compute_pattern_recall(test_edge_index, test_bin_preds, data, threshold, out_dir)`
For each of the 370 laundering patterns:
1. Identifies which pattern transactions fall in the **test set**
2. Checks if **at least one** test-set edge was predicted positive
3. If yes → pattern is "detected"

Reports:
- **Per-type recall** (e.g., CYCLE: 80%, FAN-OUT: 65%)
- **Overall pattern recall** across all types
- Results saved to `evaluation_report.json`

### Detection Criterion

A laundering pattern is considered **detected** if ≥1 of its test-set edges was predicted as laundering (score > threshold). This is a lenient but standard criterion — it answers "did the model raise any alarm for this scheme?"

### Sample Output Format

```
  Pattern Type         In Test   Detected     Recall
  ----------------------------------------------------
  BIPARTITE                 35         28      80.0%
  CYCLE                     40         32      80.0%
  FAN-IN                    30         24      80.0%
  FAN-OUT                   38         30      78.9%
  ...
  ----------------------------------------------------
  OVERALL                  280        220      78.6%
```

### Files Changed

- **`src/training/evaluate_aml.py`** — Added `parse_patterns()`, `compute_pattern_recall()`, integrated into `evaluate_aml()`, and saved results to the JSON report.

---

## 4. Cycle Pattern EDA Script (`eda_cycles.py`)

### Purpose

Created a new exploratory data analysis script specifically for **cycle laundering patterns** — the most structurally interesting pattern type in the dataset. Cycles represent closed loops where money eventually returns to the originator through intermediaries.

### Statistics Computed

The script computes 9 categories of statistics:

| # | Category | Key Findings |
|---|----------|-------------|
| 1 | **Cycle Length** | 2–12 hops per cycle, mean 5.3, median 4.0 |
| 2 | **Cycle Closure** | All 54 cycles are closed (first sender == last receiver) |
| 3 | **Temporal Span** | Mean 2.75 days per cycle, 77.8% span 2–5 days |
| 4 | **Accounts per Cycle** | 2–12 unique accounts, mean 5.3 |
| 5 | **Account Participation** | 271 unique accounts across all cycles; top accounts appear 8 times |
| 6 | **Amount Statistics** | Total ~1B moved; per-hop median ~12K |
| 7 | **Currency Usage** | USD (37.6%), EUR (25.8%), SAR (14.6%) dominate |
| 8 | **Cross-Bank Hops** | 97.2% of hops cross bank boundaries |
| 9 | **Described vs Actual** | All 54 cycles match their stated "Max N hops" |

### Key Insights for the Model

- **High cross-bank rate (97.2%)** — The `is_cross_bank` edge feature should be a strong signal.
- **Multi-day spans** — Cycles unfold over 2–5 days, validating the need for temporal awareness.
- **Variable lengths** — Cycles range from 2 to 12 hops; the model needs multi-hop neighborhood aggregation (GraphSAGE's strength).
- **Closed loops** — All cycles close, which means cycle detection algorithms could complement GNN predictions.

### Usage

```bash
python -m src.aml_pipeline.eda_cycles
```

### Files Created

- **`src/aml_pipeline/eda_cycles.py`** — New standalone script (335 lines).

---

## 5. Phase 3: Cycle Awareness Architecture & Scientific Ablation Study

Following Dr. Anbazhagan M's architectural review, Phase 3 transitioned the AML GraphSAGE system from a vanilla neighborhood aggregator to a cycle-aware topological network designed to completely dismantle laundering rings.

### Step 1: Evaluation Rigor Enhancement (Ring Completeness)
Standard binary edge recall and alert recall (catching $\ge 1$ transaction in a pattern) mask a critical operational failure: **partial evasion**. If an AML ring contains 10 hops and a bank only freezes 1 hop, the syndicate reroutes funds through the remaining 9 hops.
We implemented a 4-level evaluation hierarchy in `src/training/evaluate_aml.py`:
1. **Alert Recall ($\ge 1$ edge)**: Traditional operational alert metric.
2. **Majority Ring Recall ($\ge 50\%$)**: Majority of ring disrupted.
3. **Complete Ring Recall ($100\%$)**: Every edge in the laundering pattern detected (full ring dismantled).
4. **Mean Edge Coverage**: Average percentage of ring edges caught.

On Vanilla GraphSAGE, this revealed a **23.1% evasion gap**: while alert recall was 98.7%, full ring recall was only 76.9%.

---

### Step 2: Component A — $N$-Step Random Walk Cycle Feature Extractor
Implemented in `src/aml_pipeline/cycle_features.py`:
- Formulates $k$-step return probabilities via matrix powers $P^k$ where $P = D_{out}^{-1} A$ is the row-stochastic transition matrix.
- For directed edge $(u \to v)$, return probability is $R_k(u, v) = [P^k]_{v, u}$ for $k \in \{2, 3, 4, 5\}$.
- Computes discounted cumulative return $R_{disc} = R_2 + 0.8 R_3 + 0.64 R_4 + 0.512 R_5$ and binary existence indicators.
- **Strict Zero-Leakage Guarantee**: Transition matrix powers for training edges use *only* Days 1–8 topology. Validation and test edge features use historical graphs up to Days 9 and 18 respectively.
- Augmented PyG graph saved to `outputs/aml/graph/aml_graph_cycle.pt` (edge feature dimension expanded from 14 to 22).

---

### Step 3: Component B — Cycle-Aware Attention Layer (`CycleAwareSAGEConv`)
Implemented in `src/models/cycle_graphsage_model.py`:
- Standard GraphSAGE aggregates neighbor embeddings using unweighted mean aggregation, diluting cyclical signals.
- `CycleAwareSAGEConv` computes an edge-conditioned structural attention weight:
  $$\alpha_{j, i} = \text{Softmax}_j \left( \text{LeakyReLU}(W_\alpha [h_i \,\|\, h_j \,\|\, e_{j \to i}^{cycle}]) \right)$$
- Prioritizes message passing along high-return cyclical transaction paths before applying the GraphSAGE aggregation and target edge MLP classification.

---

### Step 4: Scalable Dynamic Negative Sampling Pipeline
Implemented in `src/training/train_cycle_aml.py`:
- Training on the complete un-sampled 4.06M training edges took ~20 minutes per epoch.
- Implemented dynamic per-epoch negative sampling (25x negatives per epoch against all 4,142 training fraud edges).
- Reduces epoch time to **~18 seconds** on Apple Silicon MPS while preserving 100% of fraud patterns and scanning new negative edges on every epoch.
- Validation and testing remain completely un-sampled across all holdout transactions.

---

### Empirical Ablation Study Results (Held-Out Test Set: Days 10–18)

Evaluated via `python predict.py --ablation`:

| Configuration | ROC-AUC | PR-AUC | Fraud Recall | Alert Recall ($\ge 1$) | Majority Ring ($\ge 50\%$) | Complete Ring ($100\%$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Vanilla GraphSAGE** | 0.9852 | 0.3235 | 87.4% | 98.7% | 97.8% | 76.9% |
| **2. + RW Cycle Features** | 0.9704 | 0.1113 | 75.3% | 96.1% | 92.6% | 63.8% |
| **3. + Cycle Attention (Full)** | **0.9889** | **0.5759** | **87.7%** | **98.7%** | **98.2%** | **85.2%** |

### Key Findings & Insights:
1. **Dramatic PR-AUC Leap (+78.0% relative gain)**: In extreme transaction imbalance ($<0.1\%$ fraud), PR-AUC jumped from 0.3235 to **0.5759**, demonstrating superior false-positive rejection.
2. **Complete Ring Dismantling (+8.3% absolute gain)**: Full ring dismantling surged from 76.9% to **85.2%** (with Cycle-type patterns reaching 85.7% and Fan-Out reaching 92.9%), slashing missed laundering rings by over 36%.
3. **The Necessity of Attention Aggregation**: Feeding cycle metrics purely as passive edge attributes (Ablation 2) failed without attention because mean aggregation dilutes localized cycle signals. Cycle-Aware Attention (Ablation 3) dynamically weights neighbor aggregation, unlocking substantial structural gains.

