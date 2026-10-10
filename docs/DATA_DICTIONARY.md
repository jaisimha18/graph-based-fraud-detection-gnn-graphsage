# Data Dictionary

> Schema reference for the IBM HI-Small AML Dataset and all derived features.

---

## Source Dataset

### `HI-Small_Trans.csv` — Transaction Records

The IBM HI-Small synthetic dataset contains ~4.92 million financial transactions simulating real-world anti-money laundering scenarios.

| Column | Type | Description | Example |
|---|---|---|---|
| `Timestamp` | datetime | Transaction timestamp | `2022/09/01 08:30` |
| `From Bank` | string | Sender's bank ID | `BANK_A` |
| `Account` | string | Sender's account ID | `8000ECA930` |
| `To Bank` | string | Recipient's bank ID | `BANK_B` |
| `Account.1` | string | Recipient's account ID | `8000F5E710` |
| `Amount Received` | float | Amount received (recipient currency) | `1500.50` |
| `Amount Paid` | float | Amount paid (sender currency) | `1500.50` |
| `Receiving Currency` | string | Currency of received amount | `US Dollar` |
| `Payment Currency` | string | Currency of payment | `Euro` |
| `Payment Format` | string | Payment method | `Wire` |
| `Is Laundering` | int (0/1) | Ground truth label | `0` |

### `HI-Small_accounts.csv` — Account Metadata

| Column | Type | Description |
|---|---|---|
| `account_id` | string | Unique account identifier |
| Additional columns | varies | Bank and account attributes |

---

## Temporal Split Strategy

| Split | Days | Purpose | Leakage Risk |
|---|---|---|---|
| **Train** | Days 1–8 (Sept 1–8) | Model training | None (all features fit here) |
| **Validation** | Day 9 (Sept 9) | Hyperparameter tuning | None (no future data used) |
| **Test** | Days 10–18 (Sept 10–18) | Final evaluation | None (strict holdout) |

---

## Engineered Features

### Node Features (8 dimensions)

| # | Feature | Source | Description |
|---|---|---|---|
| 1–8 | Account aggregate features | `preprocess.py` | Per-account statistics computed *only* from training days |

### Edge Features — Base (14 dimensions)

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `amount_received_log` | float | log1p(Amount Received) |
| 2 | `amount_paid_log` | float | log1p(Amount Paid) |
| 3 | `amount_diff_log` | float | Difference of log amounts |
| 4 | `is_cross_bank` | binary | 1 if From Bank ≠ To Bank |
| 5 | `is_self_transfer` | binary | 1 if Account == Account.1 |
| 6 | `is_cross_currency` | binary | 1 if currencies differ |
| 7 | `rec_curr_enc` | categorical | Receiving currency encoded (0–14) |
| 8 | `fmt_enc` | categorical | Payment format encoded (0–6) |
| 9 | `hour_sin` | float | sin(2π × hour / 24) |
| 10 | `hour_cos` | float | cos(2π × hour / 24) |
| 11 | `dow_sin` | float | sin(2π × day_of_week / 7) |
| 12 | `dow_cos` | float | cos(2π × day_of_week / 7) |
| 13 | `is_night` | binary | 1 if hour ∈ [22:00, 06:00) |
| 14 | `is_weekend` | binary | 1 if Saturday or Sunday |

### Edge Features — Cycle (8 dimensions, appended for cycle models)

| # | Feature | Type | Description |
|---|---|---|---|
| 15 | `return_prob_2hop` | float | P(return to source in 2 hops) |
| 16 | `return_prob_3hop` | float | P(return to source in 3 hops) |
| 17 | `return_prob_4hop` | float | P(return to source in 4 hops) |
| 18 | `return_prob_5hop` | float | P(return to source in 5 hops) |
| 19 | `discounted_cumulative_return` | float | Weighted sum of return probs |
| 20 | `has_any_cycle` | binary | 1 if any return prob > 0 |
| 21 | `cycle_intensity` | float | max(return probabilities) |
| 22 | `neighbor_cycle_density` | float | Avg cycle participation of neighbors |

---

## Currency Encoding

| Index | Currency |
|---|---|
| 0 | Australian Dollar |
| 1 | Bitcoin |
| 2 | Brazil Real |
| 3 | Canadian Dollar |
| 4 | Euro |
| 5 | Mexican Peso |
| 6 | Ruble |
| 7 | Rupee |
| 8 | Saudi Riyal |
| 9 | Shekel |
| 10 | Swiss Franc |
| 11 | UK Pound |
| 12 | US Dollar (default) |
| 13 | Yen |
| 14 | Yuan |

## Payment Format Encoding

| Index | Format |
|---|---|
| 0 | ACH |
| 1 | Bitcoin |
| 2 | Cash |
| 3 | Cheque (default) |
| 4 | Credit Card |
| 5 | Reinvestment |
| 6 | Wire |

---

## Output Artifacts

| File | Location | Size | Description |
|---|---|---|---|
| `aml_graph.pt` | `outputs/aml/graph/` | ~782 MB | Base PyG graph object |
| `aml_graph_cycle.pt` | `outputs/aml/graph/` | ~1.1 GB | Graph with cycle features |
| `best_model.pt` | `outputs/aml/model_results/` | ~2 MB | Trained baseline weights |
| `best_model_cycle_attn.pt` | `outputs/aml/model_results/` | ~3 MB | Trained cycle model weights |
| `evaluation_report.json` | `outputs/aml/model_results/` | ~2 KB | Baseline evaluation metrics |
| `evaluation_report_cycle_attn.json` | `outputs/aml/model_results/` | ~2 KB | Cycle model metrics |

---

## Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | ~4,916,000 | ~99.9% |
| Fraudulent (1) | ~4,600 | ~0.1% |

> Extreme class imbalance makes **PR-AUC** a more meaningful metric than ROC-AUC.
