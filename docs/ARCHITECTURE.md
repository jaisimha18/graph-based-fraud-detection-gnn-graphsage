# System Architecture

> Detailed technical architecture of the Graph-Based AML Fraud Detection system.

---

## Overview

The system is a three-tier architecture for detecting money laundering patterns in financial transaction networks using Graph Neural Networks (GNNs).

```
┌────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (React)                       │
│  TransactionAnalysis │ ResultsDashboard │ Methodology          │
│  Force-graph viz     │ Metrics charts   │ Pipeline overview    │
└──────────────────────────┬─────────────────────────────────────┘
                           │ HTTP (REST API)
┌──────────────────────────┴─────────────────────────────────────┐
│                    FastAPI Backend                              │
│  POST /api/predict  │  GET /api/ablation  │  GET /api/model    │
│  Real-time GNN inference on uploaded CSV transactions          │
└──────────────────────────┬─────────────────────────────────────┘
                           │ PyTorch / PyG
┌──────────────────────────┴─────────────────────────────────────┐
│                  ML Pipeline & Models                          │
│  ┌──────────┐  ┌───────────┐  ┌───────────────────┐           │
│  │Preprocess│→ │Build Graph│→ │ Train & Evaluate  │           │
│  │(temporal) │  │  (PyG)    │  │ (GraphSAGE/Cycle) │           │
│  └──────────┘  └───────────┘  └───────────────────┘           │
│  Cycle Features ─── Random Walk N-hop Return Probabilities     │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline

### 1. Preprocessing (`src/aml_pipeline/preprocess.py`)

```
Raw CSV (HI-Small_Trans.csv)
    │
    ├─ Parse timestamps, extract temporal features
    ├─ Cyclical encoding (hour, day-of-week)
    ├─ Log-transform amounts
    ├─ Categorical encoding (currencies, payment formats)
    ├─ Binary indicators (cross-bank, self-transfer, cross-currency)
    │
    ├─ Chronological Split (ZERO temporal leakage):
    │   ├─ Train: Days 1–8
    │   ├─ Validation: Day 9
    │   └─ Test: Days 10–18
    │
    └─ Output: preprocessed CSVs + account features
```

### 2. Graph Construction (`src/aml_pipeline/build_graph.py`)

```
Preprocessed Data
    │
    ├─ Accounts → Nodes (8-dim feature vectors)
    ├─ Transactions → Edges (14-dim feature vectors)
    ├─ Labels → Edge labels (Is Laundering: 0/1)
    │
    └─ Output: PyG Data object (aml_graph.pt)
        ├─ x: [N_nodes, 8] node features
        ├─ edge_index: [2, N_edges] adjacency
        ├─ edge_attr: [N_edges, 14] edge features
        └─ edge_label: [N_edges] binary labels
```

### 3. Cycle Feature Extraction (`src/aml_pipeline/cycle_features.py`)

```
Graph Adjacency (edge_index)
    │
    ├─ Build sparse adjacency matrix A
    ├─ Compute A², A³, A⁴, A⁵ (sparse matrix powers)
    │
    ├─ For each edge (u→v), extract:
    │   ├─ 2-hop return probability: A²[u,u]
    │   ├─ 3-hop return probability: A³[u,u]
    │   ├─ 4-hop return probability: A⁴[u,u]
    │   ├─ 5-hop return probability: A⁵[u,u]
    │   ├─ Discounted cumulative return
    │   ├─ Has any cycle indicator (binary)
    │   ├─ Cycle intensity (max return prob)
    │   └─ Neighbor cycle density
    │
    └─ Output: [N_edges, 8] cycle features
```

### 4. Model Architecture

#### Vanilla GraphSAGE (`src/models/graphsage_model.py`)

```
Input: Node features [N, 8], Edge features [E, 14]
    │
    ├─ SAGEConv Layer 1 (8 → 128, mean aggregation)
    ├─ BatchNorm → ReLU → Dropout(0.3)
    ├─ SAGEConv Layer 2 (128 → 64, mean aggregation)
    ├─ BatchNorm → ReLU → Dropout(0.3)
    │
    ├─ Edge Predictor MLP:
    │   ├─ Concat [src_embed, dst_embed, edge_features]
    │   ├─ Linear(142 → 64) → ReLU → Dropout
    │   └─ Linear(64 → 1) → Sigmoid
    │
    └─ Output: Fraud probability per edge
```

#### Cycle-Aware Attention GraphSAGE (`src/models/cycle_graphsage_model.py`)

```
Input: Node features [N, 8], Edge features [E, 22] (14 base + 8 cycle)
    │
    ├─ CycleAwareSAGEConv Layer 1:
    │   ├─ Standard SAGE message passing
    │   ├─ Cycle attention: α = σ(W_attn · cycle_features)
    │   └─ Re-weight neighbor messages by α
    ├─ BatchNorm → ReLU → Dropout(0.3)
    │
    ├─ CycleAwareSAGEConv Layer 2:
    │   └─ (same architecture)
    ├─ BatchNorm → ReLU → Dropout(0.3)
    │
    ├─ Edge Predictor MLP:
    │   ├─ Concat [src_embed, dst_embed, edge_features]
    │   └─ Linear → ReLU → Dropout → Linear → Sigmoid
    │
    └─ Output: Fraud probability per edge
```

---

## Training Strategy

| Aspect | Details |
|---|---|
| **Loss** | Binary Cross-Entropy with class weights |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=5) |
| **Early Stopping** | Patience=10 on validation PR-AUC |
| **Negative Sampling** | Dynamic ratio 25:1 (neg:pos) for cycle model |
| **Batching** | Mini-batch with NeighborLoader (sizes=[15, 10]) |
| **Epochs** | 50 |

---

## Evaluation Framework

### 4-Level Ring Completeness

The IBM HI-Small dataset tags laundering transactions with ring IDs. We evaluate at four levels of granularity:

1. **Alert Recall (≥1 edge):** At least one edge in the ring flagged
2. **Majority Ring Recall (≥50%):** Majority of ring edges flagged
3. **Complete Ring Recall (100%):** All edges in the ring flagged
4. **Mean Ring Edge Coverage:** Average % of edges flagged per ring

---

## API Architecture

The FastAPI backend (`src/api/main.py`) loads the best available model at startup and serves real-time inference:

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Upload CSV → returns graph JSON with fraud predictions |
| `/api/ablation` | GET | Returns evaluation reports for all model variants |
| `/api/model_info` | GET | Returns active model type, threshold, and device |

---

## Frontend Architecture

React SPA (Vite) with five main components:

| Component | Purpose |
|---|---|
| `Hero` | Landing page with project overview |
| `TransactionAnalysis` | CSV upload + force-directed graph visualization |
| `ResultsDashboard` | Ablation metrics charts |
| `Methodology` | Pipeline overview with step descriptions |
| `Insights` | Key findings and contributions |
