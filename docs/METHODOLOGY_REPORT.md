# Project Methodology Report: AML Graph-Based Fraud Detection

This document details the exact technical implementations, tools, and methodologies used in the project so far (Phases 1 & 2), transitioning from raw tabular data to an inductive GraphSAGE edge-classification model.

---

## 1. Data Preprocessing (`src/aml_pipeline/preprocess.py`)

**Objective:** Clean the raw IBM HI-Small dataset, engineer features, and prevent data leakage across time.

**What we did & How we did it:**
*   **Temporal Splitting:** Fraud detection must simulate real-world streaming. Instead of randomly shuffling data, we implemented a **strict chronological split**.
    *   **Train:** Days 1–8 (Historical data)
    *   **Validation:** Day 9 (Tuning)
    *   **Test:** Days 10–18 (Future predictions)
*   **Node Feature Engineering:** Accounts (nodes) originally only had static data (like Bank ID or Entity Type). We engineered dynamic behavior features (e.g., total amount sent, total amount received, transaction frequency).
*   **Strict Leakage Prevention:** To prevent the model from "cheating", the dynamic account features (like average money sent) were calculated **only using transactions from the Training window (Days 1–8)**. If we had used Test window data to calculate account averages, the model would be illegally looking into the future.
*   **Feature Scaling:** We used Scikit-Learn's `StandardScaler` to normalize features (mean=0, variance=1). The scaler was strictly `fit()` on the training accounts and only `transform()` was applied to the validation/test accounts.
*   **Categorical Encoding:** Currencies and payment formats (Wire, Cheque, ACH) were mapped to integer indices using `LabelEncoder`. Timestamps were converted into cyclical sine/cosine wave features to capture daily/weekly temporal patterns.

---

## 2. Graph Construction (`src/aml_pipeline/build_graph.py`)

**Objective:** Convert tabular CSVs into a tensor-based graph structure suitable for neural network message passing.

**What we did & How we did it:**
*   **Framework:** We used **PyTorch Geometric (PyG)**, the industry standard for Graph Neural Networks.
*   **Node ID Mapping:** GNNs cannot process string-based IDs (like `"Account_1234"`). We created a continuous integer mapping (0 to $N-1$) for all 518,573 accounts.
*   **Graph Structure:** We built a **Homogeneous Directed Graph** represented as a PyG `Data` object containing:
    *   `x`: Node feature matrix of shape `[518573, 8]` (Accounts).
    *   `edge_index`: A `[2, E]` tensor mapping the source node to the destination node for every transaction.
    *   `edge_attr`: A `[E, 14]` tensor holding the specific features for that transaction (amount, currency, time).
    *   `y`: A binary label tensor (0 for legitimate, 1 for laundering).
*   **Masking:** Rather than creating three entirely separate graphs, we created one massive graph and assigned boolean masks (`train_mask`, `val_mask`, `test_mask`) to the edges. This allows the model to "see" the historical graph structure during testing without training on test labels.

---

## 3. Model Architecture (`src/models/graphsage_model.py`)

**Objective:** Design an inductive Graph Neural Network capable of Edge Classification.

**What we did & How we did it:**
*   **GraphSAGE Encoder:** We chose **GraphSAGE** instead of standard GCNs because GraphSAGE is *inductive* (it learns how to aggregate neighbors, meaning it can handle entirely new accounts that appear in the future).
    *   We implemented a 2-layer `SAGEConv` architecture.
    *   It uses "Mean Aggregation"—an account's embedding is calculated by taking the mathematical average of all its immediate neighbors' features, passed through a neural network layer.
*   **Edge Classifier Head:** Because our goal is to classify *transactions* (edges), not accounts (nodes), we built an `EdgeClassifier` module. 
    *   For any given transaction from Account A to Account B, the model takes the GraphSAGE embedding of Account A, the embedding of Account B, and concatenates them together with the original transaction features.
    *   This combined vector is passed through a Multi-Layer Perceptron (MLP) to output a final probability (0.0 to 1.0) of fraud.

---

## 4. Model Training & Evaluation (`src/training/train_aml.py`)

**Objective:** Train the model efficiently without running out of RAM, while handling extreme class imbalance.

**What we did & How we did it:**
*   **Mini-Batching via LinkNeighborLoader:** A graph with 5 million edges cannot fit into a GPU's memory all at once. We used PyG's `LinkNeighborLoader` to perform **mini-batching**. 
    *   For a batch of transactions, the loader traces back and samples exactly 15 neighbors at the 1st hop, and 10 neighbors at the 2nd hop. This creates a small, manageable "sub-graph" that easily fits on the GPU (Apple Silicon MPS).
*   **Handling Class Imbalance:** Fraud represents less than 0.1% of all edges. If the model predicted "Legitimate" 100% of the time, it would be 99.9% accurate. To fix this:
    *   We used `BCEWithLogitsLoss` (Binary Cross Entropy).
    *   We calculated the ratio of negative to positive edges and applied a **`pos_weight` of ~979**. This mathematically forces the model to treat 1 missed fraud case as equal to 979 false alarms, heavily penalizing the model for missing money laundering.
*   **Evaluation Metrics:** Because Accuracy is useless here, we evaluate using:
    *   **ROC-AUC:** How well the model separates the two classes globally.
    *   **PR-AUC:** Precision-Recall Area Under Curve (the gold standard for heavy imbalance).
    *   **Pattern-Level Recall:** We wrote a custom script (`evaluate_aml.py`) to parse the `HI-Small_Patterns.txt` file and calculate exactly how many multi-hop laundering rings (Cycles, Bipartite, Fan-Out) the model successfully flagged.
