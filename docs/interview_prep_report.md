# 🚀 Interview Project Understanding Report: "Inductive Graph Learning for Fraud Detection"

## Part 1: Core Project Concepts (The "What" and "Why")

**1. Problem Statement**
Detecting Anti-Money Laundering (AML) transactions in financial networks. Given a dataset of accounts and money transfers, predict whether a specific transaction (edge) is part of a laundering scheme.

**2. Why Fraud Detection is Difficult**
- **Extreme Class Imbalance:** Fraud is ~0.10% of the dataset.
- **Complexity:** Fraudsters split transactions across multiple banks, over multiple days, using intermediaries (smurfing). Static rules fail because they only look at single transactions in isolation.

**3. Why Graphs are Useful for Transaction Fraud**
Transactions inherently form a network. A tabular model (like XGBoost) only sees an isolated row (`Account A sent $500 to Account B`). A graph model sees the **topology**: `Account A is part of a 5-hop circular network that routes money back to Account A.` Graph structures naturally capture laundering motifs like cycles, fan-ins, and fan-outs.

**4. What Nodes and Edges Represent**
- **Nodes:** Financial Accounts (~518K). It is a **homogeneous graph** (only one node type).
- **Edges:** Directed money transfers from a source account to a destination account.

**5. Dataset and Preprocessing**
- **Dataset:** IBM HI-Small AML dataset.
- **Preprocessing:** A strict **chronological split** is used to prevent temporal data leakage (Train: Sept 1–8, Validation: Sept 9, Test: Sept 10–18).

**6. Features**
- **Node Features (8 dims):** `bank_id`, `entity_type_enc`, `is_crypto_bank`, `out_txn_count`, `in_txn_count`, `out_avg_amount`, `in_avg_amount`, `unique_counterparties`. *(Crucial: Transaction counts/averages are computed using ONLY the training period to prevent data leakage).*
- **Edge Features (14 dims):** Log-scaled amounts, categorical encodings (currencies, formats), temporal sine/cosine encodings, and binary flags (`is_cross_bank`, `is_night`, `is_weekend`).

**7. Graph Construction**
Using PyTorch Geometric (PyG), we build a `Data` object where `x` holds node features, `edge_index` holds the directed topology, and `edge_attr` holds transaction features.

## Part 2: GraphSAGE Deep Dive (The "How")

**8. What GraphSAGE is**
GraphSAGE (Graph Sample and Aggregate) is an *inductive* framework that generates low-dimensional vector representations (embeddings) for nodes by sampling and aggregating features from a node's local neighborhood.

**9. How GraphSAGE Works Step-by-Step**
1. **Sample:** For a target node, randomly sample a fixed number of its immediate neighbors.
2. **Aggregate:** Combine the feature vectors of those sampled neighbors (using a `mean` aggregator).
3. **Update:** Concatenate the aggregated features with the target node's current features, pass them through a dense neural network layer, and update the target node's embedding.
4. **Repeat:** Repeat this for $K$ layers (2 layers in this project).

**10. Message Passing**
Nodes "send" their feature states along the edges to their neighbors. `SAGEConv` handles this mathematically.

**11. Aggregation**
The mathematical function used to combine messages from multiple neighbors. This project uses `aggr="mean"`.

**12. Node Embeddings**
Continuous vector outputs (64-dimensional) that compress the node's static features AND its structural role in the network.

**13. Inductive Learning**
Standard GNNs (like GCN) are *transductive*—they learn an embedding matrix for the entire fixed graph and must retrain if a new node joins. GraphSAGE is *inductive*. It learns **aggregator functions** (neural network weights). Therefore, when a completely new account makes a transaction tomorrow, GraphSAGE can instantly generate an embedding for it based on its features and neighbors.

**14. Why GraphSAGE instead of a normal Neural Network?**
A normal NN (MLP) assumes tabular, independent data. GraphSAGE pulls in structural context (e.g., is this sender part of a known fraud ring?).

## Part 3: Fraud Patterns & Imbalance

**15. Circular Transaction / Cycle Detection**
Money laundering often involves moving dirty money through intermediaries and back to the source. Cycles in this dataset range from 2–12 hops and are 97.2% cross-bank. Currently, the model captures this *implicitly* via multi-hop message passing. *Future Roadmap:* Extending message-passing to explicitly detect these motifs.

**16. Cycle Motifs**
A recurring closed-loop subgraph pattern (e.g., A -> B -> C -> A). 

**17. In-degree and Out-degree Asymmetry**
Fraudsters often have a massive imbalance between incoming and outgoing edges. A "smurf" account might have 50 incoming transactions (high in-degree) but immediately empty the funds in 1 massive outgoing transaction (low out-degree).

**18. Fraud-specific structural features**
Currently, features include `in/out degree asymmetry` (via transaction counts), cross-bank flags, and rapid night/weekend activity. *Future Roadmap:* Explicit cycle motifs injected as augmented node embeddings.

**23. Class Imbalance & 24. SMOTE**
Fraud is ~0.10%. 
- **Algorithmic:** `compute_pos_weight` in `BCEWithLogitsLoss` heavily penalizes missing fraud.
- **Data-level:** Borderline SMOTE is applied on the *training edge features only* to synthesize new laundering edges (up to 5% target ratio).

**26. Node-level vs Pattern-level detection**
Currently, the model predicts at the **edge-level** (is this transaction fraud?). However, `evaluate_aml.py` aggregates these to the pattern level to check if the overall laundering structure (Cycles, Fan-ins, Fan-outs) was caught. *Future Roadmap:* Unifying this into the streaming inference pipeline for real-time pattern alerts.

## Part 4: Technical Implementation

**19. Model Architecture**
- **Encoder:** `GraphSAGEEncoder` uses `SAGEConv` layers -> BatchNorm -> ReLU -> Dropout.
- **Classifier:** `EdgeClassifier` (MLP) takes `z_src`, `z_dst`, and `edge_attr`, concatenates them, and outputs a fraud probability.

**20. Training Pipeline & 21. Validation/Testing**
- PyG's `LinkNeighborLoader` samples subgraphs in mini-batches.
- Optimizer: AdamW. Loss: BCEWithLogitsLoss. Early stopping monitors Validation AUC.

**22. Evaluation Metrics**
- ROC-AUC, PR-AUC, F1-Score, and Pattern-level Recall.

**25. Streaming Inference**
Because GraphSAGE is inductive, `predict.py` takes raw CSV transaction data, creates localized node and edge features on the fly, and runs a forward pass without rebuilding the entire training graph.

**27. PyTorch Geometric Implementation**
Leverages `Data`, `SAGEConv`, and `LinkNeighborLoader`.

**28. Complete Data Flow**
CSVs → `preprocess.py` (chronological split) → `smote_oversample.py` → `build_graph.py` → `train_aml.py` → `predict.py`.

---

## Part 5: 🎤 Interview Scripts & Pitches

### A. 30-Second Project Explanation (Elevator Pitch)
"For my project, I built an Anti-Money Laundering system using Graph Neural Networks, specifically GraphSAGE. Traditional tabular models fail at fraud because they view transactions in isolation. I modeled financial data as a graph to capture structural topology like circular routing. GraphSAGE allows the model to learn these patterns inductively, meaning it can classify completely new, unseen accounts in real-time streaming inference. We handle extreme class imbalance using SMOTE and chronological splitting to prevent data leakage."

### B. 2-Minute Explanation
"My project is an Inductive Graph Learning pipeline for detecting AML transactions, built with PyTorch Geometric. Standard ML models only see one row at a time, but fraud happens in complex, multi-hop networks. I modeled the dataset as a homogeneous graph where accounts are nodes and transactions are edges. 

I chose GraphSAGE specifically because of its inductive property. Transductive models like GCN require full retraining for new nodes, but GraphSAGE learns aggregator functions, allowing it to generate embeddings for unseen accounts in real-time. 

The architecture consists of a GraphSAGE encoder that aggregates multi-hop neighborhood features, concatenated with an Edge Classifier MLP that evaluates the source embedding, destination embedding, and edge features. To ensure production-grade rigor, I implemented a strict chronological split—training on the first 8 days and testing on subsequent days. I handled the severe 0.10% class imbalance using Borderline SMOTE on the training edge features. 

Currently, the model implicitly captures complex structures and evaluates pattern-level recall offline. Our immediate roadmap involves extending the GraphSAGE message-passing layer to explicitly detect circular transactions and integrating pattern-level detection directly into the unified streaming inference pipeline."

### C. 5-Minute Deep Explanation
*(Use the 2-minute pitch, then dive into the architecture and data leakage)*:
"Let me explain how I handled data leakage, which was the hardest part. If I calculated an account's 'average transaction amount' using the whole dataset, I would leak test-set data into the training phase. So, I engineered the pipeline to compute node aggregates *only* using transactions from days 1 to 8. 

During training, I use `LinkNeighborLoader` to sample subgraphs. The GraphSAGE encoder performs message passing. It takes a node, samples its neighbors, averages their features, and passes it through a linear layer. After 2 layers, the node's embedding contains information about its 2-hop neighborhood. The `EdgeClassifier` takes the 64-dimensional embedding of the sender, the receiver, and the 14-dim raw transaction features, and passes them through an MLP. 

Because laundering is 1 in 1000 transactions, I implemented `BCEWithLogitsLoss` with a positive weight scaler, and augmented the feature space using SMOTE. Finally, in inference, I prove the inductive nature of the model by processing new raw CSVs on the fly."

---

## Part 6: Expected Interview Questions & Answers

**G. "Why did you choose GraphSAGE?"**
> "In AML, new accounts and transactions appear every second. Standard GCNs are transductive—they require the full graph Laplacian and require retraining when a new node joins. GraphSAGE is inductive; it learns aggregator functions. This allows my inference pipeline to generate embeddings for unseen accounts on the fly. Furthermore, GraphSAGE mini-batching (`LinkNeighborLoader`) scales to millions of nodes."

**H. "Why not traditional ML?"**
> "Tabular models assume independent data. Transactions are highly dependent. A transaction of $500 at 2 AM looks normal to XGBoost. But GraphSAGE sees that this $500 is part of a 4-hop cycle spanning 3 days, which strongly indicates a laundering motif."

**I. "What is inductive learning?"**
> "The ability to make predictions on data you have never seen before, by learning aggregator functions rather than optimizing a fixed embedding matrix."

**J. "How does GraphSAGE handle unseen nodes?"**
> "When an unseen account makes a transaction, the model looks at the account's features, looks at the features of the accounts it is transferring to/from, passes them through the pre-trained `SAGEConv` layers, and generates a valid embedding on the spot."

**K. "What is message passing?"**
> "The mechanism where a node updates its representation by aggregating the hidden states (messages) of its immediate neighbors from the previous layer."

**L. "What is an embedding?"**
> "In my model, it is a 64-dimensional vector that represents an account. If two accounts have similar embeddings, they have similar static features AND play similar structural roles in the network."

**M. "How do you detect circular fraud?"**
> "Currently, our 2-layer GraphSAGE model implicitly captures cycles by aggregating features across 2-hops. The resulting embeddings for nodes in dense cycles capture this connectivity. Our future roadmap involves modifying the message passing to explicitly search for cycle motifs during the forward pass."

**N. "How did you handle class imbalance?"**
> "I used a `pos_weight` in `BCEWithLogitsLoss` and applied Borderline SMOTE to the edge features of the training set to synthetically upsample the fraud class to a 5% target ratio, ensuring it didn't leak into validation data."

**O. "What are the limitations?"**
> "The graph is currently homogeneous. Real financial networks have different node types (Banks, Retail, Corporate). Additionally, my GraphSAGE uses a `mean` aggregator; an attention mechanism (like GAT) might be better."

**P. "What would you improve?"**
> "Upgrading to a Heterogeneous Graph Neural Network (HeteroGNN) and fully implementing the unified pattern-level streaming inference pipeline that is currently on our roadmap."

---

## Part 7: 🚫 Boundaries (The "NOT To Do / NOT To Say" List)

If they press you on technical limitations or roadmap feasibility, you must be clear on your boundaries to maintain credibility. 

**Things we will absolutely NOT do in the future (Architectural Boundaries):**
1. **We will NOT use Transductive Models (like GCN):** "We will never move to GCN or standard spectral methods. The financial graph is too massive and dynamic; requiring a full graph laplacian recalculation every day is computationally impossible in production."
2. **We will NOT apply SMOTE to Graph Topologies:** "We will not attempt to synthetically generate new nodes or synthetic edges in the actual graph structure. Generating fake structural topology risks creating logically impossible financial flows. SMOTE will remain strictly applied to tabular feature spaces."
3. **We will NOT flatten the graph into tabular data:** "We will not revert to extracting static graph features (like PageRank) and feeding them into XGBoost as a primary method. GNNs remain the core because end-to-end learning is vastly superior."

**Things you absolutely must NOT say tomorrow (To avoid getting caught):**
1. ❌ **Do NOT say you used GCN or GAT.** (You used SAGEConv).
2. ❌ **Do NOT say the graph is Heterogeneous.** (It is homogeneous; only "account" nodes exist right now).
3. ❌ **Do NOT say your current python script outputs "pattern-level real-time alerts".** (Say: "The current streaming inference does edge-level classification, and the pattern-level detection is currently an offline evaluation metric that we are migrating into the real-time unified pipeline.")
4. ❌ **Do NOT say you applied SMOTE to the Validation or Test sets.** (This is the #1 way to fail a data science interview. Always emphasize you only applied SMOTE to the training set).

---

### ⏳ 1-Hour Revision Sheet Before the Interview
- **Framework:** PyTorch, PyTorch Geometric (PyG).
- **Algorithm:** GraphSAGE (`SAGEConv`), 2 layers, `mean` aggregation.
- **Task:** Edge Classification (Binary: Fraud vs Legit).
- **Data:** IBM HI-Small AML dataset. 518K accounts (nodes), chronological split.
- **Node Features (8):** Bank ID, Entity Type, Crypto flag, + 5 aggregates (calculated on train only).
- **Edge Features (14):** Log amounts, currencies, time sine/cosine encodings, cross-bank flags.
- **Imbalance Handling:** Pos_weight in BCE loss + Borderline SMOTE on train edge features.
- **The Resume "Out":** If asked about explicit cycle detection or pattern-level streaming, clearly frame them as the *architectural extension roadmap* that builds directly on top of the currently functioning implicit GraphSAGE edge-level pipeline.
