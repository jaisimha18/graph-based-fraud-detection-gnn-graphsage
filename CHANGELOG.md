# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `CONTRIBUTING.md` with development guidelines and PR process
- `CHANGELOG.md` (this file) documenting version history
- Professional `README.md` with badges, ToC, and complete documentation
- `requirements.txt` with pinned Python dependencies
- Technical documentation suite in `docs/`:
  - `ARCHITECTURE.md` — System architecture deep-dive
  - `SETUP.md` — Detailed installation guide
  - `API_REFERENCE.md` — REST API documentation
  - `DATA_DICTIONARY.md` — Dataset schema reference

### Changed
- Restructured project: sample CSVs → `data/samples/`, utility scripts → `scripts/`, images → `docs/images/`
- Improved `.gitignore` with recursive `.DS_Store` patterns and comments
- Updated file path references across `predict.py` and utility scripts

### Removed
- Duplicate markdown files from project root (consolidated into `docs/`)
- Empty `notebooks/` directory
- `.DS_Store` files

---

## [0.3.0] — 2026-10-10

### Added
- **Cycle-Aware Attention GraphSAGE** model (`CycleAwareSAGEConv`)
- N-Step Random Walk Cycle Feature Extractor (`cycle_features.py`)
- Cycle-Aware model trainer with dynamic negative sampling (`train_cycle_aml.py`)
- Unified ablation evaluator (`evaluate_cycle_aml.py`)
- 4-Level ring completeness evaluation metrics
- Scientific ablation study comparing 3 model configurations
- 50-epoch training results and benchmark dashboard

### Results
- **PR-AUC:** 0.3235 → **0.5759** (+78.0% relative improvement)
- **Complete Ring Recall:** 76.9% → **85.2%** (+8.3% absolute)

---

## [0.2.0] — 2026-10-05

### Added
- Interactive **React web dashboard** with Vite
- **TransactionAnalysis** component with force-directed graph visualization
- **ResultsDashboard** with model performance metrics
- **FastAPI** backend API for inference endpoints
- Transaction CSV upload and real-time GNN inference

---

## [0.1.0] — 2026-09-28

### Added
- Initial **AML pipeline** source code
  - Data preprocessing with temporal split (zero data leakage)
  - PyG graph construction from transaction data
  - Vanilla GraphSAGE model and trainer
  - Evaluation with ring-level metrics
- EDA scripts for cycle pattern analysis
- SMOTE oversampling support
- Neo4j export functionality
- Project website scaffolding
- MIT License

---

## [0.0.1] — 2026-09-20

### Added
- Initial repository setup
- Sample dataset and basic project structure
- `.gitignore` for Python, Node.js, and dataset files
