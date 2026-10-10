# Setup Guide

> Detailed installation and configuration instructions for the Graph-Based AML Fraud Detection system.

---

## Prerequisites

| Requirement | Minimum Version | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| Node.js | 18 | 20 LTS |
| npm | 9 | 10+ |
| Git | 2.30 | Latest |
| RAM | 8 GB | 16 GB |
| Disk Space | 5 GB | 10 GB (with dataset) |

### Optional
- **CUDA-capable GPU** for faster training (NVIDIA with CUDA 11.8+)
- **Apple Silicon Mac** (MPS backend supported automatically)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/jaisimha18/graph-based-fraud-detection-gnn-graphsage.git
cd graph-based-fraud-detection-gnn-graphsage
```

---

## Step 2: Python Environment

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch Installation Notes

The `requirements.txt` specifies minimum versions. For GPU support, install PyTorch with the appropriate CUDA version:

```bash
# CPU only (macOS / Linux)
pip install torch torchvision torchaudio

# CUDA 11.8 (Linux / Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (Linux / Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install PyG:
```bash
pip install torch-geometric
```

> See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for platform-specific instructions.

---

## Step 3: Dataset Setup

### Download the IBM HI-Small Dataset

1. Visit the [IBM AML Dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
2. Download `HI-Small_Trans.csv` and `HI-Small_accounts.csv`
3. Place them in the `AML_DATASET/` directory:

```bash
mkdir -p AML_DATASET
mv /path/to/downloaded/HI-Small_Trans.csv AML_DATASET/
mv /path/to/downloaded/HI-Small_accounts.csv AML_DATASET/
```

### Verify Dataset

```bash
ls -lh AML_DATASET/
# Expected:
# HI-Small_Trans.csv      ~454 MB
# HI-Small_accounts.csv   ~32 MB
```

---

## Step 4: Run the Pipeline

### 4a. Preprocess Data

```bash
python -m src.aml_pipeline.preprocess
```

This creates preprocessed CSVs under `outputs/aml/preprocessed/`.

### 4b. Build Graph

```bash
python -m src.aml_pipeline.build_graph
```

This creates `outputs/aml/graph/aml_graph.pt` (~782 MB).

### 4c. Extract Cycle Features

```bash
python -m src.aml_pipeline.cycle_features
```

Computes cycle features in ~7.5 seconds and saves `aml_graph_cycle.pt`.

### 4d. Train Models

```bash
# Train Vanilla GraphSAGE Baseline
python -m src.training.train_aml --epochs 50

# Train Cycle-Aware Attention Model
python -m src.training.train_cycle_aml --mode cycle_attn --epochs 10 --neg_ratio 25
```

### 4e. Evaluate

```bash
python -m src.training.evaluate_cycle_aml --eval all
```

---

## Step 5: Frontend Setup

```bash
cd website
npm install
cd ..
```

---

## Step 6: Run the Application

### Start Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend (new terminal)

```bash
cd website
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: torch_geometric` | Install PyG: `pip install torch-geometric` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `aml_config.py` or use CPU |
| `FileNotFoundError: AML_DATASET/` | Download the dataset (see Step 3) |
| `Port 8000 already in use` | Use `--port 8001` or kill the existing process |
| `npm: command not found` | Install Node.js from https://nodejs.org/ |

### Verify Installation

```bash
# Check Python dependencies
python -c "import torch; import torch_geometric; print('✅ PyTorch', torch.__version__); print('✅ PyG', torch_geometric.__version__)"

# Check API
python -c "from src.api.main import app; print('✅ FastAPI app loaded')"
```
