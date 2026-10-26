import os
import sys
import io
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import (
    AML_OUTPUT_DIR, HIDDEN_DIM, OUTPUT_DIM, SAGE_AGGR, 
    NUM_SAGE_LAYERS, DROPOUT, PREPROCESSED_DIR, ACCOUNT_FEAT_DIM, EDGE_FEAT_DIM
)
from src.models.graphsage_model import AMLGraphSAGEModel
from src.models.cycle_graphsage_model import CycleAwareAMLModel
from src.aml_pipeline.cycle_features import compute_cycle_features_for_edges

app = FastAPI(title="GraphSAGE AML Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "GraphSAGE AML Fraud Detection API",
        "model_loaded": model is not None,
        "model_type": "cycle_attn" if is_cycle_model else ("baseline" if model else "none")
    }

# Load model globally
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")
cycle_model_path = os.path.join(out_dir, "best_model_cycle_attn.pt")
baseline_model_path = os.path.join(out_dir, "best_model.pt")

is_cycle_model = False
model = None
best_thresh = 0.5

if os.path.exists(cycle_model_path):
    print(f"[*] Loading Cycle-Aware AML Model from {cycle_model_path}")
    model = CycleAwareAMLModel(
        node_feat_dim=ACCOUNT_FEAT_DIM, edge_feat_dim=22,
        hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM,
        use_cycle_attention=True,
        num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(cycle_model_path, map_location=device))
    model.eval()
    is_cycle_model = True
    rep_path = os.path.join(out_dir, "evaluation_report_cycle_attn.json")
    if os.path.exists(rep_path):
        with open(rep_path, "r") as f:
            best_thresh = json.load(f).get("best_threshold", 0.95)
elif os.path.exists(baseline_model_path):
    print(f"[*] Loading Baseline GraphSAGE Model from {baseline_model_path}")
    model = AMLGraphSAGEModel(
        node_feat_dim=ACCOUNT_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, aggr=SAGE_AGGR,
        num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(baseline_model_path, map_location=device))
    model.eval()
    rep_path = os.path.join(out_dir, "evaluation_report.json")
    if os.path.exists(rep_path):
        with open(rep_path, "r") as f:
            best_thresh = json.load(f).get("best_threshold", 0.95)

# Helper for cyclic encoding
def cyclical(series: pd.Series, period: int):
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)

@app.post("/api/predict")
async def predict_transactions(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        return {"error": f"Failed to parse CSV file: {str(e)}"}
    
    # Check if necessary columns exist
    required = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", 
                "Amount Received", "Amount Paid", "Receiving Currency", "Payment Currency", "Payment Format"]
    
    if not all(col in df.columns for col in required):
        return {"error": "Missing required columns in CSV."}
        
    if not model:
        return {"error": "Model not loaded. Please train the model first."}
        
    # Build unique nodes
    accounts = pd.concat([df["Account"], df["Account.1"]]).unique()
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    idx_to_account = {i: acc for acc, i in account_to_idx.items()}
    
    num_nodes = len(accounts)
    
    # Try to load existing node features, otherwise use zeros
    node_features = np.zeros((num_nodes, ACCOUNT_FEAT_DIM), dtype=np.float32)
    acc_feat_path = os.path.join(PREPROCESSED_DIR, "account_features.csv")
    if os.path.exists(acc_feat_path):
        acc_df = pd.read_csv(acc_feat_path)
        # map account_id to row
        acc_df.set_index("account_id", inplace=True)
        for i, acc in enumerate(accounts):
            if acc in acc_df.index:
                node_features[i] = acc_df.loc[acc].values[:ACCOUNT_FEAT_DIM]
                
    # Build edge index
    edge_index_np = np.zeros((2, len(df)), dtype=np.int64)
    edge_index_np[0] = df["Account"].map(account_to_idx).values
    edge_index_np[1] = df["Account.1"].map(account_to_idx).values
    
    # Build edge features
    dt = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    
    amount_received_log = np.log1p(df["Amount Received"].astype(float).clip(lower=0)).astype(np.float32)
    amount_paid_log = np.log1p(df["Amount Paid"].astype(float).clip(lower=0)).astype(np.float32)
    amount_diff_log = (amount_received_log - amount_paid_log).astype(np.float32)
    
    is_cross_bank = (df["From Bank"].astype(str) != df["To Bank"].astype(str)).astype(np.float32)
    is_self_transfer = (df["Account"] == df["Account.1"]).astype(np.float32)
    is_cross_currency = (df["Receiving Currency"] != df["Payment Currency"]).astype(np.float32)
    
    # Exact categorical mapping matching training preprocessing
    currencies = ['Australian Dollar', 'Bitcoin', 'Brazil Real', 'Canadian Dollar', 'Euro', 'Mexican Peso', 'Ruble', 'Rupee', 'Saudi Riyal', 'Shekel', 'Swiss Franc', 'UK Pound', 'US Dollar', 'Yen', 'Yuan']
    formats = ['ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire']
    curr_dict = {c: i for i, c in enumerate(currencies)}
    fmt_dict = {f: i for i, f in enumerate(formats)}
    
    rec_curr_enc = df["Receiving Currency"].fillna("US Dollar").map(lambda x: curr_dict.get(x, 12)).astype(np.float32)
    fmt_enc = df["Payment Format"].fillna("Cheque").map(lambda x: fmt_dict.get(x, 3)).astype(np.float32)
    
    hour_sin, hour_cos = cyclical(df["hour"], 24)
    dow_sin, dow_cos = cyclical(df["dow"], 7)
    
    is_night = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.float32)
    is_weekend = (df["dow"] >= 5).astype(np.float32)
    
    edge_attr_np = np.column_stack([
        amount_received_log, amount_paid_log, amount_diff_log,
        is_cross_bank, is_self_transfer, is_cross_currency,
        rec_curr_enc, fmt_enc,
        hour_sin, hour_cos, dow_sin, dow_cos,
        is_night, is_weekend
    ])
    
    if is_cycle_model:
        if num_nodes > 1 and len(df) > 1:
            cycle_feat_np = compute_cycle_features_for_edges(
                num_nodes, edge_index_np[0], edge_index_np[1], edge_index_np[0], edge_index_np[1]
            )
        else:
            cycle_feat_np = np.zeros((len(df), 8), dtype=np.float32)
        edge_attr_np = np.hstack([edge_attr_np, cycle_feat_np])

    # Move to tensors
    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float).to(device)
    edge_label_index = edge_index

    with torch.no_grad():
        if is_cycle_model:
            logits = model(x, edge_index, edge_attr, edge_label_index, subgraph_edge_attr=edge_attr)
        else:
            logits = model(x, edge_index, edge_attr, edge_label_index)
        preds = torch.sigmoid(logits).cpu().numpy()
        
    # Prepare JSON response
    # Nodes: {id: account_id}
    nodes_out = [{"id": str(acc)} for acc in accounts]
    
    # Calculate actual fraud count if column exists
    actual_fraud = 0
    if "Is Laundering" in df.columns:
        actual_fraud = int(df["Is Laundering"].sum())
    
    # Links: {source: account, target: account, prediction: float, isFraud: bool}
    links_out = []
    for i, row in df.iterrows():
        source = str(row["Account"])
        target = str(row["Account.1"])
        pred_score = float(preds[i])
        is_fraud = bool(pred_score > best_thresh)
        amt = float(row["Amount Paid"])
        links_out.append({
            "source": source,
            "target": target,
            "prediction": pred_score,
            "isFraud": is_fraud,
            "amount": amt,
            "timestamp": str(row["Timestamp"]),
            "currency": str(row["Payment Currency"])
        })
        
    return {
        "nodes": nodes_out,
        "links": links_out,
        "threshold": float(best_thresh),
        "actualFraudCount": actual_fraud,
        "modelUsed": "cycle_attn" if is_cycle_model else "baseline"
    }

@app.get("/api/ablation")
def get_ablation_results():
    results = {}
    for m in ["baseline", "cycle_rw", "cycle_attn"]:
        path = os.path.join(out_dir, f"evaluation_report_{m}.json" if m != "baseline" else "evaluation_report.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                results[m] = json.load(f)
    return results

@app.get("/api/model_info")
def get_model_info():
    return {
        "active_model": "cycle_attn" if is_cycle_model else "baseline",
        "threshold": best_thresh,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
