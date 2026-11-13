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

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples"))

AVAILABLE_SAMPLES = [
    {
        "id": "sample_transactions.csv",
        "name": "Multi-Bank AML Investigation Stream",
        "description": "45 authentic cross-bank transactions embedding two flagged circular laundering rings (3-hop & 4-hop, $998K volume) and one legitimate unflagged commercial loop ($3.7K), plus 35 normal transfers.",
        "tag": "Verified Benchmark (Flagged & Unflagged)",
        "badgeColor": "purple",
        "rows": 45,
        "isCycle": True
    }
]

# Helper for cyclic encoding
def cyclical(series: pd.Series, period: int):
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)

def run_model_inference_on_df(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)

    # 1. Flexible column normalization & intelligent header mapping
    # Strip whitespace from column names
    df.rename(columns={c: str(c).strip() for c in df.columns}, inplace=True)
    
    # Drop unnamed index columns if present
    drop_unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_unnamed:
        df.drop(columns=drop_unnamed, inplace=True)

    rename_dict = {}
    for col in df.columns:
        cl = str(col).lower().replace("_", " ").strip()
        if cl in ["from account", "source account", "source_account", "src account", "origin account", "sender"]:
            rename_dict[col] = "Account"
        elif cl in ["to account", "destination account", "target account", "dest account", "dst account", "account 1", "account.1", "receiver"]:
            rename_dict[col] = "Account.1"
        elif cl in ["from bank", "source bank", "src bank", "origin bank"]:
            rename_dict[col] = "From Bank"
        elif cl in ["to bank", "target bank", "dest bank", "destination bank"]:
            rename_dict[col] = "To Bank"
        elif cl in ["amount paid", "amount_paid", "paid amount"]:
            rename_dict[col] = "Amount Paid"
        elif cl in ["amount received", "amount_received", "received amount"]:
            rename_dict[col] = "Amount Received"
        elif cl in ["amount", "txn amount", "transaction amount"]:
            if "Amount Paid" not in df.columns and "Amount Paid" not in rename_dict.values():
                rename_dict[col] = "Amount Paid"
        elif cl in ["currency", "curr"]:
            if "Payment Currency" not in df.columns and "Payment Currency" not in rename_dict.values():
                rename_dict[col] = "Payment Currency"
        elif cl in ["payment format", "format", "payment type", "txn type"]:
            rename_dict[col] = "Payment Format"
        elif cl in ["is laundering", "is fraud", "label", "fraud", "is_laundering", "is_fraud"]:
            rename_dict[col] = "Is Laundering"
    df.rename(columns=rename_dict, inplace=True)

    # If 'Account' appears duplicate times and 'Account.1' not present:
    if "Account.1" not in df.columns and list(df.columns).count("Account") >= 2:
        cols = list(df.columns)
        first_acc = cols.index("Account")
        second_acc = cols.index("Account", first_acc + 1)
        cols[second_acc] = "Account.1"
        df.columns = cols

    # Verify minimum required columns
    required = ["Account", "Account.1", "Amount Paid"]
    if not all(col in df.columns for col in required):
        return {"error": f"Missing required columns in CSV: {set(required) - set(df.columns)}. Supported columns include 'Account', 'Account.1' (or 'From Account', 'To Account'), and 'Amount Paid'."}

    if not model:
        return {"error": "Model not loaded. Please train the model first."}

    # Clean amount columns from string formatting ($ or commas)
    def clean_amount(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace(",", "").replace("$", "").strip()
        try:
            return float(val_str)
        except:
            return 0.0

    df["Amount Paid"] = df["Amount Paid"].apply(clean_amount)
    df["Amount Received"] = df["Amount Received"].apply(clean_amount)

    # Build unique nodes
    accounts = pd.concat([df["Account"].astype(str), df["Account.1"].astype(str)]).unique()
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    idx_to_account = {i: acc for acc, i in account_to_idx.items()}
    
    num_nodes = len(accounts)
    
    # Try to load existing node features, otherwise use zeros
    node_features = np.zeros((num_nodes, ACCOUNT_FEAT_DIM), dtype=np.float32)
    acc_feat_path = os.path.join(PREPROCESSED_DIR, "account_features.csv")
    if os.path.exists(acc_feat_path):
        acc_df = pd.read_csv(acc_feat_path)
        acc_df.set_index("account_id", inplace=True)
        for i, acc in enumerate(accounts):
            if acc in acc_df.index:
                row_val = acc_df.loc[acc]
                if isinstance(row_val, pd.DataFrame):
                    row_val = row_val.iloc[0]
                node_features[i] = row_val.values[:ACCOUNT_FEAT_DIM]
                
    # Build edge index
    edge_index_np = np.zeros((2, len(df)), dtype=np.int64)
    edge_index_np[0] = df["Account"].astype(str).map(account_to_idx).values
    edge_index_np[1] = df["Account.1"].astype(str).map(account_to_idx).values
    
    # Build edge features
    dt = pd.to_datetime(df["Timestamp"], errors="coerce")
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime("2022/09/10 12:00"))
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    
    amount_received_log = np.log1p(df["Amount Received"].astype(float).clip(lower=0)).astype(np.float32)
    amount_paid_log = np.log1p(df["Amount Paid"].astype(float).clip(lower=0)).astype(np.float32)
    amount_diff_log = (amount_received_log - amount_paid_log).astype(np.float32)
    
    is_cross_bank = (df["From Bank"].astype(str) != df["To Bank"].astype(str)).astype(np.float32)
    is_self_transfer = (df["Account"].astype(str) == df["Account.1"].astype(str)).astype(np.float32)
    is_cross_currency = (df["Receiving Currency"].astype(str) != df["Payment Currency"].astype(str)).astype(np.float32)
    
    # Exact categorical mapping matching training preprocessing
    currencies = ['Australian Dollar', 'Bitcoin', 'Brazil Real', 'Canadian Dollar', 'Euro', 'Mexican Peso', 'Ruble', 'Rupee', 'Saudi Riyal', 'Shekel', 'Swiss Franc', 'UK Pound', 'US Dollar', 'Yen', 'Yuan']
    formats = ['ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire']
    curr_dict = {c: i for i, c in enumerate(currencies)}
    fmt_dict = {f: i for i, f in enumerate(formats)}
    
    rec_curr_enc = df["Receiving Currency"].fillna("US Dollar").map(lambda x: curr_dict.get(x, 12)).astype(np.float32)
    fmt_enc = df["Payment Format"].fillna("ACH").map(lambda x: fmt_dict.get(x, 0)).astype(np.float32)
    
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
    else:
        cycle_feat_np = None

    # Identify directed simple cycles (rings) in the uploaded transaction graph
    adj = {}
    for _, r in df.iterrows():
        u, v = str(r["Account"]), str(r["Account.1"])
        adj.setdefault(u, []).append(v)

    seen_cycles = set()
    detected_rings = []

    def find_cycles(start, curr, path, visited):
        if len(path) > 7:
            return
        for nxt in adj.get(curr, []):
            if nxt == start and len(path) >= 2:
                cycle_tuple = tuple(path)
                min_idx = cycle_tuple.index(min(cycle_tuple))
                canon = cycle_tuple[min_idx:] + cycle_tuple[:min_idx]
                if canon not in seen_cycles:
                    seen_cycles.add(canon)
                    ring_nodes = path + [start]
                    detected_rings.append({
                        "ringId": len(detected_rings) + 1,
                        "hops": len(path),
                        "path": ring_nodes,
                        "pathDisplay": " ➔ ".join(ring_nodes),
                    })
            elif nxt not in visited and len(path) < 7:
                find_cycles(start, nxt, path + [nxt], visited | {nxt})

    for n in sorted(adj.keys()):
        find_cycles(n, n, [n], {n})

    # Set of directed edges (u, v) in any detected ring
    ring_edges = set()
    for ring in detected_rings:
        p = ring["path"]
        for k in range(len(p) - 1):
            ring_edges.add((p[k], p[k + 1]))

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
    
    # Links: {source: account, target: account, prediction: float, isFraud: bool, inCycle: bool, ...}
    links_out = []
    for i, (_, row) in enumerate(df.iterrows()):
        source = str(row["Account"])
        target = str(row["Account.1"])
        pred_score = float(preds[i])
        is_fraud = bool(pred_score > best_thresh)
        amt = float(row["Amount Paid"])
        
        in_ring = (source, target) in ring_edges
        c_prob = 0.0
        has_c = in_ring
        c_hop = 0
        if is_cycle_model and cycle_feat_np is not None:
            c_prob = float(cycle_feat_np[i, 4])
            has_c = bool(in_ring or cycle_feat_np[i, 4] > 0 or np.any(cycle_feat_np[i, 5:8] > 0))
            if cycle_feat_np[i, 5] > 0:
                c_hop = 2
            elif cycle_feat_np[i, 6] > 0:
                c_hop = 3
            elif cycle_feat_np[i, 7] > 0:
                c_hop = 4
            elif cycle_feat_np[i, 3] > 0:
                c_hop = 5
            elif in_ring:
                for r in detected_rings:
                    if (source, target) in zip(r["path"][:-1], r["path"][1:]):
                        c_hop = len(r["path"]) - 1
                        break

        links_out.append({
            "source": source,
            "target": target,
            "prediction": pred_score,
            "isFraud": is_fraud,
            "inCycle": has_c,
            "cycleHop": c_hop,
            "cycleProb": round(c_prob, 4),
            "isLaunderingCycle": bool(is_fraud and has_c),
            "amount": amt,
            "timestamp": str(row["Timestamp"]),
            "currency": str(row["Payment Currency"]),
            "fromBank": str(row["From Bank"]),
            "toBank": str(row["To Bank"]),
        })

    # Enrich rings with amount and edge predictions
    for ring in detected_rings:
        p = ring["path"]
        ring_pairs = set((p[k], p[k + 1]) for k in range(len(p) - 1))
        matching_links = [l for l in links_out if (l["source"], l["target"]) in ring_pairs]
        ring["totalAmount"] = round(sum(l["amount"] for l in matching_links), 2)
        ring["currency"] = matching_links[0]["currency"] if matching_links else "USD"
        ring["flaggedEdges"] = sum(1 for l in matching_links if l["isFraud"])
        ring["totalEdges"] = len(matching_links)
        ring["isLaundering"] = bool(ring["flaggedEdges"] > 0)
        ring["avgPrediction"] = round(float(np.mean([l["prediction"] for l in matching_links])), 4) if matching_links else 0.0
        
    return {
        "nodes": nodes_out,
        "links": links_out,
        "threshold": float(best_thresh),
        "recommendedThreshold": 0.85,
        "actualFraudCount": actual_fraud,
        "modelUsed": "cycle_attn" if is_cycle_model else "baseline",
        "cycleAnalysis": {
            "totalRingsFound": len(detected_rings),
            "totalCycleEdges": sum(1 for l in links_out if l.get("inCycle")),
            "launderingCyclesCount": sum(1 for l in links_out if l.get("isLaunderingCycle")),
            "detectedRings": detected_rings
        }
    }

@app.post("/api/predict")
async def predict_transactions(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        return {"error": f"Failed to parse CSV file: {str(e)}"}
    
    return run_model_inference_on_df(df)

@app.get("/api/samples")
def get_samples_list():
    return {"samples": AVAILABLE_SAMPLES}

@app.get("/api/sample/{sample_id}/download")
def download_sample_file(sample_id: str):
    from fastapi.responses import FileResponse
    clean_name = os.path.basename(sample_id)
    file_path = os.path.join(SAMPLES_DIR, clean_name)
    if not os.path.exists(file_path):
        return {"error": "Sample file not found"}
    return FileResponse(file_path, media_type="text/csv", filename=clean_name)

@app.post("/api/predict_sample/{sample_id}")
def predict_sample(sample_id: str):
    clean_name = os.path.basename(sample_id)
    file_path = os.path.join(SAMPLES_DIR, clean_name)
    if not os.path.exists(file_path):
        return {"error": f"Sample file {clean_name} not found"}
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to read sample file: {str(e)}"}
    res = run_model_inference_on_df(df)
    res["sampleLoaded"] = clean_name
    return res

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
