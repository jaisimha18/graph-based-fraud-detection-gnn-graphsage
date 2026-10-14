#!/usr/bin/env python3
# ============================================================
# predict.py
#
# CLI tool to run inference or view model evaluation output
# directly in the terminal.
# ============================================================

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.aml_pipeline.aml_config import (
    AML_OUTPUT_DIR, HIDDEN_DIM, OUTPUT_DIM, SAGE_AGGR,
    NUM_SAGE_LAYERS, DROPOUT, PREPROCESSED_DIR, ACCOUNT_FEAT_DIM, EDGE_FEAT_DIM
)
from src.models.graphsage_model import AMLGraphSAGEModel
from src.models.cycle_graphsage_model import CycleAwareAMLModel
from src.aml_pipeline.cycle_features import compute_cycle_features_for_edges

def load_evaluation_summary(model_name: str = "cycle_attn"):
    if model_name == "baseline":
        report_path = os.path.join(AML_OUTPUT_DIR, "model_results", "evaluation_report.json")
    else:
        report_path = os.path.join(AML_OUTPUT_DIR, "model_results", f"evaluation_report_{model_name}.json")
        if not os.path.exists(report_path):
            report_path = os.path.join(AML_OUTPUT_DIR, "model_results", "evaluation_report.json")

    if not os.path.exists(report_path):
        print(f"[!] Evaluation report not found at {report_path}")
        return None
    with open(report_path, "r") as f:
        return json.load(f)

def print_model_summary(model_name: str = "cycle_attn"):
    report = load_evaluation_summary(model_name)
    if not report:
        return

    m_title = "CYCLE-AWARE (ATTENTION)" if model_name == "cycle_attn" else ("CYCLE RW" if model_name == "cycle_rw" else "VANILLA")
    print("\n" + "=" * 65)
    print(f"       AML {m_title} GRAPHSAGE MODEL RESULTS (TEST SET)")
    print("=" * 65)
    print(f"  • Model Architecture   : {report.get('model_name', model_name).upper()}")
    print(f"  • ROC-AUC Score        : {report.get('roc_auc', 0):.4f}")
    print(f"  • PR-AUC Score         : {report.get('pr_auc', 0):.4f}")
    print(f"  • Optimal F1-Score     : {report.get('f1_score', 0):.4f}")
    print(f"  • Detection Threshold  : {report.get('best_threshold', 0.5):.2f}")

    cm = report.get("confusion_matrix", [[0, 0], [0, 0]])
    print("\n" + "-" * 65)
    print("  CONFUSION MATRIX")
    print("-" * 65)
    print(f"                     Predicted Legit    Predicted Laundering")
    print(f"  Actual Legit       {cm[0][0]:<18} {cm[0][1]:<20} (TN / FP)")
    print(f"  Actual Laundering  {cm[1][0]:<18} {cm[1][1]:<20} (FN / TP)")

    clf_rep = report.get("classification_report", {})
    if "Laundering" in clf_rep:
        laundering = clf_rep["Laundering"]
        print("\n" + "-" * 65)
        print("  FRAUD DETECTION METRICS (LAUNDERING CLASS)")
        print("-" * 65)
        print(f"  • Recall (Sensitivity) : {laundering.get('recall', 0)*100:.2f}%  (Found {cm[1][1]} out of {cm[1][0] + cm[1][1]} frauds)")
        print(f"  • Precision            : {laundering.get('precision', 0)*100:.2f}%")
        print(f"  • F1-Score             : {laundering.get('f1-score', 0):.4f}")

    pat_rec = report.get("pattern_recall", {})
    if pat_rec:
        overall = pat_rec.get("overall", {})
        print("\n" + "-" * 75)
        print(f"  LAUNDERING RING-LEVEL RECALL (TOTAL PATTERNS: {overall.get('total_in_test', 0)})")
        print(f"  • Alert Recall (≥1 edge caught)        : {overall.get('recall_any', overall.get('recall', 0)):.1f}%")
        print(f"  • Majority Ring Recall (≥50% caught)   : {overall.get('recall_majority', 0):.1f}%")
        print(f"  • Complete Ring Recall (100% caught)   : {overall.get('recall_full', 0):.1f}%")
        print(f"  • Mean Ring Edge Coverage              : {overall.get('mean_coverage', 0):.1f}%")
        print("-" * 75)
        print(f"  {'Pattern Type':<16} {'Total':>6} {'Any(≥1)':>10} {'Maj(≥50%)':>12} {'Full(100%)':>12} {'Mean Cov':>10}")
        print(f"  {'-'*70}")
        for ptype, stats in sorted(pat_rec.get("by_type", {}).items()):
            tot = stats.get('total', 0)
            r_any = stats.get('recall_any', stats.get('recall', 0))
            r_maj = stats.get('recall_majority', 0)
            r_full = stats.get('recall_full', 0)
            m_cov = stats.get('mean_coverage', 0)
            print(f"  {ptype:<16} {tot:>6} {r_any:>9.1f}% {r_maj:>11.1f}% {r_full:>11.1f}% {m_cov:>9.1f}%")
    print("=" * 75 + "\n")

def cyclical(series: pd.Series, period: int):
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)

def run_inference(csv_path: str, threshold: float = None, limit: int = 25, only_fraud: bool = False, model_name: str = "cycle_attn"):
    if not os.path.exists(csv_path):
        print(f"[!] Error: File '{csv_path}' does not exist.")
        return

    out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")
    if model_name == "baseline":
        model_path = os.path.join(out_dir, "best_model.pt")
        is_cycle = False
        use_attn = False
    elif model_name == "cycle_rw":
        model_path = os.path.join(out_dir, "best_model_cycle_rw.pt")
        is_cycle = True
        use_attn = False
    else:  # cycle_attn
        model_path = os.path.join(out_dir, "best_model_cycle_attn.pt")
        is_cycle = True
        use_attn = True

    if not os.path.exists(model_path):
        # Fallback if specific model weights not found
        fallback_path = os.path.join(out_dir, "best_model.pt")
        if os.path.exists(fallback_path):
            print(f"  [!] {model_name} weights not found. Falling back to baseline.")
            model_path = fallback_path
            model_name = "baseline"
            is_cycle = False
        else:
            print(f"[!] Error: Model weights not found at '{model_path}'. Please train first.")
            return

    rep = load_evaluation_summary(model_name)
    val_thresh = rep.get("best_threshold", 0.95) if rep else 0.95
    if threshold is None:
        threshold = val_thresh

    print(f"\n[+] Model: {model_name.upper()} | Loading transactions from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                "Amount Received", "Amount Paid", "Receiving Currency", "Payment Currency", "Payment Format"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[!] Missing required columns: {missing}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device} | Threshold: {threshold:.2f} (Val Optimal: {val_thresh:.2f})")

    # Build unique nodes
    accounts = pd.concat([df["Account"], df["Account.1"]]).unique()
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    num_nodes = len(accounts)

    node_features = np.zeros((num_nodes, ACCOUNT_FEAT_DIM), dtype=np.float32)
    acc_feat_path = os.path.join(PREPROCESSED_DIR, "account_features.csv")
    if os.path.exists(acc_feat_path):
        acc_df = pd.read_csv(acc_feat_path)
        acc_df.set_index("account_id", inplace=True)
        for i, acc in enumerate(accounts):
            if acc in acc_df.index:
                node_features[i] = acc_df.loc[acc].values[:ACCOUNT_FEAT_DIM]

    edge_index_np = np.zeros((2, len(df)), dtype=np.int64)
    edge_index_np[0] = df["Account"].map(account_to_idx).values
    edge_index_np[1] = df["Account.1"].map(account_to_idx).values

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

    edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)

    if is_cycle:
        if num_nodes > 1 and len(df) > 1:
            cycle_feat_np = compute_cycle_features_for_edges(
                num_nodes, edge_index_np[0], edge_index_np[1], edge_index_np[0], edge_index_np[1]
            )
        else:
            cycle_feat_np = np.zeros((len(df), 8), dtype=np.float32)
        edge_attr_np = np.hstack([edge_attr_np, cycle_feat_np])

    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = edge_index_t.to(device)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float).to(device)
    edge_label_index = edge_index

    # Load model
    if not is_cycle:
        model = AMLGraphSAGEModel(
            node_feat_dim=ACCOUNT_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
            hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, aggr=SAGE_AGGR,
            num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
        ).to(device)
    else:
        model = CycleAwareAMLModel(
            node_feat_dim=ACCOUNT_FEAT_DIM, edge_feat_dim=22,
            hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM,
            use_cycle_attention=use_attn,
            num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"[+] Running {model_name.upper()} forward pass...")
    with torch.no_grad():
        if is_cycle:
            logits = model(x, edge_index, edge_attr, edge_label_index, subgraph_edge_attr=edge_attr)
        else:
            logits = model(x, edge_index, edge_attr, edge_label_index)
        preds = torch.sigmoid(logits).cpu().numpy()

    df["Fraud_Prob"] = preds
    df["Predicted_Label"] = (df["Fraud_Prob"] >= threshold).astype(int)
    df["Status"] = df["Predicted_Label"].apply(lambda p: "🚨 FRAUD" if p == 1 else "✅ LEGIT")

    has_actual = "Is Laundering" in df.columns

    out_df = df
    if only_fraud:
        out_df = df[df["Predicted_Label"] == 1]

    total_txns = len(df)
    flagged_fraud = int((df["Predicted_Label"] == 1).sum())

    print("\n" + "=" * 80)
    print(f"             MODEL INFERENCE OUTPUT ({model_name.upper()})")
    print("=" * 80)
    print(f"Total Transactions: {total_txns} | Flagged as Laundering: {flagged_fraud} ({(flagged_fraud/total_txns)*100:.1f}%)")
    if has_actual:
        actual_fraud = int(df["Is Laundering"].sum())
        print(f"Actual Laundering in CSV: {actual_fraud}")
    print("-" * 80)

    # Print table header
    if has_actual:
        print(f"{'Timestamp':<17} {'From Account':<12} {'To Account':<12} {'Amount':>12} {'Prob':>8} {'Predicted':>11} {'Actual':>8}")
        print("-" * 80)
    else:
        print(f"{'Timestamp':<17} {'From Account':<12} {'To Account':<12} {'Amount':>12} {'Prob':>8} {'Predicted':>11}")
        print("-" * 80)

    shown_count = 0
    for idx, row in out_df.iterrows():
        if limit and shown_count >= limit:
            break
        ts = str(row["Timestamp"])[:16]
        src = str(row["Account"])[:10]
        dst = str(row["Account.1"])[:10]
        amt = f"${float(row['Amount Paid']):,.2f}"
        prob = f"{row['Fraud_Prob']:.4f}"
        stat = row["Status"]
        if has_actual:
            act = "FRAUD" if row["Is Laundering"] == 1 else "LEGIT"
            print(f"{ts:<17} {src:<12} {dst:<12} {amt:>12} {prob:>8} {stat:>11} {act:>8}")
        else:
            print(f"{ts:<17} {src:<12} {dst:<12} {amt:>12} {prob:>8} {stat:>11}")
        shown_count += 1

    if limit and len(out_df) > limit:
        print(f"... and {len(out_df) - limit} more transactions (use --limit 0 to show all).")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="AML GraphSAGE Model Terminal Output Viewer")
    parser.add_argument("--model", type=str, default="cycle_attn", choices=["baseline", "cycle_rw", "cycle_attn"],
                        help="Model variant to run inference on (default: cycle_attn)")
    parser.add_argument("--summary", action="store_true", help="Display evaluation metrics for the selected model")
    parser.add_argument("--ablation", action="store_true", help="Display full comparative ablation table across all models")
    parser.add_argument("--csv", type=str, default=None, help="Path to transaction CSV file to run inference on (default: data/samples/sample_trans_4_with_fraud.csv)")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold (default: optimal threshold from report)")
    parser.add_argument("--limit", type=int, default=20, help="Number of transaction rows to display in terminal (0 for all)")
    parser.add_argument("--only-fraud", action="store_true", help="Only display transactions predicted as fraud")

    args = parser.parse_args()

    if args.ablation:
        from src.training.evaluate_cycle_aml import print_ablation_table
        print_ablation_table()
        return

    if args.summary:
        print_model_summary(args.model)
        return

    csv_file = args.csv if args.csv else "data/samples/sample_trans_4_with_fraud.csv"
    if not os.path.exists(csv_file) and not args.csv:
        print_model_summary(args.model)
        return

    run_inference(csv_file, threshold=args.threshold, limit=args.limit, only_fraud=args.only_fraud, model_name=args.model)

if __name__ == "__main__":
    main()
