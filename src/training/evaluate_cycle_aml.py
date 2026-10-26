# ============================================================
# src/training/evaluate_cycle_aml.py
#
# Unified evaluation and comparative ablation reporter.
# Evaluates baseline, cycle_rw, and cycle_attn models.
# ============================================================

import os
import sys
import json
import argparse

try:
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
    from torch_geometric.loader import LinkNeighborLoader
    from tqdm import tqdm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import (
    GRAPH_DIR, AML_OUTPUT_DIR, BATCH_SIZE, NUM_NEIGHBORS,
    HIDDEN_DIM, OUTPUT_DIM, NUM_SAGE_LAYERS, DROPOUT, THRESHOLD_SWEEP,
    PATTERNS_TXT, PREPROCESSED_DIR
)

def get_device():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for model evaluation. Please install torch.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
def get_predictions(model, loader, target_edge_attr, device, is_cycle_model=False):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for model evaluation. Please install torch.")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            batch = batch.to(device)
            batch_edge_attr = target_edge_attr[batch.input_id.cpu()].to(device)

        if is_cycle_model:
            subgraph_edge_attr = batch.edge_attr.to(device) if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
            logits = model(batch.x, batch.edge_index, batch_edge_attr, batch.edge_label_index, subgraph_edge_attr)
        else:
            logits = model(batch.x, batch.edge_index, batch_edge_attr, batch.edge_label_index)

        preds = torch.sigmoid(logits).cpu().numpy()
        labels = batch.edge_label.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)

def evaluate_model(model_name: str = "cycle_attn"):
    """
    Evaluates a specific model variant: 'baseline', 'cycle_rw', or 'cycle_attn'.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and PyG are required to run model evaluation. Install torch to run inference.")
    from src.models.graphsage_model import AMLGraphSAGEModel
    from src.models.cycle_graphsage_model import CycleAwareAMLModel
    from src.training.evaluate_aml import parse_patterns, compute_pattern_recall

    print("\n" + "=" * 65)
    print(f"  EVALUATION: {model_name.upper()} MODEL")
    print("=" * 65)

    device = get_device()
    out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")

    if model_name == "baseline":
        model_path = os.path.join(out_dir, "best_model.pt")
        graph_path = os.path.join(GRAPH_DIR, "aml_graph.pt")
        is_cycle = False
        use_attn = False
    elif model_name == "cycle_rw":
        model_path = os.path.join(out_dir, "best_model_cycle_rw.pt")
        graph_path = os.path.join(GRAPH_DIR, "aml_graph_cycle.pt")
        is_cycle = True
        use_attn = False
    else:  # cycle_attn
        model_path = os.path.join(out_dir, "best_model_cycle_attn.pt")
        graph_path = os.path.join(GRAPH_DIR, "aml_graph_cycle.pt")
        is_cycle = True
        use_attn = True

    if not os.path.exists(model_path):
        print(f"  [!] Model weights not found at {model_path}.")
        return None

    print(f"  Loading graph from {graph_path}...")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    val_edge_index = data.edge_index[:, data.val_mask]
    val_edge_label = data.y[data.val_mask]
    val_edge_attr = data.edge_attr[data.val_mask]

    test_edge_index = data.edge_index[:, data.test_mask]
    test_edge_label = data.y[data.test_mask]
    test_edge_attr = data.edge_attr[data.test_mask]

    val_loader = LinkNeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, edge_label_index=val_edge_index,
        edge_label=val_edge_label, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = LinkNeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, edge_label_index=test_edge_index,
        edge_label=test_edge_label, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    node_feat_dim = data.x.size(1)
    edge_feat_dim = data.edge_attr.size(1)

    if not is_cycle:
        model = AMLGraphSAGEModel(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, aggr="mean",
            num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
        ).to(device)
    else:
        model = CycleAwareAMLModel(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM,
            use_cycle_attention=use_attn,
            num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Threshold optimization
    print("  Optimizing threshold on Validation Set...")
    val_preds, val_labels = get_predictions(model, val_loader, val_edge_attr, device, is_cycle_model=is_cycle)

    best_f1 = 0
    best_thresh = 0.5
    for thresh in THRESHOLD_SWEEP:
        bin_preds = (val_preds > thresh).astype(int)
        f1 = f1_score(val_labels, bin_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"  Best Val Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")

    # Test evaluation
    print("  Evaluating on Test Set...")
    test_preds, test_labels = get_predictions(model, test_loader, test_edge_attr, device, is_cycle_model=is_cycle)

    test_roc_auc = float(roc_auc_score(test_labels, test_preds))
    prec_arr, rec_arr, _ = precision_recall_curve(test_labels, test_preds)
    test_pr_auc = float(auc(rec_arr, prec_arr))

    test_bin_preds = (test_preds > best_thresh).astype(int)
    test_f1 = float(f1_score(test_labels, test_bin_preds))
    cm = confusion_matrix(test_labels, test_bin_preds)
    fraud_recall = float(cm[1][1] / max(cm[1][0] + cm[1][1], 1))
    fraud_precision = float(cm[1][1] / max(cm[0][1] + cm[1][1], 1))

    print(f"  Test ROC-AUC   : {test_roc_auc:.4f}")
    print(f"  Test PR-AUC    : {test_pr_auc:.4f}")
    print(f"  Fraud Recall   : {fraud_recall*100:.2f}% ({cm[1][1]}/{cm[1][0] + cm[1][1]})")
    print(f"  Fraud Prec     : {fraud_precision*100:.2f}% (FP: {cm[0][1]:,})")

    # Pattern-level recall
    pattern_recall = compute_pattern_recall(test_edge_index, test_bin_preds, data, best_thresh, out_dir)

    report_path = os.path.join(out_dir, f"evaluation_report_{model_name}.json")
    results = {
        "model_name": model_name,
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        "best_threshold": best_thresh,
        "f1_score": test_f1,
        "fraud_recall": fraud_recall,
        "fraud_precision": fraud_precision,
        "confusion_matrix": cm.tolist(),
        "pattern_recall": pattern_recall
    }
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

def print_ablation_table():
    out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")
    models = ["baseline", "cycle_rw", "cycle_attn"]
    results = {}

    for m in models:
        path = os.path.join(out_dir, f"evaluation_report_{m}.json" if m != "baseline" else "evaluation_report.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                results[m] = json.load(f)

    if not results:
        print("[!] No evaluation reports found.")
        return

    print("\n" + "=" * 90)
    print("                     CYCLE-AWARE AML GNN ABLATION TABLE")
    print("=" * 90)
    print(f"{'Configuration':<26} {'ROC-AUC':>8} {'PR-AUC':>8} {'Fraud Rec':>10} {'Alert Rec':>10} {'Maj Ring':>10} {'Full Ring':>10}")
    print("-" * 90)

    names = {
        "baseline": "1. Vanilla GraphSAGE",
        "cycle_rw": "2. + RW Cycle Features",
        "cycle_attn": "3. + Cycle Attention (Full)"
    }

    for m in models:
        if m in results:
            r = results[m]
            cm = r.get("confusion_matrix", [[0,0],[0,0]])
            rec = r.get("fraud_recall", cm[1][1] / max(cm[1][0] + cm[1][1], 1) if cm[1][0]+cm[1][1]>0 else 0) * 100
            pat = r.get("pattern_recall", {}).get("overall", {})
            r_any = pat.get("recall_any", pat.get("recall", 0))
            r_maj = pat.get("recall_majority", 0)
            r_full = pat.get("recall_full", 0)

            print(f"{names[m]:<26} {r.get('roc_auc', 0):>8.4f} {r.get('pr_auc', 0):>8.4f} {rec:>9.1f}% {r_any:>9.1f}% {r_maj:>9.1f}% {r_full:>9.1f}%")

    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="all", choices=["baseline", "cycle_rw", "cycle_attn", "all"])
    parser.add_argument("--table", action="store_true", help="Print comparative ablation table")
    args = parser.parse_args()

    if args.table:
        print_ablation_table()
    else:
        if args.eval == "all":
            for m in ["baseline", "cycle_rw", "cycle_attn"]:
                evaluate_model(m)
            print_ablation_table()
        else:
            evaluate_model(args.eval)
            print_ablation_table()
