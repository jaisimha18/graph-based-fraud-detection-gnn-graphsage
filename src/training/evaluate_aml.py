import os
import sys
import json
import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import (
    GRAPH_DIR, AML_OUTPUT_DIR, BATCH_SIZE, NUM_NEIGHBORS,
    HIDDEN_DIM, OUTPUT_DIM, SAGE_AGGR, NUM_SAGE_LAYERS, DROPOUT, THRESHOLD_SWEEP,
    PATTERNS_TXT, TRANSACTIONS_CSV, PREPROCESSED_DIR,
    TRAIN_END_DAY, VAL_DAY, TEST_START_DAY,
)
from src.models.graphsage_model import AMLGraphSAGEModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def get_predictions(model, loader, target_edge_attr, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Inference", leave=False):
        batch = batch.to(device)
        batch_edge_attr = target_edge_attr[batch.input_id.cpu()].to(device)
        
        logits = model(batch.x, batch.edge_index, batch_edge_attr, batch.edge_label_index)
        preds = torch.sigmoid(logits).cpu().numpy()
        labels = batch.edge_label.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    return np.array(all_preds), np.array(all_labels)

def evaluate_aml():
    print("=" * 60)
    print("  AML GraphSAGE Evaluation Pipeline")
    print("=" * 60)
    
    device = get_device()
    out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")
    model_path = os.path.join(out_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"  [!] Best model not found at {model_path}. Train the model first.")
        return
        
    # ── Load Data ──────────────────────────────────────────────
    print("\n  Loading graph data...")
    graph_path = os.path.join(GRAPH_DIR, "aml_graph.pt")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    
    val_edge_index = data.edge_index[:, data.val_mask]
    val_edge_label = data.y[data.val_mask]
    val_edge_attr = data.edge_attr[data.val_mask]
    
    test_edge_index = data.edge_index[:, data.test_mask]
    test_edge_label = data.y[data.test_mask]
    test_edge_attr = data.edge_attr[data.test_mask]
    
    val_loader = LinkNeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, edge_label_index=val_edge_index,
        edge_label=val_edge_label, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )
    test_loader = LinkNeighborLoader(
        data, num_neighbors=NUM_NEIGHBORS, edge_label_index=test_edge_index,
        edge_label=test_edge_label, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )
    
    # ── Load Model ─────────────────────────────────────────────
    node_feat_dim = data.x.size(1)
    edge_feat_dim = data.edge_attr.size(1)
    
    model = AMLGraphSAGEModel(
        node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
        hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, aggr=SAGE_AGGR,
        num_layers=NUM_SAGE_LAYERS, dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # ── Threshold Optimization on Validation Set ───────────────
    print("\n  Optimizing threshold on Validation Set...")
    val_preds, val_labels = get_predictions(model, val_loader, val_edge_attr, device)
    
    best_f1 = 0
    best_thresh = 0.5
    for thresh in THRESHOLD_SWEEP:
        bin_preds = (val_preds > thresh).astype(int)
        f1 = f1_score(val_labels, bin_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"  Best Val Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    
    # ── Test Set Evaluation ────────────────────────────────────
    print("\n  Evaluating on Test Set...")
    test_preds, test_labels = get_predictions(model, test_loader, test_edge_attr, device)
    
    # 1. Base Metrics
    test_roc_auc = roc_auc_score(test_labels, test_preds)
    precision_arr, recall_arr, _ = precision_recall_curve(test_labels, test_preds)
    test_pr_auc = auc(recall_arr, precision_arr)
    
    test_bin_preds = (test_preds > best_thresh).astype(int)
    test_f1 = f1_score(test_labels, test_bin_preds)
    
    print(f"  Test ROC-AUC : {test_roc_auc:.4f}")
    print(f"  Test PR-AUC  : {test_pr_auc:.4f}")
    print(f"  Test F1      : {test_f1:.4f} (at threshold {best_thresh:.2f})")
    
    # 2. Confusion Matrix & Report
    cm = confusion_matrix(test_labels, test_bin_preds)
    report = classification_report(test_labels, test_bin_preds, target_names=["Legit", "Laundering"], output_dict=True)
    
    # ── Visualizations ─────────────────────────────────────────
    print("\n  Generating plots...")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'GraphSAGE (AUC = {test_roc_auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AML Test Set')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_arr, precision_arr, label=f'GraphSAGE (AUC = {test_pr_auc:.3f})', lw=2, color='darkorange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - AML Test Set')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Laundering"], yticklabels=["Legit", "Laundering"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (Threshold = {best_thresh:.2f})')
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── Pattern-Level Recall ──────────────────────────────────
    pattern_recall = compute_pattern_recall(
        test_edge_index, test_bin_preds, data, best_thresh, out_dir
    )

    # Save Report
    report_data = {
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        "best_threshold": best_thresh,
        "f1_score": test_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "pattern_recall": pattern_recall,
    }
    with open(os.path.join(out_dir, "evaluation_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\n  Plots and report saved to {out_dir}/")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────
# PATTERN-LEVEL RECALL
# ─────────────────────────────────────────────────────────────

def parse_patterns(patterns_path: str) -> list:
    """
    Parse HI-Small_Patterns.txt and return a list of pattern dicts.

    Each pattern dict contains:
      - pattern_type : str (e.g. 'CYCLE', 'FAN-OUT', 'STACK')
      - description  : str (the full BEGIN line descriptor)
      - transactions : list of (timestamp, from_bank, from_acc, to_bank, to_acc)
    """
    patterns = []
    current_pattern = None

    with open(patterns_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                # Extract pattern type from line like:
                # "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops"
                # or "BEGIN LAUNDERING ATTEMPT - STACK" (no colon)
                match = re.match(
                    r"BEGIN LAUNDERING ATTEMPT - ([^:\n]+?)(?::\s*(.*))?$", line
                )
                if match:
                    ptype = match.group(1).strip()
                    desc = (match.group(2) or "").strip()
                else:
                    ptype = "UNKNOWN"
                    desc = line
                current_pattern = {
                    "pattern_type": ptype,
                    "description": desc,
                    "transactions": [],
                }
                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                if current_pattern and current_pattern["transactions"]:
                    patterns.append(current_pattern)
                current_pattern = None
                continue

            # Transaction line: timestamp,from_bank,from_acc,to_bank,to_acc,...
            if current_pattern is not None:
                parts = line.split(",")
                if len(parts) >= 5:
                    current_pattern["transactions"].append({
                        "timestamp": parts[0],
                        "from_bank": parts[1],
                        "from_account": parts[2],
                        "to_bank": parts[3],
                        "to_account": parts[4],
                    })

    return patterns


def compute_pattern_recall(
    test_edge_index, test_bin_preds, data, threshold, out_dir
) -> dict:
    """
    Compute pattern-level recall using HI-Small_Patterns.txt.

    For each laundering pattern that has at least one edge in the
    test set, we check if the model detected it. A pattern is
    considered "detected" if at least one of its test-set edges
    was predicted positive (score > threshold).

    Prints and returns per-pattern-type and overall recall.
    """
    print("\n" + "=" * 60)
    print("  PATTERN-LEVEL RECALL")
    print("=" * 60)

    if not os.path.exists(PATTERNS_TXT):
        print(f"  [!] Patterns file not found: {PATTERNS_TXT}")
        return {}

    # Parse patterns
    patterns = parse_patterns(PATTERNS_TXT)
    print(f"  Total laundering patterns parsed: {len(patterns)}")

    # Build node index → account_id reverse mapping
    mapping_path = os.path.join(
        os.path.dirname(PREPROCESSED_DIR.rstrip("/")),
        "graph", "node_mappings.json"
    )
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            account_to_idx = json.load(f)
    else:
        print(f"  [!] Node mappings not found at {mapping_path}")
        return {}

    # Build reverse mapping: idx -> account_id
    idx_to_account = {int(v): k for k, v in account_to_idx.items()}

    # Build a set of (from_acc, to_acc) edges that were predicted positive
    # in the test set
    test_ei_np = test_edge_index.cpu().numpy()
    predicted_positive_edges = set()
    for i in range(test_ei_np.shape[1]):
        if test_bin_preds[i] == 1:
            from_idx = int(test_ei_np[0, i])
            to_idx = int(test_ei_np[1, i])
            from_acc = idx_to_account.get(from_idx, "")
            to_acc = idx_to_account.get(to_idx, "")
            predicted_positive_edges.add((from_acc, to_acc))

    # Also build set of ALL test edges (for checking which pattern
    # transactions actually fall in the test set)
    all_test_edges = set()
    for i in range(test_ei_np.shape[1]):
        from_idx = int(test_ei_np[0, i])
        to_idx = int(test_ei_np[1, i])
        from_acc = idx_to_account.get(from_idx, "")
        to_acc = idx_to_account.get(to_idx, "")
        all_test_edges.add((from_acc, to_acc))

    # Evaluate each pattern
    results_by_type = {}  # pattern_type -> {total, detected}
    total_patterns_in_test = 0
    detected_patterns = 0

    for pat in patterns:
        ptype = pat["pattern_type"]
        if ptype not in results_by_type:
            results_by_type[ptype] = {"total": 0, "detected": 0}

        # Find which of this pattern's transactions are in the test set
        pattern_test_edges = []
        for txn in pat["transactions"]:
            edge_key = (txn["from_account"], txn["to_account"])
            if edge_key in all_test_edges:
                pattern_test_edges.append(edge_key)

        # Only count patterns that have at least one edge in the test set
        if not pattern_test_edges:
            continue

        total_patterns_in_test += 1
        results_by_type[ptype]["total"] += 1

        # Pattern is detected if ANY of its test edges were predicted positive
        detected = any(e in predicted_positive_edges for e in pattern_test_edges)
        if detected:
            detected_patterns += 1
            results_by_type[ptype]["detected"] += 1

    # Print results
    print(f"  Patterns with ≥1 test edge: {total_patterns_in_test}")
    print(f"\n  {'Pattern Type':<20} {'In Test':>8} {'Detected':>10} {'Recall':>10}")
    print(f"  {'-'*52}")
    for ptype in sorted(results_by_type.keys()):
        r = results_by_type[ptype]
        if r["total"] > 0:
            recall = r["detected"] / r["total"] * 100
            print(f"  {ptype:<20} {r['total']:>8} {r['detected']:>10} {recall:>9.1f}%")

    overall_recall = (
        detected_patterns / total_patterns_in_test * 100
        if total_patterns_in_test > 0 else 0.0
    )
    print(f"  {'-'*52}")
    print(f"  {'OVERALL':<20} {total_patterns_in_test:>8} {detected_patterns:>10} {overall_recall:>9.1f}%")
    print(f"\n  Pattern-level recall: {overall_recall:.1f}%")
    print("=" * 60)

    # Build return dict
    pattern_recall_data = {
        "overall": {
            "total_in_test": total_patterns_in_test,
            "detected": detected_patterns,
            "recall": round(overall_recall, 2),
        },
        "by_type": {},
    }
    for ptype, r in results_by_type.items():
        if r["total"] > 0:
            pattern_recall_data["by_type"][ptype] = {
                "total": r["total"],
                "detected": r["detected"],
                "recall": round(r["detected"] / r["total"] * 100, 2),
            }

    return pattern_recall_data


if __name__ == "__main__":
    evaluate_aml()
