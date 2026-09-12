import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.aml_pipeline.aml_config import (
    GRAPH_DIR, AML_OUTPUT_DIR, BATCH_SIZE, NUM_NEIGHBORS,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOP_PAT, LR_SCHED_PAT,
    HIDDEN_DIM, OUTPUT_DIM, SAGE_AGGR, NUM_SAGE_LAYERS, DROPOUT
)
from src.models.graphsage_model import AMLGraphSAGEModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss based on class distribution."""
    num_pos = y.sum().item()
    num_neg = len(y) - num_pos
    return torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float)

def train_epoch(model, loader, target_edge_attr, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Original edge attributes for the target edges
        # batch.input_id gives the index into the original target edges provided to the loader
        batch_edge_attr = target_edge_attr[batch.input_id.cpu()].to(device)
        
        # Forward pass
        # batch.edge_index is the message passing graph (subgraph)
        # batch.edge_label_index are the edges to classify (src, dst in subgraph node indices)
        logits = model(batch.x, batch.edge_index, batch_edge_attr, batch.edge_label_index)
        
        # Loss
        loss = criterion(logits, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.edge_label.size(0)
        
        # Metrics
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.edge_label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary)
    
    return avg_loss, roc_auc, f1

@torch.no_grad()
def evaluate_epoch(model, loader, target_edge_attr, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = batch.to(device)
        batch_edge_attr = target_edge_attr[batch.input_id.cpu()].to(device)
        
        logits = model(batch.x, batch.edge_index, batch_edge_attr, batch.edge_label_index)
        loss = criterion(logits, batch.edge_label.float())
        
        total_loss += loss.item() * batch.edge_label.size(0)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        labels = batch.edge_label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    avg_loss = total_loss / len(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary)
    
    return avg_loss, roc_auc, f1

def train_aml():
    print("=" * 60)
    print("  AML GraphSAGE Training Pipeline")
    print("=" * 60)
    
    device = get_device()
    print(f"  Device: {device}")
    
    # ── Load Data ──────────────────────────────────────────────
    print("\n  Loading graph data...")
    graph_path = os.path.join(GRAPH_DIR, "aml_graph.pt")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.size(1):,}")
    
    # Extract split edges for the loader
    train_edge_index = data.edge_index[:, data.train_mask]
    train_edge_label = data.y[data.train_mask]
    train_edge_attr = data.edge_attr[data.train_mask]
    
    val_edge_index = data.edge_index[:, data.val_mask]
    val_edge_label = data.y[data.val_mask]
    val_edge_attr = data.edge_attr[data.val_mask]
    
    print(f"  Train edges: {train_edge_index.size(1):,} | Val edges: {val_edge_index.size(1):,}")
    
    # ── Dataloaders ────────────────────────────────────────────
    # LinkNeighborLoader automatically samples subgraphs for the provided edges
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=train_edge_index,
        edge_label=train_edge_label,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=val_edge_index,
        edge_label=val_edge_label,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    
    # ── Model & Optimizer ──────────────────────────────────────
    node_feat_dim = data.x.size(1)
    edge_feat_dim = data.edge_attr.size(1)
    
    model = AMLGraphSAGEModel(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUTPUT_DIM,
        aggr=SAGE_AGGR,
        num_layers=NUM_SAGE_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    pos_weight = compute_pos_weight(train_edge_label).to(device)
    print(f"  Loss pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=LR_SCHED_PAT, factor=0.5)
    
    # ── Training Loop ──────────────────────────────────────────
    out_dir = os.path.join(AML_OUTPUT_DIR, "model_results")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_model.pt")
    
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}]")
        
        train_loss, train_auc, train_f1 = train_epoch(model, train_loader, train_edge_attr, optimizer, criterion, device)
        val_loss, val_auc, val_f1 = evaluate_epoch(model, val_loader, val_edge_attr, criterion, device)
        
        scheduler.step(val_auc)
        
        print(f"  Train Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auc": train_auc, "train_f1": train_f1,
            "val_loss": val_loss, "val_auc": val_auc, "val_f1": val_f1,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  [*] New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PAT:
                print(f"\n  [!] Early stopping triggered after {epoch} epochs.")
                break

    end_time = time.time()
    print(f"\n  Training completed in {(end_time - start_time) / 60:.1f} minutes.")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train_aml()
