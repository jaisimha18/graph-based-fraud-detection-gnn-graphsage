# ============================================================
# src/models/cycle_graphsage_model.py
#
# Phase 3: Cycle-Aware GraphSAGE Architecture
#
# Components:
#   1. CycleAwareSAGEConv: GraphSAGE Convolution with Cycle-Conditioned
#      Neighborhood Attention (Component B).
#   2. CycleAwareGraphSAGEEncoder: Multi-layer encoder with cycle awareness.
#   3. EdgeClassifier: MLP Edge head with 22-dim augmented edge features.
#   4. CycleAwareAMLModel: Unified End-to-End Cycle-Aware Model.
#
# Ablation Modes:
#   - use_cycle_attention=False : Vanilla GraphSAGE with RW cycle edge features (Ablation 1)
#   - use_cycle_attention=True  : Full Cycle-Aware Attention GraphSAGE (Ablation 2)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import softmax

class CycleAwareSAGEConv(MessagePassing):
    """
    GraphSAGE Convolution with Cycle-Conditioned Neighborhood Attention.
    
    Instead of uniform mean aggregation, it weights neighbor messages using
    an attention mechanism conditioned on:
      - Source node embedding h_j
      - Target node embedding h_i
      - Cycle return probability edge features e_ji
    """
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int, dropout: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.dropout = dropout

        # Linear transformations
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_channels, out_channels)

        # Attention scoring MLP: [h_j || h_i || e_ji] -> scalar score
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels * 2 + edge_channels, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        out_self = self.lin_self(x)
        
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_channels), device=x.device)
            
        out_neigh = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_self + out_neigh

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor, size_i=None) -> torch.Tensor:
        h_j = self.lin_neigh(x_j)
        h_i = self.lin_self(x_i)
        e_proj = self.lin_edge(edge_attr)

        # Condition attention on both node states and edge cycle metrics
        attn_input = torch.cat([h_j, h_i, edge_attr], dim=-1)
        scores = self.attn_mlp(attn_input)

        # Softmax normalize across incoming neighbors for each target node
        alpha = softmax(scores, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha * (h_j + e_proj)


class CycleAwareGraphSAGEEncoder(nn.Module):
    """
    Cycle-Aware GraphSAGE Node Encoder.
    Supports either Cycle-Aware Attention or Standard SAGEConv.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 edge_channels: int = 22, use_cycle_attention: bool = True,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cycle_attention = use_cycle_attention

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Layer 1
        if use_cycle_attention:
            self.convs.append(CycleAwareSAGEConv(in_channels, hidden_channels, edge_channels, dropout=dropout))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            if use_cycle_attention:
                self.convs.append(CycleAwareSAGEConv(hidden_channels, hidden_channels, edge_channels, dropout=dropout))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final Layer
        if use_cycle_attention:
            self.convs.append(CycleAwareSAGEConv(hidden_channels, out_channels, edge_channels, dropout=dropout))
        else:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="mean"))
        self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        for i in range(self.num_layers):
            if self.use_cycle_attention:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeClassifier(nn.Module):
    """
    Edge Classifier Head.
    Input: [z_src || z_dst || edge_attr] -> probability logit.
    """
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        in_dim = node_emb_dim * 2 + edge_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_repr = torch.cat([z_src, z_dst, edge_attr], dim=1)
        return self.mlp(edge_repr).squeeze(-1)


class CycleAwareAMLModel(nn.Module):
    """
    Full AML Model with Cycle-Aware Extensions.
    """
    def __init__(self, node_feat_dim: int = 8, edge_feat_dim: int = 22,
                 hidden_dim: int = 128, out_dim: int = 64,
                 use_cycle_attention: bool = True,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.use_cycle_attention = use_cycle_attention

        self.encoder = CycleAwareGraphSAGEEncoder(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            edge_channels=edge_feat_dim,
            use_cycle_attention=use_cycle_attention,
            num_layers=num_layers,
            dropout=dropout
        )

        self.classifier = EdgeClassifier(
            node_emb_dim=out_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_edge_attr: torch.Tensor, edge_label_index: torch.Tensor,
                subgraph_edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for mini-batch:
          - x: Node features for the sampled subgraph [N, node_feat_dim]
          - edge_index: Message passing edges [2, E_sub]
          - batch_edge_attr: Target edge attributes [E_target, edge_feat_dim]
          - edge_label_index: Target edge endpoint indices [2, E_target]
          - subgraph_edge_attr: Attributes for message passing edges [E_sub, edge_feat_dim]
        """
        # 1. Encode nodes using subgraph topology and cycle features
        if self.use_cycle_attention:
            z = self.encoder(x, edge_index, subgraph_edge_attr)
        else:
            z = self.encoder(x, edge_index)

        # 2. Extract endpoints for target edges
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]
        z_src = z[src_idx]
        z_dst = z[dst_idx]

        # 3. Classify edges
        logits = self.classifier(z_src, z_dst, batch_edge_attr)
        return logits
