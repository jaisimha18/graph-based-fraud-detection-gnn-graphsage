import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Node Encoder
    Takes account node features + edge_index and produces node embeddings.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 aggr: str = "mean", num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Final layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class EdgeClassifier(nn.Module):
    """
    Edge Classifier Head
    Takes concatenated embeddings (src_node, dst_node, edge_features) and outputs laundering probability (logit).
    """
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        # Input: z_src + z_dst + edge_attr
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
        # Concatenate src embedding, dst embedding, and edge features
        edge_repr = torch.cat([z_src, z_dst, edge_attr], dim=1)
        return self.mlp(edge_repr).squeeze(-1)

class AMLGraphSAGEModel(nn.Module):
    """
    Full AML Model: GraphSAGE Encoder + Edge Classifier
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int = 128, out_dim: int = 64,
                 aggr: str = "mean", num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = GraphSAGEEncoder(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            aggr=aggr,
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
                edge_attr: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a mini-batch of edges.
        
        Args:
            x: Node features for the subgraph [N, node_feat_dim]
            edge_index: Message passing edges [2, E_mp]
            edge_attr: Edge features for the target edges to predict [E_target, edge_feat_dim]
            edge_label_index: Edges to classify (src and dst node indices in the subgraph) [2, E_target]
            
        Returns:
            logits: Unnormalized prediction scores [E_target]
        """
        # 1. Encode all nodes in the sampled subgraph
        z = self.encoder(x, edge_index)
        
        # 2. Extract embeddings for the nodes involved in the target edges
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]
        
        z_src = z[src_idx]
        z_dst = z[dst_idx]
        
        # 3. Classify edges
        logits = self.classifier(z_src, z_dst, edge_attr)
        
        return logits
