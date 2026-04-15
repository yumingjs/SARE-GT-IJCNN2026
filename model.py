"""
SARE-GT: Structure-Aware Robust Enhancement Graph Transformer
for Corporate Credit Rating.

This module implements the full SARE-GT architecture consisting of:
  - Tier 1: Feature Interaction Encoder (Learnable Meta-Graph Convolution)
  - Tier 2: Local Expert (Topology-Aware Dual-Channel GAT Aggregation)
  - Tier 3: Global Expert (Multi-Layer Graph Transformer)
  - Final Verdict: Adaptive ExpertGate Fusion and Classification

Reference:
    Kong et al., "SARE-GT: Structure-Aware Robust Enhancement Graph
    Transformer for Corporate Credit Rating", IJCNN 2026.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# Tier 3 – Global Expert: Graph Transformer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer with positional encoding support.

    Treats the entire graph as a sequence and applies standard multi-head
    self-attention, enabling each node to attend to all other nodes for
    capturing long-range dependencies.

    Args:
        embed_dim (int): Embedding / hidden dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, h, data=None):
        """
        Args:
            h (Tensor): Node features of shape ``[N, embed_dim]``.
            data: Optional PyG ``Data`` object carrying positional encodings
                  in ``data.pe``.

        Returns:
            Tensor: Globally-enhanced node features of shape ``[N, embed_dim]``.
        """
        if data is not None and hasattr(data, "pe") and data.pe is not None:
            h_with_pe = h + self.pos_proj(data.pe)
        else:
            h_with_pe = h

        # Self-attention over the full graph (batch_size=1, seq_len=N)
        h_batch = h_with_pe.unsqueeze(0)
        attn_output, _ = self.attention(h_batch, h_batch, h_batch)

        h = h + self.dropout(attn_output.squeeze(0))
        h = self.norm1(h)

        h = h + self.dropout(self.ffn(h))
        return self.norm2(h)


class MultiLayerGraphTransformer(nn.Module):
    """Stack of :class:`GraphTransformerLayer` for deeper global modelling.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads per layer.
        num_layers (int): Number of transformer layers.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h, data=None):
        for layer in self.layers:
            h = layer(h, data)
        return h


# ---------------------------------------------------------------------------
# Tier 1 – Feature Interaction Encoder (Learnable Meta-Graph)
# ---------------------------------------------------------------------------

class FeatureInteractionEncoder(nn.Module):
    """Captures latent relationships among financial indicators via a
    learnable, globally-shared feature adjacency matrix.

    Implements Eq. (1) in the paper:
        X_enhanced = LayerNorm(X + MLP(X · softmax(ReLU(A_feat))))

    Args:
        num_features (int): Number of input features (``d_0``).
        gcn_hidden_channels (int): Hidden channels for the GCN-like transform.
    """

    def __init__(self, num_features: int = 39, gcn_hidden_channels: int = 39):
        super().__init__()
        self.num_features = num_features

        # Learnable feature adjacency matrix
        self.feature_adj = nn.Parameter(torch.randn(num_features, num_features))
        nn.init.xavier_uniform_(self.feature_adj)

        self.gcn_transform1 = nn.Linear(num_features, gcn_hidden_channels)
        self.gcn_transform2 = nn.Linear(gcn_hidden_channels, num_features)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        """
        Args:
            x (Tensor): Firm feature matrix of shape ``[N, num_features]``.

        Returns:
            Tensor: Enhanced feature matrix of shape ``[N, num_features]``.
        """
        adj_normalized = F.softmax(F.relu(self.feature_adj), dim=1)
        x_aggregated = torch.matmul(x, adj_normalized)

        h = self.gcn_transform1(x_aggregated)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gcn_transform2(h)

        return self.layer_norm(x + h)


# ---------------------------------------------------------------------------
# Tier 2 – Local Expert helpers
# ---------------------------------------------------------------------------

class TopologyAwareEdgeRefiner(nn.Module):
    """Topology-aware edge refinement producing a dynamic homophily
    confidence score for each edge.

    Combines semantic features, topological features, and original edge
    weights via an MLP (cf. Eq. (2) in the paper).

    Args:
        feature_dim (int): Semantic feature dimension.
        topo_dim (int): Topological feature dimension.
        hidden_dim (int): Hidden dimension of the scoring MLP.
    """

    def __init__(self, feature_dim: int, topo_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.topo_dim = topo_dim

        input_dim = (feature_dim + topo_dim) * 2 + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        if topo_dim > 0:
            self.topo_processor = nn.Sequential(
                nn.Linear(topo_dim, topo_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

    def forward(self, h_semantic, h_topo, edge_index, edge_weight=None):
        """
        Args:
            h_semantic (Tensor): Node semantic features ``[N, feature_dim]``.
            h_topo (Tensor | None): Node topological features ``[N, topo_dim]``.
            edge_index (Tensor): Edge indices ``[2, E]``.
            edge_weight (Tensor | None): Original edge weights ``[E]``.

        Returns:
            Tensor: Refined confidence scores ``[E]`` in ``(0, 1)``.
        """
        row, col = edge_index
        edge_feat = torch.cat([h_semantic[row], h_semantic[col]], dim=1)

        if self.topo_dim > 0 and h_topo is not None:
            if hasattr(self, "topo_processor"):
                h_topo = self.topo_processor(h_topo)
            edge_topo = torch.cat([h_topo[row], h_topo[col]], dim=1)
            edge_feat = torch.cat([edge_feat, edge_topo], dim=1)

        if edge_weight is not None:
            ew = edge_weight.unsqueeze(1)
        else:
            ew = torch.ones(edge_feat.size(0), 1, device=edge_feat.device)

        return self.edge_mlp(torch.cat([edge_feat, ew], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Adaptive ExpertGate
# ---------------------------------------------------------------------------

class ExpertGate(nn.Module):
    """Learns per-node gating weights to adaptively fuse local and global
    expert outputs (cf. Eq. (7) in the paper).

    Args:
        hidden_dim (int): Input feature dimension.
        num_experts (int): Number of expert branches.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 2):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, h):
        """
        Args:
            h (Tensor): Node features ``[N, hidden_dim]``.

        Returns:
            Tensor: Expert weights ``[N, num_experts]``.
        """
        return self.gate_network(h)


# ---------------------------------------------------------------------------
# Full SARE-GT Model
# ---------------------------------------------------------------------------

class SARE_GT(nn.Module):
    """SARE-GT: Structure-Aware Robust Enhancement Graph Transformer.

    Hierarchical evidence fusion architecture with three progressive tiers:

    * **Tier 1** – Intra-Feature Systemics via learnable meta-graph.
    * **Tier 2** – Local Expert with topology-aware dual-channel GAT.
    * **Tier 3** – Global Expert via multi-layer Graph Transformer.
    * **Fusion** – Adaptive ExpertGate with gating entropy regularisation.

    Args:
        input_dim (int): Raw feature dimension (default: 39).
        hidden_dim (int): Hidden dimension (default: 128).
        output_dim (int): Number of rating classes (default: 9).
        num_heads (int): Number of GAT attention heads (default: 4).
        num_layers (int): Number of GAT layers (default: 3).
        topo_dim (int): Dimension of topological features.
        pe_dim (int): Dimension of positional encodings.
        transformer_layers (int): Number of Graph Transformer layers.
    """

    def __init__(self, input_dim: int = 39, hidden_dim: int = 128,
                 output_dim: int = 9, num_heads: int = 4, num_layers: int = 3,
                 topo_dim: int = 0, pe_dim: int = 16,
                 transformer_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.topo_dim = topo_dim
        self.pe_dim = pe_dim

        # Tier 1: Feature Interaction Encoder
        self.feature_enhancer = FeatureInteractionEncoder(
            num_features=input_dim
        )
        self.initial_transform = nn.Linear(input_dim, hidden_dim)

        # Tier 2: Local Expert – Topology-aware dual-channel GATs
        self.edge_refiner = TopologyAwareEdgeRefiner(
            feature_dim=hidden_dim, topo_dim=topo_dim, hidden_dim=hidden_dim
        )
        self.local_trust_gats = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    dropout=0.1)
            for _ in range(num_layers)
        ])
        self.local_disc_gats = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fusion_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=1),
            )
            for _ in range(num_layers)
        ])
        self.local_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Tier 3: Global Expert – Graph Transformer
        self.global_transformer = MultiLayerGraphTransformer(
            embed_dim=hidden_dim,
            num_heads=max(1, num_heads // 2),
            num_layers=transformer_layers,
            dropout=0.1,
        )

        # ExpertGate & Fusion
        self.expert_gate = ExpertGate(hidden_dim, num_experts=2)
        self.expert_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Classifier: h0 ‖ h_local ‖ h_global ‖ h_fused → logits
        classifier_input_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, data):
        """
        Args:
            data: PyG ``Data`` object with attributes ``x``, ``edge_index``,
                  and optionally ``edge_weight``, ``topo_features``, ``pe``.

        Returns:
            tuple: ``(logits, h_fused)`` where *logits* has shape
            ``[N, output_dim]`` and *h_fused* has shape ``[N, hidden_dim]``.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        topo_features = getattr(data, "topo_features", None)

        # Tier 1: Feature interaction encoding
        enhanced_x = self.feature_enhancer(x)
        h0 = F.relu(self.initial_transform(enhanced_x))

        # Tier 2: Local Expert – edge refinement + dual-channel GAT
        edge_conf = self.edge_refiner(
            h_semantic=h0, h_topo=topo_features,
            edge_index=edge_index, edge_weight=edge_weight,
        )

        h_trust = h0
        h_disc = h0
        for i in range(self.num_layers):
            h_trust_new = self.local_trust_gats[i](
                h_trust, edge_index, edge_attr=edge_conf)
            h_disc_new = self.local_disc_gats[i](
                h_disc, edge_index, edge_attr=1.0 - edge_conf)

            h_cat = torch.cat([h_trust_new, h_disc_new], dim=1)
            fw = self.fusion_attentions[i](h_cat)
            h_fused = fw[:, 0:1] * h_trust_new + fw[:, 1:2] * h_disc_new

            if i > 0:
                h_fused = h_fused + h_trust
            h_fused = self.local_layer_norms[i](h_fused)
            h_trust = h_fused
            h_disc = h_fused

        h_local = h_fused

        # Tier 3: Global Expert – Graph Transformer
        h_global = self.global_transformer(h0, data)

        # Adaptive evidence fusion via ExpertGate
        gate_scores = self.expert_gate(h0)
        h_expert_fused = (gate_scores[:, 0:1] * h_local
                          + gate_scores[:, 1:2] * h_global)

        # Final classification
        final_features = torch.cat(
            [h0, h_local, h_global, h_expert_fused], dim=1)
        logits = self.classifier(final_features)

        return logits, h_expert_fused
