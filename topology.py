"""Topological feature computation for the Latent Peer Network.

Computes node-level structural descriptors (degree centrality, PageRank,
hub scores) that are used by the Topology-Aware Edge Refiner in Tier 2
of the SARE-GT architecture.
"""

import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, degree


def compute_topological_features(data, feature_list=None):
    """Compute topological features for every node in the graph.

    Args:
        data: PyG ``Data`` object with ``edge_index`` and ``num_nodes``.
        feature_list (list[str]): Features to compute.  Supported values:
            ``'degree'``, ``'pagerank'``, ``'hub_score'``.

    Returns:
        tuple: ``(topo_features, feature_names)`` where *topo_features* is a
        ``Tensor`` of shape ``[num_nodes, num_features]`` and *feature_names*
        is a list of strings.
    """
    if feature_list is None:
        feature_list = ["degree", "pagerank", "hub_score"]

    try:
        G = to_networkx(data, to_undirected=True)
    except Exception:
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))

    n_nodes = data.num_nodes
    features, names = [], []

    for name in feature_list:
        if name == "degree":
            values = _compute_degree(data, G, n_nodes)
        elif name == "pagerank":
            values = _compute_pagerank(G, n_nodes)
        elif name == "hub_score":
            values = _compute_hub_score(G, n_nodes)
        else:
            continue
        features.append(values)
        names.append(name)

    if not features:
        return torch.zeros(n_nodes, 1, dtype=torch.float), ["zero"]

    return torch.tensor(np.column_stack(features), dtype=torch.float), names


# ---- helpers ---------------------------------------------------------------

def _compute_degree(data, G, n_nodes):
    """Normalised node degree."""
    try:
        if hasattr(data, "edge_index"):
            deg = degree(data.edge_index[0], num_nodes=n_nodes).float()
            mx = deg.max()
            return (deg / mx).numpy() if mx > 0 else np.zeros(n_nodes)
        deg_dict = dict(G.degree())
        vals = [deg_dict.get(i, 0) for i in range(n_nodes)]
        mx = max(vals) if vals else 1
        return [v / mx for v in vals]
    except Exception:
        return [0.0] * n_nodes


def _compute_pagerank(G, n_nodes):
    """Normalised PageRank scores."""
    try:
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            pr = nx.pagerank(G, max_iter=100, tol=1e-4)
            vals = [pr.get(i, 0) for i in range(n_nodes)]
            mx = max(vals) if vals else 1
            return [v / mx for v in vals]
        return [1.0 / n_nodes] * n_nodes
    except Exception:
        return [1.0 / n_nodes] * n_nodes


def _compute_hub_score(G, n_nodes):
    """Normalised HITS hub scores."""
    try:
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            hubs, _ = nx.hits(G, max_iter=100)
            vals = [hubs.get(i, 0) for i in range(n_nodes)]
            mx = max(vals) if vals else 1
            return [v / mx for v in vals] if mx > 0 else [0.0] * n_nodes
        return [0.0] * n_nodes
    except Exception:
        return [0.0] * n_nodes
