"""Dataset construction for the Latent Peer Network.

Builds a k-NN-based latent peer network from raw financial features, computes
Laplacian Positional Encodings and topological node descriptors, and packages
everything as a PyG ``InMemoryDataset``.
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from topology import compute_topological_features


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_features(data):
    """Min-max normalisation with numerical stability."""
    min_vals = data.min(0)
    max_vals = data.max(0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)


def compute_laplacian_pe(edge_index, num_nodes, pe_dim=16):
    """Compute Laplacian Positional Encoding (LapPE).

    Returns the ``pe_dim`` smallest non-trivial eigenvectors of the graph
    Laplacian, following Dwivedi & Bresson (2020).

    Args:
        edge_index (ndarray): ``[2, E]`` edge indices.
        num_nodes (int): Number of nodes.
        pe_dim (int): Desired PE dimension.

    Returns:
        ndarray: Positional encoding of shape ``[num_nodes, pe_dim]``.
    """
    try:
        adj = sp.coo_matrix(
            (np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        )
        adj = adj + adj.T  # symmetrise
        D = sp.diags(np.array(adj.sum(axis=1)).flatten())
        L = D - adj

        k = min(pe_dim + 1, num_nodes - 1)
        eigenvalues, eigenvectors = eigs(L, k=k, which="SM")

        idx = np.argsort(eigenvalues.real)
        eigenvectors = eigenvectors[:, idx]

        pe = eigenvectors[:, 1:min(pe_dim + 1, eigenvectors.shape[1])].real
        if pe.shape[1] < pe_dim:
            pe = np.hstack([pe, np.zeros((num_nodes, pe_dim - pe.shape[1]))])
        return pe.astype(np.float32)
    except Exception:
        return np.zeros((num_nodes, pe_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CCRPeerNetworkDataset(InMemoryDataset):
    """PyG ``InMemoryDataset`` for the corporate credit-rating task.

    Constructs a weighted k-NN latent peer network from raw financial
    features, augments nodes with Laplacian PE and topological descriptors,
    and stores the result for efficient reuse.

    Args:
        root (str): Root directory (expects ``raw/data.npz``).
        k_neighbors (int): Number of neighbours for the k-NN graph.
        use_pe (bool): Whether to compute Laplacian Positional Encoding.
        pe_dim (int): PE dimension.
        use_topo (bool): Whether to compute topological features.
        topo_features (list[str]): Topological features to compute.
        transform: Optional PyG transform.
        pre_transform: Optional PyG pre-transform.
    """

    def __init__(self, root, k_neighbors=10, use_pe=True, pe_dim=16,
                 use_topo=True,
                 topo_features=("degree", "pagerank", "hub_score"),
                 transform=None, pre_transform=None):
        self.k_neighbors = k_neighbors
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.use_topo = use_topo
        self.topo_features = list(topo_features)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        topo_tag = ("_topo" + "".join(self.topo_features)) if self.use_topo else ""
        pe_tag = self.pe_dim if self.use_pe else 0
        return [f"CCRPeerNetwork_k{self.k_neighbors}_pe{pe_tag}{topo_tag}.pt"]

    def download(self):
        pass  # raw data must be placed manually

    def process(self):
        raw = np.load(self.raw_paths[0])
        train_x, test_x = raw["train_x"], raw["test_x"]
        train_y, test_y = raw["train_y"], raw["test_y"]

        all_x = np.concatenate([train_x, test_x])
        all_y = np.concatenate([train_y, test_y])

        norm_x = normalize_features(all_x)

        # Build weighted k-NN graph
        knn = kneighbors_graph(
            norm_x, n_neighbors=self.k_neighbors,
            mode="distance", include_self=False,
        )
        knn = knn + knn.T  # undirected
        knn.data = 1.0 / (knn.data + 1e-8)  # distance → similarity

        coo = knn.tocoo()
        edge_index = np.vstack([coo.row, coo.col])
        edge_weights = coo.data

        temp_data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=len(all_x),
        )

        # Topological features
        topo_tensor, topo_dim = None, 0
        if self.use_topo:
            try:
                topo_tensor, _ = compute_topological_features(
                    temp_data, feature_list=self.topo_features)
                topo_dim = topo_tensor.shape[1]
            except Exception:
                topo_tensor = torch.zeros(len(all_x), 1)
                topo_dim = 1

        # Laplacian Positional Encoding
        features = all_x
        if self.use_pe:
            pe = compute_laplacian_pe(edge_index, len(all_x), self.pe_dim)
            features = np.hstack([all_x, pe])

        n = len(all_x)
        train_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:len(train_x)] = True
        test_mask[len(train_x):] = True

        data_obj = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_weight=torch.tensor(edge_weights, dtype=torch.float),
            y=torch.tensor(all_y, dtype=torch.long),
            train_mask=train_mask,
            test_mask=test_mask,
        )

        if self.use_topo and topo_tensor is not None:
            data_obj.topo_features = topo_tensor
            data_obj.topo_dim = topo_dim
        else:
            data_obj.topo_features = torch.zeros(n, 1)
            data_obj.topo_dim = 0

        data, slices = self.collate([data_obj])
        torch.save((data, slices), self.processed_paths[0])
