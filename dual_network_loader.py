import pickle
from typing import List, Tuple

import numpy as np
from torch_geometric.data import Data


def load_pkl_graphs(path: str) -> List[Tuple[object, Data]]:
    """
    Load a graph snapshot pickle produced by `temporal_network_module.py` or `frm_network_module.py`.

    Expected format: list of (date, torch_geometric.data.Data) tuples.
    """
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list in {path}, got {type(graphs)}")
    return graphs


def align_graphs_by_date(
    bubble_graphs: List[Tuple[object, Data]],
    frm_graphs: List[Tuple[object, Data]],
) -> List[Tuple[np.datetime64, Data, Data]]:
    """
    Align bubble and FRM snapshots by exact date key.

    Note: these pickles contain PyG `Data` objects (node identities are integer indices), so
    alignment assumes consistent node indexing across snapshots for any downstream feature merge.
    """
    bubble_by_date = {np.datetime64(d): g for d, g in bubble_graphs}
    frm_by_date = {np.datetime64(d): g for d, g in frm_graphs}
    common_dates = sorted(set(bubble_by_date) & set(frm_by_date))
    return [(d, bubble_by_date[d], frm_by_date[d]) for d in common_dates]


def combine_node_features(bubble_g: Data, frm_g: Data) -> np.ndarray:
    """
    Combine node features by concatenating `x` along feature dimension.

    Requires matching `num_nodes` across graphs.
    """
    if bubble_g.num_nodes != frm_g.num_nodes:
        raise ValueError(f"num_nodes mismatch: bubble={bubble_g.num_nodes} frm={frm_g.num_nodes}")
    if bubble_g.x is None or frm_g.x is None:
        raise ValueError("Both graphs must contain `x` node features to combine.")
    return np.concatenate([bubble_g.x.detach().cpu().numpy(), frm_g.x.detach().cpu().numpy()], axis=1)


if __name__ == "__main__":
    bubble = load_pkl_graphs("temporal_graphs.pkl")
    frm = load_pkl_graphs("frm_graphs.pkl")
    aligned = align_graphs_by_date(bubble, frm)
    print(f"Aligned snapshots: {len(aligned)}")
