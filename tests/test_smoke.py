import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_validation_network_diagnostics_writes_nonempty_csv(tmp_path: Path) -> None:
    from bubbles_networks.network_aggregate import build_aggregate_overlap_graph
    from bubbles_networks.validation import write_network_diagnostics

    repo = _repo_root()

    bubble_file = repo / "data" / "ro" / "ResultResults_ro_bet_bubbles.xlsx"
    if bubble_file.exists():
        G = build_aggregate_overlap_graph(bubble_file=str(bubble_file))
    else:
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)

    # Minimal synthetic temporal graphs (avoid heavy recomputation in tests)
    import networkx as nx
    from torch_geometric.utils import from_networkx

    g1 = nx.DiGraph()
    g1.add_nodes_from([0, 1, 2])
    g1.add_edge(0, 1, weight=1.0)
    g2 = nx.DiGraph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edge(1, 2, weight=2.0)

    temporal = [
        (pd.Timestamp("2020-01-01"), from_networkx(g1)),
        (pd.Timestamp("2020-01-02"), from_networkx(g2)),
    ]
    temporal_pkl = tmp_path / "temporal_graphs.pkl"
    import pickle

    with temporal_pkl.open("wb") as f:
        pickle.dump(temporal, f)

    out_csv = tmp_path / "network_summary.csv"
    out_tex = tmp_path / "table_network_summary.tex"
    out_fig = tmp_path / "fig_degree_distributions.png"

    write_network_diagnostics(
        aggregate_graph=G,
        temporal_graphs_pkl=str(temporal_pkl),
        firm_mapping={0: "A", 1: "B", 2: "C"},
        out_csv=str(out_csv),
        out_tex=str(out_tex),
        out_degree_fig=str(out_fig),
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) > 0


def test_validation_centrality_diagnostics_writes_outputs(tmp_path: Path) -> None:
    from bubbles_networks.validation import write_centrality_diagnostics

    # Minimal synthetic temporal graphs + matching centrality time series
    import networkx as nx
    from torch_geometric.utils import from_networkx

    g1 = nx.DiGraph()
    g1.add_nodes_from([0, 1, 2])
    g1.add_edge(0, 1, weight=1.0)
    g2 = nx.DiGraph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edge(1, 2, weight=2.0)

    temporal = [
        (pd.Timestamp("2020-01-01"), from_networkx(g1)),
        (pd.Timestamp("2020-01-02"), from_networkx(g2)),
    ]
    temporal_pkl = tmp_path / "temporal_graphs.pkl"
    import pickle

    with temporal_pkl.open("wb") as f:
        pickle.dump(temporal, f)

    ts_csv = tmp_path / "centrality_timeseries.csv"
    df = pd.DataFrame(
        [
            {"Date": "2020-01-01", "Company": "A", "Degree": 0.1, "Betweenness": 0.0, "Eigenvector": 0.2},
            {"Date": "2020-01-02", "Company": "A", "Degree": 0.2, "Betweenness": 0.0, "Eigenvector": 0.3},
            {"Date": "2020-01-01", "Company": "B", "Degree": 0.3, "Betweenness": 0.1, "Eigenvector": 0.4},
            {"Date": "2020-01-02", "Company": "B", "Degree": 0.2, "Betweenness": 0.1, "Eigenvector": 0.5},
        ]
    )
    df.to_csv(ts_csv, index=False)

    out_csv = tmp_path / "centrality_summary.csv"
    out_tex = tmp_path / "table_centrality_summary.tex"
    out_fig = tmp_path / "fig_centrality_top_nodes.png"

    write_centrality_diagnostics(
        centrality_timeseries_csv=str(ts_csv),
        temporal_graphs_pkl=str(temporal_pkl),
        out_csv=str(out_csv),
        out_tex=str(out_tex),
        out_fig=str(out_fig),
    )

    assert out_csv.exists()
    assert out_tex.exists()
    assert out_fig.exists()


def test_make_paper_dry_run_does_not_crash() -> None:
    repo = _repo_root()
    cmd = [sys.executable, str(repo / "scripts" / "make_paper.py"), "--mode", "minimal", "--dry-run"]
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert p.returncode == 0
    assert "DRY RUN" in (p.stdout + p.stderr)
    assert "Writes:" in (p.stdout + p.stderr)

