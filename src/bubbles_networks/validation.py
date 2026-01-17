"""Validation and diagnostics for bubble network artifacts.

This module is intentionally "dataset-agnostic": callers provide paths/sheet names
for the dataset being analyzed (RO, STOXX600, etc.). The active pipeline currently
wires RO by default, but these functions should not assume RO-only conventions.

Outputs are designed to support the LaTeX paper as a living artifact by exporting
figures/tables under `documents/` (tracked) while writing intermediate artifacts
under `results/` (ignored by git policy).
"""

import csv
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ValidationError(RuntimeError):
    """Raised when required inputs/outputs are missing or inconsistent."""

    pass


def _latex_escape(text: str) -> str:
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    out = str(text)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _require_path(path: str) -> None:
    if not os.path.exists(path):
        raise ValidationError(f"Missing required file: {path}")


def _read_excel_required(path: str, sheet: str, *, nrows: Optional[int] = None) -> pd.DataFrame:
    _require_path(path)
    try:
        return pd.read_excel(path, sheet_name=sheet, nrows=nrows)
    except ValueError as e:
        raise ValidationError(f"Missing Excel sheet '{sheet}' in {path}. Error: {e}") from e


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns in {context}: {missing}. Present: {list(df.columns)}")


@dataclass(frozen=True)
class ExcelInputSpec:
    """Defines which input files/sheets/columns must exist for a dataset run."""

    dataset: str
    bubble_file: str
    bubble_sheet: str
    date_sheet: str
    covar_file: str
    covar_sheet: str
    returns_file: Optional[str] = None
    returns_sheet: Optional[str] = None
    bubble_required_columns: Tuple[str, ...] = ("Firm", "Start", "Peak", "End", "Duration")
    date_required_columns: Tuple[str, ...] = ("Date",)
    covar_required_columns: Tuple[str, ...] = ("Date",)
    returns_required_columns: Tuple[str, ...] = ("Date",)


def validate_inputs(spec: ExcelInputSpec, *, nrows: int = 5) -> None:
    """Fail-loud preflight validation for required Excel inputs.

    Parameters
    - spec: Dataset input definition (paths + sheet names).
    - nrows: Read only the first `nrows` for lightweight validation.
    """

    bubble = _read_excel_required(spec.bubble_file, spec.bubble_sheet, nrows=nrows)
    _require_columns(bubble, list(spec.bubble_required_columns), context=f"{spec.bubble_file}::{spec.bubble_sheet}")

    dates = _read_excel_required(spec.bubble_file, spec.date_sheet, nrows=nrows)
    _require_columns(dates, list(spec.date_required_columns), context=f"{spec.bubble_file}::{spec.date_sheet}")

    covar = _read_excel_required(spec.covar_file, spec.covar_sheet, nrows=nrows)
    _require_columns(covar, list(spec.covar_required_columns), context=f"{spec.covar_file}::{spec.covar_sheet}")

    if spec.returns_file and os.path.exists(spec.returns_file):
        sheet = spec.returns_sheet if spec.returns_sheet is not None else 0
        returns = pd.read_excel(spec.returns_file, sheet_name=sheet, nrows=nrows)
        _require_columns(
            returns,
            list(spec.returns_required_columns),
            context=f"{spec.returns_file}::{spec.returns_sheet or '(default)'}",
        )


def validate_ro_inputs(
    bubble_file: str = "data/ro/ResultResults_ro_bet_bubbles.xlsx",
    covar_file: str = "data/ro/ResultResults_ro_bet_covars.xlsx",
    returns_file: str = "data/ro/ResultResults_ro_bet_returns.xlsx",
    bubble_sheet: str = "Breakdowns",
    date_sheet: str = "BUB (CVM= WB, CVQ=95%, L=0)",
    covar_sheet: str = "Delta CoVaR (K=95%)",
) -> None:
    """Backward-compatible RO validator.

    Prefer using `validate_inputs(ExcelInputSpec(...))` from new code.
    """

    validate_inputs(
        ExcelInputSpec(
            dataset="ro",
            bubble_file=bubble_file,
            bubble_sheet=bubble_sheet,
            date_sheet=date_sheet,
            covar_file=covar_file,
            covar_sheet=covar_sheet,
            returns_file=returns_file,
            returns_sheet=None,
        )
    )


@dataclass(frozen=True)
class DataDictionaryRow:
    """Row definition for the exported data dictionary."""

    dataset: str
    file_path: str
    sheet: str
    required_columns: str
    column_types: str
    notes: str


def write_data_dictionary(
    *,
    rows: Optional[Sequence[DataDictionaryRow]] = None,
    out_csv: str,
    out_tex: str,
) -> List[DataDictionaryRow]:
    """Write a data dictionary as CSV (artifact) and LaTeX table (paper).

    Parameters
    - rows: Optional explicit rows; if omitted, callers should provide dataset-specific rows.
    - out_csv: CSV output path (typically under `results/metadata/`).
    - out_tex: LaTeX table output path (typically under `documents/tables/`).
    """

    if rows is None:
        rows = []

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "file_path", "sheet", "required_columns", "column_types", "notes"])
        for r in list(rows):
            w.writerow([r.dataset, r.file_path, r.sheet, r.required_columns, r.column_types, r.notes])

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{p{1.2cm} p{5.5cm} p{4.3cm} p{4.5cm}}\n")
        f.write("\\hline\n")
        f.write("Dataset & File (relative path) & Sheet & Notes \\\\\n")
        f.write("\\hline\n")
        for r in list(rows):
            file_tex = _latex_escape(r.file_path)
            sheet_tex = _latex_escape(r.sheet)
            notes_tex = _latex_escape(r.notes)
            f.write(f"{r.dataset} & \\texttt{{{file_tex}}} & {sheet_tex} & {notes_tex} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Data dictionary and provenance (high-level).}\n")
        f.write("\\label{tab:data_dictionary}\n")
        f.write("\\end{table}\n")

    return list(rows)


def _load_temporal_graphs(pkl_path: str) -> List[Tuple[pd.Timestamp, object]]:
    _require_path(pkl_path)
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, list) or not graphs:
        raise ValidationError(f"Expected a non-empty list of (date, graph) in {pkl_path}")
    return graphs


def _extract_edge_weights_from_pyg(graph) -> np.ndarray:
    if hasattr(graph, "weight") and graph.weight is not None:
        w = graph.weight
        try:
            return np.asarray(w.detach().cpu().numpy(), dtype=float)
        except Exception:
            return np.asarray(w, dtype=float)
    if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
        w = graph.edge_weight
        try:
            return np.asarray(w.detach().cpu().numpy(), dtype=float)
        except Exception:
            return np.asarray(w, dtype=float)
    edge_index = getattr(graph, "edge_index", None)
    m = int(edge_index.shape[1]) if edge_index is not None else 0
    return np.ones((m,), dtype=float)


def _union_find_components(num_nodes: int, edges: np.ndarray) -> Tuple[int, int]:
    parent = list(range(num_nodes))
    size = [1] * num_nodes

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            union(int(u), int(v))

    roots = {}
    for i in range(num_nodes):
        r = find(i)
        roots[r] = roots.get(r, 0) + 1
    n_comp = len(roots)
    largest = max(roots.values()) if roots else 0
    return n_comp, largest


def write_network_diagnostics(
    *,
    aggregate_graph,
    temporal_graphs_pkl: str = "results/temporal_graphs.pkl",
    firm_mapping: Dict[int, str],
    out_csv: str = "results/network_summary.csv",
    out_tex: str = "documents/tables/table_network_summary.tex",
    out_degree_fig: str = "documents/figures/fig_degree_distributions.png",
) -> None:
    """Compute and export baseline overlap network diagnostics.

    Writes a CSV summary (artifact), a LaTeX table (paper), and a simple degree histogram
    figure for the aggregate overlap network. Also includes per-snapshot rows for temporal
    graphs in the CSV to support robustness runs.

    Parameters
    - aggregate_graph: networkx graph for the aggregate overlap network.
    - temporal_graphs_pkl: Pickle path containing a list of (date, PyG graph) snapshots.
    - firm_mapping: Mapping from snapshot node indices -> firm identifiers.
    - out_csv/out_tex/out_degree_fig: Output paths.
    """

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    os.makedirs(os.path.dirname(out_degree_fig), exist_ok=True)

    # Aggregate overlap network
    G = aggregate_graph
    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    density = float(m / (n * (n - 1))) if n > 1 else 0.0
    w = np.array([float(G[u][v].get("weight", 1.0)) for u, v in G.edges()], dtype=float) if m else np.array([])
    w_stats = {
        "w_min": float(np.min(w)) if w.size else 0.0,
        "w_median": float(np.median(w)) if w.size else 0.0,
        "w_mean": float(np.mean(w)) if w.size else 0.0,
        "w_max": float(np.max(w)) if w.size else 0.0,
    }
    try:
        import networkx as nx

        comp_count = nx.number_weakly_connected_components(G) if n else 0
        comp_largest = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
    except Exception:
        comp_count, comp_largest = 0, 0

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_wdeg = dict(G.in_degree(weight="weight"))
    out_wdeg = dict(G.out_degree(weight="weight"))

    def top_k(d: Dict[str, float], k: int = 10) -> str:
        items = sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return "; ".join([f"{name}:{val:.3g}" for name, val in items[:k]])

    agg_row = {
        "network": "aggregate",
        "nodes": n,
        "edges": m,
        "density": density,
        "components_weak": comp_count,
        "largest_component_weak": comp_largest,
        **w_stats,
        "top_in_degree": top_k(in_deg),
        "top_out_degree": top_k(out_deg),
        "top_in_weighted_degree": top_k(in_wdeg),
        "top_out_weighted_degree": top_k(out_wdeg),
    }

    # Temporal snapshots summary
    temporal = _load_temporal_graphs(temporal_graphs_pkl)
    dates = [pd.to_datetime(d) for d, _ in temporal]
    if any(dates[i] >= dates[i + 1] for i in range(len(dates) - 1)):
        raise ValidationError(f"Snapshot dates in {temporal_graphs_pkl} are not strictly increasing.")

    snapshot_rows = []
    t_nodes = []
    t_edges = []
    t_density = []
    t_comp = []
    t_comp_largest = []
    all_w = []

    # Accumulate degrees across time for ranking (average over snapshots)
    deg_in_total: Dict[int, float] = {}
    deg_out_total: Dict[int, float] = {}
    wdeg_total: Dict[int, float] = {}

    def top_k_named(d: Dict[int, float], k: int = 10) -> str:
        items = sorted(d.items(), key=lambda kv: (-kv[1], int(kv[0])))
        parts = []
        for idx, val in items[:k]:
            parts.append(f"{firm_mapping.get(int(idx), f'Unknown_{idx}')}:{val:.3g}")
        return "; ".join(parts)

    for dt, g in temporal:
        num_nodes = int(getattr(g, "num_nodes", 0) or 0)
        edge_index = getattr(g, "edge_index", None)
        if edge_index is None:
            continue
        num_edges = int(edge_index.shape[1])

        t_nodes.append(num_nodes)
        t_edges.append(num_edges)
        t_density.append(float(num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0)

        edges = edge_index.detach().cpu().numpy().T.astype(int) if hasattr(edge_index, "detach") else np.asarray(edge_index).T
        comp_count, comp_largest = _union_find_components(num_nodes, edges)
        t_comp.append(comp_count)
        t_comp_largest.append(comp_largest)

        w = _extract_edge_weights_from_pyg(g)
        if w.size:
            all_w.extend(w.tolist())

        in_counts = np.bincount(edges[:, 1], minlength=num_nodes) if num_edges else np.zeros(num_nodes, dtype=int)
        out_counts = np.bincount(edges[:, 0], minlength=num_nodes) if num_edges else np.zeros(num_nodes, dtype=int)
        wdeg_counts = np.zeros(num_nodes, dtype=float)
        if num_edges and w.size == num_edges:
            for (u, v), ww in zip(edges, w):
                wdeg_counts[int(u)] += float(ww)
                wdeg_counts[int(v)] += float(ww)

        w_snapshot = w.astype(float) if w.size else np.array([], dtype=float)
        snapshot_rows.append(
            {
                "network": "snapshot",
                "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                "nodes": num_nodes,
                "edges": num_edges,
                "density": float(num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0,
                "components_weak": comp_count,
                "largest_component_weak": comp_largest,
                "w_min": float(np.min(w_snapshot)) if w_snapshot.size else 0.0,
                "w_median": float(np.median(w_snapshot)) if w_snapshot.size else 0.0,
                "w_mean": float(np.mean(w_snapshot)) if w_snapshot.size else 0.0,
                "w_max": float(np.max(w_snapshot)) if w_snapshot.size else 0.0,
                "top_in_degree": top_k_named({i: float(in_counts[i]) for i in range(num_nodes)}),
                "top_out_degree": top_k_named({i: float(out_counts[i]) for i in range(num_nodes)}),
                "top_in_weighted_degree": "",
                "top_out_weighted_degree": top_k_named({i: float(wdeg_counts[i]) for i in range(num_nodes)}),
            }
        )

        for i in range(num_nodes):
            deg_in_total[i] = deg_in_total.get(i, 0.0) + float(in_counts[i])
            deg_out_total[i] = deg_out_total.get(i, 0.0) + float(out_counts[i])

        if num_edges and w.size == num_edges:
            for (u, v), ww in zip(edges, w):
                wdeg_total[int(u)] = wdeg_total.get(int(u), 0.0) + float(ww)
                wdeg_total[int(v)] = wdeg_total.get(int(v), 0.0) + float(ww)

    snapshots = len(t_nodes)
    all_w_arr = np.array(all_w, dtype=float) if all_w else np.array([])

    def map_names(d: Dict[int, float]) -> Dict[str, float]:
        return {firm_mapping.get(i, f"Unknown_{i}"): v / max(1, snapshots) for i, v in d.items()}

    temporal_row = {
        "network": "temporal_avg",
        "date": "",
        "nodes": float(np.mean(t_nodes)) if t_nodes else 0.0,
        "edges": float(np.mean(t_edges)) if t_edges else 0.0,
        "density": float(np.mean(t_density)) if t_density else 0.0,
        "components_weak": float(np.mean(t_comp)) if t_comp else 0.0,
        "largest_component_weak": float(np.mean(t_comp_largest)) if t_comp_largest else 0.0,
        "w_min": float(np.min(all_w_arr)) if all_w_arr.size else 0.0,
        "w_median": float(np.median(all_w_arr)) if all_w_arr.size else 0.0,
        "w_mean": float(np.mean(all_w_arr)) if all_w_arr.size else 0.0,
        "w_max": float(np.max(all_w_arr)) if all_w_arr.size else 0.0,
        "top_in_degree": top_k(map_names(deg_in_total)),
        "top_out_degree": top_k(map_names(deg_out_total)),
        "top_in_weighted_degree": "",
        "top_out_weighted_degree": top_k(map_names(wdeg_total)),
    }

    agg_row_with_date = {**agg_row, "date": ""}
    df = pd.DataFrame([agg_row_with_date, temporal_row] + snapshot_rows)
    df.to_csv(out_csv, index=False)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l r r r r}\n\\hline\n")
        f.write("Network & Nodes & Edges & Density & Weak CCs \\\\\n\\hline\n")
        f.write(
            f"Aggregate & {int(agg_row['nodes'])} & {int(agg_row['edges'])} & {agg_row['density']:.4f} & {int(agg_row['components_weak'])} \\\\\n"
        )
        f.write(
            f"Temporal (avg) & {temporal_row['nodes']:.1f} & {temporal_row['edges']:.1f} & {temporal_row['density']:.4f} & {temporal_row['components_weak']:.1f} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Baseline network diagnostics (aggregate and temporal average).}\n")
        f.write("\\label{tab:network_summary}\n")
        f.write("\\end{table}\n\n")
        f.write("\\noindent\\textbf{Top nodes (aggregate)}\\\\\n")
        f.write("\\begin{verbatim}\n")
        f.write(f"in-degree: {agg_row['top_in_degree']}\n")
        f.write(f"out-degree: {agg_row['top_out_degree']}\n")
        f.write(f"in-weighted-degree: {agg_row['top_in_weighted_degree']}\n")
        f.write(f"out-weighted-degree: {agg_row['top_out_weighted_degree']}\n")
        f.write("\\end{verbatim}\n\n")
        f.write("\\noindent\\textbf{Top nodes (temporal average)}\\\\\n")
        f.write("\\begin{verbatim}\n")
        f.write(f"in-degree(avg): {temporal_row['top_in_degree']}\n")
        f.write(f"out-degree(avg): {temporal_row['top_out_degree']}\n")
        f.write(f"weighted-degree(avg): {temporal_row['top_out_weighted_degree']}\n")
        f.write("\\end{verbatim}\n")

    # Degree distribution figure (aggregate)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(list(in_deg.values()), bins=20, color="#4C72B0", alpha=0.9)
    plt.title("Aggregate In-degree")
    plt.xlabel("In-degree")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    plt.hist(list(out_deg.values()), bins=20, color="#55A868", alpha=0.9)
    plt.title("Aggregate Out-degree")
    plt.xlabel("Out-degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_degree_fig, dpi=300, bbox_inches="tight")
    plt.close()


def write_centrality_diagnostics(
    *,
    centrality_timeseries_csv: str = "results/centrality_timeseries.csv",
    temporal_graphs_pkl: str = "results/temporal_graphs.pkl",
    out_csv: str = "results/centrality_summary.csv",
    out_tex: str = "documents/tables/table_centrality_summary.tex",
    out_fig: str = "documents/figures/fig_centrality_top_nodes.png",
) -> None:
    """Validate and summarize temporal centrality time series.

    Parameters
    - centrality_timeseries_csv: CSV with columns: Date, Company, Degree, Betweenness, Eigenvector.
    - temporal_graphs_pkl: Temporal graphs pickle (used for date alignment validation when present).
    - out_csv/out_tex/out_fig: Output paths.
    """

    _require_path(centrality_timeseries_csv)
    df = pd.read_csv(centrality_timeseries_csv)
    _require_columns(df, ["Date", "Company", "Degree", "Betweenness", "Eigenvector"], context=centrality_timeseries_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    unique_dates = sorted(df["Date"].unique())
    if any(unique_dates[i] >= unique_dates[i + 1] for i in range(len(unique_dates) - 1)):
        raise ValidationError("Centrality time series dates are not strictly increasing.")

    if os.path.exists(temporal_graphs_pkl):
        temporal_dates = [pd.to_datetime(d) for d, _ in _load_temporal_graphs(temporal_graphs_pkl)]
        temporal_set = {d.normalize() for d in temporal_dates}
        centrality_set = {d.normalize() for d in unique_dates}
        if not centrality_set.issubset(temporal_set):
            missing = sorted(centrality_set - temporal_set)
            raise ValidationError(
                f"Centrality dates are not aligned with temporal snapshots in {temporal_graphs_pkl}. "
                f"Example missing dates: {missing[:5]}"
            )

    summary = df.groupby("Company", as_index=False).agg(
        degree_mean=("Degree", "mean"),
        degree_std=("Degree", "std"),
        betweenness_mean=("Betweenness", "mean"),
        betweenness_std=("Betweenness", "std"),
        eigenvector_mean=("Eigenvector", "mean"),
        eigenvector_std=("Eigenvector", "std"),
    )
    summary = summary.sort_values(["eigenvector_mean", "Company"], ascending=[False, True])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv, index=False)

    top10 = summary.head(10).copy()
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l r r}\n\\hline\n")
        f.write("Company & Mean eigenvector & Std eigenvector \\\\\n\\hline\n")
        for _, row in top10.iterrows():
            name = _latex_escape(str(row["Company"]))
            f.write(f"{name} & {row['eigenvector_mean']:.4f} & {0.0 if pd.isna(row['eigenvector_std']) else row['eigenvector_std']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Top-10 nodes by average eigenvector centrality (temporal).}\n")
        f.write("\\label{tab:centrality_top10}\n")
        f.write("\\end{table}\n")

    top5 = top10.head(5)["Company"].tolist()
    df_top = df[df["Company"].isin(top5)].copy()
    df_top = df_top.sort_values(["Date", "Company"])

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.figure(figsize=(10, 5))
    for company in top5:
        sub = df_top[df_top["Company"] == company]
        plt.plot(sub["Date"], sub["Eigenvector"], label=company, linewidth=2)
    plt.title("Top-5 nodes by average eigenvector centrality")
    plt.xlabel("Date")
    plt.ylabel("Eigenvector centrality")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()


def write_run_report(
    out_path: str,
    *,
    git_sha: str,
    mode: str,
    dataset: str,
    params: Dict[str, str],
    environment: Optional[Dict[str, str]] = None,
    stats: Optional[Dict[str, str]] = None,
    outputs: Sequence[str],
) -> None:
    """Write a Markdown run report for `scripts/make_paper.py`.

    Parameters
    - out_path: Output markdown path.
    - git_sha: Current git commit hash.
    - mode/dataset: Build mode + dataset label.
    - params: Key/value configuration parameters.
    - environment: Optional environment metadata (OS, python version, etc.).
    - stats: Optional numeric summaries (nodes/edges, snapshots, date range, etc.).
    - outputs: List of outputs created/updated by the run.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "# make_paper run report",
        "",
        f"- Timestamp (UTC): `{ts}`",
        f"- Git commit: `{git_sha}`",
        f"- Mode: `{mode}`",
        f"- Dataset: `{dataset}`",
        "",
        "## Environment",
    ]
    if environment:
        for k in sorted(environment):
            lines.append(f"- `{k}`: `{environment[k]}`")
    else:
        lines.append("- (not provided)")

    lines += ["", "## Stats"]
    if stats:
        for k in sorted(stats):
            lines.append(f"- `{k}`: `{stats[k]}`")
    else:
        lines.append("- (not provided)")

    lines += [
        "",
        "## Parameters",
    ]
    for k in sorted(params):
        lines.append(f"- `{k}`: `{params[k]}`")
    lines += ["", "## Outputs"]
    for p in sorted(set(outputs)):
        lines.append(f"- `{p}`")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
