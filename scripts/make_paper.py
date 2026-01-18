import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _run_cmd(cmd: List[str]) -> int:
    print(f"[make_paper] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def _git_sha(repo: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            return (p.stdout or "").strip()
    except Exception:
        pass
    return "unknown"


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _compose_two_up(top: Path, bottom: Path, out_path: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    if not top.exists() or not bottom.exists():
        return False
    a = Image.open(top).convert("RGB")
    b = Image.open(bottom).convert("RGB")
    width = max(a.width, b.width)
    height = a.height + b.height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (0, a.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


@dataclass(frozen=True)
class MakePaperArgs:
    mode: str
    dataset: str
    run_frm: bool
    run_tgnn: bool
    run_robustness: bool
    skip_temporal: bool
    skip_centrality: bool
    skip_pdf: bool
    start_date: Optional[str]
    dry_run: bool
    report: bool


def _required_paths_ok(required: Sequence[Tuple[Path, str]]) -> None:
    missing = [(str(p), hint) for p, hint in required if not p.exists()]
    if missing:
        lines = ["[make_paper] Missing required outputs:"]
        for p, hint in missing:
            lines.append(f"- {p} ({hint})")
        raise RuntimeError("\n".join(lines))


def _write_tgnn_placeholder(out_path: Path) -> None:
    placeholder = _repo_root() / "documents" / "figures" / "fig_tgnn_performance_placeholder.png"
    if not placeholder.exists():
        raise RuntimeError(
            "[make_paper] Missing TGNN placeholder: documents/figures/fig_tgnn_performance_placeholder.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(placeholder, out_path)


def _require_text_contains(path: Path, needle: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[make_paper] Expected file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if needle not in text:
        raise RuntimeError(f"[make_paper] Expected marker not found in {path}: {needle}")


def export_paper_assets(*, skip_centrality: bool, run_tgnn: bool) -> List[str]:
    repo = _repo_root()
    figures = repo / "figures"
    doc_figures = repo / "documents" / "figures"
    doc_figures.mkdir(parents=True, exist_ok=True)

    required = [
        (figures / "NoOfBubbles.png", "descriptive bubble plot"),
        (figures / "histDuration.png", "descriptive duration plot"),
        (figures / "overlapping_bubbles.png", "overlap gantt plot"),
        (figures / "bubble_network_circular.png", "aggregate overlap network plot"),
    ]
    if not skip_centrality:
        required += [
            (figures / "centrality_dynamics.png", "centrality dynamics plot"),
            (figures / "centrality_heatmap.png", "centrality heatmap plot"),
        ]
    if run_tgnn:
        required += [(figures / "tgnn_performance.png", "TGNN performance plot")]
    _required_paths_ok(required)

    two_up_ok = _compose_two_up(
        figures / "NoOfBubbles.png",
        figures / "histDuration.png",
        doc_figures / "fig_bubble_descriptives.png",
    )
    if not two_up_ok:
        _copy(figures / "NoOfBubbles.png", doc_figures / "fig_bubble_descriptives.png")

    _copy(figures / "overlapping_bubbles.png", doc_figures / "fig_overlap_gantt.png")
    _copy(figures / "bubble_network_circular.png", doc_figures / "fig_overlap_network.png")

    created = [
        "documents/figures/fig_bubble_descriptives.png",
        "documents/figures/fig_overlap_gantt.png",
        "documents/figures/fig_overlap_network.png",
    ]

    if not skip_centrality:
        _copy(figures / "centrality_dynamics.png", doc_figures / "fig_centrality_dynamics.png")
        _copy(figures / "centrality_heatmap.png", doc_figures / "fig_centrality_heatmap.png")
        created += [
            "documents/figures/fig_centrality_dynamics.png",
            "documents/figures/fig_centrality_heatmap.png",
        ]

    if run_tgnn:
        _copy(figures / "tgnn_performance.png", doc_figures / "fig_tgnn_performance.png")
    else:
        _write_tgnn_placeholder(doc_figures / "fig_tgnn_performance.png")
    created += ["documents/figures/fig_tgnn_performance.png"]

    return created


def build_pdf() -> int:
    repo = _repo_root()
    if os.name == "nt":
        return _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo / "scripts" / "build_paper.ps1"),
            ]
        )
    return _run_cmd([str(repo / "scripts" / "build_paper.sh")])


def parse_args(argv: Optional[List[str]] = None) -> MakePaperArgs:
    p = argparse.ArgumentParser(description="One-command pipeline → paper assets → PDF build.")
    p.add_argument("--mode", choices=["minimal", "full"], default="minimal")
    p.add_argument("--dataset", choices=["ro", "stoxx600"], default="ro", help="Dataset label for reporting.")
    p.add_argument("--run-frm", action="store_true", help="Run FRM module (can be slow)")
    p.add_argument("--run-tgnn", action="store_true", help="Run TGNN module (optional)")
    p.add_argument("--run-robustness", action="store_true", help="Run robustness grid for overlap network")
    p.add_argument("--start-date", type=str, default=None, help="FRM start date (YYYY-MM-DD)")
    p.add_argument("--skip-temporal", action="store_true", help="Skip temporal graph rebuild (reuse cached pkl if any)")
    p.add_argument("--skip-centrality", action="store_true", help="Skip centrality diagnostics/plots")
    p.add_argument("--skip-pdf", action="store_true", help="Skip LaTeX PDF build")
    p.add_argument("--dry-run", action="store_true", help="Print planned steps/IO and exit.")
    p.add_argument("--report", action="store_true", help="Write a Markdown run report under results/run_reports/.")
    ns = p.parse_args(argv)
    return MakePaperArgs(
        mode=ns.mode,
        dataset=ns.dataset,
        run_frm=bool(ns.run_frm),
        run_tgnn=bool(ns.run_tgnn),
        run_robustness=bool(ns.run_robustness),
        skip_temporal=bool(ns.skip_temporal),
        skip_centrality=bool(ns.skip_centrality),
        skip_pdf=bool(ns.skip_pdf),
        start_date=ns.start_date,
        dry_run=bool(ns.dry_run),
        report=bool(ns.report),
    )


def _planned_io(args: MakePaperArgs) -> Dict[str, List[str]]:
    reads = [
        "data/ro/ResultResults_ro_bet_bubbles.xlsx",
        "data/ro/ResultResults_ro_bet_covars.xlsx",
        "data/ro/ResultResults_ro_bet_returns.xlsx (optional; FRM)",
    ]
    writes = [
        "figures/NoOfBubbles.png",
        "figures/histDuration.png",
        "figures/overlapping_bubbles.png",
        "figures/bubble_network_circular.png",
        "results/temporal_graphs.pkl",
        "figures/centrality_dynamics.png",
        "figures/centrality_heatmap.png",
        "results/centrality_timeseries.csv",
        "results/metadata/data_dictionary.csv",
        "results/network_summary.csv",
        "results/centrality_summary.csv",
        "documents/tables/table_data_dictionary.tex",
        "documents/tables/table_network_summary.tex",
        "documents/tables/table_centrality_summary.tex",
        "documents/figures/fig_bubble_descriptives.png",
        "documents/figures/fig_overlap_gantt.png",
        "documents/figures/fig_overlap_network.png",
        "documents/figures/fig_degree_distributions.png",
        "documents/figures/fig_centrality_dynamics.png",
        "documents/figures/fig_centrality_heatmap.png",
        "documents/figures/fig_centrality_top_nodes.png",
        "documents/figures/fig_tgnn_performance.png",
        "results/robustness/robustness_summary.csv",
        "documents/tables/table_robustness_summary.tex",
        "documents/figures/fig_robustness_heatmap.png",
        "results/frm/frm_graphs.pkl",
        "results/frm/frm_network_summary.csv",
        "documents/tables/table_frm_network_summary.tex",
        "documents/figures/fig_frm_degree_distributions.png",
        "documents/figures/fig_frm_overlap_vs_bubble.png",
        "documents/tables/table_frm_sensitivity.tex",
        "documents/figures/fig_frm_sensitivity_heatmap.png",
        "documents/build/main.pdf",
    ]
    if args.run_tgnn:
        writes += ["figures/tgnn_performance.png"]
    if args.report:
        writes += ["results/run_reports/<timestamp>_<dataset>_<mode>.md"]
    return {"reads": reads, "writes": writes}


def _print_dry_run(args: MakePaperArgs) -> None:
    io = _planned_io(args)
    print("[make_paper] DRY RUN")
    print(f"[make_paper] mode={args.mode} dataset={args.dataset} run_frm={args.run_frm} run_tgnn={args.run_tgnn}")
    print(
        f"[make_paper] run_robustness={args.run_robustness} skip_temporal={args.skip_temporal} skip_centrality={args.skip_centrality} "
        f"skip_pdf={args.skip_pdf} report={args.report}"
    )
    print("[make_paper] Reads:")
    for p in io["reads"]:
        print(f"  - {p}")
    print("[make_paper] Writes:")
    for p in io["writes"]:
        print(f"  - {p}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _ensure_imports()

    if args.dry_run:
        _print_dry_run(args)
        return 0

    from bubbles_networks.pipeline import PipelineArgs, run_pipeline
    from bubbles_networks.validation import ExcelInputSpec, ValidationError, validate_inputs

    if args.dataset != "ro":
        print("[make_paper] ERROR: dataset=stoxx600 is not wired into the active pipeline yet.")
        return 2

    ro_spec = ExcelInputSpec(
        dataset="ro",
        bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
        bubble_sheet="Breakdowns",
        date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
        covar_file="data/ro/ResultResults_ro_bet_covars.xlsx",
        covar_sheet="Delta CoVaR (K=95%)",
        returns_file="data/ro/ResultResults_ro_bet_returns.xlsx",
    )

    try:
        validate_inputs(ro_spec)
    except ValidationError as e:
        print(f"[make_paper] INPUT VALIDATION ERROR: {e}")
        return 2

    repo = _repo_root()
    results_temporal = repo / "results" / "temporal_graphs.pkl"

    skip_temporal = args.skip_temporal
    if args.mode == "minimal" and not results_temporal.exists():
        print("[make_paper] results/temporal_graphs.pkl missing; rebuilding temporal graphs (first run may be slower).")
        skip_temporal = False
    if args.mode == "full":
        skip_temporal = False

    tgnn_args: Optional[List[str]] = None
    if args.run_tgnn:
        tgnn_args = [
            "--mode",
            "bubble",
            "--epochs",
            "1" if args.mode == "minimal" else "50",
            "--model",
            "gconvgru",
            "--out-fig",
            "figures/tgnn_performance.png",
        ]

    pipeline_args = PipelineArgs(
        skip_temporal=skip_temporal,
        run_frm=False,
        skip_centrality=args.skip_centrality,
        start_date=args.start_date,
        run_tgnn=args.run_tgnn,
        tgnn_args=tgnn_args,
    )

    rc = run_pipeline(pipeline_args)
    if rc != 0:
        return rc

    from bubbles_networks.network_aggregate import build_aggregate_overlap_graph
    from bubbles_networks.validation import (
        DataDictionaryRow,
        write_centrality_diagnostics,
        write_data_dictionary,
        write_network_diagnostics,
        write_run_report,
    )

    outputs: List[str] = []

    rows = [
        DataDictionaryRow(
            dataset="ro",
            file_path=ro_spec.bubble_file,
            sheet=ro_spec.bubble_sheet,
            required_columns="Firm,Start,Peak,End,Duration",
            column_types="Firm:str; Start:int; Peak:int; End:int; Duration:int",
            notes="Start/Peak/End are integer indices mapped to calendar dates using the date-index sheet.",
        ),
        DataDictionaryRow(
            dataset="ro",
            file_path=ro_spec.bubble_file,
            sheet=ro_spec.date_sheet,
            required_columns="Date",
            column_types="Date:date",
            notes="Provides date index mapping used by bubble episode tables.",
        ),
        DataDictionaryRow(
            dataset="ro",
            file_path=ro_spec.covar_file,
            sheet=ro_spec.covar_sheet,
            required_columns="Date,+firm columns",
            column_types="Date:date; firms:float",
            notes="Wide format: one column per firm with Delta CoVaR values.",
        ),
        DataDictionaryRow(
            dataset="ro",
            file_path=str(ro_spec.returns_file or ""),
            sheet="(default)",
            required_columns="Date,+firm columns",
            column_types="Date:date; firms:float",
            notes="Wide format returns used by FRM module (optional).",
        ),
        DataDictionaryRow(
            dataset="stoxx600",
            file_path="data/stoxx600/ResultBubbles_STOXX_Mar2025.xlsx",
            sheet="(unknown/varies)",
            required_columns="(project-specific)",
            column_types="(project-specific)",
            notes="Present for extension work; not required for the minimal paper build.",
        ),
    ]

    write_data_dictionary(
        rows=rows,
        out_csv="results/metadata/data_dictionary.csv",
        out_tex="documents/tables/table_data_dictionary.tex",
    )
    _require_text_contains(repo / "documents" / "tables" / "table_data_dictionary.tex", "\\label{tab:data_dictionary}")
    outputs += ["results/metadata/data_dictionary.csv", "documents/tables/table_data_dictionary.tex"]

    G = build_aggregate_overlap_graph(bubble_file=ro_spec.bubble_file, date_sheet=ro_spec.date_sheet, bubble_sheet=ro_spec.bubble_sheet)
    bubble_df = None
    try:
        import pandas as pd

        bubble_df = pd.read_excel(ro_spec.bubble_file, sheet_name=ro_spec.bubble_sheet)
    except Exception:
        bubble_df = None
    if bubble_df is not None and "Firm" in bubble_df.columns:
        firms = sorted(bubble_df["Firm"].dropna().unique())
    else:
        firms = sorted(list(G.nodes()))
    firm_mapping = {i: firm for i, firm in enumerate(firms)}

    write_network_diagnostics(
        aggregate_graph=G,
        temporal_graphs_pkl="results/temporal_graphs.pkl",
        firm_mapping=firm_mapping,
        out_csv="results/network_summary.csv",
        out_tex="documents/tables/table_network_summary.tex",
        out_degree_fig="documents/figures/fig_degree_distributions.png",
    )
    _require_text_contains(repo / "documents" / "tables" / "table_network_summary.tex", "\\label{tab:network_summary}")
    outputs += [
        "results/network_summary.csv",
        "documents/tables/table_network_summary.tex",
        "documents/figures/fig_degree_distributions.png",
    ]

    if not args.skip_centrality:
        write_centrality_diagnostics(
            centrality_timeseries_csv="results/centrality_timeseries.csv",
            temporal_graphs_pkl="results/temporal_graphs.pkl",
            out_csv="results/centrality_summary.csv",
            out_tex="documents/tables/table_centrality_summary.tex",
            out_fig="documents/figures/fig_centrality_top_nodes.png",
        )
        _require_text_contains(
            repo / "documents" / "tables" / "table_centrality_summary.tex", "\\label{tab:centrality_top10}"
        )
        outputs += [
            "results/centrality_summary.csv",
            "documents/tables/table_centrality_summary.tex",
            "documents/figures/fig_centrality_top_nodes.png",
        ]

    if args.run_robustness:
        from bubbles_networks.robustness import RobustnessRun, run_overlap_network_robustness

        runs = [
            RobustnessRun(min_overlap_days=m, edge_rule=r)
            for m in [0, 5, 10]
            for r in ["start_lead", "overlap_undirected_then_direct"]
        ]
        run_overlap_network_robustness(
            dataset=args.dataset,
            bubble_file=ro_spec.bubble_file,
            date_sheet=ro_spec.date_sheet,
            bubble_sheet=ro_spec.bubble_sheet,
            runs=runs,
            out_summary_csv="results/robustness/robustness_summary.csv",
            out_paper_tex="documents/tables/table_robustness_summary.tex",
            out_paper_fig="documents/figures/fig_robustness_heatmap.png",
        )
        outputs += [
            "results/robustness/robustness_summary.csv",
            "documents/tables/table_robustness_summary.tex",
            "documents/figures/fig_robustness_heatmap.png",
        ]

    frm_stats: Dict[str, str] = {}
    frm_params: Dict[str, str] = {}
    if args.run_frm:
        from bubbles_networks.frm_network import FRMConfig
        from bubbles_networks.frm_paper import run_frm_sensitivity_grid, run_frm_small_and_export_paper_outputs

        if args.mode == "minimal":
            cfg = FRMConfig(
                returns_file=str(ro_spec.returns_file or "data/ro/ResultResults_ro_bet_returns.xlsx"),
                start_date=args.start_date,
                end_date=None,
                max_days=260,
                window_size=60,
                step_size=5,
                quantile=0.05,
                alpha=0.0,
                max_firms=20,
                threshold_method="top_k_in",
                top_k=5,
                out_dir="results/frm",
                parallel=False,
            )
        else:
            cfg = FRMConfig(
                returns_file=str(ro_spec.returns_file or "data/ro/ResultResults_ro_bet_returns.xlsx"),
                start_date=args.start_date,
                end_date=None,
                max_days=None,
                window_size=250,
                step_size=1,
                quantile=0.05,
                alpha=0.0,
                max_firms=None,
                threshold_method="top_k_in",
                top_k=5,
                out_dir="results/frm",
                parallel=False,
            )

        frm_params = {
            "frm_window_size": str(cfg.window_size),
            "frm_step_size": str(cfg.step_size),
            "frm_quantile": str(cfg.quantile),
            "frm_top_k": str(cfg.top_k),
            "frm_max_firms": str(cfg.max_firms or ""),
            "frm_max_days": str(cfg.max_days or ""),
        }

        frm_stats = run_frm_small_and_export_paper_outputs(
            cfg=cfg,
            bubble_file=ro_spec.bubble_file,
            date_sheet=ro_spec.date_sheet,
            bubble_sheet=ro_spec.bubble_sheet,
            out_summary_csv="results/frm/frm_network_summary.csv",
            out_table_tex="documents/tables/table_frm_network_summary.tex",
            out_degree_fig="documents/figures/fig_frm_degree_distributions.png",
            out_compare_fig="documents/figures/fig_frm_overlap_vs_bubble.png",
        )
        _require_text_contains(
            repo / "documents" / "tables" / "table_frm_network_summary.tex", "\\label{tab:frm_network_summary}"
        )
        outputs += [
            "results/frm/frm_graphs.pkl",
            "results/frm/frm_network_summary.csv",
            "documents/tables/table_frm_network_summary.tex",
            "documents/figures/fig_frm_degree_distributions.png",
            "documents/figures/fig_frm_overlap_vs_bubble.png",
        ]

        run_frm_sensitivity_grid(
            base_cfg=FRMConfig(**{**cfg.__dict__, "max_days": 260, "max_firms": 20, "step_size": 10}),
            window_sizes=[60, 120],
            out_tex="documents/tables/table_frm_sensitivity.tex",
            out_fig="documents/figures/fig_frm_sensitivity_heatmap.png",
        )
        _require_text_contains(repo / "documents" / "tables" / "table_frm_sensitivity.tex", "\\label{tab:frm_sensitivity}")
        outputs += [
            "documents/tables/table_frm_sensitivity.tex",
            "documents/figures/fig_frm_sensitivity_heatmap.png",
        ]

    outputs += export_paper_assets(skip_centrality=args.skip_centrality, run_tgnn=args.run_tgnn)

    if not args.skip_pdf:
        rc = build_pdf()
        if rc != 0:
            return rc

    pdf = repo / "documents" / "build" / "main.pdf"
    if pdf.exists():
        print(f"[make_paper] OK: {pdf}")
        outputs += ["documents/build/main.pdf"]
    else:
        print("[make_paper] WARNING: expected PDF not found at documents/build/main.pdf")

    if args.report:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        report_path = repo / "results" / "run_reports" / f"{ts}_{args.dataset}_{args.mode}.md"
        params = {
            "mode": args.mode,
            "dataset": args.dataset,
            "run_frm": str(args.run_frm),
            "run_tgnn": str(args.run_tgnn),
            "run_robustness": str(args.run_robustness),
            "skip_temporal": str(skip_temporal),
            "skip_centrality": str(args.skip_centrality),
            "skip_pdf": str(args.skip_pdf),
            "start_date": str(args.start_date or ""),
        }
        params.update(frm_params)

        env = {
            "os": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
        }
        stats: Dict[str, str] = {
            "aggregate_nodes": str(int(G.number_of_nodes())),
            "aggregate_edges": str(int(G.number_of_edges())),
        }
        try:
            import pickle

            with open(repo / "results" / "temporal_graphs.pkl", "rb") as f:
                temporal = pickle.load(f)
            if isinstance(temporal, list) and temporal:
                dates = [d for d, _ in temporal]
                stats["temporal_snapshots"] = str(len(temporal))
                stats["temporal_date_start"] = str(min(dates))
                stats["temporal_date_end"] = str(max(dates))
        except Exception:
            pass
        if args.run_robustness:
            stats["robustness_runs"] = "6"
        if args.run_frm:
            stats.update(frm_stats)

        write_run_report(
            str(report_path),
            git_sha=_git_sha(repo),
            mode=args.mode,
            dataset=args.dataset,
            params=params,
            environment=env,
            stats=stats,
            outputs=outputs,
        )
        print(f"[make_paper] Wrote report: {report_path}")

        roadmap = repo / "PUBLICATION_REVIEW_AND_ROADMAP.md"
        if roadmap.exists():
            summary = (
                f"- make_paper report: `{report_path.as_posix()}` "
                f"(mode={args.mode}, dataset={args.dataset}, nodes={stats.get('aggregate_nodes')}, "
                f"edges={stats.get('aggregate_edges')}, snapshots={stats.get('temporal_snapshots', 'n/a')})"
            )
            _append_roadmap_dev_log(roadmap, [summary])

    return 0


def _append_roadmap_dev_log(path: Path, lines: List[str]) -> None:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = f"### {today}"
    block = [line if line.startswith("- ") else f"- {line}" for line in lines]

    try:
        dev_idx = content.index("## Development log (update on every meaningful change)")
    except ValueError:
        content += ["", "## Development log (update on every meaningful change)", "", header] + block
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        return

    # Find insertion point: under today's header if present, otherwise append a new header at end of dev-log section.
    insert_at = len(content)
    if header in content[dev_idx + 1 :]:
        h_idx = content.index(header, dev_idx + 1)
        insert_at = h_idx + 1
        while insert_at < len(content) and not content[insert_at].startswith("### "):
            insert_at += 1
        content[insert_at:insert_at] = block + [""]
    else:
        content += ["", header] + block

    path.write_text("\n".join(content) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
