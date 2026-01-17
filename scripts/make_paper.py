import argparse
import os
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
        "documents/build/main.pdf",
    ]
    if args.run_tgnn:
        writes += ["figures/tgnn_performance.png", "documents/figures/fig_tgnn_performance.png"]
    if args.report:
        writes += ["results/run_reports/<timestamp>_<dataset>_<mode>.md"]
    return {"reads": reads, "writes": writes}


def _print_dry_run(args: MakePaperArgs) -> None:
    io = _planned_io(args)
    print("[make_paper] DRY RUN")
    print(f"[make_paper] mode={args.mode} dataset={args.dataset} run_frm={args.run_frm} run_tgnn={args.run_tgnn}")
    print(
        f"[make_paper] skip_temporal={args.skip_temporal} skip_centrality={args.skip_centrality} "
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

    if args.dataset != "ro":
        print("[make_paper] ERROR: dataset=stoxx600 is not wired into the active pipeline yet.")
        return 2

    if args.dry_run:
        _print_dry_run(args)
        return 0

    from bubbles_networks.pipeline import PipelineArgs, run_pipeline
    from bubbles_networks.validation import ValidationError, validate_ro_inputs

    try:
        validate_ro_inputs()
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
        run_frm=args.run_frm,
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
        write_centrality_diagnostics,
        write_data_dictionary,
        write_network_diagnostics,
        write_run_report,
    )

    outputs: List[str] = []

    write_data_dictionary()
    outputs += ["results/metadata/data_dictionary.csv", "documents/tables/table_data_dictionary.tex"]

    G = build_aggregate_overlap_graph()
    firm_mapping = {i: firm for i, firm in enumerate(list(G.nodes()))}
    write_network_diagnostics(aggregate_graph=G, firm_mapping=firm_mapping)
    outputs += [
        "results/network_summary.csv",
        "documents/tables/table_network_summary.tex",
        "documents/figures/fig_degree_distributions.png",
    ]

    if not args.skip_centrality:
        write_centrality_diagnostics()
        outputs += [
            "results/centrality_summary.csv",
            "documents/tables/table_centrality_summary.tex",
            "documents/figures/fig_centrality_top_nodes.png",
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
            "skip_temporal": str(skip_temporal),
            "skip_centrality": str(args.skip_centrality),
            "skip_pdf": str(args.skip_pdf),
            "start_date": str(args.start_date or ""),
        }
        write_run_report(
            str(report_path),
            git_sha=_git_sha(repo),
            mode=args.mode,
            dataset=args.dataset,
            params=params,
            outputs=outputs,
        )
        print(f"[make_paper] Wrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
