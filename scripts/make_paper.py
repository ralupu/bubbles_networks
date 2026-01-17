import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _run_cmd(cmd: List[str]) -> int:
    print(f"[make_paper] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


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
    run_frm: bool
    run_tgnn: bool
    skip_temporal: bool
    skip_centrality: bool
    skip_pdf: bool
    start_date: Optional[str]


def export_paper_assets() -> None:
    repo = _repo_root()
    figures = repo / "figures"
    doc_figures = repo / "documents" / "figures"

    doc_figures.mkdir(parents=True, exist_ok=True)

    two_up_ok = _compose_two_up(
        figures / "NoOfBubbles.png",
        figures / "histDuration.png",
        doc_figures / "fig_bubble_descriptives.png",
    )
    if not two_up_ok:
        _copy(figures / "NoOfBubbles.png", doc_figures / "fig_bubble_descriptives.png")

    _copy(figures / "overlapping_bubbles.png", doc_figures / "fig_overlap_gantt.png")
    _copy(figures / "bubble_network_circular.png", doc_figures / "fig_overlap_network.png")
    _copy(figures / "centrality_dynamics.png", doc_figures / "fig_centrality_dynamics.png")
    _copy(figures / "centrality_heatmap.png", doc_figures / "fig_centrality_heatmap.png")
    _copy(figures / "tgnn_performance.png", doc_figures / "fig_tgnn_performance.png")


def export_paper_tables() -> None:
    repo = _repo_root()
    out_dir = repo / "documents" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ("ro_bubbles", "data/ro/ResultResults_ro_bet_bubbles.xlsx"),
        ("ro_covars", "data/ro/ResultResults_ro_bet_covars.xlsx"),
        ("ro_returns", "data/ro/ResultResults_ro_bet_returns.xlsx"),
        ("stoxx600", "data/stoxx600/ResultBubbles_STOXX_Mar2025.xlsx"),
    ]
    out_path = out_dir / "table_inputs.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("name,path,exists,size_bytes\n")
        for name, rel in rows:
            p = repo / rel
            f.write(f"{name},{rel},{p.exists()},{p.stat().st_size if p.exists() else ''}\n")


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
    p.add_argument("--run-frm", action="store_true", help="Run FRM module (can be slow)")
    p.add_argument("--run-tgnn", action="store_true", help="Run TGNN module (optional)")
    p.add_argument("--start-date", type=str, default=None, help="FRM start date (YYYY-MM-DD)")
    p.add_argument("--skip-temporal", action="store_true", help="Skip temporal graph rebuild (reuse cached pkl if any)")
    p.add_argument("--skip-centrality", action="store_true", help="Skip centrality figures")
    p.add_argument("--skip-pdf", action="store_true", help="Skip LaTeX PDF build")
    ns = p.parse_args(argv)
    return MakePaperArgs(
        mode=ns.mode,
        run_frm=bool(ns.run_frm),
        run_tgnn=bool(ns.run_tgnn),
        skip_temporal=bool(ns.skip_temporal),
        skip_centrality=bool(ns.skip_centrality),
        skip_pdf=bool(ns.skip_pdf),
        start_date=ns.start_date,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _ensure_imports()

    from bubbles_networks.pipeline import PipelineArgs, run_pipeline

    results_temporal = _repo_root() / "results" / "temporal_graphs.pkl"

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
            "1",
            "--model",
            "gconvgru",
            "--out-fig",
            "figures/tgnn_performance.png",
        ]
        if args.mode == "full":
            tgnn_args = [
                "--mode",
                "bubble",
                "--epochs",
                "50",
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

    export_paper_assets()
    export_paper_tables()

    if not args.skip_pdf:
        rc = build_pdf()
        if rc != 0:
            return rc

    pdf = _repo_root() / "documents" / "build" / "main.pdf"
    if pdf.exists():
        print(f"[make_paper] OK: {pdf}")
    else:
        print("[make_paper] WARNING: expected PDF not found at documents/build/main.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
