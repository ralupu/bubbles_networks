import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness grid for the aggregate overlap network.")
    p.add_argument("--bubble-file", type=str, default="data/ro/ResultResults_ro_bet_bubbles.xlsx")
    p.add_argument("--date-sheet", type=str, default="BUB (CVM= WB, CVQ=95%, L=0)")
    p.add_argument("--bubble-sheet", type=str, default="Breakdowns")
    p.add_argument("--out-csv", type=str, default="results/robustness/robustness_summary.csv")
    p.add_argument("--out-tex", type=str, default="documents/tables/table_robustness_summary.tex")
    p.add_argument("--out-fig", type=str, default="documents/figures/fig_robustness_heatmap.png")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_imports()
    ns = parse_args(argv)

    from bubbles_networks.robustness import RobustnessRun, run_overlap_network_robustness

    runs = [
        RobustnessRun(min_overlap_days=m, edge_rule=r)
        for m in [0, 5, 10]
        for r in ["start_lead", "overlap_undirected_then_direct"]
    ]

    run_overlap_network_robustness(
        bubble_file=ns.bubble_file,
        date_sheet=ns.date_sheet,
        bubble_sheet=ns.bubble_sheet,
        runs=runs,
        out_summary_csv=ns.out_csv,
        out_paper_tex=ns.out_tex,
        out_paper_fig=ns.out_fig,
    )

    print(f"[run_robustness] Wrote: {ns.out_csv}")
    print(f"[run_robustness] Wrote: {ns.out_tex}")
    print(f"[run_robustness] Wrote: {ns.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

