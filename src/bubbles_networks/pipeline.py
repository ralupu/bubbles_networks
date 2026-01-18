import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

from bubbles_networks.bubble_overlap_chart import generate_bubble_overlap_chart
from bubbles_networks.centrality_analysis import run_centrality_analysis
from bubbles_networks.descriptive_bubbles import run_descriptive_bubble_analysis
from bubbles_networks.frm_network import FRMConfig, build_frm_snapshot_sequence
from bubbles_networks.network_aggregate import run_aggregate_network_analysis
from bubbles_networks.temporal_network import build_temporal_graphs


@dataclass(frozen=True)
class PipelineArgs:
    skip_temporal: bool = False
    run_frm: bool = False
    skip_centrality: bool = False
    start_date: Optional[str] = None
    run_tgnn: bool = False
    tgnn_args: Optional[List[str]] = None


def run_pipeline(args: PipelineArgs) -> int:
    print("==== Step 1: Descriptive Bubble Analysis ====")
    run_descriptive_bubble_analysis()

    print("==== Step 2: Bubble Overlap Chart ====")
    generate_bubble_overlap_chart()

    print("==== Step 3: Aggregate Network Construction ====")
    run_aggregate_network_analysis()

    if not args.skip_temporal:
        print("==== Step 4: Build Temporal Dynamic Networks (Bubble) ====")
        build_temporal_graphs()
    else:
        print("==== Step 4 skipped. Using existing results/temporal_graphs.pkl ====")

    if args.run_frm:
        print("==== Step 5: Build FRM Dynamic Networks ====")
        cfg = FRMConfig(
            start_date=args.start_date,
            window_size=250,
            step_size=1,
            quantile=0.05,
            alpha=0.0,
            threshold_method="top_k_in",
            top_k=5,
            out_dir="results/frm",
            parallel=False,
        )
        build_frm_snapshot_sequence(cfg)
    else:
        print("==== Step 5 skipped. (Pass --run-frm to generate results/frm/frm_graphs.pkl) ====")

    if not args.skip_centrality:
        print("==== Step 6: Compute Centrality Metrics (Bubble only) ====")
        run_centrality_analysis()
    else:
        print("==== Step 6 skipped. (Omit --skip-centrality to compute centrality artifacts) ====")

    if args.run_tgnn:
        print("==== Step 7: Run TGNN Forecasting ====")
        tgnn_script = "scripts/run_tgnn.py"
        result = subprocess.run([sys.executable, tgnn_script] + (args.tgnn_args or []), check=False)
        if result.returncode != 0:
            print(f"TGNN script failed with code {result.returncode}")
            return result.returncode

    print("\nAll pipeline steps completed successfully.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end bubble network pipeline.")
    parser.add_argument("--skip-temporal", action="store_true", help="Skip temporal network generation")
    parser.add_argument("--run-frm", action="store_true", help="Generate FRM networks (results/frm/frm_graphs.pkl)")
    parser.add_argument("--start-date", type=str, default=None, help="First date to include (YYYY-MM-DD) for FRM")
    parser.add_argument("--skip-centrality", action="store_true", help="Skip centrality calculations/plots")
    parser.add_argument("--run-tgnn", action="store_true", help="Run TGNN forecasting at the end")
    parser.add_argument(
        "--tgnn-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to scripts/run_tgnn.py (e.g. --mode bubble --epochs 10 ...)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    args = PipelineArgs(
        skip_temporal=bool(ns.skip_temporal),
        run_frm=bool(ns.run_frm),
        skip_centrality=bool(ns.skip_centrality),
        start_date=ns.start_date,
        run_tgnn=bool(ns.run_tgnn),
        tgnn_args=ns.tgnn_args,
    )
    return run_pipeline(args)
