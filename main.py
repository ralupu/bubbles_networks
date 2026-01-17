import argparse
import subprocess
import sys

from bubble_overlap_chart_module import generate_bubble_overlap_chart
from centrality_analysis_module import run_centrality_analysis
from descriptive_bubbles_module import run_descriptive_bubble_analysis
from frm_network_module import build_dynamic_frm_graphs
from network_aggregate_module import run_aggregate_network_analysis
from temporal_network_module import build_temporal_graphs


def main(skip_temporal=False, run_frm=False, start_date=None, run_tgnn=False, tgnn_args=None):
    print("==== Step 1: Descriptive Bubble Analysis ====")
    run_descriptive_bubble_analysis()

    print("==== Step 2: Bubble Overlap Chart ====")
    generate_bubble_overlap_chart()

    print("==== Step 3: Aggregate Network Construction ====")
    run_aggregate_network_analysis()

    if not skip_temporal:
        print("==== Step 4: Build Temporal Dynamic Networks (Bubble) ====")
        build_temporal_graphs()
    else:
        print("==== Step 4 skipped. Using existing temporal_graphs.pkl ====")

    if run_frm:
        print("==== Step 5: Build FRM Dynamic Networks ====")
        build_dynamic_frm_graphs(
            frm_window=250,
            max_zero_frac=0.5,
            start_date=start_date,
        )
    else:
        print("==== Step 5 skipped. (Pass --run-frm to generate frm_graphs.pkl) ====")

    print("==== Step 6: Compute Centrality Metrics (Bubble only) ====")
    run_centrality_analysis()

    if run_tgnn:
        print("==== Step 7: Run TGNN Forecasting ====")
        tgnn_script = "tgnn_forecasting_module.py"
        result = subprocess.run([sys.executable, tgnn_script] + (tgnn_args or []), check=False)
        if result.returncode != 0:
            print(f"TGNN script failed with code {result.returncode}")
        else:
            print("TGNN forecasting completed successfully.")

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-temporal", action="store_true", help="Skip temporal network generation")
    parser.add_argument("--run-frm", action="store_true", help="Generate FRM networks (frm_graphs.pkl)")
    parser.add_argument("--start-date", type=str, default=None, help="First date to include (YYYY-MM-DD)")
    parser.add_argument("--run-tgnn", action="store_true", help="Run TGNN forecasting pipeline at the end")
    parser.add_argument(
        "--tgnn-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to tgnn_forecasting_module.py",
    )
    args = parser.parse_args()

    main(
        skip_temporal=args.skip_temporal,
        run_frm=args.run_frm,
        start_date=args.start_date,
        run_tgnn=args.run_tgnn,
        tgnn_args=args.tgnn_args,
    )
