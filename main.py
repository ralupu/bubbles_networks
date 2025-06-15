import argparse
from descriptive_bubbles_module import run_descriptive_bubble_analysis
from bubble_overlap_chart_module import generate_bubble_overlap_chart
from network_aggregate_module import run_aggregate_network_analysis
from temporal_network_module import build_temporal_graphs
from centrality_analysis_module import run_centrality_analysis


def main(skip_temporal=False):
    print("==== Step 1: Descriptive Bubble Analysis ====")
    run_descriptive_bubble_analysis()

    print("==== Step 2: Bubble Overlap Chart ====")
    generate_bubble_overlap_chart()

    print("==== Step 3: Aggregate Network Construction ====")
    run_aggregate_network_analysis()

    if not skip_temporal:
        print("==== Step 4: Build Temporal Dynamic Networks ====")
        build_temporal_graphs()
    else:
        print("==== Step 4 skipped. Using existing temporal_graphs.pkl ====")

    print("==== Step 5: Compute Centrality Metrics ====")
    run_centrality_analysis()

    print("\n✅ All pipeline steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-temporal', action='store_true', help='Skip temporal network generation')
    args = parser.parse_args()

    main(skip_temporal=args.skip_temporal)
