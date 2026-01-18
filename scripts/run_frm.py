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
    p = argparse.ArgumentParser(description="Run FRM dynamic network construction.")
    p.add_argument("--returns-file", type=str, default="data/ro/ResultResults_ro_bet_returns.xlsx")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--window-size", type=int, default=250)
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument("--max-firms", type=int, default=None, help="Keep only the first N firm columns for quick runs.")
    p.add_argument("--quantile", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="results/frm")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_imports()
    ns = parse_args(argv)

    from bubbles_networks.frm_network import FRMConfig, build_frm_snapshot_sequence

    cfg = FRMConfig(
        returns_file=ns.returns_file,
        start_date=ns.start_date,
        end_date=ns.end_date,
        max_days=ns.max_days,
        window_size=int(ns.window_size),
        step_size=int(ns.step_size),
        max_firms=ns.max_firms,
        quantile=float(ns.quantile),
        threshold_method="top_k_in",
        top_k=int(ns.top_k),
        out_dir=ns.out_dir,
        parallel=False,
    )

    build_frm_snapshot_sequence(cfg)
    print(f"[run_frm] Wrote: {ns.out_dir}/frm_graphs.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

