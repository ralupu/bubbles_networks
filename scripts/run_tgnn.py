import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv=None) -> int:
    _ensure_src_on_path()
    from bubbles_networks.tgnn_forecasting import main as tgnn_main

    return tgnn_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

