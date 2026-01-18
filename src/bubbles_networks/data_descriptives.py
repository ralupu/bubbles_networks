"""Paper-facing descriptive statistics for the RO dataset.

This module generates:
- Table D1: sample overview (`documents/tables/table_sample_overview.tex`)
- Table D2: bubble heterogeneity across firms (`documents/tables/table_bubble_heterogeneity.tex`)
- Figure: combined descriptives (`documents/figures/fig_data_descriptives.png`)

Important formatting requirement (journal workflow):
- The exported `.tex` files contain ONLY a `tabular` environment (no table/caption/label).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DescriptivesOutputs:
    table_sample_overview_tex: str
    table_bubble_heterogeneity_tex: str
    fig_data_descriptives_png: str


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}. Present: {list(df.columns)}")


def _fmt_date(d: pd.Timestamp) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def _fmt_int(x: object) -> str:
    try:
        return f"{int(x):d}"
    except Exception:
        return "NA"


def _fmt_float(x: object, *, ndigits: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if np.isnan(v):
        return "NA"
    return f"{v:.{ndigits}f}"


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters for plain-text table cells."""

    s = str(text)
    # Escape LaTeX special characters in remaining text.
    # Order matters: backslash first.
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("^", "\\textasciicircum{}")
    s = s.replace("~", "\\textasciitilde{}")
    return s


def load_bubble_episodes_with_dates(
    *,
    bubble_file: str,
    bubble_sheet: str,
    date_sheet: str,
) -> pd.DataFrame:
    """Load bubble episodes and map Start/End indices to calendar dates."""

    bubbles = pd.read_excel(bubble_file, sheet_name=bubble_sheet)
    _require_columns(bubbles, ["Firm", "Start", "End", "Duration"], context=f"{bubble_file}::{bubble_sheet}")

    dates_df = pd.read_excel(bubble_file, sheet_name=date_sheet)
    _require_columns(dates_df, ["Date"], context=f"{bubble_file}::{date_sheet}")
    dates_df = dates_df.copy()
    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y", errors="coerce")
    if dates_df["Date"].isna().all():
        # fallback parse
        dates_df["Date"] = pd.to_datetime(dates_df["Date"], errors="coerce")
    if dates_df["Date"].isna().all():
        raise ValueError(f"Could not parse any dates in {bubble_file}::{date_sheet} (Date column).")

    date_mapping: Dict[int, pd.Timestamp] = {i + 1: d for i, d in enumerate(dates_df["Date"]) if pd.notna(d)}

    out = bubbles.copy()
    out["Firm"] = out["Firm"].astype(str)
    out["Start_Date"] = out["Start"].map(date_mapping)
    out["End_Date"] = out["End"].map(date_mapping)
    out["Duration_Days"] = (pd.to_datetime(out["End_Date"]) - pd.to_datetime(out["Start_Date"])).dt.days + 1

    # Fallback to provided Duration if date mapping failed for some rows.
    dur_fallback = pd.to_numeric(out["Duration"], errors="coerce")
    out["Duration_Days"] = out["Duration_Days"].where(out["Duration_Days"].notna(), dur_fallback)
    out["Duration_Days"] = pd.to_numeric(out["Duration_Days"], errors="coerce")
    return out


def load_returns(
    *,
    returns_file: str,
    returns_sheet: Optional[str] = None,
) -> pd.DataFrame:
    """Load a Date-indexed returns dataframe.

    Expected wide format: Date + one column per firm.
    """

    if not os.path.exists(returns_file):
        raise FileNotFoundError(f"Returns file not found: {returns_file}")
    sheet = returns_sheet if returns_sheet is not None else 0
    df = pd.read_excel(returns_file, sheet_name=sheet)
    _require_columns(df, ["Date"], context=f"{returns_file}::{returns_sheet or '(default)'}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _restrict_firms_for_descriptives(
    *,
    bubble_firms: Sequence[str],
    returns_firms: Sequence[str],
    prefer_firms: Optional[Sequence[str]] = None,
) -> List[str]:
    """Pick a deterministic firm universe for descriptives.

    - Base universe is firms present in both bubbles and returns.
    - If prefer_firms is provided, restrict to its intersection with the base universe.
    """

    bubble_set = {str(x) for x in bubble_firms}
    returns_set = {str(x) for x in returns_firms}
    base = sorted(bubble_set & returns_set)
    if prefer_firms is None:
        return base
    prefer = [str(x) for x in prefer_firms]
    prefer_set = {str(x) for x in prefer}
    out = [f for f in base if f in prefer_set]
    return out


def compute_sample_overview(
    *,
    bubbles: pd.DataFrame,
    returns: pd.DataFrame,
    firms: Sequence[str],
) -> Dict[str, object]:
    """Compute sample-level descriptive statistics for Table D1."""

    if not firms:
        raise ValueError("No firms selected for descriptives (empty firm universe).")

    _require_columns(returns, ["Date"], context="returns")
    for f in firms:
        if f not in returns.columns:
            raise ValueError(f"Missing firm column in returns data: {f}")

    r = returns[["Date"] + list(firms)].copy()
    start = r["Date"].min()
    end = r["Date"].max()

    obs_counts = r.drop(columns=["Date"]).notna().sum(axis=0).astype(int)

    b = bubbles.copy()
    _require_columns(b, ["Firm", "Duration_Days"], context="bubbles")
    b = b[b["Firm"].astype(str).isin([str(x) for x in firms])].copy()

    episodes_total = int(len(b))
    has_bubble = b.groupby("Firm").size().reindex(list(firms), fill_value=0) > 0
    pct_with_bubble = float(100.0 * has_bubble.mean()) if len(has_bubble) else float("nan")

    durations = pd.to_numeric(b["Duration_Days"], errors="coerce").dropna()
    bubble_days_total = float(durations.sum()) if len(durations) else 0.0

    out: Dict[str, object] = {
        "Sample period (start)": _fmt_date(start) if pd.notna(start) else "NA",
        "Sample period (end)": _fmt_date(end) if pd.notna(end) else "NA",
        "Frequency": "Daily",
        "Number of firms (N)": int(len(firms)),
        "Return observations per firm (min)": int(obs_counts.min()),
        "Return observations per firm (median)": float(obs_counts.median()),
        "Return observations per firm (max)": int(obs_counts.max()),
        "Total bubble episodes": int(episodes_total),
        "Firms with at least one bubble (%)": float(pct_with_bubble),
        "Bubble duration (days) mean": float(durations.mean()) if len(durations) else float("nan"),
        "Bubble duration (days) median": float(durations.median()) if len(durations) else float("nan"),
        "Bubble duration (days) std": float(durations.std(ddof=0)) if len(durations) else float("nan"),
        "Bubble duration (days) min": float(durations.min()) if len(durations) else float("nan"),
        "Bubble duration (days) max": float(durations.max()) if len(durations) else float("nan"),
        "Total firm-days in bubble (sum)": float(bubble_days_total),
    }
    return out


def compute_bubble_heterogeneity(
    *,
    bubbles: pd.DataFrame,
    firms: Sequence[str],
) -> pd.DataFrame:
    """Compute firm-level heterogeneity table for bubbles (Table D2)."""

    b = bubbles.copy()
    _require_columns(b, ["Firm", "Duration_Days"], context="bubbles")
    b["Firm"] = b["Firm"].astype(str)
    b = b[b["Firm"].isin([str(x) for x in firms])].copy()

    rows = []
    for firm in [str(x) for x in firms]:
        bb = b[b["Firm"] == firm]
        dur = pd.to_numeric(bb["Duration_Days"], errors="coerce").dropna()
        rows.append(
            {
                "Firm": firm,
                "# bubble episodes": int(len(bb)),
                "Total bubble days": float(dur.sum()) if len(dur) else 0.0,
                "Avg bubble duration": float(dur.mean()) if len(dur) else 0.0,
                "Max bubble duration": float(dur.max()) if len(dur) else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["# bubble episodes", "Firm"], ascending=[False, True]).reset_index(drop=True)
    return df


def write_sample_overview_tex(*, overview: Dict[str, object], out_tex: str) -> None:
    """Write Table D1 as a bare tabular environment (no caption/label)."""

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    def fmt_value(k: str, v: object) -> str:
        if k.endswith("(N)"):
            return _fmt_int(v)
        if "Return observations per firm" in k:
            return _fmt_float(v, ndigits=0)
        if "Total bubble episodes" in k:
            return _fmt_int(v)
        if "Firms with ≥1 bubble" in k:
            return _fmt_float(v, ndigits=1)
        if "Bubble duration" in k:
            return _fmt_float(v, ndigits=1)
        if "Total firm-days" in k:
            return _fmt_float(v, ndigits=0)
        return str(v)

    keys = list(overview.keys())
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{p{0.62\\textwidth} r}\n")
        f.write("\\toprule\n")
        f.write("Statistic & Value \\\\\n")
        f.write("\\midrule\n")
        for k in keys:
            v = overview[k]
            safe_k = _latex_escape(str(k))
            f.write(f"{safe_k} & {fmt_value(str(k), v)} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_bubble_heterogeneity_tex(
    *,
    firm_table: pd.DataFrame,
    out_tex: str,
    top_n: int = 15,
) -> None:
    """Write Table D2 as a bare tabular environment (no caption/label)."""

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = firm_table.copy()

    if len(df) > int(top_n):
        head = df.head(int(top_n)).copy()
        tail = df.iloc[int(top_n) :].copy()
        if len(tail):
            other = {
                "Firm": "All others (avg)",
                "# bubble episodes": float(tail["# bubble episodes"].mean()),
                "Total bubble days": float(tail["Total bubble days"].mean()),
                "Avg bubble duration": float(tail["Avg bubble duration"].mean()),
                "Max bubble duration": float(tail["Max bubble duration"].mean()),
            }
            head = pd.concat([head, pd.DataFrame([other])], ignore_index=True)
        df = head

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{p{0.22\\textwidth} r r r r}\n")
        f.write("\\toprule\n")
        f.write("Firm & Episodes & Bubble days & Avg duration & Max duration \\\\\n")
        f.write("\\midrule\n")
        for _, r in df.iterrows():
            firm = _latex_escape(str(r["Firm"]))
            episodes = _fmt_float(r["# bubble episodes"], ndigits=0)
            bubble_days = _fmt_float(r["Total bubble days"], ndigits=0)
            avg_d = _fmt_float(r["Avg bubble duration"], ndigits=1)
            max_d = _fmt_float(r["Max bubble duration"], ndigits=1)
            f.write(f"{firm} & {episodes} & {bubble_days} & {avg_d} & {max_d} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def plot_data_descriptives(
    *,
    bubbles: pd.DataFrame,
    firm_table: pd.DataFrame,
    out_png: str,
    top_n_firms: int = 15,
) -> None:
    """Create a 2-panel publication-style descriptive figure."""

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    b = bubbles.copy()
    _require_columns(b, ["Duration_Days"], context="bubbles")
    durations = pd.to_numeric(b["Duration_Days"], errors="coerce").dropna()

    ft = firm_table.copy()
    ft = ft.sort_values(["# bubble episodes", "Firm"], ascending=[False, True]).reset_index(drop=True)
    ft = ft.head(int(top_n_firms))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel A: duration histogram
    ax = axes[0]
    if len(durations):
        ax.hist(durations.to_numpy(), bins=25, color="#4C78A8", edgecolor="white")
    ax.set_title("Panel A: Bubble duration distribution")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")

    # Panel B: bubble episodes per firm
    ax = axes[1]
    x = [str(v) for v in ft["Firm"].tolist()]
    y = [int(float(v)) for v in ft["# bubble episodes"].tolist()]
    ax.bar(x, y, color="#F58518", edgecolor="white")
    ax.set_title("Panel B: Bubble episodes per firm (top)")
    ax.set_xlabel("Firm")
    ax.set_ylabel("# episodes")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_ro_descriptives_for_paper(
    *,
    bubble_file: str,
    bubble_sheet: str,
    date_sheet: str,
    returns_file: str,
    returns_sheet: Optional[str],
    prefer_firms: Optional[Sequence[str]],
    outputs: DescriptivesOutputs,
) -> Tuple[List[str], Dict[str, str]]:
    """Generate RO descriptives paper assets and return (created_paths, stats)."""

    bubbles = load_bubble_episodes_with_dates(bubble_file=bubble_file, bubble_sheet=bubble_sheet, date_sheet=date_sheet)
    returns = load_returns(returns_file=returns_file, returns_sheet=returns_sheet)

    bubble_firms = sorted(bubbles["Firm"].dropna().astype(str).unique())
    returns_firms = [str(c) for c in returns.columns if c != "Date"]
    firms = _restrict_firms_for_descriptives(bubble_firms=bubble_firms, returns_firms=returns_firms, prefer_firms=prefer_firms)
    if len(firms) < 2:
        raise RuntimeError(
            "Too few firms for descriptives after firm restriction. "
            f"bubble_firms={len(bubble_firms)} returns_firms={len(returns_firms)} prefer_firms={len(prefer_firms or [])}."
        )

    overview = compute_sample_overview(bubbles=bubbles, returns=returns, firms=firms)
    firm_table = compute_bubble_heterogeneity(bubbles=bubbles, firms=firms)

    write_sample_overview_tex(overview=overview, out_tex=outputs.table_sample_overview_tex)
    write_bubble_heterogeneity_tex(firm_table=firm_table, out_tex=outputs.table_bubble_heterogeneity_tex, top_n=15)
    plot_data_descriptives(bubbles=bubbles[bubbles["Firm"].isin(firms)], firm_table=firm_table, out_png=outputs.fig_data_descriptives_png)

    stats = {
        "descriptives_firms": str(len(firms)),
        "descriptives_return_start": str(overview.get("Sample period (start)", "")),
        "descriptives_return_end": str(overview.get("Sample period (end)", "")),
        "descriptives_bubble_episodes": str(int(overview.get("Total bubble episodes", 0))),
    }
    created = [
        outputs.table_sample_overview_tex,
        outputs.table_bubble_heterogeneity_tex,
        outputs.fig_data_descriptives_png,
    ]
    return created, stats
