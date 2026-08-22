"""Report and plot a Phase 0 graph-corruption sweep.

This is a CPU-only consumer of ``scripts/corruption_sweep.py`` outputs.  It never
opens a run directory or reruns an evaluator.  The sweep JSON is expected to contain
``ordered_edges``, ``planner_names`` and ``rows``; each row contains ``k``, the edge
added at that k, ``planner_mean_utilities`` and optionally ``pooled``.

Examples
--------
    uv run python scripts/p0_curve_report.py \
        --json .temp/results/p0_corruption_envI.json \
        --edgedeltas .temp/results/p0_corruption_envI_edgedeltas.json \
        --out-fig .temp/results/p0_corruption_curve.png \
        --out-md .temp/results/p0_corruption_report.md

    uv run python scripts/p0_curve_report.py --self-check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any


PAPER_PLANNERS = ("QACM", "CEM", "MPPI", "MCTS")
ALL_SERIES = (*PAPER_PLANNERS, "pooled")
EPSILON = 1e-12


def _finite(value: Any) -> bool:
    """Return whether ``value`` is a finite numeric scalar."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _planner_label(name: str) -> str | None:
    """Map repository planner names to the paper's short labels."""
    normalized = str(name).lower().replace("_", "")
    if "qacm" in normalized:
        return "QACM"
    if "mppi" in normalized:
        return "MPPI"
    if "mcts" in normalized:
        return "MCTS"
    if "cem" in normalized:
        return "CEM"
    return None


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rows_by_k(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("sweep JSON has no non-empty 'rows' list")

    rows: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        try:
            k = int(raw["k"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("every sweep row needs an integer 'k'") from error
        if k < 0:
            raise ValueError(f"invalid negative k={k}")
        if k in rows:
            raise ValueError(f"duplicate sweep row k={k}")
        rows[k] = raw

    if 0 not in rows:
        raise ValueError("sweep JSON must contain the k=0 baseline row")

    ordered_edges = payload.get("ordered_edges", [])
    if not isinstance(ordered_edges, list):
        raise ValueError("'ordered_edges' must be a list")
    edges = [str(edge) for edge in ordered_edges]
    return rows, edges


def _edge_label(k: int, row: dict[str, Any], ordered_edges: list[str]) -> str:
    if k == 0:
        return "(none)"
    if ordered_edges and k == len(ordered_edges):
        return "all edges"
    removed = row.get("removed_edge")
    if removed:
        return str(removed)
    removed_edges = row.get("removed_edges")
    if isinstance(removed_edges, list) and len(removed_edges) >= k:
        return str(removed_edges[k - 1])
    if len(ordered_edges) >= k:
        return ordered_edges[k - 1]
    return "unknown edge"


def _series(payload: dict[str, Any], rows: dict[int, dict[str, Any]], label: str) -> list[float | None]:
    """Return a series in sorted-k order, tolerating missing/NaN values."""
    values: list[float | None] = []
    for k in sorted(rows):
        row = rows[k]
        if label == "pooled":
            value = row.get("pooled")
            if not _finite(value):
                utilities = row.get("planner_mean_utilities", {})
                if isinstance(utilities, dict):
                    planner_values = [
                        _number(value)
                        for value in utilities.values()
                        if _finite(value)
                    ]
                    value = sum(planner_values) / len(planner_values) if planner_values else None
        else:
            value = None
            utilities = row.get("planner_mean_utilities", {})
            if isinstance(utilities, dict):
                for raw_name, raw_value in utilities.items():
                    if _planner_label(str(raw_name)) == label:
                        value = raw_value
                        break
        values.append(_number(value))
    return values


def _monotonic(values: list[float | None]) -> str:
    if len(values) < 2:
        return "UNKNOWN (fewer than two finite points)"
    if any(value is None for value in values):
        return "UNKNOWN (missing/NaN point)"
    finite_values = [value for value in values if value is not None]
    monotone = all(right <= left + EPSILON for left, right in zip(finite_values, finite_values[1:]))
    return "YES" if monotone else "NO"


def _drop(values: list[float | None]) -> dict[str, Any]:
    baseline = values[0] if values else None
    maximum_k = values[-1] if values else None
    if baseline is None or maximum_k is None:
        return {
            "baseline": baseline,
            "max_k": maximum_k,
            "absolute": None,
            "relative": None,
            "reduced": None,
        }
    absolute = baseline - maximum_k
    relative = absolute / abs(baseline) if abs(baseline) > EPSILON else None
    return {
        "baseline": baseline,
        "max_k": maximum_k,
        "absolute": absolute,
        "relative": relative,
        "reduced": absolute > EPSILON,
    }


def _read_bites(path: str | Path | None, ordered_edges: list[str]) -> dict[str, Any]:
    if path is None:
        return {"status": "not supplied", "by_edge": {}}
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {"status": f"unreadable ({error})", "by_edge": {}}
    raw_edges = payload.get("edges", [])
    by_edge: dict[str, dict[str, Any]] = {}
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if isinstance(item, dict) and item.get("edge") is not None:
                by_edge[str(item["edge"])] = item
    missing = [edge for edge in ordered_edges if edge not in by_edge]
    return {
        "status": "loaded",
        "by_edge": by_edge,
        "missing": missing,
    }


def _write_csv(path: Path, payload: dict[str, Any], rows: dict[int, dict[str, Any]], edges: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["k", "removed_edge", *ALL_SERIES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        series = {label: _series(payload, rows, label) for label in ALL_SERIES}
        for index, k in enumerate(sorted(rows)):
            writer.writerow(
                {
                    "k": k,
                    "removed_edge": _edge_label(k, rows[k], edges),
                    **{
                        label: "" if series[label][index] is None else series[label][index]
                        for label in ALL_SERIES
                    },
                }
            )


def _plot(
    path: Path,
    payload: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    edges: list[str],
) -> tuple[bool, Path]:
    """Write the PNG when matplotlib is available, otherwise write a CSV fallback."""
    csv_path = path.with_suffix(".csv")
    _write_csv(csv_path, payload, rows, edges)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False, csv_path

    path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(rows)
    labels = [_edge_label(k, rows[k], edges) for k in ks]
    series = {label: _series(payload, rows, label) for label in ALL_SERIES}
    styles = {
        "QACM": {"color": "#0072B2", "marker": "o"},
        "CEM": {"color": "#D55E00", "marker": "s"},
        "MPPI": {"color": "#009E73", "marker": "^"},
        "MCTS": {"color": "#CC79A7", "marker": "D"},
        "pooled": {"color": "#222222", "marker": "o", "linestyle": "--", "linewidth": 2.2},
    }
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for label in ALL_SERIES:
        ax.plot(ks, series[label], label=label, **styles[label])
    environment = payload.get("environment", "corruption sweep")
    ax.set_title(f"P0 actuator-edge corruption curve — {environment}")
    ax.set_xlabel("Number of removed actuator edges (k)")
    ax.set_ylabel("Mean planner utility")
    ax.set_xticks(ks, [f"{k}\n{label}" for k, label in zip(ks, labels, strict=True)])
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True, csv_path


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _report(
    path: Path,
    payload: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    edges: list[str],
    bites: dict[str, Any],
    figure_written: bool,
    fallback_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(rows)
    max_k = ks[-1]
    lines = [
        f"# P0 corruption-curve report — {payload.get('environment', 'unknown environment')}",
        "",
        f"Rows: k={ks[0]}..{max_k}; k={max_k} is the all-edges floor when it equals the "
        f"ordered edge count ({len(edges)}). Utilities are the sweep's per-planner means; "
        "pooled is the unweighted mean over available planner means.",
        "",
        "## Utility degradation",
        "",
        "| Series | Utility k=0 | Utility max k | Absolute drop | Relative drop | Monotone non-increasing | Mechanism claim |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    series_data: dict[str, dict[str, Any]] = {}
    for label in ALL_SERIES:
        values = _series(payload, rows, label)
        drop = _drop(values)
        monotone = _monotonic(values)
        if drop["reduced"] is True:
            claim = "YES — utility reduced"
        elif drop["reduced"] is False:
            claim = "NO — counter-evidence (no reduction)"
        else:
            claim = "UNKNOWN — missing/NaN endpoint"
        lines.append(
            f"| {label} | {_fmt(drop['baseline'])} | {_fmt(drop['max_k'])} | "
            f"{_fmt(drop['absolute'])} | {_fmt_percent(drop['relative'])} | {monotone} | {claim} |"
        )
        series_data[label] = {**drop, "monotone": monotone}

    lines.extend(
        [
            "",
            "Absolute drop is `utility(k=0) - utility(k=max)`; relative drop divides by "
            "`abs(utility(k=0))` and is N/A when the baseline is zero. MONOTONE means "
            "non-increasing across every finite adjacent point. Missing or NaN points "
            "produce UNKNOWN rather than being interpolated.",
            "",
            "## Ordered removals",
            "",
            "| k | Removed at this step | Cumulative interpretation |",
            "| ---: | --- | --- |",
        ]
    )
    for k in ks:
        label = _edge_label(k, rows[k], edges)
        cumulative = "baseline; no edge removed" if k == 0 else (
            "all ordered actuator edges removed" if k == len(edges) else f"first {k} ordered edge(s) removed"
        )
        lines.append(f"| {k} | {label} | {cumulative} |")

    lines.extend(["", "## Edge-delta bite cross-check", ""])
    if bites["status"] != "loaded":
        lines.append(f"Edge-delta file: **{bites['status']}**. Bite status cannot be confirmed.")
    else:
        by_edge = bites["by_edge"]
        lines.extend(
            [
                "The diagnostic's `present_in_base_graph` confirms structural presence; "
                "`bites` confirms a non-zero predicted-state delta under single-edge removal.",
                "",
                "| Edge | Present in base graph | Bites |",
                "| --- | --- | --- |",
            ]
        )
        for edge in edges:
            item = by_edge.get(edge)
            if item is None:
                lines.append(f"| {edge} | N/A | N/A |")
            else:
                lines.append(
                    f"| {edge} | {str(bool(item.get('present_in_base_graph'))).upper()} | "
                    f"{str(bool(item.get('bites'))).upper()} |"
                )
        missing = bites.get("missing", [])
        present = all(bool(by_edge.get(edge, {}).get("present_in_base_graph")) for edge in edges)
        biting = all(bool(by_edge.get(edge, {}).get("bites")) for edge in edges)
        if missing:
            lines.append(f"\nMissing bite diagnostics for: {', '.join(missing)}.")
        elif present and biting:
            lines.append("\nAll ordered removals were structurally present and numerically biting.")
        elif not present:
            lines.append("\nAt least one requested removal was not structurally present in the base graph.")
        else:
            lines.append("\nAt least one structurally present removal had `bites=false`; interpret the curve cautiously.")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an inference-time structural omission in the current CDL: the predictor "
            "was trained with the full feature set and the graph mask is changed only at "
            "prediction. A utility drop supports the controlled mechanism claim for this "
            "harness; it does not establish the result for a model retrained without the "
            "removed parent.",
            "",
            f"Figure: {'written to `' + str(path.with_suffix('.png')) + '`' if figure_written else 'matplotlib unavailable; use CSV fallback `' + str(fallback_path) + '`'}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    json_path: str | Path,
    edgedeltas_path: str | Path | None,
    figure_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    payload = _load_json(json_path)
    rows, edges = _rows_by_k(payload)
    bites = _read_bites(edgedeltas_path, edges)
    figure_path = Path(figure_path)
    report_path = Path(report_path)
    figure_written, fallback_path = _plot(figure_path, payload, rows, edges)
    _report(report_path, payload, rows, edges, bites, figure_written, fallback_path)
    return {
        "environment": payload.get("environment"),
        "rows": len(rows),
        "max_k": max(rows),
        "figure_written": figure_written,
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "fallback_path": str(fallback_path),
    }


def _self_check() -> None:
    """Exercise the full JSON -> figure/report path on a monotone synthetic curve."""
    with tempfile.TemporaryDirectory(prefix="p0-curve-report-") as temp_dir:
        root = Path(temp_dir)
        json_path = root / "synthetic.json"
        bites_path = root / "synthetic_edgedeltas.json"
        figure_path = root / "synthetic.png"
        report_path = root / "synthetic.md"
        edges = ["KPI1<-P2", "KPI2<-P3", "KPI3<-P4"]
        planner_rows = [
            {"QACM": 1.0, "ModelBasedCEM": 0.8, "ModelBasedMPPI": 0.6, "ModelBasedMCTS": 0.4},
            {"QACM": 0.9, "ModelBasedCEM": 0.7, "ModelBasedMPPI": 0.5, "ModelBasedMCTS": 0.3},
            {"QACM": 0.75, "ModelBasedCEM": 0.6, "ModelBasedMPPI": 0.4, "ModelBasedMCTS": 0.2},
            {"QACM": 0.6, "ModelBasedCEM": 0.5, "ModelBasedMPPI": 0.3, "ModelBasedMCTS": 0.1},
        ]
        rows = []
        for k, utilities in enumerate(planner_rows):
            rows.append(
                {
                    "k": k,
                    "removed_edge": None if k == 0 else edges[k - 1],
                    "removed_edges": edges[:k],
                    "planner_mean_utilities": utilities,
                    "pooled": sum(utilities.values()) / len(utilities),
                }
            )
        json_path.write_text(
            json.dumps(
                {
                    "environment": "Synthetic",
                    "ordered_edges": edges,
                    "planner_names": list(planner_rows[0]),
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        bites_path.write_text(
            json.dumps(
                {
                    "edges": [
                        {"edge": edge, "present_in_base_graph": True, "bites": True}
                        for edge in edges
                    ]
                }
            ),
            encoding="utf-8",
        )
        build_report(json_path, bites_path, figure_path, report_path)
        report = report_path.read_text(encoding="utf-8")
        assert figure_path.exists() or figure_path.with_suffix(".csv").exists()
        assert "| QACM | 1.000000 | 0.600000 | 0.400000 | 40.00% | YES |" in report
        assert "| pooled | 0.700000 | 0.375000 | 0.325000 | 46.43% | YES |" in report
        assert "All ordered removals were structurally present and numerically biting." in report
    print("p0_curve_report self-check passed: monotone synthetic curve, drops, figure/report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="corruption sweep utility JSON")
    parser.add_argument("--edgedeltas", help="per-edge delta JSON; missing is reported gracefully")
    parser.add_argument("--out-fig", help="PNG figure path")
    parser.add_argument("--out-md", help="Markdown report path")
    parser.add_argument("--self-check", action="store_true", help="run the synthetic CPU self-check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    required = {"--json": args.json, "--out-fig": args.out_fig, "--out-md": args.out_md}
    missing = [name for name, value in required.items() if not value]
    if missing:
        build_parser().error("missing required arguments: " + ", ".join(missing))
    result = build_report(args.json, args.edgedeltas, args.out_fig, args.out_md)
    mode = "PNG" if result["figure_written"] else "CSV fallback"
    print(f"[done] {mode} -> {result['figure_path'] if result['figure_written'] else result['fallback_path']}")
    print(f"[done] report -> {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
