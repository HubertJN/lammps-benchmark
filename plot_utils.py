import json
from pathlib import Path

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt_num(x, *, digits: int = 3):
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return "n/a"


def top_runs_by_tps(runs, n: int):
    scored = []
    for r in runs:
        tps = safe_float((r.get("performance") or {}).get("timesteps_per_s"))
        if tps is None:
            continue
        scored.append((tps, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:n]]


def load_params_for_tag(runs_dir: Path, tag: str) -> dict:
    """Load runs/<tag>/params.json. Returns {} if missing/unreadable."""
    p = runs_dir / tag / "params.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def collect_param_columns(params_list: list[dict]) -> list[str]:
    exclude = {"run_id", "input", "lmp", "log_path", "run_dir", "time_s", "returncode"}
    keys = set()
    for d in params_list:
        for k, v in d.items():
            if k in exclude:
                continue
            if isinstance(v, (str, int, float)):
                keys.add(k)

    preferred = ["ks", "dcut", "kacc"]
    rest = sorted(k for k in keys if k not in preferred)
    return [k for k in preferred if k in keys] + rest


def format_param_value(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def merge_small_slices(items, merge_lt_pct: float):
    """Merge pie slices smaller than merge_lt_pct into a 'Remainder' slice."""
    big = [(k, float(v)) for k, v in items if float(v) >= merge_lt_pct]
    small_sum = sum(float(v) for _, v in items if float(v) < merge_lt_pct)

    labels = [k for k, _ in big]
    values = [v for _, v in big]

    remainder = small_sum
    s = sum(values) + remainder
    if s < 99.999:
        remainder += (100.0 - s)

    if remainder > 0:
        labels.append("Remainder")
        values.append(remainder)

    return labels, values


def _normalize_group_value(v):
    if v is None:
        return ("none", None)
    try:
        return ("num", float(v))
    except Exception:
        return ("str", str(v).strip())


def best_runs_by_param(runs, params_by_tag: dict, param_key: str):
    """Return one run per unique param_key, choosing the highest timesteps/s."""
    best = {}
    for r in runs:
        tps = safe_float((r.get("performance") or {}).get("timesteps_per_s"))
        if tps is None:
            continue

        tag = r.get("tag", "unknown")
        p = params_by_tag.get(tag, {})
        group_key = _normalize_group_value(p.get(param_key))

        if group_key[0] == "none":
            continue

        prev = best.get(group_key)
        if prev is None or tps > prev[0]:
            best[group_key] = (tps, r)

    scored = list(best.values())
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored]


def make_timing_pie(run, out_png: Path, merge_lt_pct: float):
    timing = run.get("timing_breakdown_pct") or {}
    items = [(k, float(v)) for k, v in timing.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels, values = merge_small_slices(items, merge_lt_pct=merge_lt_pct)

    title = f"{run.get('tag', 'unknown')}"
    ks = (run.get("kspace") or {}).get("style")
    if ks:
        title += f"  ({ks})"

    plt.figure(figsize=(3.0, 3.0), dpi=200)
    plt.title(title, fontsize=9)
    plt.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        textprops={"fontsize": 7},
        labeldistance=1.15,
        pctdistance=0.70,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def opening_paragraph(top_runs, params_by_tag, *, manual_tag: str, manual_run=None):
    if not top_runs:
        return "No runs with throughput results were found in benchmark_summary.json."

    best = top_runs[0]
    tag = best.get("tag", "unknown")
    tps = safe_float((best.get("performance") or {}).get("timesteps_per_s"))
    tps_s = f"{tps:.3f}" if tps is not None else "n/a"

    p = params_by_tag.get(tag, {})
    ks = p.get("ks") or (best.get("kspace") or {}).get("style") or "unknown"

    extra = []
    for k in sorted(p.keys()):
        if k in {"run_id", "input", "lmp"}:
            continue
        if k in {"case", "ks"}:
            continue
        if isinstance(p.get(k), (str, int, float)):
            extra.append(f"{k}={format_param_value(p.get(k))}")

    extra_s = "; ".join(extra) if extra else "No sweep parameters were found in params.json for this run."

    manual_clause = ""
    if manual_run is not None:
        mtps = safe_float((manual_run.get("performance") or {}).get("timesteps_per_s"))
        if mtps is not None and tps is not None and mtps > 0:
            speedup = tps / mtps
            manual_clause = (
                f" Manual baseline <b>{manual_tag}</b> achieved <b>{mtps:.3f} timesteps/s</b> "
                f"(speedup vs baseline: <b>{speedup:.2f}×</b>)."
            )

    return (
        f"Fastest throughput in this sweep was achieved by <b>{tag}</b> at "
        f"<b>{tps_s} timesteps/s</b>. "
        f"Run parameters: kstyle=<b>{ks}</b>. "
        f"Sweep parameters recorded in params.json: {extra_s}."
        f"{manual_clause}"
    )


def build_pdf(
    *,
    summary: dict,
    top_runs: list,
    table_runs: list,
    pie_paths: list[Path],
    params_by_tag: dict,
    out_pdf: Path,
    manual_tag: str,
):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodySmall",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    story.append(Paragraph("Performance Review", styles["Title"]))
    story.append(Spacer(1, 0.10 * inch))

    manual_run = None
    for r in (summary.get("runs") or []):
        if r.get("tag") == manual_tag:
            manual_run = r
            break

    manual_tps = None
    if manual_run is not None:
        manual_tps = safe_float((manual_run.get("performance") or {}).get("timesteps_per_s"))

    story.append(
        Paragraph(
            opening_paragraph(top_runs, params_by_tag, manual_tag=manual_tag, manual_run=manual_run),
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    params_list = [params_by_tag.get(r.get("tag", "unknown"), {}) for r in table_runs]
    param_cols = collect_param_columns(params_list)
    param_cols = [c for c in param_cols if c != "kacc"]

    header = ["Rank", "kacc", "Tag", "timesteps/s", "speedup vs manual", "walltime"] + param_cols
    rows = [header]

    for i, r in enumerate(table_runs, start=1):
        tag = r.get("tag", "unknown")
        tps = safe_float((r.get("performance") or {}).get("timesteps_per_s"))
        tps_s = f"{tps:.3f}" if tps is not None else "n/a"

        speedup_s = "n/a"
        if manual_tps is not None and tps is not None and manual_tps > 0:
            speedup_s = f"{(tps / manual_tps):.2f}x"

        walltime = r.get("total_wall_time") or "n/a"

        p = params_by_tag.get(tag, {})
        kacc_s = format_param_value(p.get("kacc"))
        row = [str(i), kacc_s, tag, tps_s, speedup_s, walltime] + [
            format_param_value(p.get(k)) for k in param_cols
        ]
        rows.append(row)

    page_w = A4[0] - (doc.leftMargin + doc.rightMargin)
    base_widths = [0.45 * inch, 0.70 * inch, 1.05 * inch, 0.85 * inch, 0.90 * inch, 0.85 * inch]
    remaining = page_w - sum(base_widths)
    extra_w = remaining / max(1, len(param_cols))
    col_widths = base_widths + [extra_w] * len(param_cols)

    tbl = Table(rows, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("ALIGN", (4, 1), (4, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.12 * inch))

    imgs = []
    for p in pie_paths:
        img = Image(str(p))
        img.drawWidth = 2.35 * inch
        img.drawHeight = 2.35 * inch
        imgs.append(img)

    while len(imgs) < 3:
        imgs.append(Spacer(1, 2.35 * inch))

    pie_tbl = Table([imgs], colWidths=[2.45 * inch, 2.45 * inch, 2.45 * inch])
    pie_tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(pie_tbl)

    timed_out = []
    for r in (summary.get("runs") or []):
        runner = r.get("runner") or {}
        if runner.get("timed_out") is True:
            timed_out.append(r)

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Timed Out Runs", styles["Heading2"]))
    story.append(Spacer(1, 0.06 * inch))

    if not timed_out:
        story.append(Paragraph("No runs timed out.", body))
    else:
        to_rows = [["Tag", "timeout_s", "elapsed_s", "returncode", "note"]]
        for r in timed_out:
            tag = r.get("tag", "unknown")
            runner = r.get("runner") or {}
            to_rows.append(
                [
                    str(tag),
                    fmt_num(runner.get("timeout_s")),
                    fmt_num(runner.get("time_s")),
                    str(runner.get("returncode")) if runner.get("returncode") is not None else "n/a",
                    str(runner.get("note") or "timeout"),
                ]
            )

        page_w2 = A4[0] - (doc.leftMargin + doc.rightMargin)
        col_widths2 = [
            1.20 * inch,
            0.85 * inch,
            0.85 * inch,
            0.80 * inch,
            page_w2 - (1.20 + 0.85 + 0.85 + 0.80) * inch,
        ]
        to_tbl = Table(to_rows, colWidths=col_widths2, hAlign="LEFT")
        to_tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.append(to_tbl)

    doc.build(story)


def generate_performance_review(
    *,
    runs_dir: Path = Path("runs"),
    json_path: Path | None = None,
    out_pdf: Path | None = None,
    top_n_runs: int = 3,
    merge_lt_pct: float = 5.0,
    manual_tag: str = "manual",
    sweep_input: str = "in.performance_test.lmp",
):
    json_path = json_path or (runs_dir / "benchmark_summary.json")
    out_pdf = out_pdf or (runs_dir / "performance_review.pdf")

    summary = json.loads(json_path.read_text())
    runs = summary.get("runs", [])

    params_by_tag = {}
    for r in runs:
        tag = r.get("tag", "unknown")
        if tag in params_by_tag:
            continue
        params_by_tag[tag] = load_params_for_tag(runs_dir, tag)

    sweep_runs = []
    for r in runs:
        tag = r.get("tag", "unknown")
        if tag == manual_tag:
            continue
        p = params_by_tag.get(tag, {})
        if p.get("input") == sweep_input:
            sweep_runs.append(r)

    top_runs = top_runs_by_tps(sweep_runs, top_n_runs)
    if not top_runs:
        raise SystemExit("No runs with performance.timesteps_per_s found in benchmark_summary.json.")

    table_runs = best_runs_by_param(sweep_runs, params_by_tag, param_key="kacc")
    if not table_runs:
        table_runs = top_runs

    tmp_dir = runs_dir / "_report_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pie_paths: list[Path] = []
    for i, r in enumerate(top_runs, start=1):
        tag = r.get("tag", "unknown")
        p = tmp_dir / f"pie_{i}_{tag}.png"
        make_timing_pie(r, p, merge_lt_pct=merge_lt_pct)
        pie_paths.append(p)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(
        summary=summary,
        top_runs=top_runs,
        table_runs=table_runs,
        pie_paths=pie_paths,
        params_by_tag=params_by_tag,
        out_pdf=out_pdf,
        manual_tag=manual_tag,
    )

    print(f"Wrote: {out_pdf}")
    return out_pdf
