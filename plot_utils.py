import json
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from reportlab.graphics.shapes import Drawing, String, Group
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics.renderPDF import GraphicsFlowable
from reportlab.graphics import renderPDF


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

def _parse_hms_walltime_s(s: str | None) -> float | None:
    """Parse walltime strings like H:MM:SS or MM:SS into seconds."""
    if not s:
        return None
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(p) for p in parts]
    except Exception:
        return None

    if len(nums) == 2:
        m, sec = nums
        return m * 60.0 + sec
    h, m, sec = nums
    return h * 3600.0 + m * 60.0 + sec

def max_wall_time(summary: dict) -> tuple[float | None, str | None, dict | None]:
    """Return (max_seconds, tag, run_obj) across all finished runs in summary."""
    best_s: float | None = None
    best_tag: str | None = None
    best_run: dict | None = None
    for r in (summary.get("runs") or []):
        tag = r.get("tag")
        runner = r.get("runner") or {}

        # Only consider runs that finished (exclude timeouts; and exclude nonzero returncode if known).
        if runner.get("timed_out") is True:
            continue
        if runner.get("returncode") is not None and runner.get("returncode") != 0:
            continue

        t = safe_float(runner.get("time_s"))
        if t is None:
            t = _parse_hms_walltime_s(r.get("total_wall_time"))
        if t is None:
            continue
        if best_s is None or t > best_s:
            best_s = t
            best_tag = tag
            best_run = r
    return best_s, best_tag, best_run


def _extract_last_loop_metrics_from_log_text(text: str) -> tuple[int | None, int | None]:
    """Return (atoms, steps) from the last 'Loop time...' line in a log.

    For multi-run scripts, the last loop block typically corresponds to the
    production run.
    """
    patt = re.compile(
        r"Loop time of\s+[0-9.]+\s+on\s+\d+\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms"
    )
    hits = patt.findall(text)
    if not hits:
        return None, None
    steps_s, atoms_s = hits[-1]
    try:
        return int(atoms_s), int(steps_s)
    except Exception:
        return None, None


def extract_production_size_from_run(run: dict, *, runs_dir: Path) -> tuple[int | None, int | None]:
    """Best-effort extraction of (atoms, production_steps) for a run."""
    atoms = run.get("atoms") if isinstance(run.get("atoms"), int) else None
    steps = run.get("steps") if isinstance(run.get("steps"), int) else None

    log_path = run.get("log_path")
    if not log_path:
        runner = run.get("runner") or {}
        log_path = runner.get("log_path")

    p = _resolve_log_path(log_path, runs_dir=runs_dir)
    if p is None:
        return atoms, steps

    try:
        atoms_last, steps_last = _extract_last_loop_metrics_from_log_text(p.read_text())
    except Exception:
        return atoms, steps

    if atoms_last is not None:
        atoms = atoms_last
    if steps_last is not None:
        steps = steps_last
    return atoms, steps

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


def _resolve_log_path(log_path: str | None, *, runs_dir: Path) -> Path | None:
    if not log_path:
        return None
    p = Path(log_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    alt = runs_dir.parent / p
    return alt


def _extract_final_total_energy_from_log_text(text: str) -> float | None:
    # Tries to read the last thermo row that contains a total-energy column.
    # Common LAMMPS thermo keys: TotEng, Etot, E_total, etotal.
    candidates = {"toteng", "etot", "e_total", "etotal"}

    last_value: float | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("Step"):
            i += 1
            continue

        cols = line.split()
        col_l = [c.strip().lower() for c in cols]
        try:
            idx = next(j for j, c in enumerate(col_l) if c in candidates)
        except StopIteration:
            i += 1
            continue

        # Consume following numeric thermo rows
        i += 1
        while i < len(lines):
            dl = lines[i].strip()
            if not dl:
                break
            parts = dl.split()
            # Thermo rows always start with Step (numeric)
            try:
                float(parts[0])
            except Exception:
                break
            if len(parts) > idx:
                try:
                    last_value = float(parts[idx])
                except Exception:
                    pass
            i += 1

        continue

    return last_value


def _extract_last_timesteps_per_s_from_log_text(text: str) -> float | None:
    # Example:
    # Performance: 4680.949 tau/day, 27.089 timesteps/s, 81.266 katom-step/s
    last: float | None = None
    for line in text.splitlines():
        if "Performance:" not in line:
            continue
        m = re.search(r"([0-9.]+)\s*timesteps/s", line)
        if not m:
            continue
        try:
            last = float(m.group(1))
        except Exception:
            pass
    return last


def _find_latest_scaling_summary(runs_dir: Path) -> Path | None:
    candidates = list(runs_dir.glob("**/scaling_summary.json"))
    if not candidates:
        return None
    try:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        candidates.sort()
    return candidates[0]


def _load_scaling_speedup_points(runs_dir: Path) -> tuple[list[tuple[int, float]], int | None, float | None]:
    """Return (points, base_cores, base_tps).

    points is a sorted list of (cores, speedup) for scaling runs that have logs.
    Speedup is computed from timesteps/s relative to the minimum-cores run.
    """

    scaling_path = _find_latest_scaling_summary(runs_dir)
    if scaling_path is None:
        return [], None, None

    try:
        scaling = json.loads(scaling_path.read_text())
    except Exception:
        return [], None, None

    best_tps_by_cores: dict[int, float] = {}
    for r in (scaling.get("runs") or []):
        try:
            cores = int(r.get("cores"))
        except Exception:
            continue

        log_path = r.get("log_path")
        p = _resolve_log_path(log_path, runs_dir=runs_dir)
        if p is None or not p.exists():
            continue

        try:
            tps = _extract_last_timesteps_per_s_from_log_text(p.read_text(errors="replace"))
        except Exception:
            continue
        if tps is None or tps <= 0:
            continue

        prev = best_tps_by_cores.get(cores)
        if prev is None or tps > prev:
            best_tps_by_cores[cores] = float(tps)

    if not best_tps_by_cores:
        return [], None, None

    base_cores = min(best_tps_by_cores.keys())
    base_tps = best_tps_by_cores.get(base_cores)
    if base_tps is None or base_tps <= 0:
        return [], None, None

    points = [(c, best_tps_by_cores[c] / base_tps) for c in sorted(best_tps_by_cores.keys())]
    return points, base_cores, base_tps


def make_scaling_speedup_plot(
    *,
    points: list[tuple[int, float]],
    base_cores: int,
    base_tps: float,
    title: str = "Scaling: speedup vs cores",
    width: float = 480,
    height: float = 240,
) -> Drawing:
    d = Drawing(width, height)
    d.add(String(width / 2, height - 14, title, textAnchor="middle", fontSize=11))

    if not points:
        d.add(String(width / 2, height / 2, "No scaling results found.", textAnchor="middle", fontSize=10))
        return d

    x_vals = [c for c, _ in points]
    min_x, max_x = min(x_vals), max(x_vals)
    max_measured = max(s for _, s in points)

    actual = [(float(c), float(s)) for c, s in points]
    ideal = [(float(c), float(c) / float(base_cores)) for c in x_vals]
    max_ideal = max(float(c) / float(base_cores) for c in x_vals)

    def _nice_step(step: float) -> float:
        if step <= 0:
            return 1.0
        # Round to 1, 2, 5 × 10^k
        import math

        k = math.floor(math.log10(step))
        base = step / (10**k)
        if base <= 1:
            nice = 1
        elif base <= 2:
            nice = 2
        elif base <= 5:
            nice = 5
        else:
            nice = 10
        return float(nice) * (10**k)

    lp = LinePlot()
    lp.x = 52
    lp.y = 38
    lp.width = width - 70
    lp.height = height - 68
    lp.data = [actual, ideal]

    lp.xValueAxis.valueMin = float(min_x)
    lp.xValueAxis.valueMax = float(max_x)
    lp.xValueAxis.valueSteps = [float(v) for v in x_vals]
    lp.xValueAxis.labelTextFormat = '%d'
    lp.xValueAxis.labels.fontSize = 8

    y_min = 1.0
    y_max = max(1.1, float(max(max_measured, max_ideal)) * 1.1)
    lp.yValueAxis.valueMin = y_min
    lp.yValueAxis.valueMax = y_max
    lp.yValueAxis.labels.fontSize = 8

    # Choose explicit y ticks so the grid and the right-side axis align.
    try:
        span = float(y_max) - float(y_min)
        step = _nice_step(span / 4.0)  # ~5 ticks including endpoints
        ticks = []
        v = float(y_min)
        # Keep ticks within [y_min, y_max]
        while v <= float(y_max) + 1e-9:
            ticks.append(v)
            v += step
        if len(ticks) >= 2:
            lp.yValueAxis.valueSteps = ticks
    except Exception:
        ticks = []

    # Grid
    for ax in (lp.xValueAxis, lp.yValueAxis):
        if hasattr(ax, "visibleGrid"):
            ax.visibleGrid = True
        if hasattr(ax, "gridStrokeColor"):
            ax.gridStrokeColor = colors.lightgrey
        if hasattr(ax, "gridStrokeWidth"):
            ax.gridStrokeWidth = 0.25

    lp.lines[0].strokeColor = colors.darkblue
    lp.lines[0].strokeWidth = 1.5
    lp.lines[0].symbol = makeMarker('FilledCircle')
    lp.lines[0].symbol.size = 4

    lp.lines[1].strokeColor = colors.grey
    lp.lines[1].strokeWidth = 1.0
    lp.lines[1].strokeDashArray = [3, 2]

    d.add(lp)
    g = Group()
    g.add(String(0, 0, "Speedup (×)", textAnchor="middle", fontSize=9))
    g.translate(30, height/2)
    g.rotate(90)
    d.add(g)
    g = Group()
    g.add(String(0, 0, "Timesteps/s", textAnchor="middle", fontSize=9))
    g.translate(width + 10, height / 2)
    g.rotate(270)
    d.add(g)
    d.add(String(width / 2, 14, "Cores", textAnchor="middle", fontSize=9))
    d.add(String(width - 12, height - 30, "Ideal", textAnchor="end", fontSize=8, fillColor=colors.grey))
    d.add(String(width - 12, height - 42, "Measured", textAnchor="end", fontSize=8, fillColor=colors.darkblue))

    # Right-side tick labels for timesteps/s (derived from speedup × base_tps).
    # This avoids needing a second plot/axis object while still giving an absolute scale.
    if base_tps and base_tps > 0:
        right_x = lp.x + lp.width + 6
        if not ticks:
            # Fallback ticks if we couldn't compute explicit valueSteps above.
            ticks = [y_min, (y_min + y_max) / 2.0, y_max]
        for s in ticks:
            try:
                frac = (float(s) - float(y_min)) / (float(y_max) - float(y_min)) if y_max != y_min else 0.0
                y_px = lp.y + frac * lp.height
                tps = float(s) * float(base_tps)
                d.add(String(right_x, y_px - 3, fmt_num(tps, digits=4), fontSize=8, fillColor=colors.black))
            except Exception:
                continue

    return d


def extract_final_total_energy_from_run(run: dict, *, runs_dir: Path) -> float | None:
    log_path = run.get("log_path")
    if not log_path:
        runner = run.get("runner") or {}
        log_path = runner.get("log_path")

    p = _resolve_log_path(log_path, runs_dir=runs_dir)
    if p is None:
        return None
    try:
        return _extract_final_total_energy_from_log_text(p.read_text())
    except Exception:
        return None


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


def make_timing_pie(
    run,
    out_path: Path,
    merge_lt_pct: float,
    *,
    title: str | None = None,
    width: float = 240,
    height: float = 240,
):
    timing = run.get("timing_breakdown_pct") or {}
    items = [(k, float(v)) for k, v in timing.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)

    labels, values = merge_small_slices(items, merge_lt_pct=merge_lt_pct)

    if title is None:
        title = f"{run.get('tag', 'unknown')}"
        ks = (run.get("kspace") or {}).get("style")
        if ks:
            title += f"  ({ks})"

    d = Drawing(width, height)
    title_h = 18
    pad = 14
    d.add(String(width / 2, height - (title_h - 3), title, textAnchor="middle", fontSize=11))

    if not values or sum(values) <= 0:
        d.add(String(width / 2, height / 2, "n/a", textAnchor="middle", fontSize=10))
    else:
        pie = Pie()
        avail_w = max(10, width - 2 * pad)
        avail_h = max(10, height - title_h - 2 * pad)
        diam = max(10, min(avail_w, avail_h))

        pie.width = diam
        pie.height = diam
        pie.x = (width - diam) / 2
        pie.y = pad + (avail_h - diam) / 2
        pie.data = values
        pie.labels = [f"{lab} ({val:.0f}%)" for lab, val in zip(labels, values)]

        # Use side labels to avoid clutter inside slices.
        pie.sideLabels = 1
        pie.simpleLabels = 0
        pie.sideLabelsOffset = 0.16
        pie.slices.strokeWidth = 0.25
        pie.slices.fontSize = 11
        pie.slices.fontName = "Helvetica"
        pie.slices.fontColor = colors.black

        palette = [
            colors.darkblue,
            colors.darkgreen,
            colors.darkred,
            colors.purple,
            colors.darkorange,
            colors.teal,
            colors.brown,
            colors.indigo,
        ]
        for i in range(len(values)):
            pie.slices[i].fillColor = palette[i % len(palette)]

        d.add(pie)

    out_pdf = out_path.with_suffix(".pdf")
    renderPDF.drawToFile(d, str(out_pdf))
    return {"pdf": out_pdf, "drawing": d}


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
                f"(speedup: <b>{speedup:.2f}×</b>)."
            )


def build_pdf(
    *,
    summary: dict,
    top_runs: list,
    table_runs: list,
    pie_drawings: list[Drawing],
    scaling_drawing: Drawing | None,
    params_by_tag: dict,
    runs_dir: Path,
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

    manual_energy = extract_final_total_energy_from_run(manual_run, runs_dir=runs_dir) if manual_run else None

    params_list = [params_by_tag.get(r.get("tag", "unknown"), {}) for r in table_runs]
    param_cols = collect_param_columns(params_list)
    param_cols = [c for c in param_cols if c not in {"kacc", "ks", "dcut"}]

    header = ["Rank", "kacc", "ks", "dcut", "Tag", "timesteps/s", "speedup", "ΔE%", "walltime"] + param_cols
    rows = [header]

    for i, r in enumerate(table_runs, start=1):
        tag = r.get("tag", "unknown")
        tps = safe_float((r.get("performance") or {}).get("timesteps_per_s"))
        tps_s = f"{tps:.3f}" if tps is not None else "n/a"

        speedup_s = "n/a"
        if manual_tps is not None and tps is not None and manual_tps > 0:
            speedup_s = f"{(tps / manual_tps):.2f}x"

        energy_diff_s = "n/a"
        if manual_energy is not None and manual_energy != 0:
            sweep_energy = extract_final_total_energy_from_run(r, runs_dir=runs_dir)
            if sweep_energy is not None:
                energy_diff_pct = (sweep_energy - manual_energy) / manual_energy * 100.0
                energy_diff_s = f"{energy_diff_pct:+.3f}%"

        walltime = r.get("total_wall_time") or "n/a"

        p = params_by_tag.get(tag, {})
        kacc_s = format_param_value(p.get("kacc"))
        ks_s = format_param_value(p.get("ks") or (r.get("kspace") or {}).get("style"))
        dcut_s = format_param_value(p.get("dcut"))
        row = [str(i), kacc_s, ks_s, dcut_s, tag, tps_s, speedup_s, energy_diff_s, walltime] + [
            format_param_value(p.get(k)) for k in param_cols
        ]
        rows.append(row)

    page_w = A4[0] - (doc.leftMargin + doc.rightMargin)
    base_widths = [
        0.45 * inch,  # Rank
        0.65 * inch,  # kacc
        0.80 * inch,  # ks
        0.65 * inch,  # dcut
        0.95 * inch,  # Tag
        0.75 * inch,  # timesteps/s
        0.70 * inch,  # speedup
        0.75 * inch,  # ΔE%
        0.80 * inch,  # walltime
    ]
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
                ("ALIGN", (5, 1), (5, -1), "RIGHT"),
                ("ALIGN", (6, 1), (6, -1), "RIGHT"),
                ("ALIGN", (7, 1), (7, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    story.append(Paragraph("Overview", styles["Heading2"]))

    mx_s, mx_tag, mx_run = max_wall_time(summary)
    if mx_s is not None:
        mx_line = f"Max wall time (finished runs): <b>{fmt_num(mx_s, digits=4)} s</b>"
        if mx_tag:
            mx_line += f" (tag: <b>{mx_tag}</b>)"
        if mx_run is not None:
            mx_atoms, mx_prod_steps = extract_production_size_from_run(mx_run, runs_dir=runs_dir)
            atoms_s = f"{mx_atoms:,}" if isinstance(mx_atoms, int) else "n/a"
            prod_steps_s = f"{mx_prod_steps:,}" if isinstance(mx_prod_steps, int) else "n/a"
            mx_line += (
                f". Size: <b>{atoms_s} atoms</b>; "
                f"production run length: <b>{prod_steps_s} timesteps</b>"
            )
        mx_line += "."
        story.append(Paragraph(mx_line, body))

    table_desc = (
    "This table shows the best-performing sweep runs for each value of kacc. "
    "Runs are ranked by timesteps/s. "
    "Speedup is relative to the manual baseline run. "
    "ΔE% is the final total energy difference relative to the manual run. "
    )
    story.append(Paragraph(table_desc, body))
    story.append(Spacer(1, 0.06 * inch))
    story.append(tbl)
    story.append(Spacer(1, 0.10 * inch))

    pie_desc = ("Timing breakdown pie charts for the runs listed in the table (one chart per kacc, titled by rank).")
    story.append(Paragraph(pie_desc, body))
    story.append(Spacer(1, 0.06 * inch))

    if pie_drawings:
        cols = 2
        pie_page_w = A4[0] - (doc.leftMargin + doc.rightMargin)
        cell_w = pie_page_w / cols
        draw_w = cell_w * 0.60
        draw_h = draw_w

        charts = []
        for d in pie_drawings:
            try:
                sx = draw_w / float(getattr(d, "width", draw_w) or draw_w)
                sy = draw_h / float(getattr(d, "height", draw_h) or draw_h)
                s = min(sx, sy)
            except Exception:
                s = 1.0

            if s and abs(s - 1.0) > 1e-6:
                d.scale(s, s)
                if hasattr(d, "width"):
                    d.width = float(getattr(d, "width", draw_w)) * s
                if hasattr(d, "height"):
                    d.height = float(getattr(d, "height", draw_h)) * s

            charts.append(GraphicsFlowable(d))

        img_rows = []
        for i in range(0, len(charts), cols):
            row = charts[i : i + cols]
            while len(row) < cols:
                row.append(Spacer(1, draw_h))
            img_rows.append(row)

        pie_tbl = Table(img_rows, colWidths=[cell_w] * cols)
        pie_tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(pie_tbl)

    # Optional scaling plot (if scaling_summary.json + logs exist).
    if scaling_drawing is not None:
        story.append(PageBreak())
        story.append(Paragraph("Scaling", styles["Heading2"]))
        story.append(Spacer(1, 0.06 * inch))
        story.append(
            Paragraph(
                "Speedup vs cores from scaling runs (speedup computed from timesteps/s relative to the smallest core count).",
                body,
            )
        )
        story.append(Spacer(1, 0.16 * inch))

        # Fit to page width.
        try:
            page_w3 = A4[0] - (doc.leftMargin + doc.rightMargin)
            sx = page_w3 / float(getattr(scaling_drawing, "width", page_w3) or page_w3)
            if sx and abs(sx - 1.0) > 1e-6:
                scaling_drawing.scale(sx, sx)
        except Exception:
            pass
        story.append(GraphicsFlowable(scaling_drawing))

    timed_out = []
    for r in (summary.get("runs") or []):
        runner = r.get("runner") or {}
        if runner.get("timed_out") is True:
            timed_out.append(r)

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Timed Out Runs", styles["Heading2"]))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(f"Timed out runs: <b>{len(timed_out)}</b>.", body))

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

    pie_drawings: list[Drawing] = []
    for i, r in enumerate(table_runs, start=1):
        tag = r.get("tag", "unknown")
        base = tmp_dir / f"pie_{i:02d}_{tag}"
        outs = make_timing_pie(r, base, merge_lt_pct=merge_lt_pct, title=f"Rank {i}")
        pie_drawings.append(outs["drawing"])

    scaling_drawing = None
    scaling_points, base_cores, base_tps = _load_scaling_speedup_points(runs_dir)
    if scaling_points and base_cores is not None and base_tps is not None:
        scaling_drawing = make_scaling_speedup_plot(points=scaling_points, base_cores=base_cores, base_tps=base_tps)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(
        summary=summary,
        top_runs=top_runs,
        table_runs=table_runs,
        pie_drawings=pie_drawings,
        scaling_drawing=scaling_drawing,
        params_by_tag=params_by_tag,
        runs_dir=runs_dir,
        out_pdf=out_pdf,
        manual_tag=manual_tag,
    )

    print(f"Wrote: {out_pdf}")
    return out_pdf
