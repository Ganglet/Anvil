import io
import datetime
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from clustering.taxonomy import VulnerabilityTaxonomy
from agent.schema import ExplanationReport
from patching.schema import PatchReport

# ── Professional palette ──────────────────────────────────────────────────────
# One accent colour family: steel blue. Red only for critical failures.
_NAVY    = HexColor("#0f1e2e")   # deepest — cover header, table headers
_NAVY2   = HexColor("#1e3a5f")   # section accents
_ACCENT  = HexColor("#2563eb")   # primary interactive blue
_LIGHT_A = HexColor("#dbeafe")   # very light blue
_SLATE   = HexColor("#64748b")   # secondary text
_BODY    = HexColor("#1e293b")   # body text
_LIGHT   = HexColor("#f8fafc")   # table alternating row
_BORDER  = HexColor("#e2e8f0")   # subtle borders
_SUCCESS = HexColor("#059669")   # emerald — used sparingly
_DANGER  = HexColor("#dc2626")   # red — failures only
_WARN    = HexColor("#d97706")   # amber — medium risk
_WHITE   = colors.white

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 4 * cm

# Matplotlib string equivalents
_M = {
    "navy":    "#0f1e2e",
    "navy2":   "#1e3a5f",
    "accent":  "#2563eb",
    "light_a": "#dbeafe",
    "slate":   "#64748b",
    "body":    "#1e293b",
    "light":   "#f8fafc",
    "border":  "#e2e8f0",
    "success": "#059669",
    "danger":  "#dc2626",
    "warn":    "#d97706",
    "white":   "#ffffff",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid":        True,
    "grid.color":       "#e2e8f0",
    "grid.linewidth":   0.6,
    "axes.facecolor":   "#f8fafc",
    "figure.facecolor": "white",
    "text.color":       "#1e293b",
    "axes.labelcolor":  "#1e293b",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})


# ── Chart builders ────────────────────────────────────────────────────────────

def _buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def _to_img(fig, w_cm, h_cm) -> Image:
    return Image(_buf(fig), width=w_cm * cm, height=h_cm * cm)


def _radar(categories, values_list, labels_list, colors_list,
           title, w_cm, h_cm) -> Image:
    """Shared polar radar builder. values_list: list of lists (one per series), all in [0, 1]."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(_M["light"])
    ax.spines["polar"].set_color(_M["border"])
    ax.grid(color=_M["border"], linewidth=0.7)
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=7.5, color=_M["body"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color=_M["slate"])

    for vals, lbl, col in zip(values_list, labels_list, colors_list):
        v_closed = list(vals) + [vals[0]]
        ax.plot(angles_closed, v_closed, color=col, linewidth=2.2, label=lbl)
        ax.fill(angles_closed, v_closed, color=col, alpha=0.18)

    ax.set_title(title, fontsize=10, fontweight="bold", color=_M["body"],
                 pad=18, y=1.1)
    if len(labels_list) > 1:
        ax.legend(fontsize=7.5, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.35, 1.2))
    fig.tight_layout()
    return _to_img(fig, w_cm, h_cm)


def _hbar(categories, values, value_fmt, title, w_cm, h_cm) -> Image:
    """Horizontal bar chart — fallback when N < 3 (too few axes for radar)."""
    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54))
    y = list(range(len(categories)))
    ax.barh(y, values, color=_M["accent"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8.5)
    mx = max(values) if values else 1
    ax.set_xlim(0, mx * 1.3 if mx > 0 else 1)
    for i, val in enumerate(values):
        ax.text(val + mx * 0.03, i, value_fmt(val), va="center",
                fontsize=8, color=_M["body"])
    ax.set_title(title, fontsize=10, fontweight="bold", color=_M["body"], pad=6)
    ax.set_facecolor(_M["light"])
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_img(fig, w_cm, h_cm)


def _chart_attack_radar(attack_rates: dict, w_cm=13, h_cm=5.5) -> Image:
    """Radar — one axis per attack type, value = success rate."""
    categories = list(attack_rates.keys())
    values = [attack_rates[c] for c in categories]
    if len(categories) < 3:
        return _hbar(categories, values, lambda v: f"{v:.1%}",
                     "Attack Success Rates", w_cm, h_cm)
    return _radar(categories, [values], ["Success Rate"],
                  [_M["accent"]], "Attack Success Rates", w_cm, h_cm)


def _chart_layer_radar(profile: dict, w_cm=13, h_cm=5.5) -> Image | None:
    """Radar — one axis per layer, value = normalised gradient norm."""
    grad = profile.get("gradient_norms", {})
    priority = profile.get("attack_priority", [])
    layers = [l for l in priority if l in grad]
    if not layers:
        return None
    raw = [grad[l] for l in layers]
    mx = max(raw) or 1.0
    norm = [v / mx for v in raw]
    if len(layers) < 3:
        return _hbar(layers, raw, lambda v: f"{v:.3f}",
                     "Layer Vulnerability (Gradient Norm)", w_cm, h_cm)
    return _radar(layers, [norm], ["Gradient Norm (norm.)"],
                  [_M["navy2"]], "Layer Vulnerability Profile", w_cm, h_cm)


def _chart_cluster_radar(taxonomy: VulnerabilityTaxonomy,
                          w_cm=13, h_cm=5.0) -> Image | None:
    """Radar — one axis per cluster, value = normalised failure count."""
    clusters = taxonomy.clusters
    if not clusters:
        return None
    names = [f"C{c.cluster_id}\n{c.dominant_attack}" for c in clusters]
    sizes = [c.size for c in clusters]
    mx = max(sizes) or 1
    norm = [s / mx for s in sizes]
    noise_note = f"  (+{taxonomy.noise_count} noise)" if taxonomy.noise_count else ""
    title = f"Vulnerability Cluster Sizes{noise_note}"
    if len(clusters) < 3:
        bar_names = [f"C{c.cluster_id} ({c.dominant_attack})" for c in clusters]
        return _hbar(bar_names, sizes, lambda v: str(int(v)), title, w_cm, h_cm)
    return _radar(names, [norm], ["Failures (norm.)"],
                  [_M["navy"]], title, w_cm, h_cm)


def _chart_patch_radar(patch_report: PatchReport, w_cm=13, h_cm=5.5) -> Image | None:
    """Multi-series radar — 4 quality dimensions, one polygon per cluster."""
    results = patch_report.results
    if not results:
        return None
    categories = ["Safety\nScore", "Resistance\nGain",
                  "Accuracy\nRetention", "Effort\n(1−retries/3)"]
    palette = [_M["accent"], _M["navy"], _M["navy2"], _M["success"], _M["warn"]]
    values_list, labels_list, colors_list = [], [], []
    for r, col in zip(results, palette * 10):
        vals = [
            min(max(r.safety_score, 0.0), 1.0),
            min(max(r.resistance_gain, 0.0), 1.0),
            min(max(1.0 - r.accuracy_drop, 0.0), 1.0),
            max(1.0 - r.retries / 3.0, 0.0),
        ]
        values_list.append(vals)
        labels_list.append(f"C{r.cluster_id}")
        colors_list.append(col)
    return _radar(categories, values_list, labels_list, colors_list,
                  "Patch Quality Profile", w_cm, h_cm)


# ── ReportLab styles ──────────────────────────────────────────────────────────

def _styles() -> dict:
    return {
        "cover_title": ParagraphStyle(
            "ct", fontName="Helvetica-Bold", fontSize=42,
            textColor=_WHITE, alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cs", fontName="Helvetica", fontSize=11,
            textColor=HexColor("#94a3b8"), alignment=TA_CENTER,
        ),
        "stat_num": ParagraphStyle(
            "sn", fontName="Helvetica-Bold", fontSize=24,
            textColor=_WHITE, alignment=TA_CENTER, leading=28,
        ),
        "stat_lbl": ParagraphStyle(
            "sl", fontName="Helvetica", fontSize=8,
            textColor=HexColor("#94a3b8"), alignment=TA_CENTER, leading=11,
        ),
        "section": ParagraphStyle(
            "sec", fontName="Helvetica-Bold", fontSize=13,
            textColor=_NAVY, spaceBefore=14, spaceAfter=4,
        ),
        "cluster_h": ParagraphStyle(
            "clh", fontName="Helvetica-Bold", fontSize=10, textColor=_WHITE,
        ),
        "cluster_meta": ParagraphStyle(
            "clm", fontName="Helvetica", fontSize=10, textColor=_WHITE,
            alignment=TA_RIGHT,
        ),
        "body": ParagraphStyle(
            "bod", fontName="Helvetica", fontSize=9, textColor=_BODY,
            leading=14, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "sm", fontName="Helvetica", fontSize=8, textColor=_SLATE,
            leading=12, spaceAfter=2,
        ),
        "label": ParagraphStyle(
            "lbl", fontName="Helvetica-Bold", fontSize=9,
            textColor=_BODY, spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Helvetica", fontSize=8.5,
            textColor=_BODY, leading=12,
        ),
        "cell_b": ParagraphStyle(
            "cellb", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=_BODY, leading=12,
        ),
        "cell_w": ParagraphStyle(
            "cellw", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=_WHITE, leading=12,
        ),
    }


def _hr():
    return HRFlowable(width="100%", thickness=2, color=_ACCENT,
                      spaceAfter=8, spaceBefore=0)


def _section(title, s):
    return [Paragraph(title, s["section"]), _hr()]


def _table(rows, widths, extra=None):
    cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  _NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_LIGHT, _WHITE]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 9),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 9),
        ("BOX",           (0, 0), (-1, -1), 0.5, _BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, _BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    if extra:
        cmds += extra
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle(cmds))
    return t


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_SLATE)
    canvas.drawString(2 * cm, 1.1 * cm,
                      "ANVIL — Adversarial Neural Vulnerability Inspection and Learning")
    canvas.drawRightString(PAGE_W - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(_BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, 1.4 * cm, PAGE_W - 2 * cm, 1.4 * cm)
    canvas.restoreState()


# ── Cover ─────────────────────────────────────────────────────────────────────

def _cover(s, model_name, date_str, vuln_score,
           num_clusters, total_failures, patched,
           total_examples, total_fooled):
    story = []

    # Header — single table, explicit row heights eliminates overlap
    hdr = Table(
        [
            [Paragraph("ANVIL", s["cover_title"])],
            [Paragraph("Adversarial Neural Vulnerability Inspection and Learning",
                       s["cover_sub"])],
        ],
        colWidths=[CONTENT_W],
        rowHeights=[3.6 * cm, 1.2 * cm],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _NAVY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(hdr)

    # Thin accent stripe under header
    stripe = Table([[""]], colWidths=[CONTENT_W], rowHeights=[0.15 * cm])
    stripe.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _ACCENT)]))
    story.append(stripe)
    story.append(Spacer(1, 0.5 * cm))

    # Stat boxes — all same navy, list of flowables per cell to avoid overlap
    bw = CONTENT_W / 3 - 0.05 * cm
    sc = _WARN if 0.25 < vuln_score <= 0.5 else (_DANGER if vuln_score > 0.5 else _SUCCESS)

    stat_rows = [[
        [Paragraph(f"{vuln_score:.3f}", s["stat_num"]),
         Spacer(1, 0.05 * cm),
         Paragraph("Vulnerability Score", s["stat_lbl"])],
        [Paragraph(str(num_clusters), s["stat_num"]),
         Spacer(1, 0.05 * cm),
         Paragraph("Clusters Found", s["stat_lbl"])],
        [Paragraph(f"{patched}/{num_clusters}", s["stat_num"]),
         Spacer(1, 0.05 * cm),
         Paragraph("Clusters Patched", s["stat_lbl"])],
    ]]
    stat = Table(stat_rows,
                 colWidths=[bw, bw, bw],
                 rowHeights=[2.4 * cm])
    stat.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _NAVY),
        ("BACKGROUND",    (0, 0), (0,  0),  sc),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEBEFORE",    (1, 0), (2, 0),   1, _WHITE),
    ]))
    story.append(stat)
    story.append(Spacer(1, 0.5 * cm))

    # Metadata grid
    meta = [
        ["Model", model_name, "Date", date_str],
        ["Inputs Attacked", str(total_examples),
         "Inputs Fooled", f"{total_fooled}  ({total_fooled / max(total_examples,1):.1%})"],
        ["Total Failures", str(total_failures), "Patched", f"{patched}/{num_clusters}"],
    ]
    cw = [2.8*cm, 5.2*cm, 2.8*cm, CONTENT_W - 10.8*cm]
    mt = Table(
        [[Paragraph(row[j], s["label"] if j % 2 == 0 else s["body"])
          for j in range(4)] for row in meta],
        colWidths=cw,
    )
    mt.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LIGHT, _WHITE]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, _BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, _BORDER),
        ("TEXTCOLOR",     (0, 0), (0, -1),  _SLATE),
        ("TEXTCOLOR",     (2, 0), (2, -1),  _SLATE),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Generated automatically by ANVIL — an autonomous adversarial ML auditing pipeline. "
        "Covers attack surface profiling, multi-strategy adversarial attacks, failure mode "
        "clustering, LLM-grounded vulnerability analysis, and autonomous patch validation.",
        s["small"],
    ))
    story.append(PageBreak())
    return story


# ── Executive Summary ─────────────────────────────────────────────────────────

def _executive_summary(s, profile, total_fooled,
                        total_examples, taxonomy, patch_report):
    story = _section("Executive Summary", s)

    ps = patch_report.summary()
    ts = taxonomy.summary()
    vuln = profile.get("vulnerability_score", 0)

    risk_key = "High" if vuln > 0.5 else "Medium" if vuln > 0.25 else "Low"
    rate = total_fooled / max(total_examples, 1)
    rate_key = "High" if rate > 0.7 else "Medium" if rate > 0.3 else "Low"

    rows_data = [
        ("Vulnerability Score",   f"{vuln:.3f}",  risk_key),
        ("Mean Saliency Score",   f"{profile.get('mean_saliency_score', 0):.3f}", "—"),
        ("Attack Success Rate",   f"{rate:.1%}",  rate_key),
        ("Vulnerability Clusters",str(ts["num_clusters"]), "—"),
        ("Noise Points",          str(ts["noise_count"]),  "Unclustered"),
        ("Clusters Patched",      f"{ps['patched']}/{ps['total_clusters']}",
         "Resolved" if ps["patched"] == ps["total_clusters"] else "Partial"),
    ]

    header = [Paragraph(t, s["cell_w"]) for t in ["Metric", "Value", "Assessment"]]
    rows = [header] + [
        [Paragraph(m, s["cell_b"]), Paragraph(v, s["cell"]), Paragraph(a, s["cell"])]
        for m, v, a in rows_data
    ]

    risk_colors = {
        "High":    _DANGER, "Medium": _WARN, "Low":   _SUCCESS,
        "Resolved":_SUCCESS, "Partial": _WARN, "Unclustered": _SLATE,
    }
    extra = []
    for i, (_, _, a) in enumerate(rows_data, 1):
        c = risk_colors.get(a)
        if c:
            extra += [("TEXTCOLOR", (2, i), (2, i), c),
                      ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold")]

    story.append(_table(rows, [7*cm, 3.5*cm, CONTENT_W - 10.5*cm], extra))
    return story


# ── Attack Surface ────────────────────────────────────────────────────────────

def _attack_surface(s, profile):
    story = [Spacer(1, 0.4 * cm)]
    story += _section("Attack Surface Profile", s)

    priority = profile.get("attack_priority", [])
    story.append(Paragraph(
        f"Profiled on <b>{profile.get('num_samples','?')}</b> samples.  "
        "Layer attack priority (highest gradient norm first): "
        + ", ".join(f"<b>{l}</b>" for l in priority[:5]),
        s["body"],
    ))

    lollipop = _chart_layer_radar(profile, w_cm=13, h_cm=5.5)
    if lollipop:
        story.append(Spacer(1, 0.2 * cm))
        story.append(lollipop)

    grad = profile.get("gradient_norms", {})
    entr = profile.get("activation_entropy", {})
    if grad:
        story.append(Spacer(1, 0.2 * cm))
        header = [Paragraph(t, s["cell_w"]) for t in
                  ["Layer", "Gradient Norm", "Entropy", "Priority"]]
        rows = [header]
        for rank, layer in enumerate(priority, 1):
            rows.append([
                Paragraph(layer, s["cell_b"]),
                Paragraph(f"{grad.get(layer, 0):.4f}", s["cell"]),
                Paragraph(f"{entr.get(layer, 0):.4f}", s["cell"]),
                Paragraph(f"#{rank}", s["cell"]),
            ])
        story.append(_table(rows, [6*cm, 3.5*cm, 3.5*cm, 2.5*cm]))
    return story


# ── Attack Results ────────────────────────────────────────────────────────────

def _attack_results(s, attack_rates, total_fooled, total_examples):
    story = [Spacer(1, 0.4 * cm)]
    story += _section("Attack Results", s)

    story.append(Paragraph(
        f"<b>{total_fooled}</b> of <b>{total_examples}</b> adversarial inputs "
        f"fooled the model — combined success rate "
        f"<b>{total_fooled / max(total_examples, 1):.1%}</b>.",
        s["body"],
    ))

    chart = _chart_attack_radar(attack_rates)
    story.append(Spacer(1, 0.2 * cm))
    story.append(chart)
    story.append(Spacer(1, 0.2 * cm))

    risk_map = {"High Risk": _DANGER, "Moderate": _WARN, "Low Risk": _SUCCESS}
    n = total_examples // max(len(attack_rates), 1)
    rows_data = []
    for name, rate in attack_rates.items():
        risk = "High Risk" if rate > 0.7 else "Moderate" if rate > 0.3 else "Low Risk"
        rows_data.append((name, f"{rate:.1%}", f"{round(rate*n)}/{n}", risk))

    header = [Paragraph(t, s["cell_w"]) for t in
              ["Attack", "Success Rate", "Fooled / Total", "Risk Level"]]
    rows = [header] + [
        [Paragraph(a, s["cell_b"]), Paragraph(r, s["cell"]),
         Paragraph(f, s["cell"]), Paragraph(k, s["cell"])]
        for a, r, f, k in rows_data
    ]
    extra = []
    for i, (_, _, _, k) in enumerate(rows_data, 1):
        c = risk_map.get(k)
        if c:
            extra += [("TEXTCOLOR", (3, i), (3, i), c),
                      ("FONTNAME",  (3, i), (3, i), "Helvetica-Bold")]
    story.append(_table(rows, [5*cm, 3.5*cm, 3.5*cm, 3.5*cm], extra))
    return story


# ── Vulnerability Clusters ────────────────────────────────────────────────────

def _vulnerability_clusters(s, taxonomy, explanation_report):
    story = [PageBreak()]
    story += _section("Vulnerability Clusters", s)

    ts = taxonomy.summary()
    story.append(Paragraph(
        f"HDBSCAN identified <b>{ts['num_clusters']}</b> distinct failure clusters "
        f"from <b>{ts['total_failures']}</b> adversarial failures "
        f"(<b>{ts['noise_count']}</b> noise points unclustered).",
        s["body"],
    ))

    lollipop = _chart_cluster_radar(taxonomy, w_cm=13, h_cm=5.0)
    if lollipop:
        story.append(Spacer(1, 0.2 * cm))
        story.append(lollipop)
        story.append(Spacer(1, 0.3 * cm))

    exp_map = {e.cluster_id: e for e in explanation_report.explanations}

    accent_cycle = [_NAVY, _NAVY2, _ACCENT, HexColor("#1e40af"), HexColor("#1d4ed8")]
    for cluster in taxonomy.clusters:
        col = accent_cycle[cluster.cluster_id % len(accent_cycle)]
        dist_str = "  |  ".join(f"{k}: {v}" for k, v in cluster.attack_distribution.items())

        ch = Table(
            [[Paragraph(f"Cluster {cluster.cluster_id}  —  {cluster.name}", s["cluster_h"]),
              Paragraph(f"n = {cluster.size}", s["cluster_meta"])]],
            colWidths=[CONTENT_W * 0.75, CONTENT_W * 0.25],
            rowHeights=[0.75 * cm],
        )
        ch.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), col),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))

        body = [
            Spacer(1, 0.1 * cm),
            Paragraph(f"<b>Attack distribution:</b>  {dist_str}", s["small"]),
        ]
        exp = exp_map.get(cluster.cluster_id)
        if exp:
            body += [
                Spacer(1, 0.1 * cm),
                Paragraph("<b>Analysis</b>", s["label"]),
                Paragraph(exp.explanation, s["body"]),
                Paragraph(
                    f"<b>Recommended strategy:</b> {exp.patch_strategy}    "
                    f"<b>Strength:</b> {exp.patch_params.get('strength','—')}    "
                    f"<b>Steps:</b> {exp.patch_params.get('steps','—')}",
                    s["small"],
                ),
            ]
            if exp.sources:
                body.append(Paragraph(
                    "<b>Research basis:</b> " + ",  ".join(exp.sources), s["small"]
                ))

        body_t = Table([[body]], colWidths=[CONTENT_W])
        body_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _LIGHT),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("BOX",           (0, 0), (-1, -1), 0.5, _BORDER),
        ]))
        story.append(KeepTogether([Spacer(1, 0.25 * cm), ch, body_t]))

    return story


# ── Patch Results ─────────────────────────────────────────────────────────────

def _patch_results(s, patch_report):
    story = [PageBreak()]
    story += _section("Patch Results", s)

    ps = patch_report.summary()
    story.append(Paragraph(
        f"<b>{ps['patched']}</b> of <b>{ps['total_clusters']}</b> clusters patched. "
        f"<b>{ps['unresolved']}</b> unresolved after 3 retry attempts "
        "(aggressive → moderate → conservative).",
        s["body"],
    ))
    story.append(Paragraph(
        "Safety gate: score = 0.6 × resistance_gain + 0.4 × accuracy_retention. "
        "Passes if score ≥ 0.70 AND clean accuracy drop ≤ 3%. "
        "Unresolved clusters represent fundamental robustness–accuracy tradeoffs "
        "requiring architectural changes or full retraining.",
        s["small"],
    ))

    dot = _chart_patch_radar(patch_report, w_cm=13, h_cm=5.5)
    if dot:
        story.append(Spacer(1, 0.2 * cm))
        story.append(dot)
        story.append(Spacer(1, 0.2 * cm))

    if not patch_report.results:
        story.append(Paragraph("No clusters to patch.", s["small"]))
        return story

    header = [Paragraph(t, s["cell_w"]) for t in
              ["Cluster", "Strategy", "Score", "Resistance", "Acc. Drop", "Retries", "Status"]]
    rows_data = patch_report.results
    rows = [header] + [
        [Paragraph(f"C{r.cluster_id}: {r.cluster_name[:20]}", s["cell_b"]),
         Paragraph(r.strategy.replace("_", " "), s["cell"]),
         Paragraph(f"{r.safety_score:.3f}", s["cell"]),
         Paragraph(f"{r.resistance_gain:.1%}", s["cell"]),
         Paragraph(f"{r.accuracy_drop:.1%}", s["cell"]),
         Paragraph(str(r.retries), s["cell"]),
         Paragraph("PASS" if r.passed else "FAIL", s["cell"])]
        for r in rows_data
    ]
    extra = [
        cmd
        for i, r in enumerate(rows_data, 1)
        for cmd in [
            ("TEXTCOLOR", (6, i), (6, i), _SUCCESS if r.passed else _DANGER),
            ("FONTNAME",  (6, i), (6, i), "Helvetica-Bold"),
        ]
    ]
    story.append(_table(
        rows, [4*cm, 3.5*cm, 1.8*cm, 2.2*cm, 2.2*cm, 1.5*cm, 1.3*cm], extra
    ))
    return story


# ── Methodology ───────────────────────────────────────────────────────────────

def _methodology(s):
    story = [PageBreak()]
    story += _section("Methodology", s)

    phases = [
        ("Phase 1 — Model Interface",
         "Wraps the target model in a model-agnostic ABC exposing predict(), "
         "get_gradients(), and get_activations(). Supports ResNet-18 and DistilBERT."),
        ("Phase 2 — Attack Surface Profiling",
         "Gradient norms, activation entropy, and input saliency per layer via Captum. "
         "Produces a vulnerability score and ranked attack priority list."),
        ("Phase 3 — Multi-Strategy Attack Engine",
         "FGSM, PGD (Madry et al. 2018), Adversarial Patch (Brown et al. 2017), "
         "Semantic attacks. Collects all successful adversarial examples."),
        ("Phase 4 — Failure Mode Clustering",
         "Activation vectors extracted from successful attacks, reduced with UMAP "
         "(non-linear manifold), clustered with HDBSCAN (auto cluster count, noise labelling)."),
        ("Phase 5 — LLM Explanation Agent",
         "LangGraph agent retrieves paper chunks via FAISS (10 adversarial ML papers) "
         "and uses Gemini 2.5 Flash to write a technical explanation and patch "
         "recommendation per cluster."),
        ("Phase 6 — Autonomous Patching",
         "Executes the recommended patch strategy (adversarial training, stylized "
         "augmentation, counterfactual generation, or targeted augmentation) with a "
         "3-attempt retry loop. Composite safety gate validates each attempt."),
        ("Phase 7 — Report Generation",
         "All phase outputs rendered into this PDF using ReportLab and matplotlib."),
    ]
    header = [Paragraph(t, s["cell_w"]) for t in ["Phase", "Description"]]
    rows = [header] + [
        [Paragraph(t, s["cell_b"]), Paragraph(d, s["cell"])]
        for t, d in phases
    ]
    story.append(_table(rows, [5.5*cm, CONTENT_W - 5.5*cm]))
    return story


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    output_path: str,
    model_name: str,
    profile: dict,
    attack_rates: Dict[str, float],
    total_fooled: int,
    total_examples: int,
    taxonomy: VulnerabilityTaxonomy,
    explanation_report: ExplanationReport,
    patch_report: PatchReport,
) -> None:
    s = _styles()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ps = patch_report.summary()
    ts = taxonomy.summary()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=f"ANVIL Audit — {model_name}",
        author="ANVIL",
    )

    story = []
    story += _cover(s, model_name, date_str,
                    profile.get("vulnerability_score", 0),
                    ts["num_clusters"], ts["total_failures"],
                    ps["patched"], total_examples, total_fooled)
    story += _executive_summary(s, profile,
                                 total_fooled, total_examples, taxonomy, patch_report)
    story += _attack_surface(s, profile)
    story += _attack_results(s, attack_rates, total_fooled, total_examples)
    story += _vulnerability_clusters(s, taxonomy, explanation_report)
    story += _patch_results(s, patch_report)
    story += _methodology(s)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
