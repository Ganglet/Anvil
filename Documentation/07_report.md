# Phase 7 — PDF Audit Report Generation

## What this phase does

Phase 6 produced a `PatchReport`. Phase 7 takes every output from all prior phases and renders them into a professional multi-page PDF audit report.

The report is the primary deliverable — the artifact a recruiter, client, or researcher actually sees. It must be self-contained: someone who has never seen the terminal output should be able to read the PDF and fully understand what the pipeline found, why it matters, and what was done about it.

---

## Pipeline

```
profile        (Phase 2)  ─┐
attack_rates   (Phase 3)  ─┤
taxonomy       (Phase 4)  ─┤──► generate_report() ──► audit_report.pdf
explanation_report (Ph 5) ─┤
patch_report   (Phase 6)  ─┘
```

---

## Output

A single PDF file at the path specified by `--output` (default: `./audit_report.pdf`).

### Pages

| Page | Section | Content |
|------|---------|---------|
| 1 | Cover | ANVIL title, 3 stat boxes (vulnerability score, clusters found, clusters patched), metadata grid (model, date, inputs attacked, inputs fooled) |
| 2 | Executive Summary | 6-row table: vulnerability score, saliency, attack success rate, clusters, noise points, patch resolution — each with risk assessment colour-coded High/Medium/Low |
| 2 | Attack Surface Profile | Layer priority, layer vulnerability radar chart (normalised gradient norms), layer detail table (gradient norm, entropy, rank) |
| 2 | Attack Results | Combined success rate callout, attack success rates radar chart, per-attack table with risk level |
| 3 | Vulnerability Clusters | Cluster size radar chart, one card per cluster (coloured header, attack distribution, LLM explanation, patch strategy, research sources) |
| 4 | Patch Results | Patch quality profile multi-series radar (4 axes per cluster), full results table (score, resistance gain, accuracy drop, retries, PASS/FAIL) |
| 5 | Methodology | 7-row table summarising each phase — what it does and which tools it uses |

---

## Key files

| File | Role |
|------|------|
| `reporter/report.py` | All logic — chart builders, section builders, `generate_report()` public API |
| `reporter/__init__.py` | Re-exports `generate_report` |
| `tests/test_reporter.py` | 7 integration tests |

---

## Charts

All charts use matplotlib with `Agg` backend (no display required), saved to `io.BytesIO`, embedded via `reportlab.platypus.Image`.

Every chart is a **radar/spider chart** — `subplot_kw=dict(polar=True)`, `np.linspace(0, 2π, N, endpoint=False)`, polygon closed by repeating the first point. When a section has fewer than 3 axes (e.g. single-cluster audit), a horizontal bar fallback is used automatically.

| Chart | Function | Axes | Series |
|-------|----------|------|--------|
| Layer Vulnerability | `_chart_layer_radar` | One per layer in attack priority | Normalised gradient norm |
| Attack Success Rates | `_chart_attack_radar` | One per attack type | Success rate (0–1) |
| Cluster Sizes | `_chart_cluster_radar` | One per cluster | Normalised failure count |
| Patch Quality | `_chart_patch_radar` | Safety Score, Resistance Gain, Accuracy Retention, Effort (1−retries/3) | One polygon per cluster |

---

## Colour palette

Single navy/blue family throughout — no rainbow gradients.

| Token | Hex | Used for |
|-------|-----|---------|
| `_NAVY` | `#0f1e2e` | Cover header, table headers |
| `_NAVY2` | `#1e3a5f` | Section accents, secondary series |
| `_ACCENT` | `#2563eb` | Primary blue — charts, HR lines, primary series |
| `_SUCCESS` | `#059669` | PASS status, Low risk |
| `_DANGER` | `#dc2626` | FAIL status, High risk |
| `_WARN` | `#d97706` | Medium risk |
| `_SLATE` | `#64748b` | Secondary text, footer |

---

## Design decisions

### ReportLab table style — `_table()` helper
ReportLab's `Table` object does not expose its style commands list publicly. Early implementation tried to extend `t._cmds` after construction — this raised `AttributeError`. Fixed by a `_table(rows, widths, extra=None)` helper that builds all style commands before constructing the table, with `extra` for per-call colour overrides.

### Stat box overlap — list of flowables per cell
Early version used a nested `Table` inside each stat box cell. With no explicit `rowHeights`, ReportLab compressed rows and the number overlapped the label. Fixed by replacing nested tables with `[Paragraph(num), Spacer, Paragraph(label)]` lists as cell content, outer table `rowHeights=[2.4*cm]`.

### Cover header overlap — single table with explicit row heights
Two separate `Table` elements with dark backgrounds placed sequentially produced title/subtitle overlap. Fixed by merging into a single `Table` with two rows and explicit `rowHeights=[3.6*cm, 1.2*cm]`.

### Radar over other chart types
First iteration used a double-ring donut (attack distribution) and horizontal lollipop charts (layer norms, cluster sizes, patch dot plot). Lollipop charts had persistent dot/label overlap at certain data scales. Dot plot placed score labels at `score + 0.025` — when `score = 0.0` this still overlapped the dot visually. All replaced with radar charts: overlap is impossible (labels are on fixed outer ring, values are interior polygon vertices), and a single multi-series radar (patch quality) communicates 4 dimensions per cluster in one glance.

---

## Tests

`tests/test_reporter.py` — 7 tests, all passing.

| Test | What it checks |
|------|---------------|
| `test_report_creates_file` | PDF file exists after `generate_report()` |
| `test_report_file_not_empty` | File size > 5 KB (not a zero-byte stub) |
| `test_report_uses_output_path` | File appears at the specified path, not a default |
| `test_report_no_clusters` | Empty taxonomy + empty reports produce a valid PDF |
| `test_report_with_passed_patches` | All-pass patch report renders without error |
| `test_report_single_cluster` | Single-cluster path (triggers bar fallback on radar) |
| `test_report_low_vulnerability_score` | Low-score cover stat box uses correct colour |

All tests use `tempfile.TemporaryDirectory()` — no files left on disk after the suite runs.

---

## `audit.py` integration

```python
# Phase 7
generate_report(
    output_path=args.output,
    model_name=model.model_name,
    profile=profile,
    attack_rates=rates,
    total_fooled=total_success,
    total_examples=len(all_examples),
    taxonomy=taxonomy,
    explanation_report=report,
    patch_report=patch_report,
)
print(f"      Report written to: {args.output}")
```

The `--output` flag (default `./audit_report.pdf`) passes through directly. No intermediate files are created.
