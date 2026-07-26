#!/usr/bin/env python3
"""Render the recorded benchmark results as SVG charts.

Standard library only -- no matplotlib. Three reasons that matters here: the repo stays
installable with nothing but a C++ toolchain and Python 3, SVG renders natively in GitHub's
markdown viewer (a PNG would need committing as a binary blob), and the numbers live in
this file as data rather than being baked into an image, so a reader can see exactly what
was plotted and regenerate it.

The data below is transcribed from docs/benchmarks.md. It is deliberately duplicated rather
than parsed out of the markdown: a parser would be more code than the data and would fail
silently if the prose were reworded.

Usage:
    python benchmarks/make_charts.py     # writes docs/charts/*.svg
"""
import pathlib
import sys

# --- recorded results (see docs/benchmarks.md) ------------------------------------

# Phase 3: throughput vs batch size, 32 requests x 32 tokens, uniform length, t=8.
BATCH_THROUGHPUT = [
    (1, 35.36, 32.69, 38.62),
    (4, 55.25, 42.38, 57.90),
    (8, 63.61, 62.52, 64.27),
    (16, 72.84, 72.05, 73.09),
    (32, 56.85, 56.59, 61.23),
]

# Phase 4: continuous vs static under mixed load (80% short, 20% long).
MIXED_LOAD = [
    ("short p50", 21.44, 8.51),
    ("short p95", 30.90, 13.24),
    ("short p99", 30.90, 16.40),
    ("long p50", 21.44, 47.12),
]

# --- tiny SVG helpers -------------------------------------------------------------

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 46, 56

# Colours chosen to stay legible on both light and dark GitHub themes.
INK = "#c9d1d9"
GRID = "#30363d"
ACCENT = "#58a6ff"
WARN = "#f0883e"
GOOD = "#3fb950"


def header(title, subtitle):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">
<style>
  .t {{ fill: {INK}; font-size: 15px; font-weight: 600; }}
  .s {{ fill: #8b949e; font-size: 11px; }}
  .a {{ fill: {INK}; font-size: 11px; }}
  .v {{ fill: {INK}; font-size: 11px; font-weight: 600; }}
</style>
<text x="{PAD_L}" y="20" class="t">{title}</text>
<text x="{PAD_L}" y="36" class="s">{subtitle}</text>
"""


def y_axis(y_max, plot_h, ticks=5):
    out = []
    for i in range(ticks + 1):
        frac = i / ticks
        y = PAD_T + plot_h - frac * plot_h
        val = y_max * frac
        out.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" class="a" text-anchor="end">{val:.0f}</text>'
        )
    return "\n".join(out)


def chart_batch_throughput():
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    y_max = 80.0
    n = len(BATCH_THROUGHPUT)
    step = plot_w / n
    bar_w = step * 0.5

    parts = [
        header(
            "Throughput vs. batch size (static batching)",
            "32 concurrent requests x 32 tokens, uniform length, 8 threads, "
            "median of 5 interleaved rounds. Bars show IQR.",
        ),
        y_axis(y_max, plot_h),
        f'<text x="16" y="{PAD_T + plot_h / 2:.0f}" class="a" '
        f'transform="rotate(-90 16 {PAD_T + plot_h / 2:.0f})" text-anchor="middle">'
        f"output tokens/sec</text>",
    ]

    for i, (batch, med, q1, q3) in enumerate(BATCH_THROUGHPUT):
        cx = PAD_L + step * i + step / 2
        x = cx - bar_w / 2
        h = med / y_max * plot_h
        y = PAD_T + plot_h - h
        peak = med == max(r[1] for r in BATCH_THROUGHPUT)
        colour = GOOD if peak else ACCENT
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'fill="{colour}" opacity="0.85" rx="2"/>'
        )
        # IQR whisker
        y1 = PAD_T + plot_h - q3 / y_max * plot_h
        y2 = PAD_T + plot_h - q1 / y_max * plot_h
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{y2:.1f}" '
            f'stroke="{INK}" stroke-width="1.5" opacity="0.7"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{y - 8:.1f}" class="v" text-anchor="middle">{med:.1f}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{PAD_T + plot_h + 18:.0f}" class="a" '
            f'text-anchor="middle">{batch}</text>'
        )

    parts.append(
        f'<text x="{PAD_L + plot_w / 2:.0f}" y="{H - 20}" class="a" '
        f'text-anchor="middle">batch size</text>'
    )
    parts.append(
        f'<text x="{W - PAD_R}" y="{H - 20}" class="s" text-anchor="end">'
        f"peak 2.06x at 16, then KV pressure regresses it</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def chart_mixed_load():
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    y_max = 50.0
    n = len(MIXED_LOAD)
    step = plot_w / n
    bar_w = step * 0.30

    parts = [
        header(
            "Continuous vs. static batching under mixed-length load",
            "80% short (16-32 tok), 20% long (256-512 tok). Lower is better. "
            "Median of 3 interleaved rounds.",
        ),
        y_axis(y_max, plot_h),
        f'<text x="16" y="{PAD_T + plot_h / 2:.0f}" class="a" '
        f'transform="rotate(-90 16 {PAD_T + plot_h / 2:.0f})" text-anchor="middle">'
        f"latency (seconds)</text>",
        # legend
        f'<rect x="{W - PAD_R - 190}" y="14" width="10" height="10" fill="{WARN}" rx="2"/>'
        f'<text x="{W - PAD_R - 175}" y="23" class="a">static</text>'
        f'<rect x="{W - PAD_R - 118}" y="14" width="10" height="10" fill="{ACCENT}" rx="2"/>'
        f'<text x="{W - PAD_R - 103}" y="23" class="a">continuous</text>',
    ]

    for i, (label, static_v, cont_v) in enumerate(MIXED_LOAD):
        cx = PAD_L + step * i + step / 2
        for j, (val, colour) in enumerate(((static_v, WARN), (cont_v, ACCENT))):
            x = cx - bar_w + j * bar_w
            h = min(val / y_max, 1.0) * plot_h
            y = PAD_T + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{h:.1f}" '
                f'fill="{colour}" opacity="0.85" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + (bar_w - 3) / 2:.1f}" y="{y - 6:.1f}" class="v" '
                f'text-anchor="middle">{val:.1f}</text>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{PAD_T + plot_h + 18:.0f}" class="a" '
            f'text-anchor="middle">{label}</text>'
        )

    parts.append(
        f'<text x="{PAD_L}" y="{H - 20}" class="s">'
        f"Short requests 2.5x faster; long requests pay for it. Fairness, not throughput."
        f"</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "docs" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, svg in (
        ("batch-throughput.svg", chart_batch_throughput()),
        ("continuous-vs-static.svg", chart_mixed_load()),
    ):
        path = out_dir / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)

    for p in written:
        print(f"wrote {p.relative_to(pathlib.Path.cwd()) if p.is_relative_to(pathlib.Path.cwd()) else p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
