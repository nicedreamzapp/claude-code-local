#!/usr/bin/env python3
"""Draw the docs chart: median time per task, claude-code-local vs Ollama,
Qwen 3.5 2B and 4B, with the pass rate on each bar.

Writes assets/vs-ollama-time-light.svg and -dark.svg (docs/images is a link to assets/) (the docs page
shows one or the other with Material's #only-light / #only-dark). Numbers
come from results/raw.jsonl, as-shipped setups only (mlx, ollama).
Colors: blue and orange, slots 1 and 2 of the reference categorical palette,
checked for color-vision separation and 3:1 contrast on both site surfaces.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets")  # docs/images links here

THEMES = {
    "light": {"ccl": "#2a78d6", "ollama": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e3df", "axis": "#8a8984", "surface": "#ffffff"},
    "dark": {"ccl": "#3987e5", "ollama": "#d95926", "ink": "#f0efec", "ink2": "#c3c2b7",
             "grid": "#34363d", "axis": "#6b6d74", "surface": "#1e2129"},
}
MODELS = [("qwen35-2b", "Qwen 3.5 2B, 8-bit"), ("qwen35-4b", "Qwen 3.5 4B, bf16")]
SETUPS = [("mlx", "Claude Code Local", "ccl"), ("ollama", "Ollama", "ollama")]


def load():
    runs = [json.loads(line) for line in open(os.path.join(HERE, "results", "raw.jsonl"))]
    data = {}
    for m, _ in MODELS:
        for s, _, _ in SETUPS:
            rs = [r for r in runs if r["model"] == m and r["setup"] == s]
            data[(m, s)] = (statistics.median(r["wall_s"] for r in rs), sum(r["passed"] for r in rs), len(rs))
    return data


def svg(theme, data):
    c = THEMES[theme]
    # Value labels get a halo in the site's page color so a gridline never runs
    # through them. Narrow canvas with model names above their bars, so the chart stays
    # readable when a phone shrinks it to the content width.
    W, left, right = 560, 2, 150
    plot_w = W - left - right
    xmax = 30.0
    bar_h, gap = 26, 2
    sx = lambda v: left + v / xmax * plot_w  # noqa: E731
    font = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif\""
    o = []
    y = 104
    body = []
    for m, mlabel in MODELS:
        body.append(f'<text x="{left}" y="{y}" font-size="15" font-weight="600" fill="{c["ink"]}">{mlabel}</text>')
        y += 10
        for s, slabel, key in SETUPS:
            med, p, n = data[(m, s)]
            x0, x1, r = sx(0), sx(med), 4
            body.append(f'<path d="M{x0:.1f},{y} H{x1 - r:.1f} Q{x1:.1f},{y} {x1:.1f},{y + r} V{y + bar_h - r} '
                        f'Q{x1:.1f},{y + bar_h} {x1 - r:.1f},{y + bar_h} H{x0:.1f} Z" fill="{c[key]}">'
                        f'<title>{slabel}: {med:.1f} s median, {p}/{n} passed</title></path>')
            body.append(f'<text x="{x1 + 8:.1f}" y="{y + 18}" font-size="14" fill="{c["ink"]}" '
                        f'stroke="{c["surface"]}" stroke-width="5" stroke-linejoin="round" paint-order="stroke">'
                        f'<tspan font-weight="600">{med:.1f} s</tspan>'
                        f'<tspan fill="{c["ink2"]}" dx="8">{p}/{n} passed</tspan></text>')
            y += bar_h + gap
        y += 34
    base = y - 34 + 6
    H = base + 26
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'role="img" aria-labelledby="t d" {font}>')
    o.append('<title id="t">Median time per task: Claude Code Local vs Ollama</title>')
    o.append('<desc id="d">' + "; ".join(
        f"{ml}, {sl}: {data[(m, s)][0]:.1f} seconds, {data[(m, s)][1]} of {data[(m, s)][2]} passed"
        for m, ml in MODELS for s, sl, _ in SETUPS) + '</desc>')
    o.append(f'<text x="0" y="22" font-size="18" font-weight="600" fill="{c["ink"]}">'
             'Median time per task, lower is faster</text>')
    o.append(f'<text x="0" y="44" font-size="13.5" fill="{c["ink2"]}">10 coding tasks, 3 runs each, '
             'Mac mini M4 Pro. Label: tasks passed.</text>')
    lx = 0
    for _, slabel, key in SETUPS:
        o.append(f'<rect x="{lx}" y="60" width="13" height="13" rx="2" fill="{c[key]}"/>')
        o.append(f'<text x="{lx + 19}" y="71.5" font-size="14" fill="{c["ink"]}">{slabel}</text>')
        lx += 19 + len(slabel) * 8 + 24
    for v in range(0, 31, 10):
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="86" x2="{x:.1f}" y2="{base}" '
                 f'stroke="{c["grid"] if v else c["axis"]}" stroke-width="1"/>')
        anchor = "start" if v == 0 else "middle"
        o.append(f'<text x="{x:.1f}" y="{base + 18}" font-size="13" text-anchor="{anchor}" '
                 f'fill="{c["ink2"]}">{v} s</text>')
    o += body
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main():
    data = load()
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"vs-ollama-time-{theme}.svg"), "w") as f:
            f.write(svg(theme, data))
    for k, v in sorted(data.items()):
        print(k, v)


if __name__ == "__main__":
    main()
