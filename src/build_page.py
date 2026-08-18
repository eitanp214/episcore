"""Generate the self-contained findings page.

Charts are hand-authored SVG against the measured numbers; the calibration
pairs are embedded as base64 so the page has no external dependencies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR  # noqa: E402

ASSETS = json.loads((OUT_DIR / "page_assets.json").read_text(encoding="utf-8"))
CONF = json.loads((OUT_DIR / "confidence_factory051_v2.json").read_text(encoding="utf-8"))
NOVELTY = json.loads((OUT_DIR / "novelty_factory051_v2.json").read_text(encoding="utf-8"))

CI_LO, CI_HI = (v * 100 for v in CONF["ci95_jackknife"])
WORKERS = sorted((k, v * 100) for k, v in CONF["per_worker"].items())
SPREAD = CONF["operator_spread"]

CURVE = [
    (0.80, 97.2), (0.85, 84.3), (0.88, 65.5), (0.90, 49.0),
    (0.92, 27.2), (0.94, 8.3), (0.96, 1.0), (0.98, 0.0),
]
SENSITIVITY = [(0, 7993, 23.6), (5, 7593, 24.7), (10, 7193, 25.7),
               (20, 6394, 27.2), (30, 5595, 27.3), (40, 4796, 27.4)]

# Per-worker recall from the workstation-identification test.
RECALL = [("w001", 92.7, 286), ("w002", 95.9, 462), ("w003", 86.1, 418),
          ("w004", 93.5, 217), ("w005", 99.7, 352), ("w006", 89.0, 418),
          ("w007", 95.5, 264)]
CHANCE = 14.3

PAD_L, PAD_R, PAD_T, PAD_B = 54, 22, 18, 38
W, H = 720, 300
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B


def cx(t: float) -> float:
    return PAD_L + (t - 0.80) / 0.18 * PLOT_W


def cy(v: float) -> float:
    return PAD_T + (1 - v / 100) * PLOT_H


def curve_svg() -> str:
    pts = " ".join(f"{cx(t):.1f},{cy(v):.1f}" for t, v in CURVE)
    area = f"{PAD_L},{PAD_T + PLOT_H} {pts} {PAD_L + PLOT_W},{PAD_T + PLOT_H}"
    grid = "".join(
        f'<line x1="{PAD_L}" y1="{cy(v):.1f}" x2="{PAD_L + PLOT_W}" y2="{cy(v):.1f}" '
        f'class="grid"/><text x="{PAD_L - 10}" y="{cy(v) + 4:.1f}" '
        f'class="tick" text-anchor="end">{v}</text>'
        for v in (0, 25, 50, 75, 100)
    )
    xlab = "".join(
        f'<text x="{cx(t):.1f}" y="{PAD_T + PLOT_H + 22}" class="tick" '
        f'text-anchor="middle">{t:.2f}</text>'
        for t, _ in CURVE
    )
    dots = "".join(
        f'<circle cx="{cx(t):.1f}" cy="{cy(v):.1f}" r="3.5" class="dot"/>'
        for t, v in CURVE
    )
    hx, hy = cx(0.92), cy(27.2)
    return f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="Redundancy falls from 97 percent at similarity 0.80 to near zero at 0.98, passing 27 percent at 0.92">
  {grid}
  <polygon points="{area}" class="area"/>
  <polyline points="{pts}" class="line"/>
  {dots}
  <line x1="{hx:.1f}" y1="{PAD_T}" x2="{hx:.1f}" y2="{PAD_T + PLOT_H}" class="marker"/>
  <circle cx="{hx:.1f}" cy="{hy:.1f}" r="6" class="dot-hi"/>
  <text x="{hx - 12:.1f}" y="{hy - 14:.1f}" class="callout" text-anchor="end">27.2% @ 0.92</text>
  {xlab}
  <text x="{PAD_L}" y="{H - 6}" class="axis">cosine similarity threshold</text>
</svg>"""


def sensitivity_svg() -> str:
    rows, y, row_h = [], 16, 34
    scale = 560 / 30.0
    for pct, frames, val in SENSITIVITY:
        bar = val * scale
        label = "none" if pct == 0 else f"−{pct}%"
        rows.append(
            f'<text x="0" y="{y + 15}" class="rowlab">{label}</text>'
            f'<rect x="62" y="{y + 3}" width="{bar:.1f}" height="17" rx="2" class="bar"/>'
            f'<text x="{62 + bar + 8:.1f}" y="{y + 16}" class="barval">{val}%</text>'
            f'<text x="700" y="{y + 16}" class="rowmeta" text-anchor="end">{frames:,}</text>'
        )
        y += row_h
    return f"""<svg viewBox="0 0 720 {y + 26}" role="img" aria-label="Redundancy stays between 23.6 and 27.4 percent across blur filter strengths from none to 40 percent">
  {''.join(rows)}
  <text x="0" y="{y + 16}" class="axis">blur gate</text>
  <text x="700" y="{y + 16}" class="axis" text-anchor="end">frames measured</text>
</svg>"""


def distribution_svg() -> str:
    lo, hi = 0.55, 0.96
    left, width = 96, 560

    def px(v: float) -> float:
        return left + (v - lo) / (hi - lo) * width

    rows = [
        ("same worker", 0.767, 0.907, 0.935, "acc"),
        ("different factory", 0.612, 0.751, 0.798, "sig"),
    ]
    out, y = [], 22
    for name, mean, p95, p99, cls in rows:
        out.append(
            f'<text x="0" y="{y + 5}" class="rowlab">{name}</text>'
            f'<line x1="{px(mean):.1f}" y1="{y}" x2="{px(p99):.1f}" y2="{y}" class="range {cls}"/>'
            f'<circle cx="{px(mean):.1f}" cy="{y}" r="5" class="pt {cls}"/>'
            f'<circle cx="{px(p95):.1f}" cy="{y}" r="3" class="pt {cls}"/>'
            f'<circle cx="{px(p99):.1f}" cy="{y}" r="3" class="pt {cls}"/>'
            f'<text x="{px(mean):.1f}" y="{y - 12}" class="tick" text-anchor="middle">{mean:.2f}</text>'
            f'<text x="{px(p99):.1f}" y="{y - 12}" class="tick" text-anchor="middle">p99 {p99:.2f}</text>'
        )
        y += 62
    thr = px(0.92)
    return f"""<svg viewBox="0 0 720 {y + 24}" role="img" aria-label="Same-worker similarity averages 0.77 and different-factory 0.61; the 0.92 threshold sits above both distributions">
  {''.join(out)}
  <line x1="{thr:.1f}" y1="4" x2="{thr:.1f}" y2="{y - 26}" class="marker"/>
  <text x="{thr:.1f}" y="{y + 2}" class="callout" text-anchor="middle">threshold 0.92</text>
</svg>"""


def recall_svg() -> str:
    """Per-workstation recall bars against the 14.3% chance line."""
    left, width, row_h = 52, 590, 30
    rows, y = [], 14
    for name, rec, n in RECALL:
        bar = rec / 100 * width
        rows.append(
            f'<text x="0" y="{y + 14}" class="rowlab">{name}</text>'
            f'<rect x="{left}" y="{y + 3}" width="{bar:.1f}" height="16" rx="2" class="bar"/>'
            f'<text x="{left + bar + 8:.1f}" y="{y + 16}" class="barval">{rec}%</text>'
            f'<text x="710" y="{y + 16}" class="rowmeta" text-anchor="end">n={n}</text>'
        )
        y += row_h
    cx_chance = left + CHANCE / 100 * width
    return f"""<svg viewBox="0 0 720 {y + 30}" role="img" aria-label="Each of seven workstations is identified with 86 to 99.7 percent recall, far above the 14.3 percent chance line">
  {''.join(rows)}
  <line x1="{cx_chance:.1f}" y1="8" x2="{cx_chance:.1f}" y2="{y - 6}" class="marker"/>
  <text x="{cx_chance + 6:.1f}" y="{y + 14}" class="callout">chance {CHANCE}%</text>
</svg>"""


def operator_svg() -> str:
    """Per-operator redundancy — the spread is the finding, not the mean."""
    left, width, row_h = 54, 560, 32
    top = max(v for _, v in WORKERS)
    rows, y = [], 14
    for name, val in WORKERS:
        bar = val / top * width
        hi = val == top
        lo = val == min(v for _, v in WORKERS)
        cls = "bar-hi" if hi else ("bar-lo" if lo else "bar")
        rows.append(
            f'<text x="0" y="{y + 14}" class="rowlab">w{name}</text>'
            f'<rect x="{left}" y="{y + 2}" width="{bar:.1f}" height="17" rx="2" class="{cls}"/>'
            f'<text x="{left + bar + 8:.1f}" y="{y + 16}" class="barval">{val:.1f}%</text>'
        )
        y += row_h
    mean_x = left + 27.2 / top * width
    return f"""<svg viewBox="0 0 720 {y + 30}" role="img" aria-label="Per-operator redundancy ranges from 7.4 percent to 46 percent around a 27.2 percent mean">
  {''.join(rows)}
  <line x1="{mean_x:.1f}" y1="6" x2="{mean_x:.1f}" y2="{y - 6}" class="marker"/>
  <text x="{mean_x + 6:.1f}" y="{y + 14}" class="callout">corpus mean 27.2%</text>
</svg>"""


def novelty_svg() -> str:
    """Marginal novelty per clip — it decays, then holds."""
    curve = NOVELTY["curve"]
    pw, ph = 640, 190
    ox, oy = 56, 16
    n = len(curve)

    def nx(i: int) -> float:
        return ox + i / max(n - 1, 1) * pw

    def ny(v: float) -> float:
        return oy + (1 - v) * ph

    pts = " ".join(f"{nx(i):.1f},{ny(r['novel_share']):.1f}"
                   for i, r in enumerate(curve))
    grid = "".join(
        f'<line x1="{ox}" y1="{ny(v):.1f}" x2="{ox + pw}" y2="{ny(v):.1f}" class="grid"/>'
        f'<text x="{ox - 10}" y="{ny(v) + 4:.1f}" class="tick" text-anchor="end">{int(v * 100)}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    xlab = "".join(
        f'<text x="{nx(i):.1f}" y="{oy + ph + 20}" class="tick" text-anchor="middle">{i + 1}</text>'
        for i in range(0, n, 10)
    )
    tail = sum(r["novel_share"] for r in curve[-10:]) / 10
    ty = ny(tail)
    return f"""<svg viewBox="0 0 720 {oy + ph + 52}" role="img" aria-label="Novelty per clip falls from 100 percent to roughly 75 percent and then stays flat through clip 52">
  {grid}
  <line x1="{ox}" y1="{ty:.1f}" x2="{ox + pw}" y2="{ty:.1f}" class="marker"/>
  <polyline points="{pts}" class="line"/>
  {xlab}
  <text x="{ox + pw}" y="{ty - 8:.1f}" class="callout" text-anchor="end">tail average {tail * 100:.0f}%</text>
  <text x="{ox}" y="{oy + ph + 44}" class="axis">clip number, per operator</text>
</svg>"""


def img(key: str, alt: str) -> str:
    return (f'<img src="data:image/jpeg;base64,{ASSETS[key]}" alt="{alt}" '
            f'loading="lazy"/>')


FAQ = [
    ("How do you measure redundancy in a video training corpus?",
     "Sample keyframes, embed each one, and for every frame record its highest "
     "cosine similarity to any earlier frame in the corpus. The share of frames "
     "above a chosen threshold is the redundancy figure. Computed in one pass "
     "against all earlier frames, the whole threshold curve derives from a "
     "single array, so the result does not depend on a cutoff picked in advance."),
    ("How much of a robotics training dataset is duplicate content?",
     "On 10,000 hours of factory-floor egocentric video, roughly a quarter: "
     f"27.2% with a 95% interval of [{CI_LO:.1f}%, {CI_HI:.1f}%]. The corpus "
     "figure is the wrong unit — individual operators ranged from 7.4% to 46.0%."),
    ("Should I filter blurry frames before measuring similarity?",
     "Yes. Motion blur destroys the detail that distinguishes frames, so blurred "
     "frames match each other and register as duplicates of unrelated content. A "
     "blur-only pair scored 0.906 while a genuine same-workstation pair scored "
     "0.872. Score sharpness with variance of the Laplacian and drop the "
     "blurriest fraction first."),
    ("Can I use bootstrap resampling for a confidence interval on redundancy?",
     "No. Resampling with replacement puts the same clip in the corpus twice, and "
     "this statistic measures duplication, so every repeated clip scores as a "
     "perfect match against its own copy. It returned a 95% interval of "
     "[49.2%, 53.4%] around a 27.2% estimate. Use a jackknife instead."),
    ("Does buying more hours from the same operator stop paying?",
     "Not within the range measured. Marginal novelty falls from 100% to roughly "
     "75% within ten clips and then holds flat through clip 52. Redundancy is a "
     "steady tax of about a quarter, not a saturation cliff."),
    ("How do I know an embedding is valid on my own footage?",
     "Check saturation and separability. Saturation: what share of random frame "
     "pairs already clears the duplicate threshold? Real footage measured 0.03% "
     "at a 0.647 median; a corpus that broke the metric measured 26.2% at 0.904. "
     "Separability: can a linear model name the operator from one frame? Below "
     "about 2x chance the embedding is not resolving the footage."),
]

SCHEMA = json.dumps({
    "@context": "https://schema.org",
    "@graph": [
        {
            "@type": "TechArticle",
            "headline": "What fraction of a video corpus is content it already contains?",
            "about": ["corpus redundancy", "operator spread", "marginal novelty",
                      "robot training data", "egocentric video", "dataset curation"],
            "abstract": (
                "Measured corpus redundancy on 10,000 hours of open egocentric "
                "factory video: 27.2% of sampled frames duplicate earlier content, "
                "with a 6.2x spread between individual operators. Includes the "
                "motion-blur confound that inverts similarity ranking, and two "
                "validation gates for applying the method to other corpora."
            ),
            "keywords": ("corpus redundancy, operator spread, marginal novelty, "
                         "dataset deduplication, embodied AI training data"),
        },
        {
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer", "text": a}}
                for q, a in FAQ
            ],
        },
    ],
}, indent=None)

HTML = f"""<title>What fraction of a video corpus is content it already contains?</title>
<script type="application/ld+json">{SCHEMA}</script>
<style>
:root {{
  --ground:      #F1F4EF;
  --surface:     #FFFFFF;
  --ink:         #191E18;
  --ink-soft:    #545C51;
  --ink-faint:   #7C8478;
  --rule:        #CCD5C8;
  --rule-soft:   #E1E7DE;
  --accent:      #3D6B4E;
  --accent-soft: #E3EDE4;
  --signal:      #2C5A87;
  --warn:        #8F5423;
  --warn-soft:   #F6EADC;
  --measure: 66ch;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:      #101310;
    --surface:     #171B16;
    --ink:         #DEE4DA;
    --ink-soft:    #9BA595;
    --ink-faint:   #6E776A;
    --rule:        #2B322A;
    --rule-soft:   #21271F;
    --accent:      #74A886;
    --accent-soft: #1B2A20;
    --signal:      #6FA0CB;
    --warn:        #C68A54;
    --warn-soft:   #2A2018;
  }}
}}
:root[data-theme="dark"] {{
  --ground:      #101310;
  --surface:     #171B16;
  --ink:         #DEE4DA;
  --ink-soft:    #9BA595;
  --ink-faint:   #6E776A;
  --rule:        #2B322A;
  --rule-soft:   #21271F;
  --accent:      #74A886;
  --accent-soft: #1B2A20;
  --signal:      #6FA0CB;
  --warn:        #C68A54;
  --warn-soft:   #2A2018;
}}

* {{ box-sizing: border-box; }}

body {{
  background: var(--ground);
  color: var(--ink);
  font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.65;
  margin: 0;
  padding: 0 24px 96px;
  -webkit-font-smoothing: antialiased;
}}

.wrap {{ max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 34px; }}

.mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace; }}

header {{ padding-top: 72px; display: flex; flex-direction: column; gap: 20px; }}

.eyebrow {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 12px; letter-spacing: .13em; text-transform: uppercase;
  color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 14px;
}}
.eyebrow span::before {{ content: "· "; color: var(--rule); }}
.eyebrow span:first-child::before {{ content: ""; }}

h1 {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: clamp(27px, 4.4vw, 40px); line-height: 1.16; font-weight: 600;
  letter-spacing: -.022em; margin: 0; text-wrap: balance;
}}

.standfirst {{ font-size: 19px; color: var(--ink-soft); margin: 0; max-width: var(--measure); }}

.status {{
  border: 1px solid var(--warn); background: var(--warn-soft);
  padding: 14px 18px; display: flex; gap: 13px; align-items: baseline;
  font-size: 15px; line-height: 1.55;
}}
.status.ok {{ border-color: var(--accent); background: var(--accent-soft); }}
.status.ok b {{ color: var(--accent); }}
.status b {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--warn); white-space: nowrap;
}}

h2 {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 15px; letter-spacing: .08em; text-transform: uppercase;
  font-weight: 600; color: var(--accent);
  margin: 22px 0 0; padding-top: 22px; border-top: 1px solid var(--rule);
  display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
}}
h2 .n {{ color: var(--ink-faint); font-weight: 400; font-size: 12px; letter-spacing: .1em; }}

h3 {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 14px; letter-spacing: .02em; font-weight: 600; margin: 0;
}}

p {{ margin: 0; max-width: var(--measure); }}
section {{ display: flex; flex-direction: column; gap: 17px; }}

.lede {{ font-size: 18.5px; }}
.aside {{
  font-size: 15px; color: var(--ink-soft); border-left: 2px solid var(--rule);
  padding-left: 16px; line-height: 1.6;
}}

.defs {{
  margin: 0; display: grid; grid-template-columns: auto 1fr;
  gap: 10px 20px; align-items: baseline; max-width: var(--measure);
  border-top: 1px solid var(--rule-soft); border-bottom: 1px solid var(--rule-soft);
  padding: 16px 0;
}}
.defs dt {{
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px; color: var(--accent); white-space: nowrap;
}}
.defs dd {{ margin: 0; font-size: 15.5px; color: var(--ink-soft); }}
@media (max-width: 560px) {{
  .defs {{ grid-template-columns: 1fr; gap: 4px 0; }}
  .defs dd {{ margin-bottom: 10px; }}
}}

.qa {{ display: flex; flex-direction: column; gap: 8px; }}
.qa h3 {{ color: var(--ink); font-size: 15px; line-height: 1.45; }}
.qa p {{ font-size: 16px; color: var(--ink-soft); }}

.figure {{
  background: var(--surface); border: 1px solid var(--rule-soft);
  padding: 22px; display: flex; flex-direction: column; gap: 14px; overflow-x: auto;
}}
.figure svg {{ width: 100%; min-width: 520px; height: auto; display: block; }}
.figcap {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-size: 12px; color: var(--ink-faint); line-height: 1.55; max-width: none;
}}

.grid {{ stroke: var(--rule-soft); stroke-width: 1; }}
.tick {{ font-family: ui-monospace, monospace; font-size: 10.5px; fill: var(--ink-faint); }}
.axis {{ font-family: ui-monospace, monospace; font-size: 10.5px; fill: var(--ink-faint);
         letter-spacing: .07em; text-transform: uppercase; }}
.rowlab {{ font-family: ui-monospace, monospace; font-size: 12px; fill: var(--ink); }}
.rowmeta {{ font-family: ui-monospace, monospace; font-size: 11px; fill: var(--ink-faint); }}
.barval {{ font-family: ui-monospace, monospace; font-size: 12px; fill: var(--ink-soft); }}
.callout {{ font-family: ui-monospace, monospace; font-size: 11.5px; fill: var(--accent); font-weight: 600; }}
.area {{ fill: var(--accent-soft); }}
.line {{ fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; }}
.dot {{ fill: var(--surface); stroke: var(--accent); stroke-width: 1.6; }}
.dot-hi {{ fill: var(--accent); }}
.bar {{ fill: var(--accent); opacity: .82; }}
.bar-hi {{ fill: var(--warn); }}
.bar-lo {{ fill: var(--signal); opacity: .8; }}
.marker {{ stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 3; opacity: .65; }}
.range {{ stroke-width: 2.5; stroke-linecap: round; }}
.range.acc {{ stroke: var(--accent); }}
.range.sig {{ stroke: var(--signal); }}
.pt.acc {{ fill: var(--accent); }}
.pt.sig {{ fill: var(--signal); }}

.pair {{ display: flex; flex-direction: column; gap: 9px; }}
.pair img {{ width: 100%; height: auto; display: block; border: 1px solid var(--rule); }}
.pair .tag {{
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px; display: flex; justify-content: space-between; gap: 14px;
  color: var(--ink-soft); flex-wrap: wrap;
}}
.pair .tag b {{ color: var(--ink); font-weight: 600; }}
.verdict-real {{ color: var(--accent); font-weight: 600; }}
.verdict-false {{ color: var(--warn); font-weight: 600; }}

table {{
  border-collapse: collapse; width: 100%; font-size: 14px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}}
th, td {{ text-align: right; padding: 9px 12px; border-bottom: 1px solid var(--rule-soft); }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{
  color: var(--ink-faint); font-weight: 500; font-size: 11px;
  letter-spacing: .08em; text-transform: uppercase; border-bottom: 1px solid var(--rule);
}}

ul {{ margin: 0; padding-left: 20px; max-width: var(--measure); display: flex;
     flex-direction: column; gap: 11px; }}
li::marker {{ color: var(--accent); }}

pre {{
  background: var(--surface); border: 1px solid var(--rule-soft);
  padding: 16px 18px; overflow-x: auto; font-size: 13.5px; line-height: 1.7;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color: var(--ink-soft); margin: 0;
}}

a {{ color: var(--accent); text-underline-offset: 2px; }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}

footer {{
  border-top: 1px solid var(--rule); padding-top: 22px; margin-top: 14px;
  font-size: 14px; color: var(--ink-faint);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">
    <span>Egocentric-10K</span><span>Apache-2.0</span>
    <span>7,993 frames</span><span>8 workers</span><span>factory 051</span>
  </div>
  <h1>What fraction of a video corpus is content it already contains?</h1>
  <p class="standfirst">Everyone agrees curation beats scale for robot training data.
  Nobody publishes the number that would let a buyer act on it. Here is one — and the
  finding that the number itself is the wrong unit.</p>
  <div class="status ok">
    <b>Method validated</b>
    <span>The redundancy figure is measured on one factory, but the embedding it
    relies on is now confirmed across six physically distinct factories: zero of
    20,000 cross-factory pairs register as duplicates. What remains untested is
    whether the <i>figure</i> generalises, not whether the method is sound.</span>
  </div>
</header>

<section>
  <h2>The measurement <span class="n">01</span></h2>
  <p class="lede">A frame is <b>redundant at threshold t</b> if its cosine similarity
  to any <i>earlier</i> frame in the corpus is at least t.</p>
  <p>That definition is computed threshold-independently, which matters more than it
  sounds. One pass records each frame's best match against everything preceding it,
  and the entire curve derives from that single array. The obvious alternative —
  greedily keeping or dropping frames — has to be re-run per threshold, and its
  answer depends on the order in which drops cascade. That is a hidden knob, and a
  number resting on a hidden knob does not survive a skeptic.</p>
  <div class="figure">
    {curve_svg()}
    <p class="figcap">Share of sampled keyframes whose content already appeared
    earlier in the corpus, after dropping the blurriest 20%. Global scope, 6,394
    frames.</p>
  </div>
  <p>At a conservative threshold, <b>27.2% of frames duplicate earlier content</b>
  — 95% CI [{CI_LO:.1f}%, {CI_HI:.1f}%]. Redundancy is almost entirely
  <i>self</i>-redundancy: within-worker measures 27.2% and the global figure is also
  27.2%. Workers repeat themselves. They do not duplicate each other.</p>
  <dl class="defs">
    <dt>corpus redundancy</dt>
    <dd>Share of frames whose cosine similarity to any <i>earlier</i> frame in
    the same corpus meets a stated threshold.</dd>
    <dt>operator spread</dt>
    <dd>Ratio between the highest and lowest per-operator redundancy in a
    corpus. Measured here at 6.2×.</dd>
    <dt>marginal novelty</dt>
    <dd>Share of a clip's frames that are novel against everything that
    operator has already delivered.</dd>
  </dl>
  <p class="aside"><b>On that interval.</b> The obvious tool — bootstrap over clips —
  is invalid for this statistic. Resampling with replacement puts the same clip in
  the corpus twice, and this metric measures duplication, so every repeated clip
  scores as a perfect match against its own copy. The first attempt returned a 95%
  interval of [49.2%, 53.4%] around a 27.2% estimate: an interval that cannot contain
  its own point, because the method manufactured what it was measuring. The figure
  above is a jackknife, which never duplicates anything.</p>
</section>

<section>
  <h2>The average is the least useful number here <span class="n">02</span></h2>
  <p class="lede">Redundancy ranges from <b>7.4% to 46.0%</b> depending on which
  operator's footage you bought — a {SPREAD['max'] / SPREAD['min']:.1f}× spread around
  the corpus mean.</p>
  <div class="figure">
    {operator_svg()}
    <p class="figcap">Per-operator redundancy, same threshold and blur gate
    throughout. Standard deviation {SPREAD['sd'] * 100:.1f} points.</p>
  </div>
  <p>That spread is roughly {(SPREAD['max'] - SPREAD['min']) * 100:.0f} points wide,
  an order of magnitude larger than the ±{(CI_HI - CI_LO) / 2:.1f}-point sampling
  error on the mean. Which means the corpus average — the number this whole exercise
  set out to produce — is the wrong unit of analysis. A buyer paying a flat rate per
  hour is exposed to <i>whose</i> footage they received far more than to any property
  of the corpus.</p>
  <p>The commercial implication is not "audit corpora." It is <b>price per
  operator</b>.</p>
</section>

<section>
  <h2>Does buying more hours stop paying? <span class="n">03</span></h2>
  <p>The natural follow-up: if an operator repeats themselves, there should be a
  point where their next clip adds nothing. So — for each clip in collection order,
  what share of its frames are novel against everything that operator has already
  delivered?</p>
  <div class="figure">
    {novelty_svg()}
    <p class="figcap">Marginal novelty per clip, averaged across operators and
    weighted by frame count. Threshold 0.92, blurriest 20% dropped.</p>
  </div>
  <p><b>It does not collapse.</b> Novelty falls from 100% to roughly 75% within the
  first ten clips and then holds there — clip 50 is still about as novel as clip 15.
  Across the whole sample, 79.2% of frames are novel against their operator's earlier
  output.</p>
  <p>This was not the expected result, and it changes the advice. There is no
  saturation point to stop at within this volume. Redundancy behaves like a steady
  tax of roughly a quarter, not a cliff — so the question for a buyer is not
  <i>when to stop buying</i> but <i>what rate they are paying</i>, which loops back to
  the operator spread above.</p>
</section>

<section>
  <h2>Motion blur is a confound, not noise <span class="n">04</span></h2>
  <p>This was not in the plan. It surfaced from rendering sampled pairs and actually
  looking at them — and it is the reason the sensitivity sweep below is published
  alongside the headline rather than produced on request.</p>

  <div class="figure">
    <div class="pair">
      {img("genuine_high", "Two near-identical frames of hands assembling a small white component on a green workbench")}
      <div class="tag"><span><b>0.956</b> · same worker, 8 seconds apart</span>
      <span class="verdict-real">genuine duplicate</span></div>
    </div>
    <p class="figcap">Unambiguous. Same bench, same paper, same hands, same part.</p>
  </div>

  <div class="figure">
    <div class="pair">
      {img("blur_false", "Two heavily motion-blurred frames showing green smears with little discernible detail")}
      <div class="tag"><span><b>0.906</b> · same worker, different clips</span>
      <span class="verdict-false">blur artifact</span></div>
    </div>
    <div class="pair">
      {img("genuine_low", "Two frames of a worker in blue coveralls at the same station handling a bin of metal parts")}
      <div class="tag"><span><b>0.872</b> · same worker, different clips</span>
      <span class="verdict-real">genuine duplicate</span></div>
    </div>
    <p class="figcap">The inversion. The 0.906 pair matches on shared blur — two fast
    camera sweeps smeared into green. The 0.872 pair is unmistakably the same
    workstation, same parts bin, same task. The metric ranked the artifact above the
    real repeat.</p>
  </div>

  <p>Blur destroys the detail that distinguishes frames, so blurred frames collapse
  toward each other in embedding space and register as duplicates of one another.
  Frames are now scored for sharpness — variance of the Laplacian — and the blurriest
  fraction is dropped before measurement.</p>
  <p>Filtering blur <i>raises</i> measured redundancy, because it removes
  low-information noise rather than real repeats. A filter that moves the headline in
  the flattering direction is exactly the kind of choice a reader should distrust, so
  here is the whole sweep:</p>

  <div class="figure">
    {sensitivity_svg()}
    <p class="figcap">Redundancy at threshold 0.92 across blur-gate strengths. The
    figure moves 3.8 points end to end and plateaus past 20% — the cutoff is not
    carrying the claim.</p>
  </div>

  <p>Blur share is independently useful. Frames too blurred to distinguish are also
  too blurred to train on, whether or not they repeat.</p>
</section>

<section>
  <h2>Does the embedding see this footage? <span class="n">05</span></h2>
  <p>The obvious objection: CLIP was trained on web photography, not fisheye
  industrial video. If factory interiors collapse into a narrow region of the
  embedding space, everything looks alike and the redundancy figure is measuring a
  blind spot rather than a corpus.</p>
  <p>The corpus answers this itself. It spans 85 physically distinct factories, which
  is a free control: frames from different plants must be clearly less similar than
  frames from the same worker. Six factories were sampled to test it.</p>
  <div class="figure">
    {distribution_svg()}
    <p class="figcap">Random-pair cosine similarity, 20,000 pairs per scope, across
    six factories. Dots mark mean, p95 and p99. Separation +0.155, and
    <b>zero</b> cross-factory pairs clear 0.92 — the threshold sits above the 99th
    percentile of same-worker pairs.</p>
  </div>
  <p>The pass criteria were written into <code>validate.py</code> before the gated
  data was accessible: separation above 0.10, and under 1% of cross-factory pairs
  clearing the threshold. Measured: 0.155, and 0.00%. Not one pair in twenty
  thousand.</p>
  <h3>A sharper version of the same question</h3>
  <p>Mean separation is a blunt instrument. The direct test: if the embedding had
  collapsed this footage, it could not tell workstations <i>inside a single factory</i>
  apart. So — can a linear classifier read a frame's embedding and name the station
  it came from?</p>
  <div class="figure">
    {recall_svg()}
    <p class="figcap">Workstation identification, 7 classes, held-out split
    <b>by clip</b> so no clip appears on both sides — frames seconds apart would
    otherwise leak the answer. Overall accuracy 92.9% against 14.3% chance.</p>
  </div>
  <p>This is arguably a <i>harder</i> test than the cross-factory one: stations inside
  one plant resemble each other far more than two different plants do. Passing it at
  92.9% says the embedding carries fine-grained discriminative structure on exactly
  this footage — which is the property the redundancy metric depends on.</p>
  <p><b>What remains open.</b> Each worker has their own fisheye intrinsics, so part of
  that signal may be lens rather than scene. And validating the method is not the same
  as validating the number: 27.2% is what factory 051 measures, and whether other
  plants land near it is a separate question this does not answer.</p>
</section>

<section>
  <h2>What this does not show <span class="n">06</span></h2>
  <ul>
    <li><b>One factory.</b> Factory work is also the easy case — repetitive manual
    labour should produce repetitive footage. This says nothing about kitchen or
    household corpora without re-running.</li>
    <li><b>Fisheye is uncorrected.</b> Per-worker intrinsics differ, so cross-worker
    comparison is confounded. Within-worker numbers are the trustworthy ones.</li>
    <li><b>Keyframe sampling</b> ties sample rate to encoder settings rather than a
    fixed interval. Fine for scene-level redundancy; wrong for anything
    motion-sensitive.</li>
    <li><b>Redundant is not worthless.</b> Repetition carries real signal for
    robustness, and failure footage is valuable for recovery training. The argument is
    that buyers should be able to <i>price</i> redundancy — not that it should be
    deleted.</li>
  </ul>
  <p>None of this is a criticism of the corpus or the people who built it. Repetitive
  manual work produces repetitive footage; that is a property of factories, not a
  failure of whoever pointed the camera. Egocentric-10K is open, documented and
  Apache-2.0, which is the only reason this analysis exists at all.</p>
</section>

<section>
  <h2>Questions this answers <span class="n">07</span></h2>

  <div class="qa">
    <h3>How do you measure redundancy in a video training corpus?</h3>
    <p>Sample keyframes, embed each one, and for every frame record its highest
    cosine similarity to any <i>earlier</i> frame in the corpus. The share of
    frames above a chosen threshold is the redundancy figure. Computing it this
    way — one pass, best-match-against-everything-earlier — means the whole
    threshold curve derives from a single array, so the result does not depend
    on a cutoff picked in advance.</p>
  </div>

  <div class="qa">
    <h3>How much of a robotics training dataset is duplicate content?</h3>
    <p>On the one corpus measured here — 10,000 hours of factory-floor
    egocentric video — roughly a quarter, 27.2% with a 95% interval of
    [{CI_LO:.1f}%, {CI_HI:.1f}%]. But the corpus figure is the wrong unit:
    individual operators in that same corpus ranged from 7.4% to 46.0%.</p>
  </div>

  <div class="qa">
    <h3>Should I filter blurry frames before measuring similarity?</h3>
    <p>Yes, and not only for image quality. Motion blur destroys the detail
    that distinguishes frames, so blurred frames match <i>each other</i> and
    register as duplicates of unrelated content. Measured here, a blur-only
    pair scored 0.906 while a genuine same-workstation pair scored 0.872 —
    the metric ranked the artifact above the real repeat. Score sharpness
    (variance of the Laplacian) and drop the blurriest fraction first.</p>
  </div>

  <div class="qa">
    <h3>Can I use bootstrap resampling for a confidence interval on this?</h3>
    <p>No. Resampling with replacement puts the same clip in the corpus twice,
    and this statistic measures duplication — so every repeated clip scores as
    a perfect match against its own copy. Attempted here, it returned a 95%
    interval of [49.2%, 53.4%] around a 27.2% estimate. Use a jackknife, which
    never duplicates anything.</p>
  </div>

  <div class="qa">
    <h3>Does buying more hours from the same operator stop paying?</h3>
    <p>Not within the range measured. Marginal novelty falls from 100% to
    roughly 75% within the first ten clips and then holds flat through clip 52.
    Redundancy behaves like a steady tax of about a quarter, not a saturation
    cliff — so the actionable question is what rate you are paying, which
    depends on which operator you bought from.</p>
  </div>

  <div class="qa">
    <h3>How do I know the embedding is valid on my own footage?</h3>
    <p>Check two things before trusting any figure. First, saturation: what
    share of <i>random</i> frame pairs already clears the duplicate threshold?
    Real footage measured 0.03% at a 0.647 median; a synthetic corpus that
    broke the metric measured 26.2% at a 0.904 median. Second, separability:
    can a linear model name the operator from a single frame? Below about 2×
    chance, the embedding is not resolving the footage. Both are implemented
    as hard gates in the tool below.</p>
  </div>
</section>

<section>
  <h2>Reproduce it <span class="n">08</span></h2>
  <p>Runs on a GTX 1050 Ti in about an hour, network-bound. Nothing is downloaded:
  shards stream from the Hub, frames decode in memory, only embeddings persist.</p>
<pre>pip install -r requirements.txt
python src/pipeline.py --tag myrun --source subset</pre>
  <p>The validation gate blocks the report if the embedding fails to discriminate on
  your footage. That is deliberate, and not skippable from the entry point.</p>
</section>

<footer>
  <p>The interesting number was never 27%. It is that two operators on the same
  factory floor, sold at the same rate, differ by 6× in what you are paying for
  twice — and nobody currently reports it.</p>
</footer>
</div>
"""

dest = OUT_DIR / "findings.html"
dest.write_text(HTML, encoding="utf-8")
print(f"wrote {dest}  ({len(HTML) / 1024:.1f} KB)")
