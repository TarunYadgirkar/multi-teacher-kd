"""Build REPORT.md and the results figure from results/.

Every number comes from saved per-item rows, so the report can be rebuilt (or the
scorers changed and rows rescored) without regenerating anything.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mtkd.data import CATEGORIES
from mtkd.run_eval import read_rows
from mtkd.stats import macro_gain, macro_gain_boot, mean_ci, paired_diff_ci, summarize_draws

PRIMARY = ("task", "learned")
REFERENCE = "instruct"
METRIC = {"math": "GSM8K exact match", "code": "MBPP pass@1", "qa": "ARC-C accuracy (log-lik)",
          "summ": "SamSum ROUGE-L F1"}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(d: dict, signed: bool = False) -> str:
    fmt = (lambda v: f"{100 * v:+.1f}") if signed else _pct
    return f"{fmt(d['mean'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"


def _per_item(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[row["category"]][row["id"]] = row["score"]
    return out


class Results:
    """Seed-averaged per-item scores for every evaluated run, aligned on item ids."""

    def __init__(self, cfg: dict):
        root = Path(cfg["paths"]["results"]) / "eval"
        self.cfg = cfg
        self.summaries = {p.stem: json.loads(p.read_text()) for p in sorted(root.glob("*.json"))}
        self.rows = {name: read_rows(root / f"{name}.jsonl.gz") for name in self.summaries}
        base = _per_item(self.rows["base"])
        self.ids = {c: sorted(base[c]) for c in base}
        self.base = {c: np.array([base[c][i] for i in self.ids[c]]) for c in base}
        self.seeds: dict[str, list[int]] = defaultdict(list)
        for summary in self.summaries.values():
            if "condition" in summary:
                self.seeds[summary["condition"]].append(summary["seed"])
        self.conditions = [c for c in cfg["conditions"] if c in self.seeds]
        self.scores = {c: self._seed_mean(c) for c in self.conditions}
        self.teachers = [k for k in cfg["teachers"] if f"teacher-{k}" in self.summaries]
        for key in self.teachers:
            self.scores[f"teacher-{key}"] = self._aligned(f"teacher-{key}")

    def _aligned(self, name: str) -> dict[str, np.ndarray]:
        items = _per_item(self.rows[name])
        return {c: np.array([items[c][i] for i in self.ids[c]]) for c in self.ids}

    def _seed_mean(self, condition: str) -> dict[str, np.ndarray]:
        runs = [self._aligned(f"{condition}-s{s}") for s in sorted(self.seeds[condition])]
        return {c: np.mean([r[c] for r in runs], axis=0) for c in self.ids}

    def seed_gains(self, condition: str) -> list[float]:
        return [macro_gain(self._aligned(f"{condition}-s{s}"), self.base)
                for s in sorted(self.seeds[condition])]

    def gain(self, name: str) -> dict:
        return summarize_draws(macro_gain(self.scores[name], self.base),
                               macro_gain_boot(self.scores[name], self.base))

    def gain_diff(self, a: str, b: str, alpha: float = 0.05) -> dict:
        point = macro_gain(self.scores[a], self.base) - macro_gain(self.scores[b], self.base)
        draws = macro_gain_boot(self.scores[a], self.base) - macro_gain_boot(self.scores[b], self.base)
        return summarize_draws(point, draws, alpha)


def _probe_section(cfg: dict) -> list[str]:
    path = Path(cfg["paths"]["results"]) / "probe" / "summary.json"
    if not path.exists():
        return []
    probe = json.loads(path.read_text())
    lines = ["## Teacher probe", "",
             "Each teacher answers the probe split under the student's chat template. Mean score (%) "
             "with a 95% bootstrap CI; trunc is the share of outputs that hit the token limit.", "",
             "| Teacher | " + " | ".join(METRIC[c] for c in CATEGORIES) + " |",
             "|---|" + "---|" * len(CATEGORIES)]
    for key, scores in probe["scores"].items():
        cells = [f"{_ci(scores[c])}, trunc {_pct(scores['truncated'][c])}%" for c in CATEGORIES]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", f"Learned map (the probe winner only when its paired CI against `{cfg['default_teacher']}` "
              "excludes zero):", ""]
    for c in CATEGORIES:
        ev = probe["evidence"][c]
        vs = ev["vs_default"]
        why = "the default teacher won" if vs is None else f"winner {ev['winner']}, difference {_ci(vs, True)} pts"
        lines.append(f"- {c}: **{probe['learned_map'][c]}** ({why})")
    return lines + [""]


def _main_table(res: Results) -> list[str]:
    lines = ["## Main results", "",
             "Seed-averaged eval scores (%) with 95% item-bootstrap CIs. G is the macro normalized "
             "gain over the untrained student, (score - base) / (1 - base) averaged over the four "
             "categories, in points of headroom; seed sd is the spread of per-seed G.", "",
             "| Run | Seeds | " + " | ".join(METRIC[c] for c in CATEGORIES) + " | G | seed sd of G |",
             "|---|---|" + "---|" * len(CATEGORIES) + "---|---|"]
    base_cells = " | ".join(_ci(mean_ci(res.base[c])) for c in CATEGORIES)
    lines.append(f"| base (untrained) | - | {base_cells} | 0 | - |")
    for name in [*res.conditions, *[f"teacher-{k}" for k in res.teachers]]:
        seeds = res.seeds.get(name, [])
        sd = f"{100 * np.std(res.seed_gains(name), ddof=1):.1f}" if len(seeds) > 1 else "-"
        cells = " | ".join(_ci(mean_ci(res.scores[name][c])) for c in CATEGORIES)
        lines.append(f"| {name} | {len(seeds) or '-'} | {cells} | {_ci(res.gain(name), True)} | {sd} |")
    return lines + [""]


def _comparison_rows(res: Results, names: list[str], alpha: float) -> list[str]:
    rows = []
    for name in names:
        if name not in res.scores or name == REFERENCE:
            continue
        diff = res.gain_diff(name, REFERENCE, alpha)
        per_cat = " | ".join(_ci(paired_diff_ci(res.scores[name][c], res.scores[REFERENCE][c]), True)
                             for c in CATEGORIES)
        rows.append(f"| {name} - {REFERENCE} | {_ci(diff, True)} | {diff['p_le_0']:.3f} | {per_cat} |")
    return rows


def _comparisons(res: Results) -> list[str]:
    if REFERENCE not in res.scores:
        return []
    header = ["| Comparison | dG | share of draws <= 0 | " + " | ".join(f"d {c}" for c in CATEGORIES) + " |",
              "|---|---|---|" + "---|" * len(CATEGORIES)]
    secondary = [c for c in res.conditions if c not in PRIMARY]
    return ["## Primary comparisons (pre-registered)", "",
            "Difference in G with a 97.5% paired-bootstrap CI (Bonferroni over the two comparisons). "
            "Per-category differences are in score points with 95% CIs.", "", *header,
            *_comparison_rows(res, list(PRIMARY), 0.025), "",
            "## Secondary comparisons", "", "Same layout, 95% CIs, no multiplicity correction.", "",
            *header, *_comparison_rows(res, secondary, 0.05), ""]


def _diagnostics(res: Results, cfg: dict) -> list[str]:
    limits = ", ".join(f"{c} {cfg['generation']['max_new_tokens'][c]}" for c in CATEGORIES)
    lines = ["## Generation diagnostics", "",
             f"Share of eval outputs that hit the token limit ({limits}); mean math output tokens; "
             "tool-style math answers, i.e. outputs that write Python and then a ```output block the "
             "model made up itself, with their accuracy; ARC accuracy scored from the generated text "
             "instead of log-likelihood; SamSum detail.", "",
             "| Run | trunc math | trunc code | trunc summ | math tokens | tool-style math (acc) | "
             "ARC generative | ROUGE-1 / ROUGE-2 | summary words |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, summary in res.summaries.items():
        s = summary["summary"]
        share, acc = tool_style(res.rows[name])
        lines.append(f"| {name} | {_pct(s['math']['truncated'])}% | {_pct(s['code']['truncated'])}% | "
                     f"{_pct(s['summ']['truncated'])}% | {s['math']['mean_tokens']:.0f} | "
                     f"{_pct(share)}% ({_pct(acc) if acc is not None else '-'}) | {_pct(s['qa']['gen_score'])} | "
                     f"{_pct(s['summ']['rouge1'])} / {_pct(s['summ']['rouge2'])} | {s['summ']['words']:.1f} |")
    return lines + [""]


def tool_style(rows: list[dict]) -> tuple[float, float | None]:
    """Share of math outputs in Qwen2.5-Math's tool-integrated format (code followed by a
    ```output block), and their accuracy. Nothing executes that code during generation."""
    math = [r for r in rows if r["category"] == "math"]
    tool = [r for r in math if "```output" in r["output"]]
    acc = sum(r["score"] for r in tool) / len(tool) if tool else None
    return len(tool) / len(math), acc


def _training(cfg: dict) -> list[str]:
    metas = sorted((Path(cfg["paths"]["results"]) / "train").glob("*.json"))
    if not metas:
        return []
    lines = ["## Training runs", "",
             "| Run | Steps | Minutes | Teacher loads | Loss, first 8 steps | Loss, last 8 steps | Mean completion tokens |",
             "|---|---|---|---|---|---|---|"]
    for path in metas:
        meta = json.loads(path.read_text())
        log = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        first = np.mean([r["loss"] for r in log[:8]])
        last = np.mean([r["loss"] for r in log[-8:]])
        tokens = np.mean([r["completion_tokens"] for r in log])
        lines.append(f"| {path.stem} | {meta['steps']} | {meta['minutes']} | {meta['teacher_loads']} | "
                     f"{first:.4f} | {last:.4f} | {tokens:.0f} |")
    return lines + ["", "Losses are not comparable across conditions: each is measured against its own "
                    "teacher(s), and SFT reports cross-entropy.", ""]


def _provenance(res: Results) -> list[str]:
    shas = sorted({s["provenance"]["git_sha"][:7] for s in res.summaries.values()})
    first = next(iter(res.summaries.values()))["provenance"]
    dirty = any(s["provenance"]["git_dirty"] for s in res.summaries.values())
    versions = ", ".join(f"{k} {v}" for k, v in first["versions"].items())
    return ["## Provenance", "",
            f"- Code: git {', '.join(shas)}{' (some runs from a tree with uncommitted changes)' if dirty else ''}",
            f"- Hardware: {first['chip']}, device `{first['device']}`, {first['platform']}",
            f"- Software: Python {first['python']}, {versions}", ""]


def _collapsed(results_dir: Path) -> list[str]:
    path = results_dir / "collapsed.json"
    if not path.exists():
        return []
    notes = [f"- `{name}` routes every category exactly like `{twin}`, so it was not trained separately."
             for name, twin in json.loads(path.read_text()).items() if name != "learned_map"]
    return ["## Collapsed conditions", "", *notes, ""]


def build_report(cfg: dict) -> None:
    results_dir = Path(cfg["paths"]["results"])
    if not (results_dir / "eval" / "base.json").exists():
        print("report: no base eval yet")
        return
    res = Results(cfg)
    lines = ["# Report", "", "Generated by `uv run mtkd report` from `results/`. Do not edit by hand.", "",
             *_collapsed(results_dir), *_probe_section(cfg), *_main_table(res), *_comparisons(res),
             *_diagnostics(res, cfg), *_training(cfg), *_provenance(res)]
    out = Path(cfg.get("report_path", "REPORT.md"))
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    if res.conditions:
        figure = Path(cfg["paths"]["figures"]) / "results.png"
        plot_results(res, figure)
        print(f"wrote {figure}")


SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, DEEMPH = "#2a78d6", "#a3a29c"


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)


OUTLIER_PTS = 12.0


def _panel_limits(intervals: list[dict]) -> tuple[float, float]:
    """x-range from rows whose mean is within OUTLIER_PTS of the median row, so one extreme
    condition does not flatten every other row; outliers are drawn at the edge."""
    means = sorted(100 * d["mean"] for d in intervals)
    median = means[len(means) // 2]
    core = [d for d in intervals if abs(100 * d["mean"] - median) <= OUTLIER_PTS]
    lo = min(0.0, min(100 * d["lo"] for d in core))
    hi = max(0.0, max(100 * d["hi"] for d in core))
    span = hi - lo
    below = any(100 * d["mean"] < median - OUTLIER_PTS for d in intervals)
    above = any(100 * d["mean"] > median + OUTLIER_PTS for d in intervals)
    return lo - (0.3 if below else 0.08) * span, hi + (0.3 if above else 0.08) * span


def _draw_row(ax, y: int, d: dict, color: str, limits: tuple[float, float]) -> None:
    lo, hi = limits
    mean = 100 * d["mean"]
    if lo <= mean <= hi:
        ax.plot([max(lo, 100 * d["lo"]), min(hi, 100 * d["hi"])], [y, y], color=color, lw=2,
                solid_capstyle="round")
        ax.plot(mean, y, "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=3)
        return
    inset = 0.03 * (hi - lo)
    edge, marker, align = (lo + inset, "<", "left") if mean < lo else (hi - inset, ">", "right")
    ax.plot(edge, y, marker, color=color, ms=8, zorder=3)
    ax.annotate(f"{mean:+.1f}", (edge, y), xytext=(8 if align == "left" else -8, -4),
                textcoords="offset points", ha=align, va="top", fontsize=9, color=MUTED)


def plot_results(res: Results, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res.conditions)
    panels = [("Macro gain G", None), *[(METRIC[c], c) for c in CATEGORIES]]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.45 * len(names) + 1.7), sharey=True,
                             facecolor=SURFACE)
    for ax, (title, category) in zip(axes, panels):
        intervals = [res.gain(n) if category is None
                     else paired_diff_ci(res.scores[n][category], res.base[category]) for n in names]
        limits = _panel_limits(intervals)
        for y, (name, d) in enumerate(zip(names, intervals)):
            _draw_row(ax, y, d, ACCENT if name in (REFERENCE, *PRIMARY) else DEEMPH, limits)
        ax.set_xlim(*limits)
        _style(ax, title, "points of headroom" if category is None else "points vs untrained")
    axes[0].set_yticks(range(len(names)), names, fontsize=10, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Change over the untrained student (seed mean, 95% CI). Blue: Instruct-only and the "
                 "two pre-registered routing policies. Arrows mark values beyond the axis.",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
