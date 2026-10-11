"""README figures from one or more results files.

    python -m stale_rag.figures results/hf__Qwen_Qwen2.5-0.5B-Instruct.jsonl [more.jsonl ...]

conditions.png  share of answers naming the current holder, per context and
                date condition, one panel per instruction (one row per model)
contrasts.png   the headline contrasts with 95% bootstrap intervals, models
                side by side
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analyze import contrast, contrasts, load, summarise, _cell  # noqa: E402
from .config import ROOT  # noqa: E402

FIGS = ROOT / "figures"

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3

CONTEXTS = [("closed", "no passage"), ("new", "current only"), ("old", "stale only"),
            ("old_new", "stale, then\ncurrent"), ("new_old", "current, then\nstale")]
DATES = [("none", "no dates"), ("true", "true dates"), ("swapped", "dates swapped")]
INSTRUCTS = [("plain", "Plain question"), ("recency", '"Trust the most recent source"')]


def model_name(path):
    return Path(path).stem.split("__", 1)[-1].replace("Qwen_", "").replace("-Instruct", "")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def conditions_figure(frames, out):
    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 3.6 * len(frames)),
                             squeeze=False, sharey=True, facecolor=SURFACE)
    for row, (name, df) in enumerate(frames):
        for col, (ins, title) in enumerate(INSTRUCTS):
            ax = axes[row][col]
            _style(ax)
            for di, (dates, label) in enumerate(DATES):
                xs, ys, los, his = [], [], [], []
                for ci, (ctx, _) in enumerate(CONTEXTS):
                    if ctx == "closed" and dates != "none":
                        continue
                    cell = _cell(df, context=ctx, dates=dates, instruct=ins)
                    ci95 = contrast(cell, cell * 0)
                    offset = 0 if ctx == "closed" else (di - 1) * 0.22
                    xs.append(ci + offset)
                    ys.append(ci95["diff"] * 100)
                    los.append(ci95["lo"] * 100)
                    his.append(ci95["hi"] * 100)
                color = SERIES[di]
                ax.vlines(xs, los, his, color=color, linewidth=2)
                ax.scatter(xs, ys, s=42, color=color, edgecolor=SURFACE, linewidth=1.5,
                           zorder=3, label=label)
            ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.set_xticks(range(len(CONTEXTS)), [c[1] for c in CONTEXTS])
            ax.set_ylim(-3, 103)
            ax.set_title(f"{title}  ({name})", color=INK, fontsize=10.5, loc="left")
            if col == 0:
                ax.set_ylabel("answers naming the current holder (%)", color=INK2, fontsize=9)
    axes[0][1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# The stale-override contrast (~+95) is left out: it would crush this scale, and
# conditions.png already shows it.
LABELS = {
    "order effect (plain)": "Order: current first minus stale first",
    "order effect (recency)": "Order, with recency instruction",
    "date effect (plain)": "Dates: true minus swapped",
    "date effect (recency)": "Dates, with recency instruction",
    "recency instruction, mixed, true dates": "Recency instruction (both passages, true dates)",
}


def contrasts_figure(frames, out):
    rows = list(LABELS)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    n = len(frames)
    for mi, (name, df) in enumerate(frames):
        c = contrasts(df).reindex(rows)
        y = np.arange(len(rows)) + (mi - (n - 1) / 2) * 0.25
        ax.hlines(y, c["lo"] * 100, c["hi"] * 100, color=SERIES[mi], linewidth=2)
        ax.scatter(c["diff"] * 100, y, s=42, color=SERIES[mi], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=name)
        for yy, d in zip(y, c["diff"]):
            ax.annotate(f"{d * 100:+.0f}", (d * 100, yy), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [LABELS[r] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("difference in answers naming the current holder (percentage points)",
                  color=INK2, fontsize=9)
    if n > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    else:
        ax.set_title(frames[0][0], color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    paths = sys.argv[1:]
    frames = [(model_name(p), summarise(load(p))[0]) for p in paths]
    FIGS.mkdir(exist_ok=True)
    conditions_figure(frames, FIGS / "conditions.png")
    contrasts_figure(frames, FIGS / "contrasts.png")
    print(f"wrote {FIGS / 'conditions.png'} and {FIGS / 'contrasts.png'}")


if __name__ == "__main__":
    main()
