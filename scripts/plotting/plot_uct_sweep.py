"""Figures for the pure-UCT simulation sweep.

Both figure kinds carry the same three panels, sharing a y-axis:

  A  greedy R^2            -- deterministic, no noise, tau=0 at selection
  B  avg self-play R^2     -- mean terminal reward over the iteration's games
  C  best-so-far R^2       -- best structure any game has emitted up to here

`--figures family` draws one figure per target family, hue = target. This is
the encoding for a single-budget run, where the comparison that matters is
across the family; it is what the notes embed.

`--figures per-target` draws one figure per target, hue = simulation budget.
The budget is an *ordered magnitude*, so it rides a single-hue sequential ramp
(light = shallow, dark = deep), never a categorical rainbow. Informative only
once a multi-budget sweep has run -- at one budget the hue channel is
degenerate and the legend has a single entry.

Coincident series. Targets often sit at exactly the same value (panels A and
C), so equal-weight lines would show only whichever hue was drawn last. The
family figure descends line width with draw order instead, nesting a
coincident bundle into visible concentric bands. Backup rules that agree
numerically are detected and stated in a corner note rather than drawn twice --
an overplotted dashed line is invisible, not evidence.

Usage:
    python scripts/plotting/plot_uct_sweep.py --sweep-dir experiments/symreg_uct/<run>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sraz.instances.symreg.game import fit_expression
from sraz.instances.symreg.targets import family_targets, get_target


PANELS = [
    ("greedy_r2", "A  greedy $R^2$  ($\\tau=0$, no noise)"),
    ("avg_train_reward", "B  avg self-play $R^2$  ($\\tau=1$)"),
    ("best_r2_so_far", "C  best-so-far $R^2$"),
]

# The exactly-expressible structure for each family: the sentence whose fitted
# form equals the target. Used as the per-target reference line.
EXACT_STRUCTURE = {
    "linear": "+ C0 * C1 x",
    "quadratic": "+ C0 + * C1 x * C2 * x x",
}


def load_sweep(sweep_dir: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in (sweep_dir / "sweep.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((sweep_dir / "sweep_meta.json").read_text())
    return rows, meta


def exact_reference(target_name: str, max_nfev: int | None) -> tuple[float, str]:
    """R^2 of the structure that can represent the target exactly."""
    t = get_target(target_name)
    rule = EXACT_STRUCTURE[t.family]
    return fit_expression(rule, t.xs(), t.ys(t.xs()), max_nfev=max_nfev), rule


def plot_target(rows: list[dict], meta: dict, target_name: str, out_path: Path):
    sims = sorted({r["n_simulations"] for r in rows})
    rules = [r for r in ("mean", "max", "topk", "softmax") if r in {x["backup_rule"] for x in rows}]

    # The simulation budget is an ordered magnitude, so it rides a sequential
    # single-hue ramp (shallow -> light, deep -> dark), never a categorical
    # rainbow. Truncated at 0.45 so the lightest curve still reads against
    # white; a lone budget takes a mid-dark step rather than the light end,
    # which would otherwise render a single-cell sweep nearly invisible.
    cmap = plt.get_cmap("Blues")
    if len(sims) == 1:
        colors = {sims[0]: cmap(0.8)}
    else:
        colors = {s: cmap(0.45 + 0.55 * i / (len(sims) - 1)) for i, s in enumerate(sims)}
    styles = {"mean": "-", "max": "--", "topk": ":", "softmax": "-."}

    series = defaultdict(list)
    for r in rows:
        series[(r["n_simulations"], r["backup_rule"])].append(r)
    for v in series.values():
        v.sort(key=lambda r: r["iteration"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    r2_exact, exact_rule = exact_reference(target_name, meta.get("lmfit_max_nfev"))
    target = get_target(target_name)

    for ax, (key, title) in zip(axes, PANELS):
        for s in sims:
            for rule in rules:
                pts = series.get((s, rule))
                if not pts:
                    continue
                ax.plot([p["iteration"] for p in pts], [p[key] for p in pts],
                        styles[rule], color=colors[s], linewidth=2.0,
                        marker="o" if rule == rules[0] else None,
                        markersize=3.5, alpha=0.9)
        ax.axhline(r2_exact, color="0.35", linewidth=1.2, linestyle=(0, (6, 4)), zorder=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("$R^2$")
    axes[-1].annotate(f"exact structure  $R^2$={r2_exact:.4f}",
                      xy=(0.98, r2_exact), xycoords=("axes fraction", "data"),
                      ha="right", va="bottom", fontsize=8, color="0.35")

    # Legend: hue = budget (ordered), style = backup rule. Both spelled out, so
    # identity never rests on colour alone.
    sim_handles = [plt.Line2D([], [], color=colors[s], linewidth=2.4, label=f"{s}")
                   for s in sims]
    rule_handles = [plt.Line2D([], [], color="0.35", linewidth=2.0, linestyle=styles[r], label=r)
                    for r in rules]
    leg1 = fig.legend(handles=sim_handles, title="MCTS simulations", loc="center left",
                      bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=8, title_fontsize=8)
    fig.add_artist(leg1)
    fig.legend(handles=rule_handles, title="backup rule", loc="center left",
               bbox_to_anchor=(1.0, 0.26), frameon=False, fontsize=8, title_fontsize=8)

    coef = ", ".join(f"$c_{i}$={c:g}" for i, c in enumerate(target.coeffs))
    fig.suptitle(
        f"Pure UCT on the SR grammar game — {target_name} ({coef})   |   "
        f"{target.label}",
        fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_family_panels(rows: list[dict], meta: dict, family: str, out_path: Path):
    """One figure per family: three panels, one line per target.

    The per-target figure above encodes the simulation budget as hue, which is
    the right encoding for a sweep. At a single budget that channel is
    degenerate, and the comparison the reader actually needs is across the
    family: hue therefore carries the *target*, and the four members share axes
    so that a flat panel A reads as flat everywhere rather than four times over.
    """
    targets = [t for t in family_targets(family) if t.name in {r["target"] for r in rows}]
    rules = [r for r in ("mean", "max", "topk", "softmax") if r in {x["backup_rule"] for x in rows}]
    cmap = plt.get_cmap("tab10")
    colors = {t.name: cmap(i) for i, t in enumerate(targets)}

    series = defaultdict(list)
    for r in rows:
        series[(r["target"], r["backup_rule"])].append(r)
    for v in series.values():
        v.sort(key=lambda r: r["iteration"])

    # Panels A and C pin several targets to the same value, so equal-weight
    # lines would show only whichever hue was drawn last. Widths instead
    # descend with draw order, so a coincident bundle nests into visible
    # concentric bands while a separated line stays solid and traceable.
    # Markers are staggered around the cycle for the same reason.
    n = max(len(targets), 1)
    widths = {t.name: w for t, w in zip(targets, np.linspace(5.0, 1.4, n))}

    # `mean` and `max` coincide exactly whenever every leaf backs up a constant
    # (see the note's Section 6). Drawing the second rule on top of the first
    # would render an invisible line and a legend entry for a style that never
    # appears, so the degeneracy is detected and stated instead of drawn.
    def rule_series(rule):
        return {(t.name, p["iteration"]): p for t in targets for p in series.get((t.name, rule), [])}

    base = rules[0]
    ref = rule_series(base)
    identical = [r for r in rules[1:]
                 if {k: {p: v[p] for p in ("greedy_r2", "avg_train_reward", "best_r2_so_far")}
                     for k, v in rule_series(r).items()}
                 == {k: {p: v[p] for p in ("greedy_r2", "avg_train_reward", "best_r2_so_far")}
                     for k, v in ref.items()}]
    drawn = [base] + [r for r in rules[1:] if r not in identical]
    styles = {"mean": "-", "max": "--", "topk": ":", "softmax": "-."}

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)

    exact = [exact_reference(t.name, meta.get("lmfit_max_nfev"))[0] for t in targets]
    r2_exact = float(np.max(exact))

    for ax, (key, title) in zip(axes, PANELS):
        for i, t in enumerate(targets):
            for rule in drawn:
                pts = series.get((t.name, rule))
                if not pts:
                    continue
                ax.plot([p["iteration"] for p in pts], [p[key] for p in pts],
                        color=colors[t.name], linewidth=widths[t.name],
                        linestyle="-" if rule == base else styles[rule],
                        marker="o" if rule == base else None,
                        markevery=(i, n), markersize=4.5,
                        solid_capstyle="butt", alpha=0.95)
        ax.axhline(r2_exact, color="0.35", linewidth=1.2, linestyle=(0, (6, 4)), zorder=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("$R^2$")
    axes[-1].annotate(f"exact structure  $R^2$={r2_exact:.4f}",
                      xy=(0.98, r2_exact), xycoords=("axes fraction", "data"),
                      ha="right", va="bottom", fontsize=8, color="0.35")

    # Hue = target, spelled out by design intent rather than by dict key. The
    # legend swatch carries the same dash slice as the line it names.
    tgt_handles = [plt.Line2D([], [], color=colors[t.name], linewidth=widths[t.name],
                              marker="o", markersize=4.5, label=t.short_label)
                   for t in targets]
    leg1 = fig.legend(handles=tgt_handles, title="target shape", loc="center left",
                      bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=8, title_fontsize=8)
    fig.add_artist(leg1)

    if identical:
        fig.text(1.0, 0.30, "backup rule\n" + " ≡ ".join([base] + identical)
                 + "\n(identical to 1e-12;\ndrawn once)",
                 fontsize=8, va="center", ha="left", color="0.35")
    elif len(drawn) > 1:
        rule_handles = [plt.Line2D([], [], color="0.35", linewidth=2.0,
                                   linestyle=styles[r], label=r) for r in drawn]
        fig.legend(handles=rule_handles, title="backup rule", loc="center left",
                   bbox_to_anchor=(1.0, 0.24), frameon=False, fontsize=8, title_fontsize=8)

    sims = sorted({r["n_simulations"] for r in rows})
    budget = f"{sims[0]}" if len(sims) == 1 else f"{min(sims)}–{max(sims)}"
    fig.suptitle(f"Pure UCT on the {family} family — {budget} simulations, "
                 f"{len(targets)} targets", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.82, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_family_summary(rows: list[dict], meta: dict, out_path: Path):
    """Final best-so-far R^2 against simulation budget, one line per target."""
    targets = sorted({r["target"] for r in rows})
    rules = sorted({r["backup_rule"] for r in rows})
    styles = {"mean": "-", "max": "--", "topk": ":", "softmax": "-."}
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ti, t in enumerate(targets):
        for rule in rules:
            pts = defaultdict(float)
            for r in rows:
                if r["target"] == t and r["backup_rule"] == rule:
                    pts[r["n_simulations"]] = max(pts[r["n_simulations"]], r["best_r2_so_far"])
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], styles[rule], color=cmap(ti),
                    marker="o", markersize=5, linewidth=2.0,
                    label=f"{t} ({rule})" if len(rules) > 1 else t)
    ax.set_xlabel("MCTS simulations")
    ax.set_ylabel("final best-so-far $R^2$")
    ax.set_title("Search budget vs best structure found", fontsize=11, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Plot the pure-UCT sweep")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Default: <sweep-dir>/figures")
    p.add_argument("--prefix", default="uct", help="Output filename prefix")
    p.add_argument("--figures", choices=("family", "per-target", "all"), default="all",
                   help="family: one 3-panel figure per family (the note's figures). "
                        "per-target: one figure per target, hue = simulation budget "
                        "(informative only once a multi-budget sweep has run).")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "figures"
    rows, meta = load_sweep(sweep_dir)

    written = []
    if args.figures in ("family", "all"):
        for fam in sorted({get_target(r["target"]).family for r in rows}):
            frows = [r for r in rows if get_target(r["target"]).family == fam]
            written.append(plot_family_panels(frows, meta, fam,
                                              out_dir / f"{args.prefix}_{fam}.png"))
    if args.figures in ("per-target", "all"):
        for t in sorted({r["target"] for r in rows}):
            trows = [r for r in rows if r["target"] == t]
            written.append(plot_target(trows, meta, t, out_dir / f"{args.prefix}_{t}.png"))
        written.append(plot_family_summary(rows, meta, out_dir / f"{args.prefix}_summary.png"))

    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
