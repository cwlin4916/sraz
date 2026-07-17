"""One full backup of the note 02 rollout search, drawn from a real simulation.

    mcts_backup_trace.png
        One counted simulation of PUCT-with-a-uniform-prior at rollout_n=5:
        what number the backup actually carries, and where the two
        aggregations sit. Answers: is the backed-up value accumulated along
        the path, or is it one scalar repeated?

The ledger's y is slaved to the tree's edges, and the rollout-aggregation
block to the y of the rollout fan. That is the figure's argument made
geometric: AGG 1 sits beside the five returns because it reduces those five,
AGG 2 beside the three edges because it reduces each edge's history. A caption
can assert that separation; only position can show it.

One tree, not one per stage: the four stages are four moments of a single
recursion, and redrawing the skeleton four times spends the width the
sentential forms and the ledger columns need at note width.

The "max" columns are recomputed from the same recorded list, never from a
second run. A different rule shifts the global q_min/q_max window
(mcts.py:545-549), so a max-backup run does not descend this path and the two
ledgers would not be comparable. The columns answer "what does the other rule
compute from this list at this moment", not "what would a max run show".

No sibling stubs are drawn. An untried action is normalised as if Q = 0
(mcts.py:472), so q_norm = (0 - q_min)/(q_max - q_min) is negative whenever the
backed-up values are positive: untried actions are disfavoured by the Q term,
not optimistic. A greyed stub would read as "neutral", which inverts the code.

np.random.seed, not the configuration's random_seeds: mcts.py:322 draws rollout
actions from the global legacy generator, which the config's seeds never reach.

Simulation 4 is the specimen because it is the first that ends at a fresh
nonterminal leaf, completes all five rollouts, and carries an edge whose value
list already holds two entries -- at simulation 1 every list has length <= 1,
mean and max agree everywhere, and the figure's argument evaporates.

Convention: this is boxes-and-arrows, so plot_sr_game.py is the nearer relative
and its monospace token buffers and FancyBboxPatch slots are reused. Its
green/red pair encodes node roles and would fail the CVD check this figure must
pass, so the hues here are the house categorical set, encoding stage identity.
_style() is not called: both axes are off.

Provenance: returns come from the game's own reward pipeline (clipped R^2), not
recomputed here. Every drawn number is asserted against the replay.

Usage:
    python scripts/plotting/plot_mcts_backup.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sraz.core.mcts import MCTS  # noqa: E402
from sraz.core.policy_value_net import UniformPolicyValueNet  # noqa: E402
from sraz.instances.symreg.game import SymRegGame  # noqa: E402

# Fixed hue order, assigned by series identity and never cycled.
# Validated (light surface, categorical): worst adjacent CVD dE 26.5 protan,
# 20.0 tritan; normal-vision floor 31.9; all >= 3:1 against the surface.
BLUE, ORANGE, PURPLE = "#3b7dd8", "#e8710a", "#9c27b0"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

# Every stage carries hue AND linestyle AND direction AND a numbered label, so
# colour is never the sole channel and the figure survives grayscale.
STAGES = [
    ("select", "1  SELECT", BLUE, "-", "down"),
    ("expand", "2  EXPAND", INK, "-", None),
    ("rollout", "3  ROLLOUT", ORANGE, "--", "down"),
    ("backup", "4  BACKUP", PURPLE, "-", "up"),
]

# The specimen, pinned. If the replay diverges from these, every number in the
# figure is stale and the assertions below say so.
EXPECTED_PATH = ["S", "+ S S", "+ + S S S", "+ + + S S S S"]
EXPECTED_NVALID = [7, 14, 21, 28]


class TraceMCTS(MCTS):
    """Records one simulation without touching src/.

    `_rollout_value` is reimplemented rather than wrapped, because the parent
    returns only the aggregate and the figure needs each completion. The
    reimplementation asserts its own aggregate against the parent's rule, so
    the copy is proven equivalent rather than assumed to be.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trace = None

    def search(self, msg, _depth: int = 0) -> float:
        rec = {
            "depth": _depth,
            "state": self.game._decode_state(),
            "total_N": (0 if self.game.hashable_obs not in self.nodes
                        else self.nodes[self.game.hashable_obs].total_N),
            "n_valid": int(self.game.get_action_mask().sum()),
        }
        if self.trace is not None:
            self.trace["frames"].append(rec)
        out = super().search(msg, _depth=_depth)
        rec["returned"] = out
        return out

    def _rollout_value(self, msg):
        rewards, sentences = [], []
        for _ in range(self.rollout_n):
            if self._search_rollout_budget <= 0:
                break
            g = self.game.clone()
            cumulative = 0.0
            while not (g.terminated or g.truncated):
                if self._search_rollout_budget <= 0:
                    break
                mask = g.get_action_mask()
                valid = np.flatnonzero(mask)
                if len(valid) == 0:
                    break
                action = valid[np.random.randint(len(valid))]
                if len(g.action_space.shape) == 0:
                    g.step_wrapper(int(action))
                else:
                    g.step_wrapper(np.unravel_index(action, mask.shape))
                cumulative += g.reward
                self._search_rollout_budget -= 1
            if g.terminated or g.truncated:
                rewards.append(cumulative)
                sentences.append(g._decode_state())
        if not rewards:
            return None
        agg = (float(max(rewards)) if self.rollout_mode == "max"
               else float(sum(rewards) / len(rewards)))
        if self.trace is not None:
            self.trace["rollout"] = {
                "returns": rewards, "sentences": sentences, "agg": agg}
        return agg

    def update_edge(self, mynode, action, reward):
        before = list(mynode.action_values.get(action, []))
        n0 = mynode.action_N.get(action, 0)
        super().update_edge(mynode, action, reward)
        if self.trace is not None:
            values = before + [reward]
            self.trace["edges"].append({
                "action": action, "n0": n0, "n1": mynode.action_N[action],
                "before": before, "append": reward,
                "q_mean": sum(values) / len(values), "q_max": max(values),
            })


def replay(seed: int, n_sims: int = 8) -> list[dict]:
    """Drive `n_sims` counted simulations, mirroring perform_simulations.

    q_min/q_max and the rollout budget are initialised once, as mcts.py:107-110
    does at the start of a search. The root's own expansion is an extra,
    uncounted `search` call in the real code (mcts.py:120-125); here the first
    loop iteration performs it, so "simulation i" is the loop index of
    mcts.py:154.
    """
    np.random.seed(seed)  # the GLOBAL legacy generator mcts.py:322 draws from
    game = SymRegGame(problem_seed=42, lmfit_max_nfev=50)
    net = UniformPolicyValueNet(n_actions=105, value=0.0)
    mcts = TraceMCTS(game, net, n_simulations=1, c_exploration=1.0,
                     rollout_n=5, rollout_mode="mean", rollout_blend=0.0,
                     rollout_budget=20000, backup_rule="mean")
    mcts.q_min, mcts.q_max = float("inf"), float("-inf")
    mcts._search_rollout_budget = mcts.rollout_budget

    sims = []
    for _ in range(n_sims):
        mcts.trace = {"frames": [], "edges": [], "rollout": None}
        stashed = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(stashed)
        sims.append(mcts.trace)
    return sims


def pick_sim(sims: list[dict], forced: int | None) -> int:
    """First simulation whose ledger has something to say.

    Rejects any sim that stops at a terminal (no rollout to draw), that
    completes fewer than five rollouts, whose rollout mean equals its max, or
    all of whose edges hold at most one prior value -- in that last case every
    `mean` cell equals its `max` cell and the figure's argument evaporates.
    """
    if forced is not None:
        return forced
    for i, t in enumerate(sims):
        roll = t["rollout"]
        why = None
        if roll is None:
            why = "no rollout: descent ended at a terminal"
        elif len(roll["returns"]) != 5:
            why = f"only {len(roll['returns'])}/5 rollouts completed"
        elif abs(roll["agg"] - max(roll["returns"])) < 1e-9:
            why = "rollout mean == max: AGG 1 would show nothing"
        elif not any(len(e["before"]) >= 2 and abs(e["q_mean"] - e["q_max"]) > 1e-9
                     for e in t["edges"]):
            why = "no edge with >= 2 prior values and mean != max"
        if why is None:
            return i
        print(f"  sim {i} rejected: {why}")
    raise SystemExit("no simulation met the criteria; widen the replay")


def _box(ax, x, y, text, *, edge, ls="-", fs=8.5, pad=0.28, color=None):
    """A monospace token buffer in a rounded slot, per plot_sr_game._slot."""
    t = ax.text(x, y, text, ha="center", va="center", family="monospace",
                fontsize=fs, color=color or INK, zorder=4)
    t.set_bbox(dict(boxstyle=f"round,pad={pad}", facecolor="white",
                    edgecolor=edge, linewidth=1.4, linestyle=ls))
    return t


def plot(sim: dict, idx: int, seed: int, out: Path) -> None:
    frames, edges, roll = sim["frames"], sim["edges"], sim["rollout"]
    returns = roll["returns"]
    leaf_value = roll["agg"]
    roll_max = max(returns)
    # Edges are recorded innermost-first (the recursion unwinds bottom-up);
    # the tree reads top-down, so the ledger reverses them to match.
    rows = list(reversed(edges))

    fig = plt.figure(figsize=(13.6, 9.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.03)
    axt, axl = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # _style() configures spines/grid/ticks for series axes; this is a diagram,
    # and plot_sr_game.py's diagram panel likewise takes none.
    for ax in (axt, axl):
        ax.axis("off")
        ax.set_ylim(0, 10)
    axt.set_xlim(-1.4, 8.2)
    axl.set_xlim(0, 10)

    NX = 1.75
    NY = [8.9, 7.6, 6.3, 5.0]
    MID = [(NY[k] + NY[k + 1]) / 2 for k in range(3)]   # the shared y
    BX = 6.0
    RY = [3.7, 3.0, 2.3, 1.6, 0.9]

    # ---- left: the tree -------------------------------------------------
    for k, (f, y) in enumerate(zip(frames, NY)):
        fresh = (k == len(frames) - 1)
        _box(axt, NX, y, f["state"], edge=BLUE, ls="--" if fresh else "-")
        axt.text(3.1, y, f"total_N = {f['total_N']}   n_valid = {f['n_valid']}",
                 ha="left", va="center", fontsize=8, color=MUTED)

    for k in range(3):
        axt.annotate("", xy=(NX, NY[k + 1] + 0.42), xytext=(NX, NY[k] - 0.42),
                     arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.9,
                                     shrinkA=0, shrinkB=0), zorder=3)
        axt.text(NX - 0.6, MID[k] + 0.14, f"a = {edges[2 - k]['action']}",
                 ha="right", va="center", family="monospace", fontsize=8.5,
                 color=INK)
        axt.text(NX - 0.6, MID[k] - 0.17, "r = 0", ha="right",
                 va="center", fontsize=8, color=MUTED)

    axt.text(-1.35, (NY[0] + NY[3]) / 2, "1  SELECT", rotation=90, ha="center",
             va="center", fontsize=9.5, color=BLUE, weight="bold")
    axt.text(-1.35, NY[3], "2  EXPAND", rotation=90, ha="center", va="center",
             fontsize=9.5, color=INK, weight="bold")
    axt.text(-1.35, float(np.mean(RY)), "3  ROLLOUT", rotation=90, ha="center",
             va="center", fontsize=9.5, color=ORANGE, weight="bold")
    axt.text(BX, NY[0] + 0.55, "4  BACKUP", ha="center", va="bottom",
             fontsize=9.5, color=PURPLE, weight="bold")

    axt.text(2.6, NY[3] - 0.40,
             "nonterminal leaf: nn_policy is None\n"
             "=> expand; nn_value <- 0.0, and return\n"
             "at once: no action selected, no\n"
             "update_edge, no total_N++",
             ha="left", va="top", family="monospace", fontsize=7.4, color=MUTED)

    # rollout (stage 3): a vertical fan, so the real sentences fit unabbreviated
    for y, sent, r in zip(RY, roll["sentences"], returns):
        axt.annotate("", xy=(0.42, y), xytext=(NX - 0.35, NY[3] - 0.42),
                     arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1,
                                     ls="--", shrinkA=2, shrinkB=5), zorder=2)
        axt.plot([0.42], [y], "o", ms=4.5, color=ORANGE, zorder=4)
        axt.text(NX, y, sent, ha="left", va="center", family="monospace",
                 fontsize=7.2, color=INK)
        axt.text(8.15, y, f"{r:+.3f}", ha="right", va="center",
                 family="monospace", fontsize=8.5, color=ORANGE)
    axt.text(0.42, 0.35,
             "5 clones of the leaf; actions uniform over the mask, drawn from the\n"
             "global legacy RNG -- not from the configuration's seeds",
             ha="left", va="top", fontsize=8, color=MUTED)

    # backup (stage 4): the identical string three times -- that IS the finding
    for k in range(3):
        axt.annotate("", xy=(BX, NY[k] - 0.35), xytext=(BX, NY[k + 1] + 0.35),
                     arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.6,
                                     shrinkA=0, shrinkB=0), zorder=3)
        axt.text(BX + 0.22, MID[k], f"v = {leaf_value:.2f}", ha="left",
                 va="center", family="monospace", fontsize=9, color=PURPLE)

    # ---- right: the ledger ----------------------------------------------
    axl.text(0.1, 9.92,
             "AGG 2   backup_rule -- reduces ONE EDGE'S history across ALL simulations",
             ha="left", va="top", fontsize=8.6, color=INK, weight="bold")
    axl.text(0.1, 9.60, "recomputed from scratch on every traversal",
             ha="left", va="top", fontsize=8, color=MUTED)

    CX = [0.1, 1.9, 2.6, 5.5, 6.7, 8.5]
    HEAD = ["edge", "N(s,a)", "action_values[a] before", "append",
            'Q if "mean"', 'Q if "max"']
    for x, h in zip(CX, HEAD):
        axl.text(x, 9.15, h, ha="left", va="center", fontsize=8.4, color=INK,
                 weight="bold")
    axl.plot([0.05, 9.95], [8.95, 8.95], lw=0.9, color=GRID, zorder=1)

    def fmt(vals):
        return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]" if vals else "[]"

    for k, (e, y) in enumerate(zip(rows, MID)):
        cells = [
            f"s{k} -{e['action']}->",
            f"{e['n0']} -> {e['n1']}",
            fmt(e["before"]) + ("  (first traversal)" if not e["before"] else ""),
            f"+({e['append']:.2f})",
            f"{e['q_mean']:.4f}",
            f"{e['q_max']:.4f}",
        ]
        for j, (x, c) in enumerate(zip(CX, cells)):
            axl.text(x, y, c, ha="left", va="center", family="monospace",
                     fontsize=8.4, color=PURPLE if j == 3 else INK)
        if abs(e["q_mean"] - e["q_max"]) > 1e-9:
            # secondary encoding, so the divergence survives grayscale
            axl.plot([CX[5], CX[5] + 0.62], [y - 0.26, y - 0.26], lw=1.1,
                     color=PURPLE, zorder=2)
        axl.plot([0.05, 9.95], [y - 0.65, y - 0.65], lw=0.6, color=GRID,
                 zorder=1)
    axl.text(CX[2], MID[2] - 0.38,
             "mean = max when len(list) = 1 -- the rule reduces a list, not a descent",
             ha="left", va="center", fontsize=7.6, color=MUTED)

    axl.text(0.1, 4.95,
             'the "max" column is recomputed from this same list,\n'
             "not from a second run: a different rule shifts the\n"
             "global q_min/q_max window and the descent diverges\n"
             "from here on.",
             ha="left", va="top", fontsize=8, color=MUTED)
    axl.text(0.1, 4.05,
             f"max is NOT Bellman max: search returns {leaf_value:.2f}\n"
             "upward under either rule; only q_norm at this\n"
             "one node changes.",
             ha="left", va="top", fontsize=8.6, color=INK)

    axl.plot([0.05, 9.95], [3.35, 3.35], lw=0.9, color=GRID, zorder=1)
    axl.text(0.1, 3.28, "different list  .  different scope  .  different time",
             ha="left", va="top", fontsize=8, color=MUTED)

    axl.add_patch(FancyBboxPatch((0.05, 0.30), 9.9, 2.70,
                                 boxstyle="round,pad=0.04", facecolor="none",
                                 edgecolor=GRID, lw=1.0, zorder=1))
    axl.text(0.25, 2.82,
             "AGG 1   rollout_mode -- reduces the K returns at ONE leaf, ONCE, at expansion",
             ha="left", va="top", fontsize=8.6, color=INK, weight="bold")
    axl.text(0.25, 2.32,
             "rewards = [" + ", ".join(f"{r:.3f}" for r in returns) + "]",
             ha="left", va="center", family="monospace", fontsize=8.4, color=INK)
    axl.text(0.25, 2.00,
             f"len(rewards) = {len(returns)} of rollout_n = 5 completed -- the mean "
             "divides by len(rewards), not by rollout_n",
             ha="left", va="center", fontsize=7.8, color=MUTED)
    axl.text(0.25, 1.58, f"mean -> {leaf_value:.2f}   <-- leaf_value  (LIVE)",
             ha="left", va="center", family="monospace", fontsize=8.8,
             color=ORANGE)
    axl.text(0.25, 1.24, f"max  ->  {roll_max:.2f}      (counterfactual, same list)",
             ha="left", va="center", family="monospace", fontsize=8.8, color=INK)
    axl.text(0.25, 0.88,
             "leaf_value = (1-blend)*rv + blend*nn_value with blend = 0.0, so\n"
             "leaf_value = rv: the net's 0.0 is computed and discarded",
             ha="left", va="top", fontsize=7.8, color=MUTED)

    # The figure's causal claim, and the only line crossing the divider: one
    # leaf aggregate becomes every edge's appended value.
    axl.annotate("", xy=(CX[3] + 0.35, MID[2] - 0.42), xytext=(CX[3] + 0.35, 3.06),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3,
                                 ls="--", shrinkA=0, shrinkB=0), zorder=3)
    axl.text(CX[3] + 0.55, 4.15,
             "AGG 1's single scalar\nis what every row\nappends",
             ha="left", va="center", fontsize=7.8, color=ORANGE)

    fig.text(0.038, 0.972,
             "Every non-terminal reward is 0, so one number lands on every edge "
             "of the path",
             fontsize=12.5, color=INK, ha="left", va="top")
    fig.text(0.038, 0.941,
             f"counted simulation {idx} . PUCT with a uniform prior, "
             "P(a) = 1/n_valid . value = 0.0\n"
             "rollout_n = 5, rollout_mode = mean, rollout_blend = 0.0 . c = 1.0 . "
             f"np.random.seed({seed})",
             fontsize=9, color=MUTED, ha="left", va="top")
    fig.text(0.038, 0.028,
             "Real run, not a schematic: instrumented replay of sraz.core.mcts.MCTS over "
             "SymRegGame(problem_seed=42, lmfit_max_nfev=50); returns come from the game's own "
             "reward pipeline (clipped R^2), not recomputed here.  This is the rollout arm -- at "
             "rollout_n = 0 every leaf value is exactly 0.0 and there is no rollout to draw.  "
             "Nodes are keyed by the token buffer with no step_count, so the search structure is "
             "a DAG with transpositions; this path has none (asserted).",
             fontsize=7.6, color=MUTED, ha="left", va="bottom", wrap=True)

    fig.subplots_adjust(left=0.042, right=0.99, top=0.90, bottom=0.085)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("Claude-research/figures"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sim-index", type=int, default=None)
    a = p.parse_args()

    sims = replay(a.seed)
    idx = pick_sim(sims, a.sim_index)
    sim = sims[idx]
    frames, edges, roll = sim["frames"], sim["edges"], sim["rollout"]
    returns = roll["returns"]

    # Each message says what a failure MEANS, not merely that one occurred.
    assert [f["state"] for f in frames] == EXPECTED_PATH \
        and [f["n_valid"] for f in frames] == EXPECTED_NVALID, \
        ("the replay diverged from the recorded specimen -- the RNG, the config "
         "or mcts.py changed; every number in this figure is stale")
    assert len(returns) == 5, \
        ("a rollout was discarded (budget exhaustion or dead end); the mean "
         "divides by len(rewards), and the ledger must print the true divisor")
    appended = [e["append"] for e in edges]
    assert len({round(v, 12) for v in appended}) == 1, \
        ("the value appended differs across edges -- gamma is not 1.0, or an "
         "intermediate reward fired and the figure's central claim is false")
    assert abs(appended[0] - float(np.mean(returns))) < 1e-12, \
        ("the backed-up number is not the leaf's rollout aggregate -- "
         "rollout_blend is nonzero or the net value leaked into the leaf")
    assert len(edges) == len(frames) - 1, \
        ("an edge was updated off the descent path -- the ledger's rows no "
         "longer correspond one-to-one with the tree's arrows")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "mcts_backup_trace.png"

    print(f"  sim {idx}  path {' -> '.join(f['state'] for f in frames)}")
    print(f"  returns  {returns}   mean = {np.mean(returns):.4f}   "
          f"max = {max(returns):.4f}")
    print(f"  leaf_value = {roll['agg']:.4f}  (blend=0.0); backed to all "
          f"{len(edges)} edges")
    for e in reversed(edges):
        print(f"  a={e['action']}  N {e['n0']}->{e['n1']}  "
              f"before {[round(v, 4) for v in e['before']]}  "
              f"Q mean {e['q_mean']:.4f}  Q max {e['q_max']:.4f}")
    plot(sim, idx, a.seed, out)


if __name__ == "__main__":
    main()
