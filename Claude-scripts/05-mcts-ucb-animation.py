#!/usr/bin/env python
"""Animate the root UCB terms of a pure-MCTS search.

Works on sine (the default, 7 first-move actions) and on any named family
target (ADDITIVE_GRAMMAR at L=12, 4 first-move actions) -- see --problem.

Companion to 04-mcts-tree-animation.py. Same search, same fidelity, but the
focus moves from "which node gets expanded" to "why that node got picked":
every frame decomposes the selection score of every first-move action into
its two summands, exactly as calc_masked_ucbs computes them (mcts.py:479):

    UCB(a) = Qtilde(a)  +  c * P(a) * sqrt(N_tot + EPS) / (1 + N(a))
             ^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             exploit        explore

with the min-max normalisation

    Qtilde(a) = 0                                   if nothing backed up yet
              = 0.5                                 if q_max == q_min
              = (Q(a) - q_min) / (q_max - q_min)     otherwise

and q_min / q_max taken over the WHOLE tree, reset once per move.

Each frame shows the tree and the UCB numbers as they stand *before* the
simulation runs -- i.e. the inputs to that simulation's decision -- then names
the action the argmax therefore selects and the leaf value the simulation
brought back. So a frame is one complete decision, cause and effect.

Panels:
  top          depth-2 derivation tree, current descent path highlighted
  bottom-left  stacked bars: Qtilde (exploit) + u (explore) = UCB, per action
  bottom-right UCB per action vs simulation index, with a sweeping cursor --
               the crossover where the certain terminal `* C1 x` overtakes the
               decaying `+ S S` is a visible line crossing

Fidelity: the real MCTS class, driven one search() at a time. The root is
expanded first (as perform_simulations does at mcts.py:130-136) so frame 1
already has UCB numbers, and q-normalisation stats plus the shared rollout
budget are reset once and then accumulate -- faithful to one perform_simulations
of `--sims` simulations. Axis limits are computed in a first pass so nothing
rescales mid-animation. Fits are memoised from the seed-42 cache.

NOTE ON WHICH ALGORITHM THIS IS. The rule above is AlphaZero-style PUCT, the
searcher this repo ships. It is NOT the writeup's eq. (17) Mean-UCT, which uses
Qbar + sqrt(2 log N(s) / N(s,a)) on raw rewards and gives unvisited actions an
infinite score. Three differences matter: sqrt(N) vs sqrt(log N); `1 + N(a)` in
the denominator, so an unvisited action is NOT forced and can be skipped
indefinitely; and the min-max normalisation of Q, which silently rescales the
constant the writeup holds fixed. Conclusions drawn from these animations
describe the shipped searcher, not the writeup's algorithm.

Run from repo root:
    .venv/bin/python Claude-scripts/05-mcts-ucb-animation.py --sims 40 --fps 1.5
    .venv/bin/python Claude-scripts/05-mcts-ucb-animation.py \
        --problem lin_A --sims 64 --rollout-n 1 --rollout-budget 10000000
"""

from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image

from sraz.core.mcts import MCTS, EPS
from sraz.instances.symreg.config import SymRegConfig

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-5"


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(
        fname.replace("-", "_").removesuffix(".py"), Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse 04's tree/layout/mapping helpers rather than re-deriving them
anim = _load("04-mcts-tree-animation.py")
svg = anim.svg

CMAP = matplotlib.colormaps["RdYlGn"]
NORM = Normalize(vmin=-1.0, vmax=1.0)
LINE_COLORS = matplotlib.colormaps["tab10"].colors
C_EXPLOIT = "#3d6fa5"
C_EXPLORE = "#e0a03c"


# --------------------------------------------------------------------------
# The UCB decomposition -- a faithful split of calc_masked_ucbs
# --------------------------------------------------------------------------
def root_terms(mcts: MCTS, root_obs) -> dict | None:
    node = mcts.nodes.get(root_obs)
    if node is None or node.nn_policy is None:
        return None
    P = np.asarray(node.nn_policy, dtype=float)
    Q = np.zeros_like(P)
    N = np.zeros_like(P)
    for k, v in node.action_Q.items():
        Q[int(k[0]) if isinstance(k, tuple) else int(k)] = v
    for k, v in node.action_N.items():
        N[int(k[0]) if isinstance(k, tuple) else int(k)] = v

    qmin, qmax = mcts.q_min, mcts.q_max
    if qmin == float("inf") or qmax == float("-inf"):
        qt = np.zeros_like(Q)
        regime = "cold start: nothing backed up yet  ->  Qtilde = 0 for all"
    elif qmax > qmin:
        qt = (Q - qmin) / (qmax - qmin)
        regime = (f"Qtilde = (Q - q_min) / (q_max - q_min)"
                  f"  =  (Q {qmin:+.3f}) / {qmax - qmin:.3f}")
    else:
        qt = np.full_like(Q, 0.5)
        regime = f"q_max == q_min == {qmin:+.3f}  ->  Qtilde = 0.5 for all"

    u = mcts.c_exploration * P * np.sqrt(node.total_N + EPS) / (1 + N)
    return {"qt": qt, "u": u, "ucb": qt + u, "Q": Q, "N": N, "P": P,
            "qmin": qmin, "qmax": qmax, "total_N": node.total_N,
            "regime": regime}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(snap, ctx, lim, n_sims, seed) -> Image.Image:
    nodes, pos = ctx["nodes"], ctx["pos"]
    node_state, edge_action = ctx["node_state"], ctx["edge_action"]
    root_children, expr = ctx["root_children"], ctx["expr"]
    acts, lab, obs_to_id = ctx["acts"], ctx["lab"], ctx["obs_to_id"]

    T, est = snap["terms"], snap["estats"]
    path_ids = [obs_to_id.get(o) for o in snap["path"]]
    path_edges = {(a, b) for a, b in zip(path_ids, path_ids[1:])
                  if a is not None and b is not None}
    leaf_id = obs_to_id.get(snap["leaf_obs"])
    pick = acts[int(np.argmax([T["ucb"][a] for a in acts]))]

    fig = plt.figure(figsize=(16, 10), dpi=ctx["dpi"])
    gs = GridSpec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.05, 1.0],
                  hspace=0.30, wspace=0.20,
                  left=0.055, right=0.985, top=0.885, bottom=0.075)
    axT = fig.add_subplot(gs[0, :])
    axU = fig.add_subplot(gs[1, 0])
    axH = fig.add_subplot(gs[1, 1])

    # ---------------- top: the depth-2 tree ----------------
    def xy(nid):
        return pos[nid][0], -float(nodes[nid]["depth"])

    maxN = max(lim["maxN"], 1)
    for n in nodes:
        pid = n["parent"]
        if pid is None:
            continue
        cid = n["id"]
        eN, eQ = est.get((pid, cid), (0, None))
        x0, y0 = xy(pid)
        x1, y1 = xy(cid)
        hot = (pid, cid) in path_edges
        col = CMAP(NORM(eQ)) if eQ is not None else "0.84"
        lw = 0.6 + 3.4 * (eN / maxN)
        if hot:
            axT.plot([x0, x1], [y0, y1], "-", color="black", lw=lw + 3.2,
                     zorder=3, solid_capstyle="round")
        axT.plot([x0, x1], [y0, y1], "-", color=col, lw=max(lw, 0.6),
                 zorder=4 if hot else 1, solid_capstyle="round")

    for n in nodes:
        if n["depth"] != 2:
            continue
        cid, pid = n["id"], n["parent"]
        eN, eQ = est.get((pid, cid), (0, None))
        x, y = xy(cid)
        axT.scatter([x], [y], s=18 + 52 * (eN / maxN),
                    c=[CMAP(NORM(eQ)) if eQ is not None else "white"],
                    edgecolors="black" if cid == leaf_id else "0.55",
                    linewidths=1.5 if cid == leaf_id else 0.5, zorder=5)

    for nid in [0] + root_children:
        x, y = xy(nid)
        if nid == 0:
            txt, fc = "S", "0.95"
        else:
            eN, eQ = est.get((0, nid), (0, None))
            txt = expr[nid] + f"\nN={eN}" + (f"  Q={eQ:+.2f}" if eQ is not None else "")
            fc = CMAP(NORM(eQ)) if eQ is not None else "0.95"
        axT.annotate(txt + ("  *" if nid == leaf_id else ""), (x, y),
                     ha="center", va="center", zorder=6,
                     fontsize=9 if nid == 0 else 7.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="black",
                               lw=1.4 if nid == leaf_id else 0.5))

    xs = [xy(n["id"])[0] for n in nodes]
    axT.set_xlim(min(xs) - 1, max(xs) + 1)
    axT.set_ylim(-2.5, 0.6)
    axT.axis("off")

    lv = snap["leaf_val"]
    lvtxt = f"leaf value backed up = {lv:+.3f}" if lv is not None else "terminal (exact)"
    fig.suptitle(
        f"Pure MCTS root selection  |  {ctx['label']}  |  "
        f"c = {ctx['c']:g}  |  "
        f"simulation {snap['t']}/{n_sims}   (state shown = before this simulation)\n"
        f"argmax UCB selects:  {lab[pick]}          {lvtxt}          "
        f"greedy move so far:  {snap['greedy']}",
        fontsize=11.5)

    # ---------------- bottom-left: the UCB decomposition ----------------
    ys = np.arange(len(acts))[::-1]
    for a, y in zip(acts, ys):
        qt, u = T["qt"][a], T["u"][a]
        axU.barh(y, qt, height=0.60, color=C_EXPLOIT, edgecolor="black", lw=0.5,
                 zorder=2)
        axU.barh(y, u, left=qt, height=0.60, color=C_EXPLORE, edgecolor="black",
                 lw=0.5, hatch="///", zorder=2)
        axU.plot([qt + u], [y], "D", color="black", ms=4.5, zorder=4)
        if a == pick:
            axU.barh(y, qt + u, height=0.60, color="none", edgecolor="crimson",
                     lw=2.4, zorder=5)
            axU.annotate("selected", (qt + u, y), xytext=(10, 0),
                         textcoords="offset points", va="center",
                         fontsize=8.5, color="crimson", fontweight="bold")
        axU.text(lim["ulo"] + 0.01 * (lim["uhi"] - lim["ulo"]), y + 0.30,
                 f"{lab[a]}", fontsize=8, family="monospace", va="bottom")
        axU.text(lim["ulo"] + 0.01 * (lim["uhi"] - lim["ulo"]), y - 0.32,
                 f"   N={int(T['N'][a])}  Q={T['Q'][a]:+.3f}  "
                 f"Qtilde={qt:+.3f}  u={u:.3f}  UCB={qt+u:+.3f}",
                 fontsize=7, family="monospace", va="top", color="0.35")

    axU.axvline(0, color="0.4", lw=0.8, zorder=1)
    axU.set_xlim(lim["ulo"], lim["uhi"])
    axU.set_ylim(-0.8, len(acts) - 0.2)
    axU.set_yticks([])
    axU.set_xlabel("UCB score")
    axU.set_title(
        f"UCB(a) = Qtilde(a) + c*P(a)*sqrt(N_tot+EPS)/(1+N(a))   "
        f"[c={ctx['c']:g}, P=1/{len(acts)}, N_tot={T['total_N']}]\n"
        f"{T['regime']}", fontsize=9, family="monospace")
    axU.bar(0, 0, color=C_EXPLOIT, label="Qtilde(a)  -- exploit (normalised Q)")
    axU.bar(0, 0, color=C_EXPLORE, hatch="///", label="u(a)  -- explore bonus")
    axU.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # ---------------- bottom-right: UCB history ----------------
    hist = snap["hist"]                      # list of per-action ucb arrays
    xs_h = np.arange(1, len(hist) + 1)
    for i, a in enumerate(acts):
        axH.plot(xs_h, [h[a] for h in hist], "-", lw=1.7,
                 color=LINE_COLORS[i % 10], label=lab[a], zorder=2)
    picks = snap["picks"]
    for i, a in enumerate(acts):
        sel = [x for x, p in zip(xs_h, picks) if p == a]
        if sel:
            axH.plot(sel, [hist[x - 1][a] for x in sel], "o", ms=4.0,
                     color=LINE_COLORS[i % 10], mec="black", mew=0.6, zorder=3)
    axH.axvline(snap["t"], color="0.35", ls="--", lw=1.1, zorder=1)
    axH.axhline(0, color="0.75", lw=0.7, zorder=0)
    axH.set_xlim(0.5, n_sims + 0.5)
    axH.set_ylim(lim["hlo"], lim["hhi"])
    axH.set_xlabel("simulation index")
    axH.set_ylabel("UCB(a) at the root")
    axH.set_title("UCB per first-move action over the search\n"
                  "(filled dot = this action was the argmax that simulation)",
                  fontsize=9)
    axH.grid(alpha=0.25)
    axH.legend(fontsize=7, loc="upper right", framealpha=0.95, ncol=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="sine")
    ap.add_argument("--problem-seed", type=int, default=42)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42, help="MCTS rollout RNG seed")
    ap.add_argument("--fps", type=float, default=1.5,
                    help="frames per second (default 1.5 = ~0.67s per frame)")
    ap.add_argument("--c-exploration", type=float, default=None,
                    help="override c in the UCB rule (config default is 1.0)")
    ap.add_argument("--rollout-budget", type=int, default=None,
                    help="override the shared per-move rollout step budget "
                         "(MCTS default 500 starves after ~6 leaf evaluations)")
    ap.add_argument("--rollout-n", type=int, default=None,
                    help="random completions per leaf evaluation. Left unset "
                         "the pure_mcts config default of 20 applies, which is "
                         "what the existing sine GIFs used. Pass 1 to make one "
                         "simulation cost exactly one terminal evaluation, as "
                         "the writeup's evaluation protocol requires.")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: the 8-5 experiment folder)")
    ap.add_argument("--dpi", type=int, default=88,
                    help="figure dpi; lower it for long animations to cap GIF size")
    args = ap.parse_args()

    cache_key = (f"{args.problem}_seed{args.problem_seed}"
                 if args.problem == "sine" else args.problem)
    anim.install_fit_memo(cache_key)

    # `problems.PROBLEMS` only knows "sine" and "additive_quadratic", so a
    # family target name cannot be the config's `problem` -- get_problem would
    # raise. SymRegConfig.build() already handles "named target + problem
    # supplies the grammar" (config.py:96-100); it just has to be fed that way.
    if args.problem == "sine":
        cfg = SymRegConfig(problem="sine", pure_mcts=True)
    else:
        cfg = SymRegConfig(problem="additive_quadratic", pure_mcts=True)
        cfg.game.kwargs["target"] = args.problem
        cfg.game.kwargs["max_len"] = 12
    cfg.game.kwargs["problem_seed"] = args.problem_seed
    cfg.game.kwargs["redraw_constants"] = False
    _, net, agent, _ = cfg.build()
    params = dict(agent.mcts_params)

    root = svg.build_game(args.problem, args.problem_seed)
    root.reset_wrapper()
    print(f"target: {root.target_infix}")

    nodes = anim.build_tree(root, 2)
    pos = anim.layout(nodes)
    node_state, edge_action = anim.build_maps(root.clone(), nodes)
    root_children = nodes[0]["children"]
    expr = {n["id"]: svg.decode(n["tokens"], root.grammar.tokenlist) for n in nodes}
    acts = sorted(edge_action[(0, c)] for c in root_children)
    child_of = {edge_action[(0, c)]: c for c in root_children}
    lab = {a: expr[child_of[a]] for a in acts}

    overrides = {"n_simulations": 1, "temperature": 0.01}
    if args.c_exploration is not None:
        overrides["c_exploration"] = float(args.c_exploration)
    if args.rollout_budget is not None:
        overrides["rollout_budget"] = int(args.rollout_budget)
    if args.rollout_n is not None:
        overrides["rollout_n"] = int(args.rollout_n)
    mcts = MCTS(root, net, rng=np.random.default_rng(args.seed),
                **{**params, **overrides})
    print(f"[mcts] c_exploration={mcts.c_exploration:g}  "
          f"rollout_n={mcts.rollout_n}  rollout_budget={mcts.rollout_budget}")

    # Instrument leaf evaluation so the report can distinguish real rollout
    # estimates from budget-exhausted fallbacks to the net's constant value.
    tally = {"rollout": 0, "fallback": 0}
    _orig_rv = mcts._rollout_value

    def _counted(msg):
        v = _orig_rv(msg)
        tally["rollout" if v is not None else "fallback"] += 1
        return v

    mcts._rollout_value = _counted
    mcts.q_min, mcts.q_max = float("inf"), float("-inf")
    mcts._search_rollout_budget = mcts.rollout_budget
    root_obs = root.hashable_obs

    # Expand the root first, exactly as perform_simulations does, so that
    # simulation 1 already has UCB numbers to show.
    old = mcts.game.stash_state()
    mcts.search("")
    mcts.game = mcts.game.unstash_state(old)
    print(f"[root] expanded; leaf value {mcts.nodes[root_obs].nn_value:+.4f}")

    # ---- pass 1: run the search, snapshotting the pre-simulation state ----
    parents = sorted({p for p, _ in edge_action})
    snaps, hist, picks = [], [], []
    for t in range(args.sims):
        T = root_terms(mcts, root_obs)
        assert T is not None, "root must be expanded before the loop"
        # self-check: the split must reproduce the real selection scores
        real = mcts.calc_masked_ucbs(mcts.nodes[root_obs], "")
        assert np.allclose([T["ucb"][a] for a in acts], [real[a] for a in acts]), \
            "UCB decomposition does not match calc_masked_ucbs"

        pick = acts[int(np.argmax([T["ucb"][a] for a in acts]))]
        hist.append({a: T["ucb"][a] for a in acts})
        picks.append(pick)

        estats = {}
        for pid in parents:
            pN, pQ = anim._edge_stats(mcts.nodes.get(node_state.get(pid)))
            for (p, cid), a in edge_action.items():
                if p == pid:
                    estats[(pid, cid)] = (pN.get(a, 0), pQ.get(a, None))

        path = anim.replay_selection(mcts, root)
        gN = {a: T["N"][a] for a in acts}
        greedy = lab[max(acts, key=lambda a: gN[a])] if max(gN.values()) > 0 else "(none yet)"

        old = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(old)

        ln = mcts.nodes.get(path[-1])
        snaps.append({"t": t + 1, "terms": T, "estats": estats, "path": path,
                      "leaf_obs": path[-1], "greedy": greedy,
                      "leaf_val": (None if ln is None or ln.is_terminal_state
                                   else ln.nn_value)})

    # attach the running history (each frame sees only its own prefix)
    for i, s in enumerate(snaps):
        s["hist"] = hist[:i + 1]
        s["picks"] = picks[:i + 1]

    # ---- fixed axis limits so nothing rescales mid-animation ----
    allu = np.array([[s["terms"]["ucb"][a] for a in acts] for s in snaps])
    allq = np.array([[s["terms"]["qt"][a] for a in acts] for s in snaps])
    span = max(allu.max() - min(0.0, allq.min()), 0.5)
    lim = {
        "ulo": min(0.0, allq.min()) - 0.34 * span,   # room for the text labels
        "uhi": allu.max() + 0.10 * span,
        "hlo": allu.min() - 0.06 * (allu.max() - allu.min() + 1e-9),
        "hhi": allu.max() + 0.16 * (allu.max() - allu.min() + 1e-9),
        "maxN": max(1, max(n for s in snaps for n, _ in s["estats"].values())),
    }

    # The title used to hard-code "sine"; a family run needs its own label, and
    # the sine label must come out byte-identical so existing GIFs still match.
    label = (f"sine (seed {args.problem_seed})" if args.problem == "sine"
             else f"{args.problem}   y = {root.target_infix}")
    ctx = {"label": label,
           "nodes": nodes, "pos": pos, "node_state": node_state,
           "edge_action": edge_action, "root_children": root_children,
           "expr": expr, "acts": acts, "lab": lab,
           "obs_to_id": {st: nid for nid, st in node_state.items()},
           "c": mcts.c_exploration, "dpi": args.dpi}

    # ---- pass 2: render ----
    frames = [render(s, ctx, lim, args.sims, args.problem_seed) for s in snaps]

    # ---- report ----
    T = root_terms(mcts, root_obs)
    print("\nfinal root state:")
    for a in sorted(acts, key=lambda a: -T["N"][a]):
        print(f"  N={int(T['N'][a]):>3}  Q={T['Q'][a]:+.3f}  "
              f"Qtilde={T['qt'][a]:+.3f}  u={T['u'][a]:.3f}  "
              f"UCB={T['ucb'][a]:+.3f}   {lab[a]}")
    print(f"  q_min={T['qmin']:+.4f}  q_max={T['qmax']:+.4f}")
    print(f"  rollout budget: {mcts._search_rollout_budget}/{mcts.rollout_budget} "
          f"steps left  |  leaf evals: {tally['rollout']} rollout-estimated, "
          f"{tally['fallback']} starved (fell back to nn_value)")
    print(f"=> greedy first move: {lab[max(acts, key=lambda a: T['N'][a])]}")
    first_lin = next((i + 1 for i, p in enumerate(picks)
                      if lab[p].strip() == "* C1 x"), None)
    print(f"=> `* C1 x` first selected at simulation {first_lin}")

    out_dir = (Path(args.out_dir) if args.out_dir else OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctag = f"_c{mcts.c_exploration:g}".replace(".", "p")
    rtag = "" if args.rollout_budget is None else f"_rb{args.rollout_budget}"
    # rollout_n changes what one simulation COSTS (n terminal evaluations), so it
    # belongs in the name. Empty when unset, keeping existing sine names valid.
    ntag = "" if args.rollout_n is None else f"_rn{args.rollout_n}"
    gif = out_dir / f"mcts-ucb_{cache_key}{ctag}{rtag}{ntag}_s{args.sims}.gif"
    hold = [frames[-1]] * max(1, int(round(args.fps * 3)))
    seq = frames + hold
    seq[0].save(gif, save_all=True, append_images=seq[1:],
                duration=int(round(1000 / args.fps)), loop=0, optimize=True)
    print(f"[saved] {gif.relative_to(REPO)}  ({len(frames)} frames, "
          f"{args.fps} fps = {int(round(1000/args.fps))} ms/frame)")


if __name__ == "__main__":
    main()
