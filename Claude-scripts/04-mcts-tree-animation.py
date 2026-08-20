#!/usr/bin/env python
"""Animate a pure-MCTS search on the sine task and watch it converge to C1*x.

The linear answer `* C1 x` is a *first-move* terminal (one production from the
root S), so "MCTS converges to the linear equation" is decided entirely within
the root's move-1 search. This script drives that one search ONE simulation at
a time and renders a frame per simulation, producing a GIF.

Each frame shows:
  * the depth-2 derivation tree (root S -> its 7 children -> their children).
    Depth 0/1 edges are the exact root statistics; edge width encodes visit
    count N, edge/node colour encodes the backed-up value estimate Q (the
    quantity mean-backup optimises). Depth-2 nodes are small dots (visits by
    size) so the tree shape stays legible.
  * the current simulation's UCB descent path, highlighted -- selection, then a
    star at the leaf it expands + the random-rollout value it backed up.
  * a bar panel of the 7 first-move options with their EXACT visit counts and Q
    (the root has a single nonterminal, so its action stats are complete, not a
    leftmost approximation). The most-visited bar = the current greedy move.

The drama: `* C1 x` (certain R^2=0.873) out-competes the additive `+ S S`
branch (whose TRUE optimum is 0.997 but whose random-rollout value looks like
~0.56), so visits pile onto the linear terminal and the search never pursues
the sine.

Fidelity: this is the real MCTS class driven exactly as one perform_simulations
call (q-normalisation stats and the shared rollout budget are reset once, then
accumulate across the T sims). Fits are memoised from the seed-42 cache so
rollouts are dict lookups.

Run from repo root:
    .venv/bin/python Claude-scripts/04-mcts-tree-animation.py --sims 64
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image

import sraz.instances.symreg.game as game_mod
from sraz.core.mcts import MCTS
from sraz.instances.symreg.config import SymRegConfig

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-5"
CACHE_DIR = OUT_DIR / "cache"

_spec = importlib.util.spec_from_file_location(
    "state_value_gap", Path(__file__).with_name("01-state-value-gap.py"))
svg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svg)

CMAP = matplotlib.colormaps["RdYlGn"]
NORM = Normalize(vmin=-1.0, vmax=1.0)          # the reward's clip range


# --------------------------------------------------------------------------
# fit memo (identical to 03) so rollouts are ~free and R^2 matches V*
# --------------------------------------------------------------------------
def install_fit_memo(cache_key: str) -> None:
    cpath = CACHE_DIR / f"fit_cache_{cache_key}.json"
    memo = json.loads(cpath.read_text()) if cpath.exists() else {}
    print(f"[memo] preloaded {len(memo)} fits")
    orig = game_mod.fit_expression

    def memoised(rule, xs, exact_ys, max_nfev=None):
        v = memo.get(rule)
        if v is None:
            v = orig(rule, xs, exact_ys, max_nfev=max_nfev)
            memo[rule] = v
        return v

    game_mod.fit_expression = memoised


# --------------------------------------------------------------------------
# depth-2 derivation tree (leftmost expansion), layout, and MCTS<->tree maps
# --------------------------------------------------------------------------
def build_tree(game, max_depth: int) -> list[dict]:
    g = game.grammar
    nodes: list[dict] = []

    def add(tokens, depth, parent):
        nid = len(nodes)
        term = svg.leftmost_nt(tokens, g.nonterms) == -1
        nodes.append({"id": nid, "tokens": tuple(tokens), "depth": depth,
                      "parent": parent, "children": [], "terminal": term})
        if parent is not None:
            nodes[parent]["children"].append(nid)
        if not term and depth < max_depth:
            i = svg.leftmost_nt(tokens, g.nonterms)
            for j in g.proddict[tokens[i]]:
                rhs = tuple(g.productions[j])
                if len(tokens) + len(rhs) - 1 < game.state_len:
                    add(tokens[:i] + rhs + tokens[i + 1:], depth + 1, nid)
        return nid

    add((g.symdict[g.start],), 0, None)
    return nodes


def layout(nodes: list[dict]) -> dict:
    pos, counter = {}, [0]

    def assign(nid):
        ch = nodes[nid]["children"]
        if not ch:
            x = float(counter[0]); counter[0] += 1
        else:
            x = float(np.mean([assign(c) for c in ch]))
        pos[nid] = (x, -float(nodes[nid]["depth"]))
        return x

    assign(0)
    return pos


def build_maps(root, nodes):
    """Reach every displayed node by stepping, matching MCTS's own obs.

    Returns:
        node_state : id -> hashable_obs (the MCTS node key for that state)
        edge_action: (parent_id, child_id) -> flat action int (leftmost expand)
    """
    node_game = {0: root}
    node_state = {0: root.hashable_obs}
    edge_action: dict[tuple, int] = {}
    expr = {n["id"]: svg.decode(n["tokens"], root.grammar.tokenlist) for n in nodes}
    for n in sorted(nodes, key=lambda n: n["depth"]):
        pid = n["id"]
        if n["terminal"] or not n["children"]:
            continue
        pg = node_game[pid]
        for a in np.flatnonzero(pg.get_action_mask()):
            cg = pg.clone()
            cg.step_wrapper(int(a))
            cstr = cg._decode_state()
            for cid in n["children"]:
                if cid not in node_game and expr[cid] == cstr:
                    node_game[cid] = cg
                    node_state[cid] = cg.hashable_obs
                    edge_action[(pid, cid)] = int(a)
                    break
    return node_state, edge_action


# --------------------------------------------------------------------------
# live MCTS helpers
# --------------------------------------------------------------------------
def _edge_stats(node):
    """Normalise a node's action_N / action_Q (keys are 1-tuples) to int->val."""
    N, Q = {}, {}
    if node is not None:
        for k, v in node.action_N.items():
            N[int(k[0]) if isinstance(k, tuple) else int(k)] = v
        for k, v in node.action_Q.items():
            Q[int(k[0]) if isinstance(k, tuple) else int(k)] = v
    return N, Q


def replay_selection(mcts, root):
    """The UCB descent path this simulation will take (pre-sim tree).

    Mirrors search()'s selection: argmax masked-UCB until reaching a state not
    yet in the tree (the expansion leaf) or a terminal. Returns the list of
    hashable_obs visited, root first.
    """
    g = root.clone()
    path = [g.hashable_obs]
    while True:
        node = mcts.nodes.get(g.hashable_obs)
        if node is None or node.is_terminal_state or node.nn_policy is None:
            break
        ucbs = mcts.calc_masked_ucbs(node, "")
        a = int(np.argmax(ucbs))
        g.step_wrapper(a)
        path.append(g.hashable_obs)
    return path


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render(mcts, root, nodes, pos, node_state, edge_action, root_children,
           expr, path, sim_idx, n_sims, label):
    obs_to_id = {st: nid for nid, st in node_state.items()}
    path_ids = [obs_to_id.get(st) for st in path]
    path_edges = {(a, b) for a, b in zip(path_ids, path_ids[1:])
                  if a is not None and b is not None}
    leaf_obs = path[-1]
    leaf_node = mcts.nodes.get(leaf_obs)
    leaf_val = (leaf_node.nn_value if leaf_node is not None else None)
    leaf_id = obs_to_id.get(leaf_obs)

    root_node = mcts.nodes.get(node_state[0])
    rN, rQ = _edge_stats(root_node)

    fig = plt.figure(figsize=(15, 9), dpi=100)
    gs = GridSpec(2, 1, height_ratios=[2.6, 1.0], hspace=0.16)
    axT = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ---- tree ----
    def xy(nid):
        x, d = pos[nid][0], nodes[nid]["depth"]
        return x, -d * 1.0

    maxN = max(rN.values()) if rN else 1
    for n in nodes:                                    # edges
        pid = n["parent"]
        if pid is None:
            continue
        cid = n["id"]
        a = edge_action.get((pid, cid))
        pnode = mcts.nodes.get(node_state[pid])
        pN, pQ = _edge_stats(pnode)
        eN = pN.get(a, 0)
        eQ = pQ.get(a, None)
        x0, y0 = xy(pid); x1, y1 = xy(cid)
        hot = (pid, cid) in path_edges
        col = CMAP(NORM(eQ)) if eQ is not None else "0.82"
        lw = 0.5 + 3.5 * (eN / maxN) if maxN else 0.5
        ax_z = 4 if hot else 1
        if hot:
            axT.plot([x0, x1], [y0, y1], "-", color="black", lw=lw + 3.5,
                     zorder=3, solid_capstyle="round")
        axT.plot([x0, x1], [y0, y1], "-", color=col, lw=max(lw, 0.6),
                 zorder=ax_z, solid_capstyle="round")

    # depth-2 nodes as dots (size = visits through them)
    for n in nodes:
        if n["depth"] != 2:
            continue
        cid = n["id"]
        st = node_state.get(cid)
        node = mcts.nodes.get(st)
        pid = n["parent"]
        a = edge_action.get((pid, cid))
        pN, pQ = _edge_stats(mcts.nodes.get(node_state[pid]))
        eN, eQ = pN.get(a, 0), pQ.get(a, None)
        x, y = xy(cid)
        size = 20 + 55 * (eN / maxN if maxN else 0)
        fc = CMAP(NORM(eQ)) if eQ is not None else "white"
        edge_c = "black" if cid == leaf_id else "0.5"
        axT.scatter([x], [y], s=size, c=[fc], edgecolors=edge_c,
                    linewidths=1.4 if cid == leaf_id else 0.5, zorder=5)

    # depth 0/1 as labelled boxes
    for nid in [0] + root_children:
        x, y = xy(nid)
        a = edge_action.get((0, nid))
        eN, eQ = ((rN.get(a, 0), rQ.get(a, None)) if nid != 0
                  else (root_node.total_N if root_node else 0, None))
        fc = CMAP(NORM(eQ)) if eQ is not None else "0.95"
        txt = expr[nid] if nid != 0 else "S"
        if nid != 0:
            txt += f"\nN={eN}" + (f"  Q={eQ:+.2f}" if eQ is not None else "")
        star = "  ★" if nid == leaf_id else ""
        axT.annotate(txt + star, (x, y), ha="center", va="center",
                     fontsize=7.5 if nid != 0 else 9, family="monospace",
                     zorder=6,
                     bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="black",
                               lw=1.2 if nid == leaf_id else 0.5))
    xs = [xy(n["id"])[0] for n in nodes]
    axT.set_xlim(min(xs) - 1, max(xs) + 1)
    axT.set_ylim(-2.6, 0.7)
    axT.axis("off")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
    fig.colorbar(sm, ax=axT, label="backed-up value Q (what mean-backup optimises)",
                 shrink=0.55, pad=0.005)

    greedy = (max(root_children, key=lambda c: rN.get(edge_action.get((0, c)), 0))
              if root_node and rN else None)
    gtxt = expr[greedy] if greedy is not None else "(none yet)"
    rvtxt = (f"leaf rollout R² = {leaf_val:+.3f}"
             if leaf_val is not None else "leaf: net value")
    axT.set_title(
        f"Pure MCTS · {label} · first-move search · "
        f"simulation {sim_idx}/{n_sims}\n"
        f"greedy first move so far:  {gtxt}     |     this sim: {rvtxt}",
        fontsize=11)

    # ---- bar panel: exact first-move visit counts ----
    order = list(root_children)
    ys = np.arange(len(order))[::-1]
    Ns = [rN.get(edge_action.get((0, c)), 0) for c in order]
    Qs = [rQ.get(edge_action.get((0, c)), None) for c in order]
    cols = [CMAP(NORM(q)) if q is not None else "0.85" for q in Qs]
    axB.barh(ys, Ns, color=cols, edgecolor="black", linewidth=0.6)
    for y, c, nvis, q in zip(ys, order, Ns, Qs):
        lab = expr[c] + (f"   Q={q:+.2f}" if q is not None else "")
        axB.text(-0.02 * (max(Ns) + 1), y, lab, ha="right", va="center",
                 fontsize=8, family="monospace")
        if nvis:
            axB.text(nvis + 0.01 * (max(Ns) + 1), y, str(nvis), ha="left",
                     va="center", fontsize=8)
        if greedy is not None and c == greedy:
            axB.barh([y], [nvis], color="none", edgecolor="crimson",
                     linewidth=2.2)
    axB.set_xlim(0, max(max(Ns) + 1, 2))
    axB.set_ylim(-0.7, len(order) - 0.3)
    axB.set_yticks([])
    axB.set_xlabel("visit count N of each first-move option (exact); "
                   "red outline = current greedy move")
    axB.spines[["top", "right", "left"]].set_visible(False)

    fig.subplots_adjust(left=0.28, right=0.98, top=0.9, bottom=0.08)
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
    ap.add_argument("--sims", type=int, default=64, help="simulations to animate")
    ap.add_argument("--seed", type=int, default=42, help="MCTS rollout RNG seed")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--c-exploration", type=float, default=None,
                    help="override the config's exploration constant")
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
    args = ap.parse_args()

    cache_key = (f"{args.problem}_seed{args.problem_seed}"
                 if args.problem == "sine" else args.problem)
    install_fit_memo(cache_key)

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
    print(f"[mcts] params: {params}")

    root = svg.build_game(args.problem, args.problem_seed)
    root.reset_wrapper()
    print(f"target: {root.target_infix}  constants: "
          + str({k: round(v, 3) for k, v in root.constants.items()}))

    # The title used to hard-code "sine"; a family run needs its own label, and
    # the sine label must come out byte-identical so existing GIFs still match.
    label = (f"sine (seed {args.problem_seed})" if args.problem == "sine"
             else f"{args.problem}   y = {root.target_infix}")

    nodes = build_tree(root, 2)
    pos = layout(nodes)
    node_state, edge_action = build_maps(root.clone(), nodes)
    root_children = nodes[0]["children"]
    expr = {n["id"]: svg.decode(n["tokens"], root.grammar.tokenlist) for n in nodes}
    print(f"[tree] {len(nodes)} nodes; {len(root_children)} first-move options")

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
    # drive exactly one perform_simulations of `sims` searches: reset the shared
    # rollout budget and q-normalisation stats once, then accumulate.
    mcts.q_min, mcts.q_max = float("inf"), float("-inf")
    mcts._search_rollout_budget = mcts.rollout_budget

    frames = []
    for t in range(args.sims):
        path = replay_selection(mcts, root)          # this sim's descent
        old = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(old)
        frames.append(render(mcts, root, nodes, pos, node_state, edge_action,
                             root_children, expr, path, t + 1, args.sims,
                             label))

    # report final root distribution
    rN, rQ = _edge_stats(mcts.nodes.get(node_state[0]))
    greedy = max(root_children, key=lambda c: rN.get(edge_action.get((0, c)), 0))
    print("\nfinal first-move visit counts:")
    for c in sorted(root_children,
                    key=lambda c: -rN.get(edge_action.get((0, c)), 0)):
        a = edge_action.get((0, c))
        print(f"  N={rN.get(a,0):>3}  Q={rQ.get(a, float('nan')):+.3f}  {expr[c]}")
    print(f"=> greedy first move: {expr[greedy]}")

    out_dir = (Path(args.out_dir) if args.out_dir else OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctag = f"_c{mcts.c_exploration:g}".replace(".", "p")
    rtag = "" if args.rollout_budget is None else f"_rb{args.rollout_budget}"
    # rollout_n changes what one simulation COSTS (n terminal evaluations), so it
    # belongs in the name. Empty when unset, keeping existing sine names valid.
    ntag = "" if args.rollout_n is None else f"_rn{args.rollout_n}"
    gif = out_dir / f"mcts-tree_{cache_key}{ctag}{rtag}{ntag}_s{args.sims}.gif"
    hold = [frames[-1]] * int(args.fps * 2)          # linger on the final frame
    (frames + hold)[0].save(
        gif, save_all=True, append_images=(frames + hold)[1:],
        duration=int(1000 / args.fps), loop=0)
    print(f"[saved] {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
