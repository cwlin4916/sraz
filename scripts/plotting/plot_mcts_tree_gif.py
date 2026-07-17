"""Animate one MCTS search as the simulation budget grows: 1, 2, ..., N sims.

Runs a *single* search from the root of --problem, snapshotting the tree after
every simulation, then renders each snapshot as a GIF frame. The layout is
computed once from the final tree, so nodes keep fixed positions and the eye
tracks visit counts rather than moving boxes.

The right-hand panel shows the root's action statistics (N, Q, prior P) with the
current argmax-visits action highlighted -- this is where the "first action from
the root" flip is visible, together with the Q values that drive it.

Run from repo root:
    python scripts/plotting/plot_mcts_tree_gif.py --problem additive_quadratic --max-sims 15
"""
from sraz.utils import disable_numpy_multithreading

disable_numpy_multithreading()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sraz.core.mcts import MCTS  # noqa: E402
from sraz.core.policy_value_net import UniformPolicyValueNet  # noqa: E402
from sraz.instances.symreg.game import SymRegGame  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402

from plot_mcts_tree import child_tokens, pad_key, reconstruct  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_snapshots(mcts, game, root_tokens, max_sims, max_depth, min_visits):
    """Run max_sims simulations one at a time, reconstructing the tree after each."""
    mystate = game.hashable_obs
    mcts.q_min, mcts.q_max = float("inf"), float("-inf")
    mcts._search_rollout_budget = mcts.rollout_budget
    if mystate not in mcts.nodes:  # root expansion, as perform_simulations() does
        old = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(old)

    snaps = []
    for _ in range(max_sims):
        old = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(old)
        assert mystate == mcts.game.hashable_obs
        nodes, edges = reconstruct(mcts, game, root_tokens, max_depth, min_visits)
        root = mcts.nodes[pad_key(root_tokens, game)]
        acts = [(a, n, root.action_Q.get(a, 0.0), float(root.nn_policy.ravel()[a]))
                for a, n in root.action_N.items()]
        snaps.append({"nodes": nodes, "edges": edges, "root_actions": acts})
    return snaps


def global_layout(snaps):
    """Layout from the final (largest) tree; earlier frames reuse those positions.

    Trees only grow, so the last snapshot's node set is a superset of every
    earlier one. Keying by token tuple keeps a node pinned across frames.
    """
    nodes, edges = snaps[-1]["nodes"], snaps[-1]["edges"]
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        children[e[0]].append(e[1])
    pos, counter = {}, [0]

    def assign(nid):
        ch = children[nid]
        if not ch:
            x = float(counter[0])
            counter[0] += 1
        else:
            x = float(np.mean([assign(c) for c in ch]))
        pos[nid] = (x, -float(nodes[nid]["depth"]))
        return x

    assign(0)
    return {nodes[nid]["tokens"]: xy for nid, xy in pos.items()}


def expr_of(tokens, game):
    return " ".join(game.grammar.tokenlist[t] for t in tokens)


def draw_frame(game, snap, xy, out_path, title, maxN, sim, max_sims):
    nodes, edges, acts = snap["nodes"], snap["edges"], snap["root_actions"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(18, 8),
                                 gridspec_kw={"width_ratios": [3, 1]})
    best = max(acts, key=lambda t: (t[1], t[2]))[0] if acts else None

    for (p, c, n, q, prior) in edges:
        x0, y0 = xy[nodes[p]["tokens"]]
        x1, y1 = xy[nodes[c]["tokens"]]
        root_edge = p == 0
        hot = root_edge and _same_action(nodes, p, c, game, best)
        ax.plot([x0, x1], [y0, y1], "-",
                color="crimson" if hot else "steelblue",
                lw=0.5 + 4.0 * n / maxN, alpha=0.85 if hot else 0.6, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        lbl = f"N={n}\nQ={q:+.3f}\nP={prior:.2f}" if root_edge else f"N={n}"
        ax.annotate(lbl, (mx, my), fontsize=6,
                    color="crimson" if hot else "darkblue", ha="center", zorder=3)

    for node in nodes:
        x, y = xy[node["tokens"]]
        fc = "lightgreen" if node["terminal"] else "lightyellow"
        ax.annotate(f"{expr_of(node['tokens'], game)}\nN={node['total_N']}", (x, y),
                    ha="center", va="center", fontsize=7, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="black", alpha=0.9))

    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    _draw_root_panel(bx, game, acts, best, sim, max_sims)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _same_action(nodes, parent_id, child_id, game, best_action):
    if best_action is None:
        return False
    return nodes[child_id]["tokens"] == child_tokens(
        nodes[parent_id]["tokens"], best_action, game)


def _draw_root_panel(bx, game, acts, best, sim, max_sims):
    """Root action stats: visit bars, Q values, prior -- the 'first action' story.

    Rows are fixed by _ROW_ORDER (the final frame's actions) so a bar stays on
    its own row for the whole animation and only its length changes.
    """
    stats = {a: (n, q, p) for a, n, q, p in acts}
    labels, Ns = [], []
    for a in _ROW_ORDER:
        n, q, p = stats.get(a, (0, 0.0, _PRIOR.get(a, 0.0)))
        labels.append(f"{n:>2}  Q={q:+.3f}  P={p:.2f}  {_action_expr(a, game)}")
        Ns.append(n)
    y = np.arange(len(_ROW_ORDER))[::-1]
    colors = ["crimson" if a == best else "steelblue" for a in _ROW_ORDER]
    bx.barh(y, Ns, color=colors, alpha=0.75, height=0.5)
    bx.set_yticks(y)
    bx.set_yticklabels(labels, fontsize=7, family="monospace")
    bx.set_xlabel("root visits N(a)", fontsize=8)
    bx.set_xlim(0, max(_MAX_ROOT_N[0], 1) * 1.15)  # fixed scale: bars grow, axis doesn't
    bx.set_ylim(-0.8, max(len(_ROW_ORDER) - 1, 0) + 0.8)
    bx.set_title(f"root actions after {sim}/{max_sims} sims\n"
                 f"argmax-N first action in red", fontsize=9)
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(axis="x", labelsize=7)


_ACTION_EXPR_CACHE = {}
_ROW_ORDER = []   # root actions, fixed row order across frames (final-frame N desc)
_PRIOR = {}       # action -> prior P, for rows not yet visited in early frames
_MAX_ROOT_N = [1]  # largest root visit count over all frames, for a fixed bar scale


def _action_expr(a, game):
    return _ACTION_EXPR_CACHE.get(a, "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--max-sims", type=int, default=15)
    ap.add_argument("--rollout-n", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-visits", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=0.8, help="frames per second (slow by default)")
    ap.add_argument("--hold-last", type=int, default=4, help="extra copies of the final frame")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    prob = get_problem(args.problem)
    game = SymRegGame(problem_seed=args.seed, **prob.game_kwargs())
    game.reset_wrapper(seed=args.seed)
    n_actions = game.state_len * game.grammar.nprods
    rng = np.random.default_rng(args.seed)
    net = UniformPolicyValueNet(n_actions=n_actions)
    mcts = MCTS(game, net, n_simulations=args.max_sims,
                rollout_n=args.rollout_n, rollout_blend=0.0, rng=rng)
    kind = f"pure MCTS (uniform prior + {args.rollout_n} rollouts/leaf)"

    root_tokens = tuple(int(t) for t in game.state[:game.real_state_len])
    snaps = run_snapshots(mcts, game, root_tokens, args.max_sims,
                          args.max_depth, args.min_visits)

    final_acts = sorted(snaps[-1]["root_actions"], key=lambda t: (-t[1], -t[2]))
    for a, n, q, p in final_acts:
        _ACTION_EXPR_CACHE[a] = expr_of(child_tokens(root_tokens, a, game), game)
        _ROW_ORDER.append(a)
        _PRIOR[a] = p
    _MAX_ROOT_N[0] = max((n for _, n, _, _ in final_acts), default=1)

    xy = global_layout(snaps)
    maxN = max((e[2] for e in snaps[-1]["edges"]), default=1)
    frames_dir = REPO_ROOT / "Claude-research" / "figures" / f"_frames_{args.problem}"
    paths = []
    for i, snap in enumerate(snaps, start=1):
        title = (f"MCTS search tree — {args.problem}, {kind}, "
                 f"{i} sim{'s' if i > 1 else ''} (edge width ~ visits; seed {args.seed})")
        p = frames_dir / f"sim{i:03d}.png"
        draw_frame(game, snap, xy, p, title, maxN, i, args.max_sims)
        paths.append(p)
        best = max(snap["root_actions"], key=lambda t: (t[1], t[2]))
        print(f"sim {i:2d}: argmax first action -> {_ACTION_EXPR_CACHE.get(best[0], '?')}"
              f"  (N={best[1]}, Q={best[2]:+.3f})")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "Claude-research" / "figures" /
        f"{args.problem}_mcts_tree_sims1to{args.max_sims}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [imageio.imread(p) for p in paths]
    imgs += [imgs[-1]] * args.hold_last
    imageio.mimsave(out, imgs, duration=1.0 / args.fps, loop=0)
    print(f"frames: {len(imgs)}  gif: {out}")


if __name__ == "__main__":
    main()
