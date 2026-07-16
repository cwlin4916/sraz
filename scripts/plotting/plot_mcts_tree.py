"""Visualize one MCTS search (the root decision) on an SR instance.

Runs a single perform_simulations() from the root of --problem with
--n-simulations, then reconstructs and draws the search tree it built: edge
width ~ visit count N(a), each node labeled with its partial expression and
total visits, and the root actions also labeled with N / Q / prior P.

Default is *pure* MCTS (uniform prior + random rollouts) so no trained net is
needed; pass --alphazero to use the (untrained) learned net instead.

Run from repo root:
    python scripts/plotting/plot_mcts_tree.py --problem additive_quadratic --n-simulations 100
"""
from sraz.utils import disable_numpy_multithreading

disable_numpy_multithreading()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sraz.core.mcts import MCTS  # noqa: E402
from sraz.core.policy_value_net import UniformPolicyValueNet  # noqa: E402
from sraz.instances.symreg.game import SymRegGame  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def pad_key(tokens, game):
    buf = np.full(game.state_len, game.grammar.pad_tok, dtype=np.int64)
    buf[:len(tokens)] = np.array(tokens, dtype=np.int64)
    return buf.tobytes()


def child_tokens(parent_tokens, flat_action, game):
    g = game.grammar
    # MCTS keys edges by the np.unravel_index result, i.e. a 1-tuple (idx,) for
    # the 1-D Discrete action space; accept either that or a bare int.
    flat = flat_action[0] if isinstance(flat_action, tuple) else flat_action
    pos, prod = divmod(int(flat), g.nprods)
    rhs = g.productions[prod]
    return tuple(parent_tokens[:pos]) + tuple(rhs) + tuple(parent_tokens[pos + 1:])


def reconstruct(mcts, game, root_tokens, max_depth, min_visits):
    nodes, edges = [], []

    def add(tokens, depth):
        nid = len(nodes)
        node = mcts.nodes.get(pad_key(tokens, game))
        total_N = node.total_N if node is not None else 0
        terminal = node.is_terminal_state if node is not None else True
        nodes.append({"id": nid, "tokens": tokens, "total_N": total_N,
                      "terminal": terminal, "depth": depth})
        if node is None or terminal or depth >= max_depth:
            return nid
        thresh = 1 if depth == 0 else min_visits
        for a, n in sorted(node.action_N.items(), key=lambda kv: -kv[1]):
            if n < thresh:
                continue
            ct = child_tokens(tokens, a, game)
            if pad_key(ct, game) in mcts.nodes:
                q = node.action_Q.get(a, 0.0)
                p = float(node.nn_policy.ravel()[a]) if node.nn_policy is not None else 0.0
                cid = add(ct, depth + 1)
                edges.append((nid, cid, n, q, p))
        return nid

    add(root_tokens, 0)
    return nodes, edges


def layout(nodes, edges):
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        children[e[0]].append(e[1])
    pos, counter = {}, [0]

    def assign(nid):
        ch = children[nid]
        if not ch:
            x = float(counter[0]); counter[0] += 1
        else:
            x = float(np.mean([assign(c) for c in ch]))
        pos[nid] = (x, -float(nodes[nid]["depth"]))
        return x

    assign(0)
    return pos


def draw(game, nodes, edges, pos, out_path, title, root_id=0):
    g = game.grammar
    fig, ax = plt.subplots(figsize=(16, 8))
    maxN = max((e[2] for e in edges), default=1)
    for (p, c, n, q, prior) in edges:
        x0, y0 = pos[p]; x1, y1 = pos[c]
        ax.plot([x0, x1], [y0, y1], "-", color="steelblue",
                lw=0.5 + 4.0 * n / maxN, alpha=0.6, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        lbl = f"N={n}" if p != root_id else f"N={n}\nQ={q:+.3f}\nP={prior:.2f}"
        ax.annotate(lbl, (mx, my), fontsize=6, color="darkblue", ha="center", zorder=3)
    for node in nodes:
        x, y = pos[node["id"]]
        expr = " ".join(g.tokenlist[t] for t in node["tokens"])
        fc = "lightgreen" if node["terminal"] else "lightyellow"
        ax.annotate(f"{expr}\nN={node['total_N']}", (x, y), ha="center",
                    va="center", fontsize=7, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="black", alpha=0.9))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--rollout-n", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-visits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphazero", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    prob = get_problem(args.problem)
    game = SymRegGame(problem_seed=args.seed, **prob.game_kwargs())
    game.reset_wrapper(seed=args.seed)
    n_actions = game.state_len * game.grammar.nprods
    rng = np.random.default_rng(args.seed)
    if args.alphazero:
        from sraz.instances.symreg.network import SymRegPolicyValueNet
        net = SymRegPolicyValueNet(state_len=game.state_len,
                                   n_tokens=game.grammar.nsym + 1,
                                   n_actions=n_actions, random_seed=args.seed)
        mcts = MCTS(game, net, n_simulations=args.n_simulations, rng=rng)
        kind = "AlphaZero (untrained net)"
    else:
        net = UniformPolicyValueNet(n_actions=n_actions)
        mcts = MCTS(game, net, n_simulations=args.n_simulations,
                    rollout_n=args.rollout_n, rollout_blend=0.0, rng=rng)
        kind = f"pure MCTS (uniform prior + {args.rollout_n} rollouts/leaf)"

    root_tokens = tuple(int(t) for t in game.state[:game.real_state_len])
    mcts.perform_simulations("")
    root = mcts.nodes[pad_key(root_tokens, game)]

    g = game.grammar
    print(f"root actions after {args.n_simulations} sims ({kind}):")
    for a, n in sorted(root.action_N.items(), key=lambda kv: -kv[1]):
        expr = " ".join(g.tokenlist[t] for t in child_tokens(root_tokens, a, game))
        q = root.action_Q.get(a, 0.0)
        p = float(root.nn_policy.ravel()[a])
        print(f"  N={n:3d}  Q={q:+.3f}  P={p:.3f}  -> {expr}")

    nodes, edges = reconstruct(mcts, game, root_tokens, args.max_depth, args.min_visits)
    pos = layout(nodes, edges)
    out = (Path(args.out) if args.out else
           REPO_ROOT / "Claude-research" / "figures" /
           f"{args.problem}_mcts_tree_sim{args.n_simulations}.png")
    title = (f"MCTS search tree — {args.problem}, {kind}, "
             f"{args.n_simulations} sims (edge width ~ visits; seed {args.seed})")
    draw(game, nodes, edges, pos, out, title)
    print(f"nodes drawn: {len(nodes)}  edges: {len(edges)}")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
