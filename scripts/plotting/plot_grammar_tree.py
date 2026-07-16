"""Grammar derivation tree for an SR instance, annotated with each node's mean
randomly-sampled (rollout) value.

Each node is a partial derivation (sentential form). Terminal nodes show their
exact fit R^2; partial nodes show the MEAN R^2 over K uniform-random completions
-- i.e. the value a mean-backup / random-rollout MCTS estimates for that node.
This visualizes why the search sticks: the compose branch (+ S S) scores BELOW
the C2*x^2 terminal under random completion, even though its best reachable
value is higher.

Run from repo root:
    python scripts/plotting/plot_grammar_tree.py --problem additive_quadratic --depth 2
"""

from sraz.utils import disable_numpy_multithreading

disable_numpy_multithreading()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from sraz.instances.symreg.game import SymRegGame  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def has_nonterm(tokens, g):
    return any(t in g.nonterms for t in tokens)


def set_partial_state(game, tokens):
    g = game.grammar
    buf = np.full(game.state_len, g.pad_tok, dtype=np.int64)
    buf[:len(tokens)] = np.array(tokens, dtype=np.int64)
    game.state = buf
    game.real_state_len = len(tokens)
    game.obs = buf.copy()
    game.reward = None
    game.terminated = False
    game.truncated = False


def node_value(game, tokens, rng, k):
    """(mean R^2, std): terminal -> exact fit; partial -> mean over k random completions."""
    g = game.grammar
    if not has_nonterm(tokens, g):
        rule = " ".join(g.tokenlist[t] for t in tokens)
        return game._fit_cached(rule), 0.0
    vals = []
    for _ in range(k):
        set_partial_state(game, tokens)
        for _ in range(game.state_len + 5):
            if game.terminated or game.truncated:
                break
            valid = np.flatnonzero(game.get_action_mask().ravel())
            if len(valid) == 0:
                break
            game.step_wrapper(int(valid[rng.integers(len(valid))]))
        vals.append(float(game.reward) if game.reward is not None else -1.0)
    return float(np.mean(vals)), float(np.std(vals))


def build_tree(game, max_depth):
    g = game.grammar
    nodes = []

    def add(tokens, depth, parent):
        nid = len(nodes)
        term = not has_nonterm(tokens, g)
        nodes.append({"id": nid, "tokens": tuple(tokens), "depth": depth,
                      "parent": parent, "children": [], "terminal": term})
        if parent is not None:
            nodes[parent]["children"].append(nid)
        if not term and depth < max_depth:
            i = next(k for k, t in enumerate(tokens) if t in g.nonterms)
            for j in g.proddict[tokens[i]]:
                rhs = g.productions[j]
                if len(tokens) + len(rhs) - 1 < game.state_len:
                    child = tuple(tokens[:i]) + tuple(rhs) + tuple(tokens[i + 1:])
                    add(child, depth + 1, nid)
        return nid

    add((g.symdict[g.start],), 0, None)
    return nodes


def layout(nodes):
    pos = {}
    counter = [0]

    def assign(nid):
        ch = nodes[nid]["children"]
        if not ch:
            x = float(counter[0])
            counter[0] += 1
        else:
            x = float(np.mean([assign(c) for c in ch]))
        pos[nid] = (x, -float(nodes[nid]["depth"]))
        return x

    assign(0)
    return pos


def draw(game, nodes, pos, values, out_path, problem, k):
    g = game.grammar
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=0.6, vmax=1.0)
    fig, ax = plt.subplots(figsize=(15, 7))
    for node in nodes:
        x0, y0 = pos[node["id"]]
        for c in node["children"]:
            x1, y1 = pos[c]
            ax.plot([x0, x1], [y0, y1], "-", color="gray", lw=0.8, zorder=1)
    for node in nodes:
        x, y = pos[node["id"]]
        v, _sd = values[node["id"]]
        expr = " ".join(g.tokenlist[t] for t in node["tokens"])
        tag = "R2" if node["terminal"] else "mean R2"
        ax.annotate(f"{expr}\n{tag}={v:.3f}", (x, y), ha="center", va="center",
                    fontsize=7.5, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.3", fc=cmap(norm(v)),
                              ec="black", alpha=0.9))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="value (R²)", shrink=0.7)
    ax.set_title(f"Derivation tree: {problem} — node = partial expression, "
                 f"color/label = mean random-completion R² (K={k}, seed 42)")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--completions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    prob = get_problem(args.problem)
    game = SymRegGame(problem_seed=args.seed, **prob.game_kwargs())
    rng = np.random.default_rng(args.seed)
    nodes = build_tree(game, args.depth)
    values = {n["id"]: node_value(game, n["tokens"], rng, args.completions)
              for n in nodes}
    pos = layout(nodes)
    out = (Path(args.out) if args.out else
           REPO_ROOT / "Claude-research" / "figures" / f"{args.problem}_grammar_tree.png")
    draw(game, nodes, pos, values, out, args.problem, args.completions)

    print(f"nodes: {len(nodes)}")
    for n in nodes:
        expr = " ".join(game.grammar.tokenlist[t] for t in n["tokens"])
        v, sd = values[n["id"]]
        kind = "term" if n["terminal"] else "part"
        print(f"  d{n['depth']} [{kind}] R2={v:+.3f}+/-{sd:.3f}  {expr}")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
