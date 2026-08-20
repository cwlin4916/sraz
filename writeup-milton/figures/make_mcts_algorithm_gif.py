"""Animation for the elaboration note on the pure-MCTS searcher.

``pure-mcts.gif`` shows what ONE ``perform_simulations`` call actually does, one
algorithm phase per frame, on the real ``MCTS`` object from ``sraz.core.mcts``:

    SELECT    descend by argmax UCB, showing the exploit/explore split that
              ``calc_masked_ucbs`` computed for every legal action
    EXPAND    the first state with no policy yet becomes a node
    EVALUATE  ``_rollout_value`` plays one uniform-random completion and the
              terminal's R^2 comes back as the leaf value
    BACKUP    ``update_edge`` folds that value into Q along the descent path

Nothing here reimplements the search. The four phases are recovered by wrapping
``search``, ``calc_masked_ucbs``, ``_rollout_value`` and ``update_edge`` with
recorders that delegate to the originals, so the frames describe the shipped
code by construction. The wrappers are asserted transparent: the UCB array drawn
in a SELECT frame is the array ``calc_masked_ucbs`` returned, and the action
marked is ``np.argmax`` of it -- exactly what ``search`` steps on.

Two passes. Pass 1 runs the search and records the trace plus every state it
will ever touch; pass 2 lays those states out ONCE and replays the trace, so
nodes appear in place instead of the graph reflowing under the viewer.

Target ``lin_B`` (y = 5 + x). Chosen for the *algorithm*, not the pathology:
its two multiplicative one-move terminals both score -1, so the root prefers the
continue branch ``+ S S`` and the descent actually deepens -- which is what makes
SELECT, EXPAND, EVALUATE and BACKUP each visible. On ``lin_A`` the argmax is the
one-move terminal ``* C1 x``, so ``search`` returns at depth 1 almost every
simulation and the animation degenerates to a single edge. The trap that makes
``lin_A`` interesting is shown instead by
``Claude-experiments/8-17/lin_A/mcts-ucb_lin_A_c1_rb1000_rn1_s64.gif``.

Written to ``../elaborations/gifs/``.

    python3 writeup-milton/figures/make_mcts_algorithm_gif.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.gridspec import GridSpec                           # noqa: E402
from matplotlib.colors import Normalize                            # noqa: E402
from PIL import Image                                              # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import sraz.instances.symreg.game as game_mod                      # noqa: E402
from sraz.instances.symreg.config import SymRegConfig              # noqa: E402
from sraz.core.mcts import MCTS, EPS                               # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "elaborations" / "gifs"

TARGET = "lin_B"
N_SIMS = 10
C_EXPLORATION = 1.0
ROLLOUT_N = 1
ROLLOUT_BUDGET = 10_000          # provably non-binding: 10 sims x 1 x 11 = 110
SEED = 42
FPS = 1.4

CMAP = matplotlib.colormaps["RdYlGn"]
NORM = Normalize(-1.0, 1.0)
C_EXPLOIT = "#4a7fb5"
C_EXPLORE = "#e8a33d"

PHASES = {
    "SELECT":   ("#2c6fad", "descend by argmax UCB"),
    "EXPAND":   ("#7b5aa6", "add the leaf as a node"),
    "EVALUATE": ("#c8781e", "one random completion -> R²"),
    "BACKUP":   ("#2e8b57", "fold the value into Q along the path"),
}

# The lines of MCTS.search(), abridged to what the pure-MCTS path executes.
# `key` ties a line to the phase that is running when it is highlighted.
PSEUDO = [
    ("def search(self):", None),
    ("    if state not in self.nodes:", None),
    ("        self.nodes[state] = MCTSTreeNode(reward, is_terminal)", "EXPAND"),
    ("    if node.is_terminal_state:", None),
    ("        return 0.0            # reward already on the parent edge", "TERMINAL"),
    ("    if node.nn_policy is None:                      # a leaf", "EXPAND"),
    ("        node.nn_policy = uniform prior * action_mask", "EXPAND"),
    ("        leaf = self._rollout_value()                # random play", "EVALUATE"),
    ("        return leaf", "EVALUATE"),
    ("    ucbs = self.calc_masked_ucbs(node)              # Qtilde + u", "SELECT"),
    ("    a = argmax(ucbs);  self.game.step_wrapper(a)", "SELECT"),
    ("    r = self.game.reward                            # 0 until terminal", "SELECT"),
    ("    v = self.search()                               # recurse", None),
    ("    self.update_edge(node, a, r + v)                # Q <- mean", "BACKUP"),
    ("    node.total_N += 1", "BACKUP"),
    ("    return r + v", "BACKUP"),
]


# ---------------------------------------------------------------------------
# Pass 1: run the real search, recording a phase trace
# ---------------------------------------------------------------------------
def build():
    cfg = SymRegConfig(problem="additive_quadratic", pure_mcts=True)
    cfg.game.kwargs.update(target=TARGET, max_len=12, redraw_constants=False)
    cfg.agent.random_seeds = {"mcts": SEED, "train": SEED + 1,
                              "eval": SEED + 2, "external_policy": SEED + 3}
    game, net, agent, _ = cfg.build()
    game.reset_wrapper()
    return game, net, agent


def trace_search(game, net, agent):
    params = dict(agent.mcts_params)
    params.update(n_simulations=1, temperature=0.01,
                  c_exploration=C_EXPLORATION, rollout_n=ROLLOUT_N,
                  rollout_budget=ROLLOUT_BUDGET)
    mcts = MCTS(game, net, rng=np.random.default_rng(SEED), **params)
    mcts.q_min, mcts.q_max = float("inf"), float("-inf")
    mcts._search_rollout_budget = mcts.rollout_budget
    grammar = game.grammar

    events: list[dict] = []
    path: list = []                 # states currently on the descent stack
    state_depth: dict = {}
    edges: set = set()
    child_of: dict = {}             # (parent_state, action) -> child_state
    expr: dict = {}                 # state -> expression string
    ctx = {"sim": 0, "last_pick": None}

    orig_search = mcts.search
    orig_ucb = mcts.calc_masked_ucbs
    orig_roll = mcts._rollout_value
    orig_edge = mcts.update_edge
    orig_qnm = mcts.query_net_masked

    def w_search(msg, _depth=0):
        st = mcts.game.hashable_obs
        state_depth.setdefault(st, _depth)
        # Record the label here, where the game object still knows
        # real_state_len; hashable_obs alone cannot be decoded.
        expr.setdefault(st, mcts.game._decode_state())
        if path:
            edges.add((path[-1], st))
            # The action that produced this state was chosen by the UCB call
            # one level up, which ran immediately before this entry.
            if ctx["last_pick"] is not None:
                child_of[(path[-1], ctx["last_pick"])] = st
                ctx["last_pick"] = None
        path.append(st)
        fresh = st not in mcts.nodes
        was_leaf = fresh or mcts.nodes[st].nn_policy is None
        try:
            return orig_search(msg, _depth)
        finally:
            node = mcts.nodes.get(st)
            if was_leaf and node is not None and node.is_terminal_state:
                events.append({"phase": "TERMINAL", "sim": ctx["sim"],
                               "path": list(path), "focus": st,
                               "note": "terminal state: returns 0.0, the "
                                       "reward is already on the parent edge"})
            path.pop()

    def w_qnm(msg):
        pol, val, mask = orig_qnm(msg)
        st = mcts.game.hashable_obs
        events.append({"phase": "EXPAND", "sim": ctx["sim"], "path": list(path),
                       "focus": st, "n_legal": int(mask.sum()),
                       "prior": float(pol[mask.astype(bool)][0]),
                       "net_value": float(val)})
        return pol, val, mask

    def w_ucb(node, msg):
        ucbs = orig_ucb(node, msg)
        st = mcts.game.hashable_obs
        mask = node.action_mask.astype(bool)
        size = mask.shape[0]
        q = np.zeros(size); n = np.zeros(size)
        for a, v in node.action_Q.items():
            q[a] = v
        for a, v in node.action_N.items():
            n[a] = v
        if mcts.q_min == float("inf") or mcts.q_max == float("-inf"):
            qn = np.zeros(size)
        elif mcts.q_max > mcts.q_min:
            qn = (q - mcts.q_min) / (mcts.q_max - mcts.q_min)
        else:
            qn = np.full(size, 0.5)
        u = mcts.c_exploration * node.nn_policy * np.sqrt(node.total_N + EPS) / (1 + n)
        # The drawn split must reconstruct exactly what the real method returned.
        rebuilt = np.where(mask, qn + u, -np.inf)
        assert np.allclose(rebuilt[mask], ucbs[mask], atol=1e-12), \
            "UCB decomposition does not match calc_masked_ucbs"
        pick = int(np.argmax(ucbs))
        # search() keys edges by np.unravel_index(...), a 1-tuple for a 1-D
        # action space -- so action_N/action_Q are tuple-keyed, and child_of
        # must use the same key to line up.
        ctx["last_pick"] = np.unravel_index(pick, ucbs.shape)
        events.append({"phase": "SELECT", "sim": ctx["sim"], "path": list(path),
                       "focus": st, "ucbs": ucbs.copy(), "qn": qn.copy(),
                       "u": u.copy(), "n": n.copy(), "q": q.copy(),
                       "mask": mask.copy(), "pick": pick,
                       "total_N": node.total_N,
                       "qlo": mcts.q_min, "qhi": mcts.q_max})
        return ucbs

    def w_roll(msg):
        before = list(reached["rules"])
        v = orig_roll(msg)
        new = [r for r in reached["rules"] if r not in before] or reached["rules"][-1:]
        events.append({"phase": "EVALUATE", "sim": ctx["sim"], "path": list(path),
                       "focus": mcts.game.hashable_obs, "value": v,
                       "completion": new[-1] if new else None})
        return v

    def w_edge(node, action, reward):
        orig_edge(node, action, reward)
        parent = path[-1] if path else None
        events.append({"phase": "BACKUP", "sim": ctx["sim"], "path": list(path),
                       "focus": child_of.get((parent, action), parent),
                       "parent": parent, "action": action,
                       "child": child_of.get((parent, action)),
                       "depth": len(path),
                       "backed": reward, "newQ": node.action_Q[action],
                       "newN": node.action_N[action]})

    # Every terminal the game scores passes through _fit_cached, so hooking it
    # captures the rollout's completion without touching _rollout_value's body.
    reached = {"rules": []}
    orig_fit = type(game)._fit_cached

    def w_fit(self, rule):
        reached["rules"].append(rule)
        return orig_fit(self, rule)

    type(game)._fit_cached = w_fit
    mcts.search = w_search
    mcts.query_net_masked = w_qnm
    mcts.calc_masked_ucbs = w_ucb
    mcts._rollout_value = w_roll
    mcts.update_edge = w_edge

    try:
        # Expand the root first, exactly as perform_simulations does.
        old = mcts.game.stash_state()
        mcts.search("")
        mcts.game = mcts.game.unstash_state(old)
        for i in range(N_SIMS):
            ctx["sim"] = i + 1
            old = mcts.game.stash_state()
            mcts.search("")
            mcts.game = mcts.game.unstash_state(old)
    finally:
        type(game)._fit_cached = orig_fit

    root = game.hashable_obs
    return mcts, events, state_depth, edges, root, grammar, child_of, expr


# ---------------------------------------------------------------------------
# Pass 2: fixed layout, then replay
# ---------------------------------------------------------------------------
def layout(state_depth, edges, root):
    """x by ordering within a depth level, y by depth. Computed once."""
    by_depth: dict[int, list] = {}
    for st, d in state_depth.items():
        by_depth.setdefault(d, []).append(st)
    kids: dict = {}
    for p, c in edges:
        kids.setdefault(p, []).append(c)
    order: list = []

    def walk(st):                      # depth-first so siblings stay adjacent
        order.append(st)
        for c in sorted(kids.get(st, []), key=lambda s: state_depth[s]):
            if c not in order:
                walk(c)
    walk(root)
    for st in state_depth:
        if st not in order:
            order.append(st)
    rank = {st: i for i, st in enumerate(order)}
    pos = {}
    for d, sts in by_depth.items():
        sts = sorted(sts, key=lambda s: rank[s])
        for i, st in enumerate(sts):
            x = (i + 0.5) / len(sts)
            pos[st] = (x, -d)
    return pos


def draw_frame(ev, mcts, pos, edges, root, grammar, expr, snap, idx, total):
    phase = ev["phase"]
    colour, blurb = PHASES.get(phase, ("#555555", ""))
    fig = plt.figure(figsize=(15.0, 8.6), dpi=92)
    gs = GridSpec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1.0, 0.92],
                  hspace=0.22, wspace=0.16)
    axT = fig.add_subplot(gs[:, 0])
    axP = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # ---------------- the graph MCTS has built so far --------------------
    live = snap["live"]
    active = [s for s in ev["path"] if s in pos]
    active_edges = {(a, b) for a, b in zip(active, active[1:])}
    for (p, c) in edges:
        if p not in live or c not in live:
            continue
        x0, y0 = pos[p]; x1, y1 = pos[c]
        eN, eQ = snap["edge"].get((p, c), (0, None))
        hot = (p, c) in active_edges
        col = CMAP(NORM(eQ)) if eQ is not None else "0.80"
        lw = 0.7 + 3.6 * (eN / max(snap["maxN"], 1))
        if hot:
            axT.plot([x0, x1], [y0, y1], "-", color=colour, lw=lw + 4.0,
                     zorder=3, solid_capstyle="round", alpha=0.95)
        axT.plot([x0, x1], [y0, y1], "-", color=col, lw=max(lw, 0.7), zorder=4,
                 solid_capstyle="round")
    for st in live:
        x, y = pos[st]
        expanded = snap["expanded"].get(st, False)
        term = snap["terminal"].get(st, False)
        face = "#ffffff" if not expanded else "#eef3f8"
        edge = colour if st in active else ("#444444" if expanded else "0.68")
        w = 2.6 if st == ev.get("focus") else (1.6 if st in active else 0.9)
        shape = "s" if term else "o"
        axT.plot([x], [y], shape, ms=13 if st == ev.get("focus") else 10,
                 mfc=face, mec=edge, mew=w, zorder=5)
        lab = expr[st] or "S"
        axT.text(x, y - 0.20, lab, ha="center", va="top", fontsize=7.4,
                 family="monospace",
                 color="#111111" if st in active else "0.35", zorder=6)
    axT.set_axis_off()
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    axT.set_xlim(min(xs) - 0.12, max(xs) + 0.12)
    axT.set_ylim(min(ys) - 0.55, 0.42)
    axT.set_title(
        f"the search graph after {snap['done']} of {N_SIMS} simulations\n"
        f"square = terminal, hollow = not yet expanded, edge width = N, "
        f"edge colour = backed-up Q",
        fontsize=9.2)

    # ---------------- pseudocode with the live line lit ------------------
    axP.set_axis_off()
    axP.set_xlim(0, 1); axP.set_ylim(0, 1)
    axP.text(0.0, 1.02, "MCTS.search()  (sraz/core/mcts.py)", fontsize=9.6,
             family="monospace", fontweight="bold", va="bottom")
    lit = ev.get("lines", set())
    for i, (line, key) in enumerate(PSEUDO):
        y = 0.955 - i * 0.0625
        on = i in lit
        if on:
            axP.add_patch(plt.Rectangle((-0.012, y - 0.019), 1.02, 0.045,
                                        color=colour, alpha=0.17, zorder=1))
        axP.text(0.0, y, line, fontsize=8.0, family="monospace", va="center",
                 color="#101010" if on else "0.52",
                 fontweight="bold" if on else "normal", zorder=2)

    # ---------------- per-phase context panel ----------------------------
    axC.set_axis_off()
    if phase == "SELECT":
        mask = ev["mask"]
        acts = np.flatnonzero(mask)
        labs = [expr.get(snap["child_of"].get((ev["focus"], (int(a),))), "(unvisited)")
                for a in acts]
        ypos = np.arange(len(acts))[::-1]
        axC.barh(ypos, ev["qn"][acts], color=C_EXPLOIT, height=0.52,
                 label="$\\tilde Q(a)$  exploit")
        axC.barh(ypos, ev["u"][acts], left=ev["qn"][acts], color=C_EXPLORE,
                 height=0.52, hatch="///", label="$u(a)$  explore")
        for y, a in zip(ypos, acts):
            tot = ev["qn"][a] + ev["u"][a]
            star = "  ← argmax" if int(a) == ev["pick"] else ""
            axC.text(tot + 0.02, y, f"{tot:.3f}{star}", va="center",
                     fontsize=7.6,
                     fontweight="bold" if int(a) == ev["pick"] else "normal")
        axC.set_yticks(ypos)
        axC.set_yticklabels([f"{l}   N={int(ev['n'][a])}"
                             for l, a in zip(labs, acts)], fontsize=7.4,
                            family="monospace")
        axC.set_xlim(0, max(1.02, float((ev["qn"] + ev["u"])[acts].max()) * 1.22))
        axC.set_axis_on()
        axC.spines[["top", "right"]].set_visible(False)
        axC.set_xlabel("selection score", fontsize=8.4)
        axC.legend(fontsize=7.2, loc="lower right", frameon=False)
        axC.set_title(
            f"$\\mathrm{{UCB}}(a)=\\tilde Q(a)+c\\,P(a)\\sqrt{{N_{{tot}}}}/(1+N(a))$"
            f"    $c={C_EXPLORATION:g}$, $P(a)={1/len(acts):.2f}$, "
            f"$N_{{tot}}={ev['total_N']}$\n"
            f"$\\tilde Q$ = min-max normalised over "
            f"[{ev['qlo']:+.3f}, {ev['qhi']:+.3f}] seen so far; an unvisited "
            f"action is scored at raw $Q=0$",
            fontsize=8.0)
    else:
        axC.set_xlim(0, 1); axC.set_ylim(0, 1)
        txt = []
        if phase == "EXPAND":
            txt = [("nn_policy was None, so this leaf becomes a node:", "0.25"),
                   (f"    {expr[ev['focus']] or 'S'}", "#101010"),
                   ("", "0.25"),
                   (f"    legal actions   {ev['n_legal']}", "0.25"),
                   (f"    prior P(a)      {ev['prior']:.4f}  = 1/{ev['n_legal']}",
                    colour),
                   (f"    net value       {ev['net_value']:+.4f}  (never used:", "0.25"),
                   ("                    rollout_blend = 0)", "0.25"),
                   ("", "0.25"),
                   ("UniformPolicyValueNet never learns, so the prior is a", "0.25"),
                   ("constant and selection is driven entirely by backed-up Q.", "0.25")]
        elif phase == "EVALUATE":
            txt = [("_rollout_value(): one uniform-random completion", "0.25"),
                   (f"    from  {expr[ev['focus']] or 'S'}", "#101010"),
                   (f"    to    {ev['completion']}", "#101010"),
                   ("", "0.25"),
                   (f"leaf value returned = {ev['value']:+.4f}", colour),
                   ("", "0.25"),
                   ("at rollout_n=1 one simulation costs exactly one", "0.25"),
                   ("terminal evaluation, so this IS the sentence's R².", "0.25")]
        elif phase == "BACKUP":
            txt = [("update_edge(): append the value, recompute the mean", "0.25"),
                   (f"    edge into  {expr[ev['focus']] or 'S'}", "#101010"),
                   (f"    backed up  {ev['backed']:+.4f}", "#101010"),
                   (f"    N(s,a) -> {ev['newN']}", "0.25"),
                   (f"    Q(s,a) -> {ev['newQ']:+.4f}   (mean backup)", colour),
                   ("", "0.25"),
                   ("the mean is why a branch full of poor completions", "0.25"),
                   ("scores below a mediocre one-move terminal.", "0.25")]
        elif phase == "TERMINAL":
            txt = [("the descent hit a terminal state:", "0.25"),
                   (f"    {expr[ev['focus']] or 'S'}", "#101010"),
                   ("", "0.25"),
                   ("search() returns 0.0 here -- the R² for entering", "0.25"),
                   ("it was already captured as the parent's immediate", "0.25"),
                   ("reward, so counting it again would double it.", "0.25")]
        for i, (line, col) in enumerate(txt):
            axC.text(0.02, 0.90 - i * 0.108, line, fontsize=9.0, va="top",
                     family="monospace", color=col,
                     fontweight="bold" if col == colour else "normal")

    fig.suptitle(
        f"Pure MCTS on {TARGET}   y = 5 + x   |   simulation "
        f"{max(ev['sim'],0)}/{N_SIMS}   |   frame {idx}/{total}\n"
        f"$\\bf{{{phase}}}$ — {blurb}",
        fontsize=12.0, color=colour)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.875, bottom=0.075)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    game, net, agent = build()
    (mcts, events, state_depth, edges, root, grammar, child_of,
     expr) = trace_search(game, net, agent)
    print(f"target {TARGET}: y = {game.target_infix}")
    print(f"[trace] {len(events)} phase events over {N_SIMS} simulations; "
          f"{len(state_depth)} states touched, {len(edges)} edges")

    pos = layout(state_depth, edges, root)

    # which pseudocode lines to light per phase
    lines_for = {
        "SELECT":   {9, 10, 11},
        "EXPAND":   {1, 2, 5, 6},
        "EVALUATE": {7, 8},
        "BACKUP":   {13, 14, 15},
        "TERMINAL": {3, 4},
    }
    for ev in events:
        ev["lines"] = lines_for.get(ev["phase"], set())

    # Rebuild the incremental view: which nodes exist / are expanded, and the
    # edge stats, as of each event. Replayed from the trace so the drawn state
    # is the state at that instant, not the final tree.
    live = {root}
    expanded: dict = {}
    terminal: dict = {}
    edge: dict = {}
    frames = []
    done = 0
    for i, ev in enumerate(events, start=1):
        for st in ev["path"]:
            live.add(st)
        if ev["phase"] in ("EXPAND", "EVALUATE"):
            expanded[ev["focus"]] = True
        if ev["phase"] == "TERMINAL":
            terminal[ev["focus"]] = True
            expanded[ev["focus"]] = True
        if ev["phase"] == "BACKUP" and ev.get("child") is not None:
            edge[(ev["parent"], ev["child"])] = (ev["newN"], ev["newQ"])
        if ev["phase"] == "BACKUP" and ev["depth"] == 1:
            done = ev["sim"]        # backup reached the root: simulation over
        snap = {"live": set(live), "expanded": dict(expanded),
                "terminal": dict(terminal), "edge": dict(edge),
                "maxN": max([n for n, _ in edge.values()] + [1]),
                "done": done, "child_of": dict(child_of)}
        frames.append(draw_frame(ev, mcts, pos, edges, root, grammar, expr,
                                 snap, i, len(events)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gif = OUT_DIR / "pure-mcts.gif"
    hold = [frames[-1]] * max(1, int(round(FPS * 3)))
    seq = frames + hold
    seq[0].save(gif, save_all=True, append_images=seq[1:],
                duration=int(1000 / FPS), loop=0)
    print(f"[saved] {gif}  ({len(frames)} frames, {FPS} fps)")

    rN = {a: n for a, n in mcts.nodes[root].action_N.items()}
    rQ = mcts.nodes[root].action_Q
    print("\nfinal root edges:")
    for a in sorted(rN, key=lambda a: -rN[a]):
        st = child_of.get((root, a))
        print(f"  N={rN[a]:>3}  Q={rQ[a]:+.4f}  {expr.get(st, '?')}")


if __name__ == "__main__":
    main()
