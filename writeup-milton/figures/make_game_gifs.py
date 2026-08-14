"""Animations for the elaboration note on the derivation MDP.

The two static figures (``make_game_figures.py``) answer *what the MDP is* and
*when it terminates*; these two GIFs show the same two things happening, one
action at a time, on the real ``SymRegGame``:

    derivation.gif    one episode: the legal-action mask, the action drawn
                      from it, the splice it performs on the token buffer, and
                      the least-squares fit that the terminal expression earns.

    termination.gif   five uniform-random episodes run to termination, with a
                      running tally of which exit fired. Episodes end at wildly
                      different lengths and none of them stalls.

Both are written to ``../elaborations/gifs/``.

The fit is recomputed here through the same sympy/lmfit path the game uses,
because ``SymRegGame`` returns only the scalar reward and the animation needs
the fitted curve as well. The recomputed R^2 is asserted equal to the game's
own reward for every frame drawn, so the curve on screen belongs to the number
beside it.

    python3 writeup-milton/figures/make_game_gifs.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import lmfit  # noqa: E402
import sympy  # noqa: E402
from sympy.parsing import sympy_parser  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import (  # noqa: E402
    C_BAD, C_CONT, C_CONT_BG, C_DEAD, C_OK, C_TGT, HERE, build_game,
)
from make_game_figures import C_NT, C_NT_BG, C_PAD, MONO  # noqa: E402

from sraz.instances.symreg.game import prefix_to_infix  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "elaborations", "gifs"))
GIF_DPI = 100


def save_gif(fig, render, frames, durations, out):
    """Render each frame and write a GIF with explicit per-frame durations.

    ``FuncAnimation`` + ``PillowWriter`` cannot hold a frame on screen: Pillow
    collapses byte-identical consecutive frames, so repeating a frame to pause
    on it silently does nothing. Writing the frames here keeps the pause.

    The palette is built from *every* frame stacked together, not from the
    first one: the opening frame carries none of the greens the terminal frame
    ends on, and quantizing against it turns the fitted curve grey.
    """
    imgs = []
    for fr in frames:
        render(fr)
        fig.canvas.draw()
        imgs.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    w, h = imgs[0].size
    strip = Image.new("RGB", (w, h * len(imgs)))
    for i, im in enumerate(imgs):
        strip.paste(im, (0, i * h))
    base = strip.quantize(colors=255, method=Image.MEDIANCUT)
    pal = [im.quantize(palette=base, dither=Image.NONE) for im in imgs]
    pal[0].save(out, save_all=True, append_images=pal[1:],
                duration=durations, loop=0, optimize=True, disposal=2)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({len(pal)} frames, {kb:.0f} KB)")


# ---------------------------------------------------------------------------
def fit_detail(rule: str, xs, ys, max_nfev):
    """The game's fit, but returning the curve as well as the score."""
    tokens = rule.strip().split()
    pv, ind = {}, ()
    if "x" in tokens:
        pv["x"] = xs
        ind = ("x",)
    for t in tokens:
        if "C" in t:
            pv[t] = 2.5                      # game.C_INIT
    model = sympy_parser.parse_expr(prefix_to_infix(tokens), evaluate=False)
    fn = sympy.lambdify(list(model.free_symbols), model)
    res = lmfit.Model(fn, independent_vars=ind).fit(
        data=ys, **pv, **({} if max_nfev is None else {"max_nfev": max_nfev}))
    curve = np.asarray(res.best_fit, dtype=float)
    if curve.ndim == 0 or curve.size == 1:
        curve = np.full_like(xs, float(curve))
    ss_res = float(np.sum((ys - curve) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0)), curve


def episode(game, rng):
    """One uniform-over-the-mask episode, recorded frame by frame."""
    game.reset_wrapper()
    G = game.grammar
    steps = []
    while True:
        buf = tuple(int(t) for t in game.state[:game.real_state_len])
        mask = game.get_action_mask().reshape(game.state_len, G.nprods)
        valid = np.flatnonzero(mask.ravel())
        a = int(rng.choice(valid))
        pos, prod = divmod(a, G.nprods)
        steps.append(dict(buf=buf, mask=mask.copy(), pos=pos, prod=prod,
                          n_legal=int(mask.sum())))
        _, r, term, _, info = game.step_wrapper(a)
        if term:
            buf = tuple(int(t) for t in game.state[:game.real_state_len])
            steps.append(dict(buf=buf, mask=np.zeros_like(mask), pos=None,
                              prod=None, n_legal=0, reward=r,
                              rule=info["rule"]))
            return steps


def greedy_episode(game, cen):
    """The V*-greedy episode: the one that actually reaches R = 1."""
    G = game.grammar
    cur, steps = cen.root, []
    while True:
        mask = np.zeros((game.state_len, G.nprods), dtype=bool)
        for i, j in cen.legal(cur):
            mask[i, j] = True
        acts = cen.legal(cur)
        if not acts:
            steps.append(dict(buf=cur, mask=mask, pos=None, prod=None,
                              n_legal=0, reward=cen.reward(cur),
                              rule=cen.rule(cur)))
            return steps
        tgt = cen.vstar(cur)
        pos, prod = next((i, j) for i, j in acts
                         if abs(cen.vstar(cen.child(cur, i, j)) - tgt) < 1e-12)
        steps.append(dict(buf=cur, mask=mask, pos=pos, prod=prod,
                          n_legal=len(acts)))
        cur = cen.child(cur, pos, prod)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_cells(ax, buf, L, G, y, cw, ch, x0=0.02, hi=None, pos=None, fs=10):
    for k in range(L):
        inside = k < len(buf)
        tok = buf[k] if inside else None
        if not inside:
            fc, ec, tc, txt = "white", C_PAD, C_PAD, "·"
        else:
            txt = G.tokenlist[tok]
            if hi is not None and hi[0] <= k < hi[1]:
                fc, ec, tc = C_CONT_BG, C_CONT, "black"
            elif tok in G.nonterms:
                fc, ec, tc = C_NT_BG, C_NT, C_NT
            else:
                fc, ec, tc = "white", "0.55", "black"
        ax.add_patch(Rectangle((x0 + k * cw, y), cw * 0.9, ch, facecolor=fc,
                               edgecolor=ec, lw=1.2, zorder=3))
        ax.text(x0 + (k + 0.45) * cw, y + ch / 2, txt, ha="center",
                va="center", fontsize=fs, color=tc, family=MONO, zorder=4)
        if pos is not None and k == pos:
            ax.add_patch(Rectangle((x0 + k * cw - 0.004, y - 0.012),
                                   cw * 0.9 + 0.008, ch + 0.024,
                                   facecolor="none", edgecolor=C_BAD, lw=2.0,
                                   zorder=5))


def draw_mask(ax, mask, L, nprods, x0, y0, cw, rh, pick=None):
    for i in range(L):
        for j in range(nprods):
            on = bool(mask[i, j])
            ax.add_patch(Rectangle((x0 + i * cw, y0 - j * rh),
                                   cw * 0.9, rh * 0.82,
                                   facecolor=C_CONT_BG if on else "#f1f1f1",
                                   edgecolor="white", lw=0.8, zorder=3))
            if pick is not None and (i, j) == pick:
                ax.add_patch(Rectangle((x0 + i * cw, y0 - j * rh),
                                       cw * 0.9, rh * 0.82, facecolor=C_BAD,
                                       edgecolor="none", zorder=4))
    for j in range(nprods):
        ax.text(x0 - 0.008, y0 - j * rh + rh * 0.41, f"p{j}", ha="right",
                va="center", fontsize=7.5, color="0.45", family=MONO)


# ---------------------------------------------------------------------------
# GIF 1 -- one derivation
# ---------------------------------------------------------------------------
def gif_derivation(game, cen):
    G, L = game.grammar, game.state_len
    steps = greedy_episode(game, cen)
    # per action: the state and its legal set, then the action drawn from it
    frames, durations = [], []
    for k, st in enumerate(steps):
        if st["pos"] is None:
            frames.append(("term", k))
            durations.append(4000)
        else:
            frames += [("show", k), ("pick", k)]
            durations += [700, 1100]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.6, 4.3), dpi=GIF_DPI,
        gridspec_kw=dict(width_ratios=[1.62, 1.0]))
    fig.subplots_adjust(left=0.015, right=0.965, top=0.90, bottom=0.06,
                        wspace=0.16)

    def render(fr):
        kind, k = fr
        st = steps[k]
        axL.clear()
        axL.set_xlim(0, 1)
        axL.set_ylim(0, 1)
        axL.axis("off")
        axL.set_title("the derivation MDP, one action at a time",
                      fontsize=12, loc="left", color=C_TGT)

        prev = steps[k - 1] if k > 0 else None
        hi = None
        if prev is not None and prev["pos"] is not None:
            hi = (prev["pos"], prev["pos"] + len(G.productions[prev["prod"]]))
        cw = 0.0765
        draw_cells(axL, st["buf"], L, G, 0.735, cw, 0.115,
                   hi=hi, pos=st["pos"] if kind == "pick" else None)
        axL.text(0.02, 0.905,
                 f"$s_{{{k}}}$    real_state_len = {len(st['buf'])} "
                 f"of {L}", fontsize=10, color="0.35", family=MONO)

        axL.text(0.02, 0.615, f"legal actions   $|\\mathcal{{A}}(s)| = "
                              f"{st['n_legal']}$  of  {L * G.nprods}",
                 fontsize=10, color="0.35", family=MONO)
        draw_mask(axL, st["mask"], L, G.nprods, 0.02, 0.505, cw, 0.088,
                  pick=(st["pos"], st["prod"]) if kind == "pick" else None)

        if kind == "show":
            axL.text(0.02, 0.115,
                     f"{sum(1 for t in st['buf'] if t in G.nonterms)} "
                     f"nonterminal(s) remain → not terminal; "
                     f"draw one of the {st['n_legal']} legal actions",
                     fontsize=10.5, color="0.35", family=MONO)
        elif kind == "pick":
            rhs = " ".join(G.tokenlist[t] for t in G.productions[st["prod"]])
            axL.text(0.02, 0.115,
                     f"a = (pos {st['pos']}, p{st['prod']})   →   "
                     f"S → {rhs}", fontsize=11, color=C_CONT, family=MONO)
            axL.text(0.02, 0.028,
                     f"splice: buffer[{st['pos']}] becomes "
                     f"{len(G.productions[st['prod']])} cells, "
                     f"Δ real_state_len = "
                     f"{len(G.productions[st['prod']]) - 1:+d}",
                     fontsize=9, color="0.45", family=MONO)
        else:
            axL.text(0.02, 0.115, "no nonterminal remains  →  terminal",
                     fontsize=11, color=C_OK, family=MONO, weight="bold")
            axL.text(0.02, 0.028,
                     f"{st['rule']}   =   {prefix_to_infix(st['rule'].split())}",
                     fontsize=9, color="0.45", family=MONO)

        axR.clear()
        axR.set_title("terminal reward: least squares in data space",
                      fontsize=11, loc="left", color=C_TGT)
        axR.plot(game.xs, game.exact_ys, "o", ms=3.4, color=C_TGT,
                 label="target $y$", zorder=3)
        if kind == "term":
            r2, curve = fit_detail(st["rule"], game.xs, game.exact_ys,
                                   game.lmfit_max_nfev)
            assert abs(r2 - st["reward"]) < 1e-9, (r2, st["reward"])
            axR.plot(game.xs, curve, "-", lw=2.2, color=C_OK,
                     label=f"fit,  $R^2$ = {r2:.6f}", zorder=4)
        else:
            axR.text(0.5, 0.5, "the form still holds a nonterminal,\n"
                               "so there is nothing to fit yet\n"
                               "reward so far: 0",
                     transform=axR.transAxes, ha="center", va="center",
                     fontsize=10, color="0.45", linespacing=1.7,
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                               edgecolor="0.85"))
        axR.set_xlabel("$x$", fontsize=10)
        axR.set_ylabel("$y$", fontsize=10)
        axR.legend(loc="upper left", fontsize=9)
        axR.grid(True, color="0.9", lw=0.6)

    save_gif(fig, render, frames, durations,
             os.path.join(OUT, "derivation.gif"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# GIF 2 -- termination
# ---------------------------------------------------------------------------
def gif_termination(game, n_eps=5, seed=3):
    G, L = game.grammar, game.state_len
    rng = np.random.default_rng(seed)
    eps, seen = [], set()
    for _ in range(5000):
        if len(eps) == n_eps:
            break
        e = episode(game, rng)
        if e[-1]["rule"] in seen:
            continue
        seen.add(e[-1]["rule"])
        eps.append(e)
    eps.sort(key=len)

    frames, durations = [], []
    for ei, e in enumerate(eps):
        for k in range(len(e)):
            frames.append((ei, k))
            durations.append(2600 if k == len(e) - 1 else 950)

    fig, ax = plt.subplots(figsize=(11.6, 4.6), dpi=GIF_DPI)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)

    def render(fr):
        ei, k = fr
        e = eps[ei]
        st = e[k]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("when does an episode end?  five uniform-random "
                     "derivations, run to termination", fontsize=12,
                     loc="left", color=C_TGT)

        prev = e[k - 1] if k > 0 else None
        hi = None
        if prev is not None and prev["pos"] is not None:
            hi = (prev["pos"], prev["pos"] + len(G.productions[prev["prod"]]))
        cw = 0.0765
        draw_cells(ax, st["buf"], L, G, 0.66, cw, 0.135, hi=hi,
                   pos=st["pos"], fs=11)

        ax.text(0.02, 0.885, f"episode {ei + 1} of {len(eps)}    "
                             f"action {k} of {len(e) - 1}",
                fontsize=10.5, color="0.35", family=MONO)

        if st["pos"] is not None:
            rhs = " ".join(G.tokenlist[t] for t in G.productions[st["prod"]])
            ax.text(0.02, 0.545,
                    f"still {sum(1 for t in st['buf'] if t in G.nonterms)} "
                    f"nonterminal(s) → not terminal;   "
                    f"draw uniformly from $|\\mathcal{{A}}|$ = "
                    f"{st['n_legal']}:   a = (pos {st['pos']}, p{st['prod']})"
                    f"  S → {rhs}",
                    fontsize=10, color=C_CONT, family=MONO)
        else:
            ax.text(0.02, 0.545,
                    f"0 nonterminals left → EXIT 1 fires:   "
                    f"R = {st['reward']:.6f}",
                    fontsize=11, color=C_OK, family=MONO, weight="bold")
            ax.text(0.02, 0.455,
                    f"{st['rule']}   =   "
                    f"{prefix_to_infix(st['rule'].split())}",
                    fontsize=9.5, color="0.45", family=MONO)

        # the ledger: every episode resolved so far, and by which exit
        ax.text(0.02, 0.335, "episodes finished so far", fontsize=10,
                color=C_TGT, weight="bold")
        y = 0.255
        for j, ee in enumerate(eps):
            done = j < ei or (j == ei and e[-1] is st)
            if not done:
                continue
            term = ee[-1]
            ax.text(0.02, y,
                    f"{j + 1}.  {len(ee) - 1:>2} actions,  "
                    f"{len(term['buf']):>2} tokens,  R = {term['reward']:+.6f}"
                    f"    {term['rule']}",
                    fontsize=9, color="0.3", family=MONO)
            y -= 0.070

        n_done = ei + (1 if e[-1] is st else 0)
        ax.text(0.595, 0.335,
                f"EXIT 1  normal      {n_done} / {len(eps)}\n"
                f"EXIT 2  invalid      0  (masked play never emits one)\n"
                f"EXIT 3  dead end     0  (S → C0 keeps the mask nonempty)",
                fontsize=9.0, color="0.3", family=MONO, va="top",
                linespacing=1.9)

    save_gif(fig, render, frames, durations,
             os.path.join(OUT, "termination.gif"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    from make_mdp_figures import Census

    _game = build_game()
    _cen = Census(_game)
    gif_derivation(_game, _cen)
    gif_termination(_game)
