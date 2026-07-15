"""Symbolic-regression grammar game.

Port of the AlphaGrammar symbolic-regression game
(AlphaZero_PP: archive/nsai_experiments/new_games_old_engine/{srgame,grammargame}.py)
onto the sraz single-player AlphaZero engine.

Two layers:
- a grammar-derivation MDP: the state is a fixed-length token buffer holding a
  sentential form; an action picks (position, production) and splices the
  production's RHS over the nonterminal at that position; the episode ends when
  no nonterminals remain.
- an SR evaluator: the finished sentence is a prefix-notation expression whose
  C* tokens are free constants; it is converted to infix, parsed by sympy, and
  its constants fitted to the target data with lmfit. Terminal reward = R² of
  the fit, clipped to [-1, 1]. Intermediate reward is 0.
"""

from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces
from sympy.parsing import sympy_parser
import sympy
import lmfit

from sraz.core.game import Game


# ---------------------------------------------------------------------------
# Prefix -> infix conversion (ported verbatim from srgame.py)
# ---------------------------------------------------------------------------

operators = ['/', '*', '+', '-', '**']
functions = ['cos', 'sin']


def prefix_to_infix(tokens: list[str]) -> str:
    """Convert a prefix token list to a parenthesized infix string."""
    stack = []
    for element in reversed(tokens):
        if element in operators:
            first = stack.pop()
            second = stack.pop()
            stack.append("(" + first + element + second + ")")
        elif element in functions:
            first = stack.pop()
            stack.append(element + "(" + first + ")")
        elif element != "":
            stack.append(element)
    return stack.pop()


# ---------------------------------------------------------------------------
# Grammar compilation (replaces nltk.CFG.fromstring in the original)
# ---------------------------------------------------------------------------

# The SR grammar from srgame.py. In the original grammar string C3/C4 were
# accidentally unquoted (nonterminals with no productions); here every RHS
# entry that has no productions of its own is simply a terminal.
SR_GRAMMAR = {
    "S": [
        ["+", "S", "S"],
        ["C0"],
        ["*", "C1", "x"],
        ["*", "C2", "*", "x", "x"],
        ["*", "S", "S"],
        ["/", "S", "S"],
        ["*", "C3", "sin", "*", "C4", "x"],
    ],
}


@dataclass
class Grammar:
    start: str
    symdict: dict[str, int] = field(default_factory=dict)      # symbol -> token
    tokenlist: list[str] = field(default_factory=list)         # token -> symbol
    nonterms: set[int] = field(default_factory=set)            # tokens with productions
    productions: list[list[int]] = field(default_factory=list) # global tokenized RHS list
    proddict: dict[int, list[int]] = field(default_factory=dict)  # nonterm token -> global prod indices
    nsym: int = 0
    pad_tok: int = 0
    nprods: int = 0


def compile_grammar(rules: dict[str, list[list[str]]], start: str = "S") -> Grammar:
    """Build the token/production tables the derivation MDP runs on.

    Symbol numbering follows the original init_grammar: walk productions in
    order, assigning indices to the LHS first, then to unseen RHS symbols.
    """
    g = Grammar(start=start)

    def intern(sym: str) -> int:
        if sym not in g.symdict:
            g.symdict[sym] = len(g.tokenlist)
            g.tokenlist.append(sym)
        return g.symdict[sym]

    for lhs, rhs_list in rules.items():
        lhs_tok = intern(lhs)
        g.nonterms.add(lhs_tok)
        g.proddict.setdefault(lhs_tok, [])
        for rhs in rhs_list:
            g.proddict[lhs_tok].append(len(g.productions))
            g.productions.append([intern(s) for s in rhs])

    g.nsym = len(g.tokenlist)
    g.pad_tok = g.nsym
    g.nprods = len(g.productions)
    return g


# ---------------------------------------------------------------------------
# SR evaluator (ported from SRGameEnv.fit / prefix_str_to_py_fn)
# ---------------------------------------------------------------------------

C_MIN, C_MAX = 1.0, 4.0
C_INIT = 0.5 * (C_MIN + C_MAX)
X_MIN, X_MAX, N_X = 1.0, 3.0, 41

# Ground truth from srgame.py's generate_new_function:  4*sin(4*x) + C0 + C1*x + C2*x**2
TARGET_INFIX = "4*sin(4*x) + C0 + C1*x + C2*x**2"


def target_ys(xs: np.ndarray, constants: dict[str, float]) -> np.ndarray:
    return (4.0 * np.sin(4.0 * xs)
            + constants["C0"] + constants["C1"] * xs + constants["C2"] * xs ** 2)


# ---------------------------------------------------------------------------
# Additive "best-corner" instance: a pruned, division/sine-free grammar whose
# expressions are sums of monomials, and a target that is exactly expressible.
#
# Because the reward is R^2 of a least-squares fit, adding a term = adding a
# free parameter, which can only improve or equal the fit -> the reward
# gradient points monotonically toward richer sums (non-deceptive), and with
# no division/sine there are no -1 fit failures, so partial structures score
# cleanly under weak completion (informative). See
# Claude-research/02-informativeness-and-deception.md.
# ---------------------------------------------------------------------------

ADDITIVE_GRAMMAR = {
    "S": [
        ["+", "S", "S"],        # compose by addition
        ["C0"],                 # constant
        ["*", "C1", "x"],       # linear
        ["*", "C2", "*", "x", "x"],  # quadratic
    ],
}

ADDITIVE_INFIX = "C0 + C1*x + C2*x**2"


def target_ys_additive(xs: np.ndarray, constants: dict[str, float]) -> np.ndarray:
    return (constants["C0"] + constants["C1"] * xs + constants["C2"] * xs ** 2)


def fit_expression(rule: str, xs: np.ndarray, exact_ys: np.ndarray) -> float:
    """Fit the C* constants of a prefix expression to the data; return clipped R².

    Any failure (unparseable expression, failed lmfit solve, non-finite result)
    yields -1.0 rather than raising — the original crashed on those paths.
    """
    r2 = -1.0
    try:
        tokens = rule.strip().split()
        param_values = {}
        ind_vars = ()
        if "x" in tokens:
            param_values["x"] = xs
            ind_vars = ("x",)
        for term in tokens:
            if "C" in term:
                param_values[term] = C_INIT
        infix = prefix_to_infix(tokens)
        model = sympy_parser.parse_expr(infix, evaluate=False)
        fn = sympy.lambdify(list(model.free_symbols), model)
        lm_mod = lmfit.Model(fn, independent_vars=ind_vars)
        res = lm_mod.fit(data=exact_ys, **param_values)
        if res.success:
            ss_res = np.sum((res.data - res.best_fit) ** 2)
            ss_tot = np.sum((res.data - np.mean(res.data)) ** 2)
            r2 = 1.0 - ss_res / ss_tot
        if not np.isfinite(r2):
            r2 = -1.0
    except Exception:
        r2 = -1.0
    # sign-preserving |r2|^p with p=1, as in the original, then clipped so that
    # pathological division expressions cannot poison the value target
    return float(np.clip(r2, -1.0, 1.0))


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------

class SymRegGame(Game[np.ndarray, int]):
    """Grammar-derivation MDP with terminal SR reward.

    Observation: MultiDiscrete token buffer of length `max_len` (pad token
    included in the space). Action: Discrete(max_len * nprods), decoded as
    (position, production). Note the action mask forbids derivations that
    would reach exactly `max_len` tokens (a faithful `>=` from the original),
    so finished expressions have at most `max_len - 1` tokens.
    """

    def __init__(self, max_len: int = 15, redraw_constants: bool = False,
                 problem_seed: int = 0,
                 grammar_rules: dict | None = None,
                 target_ys_fn=None,
                 constant_names: tuple | None = None,
                 target_infix: str | None = None,
                 x_min: float = X_MIN, x_max: float = X_MAX, n_x: int = N_X):
        super().__init__()
        # (grammar, target) default to the sine instance, so the shipped game is
        # unchanged; pass them to build any other instance (see problems.py).
        self.grammar = compile_grammar(grammar_rules if grammar_rules is not None
                                       else SR_GRAMMAR)
        self._target_ys_fn = target_ys_fn if target_ys_fn is not None else target_ys
        self.constant_names = (tuple(constant_names) if constant_names is not None
                               else ("C0", "C1", "C2"))
        self.target_infix = target_infix if target_infix is not None else TARGET_INFIX
        g = self.grammar
        self.state_len = max_len
        self.observation_space = spaces.MultiDiscrete([g.nsym + 1] * max_len)
        self.action_space = spaces.Discrete(max_len * g.nprods)

        self.redraw_constants = redraw_constants
        self.problem_seed = problem_seed
        self.xs = np.linspace(x_min, x_max, n_x)
        self._redraw_rng = np.random.default_rng(problem_seed)
        self.constants = self._draw_constants(np.random.default_rng(problem_seed))
        self.exact_ys = self._target_ys_fn(self.xs, self.constants)
        self._fit_cache: dict[str, float] = {}

        self.state = None
        self.real_state_len = 0
        self.reset_wrapper()

    def _draw_constants(self, rng: np.random.Generator) -> dict[str, float]:
        return {name: float(C_MIN + (C_MAX - C_MIN) * rng.random())
                for name in self.constant_names}

    def reset(self, seed=None):
        g = self.grammar
        self.state = np.full(self.state_len, g.pad_tok, dtype=np.int64)
        self.state[0] = g.symdict[g.start]
        self.real_state_len = 1
        if self.redraw_constants:
            rng = np.random.default_rng(seed) if seed is not None else self._redraw_rng
            self.constants = self._draw_constants(rng)
            self.exact_ys = self._target_ys_fn(self.xs, self.constants)
            self._fit_cache = {}
        return self.state.copy(), {}

    def get_action_mask(self) -> np.ndarray:
        g = self.grammar
        mask = np.zeros((self.state_len, g.nprods), dtype=bool)
        ll = self.real_state_len
        for i in range(ll):
            tok = int(self.state[i])
            if tok in g.nonterms:
                for j in g.proddict[tok]:
                    if ll + len(g.productions[j]) - 1 < self.state_len:
                        mask[i, j] = True
        return mask.flatten()

    def step(self, action: int):
        g = self.grammar
        pos, prod = divmod(int(action), g.nprods)
        # Defensive: masked play never reaches these, but fail closed if it does
        # (the original either crashed on overflow or no-op'd, risking loops).
        invalid = (pos >= self.real_state_len
                   or int(self.state[pos]) not in g.nonterms
                   or prod not in g.proddict[int(self.state[pos])])
        rhs = g.productions[prod]
        new_len = self.real_state_len + len(rhs) - 1
        if invalid or new_len > self.state_len:
            return self.state.copy(), -1.0, True, False, {"rule": None}

        tail = self.state[pos + 1:self.real_state_len].copy()
        self.state[pos:pos + len(rhs)] = rhs
        self.state[pos + len(rhs):pos + len(rhs) + len(tail)] = tail
        self.state[new_len:] = g.pad_tok
        self.real_state_len = new_len

        terminated = not self._has_nonterms()
        rule = self._decode_state() if terminated else None
        reward = self._fit_cached(rule) if terminated else 0.0
        return self.state.copy(), reward, terminated, False, {"rule": rule}

    def _has_nonterms(self) -> bool:
        g = self.grammar
        return any(int(t) in g.nonterms for t in self.state[:self.real_state_len])

    def _decode_state(self) -> str:
        g = self.grammar
        return " ".join(g.tokenlist[int(t)] for t in self.state[:self.real_state_len])

    def _fit_cached(self, rule: str) -> float:
        if rule not in self._fit_cache:
            self._fit_cache[rule] = fit_expression(rule, self.xs, self.exact_ys)
        return self._fit_cache[rule]

    # MCTS stashes/unstashes once per simulation; the default deepcopy(self)
    # would copy the fit cache and data arrays every time. Only the derivation
    # state and the Game bookkeeping fields actually change between stash and
    # unstash — the cache deliberately survives (it only ever grows, and its
    # entries stay valid for the current constants).
    def stash_state(self):
        return {
            "state": self.state.copy(),
            "real_state_len": self.real_state_len,
            "obs": None if self.obs is None else np.array(self.obs, copy=True),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": None if self.info is None else dict(self.info),
            "step_count": self.step_count,
        }

    def unstash_state(self, stash):
        self.state = stash["state"].copy()
        self.real_state_len = stash["real_state_len"]
        self.obs = None if stash["obs"] is None else np.array(stash["obs"], copy=True)
        self.reward = stash["reward"]
        self.terminated = stash["terminated"]
        self.truncated = stash["truncated"]
        self.info = None if stash["info"] is None else dict(stash["info"])
        self.step_count = stash["step_count"]
        return self
