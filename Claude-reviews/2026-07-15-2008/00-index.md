# Correctness review + test reorganization — index

*Session 2026-07-15. Scope: whole-codebase correctness review of `sraz` and a
component-oriented reorganization of the test suite.*

## What was done

1. **Correctness review** (adversarial, with executable repros): 8 confirmed
   findings — 3 medium, 5 low. See [01-correctness-findings](01-correctness-findings.md).
2. **Fixed the 3 medium bugs** with regression tests (user-approved scope).
3. **Reorganized the tests** into subfolders mirroring `src/`, split the mixed
   files, and added a navigation index (`tests/README.md`). See
   [02-test-reorganization](02-test-reorganization.md).

## Method

A fan-out review (52 agents over 18 component/focus areas) proposed findings;
each was independently verified by two skeptics, one of which attempted an
executable repro in a throwaway venv. Only findings that survived verification
are reported here (0 were refuted). Baseline and final suites were run in a
project-local `.venv` (CPU torch).

## Bottom line

- Baseline: 257 passed, 1 skipped. Final: **263 passed, 1 skipped** (+6
  regression tests), fully green.
- The three medium bugs were all latent for the *shipped* seed-42 sequential
  symreg run **except** the reproducibility gap (#3), which affected the real
  driver. After the fix, two identical `run_symreg.py --seed 42` invocations are
  byte-identical; before, MCTS exploration noise was drawn from an unseeded
  process-global `np.random`.
- The 5 low-severity findings are reported, not changed (per approved scope);
  one (#4) is currently pinned by a test as intended behavior.
