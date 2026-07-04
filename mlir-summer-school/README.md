# MLIR Summer School — Module: *Transformations*

Teaching material for three 90-minute sessions on MLIR transformations, aimed
at students / MLIR beginners who have already seen compiler basics, the MLIR IR
structure, and ODS/TableGen.

| Session | Title | Deck | Exercise |
|---|---|---|---|
| 1 | *Your First Pass* — IR surgery with the C++ API | `slides/lecture1-passes.md` | `exercises/exercise1.md` |
| 2 | *Rewrite Patterns & Dialect Conversion* | `slides/lecture2-patterns.md` | `exercises/exercise2.md` |
| 3 | *The Free Lunch* — Canonicalization, Folding, CSE, DCE | `slides/lecture3-canonicalization.md` | `exercises/exercise3.md` |

Start with **`outline.md`** — it contains the module goals, the pedagogical arc,
per-session learning objectives, minute-level content plans, all quizzes, and
the exercise checkpoint design.

## Slides

The decks are [Marp](https://marp.app/) markdown: one file per session, one
slide per `---` section, with full speaker notes (including quiz answers and
exact live-demo commands) in HTML comments on every slide. Each deck has a
≈40-minute lecture core path; slides marked ⏱ are flex slides (skipped in a
deck-specified order when running behind), and demos are marked 🔴 (run live)
or 📸 (pre-captured output — shown, not run).

Render to PowerPoint / PDF / HTML:

```bash
npx @marp-team/marp-cli slides/lecture1-passes.md --pptx   # or --pdf / --html
```

or present directly from VS Code with the *Marp for VS Code* extension.

The decks were written against `llvm-project` HEAD (July 2026): all C++ uses
the current APIs (`OpTy::create(builder, ...)`, `applyPatternsGreedily`,
`modifyOpInPlace`, …) and all before/after IR shown on slides is real output
captured from this checkout's `mlir-opt`.

## Exercises

`exercises/` is a self-contained out-of-tree MLIR project (a small `school`
dialect + a `school-opt` tool + lit/FileCheck tests) used by all three
sessions. See `exercises/README.md` for setup, and `exercises/exerciseN.md`
for the student task sheets. Reference solutions live in
`exercises/solutions/` and can be applied per session with
`exercises/apply-solution.sh`.

The project builds against any MLIR build or install tree
(`-DMLIR_DIR=<prefix>/lib/cmake/mlir`); a plain Release+assertions build of
LLVM/MLIR is recommended for the class. Both the starter state and all
solution states are verified to build; the starter's exercise tests are red
by design, and after `apply-solution.sh all` the whole suite is green.

## Demos

The demos in the decks — both the 🔴 live ones and the 📸 pre-captured
outputs — run against a prebuilt `mlir-opt`: built with
`MLIR_INCLUDE_TESTS=ON` (the in-tree default) for the test-pass demos
(`--test-print-nesting` / `--test-print-defuse` in Session 1, and the
pre-captured `--test-walk-pattern-rewrite-driver` output in Session 2) and
with assertions enabled for the `--debug-only=...` demos/captures in
Session 2. One exception: Session 1's
worked-example demo loads a small pass plugin (`mul_to_shift.so`) compiled
beforehand — the build command is in that slide's speaker notes. Demo inputs
are either on the slides or reference files under `mlir/test/` in the
monorepo.
