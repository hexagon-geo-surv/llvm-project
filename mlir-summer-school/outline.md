# MLIR Summer School — Module: *Transformations* (3 × 90 min)

**Audience:** students / MLIR beginners. They have already had sessions on compiler
basics (SSA, IR, "what is a pass"), an introduction to MLIR (operations, IR
structure, regions, blocks, values), IR design, and ODS/TableGen. They have
**not** yet written a single line of C++ against the MLIR API.

**Sessions after this module:** dataflow analysis, mem2reg, interfaces,
transform dialect, GPU. We therefore deliberately *point at* (but do not cover)
interfaces internals, dataflow-based passes (SCCP), and mem2reg/SROA.

**The arc of the three sessions:**

| # | Title | One-sentence goal |
|---|-------|-------------------|
| 1 | *Your First Pass* — IR surgery with the C++ API | Change IR by hand: navigate, walk, create, replace, erase — inside a real pass registered in an `opt` tool. |
| 2 | *Rewrite Patterns & Dialect Conversion* | Express rewrites as composable patterns and let drivers (walk / greedy / conversion) do the orchestration. |
| 3 | *The Free Lunch* — Canonicalization, Folding, CSE, DCE | Understand the cleanup transformations every pipeline gets "for free", and plug your own dialect into them. |

The running theme: **each session removes hand-written machinery from the
previous one.** Session 1 writes a rewrite by hand with a manual walk. Session 2
re-expresses it as patterns and lets a driver run it. Session 3 shows that for
many rewrites you don't even write a pass — you attach folders/canonicalization
patterns to your ops and the standard passes pick them up.

---

## Session format & interactivity

Every session follows the same 90-minute shape — the canonical budget all
three decks are timed against:

| Time | What |
|------|------|
| 0:00–0:05 | Warm-up quiz (1 slide, show of hands / sticky notes / polling tool) that recaps the previous session |
| 0:05–0:45 | Lecture (≈40 min core path) with **3–4 embedded quizzes** (predict-the-IR, spot-the-bug, "will this terminate?") and 2–3 **live `mlir-opt` demos** |
| 0:45–0:50 | Exercise briefing (task, checkpoints, where the hints are) |
| 0:50–1:20 | Hands-on exercise (30 min). Instructors circulate. Checkpoints let fast students go on to stretch goals and slow students still reach a working state. |
| 1:20–1:30 | Solution walkthrough (live-code or prepared diff) + wrap-up + teaser for next session |

Each deck's lecture has a ≈40-minute **core path**, and slides marked ⏱ are
the pressure-release valves — skipped in a deck-specified order when a session
runs behind, presented when it runs ahead; decks also distinguish 🔴 live
demos (run them) from 📸 pre-captured output (show, don't run).

Interactivity principles used throughout the decks:

- **Predict-then-run.** Show input IR, ask the room what a pass produces, *then*
  run the real `mlir-opt` live. Wrong guesses are the teaching moments.
- **Spot-the-bug.** Show realistic-but-broken pass/pattern code (the bugs are
  the ones beginners actually write: erasing during iteration, mutating on the
  failure path, bypassing the rewriter).
- **Real-bug provenance.** Several ⏱ flex quizzes are distilled from actual
  upstream bug-fix commits (crashes on block arguments, phantom pattern
  success, unmaterializable fold results, dropped fastmath flags, region ops
  DCE'd together with their side effects): the slides show the pre-fix code or
  behavior, and the speaker notes cite the fixing commits and the in-tree
  guardrails that exist because of each mistake.
- **Every quiz has its answer + explanation in the speaker notes**, and quiz
  slides are followed by an answer slide so decks are self-contained handouts.
- **Exercises are checkpointed.** Each exercise part comes with provided
  FileCheck tests, and the task sheets map tests to checkpoints (some
  checkpoints share a test file; Exercise 1's checkpoint 1 is verified by eye
  from its stderr prints), so students always know whether they are on track.

---

## Exercise infrastructure (shared by all three sessions)

One self-contained out-of-tree MLIR project, `mlir-summer-school/exercises/`,
used in all three sessions (modeled on `mlir/examples/standalone`):

- A tiny **`school` dialect** (defined with ODS — connects to their previous
  session): `school.max` (signed max of two `i32`) and `school.mac`
  (multiply-accumulate, `a*b+c`). The dialect is *deliberately imperfect* in the
  starter state (no `Pure` trait, no folders, no canonicalization patterns) —
  fixing that **is** Exercise 3.
- A **`school-opt`** tool (like `mlir-opt`, with the school dialect and the
  exercise passes registered).
- **Pass stubs with `TODO(exercise N, step K)` markers** for each exercise. The
  starter always compiles; students fill in bodies.
- **FileCheck/lit tests per exercise** (`test/exercise1/`, …), with the task
  sheets mapping tests to checkpoints. Students run a single command to check
  their progress.
- **`solutions/`** directory mirroring the files students edit (reference
  solutions, verified to build and pass all tests).

Setup (verified against this repo's prebuilt tree): students receive a prebuilt
LLVM/MLIR (or build one before the school); the exercise project configures with
`-DMLIR_DIR=<build>/lib/cmake/mlir`. The first build takes a few minutes (it
links the MLIR static libraries); incremental rebuilds take seconds, so the
edit-compile-test loop during the session is fast.

Session 1's exercise only touches `arith` ops, so students who struggled with
the ODS session are not blocked; the custom dialect only becomes central in
Sessions 2b and 3.

---

# Session 1 — *Your First Pass*: IR surgery with the C++ API

## Learning objectives

After this session students can:

1. Explain the C++ object model of MLIR IR (`Operation`, `Region`, `Block`,
   `Value`, `OpResult`, `BlockArgument`) and navigate between these objects.
2. Distinguish generic `Operation *` from generated op classes and move between
   them with `isa`/`dyn_cast`/`cast`.
3. Follow and edit use-def chains (`getDefiningOp`, `getUsers`,
   `replaceAllUsesWith`).
4. Traverse IR with `walk` (including early exit and walk order) and mutate
   safely (collect-then-mutate, erase rules).
5. Create new operations with `OpBuilder` / `OpTy::create`, manage insertion
   points, and propagate locations.
6. Define a pass with TableGen (`Passes.td`, `GEN_PASS_DEF`), implement
   `runOnOperation`, register it in an `opt` tool.
7. Compose and run pass pipelines (`-pass-pipeline`, nesting, `addNestedPass`)
   and use the standard debugging flags.

## Lecture content (≈40 min core path + ⏱ flex slides)

1. **Where we are** (2', inside the 0:00–0:05 warm-up block). Recap of the
   school so far; today: *changing* IR. The three-session roadmap slide.
2. **Warm-up quiz** (3', closes the warm-up block). An annotated snippet (an
   `arith.muli` carrying an attribute, plus an `scf.for` with a region):
   "point at: an operation, a value, a block argument, an attribute, a
   region." Recaps previous sessions in their own vocabulary.
3. **The C++ view of IR** (5'; the navigation-map slide is ⏱). Everything at
   runtime is `Operation *`; ops own
   regions → regions own blocks → blocks own ops (the recursive structure, now
   as a C++ object diagram). `Value` is either an `OpResult` or a
   `BlockArgument`. Key navigation calls on one "map" slide
   (`getParentOp`, `getBlock`, `getOps<T>()`, `getOperand`/`getResult`, …).
4. **Generic vs. generated ops** (4'). `arith::AddIOp` as a typed "view" over
   `Operation *`; `isa<>`/`dyn_cast<>`; ODS-generated named accessors
   (`getLhs()`) vs. generic `getOperand(0)`. When to use which (generic code vs.
   op-specific code). **Flex quiz (from a real upstream crash):** `isa<>` on
   `getDefiningOp()` segfaults on block arguments — the null-safe templated
   `getDefiningOp<OpTy>()` is the idiom.
5. **Use-def chains** (7' incl. quiz). `getDefiningOp`, `getUsers()`/`getUses()`,
   `hasOneUse`, `use_empty`; `replaceAllUsesWith`. Diagram of the use-list.
   **Quiz:** given a 4-op snippet, "after `%x.replaceAllUsesWith(%y)`, which ops
   changed? Can we erase the definer of `%x` now?"
6. **Walking the IR** (6' incl. 🔴 live demo; the manual-traversal and
   `WalkResult` slides are ⏱). Manual nested loops vs. `op->walk(...)`; default
   post-order; `WalkResult::interrupt()`/`advance()`/`skip()`; pre-order walks;
   *what you may and may not mutate during a walk* (erasing the current op in a
   post-order walk is fine; erasing not-yet-visited ops is not; when in doubt:
   collect-then-mutate).
7. **Creating ops: `OpBuilder`** (3'; the `InsertionGuard` and
   constants/`Location`-discipline slides are ⏱). Builder = insertion point +
   context;
   `OpTy::create(builder, loc, ...)` (note: `builder.create<OpTy>` is the *old*,
   now-deprecated spelling — students will see it in old blog posts); creating
   constants; attribute/type getters; `Location` discipline (reuse the location
   of the op you're replacing — never `UnknownLoc` if you can help it);
   `OpBuilder::InsertionGuard`.
8. **Worked example** (8' incl. spot-the-bug quiz + 🔴 live run). Full ~20-line
   `runOnOperation`: strength-reduce
   `arith.muli %x, c` (c a power of two) → `arith.shli` (code verified to
   compile and run against this checkout). This is a *guided preview of the
   exercise pattern*: match (`matchPattern` + `m_ConstantInt` + `isPowerOf2`) →
   build replacement → RAUW → erase. **Spot-the-bug quiz variant:** the first
   version shown erases an op that still has uses / erases during iteration —
   students find the bug. The real output IR deliberately leaves the old
   constant dead — segue to Session 3 ("who cleans this up?").
9. **What is a pass, mechanically** (4'; the pass-options/statistics,
   `signalPassFailure`, and walk-plus-library-call slides are ⏱). Anatomy:
   `Passes.td` definition →
   generated base class (`GEN_PASS_DEF`) → `runOnOperation()`;
   `OperationPass<func::FuncOp>` vs. any-op passes; `signalPassFailure`; pass
   options and statistics (one slide, exercise stretch goal uses a statistic);
   the IR must verify after your pass. **Flex quiz:** a rewrite that creates
   its replacement ops at the block terminator — RAUW rewires a user *above*
   the cursor, and the after-pass verifier reports *operand does not dominate
   this use* (insertion-point discipline; the verifier safety net, made
   concrete).
10. **Pass manager & pipelines** (⏱ flex block, 0' core — presented as time
    allows). Nesting tree diagram
    (`PassManager` rooted at `builtin.module`, `addNestedPass<func::FuncOp>`);
    textual pipelines: `-pass-pipeline="builtin.module(func.func(canonicalize,cse))"`;
    **threading:** the pass manager runs your func pass on all functions in
    parallel → a pass must only touch IR nested under the op it runs on (this is
    why `IsolatedFromAbove` matters). **Quiz:** three pipeline strings — which
    are valid, and what runs on what?
11. **The debugging toolkit** (3' incl. 🔴 live demo; the second toolkit slide
    is ⏱). The flags they will use every day:
    `--mlir-print-ir-before/after(-all)`, `--mlir-print-ir-after-change`,
    `--mlir-timing`, `--mlir-pass-statistics`, `--debug-only=...`, `op->dump()`,
    crash reproducers. Live demo: run the worked example with
    `--mlir-print-ir-after-all`.
12. **Exercise briefing** (the 0:45–0:50 briefing block).

## Exercise 1 (30 min): `-school-strength-reduce`

Rewrite `arith.muli %x, C` where `C` is a power of two into `arith.shli %x,
log2(C)`, as a hand-written walk inside a provided pass stub.

- **Checkpoint 1:** walk + match: print (via `llvm::errs()`) every `arith.muli`
  whose RHS is a constant power of two. Verified by eye from the stderr prints
  (no lit test for this checkpoint). (Teaches: walk, `dyn_cast`, matching a
  constant operand with `matchPattern`/`m_ConstantInt`.)
- **Checkpoint 2:** the rewrite: build the `arith.constant` for the shift
  amount + `arith.shli` with `OpTy::create`, `replaceAllUsesWith`, erase the
  `muli`. Covered by the provided FileCheck test (`strength-reduce.mlir`,
  shared with checkpoint 3).
- **Checkpoint 3:** robustness: don't crash on `muli` without constant RHS, on
  non-power-of-two constants; handle multiple `muli`s in one function
  (collect-then-mutate or erase-current-op-in-post-order-walk).
- **Stretch goals:** (a) count rewrites in the pre-declared `numRewrites` pass
  statistic and see it with `--mlir-pass-statistics`; (b) handle a constant on
  the *left* side too — then
  discuss in the debrief why canonicalization normally makes that unnecessary
  (teaser for Session 3); (c) `muli %x, 1` → just RAUW with `%x`, no new op.

Debrief points: how many re-implemented a worklist / iterated until no change?
How annoying was the manual bookkeeping? → Session 2 motivation.

---

# Session 2 — *Rewrite Patterns & Dialect Conversion*

## Learning objectives

1. Express a rewrite as an `OpRewritePattern` and honor the `matchAndRewrite`
   contract (failure ⇒ no changes; all mutations through the `PatternRewriter`).
2. Use the two main drivers — `walkAndApplyPatterns` (single sweep) and
   `applyPatternsGreedily` (fixpoint) — and explain what the greedy driver does
   under the hood (worklist, folding, DCE) and when it fails to converge.
3. Explain why lowering needs a stronger tool: legality targets and type
   conversion; use `ConversionTarget`, `OpConversionPattern` (adaptors!), and
   `applyPartialConversion`; understand the difference between partial and full
   conversion.
4. Pick the right driver for a job (decision table).

## Lecture content (≈40 min core path + ⏱ flex slides)

1. **Recap & motivation** (1' core; the Exercise-1-solution recap slide is ⏱).
   Show the Exercise-1 solution; list what we
   hand-rolled (matching, traversal order, "did anything change?" bookkeeping,
   composing several rewrites). Patterns = *just the local rewrite*, drivers =
   everything else. Also: patterns are how canonicalization, conversion, and
   most upstream passes are built — one concept, used everywhere.
2. **Pattern anatomy** (9' incl. 🔴 live demo). `OpRewritePattern<arith::MulIOp>`
   version of the
   Session-1 rewrite, side by side with the manual version (it's shorter and
   only contains the interesting part). `matchAndRewrite` returning
   `success()`/`failure()`; **the contract:** if you return failure you must not
   have changed anything; match and rewrite must not be split arbitrarily.
3. **The `PatternRewriter`** (5' incl. spot-the-bug quiz; the full API table is
   ⏱). Why every mutation must go through the
   rewriter: drivers observe mutations (to update their worklist, to roll back
   in conversion). The core API: `replaceOp`, `replaceOpWithNewOp`, `eraseOp`,
   `modifyOpInPlace`, creating ops with the rewriter as builder,
   `notifyMatchFailure` (debuggability!). **Spot-the-bug quiz:** a pattern that
   calls `op->erase()` directly and one that mutates then returns failure.
4. **Pattern sets & benefits** (⏱ flex, 0' core). `RewritePatternSet`,
   `patterns.add<...>`,
   `PatternBenefit` (ordering hint among *matching* patterns), the upstream
   `populateXxxPatterns(...)` convention, `FrozenRewritePatternSet` (why frozen:
   built once, shared across threads).
5. **Driver #1: `walkAndApplyPatterns`** (⏱ flex, 0' core; the driver demo is
   📸 pre-captured). One post-order sweep, applies
   patterns as it goes; cheap, predictable; **no fixpoint, no folding, no DCE,
   modified/new ops are not revisited**; patterns may only erase the matched op
   and IR nested under it. Use when one pass over the IR is enough.
6. **Driver #2: the greedy driver** (4' core incl. 🔴 convergence demo; the
   worklist animation, `GreedyRewriteConfig`, and the termination quiz are ⏱).
   `applyPatternsGreedily`: worklist
   algorithm step-by-step on a 3-slide "animation" (op popped, patterns tried in
   benefit order, new/modified ops pushed back); it *also* folds ops and erases
   trivially dead ops (first meeting with folding — teaser for Session 3);
   fixpoint semantics; `GreedyRewriteConfig` (iteration limits, top-down);
   what `failed(...)` means (did **not** converge — not "nothing matched").
   **Quiz: "will this terminate?"** — pattern set {`A→B`, `B→A`}. Surprise
   answer (verified against the implementation): with default config it
   *hangs* — each rewrite re-enqueues the new op, so the inner worklist never
   empties and the iteration limit (which only bounds *outer* iterations) never
   triggers; `setMaxNumRewrites` is the safety net that turns the hang into a
   clean "did not converge" failure. Lesson: *every pattern must strictly
   reduce something*. **Flex quiz (real upstream bug):** the *phantom
   success* — a pattern that returns `success()` on its nothing-to-do path
   fires no listener events (so no hang), but books progress every iteration:
   10 silent full iterations, then a `failure()` that `-canonicalize` swallows;
   expensive-checks builds abort with *pattern returned success but IR did not
   change*. Every "nothing to do" path must return `failure()`.
7. **Debugging patterns** (⏱ flex, 0' core). `--debug-only=greedy-rewriter` /
   `--debug-only=pattern-application` (📸 pre-captured output): see which
   patterns fire and
   which fail and why (`notifyMatchFailure` messages appear here).
8. **From optimization to lowering** (8' incl. the which-driver-fails quiz).
   New problem statement: *everything*
   from dialect X must go; types may change; failure must be loud. Greedy
   doesn't give us that. Concepts: **legality** (`ConversionTarget`), partial
   vs. full conversion (exact semantics).
9. **Conversion mechanics** (11' incl. adaptor quiz + 🔴 legalization-trace
   demo). Reading the "failed to legalize" error;
   `OpConversionPattern<school::MaxOp>`; the
   **adaptor**: `adaptor.getOperands()` gives *already-converted* operands
   (with the type-converted types), while `op` still has the *old* operands —
   the #1 beginner confusion; `ConversionPatternRewriter` dos and don'ts (create
   and replace through the rewriter; don't walk around and inspect neighboring
   IR and expect it to be converted). Full `runOnOperation` for a small
   conversion pass on a slide: build target + patterns + `applyPartialConversion`.
   **Flex quiz:** what the (real, current) `complex.neg` lowering silently
   loses — `fastmath` flags; created ops carry only what you pass, and neither
   dropping all flags nor forwarding them all is right (attribute propagation
   is a per-op decision; the over-forwarding direction broke `complex.abs`
   upstream).
10. **Type conversion, gently** (⏱ flex, 0' core; cast examples are 📸
    pre-captured). `TypeConverter::addConversion`;
    where materializations come from and what `builtin.unrealized_conversion_cast`
    is (glue at the boundary between converted and unconverted IR); function
    signature conversion exists as a helper (pointer only). Keep at concept
    level — beginners need the mental model, not the full API.
11. **Choosing a driver** (2'). Decision table: walk / greedy / conversion ×
    (fixpoint? types change? completeness guarantee? rollback?).
12. **Exercise briefing** (the 0:45–0:50 briefing block).

## Exercise 2 (30 min): patterns, then a real lowering

**Part A — patterns + greedy (≈15').** Port Session 1's rewrite to an
`OpRewritePattern`, and add a second pattern
`shli(shli(x, c1), c2) → shli(x, c1+c2)`. Run both with
`applyPatternsGreedily` in the provided `-school-peephole` pass stub. The test
input contains `((x*4)*8)` chains, so the two patterns *compose to a fixpoint*
that neither achieves alone — students see the greedy driver earn its keep.
Checkpoints: (1) pattern A fires; (2) pattern B fires (happy path); (3)
composed chain fully reduced to a single `shli`. *If time / stretch-track:*
the overflow guard on pattern B — no merge when `c1+c2` reaches the bit width,
that shift would be poison (tested by `@no_merge_overflow`, which may stay red
until then).

**Part B — dialect conversion (≈15').** Implement `-convert-school-to-arith`:
`school.max` → `arith.cmpi sgt` + `arith.select`, with `ConversionTarget`
marking the school dialect illegal / `arith` legal, via
`applyPartialConversion`. Checkpoints: (1) conversion succeeds on max-only
input; (2) run the provided input containing `school.mac` **before** writing a
pattern for it → read and understand the "failed to legalize" error; (3, *if
time / stretch-track*) add the `school.mac` → `muli`+`addi` pattern.
**Stretch:** switch to `applyFullConversion` and explain the difference on an
input with an unknown op; temporarily *invert* the max pattern's guard
(pretend `i32` is unsupported) to see `notifyMatchFailure` refuse the match,
and observe the driver's error under `--debug-only=dialect-conversion`.

Debrief: show the solution; emphasize how little code the conversion pass is;
tease Session 3 with "you wrote `x*1→x` as a pattern — upstream this is not
even a pattern, it's a *folder*, and you get it for free."

---

# Session 3 — *The Free Lunch*: Canonicalization, Folding, CSE, DCE

This session is more conceptual than 1–2, so it leans harder on
predict-then-run quizzes; the coding exercise is smaller but ties the whole
module together (and reconnects to their ODS session).

**Format deviation:** Session 3 deliberately opens with a *predict-then-run*
quiz instead of a pure recap quiz — the recap is a 30-second Session-2 bridge
question ("which Session-2 driver do you bet powers `-canonicalize`?") asked
before predictions are collected.

## Learning objectives

1. Explain what `-canonicalize` and `-cse` actually do, and their limits.
2. Write `fold` implementations honoring the fold contract, and explain
   folding vs. rewrite patterns (when to use which).
3. Explain how side-effect modeling (`Pure`, `MemoryEffects`) gates DCE and
   CSE, and why unannotated ops are conservatively kept.
4. Attach canonicalization patterns to ops via ODS hooks so that *upstream*
   passes optimize *your* dialect.
5. Name the other core passes (inliner, SCCP, LICM, mem2reg, SROA, symbol-dce)
   and know where cleanup passes sit in real pipelines.

## Lecture content (≈40 min core path + ⏱ flex slides)

1. **Opening predict-then-run quiz** (3' in the 0:00–0:05 warm-up slot + 4'
   for the two 🔴 live reveal runs that open the lecture). ~8 lines of IR
   containing: two
   identical `arith.addi`, an `addi %x, 0`, a dead pure op, a dead
   `memref.store`-like op, constant-foldable ops. "What comes out of
   `-canonicalize -cse`?" Collect predictions (after the 30-second Session-2
   bridge question), run live, count surprises. Each
   surprise is a section of this lecture.
2. **The cleanup stack** (2'). One diagram: folding → canonicalization
   patterns → DCE → region simplification → CSE; which pass runs which; the key
   economic argument: *implement op hooks once; every pass in every pipeline
   benefits* (and: this is why upstream reviewers insist on folders).
3. **Folding** (9'). The most restricted, most-used rewrite: `fold` may return
   (a) an existing `Value`, (b) an `Attribute` (constant), or (c) nothing — it
   may **not** create ops or mutate IR. `fold(FoldAdaptor)` and what the adaptor
   contains (an `Attribute` for each constant operand, null otherwise). Real
   `arith` folders on slides (`addi(x,0)→x`; constant-constant via APInt).
   Where folding runs: the greedy driver (hence `-canonicalize`),
   `createOrFold`, … — "folders run *everywhere*, so they must be fast and
   always-correct". **Flex quizzes (both from real upstream crashes):** (a) a
   null-checked `FoldAdaptor` folder that still crashes — `ub.poison` /
   `dense_resource` deliver *non-null* attributes of an unexpected class
   (`dyn_cast_if_present` covers both traps in one call); (b) `subi(x,x) → 0`
   on `tensor<?xi32>` — an `Attribute` fold result claims materializability as
   one constant of exactly the result type, which a dynamic shape can't
   satisfy.
4. **Constants & materialization** (2'; the dedup/hoisting slide is ⏱). Why
   returning an `Attribute` is
   enough: `materializeConstant` dialect hook turns it back into an op;
   `ConstantLike`; constant dedup/hoisting by the folding infrastructure. (This
   answers "who creates the `arith.constant` I never built?")
5. **Quiz: legal fold or not?** (3'). Four candidates: `addi(x,0)→x` ✓;
   `muli(x,2)→shli(x,1)` ✗ (creates an op → pattern); "fold that swaps operands
   in place" (in-place folds — allowed via returning `getResult()`); "fold that
   looks at a value's *other* users" ✗ (not local, and fold must be cheap).
6. **Canonicalization** (6'; the `hasCanonicalizeMethod` slide is ⏱). What
   "canonical form" buys: every pass downstream
   matches *one* form, not N (recall Exercise 1 stretch goal: constant-on-left).
   ODS hooks: `hasCanonicalizer`/`getCanonicalizationPatterns` and
   `hasCanonicalizeMethod`; a real short upstream canonicalization pattern.
   Design rules (quoting `docs/Canonicalization.md`): patterns must converge
   ("unstable or cyclic rewrites are considered a bug"); prefer `fold` when
   possible; prefer forms with fewer uses of a value (the docs explicitly bless
   `x + x → x * 2`); no expensive/O(n) patterns or cost models; canonicalization
   is about *canonical*, not *optimal* — pipelines must stay correct with every
   canonicalize pass removed. **Discussion quiz:** "is `addi(x,x) → muli(x,2)`
   a good canonicalization? is `muli(x,4) → shli(x,2)`?" (first: yes, per the
   fewer-uses guideline; second: debatable target-dependent strength reduction —
   the point is *pick one form globally and don't optimize prematurely*).
7. **The canonicalize pass is tiny** (⏱ flex, 0' core). Show (essentially all
   of)
   `Canonicalizer.cpp`: collect every registered canonicalization pattern +
   run greedy driver. Everything they learned in Session 2, reused. Pass options
   (`top-down`, `region-simplify`, iteration limits) — also ⏱.
8. **DCE & side effects** (7' incl. quiz; the per-operand-effects slide is ⏱).
   `isOpTriviallyDead` = no users **and** no
   observable effects; how MLIR knows: `MemoryEffectsOpInterface`, the `Pure`
   trait in ODS (= `NoMemoryEffect` + `AlwaysSpeculatable`),
   `RecursiveMemoryEffects`; **unknown ops are conservatively assumed to have
   effects** (safety default!). Where DCE runs: built into the greedy driver
   (→ `-canonicalize`), built into `-cse`, and the (new) dedicated
   `-trivial-dce` pass. `memref.load` vs `memref.store` as the canonical
   example; fun subtlety: a dead `memref.alloc` *is* removed (an Allocate
   effect on the op's own result doesn't keep it alive).
   **Quiz:** 5 ops — which can be erased? (dead pure op ✓, dead load ✓(with the
   right effects), store ✗, dead call ✗/depends, unregistered op ✗).
   **Flex quiz (real upstream miscompile):** a `linalg.map` whose body prints
   was fully DCE'd when effects were computed from operands only — region
   bodies are consulted solely under `RecursiveMemoryEffects`; leaving the
   trait off is either a miscompile (incomplete effect interface) or a
   pessimization (nothing declared ⇒ unknown ⇒ conservative).
9. **Region simplification** (3' incl. 🔴 live demo). Unreachable block
   elimination, dead
   block-argument elimination, identical block merging; part of `-canonicalize`.
   Note the verified default: the pass option `region-simplify` defaults to
   `normal` — block merging only happens with
   `-canonicalize="region-simplify=aggressive"` (nice predict-then-run demo).
10. **CSE** (3' incl. quiz; the dominance-algorithm and 📸 pass-statistics
    slides are ⏱). Idea + algorithm sketch: structural equivalence (op name,
    attributes, operands, result types) + dominance-scoped hash table (walk the
    dominator tree, reuse the first equivalent op that dominates); eligibility:
    side-effect-free ops (again `Pure` pays off), plus a restricted same-block
    regime for read-only ops. Why `-cse` is a separate pass and not a pattern
    (needs global dominance context, not a local rewrite).
    **Quiz:** which merge? (identical constants ✓; identical `memref.alloc` ✗ —
    distinct buffers!; two identical loads — only with no side-effecting op in
    between; commutative twins `addi(a,b)`/`addi(b,a)` ✓; same op with
    different attributes ✗.)
11. **The wider zoo & real pipelines** (1' core — the pipelines slide; the zoo
    map, `-remove-dead-values`, `-symbol-dce`, and LICM-payoff slides are ⏱).
    One map slide: inliner, SCCP
    (→ dataflow session), LICM, mem2reg/SROA (→ later session), symbol-dce (why:
    symbols aren't SSA). One realistic pipeline slide showing
    canonicalize/CSE interleaved between lowering stages, plus the pragmatic
    idiom: *emit naive IR from your pass; let canonicalize clean up; don't
    re-implement folding inside your pass* (`createOrFold` if you must).
12. **Exercise briefing** (the 0:45–0:50 briefing block).

## Exercise 3 (30 min): make the `school` dialect a good citizen

Students edit the dialect from Exercise 2 (ODS + C++ — connects back to their
ODS session):

- **Checkpoint 1 — DCE and CSE need effects:** run the provided test: a dead
  `school.max` is *not* removed by `-canonicalize`, and `-cse` does not merge
  two identical ones. Fix: add `Pure` to the op definition in ODS. Re-run: both
  work. (Visceral demonstration of effect-gating; ~5 min.)
- **Checkpoint 2 — folder:** `let hasFolder = 1` + implement `MaxOp::fold`:
  `max(x,x) → x`. Watch `-canonicalize` (which they now know is just the greedy
  driver) pick it up.
- **Checkpoint 3 — constant folding + materialization:** extend the folder to
  constant-fold `max(c1,c2)` via the adaptor's attributes… and observe that
  *nothing happens*. Designed teaching beat (the classic custom-dialect bug):
  attribute fold results are silently dropped unless the dialect implements
  `materializeConstant`. Fix: `let hasConstantMaterializer = 1` + materialize
  via `arith::ConstantOp`. Re-run: works.
- **Stretch:** (a) add the `Commutative` trait and watch `max(c, x) → max(x, c)`
  happen *for free* (trait folding moves constants right — this is why upstream
  folders only check the RHS); (b) reassociation
  `max(max(x, c1), c2) → max(x, c3)` as a real canonicalization pattern via
  `let hasCanonicalizer = 1` + `getCanonicalizationPatterns` (Session 2 skill,
  now hooked into `-canonicalize`) — relies on (a)'s normal form, which is
  exactly the point of canonical forms; (c) discuss folder candidates for
  `school.mac` (`mac(a, b, 0)`? `mac(a, 1, c)`? — which are folds, which are
  patterns?).

Wrap-up of the module (5', inside the 1:20–1:30 walkthrough block): the
three-session arc on one slide — manual
surgery → patterns+drivers → op hooks; "you now know how ~80% of upstream MLIR
transformation code is structured"; pointers: interfaces session (how
effects/inlining really work), dataflow (SCCP), transform dialect (orchestrating
patterns from IR).

---

## Deliverables in this directory

| Path | Content |
|------|---------|
| `outline.md` | This document |
| `slides/lecture1-passes.md` | Slide deck 1 (Marp markdown, speaker notes included) |
| `slides/lecture2-patterns.md` | Slide deck 2 |
| `slides/lecture3-canonicalization.md` | Slide deck 3 |
| `exercises/` | The `school` exercise project (starter + tests + solutions) |
| `exercises/exercise{1,2,3}.md` | Student-facing task sheets |

Slides are Marp markdown: render with
`npx @marp-team/marp-cli slides/lecture1-passes.md --pptx` (or `--pdf`, or
present directly from VS Code with the Marp extension). Every slide carries
speaker notes (HTML comments) with quiz answers and demo commands.
