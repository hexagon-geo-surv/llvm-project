# Website abstracts — Transformations module (3 sessions)

## Session 1 — Your First Pass: Rewriting IR with the MLIR C++ API

You know what MLIR IR looks like — now you'll change it. This session introduces
the C++ view of the IR you've been reading all week: how operations, regions,
blocks, and values are represented at runtime, how to navigate use-def chains,
how to traverse IR with `walk()`, and how to build new operations with
`OpBuilder` — including the rules for mutating IR without pulling the rug out
from under your own iteration. We then wrap these skills into a real compiler
pass: defining it in TableGen, implementing `runOnOperation()`, composing pass
pipelines on the `mlir-opt` command line, and using the debugging flags that
MLIR developers reach for every day.

The session is hands-on: after a guided worked example, you will implement your
own optimization pass — a strength reduction that turns multiplications by
powers of two into shifts — in a prepared out-of-tree project, and validate it
with FileCheck tests exactly the way upstream MLIR does. Expect live demos,
predict-the-output quizzes, and at least one deliberately planted bug for you to
find.

## Session 2 — Rewrite Patterns and Dialect Conversion

Yesterday's pass hand-rolled everything: traversal, matching, bookkeeping.
This session introduces MLIR's pattern infrastructure, which lets you write
just the interesting part — a local rewrite — and leave the orchestration to a
driver. We dissect real upstream patterns, learn the `matchAndRewrite` contract
and why every change must go through the `PatternRewriter`, and study the
greedy pattern rewrite driver: how its worklist reaches a fixpoint, how
patterns compose to achieve rewrites none of them can do alone, and what
happens when they don't converge (spoiler: bring a Ctrl-C). From there we step
up to lowering: dialect conversion adds legality targets and type conversion on
top of patterns, and we walk through a real conversion pass end to end —
including the single most common beginner bug, reading operands from the op
instead of the adaptor.

In the exercise you'll do both halves yourself: first re-express Session 1's
rewrite as patterns and watch the greedy driver compose them into something
neither pattern achieves alone, then write your first real lowering — compiling
a small custom dialect away into standard arithmetic, and learning to read the
"failed to legalize" error you will meet many times in your MLIR career.

## Session 3 — The Free Lunch: Canonicalization, Folding, CSE, and DCE

Every MLIR pipeline is sprinkled with passes nobody writes application code
for: `-canonicalize`, `-cse`, dead code elimination. This session explains
where that "free" cleanup comes from — and what it costs to join in. We cover
folding (the most restricted and most ubiquitous rewrite in MLIR), what makes a
good canonicalization pattern, why the canonicalize pass is only ~20 lines of
code, and how side-effect modeling decides what CSE and DCE may touch: get the
`Pure` trait wrong and the optimizer either ignores your ops — or worse,
deletes the ones that matter. Throughout, we predict what real passes do to
real IR before running them live, using the actual upstream implementation as
our reference.

The exercise closes the loop of the whole module: you make your own dialect
from Session 2 a good citizen by adding effect annotations, folders, and
canonicalization patterns — then watch the *stock* upstream passes optimize
your custom ops without a single new pass being written. Includes one designed
mystery: a fold that silently refuses to happen until you discover the missing
piece.
