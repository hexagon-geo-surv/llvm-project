---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 25px; }
  section pre { font-size: 0.6em; line-height: 1.2; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8em; }
  section.lead { text-align: center; }
---

<!-- _class: lead -->

# Rewrite Patterns & Dialect Conversion

## MLIR Summer School — Transformations (2/3)

*Express rewrites as composable patterns. Let drivers do the orchestration.*

<!-- Speaker notes:
Welcome back — session 2 of the Transformations module. ~1 min.
SESSION BUDGET (canonical): 0:00-0:05 warm-up quiz | 0:05-0:45 lecture incl. embedded quizzes+demos (~40 min core path) | 0:45-0:50 exercise briefing | 0:50-1:20 hands-on (30 min) | 1:20-1:30 solution walkthrough + wrap-up.
Core path ≈40 min; slides marked ⏱ are the pressure-release valves, in this order: Exercise 1 recap, Pattern #2 (cond_br), PatternRewriter API table, RewritePatternSet, PatternBenefit/Frozen, walkAndApplyPatterns, 📸 walk-driver demo, greedy mechanics 1/3, greedy mechanics 2/3, GreedyRewriteConfig knobs, termination quiz + answer, 📸 greedy-rewriter debug, type conversion, 📸 casts 1/2 + 2/2, signature pointer, rollback, phantom-success quiz + answer, lost-fastmath quiz + answer. Skip from wherever you are when a timing check says you're behind; never skip conversion-half core slides.
Yesterday students wrote a pass that changes IR by hand: walk, match, create, RAUW, erase.
Today's promise: we delete most of that code. The rewrite itself stays; everything around it becomes framework.
Two halves today: (1) rewrite patterns + the walk/greedy drivers, (2) dialect conversion for lowering.
Demos in this deck were captured with build/bin/mlir-opt from this checkout (assertions enabled — needed for -debug-only flags). Exactly three demos stay live (🔴): FoldSelfCopy under -canonicalize, the convergence exit codes, and the dialect-conversion trace. 📸 slides are pre-captured — do not run them live unless ahead of schedule.
-->

---

# Where we are

| # | Session | What you can do afterwards |
|---|---------|----------------------------|
| 1 | Your First Pass | Change IR by hand inside a real pass |
| **2** | **Patterns & Conversion** | **Write local rewrites; pick the right driver** |
| 3 | The Free Lunch | Plug your dialect into canonicalize/CSE/DCE |

Today's route:

1. Patterns: the local rewrite, isolated
2. The `PatternRewriter` contract (and how to break it)
3. Drivers: `walkAndApplyPatterns` vs. `applyPatternsGreedily`
4. Lowering: `ConversionTarget`, adaptors, type conversion

<!-- Speaker notes:
~1 min — part of the 0:00-0:05 warm-up block; keep it tight. Recap the module arc: each session removes hand-written machinery from the previous one.
Point out that patterns are not "one feature among many": canonicalization, conversion, and most upstream passes are built from them. One concept, used everywhere.
Then straight into the warm-up quiz.
-->

---

# 🧠 Quiz: warm-up — spot the bug(s)

A colleague "fixed" the Exercise 1 pass. It still crashes. Find **two** bugs — plus **one line that is merely wasteful today but becomes fatal this afternoon**:

```cpp
funcOp.walk([&](arith::MulIOp op) {
  APInt c;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&c)))
    return;
  OpBuilder builder(op);
  Value shamt = /* arith.constant with log2(C), elided */;
  if (!c.isPowerOf2())
    return;                                // (3) bail out — nothing cleaned up
  Value shl = arith::ShLIOp::create(builder, op.getLoc(),
                                    op.getLhs(), shamt);
  op.erase();                              // (1)
  op.getResult().replaceAllUsesWith(shl);  // (2)
});
```

Raise your hand when you have all three.

<!-- Speaker notes:
~2 min incl. discussion (this is the 0:00-0:05 warm-up slot). Recaps Session 1 (walks, use-def chains, erase rules) and plants one seed for today.
Answer on the next slide. Let students talk to their neighbor for ~60 seconds first.
Bug 1: line (1) erases an op whose result still has uses — that is illegal (assertion in debug builds: "operation destroyed but still has uses").
Bug 2: line (2) then calls a method on the erased op — use-after-free.
Correct order: replaceAllUsesWith FIRST, erase SECOND.
The twist, line (3): the shift-amount constant is created BEFORE the power-of-two check, and the non-match path returns early without cleaning it up — the IR was already changed on a "no match". In a plain pass that's just litter; in a pattern it violates today's central contract. Don't resolve it fully here — say "hold that thought".
-->

---

# ✅ Warm-up answer

- **(1)** erases an op whose result **still has uses** — assert/UB. Erase only when `use_empty()`.
- **(2)** touches `op` **after** it was erased — use-after-free.
- Correct choreography (Session 1): **create → RAUW → erase**, in that order.

```cpp
  op.getResult().replaceAllUsesWith(shl);  // rewire uses first
  op.erase();                              // now it's dead — safe
```

**The twist (3):** on the non-power-of-two path the code **already changed the IR** (the dead `shamt` constant) and then returned. In a plain pass: merely wasteful. In a *pattern*: **fatal** — never mutate on a failure path. Today explains why.

Keep this choreography in mind — in 10 minutes it becomes **one line**.

<!-- Speaker notes:
~1 min. Reinforce: erasing the *current* op of a post-order walk is fine (Session 1 rule); erasing an op with live uses never is.
Twist answer: on the failure path the IR was already changed — fine (if sloppy) in a plain pass, fatal in a pattern; today explains why. The matchAndRewrite-contract slide and the spot-the-bug quiz pick up this exact shape again.
Foreshadow: rewriter.replaceOpWithNewOp does create+RAUW+erase in one call, and does it in the right order for you.
-->

---

# Exercise 1 recap: the hand-written pass ⏱

```cpp
struct SchoolStrengthReduce
    : impl::SchoolStrengthReduceBase<SchoolStrengthReduce> {
  void runOnOperation() override {
    // Collect first, mutate after: don't erase while iterating.
    SmallVector<arith::MulIOp> candidates;
    getOperation()->walk([&](arith::MulIOp op) { candidates.push_back(op); });

    for (arith::MulIOp op : candidates) {
      APInt rhs;
      if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) || !rhs.isPowerOf2())
        continue;
      OpBuilder b(op); // insertion point: right before `op`
      Value shift = arith::ConstantOp::create(
          b, op.getLoc(), b.getIntegerAttr(op.getType(), rhs.logBase2()));
      Value shl = arith::ShLIOp::create(b, op.getLoc(), op.getLhs(), shift);
      op->replaceAllUsesWith(ValueRange{shl}); // RAUW
      op->erase();                             // now safe: use_empty()
    }
  }
};
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "yesterday's pass: collect, match, build, RAUW, erase — all by hand; the side-by-side in a few slides shows the same code condensed". (+2 if presented.)
This is yesterday's worked example, wrapped in the exercise pass class (the reference solution exercises/solutions/exercise1/StrengthReduce.cpp is the same code with fuller comments, longer variable names — worklist/rhsValue/shiftAmount — and a ++numRewrites pass statistic): `arith.muli %x, C` with C a power of two becomes `arith.shli`.
Walk through the phases: collect (collect-then-mutate, Session 1's robust idiom), match (matchPattern + m_ConstantInt + isPowerOf2), build, RAUW, erase.
Ask the debrief questions from yesterday: who iterated until nothing changed? Who wanted to run a second rewrite on the result of the first? That pain is today's topic.
-->

---

# What did we hand-roll?

| We wrote... | ...but only this part was *ours* |
|---|---|
| the traversal (`walk`) | |
| the match checks | ✅ interesting |
| builder setup + insertion point | |
| the replacement ops | ✅ interesting |
| RAUW + erase choreography | |
| "did anything change?" bookkeeping | |
| composing several rewrites | |

**Pattern** = just the match + the replacement. **Driver** = everything else.

Upstream runs on this split: **700+** hand-written `OpRewritePattern` subclasses in `mlir/lib`, driven by a handful of drivers. Canonicalization, dialect conversion, most passes — all patterns.

<!-- Speaker notes:
~1 min. The core idea of the session on one slide — deliver the split (pattern = match + replacement; driver = the rest) and move on.
The number was counted in this checkout (July 2026): 725 struct/class declarations inheriting from OpRewritePattern<...> in mlir/lib (multiline-aware regex; naive one-line greps give 500-800 depending on methodology, hence the deliberately round "700+"). Other fun numbers: ~1820 matchAndRewrite occurrences in mlir/lib.
Message: you are about to learn the single most common code shape in all of MLIR.
-->

---

# Your first pattern (a real one, all 13 lines)

<div class="columns">
<div>

```cpp
/// Fold memref.copy(%x, %x).
struct FoldSelfCopy : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.getSource() != copyOp.getTarget())
      return failure();

    rewriter.eraseOp(copyOp);
    return success();
  }
};
```
<sub>mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp:828</sub>

</div>
<div>

**Every part, annotated:**

- `OpRewritePattern<CopyOp>` — this pattern *roots* at `memref.copy`; the driver only calls it on that op, already typed.
- `using ...::OpRewritePattern;` — inherit the constructor (context, benefit). No boilerplate ctor.
- `matchAndRewrite` — match **and** rewrite in one function.
- The guard: not our case → `return failure()` — *having changed nothing*.
- `rewriter.eraseOp(copyOp)` — all mutation goes through the **rewriter**, never `op->erase()`.
- `return success()` — "I changed the IR."

</div>
</div>

<!-- Speaker notes:
~3 min — the template for everything that follows; linger here rather than on the slides after it.
Copying a memref onto itself is a no-op, so the op can be deleted. One guard, one erase.
Q for the room: why is this a pattern and not a "fold"? (copy has no results — nothing to fold to; folds can't erase arbitrary ops. Folding comes in Session 3.)
matchAndRewrite returns LogicalResult: success = IR changed, failure = IR untouched. This is a *contract*, formalized in a few slides.
The rewriter parameter is the only legal mutation channel — reason coming up (drivers listen to it).
-->

---

# 🔴 Live demo: watching `FoldSelfCopy` fire

```bash
$ mlir-opt selfcopy.mlir --canonicalize
```

<div class="columns">
<div>

**Input** (`selfcopy.mlir`):

```mlir
func.func @self_copy(%m: memref<4xf32>) {
  memref.copy %m, %m
      : memref<4xf32> to memref<4xf32>
  return
}
```

</div>
<div>

**Real output:**

```mlir
module {
  func.func @self_copy(%arg0: memref<4xf32>) {
    return
  }
}
```

</div>
</div>

`FoldSelfCopy` is registered as a *canonicalization pattern* of `memref.copy` — so plain `-canonicalize` runs it. (How that registration works: Session 3.)

<!-- Speaker notes:
~2 min. Timing check: ~6 min into the core path by the end of this slide (≈0:11 wall clock).
Exact command: build/bin/mlir-opt selfcopy.mlir --canonicalize (input as shown; output captured from a real run of this checkout).
Teaching beat: students just saw a 13-line struct they fully understand run inside a stock upstream pass. Patterns plug into existing infrastructure; you rarely write the pass around them.
Optional flourish: change the input to copy between two different memrefs and show nothing happens (the guard returns failure).
-->

---

# Pattern #2: rewrite instead of erase ⏱

```cpp
/// cf.cond_br true, ^bb1, ^bb2 -> br ^bb1;  cf.cond_br false, ... -> br ^bb2   (comment condensed)
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(condbr.getCondition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueOperands());
      return success();
    }
    if (matchPattern(condbr.getCondition(), m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseOperands());
      return success();
    }
    return failure();
  }
};
```
<sub>mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp:309</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "replaceOpWithNewOp does the create+RAUW+erase choreography in one correctly-ordered call, and matchPattern/m_Zero are the idiomatic constant matchers — both appear on the next slide anyway". (+2 if presented.)
Second real upstream pattern, two new tools:
1. matchPattern + m_Zero()/m_NonZero(): the idiomatic constant matchers (from mlir/include/mlir/IR/Matchers.h). Don't dyn_cast to arith::ConstantOp — matchers also catch other ConstantLike ops.
2. rewriter.replaceOpWithNewOp<BranchOp>(condbr, ...): creates the new op at condbr's location and replaces condbr with it — the create+RAUW+erase choreography from the warm-up quiz, as ONE call, in the correct order.
Note the shape that repeats in all patterns: guard(s), then a rewriter call, then success; final return failure.
-->

---

# Exercise 1, re-expressed as a pattern

<div class="columns">
<div>

**Session 1 (manual, condensed):**

```cpp
SmallVector<arith::MulIOp> candidates;
getOperation()->walk([&](arith::MulIOp op) {
  candidates.push_back(op);
});
for (arith::MulIOp op : candidates) {
  APInt rhs;
  if (!matchPattern(op.getRhs(),
                    m_ConstantInt(&rhs)) ||
      !rhs.isPowerOf2())
    continue;
  OpBuilder b(op);
  Value shift = arith::ConstantOp::create(
      b, op.getLoc(),
      b.getIntegerAttr(op.getType(),
                       rhs.logBase2()));
  Value shl = arith::ShLIOp::create(
      b, op.getLoc(), op.getLhs(), shift);
  op->replaceAllUsesWith(ValueRange{shl});
  op->erase();
}
```

</div>
<div>

**Session 2 (pattern):**

```cpp
struct MulByPow2ToShl
    : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      arith::MulIOp op,
      PatternRewriter &rewriter) const override {
    APInt rhsValue;
    if (!matchPattern(op.getRhs(),
                      m_ConstantInt(&rhsValue)))
      return rewriter.notifyMatchFailure(
          op, "rhs is not a constant integer");
    if (!rhsValue.isPowerOf2())
      return rewriter.notifyMatchFailure(
          op, "rhs is not a power of two");
    Value shiftAmount = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(op.getType(),
                                rhsValue.logBase2()));
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(
        op, op.getLhs(), shiftAmount);
    return success();
  }
};
```

</div>
</div>

<sub>Old code/blog posts spell op creation `rewriter.create<arith::ShLIOp>(loc, ...)` — that form is **deprecated**; always write `OpTy::create(rewriter, loc, ...)`.</sub>

<!-- Speaker notes:
~2 min. Same rewrite, but: no walk, no OpBuilder setup, no RAUW/erase choreography, no traversal-safety worries. Only the match and the replacement remain.
Right column is the reference solution minus comments, re-wrapped for the column (exercises/solutions/exercise2/Peephole.cpp); left column is the recap slide's loop body, condensed.
New bits: (a) the rewriter IS an OpBuilder — create ops with OpTy::create(rewriter, loc, ...); (b) notifyMatchFailure(op, "reason") returns failure() AND records a human-readable reason — visible under -debug-only (demo later). Use it instead of bare failure() for anything non-obvious.
ONE-TIME history note: in older code and blog posts you will see rewriter.create<arith::ShLIOp>(loc, ...). That spelling is deprecated in this checkout ([[deprecated]] at mlir/include/mlir/IR/Builders.h:507) — always write OpTy::create(rewriter, loc, ...). Say it once, then never use the old form.
This exact pattern (MulByPow2ToShl) is Exercise 2 Part A, checkpoint 1.
-->

---

# The `matchAndRewrite` contract

From the header — this is the law of the land:

```cpp
/// Attempt to match against code rooted at the specified operation,
/// which is the same operation code as getRootKind(). If successful, perform
/// the rewrite.
///
/// Note: Implementations must modify the IR if and only if the function
/// returns "success".
virtual LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const = 0;
```
<sub>mlir/include/mlir/IR/PatternMatch.h:242</sub>

- `docs/PatternRewriter.md` adds: on success, the **root op** must be updated in-place, replaced, or erased.
- Violations are **silent** in normal builds (symptom: infinite loops, missed rewrites).
- Build with `-DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON` → the driver fingerprints the IR and aborts with *"pattern returned failure but IR did change"*.

<!-- Speaker notes:
~2 min. Quote is verbatim from PatternMatch.h:242-248 in this checkout.
Call back the warm-up twist: creating shamt before the isPowerOf2 check and bailing out is precisely a modify-then-return-failure violation once that code becomes a pattern.
Why so strict? Drivers make decisions based on the return value: "failure" means "nothing happened, don't re-enqueue anything". If you mutated and said failure, the driver's model of the IR is now wrong.
The expensive-checks build catches both directions ("returned failure but IR did change" / "returned success but IR did not change") — GreedyPatternRewriteDriver.cpp implements the fingerprinting. Recommend students enable it for their exercise builds if they hit weirdness.
Practical rule: do ALL your checks before the FIRST rewriter call.
-->

---

# The `PatternRewriter` API, grouped by intent ⏱

| Intent | Call | One-liner |
|---|---|---|
| Replace | `replaceOp(op, newValues)` | rewire all uses, erase `op` |
| | `replaceOpWithNewOp<OpTy>(op, args...)` | create + replace in one call |
| | `replaceAllUsesWith(v, newV)` | value-level RAUW, driver-visible |
| Erase | `eraseOp(op)` / `eraseBlock(b)` | `op` must have no uses |
| In-place | `modifyOpInPlace(op, [&]{ ... })` | announce an in-place edit |
| Create | `OpTy::create(rewriter, loc, ...)` | rewriter *is* an `OpBuilder` |
| Structure | `inlineBlockBefore`, `mergeBlocks`, `moveOpBefore`, `createBlock` | region/block surgery |
| Diagnose | `return rewriter.notifyMatchFailure(op, "why")` | failure + debuggable reason |

`modifyOpInPlace` is just a transaction wrapper:

```cpp
template <typename CallableT>
void modifyOpInPlace(Operation *root, CallableT &&callable) {
  startOpModification(root);
  callable();
  finalizeOpModification(root);
}
```
<sub>mlir/include/mlir/IR/PatternMatch.h:643</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "the full rewriter API table is on the cheat sheet — today you need replaceOp/replaceOpWithNewOp, eraseOp, modifyOpInPlace and notifyMatchFailure". (+2 if presented.)
Don't read every row — group them: "replace things, erase things, edit in place, create things, move structure around, explain failures".
modifyOpInPlace: whenever you change an op without replacing it (set an attribute, swap operands), wrap the edit in this call so the driver hears about it. (Historical note if asked: this used to be called updateRootInPlace — gone now.)
All of these live on RewriterBase (PatternMatch.h) which extends OpBuilder — hence "rewriter is a builder".
Next slide answers WHY everything must go through the rewriter.
-->

---

# Why the rewriter? The driver is *listening*

```text
 your pattern ──calls──▶ PatternRewriter ──notifies──▶ driver (a Listener)
                            │                             │
                            ▼                             ▼
                        mutates IR               updates its worklist
```

`RewriterBase::Listener` callbacks (what drivers subscribe to):

- `notifyOperationInserted` / `notifyOperationModified`
- `notifyOperationReplaced` / `notifyOperationErased` / `notifyBlockErased`
- `notifyPatternBegin` / `notifyPatternEnd` / `notifyMatchFailure`

<sub>mlir/include/mlir/IR/PatternMatch.h:370–426 (`notifyOperationInserted` is inherited from `OpBuilder::Listener`, Builders.h:300)</sub>

Bypass the rewriter (`op->erase()`, raw `setOperand`, raw RAUW) → the driver's worklist keeps **dangling pointers** and misses **newly created work**. Typical symptoms: use-after-free crash, or a rewrite that "mysteriously never fires".

<!-- Speaker notes:
~2 min. THE architectural slide of part 1.
The greedy driver literally inherits from RewriterBase::Listener (GreedyPatternRewriteDriver.cpp). Every rewriter method fires notifications; the driver uses them to (a) enqueue new/modified ops for more rewriting, (b) remove erased ops from its worklist, (c) roll back in dialect conversion later today.
There's even a comment in upstream test code (mlir/test/lib/Dialect/Test/TestPatterns.cpp:671) noting that using raw replaceAllUsesWith instead of rewriter.replaceAllUsesWith "would make the test fail".
Expensive-checks builds catch bypasses as "operation finger print changed".
Now: quiz.
-->

---

# 🧠 Quiz: spot the bug (one per pattern)

<div class="columns">
<div>

**Pattern A** — `x * 1 → x`:

```cpp
LogicalResult
matchAndRewrite(arith::MulIOp op,
    PatternRewriter &rewriter) const override {
  if (!matchPattern(op.getRhs(), m_One()))
    return failure();
  op.getResult().replaceAllUsesWith(
      op.getLhs());
  op->erase();
  return success();
}
```

</div>
<div>

**Pattern B** — normalize, then strength-reduce:

```cpp
LogicalResult
matchAndRewrite(arith::MulIOp op,
    PatternRewriter &rewriter) const override {
  // First move a constant LHS to the right.
  if (matchPattern(op.getLhs(), m_Constant())) {
    Value lhs = op.getLhs();
    rewriter.modifyOpInPlace(op, [&] {
      op->setOperand(0, op.getRhs());
      op->setOperand(1, lhs);
    });
  }
  APInt cst;
  if (!matchPattern(op.getRhs(),
                    m_ConstantInt(&cst)) ||
      !cst.isPowerOf2())
    return failure();
  rewriter.replaceOpWithNewOp<arith::ShLIOp>(
      op, op.getLhs(), /*shamt=*/...);
  return success();
}
```

</div>
</div>

<!-- Speaker notes:
~2 min. Give the room ~60 seconds. Both compile fine; both are broken. These are THE two beginner bugs.
A: mutations bypass the rewriter — raw RAUW and op->erase(). The driver never hears about the erasure (dangling worklist pointer → use-after-free) or the rewiring (users never re-enqueued).
B: the operand swap is announced correctly via modifyOpInPlace — but if the pow-2 check then fails, the pattern returns failure() AFTER having modified the IR. Contract violation — the warm-up twist reborn as a pattern.
Answers next slide.
-->

---

# ✅ Quiz answers

**Pattern A:** mutates IR **behind the driver's back**.
Fix — one rewriter call replaces all three lines (and fixes the order for free):

```cpp
rewriter.replaceOp(op, op.getLhs());   // rewires uses, erases op, notifies
return success();
```

**Pattern B:** modifies IR (the announced swap!), **then returns `failure()`**.
"Modify if and only if success" — even *properly announced* modifications are forbidden on the failure path. Fix: do **all** checks before the **first** rewriter call.

- Normal build: A crashes eventually; B silently confuses the driver.
- Expensive-checks build: A → *"operation finger print changed"*, B → *"pattern returned failure but IR did change"*.

(And yes — constants-to-the-right shouldn't be *your* job anyway. Session 3: the `Commutative` trait does it for free.)

<!-- Speaker notes:
~1 min. Timing check: ~15 min into the core path.
Emphasize the asymmetry: A is about the mutation CHANNEL (must be the rewriter), B is about the mutation TIMING (only after the last possible failure exit).
The parenthetical teases Exercise 3's stretch goal: adding Commutative to school.max makes constant operands move right without any pattern.
-->

---

# Bundling patterns: `RewritePatternSet` ⏱

```cpp
// Exercise 2A, directly in the pass (Peephole.cpp):
RewritePatternSet patterns(&getContext());
patterns.add<MulByPow2ToShl, MergeConsecutiveShl>(&getContext());

// The upstream convention for reusable collections: a populate function...
void populateSchoolPeepholePatterns(RewritePatternSet &patterns) {
  patterns.add<MulByPow2ToShl, MergeConsecutiveShl>(patterns.getContext());
}
// ...so other passes can mix & match:
patterns.add<SomeOtherPattern>(&getContext(), /*benefit=*/2);
```

- `add<A, B, ...>(args...)` instantiates each pattern type with the same ctor args.
- `OpRewritePattern` ctor order is **`(MLIRContext *, PatternBenefit benefit = 1)`** — context first. Extra args from `add` are forwarded to your ctor.
- ~400 `populateXxxPatterns(...)` functions upstream — that's how passes mix & match pattern collections.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "patterns are bundled with patterns.add<...> into a RewritePatternSet — the exercise stub already contains that code, and the greedy-driver slide shows the recipe". (+1 if presented.)
RewritePatternSet is just a builder for a list of pattern instances.
The first two lines are the real Exercise 2A pass; our exercise is small enough that it skips the populate helper — the helper shown is what the same code looks like packaged the upstream way (~397 'void populate' declarations in mlir/include, July 2026).
Gotcha worth saying out loud: the docs' raw-RewritePattern example uses (benefit, context) order, but OpRewritePattern is (context, benefit) — copy-pasting between the two gives confusing template errors (PatternMatch.h:322 is authoritative).
populate* convention: name your own helpers that way; graders^W reviewers expect it.
-->

---

# `PatternBenefit` and `FrozenRewritePatternSet` ⏱

**Benefit** — an ordering hint *among patterns matching the same op*:

- Default `1` is almost always right; convention: "number of ops matched".
- Only matters when two patterns match the same root and one should win.
- It does **not** decide which *op* gets rewritten first — only which *pattern* is tried first on a given op.

**Frozen** — drivers take a `FrozenRewritePatternSet`:

- Built once from `RewritePatternSet&&` (that's why call sites say `std::move(patterns)`), indexed per op name, immutable.
- Cheap to copy, **shareable across threads** — remember: your func pass runs on all functions in parallel.
- Long-lived passes freeze once in `initialize()` and reuse per `runOnOperation()` — the `-canonicalize` pass does exactly this.
  <sub>`initialize()` = a pass hook that runs once before any `runOnOperation()`; you'll see it in Session 3's canonicalizer source.</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "benefit orders patterns competing on the same op — default 1 is almost always right; drivers take a frozen, immutable, thread-safe copy of the set, which is why every call site says std::move(patterns)". (+2 if presented.)
Two small but load-bearing concepts.
Benefit: de-dramatize it. Students tend to invent elaborate benefit schemes; almost nobody needs them. PatternBenefit range is 0..65535 with 65535 reserved as "impossibleToMatch".
Frozen: connect to Session 1's threading discussion — the pass manager runs function passes in parallel; a frozen set is the immutable, thread-safe "compiled" form. Freezing isn't free, hence initialize().
The implicit conversion from RewritePatternSet&& is why every driver call you'll ever see contains std::move(patterns).
-->

---

# Driver #1: `walkAndApplyPatterns` ⏱

The author's own words:

```cpp
/// A fast walk-based pattern rewrite driver. Rewrites ops nested under the
/// given operation by walking it and applying the highest benefit patterns.
/// This rewriter *does not* wait until a fixpoint is reached and *does not*
/// visit modified or newly replaced ops. Also *does not* perform folding or
/// dead-code elimination.
///
/// This is intended as the simplest and most lightweight pattern rewriter in
/// cases when a simple walk gets the job done.
void walkAndApplyPatterns(Operation *op,
                          const FrozenRewritePatternSet &patterns,
                          RewriterBase::Listener *listener = nullptr);
```
<sub>mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h:21</sub>

- One post-order sweep. Predictable, fast, `void` — nothing to check.
- Restriction: a pattern may only erase the **matched op and IR nested under it** (never siblings/users) — enforced only in expensive-checks builds; otherwise silent UB.
- Use when each op needs **at most one** rewrite and results need no re-matching.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "there is also a lightweight one-sweep driver, walkAndApplyPatterns — no fixpoint, no folding, no DCE; the choosing-a-driver table at the end contrasts it with greedy". (+2 if presented.)
Quote is verbatim from the header (two comment lines trimmed: the unreachable-blocks note and the "does not apply patterns to the given operation itself" note — both stated in the bullets/notes below).
Include: #include "mlir/Transforms/WalkPatternRewriteDriver.h".
When is it enough? E.g. arith's unsigned-when-equivalent pass: analysis results must stay valid, no cascading wanted, one deterministic sweep — it uses exactly this driver.
The erasure restriction exists because the walk iterator would be invalidated; the driver pre-advances past the matched op so erasing *it* is fine.
Also mention: like all drivers, it does NOT apply patterns to the passed op itself, only nested ops.
-->

---

# 📸 Captured output: the walk driver does *not* fold ⏱

```bash
$ mlir-opt simple.mlir --test-walk-pattern-rewrite-driver
```

<div class="columns">
<div>

**Input** (`simple.mlir`):

```mlir
func.func @fold_me() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return %sum : i32
}
```

</div>
<div>

**Real output — `addi` survives:**

```mlir
module {
  func.func @fold_me() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.addi %c1_i32, %c2_i32 : i32
    return %0 : i32
  }
}
```

</div>
</div>

The walk driver applies **patterns only** — no folding, no DCE, no fixpoint.
Spoiler: the next driver folds this whole function down to `return %c3_i32`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "on the 1+2 input the walk driver leaves the addi and the dead constants untouched — patterns only, no fold/DCE; the greedy demo in a moment folds the same input to a single constant". Pre-captured — do not run live unless ahead of schedule; command: build/bin/mlir-opt simple.mlir --test-walk-pattern-rewrite-driver (test pass available in mlir-opt because this build has MLIR_INCLUDE_TESTS on). (+1 if presented.)
The test pass carries a few test-only patterns; none match here — the point is what does NOT happen: 1+2 is not folded, dead constants are not removed.
Contrast coming: --canonicalize (greedy driver) folds the addi and DCEs the dead constants.
-->

---

# Driver #2: `applyPatternsGreedily`

A complete, real pattern pass — this shape is ~90% of pattern passes upstream:

```cpp
void FoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```
<sub>mlir/lib/Dialect/MemRef/Transforms/FoldMemRefAliasOps.cpp:615</sub>

```cpp
LogicalResult applyPatternsGreedily(       // Transforms/GreedyPatternRewriteDriver.h
    Operation *op, const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig(), bool *changed = nullptr);
```

- Runs patterns **to a fixpoint**, and by default also **folds** ops and **erases trivially dead** ops.
- Applies to ops **nested under** `op` — not to `op` itself.
- The `(void)` is deliberate — the return value means something surprising (3 slides from now).

<!-- Speaker notes:
~1 min. The workhorse. Name history (say once): this was applyPatternsAndFoldGreedily in older code — the old name is gone from this checkout.
Point at the (void): failure() does NOT mean "no pattern matched". Hold the suspense; resolved on the fixpoint slide.
"Not applied to op itself": a pattern rooted at the pass anchor op will never fire — classic confusion.
Next: what "greedily" actually does, in three slides.
-->

---

# Greedy mechanics (1/3): fill the worklist ⏱

```text
iteration start:
  worklist ← all ops under the root, collected by a walk
             (default: bottom-up = post-order;
              config.setUseTopDownTraversal(true) flips it)
  known constants are CSE'd on the way in
  unreachable blocks erased
```

- The worklist is processed **LIFO** — with the default (post-order) initialization, ops are popped **bottom-up**: the last op of the region comes off first.
- Every *successful* change notifies the driver (it's the Listener!) which pushes **affected ops** back on: the new ops, their users, operands whose ops may have become dead.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "the driver seeds a worklist with all ops under the root (bottom-up by default) and every successful change re-enqueues the affected ops". (+1 if presented.)
Keep this light — the point is "worklist algorithm", not the exact traversal maths.
Implementation reference if asked: RegionPatternRewriteDriver::simplify in mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp.
Fine print if a sharp student checks with -debug-only: bottom-up (false) is the API default of GreedyRewriteConfig, but the -canonicalize PASS overrides it — its top-down option defaults to TRUE (mlir/include/mlir/Transforms/Passes.td), so canonicalize demos process ops top-down. Verified live both ways with --canonicalize='top-down=false'.
The constant-CSE-on-insert is why duplicate constants disappear under -canonicalize even without a CSE pass — it's the greedy driver's OperationFolder.
-->

---

# Greedy mechanics (2/3): process one op ⏱

For each op popped off the worklist:

```text
1. trivially dead (no users, no side effects)?  → erase it. done.     [DCE]
2. try op->fold(...)                            → folded? done.       [folding]
   (constant results materialized via the dialect's constant hook)
3. try patterns on this op, in decreasing benefit order
   (via PatternApplicator — the same engine the walk driver uses)
4. any change → affected ops get pushed back on the worklist
```

Steps 1–2 run **by default**: the greedy driver gives you DCE + folding for free.

```cpp
config.enableFolding(false);      // patterns only, no folding
config.enableConstantCSE(false);  // don't dedupe constants
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "per op the driver tries DCE, then fold(), then patterns in benefit order — DCE and folding are on by default, which is why your test IR simplifies more than your pattern alone explains". (+2 if presented.)
THE detail slide about greedy — present it whenever time allows.
Students testing "just my pattern" are often surprised that their test IR also gets constant-folded and dead-code-eliminated — that's steps 1-2, switchable via the config.
"Fold" is Session 3 material; today's takeaway: ops can carry a cheap local simplifier called fold(), and this driver invokes it before patterns.
Benefit order: this is where PatternBenefit matters — pattern choice per op, nothing more.
Source: processWorklist in GreedyPatternRewriteDriver.cpp:443-639.
-->

---

# Greedy mechanics (3/3): the fixpoint

```cpp
// RegionPatternRewriteDriver::simplify (condensed)
do {
  if (++iteration > config.getMaxIterations() &&
      config.getMaxIterations() != GreedyRewriteConfig::kNoLimit)
    break;
  worklist.clear();
  // ...repopulate worklist, erase unreachable blocks...
  continueRewrites = processWorklist();   // DCE + fold + patterns
  // ...optional region simplification...
} while (continueRewrites);
return success(!continueRewrites);
```
<sub>mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp:842 (condensed)</sub>

- **Converged** = one full iteration made **zero changes**. So *any* change forces ≥ 2 iterations — the last one just confirms the fixpoint.
- `success()` = converged. **`failure()` = did NOT converge** within `maxIterations` (default 10).
- `failure()` does **not** mean "nothing matched" — a run with zero matches converges immediately and returns `success()`.

<!-- Speaker notes:
~1 min. Hit only the two bolded semantics: converged = one change-free iteration; failure() = did NOT converge (never "nothing matched").
Resolve the (void) suspense: most passes ignore the result because non-convergence is usually "best effort was good enough", not an error. The canonicalizer itself only fails on non-convergence under its test-only test-convergence option.
The "≥2 iterations" fact is demoed live on the next core slide with real exit codes.
Ask the room before revealing: "what would failure() mean?" — most will guess "no pattern matched". Correcting that is the point of the slide.
-->

---

# `GreedyRewriteConfig` — the knobs ⏱

```cpp
GreedyRewriteConfig config;
config.setUseTopDownTraversal(useTopDownTraversal)
    .setMaxIterations(this->maxIterations)
    .enableFolding(this->fold)
    .enableConstantCSE(this->cseConstants);
(void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
```
<sub>mlir/test/lib/Dialect/Test/TestPatterns.cpp:463</sub>

| Setter | Default |
|---|---|
| `setUseTopDownTraversal(bool)` | `false` (bottom-up) |
| `setMaxIterations(int64_t)` | `10` — bounds *outer* iterations (`kNoLimit` = -1) |
| `setMaxNumRewrites(int64_t)` | `kNoLimit` — caps rewrites *within one* iteration |
| `setRegionSimplificationLevel(...)` | `Aggressive` (includes block merging) |
| `enableFolding(bool)` / `enableConstantCSE(bool)` | `true` / `true` |

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "GreedyRewriteConfig holds the knobs — traversal order, maxIterations (bounds the outer loop), maxNumRewrites (bounds one iteration), folding/CSE toggles". (+1 if presented.)
Fluent setters, all chainable (fields are private; no direct member access).
Flag the maxIterations vs maxNumRewrites distinction now — the quiz on the next slide hinges on it: maxIterations bounds the OUTER do-while; maxNumRewrites bounds rewrites INSIDE a single iteration.
Full list in mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h:43-176 (also: scope, strictness, listener, CSE between iterations).
-->

---

# 🧠 Quiz: will this terminate? ⏱

Two well-formed patterns, both honoring the `matchAndRewrite` contract:

```cpp
// Pattern 1:  test.foo  →  test.bar   (replaceOpWithNewOp, returns success)
// Pattern 2:  test.bar  →  test.foo   (replaceOpWithNewOp, returns success)

RewritePatternSet patterns(ctx);
patterns.add<FooToBar, BarToFoo>(ctx);
(void)applyPatternsGreedily(getOperation(), std::move(patterns));
```

Input contains one `test.foo`. With the **default config**, what happens?

**(a)** converges — the driver detects the cycle and stops
**(b)** returns `failure()` after `maxIterations` = 10 iterations
**(c)** hangs forever

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip the answer slide with it); if skipped say: "two patterns that undo each other make iteration 1 spin forever — the driver has no cycle detection; every pattern must strictly reduce something". (+2 if presented.)
Voting quiz. Most students pick (b) — "there's an iteration limit, so it fails gracefully". That's the trap.
Answer: (c). Explanation next slide.
-->

---

# ✅ It **hangs** (c) — here's why ⏱

- Each successful rewrite **re-enqueues** the new op → the *inner* worklist loop of iteration #1 **never empties**.
- `maxIterations` bounds the **outer** loop only — iteration #1 never finishes, so the limit never triggers.
- The greedy driver has **no cycle detection**.

**The safety net:**

```cpp
config.setMaxNumRewrites(1000);   // default: kNoLimit (unlimited)
```

forces iteration boundaries → the driver then hits `maxIterations` and returns a clean `failure()` (= did not converge).

**The real lesson:** every pattern must **strictly reduce something** (op count, a cost, a lexicographic measure). If two of your patterns can undo each other, your pattern set is buggy — no config flag fixes that.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; pairs with the previous quiz slide — skip or present both; if skipped say: "the lesson stands: every pattern must strictly reduce something, and setMaxNumRewrites is the seatbelt that turns a hang into a clean failure()". (+2 if presented.)
Verified against the implementation in this checkout: the worklist loop in GreedyPatternRewriteDriver.cpp keeps processing as long as the worklist is non-empty; only maxNumRewrites (default unlimited) can cut a single iteration short.
Side note if asked about Pattern::hasBoundedRewriteRecursion: that flag is only consulted by the DIALECT CONVERSION driver, not by greedy/walk. For greedy, the honest statement is: self-reinforcing patterns loop until maxNumRewrites/maxIterations — or hang with defaults.
Segue: "converged" is checkable — next demo.
-->

---

# 🔴 Live demo: convergence is a *verified* fixpoint

```bash
$ mlir-opt simple.mlir --canonicalize='max-iterations=1 test-convergence=true'
$ echo $?
1        # changed IR in iteration 1 → no change-free iteration happened → FAIL

$ mlir-opt simple.mlir --canonicalize='max-iterations=2 test-convergence=true'
```

```mlir
module {
  func.func @fold_me() -> i32 {
    %c3_i32 = arith.constant 3 : i32
    return %c3_i32 : i32
  }
}
```

```bash
$ echo $?
0        # iteration 2 made no changes → fixpoint confirmed
```

- Non-convergence is **silent** by default — `test-convergence` turns it into a pass failure. Upstream canonicalization tests run `canonicalize{test-convergence}` to catch cyclic patterns.

<!-- Speaker notes:
~2 min. Timing check: ~19 min into the core path — part 1 ends here; the conversion half must start by ~min 22, so if you're past that, skip ⏱ slides aggressively from here on.
If the ⏱ walk-demo slide was skipped, flash its input first (simple.mlir: constants 1 and 2, one addi, in a func).
Exact commands (run for real on this checkout, exit codes as shown):
  build/bin/mlir-opt simple.mlir --canonicalize='max-iterations=1 test-convergence=true'   → exit 1
  build/bin/mlir-opt simple.mlir --canonicalize='max-iterations=2 test-convergence=true'   → exit 0, output as shown
simple.mlir is the same 1+2 input from the walk demo. One change (the fold) happened in iteration 1, so with max-iterations=1 the fixpoint is never CONFIRMED — even though the IR looks done.
Also note: without test-convergence, non-convergence produces NO diagnostic at all (only a debug-log line) — that's the "(void)" culture.
And: compare with the walk-driver demo — same input, now fully folded, dead constants gone.
-->

---

# 🧠 Quiz: the polite pattern ⏱

The rewrite is correct; every FileCheck test is green. Approve the review?

```cpp
LogicalResult matchAndRewrite(arith::MulIOp op,
                              PatternRewriter &rewriter) const override {
  APInt c;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&c)))
    return success();                       // not our case — and no harm done!
  if (!c.isPowerOf2())
    return success();                       // ditto
  Value shift = arith::ConstantOp::create(rewriter, op.getLoc(),
      rewriter.getIntegerAttr(op.getType(), c.logBase2()));
  rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(), shift);
  return success();
}
```

Input: one `arith.muli %x, %c7` — 7 is *not* a power of two. Under `applyPatternsGreedily` (default build, default config), this…

① converges immediately — nothing to rewrite, nothing happens
② hangs — the inner worklist never empties
③ runs **all 10 iterations**, then returns `failure()` — and by default nobody tells you
④ aborts with a fatal error

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip the answer slide with it); if skipped say: "returning success() on a nothing-to-do path is the other contract violation — the driver books phantom progress and never confirms the fixpoint; every no-op path must return failure()". (+2 if presented.)
Give ~60 seconds; have the room commit before revealing.
Answer: ③ — with ④ true only under MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS (which is why the stem pins a default build; a student answering ④ remembered the contract slide — credit it, then narrow to the default build). Mechanics: success() means "I changed the IR". No rewriter call was made, so no listener event fires and nothing is re-enqueued — the ITERATION finishes (that's why ② is wrong: no hang) — but the iteration is booked as "made progress" (processWorklist sets changed=true on any pattern success, GreedyPatternRewriteDriver.cpp:623-630), so the outer loop re-seeds the worklist, phantom-succeeds again, and only stops when maxIterations (10) runs out. applyPatternsGreedily returns failure() = did not converge; the canonicalizer swallows that by design ("Canonicalization is best-effort. Non-convergence is not a pass failure.", Canonicalizer.cpp:89).
Provenance (verified in this checkout's history): commit 91c0ba6de8e7 "[OpenACC] Fix pattern API check failures in acc-loop-tiling pass (#188968)", April 2026 — a shipped pass returned success() when there was nothing to tile; caught when the pass ran under MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS ("pattern returned success but IR did not change"). The fix for this bug is literally one word; the same commit also fixes a second violation — IR moved via splice()/replaceAllUsesInRegionWith without notifying the rewriter, caught as "operation fingerprint changed" — a neat callback to the driver-is-listening slide.
-->

---

# ✅ Ten silent iterations (③) — the phantom success ⏱

- `success()` tells the driver **"I changed the IR."** No rewriter call happened, so nothing is re-enqueued — the *iteration* finishes (no hang) — but it's booked as progress, so the fixpoint is never confirmed: re-seed, "succeed", repeat… **10 iterations, then `failure()`** (= did not converge — remember?).
- Default visibility: **none.** Canonicalization is best-effort; you pay ~10× pattern time on every function containing one such op, forever, silently.
- The builds that *do* tell you:
  - expensive-checks: `"pattern returned success but IR did not change"` <sub>mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp:90-92</sub>
  - `canonicalize{test-convergence=true}` → pass failure (the upstream test-suite setting)
- **The contract really is *iff*.** Mutate-then-`failure()` (warm-up quiz) is one violation; **no-mutation-then-`success()` is the other.** Every "nothing to do" path must `return failure()` — or better:

```cpp
return rewriter.notifyMatchFailure(op, "rhs is not a constant power of two");
```

Real: a shipped upstream pass had exactly this bug until April 2026 (#188968); the fix for it is literally one word.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; pairs with the previous quiz slide — skip or present both; if skipped say: "success() means 'I changed the IR' — a nothing-to-do success burns all 10 iterations and the resulting failure() is silently swallowed; expensive-checks builds abort on it". (+1 if presented.)
Contrast with the termination quiz explicitly (if it was presented) — two different non-termination shapes: A→B/B→A re-enqueues via listener events and spins INSIDE iteration 1 (hang); the phantom success fires no events, so it burns OUTER iterations and exits with a quiet failure(). Understanding the difference proves you understood the worklist.
The "no harm done" comments in the quiz code are the trap: the code really does no harm to the IR — it lies to the DRIVER, not to the IR. The contract is about information flow, not just mutation safety.
Segue: notifyMatchFailure costs nothing in release and gives you the "** Match Failure : <reason>" lines in the --debug-only=greedy-rewriter trace for free (GreedyPatternRewriteDriver.cpp:770-776) — the next slide's captured output shows where those messages surface.
-->

---

# 📸 Captured output: `--debug-only=greedy-rewriter` ⏱

```bash
$ mlir-opt simple.mlir --canonicalize --debug-only=greedy-rewriter
```

```text
Processing operation : 'arith.addi'(0x50f00000d430) {
  %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
} -> success : operation was folded
** Insert  : 'arith.constant'(0x5080000061b0)
** Replace : 'arith.addi'(0x50f00000d430)
** Modified: 'func.return'(0x50b000038900)
** Erase   : 'arith.addi'(0x50f00000d430)
...
Processing operation : 'arith.constant'(0x508000003630) {
  %1 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  ** Erase   : 'arith.constant'(0x508000003630)
} -> success : operation is trivially dead
```

And `--debug-only=pattern-application` names the patterns:

```text
Trying to match "(anonymous namespace)::CombineIfs"
 -> matchAndRewrite failed
...
Trying to match "(anonymous namespace)::ConvertTrivialIfToSelect"
 -> matchAndRewrite successful
```

Your `notifyMatchFailure(op, "reason")` messages appear **only** here — never in a normal run.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "--debug-only=greedy-rewriter and =pattern-application trace every fold, pattern attempt and listener event — the flags are on the cheat sheet, use them in the exercise". Pre-captured — do not run live unless ahead of schedule; commands: build/bin/mlir-opt simple.mlir --canonicalize --debug-only=greedy-rewriter, and build/bin/mlir-opt scfif.mlir --canonicalize --debug-only=pattern-application (scfif.mlir: scf.if with a constant true condition). (+1 if presented.)
Real captured output (this checkout); the uniform "[greedy-rewriter:1] " / "[pattern-application ...]" log prefixes were stripped, blank lines dropped, and the traces trimmed at "..." (the second "..." hides two more failed patterns: CombineNestedIfs, ConditionPropagation). The 0x... pointers vary from run to run — don't expect them to match live.
Point out in the trace: the fold, every Listener notification (Insert/Replace/Modified/Erase — the callbacks from the listener slide, live!), and built-in DCE ("trivially dead").
notifyMatchFailure messages ALSO only appear under these flags — they print nothing in a normal run. Don't promise users error messages that only exist under -debug-only.
These flags need an assertions-enabled build; release builds silently ignore -debug-only.
-->

---

# From optimization to lowering

So far: *make the IR better*. Now the job changes: *get rid of dialect X entirely*.

| Requirement | Greedy driver | What we need |
|---|---|---|
| "every `school.*` op must be gone" | can't promise — patterns fire where they match | **legality target** + loud failure |
| types change (`i32` → `i64`, `memref` → `!llvm.struct`) | no support — patterns see old-typed operands | **type conversion** + glue at boundaries |
| multi-step lowering `A → B → C` | only if patterns happen to cascade | **legality-driven** pattern chaining |
| partial failure | half-rewritten IR | **rollback** |

This is the **dialect conversion** framework: `ConversionTarget` + `ConversionPattern`s + (optionally) a `TypeConverter`, run by `applyPartialConversion` / `applyFullConversion`.

<!-- Speaker notes:
~1 min. Timing check: ~20 min into the core path when you finish this slide. This is the PROTECTED milestone — the conversion half must begin by ~min 22 of lecture time; if you land here later than that, recover by skipping ⏱ slides ahead, never by cutting conversion-half core slides.
The hinge of the session. Lowering is a different *problem*, not just a different driver: completeness ("must all go"), type changes, and loud failure are non-negotiable for a compiler pipeline.
Vocabulary: "conversion" and "lowering" are used interchangeably in MLIR.
Three ingredients to remember: target (what is legal), patterns (how to rewrite), type converter (how types change).
-->

---

# `ConversionTarget`: declaring legality

Every op is in one of four buckets:

- **Legal** — never touched: `target.addLegalDialect<arith::ArithDialect>();`
- **Illegal** — *must* be rewritten: `target.addIllegalDialect<school::SchoolDialect>();`
- **Dynamically legal** — decided per op by a callback:

```cpp
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return llvm::none_of(op->getOperandTypes(),
                       [](Type type) { return llvm::isa<TensorType>(type); });
});
```
<sub>mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp:344</sub>

- **Unknown** — no registered action. Default fate depends on the driver (next slide); opt out via:

```cpp
target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
```
<sub>mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp:738</sub>

<!-- Speaker notes:
~2 min. The target is a declarative spec: you state WHERE the IR must end up; the driver figures out which patterns to run to get there.
Toy example reads: "toy.print is fine — but only once its operands are no longer tensors" — dynamic legality expresses "legal after conversion did its job around it".
markUnknownOpDynamicallyLegal([](...) { return true; }) is the "everything I didn't mention is fine" idiom (SCF→CF uses it, since it touches only control flow ops).
Registration API also has addLegalOp/addIllegalOp variadic forms: addIllegalOp<scf::ForOp, scf::IfOp>().
-->

---

# Partial vs. full conversion — exact semantics

| | `applyPartialConversion` | `applyFullConversion` |
|---|---|---|
| explicitly **illegal** op can't be legalized | **error, pass fails** | error, pass fails |
| **unknown** op can't be legalized | **survives untouched** | error, pass fails |
| use case | one lowering stage of a pipeline | final "nothing may remain" stage |

Marking an op illegal does **not** delete it — with no matching pattern, the conversion just fails. The framework never invents rewrites.

<!-- Speaker notes:
~2 min. Timing check: ~24 min into the core path.
Partial is NOT "best effort on everything": it still hard-fails on explicitly-illegal leftovers. It's lenient only about UNKNOWN ops — that's what makes staged pipelines possible (func.func and arith survive a memref-to-llvm stage untouched).
Full additionally fails on unknown leftovers (and on unreachable blocks).
The quiz on the next slide checks exactly this distinction — don't pre-empt the answer.
-->

---

# 🧠 Quiz: which driver fails?

The input contains **two** interesting ops:

```mlir
%v = memref.load %m[%i] : memref<4xi32>
%r = school.max %v, %c : i32
```

Setup:

- Target: `school` dialect **illegal**, `arith` dialect **legal**. Nothing registered for `memref` or `func`.
- Patterns: exactly one — `school.max` → `arith.cmpi` + `arith.select`.

Which driver fails — `applyPartialConversion` or `applyFullConversion` — and **on which op**?

<!-- Speaker notes:
~2 min, quick vote. Pure reasoning quiz — no tool output to remember; apply the table from the previous slide.
Have students commit to an answer for each driver before revealing. The traps: "partial fails because memref.load has no pattern" (wrong — unknown ops survive partial) and "full fails on school.max" (wrong — that one IS convertible).
-->

---

# ✅ Which driver fails — answer

- **`applyPartialConversion`: succeeds.** `school.max` is explicitly illegal → must be converted → the pattern handles it. `memref.load` (and `func.func`, `builtin.module`) are **unknown** → they survive untouched. That's the property that makes staged pipelines work.
- **`applyFullConversion`: fails.** In a full conversion, *unknown* is not good enough — everything left must be legalizable. With no patterns and no legality rule for them, the **first non-legalizable op the driver visits** — the enclosing `func.func`/`builtin.module` or the `memref.load` — is reported, and the *failed to legalize* error **names that op**.

Same reasoning returns in Exercise 2 Part B's stretch goal (full conversion failing on `builtin.module`).

<!-- Speaker notes:
~1 min. Answer stays in prose deliberately — no invented tool output; the real error format is on the next slide.
Reinforce the one-line summary: partial = "unknown survives", full = "nothing may remain". Illegal-but-convertible is fine for both drivers.
-->

---

# Failure is *loud*: reading the error

Real output (`complex.tanh` has no `ComplexToLLVM` pattern):

```text
$ mlir-opt tanh.mlir --convert-complex-to-llvm
tanh.mlir:2:8: error: failed to legalize operation 'complex.tanh' that was
explicitly marked illegal: %0 = "complex.tanh"(%arg0) ... : (complex<f32>) -> complex<f32>
  %0 = complex.tanh %z : complex<f32>
       ^
```

The error names the op, its legality status (*explicitly marked illegal*), and the source location. Learn to read it now — you will produce it on purpose in Exercise 2B.

<!-- Speaker notes:
~1 min. Error captured for real: build/bin/mlir-opt tanh.mlir --convert-complex-to-llvm on a func containing complex.tanh (error line wrapped, the fastmath attribute elided with ..., and the trailing "note: see current operation: ..." diagnostic line dropped for the slide).
Students will reproduce exactly this error in Exercise 2B, on purpose, with school.mac.
-->

---

# `OpConversionPattern`: a real one

```cpp
struct NegOpConversion : public OpConversionPattern<complex::NegOp> {
  using OpConversionPattern<complex::NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());

    Value real =
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value negReal = arith::NegFOp::create(rewriter, loc, real);
    Value negImag = arith::NegFOp::create(rewriter, loc, imag);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, negReal, negImag);
    return success();
  }
};
```
<sub>mlir/lib/Conversion/ComplexToStandard/ComplexToStandard.cpp:586</sub>

`-(a+bi) = (-a) + (-b)i` — extract re/im, negate both, repack.

<!-- Speaker notes:
~2 min. Differences from OpRewritePattern: (1) base class OpConversionPattern<OpTy>; (2) matchAndRewrite takes THREE args — op, ADAPTOR, and a ConversionPatternRewriter; (3) same contract, same rewriter discipline.
Note the op creation style throughout: OpTy::create(rewriter, loc, ...) — this is current upstream code, four creates in a row.
The mysterious second parameter — the adaptor — is the single most important (and most misunderstood) thing in dialect conversion. Next slide is dedicated to it. For this pattern types don't change, so adaptor.getComplex() == "the complex operand" and the distinction is invisible — which is exactly how beginners get away with wrong mental models until types DO change.
-->

---

# ⚠️ THE ADAPTOR. Read this slide twice. ⚠️

<br/>

# `op.getLhs()` → the **ORIGINAL** operand (old type)
# `adaptor.getLhs()` → the **CONVERTED** operand (new type)

<br/>

- The driver may have already rewritten the ops *around* you. The **adaptor** hands you the up-to-date, type-converted replacement values. **Take operands from the adaptor. Always.**
- Guaranteed: adaptor values have the **right types**. Nothing else!
- The value may be a `builtin.unrealized_conversion_cast` (a temporary placeholder op the driver inserts to bridge type mismatches — fully explained in a few slides) — so **never** pattern-match `adaptor.getLhs().getDefiningOp<...>()`.
- Use `op` for everything that is *not* an operand: attributes, location, original result types.

**This is the #1 dialect-conversion bug. You will write it anyway. Come back to this slide when you do.**

<!-- Speaker notes:
~2 min. Timing check: ~32 min into the core path.
Deliberately the loudest slide in the deck — say so.
Why does op still have old operands at all? Because in (default) rollback mode, replacements are applied lazily at the END of conversion; the original IR stays materially in place so the driver can backtrack. Your op is physically unchanged; the adaptor is the driver's view of "what your operands will have become".
The defining-op trap concretely: adaptor.getLhs().getDefiningOp<arith::ConstantOp>() may see an unrealized_conversion_cast instead of the constant — works in your test, breaks in the pipeline.
The adaptor type is ODS-generated (SourceOp::Adaptor) — named getters mirror the op's.
-->

---

# 🧠 Quiz: adaptor edition

A conversion uses a `TypeConverter` that rewrites `i32 → i64`. The driver is
converting this op (function signature already converted):

```mlir
%m = school.max %a, %b : i32     // %a, %b were i32 block args, now i64
```

Inside `matchAndRewrite(school::MaxOp op, OpAdaptor adaptor, ...)`:

1. `adaptor.getLhs().getType()` is ... ?
2. `op.getLhs().getType()` is ... ?
3. True or false: `adaptor.getLhs().getDefiningOp()` is reliably the op that will define this value in the final IR.

<!-- Speaker notes:
~1 min, quick vote. Answers next slide: i64 / i32 / false.
If students hesitate on 2: the op itself has not been touched yet — the driver rewrites uses lazily at the end; op.getLhs() still points at the original i32 value.
-->

---

# ✅ Adaptor answers

1. **`i64`** — adaptor values carry the **converted** types. That's the guarantee.
2. **`i32`** — `op` still sees the original, untouched operands.
3. **False** — the adaptor value may be a
   `builtin.unrealized_conversion_cast` (a temporary placeholder op the driver inserts to bridge type mismatches — fully explained in a few slides) that disappears before the conversion finishes. Only its **type** is meaningful.

Rule of thumb inside a conversion pattern:

| Need | Take it from |
|---|---|
| operand **values** | `adaptor` |
| attributes, location | `op` |
| **result** types (new) | `getTypeConverter()->convertType(op.getType())` |
| surrounding IR | ⚠️ don't — it's a mix of converted and unconverted state |

<!-- Speaker notes:
~1 min. The table is the take-home. Last row: conversion is a pre-order, lazy process — walking neighbors, getUsers(), or dominance queries during conversion observe an inconsistent mix; results are unreliable. Patterns should be local.
getTypeConverter() is available on the pattern when it was constructed with a TypeConverter (pattern ctor (const TypeConverter&, MLIRContext*, benefit)).
-->

---

# 🧠 Quiz: what did `NegOpConversion` lose? ⏱

The real upstream pattern from a few slides ago, fed a flagged input — real output of `--convert-complex-to-standard` on this checkout:

<div class="columns">
<div>

**Input**

```mlir
%n = complex.neg %arg0 fastmath<fast>
       : complex<f32>
```

</div>
<div>

**Output**

```mlir
%0 = complex.re %arg0 : complex<f32>
%1 = complex.im %arg0 : complex<f32>
%2 = arith.negf %0 : f32
%3 = arith.negf %1 : f32
%4 = complex.create %2, %3 : complex<f32>
```

</div>
</div>

*(`fastmath<fast>` = the user's permission for aggressive FP optimization. The attribute defaults to `fastmath<none>` — and the printer elides defaults.)*

It verifies. Every FileCheck test is green. The user who wrote `fastmath<fast>` is unhappy — **why**, and **which tool in your toolbox would have told you**?

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip the answer slide with it); if skipped say: "newly created ops carry ONLY what you pass to create() — the complex.neg lowering silently drops fastmath flags to this day; attribute propagation is a deliberate per-op decision, and no tool flags a dropped attribute". (+2 if presented.)
Give ~60 seconds. Someone will spot it quickly: the arith.negf ops carry NO fastmath — i.e. fastmath<none>, the default (the on-slide gloss hands them the elision fact; the inference is still theirs to make).
Both the input and output shown are real and were verified live on this checkout: build/bin/mlir-opt --convert-complex-to-standard on a func containing complex.neg with fastmath<fast>. complex ops carry a standard arith fastmath attribute (ComplexOps.td, ArithFastMathInterface).
Second question's answer: NOTHING tells you. Not the verifier (the IR is valid), not the tests (unless a CHECK-SAME looks for the flag), not any -debug-only flag. Only a reviewer or a deliberate CHECK line. That's the point of the quiz.
-->

---

# ✅ The flags are gone — and nothing will ever tell you ⏱

- `arith.negf %0 : f32` means **`fastmath<none>`** — the default. `OpTy::create` gives the new op **exactly what you pass; nothing rides along** from the op you matched: not `fastmath`, not overflow flags, not discardable attributes (`replaceOpWithNewOp` transfers nothing either).
- Failure mode: **silence.** No assert, no verifier error, no test diff — downstream, every FP optimization that needed `fast` is simply off.
- Upstream re-fixed this class about a dozen times in `ComplexToStandard` alone (`complex.mul` propagates its flags today — the `neg` pattern on our slide *still* drops them in this checkout).
- **The overcorrection is real, too:** blindly forwarding flags onto *every* op you create broke `complex.abs` — its internal NaN-detection ops must **not** promise `nnan`/`ninf`:

```cpp
// The lowering below requires NaNs and infinities to work correctly.
arith::FastMathFlags fmfWithNaNInf = arith::bitEnumClear(
    fmf, arith::FastMathFlags::nnan | arith::FastMathFlags::ninf);
```

<sub>mlir/lib/Conversion/ComplexToStandard/ComplexToStandard.cpp:42-44</sub>

**Rule:** attributes are per-op *promises*. For each op you create, decide which of the matched op's promises still hold — don't drop them all by default, don't forward them all by reflex.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; pairs with the previous quiz slide — skip or present both; if skipped say: "ops you create carry only what you pass — the neg lowering drops fastmath silently to this day; decide per created op which of the matched op's promises still hold". (+2 if presented.)
Provenance (verified in this checkout's history): a representative early drop-side fix is be5b66670d86 "[mlir][complex] Support fastmath in the binary op conversion (#65702)" — threading getFastMathFlagsAttr() through the lowering, one pattern at a time; the very first was complex.abs in Aug 2023 (653f77690bb2, relanded as 2e53e1548074), and roughly a dozen sibling commits followed (exp/expm1, log, mulf, div, sqrt, trig, tanh, angle). The over-propagation fix is 9b225d01f8ed "Fix complex abs with nnan/ninf (#95080)", source of the bitEnumClear snippet.
The two failure directions in one breath: dropping a flag is always SOUND but pessimizing (fewer promises); forwarding a flag can MISCOMPILE (a promise the new op doesn't keep — computeAbs relies on NaN propagation internally, so nnan on its own ops lets the canonicalizer fold its NaN-fixup path away: silent wrong answers on edge inputs).
Exercise 2A tie-in: arith.shli carries overflow flags; the merged shli your MergeConsecutiveShl pattern creates has overflow<none> — which is exactly right here, because nsw/nuw on the merged shift would need a proof. Conservative-by-default is correct; propagate only what you can justify.
Discardable attributes: same story, contractually weaker — any pass MAY drop them, so pipelines must never hang correctness on one; when a canonicalization rebuilds an op and you need metadata to survive, rewriter.clone + in-place mutation or an explicit setDiscardableAttrs is the pattern (upstream examples: tensor.pack canonicalization #111261, linalg generalization Generalization.cpp:68-73).
-->

---

# The driver side: a complete conversion pass

```cpp
void ConvertComplexToStandardPass::runOnOperation() {
  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns, complexRange);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addLegalOp<complex::CreateOp, complex::ImOp, complex::ReOp>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```
<sub>mlir/lib/Conversion/ComplexToStandard/ComplexToStandard.cpp:1161</sub>

Same recipe every time: **patterns + target + apply + `signalPassFailure`**.
(Here `complex.*` ops are *unknown* rather than illegal — each pattern that fires
still must produce legal ops, but unmatched complex ops survive. `create/im/re`
are declared legal because the patterns themselves emit them.)

<!-- Speaker notes:
~1 min. Note how little code the pass is — the "everything else" is the framework.
Contrast with conversion failure semantics: here complex ops aren't explicitly illegal, so a complex op that no pattern matched would SURVIVE this partial conversion. (Don't say "complex.tanh" here — this pass DOES have a tanh pattern, TanTanhOpConversion. The loud tanh error earlier came from -convert-complex-to-llvm, which has no tanh pattern AND marks the whole dialect illegal.)
Also point out: unlike greedy, here failed(...) => signalPassFailure() is mandatory etiquette. A silent half-lowering is the worst outcome.
Exercise 2B is exactly this recipe with school ops: target marks school illegal, arith legal, applyPartialConversion.
-->

---

# 🔴 Live demo: how legalization actually works

```bash
$ mlir-opt neg.mlir --convert-complex-to-standard --debug-only=dialect-conversion
```

```text
Legalizing operation : 'complex.neg' (0x50d000000ef0) {
  %0 = "complex.neg"(%arg0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
  * Fold {
  } -> FAILURE : unable to fold
  * Pattern : 'complex.neg -> ()' {
    ** Insert  : 'complex.re' (0x50d000001160) (was detached)
    ** Insert  : 'complex.im' (0x50d000001230) (was detached)
    ...
    ** Replace : 'complex.neg'(0x50d000000ef0)
    Legalizing operation : 'complex.re' (0x50d000001160) {
      %0 = "complex.re"(%arg0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> f32
    } -> SUCCESS : operation marked legal by the target
    ...
  } -> SUCCESS : pattern applied successfully
} -> SUCCESS
```

The algorithm, per op (pre-order walk): **legal already?** → done. Else **try folding** (yes, *before* patterns — the default). Else **try patterns**; everything a pattern creates is **recursively legalized**. Any dead end → **roll back** that pattern and try the next: a backtracking search guided by legality.

<!-- Speaker notes:
~3 min. Timing check: ~38 min into the core path.
Real trace (this checkout), "[dialect-conversion:1] " prefix stripped, trimmed at "...".
Exact command: build/bin/mlir-opt neg.mlir --convert-complex-to-standard --debug-only=dialect-conversion
neg.mlir = func.func with one complex.neg (same input as the NegOpConversion slide). Final output IR (also verified): complex.re/im + two arith.negf + complex.create.
Walk the trace: fold attempt first (foldingMode=BeforePatterns is the default — constant-like illegal ops may vanish by folding before your pattern ever runs!), then the pattern, then each created op is immediately legalized (all "marked legal by the target" here).
Recursive legalization is why patterns don't need to emit directly-legal ops — A→B→C chains work, driven by legality. A pattern can't be applied twice in the same recursion stack (unless it opts in via setHasBoundedRewriteRecursion).
-->

---

# Type conversion, gently ⏱

**`TypeConverter`** — how types change:

```cpp
void mlir::populateEmitCSizeTTypeConversions(TypeConverter &converter) {
  converter.addConversion(
      [](IndexType type) { return emitc::SizeTType::get(type.getContext()); });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
}
```
<sub>mlir/lib/Dialect/EmitC/Transforms/TypeConversions.cpp:30</sub>

- `addConversion`: callback per type; unhandled types fall through to the next registered callback. Legal type = converts to itself.
- **Materializations** = glue where converted and unconverted IR meet:
  - **target**: make a value of the *converted* type for a pattern's adaptor,
  - **source**: convert a replacement *back* to the original type for surviving old-typed uses.
- Default glue: the driver inserts `builtin.unrealized_conversion_cast` — a typed placeholder with no semantics. Cast pairs that cancel (`A→B→A`) are reconciled away at the end.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "a TypeConverter maps old types to new via addConversion callbacks, and where converted and unconverted IR meet, the driver stitches with unrealized_conversion_cast — the placeholder from the adaptor slide; point students here for self-study since the exercise needs no TypeConverter". (+2 if presented.)
Keep at concept level, per the outline — beginners need the mental model.
Wire to the pattern side: constructing patterns with (typeConverter, ctx) is what makes the adaptor deliver converted-type values.
unrealized_conversion_cast: "I owe you a value of this type" — an IOU. It's how the framework stitches boundaries while both worlds coexist. If any survive to the end of the *pipeline*, something is missing (a pattern or a materialization).
Heads-up for readers of old tutorials: addArgumentMaterialization no longer exists in this checkout — only source and target materializations.
Next: see the IOUs appear and disappear, live.
-->

---

# 📸 Captured output: casts at the boundary (1/2) ⏱

```bash
$ mlir-opt load.mlir --finalize-memref-to-llvm
```

<div class="columns">
<div>

```mlir
// load.mlir
func.func @load(%m: memref<4xf32>,
                %i: index) -> f32 {
  %v = memref.load %m[%i] : memref<4xf32>
  return %v : f32
}
```

This pass converts `memref` ops but **not** `func.func` — the block arguments keep their old types.

</div>
<div>

```mlir
module {
  func.func @load(%arg0: memref<4xf32>, %arg1: index) -> f32 {
    %0 = builtin.unrealized_conversion_cast %arg1
           : index to i64
    %1 = builtin.unrealized_conversion_cast %arg0
           : memref<4xf32> to !llvm.struct<(ptr, ptr,
             i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %1[1] : ...
    %3 = llvm.getelementptr inbounds|nuw %2[%0]
           : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %4 = llvm.load %3 : !llvm.ptr -> f32
    return %4 : f32
  }
}
```

</div>
</div>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "mid-pipeline you WILL see unrealized_conversion_cast bridging still-unconverted func args to converted llvm ops — expected state, not an error". Pre-captured — do not run live unless ahead of schedule; command: build/bin/mlir-opt load.mlir --finalize-memref-to-llvm. (+1 if presented.)
Real output (this checkout), the struct type in %2's annotation shortened to "..." to fit; SSA names exactly as printed.
Read it with the room: the llvm.load wants LLVM-typed inputs, but %arg0/%arg1 are still memref/index (this pass doesn't convert function signatures). The driver bridged the gap with two unrealized_conversion_cast ops — target materializations at the boundary between unconverted (func) and converted (llvm) IR.
This is EXPECTED mid-pipeline state, not an error.
-->

---

# 📸 Captured output: casts at the boundary (2/2) ⏱

Finish the job — convert the function signature too, then reconcile:

```bash
$ mlir-opt load.mlir --finalize-memref-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts
```

```mlir
module {
  llvm.func @load(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64,
                  %arg4: i64, %arg5: i64) -> f32 {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, ...)>
    ...
    %6 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, ...)>
    %7 = llvm.getelementptr inbounds|nuw %6[%arg5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    llvm.return %8 : f32
  }
}
```

All casts gone. Standard pipeline idiom: run your conversion stages, then
**`-reconcile-unrealized-casts` last**. Casts that *survive* it = a missing
pattern or materialization somewhere.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "run -reconcile-unrealized-casts last in every conversion pipeline — casts that survive it mean a missing pattern or materialization". Pre-captured — do not run live unless ahead of schedule; command: build/bin/mlir-opt load.mlir --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts. (+1 if presented.)
Real output; insertvalue chain trimmed with "..." and long struct types shortened; SSA names as printed.
convert-func-to-llvm converted the signature (memref expands into ptr+ptr+offset+sizes+strides — that's why 6 args), which made every cast a cancelling A→B→A pair; reconcile folded them away.
Diagnostic value: leftover unrealized_conversion_cast at the END of a pipeline is your "something is missing" smoke detector.
-->

---

# Pointer: function signatures & block arguments ⏱

- Block argument types are **not** converted automatically — an op-conversion pattern only converts *the op*.
- For function-like ops there's a ready-made pattern:

```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                               typeConverter);
```
<sub>mlir/include/mlir/Transforms/DialectConversion.h:816</sub>

- For your own region-carrying ops, patterns call
  `rewriter.applySignatureConversion(...)` / `rewriter.convertRegionTypes(...)`.
- Forgetting this is the classic *"failed to legalize unresolved materialization"* error.

That's all we'll say — you now know these exist and where to look.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "block-argument and signature types need extra machinery — populateFunctionOpInterfaceTypeConversionPattern / applySignatureConversion; it exists, docs/DialectConversion.md has it". (+1 if presented.)
Pointer slide only, per the outline. If a student hits region-carrying ops in a conversion (not in our exercises), send them to docs/DialectConversion.md "Region Signature Conversion".
convert-func-to-llvm from the previous demo uses exactly this machinery under the hood.
-->

---

# Rollback — and where the framework is going ⏱

- Default today: `ConversionConfig::allowPatternRollback = true`.
  - Pattern fails, or its products can't be legalized → **all** its changes are undone. Legalization = backtracking search.
  - Cost: `replaceOp`/`eraseOp` are *delayed* until the end; the driver keeps heavy bookkeeping.
- The docs are blunt: rollback mode *"has a significant toll on compilation time, is error-prone and makes debugging conversion passes more complicated. Therefore, programmers are encouraged to run in no-rollback mode when possible."* <sub>mlir/docs/DialectConversion.md</sub>
- No-rollback (`allowPatternRollback = false`, still marked *experimental*): all rewrites applied immediately; anything that *would* roll back is a fatal error. This is the **"One-Shot Dialect Conversion"** direction (see the RFC linked in `DialectConversion.h`).
- Some passes already expose it as a pass option, e.g. `convert-scf-to-cf{allow-pattern-rollback=false}`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped say: "by default the conversion driver can roll back failed patterns; no-rollback is where the framework is going — the takeaway is on the next slide's decision table". (+2 if presented.)
Keep factual — this is an evolving area. Current state in this checkout (July 2026): default is rollback=true; false is documented as experimental (ConversionConfig, DialectConversion.h:1444-1463) and references the discourse RFC "A new One-Shot Dialect Conversion driver".
Practical takeaway for students: write patterns that don't RELY on rollback (check first, mutate after — the contract again) and they'll be future-proof.
-->

---

# Choosing a driver

| | `walkAndApplyPatterns` | `applyPatternsGreedily` | dialect conversion |
|---|---|---|---|
| traversal | one post-order sweep | worklist → fixpoint | pre-order, legality-driven |
| revisits new/changed ops | ✗ | ✓ | ✓ (recursive legalization) |
| folding / DCE | ✗ / ✗ | ✓ / ✓ (default) | folds illegal ops / ✗ |
| types may change | ✗ | ✗ | ✓ (`TypeConverter`) |
| completeness guarantee | none | fixpoint or `failure()` | legality or loud error |
| rollback | ✗ | ✗ | ✓ (default) |
| use it for | one-shot cleanups | composing local rewrites | lowering out a dialect |

Rules of thumb:

- Patterns enable each other, or you want fold+DCE for free → **greedy**.
- Each op rewritten at most once, output needs no re-matching → **walk** (faster, deterministic).
- "Dialect X must be gone", or types change → **conversion**. Full stop.
- Conversion may **roll back** failed patterns — do all checks **before the first rewriter call** and you're future-proof.

<!-- Speaker notes:
~2 min. Timing check: ~40 min — the core path ends here; hand off to the exercise briefing at 0:45.
The take-home table; also on the cheat sheet.
One nuance if asked: plain OpRewritePatterns CAN run inside the conversion driver (SCF→CF does it) — fine when types don't change. The reverse is not true: never try to do type-changing rewrites in greedy.
-->

---

# Exercise 2 — Part A: `-school-peephole` (~15 min)

Port yesterday's rewrite to patterns and let the greedy driver earn its keep.

1. **Checkpoint 1** — `MulByPow2ToShl` as an `OpRewritePattern` (you saw the solution today — write it from memory).
2. **Checkpoint 2** — second pattern `MergeConsecutiveShl`:
   `shli(shli(x, c1), c2) → shli(x, c1 + c2)`.
   *If time / stretch-track:* the overflow guard — reject when `c1 + c2 ≥` the
   bit width (that shift is poison; the `@no_merge_overflow` test checks your guard).
3. **Checkpoint 3** — the driver (already wired: `applyPatternsGreedily` in the
   stub) composes them: the `((x * 4) * 8)` test chain collapses into a
   **single** `shli` — **neither pattern alone** does that.

```bash
# check your progress (from mlir-summer-school/exercises/):
ninja -C build && <llvm-build>/bin/llvm-lit -v build/test/exercise2
```

<!-- Speaker notes:
~2 min briefing (0:45-0:50 slot, shared with Part B). Core for the 30-min hands-on: checkpoint 1 + checkpoint 2 happy path + the checkpoint 3 composition. The overflow guard is the if-time/stretch track — @no_merge_overflow staying red until then is fine; same checkpoint names and tests, just re-tiered. The task sheet (exercise2.md) carries matching tiering.
Stubs in mlir-summer-school/exercises/, pass -school-peephole anchored on func.func; "TODO(exercise 2A, step N)" markers in lib/School/Peephole.cpp show where the two matchAndRewrite bodies go — the pattern set + driver call are already written.
The composition beat is the point of part A: mul→shl produces the input for shl+shl merging; only the fixpoint driver discovers the chain. Ask fast students: which pattern fires first on ((x*4)*8)? (depends on traversal — that it doesn't MATTER is the lesson.)
Hint for checkpoint 2: both shift amounts must be constants; use matchPattern + m_ConstantInt twice; the outer shli's operand comes from the inner one via getDefiningOp<arith::ShLIOp>(); for the guard use APInt::uadd_ov + APInt::uge (@no_merge_overflow tests it).
Debugging tip (also in exercise2.md): school-opt with --debug-only=greedy-rewriter --mlir-disable-threading — the pass is func-anchored and runs in parallel, so without the latter flag the debug output interleaves.
-->

---

# Exercise 2 — Part B: `-convert-school-to-arith` (~15 min)

Your first real lowering. `school` must go; `arith` may stay.

1. **Checkpoint 1** — `ConversionTarget`: `school` illegal, `arith` legal.
   Pattern: `school.max` → `arith.cmpi sgt` + `arith.select`, via
   `applyPartialConversion`. Test input contains only `school.max`.
2. **Checkpoint 2** — run the *provided* input containing `school.mac`
   **before writing any mac pattern**. Read the error. You've seen it today:
   *`failed to legalize operation 'school.mac' that was explicitly marked illegal`*.
   Understand *why* partial conversion still fails here.
3. **Checkpoint 3** *(if time / stretch-track)* — add the `school.mac` → `arith.muli` + `arith.addi` pattern.

**Stretch:** (a) swap in `applyFullConversion` and re-run the max test — it now fails on `builtin.module` itself! Why? (In a *full* conversion, "unknown" is not good enough: nothing in our target legalizes `builtin.module` or `func.func`.) (b) *see a match refusal in action*: `school.max` is ODS-constrained to `i32`, so a real non-`i32` guard could never fire — instead, temporarily **invert** a guard (`if (op.getType().isInteger(32)) return rewriter.notifyMatchFailure(...);`), watch the refused match and the legalization failure under `--debug-only=dialect-conversion`, then revert.

<!-- Speaker notes:
~3 min briefing (0:45-0:50 slot). Core for the 30-min hands-on: checkpoint 1 (the max conversion) + checkpoint 2 (reading the failed-to-legalize error). Checkpoint 3 (the mac pattern) and the stretch goals are the if-time/stretch track; same checkpoint names and tests, just re-tiered. The task sheet (exercise2.md) carries matching tiering, including the inverted-guard experiment for stretch (b).
Emphasize checkpoint 2 is deliberate: hitting the legalization error with your own dialect, then fixing it, is the lesson — school.mac is explicitly illegal (whole dialect), so PARTIAL conversion still fails loudly.
Use OpConversionPattern<school::MaxOp> with the (op, adaptor, rewriter) signature; take operands from the ADAPTOR. (Honesty note: types don't change here, so op.getLhs() would produce the same final IR — the driver rewires replaced values at commit. It's about building the habit before a TypeConverter is in play; the lit tests can't tell the difference.)
Solutions in exercises/solutions/. Same lit command as part A.
Debrief material: show how small the finished conversion pass is (≈ the ComplexToStandard slide).
-->

---

# Recap

- **Pattern = the local rewrite only.** Match checks first, then mutate — *through the rewriter*, which the driver listens to.
- **The contract:** modify IR **iff** you return `success()`. Expensive-checks builds enforce it; normal builds let you suffer.
- **`walkAndApplyPatterns`** — one sweep, no fixpoint/fold/DCE. **`applyPatternsGreedily`** — worklist to a fixpoint, folds + DCEs by default; `failure()` = didn't converge; cyclic patterns **hang**.
- **Every pattern must strictly reduce something.**
- **Lowering = dialect conversion:** target declares legality, patterns rewrite, driver searches (fold → patterns → recursive legalization, with rollback). Partial spares unknown ops; full spares nothing.
- **Operands come from the adaptor.** Say it with me.
- `builtin.unrealized_conversion_cast` = boundary IOU; `-reconcile-unrealized-casts` collects the debts.

<!-- Speaker notes:
~2 min. Wrap-up slot (1:20-1:30), after the solution walkthrough. Run through quickly; each bullet maps to a slide students can revisit.
Actually make the room say "operands come from the adaptor" out loud. It works.
-->

---

# Cheat sheet: today's API surface

```cpp
// Pattern
struct P : OpRewritePattern<OpTy> {            // ctor: (MLIRContext*, benefit=1)
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rw) const override;
};
// Rewriter: rw.replaceOp / replaceOpWithNewOp<T>(op,...) / eraseOp
//           rw.modifyOpInPlace(op, [&]{...});  OpTy::create(rw, loc, ...)
//           return rw.notifyMatchFailure(op, "why");

// Sets & drivers
RewritePatternSet patterns(ctx);  patterns.add<A, B>(ctx);
walkAndApplyPatterns(op, std::move(patterns));            // one sweep
(void)applyPatternsGreedily(op, std::move(patterns), config); // fixpoint
GreedyRewriteConfig().setMaxIterations(10).setMaxNumRewrites(n)
    .enableFolding(true).enableConstantCSE(true);

// Conversion
struct C : OpConversionPattern<OpTy> {   // matchAndRewrite(op, ADAPTOR, rw)
};
ConversionTarget target(ctx);   // addLegalDialect / addIllegalDialect /
                                // addDynamicallyLegalOp / markUnknownOpDynamicallyLegal
TypeConverter tc;               // addConversion / addSource~/addTargetMaterialization
applyPartialConversion(op, target, std::move(patterns));  // or applyFullConversion
```

Debug flags: `--debug-only=greedy-rewriter | pattern-application | walk-rewriter | dialect-conversion`

<!-- Speaker notes:
~1 min. Screenshot-this slide. Everything shown compiles against this checkout's headers (PatternMatch.h, GreedyPatternRewriteDriver.h, WalkPatternRewriteDriver.h, DialectConversion.h).
-->

---

# Further reading

- `mlir/docs/PatternRewriter.md` — patterns, both drivers, debug flags; current and accurate.
- `mlir/docs/DialectConversion.md` — targets, adaptors, materializations, rollback vs. no-rollback (incl. the delayed-modification table).
- `mlir/docs/Tutorials/QuickstartRewrites.md` — pattern quickstart (already uses `OpTy::create`).
- `mlir/docs/Tutorials/MlirOpt.md` — the `mlir-opt` flag tour used in these demos.
- `mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp` — partial lowering, dynamic legality.
- `mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp` — full conversion, `TypeConverter`, `populate*` composition.
- Headers are documentation: `mlir/include/mlir/IR/PatternMatch.h`, `mlir/include/mlir/Transforms/DialectConversion.h`.

<!-- Speaker notes:
~30 s. All paths verified to exist in this checkout; both docs files were checked against the code while preparing this deck (PatternRewriter.md documents the current driver names; DialectConversion.md documents current adaptor/materialization semantics).
-->

---

<!-- _class: lead -->

# Next session: The Free Lunch

You wrote `x * 1 → x` as a pattern today.

Upstream, that's **not even a pattern** — it's a *folder*, and every pass gets it **for free**.

*Canonicalization, folding, CSE, DCE — and how to plug **your** dialect into all of them.*

<!-- Speaker notes:
~1 min. Tease: -canonicalize, which we used all day as a demo vehicle, is nothing but the greedy driver + every registered canonicalization pattern + folders — Session 3 opens the lid (the whole pass is ~20 relevant lines).
Also promised: why a dead school.max is NOT removed by canonicalize today (no Pure trait!), and the mystery of the arith.constant you never created (constant materialization).
Exercise 3 makes the school dialect a good citizen: Pure, hasFolder, hasConstantMaterializer, Commutative.
-->
