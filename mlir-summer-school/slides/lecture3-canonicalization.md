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

# Session 3 — The Free Lunch
## Canonicalization, Folding, CSE, DCE

**MLIR Summer School — Transformations (3/3)**

<!-- Speaker notes:
Welcome back. ~1 min. The arc so far: Session 1 = IR surgery by hand (walk, create, RAUW, erase). Session 2 = the same rewrite as a pattern, drivers do the orchestration. Today we remove the last piece of machinery: for a whole class of rewrites you don't write a pass OR a driver call — you attach tiny hooks to your ops, and the standard cleanup passes (-canonicalize, -cse) pick them up in every pipeline, forever. That's the "free lunch". Today is more conceptual than sessions 1-2, so it leans hard on predict-then-run quizzes: have your prediction ready before every run.
Timing (canonical session budget): 0:00-0:05 warm-up quiz (this slide + next two), 0:05-0:45 lecture — core path ≈40 min, 0:45-0:50 exercise briefing, 0:50-1:20 hands-on (30 min), 1:20-1:30 solution walkthrough + wrap-up. Slides marked ⏱ are the pressure-release valves — skip them in this order if behind schedule: constants dedup/hoisting, hasCanonicalizeMethod, the canonicalize pass source, pass options, per-operand effects, CSE dominance algorithm, CSE pass-statistics capture, LICM payoff, cleanup zoo, -remove-dead-values, -symbol-dce, fun stats. Per-slide "~N min" estimates are brisk targets; the "Timing check" markers in the notes are cumulative core-path minutes (flex slides not counted).
-->

---

## Where we are

| # | Session | You wrote |
|---|---------|-----------|
| 1 | Your First Pass | a hand-written walk + RAUW + erase |
| 2 | Patterns & Dialect Conversion | `OpRewritePattern`s, a driver ran them |
| **3** | **The Free Lunch** | **op hooks — upstream passes run them for you** |

Today's questions:

- What do `-canonicalize` and `-cse` *actually* do? Where do they stop?
- What is a **folder**, and why is it not just a small pattern?
- How does MLIR decide an op is safe to **delete** or **merge**? (side effects!)
- How do I plug **my dialect** into all of this? (→ Exercise 3)

<!-- Speaker notes:
~1 min (warm-up slot, 0:00-0:05). Remind them of the Session 2 debrief teaser: "you wrote x*1 -> x as a pattern; upstream that is not even a pattern, it's a folder, and you get it for free." Also recall the Session 1 stretch goal (constant on the LEFT of muli) — today they learn why upstream never bothers handling that case. Set expectations: smaller coding exercise today, but it ties ODS + patterns + this session together.
-->

---

## 🧠 Quiz: predict the output

```mlir
func.func @quiz(%x: i32, %m: memref<i32>) -> (i32, i32, i32) {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %sum = arith.addi %c2, %c3 : i32       // A: constant + constant
  %a = arith.addi %x, %x : i32           // B: twin #1
  %b = arith.addi %x, %x : i32           // B: twin #2 (identical!)
  %c0 = arith.constant 0 : i32
  %y = arith.addi %x, %c0 : i32          // C: x + 0
  %dead = arith.muli %x, %x : i32        // D: no users
  memref.store %sum, %m[] : memref<i32>  // E: a store
  return %a, %b, %y : i32, i32, i32
}
```

**Recap first (30 s):** which Session-2 driver do you bet powers `-canonicalize` —
and what does it do *besides* applying patterns?

Then, two separate runs — write down your predictions for each:

1. What does `-canonicalize` leave behind?
2. What does `-cse` change?

<!-- Speaker notes:
~3 min incl. collecting predictions (closes the 0:00-0:05 warm-up slot). Open with the 30-second Session-2 recap bridge: answer = the greedy driver (applyPatternsGreedily); besides patterns it also folds, trivially-DCEs, and simplifies regions. Take two or three shouts, don't resolve it fully — the cleanup-stack slide confirms it in a few minutes. Then give the room 90 seconds, then collect predictions with a show of hands per line: "Who thinks the twins get merged by -canonicalize?" "Who thinks addi(x,0) survives -cse?" "Does the store survive?" The common wrong guesses: (a) canonicalize merges the twins (it does NOT — CSE's job), (b) cse folds x+0 (it does NOT — folding is canonicalize's job), (c) the dead muli survives cse (it does not — cse has built-in DCE). Every one of these surprises is a section of this lecture. Input file: quiz.mlir with exactly this content.
-->

---

## 🔴 Run 1 (live): `-canonicalize`

```bash
build/bin/mlir-opt quiz.mlir -canonicalize
```

```mlir
func.func @quiz(%arg0: i32, %arg1: memref<i32>) -> (i32, i32, i32) {
  %c5_i32 = arith.constant 5 : i32
  %0 = arith.addi %arg0, %arg0 : i32
  %1 = arith.addi %arg0, %arg0 : i32
  memref.store %c5_i32, %arg1[] : memref<i32>
  return %0, %1, %arg0 : i32, i32, i32
}
```

- **A** `2+3` became `%c5_i32` → **folding** (constant folding)
- **C** `x+0` vanished, `return` uses `%arg0` → **folding** (identity)
- **D** dead `muli` erased, store survived → **DCE + side effects**
- **B** the twins are **both still there!** Canonicalize does *not* value-number.

<!-- Speaker notes:
~2 min (core path: 2/40). Keep this run LIVE — it resolves the warm-up predictions. Real output from this checkout (build/bin/mlir-opt; module wrapper trimmed on the slide). Walk through each mapping: A and C are the folding section; D is the DCE/side-effects section; B is the punchline — canonicalize = greedy driver = folding + patterns + trivial DCE + region simplification, but NOT common-subexpression elimination. Also point out all three original constants are gone and one new %c5_i32 appeared: fold produced an *attribute* and the infrastructure materialized a constant op — that's the constants section.
-->

---

## 🔴 Run 2 (live): `-cse` (and both together)

<div class="columns">
<div>

```bash
build/bin/mlir-opt quiz.mlir -cse
```

```mlir
func.func @quiz(%arg0: i32, %arg1: memref<i32>)
    -> (i32, i32, i32) {
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.addi %c2_i32, %c3_i32 : i32
  %1 = arith.addi %arg0, %arg0 : i32
  %c0_i32 = arith.constant 0 : i32
  %2 = arith.addi %arg0, %c0_i32 : i32
  memref.store %0, %arg1[] : memref<i32>
  return %1, %1, %2 : i32, i32, i32
}
```

</div>
<div>

```bash
build/bin/mlir-opt quiz.mlir -canonicalize -cse
```

```mlir
func.func @quiz(%arg0: i32, %arg1: memref<i32>)
    -> (i32, i32, i32) {
  %c5_i32 = arith.constant 5 : i32
  %0 = arith.addi %arg0, %arg0 : i32
  memref.store %c5_i32, %arg1[] : memref<i32>
  return %0, %0, %arg0 : i32, i32, i32
}
```

</div>
</div>

- `-cse`: twins merged (`return %1, %1`), dead `muli` gone (CSE has built-in DCE) — but **no folding**: `2+3` and `x+0` untouched
- Together: everything cleaned. That's why real pipelines run **both**.

<!-- Speaker notes:
~2 min (core path: 4/40). Keep this run LIVE — second half of the warm-up reveal. Real outputs (module wrapper trimmed, signatures wrapped to fit the columns). Count surprises in the room. Key contrast table to say out loud: canonicalize FOLDS but doesn't MERGE; cse MERGES but doesn't FOLD; both DCE. Fun check: `-cse -mlir-pass-statistics` on this input reports "1 num-cse'd, 1 num-dce'd" (verified). Today's lecture = explaining every line of these two outputs, then making your own dialect participate.
-->

---

## The cleanup stack: who runs what

```text
-canonicalize  ==  the greedy driver (Session 2!) with ALL registered patterns:
 ┌──────────────────────────────────────────────────────────────┐
 │ per op popped from the worklist:                             │
 │   1. trivial DCE      isOpTriviallyDead? → erase             │
 │   2. fold()           op's fold hook; results materialized   │
 │   3. patterns         canonicalization RewritePatterns       │
 │ per iteration, until fixpoint:                               │
 │   4. region simplify  unreachable blocks, dead block args,   │
 │                       (block merging at "aggressive")        │
 │ + constants deduped & hoisted to the entry block             │
 └──────────────────────────────────────────────────────────────┘
-cse           separate pass: dominance-scoped value numbering (+ trivial DCE)
-trivial-dce   just step 1 + unreachable blocks
```

**The economic argument:** implement `fold`/patterns/effects *once per op* —
every pass in every pipeline benefits. This is why upstream reviewers insist on folders.

<!-- Speaker notes:
~2 min (core path: 6/40). This is the roadmap slide; each box is a section. It also resolves the warm-up recap question: the greedy driver powers -canonicalize. Connect to Session 2: they already know the greedy driver's worklist loop — steps 1 and 2 are the parts we said "teaser for Session 3" about. Emphasize the economics: a lowering pass does not need to handle addi(x,0) because SOMEONE put that knowledge on arith.addi itself, once, in 2019 — and every pipeline since gets it. When you add a dialect, YOU are that someone. -trivial-dce is a recent addition; the old folklore "MLIR has no standalone DCE pass" is outdated in this checkout.
-->

---

## Folding: the most restricted rewrite

The contract, from `mlir/docs/Canonicalization.md`:

> `fold` has the restriction that **no new operations may be created**, and **only the
> root operation may be replaced (but not erased)**. It allows for **updating an
> operation in-place**, or returning a set of **pre-existing values (or attributes)**
> to replace the operation with. This ensures that the `fold` method is a truly
> **"local"** transformation, and can be invoked **without the need for a pattern
> rewriter**.

<sub>mlir/docs/Canonicalization.md:184-189</sub>

ODS: `let hasFolder = 1;` → generates a declaration for you to implement:

```cpp
// single-result op:
OpFoldResult MyOp::fold(FoldAdaptor adaptor);
// zero- or multi-result op:
LogicalResult MyOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results);
```

<!-- Speaker notes:
~2 min (core path: 8/40). Why so restricted? BECAUSE it's so restricted, it can run anywhere — no rewriter, no driver, no worklist needed (we'll list the places soon). Contrast with patterns: a pattern gets a PatternRewriter and can build arbitrary IR; a fold cannot create a single op. Note the two signatures: single-result ops get the compact OpFoldResult form; everything else fills a results vector 1:1 (partial folding not supported, and a fold can never REMOVE a 0-result op — the doc says so explicitly; erasing memref.copy(%x,%x) has to be a pattern). Beginners only need the single-result form today.
-->

---

## `OpFoldResult`: three ways to answer

```cpp
/// This class represents a single result from folding an operation.
class OpFoldResult : public PointerUnion<Attribute, Value> { ... };
```

<sub>mlir/include/mlir/IR/OpDefinition.h:273</sub>

What you return from `OpFoldResult MyOp::fold(FoldAdaptor)`:

| Return | Meaning | Caller does |
|---|---|---|
| `{}` (null) | no fold applies | tries patterns next |
| an **existing** `Value` | replace me with that value | RAUW + erase op |
| an `Attribute` | I am the constant *value* | materialize a constant op, RAUW + erase |
| `getResult()` (my *own* result) | I **mutated myself in place** | keeps op, notes progress |

The in-place trick: rewire your own operands, then return your own result —
e.g. `arith.extui(extui(x))` collapses to one `extui` this way.

<!-- Speaker notes:
~2 min (core path: 10/40). OpFoldResult is literally a PointerUnion<Attribute, Value> — worth showing, it demystifies everything. The three-way semantics (+the in-place special case of the Value branch) is the whole API. In-place example verified live (build/bin/mlir-opt inplace.mlir -canonicalize): extui i8->i16->i32 becomes one extui i8->i32; ExtUIOp::fold does getInMutable().assign(lhs.getIn()); return getResult(); (ArithOps.cpp:1617-1630). Warning worth stating: returning getResult() WITHOUT changing anything counts as "progress" — the driver re-enqueues you forever and canonicalize won't converge (caught by -canonicalize=test-convergence).
-->

---

## `FoldAdaptor` and the null-attribute gotcha

The adaptor hands you an `Attribute` per operand: the constant value **if** that
operand is defined by a `ConstantLike` op, **null otherwise**.

> If any of the operands are non-constant, a null `Attribute` value is provided instead.

<sub>mlir/docs/Canonicalization.md:243-249</sub>

```cpp
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  // 💥 CRASH-PRONE: getRhs() is null whenever %rhs is not a constant!
  auto bad = cast<IntegerAttr>(adaptor.getRhs());      // asserts on null

  // ✅ the idioms that survive non-constant operands:
  if (matchPattern(adaptor.getRhs(), m_Zero())) ...    // matcher handles null
  auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (!rhs) return {};                                  // no fold
  ...
}
```

Note: `adaptor.getRhs()` (an `Attribute`) vs. `getRhs()` (the SSA `Value`) — you can
use **both** in one fold.

<!-- Speaker notes:
~1 min (core path: 11/40). This is THE beginner crash in folders: fold runs on every op instance, including ones with zero constant operands — the adaptor is all nulls then. cast<> on a null Attribute asserts. Teach the two safe idioms: matchPattern with m_Zero/m_One (null-safe), and dyn_cast_if_present. The last bullet prevents a subtle confusion in the next slide: inside fold, plain getRhs() still gives the operand Value, adaptor.getRhs() gives the Attribute-or-null. Both are useful in the same function.
-->

---

## A real folder: `arith.addi`

```cpp
OpFoldResult arith::AddIOp::fold(FoldAdaptor adaptor) {
  // addi(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();                              // ← existing Value

  // addi(subi(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubIOp>())
    if (getRhs() == sub.getRhs())
      return sub.getLhs();                        // ← reading neighbors is fine!

  // addi(b, subi(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubIOp>())
    if (getLhs() == sub.getRhs())
      return sub.getLhs();

  return constFoldBinaryOp<IntegerAttr>(          // ← Attribute (or null)
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) + b; });
}
```

<sub>mlir/lib/Dialect/Arith/IR/ArithOps.cpp:419-437 (arrow annotations ours)</sub>

<!-- Speaker notes:
~2 min (core path: 13/40). Line-by-line: (1) identity fold returns an existing Value — this is quiz line C. (2)+(3) folds may TRAVERSE the IR read-only: getDefiningOp walks up the use-def chain (Session 1 skill) — creating/mutating other ops is forbidden, reading them is normal. (4) constFoldBinaryOp (mlir/include/mlir/Dialect/CommonFolders.h) is the upstream helper for element-wise constant folding: give it a lambda on APInt, it handles splat/dense/poison and returns an Attribute or null — this is quiz line A, producing %c5_i32. Question to plant for two slides from now: why does the zero check only look at the RHS? (Answer: Commutative trait — constants are already moved right.)
-->

---

## Folds can be one-liners

```cpp
// The shortest fold in the tree — a constant "folds" to its own value:
OpFoldResult arith::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }
```

<sub>mlir/lib/Dialect/Arith/IR/ArithOps.cpp:274 (intro comment ours)</sub>

```cpp
// Rank of a statically ranked memref is a compile-time constant:
OpFoldResult RankOp::fold(FoldAdaptor adaptor) {
  // Constant fold rank when the rank of the operand is known.
  auto type = getOperand().getType();
  auto shapedType = llvm::dyn_cast<ShapedType>(type);
  if (shapedType && shapedType.hasRank())
    return IntegerAttr::get(IndexType::get(getContext()), shapedType.getRank());
  return IntegerAttr();   // null attr == no fold
}
```

<sub>mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp:2001-2008 (intro + "null attr" comments ours)</sub>

<!-- Speaker notes:
~1 min (core path: 14/40). Two messages, one breath each. (1) ConstantOp::fold returning its own attribute looks silly but is load-bearing: it is HOW the whole system extracts values from constants — the m_Constant matcher literally calls fold on ConstantLike ops to get the attribute out. (2) RankOp shows folds don't need constant operands at all: folds can use TYPES — type information is compile-time knowledge too. Same story for tensor.dim on tensor<8x4xf32> folding to 4 (upstream test canonicalize.mlir:28, verified). Note the "return IntegerAttr()" spelling of "no fold" — a default-constructed attribute is null.
-->

---

## Constants: who builds the op my fold never created?

Fold returned `IntegerAttr 5` — but IR needs an *operation*. The dialect provides
the recipe:

```cpp
// ODS (on the dialect):  let hasConstantMaterializer = 1;   overrides this:
virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                       Type type, Location loc) {
  return nullptr;
}
```

<sub>mlir/include/mlir/IR/Dialect.h:83-86 (ODS comment ours)</sub>

```cpp
/// Materialize an integer or floating point constant.
Operation *arith::ArithDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return ub::PoisonOp::create(builder, loc, type, poison);
  return ConstantOp::materialize(builder, value, type, loc);
}
```

<sub>mlir/lib/Dialect/Arith/IR/ArithDialect.cpp:64-72</sub>

The produced op must be **`ConstantLike`** (a trait: one result, zero operands,
no side effects) — the driver asserts this.

⚠️ **The default `materializeConstant` returns `nullptr`** — and then attribute
fold results are **silently dropped**. No error, no fold. You will hit this
head-on in Exercise 3, Checkpoint 3.

<!-- Speaker notes:
~2 min (core path: 16/40). Closes the Attribute loop: fold produces the VALUE, the dialect's materializeConstant produces the OP. Returning an Attribute is "the sanctioned way to create a constant" from inside a fold. Note the modern builder style in real upstream code: ub::PoisonOp::create(builder, loc, ...) — ops are created via OpTy::create. ConstantLike is the trait that marks constant ops (arith.constant has it); the greedy driver asserts the materialized op has it and that the result type matches (GreedyPatternRewriteDriver.cpp:553-556). The warning is the classic custom-dialect bug: the greedy driver's materialization-failure path (GreedyPatternRewriteDriver.cpp:535-551) just cleans up and moves on — deliberately no diagnostic, because "cannot materialize" is a legitimate answer. Foreshadow: this exact silence is the designed teaching beat of Exercise 3 checkpoint 3.
-->

---

## Constants: dedup, hoisting — and one infinite loop ⏱

```mlir
// INPUT                                   // AFTER -canonicalize (verified)
func.func @dedup(%m: memref<i32>) -> i32 { //  func.func @dedup(...) -> i32 {
  %a = arith.constant 5 : i32              //    %c5_i32 = arith.constant 5 : i32
  memref.store %a, %m[] : memref<i32>      //    memref.store %c5_i32, %arg0[] ...
  %b = arith.constant 5 : i32              //    return %c5_i32 : i32
  return %b : i32                          //  }
}
```

Globally applied rules (`docs/Canonicalization.md:110-125`): constant-like ops are
**uniqued** (one op per `(dialect, attribute, type)`) and **hoisted** into the entry
block of the enclosing isolated-from-above region (e.g. the function).

**Why does the driver refuse to fold `ConstantLike` ops themselves?**
`constant` folds to its attribute → driver materializes a new constant →
new constant folds to its attribute → ... The driver skips them:

```cpp
// Try to fold this op. Do not fold constant ops. That would lead to an
// infinite folding loop ...
if (config.isFoldingEnabled() && !op->hasTrait<OpTrait::ConstantLike>()) {
```

<sub>mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp:494-498</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: constants are automatically uniqued and hoisted to the entry block of the function, and the driver refuses to fold ConstantLike ops themselves to avoid an infinite fold→materialize loop. (+2 min if presented.) Output on the slide is pre-captured (dedup.mlir; command: build/bin/mlir-opt dedup.mlir -canonicalize) — don't run live. Two identical %c5 become one used by both store and return. Whiteboard moment: ask the room why folding arith.constant itself would loop before revealing the driver comment — it's a nice comprehension check of the fold->materialize cycle. Skip the DialectFoldInterface/shouldMaterializeInto barrier-region detail; "entry block of the function" is the right mental model at this level.
-->

---

## Where does `fold` actually run? Everywhere.

1. **The greedy driver** — hence `-canonicalize`: fold is tried on every worklist
   op *before* patterns.
2. **`OpBuilder::createOrFold<OpTy>(...)`** — eager folding at IR-*construction*
   time: builds the op, folds it, erases it again if it folded.
   <sub>mlir/include/mlir/IR/Builders.h:527-573</sub>
3. **Dialect conversion** — illegal ops are legalized *by folding first*, before
   your conversion patterns run (`ConversionConfig::foldingMode`, default
   `BeforePatterns`). <sub>mlir/include/mlir/Transforms/DialectConversion.h:1466-1467</sub>
   → the classic head-scratcher: *"my conversion pattern never ran!"* — the op folded away.
4. **`m_Constant(&attr)`** — the matcher extracts constant values by calling `fold`
   on `ConstantLike` ops. <sub>mlir/include/mlir/IR/Matchers.h:74-101</sub>

**Consequence:** folds must be **fast** (they run millions of times) and
**always-correct** (there is no flag to turn them off in practice).

<!-- Speaker notes:
~1 min (core path: 17/40). The payoff of the restrictive contract: no rewriter needed means fold can be invoked from anywhere, and it is. Point 3 is a debugging war story worth telling: students in Session 2 wrote a conversion pattern for school.mac; if mac had a folder that applied, the pattern would silently never fire — dialect conversion tries fold on illegal ops by default (BeforePatterns; can be set to Never/AfterPatterns via ConversionConfig). Point 2 foreshadows the pipeline advice at the end: passes that emit naive IR can use createOrFold to avoid emitting garbage in the first place. Timing check: ~17 min of the 40-min core path (≈22 min wall clock incl. the warm-up quiz).
-->

---

## 🧠 Quiz: legal fold or not?

Which of these may be implemented as a `fold` method?

1. `arith.addi(%x, %c0) → %x`

2. `arith.muli(%x, %c2) → arith.shli(%x, %c1)`

3. *"my op is commutative — swap the operands so the constant is on the right"*

4. `memref.copy(%a, %a)` — can `CopyOp::fold` erase the self-copy?

<!-- Speaker notes:
~1 min (core path: 18/40). Have them vote per candidate. Answers on the next slide. Watch for candidate 2 — after two sessions of strength reduction they WANT it to be legal. Candidate 4 should ring a bell: erasing the self-copy was literally their first pattern in Session 2.
-->

---

## ✅ Legal fold or not — answers

1. `addi(x, 0) → x` — **✅ fold.** Returns an existing value. The textbook case.

2. `muli(x, 2) → shli(x, 1)` — **❌ pattern.** Creates a *new operation* —
   forbidden in fold. (Your Exercise 2A pattern stays a pattern!)

3. Swap operands in place — **✅ fold** (the in-place variant: mutate your own
   operands, return `getResult()`). In fact you don't even write it: the
   `Commutative` *trait* does exactly this — next slide.

4. `memref.copy(%a, %a)` — **❌ pattern.** `memref.copy` has **zero results**, so
   there is nothing for a fold to replace — the only useful rewrite is *erasing*
   the op, and folds may **not erase 0-result ops**. That is exactly why Session
   2's first pattern (`FoldSelfCopy`) exists as a `RewritePattern`.

Rule of thumb from the docs:

> A canonicalization should always be implemented as a `fold` method if it can
> be, otherwise it should be implemented as a `RewritePattern`.

<sub>mlir/docs/Canonicalization.md:300-301</sub>

<!-- Speaker notes:
~2 min (core path: 20/40). Candidate 2 is the fold-vs-pattern litmus test: need to CREATE ops => pattern; need to ERASE a 0-result op => pattern; replace-with-existing-value-or-constant => fold. Candidate 3: the in-place fold mechanism is legal and real, but for commutativity specifically MLIR gives it away as a trait. Candidate 4 is the Session-2 tie-in: FoldSelfCopy was their first pattern, and now they know why it could never be a folder — the fold contract only lets you replace results (a 0-result op has none) or mutate in place, never erase the op; the multi-result fold signature fills a results vector 1:1. Same fact on the fold-contract slide's notes and the hasCanonicalizeMethod slide (cf.assert erase). Close with the doc quote — it's the decision rule for Exercise 3's stretch discussion (mac folder candidates).
-->

---

## Canonicalization: why one form beats N forms

**The point of a canonical form:** every pass downstream matches *one* shape
instead of all equivalent shapes.

Session 1 stretch goal: handle `arith.muli %c4, %x` (constant on the *left*).
Upstream never writes that code. Why?

```mlir
// INPUT                                // AFTER -canonicalize (verified)
%c4 = arith.constant 4 : i32            //  %c4_i32 = arith.constant 4 : i32
%0 = arith.addi %c4, %x : i32           //  %0 = arith.addi %arg0, %c4_i32 : i32
```

Constants move right **for free**: ops marked `Commutative` in ODS get a *trait
fold* (`impl::foldCommutative`) — even ops with **no fold method at all**.

<sub>mlir/include/mlir/IR/OpDefinition.h:1165-1171, mlir/lib/IR/Operation.cpp:862</sub>

**That's why every upstream folder only checks `adaptor.getRhs()`** — after
canonicalization, the constant *is* on the right. A LHS check would be dead code.

<!-- Speaker notes:
~2 min (core path: 22/40). This slide answers the question planted at AddIOp::fold. Output shown is pre-captured (commut.mlir; command: build/bin/mlir-opt commut.mlir -canonicalize) — don't run live: addi %c4, %x really becomes addi %x, %c4. Mechanism: the ODS-generated fold hook falls through to trait folding even when there's no hasFolder; Commutative contributes foldCommutative which moves constant operands right as an in-place fold. Generalize: canonical form is a CONTRACT between rewrites — "constants on the right" means N patterns each save half their match code. Exercise 3 stretch (a) has them add Commutative to school.max and watch max(c, x) normalize for free.
-->

---

## Attaching patterns to ops: the ODS hooks

```tablegen
def Arith_AddIOp : Arith_IntBinaryOpWithOverflowFlags<"addi", [Commutative, ...]> {
  ...
  let hasFolder = 1;         // OpFoldResult fold(FoldAdaptor);
  let hasCanonicalizer = 1;  // static void getCanonicalizationPatterns(
                             //     RewritePatternSet &, MLIRContext *);
}
```

<sub>mlir/include/mlir/Dialect/Arith/IR/ArithOps.td:271,306-307</sub>

```cpp
// You implement it with Session-2 patterns:
void arith::AddIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<AddIAddConstant, AddISubConstantRHS, AddISubConstantLHS,
               AddIMulNegativeOneRhs, AddIMulNegativeOneLhs>(context);
}
```

<sub>mlir/lib/Dialect/Arith/IR/ArithOps.cpp:439-443 (intro comment ours)</sub>

`-canonicalize` collects these hooks from **every registered op** — your patterns
run in every pipeline that canonicalizes, without you owning a pass.

<!-- Speaker notes:
~1 min (core path: 23/40). Bridge from Session 2: getCanonicalizationPatterns is just patterns.add<...> — the exact API they used in the -school-peephole exercise, except the driver invocation is now upstream's problem. This is the "free lunch" mechanism for patterns (hasFolder was the one for folds). The pattern names in the add<> call are a mix of DRR-generated (next-next slide) and hand-written classes — the pattern set doesn't care.
-->

---

## The lighter hook: `hasCanonicalizeMethod` ⏱

Exactly one simple pattern? Skip the pattern class entirely:

```tablegen
// mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td:60
let hasCanonicalizeMethod = 1;
```

```cpp
LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  // Erase assertion if argument is constant true.
  if (matchPattern(op.getArg(), m_One())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}
```

<sub>mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp:80-87</sub>

ODS wraps this static method in a pattern for you. Note: *this* rewrite could
never be a fold — `cf.assert` has **zero results**, and it's being **erased**.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: there is a middle tier, hasCanonicalizeMethod, which wraps a single static canonicalize method in a pattern for you — it's on the cheat sheet. (+1 min if presented.) Tier list so far: fold (most restricted, runs everywhere) < canonicalize method (one pattern, no boilerplate) < getCanonicalizationPatterns (full pattern list). Only 19 ops upstream use hasCanonicalizeMethod vs 183 with hasCanonicalizer — it's the niche middle tier, but perfect for exactly-one-C++-pattern cases. The closing note reinforces the fold-vs-pattern rule: erasing a 0-result op requires a rewriter, hence a pattern.
-->

---

## You'll also see: declarative patterns (DRR)

```tablegen
// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
def AddIAddConstant :
    Pat<(Arith_AddIOp:$res
          (Arith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2))>;
```

<sub>mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:63-69</sub>

- TableGen `Pat<(match...), (replace...)>` → generates a `RewritePattern` class,
  registered via the same `patterns.add<AddIAddConstant>` you just saw
- This exact pattern powered `(x+3)+4 → x+7` in demos you'll see today
- **Recognize it; you don't need to write it.** C++ patterns can do everything DRR
  can. (`mlir/docs/DeclarativeRewrites.md` if you're curious)

<!-- Speaker notes:
~1 min (core path: 24/40). This is a 45-second RECOGNIZE-NOT-WRITE slide — deliberately shallow: students WILL bump into .td pattern files when reading arith (82 DRR defs in ArithCanonicalization.td alone), so they should recognize the shape and know it compiles down to the same RewritePattern machinery. Do not teach the syntax; do not take questions on it — defer to DeclarativeRewrites.md. Moving on.
-->

---

## Canonicalization design rules (the docs mean it)

From `mlir/docs/Canonicalization.md:36-64` — the four you must internalize:

- **Convergence:** *"Repeated applications of patterns should converge. Unstable or
  cyclic rewrites are considered a bug"* (Session 2's A→B, B→A hang!)
- **Fewer uses of a value:** *"it is generally good to canonicalize `x + x` into
  `x * 2`, because this reduces the number of uses of x by one."*
- **Cheap only:** *"Patterns with expensive running time (i.e. have O(n) complexity)
  or complicated cost models don't belong to canonicalization"* — it runs to fixpoint!
- **Canonical ≠ optimal:** *"performance improvements are not necessary for
  canonicalization"* — the goal is enabling analyses.

And the one that surprises everyone:

> Pass pipelines should not rely on the canonicalizer pass for correctness. They
> should work correctly with all instances of the canonicalization pass removed.

<!-- Speaker notes:
~2 min (core path: 26/40). Discussion quiz to run verbally here (keep it to two quick opinions): "Is addi(x,x) -> muli(x,2) a good canonicalization? Is muli(x,4) -> shli(x,2)?" First: yes — the docs explicitly bless it via the fewer-uses rule. Second: debatable! It's target-dependent strength reduction; whether shl is cheaper than mul is a backend concern, and 'canonical' is about picking ONE form globally, not about being fast. There is no formally defined canonical form in MLIR — the de-facto form evolves (docs:68-70). The last quote is exam material: canonicalize is best-effort, so a pipeline that BREAKS without it has a correctness bug. Also connect convergence to Session 2's "will this terminate" quiz — a cyclic canonicalization is that bug, shipped to everyone who runs -canonicalize.
-->

---

## The canonicalize pass is tiny ⏱

```cpp
struct Canonicalizer : public impl::CanonicalizerPassBase<Canonicalizer> {
  LogicalResult initialize(MLIRContext *context) override {
    // ... (config + `filter-dialects` option handling elided) ...
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      if (isAllowed(dialect))
        dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      if (isAllowed(&op.getDialect()))
        op.getCanonicalizationPatterns(owningPatterns, context);
    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    LogicalResult converged =
        applyPatternsGreedily(getOperation(), *patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();
  }
};
```

<sub>mlir/lib/Transforms/Canonicalizer.cpp:30-95 (option plumbing trimmed; 105 lines total)</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the whole pass is ~20 relevant lines — initialize() collects canonicalization patterns from every registered dialect and op into a frozen set, runOnOperation() calls applyPatternsGreedily, and non-convergence is deliberately NOT a pass failure. (+2 min if presented.) The reveal: the pass you've been running since slide 4 is ~20 relevant lines, and every piece is Session-2 vocabulary. initialize() runs once — patterns collected from all dialects + all registered ops, frozen (FrozenRewritePatternSet: built once, shared across threads). runOnOperation() = applyPatternsGreedily. Folding is NOT collected as patterns — it lives inside the greedy driver, remember the cleanup-stack slide. And note the convergence philosophy in code: failed() means "didn't converge in max-iterations", and that is deliberately NOT a pass failure — only the test-only flag turns it into one (upstream tests run canonicalize{test-convergence} to catch cyclic patterns).
-->

---

## `-canonicalize` pass options (verified defaults) ⏱

| Option | Default | Meaning |
|---|---|---|
| `top-down` | `true` | seed the worklist in top-down order |
| `region-simplify` | `normal` | `disabled` \| `normal` (dead args etc.) \| `aggressive` (+ block merging) |
| `max-iterations` | `10` | outer fixpoint iterations |
| `max-num-rewrites` | `-1` (no limit) | rewrites within one iteration |
| `test-convergence` | `false` | *test only:* fail the pass on non-convergence |

<sub>mlir/include/mlir/Transforms/Passes.td:19-63</sub>

```bash
mlir-opt in.mlir -canonicalize="region-simplify=aggressive max-iterations=2"
```

Careful: the *C++* `GreedyRewriteConfig` default for region simplification is
`Aggressive`, but the *pass option* default is `normal` — plain `-canonicalize`
does **not** merge identical blocks. Demo coming up.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: defaults worth knowing are top-down=true, region-simplify=normal, max-iterations=10 — and note the C++ GreedyRewriteConfig default for region simplification is Aggressive while the pass-option default is normal, which the upcoming live demo exploits; the table stays in the deck for reference. (+1 min if presented.) All defaults verified against Passes.td and mlir-opt --help on this build. Also exists: cse-between-iterations (default false — canonicalize can optionally run full CSE between iterations; that's the exception to "canonicalize never CSEs"), filter-dialects, disable-patterns/enable-patterns. Don't dwell; the region-simplify mismatch note sets up the region-simplify live demo later in the deck (after the DCE section).
-->

---

## DCE: when may the compiler delete an op?

```cpp
bool mlir::isOpTriviallyDead(Operation *op) {
  return op->use_empty() && wouldOpBeTriviallyDead(op);
}

bool mlir::wouldOpBeTriviallyDead(Operation *op) {
  if (op->mightHaveTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<SymbolOpInterface>(op))    // symbols are referenced by *name*
    return false;                    // (annotation ours)
  return wouldOpBeTriviallyDeadImpl(op);
}
```

<sub>mlir/lib/Interfaces/SideEffectInterfaces.cpp:35-37, 312-318</sub>

`wouldOpBeTriviallyDeadImpl`: every effect of the op must be a **Read**, or an
**Allocate of one of the op's own results**; recurse into regions for
`RecursiveMemoryEffects` ops. **No effect info at all ⇒ conservatively NOT dead.**

So: *dead* means **no users AND no observable effects** — and MLIR needs the op
to *declare* its effects.

<!-- Speaker notes:
~2 min (core path: 28/40). This tiny predicate is THE definition of "dead" used by canonicalize, cse, and trivial-dce alike. Three exclusions to call out: terminators (structurally required), symbol ops (func.func has zero SSA uses by construction — its references are attributes; that's why -symbol-dce exists, later slide), and unknown-effect ops (the safety default: MLIR hosts arbitrary dialects, so "I don't know" must mean "don't touch"). The Allocate-of-own-result clause is the fun subtlety — quiz in three slides.
-->

---

## Declaring effects in ODS: `Pure` and friends

```tablegen
// Op has no effect on memory but may have undefined behavior.
def NoMemoryEffect : MemoryEffects<[]>;

// Op has recursively computed side effects.
def RecursiveMemoryEffects : NativeOpTrait<"HasRecursiveMemoryEffects">;

// Marks an Operation as always speculatable.
def AlwaysSpeculatable : TraitList<[
    ConditionallySpeculatable, AlwaysSpeculatableImplTrait]>;

// Always speculatable operation that does not touch memory.  These operations
// are always legal to hoist or sink.
def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>;
```

<sub>mlir/include/mlir/Interfaces/SideEffectInterfaces.td:94-147 (elided)</sub>

- Effect kinds: **Read / Write / Allocate / Free** (on a *resource*, optionally tied to a value)
- `Pure` = `NoMemoryEffect` **+** `AlwaysSpeculatable`
- ⚠️ `NoMemoryEffect` alone may still have **UB** (divide by zero!) → dead-code-removable, but **not hoistable**

<!-- Speaker notes:
~2 min (core path: 30/40). Timing check: ~30 min of the 40-min core path — if you're past 0:35 wall clock here, start skipping the remaining ⏱ slides. The trait algebra matters: Pure is a TraitList of two orthogonal claims — "touches no memory" and "safe to execute speculatively (no UB, no trap)". arith.addi is Pure. arith.divsi is NoMemoryEffect + only ConditionallySpeculatable (division by zero is UB) — verified in ArithOps.td:642-643 + base class. Consequence teased here, demoed on the LICM slide (flex): a DEAD divsi is removed (removal never executes anything) but LICM will NOT hoist a divsi out of a loop (hoisting = speculating that the loop body would have run — the loop may have zero iterations precisely to guard the division). Also: RecursiveMemoryEffects for region ops (scf.for/scf.if use it — "my effects are my body's effects"); MemoryEffects<[]> on a region op would wrongly assert the whole body is effect-free.
-->

---

## Per-operand effects & the unknown-op default ⏱

Effects can be tied to a *specific operand* with the `Arg<...>` form:

```tablegen
// memref.load — read effect on the $memref operand:
let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                         [MemRead]>:$memref,
                     Variadic<Index>:$indices, ...);

// memref.store — write effect on the $memref operand:
let arguments = (ins AnyType:$value,
                     Arg<AnyMemRef, "the reference to store to",
                         [MemWrite]>:$memref,
                     Variadic<Index>:$indices, ...);
```

<sub>mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td:1313-1315, 2088-2091</sub>

```cpp
} else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
  // No interface, no recursive trait: cannot know -> not effect free.
  return false;
```

<sub>mlir/lib/Interfaces/SideEffectInterfaces.cpp (isMemoryEffectFree, condensed)</sub>

**Forgetting to annotate = silently disabling DCE, CSE, and LICM for your op.**
This is Exercise 3, Checkpoint 1.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: effects can also be tied to a specific operand (Arg<..., [MemRead]> — how memref.load declares its read; spelling is on the cheat sheet), and remember the default for unannotated ops is "unknown = effectful" — forgetting the annotation silently disables DCE/CSE/LICM for your op, which is exactly Exercise 3 checkpoint 1. (+1 min if presented.) Two ODS spellings: op-level trait list ([Pure], MemoryEffects<[MemRead, MemWrite]>) and the per-operand Arg<> decorator, which ties the effect to that operand's VALUE (enables per-value queries — how CSE knows which memref a load reads). Then the conservative default, in code. The #1 beginner footgun, both directions: forgetting Pure silently disables every optimization for your op (the visceral Exercise 3 checkpoint 1); slapping Pure on an op that traps or writes memory makes DCE/CSE/LICM miscompile SILENTLY — effects are trusted, never verified.
-->

---

## 🧠 Quiz: which of these can be erased?

All five ops have **zero users**:

```mlir
func.func private @helper(i32) -> i32

func.func @dce_quiz(%m: memref<4xi32>, %i: index, %v: i32) {
  %dead1 = arith.addi %v, %v : i32              // 1
  %dead2 = memref.load %m[%i] : memref<4xi32>   // 2
  memref.store %v, %m[%i] : memref<4xi32>       // 3 (0 results = 0 users!)
  %dead3 = func.call @helper(%v) : (i32) -> i32 // 4
  %dead4 = "mystery.op"(%v) : (i32) -> i32      // 5 (unregistered dialect)
  %dead5 = memref.alloc() : memref<8xf32>       // 6 — bonus
  return
}
```

Which lines does `-canonicalize` delete?

<!-- Speaker notes:
~1 min (core path: 31/40). Vote line by line, fast. The interesting fights: line 2 (a LOAD — "reads are effects, right?"), line 4 (a call to a function that, if you read @helper's body... there is no body — it's a declaration), line 5 (unregistered), line 6 (alloc "allocates memory — surely an effect"). Input file dce.mlir; needs -allow-unregistered-dialect for mystery.op.
-->

---

## ✅ DCE quiz — real output

```bash
build/bin/mlir-opt dce.mlir -canonicalize -allow-unregistered-dialect
```

```mlir
func.func @dce_quiz(%arg0: memref<4xi32>, %arg1: index, %arg2: i32) {
  memref.store %arg2, %arg0[%arg1] : memref<4xi32>
  %0 = call @helper(%arg2) : (i32) -> i32
  %1 = "mystery.op"(%arg2) : (i32) -> i32
  return
}
```

| Op | Fate | Why |
|---|---|---|
| dead `addi` | ✅ erased | `Pure`, no users |
| dead `load` | ✅ erased | only effect is a **Read** → unobservable if unused |
| `store` | ❌ kept | **Write** effect — zero results ≠ dead |
| dead `call` | ❌ kept | callee may have effects → conservative |
| `"mystery.op"` | ❌ kept | unregistered ⇒ **unknown = effectful** |
| dead `alloc` | ✅ erased | **Allocate of its own result** is ignorable |

<!-- Speaker notes:
~2 min (core path: 33/40). Output is pre-captured — don't run live (command on the slide). Real output (module wrapper + @helper declaration trimmed; same result with -trivial-dce, verified). The two aha moments: (1) a dead load IS erased — Read effects are fine for deadness, the read is unobservable if nobody uses the value; contrast with CSE later where reads need care because of intervening writes. (2) dead memref.alloc IS erased despite having an (Allocate) effect — wouldOpBeTriviallyDeadImpl explicitly ignores Allocate effects on the op's OWN results: allocating something nobody can ever see is not observable. (Deallocation-side note: upstream even has a SimplifyDeadAlloc pattern that erases alloc+load+dealloc groups.) Where DCE runs: inside the greedy driver (-canonicalize), inside -cse, and standalone as -trivial-dce (this checkout; 37-line pass). None of them do liveness analysis — dead use-def CYCLES need region simplification's liveness fixpoint or -remove-dead-values.
-->

---

## Region simplification: three rules

Part of `-canonicalize` (step 4 of the cleanup stack). Three jobs, in order:

1. **Unreachable blocks are erased** — any non-entry block with no
   predecessors is gone.

2. **Region DCE** — a backward **liveness** fixpoint: kills dead use-def
   *cycles* that trivial DCE can't see, and drops dead **non-entry block
   arguments**.

3. **Identical blocks are merged** (+ redundant block-arg dropping) — **only
   at `region-simplify=aggressive`**, not at the default `normal`.

<!-- Speaker notes:
~1 min (core path: 34/40). Rules restated from mlir::simplifyRegions — open mlir/lib/Transforms/Utils/RegionUtils.cpp:1214-1228 on demand: eraseUnreachableBlocks, then runRegionDCE, then mergeIdenticalBlocks + dropRedundantArguments gated on the mergeBlocks flag. Step 2 is more powerful than trivial DCE: it's an optimistic liveness fixpoint, so mutually-dependent dead ops (a cycle through block args) die too, and dead non-entry block arguments are dropped (entry block args are the enclosing op's interface — removing FUNCTION arguments is -remove-dead-values' job). Step 3 is where the pass-option default matters — demo next.
-->

---

## 🔴 Live demo: `region-simplify=normal` vs `aggressive`

```bash
build/bin/mlir-opt blocks.mlir -canonicalize
build/bin/mlir-opt blocks.mlir -canonicalize="region-simplify=aggressive"
```

<div class="columns">
<div>

```mlir
// INPUT (^bb3 unreachable;
//        ^bb1 == ^bb2)
func.func @blocks(%cond: i1, %a: i32) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = arith.muli %a, %a : i32
  return %0 : i32
^bb2:
  %1 = arith.muli %a, %a : i32
  return %1 : i32
^bb3:
  return %a : i32
}
```

</div>
<div>

```mlir
// -canonicalize (normal): ^bb3 gone, twins kept
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %0 = arith.muli %arg1, %arg1 : i32
  return %0 : i32
^bb2:  // pred: ^bb0
  %1 = arith.muli %arg1, %arg1 : i32
  return %1 : i32

// region-simplify=aggressive: fully merged!
  %0 = arith.muli %arg1, %arg1 : i32
  return %0 : i32
```

</div>
</div>

<!-- Speaker notes:
~2 min (core path: 36/40). This is the ONE demo besides the opening quiz runs that stays live — run both commands for real; outputs above are the fallback (real, lightly trimmed to the function body for the right column). Prediction to collect first: "what does plain -canonicalize do to ^bb1/^bb2?" Most say merged — surprise: default is normal, only the unreachable ^bb3 dies. With aggressive, the identical blocks merge, which makes the cond_br's two targets equal, which lets a cf pattern fold the branch away — a nice cascade of block merging + patterns cooperating inside one greedy fixpoint. Why isn't aggressive the default for the pass? Block merging can hurt debuggability/readability of mid-pipeline IR and costs compile time; pipelines opt in.
-->

---

## 🧠 Quiz: CSE — what merges?

**The idea:** two *structurally identical* ops computing the same thing? Keep the
first, replace all uses of the second. From the real upstream test suite:

```mlir
func.func @simple_constant() -> (i32, i32) {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}

func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
  %0 = memref.alloc() : memref<2x1xf32>
  %1 = memref.alloc() : memref<2x1xf32>
  return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
}
```

<sub>mlir/test/Transforms/cse.mlir:4, 110 (CHECK lines stripped)</sub>

Identical constants, identical allocs. What does `-cse` merge?

<!-- Speaker notes:
~1 min (core path: 37/40). Collect votes quickly. The constants merge (one %c1_i32, returned twice). The allocs do NOT — and this time it's not conservatism about unknown ops: memref.alloc declares its effects perfectly well. Ask WHY before flipping the slide: two allocs are semantically DIFFERENT values — distinct buffers! Merging them would alias every user. "Identical text" is not "same value" once effects are involved.
-->

---

## ✅ CSE and effects: three regimes, three rules

Every non-terminator op lands in exactly one bucket:

1. **Effect-free** (`Pure` pays off again) → deduped via **hash table +
   dominance**. This merged the constants — and the twin `addi`s from the
   opening quiz.

2. **Read-only** (single **Read** effect, e.g. `memref.load`) → merged **only
   within one block**, and only if **nothing side-effecting sits in between**
   the two reads. No alias analysis — unknown-effect ops count as writes.

3. **Everything else** → **never CSE'd**: writes, frees, *allocates* (our
   quiz! two `memref.alloc`s are two **distinct buffers**), unknown ops.

<!-- Speaker notes:
~2 min (core path: 39/40). Rules restated from CSEDriver::simplifyOperation — open mlir/lib/Transforms/Utils/CSE.cpp:249-295 on demand (also excluded there: terminators and ops with multi-block regions). Regime 2: CSE is not an alias-analysis framework; it scans linearly between the two reads and treats unknown-effect ops as writes. Regime 3 answers the quiz: memref.alloc declares its effects perfectly well — merging two allocs would alias every user, so "identical text" is not "same value" once effects are involved. Verified demo (cse3.mlir): two loads merge with nothing in between; a store in between blocks the merge. Also verified: addi(%a,%b) and addi(%b,%a) DO merge — equivalence is commutativity-aware for Commutative ops.
-->

---

## CSE: the algorithm — dominance-scoped value numbering ⏱

**Equivalence** (`OperationEquivalence`): same op name, same attributes &
properties, same operand `Value`s (order-insensitive for `Commutative` ops),
same result types, regions compared recursively. **Locations ignored.**

**Scoping:** one hash-table scope per dominator-tree node — an op can only be
replaced by an equivalent op that **dominates** it:

```text
      ^entry                     scope stack while visiting ^right:
      /    \                        { ^entry's ops }        ← visible
  ^left    ^right                   { ^right's ops }        ← visible
                                 ^left's ops: NOT visible
                                 (^left doesn't dominate ^right)
```

- Walk the dom tree depth-first; push a scope per node, pop on the way out
- `IsolatedFromAbove` regions get a **fresh** table (no implicit captures)
- Why a *pass* and not a pattern? Needs **global state**: the dominator tree +
  the cross-block hash table. Patterns are local rewrites.

<sub>mlir/lib/Transforms/Utils/CSE.cpp:359-385; OperationSupport.cpp:837-903</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: equivalence is structural (same op name, attributes, operand Values — commutativity-aware — and result types), a match must DOMINATE the op it replaces (one hash-table scope per dom-tree node), and the global state (dom tree + table) is why CSE is a pass and not a pattern. (+2 min if presented.) Draw the dom tree on the whiteboard if possible. Key points: (1) equivalence is STRUCTURAL — same operands as SSA values, all inherent+discardable attributes equal; two addis with different overflow flags don't merge; only locations are ignored. (2) dominance scoping is what makes "replace uses of B with A" safe: A's result must be available wherever B's users are. An op inside an scf.if CAN reuse a dominating op from outside (non-isolated regions share the scope), but not vice versa, and never across sibling branches. (3) The "why a pass" beat answers the cleanup-stack question: canonicalize could not do this with a local pattern — though canonicalize has an optional cse-between-iterations mode that calls this same library function. CSE.cpp the PASS is 57 lines; the algorithm lives in Utils/CSE.cpp as eliminateCommonSubExpressions.
-->

---

## 📸 Captured output: CSE regimes + pass statistics ⏱

```bash
build/bin/mlir-opt cse3.mlir -cse -mlir-pass-statistics
```

```mlir
// loads with a store in between — NOT merged:
%0 = memref.load %arg0[%arg1] : memref<4xi32>
memref.store %arg2, %arg0[%arg1] : memref<4xi32>
%1 = memref.load %arg0[%arg1] : memref<4xi32>

// loads with nothing in between — merged:
%0 = memref.load %arg0[%arg1] : memref<4xi32>
%1 = arith.addi %0, %0 : i32          // was: addi %0, %1

// commutative twins  addi(%a,%b) / addi(%b,%a) — merged!
```

```text
===-------------------------------------------------------------------------===
                         ... Pass statistics report ...
===-------------------------------------------------------------------------===
CSEPass
  (S) 2 num-cse'd - Number of operations CSE'd
  (S) 0 num-dce'd - Number of operations DCE'd
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, just mention that -mlir-pass-statistics exists and reports what a cleanup pass did (on the opening quiz input: 1 num-cse'd / 1 num-dce'd). Pre-captured — do not run live unless ahead of schedule; command: build/bin/mlir-opt cse3.mlir -cse -mlir-pass-statistics. (+1 min if presented.) Real (trimmed) output of that command; cse3.mlir contains @twins, @loads_store_between, @loads_no_store, @allocs. Show -mlir-pass-statistics: passes can export counters (Session 1 stretch goal used one) and CSE reports both its CSE count and its built-in DCE count. Good flag to remember when you wonder whether a cleanup pass actually did anything.
-->

---

## Effects payoff: LICM is a two-line pass ⏱

```cpp
void LoopInvariantCodeMotion::runOnOperation() {
  // Walk all loops in a function in innermost-loop-first order.
  getOperation()->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}
```

<sub>mlir/lib/Transforms/LoopInvariantCodeMotion.cpp:41-47 (comment abridged)</sub>

The hoisting predicate inside `moveLoopInvariantCode` is exactly **`isPure(op)`**
(+ operands defined outside the loop). Real output of
`build/bin/mlir-opt licm.mlir -loop-invariant-code-motion` (input: `%sum = addi`,
`%quot = divsi`, two stores in an `scf.for`):

```mlir
%0 = arith.addi %arg1, %arg2 : i32                //  ← hoisted out of the loop
scf.for %arg4 = %c0 to %arg0 step %c1 {
  %1 = arith.divsi %arg1, %arg2 : i32             //  ← NOT hoisted!
  memref.store %0, %arg3[%arg4] : memref<?xi32>
  memref.store %1, %arg3[%arg4] : memref<?xi32>
}
```

`divsi` is `NoMemoryEffect` but **not speculatable** (÷0 is UB): a *dead* `divsi`
is erased, but LICM refuses to hoist it — the loop may run **zero times**
precisely to guard the division. `Pure` = both halves.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: effects pay off beyond DCE/CSE — LICM is a two-line walk that hoists isPure ops out of loops, and divsi (NoMemoryEffect but not speculatable, ÷0 is UB) is erasable when dead yet never hoisted, because the loop may run zero times precisely to guard the division — Pure = both halves. (+2 min if presented.) The payoff slide for the whole effects section: because ops declare Pure/effects through interfaces, a real, dialect-agnostic optimization is a 2-line walk (works on ANY dialect's loops via LoopLikeOpInterface — Session 1's walk-with-interface-filter trick). Output shown is pre-captured (licm.mlir; command: build/bin/mlir-opt licm.mlir -loop-invariant-code-motion) — don't run live: addi hoisted above scf.for, divsi stays inside (real output, verified; and dead-divsi removal by -canonicalize also verified). This resolves the NoMemoryEffect-vs-Pure distinction concretely: deletion never executes anything (UB can't happen), hoisting EXECUTES the op in situations where the original program wouldn't have. divsi's ODS: ConditionallySpeculatable + NoMemoryEffect via the base class, NOT Pure (ArithOps.td:642).
-->

---

## The wider cleanup zoo ⏱

| Pass | What it does | Watch out / covered in |
|---|---|---|
| `-canonicalize` | folds + patterns + DCE + region simplify | best-effort |
| `-cse` | dominance-scoped dedup + DCE | effects gate it |
| `-trivial-dce` | just trivially-dead ops + unreachable blocks | no liveness |
| `-inline` | inlines calls (runs `canonicalize` on callees) | interfaces session |
| `-sccp` | constant propagation *through branches* | → dataflow session |
| `-loop-invariant-code-motion` | hoists `isPure` ops from loops | speculation! |
| `-control-flow-sink` | sinks ops *into* conditional regions | LICM's mirror |
| `-mem2reg`, `-sroa` | memory → SSA, split aggregates | → later session |
| `-symbol-dce` | removes unreferenced symbols | two slides ahead |
| `-remove-dead-values` | liveness-based, **changes function signatures** | ABI hazard |

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: there is a wider cleanup zoo — -sccp, -inline, LICM, -symbol-dce, -remove-dead-values and friends; the table stays in the deck and the pass names are on the cheat sheet, and the two to internalize today are -canonicalize and -cse. (+1 min if presented.) A map, not a lecture: name each, one sentence, and where the school covers it. SCCP is the "smarter constant folding" — it propagates constants through control flow using the dataflow framework (next module). control-flow-sink is the opposite of LICM: move an op INTO the only branch that uses it, so the other branch doesn't pay for it (never duplicates). mem2reg/SROA: interface-driven memory-to-SSA — a later session. remove-dead-values gets its own (flex) slide next.
-->

---

## `-remove-dead-values`: the ABI hazard ⏱

- **Liveness-based** DCE with teeth: reaches what trivial DCE can't —
  including **dead function arguments and results** — and does so *module-wide*.
- That means it **changes function signatures**: great for closed IR,
  **unsafe if external callers rely on the ABI**.
- Division of labor: region simplification drops dead *non-entry block*
  arguments; dropping *function* arguments is `-remove-dead-values`' job.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, the zoo table's row already says it: -remove-dead-values rewrites function signatures module-wide — an ABI hazard unless the IR is closed. (+1 min if presented.) It is interprocedural and rewrites signatures — the only pass in the zoo table that changes a function's ABI. Ties back to the region-simplification slide: entry-block/function arguments are the enclosing op's interface, which is exactly why the in-canonicalize DCE never touches them.
-->

---

## `-symbol-dce`: DCE for things that have no uses ⏱

Why doesn't `-canonicalize` remove an unused `func.func`?

- A function's "uses" are **`SymbolRefAttr`s** (e.g. `func.call @used_helper`) —
  attributes, **not SSA uses**. `use_empty()` is *always true* for a function!
- That's why `wouldOpBeTriviallyDead` hard-excludes `SymbolOpInterface` ops.

```mlir
// INPUT                                    // AFTER -symbol-dce (verified):
module {                                    // @unused_helper is gone;
  func.func private @unused_helper(...)     // @used_helper survives (referenced),
  func.func private @used_helper(...)       // @main survives (public =
  func.func @main(%a: i32) -> i32 {         //   visible externally = live root)
    %0 = call @used_helper(%a) : (i32) -> i32
    return %0 : i32
  }
}
```

`-symbol-dce` computes a live-symbol set from the *visibility roots* (public
symbols) plus everything they reference — then erases the rest.
(`-canonicalize` on the same input keeps all three — verified.)

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: functions have no SSA uses — references are SymbolRefAttrs — so normal DCE never touches them; -symbol-dce does mark-and-sweep from the public symbols, and nothing dies if everything is public. (+1 min if presented.) The non-SSA analog of DCE. Mark-and-sweep over symbol references: roots = symbols visible beyond this IR (public); edges = SymbolRefAttrs found in attributes. private + unreferenced => erased. Practical notes: must run on a SymbolTable op like builtin.module (scheduling it on func.func is a pass error), and if your functions are all public, nothing ever dies — visibility matters.
-->

---

## Where cleanup sits in real pipelines

```bash
# the classic sandwich — cleanup between lowering stages:
mlir-opt in.mlir \
  -pass-pipeline='builtin.module(
      func.func(my-lowering-pass, canonicalize, cse),
      convert-to-next-dialect,
      func.func(canonicalize, cse))'
```

The pragmatic idiom for *your* passes:

- **Emit naive IR.** Don't fold `x+0`, don't dedup constants, don't DCE inside
  your lowering pass — the next `canonicalize`/`cse` does it better.
- If naive IR bothers you at construction time: `builder.createOrFold<OpTy>(...)`
  folds ops as you build them.
- But remember the design rule: the pipeline must stay **correct** with every
  `canonicalize` removed — cleanup is a *performance* aid, never a *semantics* step.

<!-- Speaker notes:
~1 min (core path: 40/40). Timing check: ~40 min — end of the core path; wall clock should be ≈0:45, exercise briefing next. This is how upstream pipelines actually look (e.g. most conversion pipelines interleave canonicalize+cse between phases). The "emit naive IR" advice saves beginners hundreds of lines: your Session-1 pass deliberately left a dead constant behind, and now you know who cleans it up and why that's the intended division of labor. createOrFold is the middle ground when a pass would otherwise emit embarrassing amounts of garbage. Close with the correctness rule to tie back to the design-guidelines slide.
-->

---

## `-canonicalize` is a crowd, and you're joining it ⏱

Counted in this checkout (July 2026):

| What | Count |
|---|---|
| upstream passes (`def ... : Pass<` in .td) | **~256** |
| hand-written `OpRewritePattern` subclasses | **~750** (725 in `mlir/lib` alone) |
| `fold(FoldAdaptor)` implementations | **341** |
| ops with `let hasFolder = 1` | **339** (52 in arith alone) |
| ops with `let hasCanonicalizer = 1` | **183** |
| ops with `let hasCanonicalizeMethod = 1` | **19** |
| DRR patterns just for arith canonicalization | **82** |

`-canonicalize` ≈ **~1000 tiny rewrites**, each as small as the ones you wrote
yesterday. Nobody wrote "the canonicalizer" — everybody contributed three lines.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say while bringing up the exercise: -canonicalize is roughly a thousand tiny crowdsourced rewrites — ~340 folders, ~180 hasCanonicalizer ops — and Exercise 3 makes you one of the contributors. (+1 min if presented.) Motivation before the exercise: the free lunch is crowdsourced. Every folder on this table is a 5-15 line function like AddIOp::fold; every pattern like the ones from Session 2. When students upstream a dialect, reviewers will ask for folders and canonicalization patterns — now they know why, and how little it costs. Counts verified by grep over mlir/include + mlir/lib at this commit; treat as ballpark.
-->

---

## Exercise 3: make `school` a good citizen

You edit **the dialect itself** (ODS + C++) — no new pass, no driver call.
The starter `school.max` / `school.mac` are *deliberately* bad citizens.

**Checkpoint 1 — effects.** Run the provided test: a dead `school.max` survives
`-canonicalize`; `-cse` won't merge two identical ones.
Fix: add **`Pure`** to both ops' trait lists in `SchoolOps.td`. Re-run. 🎉

**Checkpoint 2 — folder.** `let hasFolder = 1;` and implement:

```cpp
OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  // max(x, x) -> x
  if (getLhs() == getRhs())
    return getLhs();
  return {};
}
```

Watch `school-opt -canonicalize` (the greedy driver you now know) pick it up.

<!-- Speaker notes:
~2 min briefing (0:45-0:50 briefing slot, together with the next slide + recap). The exercise itself is 28-30 min of work and fits the canonical 30-minute hands-on slot (0:50-1:20) — announce it as "30 minutes". Checkpoint 1 is deliberately visceral: they stare at an op that "obviously" should die and learn that MLIR won't touch what it can't reason about. One trait fixes DCE and CSE at once — that's the "implement once, every pass benefits" argument made personal. Checkpoint tests (from exercises/): ninja -C build check-school, or <llvm-build>/bin/llvm-lit -v build/test/exercise3 for just this exercise. Remind them: fold returns {} for "no fold", never nullptr-cast games.
-->

---

## Exercise 3 (continued): the materializer teaching beat

**Checkpoint 3 — constant folding.** Extend the folder:

```cpp
  // max(c1, c2) -> constant
  auto lhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (lhsCst && rhsCst)
    return lhsCst.getValue().sgt(rhsCst.getValue()) ? lhsCst : rhsCst;
```

Run the test… **nothing happens.** 🤔 No error, no change. *(Designed bug — the
classic custom-dialect gotcha: attribute fold results are silently dropped
unless the dialect can materialize constants!)* Fix in `SchoolDialect.td` + C++:

```cpp
// ODS:  let hasConstantMaterializer = 1;
Operation *SchoolDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
```

**Stretch:** (a) add `Commutative` → `max(c, x)` normalizes *for free*;
(b) `max(max(x, c1), c2) → max(x, c3)` via `let hasCanonicalizer = 1;`
(c) discuss: which `school.mac` simplifications are folds, which are patterns?

<!-- Speaker notes:
~2 min briefing (still inside the 0:45-0:50 slot). Checkpoint 3 is the centerpiece: let them hit the silent no-op for real before explaining. Mechanism (they saw the driver code): fold returns an Attribute -> driver calls op->getDialect()->materializeConstant(...) -> default implementation returns nullptr -> driver DROPS the fold result and moves on. No diagnostic. The fix delegates to arith::ConstantOp::materialize — perfectly legal for a dialect to materialize another dialect's constant op. Stretch (a) connects to the Commutative slide (then the folder only needs to check one side); (b) is a Session-2 pattern hooked into -canonicalize, and it RELIES on (a)'s normal form — which is exactly the point of canonical forms; (c) answers: mac(a,b,0) -> muli is a PATTERN (creates an op); mac(a,1,c) -> addi likewise; mac(a,0,c) -> c is a FOLD (returns existing value); mac(c1,c2,c3) -> constant is a FOLD (attribute). Solution in exercises/solutions/.
-->

---

## Recap

- **Folding**: the most restricted rewrite — no new ops, root-only, returns
  `{}` / existing `Value` / `Attribute` / own result (in-place). Runs *everywhere*:
  greedy driver, `createOrFold`, dialect conversion, `m_Constant`.
- **Constants**: fold returns the *value*; `materializeConstant` builds the op.
  No materializer ⇒ attribute folds **silently dropped**. Dedup + hoisting is automatic.
- **Canonicalization**: patterns attached to ops (`hasCanonicalizer` /
  `hasCanonicalizeMethod`); must converge, must be cheap, canonical ≠ optimal;
  `-canonicalize` = collect all hooks + greedy driver.
- **DCE**: dead = no users **and** no observable effects. Effects are declared
  (`Pure`, `MemoryEffects`, `Arg<..., [MemRead]>`); **unknown = effectful**.
- **CSE**: separate pass; structural equivalence + dominance scoping;
  effect-free ops only (reads: same block, no write in between).
- **The free lunch**: annotate your ops once — every pipeline optimizes your
  dialect forever.

<!-- Speaker notes:
~1 min, closing the 0:45-0:50 briefing slot right before hands-on starts. Point back at the opening quiz: every line of both outputs should now be explainable by someone in the room — quiz them once more on the twins (-cse, not -canonicalize) and x+0 (-canonicalize, not -cse) as a closing check.
-->

---

## Cheat sheet: today's API surface

```tablegen
// ODS — op:                              // ODS — dialect:
let hasFolder = 1;                        let hasConstantMaterializer = 1;
let hasCanonicalizer = 1;
let hasCanonicalizeMethod = 1;
def MyOp : My_Op<"...", [Pure, Commutative]>          // traits
Arg<AnyMemRef, "desc", [MemRead]>:$ref                // per-operand effect
```

```cpp
OpFoldResult MyOp::fold(FoldAdaptor a);           // {} | Value | Attribute | getResult()
LogicalResult MyOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &);
static void MyOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *);
static LogicalResult MyOp::canonicalize(MyOp, PatternRewriter &);
Operation *MyDialect::materializeConstant(OpBuilder &, Attribute, Type, Location);
matchPattern(a.getRhs(), m_Zero());  dyn_cast_if_present<IntegerAttr>(...);
constFoldBinaryOp<IntegerAttr>(a.getOperands(), [](APInt x, const APInt &y) {...});
isOpTriviallyDead(op); isMemoryEffectFree(op); isPure(op);
builder.createOrFold<OpTy>(loc, ...);   // build + fold eagerly
```

```bash
-canonicalize="region-simplify=aggressive"   -cse -mlir-pass-statistics
-trivial-dce   -symbol-dce   -loop-invariant-code-motion   -sccp   -inline
```

(One legacy note: old code writes `builder.create<OpTy>(loc, ...)` — deprecated;
today it's `OpTy::create(builder, loc, ...)`.)

<!-- Speaker notes:
Handout slide, no talking needed — leave it up during the exercise. The legacy note keeps students from being confused by older blog posts and LLM output trained on them.
-->

---

## Further reading

- **`mlir/docs/Canonicalization.md`** — *the* document for this session: the fold
  contract, ODS hooks, globally applied rules, design guidelines. Short; read it all.
- **`mlir/docs/Rationale/SideEffectsAndSpeculation.md`** — why effects and
  speculatability are separate axes; how to model your ops' effects honestly.
- **`mlir/docs/PatternRewriter.md`** — (from Session 2) drivers, debugging flags.
- **`mlir/docs/DeclarativeRewrites.md`** — DRR, if the TableGen patterns intrigued you.
- Code worth reading (all short):
  - `mlir/lib/Transforms/Canonicalizer.cpp` (105 lines)
  - `mlir/lib/Transforms/CSE.cpp` (57) + `mlir/lib/Transforms/Utils/CSE.cpp`
  - `mlir/lib/Interfaces/SideEffectInterfaces.cpp` (`isOpTriviallyDead` & co.)
  - `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` (grep `::fold` — 50+ real folders)

<!-- Speaker notes:
Handout slide — no core-path time; flash it during the briefing. Point out that unlike many projects, these MLIR docs are current — the fold contract and the pass sources on today's slides were quoted verbatim from this checkout.
-->

---

<!-- _class: lead -->

## The module in one slide

**Session 1:** manual surgery — `walk`, `create`, RAUW, `erase`, a registered pass.
**Session 2:** the rewrite becomes a *pattern*; drivers do the orchestration.
**Session 3:** the pattern becomes an *op hook*; upstream passes do everything.

Each session deleted machinery from the previous one.
**You now know how ~80% of upstream MLIR transformation code is structured.**

Coming up in the school:
**dataflow analysis** (how `-sccp` really works) · **mem2reg & SROA** ·
**interfaces** (how effects & inlining really work) · **transform dialect** ·
**GPU**

Thanks — now go make your dialect a good citizen. 🎓

<!-- Speaker notes:
~3 min, the tail of the 1:20-1:30 solution-walkthrough + wrap-up slot. The arc: strength-reduce was a hand-written walk (S1), then MulByPow2ToShl pattern (S2), and today x*1 -> x turned out to be a folder upstream. Tease the next modules by tying to today's loose ends: SCCP = the dataflow session; "how does the inliner know what it may inline" and "how do effects interfaces work under the hood" = interfaces session; transform dialect = orchestrating today's patterns from IR itself. If time remains, live-code the Exercise 3 solution diff (Pure + fold + materializer is ~15 lines total) as the closing demonstration of the free lunch.
-->
