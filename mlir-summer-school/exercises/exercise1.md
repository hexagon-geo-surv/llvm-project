# Exercise 1 — `-school-strength-reduce`: IR surgery by hand

**Session 1: Your First Pass.** You will change IR with the raw C++ API:
walk, match, build, replace, erase. No pattern framework — that comes next
session, and it will feel *great* after this.

## Goal

Rewrite every `arith.muli %x, %c` where `%c` is a **constant power of two**
into `arith.shli %x, %log2(c)`, inside the provided pass stub
`lib/School/StrengthReduce.cpp`.

```mlir
// before                          // after
%c8 = arith.constant 8 : i32      %c8 = arith.constant 8 : i32   // now dead -- fine!
%r  = arith.muli %x, %c8 : i32    %c3 = arith.constant 3 : i32
                                  %r  = arith.shli %x, %c3 : i32
```

## Background

A pass = a TableGen entry (`include/School/SchoolPasses.td`, already written)
plus a `runOnOperation()` (yours to fill in). This pass is anchored on
`func.func`, so the pass manager calls your `runOnOperation()` once per
function — potentially in parallel, which is why you must only touch IR
inside `getOperation()`.

## Build & check

```bash
ninja -C build   # rebuild after every edit (seconds)
<llvm-build>/bin/llvm-lit -v build/test/exercise1     # FileCheck test (covers checkpoints 2-3)
build/bin/school-opt test/exercise1/strength-reduce.mlir \
  -pass-pipeline="builtin.module(func.func(school-strength-reduce))"   # by hand
```

## Checkpoint 1 — find the candidates (print only)

Walk the function and **print** (to `llvm::errs()`) every `arith.muli` whose
rhs is a constant power of two. This checkpoint has no lit test — you verify
it by eye: run the pass by hand on the test input and you
should see the muls from `@mul_by_8` and `@two_rewrites`, but not from
`@mul_by_7` or `@mul_by_dynamic`. (Add `--mlir-disable-threading` if the
printed lines interleave — the functions are processed in parallel.)

<details><summary>Hint 1 (which APIs)</summary>

`getOperation()->walk(...)` with a *typed* callback filters for you.
`matchPattern(value, m_ConstantInt(&apint))` (from `mlir/IR/Matchers.h`)
answers "is this value a constant integer?" and binds it; then
`apint.isPowerOf2()`.
</details>

<details><summary>Hint 2 (skeleton)</summary>

```cpp
getOperation()->walk([&](arith::MulIOp op) {
  APInt rhsValue;
  if (/* rhs is a constant */ && /* power of two */)
    llvm::errs() << op << "\n";
});
```
</details>

## Checkpoint 2 — the rewrite

Replace each candidate: build an `arith.constant` holding `log2(c)` and an
`arith.shli`, redirect all uses of the muli to the shli, erase the muli.
`llvm-lit -v build/test/exercise1` should now get close to green (checkpoints
2 and 3 share the one test file, so it may stay red until the checkpoint 3
cases stop biting — read
FileCheck's output to see how far you got).

<details><summary>Hint 1 (which APIs)</summary>

`OpBuilder b(op);` puts the insertion point right **before** `op`.
Create ops with `OpTy::create(b, loc, ...)` — e.g.
`arith::ConstantOp::create(b, op.getLoc(), b.getIntegerAttr(op.getType(), n))`
and `arith::ShLIOp::create(b, op.getLoc(), lhs, amount)`.
Then `op->replaceAllUsesWith(ValueRange{newValue});` and `op->erase();`.
Reuse `op.getLoc()` for the new ops — never invent an `UnknownLoc`.
</details>

<details><summary>Hint 2 (skeleton)</summary>

```cpp
OpBuilder b(op);
Value amount = arith::ConstantOp::create(
    b, op.getLoc(), b.getIntegerAttr(op.getType(), rhsValue.logBase2()));
Value shifted = arith::ShLIOp::create(b, op.getLoc(), op.getLhs(), amount);
op->replaceAllUsesWith(ValueRange{shifted});  // 1. reroute uses
op->erase();                                  // 2. THEN erase
```
</details>

## Checkpoint 3 — robustness

All four test functions green: no crash on non-constant rhs, no rewrite for
non-powers-of-two, and **both** muls in `@two_rewrites` rewritten. If your
pass crashes or misses the second mul, you are probably erasing ops during
the walk.

<details><summary>Hint (the safe idiom)</summary>

Collect-then-mutate: walk once, push candidates into a
`SmallVector<arith::MulIOp>`, then loop over the vector and rewrite. (Erasing
the *currently visited* op in a post-order walk is also legal, but the
vector version is the habit that never bites.)
</details>

## Stretch goals

1. **Statistics.** Count rewrites in the `numRewrites` statistic (it is
   already declared in `SchoolPasses.td`; the generated base class gives you
   a member `numRewrites` — just `++numRewrites;`). See it with
   `school-opt ... --mlir-pass-statistics`.
2. **Constant on the left.** Handle `arith.muli %c8, %x` too. Then note how
   much duplicate code that costs — Session 3 shows why canonicalization
   makes this unnecessary (constants drift to the right on their own).
3. **`muli %x, 1`.** `1 = 2^0` — your pass currently emits `shli %x, 0`.
   Special-case it: no new ops at all, just `replaceAllUsesWith(op.getLhs())`.

## Common pitfalls

- **Erase before RAUW** → `operation destroyed but still has uses` fatal
  error. Always replace uses first.
- **Erasing other ops mid-walk** → iterator invalidation, crashes or skipped
  ops. Collect-then-mutate.
- **`builder.create<OpTy>(...)`** — you will see this in old blog posts; it
  is deprecated. Current API: `OpTy::create(builder, loc, ...)`.
- **`getDefiningOp()` returns null** for block arguments (including function
  arguments!). `matchPattern` handles that for you; manual
  `dyn_cast<arith::ConstantOp>(v.getDefiningOp())` does not.
- The dead `%c8` after the rewrite is **expected**. Deleting it by hand is
  possible (`if (cst.use_empty()) cst.erase();`) but unnecessary — standard
  cleanup passes exist, as Session 3 shows.
