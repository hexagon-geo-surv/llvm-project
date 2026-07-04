# Exercise 2 — Rewrite patterns, then a real lowering

**Session 2: Rewrite Patterns & Dialect Conversion.** Part A re-expresses
Exercise 1 as patterns and lets the greedy driver do the orchestration.
Part B is a real lowering with the dialect conversion framework.

## Build & check

```bash
ninja -C build
<llvm-build>/bin/llvm-lit -v build/test/exercise2
```

---

## Part A — `-school-peephole` (patterns + greedy driver)

Edit `lib/School/Peephole.cpp`. The pass skeleton (pattern set + driver) is
already there; you implement two `matchAndRewrite` bodies:

1. **`MulByPow2ToShl`**: `muli(x, C)` → `shli(x, log2 C)` — your Exercise 1
   rewrite, now ~10 lines instead of a whole pass.
2. **`MergeConsecutiveShl`**: `shli(shli(x, C1), C2)` → `shli(x, C1+C2)`.
   *If time / stretch-track:* the overflow guard — do **not** merge when
   `C1+C2 >=` the bit width, because shifting past the width is poison
   (poison = a garbage value the compiler may assume never happens —
   producing it where the original program didn't is a miscompile; hence the
   guard).

The test input contains `((x*4)*8)`. Neither pattern alone reduces that to
one shift; run to a fixpoint they do — this is the greedy driver earning its
keep. Also notice: the dead constants disappear. You wrote no cleanup code;
the driver folds and DCEs as it goes.

### Checkpoints

Core for the 30-minute hands-on slot: checkpoint 1, checkpoint 2's happy
path, and checkpoint 3. The overflow guard is the *if time / stretch* track.

1. `@mul_by_16` green — pattern A fires.
2. `@shl_shl` green — pattern B fires. *(If time / stretch-track:
   `@no_merge_overflow` green — your guard works. It may stay red until
   then; that is fine.)*
3. `@mul_chain` green — composition to a single `shli` by 5.

<details><summary>Hint 1 (which APIs)</summary>

The match part is Exercise 1 (`matchPattern` + `m_ConstantInt`). For pattern
B, "is my operand produced by a shli?" is
`op.getLhs().getDefiningOp<arith::ShLIOp>()` (null-safe).
For the rewrite: create the constant with
`arith::ConstantOp::create(rewriter, loc, ...)` — the rewriter *is* a
builder — then `rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, x, amount)`
creates the shli **and** replaces + erases the muli in one call.
For clean shift-amount arithmetic: `APInt::uadd_ov` + `APInt::uge`.
</details>

<details><summary>Hint 2 (skeleton for pattern A)</summary>

```cpp
APInt rhsValue;
if (!matchPattern(op.getRhs(), m_ConstantInt(&rhsValue)))
  return rewriter.notifyMatchFailure(op, "rhs not constant");
if (!rhsValue.isPowerOf2())
  return rewriter.notifyMatchFailure(op, "not a power of two");
Value amount = arith::ConstantOp::create(
    rewriter, op.getLoc(),
    rewriter.getIntegerAttr(op.getType(), rhsValue.logBase2()));
rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(), amount);
return success();
```
</details>

**Debugging patterns:** run with
`--debug-only=greedy-rewriter --mlir-disable-threading` to watch the driver
process ops and your patterns fire/fail (your `notifyMatchFailure` messages
appear there — nowhere else; without `--mlir-disable-threading` the output
from parallel function pipelines interleaves).

### The two contracts (the graders of this exercise)

- **failure ⇒ untouched IR.** Do *all* match checks before the first
  mutation. Mutating and then returning `failure()` corrupts the driver
  state (only expensive-checks builds catch it for you).
- **All mutation through the rewriter.** `op->erase()` inside a pattern is
  a bug even when it "works": the driver tracks changes via the rewriter to
  maintain its worklist.

---

## Part B — `-convert-school-to-arith` (dialect conversion)

Edit `lib/School/ConvertSchoolToArith.cpp`. Lowering is a different job:
*everything* from the school dialect must go, and failure must be loud.
That is what `ConversionTarget` + `applyPartialConversion` give you.

Core for the 30-minute slot: checkpoint 1 (the max conversion) and
checkpoint 2 (reading the failed-to-legalize error). Checkpoint 3 and the
stretch goals are the *if time / stretch* track (`convert-mac.mlir` may stay
red until then).

### Checkpoint 1 — target + max pattern

Mark `SchoolDialect` illegal and `arith::ArithDialect` legal, and implement
`MaxOpLowering`: `school.max` → `arith.cmpi sgt` + `arith.select`.
`convert-max.mlir` turns green.

<details><summary>Hint 1 (which APIs)</summary>

`target.addIllegalDialect<SchoolDialect>();`
`target.addLegalDialect<arith::ArithDialect>();`
In the pattern: `arith::CmpIOp::create(rewriter, loc,
arith::CmpIPredicate::sgt, a, b)` then
`rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, a, b)`.
Take `a`/`b` from **`adaptor`**, not from `op` — the adaptor carries the
already-converted operands (with converted types, once type conversion is in
play).
</details>

### Checkpoint 2 — read a legalization error

*Before* writing the mac pattern, run:

```bash
build/bin/school-opt test/exercise2/convert-mac.mlir \
  -pass-pipeline="builtin.module(convert-school-to-arith)"
```

You should see:

```
error: failed to legalize operation 'school.mac' that was explicitly marked illegal
```

Read it carefully — this is the everyday error of lowering work. "Explicitly
marked illegal" (your `addIllegalDialect`) + no pattern = hard failure. Ops
the target says nothing about (`func.func`, ...) survive a *partial*
conversion silently; that is the partial/full distinction.

### Checkpoint 3 — mac pattern *(if time / stretch-track)*

Add `MacOpLowering` (`school.mac %a,%b,%c` → `arith.muli` + `arith.addi`),
register it, and `convert-mac.mlir` turns green.

### Stretch goals

1. Swap in `applyFullConversion` and re-run `convert-max.mlir`. It now fails
   with `failed to legalize operation 'builtin.module'`! Why? (In a *full*
   conversion, "unknown" is not good enough — every op must be legal,
   including `builtin.module` and `func.func`, which nothing in our target
   legalizes.) Swap back afterwards.
2. Watch a pattern *refuse* to match. (`school.max` is ODS-constrained to
   `i32`, so a "bail out on non-`i32`" guard could never fire on
   verifier-clean IR.) Instead, temporarily **invert** the guard — pretend
   `i32` is unsupported — at the top of `MaxOpLowering::matchAndRewrite`:

   ```cpp
   if (op.getType().isInteger(32))
     return rewriter.notifyMatchFailure(op, "i32 not supported (experiment)");
   ```

   Re-run `convert-max.mlir` under `--debug-only=dialect-conversion`: the
   trace shows your message as the pattern refuses, and the driver then
   reports the familiar `failed to legalize` error — the only pattern for an
   illegal op declined to match. Revert the guard afterwards.
3. Run with `--debug-only=dialect-conversion` and find the legalization tree:
   target check → fold attempt → your pattern → recursive legalization of
   the ops your pattern created.

## Common pitfalls

- **`op.getLhs()` vs `adaptor.getLhs()`** — the #1 conversion bug. The op
  holds *original* operands; the adaptor holds *converted* ones. Always use
  the adaptor in conversion patterns.
- Patterns need not produce *directly* legal ops — the framework recursively
  legalizes what you create. But here `arith` is legal, so it is one step.
- An illegal op is **not auto-deleted**: no pattern, no conversion — just an
  error. The framework never invents rewrites.
- `applyPatternsGreedily` returning `failure()` means "did not converge",
  not "nothing matched". Zero matches is a successful fixpoint.
- Don't walk around and inspect neighboring IR from inside a conversion
  pattern: mid-conversion, the IR is a mix of old and new state.
