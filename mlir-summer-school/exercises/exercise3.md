# Exercise 3 — Make the `school` dialect a good citizen

**Session 3: The Free Lunch.** No pass code this time. You edit the *dialect
itself* (ODS + a few lines of C++) so that the standard passes —
`-canonicalize`, `-cse` — start optimizing `school` ops. Implement the hooks
once; every pipeline benefits.

You edit: `include/School/SchoolOps.td`, `include/School/SchoolDialect.td`,
`lib/School/SchoolOps.cpp`, `lib/School/SchoolDialect.cpp`. The `TODO`
markers show the exact spots.

## Build & check

```bash
ninja -C build     # .td edits re-run TableGen -- still seconds
<llvm-build>/bin/llvm-lit -v build/test/exercise3
```

The stretch tests (`commutative.mlir`, `reassociate.mlir`) stay red until
the stretch goals; that is fine.

## Checkpoint 1 — DCE and CSE are gated on effects (`dce-cse.mlir`)

First, *observe the problem*:

```bash
build/bin/school-opt test/exercise3/dce-cse.mlir -canonicalize
build/bin/school-opt test/exercise3/dce-cse.mlir -cse
```

The dead `school.max` survives `-canonicalize`, and `-cse` refuses to merge
two identical maxes. Why: unannotated ops are **conservatively assumed to
have side effects** — deleting or merging them would be unsound. That safety
default is a feature; your job is to declare that `school.max`/`school.mac`
are harmless.

Fix: one word per op in `SchoolOps.td` (the `TODO(exercise 3.1)` markers).
Re-run both commands — no C++ changed, and both transformations now happen.

<details><summary>Hint</summary>

The trait is `Pure` (= no memory effects + always speculatable). It goes in
the trait list: `School_Op<"max", [Pure]>`.
</details>

## Checkpoint 2 — a folder (`fold-idempotent.mlir`)

`max(x, x)` is just `x`. That rewrite needs no new ops, so it is a **fold**,
the most restricted (and most-run) kind of rewrite: return an existing
value, an attribute, or nothing. Declare `let hasFolder = 1;` on
`School_MaxOp` (TODO 3.2) and implement in `SchoolOps.cpp`:

<details><summary>Hint 1 (signature)</summary>

For a single-result op ODS declares
`OpFoldResult MaxOp::fold(FoldAdaptor adaptor);`
Return `getLhs()` when both operands are the same `Value`; return `{}` for
"no fold". `-canonicalize` (the greedy driver) calls it — no school pass
involved.
</details>

<details><summary>Hint 2 (skeleton)</summary>

```cpp
OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return getLhs();
  return {};
}
```
</details>

## Checkpoint 3 — constant folding needs a materializer (`fold-constants.mlir`)

Extend the folder: when **both** operands are constants, return the larger
one. The adaptor gives you the constant values: `adaptor.getLhs()` is an
`Attribute` — the constant if the operand is defined by a constant op,
**null otherwise**.

<details><summary>Hint (skeleton)</summary>

```cpp
auto lhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
auto rhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
if (lhsCst && rhsCst)
  return lhsCst.getValue().sgt(rhsCst.getValue()) ? lhsCst : rhsCst;
```
(`sgt` because school.max is a *signed* max — `fold-constants.mlir` checks a
negative-number case.)
</details>

Rebuild, run the test... **and nothing happens.** The fold is correct, yet
`school.max %c3, %c5` survives `-canonicalize`. This is the classic
custom-dialect bug: your fold returns an *Attribute*, and somebody must turn
that attribute back into a constant *op*. That somebody is the dialect's
**constant materializer** — and if the dialect doesn't have one, attribute
fold results are **silently dropped**.

Fix (TODO 3.3): in `SchoolDialect.td`, declare the hook; in
`SchoolDialect.cpp`, define it.

<details><summary>Hint 1 (which hooks)</summary>

`let hasConstantMaterializer = 1;` on the dialect declares
`Operation *materializeConstant(OpBuilder &, Attribute, Type, Location);`.
The school dialect has no constant op — borrow `arith.constant`.
</details>

<details><summary>Hint 2 (the whole fix)</summary>

```cpp
Operation *SchoolDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
```
plus `#include "mlir/Dialect/Arith/IR/Arith.h"`. Since the dialect now
creates arith ops, also add
`let dependentDialects = ["::mlir::arith::ArithDialect"];` in the .td.
</details>

## Stretch goals

**(a) `Commutative` (`commutative.mlir`, TODO 3.4).** Add the trait to
`school.max` (not `mac` — why not?) and run the test: `max(%c5, %x)` becomes
`max(%x, %c5)` with zero code written. Trait folding moves constants to the
right — this is why upstream folders only ever check the rhs for constants.

**(b) Reassociation (`reassociate.mlir`, TODO 3.5).**
`max(max(x, c1), c2) → max(x, max(c1,c2))` creates a new constant op, so it
*cannot* be a fold — it is a real `RewritePattern` (Session 2 skill),
attached to the op via `let hasCanonicalizer = 1;` +
`MaxOp::getCanonicalizationPatterns` so that `-canonicalize` picks it up.
Thanks to (a) you only need to handle constants on the rhs — that is
exactly what canonical forms are for.

<details><summary>Hint (registration)</summary>

```cpp
void MaxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ReassociateConstantMax>(context);
}
```
with `ReassociateConstantMax` an `OpRewritePattern<MaxOp>`: match rhs
constant + lhs defined by another `MaxOp` with rhs constant; build the
merged constant;
`rewriter.replaceOpWithNewOp<MaxOp>(op, op.getType(), innerLhs, merged)`
(there is no builder that infers the result type, so pass it explicitly).
Make sure each application strictly shrinks the chain — cyclic
canonicalizations are a bug (the greedy driver will not converge).
</details>

**(c) Discussion (no code).** Which of these `school.mac` simplifications
could be folds, and which must be patterns?
`mac(a, b, 0)`; `mac(a, 1, c)`; `mac(a, 0, c)`; `mac(c1, c2, c3)`.
(Folds may return an existing value or an attribute, but may not create
ops. `mac(a, 1, c) = a + c` needs a new `arith.addi`...)

## Common pitfalls

- **`cast<IntegerAttr>(adaptor.getLhs())` crashes** when the operand is not
  constant — adaptor entries are null then. Use `dyn_cast_if_present`.
- **Folds must not create ops** — no builders inside `fold`. Returning an
  attribute is the sanctioned way to "create" a constant; a pattern is the
  tool when you truly need new ops.
- **Silently dropped attribute folds**: no materializer, no constant. If a
  fold "doesn't fire", check the dialect's `hasConstantMaterializer` first.
- **Non-converging canonicalizations**: returning `getResult()` from a fold
  without changing anything, or patterns that undo each other, make
  `-canonicalize` spin. `-canonicalize="test-convergence=true"` exists to
  catch this in tests.
- After `.td` edits, stale-looking errors usually mean you forgot to
  rebuild: `ninja -C build` re-runs TableGen.
