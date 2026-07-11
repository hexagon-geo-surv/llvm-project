# MLIR C++ API Cheat Sheet

Quick reference for the core IR classes: `Operation`, `Block`, `Region`, `Value`,
`Attribute`/`Type`, `OpBuilder`, and `RewriterBase`/`PatternRewriter`.

**The object model, in one line:** `Operation` → owns `Region`s → own `Block`s →
own `Operation`s (recursion). `Value` = `OpResult` (owner: defining `Operation*`) |
`BlockArgument` (owner: `Block*`).

**Handle discipline:** `Operation`/`Block`/`Region` are heap objects you always
handle as `Operation*`/`Block*`/`Region*` — pointer only, never by value.
`Value`/`Attribute`/`Type` are the opposite: thin, one-pointer-wide wrappers
around interned/uniqued storage — always **pass/store/return them by value**
(`Value v`, not `Value *v` or `Value &v`), copy them freely, compare with `==`.
Writing `Value*` or copying an `Operation` are both the classic beginner bugs.

## Casting (works on `Operation*`, `Value`, `Attribute`, `Type`, interfaces)

| Call | Behavior |
|---|---|
| `isa<T>(x)` | is it a `T`? |
| `dyn_cast<T>(x)` | `T`, or null on mismatch — the workhorse |
| `cast<T>(x)` | `T`, asserts on mismatch — only when you *know* |
| `dyn_cast_or_null<T>(x)` / `isa_and_nonnull<T>(x)` | like above, also tolerates null `x` |

## Querying & navigation

| I have… | I want… | Call |
|---|---|---|
| `Operation *op` | its parent op / typed ancestor | `op->getParentOp()`, `op->getParentOfType<FuncOp>()` |
| `Operation *op` | its block / region | `op->getBlock()`, `op->getParentRegion()` |
| `Operation *op` | nested content | `op->getRegions()` → `region.getBlocks()` → `block.getOperations()` |
| `Operation *op` | operands / results | `op->getOperand(i)`/`getOperands()`, `op->getResult(i)`/`getResults()`, `getNumOperands/Results()` |
| `Operation *op` | operand / result types | `op->getOperandTypes()`, `op->getResultTypes()` |
| `Operation *op` | attributes | `op->getAttrs()`, `op->getAttrOfType<IntegerAttr>("x")`, `op->hasAttr("x")` |
| `Operation *op` | name / context / location | `op->getName()`, `op->getContext()`, `op->getLoc()` |
| `Operation *op` | is it typed op `T`? | `isa<T>(op)`, `dyn_cast<T>(op)` (see above) |
| `Block &b` | its parent op / region | `b.getParentOp()`, `b.getParent()` |
| `Block &b` | ops, filtered by type | `b.getOps<arith::ConstantOp>()`, `b.without_terminator()` |
| `Block &b` | first/last op, terminator | `b.front()`, `b.back()`, `b.getTerminator()` |
| `Block &b` | its arguments | `b.getArguments()`, `b.getArgument(i)`, `b.getNumArguments()` |
| `Block &b` | predecessors / successors | `b.getPredecessors()`, `b.getSuccessors()`, `b.getSinglePredecessor()` |
| `Region &r` | its blocks / entry block | `r.getBlocks()`, `r.front()`, `r.hasOneBlock()` |
| `Region &r` | ops one level in, filtered | `r.getOps<T>()` |
| `Region &r` | its parent op / typed ancestor | `r.getParentOp()`, `r.getParentOfType<FuncOp>()` |
| `Operation *op` | is it inside `other`? / which comes first? | `other->isAncestor(op)`, `op->isBeforeInBlock(other)` *(same block only)* |
| `Operation *op`, `Block *block` | the op/ancestor lying in `block` | `block->findAncestorOpInBlock(*op)` (also `Region::findAncestorOpInRegion`) |
| `Value v` | who defines it | `v.getDefiningOp()` *(null for block args!)*, `v.getDefiningOp<T>()` (typed + null-safe) |
| `Value v` | its type | `v.getType()` |
| `Value v` | who uses it | `v.getUsers()` (ops), `v.getUses()` (`OpOperand&` edges) |
| `Value v` | cheap use queries | `v.use_empty()`, `v.hasOneUse()`, `v.hasNUses(n)` *(`getNumUses()` is linear — avoid)* |
| `Value v` | is it a block arg / op result? | `isa<BlockArgument>(v)` / `isa<OpResult>(v)`, then `cast<...>(v).getArgNumber()` / `.getOwner()` |
| `Attribute a` | the payload | `cast<IntegerAttr>(a).getInt()`, `cast<StringAttr>(a).getValue()`, `cast<ArrayAttr>(a).getValue()` |
| any op | is it a constant? (any dialect) | `matchPattern(v, m_ConstantInt(&apInt))`, `m_Constant()`, `m_Zero()`, `m_One()` |
| anything | debug print | `op->dump()` / `block.dump()` / `v.dump()`, or `llvm::errs() << x << "\n"` |

## Modification

Three tiers, each layered on top of the last:

- **Direct (`Operation`/`Value`)** — the raw model. Mutates immediately, in
  place, notifies no one. Fine in a plain pass; forbidden inside a pattern.
- **`OpBuilder`** — adds an insertion *cursor*: things it creates or clones are
  placed at that cursor. Still notifies no one.
- **`RewriterBase`** — `OpBuilder` + driver notifications (`Listener`
  callbacks for insert/erase/replace/modify). This is what you're actually
  handed in a pattern: a `PatternRewriter` in `OpRewritePattern`, a
  `ConversionPatternRewriter` in `OpConversionPattern` — both are-a
  `RewriterBase`. **Inside a pattern, always go through the `rewriter`, never
  the direct calls** — the driver's worklist / rollback state depends on it.

| Intent | Direct (`Operation`/`Value`) | `OpBuilder` | `RewriterBase` |
|---|---|---|---|
| Position the insertion cursor | — | `b.setInsertionPoint(op)` *(before op!)*, `setInsertionPointAfter(op)`, `setInsertionPointToStart/End(&block)`, `InsertionGuard g(b);` | same (inherited) |
| Create an op, detached (no insertion) | `OperationState state(loc, name);`<br>`state.addOperands(...); state.addTypes(...); state.addAttribute(...);`<br>`Operation *op = Operation::create(state);` — fully detached, no builder involved | — *(builder always inserts — see below)* | — |
| Create an op (build + insert at cursor) | — | `OpTy::create(b, loc, args...)` | `OpTy::create(rewriter, loc, args...)` (same) |
| Create + replace in one call | — | — | `rewriter.replaceOpWithNewOp<OpTy>(op, args...)` |
| Clone | `op->clone()` / `clone(mapper)` — **detached, inserted nowhere** | `b.clone(*op)` / `clone(*op, mapper)` — clones **and inserts** at cursor | same (inherited) |
| Rewire uses (value-level) | `oldVal.replaceAllUsesWith(newVal)` — silent | — | `rewriter.replaceAllUsesWith(oldVal, newVal)` — notified |
| Replace an op's results | `op->replaceAllUsesWith(newOp->getResults())`, then erase yourself | — | `rewriter.replaceOp(op, newValues)` (RAUW+erase, notified) |
| Erase | `op->erase()` *(only if `use_empty()`)*, silent | — | `rewriter.eraseOp(op)`, `rewriter.eraseBlock(b)` — notified |
| Edit in place (attrs, operands) | `op->setAttr(name, attr)`, `op->setOperand(i, v)` — silent | — | same calls, wrapped: `rewriter.modifyOpInPlace(op, [&]{ ... })` |
| Move an op | `op->moveBefore(other)` / `moveAfter(other)` — silent | — | `rewriter.moveOpBefore(op, other)` / `moveOpAfter(...)` — notified |
| Create a block | manual: `new Block()` + splice in | `b.createBlock(region, ...)` — creates, inserts, moves cursor in | same (inherited) |
| Inline one block's ops into another, dropping the source block | manual: `dest->getOperations().splice(before, source->getOperations())` + RAUW each block arg + `source->erase()` — fiddly, easy to get wrong | — | `rewriter.inlineBlockBefore(source, dest, before, argValues)`, `rewriter.mergeBlocks(source, dest, argValues)` — does the arg remap + erase for you, notified |
| Move a region's blocks into another region | manual: `parent.getBlocks().splice(before, region.getBlocks())` | `b.cloneRegionBefore(region, ...)` — **copies**, doesn't move | `rewriter.inlineRegionBefore(region, parent, before)` — moves, notified |
| Report a failed match | — | — | `return rewriter.notifyMatchFailure(op, "why");` |
| Factory helpers (types/attrs/locs) | — *(need a `Builder`)* | `b.getI32Type()`, `getIntegerAttr(ty, v)`, `getI32IntegerAttr(v)`, `getStringAttr(s)`, `getUnknownLoc()` | same (inherited from `Builder`) |

**Iron rules:**
- **RAUW before `erase()`.** Erasing an op with live uses is a fatal error (assert
  builds) / memory corruption (release builds).
- **`OpBuilder(op)` / `OpBuilder b(op)` inserts *before* `op`**, not after — perfect
  for building the replacement of `op`.
- **In a pattern, mutate iff you return `success()`.** Do all checks before the
  first `rewriter` call; never mutate on a path that returns `failure()`.
- **In dialect conversion**, take operand *values* from the `adaptor`, everything
  else (attributes, location, result types via `getTypeConverter()`) from `op`.
