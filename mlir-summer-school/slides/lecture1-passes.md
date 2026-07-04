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

# Your First Pass
## IR surgery with the C++ API

**MLIR Summer School — Transformations (1/3)**

<!-- Speaker notes:
Welcome to the Transformations module. Session budget (canonical): 0:00-0:05 warm-up quiz | 0:05-0:45 lecture incl. embedded quizzes+demos (~40 min core path) | 0:45-0:50 exercise briefing | 0:50-1:20 hands-on (30 min) | 1:20-1:30 solution walkthrough + wrap-up.
Core path ≈40 min: slides marked ⏱ are the pressure-release valves — skip them if behind schedule, in this order: LICM walk+library-call (+2), pass options & statistics (+2), InsertionGuard (+1), debugging toolkit (2) (+2), pipeline quiz + answers as a pair (+3), why-anchors-matter threading (+2), textual pipeline syntax (+2), PassManager tree (+2), failing & staying honest (+2), WalkResult (+1), traversal level 0 (+1), constants/Location (+1), nesting gotcha (+1), navigation map (+1).
Prerequisites the audience already has: compiler basics (SSA, "what is a pass"), MLIR IR structure (ops/regions/blocks/values), ODS/TableGen. They have NOT written C++ against the MLIR API yet — today is that day.
Setup check (do it BEFORE 0:00): everyone should have the prebuilt LLVM/MLIR tree and the exercises/ project configured. All demos in this deck run with build/bin/mlir-opt from the llvm-project checkout root.
~30 s.
-->

---

# The arc of this module

| # | Session | One-sentence goal |
|---|---------|-------------------|
| **1** | **Your First Pass** | Change IR **by hand**: navigate, walk, create, replace, erase — inside a real pass. |
| 2 | Rewrite Patterns & Dialect Conversion | Express rewrites as **patterns**, let **drivers** orchestrate them. |
| 3 | The Free Lunch | Canonicalization, folding, CSE, DCE — plug *your* dialect into the standard cleanup. |

**The running theme:** each session deletes hand-written machinery from the previous one.

- Today we hand-roll everything: traversal, matching, replacement, cleanup.
- Session 2: the *driver* does traversal + bookkeeping; you write only the rewrite.
- Session 3: for many rewrites you don't even write a pass — op hooks do it.

<!-- Speaker notes:
Sell the arc: today will feel slightly laborious ON PURPOSE. At the end of session 3 they'll look back at today's code and understand why upstream MLIR is structured the way it is. One concrete promise: the ~20-line function we write today shrinks to ~10 lines in session 2 and partially to zero lines in session 3.
~1 min.
-->

---

# Today's route

1. **Warm-up quiz** — IR anatomy in your own vocabulary
2. **The C++ object model** — `Operation*`, `Region`, `Block`, `Value`
3. **Generic vs. generated ops** — `isa` / `dyn_cast` / `cast`
4. **Use-def chains** — follow the wires, then rewire them
5. **`walk()`** — traversing IR without writing loops
6. **`OpBuilder`** — creating new ops at the right place
7. **Worked example** — strength-reduce `muli` → `shli`
8. **Pass anatomy** — TableGen def → generated base → `runOnOperation`
9. **PassManager & pipelines** — nesting, threading, the CLI
10. **Debugging toolkit** — the flags you'll use every day
11. **Exercise 1** — you build the pass yourself

<!-- Speaker notes:
Quick orientation, don't dwell. Points 2-7 are "the IR mutation API", points 8-10 are "the pass infrastructure around it". The exercise combines both.
~30 s. Leaving this slide you should be ~2 min into the session; the warm-up quiz + answers fill the rest of the 0:00-0:05 block.
-->

---

# 🧠 Quiz: warm-up — point at the pieces

```mlir
func.func @warmup(%a: i32, %n: index) -> i32 {
  %c2 = arith.constant 2 : i32
  %r = arith.muli %a, %c2 {answer = 42 : i64} : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = scf.for %i = %c0 to %n step %c1
      iter_args(%acc = %r) -> (i32) {
    %next = arith.addi %acc, %a : i32
    scf.yield %next : i32
  }
  return %sum : i32
}
```

Point at: **① an operation ② a value ③ a block argument ④ an attribute ⑤ a region**

Bonus: how many operations are written here?

<!-- Speaker notes:
Show of hands / call on people, one item at a time. This recaps their IR-structure session in their own words. The snippet round-trips through mlir-opt unchanged (verified).
Answers (also on next slide):
① Operations: func.func, arith.constant (x3), arith.muli, scf.for, arith.addi, scf.yield, func.return.
② Values: %a, %n, %c2, %r, %c0, %c1, %sum, %i, %acc, %next — every %-thing.
③ Block arguments: %a, %n (entry block of the func body region), and %i, %acc (scf.for body block). The loop IV being a block argument surprises people — good discussion moment.
④ Attributes: {answer = 42 : i64} on the muli; also the 2 : i32 payload of arith.constant, the function name @warmup (sym_name), and the function type — attributes are everywhere, not just in braces.
⑤ Regions: the func body, and the scf.for body.
Bonus: 9 ops as written; 10 counting the implicit builtin.module that mlir-opt wraps around it.
~2 min — keep it rapid-fire (one item at a time); the warm-up block ends at 0:05.
-->

---

# ✅ Warm-up answers

```mlir
func.func @warmup(%a: i32, %n: index) -> i32 {   // op; %a, %n = BLOCK ARGUMENTS
  %c2 = arith.constant 2 : i32                   // op; "2 : i32" is an ATTRIBUTE
  %r = arith.muli %a, %c2 {answer = 42} : i32    // op; {answer=42} = attribute
  ...
  %sum = scf.for %i = %c0 to %n step %c1         // op; %sum = OpResult (a Value)
      iter_args(%acc = %r) -> (i32) {            // { ... } = REGION
    %next = arith.addi %acc, %a : i32            // %i, %acc = block arguments!
    scf.yield %next : i32
  }
  return %sum : i32
}
```

- **9 operations** as written (10 with the implicit `builtin.module`).
- Every `%name` is a **`Value`** — either an op result or a block argument.
- The loop induction variable `%i` and `%acc` are **block arguments** of the region's entry block — not results of `scf.for`.

<!-- Speaker notes:
Key takeaways to say out loud: (1) attributes are compile-time constant data attached to an op — not just the {…} dictionary, also constant payloads and function names; (2) the loop IV is a block argument — MLIR has no phi nodes, block arguments play that role; (3) values come in exactly two kinds — that's literally the C++ class design, which is the next slide.
Presenter flourish for the bonus: 9 ops are written here — then reveal the 10th, the implicit builtin.module that mlir-opt wraps around top-level IR.
~1 min. This closes the 0:00-0:05 warm-up block; the ~40-min lecture core path starts on the next slide.
-->

---

# The C++ object model: it's `Operation*` all the way down

At runtime, *every* op — `arith.addi`, `scf.for`, `func.func`, even `builtin.module` — is a `mlir::Operation`:

```text
Operation                     (heap object, always handled as Operation*)
 ├─ name        "arith.muli"
 ├─ location    Location (where it came from — file:line, or unknown)
 ├─ operands    [Value, Value, ...]      ← uses of values defined elsewhere
 ├─ results     [OpResult, ...]          ← the values THIS op defines
 ├─ attributes  {answer = 42 : i64, ...} ← compile-time constant data
 └─ regions     [Region, ...]
      Region
       └─ blocks [Block, ...]
            Block
             ├─ arguments  [BlockArgument, ...]
             └─ operations [Operation, ...]   ← recursion!
```

The nesting you know from the textual IR **is** the C++ ownership structure: ops own regions, regions own blocks, blocks own ops.

<!-- Speaker notes:
This is the single most important slide of the first half. The textual IR they've been reading is a 1:1 serialization of this object graph. There is no separate AST: what you print is what's in memory.
Emphasize: Operation is uniform — a module and an addi are the same C++ class, differing only in name, operand/result counts, attributes, and regions. This uniformity is what makes generic passes possible.
Handle discipline: Operation is always passed as Operation* (pointer) or Operation& — never by value.
~3 min.
-->

---

# `Value`: exactly two kinds

```text
            Value            (one pointer; pass BY VALUE; compare with ==)
           /     \
     OpResult   BlockArgument
     "defined    "defined by
      by an op"   a block header"
```

- `OpResult` → `getOwner()` returns the defining `Operation*`, `getResultNumber()`
- `BlockArgument` → `getOwner()` returns the `Block*`, `getArgNumber()`
- `val.getDefiningOp()` — the defining op, **or `nullptr` if it's a block argument** (function arguments included!)

```cpp
if (Operation *def = operand.getDefiningOp())
  llvm::outs() << "defined by " << def->getName() << "\n";
else  // must be a block argument
  llvm::outs() << "block arg #" << cast<BlockArgument>(operand).getArgNumber() << "\n";
```

**Handle discipline:** `Value` has value semantics (copy it freely, compare with `==`). `Operation` is pointer-only. Writing `Value*` or copying `Operation` — both wrong.

<!-- Speaker notes:
The getDefiningOp-returns-null gotcha is the #1 beginner segfault: they walk a function, call getDefiningOp on an operand that happens to be a function argument, and dereference nullptr. Drill it now: ALWAYS null-check, or use the templated form val.getDefiningOp<OpTy>() (shown later) which folds the null check in.
Also mention: BlockArgument and OpResult are subclasses of Value, and you can cast a Value with cast<BlockArgument>(v) — the LLVM casting machinery works on values too.
~2 min.
-->

---

# The navigation map ⏱

| I have… | I want… | Call |
|---|---|---|
| `Operation *op` | its parent | `op->getParentOp()`, `op->getParentOfType<func::FuncOp>()` |
| `Operation *op` | its block / region | `op->getBlock()`, `op->getParentRegion()` |
| `Operation *op` | what's nested inside | `op->getRegions()` → `region.getBlocks()` → `block.getOperations()` |
| `Block &b` | only ops of one type | `b.getOps<arith::ConstantOp>()` |
| `Operation *op` | inputs / outputs | `op->getOperand(i)`, `op->getOperands()`, `op->getResult(i)`, `op->getResults()` |
| `Operation *op` | attributes | `op->getAttrs()`, `op->getAttrOfType<IntegerAttr>("answer")` |
| `Operation *op` | context / location | `op->getContext()`, `op->getLoc()` |
| `Value v` | who defines it | `v.getDefiningOp()` *(null for block args)* |
| `Value v` | who uses it | `v.getUsers()`, `v.getUses()` |

Print anything while debugging: `op->dump()`, or stream it: `llvm::errs() << *op << "\n";`

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: this table is your navigation reference card for the exercise — it reappears on the cheat sheet at the end, use it from there.
Don't read the table aloud — tell them this slide is their reference card for the exercise (it reappears in the cheat sheet at the end). Walk through just two rows: op->getParentOfType<func::FuncOp>() ("climb until you hit a function") and block.getOps<T>() ("filtered iteration over one block, no casting needed").
op->dump() prints to stderr with a trailing newline — the debugging workhorse, works in a debugger too ("call op->dump()" in gdb/lldb).
(+1 if presented.)
-->

---

# Two views of one op: generic vs. generated

The same in-memory `Operation` node can be looked at through two lenses:

- **Generic lens** — `Operation *`: uniform API, works for *any* op, even unregistered ones. `getOperand(0)`, `getNumResults()`, `getAttrs()`.
- **Typed lens** — the ODS-generated class, e.g. `arith::AddIOp`: a thin **value-semantic wrapper around the same `Operation*`** that adds named accessors.

```cpp
arith::AddIOp add = ...;
Value lhs = add.getLhs();      // named accessor, from ODS arg name "lhs"
Value lhs2 = add->getOperand(0); // identical value, generic spelling
Operation *raw = add.getOperation(); // unwrap the lens
```

- ODS argument names → accessors: `$lhs` → `getLhs()`, plus `getLhsMutable()` to reassign the operand in place.
- The `->` operator on a typed op forwards to the underlying `Operation`, so *everything* from the previous slides works on typed ops too.

**When to use which?** Generic for op-agnostic code (walking, printing, moving); typed the moment you care what the op *means*.

<!-- Speaker notes:
The mental model to repeat: the typed op class is a LENS, not a different object. arith::AddIOp is one pointer wide — it holds the Operation* and nothing else. Creating/copying op classes is free.
Connect to their ODS session: the accessors come from the names THEY wrote in the .td arguments list. This is where TableGen output meets hand-written C++.
~2 min.
-->

---

# Switching lenses: `isa` / `dyn_cast` / `cast`

MLIR uses LLVM's casting infrastructure (not C++ RTTI — MLIR builds with `-fno-rtti`, `dynamic_cast` won't even compile):

```cpp
void inspect(Operation *op) {
  // Generic API: works for ANY op, registered or not.
  llvm::outs() << op->getName().getStringRef() << " has "
               << op->getNumOperands() << " operands, "
               << op->getNumResults() << " results\n";
  for (NamedAttribute attr : op->getAttrs())
    llvm::outs() << "  attr " << attr.getName() << " = " << attr.getValue() << "\n";

  // Typed API: cast to the ODS-generated class for named accessors.
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    Value lhs = add.getLhs();    // == op->getOperand(0)
    Value res = add.getResult(); // == op->getResult(0)
  }
}
```

| Call | Behavior |
|---|---|
| `isa<AddIOp>(op)` | `bool` — is it one? |
| `dyn_cast<AddIOp>(op)` | typed op, or *null op* on mismatch — **the workhorse** |
| `cast<AddIOp>(op)` | asserts on mismatch — only when you *know* |
| `dyn_cast_or_null<T>(x)` | like `dyn_cast`, but also tolerates null input |

<!-- Speaker notes:
Two gotchas to say out loud:
1. isa/dyn_cast on a possibly-NULL Operation* asserts — that's what dyn_cast_or_null / isa_and_nonnull are for. Classic combo with getDefiningOp: val.getDefiningOp<arith::ConstantOp>() does the null-safe dyn_cast in one step.
2. op->getName() returns an OperationName object, not a string. Streaming it works; comparing needs getName().getStringRef() — but if you're comparing op names as strings, you almost always want isa<> instead.
The inspect() snippet is verified to compile against this checkout.
~2 min. Core-path check: leaving this slide you should be ~9 min into the lecture (≈0:14 on the session clock).
-->

---

# Use-def chains: the wires between ops

SSA in memory: every `Value` keeps a **list of its uses**. A *use* is an `OpOperand` — an edge that knows both endpoints:

```text
          defines                      operand #0
  ┌────────────┐   use-list   ┌─────────────────────┐
  │ %x = addi  │ ───────────► │ OpOperand            │──► owner: muli
  └────────────┘        │     └─────────────────────┘
                        │     ┌─────────────────────┐
                        └───► │ OpOperand            │──► owner: return
                              └─────────────────────┘
```

```cpp
for (Operation *user : x.getUsers())       // the ops using %x
  llvm::outs() << "used by " << user->getName() << "\n";
for (OpOperand &use : x.getUses())         // the edges: owner + operand index
  llvm::outs() << "operand #" << use.getOperandNumber()
               << " of " << use.getOwner()->getName() << "\n";
```

Cheap queries: `x.use_empty()`, `x.hasOneUse()`, `x.hasNUses(n)`.
**Trap:** `x.getNumUses()` walks the whole list — *linear time* (documented!). Don't count when you can ask.

<!-- Speaker notes:
The two directions to keep straight: operands point "up" to values (data flowing in); the use-list points "down" from a value to everyone consuming it (this is the extra bookkeeping SSA buys you — you can find all users in O(uses)).
getUsers() vs getUses(): users gives Operation*, uses gives the edges (OpOperand&) — you need the edge when you care WHICH operand slot uses the value, or want to reassign just that slot (use.set(newValue)).
Subtle gotcha: getUsers() can yield the same op twice if it uses the value in two operands (muli %x, %x). It iterates uses, not a deduplicated set.
Linear getNumUses is at mlir/include/mlir/IR/Value.h:189-193.
~2 min.
-->

---

# Rewiring: `replaceAllUsesWith` and friends

“RAUW” = replace all uses with. Three flavors, one inverse:

```cpp
// 1. Value-level: every use of one SSA value now uses another.
oldVal.replaceAllUsesWith(newVal);

// 2. Conditional: only the uses your predicate accepts.
oldVal.replaceUsesWithIf(newVal, [&](OpOperand &use) {
  return use.getOwner()->getBlock() == someBlock;
});

// 3. Operation-level: all results at once — then the op is dead.
oldOp->replaceAllUsesWith(newOp->getResults());
oldOp->erase();   // safe ONLY once use_empty()

// Inverse direction: rewrite ONE op's own operands.
userOp->replaceUsesOfWith(/*from=*/oldVal, /*to=*/newVal);
```

**Two directions — don't mix them up:**
- `replaceAllUsesWith` edits *other ops'* operands (everyone pointing at me).
- `replaceUsesOfWith` edits *this op's* operands (what I point at).

**Iron rule: RAUW first, `erase()` second.** Erasing an op whose results still have uses is a fatal error in assert builds (`"operation destroyed but still has uses"`) — and silent memory corruption in release builds.

<!-- Speaker notes:
This slide is the heart of IR surgery. The RAUW-then-erase ordering rule will come back in the spot-the-bug quiz.
The fatal error is in Operation's destructor (mlir/lib/IR/Operation.cpp:176-186) — in debug builds it even prints each offending user before dying. Show appreciation for assert builds: this is why we teach on an assertions-enabled tree.
Fine print: op->replaceAllUsesWith is a template over "values"; for a single Value wrap it — op->replaceAllUsesWith(ValueRange{v}) — or go through op->getResult(0).replaceAllUsesWith(v). Mismatched result counts assert at runtime.
~2 min.
-->

---

# 🧠 Quiz: after the RAUW

```mlir
func.func @quiz(%a: i32, %b: i32) -> i32 {
  %x = arith.addi %a, %b : i32
  %y = arith.subi %a, %b : i32
  %u = arith.muli %x, %x : i32
  %v = arith.addi %u, %y : i32
  return %v : i32
}
```

Suppose a pass runs: `x.replaceAllUsesWith(y);` *(where `x`, `y` are the C++ `Value`s for `%x`, `%y`)*

1. Which operations had their operands changed?
2. Can we now safely call `erase()` on the op defining `%x`?
3. Can we safely `erase()` the op defining `%y`?
4. What does `y.getUsers()` yield now?

<!-- Speaker notes:
Give ~90 seconds of think time, collect answers by show of hands per question.
Answers (next slide has them too):
1. Exactly ONE op changed: the muli — both of its operand slots (%x, %x) now hold %y. Nothing else used %x. %v is untouched: it uses %u and %y, and RAUW never touches the definition of %y.
2. YES — %x is now use_empty(), so erasing the addi is legal.
3. NO — %y now has three uses (muli twice + the final addi). Erase would hit the "still has uses" fatal error.
4. Three entries: muli, muli, addi — getUsers() iterates USES, so an op using the value twice appears twice. If someone answers "two ops", that's the teachable moment.
~2 min with discussion.
-->

---

# ✅ RAUW quiz answers

```mlir
  %x = arith.addi %a, %b : i32   // ← now use_empty(): erasable ✔
  %y = arith.subi %a, %b : i32   // ← now has THREE uses: NOT erasable ✘
  %u = arith.muli %y, %y : i32   // ← the only op that changed (both slots)
  %v = arith.addi %u, %y : i32   // ← unchanged (never used %x)
  return %v : i32
```

1. **Only the `muli` changed** — RAUW rewrites the *uses* of `%x`, nothing else.
2. **Yes** — `%x` is `use_empty()`; `erase()` is safe.
3. **No** — `%y` now has 3 uses; erasing its definer would be the fatal `"operation destroyed but still has uses"`.
4. `y.getUsers()` yields **`muli`, `muli`, `addi`** — one entry *per use*, so `muli` appears twice.

<!-- Speaker notes:
Reinforce: RAUW is O(uses of %x) — it splices the use-list, it doesn't scan the function. That's what the use-list bookkeeping buys.
If time allows, foreshadow: "erase the addi defining %x" is exactly what DCE does automatically for side-effect-free ops — Session 3.
~1 min. Core-path check: ~16 min into the lecture (≈0:21).
-->

---

# Traversal, level 0: follow the nesting by hand ⏱

The object model gives you the loops for free (this is the core of the real `-test-print-nesting` pass):

```cpp
void printOperation(Operation *op) {
  llvm::outs() << "op: " << op->getName() << "\n";
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())   // Region is iterable too
      for (Operation &nested : block)          // == block.getOperations()
        printOperation(&nested);
}

// Filtered, one nesting level only:
for (arith::ConstantOp c : block.getOps<arith::ConstantOp>())
  llvm::outs() << "constant: " << c.getValue() << "\n";
```

Fine for one level. Tedious for "find every `muli` anywhere in this function" — regions nest arbitrarily deep (`scf.for` inside `scf.if` inside …).

<sub>adapted from mlir/test/lib/IR/TestPrintNesting.cpp:32-74</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: you can hand-roll nested for-loops over regions/blocks/ops — that's what -test-print-nesting does — but walk() on the next slide is what you'll actually write.
Point out this is real upstream code (the test pass we'll run in the demo shortly), just condensed. The recursion mirrors the ownership diagram from earlier — nothing new, that's the point.
Then set up the pain: any interesting matcher needs to see through nesting, and hand-rolled recursion everywhere is noise. Enter walk().
(+1 if presented.)
-->

---

# Traversal, level 1: `walk()`

`walk` = “call my callback on every op nested under (and including!) this one”:

```cpp
// Every op, post-order (children before parents), lexical order in blocks:
root->walk([](Operation *op) { llvm::outs() << op->getName() << "\n"; });

// Typed filtering — the callback's parameter type selects the ops:
root->walk([](arith::MulIOp op) { /* only muli ops, already typed! */ });

// Parents before children instead:
root->walk<WalkOrder::PreOrder>([](Operation *op) { /* ... */ });
```

- Default order is **post-order** — by the time you see an op, its regions are done.
- The typed-callback filter is not a cast — it *silently skips* every non-matching op. Interfaces work too: `walk([](LoopLikeOpInterface loop) {...})`.
- `Block::walk` and `Region::walk` exist as well.

<!-- Speaker notes:
Two things students misread:
1. walk visits the root itself ("nested under AND INCLUDING") — a walk started at a ModuleOp calls the callback on the module too. Surprising when the callback assumes ops with a parent.
2. The typed parameter is a FILTER, not a cast — beginners think it's sugar for cast<> and are confused why "nothing happens" for other ops. It's implemented with dyn_cast internally; non-matching ops are skipped.
Post-order default is documented in Operation.h (walk template default WalkOrder::PostOrder, Operation.h:817).
~2 min.
-->

---

# Controlling a walk: `WalkResult` ⏱

Return `WalkResult` from the callback to steer the traversal — `advance` continues, `interrupt` aborts everything, `skip` prunes the subtree:

```cpp
WalkResult result = root->walk([](Operation *op) {
  if (isa<gpu::LaunchOp>(op))
    return WalkResult::interrupt();  // abort the entire walk
  if (isa<scf::ForOp>(op))
    return WalkResult::skip();       // don't descend into loops
  return WalkResult::advance();      // keep going (the default)
});
if (result.wasInterrupted()) { /* found one */ }
```

- A `void` callback = always advance.
- `WalkResult` implicitly converts from `LogicalResult` (failure ⇒ interrupt) — handy for verifier-style walks.

<sub>mlir/include/mlir/Support/WalkResult.h:29-55</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the walk callback can return WalkResult::interrupt/skip/advance to abort or prune the traversal — look this slide up when you need it.
interrupt vs skip in one sentence: interrupt = "I'm done with the whole walk" (e.g. found what I searched for); skip = "don't look inside THIS op, continue with its siblings" (e.g. don't descend into this loop).
Note skip only makes sense in pre-order walks — in post-order the nested ops were already visited by the time the callback runs on the parent.
(+1 if presented.)
-->

---

# Mutating during a walk: the rules

The documented guarantee (Operation.h): the callback may **erase the op it is currently visiting** if

- the walk is **post-order** (the default), **or**
- the walk is pre-order **and** you return `WalkResult::skip()` after erasing.

Everything else — erasing *other* not-yet-visited ops, erasing your parent — invalidates the iteration. Undefined behavior, often "works" until it doesn't.

**The robust beginner idiom: collect, then mutate.**

```cpp
SmallVector<arith::MulIOp> candidates;
getOperation()->walk([&](arith::MulIOp op) { candidates.push_back(op); });

for (arith::MulIOp op : candidates) {
  // ... create replacement, RAUW, op->erase() — no live iterator to break
}
```

Two phases, no aliasing between iteration and mutation. Costs one `SmallVector`; saves one afternoon of debugging.

<sub>guarantee: mlir/include/mlir/IR/Operation.h:796-799</sub>

<!-- Speaker notes:
Why the guarantee holds: the walk implementation iterates blocks with an early-increment iterator — it steps PAST the current op before invoking the callback, so deleting the current op is fine, but the "next" pointer already taken can be invalidated by deleting other ops.
Message to hammer: when in doubt, collect-then-mutate. It also separates "what to change" from "how to change it", which makes the code reviewable. The exercise starter nudges toward this shape.
Ops you CREATE during a walk before the current position are fine — they simply won't be visited (post-order already passed them).
~2 min.
-->

---

# 🔴 Live demo: seeing the structure with a real tool

```bash
build/bin/mlir-opt -test-print-nesting -allow-unregistered-dialect \
    mlir/test/IR/print-ir-nesting.mlir -o /dev/null
```

```text
visiting op: 'builtin.module' with 0 operands and 0 results
 1 nested regions:
  Region with 1 blocks:
    Block with 0 arguments, 0 successors, and 2 operations
      visiting op: 'dialect.op1' with 0 operands and 4 results
      1 attributes:
       - 'attribute name' : '42 : i32'
       0 nested regions:
      visiting op: 'dialect.op2' with 0 operands and 0 results
      ...
        Region with 3 blocks:
          Block with 0 arguments, 2 successors, and 2 operations
```

<!-- Speaker notes:
Run it live from the llvm-project checkout root; output above is the real (trimmed) output of this exact command on this tree.
Narrate the correspondence: this text is printOperation/printRegion/printBlock from the traversal slides, executing. The source is mlir/test/lib/IR/TestPrintNesting.cpp — genuinely readable, send them there.
One-line mention, don't run it: the sibling pass --test-print-defuse prints producer/consumer info for every value (mlir/test/lib/IR/TestPrintDefUse.cpp).
Note the -o /dev/null: these passes only print; we discard the IR output.
These test-* flags exist because mlir-opt is built with MLIR_INCLUDE_TESTS=ON (in-tree default); they're not in libMLIR.
~2 min. Core-path check: ~22 min into the lecture (≈0:27).
-->

---

# `OpBuilder`: a cursor into the IR

To create ops you need two things: an `MLIRContext` and a **place**. `OpBuilder` bundles both — it's a cursor (`Block*` + position) where every newly created op is inserted:

```cpp
OpBuilder b(ctx);                    // ⚠ NO insertion point yet!
b.setInsertionPointToStart(&block);  // cursor: start of block
b.setInsertionPoint(op);             // cursor: right BEFORE op
b.setInsertionPointAfter(op);        // cursor: right after op

OpBuilder before(op);                // shorthand: constructor == setInsertionPoint(op)
OpBuilder atEnd = OpBuilder::atBlockEnd(&block);
```

**Gotcha #1:** `OpBuilder b(op)` inserts **BEFORE** `op` — not after. (Perfect when replacing `op`; surprising otherwise.)

**Gotcha #2:** `OpBuilder b(ctx)` alone has no insertion point. Creating an op then does **not** crash — the op is built *detached*, inserted nowhere: it silently never appears in your output (and leaks). The docs call this an error; the API won't stop you.

<sub>mlir/include/mlir/IR/Builders.h:215-258, 400-440</sub>

<!-- Speaker notes:
Analogy that lands: the builder is a text-editor cursor. Constructors and setInsertionPoint* move the cursor; creating an op types at the cursor (and the cursor advances past what was typed, so consecutive creates appear in order).
Gotcha #2 detail (verified on this tree): OpBuilder::insert does `if (block) ...` and silently skips insertion when no insertion point is set (mlir/lib/IR/Builders.cpp, OpBuilder::insert) — no assert fires, even in debug builds. The op is simply detached. Do NOT tell students it crashes; the danger is precisely that it doesn't.
The BEFORE-not-after choice makes sense once you see the replacement idiom: new ops go before the op being replaced so its uses can be redirected to them.
~2 min.
-->

---

# Saving your place: `InsertionGuard` ⏱

Moving the cursor deep into a region and forgetting to move it back is the classic “my ops ended up in the wrong block” bug. RAII to the rescue:

```cpp
{
  OpBuilder::InsertionGuard guard(b);   // saves the current insertion point
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(42));
} // cursor restored here, automatically
```

- Use it whenever a helper function temporarily redirects a builder it doesn't own.
- Nested guards nest correctly (it's just save/restore).

<sub>mlir/include/mlir/IR/Builders.h:350-376</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: OpBuilder::InsertionGuard is RAII save/restore for the insertion point — open any helper that moves a borrowed builder's cursor with one; it's on the cheat sheet.
Short slide — one habit to install: any function that takes OpBuilder& and moves the insertion point should open with an InsertionGuard, unless moving the point IS its contract.
(+1 if presented.)
-->

---

# Creating ops: `OpTy::create(builder, loc, ...)`

Every ODS op gets static `create` methods — one per `build(...)` overload you saw in the ODS session:

```cpp
OpBuilder b(op);                    // insert before the op we're replacing
Location loc = op.getLoc();         // reuse its location (more in a second)

Value shift = arith::ConstantOp::create(
    b, loc, b.getIntegerAttr(op.getType(), 3));
Value shl = arith::ShLIOp::create(b, loc, op.getLhs(), shift);
```

> **Heads-up for old code:** you will see `builder.create<arith::ShLIOp>(loc, ...)` in blog posts, pre-2025 tutorials, and LLM output. That spelling is **deprecated** (since Oct 2025) — same behavior, but new code must use `OpTy::create(builder, loc, ...)`. Mentally rewrite the old form when you see it.

<sub>deprecation: mlir/include/mlir/IR/Builders.h:506-508</sub>

<!-- Speaker notes:
This deck mentions the deprecation exactly once — here. If students copy-paste the old form they'll get -Wdeprecated warnings, not errors, so tell them to treat the warning as a stop sign.
The create call inserts at the builder's cursor AND returns the typed op; assigning to Value implicitly takes result 0 (single-result ops convert to their result Value).
~1 min.
-->

---

# Constants, attributes, and `Location` discipline ⏱

`OpBuilder` inherits a bag of factory helpers from `Builder` (types, attributes, locations):

```cpp
IntegerAttr a  = b.getI32IntegerAttr(7);          // 7 : i32
IntegerAttr ix = b.getIndexAttr(4);               // 4 : index
Type i32       = b.getI32Type();
IntegerAttr c  = b.getIntegerAttr(someType, 3);   // typed constant payload

// A constant op is: attribute payload + arith.constant
Value cst = arith::ConstantOp::create(b, loc, b.getIntegerAttr(ty, 3));
```

**Location discipline.** Every op carries a `Location` (diagnostics + debug info survive on it):

- When replacing an op, **reuse `op.getLoc()`** for everything you create in its place.
- `b.getUnknownLoc()` is the last resort — an op with unknown location can't point users anywhere when a diagnostic fires.

One-liner to know: `b.createOrFold<OpTy>(loc, ...)` — create the op *and immediately try to fold it*; returns a `Value` that may be a pre-existing constant instead of a new op. (Folding is Session 3's topic.)

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the builder factory helpers (getIntegerAttr & co.) appear in the worked example in a minute, and the one location rule is mechanical — new ops that replace an old op reuse op.getLoc().
Locations feel like bureaucracy until the first time a verifier error says "unknown location" and you have 40k ops. Make the rule mechanical: new op replaces old op => new op gets old op's loc. There are FusedLoc/NameLoc for merging several source ops into one result op — mention only if asked.
createOrFold: don't deep-dive; plant the name so Session 3 can pick it up ("who folded my op? you asked for it").
(+1 if presented.) The worked-example section is next — it's the payoff, give it room.
-->

---

# 🧠 Quiz: spot the bug

Goal: rewrite `muli %x, C` (C a power of two) into `shli`. Does this work?

```cpp
void runOnOperation() override {
  getOperation()->walk([&](arith::MulIOp op) {
    APInt rhs;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) || !rhs.isPowerOf2())
      return;
    OpBuilder b(op);
    Value shift = arith::ConstantOp::create(
        b, op.getLoc(), b.getIntegerAttr(op.getType(), rhs.logBase2()));
    Value shl = arith::ShLIOp::create(b, op.getLoc(), op.getLhs(), shift);
    op->erase();                              // (A)
    op->replaceAllUsesWith(ValueRange{shl});  // (B)
  });
}
```

Three candidate concerns — which are real bugs?
① The erase/RAUW order (A)/(B) &nbsp; ② erasing `op` *inside* the walk &nbsp; ③ creating new ops *inside* the walk

<sub>First use: `matchPattern(v, m_ConstantInt(&rhs))` = "is `v` a constant integer? bind its value into `rhs`" — details on a later slide.</sub>

<!-- Speaker notes:
Give ~90 seconds. matchPattern/m_ConstantInt is glossed in the on-slide footnote (read it aloud so the quiz is self-contained) and explained fully on the worked-example slide.
Answers:
① REAL BUG, twice over. erase() while the result still has uses = fatal "operation destroyed but still has uses" in assert builds / memory corruption in release. And line (B) then calls a method on freed memory — use-after-free. Correct order: RAUW, THEN erase.
② NOT a bug — erasing the currently-visited op in a (default) post-order walk is explicitly documented as allowed (Operation.h:796-799). But fragile under refactoring (someone flips it to pre-order…), which is why the model solution collects first.
③ NOT a bug — the new ops are inserted before the current op; a post-order walk has already moved past that position, they simply won't be visited.
Poll each concern separately with a show of hands — ② splits the room nicely.
~2 min including discussion.
-->

---

# ✅ Spot-the-bug answers

**① is the bug** — and it's two bugs in one:

- `erase()` while the `muli`'s result still has uses → fatal error in assert builds:
  `"operation destroyed but still has uses"` (release builds: silent corruption).
- Line (B) then touches the *freed* op — use-after-free.

**Fix: swap the two lines.** RAUW first, erase second. Always.

**② is legal** — a post-order walk may erase the *currently visited* op (documented guarantee). Erasing *other* ops during the walk would be the bug.

**③ is fine** — new ops inserted before the current op are simply never visited by the ongoing post-order walk.

Still, the model solution uses **collect-then-mutate**: it doesn't depend on walk-order fine print, and it survives refactoring.

<!-- Speaker notes:
Repeat the iron rule slowly: RAUW, then erase. It's the single most common crash in exercise submissions.
Transition: "now let's see the correct version — in full, and running."
~1 min.
-->

---

# Worked example: `MulToShift`, the real thing

```cpp
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
```

- `matchPattern(value, m_ConstantInt(&rhs))` — "is this value a constant integer? if so, bind it into `rhs`" (an `APInt`). Matches *any* constant-like op, not just `arith.constant`.
- `rhs.isPowerOf2()`, `rhs.logBase2()` — `APInt` does the math.

<!-- Speaker notes:
This exact code is compiled and run against this checkout (as an mlir-opt pass plugin) — the next slide shows its real output. It is also, deliberately, the shape of Exercise 1: walk+collect (checkpoint 1), match, build, RAUW, erase (checkpoint 2), and the guards that make it robust (checkpoint 3).
Why matchPattern instead of dyn_cast<arith::ConstantOp>? It sees through anything ConstantLike (other dialects' constants, splat constants) — the idiomatic upstream way to test for constants (mlir/include/mlir/IR/Matchers.h).
Walk through the data flow once: candidates -> guard -> builder before op -> two new ops -> RAUW -> erase. Every line uses an API from the last 20 slides. Nothing new except matchPattern (glossed on the quiz slide already).
~2 min.
-->

---

# 🔴 Live demo: run it

```bash
build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
    --pass-pipeline='builtin.module(func.func(mul-to-shift))'
```

<div class="columns">
<div>

**Before** (`mul.mlir`)

```mlir
func.func @mul_by_8(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %r = arith.muli %x, %c8 : i32
  return %r : i32
}

func.func @mul_by_7(%x: i32) -> i32 {
  %c7 = arith.constant 7 : i32
  %r = arith.muli %x, %c7 : i32
  return %r : i32
}
```

</div>
<div>

**After** (real output)

```mlir
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c8_i32 = arith.constant 8 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}

func.func @mul_by_7(%arg0: i32) -> i32 {
  %c7_i32 = arith.constant 7 : i32
  %0 = arith.muli %arg0, %c7_i32 : i32
  return %0 : i32
}
```

</div>
</div>

`@mul_by_7` untouched (7 isn't a power of two) — but look at `@mul_by_8`: **`%c8_i32` is now dead.** We never erased it. Who cleans that up? → Session 3. (Spoiler: append `,canonicalize` to the pipeline.)

<!-- Speaker notes:
Demo prep (do before class; total ~30s): compile the pass as a plugin so stock mlir-opt can run it —
  c++ -shared -fPIC -fno-rtti -std=c++17 -O1 \
    -I mlir/include -I build/tools/mlir/include -I llvm/include -I build/include \
    mul_to_shift.cpp -o mul_to_shift.so
(mul_to_shift.cpp = the runOnOperation from the previous slide wrapped in a PassWrapper + mlirGetPassPluginInfo; keep that boilerplate off-slide — students get the pass pre-registered in school-opt and never need plugins.)
The "After" pane is the exact mlir-opt output of this command on this machine (module wrapper trimmed).
Things to narrate: (1) %x became %arg0 — the printer regenerates value names; names are display-only, not identity; (2) the shift constant %c3_i32 landed right before the shli — that's OpBuilder b(op) inserting BEFORE the muli; (3) THE DEAD %c8_i32 — we replaced+erased the muli but nobody asked us to erase the constant. Run the punchline live:
  build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
    --pass-pipeline='builtin.module(func.func(mul-to-shift,canonicalize))'
→ the dead constant vanishes (verified). One sentence: "canonicalize includes dead-code elimination for side-effect-free ops — Session 3 explains exactly why and when."
~3 min. Core-path check: ~33 min into the lecture (≈0:38). If over, skip the canonicalize punchline run and just tell it — the print-ir-after-all demo at the end reuses this exact run and shows the same story.
-->

---

# From snippet to pass: what's still missing

Our `runOnOperation` needs a home. A **pass** needs:

- a **name** and a **CLI flag** (`-mul-to-shift`)
- an **anchor**: which op does it run on? (`func.func` for us)
- **registration** so `mlir-opt` / `school-opt` can find it
- plumbing: `clonePass`, options, statistics, …

You write **none of that by hand.** The modern recipe:

```text
Passes.td                 TableGen definition (flag, anchor, docs, options)
   │  mlir-tblgen -gen-pass-decls
   ▼
Passes.h.inc              generated: impl::<Name>Base<Derived> CRTP base class
   │  you derive from it
   ▼
YourPass.cpp              #define GEN_PASS_DEF_<NAME> + runOnOperation()
```

<!-- Speaker notes:
Frame it as: the pass *body* is what you saw; the pass *identity* is declarative. Same philosophy as ODS for ops — declare in TableGen, generate the boilerplate, implement only the interesting method.
Historical note if asked: you'll find PassWrapper-based passes in test code and old tutorials — that's the manual non-TableGen path, only for quick hacks (the demo plugin uses it for compactness).
~1 min.
-->

---

# The smallest real pass: `-strip-debuginfo`

The **entire** implementation, from upstream (license, includes, `using namespace mlir;`, and one inner loop trimmed):

```cpp
namespace mlir {
#define GEN_PASS_DEF_STRIPDEBUGINFOPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
struct StripDebugInfo : public impl::StripDebugInfoPassBase<StripDebugInfo> {
  void runOnOperation() override;
};
} // namespace

void StripDebugInfo::runOnOperation() {
  auto unknownLoc = UnknownLoc::get(&getContext());
  getOperation()->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    // ... also resets block-argument locations (three nested loops) ...
  });
}
```

<sub>mlir/lib/Transforms/StripDebugInfo.cpp:14-42</sub>

And its TableGen side — this is *all* of it:

```tablegen
def StripDebugInfoPass : Pass<"strip-debuginfo"> {
  let summary = "Strip debug info from all operations";
  let description = [{
    This pass strips the IR of any location information, by replacing all
    operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).
  }];
}
```

<sub>mlir/include/mlir/Transforms/Passes.td:488-494</sub>

<!-- Speaker notes:
Let this sink in: a shipping upstream pass is 42 lines including the license header. The recipe: (1) GEN_PASS_DEF macro + include generates impl::StripDebugInfoPassBase; (2) derive from it (CRTP — the base is templated on your class); (3) override runOnOperation. That's it — flag name, description, clonePass, registration all come from TableGen.
If curious, the generated base lives in build/tools/mlir/include/mlir/Transforms/Passes.h.inc — and note clonePass — foreshadows threading: the pass manager CLONES passes to run them in parallel, which is why passes must be copy-constructible.
Naming convention: .td def name ends in "Pass" (StripDebugInfoPass) → generated base is StripDebugInfoPassBase. Old tutorials show defs without the suffix — the convention changed.
Pass<"strip-debuginfo"> with no second argument = op-agnostic pass (can be scheduled on any op). Anchored version on the exercise slide in a minute.
~2 min.
-->

---

# A pass is often just a walk + a library call ⏱

`-loop-invariant-code-motion`, complete `runOnOperation`:

```cpp
void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}
```

<sub>mlir/lib/Transforms/LoopInvariantCodeMotion.cpp:41-47</sub>

- The walk callback filters by **interface** — this pass hoists out of `scf.for`, `affine.for`, *and any future loop op* without knowing them.
- Post-order = innermost loops first — the comment relies on the default walk order you learned 10 slides ago.
- The heavy lifting (`moveLoopInvariantCode`) is a reusable library function.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: many real upstream passes are just a walk plus one library call — LICM's entire runOnOperation is five lines; read it tonight (mlir/lib/Transforms/LoopInvariantCodeMotion.cpp).
Purpose of this slide: calibrate expectations. Students assume "real passes" are thousands of lines of arcana; many are a walk plus a call into a transform library. The interface-based filter is the teaser for the later interfaces session — one sentence only: "LoopLikeOpInterface is a contract that loop-ish ops implement; walk can filter on contracts, not just concrete ops."
(+2 if presented.)
-->

---

# Your exercise starter: `-school-strength-reduce`

```tablegen
// exercises: include/School/SchoolPasses.td (description + statistics elided)
def SchoolStrengthReduce : Pass<"school-strength-reduce", "::mlir::func::FuncOp"> {
  let summary = "Rewrite muli-by-power-of-two into shli (hand-written walk)";
}
```

```cpp
// exercises: lib/School/StrengthReduce.cpp — your Exercise 1 starter (TODO text abridged)
namespace mlir::school {
#define GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE
#include "School/SchoolPasses.h.inc"

namespace {
struct SchoolStrengthReduce
    : public impl::SchoolStrengthReduceBase<SchoolStrengthReduce> {
  void runOnOperation() override {
    // TODO(exercise 1, step 1): Find the candidates (walk + match).
    // TODO(exercise 1, step 2): Build the replacement ops.
    // TODO(exercise 1, step 3): Replace and erase.
  }
};
} // namespace
} // namespace mlir::school
```

The second `Pass<...>` argument **anchors** the pass: it runs on (and gets scheduled per) `func.func`. `getOperation()` is typed accordingly — it returns a `func::FuncOp`.

<!-- Speaker notes:
This is the actual shape of the starter code in the exercises/ repo — students only fill the TODO body; the .td entry, CMake, and registration in school-opt are already wired. The real .td entry additionally declares a numRewrites statistic (used by stretch goal a) and its dependent dialects.
Naming: this project's def name has no "Pass" suffix (unlike the current upstream convention mentioned on the StripDebugInfo slide) — the generated base is always <DefName>Base, hence GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE and impl::SchoolStrengthReduceBase here.
Anchoring: Pass<"flag"> = op-agnostic; Pass<"flag", "::mlir::func::FuncOp"> = runs once per function. Inside an anchored pass, getOperation() returns the typed anchor op (func::FuncOp here) — no casting.
Why anchor on functions and not modules? Two reasons, both coming up: (1) parallelism — the PM runs the pass on all functions concurrently; (2) scoping — the pass may only touch IR inside its anchor.
~1 min.
-->

---

# Pass options and statistics ⏱

Declared in TableGen; materialize as *member variables* of the generated base:

```tablegen
// real upstream examples (summary/description fields elided)
def TrivialDeadCodeEliminationPass : Pass<"trivial-dce"> {
  let options = [
    Option<"recursive", "recursive", "bool", /*default=*/"true",
           "Recursively visit nested regions">,
    Option<"removeBlocks", "remove-blocks", "bool", /*default=*/"true",
           "Remove unreachable blocks">];
}
def CSEPass : Pass<"cse"> {
  let statistics = [
    Statistic<"numCSE", "num-cse'd", "Number of operations CSE'd">,
    Statistic<"numDCE", "num-dce'd", "Number of operations DCE'd">];
}
```

<sub>mlir/include/mlir/Transforms/Passes.td:103-125, 88-101</sub>

In C++: read `recursive` like a bool; assign/increment `numCSE` like an int. On the CLI (📸 captured output):

```text
$ mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(cse))" --mlir-pass-statistics
'func.func' Pipeline
  CSEPass
    (S) 2 num-cse'd - Number of operations CSE'd
    (S) 0 num-dce'd - Number of operations DCE'd
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: options and statistics are declared in the pass's TableGen entry and become plain member variables — stretch goal (a) uses the pre-declared numRewrites statistic exactly like CSE's counters.
Pre-captured — do not run live unless ahead of schedule; command: mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(cse))" --mlir-pass-statistics.
Options: Option<"cppName", "cli-arg", type, default, description> → a member you read as a plain variable inside runOnOperation, settable from the pipeline string: trivial-dce{recursive=false}.
Statistics: same idea, countable — Exercise 1 stretch goal (a) fills the pre-declared numRewrites statistic and observes it exactly like the CSE output shown (that output block is real, captured from this tree; the three '=== Pass statistics report ===' banner lines above it are trimmed).
(+2 if presented.)
-->

---

# Failing, and staying honest ⏱

**Failing:** something's broken → `signalPassFailure()`.

```cpp
if (failed(somethingImportant()))
  return signalPassFailure();   // note the return!
```

- It **sets a bit — it does not return, throw, or stop your code.** The idiom is `return signalPassFailure();`
- The pipeline stops after your pass; `PassManager::run` returns failure; the IR may be left in an invalid state.

**Staying honest:** after *every* pass, the pass manager runs the **verifier** on the result (`--verify-each`, on by default).

- Your pass must leave IR that verifies — broken dominance, wrong operand types, malformed ops get caught right after *your* pass ran, not three passes later.
- The error names the offending *op* (e.g. `error: 'test.foo' op requires attribute 'attr'`); to see which pass broke it, add `--mlir-print-ir-after-failure` or `--mlir-print-ir-after-all`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: two one-liners — signalPassFailure() only sets a bit (idiom: return signalPassFailure();), and the verifier runs after every pass, so a broken rewrite is caught right after YOUR pass ran.
signalPassFailure NOT returning is a real bug generator: students call it in an if and then fall through into code that assumes success. Make them chant the idiom.
Verifier framing: it's a safety net you'll be grateful for during the exercise — if your rewrite produces type-incorrect IR, mlir-opt tells you right after your pass runs rather than after canonicalize mangles it further. Disabling (--verify-each=false) is for performance measurements, not for making errors go away.
Live proof if asked (real, verified): mlir-opt in.mlir --pass-pipeline='builtin.module(func.func(test-pass-create-invalid-ir))' → "error: 'test.any_attr_of_i32_str' op requires attribute 'attr'" — the verifier fires between passes; the message names the op, and the IR-printing flags reveal the guilty pass.
Note the distinction: erase()-with-uses crashes IMMEDIATELY inside your pass (destructor assertion from earlier), it doesn't wait for the verifier.
(+2 if presented.)
-->

---

# PassManager: a tree of pipelines ⏱

Pipelines are **anchored** and **nested**, mirroring the IR structure:

```text
PassManager  (anchor: builtin.module)
 ├─ inline                          ← runs on the module
 └─ OpPassManager (func.func)       ← nested pipeline
     ├─ canonicalize                ← runs on EVERY func.func …
     └─ cse                         ← … potentially in parallel
```

```cpp
mlir::PassManager pm(module.get()->getName());
pm.addPass(mlir::createInlinerPass());              // module-level

mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());     // function-level
optPM.addPass(mlir::createCSEPass());

if (mlir::failed(pm.run(*module)))
  return 4;
```

<sub>adapted from mlir/examples/toy/Ch6/toyc.cpp:149-196</sub>

Shorthand for one-offs: `pm.addNestedPass<func::FuncOp>(createCSEPass());`

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: pipelines are anchored and nested exactly like the IR — you already saw the anchor idea on the exercise starter; in C++ the shorthand is pm.addNestedPass<func::FuncOp>(...).
Key vocabulary: ANCHOR = the op type a pipeline runs on. The tree mirrors IR nesting: module pipeline contains a func.func pipeline, just like modules contain functions.
The C++ snippet is real Toy tutorial code (condensed). Emphasize run() returns LogicalResult — failure propagates from signalPassFailure.
(+2 if presented.)
-->

---

# The nesting gotcha (you *will* hit this today) ⏱

Adding a `func.func`-anchored pass directly to a module-level pipeline is a **hard error**, not an auto-fix:

```text
$ mlir-opt in.mlir --pass-pipeline="builtin.module(affine-loop-tile)"
error: Can't add pass 'AffineLoopTiling' restricted to 'func.func' on a
PassManager intended to run on 'builtin.module', did you intend to nest?
```

…and running an anchored pipeline on the wrong op fails too:

```text
$ mlir-opt in.mlir --pass-pipeline="func.func(cse)"
error: can't run 'func.func' pass manager on 'builtin.module' op
```

**Confusingly:** the bare flag `mlir-opt -affine-loop-tile in.mlir` *works* — bare flags use **implicit nesting** and silently build `builtin.module(func.func(affine-loop-tile))`.

See what was actually built: `--dump-pass-pipeline`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: if mlir-opt greets you with "did you intend to nest?", come back to this slide — explicit --pass-pipeline strings never auto-nest, bare pass flags do.
Both error messages are real output from this tree (the "<unknown>:0:" / "in.mlir:0:0:" location prefixes and a trailing "failed to add `affine-loop-tile` with options ``" line are trimmed) — read them aloud, students will see them within minutes of starting the exercise and pattern-matching on the text saves TA round-trips.
The asymmetry: --pass-pipeline is EXPLICIT nesting (you must write the full tree, starting from the top-level anchor op or `any`); bare pass flags are IMPLICIT nesting (mlir-opt inserts the func.func level for you). Same in C++: default is Nesting::Explicit — pm.addPass(funcPass) on a module PM is a fatal error; use addNestedPass.
--dump-pass-pipeline is the truth serum: it prints the canonicalized tree including implicit nesting and all default option values. Teach it early.
(+1 if presented.)
-->

---

# Textual pipelines: the syntax ⏱

```bash
build/bin/mlir-opt in.mlir \
  --pass-pipeline="builtin.module(func.func(canonicalize{top-down=true max-iterations=5},cse))"
```

- Starts with the **top-level anchor** (`builtin.module`, or `any` for op-agnostic).
- Parentheses = nesting. **Commas separate passes.**
- Options in braces, **space-separated** `key=value` pairs — *not* commas!

```text
$ ... --dump-pass-pipeline
Pass Manager with 1 passes:
builtin.module(
  func.func(
    canonicalize{ ... max-iterations=5 ... top-down=true},
    cse
  )
)
```

(Real output, trimmed: the dump spells out *all* option values, including defaults you didn't set.)

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: in pipeline strings, commas separate passes and spaces separate options inside braces — and --dump-pass-pipeline prints the tree that was actually built.
The comma-vs-space trap: commas separate PASSES, spaces separate OPTIONS inside braces. canonicalize{top-down=true,max-iterations=5} is a parse error; canonicalize{top-down=true max-iterations=5} is right.
The dump output shown is captured from this tree with exactly the command above.
Also mention: --pass-pipeline cannot be combined with bare pass flags in the same invocation.
(+2 if presented.)
-->

---

# 🧠 Quiz: three pipelines ⏱

What does each one do — run, or error (and why)?

**A**
```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(canonicalize{top-down=true max-iterations=5},cse))"
```

**B**
```bash
mlir-opt in.mlir --pass-pipeline="func.func(cse)"
```

**C**
```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(affine-loop-tile)"
```

Bonus: what does plain `mlir-opt -affine-loop-tile in.mlir` do?

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip together with the answers slide that follows); if skipped, say: rule of thumb — --pass-pipeline means you spell out the whole tree, bare flags mean mlir-opt guesses it; when surprised, --dump-pass-pipeline.
~90 seconds think time. Answers (also next slide):
A: VALID. canonicalize (with those two options) then cse, on every func.func in the module; functions processed potentially in parallel.
B: ERROR — "can't run 'func.func' pass manager on 'builtin.module' op". The pipeline string must start from the top-level op of the input (builtin.module) or `any`.
C: ERROR — "Can't add pass 'AffineLoopTiling' restricted to 'func.func' ... did you intend to nest?" — affine-loop-tile is func-anchored; explicit pipelines don't auto-nest.
Bonus: WORKS — bare flags use implicit nesting, building builtin.module(func.func(affine-loop-tile)). Verify claim live with --dump-pass-pipeline if the room is skeptical.
(+2 if presented.)
-->

---

# ✅ Pipeline quiz answers ⏱

| | Verdict | Why |
|---|---|---|
| **A** | ✔ runs | Full explicit tree; `canonicalize` + `cse` on every `func.func`. |
| **B** | ✘ error | `can't run 'func.func' pass manager on 'builtin.module' op` — the string must start at the top-level anchor. |
| **C** | ✘ error | `…restricted to 'func.func' … did you intend to nest?` — explicit pipelines never auto-nest. |
| bonus | ✔ runs | Bare flags use **implicit** nesting → `builtin.module(func.func(affine-loop-tile))`. |

Rule of thumb: **`--pass-pipeline` = you spell out the whole tree. Bare flags = mlir-opt guesses the tree.** When surprised: `--dump-pass-pipeline`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; paired with the quiz slide before it — skip or present the pair together.
If anyone asks why explicit mode refuses to guess: pipelines are contracts; silently relocating a pass to a different anchor changes what it can see and touch (next slide: threading). The error text is deliberately educational — "did you intend to nest?".
(+1 if presented.)
-->

---

# Why anchors matter: threading ⏱

The pass manager runs an anchored pipeline **in parallel over sibling anchor ops** — all `func.func` in a module at once, one thread each (pipeline copies per thread; the whole pipeline runs per function, function by function).

From the `OperationPass` class comment — the law:

```cpp
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
```

<sub>mlir/include/mlir/Pass/Pass.h:354-360</sub>

Practical rules for your pass:

- Touch only IR **nested under `getOperation()`**. Never sibling functions, never the parent module. (Reading ancestors is permitted; mutating them is not.)
- No mutable state across `runOnOperation` calls; no global mutable state. Passes must be **copy-constructible** (remember `clonePass`?).
- Anchor ops must be **`IsolatedFromAbove`** — they can't reference SSA values from enclosing regions, so sibling-parallel passes can't race through use-def chains. That's why `func.func` carries the trait, and why passes can only be anchored on such ops (the PM enforces it at runtime).

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the pass manager runs anchored pipelines in parallel over sibling functions, so a pass may only touch IR under its own getOperation() — and "works with --mlir-disable-threading but crashes without" means that rule was broken.
Concrete mental picture: module with 100 functions, pipeline canonicalize+cse → the PM clones the pipeline once per worker thread and parallel-for-eaches over the functions. Each function gets the ENTIRE pipeline before the thread moves on (cache-friendly), so it's NOT "canonicalize everywhere, then cse everywhere".
IsolatedFromAbove in one sentence: an op is isolated if the IR inside it cannot use values defined outside it — so two threads mutating two sibling functions can never touch the same use-list. Scheduling a pass on a non-isolated or unregistered op is a runtime error ("trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'").
If someone's pass "works with --mlir-disable-threading but crashes without" — they broke one of these rules. That flag is a diagnostic tool, not a fix.
(+2 if presented.)
-->

---

# The debugging toolkit (1): watch the IR move

| Flag | What it does |
|---|---|
| `--mlir-print-ir-before=cse` | dump IR before a specific pass |
| `--mlir-print-ir-after=cse` | dump IR after a specific pass |
| `--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` | dump around *every* pass |
| `--mlir-print-ir-after-change` | after-dumps only when the pass changed something |
| `--mlir-print-ir-after-failure` | dump only when a pass fails |
| `--mlir-print-ir-tree-dir=/tmp/dumps` | one file per pass, in a directory tree |

Banner format (real):

```text
// -----// IR Dump After CSEPass: cse //----- //
```

First reflex when a pipeline misbehaves: `--mlir-print-ir-after-all` and read the story top to bottom.

<!-- Speaker notes:
Rank for beginners: after-all first (bisect by eye), then before/after=<pass> once suspicion narrows, after-change to cut noise in long pipelines, tree-dir when output gets too big for a terminal.
Fine print: --mlir-print-ir-module-scope (print the whole module instead of just the anchor op) requires --mlir-disable-threading — you can't safely print IR other threads are mutating. Nice callback to the threading slide (if presented).
~1 min.
-->

---

# The debugging toolkit (2): time, count, trace, reproduce ⏱

| Flag | What it does |
|---|---|
| `--mlir-timing` | wall time per pass (`--mlir-timing-display=list\|tree`) |
| `--mlir-pass-statistics` | print pass `Statistic`s (like CSE's `num-cse'd`) |
| `--debug-only=greedy-rewriter` | targeted `LLVM_DEBUG` tracing (`--debug` = firehose) |
| `--dump-pass-pipeline` | show the pipeline tree that was actually built |
| `--verify-each=false` | skip inter-pass verification (perf runs only!) |
| `--mlir-pass-pipeline-crash-reproducer=repro.mlir` | on failure/crash: write input IR + pipeline config into one file |

In C++ / gdb: `op->dump()` prints any op, block, region, or value to stderr.

Reproducer round-trip (📸 captured output — don't run live):

```bash
mlir-opt big.mlir --pass-pipeline="..." --mlir-pass-pipeline-crash-reproducer=repro.mlir
mlir-opt repro.mlir --run-reproducer     # replays the embedded pipeline
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: this table is on the cheat sheet — the one flag to remember is --mlir-pass-pipeline-crash-reproducer, which writes the input IR + pipeline into one replayable repro.mlir.
Pre-captured — do not run live unless ahead of schedule; command: mlir-opt big.mlir --pass-pipeline="..." --mlir-pass-pipeline-crash-reproducer=repro.mlir, then mlir-opt repro.mlir --run-reproducer.
Crash reproducers: the generated file is the input IR plus an embedded `mlir_reproducer` resource blob recording pipeline string, threading and verify flags; --run-reproducer replays it exactly. Add --mlir-pass-pipeline-local-reproducer (needs --mlir-disable-threading) to shrink it to just the failing pass with the IR immediately before it. This is also how you ask for help: attach repro.mlir, not a 200-line shell history.
--debug-only names to know already: pass-manager (scheduling), greedy-rewriter (Session 2's driver). Values are comma-separable.
(+2 if presented.)
-->

---

# 🔴 Live demo: `--mlir-print-ir-after-all`

```bash
build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
  --pass-pipeline='builtin.module(func.func(mul-to-shift,canonicalize))' \
  --mlir-print-ir-after-all -o /dev/null
```

```text
// -----// IR Dump After {anonymous}::MulToShiftPass: mul-to-shift //----- //
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c8_i32 = arith.constant 8 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}

// -----// IR Dump After CanonicalizerPass: canonicalize{...} //----- //
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}
```

The dead `%c8_i32` visibly survives our pass and dies in `canonicalize` — the whole story in two dumps.

<!-- Speaker notes:
Same live run as the worked-example demo — identical command with --mlir-print-ir-after-all appended; up-arrow in the same shell, nothing new to set up.
Real output (trimmed: @mul_by_7 dumps and the full canonicalize option list omitted; the banner actually reads canonicalize{cse-between-iterations=false max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}).
Things to narrate: (1) one dump per (pass, anchor-op) pair — you'll see each func dumped separately, and the func order may vary: threads! (2) the banner shows pass class name + full canonicalized pass options; (3) this flag is how you debug the exercise — school-opt supports all the same flags because they come from the PassManager, not from mlir-opt.
If time allows, re-run with --mlir-timing appended and show the per-pass timing table.
~2 min. End of lecture content — core path total ≈40 min; you should be at ≈0:45, ready for the exercise briefing.
-->

---

# Exercise 1: `-school-strength-reduce`

**Task:** in the `exercises/` project, implement the pass stub: rewrite `arith.muli %x, C` (C a power of two) → `arith.shli %x, log2(C)`. Hand-written walk — exactly today's material.

**Where:** the `// TODO(exercise 1, step N)` markers in `lib/School/StrengthReduce.cpp`; run via **`school-opt`**. Hints in `exercises/exercise1.md`.

**Checkpoints** (2–3 covered by the FileCheck test `test/exercise1/strength-reduce.mlir`; 1 verified by eye — stderr prints):

1. **Match & print** — `llvm::errs()` every `arith.muli` whose RHS is a constant power of two. *(walk, `matchPattern` + `m_ConstantInt`)*
2. **Rewrite** — build the shift-amount constant + `arith.shli` with `OpTy::create`, RAUW, erase the `muli`.
3. **Robustness** — non-constant RHS, non-power-of-two, several `muli`s per function. No crashes.

**Check your progress** (from `exercises/`):

```bash
ninja -C build && <llvm-build>/bin/llvm-lit -v build/test/exercise1
```

**Stretch goals:** (a) count rewrites in the pre-declared `numRewrites` `Statistic`, watch `--mlir-pass-statistics`; (b) handle constant on the *left* too; (c) `muli %x, 1` → just RAUW with `%x`, no new op.

<!-- Speaker notes:
Briefing ~5 min (0:45-0:50), then 30 min hands-on (0:50-1:20); instructors circulate. Solution walkthrough + wrap-up fill 1:20-1:30.
Checkpoint verification, say it precisely: checkpoints 2-3 are covered by test/exercise1/strength-reduce.mlir; checkpoint 1 has no lit test — verify it by eye from the llvm::errs() prints on stderr.
Common failure modes to watch for (from the quiz): erase before RAUW (fatal error names the op — teach them to READ it); erasing non-visited ops inside the walk (nudge to collect-then-mutate); forgetting the null-check on getDefiningOp when they try manual constant matching instead of matchPattern; getIntegerAttr with the wrong type (shift amount must have the muli's result type — the verifier catches it, point them at the error).
Checkpoint pacing: everyone should reach checkpoint 2; checkpoint 3 separates the careful from the fast; stretch (a) revisits the statistics slide; stretch (b) sets up the Session 3 debrief question "why does upstream not need the LHS case?" (canonicalization moves constants right — don't spoil it yet).
Debrief questions for the wrap-up: Who iterated until fixpoint? Who re-implemented "did anything change" bookkeeping? How annoying was the manual walk? → that pain is Session 2's motivation.
-->

---

# Recap: what you can do now

- **Model** — IR is `Operation*` all the way down: ops → regions → blocks → ops; `Value` = `OpResult` | `BlockArgument`.
- **Navigate** — parent/child accessors, `getDefiningOp` (null for block args!), `getUsers`/`getUses`.
- **Traverse** — `walk()` with typed callbacks, `WalkResult`, post-order default; collect-then-mutate.
- **Create** — `OpBuilder` insertion points (`OpBuilder(op)` = *before* `op`), `OpTy::create(b, loc, ...)`, reuse locations.
- **Rewire** — `replaceAllUsesWith` *then* `erase()`. Always in that order.
- **Package** — `Passes.td` → `GEN_PASS_DEF` → `impl::...Base` → `runOnOperation()`; options & statistics for free.
- **Run** — `--pass-pipeline="builtin.module(func.func(...))"`, nesting rules, parallel-over-functions, and the debug flags.

<!-- Speaker notes:
Run through fast — each bullet should now feel earned rather than new. Ask for a show of hands per bullet: "comfortable / shaky?" Shaky bullets tell you what to revisit in the next session's warm-up quiz.
~2 min — part of the 1:20-1:30 solution-walkthrough + wrap-up block.
-->

---

# Cheat sheet: today's API surface

```cpp
// Navigate                                  // Traverse
op->getParentOp();  op->getBlock();          root->walk([](arith::MulIOp m) { ... });
op->getOperand(0);  op->getResults();        root->walk([](Operation *o) {
v.getDefiningOp<arith::ConstantOp>();          return WalkResult::interrupt(); });
v.getUsers();  v.use_empty();  v.hasOneUse();

// Match constants                           // Create
APInt c;                                     OpBuilder b(op);        // BEFORE op!
matchPattern(v, m_ConstantInt(&c));          OpBuilder::InsertionGuard g(b);
c.isPowerOf2(); c.logBase2();                auto s = arith::ShLIOp::create(b, loc, x, y);

// Rewire (in this order!)                   // Pass
oldOp->replaceAllUsesWith(newOp->getResults());  struct P : impl::PPassBase<P> {
oldOp->erase();                                    void runOnOperation() override; };
                                                 return signalPassFailure();
```

```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(my-pass{opt=1},cse))" \
  --dump-pass-pipeline --mlir-print-ir-after-all --mlir-timing --mlir-pass-statistics
```

<!-- Speaker notes:
This slide is designed to be the one screenshot students keep open during the exercise — point students at it at the end of the briefing, then return to it during wrap-up. Point at the two "in this order!" lines one last time.
~1 min (wrap-up block).
-->

---

# Further reading (in your checkout)

- **`mlir/docs/Tutorials/UnderstandingTheIRStructure.md`** — today's traversal/def-use content, with diagrams; maps 1:1 to `-test-print-nesting` / `-test-print-defuse`.
  ⚠ Its walk examples call `getFunction()` — that accessor is gone; read it as `getOperation()`.
- **`mlir/docs/PassManagement.md`** — the authoritative pass-infrastructure reference (anchoring, threading contract, options, statistics, reproducers).
  ⚠ Some console-output examples predate current behavior (old dump banners, unanchored `-pass-pipeline` strings) — trust the binary over the prose.
- **`mlir/docs/Tutorials/MlirOpt.md`** — a guided tour of `mlir-opt` flags with runnable examples.
- Source worth reading whole: `mlir/lib/Transforms/StripDebugInfo.cpp`, `LoopInvariantCodeMotion.cpp`, `mlir/test/lib/IR/TestPrintNesting.cpp`.

<!-- Speaker notes:
The staleness warnings are deliberate: teaching students that docs drift and binaries don't is itself a lesson. "When docs and mlir-opt --help disagree, the binary wins."
~1 min (wrap-up block).
-->

---

# Next session: stop writing the boring parts

Today you hand-wrote: the traversal, the match, the RAUW, the erase — and left a dead constant behind.

```cpp
// Session 2 preview: the SAME rewrite as a pattern (Exercise 2A's MulByPow2ToShl)
struct MulByPow2ToShl : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    // ... just the match + the replacement. Nothing else.
  }
};
```

A **driver** (`applyPatternsGreedily`) brings: the worklist, re-visiting changed ops, fixpoint iteration, folding, **and dead-code cleanup** — your leftover constant disappears for free.

**Session 2: Rewrite Patterns & Dialect Conversion.** Bring your Exercise 1 solution — we'll port it in ten lines.

<!-- Speaker notes:
End on the hook: everything that was fiddly today becomes declarative next time. The dead-constant cliffhanger from the demo resolves in Session 2 (greedy driver DCE) and fully in Session 3 (why folding/canonicalization exist as op hooks).
Logistics: remind students to keep their exercises/ build directory — Sessions 2 and 3 build on the same project (Exercise 2 ports this rewrite; Exercise 3 improves the school dialect itself).
~1 min (wrap-up block; session ends at 1:30). Done — thanks!
-->
